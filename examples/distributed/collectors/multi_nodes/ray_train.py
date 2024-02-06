"""
Train example with a distributed collector
==========================================

This script reproduces the PPO example in https://pytorch.org/rl/tutorials/coding_ppo.html
with a RayCollector.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.collectors.distributed.ray import RayCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm


if __name__ == "__main__":

    # 1. Define Hyperparameters
    device = "cpu"  # if not torch.cuda.device_count() else "cuda:0"
    num_cells = 256
    max_grad_norm = 1.0
    frame_skip = 1
    num_collectors = 2
    lr = 3e-4
    frames_per_batch = 1000 // frame_skip
    total_frames = 50_000 // frame_skip
    sub_batch_size = 64
    num_epochs = 10
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    # 2. Define Environment
    base_env = GymEnv("InvertedDoublePendulum-v4", device=device, frame_skip=frame_skip)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    check_env_specs(env)

    # 3. Define actor and critic
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    policy_module(env.reset())
    value_module(env.reset())

    # 4. Distributed collector
    remote_config = {
        "num_cpus": 1,
        "num_gpus": 0.1,
        "memory": 1024**3,
        "object_store_memory": 1024**3,
    }
    collector = RayCollector(
        create_env_fn=[env] * num_collectors,
        policy=policy_module,
        collector_class=SyncDataCollector,
        collector_kwargs={
            "max_frames_per_traj": 50,
            "device": device,
        },
        remote_configs=remote_config,
        num_collectors=num_collectors,
        total_frames=total_frames,
        sync=False,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        update_after_each_batch=True,
    )

    # 5. Define replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # 6. Define loss
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic_network=value_module,
        advantage_key="advantage",
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,  # these keys match by default but we set this for completeness
        value_target_key=advantage_module.value_target_key,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=0.99)

    # 7. Define optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    # 8. Define training loop
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames * frame_skip)
    eval_str = ""
    # We iterate over the distributed_collector until it reaches the total number of frames it was
    # designed to collect:
    for tensordict_data in collector:
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata, *_ = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel() * frame_skip)
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        with set_exploration_mode("mean"), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} (init: {logs['eval reward (sum)'][0]: 4.4f}), eval step-count: {logs['eval step_count'][-1]}"
            del eval_rollout
        pbar.set_description(
            ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
        )

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    # 9. Plot results
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    save_name = "/tmp/results.jpg"
    plt.savefig(save_name)
    torchrl_logger.info(f"results saved in {save_name}")
