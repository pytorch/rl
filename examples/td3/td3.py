# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra

import numpy as np
import torch
import torch.cuda
import tqdm

from torch import nn, optim
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import (
    AdditiveGaussianWrapper,
    MLP,
    ProbabilisticActor,
    SafeModule,
    ValueOperator,
)
from torchrl.modules.distributions import TanhDelta

from torchrl.objectives import SoftUpdate
from torchrl.objectives.td3 import TD3Loss
from torchrl.record.loggers import generate_exp_name, get_logger


def env_maker(task, frame_skip=1, device="cpu", from_pixels=False):
    return GymEnv(
        task, "run", device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )


def apply_env_transforms(env, reward_scaling=1.0):
    transformed_env = TransformedEnv(
        env,
        Compose(
            RewardScaling(loc=0.0, scale=reward_scaling),
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
        ),
    )
    return transformed_env


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir="/tmp/",
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    exp_name = generate_exp_name("TD3", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="td3_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    parallel_env = ParallelEnv(
        cfg.env_per_collector, EnvCreator(lambda: env_maker(task=cfg.env_name))
    )
    parallel_env.set_seed(cfg.seed)

    train_env = apply_env_transforms(parallel_env)

    train_env.transform[1].init_stats(
        num_iter=cfg.init_env_steps, reduce_dim=(0, 1), cat_dim=0
    )
    # check the shape of our summary stats
    print("normalization constant shape:", train_env.transform[1].loc.shape)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.env_per_collector, EnvCreator(lambda: env_maker(task=cfg.env_name))
        ),
        train_env.transform.clone(),
    )
    assert (eval_env.transform[1].loc == train_env.transform[1].loc).all()

    # Create Agent

    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": action_spec.shape[-1],
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhDelta
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
    }

    in_keys_actor = in_keys
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "param",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["param"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    # Exploration wrappers:
    # actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
    #     actor,
    #     annealing_num_steps=1_000_000,
    # ).to(device)

    actor_model_explore = AdditiveGaussianWrapper(
        actor,
        sigma_init=1,
        sigma_end=1,
        mean=0,
        std=0.01,
    ).to(device)

    # Create TD3 loss
    loss_module = TD3Loss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        gamma=cfg.gamma,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, cfg.target_update_polyak)

    # Make Off-Policy Collector
    collector = MultiSyncDataCollector(
        # we'll just run one ParallelEnvironment. Adding elements to the list would increase the number of envs run in parallel
        [
            train_env,
        ],
        actor_model_explore,
        frames_per_batch=cfg.frames_per_batch,
        max_frames_per_traj=cfg.max_frames_per_traj,
        total_frames=cfg.total_frames,
    )
    collector.set_seed(cfg.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.batch_size,
        prb=cfg.prb,
        buffer_size=cfg.buffer_size,
        device=device,
    )

    # Optimizers
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(actor_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer_critic = optim.Adam(
        critic_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    r0 = None
    q_loss = None

    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["reward"].sum(-1).mean().item()
        pbar.update(tensordict.numel())

        # extend the replay buffer with the new data
        if ("collector", "mask") in tensordict.keys(True):
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["collector", "mask"].sum()
            tensordict = tensordict[tensordict.get(("collector", "mask")).squeeze(-1)]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # optimization steps
        if collected_frames >= cfg.init_random_frames:
            (
                actor_losses,
                q_losses,
            ) = ([], [])
            for i in range(
                int(cfg.env_per_collector * cfg.frames_per_batch * cfg.utd_ratio)
            ):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]

                optimizer_critic.zero_grad()
                q_loss.backward(retain_graph=True)
                optimizer_critic.step()
                q_losses.append(q_loss.item())

                if i % cfg.policy_update_delay == 0:
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()
                    actor_losses.append(actor_loss.item())

                    # update qnet_target params
                    target_net_updater.step()

                # update priority
                if cfg.prb:
                    replay_buffer.update_priority(sampled_tensordict)

        rewards.append((i, tensordict["reward"].sum().item() / cfg.env_per_collector))
        train_log = {
            "train_reward": rewards[-1][1],
            "collected_frames": collected_frames,
        }
        if q_loss is not None:
            train_log.update(
                {
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                }
            )
        for key, value in train_log.items():
            logger.log_scalar(key, value, step=collected_frames)

        with set_exploration_mode("mean"), torch.no_grad():
            eval_rollout = eval_env.rollout(
                cfg.max_frames_per_traj // cfg.frame_skip,
                actor_model_explore,
                auto_cast_to_device=True,
            )
            eval_reward = eval_rollout["reward"].sum(-2).mean().item()
            rewards_eval.append((i, eval_reward))
            eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
            logger.log_scalar("test_reward", rewards_eval[-1][1], step=collected_frames)
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str
            )

    collector.shutdown()


if __name__ == "__main__":
    main()
