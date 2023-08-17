import gym
import time
import tqdm
import torch
import numpy as np
from torch import nn, optim

from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    RewardSum,
    DoubleToFloat,
    TransformedEnv,
    ExplorationType,
    set_exploration_type,
)
from torchrl.record.loggers import generate_exp_name, get_logger


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu"):
    env = gym.make(env_name)
    env = GymWrapper(env, device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_sac_agent(env_name, device):
    """Make SAC agent."""

    proof_environment = make_env(env_name=env_name, device=device)

    # Define Actor Network
    in_keys = ["observation"]
    action_spec = proof_environment.action_spec
    if proof_environment.batch_size:
        action_spec = action_spec[(0,) * len(proof_environment.batch_size)]
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class":  nn.ReLU,
    }
    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping="biased_softplus_1.0",
        scale_lb=0.1,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class":  nn.ReLU,
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
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = proof_environment.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    proof_environment.close()

    return model, model[0]


# ====================================================================
# Collector utils
# --------------------------------------------------------------------

def make_collector(env_name, policy, device):
    """Make collector."""
    collector = SyncDataCollector(
        make_env(env_name, device),
        policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=-1,
        total_frames=total_frames,
        device=device,
    )
    collector.set_seed(seed)
    return collector


# ====================================================================
# Collector and replay buffer utils
# --------------------------------------------------------------------


def make_replay_buffer(
        batch_size,
        buffer_size=1_000_000,
        buffer_scratch_dir="/tmp/",
        prefetch=3,
):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=prefetch,
        storage=LazyMemmapStorage(
            max_size=buffer_size,
            scratch_dir=buffer_scratch_dir,
            device=device,
        ),
        batch_size=batch_size,
    )
    return replay_buffer


# ====================================================================
# SAC Loss
# --------------------------------------------------------------------


def make_loss_module(model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=loss_function,
        delay_actor=False,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(
        loss_module, eps=target_update_polyak
    )
    return loss_module, target_net_updater


# ====================================================================
# Other component utils
# --------------------------------------------------------------------


def make_optimizers(cfg, loss_module):
    """Make SAC optimizers."""
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    policy_optimizer = optim.Adam(
        actor_params,
        lr=cfg.optimization.lr,

    )
    qvalue_optimizer = optim.Adam(
        critic_params,
        lr=cfg.optimization.lr,

    )
    alpha_optimizer = optim.Adam(
        list([loss_module.log_alpha]),
        lr=cfg.optimization.lr,

    )
    return policy_optimizer, qvalue_optimizer, alpha_optimizer


def make_logger(backend="csv"):
    exp_name = generate_exp_name("SAC", f"Mujoco_haarnoja18_{env_name}")
    logger = get_logger(backend, logger_name="sac", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "Ant-v3"
    record_interval = 1_000_000
    total_frames = 1_000_000
    init_random_frames = 10_000
    frames_per_batch = 1000
    num_updates = 1000
    max_frames_per_traj = 1000
    init_env_steps = 1000
    lr = 3e-4
    weight_decay = 2e-4
    gamma = 0.99
    batch_size = 256
    target_update_polyak = 0.995
    loss_function = "smooth_l1"
    logger_backend = "csv"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the components
    model, model_explore = make_sac_agent(env_name, device)
    collector = make_collector(env_name, model_explore, device)
    replay_buffer = make_replay_buffer(batch_size)
    loss_module, target_net_updater = make_loss_module(model)
    policy_optimizer, qvalue_optimizer, alpha_optimizer = make_optimizers(loss_module)
    logger = make_logger(logger_backend)
    test_env = make_env(env_name, device)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for i, data in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames

        # optimization steps
        if collected_frames >= init_random_frames:

            q_losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):

                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample().clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                policy_optimizer.zero_grad()
                qvalue_optimizer.zero_grad()
                alpha_optimizer.zero_grad()

                actor_loss.backward()
                q_loss.backward()
                alpha_loss.backward()

                policy_optimizer.step()
                qvalue_optimizer.step()
                alpha_optimizer.step()

                # update qnet_target params
                target_net_updater.step()

                q_losses[j] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha").detach()

            q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in q_losses.items():
                logger.log_scalar(key, value.item(), collected_frames)

        # Test logging
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_per_batch) // record_interval < (collected_frames // record_interval):
                model.eval()
                test_rewards = []
                for i in range(30):
                    td_test = test_env.rollout(
                        policy=model,
                        auto_reset=True,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                        max_steps=10_000_000,
                    )
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards = np.append(test_rewards, reward.cpu().numpy())
                    del td_test
                logger.log_scalar("reward_test", test_rewards.mean(), collected_frames)
                model.train()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
