# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import InteractionType

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictReplayBuffer,
)

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import EnvCreator, ParallelEnv

from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, SafeModule
from torchrl.modules.distributions import OneHotCategorical

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.record.loggers import generate_exp_name, get_logger


def env_maker(env_name, frame_skip=1, device="cpu", from_pixels=False):
    return GymEnv(
        env_name, device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )


def env_factory(env_name, num_workers):
    """Creates an instance of the environment."""

    # 1.2 Create env vector
    vec_env = ParallelEnv(
        create_env_fn=EnvCreator(lambda: env_maker(env_name=env_name)),
        num_workers=num_workers,
    )

    return vec_env


def make_replay_buffer(
        buffer_size=1000000,
        batch_size=256,
        buffer_scratch_dir="/tmp/",
        device="cpu",
        prefetch=3,
):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        batch_size=batch_size,
        prefetch=prefetch,
        storage=LazyMemmapStorage(
            buffer_size,
            scratch_dir=buffer_scratch_dir,
            device=device,
        ),
    )
    return replay_buffer


def make_logger(backend="csv"):
    exp_name = generate_exp_name("Discrete_SAC", f"Atari_christodoulou19_{env_name}")
    logger = get_logger(backend, logger_name="discrete_sac", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "PongNoFrameskip-v4"
    max_frames_per_traj = 500
    total_frames = 1_000_000
    record_interval = 1_000_000
    init_random_frames = 5000
    frames_per_batch = 500
    num_updates = 500
    buffer_size = 1_000_000
    env_per_collector = 1
    gamma = 0.99
    batch_size = 256
    lr = 3.0e-4
    weight_decay = 0.0
    target_update_polyak = 0.995
    target_entropy_weight = 0.2
    logger_backend = "csv"
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_env = env_factory(env_name, num_workers=1)
    num_actions = test_env.action_spec.space.n

    # Create Agent
    # Define Actor Network
    in_keys = ["observation"]

    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": num_actions,
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    actor_module = SafeModule(
        module=actor_net,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=test_env.action_spec),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    ).to(device)

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": num_actions,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=in_keys,
        module=qvalue_net,
    ).to(device)

    # init nets
    with torch.no_grad():
        td = test_env.reset()
        td = td.to(device)
        actor(td)
        qvalue(td)

    del td
    test_env.close()
    test_env.eval()

    model = torch.nn.ModuleList([actor, qvalue])

    # Create SAC loss
    loss_module = DiscreteSACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_actions=num_actions,
        num_qvalue_nets=2,
        target_entropy_weight=target_entropy_weight,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=target_update_polyak)

    # Make Off-Policy Collector
    collector = SyncDataCollector(
        env_factory,
        create_env_kwargs={"num_workers": env_per_collector},
        policy=model[0],
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=max_frames_per_traj,
        total_frames=total_frames,
        device=device,
    )
    collector.set_seed(seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device,
    )

    # Optimizers
    params = list(loss_module.parameters())
    optimizer_actor = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    logger = make_logger(backend=logger_backend)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames)
    for i, tensordict in enumerate(collector):

        collector.update_policy_weights_()
        pbar.update(tensordict.numel())

        # Train loging
        episode_rewards = tensordict["next", "episode_reward"][tensordict["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)

        # Extend the replay buffer with the new data
        if "mask" in tensordict.keys():
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["mask"].sum()
            tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()

        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames
        total_collected_epochs = tensordict["collector"]["traj_ids"].max().item()

        # optimization steps
        if collected_frames >= init_random_frames:

            losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):
                sampled_tensordict = replay_buffer.sample().clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                loss = actor_loss + q_loss + alpha_loss
                optimizer_actor.zero_grad()
                loss.backward()
                optimizer_actor.step()

                # update qnet_target params
                target_net_updater.step()

                losses[j] = loss_td.detach()

            losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses.items():
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

