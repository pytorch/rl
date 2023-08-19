# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict.nn import InteractionType

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictPrioritizedReplayBuffer,
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
    total_frames = 1000000
    init_random_frames = 5000
    frames_per_batch = 500
    num_updates = 500
    buffer_size = 1000000
    env_per_collector = 1
    gamma: 0.99
    batch_size: 256
    lr: 3.0e-4
    weight_decay: 0.0
    target_update_polyak: 0.995
    target_entropy_weight: 0.2
    logger_backend: "csv"
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_name = generate_exp_name("Discrete_SAC", cfg.exp_name)
    logger = make_logger(backend=logger_backend)
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

    rewards = []
    rewards_eval = []

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames)
    r0 = None
    loss = None

    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        new_collected_epochs = len(np.unique(tensordict["collector"]["traj_ids"]))
        if r0 is None:
            r0 = tensordict["next", "reward"].sum().item() / new_collected_epochs
        pbar.update(tensordict.numel())

        # extend the replay buffer with the new data
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
            (
                total_losses,
                actor_losses,
                q_losses,
                alpha_losses,
                alphas,
                entropies,
            ) = ([], [], [], [], [], [])
            for _ in range(num_updates):
                # sample from replay buffer
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

                total_losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                q_losses.append(q_loss.item())
                alpha_losses.append(alpha_loss.item())
                alphas.append(loss_td["alpha"].item())
                entropies.append(loss_td["entropy"].item())

        rewards.append(
            (
                i, tensordict["next", "reward"].sum().item() / new_collected_epochs,
            )
        )
        metrics = {
            "train_reward": rewards[-1][1],
            "collected_frames": collected_frames,
            "epochs": total_collected_epochs,
        }

        if loss is not None:
            metrics.update(
                {
                    "total_loss": np.mean(total_losses),
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                    "alpha_loss": np.mean(alpha_losses),
                    "alpha": np.mean(alphas),
                    "entropy": np.mean(entropies),
                }
            )

        with set_exploration_type(
                ExplorationType.RANDOM
        ), torch.no_grad():  # TODO: exploration mode to mean causes nans

            eval_rollout = test_env.rollout(
                max_steps=-1,
                policy=actor,
                auto_cast_to_device=True,
            ).clone()
            eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
            rewards_eval.append((i, eval_reward))
            eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
            metrics.update({"test_reward": rewards_eval[-1][1]})
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str
            )

        # log metrics
        for key, value in metrics.items():
            logger.log_scalar(key, value, step=collected_frames)

    collector.shutdown()

