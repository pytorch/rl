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


def make_replay_buffer(
    prb=False,
    buffer_size=1000000,
    batch_size=256,
    buffer_scratch_dir="/tmp/",
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            batch_size=batch_size,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
        )
    else:
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    exp_name = generate_exp_name("Discrete_SAC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="dSAC_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def env_factory(num_workers):
        """Creates an instance of the environment."""

        # 1.2 Create env vector
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda: env_maker(env_name=cfg.env_name)),
            num_workers=num_workers,
        )

        return vec_env

    # Sanity check
    test_env = env_factory(num_workers=5)
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
        gamma=cfg.gamma,
        target_entropy_weight=cfg.target_entropy_weight,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, cfg.target_update_polyak)

    # Make Off-Policy Collector
    collector = SyncDataCollector(
        env_factory,
        create_env_kwargs={"num_workers": cfg.env_per_collector},
        policy=model[0],
        frames_per_batch=cfg.frames_per_batch,
        max_frames_per_traj=cfg.max_frames_per_traj,
        total_frames=cfg.total_frames,
        device=cfg.device,
    )
    collector.set_seed(cfg.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        prb=cfg.prb,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        device=device,
    )

    # Optimizers
    params = list(loss_module.parameters())
    optimizer_actor = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    r0 = None
    loss = None

    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        new_collected_epochs = len(np.unique(tensordict["collector"]["traj_ids"]))
        if r0 is None:
            r0 = (
                tensordict["next", "reward"].sum().item()
                / new_collected_epochs
                / cfg.env_per_collector
            )
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
        if collected_frames >= cfg.init_random_frames:
            (
                total_losses,
                actor_losses,
                q_losses,
                alpha_losses,
                alphas,
                entropies,
            ) = ([], [], [], [], [], [])
            for _ in range(cfg.frames_per_batch * int(cfg.utd_ratio)):
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

                # update priority
                if cfg.prb:
                    replay_buffer.update_priority(sampled_tensordict)

                total_losses.append(loss.item())
                actor_losses.append(actor_loss.item())
                q_losses.append(q_loss.item())
                alpha_losses.append(alpha_loss.item())
                alphas.append(loss_td["alpha"].item())
                entropies.append(loss_td["entropy"].item())

        rewards.append(
            (
                i,
                tensordict["next", "reward"].sum().item()
                / cfg.env_per_collector
                / new_collected_epochs,
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
                max_steps=cfg.max_frames_per_traj,
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


if __name__ == "__main__":
    main()
