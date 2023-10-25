# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    DMControlEnv,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

from torchrl.objectives import SoftUpdate
from torchrl.objectives.iql import IQLLoss
from torchrl.record.loggers import generate_exp_name, get_logger


def env_maker(cfg, device="cpu"):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                frame_skip=cfg.env.frame_skip,
            )
    elif lib == "dm_control":
        env = DMControlEnv(cfg.env.name, cfg.env.task, frame_skip=cfg.env.frame_skip)
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir=None,
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


@hydra.main(version_base="1.1", config_path=".", config_name="online_config")
def main(cfg: "DictConfig"):  # noqa: F821

    device = torch.device(cfg.network.device)

    exp_name = generate_exp_name("Online_IQL", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="iql_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode},
        )

    torch.manual_seed(cfg.optim.seed)
    np.random.seed(cfg.optim.seed)

    def env_factory(num_workers):
        """Creates an instance of the environment."""

        # 1.2 Create env vector
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda cfg=cfg: env_maker(cfg=cfg)),
            num_workers=num_workers,
        )

        return vec_env

    # Sanity check
    test_env = env_factory(num_workers=cfg.collector.env_per_collector)
    num_actions = test_env.action_spec.shape[-1]

    # Create Agent
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = test_env.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * num_actions,
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.low[-1],
        "max": action_spec.space.high[-1],
        "tanh_loc": cfg.network.tanh_loc,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
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
        default_interaction_type=ExplorationType.RANDOM,
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

    # Define Value Network
    value_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }
    value_net = MLP(**value_net_kwargs)
    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )

    model = nn.ModuleList([actor, qvalue, value]).to(device)

    # init nets
    with torch.no_grad():
        td = test_env.reset()
        td = td.to(device)
        actor(td)
        qvalue(td)
        value(td)

    del td
    test_env.close()
    test_env.eval()

    # Create IQL loss
    loss_module = IQLLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        value_network=model[2],
        num_qvalue_nets=2,
        temperature=cfg.loss.temperature,
        expectile=cfg.loss.expectile,
        loss_function=cfg.loss.loss_function,
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

    # Make Off-Policy Collector
    collector = SyncDataCollector(
        env_factory,
        create_env_kwargs={"num_workers": cfg.collector.env_per_collector},
        policy=model[0],
        frames_per_batch=cfg.collector.frames_per_batch,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.optim.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        buffer_size=cfg.buffer.size,
        device="cpu",
        batch_size=cfg.buffer.batch_size,
        prefetch=cfg.buffer.prefetch,
        prb=cfg.buffer.prb,
    )

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(
        params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, eps=cfg.optim.eps
    )

    rewards = []
    rewards_eval = []

    # Main loop
    collected_frames = 0

    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    r0 = None
    loss = None
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    env_per_collector = cfg.collector.env_per_collector
    prb = cfg.buffer.prb
    max_frames_per_traj = cfg.collector.max_frames_per_traj

    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["next", "reward"].sum(-1).mean().item()
        pbar.update(tensordict.numel())

        if "mask" in tensordict.keys():
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["mask"].sum()
            tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        (
            actor_losses,
            q_losses,
            value_losses,
        ) = ([], [], [])
        # optimization steps
        for _ in range(num_updates):
            # sample from replay buffer
            sampled_tensordict = replay_buffer.sample()
            if sampled_tensordict.device == device:
                sampled_tensordict = sampled_tensordict.clone()
            else:
                sampled_tensordict = sampled_tensordict.to(device, non_blocking=True)

            loss_td = loss_module(sampled_tensordict)

            actor_loss = loss_td["loss_actor"]
            q_loss = loss_td["loss_qvalue"]
            value_loss = loss_td["loss_value"]

            loss = actor_loss + q_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            q_losses.append(q_loss.item())
            actor_losses.append(actor_loss.item())
            value_losses.append(value_loss.item())

            # update qnet_target params
            target_net_updater.step()

            # update priority
            if prb:
                replay_buffer.update_priority(sampled_tensordict)

        rewards.append(
            (i, tensordict["next", "reward"].sum().item() / env_per_collector)
        )
        train_log = {
            "train_reward": rewards[-1][1],
            "collected_frames": collected_frames,
        }
        if q_loss is not None:
            train_log.update(
                {
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                    "value_loss": np.mean(value_losses),
                }
            )
        if logger is not None:
            for key, value in train_log.items():
                logger.log_scalar(key, value, step=collected_frames)

        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            eval_rollout = test_env.rollout(
                max_steps=max_frames_per_traj,
                policy=model[0],
                auto_cast_to_device=True,
            ).clone()
            eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
            rewards_eval.append((i, eval_reward))
            eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
            if logger is not None:
                logger.log_scalar(
                    "test_reward", rewards_eval[-1][1], step=collected_frames
                )
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str
            )

    collector.shutdown()


if __name__ == "__main__":
    main()
