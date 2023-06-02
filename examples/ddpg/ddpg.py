# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DDPG Example.

This is a simple self-contained example of a DDPG training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""

import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    make_collector,
    make_ddpg_agent,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
)


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = torch.device(cfg.network.device)

    exp_name = generate_exp_name("DDPG", cfg.env.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ddpg_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create Environments
    train_env, eval_env = make_environment(cfg)

    # Create Agent
    model, exploration_policy = make_ddpg_agent(cfg, train_env, eval_env, device)

    # Create Loss Module and Target Updater
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Make Off-Policy Collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optimization.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device=device,
    )

    # Make Optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

    rewards = []
    rewards_eval = []

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    r0 = None
    q_loss = None

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optimization.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    env_per_collector = cfg.collector.env_per_collector
    frames_per_batch, frame_skip = cfg.collector.frames_per_batch, cfg.env.frame_skip
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.collector.max_frames_per_traj // frame_skip

    for i, tensordict in enumerate(collector):
        exploration_policy.step(tensordict.numel())
        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["next", "reward"].sum(-1).mean().item()
        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # optimization steps
        if collected_frames >= init_random_frames:
            (
                actor_losses,
                q_losses,
            ) = ([], [])
            for _ in range(num_updates):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample().clone()

                loss_td = loss_module(sampled_tensordict)

                optimizer_critic.zero_grad()
                optimizer_actor.zero_grad()

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_value"]
                (actor_loss + q_loss).backward()

                optimizer_critic.step()
                q_losses.append(q_loss.item())

                optimizer_actor.step()
                actor_losses.append(actor_loss.item())

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
                }
            )
        if logger is not None:
            for key, value in train_log.items():
                logger.log_scalar(key, value, step=collected_frames)
        if abs(collected_frames % eval_iter) < frames_per_batch * frame_skip:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    exploration_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                if logger is not None:
                    logger.log_scalar(
                        "evaluation_reward", rewards_eval[-1][1], step=collected_frames
                    )
        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f})," + eval_str
            )

    collector.shutdown()


if __name__ == "__main__":
    main()
