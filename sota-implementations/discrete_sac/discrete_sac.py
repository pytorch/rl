# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Discrete SAC Example.

This is a simple self-contained example of a discrete SAC training script.

It supports gym state environments like CartPole.

The helper functions are coded in the utils.py associated with this script.
"""
import time

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from torchrl._utils import logger as torchrl_logger

from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_sac_agent,
)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = torch.device(cfg.network.device)

    # Create logger
    exp_name = generate_exp_name("DiscreteSAC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="DiscreteSAC_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model = make_sac_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create off-policy collector
    collector = make_collector(cfg, train_env, model[0])

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    optimizer_actor, optimizer_critic, optimizer_alpha = make_optimizer(
        cfg, loss_module
    )

    # Main loop
    start_time = time.time()
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    eval_rollout_steps = cfg.env.max_episode_steps
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch

    sampling_start = time.time()
    for tensordict in collector:
        sampling_time = time.time() - sampling_start

        # Update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            (
                actor_losses,
                q_losses,
                alpha_losses,
            ) = ([], [], [])
            for _ in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                # Compute loss
                loss_out = loss_module(sampled_tensordict)

                actor_loss, q_loss, alpha_loss = (
                    loss_out["loss_actor"],
                    loss_out["loss_qvalue"],
                    loss_out["loss_alpha"],
                )

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()
                q_losses.append(q_loss.item())

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                actor_losses.append(actor_loss.item())

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                alpha_losses.append(alpha_loss.item())

                # Update target params
                target_net_updater.step()

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )

        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = np.mean(q_losses)
            metrics_to_log["train/a_loss"] = np.mean(actor_losses)
            metrics_to_log["train/alpha_loss"] = np.mean(alpha_losses)
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time
        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
