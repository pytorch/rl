# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3 Example.

This is a simple self-contained example of a TD3 training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""

import time

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm

from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_td3_agent,
)


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = torch.device(cfg.network.device)

    # Create Logger
    exp_name = generate_exp_name("TD3", cfg.env.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create Environments
    train_env, eval_env = make_environment(cfg)

    # Create Agent
    model, exploration_policy = make_td3_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create Off-Policy Collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Create Replay Buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device=device,
    )

    # Create Optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

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
    delayed_updates = cfg.optim.policy_update_delay
    prb = cfg.replay_buffer.prb
    eval_rollout_steps = cfg.collector.max_frames_per_traj // cfg.env.frame_skip
    eval_iter = cfg.logger.eval_iter
    frames_per_batch, frame_skip = cfg.collector.frames_per_batch, cfg.env.frame_skip
    update_counter = 0

    sampling_start = time.time()
    for tensordict in collector:
        sampling_time = time.time() - sampling_start
        exploration_policy.step(tensordict.numel())

        # update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            (
                actor_losses,
                q_losses,
            ) = ([], [])
            for _ in range(num_updates):
                update_counter += 1
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample().clone()

                # compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]

                # update critic
                optimizer_critic.zero_grad()
                update_actor = update_counter % delayed_updates == 0
                q_loss.backward(retain_graph=update_actor)
                optimizer_critic.step()
                q_losses.append(q_loss.item())

                # update actor
                if update_actor:
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    actor_losses.append(actor_loss.item())

                    # update target params
                    target_net_updater.step()

                # update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_rewards = tensordict["next", "episode_reward"][
            tensordict["next", "done"]
        ]

        # logging
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][
                tensordict["next", "done"]
            ]
            logger.log_scalar(
                "train/reward", episode_rewards.mean().item(), collected_frames
            )
            logger.log_scalar(
                "train/episode_length",
                episode_length.sum().item() / len(episode_length),
                collected_frames,
            )

        if collected_frames >= init_random_frames:
            logger.log_scalar("train/q_loss", np.mean(q_losses), step=collected_frames)
            if update_actor:
                logger.log_scalar(
                    "train/a_loss", np.mean(actor_losses), step=collected_frames
                )
            logger.log_scalar("train/sampling_time", sampling_time, collected_frames)
            logger.log_scalar("train/training_time", training_time, collected_frames)

        # evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch * frame_skip:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    exploration_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                logger.log_scalar("eval/reward", eval_reward, step=collected_frames)
                logger.log_scalar("eval/time", eval_time, step=collected_frames)

        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
