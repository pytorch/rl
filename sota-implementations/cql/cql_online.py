# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CQL Example.

This is a self-contained example of an online CQL training script.

It works across Gym and MuJoCo over a variety of tasks.

The helper functions are coded in the utils.py associated with this script.

"""
import time

import hydra
import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    log_metrics,
    make_collector,
    make_continuous_cql_optimizer,
    make_continuous_loss,
    make_cql_model,
    make_environment,
    make_replay_buffer,
)


@hydra.main(version_base="1.1", config_path="", config_name="online_config")
def main(cfg: "DictConfig"):  # noqa: F821
    # Create logger
    exp_name = generate_exp_name("CQL-online", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="cql_logging",
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
    device = torch.device(cfg.optim.device)

    # Create env
    train_env, eval_env = make_environment(
        cfg,
        cfg.env.train_num_envs,
        cfg.env.eval_num_envs,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device="cpu",
    )

    # create agent
    model = make_cql_model(cfg, train_env, eval_env, device)

    # Create collector
    collector = make_collector(cfg, train_env, actor_model_explore=model[0])

    # Create loss
    loss_module, target_net_updater = make_continuous_loss(cfg.loss, model)

    # Create optimizer
    (
        policy_optim,
        critic_optim,
        alpha_optim,
        alpha_prime_optim,
    ) = make_continuous_cql_optimizer(cfg, loss_module)
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
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.collector.max_frames_per_traj

    sampling_start = time.time()
    for tensordict in collector:
        sampling_time = time.time() - sampling_start
        pbar.update(tensordict.numel())
        # update weights of the inference policy
        collector.update_policy_weights_()

        tensordict = tensordict.view(-1)
        current_frames = tensordict.numel()
        # add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            log_loss_td = TensorDict({}, [num_updates])
            for j in range(num_updates):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                cql_loss = loss_td["loss_cql"]
                q_loss = q_loss + cql_loss
                alpha_loss = loss_td["loss_alpha"]
                alpha_prime_loss = loss_td["loss_alpha_prime"]

                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                policy_optim.zero_grad()
                actor_loss.backward()
                policy_optim.step()

                if alpha_prime_optim is not None:
                    alpha_prime_optim.zero_grad()
                    alpha_prime_loss.backward(retain_graph=True)
                    alpha_prime_optim.step()

                critic_optim.zero_grad()
                q_loss.backward(retain_graph=False)
                critic_optim.step()

                log_loss_td[j] = loss_td.detach()

                # update qnet_target params
                target_net_updater.step()

                # update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_rewards = tensordict["next", "episode_reward"][
            tensordict["next", "done"]
        ]
        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][
                tensordict["next", "done"]
            ]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/loss_actor"] = log_loss_td.get("loss_actor").mean()
            metrics_to_log["train/loss_qvalue"] = log_loss_td.get("loss_qvalue").mean()
            metrics_to_log["train/loss_alpha"] = log_loss_td.get("loss_alpha").mean()
            metrics_to_log["train/loss_alpha_prime"] = log_loss_td.get(
                "loss_alpha_prime"
            ).mean()
            metrics_to_log["train/entropy"] = log_loss_td.get("entropy").mean()
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

        log_metrics(logger, metrics_to_log, collected_frames)
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")

    collector.shutdown()


if __name__ == "__main__":
    main()
