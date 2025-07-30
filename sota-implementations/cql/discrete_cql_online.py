# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Discrete (DQN) CQL Example.

This is a simple self-contained example of a discrete CQL training script.

It supports state environments like gym and gymnasium.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict.nn import CudaGraphModule
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_discrete_cql_optimizer,
    make_discrete_loss,
    make_discretecql_model,
    make_environment,
    make_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="discrete_online_config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("DiscreteCQL", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="discretecql_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, explore_policy = make_discretecql_model(cfg, train_env, eval_env, device)

    # Create loss
    loss_module, target_net_updater = make_discrete_loss(cfg.loss, model, device=device)

    compile_mode = None
    if cfg.compile.compile:
        if cfg.compile.compile_mode not in (None, ""):
            compile_mode = cfg.compile.compile_mode
        elif cfg.compile.cudagraphs:
            compile_mode = "default"
        else:
            compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg,
        train_env,
        explore_policy,
        compile=cfg.compile.compile,
        compile_mode=compile_mode,
        cudagraph=cfg.compile.cudagraphs,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    optimizer = make_discrete_cql_optimizer(cfg, loss_module)

    def update(sampled_tensordict):
        # Compute loss
        optimizer.zero_grad(set_to_none=True)
        loss_dict = loss_module(sampled_tensordict)

        q_loss = loss_dict["loss_qvalue"]
        cql_loss = loss_dict["loss_cql"]
        loss = q_loss + cql_loss

        # Update model
        loss.backward()
        optimizer.step()

        # Update target params
        target_net_updater.step()
        return loss_dict.detach()

    if compile_mode:
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_rollout_steps = cfg.env.max_episode_steps
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch

    c_iter = iter(collector)
    total_iter = len(collector)
    for _ in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)
        with timeit("collecting"):
            torch.compiler.cudagraph_mark_step_begin()
            tensordict = next(c_iter)

        # Update exploration policy
        explore_policy[1].step(tensordict.numel())

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        tensordict = tensordict.reshape(-1)
        with timeit("rb - extend"):
            # Add to replay buffer
            replay_buffer.extend(tensordict)
        collected_frames += current_frames

        # Optimization steps
        if collected_frames >= init_random_frames:
            tds = []
            for _ in range(num_updates):
                # Sample from replay buffer
                with timeit("rb - sample"):
                    sampled_tensordict = replay_buffer.sample()
                    sampled_tensordict = sampled_tensordict.to(device)
                with timeit("update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    loss_dict = update(sampled_tensordict).clone()
                tds.append(loss_dict)

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        metrics_to_log = {}
        # Evaluation
        with timeit("eval"):
            if collected_frames % eval_iter < frames_per_batch:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    eval_rollout = eval_env.rollout(
                        eval_rollout_steps,
                        model,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                    metrics_to_log["eval/reward"] = eval_reward

        # Logging
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
            metrics_to_log["train/epsilon"] = explore_policy[1].eps

        if collected_frames >= init_random_frames:
            tds = torch.stack(tds, dim=0).mean()
            metrics_to_log["train/q_loss"] = tds["loss_qvalue"]
            metrics_to_log["train/cql_loss"] = tds["loss_cql"]

        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()


if __name__ == "__main__":
    main()
