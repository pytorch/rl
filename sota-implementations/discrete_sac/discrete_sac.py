# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Discrete SAC Example.

This is a simple self-contained example of a discrete SAC training script.

It supports gym state environments like CartPole.

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
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_sac_agent,
)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

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
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model = make_sac_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

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
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update(sampled_tensordict):
        optimizer.zero_grad(set_to_none=True)

        # Compute loss
        loss_out = loss_module(sampled_tensordict)

        actor_loss, q_loss, alpha_loss = (
            loss_out["loss_actor"],
            loss_out["loss_qvalue"],
            loss_out["loss_alpha"],
        )

        # Update critic
        (q_loss + actor_loss + alpha_loss).backward()
        optimizer.step()

        # Update target params
        target_net_updater.step()

        return loss_out.detach()

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Create off-policy collector
    collector = make_collector(
        cfg,
        train_env,
        model[0],
        compile=compile_mode is not None,
        compile_mode=compile_mode,
        cudagraphs=cfg.compile.cudagraphs,
    )

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
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)
        with timeit("collecting"):
            collected_data = next(c_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()
        current_frames = collected_data.numel()

        pbar.update(current_frames)

        collected_data = collected_data.reshape(-1)
        with timeit("rb - extend"):
            # Add to replay buffer
            replay_buffer.extend(collected_data)
        collected_frames += current_frames

        # Optimization steps
        if collected_frames >= init_random_frames:
            tds = []
            for _ in range(num_updates):
                with timeit("rb - sample"):
                    # Sample from replay buffer
                    sampled_tensordict = replay_buffer.sample()

                with timeit("update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    sampled_tensordict = sampled_tensordict.to(device)
                    loss_out = update(sampled_tensordict).clone()

                tds.append(loss_out)

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)
            tds = torch.stack(tds).mean()

        # Logging
        episode_end = (
            collected_data["next", "done"]
            if collected_data["next", "done"].any()
            else collected_data["next", "truncated"]
        )
        episode_rewards = collected_data["next", "episode_reward"][episode_end]

        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = collected_data["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )

        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = tds["loss_qvalue"]
            metrics_to_log["train/a_loss"] = tds["loss_actor"]
            metrics_to_log["train/alpha_loss"] = tds["loss_alpha"]

        # Evaluation
        prev_test_frame = ((i - 1) * frames_per_batch) // eval_iter
        cur_test_frame = (i * frames_per_batch) // eval_iter
        final = current_frames >= collector.total_frames
        if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
