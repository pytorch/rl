# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO-style RL fine-tuning of a token-head VLA policy.

This is the SimpleVLA-RL recipe
(`arXiv:2509.09674 <https://arxiv.org/abs/2509.09674>`_): a token-head VLA
policy emits a whole action chunk per forward, trajectories are collected in
same-initial-state groups, the trajectory-level binary success return is
normalized within each group, and PPO updates the sampled action tokens with an
asymmetric clipping objective.
"""
from __future__ import annotations

import os
import warnings

import hydra
import torch
import tqdm
from torchrl._utils import logger as torchrl_logger, timeit
from utils import (
    advantage_metrics,
    auto_device,
    candidate_group_size,
    iteration_metrics,
    load_checkpoint,
    log_iteration_summary,
    log_metrics,
    make_action_tokenizer,
    make_collector,
    make_evaluator,
    make_logger,
    make_loss_module,
    make_optimizer,
    make_policy,
    make_replay_buffer,
    replay_ready_target,
    reset_collection_state,
    save_checkpoint,
    sync_policy_server,
    update,
    wait_for_replay,
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensordict")


@hydra.main(config_path="config", config_name="vla_grpo_toy", version_base="1.1")
def main(cfg):  # noqa: F821
    torch.manual_seed(cfg.env.seed)
    train_device = auto_device(cfg.policy.device)
    rollout_device = (
        torch.device(cfg.collector.policy_device)
        if cfg.collector.policy_device
        else train_device
    )
    eval_device = (
        torch.device(cfg.logger.eval_device)
        if cfg.logger.eval_device
        else rollout_device
    )
    buffer_device = (
        torch.device(cfg.buffer.device) if cfg.buffer.device else train_device
    )
    logger = make_logger(cfg)
    policy = make_policy(cfg, train_device)
    tokenizer = make_action_tokenizer(cfg, policy)
    replay_buffer, advantage_transform = make_replay_buffer(cfg, buffer_device)
    loss_module = make_loss_module(cfg, policy)
    optim, scheduler = make_optimizer(cfg, loss_module)
    collector, policy_server, eval_policy = make_collector(
        cfg,
        policy,
        rollout_device,
        tokenizer=tokenizer,
        replay_buffer=replay_buffer,
    )
    evaluator = make_evaluator(
        cfg,
        tokenizer,
        eval_policy,
        logger,
        eval_device,
    )

    start_iter = 0
    if cfg.checkpoint.resume:
        start_iter = load_checkpoint(
            cfg.checkpoint.resume, policy, optim, scheduler, train_device
        )
        sync_policy_server(policy_server, policy)
        torchrl_logger.info(
            "Resumed from %s at iteration %d.", cfg.checkpoint.resume, start_iter
        )

    target_replay_decisions = replay_ready_target(cfg)
    episodes_per_iter = cfg.collector.groups_per_iter * candidate_group_size(cfg)
    total_episodes = start_iter * episodes_per_iter
    sync_policy_server(policy_server, policy)
    collector.start()

    pbar = tqdm.tqdm(total=cfg.collector.total_iters, initial=start_iter)
    try:
        for iteration in range(start_iter, cfg.collector.total_iters):
            with timeit("collect"):
                collect_metrics = wait_for_replay(
                    replay_buffer,
                    min_replay_decisions=target_replay_decisions,
                    poll_interval_s=cfg.collector.replay_wait_s,
                    log_interval_s=cfg.collector.replay_log_s,
                    iteration=iteration,
                )

            with collector.pause():
                num_decisions = len(replay_buffer)
                train_metrics = update(
                    replay_buffer,
                    loss_module,
                    optim,
                    scheduler,
                    cfg,
                    train_device,
                    logger=logger,
                    iteration=iteration,
                )
                group_metrics = advantage_metrics(advantage_transform, collector)
                reset_collection_state(advantage_transform, collector)
                sync_policy_server(policy_server, policy)

            eval_metrics = {}
            if iteration % cfg.logger.eval_iter == 0:
                if cfg.logger.eval_async:
                    evaluator.trigger_eval(weights=None, step=iteration)
                else:
                    eval_metrics = evaluator.evaluate(weights=None, step=iteration)
                    log_metrics(logger, eval_metrics, iteration)
            if cfg.logger.eval_async:
                ready_eval = evaluator.poll(timeout=0)
                if ready_eval is not None:
                    eval_metrics = ready_eval
                    log_metrics(logger, eval_metrics, iteration)

            timings = timeit.todict(prefix="time")
            timeit.erase()
            completed_trajectories = int(
                group_metrics["collector/completed_trajectories"]
            )
            total_episodes += completed_trajectories
            metrics, train_success = iteration_metrics(
                cfg,
                num_decisions=num_decisions,
                total_episodes=total_episodes,
                collect_metrics=collect_metrics,
                group_metrics=group_metrics,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
                timings=timings,
            )
            metrics["train/lr"] = scheduler.get_last_lr()[0]
            log_metrics(logger, metrics, iteration)
            log_iteration_summary(
                iteration,
                train_success=train_success,
                num_decisions=num_decisions,
                group_metrics=group_metrics,
                timings=timings,
            )
            pbar.update(1)
            pbar.set_description(
                f"success {train_success:.2f} decisions {num_decisions}"
            )

            if (
                cfg.checkpoint.save_iter
                and (iteration + 1) % cfg.checkpoint.save_iter == 0
            ):
                save_checkpoint(
                    os.path.join(os.getcwd(), "checkpoint_latest"),
                    policy,
                    optim,
                    scheduler,
                    iteration,
                )

        if cfg.checkpoint.save_iter:
            save_checkpoint(
                os.path.join(os.getcwd(), "checkpoint_latest"),
                policy,
                optim,
                scheduler,
                cfg.collector.total_iters - 1,
            )
        if cfg.logger.eval_async:
            final_async = evaluator.wait(timeout=cfg.logger.eval_timeout_s)
            if final_async is not None:
                log_metrics(logger, final_async, cfg.collector.total_iters)
        final_eval = evaluator.evaluate(weights=None, step=cfg.collector.total_iters)
        log_metrics(logger, final_eval, cfg.collector.total_iters)
        torchrl_logger.info(
            "Final greedy success rate: %.3f", final_eval["eval/success_rate"]
        )
    finally:
        pbar.close()
        evaluator.shutdown()
        collector.shutdown()
        policy_server.shutdown()
        policy_server.transport.close()


if __name__ == "__main__":
    main()
