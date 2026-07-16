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
from functools import partial

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
    make_inference_policy,
    make_logger,
    make_loss_module,
    make_optimizer,
    make_policy,
    make_replay_buffer,
    make_video_evaluator,
    replay_ready_targets,
    reset_collection_state,
    save_checkpoint,
    sync_policy_server,
    update,
    video_eval_episodes,
    wait_for_replay,
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensordict")


@hydra.main(config_path="config", config_name="vla_grpo_libero", version_base="1.1")
def main(cfg):  # noqa: F821
    torchrl_logger.info("Initializing VLA-GRPO with seed %d.", cfg.env.seed)
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
    torchrl_logger.info(
        "Resolved devices: learner=%s rollout=%s evaluation=%s replay=%s.",
        train_device,
        rollout_device,
        eval_device,
        buffer_device,
    )
    torchrl_logger.info(
        "Initializing experiment logger: backend=%s mode=%s.",
        cfg.logger.backend,
        cfg.logger.mode,
    )
    logger = make_logger(cfg)
    torchrl_logger.info("Experiment logger is ready.")
    torchrl_logger.info("Building environment-side action tokenizer.")
    tokenizer = make_action_tokenizer(cfg)
    if cfg.policy.backend == "openvla":
        torchrl_logger.info(
            "Loading learner policy %s on %s.",
            cfg.policy.checkpoint,
            train_device,
        )
    else:
        torchrl_logger.info(
            "Building learner policy backend=%s on %s.",
            cfg.policy.backend,
            train_device,
        )
    learner_policy = make_policy(cfg, train_device)
    torchrl_logger.info("Learner policy is ready; building replay and PPO state.")
    replay_buffer, advantage_transform = make_replay_buffer(cfg, buffer_device)
    loss_module = make_loss_module(cfg, learner_policy)
    optim, scheduler = make_optimizer(cfg, loss_module)
    rollout_policy_factory = partial(
        make_inference_policy,
        cfg,
        rollout_device,
        policy_micro_batch_size=cfg.collector.policy_micro_batch_size,
    )
    total_rollout_envs = int(cfg.collector.num_collectors) * int(
        cfg.collector.envs_per_collector
    )
    torchrl_logger.info(
        "Starting inference server and collectors: collectors=%d "
        "envs_per_collector=%d total_envs=%d rollout_device=%s "
        "policy_micro_batch_size=%s.",
        cfg.collector.num_collectors,
        cfg.collector.envs_per_collector,
        total_rollout_envs,
        rollout_device,
        cfg.collector.policy_micro_batch_size,
    )
    collector, policy_server, eval_policy = make_collector(
        cfg,
        rollout_device,
        policy_factory=rollout_policy_factory,
        tokenizer=tokenizer,
        replay_buffer=replay_buffer,
    )
    torchrl_logger.info(
        "Inference server and collectors are ready at policy version %d.",
        policy_server.policy_version,
    )
    torchrl_logger.info(
        "Building evaluator: episodes=%d envs=%d backend=%s device=%s.",
        cfg.logger.eval_episodes,
        cfg.env.eval_num_envs,
        cfg.logger.eval_backend,
        eval_device,
    )
    evaluator = make_evaluator(
        cfg,
        tokenizer,
        eval_policy,
        logger,
        eval_device,
    )
    video_evaluator = make_video_evaluator(
        cfg,
        tokenizer,
        eval_policy,
        logger,
        eval_device,
    )
    if video_evaluator is None:
        torchrl_logger.info("Evaluator is ready; eval video recording is disabled.")
    else:
        torchrl_logger.info(
            "Evaluator is ready; video evaluator will record %d episode(s) "
            "with %s parallel env(s) per scheduled eval.",
            video_eval_episodes(cfg),
            cfg.logger.video_num_envs,
        )

    start_iter = 0
    resumed_episodes = None
    if cfg.checkpoint.resume:
        torchrl_logger.info("Loading checkpoint from %s.", cfg.checkpoint.resume)
        start_iter, resumed_episodes = load_checkpoint(
            cfg.checkpoint.resume,
            learner_policy,
            optim,
            scheduler,
            train_device,
        )
        torchrl_logger.info(
            "Resumed from %s at iteration %d.", cfg.checkpoint.resume, start_iter
        )

    target_replay_groups, target_replay_decisions = replay_ready_targets(cfg)
    episodes_per_iter = cfg.collector.groups_per_iter * candidate_group_size(cfg)
    total_episodes = (
        start_iter * episodes_per_iter if resumed_episodes is None else resumed_episodes
    )
    trajectory_budget = cfg.collector.get("total_trajectories", None)
    if trajectory_budget is not None:
        trajectory_budget = int(trajectory_budget)
    weight_sync_trainable_only = bool(cfg.train.weight_sync_trainable_only)
    torchrl_logger.info(
        "Publishing initial learner weights to the inference server "
        "(trainable_only=%s).",
        weight_sync_trainable_only,
    )
    sync_policy_server(
        policy_server,
        learner_policy,
        trainable_only=weight_sync_trainable_only,
    )
    torchrl_logger.info(
        "Initial weights published; inference policy version is %d.",
        policy_server.policy_version,
    )
    latest_eval_metrics = {}
    last_eval_step = None
    last_video_step = None

    def record_eval_video(step: int) -> None:
        nonlocal last_video_step
        if video_evaluator is None or last_video_step == step:
            return
        torchrl_logger.info(
            "Recording eval video for step %d over %d episode(s).",
            step,
            video_eval_episodes(cfg),
        )
        video_metrics = video_evaluator.evaluate(weights=None, step=step)
        last_video_step = step
        torchrl_logger.info(
            "Eval video for step %d complete: success_rate=%.3f.",
            step,
            video_metrics["eval/success_rate"],
        )

    if start_iter == 0 and bool(cfg.logger.eval_before_train):
        if cfg.logger.eval_async:
            torchrl_logger.info(
                "Submitting asynchronous frozen-policy baseline evaluation over "
                "%d episodes.",
                cfg.logger.eval_episodes,
            )
            evaluator.trigger_eval(weights=None, step=0)
            record_eval_video(0)
        else:
            torchrl_logger.info(
                "Starting frozen-policy baseline evaluation over %d episodes.",
                cfg.logger.eval_episodes,
            )
            baseline_metrics = evaluator.evaluate(weights=None, step=0)
            log_metrics(logger, baseline_metrics, 0)
            latest_eval_metrics = baseline_metrics
            last_eval_step = 0
            torchrl_logger.info(
                "Frozen-policy success rate: %.3f",
                baseline_metrics["eval/success_rate"],
            )
            record_eval_video(0)
    torchrl_logger.info("Starting asynchronous rollout collection.")
    collector.start()
    torchrl_logger.info("Rollout collectors are running.")

    pbar = tqdm.tqdm(total=cfg.collector.total_iters, initial=start_iter)
    completed_updates = start_iter
    try:
        for iteration in range(start_iter, cfg.collector.total_iters):
            if trajectory_budget is not None and total_episodes >= trajectory_budget:
                torchrl_logger.info(
                    "Reached completed-trajectory budget: %d/%d.",
                    total_episodes,
                    trajectory_budget,
                )
                break
            update_step = iteration + 1
            torchrl_logger.info(
                "Iteration %d/%d: collecting for at least %d useful groups "
                "and %d decisions with policy version %d.",
                update_step,
                cfg.collector.total_iters,
                target_replay_groups,
                target_replay_decisions,
                policy_server.policy_version,
            )
            with timeit("collect"):
                remaining_trajectories = (
                    None
                    if trajectory_budget is None
                    else trajectory_budget - total_episodes
                )
                collect_metrics = wait_for_replay(
                    replay_buffer,
                    advantage_transform,
                    collector,
                    min_replay_groups=target_replay_groups,
                    min_replay_decisions=target_replay_decisions,
                    poll_interval_s=cfg.collector.replay_wait_s,
                    log_interval_s=cfg.collector.replay_log_s,
                    iteration=update_step,
                    max_completed_trajectories=remaining_trajectories,
                )
            torchrl_logger.info(
                "Iteration %d collection ready: kept_groups=%d decisions=%d "
                "completed_trajectories=%d wait_s=%.1f.",
                update_step,
                collect_metrics["buffer/kept_groups_before_update"],
                collect_metrics["buffer/decisions_before_update"],
                collect_metrics["collector/completed_trajectories_before_update"],
                collect_metrics["buffer/wait_s"],
            )

            if (
                collect_metrics["buffer/trajectory_budget_reached"]
                and collect_metrics["buffer/kept_groups_before_update"]
                < target_replay_groups
            ):
                with collector.pause():
                    group_metrics = advantage_metrics(advantage_transform, collector)
                    reset_collection_state(advantage_transform, collector)
                total_episodes += int(group_metrics["collector/completed_trajectories"])
                torchrl_logger.info(
                    "Reached completed-trajectory budget without another full "
                    "useful policy batch: %d/%d.",
                    total_episodes,
                    trajectory_budget,
                )
                timeit.erase()
                break

            with collector.pause():
                num_decisions = len(replay_buffer)
                behavior_policy_version = policy_server.policy_version
                torchrl_logger.info(
                    "Iteration %d: collectors paused; starting PPO over %d "
                    "decisions from behavior policy version %d.",
                    update_step,
                    num_decisions,
                    behavior_policy_version,
                )
                train_metrics = update(
                    replay_buffer,
                    loss_module,
                    optim,
                    scheduler,
                    cfg,
                    train_device,
                    logger=logger,
                    iteration=update_step,
                    current_policy_version=behavior_policy_version,
                )
                torchrl_logger.info(
                    "Iteration %d PPO complete: loss=%.6f grad_norm=%.6f "
                    "optimizer_steps=%d max_staleness=%s.",
                    update_step,
                    train_metrics["train/loss_objective"],
                    train_metrics["train/grad_norm"],
                    train_metrics["train/optim_steps"],
                    train_metrics.get("train/policy_staleness_max", "unavailable"),
                )
                group_metrics = advantage_metrics(advantage_transform, collector)
                reset_collection_state(advantage_transform, collector)
                torchrl_logger.info(
                    "Iteration %d: publishing updated learner weights.", update_step
                )
                sync_policy_server(
                    policy_server,
                    learner_policy,
                    trainable_only=weight_sync_trainable_only,
                )
                torchrl_logger.info(
                    "Iteration %d: inference policy advanced to version %d; "
                    "collectors will resume after the update boundary.",
                    update_step,
                    policy_server.policy_version,
                )

            eval_metrics = {}
            if update_step % cfg.logger.eval_iter == 0:
                record_eval_video(update_step)
                if cfg.logger.eval_async:
                    torchrl_logger.info(
                        "Iteration %d: submitting asynchronous evaluation over "
                        "%d episodes.",
                        update_step,
                        cfg.logger.eval_episodes,
                    )
                    evaluator.trigger_eval(weights=None, step=update_step)
                else:
                    torchrl_logger.info(
                        "Iteration %d: starting synchronous evaluation over %d "
                        "episodes.",
                        update_step,
                        cfg.logger.eval_episodes,
                    )
                    eval_metrics = evaluator.evaluate(weights=None, step=update_step)
                    latest_eval_metrics = eval_metrics
                    last_eval_step = update_step
                    torchrl_logger.info(
                        "Iteration %d evaluation complete: success_rate=%.3f.",
                        update_step,
                        eval_metrics["eval/success_rate"],
                    )
            if cfg.logger.eval_async:
                ready_eval = evaluator.poll(timeout=0)
                if ready_eval is not None:
                    eval_metrics = ready_eval
                    latest_eval_metrics = ready_eval
                    last_eval_step = int(ready_eval["eval/step"])
                    if last_eval_step != update_step:
                        log_metrics(logger, ready_eval, last_eval_step)
                    torchrl_logger.info(
                        "Asynchronous evaluation for iteration %d complete: "
                        "success_rate=%.3f.",
                        last_eval_step,
                        ready_eval["eval/success_rate"],
                    )

            timings = timeit.todict(prefix="time")
            timeit.erase()
            completed_trajectories = int(
                group_metrics["collector/completed_trajectories"]
            )
            total_episodes += completed_trajectories
            completed_updates = update_step
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
            log_metrics(logger, metrics, update_step)
            log_iteration_summary(
                update_step,
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
                checkpoint_path = os.path.join(os.getcwd(), "checkpoint_latest")
                torchrl_logger.info(
                    "Iteration %d: saving checkpoint to %s.",
                    update_step,
                    checkpoint_path,
                )
                save_checkpoint(
                    checkpoint_path,
                    learner_policy,
                    optim,
                    scheduler,
                    iteration,
                    total_episodes=total_episodes,
                )
                torchrl_logger.info("Iteration %d checkpoint saved.", update_step)

        if cfg.checkpoint.save_iter:
            checkpoint_path = os.path.join(os.getcwd(), "checkpoint_latest")
            torchrl_logger.info("Saving final checkpoint to %s.", checkpoint_path)
            save_checkpoint(
                checkpoint_path,
                learner_policy,
                optim,
                scheduler,
                completed_updates - 1,
                total_episodes=total_episodes,
            )
        if cfg.logger.eval_async and evaluator.pending:
            torchrl_logger.info("Waiting for the pending asynchronous evaluation.")
            final_async = evaluator.wait(timeout=cfg.logger.eval_timeout_s)
            if final_async is not None:
                last_eval_step = int(final_async["eval/step"])
                latest_eval_metrics = final_async
                log_metrics(logger, final_async, last_eval_step)
        record_eval_video(completed_updates)
        if last_eval_step == completed_updates:
            final_eval = latest_eval_metrics
        else:
            torchrl_logger.info(
                "Starting final evaluation at iteration %d over %d episodes.",
                completed_updates,
                cfg.logger.eval_episodes,
            )
            final_eval = evaluator.evaluate(weights=None, step=completed_updates)
            log_metrics(logger, final_eval, completed_updates)
        torchrl_logger.info(
            "Final greedy success rate: %.3f", final_eval["eval/success_rate"]
        )
    finally:
        torchrl_logger.info("Shutting down VLA-GRPO services.")
        pbar.close()
        try:
            evaluator.shutdown()
        finally:
            try:
                if video_evaluator is not None:
                    video_evaluator.shutdown()
            finally:
                try:
                    collector.shutdown()
                finally:
                    try:
                        policy_server.shutdown()
                    finally:
                        try:
                            policy_server.transport.close()
                        finally:
                            if logger is not None:
                                logger.shutdown()
        torchrl_logger.info("VLA-GRPO shutdown complete.")


if __name__ == "__main__":
    main()
