# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Training loops for async PPO.

Two modes:
  - train_start: fully async (collector.start(), trainer reads from buffer)
  - train_iterate: semi-async (for data in collector, gated on collector)
"""
from __future__ import annotations

import logging
import multiprocessing
import time
from functools import partial

import torch
import tqdm

from torchrl.collectors import Evaluator, MultiaSyncDataCollector
from torchrl.weight_update import SharedMemWeightSyncScheme
from utils_mujoco import (
    ActorWithCritic,
    LearnerPostproc,
    make_env,
    make_eval_env,
    make_ppo_models,
    make_shared_vecnorm_data,
    WorkerGAEPostproc,
)

log = logging.getLogger(__name__)


def _check_finite(tensor, name, step):
    """Return True if *tensor* is all finite, log warning otherwise."""
    if torch.isfinite(tensor).all():
        return True
    n_bad = (~torch.isfinite(tensor)).sum().item()
    log.warning(
        "Non-finite values in %s at step %d: %d/%d elements "
        "(inf=%d, nan=%d) — skipping batch",
        name,
        step,
        n_bad,
        tensor.numel(),
        tensor.isinf().sum().item(),
        tensor.isnan().sum().item(),
    )
    return False


def _make_eval_policy(env, env_name, device):
    """Picklable policy factory for the process-based Evaluator."""
    return make_ppo_models(env_name, device)[0]


def train_start(
    *,
    cfg,
    actor,
    critic,
    adv_module,
    loss_module,
    optim,
    sampler,
    data_buffer,
    advantage_on,
    device,
    collect_device,
    logger,
    eval_device,
    num_eval_envs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    cfg_loss_gamma,
    test_interval,
    total_frames,
    total_network_updates,
):
    """Fully async training: collector.start() fills buffer independently."""
    # Shared VecNormV2 state so collector and evaluator use the same stats
    shared_vecnorm = make_shared_vecnorm_data(cfg.env.env_name)

    # Shared version counter (readable by workers via postproc)
    version_counter = multiprocessing.Value("i", 0)

    # Build collector policy and postproc based on advantage_on mode
    weight_sync_schemes = None
    if advantage_on == "worker":
        # Workers compute GAE via postproc. A dedicated weight sync scheme
        # keeps the worker's adv_module in sync with the trainer's critic.
        collector_policy = actor
        postproc = WorkerGAEPostproc(adv_module, version_counter)
        weight_sync_schemes = {
            "policy": SharedMemWeightSyncScheme(),
            "postproc.adv_module": SharedMemWeightSyncScheme(),
        }
    else:
        collector_policy = actor
        postproc = LearnerPostproc(version_counter)

    create_env_fn = [
        partial(
            make_env,
            cfg.env.env_name,
            collect_device,
            cfg.env.num_envs,
            cfg.env.compile,
            shared_vecnorm=shared_vecnorm,
        )
    ]

    # Collector init compiles the collector env in a subprocess
    collector = MultiaSyncDataCollector(
        create_env_fn=create_env_fn,
        policy=collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=collect_device,
        storing_device=device,
        max_frames_per_traj=-1,
        replay_buffer=data_buffer,
        postproc=postproc,
        local_init_rb=True,
        weight_sync_schemes=weight_sync_schemes,
    )

    # Async evaluator in a separate process (avoids CUDA stream contention)
    # logger=None: we log eval metrics ourselves in the training loop so that
    # eval and training metrics land in the same WandB row.
    evaluator = Evaluator(
        env=partial(
            make_eval_env,
            cfg.env.env_name,
            eval_device,
            num_eval_envs,
            shared_vecnorm=shared_vecnorm,
        ),
        policy_factory=partial(
            _make_eval_policy, env_name=cfg.env.env_name, device=eval_device
        ),
        num_trajectories=num_eval_envs,
        max_steps=1000,
        backend="process",
    )

    # Start collection and evaluation
    collector.start()
    evaluator.trigger_eval(actor, step=0)

    policy_version = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()
    train_start_time = None
    last_write_count = 0
    last_trained_wc = 0
    pending_eval_metrics = None
    # FPS tracking
    last_fps_time = time.time()
    last_fps_wc = 0
    eval_trigger_time = time.time()
    last_eval_wc = 0  # frames at which we last triggered eval

    while True:
        current_wc = data_buffer.write_count
        if current_wc > last_write_count:
            if train_start_time is None:
                train_start_time = time.time()
            pbar.update(current_wc - last_write_count)
            last_write_count = current_wc
        if current_wc >= total_frames:
            break

        # Trigger next eval; capture previous result for logging
        if not evaluator.pending and (current_wc - last_eval_wc >= test_interval):
            prev_eval = evaluator.trigger_eval(actor, step=current_wc)
            if prev_eval is not None:
                pending_eval_metrics = prev_eval
            eval_trigger_time = time.time()
            last_eval_wc = current_wc

        if current_wc <= last_trained_wc or len(data_buffer) < cfg_buffer_min_fill:
            time.sleep(0.05)
            continue
        last_trained_wc = current_wc

        metrics_to_log = {}

        # Log buffer reward stats
        buf_reward = data_buffer[:]["next", "reward"]
        metrics_to_log["buffer/reward_mean"] = buf_reward.mean().item()
        metrics_to_log["buffer/reward_std"] = buf_reward.std().item()

        batch, info = data_buffer.sample(return_info=True)
        batch = batch.to(device)
        batch_staleness = info.get("staleness")

        if advantage_on == "learner":
            with torch.no_grad():
                state_value = critic(batch).get("state_value")
                next_state_value = critic(batch.get("next")).get("state_value")
                reward = batch.get(("next", "reward"))
                done = batch.get(("next", "done")).float()
                value_target = reward + cfg_loss_gamma * (1 - done) * next_state_value
                advantage = value_target - state_value
                batch.set("advantage", advantage)
                batch.set("value_target", value_target)

        # Guard: check inputs to policy are finite — skip corrupted batches
        batch_ok = _check_finite(
            batch["observation"], "batch/observation", current_wc
        ) and _check_finite(
            batch["action_log_prob"], "batch/action_log_prob", current_wc
        )
        adv = batch.get("advantage", None)
        if adv is not None:
            batch_ok = batch_ok and _check_finite(adv, "batch/advantage", current_wc)
        if not batch_ok:
            continue

        alpha = 1.0
        if cfg_optim_anneal_lr:
            alpha = 1 - (current_wc / total_frames)
            for group in optim.param_groups:
                group["lr"] = cfg_optim_lr * alpha
        if cfg_loss_anneal_clip_eps:
            loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

        optim.zero_grad(set_to_none=True)
        loss = loss_module(batch)
        total_loss = loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]

        # Guard: check loss outputs are finite — skip if loss is NaN
        if not _check_finite(total_loss, "loss/total", current_wc):
            continue

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
        )
        optim.step()
        num_network_updates += 1

        # Sync both actor and adv_module (critic) when in worker mode
        if advantage_on == "worker":
            collector.update_policy_weights_(
                weights_dict={"policy": None, "postproc.adv_module": None}
            )
        else:
            collector.update_policy_weights_()
        policy_version += 1
        version_counter.value = policy_version
        if hasattr(sampler, "consumer_version"):
            sampler.consumer_version = policy_version

        # Batch reward stats (if reward is present, e.g. from worker GAE)
        batch_reward = batch.get(("next", "reward"), None)
        if batch_reward is not None:
            metrics_to_log["train/batch_reward_mean"] = batch_reward.mean().item()
            metrics_to_log["train/batch_reward_std"] = batch_reward.std().item()

        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": alpha * cfg_optim_lr.item(),
                "train/clip_epsilon": (alpha * cfg_loss_clip_epsilon)
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
                "train/ESS": loss["ESS"].item(),
                "train/clip_fraction": loss["clip_fraction"].item(),
                "train/kl_approx": loss["kl_approx"].item(),
                "train/max_ratio": loss["max_ratio"].item(),
                "train/mean_ratio": loss["mean_ratio"].item(),
                "staleness/consumer_version": getattr(sampler, "consumer_version", 0),
                "staleness/policy_version": policy_version,
                "staleness/batch_mean": batch_staleness.float().mean().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_max": batch_staleness.max().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_min": batch_staleness.min().item()
                if batch_staleness is not None
                else 0,
                "buffer/size": len(data_buffer),
                "buffer/write_count": current_wc,
                "collector/collected_frames": current_wc,
                "time/train_time": time.time() - train_start_time
                if train_start_time is not None
                else 0.0,
                "time/wall_time": time.time() - start_time,
            }
        )

        # Collector FPS
        now = time.time()
        dt = now - last_fps_time
        if dt > 0:
            metrics_to_log["collector/fps"] = (current_wc - last_fps_wc) / dt
        wall = now - start_time
        if wall > 0:
            metrics_to_log["collector/fps_cumulative"] = current_wc / wall
        last_fps_time = now
        last_fps_wc = current_wc

        # Merge any pending eval metrics into this log step
        if pending_eval_metrics is not None:
            eval_dt = time.time() - eval_trigger_time
            if eval_dt > 0:
                eval_frames = num_eval_envs * 1000
                pending_eval_metrics["eval/fps"] = eval_frames / eval_dt
            metrics_to_log.update(pending_eval_metrics)
            pending_eval_metrics = None
        else:
            # Poll in case result arrived since last trigger_eval
            eval_result = evaluator.poll(0)
            if eval_result is not None:
                eval_dt = time.time() - eval_trigger_time
                if eval_dt > 0:
                    eval_frames = num_eval_envs * 1000
                    eval_result["eval/fps"] = eval_frames / eval_dt
                metrics_to_log.update(eval_result)

        if logger:
            logger.log_metrics(metrics_to_log, current_wc)

    # Log any final eval result
    final_eval = evaluator.wait(timeout=120)
    if final_eval is not None and logger:
        logger.log_metrics(final_eval, last_write_count)

    pbar.close()
    evaluator.shutdown()
    collector.shutdown()

    elapsed = time.time() - start_time
    train_elapsed = time.time() - train_start_time if train_start_time else 0
    print(  # noqa: T001
        f"Training took {train_elapsed:.2f}s ({elapsed:.2f}s wall, "
        f"mode=start, advantage_on={advantage_on})"
    )


def train_iterate(
    *,
    cfg,
    actor,
    critic,
    adv_module,
    loss_module,
    optim,
    sampler,
    data_buffer,
    device,
    collect_device,
    logger,
    eval_device,
    num_eval_envs,
    cfg_loss_ppo_epochs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    test_interval,
    total_frames,
    total_network_updates,
):
    """Semi-async training: for data in collector, gated on collector output."""
    # Shared VecNormV2 state so collector and evaluator use the same stats
    shared_vecnorm = make_shared_vecnorm_data(cfg.env.env_name)

    collector_policy = ActorWithCritic(actor, critic)

    create_env_fn = [
        partial(
            make_env,
            cfg.env.env_name,
            collect_device,
            cfg.env.num_envs,
            cfg.env.compile,
            shared_vecnorm=shared_vecnorm,
        )
    ]

    # Collector init compiles the collector env in a subprocess
    collector = MultiaSyncDataCollector(
        create_env_fn=create_env_fn,
        policy=collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=collect_device,
        storing_device=device,
        max_frames_per_traj=-1,
        update_at_each_batch=True,
        postproc=adv_module,
    )

    # Async evaluator in a separate process (avoids CUDA stream contention)
    # logger=None: we log eval metrics ourselves in the training loop so that
    # eval and training metrics land in the same WandB row.
    evaluator = Evaluator(
        env=partial(
            make_eval_env,
            cfg.env.env_name,
            eval_device,
            num_eval_envs,
            shared_vecnorm=shared_vecnorm,
        ),
        policy_factory=partial(
            _make_eval_policy, env_name=cfg.env.env_name, device=eval_device
        ),
        num_trajectories=num_eval_envs,
        max_steps=1000,
        backend="process",
    )

    # Start continuous eval immediately
    evaluator.trigger_eval(actor, step=0)

    policy_version = 0
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()
    train_start_time = None
    # FPS tracking
    last_fps_time = time.time()
    last_fps_frames = 0
    eval_trigger_time = time.time()
    last_eval_frames = 0  # frames at which we last triggered eval

    for _i, data in enumerate(collector):
        if train_start_time is None:
            train_start_time = time.time()

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],), float(policy_version), device=device
        )
        data_buffer.extend(data_flat)

        if len(data_buffer) < cfg_buffer_min_fill:
            if logger:
                metrics_to_log["buffer/size"] = len(data_buffer)
                logger.log_metrics(metrics_to_log, collected_frames)
            continue

        # Log buffer reward stats
        buf_reward = data_buffer[:]["next", "reward"]
        metrics_to_log["buffer/reward_mean"] = buf_reward.mean().item()
        metrics_to_log["buffer/reward_std"] = buf_reward.std().item()

        for _epoch in range(cfg_loss_ppo_epochs):
            batch, info = data_buffer.sample(return_info=True)
            batch = batch.to(device)
            batch_staleness = info.get("staleness")

            # Guard: check inputs to policy are finite — skip corrupted batches
            batch_ok = _check_finite(
                batch["observation"], "batch/observation", collected_frames
            ) and _check_finite(
                batch["action_log_prob"],
                "batch/action_log_prob",
                collected_frames,
            )
            adv = batch.get("advantage", None)
            if adv is not None:
                batch_ok = batch_ok and _check_finite(
                    adv, "batch/advantage", collected_frames
                )
            if not batch_ok:
                continue

            alpha = 1.0
            if cfg_optim_anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in optim.param_groups:
                    group["lr"] = cfg_optim_lr * alpha
            if cfg_loss_anneal_clip_eps:
                loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

            optim.zero_grad(set_to_none=True)
            loss = loss_module(batch)
            total_loss = (
                loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
            )

            # Guard: check loss outputs are finite — skip if loss is NaN
            if not _check_finite(total_loss, "loss/total", collected_frames):
                continue

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
            )
            optim.step()
            num_network_updates += 1

        collector.update_policy_weights_()
        policy_version += 1
        if hasattr(sampler, "consumer_version"):
            sampler.consumer_version = policy_version

        # Batch reward stats (from last epoch's batch)
        batch_reward = batch.get(("next", "reward"), None)
        if batch_reward is not None:
            metrics_to_log["train/batch_reward_mean"] = batch_reward.mean().item()
            metrics_to_log["train/batch_reward_std"] = batch_reward.std().item()

        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": alpha * cfg_optim_lr.item(),
                "train/clip_epsilon": (alpha * cfg_loss_clip_epsilon)
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
                "train/ESS": loss["ESS"].item(),
                "train/clip_fraction": loss["clip_fraction"].item(),
                "train/kl_approx": loss["kl_approx"].item(),
                "train/max_ratio": loss["max_ratio"].item(),
                "train/mean_ratio": loss["mean_ratio"].item(),
                "staleness/consumer_version": getattr(sampler, "consumer_version", 0),
                "staleness/policy_version": policy_version,
                "staleness/batch_mean": batch_staleness.float().mean().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_max": batch_staleness.max().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_min": batch_staleness.min().item()
                if batch_staleness is not None
                else 0,
                "buffer/size": len(data_buffer),
                "collector/collected_frames": collected_frames,
                "time/train_time": time.time() - train_start_time
                if train_start_time is not None
                else 0.0,
                "time/wall_time": time.time() - start_time,
            }
        )

        # Collector FPS
        now = time.time()
        dt = now - last_fps_time
        if dt > 0:
            metrics_to_log["collector/fps"] = (collected_frames - last_fps_frames) / dt
        wall = now - start_time
        if wall > 0:
            metrics_to_log["collector/fps_cumulative"] = collected_frames / wall
        last_fps_time = now
        last_fps_frames = collected_frames

        # Trigger next eval and merge previous result into this log step
        if not evaluator.pending and (
            collected_frames - last_eval_frames >= test_interval
        ):
            prev_eval = evaluator.trigger_eval(actor, step=collected_frames)
            if prev_eval is not None:
                eval_dt = time.time() - eval_trigger_time
                if eval_dt > 0:
                    eval_frames = num_eval_envs * 1000
                    prev_eval["eval/fps"] = eval_frames / eval_dt
                metrics_to_log.update(prev_eval)
            eval_trigger_time = time.time()
            last_eval_frames = collected_frames
        else:
            # Poll in case result arrived since last trigger_eval
            eval_result = evaluator.poll(0)
            if eval_result is not None:
                eval_dt = time.time() - eval_trigger_time
                if eval_dt > 0:
                    eval_frames = num_eval_envs * 1000
                    eval_result["eval/fps"] = eval_frames / eval_dt
                metrics_to_log.update(eval_result)

        if logger:
            logger.log_metrics(metrics_to_log, collected_frames)

    # Log any final eval result
    final_eval = evaluator.wait(timeout=120)
    if final_eval is not None and logger:
        logger.log_metrics(final_eval, collected_frames)

    pbar.close()
    evaluator.shutdown()
    collector.shutdown()

    train_elapsed = time.time() - train_start_time if train_start_time else 0
    elapsed = time.time() - start_time
    print(  # noqa: T001
        f"Training took {train_elapsed:.2f}s ({elapsed:.2f}s wall, mode=iterate)"
    )
