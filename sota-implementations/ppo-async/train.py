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

import multiprocessing
import time
from functools import partial

import torch
import tqdm

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.weight_update import SharedMemWeightSyncScheme
from utils_mujoco import ActorWithCritic, LearnerPostproc, make_env, WorkerGAEPostproc


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
    evaluator,
    cfg_loss_ppo_epochs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_logger_test_interval,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    cfg_loss_gamma,
    total_frames,
    total_network_updates,
):
    """Fully async training: collector.start() fills buffer independently."""
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

    # Kick off eval env creation + compilation on the evaluator thread so it
    # runs concurrently with collector env compilation below.
    evaluator.trigger_eval(actor, step=0)

    create_env_fn = [
        partial(
            make_env,
            cfg.env.env_name,
            collect_device,
            cfg.env.num_envs,
            cfg.env.compile,
        )
    ]

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

    collector.start()

    policy_version = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()
    train_start_time = None
    last_write_count = 0
    last_test_frames = 0
    last_trained_wc = 0

    while True:
        current_wc = data_buffer.write_count
        if current_wc > last_write_count:
            if train_start_time is None:
                train_start_time = time.time()
            pbar.update(current_wc - last_write_count)
            last_write_count = current_wc
        if current_wc >= total_frames:
            break

        if current_wc <= last_trained_wc or len(data_buffer) < cfg_buffer_min_fill:
            time.sleep(0.05)
            continue
        last_trained_wc = current_wc

        metrics_to_log = {}

        for _epoch in range(cfg_loss_ppo_epochs):
            batch, info = data_buffer.sample(return_info=True)
            batch = batch.to(device)
            batch_staleness = info.get("staleness")

            if advantage_on == "learner":
                with torch.no_grad():
                    state_value = critic(batch).get("state_value")
                    next_state_value = critic(batch.get("next")).get("state_value")
                    reward = batch.get(("next", "reward"))
                    done = batch.get(("next", "done")).float()
                    value_target = (
                        reward + cfg_loss_gamma * (1 - done) * next_state_value
                    )
                    advantage = value_target - state_value
                    batch.set("advantage", advantage)
                    batch.set("value_target", value_target)

            alpha = 1.0
            if cfg_optim_anneal_lr:
                alpha = 1 - (current_wc / total_frames)
                for group in optim.param_groups:
                    group["lr"] = cfg_optim_lr * alpha
            if cfg_loss_anneal_clip_eps:
                loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

            optim.zero_grad(set_to_none=True)
            loss = loss_module(batch)
            total_loss = (
                loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
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

        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
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

        if (current_wc // cfg_logger_test_interval) > (
            last_test_frames // cfg_logger_test_interval
        ):
            if not evaluator.pending:
                evaluator.trigger_eval(actor, step=current_wc)
            last_test_frames = current_wc
        eval_metrics = evaluator.poll()
        if eval_metrics is not None:
            metrics_to_log.update(eval_metrics)

        if logger:
            logger.log_metrics(metrics_to_log, current_wc)

    pbar.close()
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
    evaluator,
    cfg_loss_ppo_epochs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_logger_test_interval,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    total_frames,
    total_network_updates,
):
    """Semi-async training: for data in collector, gated on collector output."""
    # Kick off eval env creation + compilation on the evaluator thread so it
    # runs concurrently with collector env compilation below.
    evaluator.trigger_eval(actor, step=0)

    collector_policy = ActorWithCritic(actor, critic)

    create_env_fn = [
        partial(
            make_env,
            cfg.env.env_name,
            collect_device,
            cfg.env.num_envs,
            cfg.env.compile,
        )
    ]

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

    policy_version = 0
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()
    train_start_time = None

    for i, data in enumerate(collector):
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

        for _epoch in range(cfg_loss_ppo_epochs):
            batch, info = data_buffer.sample(return_info=True)
            batch = batch.to(device)
            batch_staleness = info.get("staleness")

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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
            )
            optim.step()
            num_network_updates += 1

        collector.update_policy_weights_()
        policy_version += 1
        if hasattr(sampler, "consumer_version"):
            sampler.consumer_version = policy_version

        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
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

        if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
            i * frames_in_batch
        ) // cfg_logger_test_interval:
            if not evaluator.pending:
                evaluator.trigger_eval(actor, step=collected_frames)
        eval_metrics = evaluator.poll()
        if eval_metrics is not None:
            metrics_to_log.update(eval_metrics)

        if logger:
            logger.log_metrics(metrics_to_log, collected_frames)

    pbar.close()
    collector.shutdown()

    train_elapsed = time.time() - train_start_time if train_start_time else 0
    elapsed = time.time() - start_time
    print(  # noqa: T001
        f"Training took {train_elapsed:.2f}s ({elapsed:.2f}s wall, mode=iterate)"
    )
