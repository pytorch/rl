# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dreamer v1 training with IsaacLab environments.

CRITICAL: IsaacLab requires AppLauncher to be initialized before importing torch.
This is why this script has a non-standard import order -- the AppLauncher init
MUST happen at the very top, before any torch/torchrl imports.

IsaacLab environments are:
- Pre-vectorized (e.g., 4096 parallel envs in a single GPU simulation)
- GPU-native (always on cuda:0)
- State-based (observations are vectors, not pixels)

This script uses a single synchronous Collector (not MultiCollector) because
IsaacLab's built-in vectorization already provides massive throughput.
"""
from __future__ import annotations

# ============================================================================
# STEP 1: Initialize IsaacLab AppLauncher BEFORE importing torch
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dreamer + IsaacLab Training")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
app_launcher = AppLauncher(args_cli)

# ============================================================================
# STEP 2: Now safe to import torch, torchrl, and everything else
# ============================================================================
import contextlib
import time

import hydra
import torch
import torch.cuda
import tqdm
from dreamer_utils import (
    _default_device,
    _make_env,
    log_metrics,
    make_dreamer,
    make_replay_buffer,
    transform_env,
)
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.autograd.profiler import record_function
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import Collector
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger


@hydra.main(version_base="1.1", config_path="", config_name="config_isaac")
def main(cfg: DictConfig):
    device = _default_device(cfg.networks.device)
    assert device.type == "cuda", "Dreamer + IsaacLab requires CUDA"

    # Create logger
    exp_name = generate_exp_name("Dreamer-Isaac", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="dreamer_isaac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "project": cfg.logger.project,
            },
        )
        if hasattr(logger, "log_hparams"):
            logger.log_hparams(cfg)

    # ========================================================================
    # Environment setup
    # ========================================================================
    # IsaacLab envs are pre-vectorized and GPU-native -- no ParallelEnv needed.
    torchrl_logger.info(f"Creating IsaacLab env: {cfg.env.name}")
    train_env = _make_env(cfg, device=device)
    train_env = transform_env(cfg, train_env)
    train_env.set_seed(cfg.env.seed)
    torchrl_logger.info(
        f"IsaacLab env created: batch_size={train_env.batch_size}, "
        f"obs_spec keys={list(train_env.observation_spec.keys())}, "
        f"action_shape={train_env.action_spec.shape}"
    )

    # ========================================================================
    # Dreamer components (world model, actor, value, model-based env)
    # ========================================================================
    action_key = "action"
    value_key = "state_value"
    (
        world_model,
        model_based_env,
        model_based_env_eval,
        actor_model,
        value_model,
        policy,
    ) = make_dreamer(
        cfg=cfg,
        device=device,
        action_key=action_key,
        value_key=value_key,
        use_decoder_in_env=False,  # No video for Isaac (state-based)
        logger=logger,
        test_env=train_env,
    )

    # ========================================================================
    # Losses
    # ========================================================================
    world_model_loss = DreamerModelLoss(world_model)
    # IsaacLab uses "policy" as observation key (state-based, not pixels)
    world_model_loss.set_keys(pixels="policy", reco_pixels="reco_policy")

    actor_loss = DreamerActorLoss(
        actor_model,
        value_model,
        model_based_env,
        imagination_horizon=cfg.optimization.imagination_horizon,
        discount_loss=True,
    )
    actor_loss.make_value_estimator(
        gamma=cfg.optimization.gamma, lmbda=cfg.optimization.lmbda
    )
    value_loss = DreamerValueLoss(
        value_model, discount_loss=True, gamma=cfg.optimization.gamma
    )

    # ========================================================================
    # Replay buffer
    # ========================================================================
    batch_size = cfg.replay_buffer.batch_size
    batch_length = cfg.replay_buffer.batch_length
    replay_buffer = make_replay_buffer(
        batch_size=batch_size,
        batch_seq_len=batch_length,
        buffer_size=cfg.replay_buffer.buffer_size,
        buffer_scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
        prefetch=cfg.replay_buffer.prefetch,
        pixel_obs=False,
        grayscale=False,
        image_size=64,  # unused for state-based
    )

    # ========================================================================
    # Collector (single, synchronous -- IsaacLab is already vectorized)
    # ========================================================================
    collector = Collector(
        create_env_fn=train_env,
        policy=policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=-1,
        init_random_frames=cfg.collector.init_random_frames,
        storing_device="cpu",
        no_cuda_sync=True,  # Critical for CUDA-native IsaacLab envs
    )
    collector.set_seed(cfg.env.seed)

    # ========================================================================
    # Optimizers
    # ========================================================================
    use_fused = device.type == "cuda"
    world_model_opt = torch.optim.Adam(
        world_model.parameters(), lr=cfg.optimization.world_model_lr, fused=use_fused
    )
    actor_opt = torch.optim.Adam(
        actor_model.parameters(), lr=cfg.optimization.actor_lr, fused=use_fused
    )
    value_opt = torch.optim.Adam(
        value_model.parameters(), lr=cfg.optimization.value_lr, fused=use_fused
    )

    # ========================================================================
    # Mixed precision
    # ========================================================================
    autocast_cfg = cfg.optimization.autocast
    if autocast_cfg in (False, "false", "False"):
        autocast_dtype = None
    elif autocast_cfg in (True, "true", "True", "bfloat16"):
        autocast_dtype = torch.bfloat16
    elif autocast_cfg == "float16":
        autocast_dtype = torch.float16
    else:
        raise ValueError(
            f"Invalid autocast value: {autocast_cfg}. "
            "Use false, true, float16, or bfloat16."
        )

    if autocast_dtype is not None:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
        scaler3 = GradScaler()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ========================================================================
    # torch.compile
    # ========================================================================
    compile_cfg = cfg.optimization.compile
    compile_enabled = compile_cfg.enabled
    compile_losses = set(compile_cfg.losses)
    compile_warmup = 0
    if compile_enabled:
        torch._dynamo.config.capture_scalar_outputs = True
        compile_warmup = 3
        torchrl_logger.info(f"Compiling loss modules with warmup={compile_warmup}")
        backend = compile_cfg.backend
        compile_options = {"triton.cudagraphs": compile_cfg.cudagraphs}

        if "world_model" in compile_losses:
            world_model_loss = compile_with_warmup(
                world_model_loss,
                backend=backend,
                fullgraph=False,
                warmup=compile_warmup,
                options=compile_options,
            )
        if "actor" in compile_losses:
            actor_loss = compile_with_warmup(
                actor_loss,
                backend=backend,
                fullgraph=False,
                warmup=compile_warmup,
                options=compile_options,
            )
        if "value" in compile_losses:
            value_loss = compile_with_warmup(
                value_loss,
                backend=backend,
                fullgraph=False,
                warmup=compile_warmup,
                options=compile_options,
            )

    # ========================================================================
    # Training config
    # ========================================================================
    total_optim_steps = cfg.optimization.total_optim_steps
    log_every = cfg.optimization.log_every
    grad_clip = cfg.optimization.grad_clip
    optim_steps_per_collect = cfg.collector.optim_steps_per_collect

    pbar = tqdm.tqdm(total=total_optim_steps, desc="Optim steps")
    t_log_start = time.time()
    frames_at_log_start = 0

    # ========================================================================
    # Main training loop: synchronous collect-then-train
    # ========================================================================
    optim_step = 0
    collected_frames = 0

    torchrl_logger.info(
        f"Starting training: {total_optim_steps} optim steps, "
        f"{optim_steps_per_collect} optim steps per collection, "
        f"frames_per_batch={cfg.collector.frames_per_batch}, "
        f"init_random_frames={cfg.collector.init_random_frames}"
    )

    for data in collector:
        data_frames = data.numel()
        collected_frames += data_frames

        # Extend replay buffer with collected data (already on CPU via storing_device)
        with timeit("train/extend"):
            replay_buffer.extend(data)

        # Track episode rewards from completed episodes
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            episode_rewards = data["next", "episode_reward"][done_mask]
            mean_episode_reward = episode_rewards.mean().item()
            torchrl_logger.info(
                f"Episodes completed: {done_mask.sum().item()}, "
                f"mean_reward={mean_episode_reward:.2f}"
            )

        # Wait for enough data before training
        if collected_frames < cfg.collector.init_random_frames:
            torchrl_logger.info(
                f"Random collection: {collected_frames}/{cfg.collector.init_random_frames} frames"
            )
            continue

        # ================================================================
        # Training: multiple optim steps per collection step
        # ================================================================
        for _ in range(optim_steps_per_collect):
            if optim_step >= total_optim_steps:
                break

            pbar.update(1)

            # Sample from replay buffer
            with timeit("train/sample"), record_function("## train/sample ##"):
                sampled_tensordict = replay_buffer.sample()
                # Flatten env batch dims (if any) and reshape to (num_slices, batch_length)
                sampled_tensordict = sampled_tensordict.reshape(-1, batch_length)

            # --- World model update ---
            with timeit("train/world_model-forward"), record_function(
                "## world_model/forward ##"
            ):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    model_loss_td, sampled_tensordict = world_model_loss(
                        sampled_tensordict
                    )
                    loss_world_model = (
                        model_loss_td["loss_model_kl"]
                        + model_loss_td["loss_model_reco"]
                        + model_loss_td["loss_model_reward"]
                    )

            with timeit("train/world_model-backward"), record_function(
                "## world_model/backward ##"
            ):
                world_model_opt.zero_grad()
                if autocast_dtype:
                    scaler1.scale(loss_world_model).backward()
                    scaler1.unscale_(world_model_opt)
                else:
                    loss_world_model.backward()
                world_model_grad = clip_grad_norm_(world_model.parameters(), grad_clip)
                if autocast_dtype:
                    scaler1.step(world_model_opt)
                    scaler1.update()
                else:
                    world_model_opt.step()

            # --- Actor update ---
            with timeit("train/actor-forward"), record_function("## actor/forward ##"):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    actor_loss_td, sampled_tensordict = actor_loss(
                        sampled_tensordict.reshape(-1)
                    )

            with timeit("train/actor-backward"), record_function(
                "## actor/backward ##"
            ):
                actor_opt.zero_grad()
                if autocast_dtype:
                    scaler2.scale(actor_loss_td["loss_actor"]).backward()
                    scaler2.unscale_(actor_opt)
                else:
                    actor_loss_td["loss_actor"].backward()
                actor_model_grad = clip_grad_norm_(actor_model.parameters(), grad_clip)
                if autocast_dtype:
                    scaler2.step(actor_opt)
                    scaler2.update()
                else:
                    actor_opt.step()

            # --- Value update ---
            with timeit("train/value-forward"), record_function("## value/forward ##"):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)

            with timeit("train/value-backward"), record_function(
                "## value/backward ##"
            ):
                value_opt.zero_grad()
                if autocast_dtype:
                    scaler3.scale(value_loss_td["loss_value"]).backward()
                    scaler3.unscale_(value_opt)
                else:
                    value_loss_td["loss_value"].backward()
                critic_model_grad = clip_grad_norm_(value_model.parameters(), grad_clip)
                if autocast_dtype:
                    scaler3.step(value_opt)
                    scaler3.update()
                else:
                    value_opt.step()

            optim_step += 1

            # ============================================================
            # Logging
            # ============================================================
            if optim_step % log_every == 0:
                t_log_end = time.time()
                log_interval_time = t_log_end - t_log_start
                frames_this_interval = collected_frames - frames_at_log_start

                fps = (
                    frames_this_interval / log_interval_time
                    if log_interval_time > 0
                    else 0
                )
                ops = log_every / log_interval_time if log_interval_time > 0 else 0
                opf = optim_step / collected_frames if collected_frames > 0 else 0

                pbar.set_postfix(
                    fps=f"{fps:.1f}",
                    ops=f"{ops:.1f}",
                    opf=f"{opf:.2f}",
                    frames=collected_frames,
                )

                sampled_reward = sampled_tensordict.get(("next", "reward"))
                reward_mean = sampled_reward.mean().item()
                reward_std = sampled_reward.std().item()

                metrics = {
                    "loss_model_kl": model_loss_td["loss_model_kl"].item(),
                    "loss_model_reco": model_loss_td["loss_model_reco"].item(),
                    "loss_model_reward": model_loss_td["loss_model_reward"].item(),
                    "loss_actor": actor_loss_td["loss_actor"].item(),
                    "loss_value": value_loss_td["loss_value"].item(),
                    "world_model_grad": world_model_grad,
                    "actor_model_grad": actor_model_grad,
                    "critic_model_grad": critic_model_grad,
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "throughput/fps": fps,
                    "throughput/ops": ops,
                    "throughput/opf": opf,
                    "collected_frames": collected_frames,
                    **timeit.todict(prefix="time"),
                }

                if logger is not None:
                    log_metrics(logger, metrics, collected_frames)

                t_log_start = time.time()
                frames_at_log_start = collected_frames

        # Update policy weights in collector after training
        policy[1].step(data_frames)
        collector.update_policy_weights_()
        torchrl_logger.debug(
            f"Policy weights updated after {optim_steps_per_collect} optim steps"
        )

        if optim_step >= total_optim_steps:
            break

    # ========================================================================
    # Cleanup
    # ========================================================================
    pbar.close()
    if not train_env.is_closed:
        train_env.close()
    collector.shutdown()


if __name__ == "__main__":
    main()
