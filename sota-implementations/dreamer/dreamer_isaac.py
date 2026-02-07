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

GPU strategy:
- GPU 0 ("sim_device"): IsaacLab simulation + collection policy inference
- GPU 1 ("train_device"): Model training (world model, actor, value gradients)
- Collection and training alternate synchronously
- Falls back to single-GPU if only 1 GPU is available
"""
from __future__ import annotations

# ============================================================================
# STEP 1: Initialize IsaacLab AppLauncher BEFORE importing torch
# ============================================================================
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dreamer + IsaacLab Training")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args(["--headless"])
app_launcher = AppLauncher(args_cli)

# ============================================================================
# STEP 2: Now safe to import torch, torchrl, and everything else
# ============================================================================
import contextlib
import copy
import math
import time

import hydra
import torch
import torch.cuda
import tqdm
from dreamer_utils import (
    _make_env,
    log_metrics,
    make_dreamer,
    make_eval_policy_factory,
    make_isaac_eval_env_factory,
    make_isaac_init_fn,
    make_replay_buffer,
    transform_env,
)
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.amp import GradScaler
from torch.autograd.profiler import record_function
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors import Collector
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger


def _td_stats(td, key):
    """Return a compact stats string for a tensor in a TensorDict."""
    try:
        t = td[key]
    except KeyError:
        return f"{key}: MISSING"
    has_nan = t.isnan().any().item()
    has_inf = t.isinf().any().item()
    return (
        f"{key}: shape={list(t.shape)} dtype={t.dtype} device={t.device} "
        f"min={t.min().item():.4g} max={t.max().item():.4g} "
        f"mean={t.float().mean().item():.4g} std={t.float().std().item():.4g} "
        f"nan={has_nan} inf={has_inf}"
    )


def _param_stats(name, module):
    """Return stats for module parameters."""
    lines = []
    for pname, p in module.named_parameters():
        has_nan = p.isnan().any().item()
        has_inf = p.isinf().any().item()
        grad_info = "no_grad"
        if p.grad is not None:
            grad_nan = p.grad.isnan().any().item()
            grad_inf = p.grad.isinf().any().item()
            grad_info = (
                f"grad_min={p.grad.min().item():.4g} "
                f"grad_max={p.grad.max().item():.4g} "
                f"grad_nan={grad_nan} grad_inf={grad_inf}"
            )
        lines.append(
            f"  {name}.{pname}: min={p.min().item():.4g} max={p.max().item():.4g} "
            f"nan={has_nan} inf={has_inf} {grad_info}"
        )
    return "\n".join(lines)


def _check_nan_grads(module, name, optim_step):
    """Check for NaN in gradients. Abort if found."""
    for pname, p in module.named_parameters():
        if p.grad is not None and p.grad.isnan().any():
            torchrl_logger.error(
                f"NaN gradient detected at optim_step={optim_step} "
                f"in {name}.{pname} (shape={list(p.grad.shape)})\n"
                f"Param stats: min={p.min().item():.4g} max={p.max().item():.4g} "
                f"nan={p.isnan().any().item()}\n"
                f"Full param diagnostics:\n{_param_stats(name, module)}"
            )
            torchrl_logger.error("ABORTING: NaN gradients are not recoverable.")
            sys.exit(1)


@hydra.main(version_base="1.1", config_path="", config_name="config_isaac")
def main(cfg: DictConfig):
    # Force DEBUG logging level
    import logging

    torchrl_logger.setLevel(logging.DEBUG)
    for handler in torchrl_logger.handlers:
        handler.setLevel(logging.DEBUG)

    # ========================================================================
    # Device setup: sim on cuda:0, training on cuda:1 (or cuda:0 if single GPU)
    # ========================================================================
    sim_device = torch.device("cuda:0")  # IsaacLab always binds to cuda:0
    num_gpus = torch.cuda.device_count()
    train_device = torch.device("cuda:1") if num_gpus > 1 else sim_device

    torchrl_logger.info(
        f"GPU setup: {num_gpus} GPUs available, "
        f"sim_device={sim_device}, train_device={train_device}"
    )

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
    # Environment setup (on sim_device = cuda:0)
    # ========================================================================
    torchrl_logger.info(f"Creating IsaacLab env: {cfg.env.name}")
    train_env = _make_env(cfg, device=sim_device)
    train_env = transform_env(cfg, train_env)
    train_env.set_seed(cfg.env.seed)

    # Normalize observations to ~N(0,1).  Without this, the raw IsaacLab
    # observations (std ~1.7, range [-30, 30]) produce an initial reco loss
    # of ~150 000 whose gradients are clipped to ~100 (vs. norms of ~700 000),
    # effectively reducing the world-model learning rate by ~7000x.
    from torchrl.envs import ObservationNorm

    obs_norm = ObservationNorm(
        in_keys=["policy"],
        standard_normal=True,
    )
    train_env.append_transform(obs_norm)
    torchrl_logger.info("Initialising ObservationNorm stats (1 rollout)...")
    obs_norm.init_stats(num_iter=1, reduce_dim=0, cat_dim=0)
    torchrl_logger.info(
        f"ObservationNorm: loc range=[{obs_norm.loc.min():.2f}, {obs_norm.loc.max():.2f}], "
        f"scale range=[{obs_norm.scale.min():.2f}, {obs_norm.scale.max():.2f}]"
    )

    torchrl_logger.info(
        f"IsaacLab env created: batch_size={train_env.batch_size}, "
        f"obs_spec keys={list(train_env.observation_spec.keys())}, "
        f"action_shape={train_env.action_spec.shape}"
    )

    # ========================================================================
    # Dreamer components (on train_device for gradient computation)
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
        device=train_device,
        action_key=action_key,
        value_key=value_key,
        use_decoder_in_env=False,  # No video for Isaac (state-based)
        logger=logger,
        test_env=train_env,  # env on sim_device; make_dreamer transfers init data
    )

    # ========================================================================
    # Collector policy: deep copy on sim_device for collection
    # ========================================================================
    collector_policy = copy.deepcopy(policy)
    if train_device != sim_device:
        collector_policy = collector_policy.to(sim_device)
        torchrl_logger.info(
            f"Created collector policy copy on {sim_device} "
            f"(training models on {train_device})"
        )

    # ========================================================================
    # Async eval worker (on a dedicated GPU via Ray)
    # ========================================================================
    eval_worker = None
    if cfg.logger.video and num_gpus >= 3:
        import ray

        from torchrl.eval import RayEvalWorker

        ray.init(num_gpus=num_gpus - 2)
        eval_worker = RayEvalWorker(
            init_fn=make_isaac_init_fn(),
            env_maker=make_isaac_eval_env_factory(
                cfg,
                obs_norm_loc=obs_norm.loc.cpu(),
                obs_norm_scale=obs_norm.scale.cpu(),
            ),
            policy_maker=make_eval_policy_factory(cfg),
            num_gpus=1,
        )
        torchrl_logger.info(
            f"Eval worker created: eval_every={cfg.logger.eval_every}, "
            f"eval_rollout_steps={cfg.logger.eval_rollout_steps}, "
            f"eval_num_envs={cfg.logger.eval_num_envs}"
        )
    elif cfg.logger.video:
        torchrl_logger.warning(
            f"Video rendering requested but only {num_gpus} GPUs available. "
            "Need >= 3 GPUs (sim + train + eval). Skipping eval worker."
        )

    # ========================================================================
    # Losses (on train_device)
    # ========================================================================
    # global_average=True: for state-based observations the tensor is 3D
    # [B, T, F], and the default sum((-3,-2,-1)) accidentally sums over
    # batch and time too, inflating the loss ~40000x and rendering the
    # gradient clipping far too aggressive.
    world_model_loss = DreamerModelLoss(world_model, global_average=True)
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
    # Replay buffer (always CPU storage for thread safety)
    # ========================================================================
    batch_size = cfg.replay_buffer.batch_size
    batch_length = cfg.replay_buffer.batch_length
    replay_buffer = make_replay_buffer(
        batch_size=batch_size,
        batch_seq_len=batch_length,
        buffer_size=cfg.replay_buffer.buffer_size,
        buffer_scratch_dir=cfg.replay_buffer.scratch_dir,
        device=train_device,
        prefetch=cfg.replay_buffer.prefetch,
        pixel_obs=False,
        grayscale=False,
        image_size=64,  # unused for state-based
        gpu_storage=False,
    )
    torchrl_logger.info(
        f"Replay buffer: batch_size={batch_size}, batch_length={batch_length}, "
        f"device={train_device}"
    )

    # ========================================================================
    # Collector (on sim_device, uses collector_policy)
    # ========================================================================
    collector = Collector(
        create_env_fn=train_env,
        policy=collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=-1,
        init_random_frames=cfg.collector.init_random_frames,
        storing_device="cpu",
        no_cuda_sync=True,  # Critical for CUDA-native IsaacLab envs
    )
    collector.set_seed(cfg.env.seed)

    # ========================================================================
    # Optimizers (on train_device)
    # ========================================================================
    use_fused = train_device.type == "cuda"
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

    # GradScaler is only needed for float16 (limited dynamic range).
    # bfloat16 has the same exponent range as float32, so no scaling needed.
    use_scaler = autocast_dtype == torch.float16
    if use_scaler:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
        scaler3 = GradScaler()

    if train_device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ========================================================================
    # Training config
    # ========================================================================
    total_optim_steps = cfg.optimization.total_optim_steps
    log_every = cfg.optimization.log_every
    grad_clip = cfg.optimization.grad_clip
    optim_steps_per_collect = cfg.collector.optim_steps_per_collect

    eval_every = cfg.logger.eval_every
    next_eval_step = eval_every

    pbar = tqdm.tqdm(total=total_optim_steps, desc="Optim steps")
    t_log_start = time.time()
    frames_at_log_start = 0
    collected_frames = 0
    optim_step = 0
    last_mean_reward = float("nan")

    torchrl_logger.info(
        f"Starting synchronous training: {total_optim_steps} optim steps, "
        f"optim_steps_per_collect={optim_steps_per_collect}, "
        f"frames_per_batch={cfg.collector.frames_per_batch}, "
        f"init_random_frames={cfg.collector.init_random_frames}"
    )

    # ========================================================================
    # Main training loop: synchronous collect -> train
    # ========================================================================
    for i_collect, data in enumerate(collector):
        # ---- Extend replay buffer ----
        with timeit("collect/extend"):
            replay_buffer.extend(data)
        collected_frames += data.numel()

        # DEBUG: log collected data stats
        torchrl_logger.debug(
            f"[collect #{i_collect}] frames={data.numel()}, "
            f"total_collected={collected_frames}\n"
            f"  {_td_stats(data, 'policy')}\n"
            f"  {_td_stats(data, 'action')}\n"
            f"  {_td_stats(data, ('next', 'reward'))}\n"
            f"  {_td_stats(data, ('next', 'done'))}"
        )

        # Track episode rewards from completed episodes
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            episode_rewards = data["next", "episode_reward"][done_mask]
            last_mean_reward = episode_rewards.mean().item()

        # Skip training on first batch (random exploration data)
        if i_collect == 0:
            torchrl_logger.info(
                f"Init data collected: {collected_frames} frames. "
                f"Starting training on {train_device}."
            )
            continue

        # ---- Train for optim_steps_per_collect steps ----
        for _j in range(optim_steps_per_collect):
            if optim_step >= total_optim_steps:
                break
            pbar.update(1)

            # Sample from replay buffer
            with timeit("train/sample"), record_function("## train/sample ##"):
                sampled_tensordict = replay_buffer.sample()
                # With strict_length=False, the sample numel may not be
                # exactly divisible by batch_length. Truncate to make it so.
                numel = sampled_tensordict.numel()
                usable = (numel // batch_length) * batch_length
                if usable < numel:
                    sampled_tensordict = sampled_tensordict[:usable]
                sampled_tensordict = sampled_tensordict.reshape(-1, batch_length)

            # DEBUG: log sampled data stats
            torchrl_logger.debug(
                f"[optim_step={optim_step}] Sampled batch "
                f"shape={list(sampled_tensordict.shape)}\n"
                f"  {_td_stats(sampled_tensordict, 'policy')}\n"
                f"  {_td_stats(sampled_tensordict, 'action')}\n"
                f"  {_td_stats(sampled_tensordict, ('next', 'policy'))}\n"
                f"  {_td_stats(sampled_tensordict, ('next', 'reward'))}\n"
                f"  {_td_stats(sampled_tensordict, 'state')}\n"
                f"  {_td_stats(sampled_tensordict, 'belief')}"
            )

            # --- World model update ---
            with timeit("train/world_model-forward"), record_function(
                "## world_model/forward ##"
            ):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=train_device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    model_loss_td, sampled_tensordict = world_model_loss(
                        sampled_tensordict
                    )
                    loss_world_model = (
                        model_loss_td["loss_model_kl"]
                        + model_loss_td["loss_model_reco"]
                        + model_loss_td["loss_model_reward"]
                    )

            # DEBUG: log world model losses
            kl_val = model_loss_td["loss_model_kl"].item()
            reco_val = model_loss_td["loss_model_reco"].item()
            reward_val = model_loss_td["loss_model_reward"].item()
            total_val = loss_world_model.item()
            torchrl_logger.debug(
                f"[optim_step={optim_step}] World model loss: "
                f"kl={kl_val:.4g} reco={reco_val:.4g} reward={reward_val:.4g} "
                f"total={total_val:.4g} "
                f"nan={math.isnan(total_val)} inf={math.isinf(total_val)}"
            )

            if math.isnan(total_val) or math.isinf(total_val):
                # Log posterior/prior stats from the world model output
                torchrl_logger.error(
                    f"NaN/Inf in world model loss at optim_step={optim_step}!\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'prior_mean'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'prior_std'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'posterior_mean'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'posterior_std'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'state'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'belief'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'reco_policy'))}\n"
                    f"  {_td_stats(sampled_tensordict, ('next', 'reward'))}"
                )
                torchrl_logger.error(
                    "ABORTING: NaN/Inf in world model loss is not recoverable."
                )
                sys.exit(1)

            with timeit("train/world_model-backward"), record_function(
                "## world_model/backward ##"
            ):
                world_model_opt.zero_grad()
                if use_scaler:
                    scaler1.scale(loss_world_model).backward()
                    scaler1.unscale_(world_model_opt)
                else:
                    loss_world_model.backward()
                world_model_grad = clip_grad_norm_(world_model.parameters(), grad_clip)

                torchrl_logger.debug(
                    f"[optim_step={optim_step}] World model grad_norm="
                    f"{world_model_grad:.4g}"
                )
                _check_nan_grads(world_model, "world_model", optim_step)

                if use_scaler:
                    scaler1.step(world_model_opt)
                    scaler1.update()
                else:
                    world_model_opt.step()

            # --- Actor update ---
            with timeit("train/actor-forward"), record_function("## actor/forward ##"):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=train_device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    actor_loss_td, sampled_tensordict = actor_loss(
                        sampled_tensordict.reshape(-1)
                    )

            actor_val = actor_loss_td["loss_actor"].item()
            torchrl_logger.debug(
                f"[optim_step={optim_step}] Actor loss: {actor_val:.4g} "
                f"nan={math.isnan(actor_val)}"
            )

            if math.isnan(actor_val) or math.isinf(actor_val):
                torchrl_logger.error(
                    f"NaN/Inf in actor loss at optim_step={optim_step}!"
                )
                torchrl_logger.error(
                    "ABORTING: NaN/Inf in actor loss is not recoverable."
                )
                sys.exit(1)

            with timeit("train/actor-backward"), record_function(
                "## actor/backward ##"
            ):
                actor_opt.zero_grad()
                if use_scaler:
                    scaler2.scale(actor_loss_td["loss_actor"]).backward()
                    scaler2.unscale_(actor_opt)
                else:
                    actor_loss_td["loss_actor"].backward()
                actor_model_grad = clip_grad_norm_(actor_model.parameters(), grad_clip)

                torchrl_logger.debug(
                    f"[optim_step={optim_step}] Actor grad_norm="
                    f"{actor_model_grad:.4g}"
                )
                _check_nan_grads(actor_model, "actor_model", optim_step)

                if use_scaler:
                    scaler2.step(actor_opt)
                    scaler2.update()
                else:
                    actor_opt.step()

            # --- Value update ---
            with timeit("train/value-forward"), record_function("## value/forward ##"):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(
                    device_type=train_device.type, dtype=autocast_dtype
                ) if autocast_dtype else contextlib.nullcontext():
                    value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)

            value_val = value_loss_td["loss_value"].item()
            torchrl_logger.debug(
                f"[optim_step={optim_step}] Value loss: {value_val:.4g} "
                f"nan={math.isnan(value_val)}"
            )

            if math.isnan(value_val) or math.isinf(value_val):
                torchrl_logger.error(
                    f"NaN/Inf in value loss at optim_step={optim_step}!"
                )
                torchrl_logger.error(
                    "ABORTING: NaN/Inf in value loss is not recoverable."
                )
                sys.exit(1)

            with timeit("train/value-backward"), record_function(
                "## value/backward ##"
            ):
                value_opt.zero_grad()
                if use_scaler:
                    scaler3.scale(value_loss_td["loss_value"]).backward()
                    scaler3.unscale_(value_opt)
                else:
                    value_loss_td["loss_value"].backward()
                critic_model_grad = clip_grad_norm_(value_model.parameters(), grad_clip)

                torchrl_logger.debug(
                    f"[optim_step={optim_step}] Value grad_norm="
                    f"{critic_model_grad:.4g}"
                )
                _check_nan_grads(value_model, "value_model", optim_step)

                if use_scaler:
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
                    "train/episode_reward": last_mean_reward,
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

        # ---- Sync weights to collector policy ----
        weights = TensorDict.from_module(policy)
        collector.update_policy_weights_(weights)
        policy[1].step(optim_steps_per_collect)  # Update exploration noise schedule

        # ---- Async eval: poll results & submit new rollout ----
        if eval_worker is not None:
            eval_result = eval_worker.poll()
            if eval_result is not None:
                eval_metrics = {"eval/reward": eval_result["reward"]}
                if logger is not None:
                    log_metrics(logger, eval_metrics, collected_frames)
                    if eval_result["frames"] is not None:
                        logger.log_video(
                            "eval/video",
                            eval_result["frames"],
                            step=collected_frames,
                        )
                torchrl_logger.info(
                    f"Eval result: reward={eval_result['reward']:.4f}, "
                    f"has_video={eval_result['frames'] is not None}"
                )

            if optim_step >= next_eval_step:
                next_eval_step += eval_every
                eval_weights = TensorDict.from_module(policy).data.detach().cpu()
                eval_worker.submit(
                    eval_weights,
                    max_steps=cfg.logger.eval_rollout_steps,
                    # Isaac envs auto-reset, so done.any() fires as soon as a
                    # single sub-env finishes.  Run the full rollout instead.
                    break_when_any_done=False,
                )

        if optim_step >= total_optim_steps:
            break

    # ========================================================================
    # Cleanup
    # ========================================================================
    pbar.close()
    if eval_worker is not None:
        eval_worker.shutdown()
    if not train_env.is_closed:
        train_env.close()
    collector.shutdown()


if __name__ == "__main__":
    main()
