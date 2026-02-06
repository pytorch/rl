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

GPU strategy (2-GPU async pipeline):
- GPU 0 ("sim_device"): IsaacLab simulation + collection policy inference
- GPU 1 ("train_device"): Model training (world model, actor, value gradients)
- Collection runs in a background thread; training runs in the main thread
- Policy weights are synced periodically from train_device to sim_device
- Falls back to single-GPU if only 1 GPU is available
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
import copy
import threading
import time

import hydra
import torch
import torch.cuda
import tqdm
from dreamer_utils import (
    _make_env,
    log_metrics,
    make_dreamer,
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


def _collect_loop(collector, replay_buffer, stop_event, stats):
    """Background collection loop.

    Runs the collector and extends the replay buffer continuously.
    Tracks episode statistics for logging.
    """
    for data in collector:
        replay_buffer.extend(data)
        stats["collected_frames"] += data.numel()

        # Track episode rewards from completed episodes
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            episode_rewards = data["next", "episode_reward"][done_mask]
            stats["last_mean_reward"] = episode_rewards.mean().item()
            stats["last_episodes_done"] = done_mask.sum().item()

        if stop_event.is_set():
            break


@hydra.main(version_base="1.1", config_path="", config_name="config_isaac")
def main(cfg: DictConfig):
    # ========================================================================
    # Device setup: sim on cuda:0, training on cuda:1 (or cuda:0 if single GPU)
    # ========================================================================
    sim_device = torch.device("cuda:0")  # IsaacLab always binds to cuda:0
    num_gpus = torch.cuda.device_count()
    train_device = torch.device("cuda:1") if num_gpus > 1 else sim_device
    async_mode = train_device != sim_device

    torchrl_logger.info(
        f"GPU setup: {num_gpus} GPUs available, "
        f"sim_device={sim_device}, train_device={train_device}, "
        f"async={'yes' if async_mode else 'no (single GPU)'}"
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
    # Collector policy: deep copy on sim_device for parallel collection
    # ========================================================================
    collector_policy = copy.deepcopy(policy)
    if async_mode:
        collector_policy = collector_policy.to(sim_device)
        torchrl_logger.info(
            f"Created collector policy copy on {sim_device} "
            f"(training models on {train_device})"
        )

    # ========================================================================
    # Losses (on train_device)
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
    gpu_storage = cfg.replay_buffer.get("gpu_storage", False)
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
        gpu_storage=gpu_storage,
    )
    torchrl_logger.info(
        f"Replay buffer: batch_size={batch_size}, batch_length={batch_length}, "
        f"gpu_storage={gpu_storage}, device={train_device}"
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

    if autocast_dtype is not None:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
        scaler3 = GradScaler()

    if train_device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ========================================================================
    # torch.compile -- compile individual MLP sub-modules, not full losses.
    # The full loss modules use heavy TensorDict operations that dynamo can't
    # trace reliably with the container's tensordict version. Instead, we
    # compile the pure-tensor MLP internals (encoder, decoder, reward) and
    # set suppress_errors=True so dynamo falls back to eager for anything
    # it can't trace.
    # ========================================================================
    compile_cfg = cfg.optimization.compile
    if compile_cfg.enabled:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_scalar_outputs = True
        backend = compile_cfg.backend

        # Access inner MLP modules through the world model structure:
        #   world_model = WorldModelWrapper(transition_model, reward_model)
        #   transition_model = TensorDictSequential(encoder_td, rssm_rollout, decoder_td)
        transition_model = world_model.module[0]
        reward_td = world_model.module[1]

        # Encoder MLP: transition_model[0].module
        encoder_mlp = transition_model.module[0].module
        transition_model.module[0].module = torch.compile(encoder_mlp, backend=backend)

        # Decoder MLP: transition_model[2][0].module
        # (decoder is ProbabilisticTensorDictSequential, first element has the MLP)
        decoder_mlp = transition_model.module[2].module[0].module
        transition_model.module[2].module[0].module = torch.compile(
            decoder_mlp, backend=backend
        )

        # Reward MLP: reward_model[0].module
        reward_mlp = reward_td.module[0].module
        reward_td.module[0].module = torch.compile(reward_mlp, backend=backend)

        # Value model MLP: value_model[0].module
        value_mlp = value_model.module[0].module
        value_model.module[0].module = torch.compile(value_mlp, backend=backend)

        torchrl_logger.info(
            f"Compiled 4 MLP sub-modules with backend={backend}, "
            f"suppress_errors=True (RSSM + loss modules stay eager)"
        )

    # ========================================================================
    # Training config
    # ========================================================================
    total_optim_steps = cfg.optimization.total_optim_steps
    log_every = cfg.optimization.log_every
    grad_clip = cfg.optimization.grad_clip
    # How often to sync weights from training policy to collector policy
    weight_sync_every = cfg.collector.optim_steps_per_collect

    pbar = tqdm.tqdm(total=total_optim_steps, desc="Optim steps")
    t_log_start = time.time()
    frames_at_log_start = 0

    # ========================================================================
    # Start async collection in background thread
    # ========================================================================
    collection_stats = {
        "collected_frames": 0,
        "last_mean_reward": float("nan"),
        "last_episodes_done": 0,
    }
    stop_event = threading.Event()
    collector_thread = threading.Thread(
        target=_collect_loop,
        args=(collector, replay_buffer, stop_event, collection_stats),
        daemon=True,
    )

    torchrl_logger.info(
        f"Starting async training: {total_optim_steps} optim steps, "
        f"weight_sync_every={weight_sync_every}, "
        f"frames_per_batch={cfg.collector.frames_per_batch}, "
        f"init_random_frames={cfg.collector.init_random_frames}"
    )
    collector_thread.start()

    # Wait for enough data before starting training
    init_random_frames = cfg.collector.init_random_frames
    while collection_stats["collected_frames"] < init_random_frames:
        time.sleep(0.1)

    torchrl_logger.info(
        f"Init data collected: {collection_stats['collected_frames']} frames. "
        f"Starting training on {train_device}."
    )

    # ========================================================================
    # Main training loop (runs on train_device, async with collection)
    # ========================================================================
    for optim_step in range(total_optim_steps):
        pbar.update(1)

        # Sample from replay buffer (prefetched to train_device)
        with timeit("train/sample"), record_function("## train/sample ##"):
            sampled_tensordict = replay_buffer.sample()
            # With strict_length=False, the sample numel may not be
            # exactly divisible by batch_length. Truncate to make it so.
            numel = sampled_tensordict.numel()
            usable = (numel // batch_length) * batch_length
            if usable < numel:
                sampled_tensordict = sampled_tensordict[:usable]
            sampled_tensordict = sampled_tensordict.reshape(-1, batch_length)

        # --- World model update ---
        with timeit("train/world_model-forward"), record_function(
            "## world_model/forward ##"
        ):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=train_device.type, dtype=autocast_dtype
            ) if autocast_dtype else contextlib.nullcontext():
                model_loss_td, sampled_tensordict = world_model_loss(sampled_tensordict)
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
                device_type=train_device.type, dtype=autocast_dtype
            ) if autocast_dtype else contextlib.nullcontext():
                actor_loss_td, sampled_tensordict = actor_loss(
                    sampled_tensordict.reshape(-1)
                )

        with timeit("train/actor-backward"), record_function("## actor/backward ##"):
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
                device_type=train_device.type, dtype=autocast_dtype
            ) if autocast_dtype else contextlib.nullcontext():
                value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)

        with timeit("train/value-backward"), record_function("## value/backward ##"):
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

        # Sync training weights to collector policy
        if (optim_step + 1) % weight_sync_every == 0:
            with timeit("train/weight_sync"):
                weights = TensorDict.from_module(policy)
                collector.update_policy_weights_(weights)
                policy[1].step(weight_sync_every)  # Update exploration noise schedule

        # ============================================================
        # Logging
        # ============================================================
        collected_frames = collection_stats["collected_frames"]
        if (optim_step + 1) % log_every == 0:
            t_log_end = time.time()
            log_interval_time = t_log_end - t_log_start
            frames_this_interval = collected_frames - frames_at_log_start

            fps = (
                frames_this_interval / log_interval_time if log_interval_time > 0 else 0
            )
            ops = log_every / log_interval_time if log_interval_time > 0 else 0
            opf = (optim_step + 1) / collected_frames if collected_frames > 0 else 0

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
                "train/episode_reward": collection_stats["last_mean_reward"],
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

    # ========================================================================
    # Cleanup
    # ========================================================================
    pbar.close()
    stop_event.set()
    collector_thread.join(timeout=30)
    if not train_env.is_closed:
        train_env.close()
    collector.shutdown()


if __name__ == "__main__":
    main()
