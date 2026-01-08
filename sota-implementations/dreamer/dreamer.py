# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import time

import hydra
import torch
import torch.cuda
import tqdm

from dreamer_utils import (
    _default_device,
    DreamerProfiler,
    dump_video,
    log_metrics,
    make_collector,
    make_dreamer,
    make_environments,
    make_replay_buffer,
    make_storage_transform,
)
from omegaconf import DictConfig

# mixed precision training
from torch.amp import GradScaler
from torch.autograd.profiler import record_function
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.llm.transforms import PolicyVersion
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    # cfg = correct_for_frame_skip(cfg)

    device = _default_device(cfg.networks.device)
    assert device.type == "cuda", "Dreamer only supports CUDA devices"

    # Create logger
    exp_name = generate_exp_name("Dreamer", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="dreamer_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode},  # "config": cfg},
        )

    # make_environments returns (train_env_factory, test_env) for async collection
    train_env_factory, test_env = make_environments(
        cfg=cfg,
        parallel_envs=cfg.env.n_parallel_envs,
        logger=logger,
    )

    # Make dreamer components
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
        use_decoder_in_env=cfg.logger.video,
        logger=logger,
    )
    # Losses
    world_model_loss = DreamerModelLoss(world_model)
    # Adapt loss keys to gym backend
    if cfg.env.backend == "gym":
        world_model_loss.set_keys(pixels="observation", reco_pixels="reco_observation")

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

    # Make replay buffer with minimal sample-time transforms
    # Note: Buffer must be created BEFORE collector for true async collection
    batch_size = cfg.replay_buffer.batch_size
    batch_length = cfg.replay_buffer.batch_length
    buffer_size = cfg.replay_buffer.buffer_size
    scratch_dir = cfg.replay_buffer.scratch_dir
    prefetch = cfg.replay_buffer.prefetch
    profiling_enabled = cfg.profiling.enabled
    replay_buffer = make_replay_buffer(
        batch_size=batch_size,
        batch_seq_len=batch_length,
        buffer_size=buffer_size,
        buffer_scratch_dir=scratch_dir,
        device=device,
        prefetch=prefetch if not profiling_enabled else None,
        pixel_obs=cfg.env.from_pixels,
        grayscale=cfg.env.grayscale,
        image_size=cfg.env.image_size,
    )

    # Create storage transform for extend-time processing (applied once per frame)
    storage_transform = make_storage_transform(
        pixel_obs=cfg.env.from_pixels,
        grayscale=cfg.env.grayscale,
        image_size=cfg.env.image_size,
    )

    # Create policy version tracker for async collection
    # This tracks policy versions so we can correlate collected data with policy updates
    policy_version = PolicyVersion(version_type="int")

    # Make async multi-collector with replay buffer for true async collection
    # Device allocation: cuda:0 for training, cuda:1+ for collectors (if multi-GPU)
    collector = make_collector(
        cfg,
        train_env_factory,
        policy,
        training_device=device,
        replay_buffer=replay_buffer,
        storage_transform=storage_transform,
        track_policy_version=policy_version,
    )

    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    # Make optimizer (fused=True for faster GPU execution)
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

    # Grad scaler for mixed precision training https://pytorch.org/docs/stable/amp.html
    # autocast can be: false, true (=bfloat16), float16, bfloat16
    autocast_cfg = cfg.optimization.autocast
    if autocast_cfg in (False, "false", "False"):
        autocast_dtype = None
    elif autocast_cfg in (True, "true", "True", "bfloat16"):
        autocast_dtype = torch.bfloat16
    elif autocast_cfg == "float16":
        autocast_dtype = torch.float16
    else:
        raise ValueError(
            f"Invalid autocast value: {autocast_cfg}. Use false, true, float16, or bfloat16."
        )

    if autocast_dtype is not None:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
        scaler3 = GradScaler()

    optim_steps_per_batch = cfg.optimization.optim_steps_per_batch
    grad_clip = cfg.optimization.grad_clip
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.logger.eval_rollout_steps

    # Enable TensorFloat32 for better performance on Ampere+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    compile_cfg = cfg.optimization.compile
    compile_enabled = compile_cfg.enabled
    compile_losses = set(compile_cfg.losses)
    if compile_enabled:
        torch._dynamo.config.capture_scalar_outputs = True

        compile_warmup = 3
        torchrl_logger.info(f"Compiling loss modules with warmup={compile_warmup}")
        backend = compile_cfg.backend
        mode = compile_cfg.mode

        # Note: We do NOT compile rssm_prior/rssm_posterior here because they are
        # shared with the policy used in the collector. Compiling them would cause
        # issues with the MultiCollector workers.
        #
        # Instead, we compile the loss modules themselves which wraps the forward pass.
        # fullgraph=False allows graph breaks which can help with inductor issues.
        # warmup=compile_warmup runs eagerly for first `compile_warmup` calls before compiling.
        if "world_model" in compile_losses:
            world_model_loss = compile_with_warmup(
                world_model_loss,
                backend=backend,
                mode=mode,
                fullgraph=False,
                warmup=compile_warmup,
            )
        if "actor" in compile_losses:
            actor_loss = compile_with_warmup(
                actor_loss, backend=backend, mode=mode, warmup=compile_warmup
            )
        if "value" in compile_losses:
            value_loss = compile_with_warmup(
                value_loss, backend=backend, mode=mode, warmup=compile_warmup
            )
    else:
        compile_warmup = 0

    # Throughput tracking
    t_iter_start = time.time()

    # Profiling setup (encapsulated in helper class)
    profiler = DreamerProfiler(cfg, device, pbar, compile_warmup=compile_warmup)

    # Calculate total optimization steps based on total frames and collection rate
    # We do optim_steps_per_batch optimization steps per frames_per_batch collected frames
    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    total_optim_steps = (total_frames // frames_per_batch) * optim_steps_per_batch

    # Start async collection - collector fills the buffer in background
    torchrl_logger.info("Starting async collection...")
    collector.start()

    # Wait for enough samples to start training
    # Note: We don't pass init_random_frames to collector (not supported with start()),
    # but we still wait for it here. The untrained policy is effectively random anyway.
    min_frames_to_start = cfg.collector.init_random_frames
    torchrl_logger.info(
        f"Waiting for {min_frames_to_start} initial frames before training..."
    )
    prev_collected_frames = 0
    while replay_buffer.write_count < min_frames_to_start:
        time.sleep(0.1)
        collected_frames = replay_buffer.write_count
        if collected_frames > prev_collected_frames:
            pbar.update(collected_frames - prev_collected_frames)
            prev_collected_frames = collected_frames

    torchrl_logger.info(
        f"Collected {replay_buffer.write_count} frames. Starting training..."
    )

    # Track frames for FPS calculation over logging interval
    frames_at_log_start = prev_collected_frames

    # Main training loop - iterate over optimization steps
    for optim_step in range(total_optim_steps):
        # Track collected frames from buffer write count
        collected_frames = replay_buffer.write_count
        frames_delta = collected_frames - prev_collected_frames
        if frames_delta > 0:
            pbar.update(frames_delta)
            prev_collected_frames = collected_frames

        # Check if we've collected enough frames
        if collected_frames >= total_frames:
            torchrl_logger.info(
                f"Collected {collected_frames} frames (target: {total_frames}). Stopping."
            )
            break

        # sample from replay buffer
        with timeit("train/sample"), record_function("## train/sample ##"):
            sampled_tensordict = replay_buffer.sample().reshape(-1, batch_length)
            if profiling_enabled:
                torch.cuda.synchronize()

        # update world model
        with timeit("train/world_model-forward"), record_function(
            "## world_model/forward ##"
        ):
            # Mark step begin for CUDAGraph to prevent tensor overwrite issues
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
            ) if autocast_dtype else contextlib.nullcontext():
                assert (
                    sampled_tensordict.device.type == "cuda"
                ), "sampled_tensordict should be on CUDA"
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
            torchrl_logger.debug("world_model_loss backward OK")
            world_model_grad = clip_grad_norm_(world_model.parameters(), grad_clip)
            if autocast_dtype:
                scaler1.step(world_model_opt)
                scaler1.update()
            else:
                world_model_opt.step()

        # update actor network
        with timeit("train/actor-forward"), record_function("## actor/forward ##"):
            # Mark step begin for CUDAGraph to prevent tensor overwrite issues
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype
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
            torchrl_logger.debug("actor_loss backward OK")
            actor_model_grad = clip_grad_norm_(actor_model.parameters(), grad_clip)
            if autocast_dtype:
                scaler2.step(actor_opt)
                scaler2.update()
            else:
                actor_opt.step()

        # update value network
        with timeit("train/value-forward"), record_function("## value/forward ##"):
            # Mark step begin for CUDAGraph to prevent tensor overwrite issues
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype
            ) if autocast_dtype else contextlib.nullcontext():
                value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)

        with timeit("train/value-backward"), record_function("## value/backward ##"):
            value_opt.zero_grad()
            if autocast_dtype:
                scaler3.scale(value_loss_td["loss_value"]).backward()
                scaler3.unscale_(value_opt)
            else:
                value_loss_td["loss_value"].backward()
            torchrl_logger.debug("value_loss backward OK")
            critic_model_grad = clip_grad_norm_(value_model.parameters(), grad_clip)
            if autocast_dtype:
                scaler3.step(value_opt)
                scaler3.update()
            else:
                value_opt.step()

        # Step profiler (returns True if profiling complete)
        if profiler.step():
            break

        # Check if profiling is complete and we should exit
        if profiler.should_exit():
            torchrl_logger.info("Profiling complete. Exiting training loop.")
            break

        # Log metrics periodically (every optim_steps_per_batch steps)
        if (optim_step + 1) % optim_steps_per_batch == 0:
            # Compute throughput metrics
            t_iter_end = time.time()
            iter_time = t_iter_end - t_iter_start

            # SPS: Samples (batch elements) processed per second
            total_samples = optim_steps_per_batch * batch_size
            sps = total_samples / iter_time if iter_time > 0 else 0

            # UPS: Updates (gradient steps) per second
            # 3 updates per optim step (world_model, actor, value)
            total_updates = optim_steps_per_batch * 3
            ups = total_updates / iter_time if iter_time > 0 else 0

            # FPS: Frames collected per second (measured from buffer over logging interval)
            frames_collected_this_interval = collected_frames - frames_at_log_start
            fps = frames_collected_this_interval / iter_time if iter_time > 0 else 0

            # Get reward stats from sampled data (since we don't iterate over collector directly)
            sampled_reward = sampled_tensordict.get(("next", "reward"))
            reward_mean = sampled_reward.mean().item()
            reward_std = sampled_reward.std().item()

            loss_metrics = {
                "loss_model_kl": model_loss_td["loss_model_kl"].item(),
                "loss_model_reco": model_loss_td["loss_model_reco"].item(),
                "loss_model_reward": model_loss_td["loss_model_reward"].item(),
                "loss_actor": actor_loss_td["loss_actor"].item(),
                "loss_value": value_loss_td["loss_value"].item(),
                "world_model_grad": world_model_grad,
                "actor_model_grad": actor_model_grad,
                "critic_model_grad": critic_model_grad,
                # Reward stats from sampled batch
                "train/reward_mean": reward_mean,
                "train/reward_std": reward_std,
                # Throughput metrics
                "throughput/fps": fps,  # Frames per second (collection)
                "throughput/sps": sps,  # Samples per second (training)
                "throughput/ups": ups,  # Updates per second (gradient steps)
                "throughput/iter_time": iter_time,  # Total iteration time
                # Policy version tracking
                "policy_version": policy_version.version,
                # Detailed timing from timeit (some metrics may be empty when compiling)
                **timeit.todict(prefix="time"),
            }

            if logger is not None:
                log_metrics(logger, loss_metrics, collected_frames)

            # Reset timer and frame counter for next logging interval
            t_iter_start = time.time()
            frames_at_log_start = collected_frames

            # Update policy weights in collector (for async collection)
            policy[1].step(frames_delta)
            collector.update_policy_weights_()
            # Increment policy version after weight update
            collector.increment_version()

        # Evaluation (every eval_iter * optim_steps_per_batch optimization steps)
        eval_freq = eval_iter * optim_steps_per_batch
        if (optim_step + 1) % eval_freq == 0:
            # Real env
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = test_env.rollout(
                    eval_rollout_steps,
                    policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                test_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                eval_metrics = {"eval/reward": eval_reward}
                if logger is not None:
                    log_metrics(logger, eval_metrics, collected_frames)
            # Simulated env
            if model_based_env_eval is not None:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    eval_rollout = model_based_env_eval.rollout(
                        eval_rollout_steps,
                        policy,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                        auto_reset=False,
                        tensordict=eval_rollout[..., 0]
                        .exclude("next", "action")
                        .to(device),
                    )
                    model_based_env_eval.apply(dump_video)
                    eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                    eval_metrics = {"eval/simulated_reward": eval_reward}
                    if logger is not None:
                        log_metrics(logger, eval_metrics, collected_frames)

    if not test_env.is_closed:
        test_env.close()
    # Shutdown async collector (use async_shutdown since we used start())
    collector.async_shutdown()

    del test_env
    del collector


if __name__ == "__main__":
    main()
