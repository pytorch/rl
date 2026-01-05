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
from torch.autograd.profiler import record_function
from omegaconf import DictConfig

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

# mixed precision training
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
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

    # Make async multi-collector (uses env factory for worker processes)
    collector = make_collector(cfg, train_env_factory, policy)

    # Make replay buffer with minimal sample-time transforms
    batch_size = cfg.replay_buffer.batch_size
    batch_length = cfg.replay_buffer.batch_length
    buffer_size = cfg.replay_buffer.buffer_size
    scratch_dir = cfg.replay_buffer.scratch_dir
    prefetch = cfg.replay_buffer.prefetch
    replay_buffer = make_replay_buffer(
        batch_size=batch_size,
        batch_seq_len=batch_length,
        buffer_size=buffer_size,
        buffer_scratch_dir=scratch_dir,
        device=device,
        prefetch=prefetch,
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

    init_random_frames = cfg.collector.init_random_frames
    optim_steps_per_batch = cfg.optimization.optim_steps_per_batch
    grad_clip = cfg.optimization.grad_clip
    eval_iter = cfg.logger.eval_iter
    eval_rollout_steps = cfg.logger.eval_rollout_steps

    # Enable TensorFloat32 for better performance on Ampere+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if cfg.optimization.compile:
        torch._dynamo.config.capture_scalar_outputs = True

        torchrl_logger.info("Compiling loss modules with warmup=3")
        backend = cfg.optimization.compile_backend

        # Note: We do NOT compile rssm_prior/rssm_posterior here because they are
        # shared with the policy used in the collector. Compiling them would cause
        # issues with the MultiCollector workers.
        #
        # Instead, we compile the loss modules themselves which wraps the forward pass.
        # fullgraph=False allows graph breaks which can help with inductor issues.
        # warmup=3 runs eagerly for first 3 calls before compiling.
        world_model_loss = compile_with_warmup(
            world_model_loss, backend=backend, mode=cfg.optimization.compile_mode, fullgraph=False, warmup=3
        )
        actor_loss = compile_with_warmup(actor_loss, backend=backend, mode=cfg.optimization.compile_mode, warmup=3)
        value_loss = compile_with_warmup(value_loss, backend=backend, mode=cfg.optimization.compile_mode, warmup=3)

    # Throughput tracking
    t_iter_start = time.time()

    # Profiling setup (encapsulated in helper class)
    profiler = DreamerProfiler(cfg, device, pbar)

    for i, tensordict in enumerate(collector):
        # Note: Collection time is implicitly measured by the collector's iteration
        # The time between loop iterations that isn't training is effectively collection time
        with timeit("collect/preproc"):
            pbar.update(tensordict.numel())
            current_frames = tensordict.numel()
            collected_frames += current_frames

            ep_reward = tensordict.get("episode_reward")[..., -1, 0]
            # Apply storage transforms (ToTensorImage, Resize, GrayScale) once at extend-time
            tensordict_cpu = tensordict.cpu()
            if storage_transform is not None:
                tensordict_cpu = storage_transform(tensordict_cpu)
            replay_buffer.extend(tensordict_cpu)

        if collected_frames >= init_random_frames:
            for _ in range(optim_steps_per_batch):
                # sample from replay buffer
                with timeit("train/sample"), record_function(
                    "## train/sample ##"
                ):
                    sampled_tensordict = replay_buffer.sample().reshape(
                        -1, batch_length
                    )
                    # Ensure all tensors are on the correct device and contiguous
                    # The clone() ensures NCHW layout for torch.compile compatibility
                    sampled_tensordict = sampled_tensordict.to(device).clone()

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
                        model_loss_td, sampled_tensordict = world_model_loss(
                            sampled_tensordict
                        )
                        loss_world_model = (
                            model_loss_td["loss_model_kl"]
                            + model_loss_td["loss_model_reco"]
                            + model_loss_td["loss_model_reward"]
                        )
                    torchrl_logger.debug(
                        f"world_model_loss forward OK, loss={loss_world_model.item():.4f}"
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
                    world_model_grad = clip_grad_norm_(
                        world_model.parameters(), grad_clip
                    )
                    if autocast_dtype:
                        scaler1.step(world_model_opt)
                        scaler1.update()
                    else:
                        world_model_opt.step()

                # update actor network
                with timeit("train/actor-forward"), record_function(
                    "## actor/forward ##"
                ):
                    # Mark step begin for CUDAGraph to prevent tensor overwrite issues
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast(
                        device_type=device.type, dtype=autocast_dtype
                    ) if autocast_dtype else contextlib.nullcontext():
                        actor_loss_td, sampled_tensordict = actor_loss(
                            sampled_tensordict.reshape(-1)
                        )
                    torchrl_logger.debug(
                        f"actor_loss forward OK, loss={actor_loss_td['loss_actor'].item():.4f}"
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
                    torchrl_logger.debug("actor_loss backward OK")
                    actor_model_grad = clip_grad_norm_(
                        actor_model.parameters(), grad_clip
                    )
                    if autocast_dtype:
                        scaler2.step(actor_opt)
                        scaler2.update()
                    else:
                        actor_opt.step()

                # update value network
                with timeit("train/value-forward"), record_function(
                    "## value/forward ##"
                ):
                    # Mark step begin for CUDAGraph to prevent tensor overwrite issues
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast(
                        device_type=device.type, dtype=autocast_dtype
                    ) if autocast_dtype else contextlib.nullcontext():
                        value_loss_td, sampled_tensordict = value_loss(
                            sampled_tensordict
                        )
                    torchrl_logger.debug(
                        f"value_loss forward OK, loss={value_loss_td['loss_value'].item():.4f}"
                    )

                with timeit("train/value-backward"), record_function(
                    "## value/backward ##"
                ):
                    value_opt.zero_grad()
                    if autocast_dtype:
                        scaler3.scale(value_loss_td["loss_value"]).backward()
                        scaler3.unscale_(value_opt)
                    else:
                        value_loss_td["loss_value"].backward()
                    torchrl_logger.debug("value_loss backward OK")
                    critic_model_grad = clip_grad_norm_(
                        value_model.parameters(), grad_clip
                    )
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

        # Compute throughput metrics
        t_iter_end = time.time()
        iter_time = t_iter_end - t_iter_start

        # FPS: Frames (env steps) collected per second
        fps = current_frames / iter_time if iter_time > 0 else 0

        metrics_to_log = {"reward": ep_reward.mean().item()}
        if collected_frames >= init_random_frames:
            # SPS: Samples (batch elements) processed per second
            # Each optim step processes batch_size samples
            total_samples = optim_steps_per_batch * batch_size
            sps = total_samples / iter_time if iter_time > 0 else 0

            # UPS: Updates (gradient steps) per second
            # 3 updates per optim step (world_model, actor, value)
            total_updates = optim_steps_per_batch * 3
            ups = total_updates / iter_time if iter_time > 0 else 0

            loss_metrics = {
                "loss_model_kl": model_loss_td["loss_model_kl"].item(),
                "loss_model_reco": model_loss_td["loss_model_reco"].item(),
                "loss_model_reward": model_loss_td["loss_model_reward"].item(),
                "loss_actor": actor_loss_td["loss_actor"].item(),
                "loss_value": value_loss_td["loss_value"].item(),
                "world_model_grad": world_model_grad,
                "actor_model_grad": actor_model_grad,
                "critic_model_grad": critic_model_grad,
                # Throughput metrics
                "throughput/fps": fps,  # Frames per second (collection)
                "throughput/sps": sps,  # Samples per second (training)
                "throughput/ups": ups,  # Updates per second (gradient steps)
                "throughput/iter_time": iter_time,  # Total iteration time
                # Detailed timing from timeit (some metrics may be empty when compiling)
                **timeit.todict(prefix="time"),
            }
            metrics_to_log.update(loss_metrics)
        else:
            # During random collection phase, only log FPS
            metrics_to_log["throughput/fps"] = fps
            metrics_to_log["throughput/iter_time"] = iter_time

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)

        # Reset timer for next iteration
        t_iter_start = time.time()

        policy[1].step(current_frames)
        collector.update_policy_weights_()
        # Evaluation
        if (i % eval_iter) == 0:
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
    # Note: train envs are managed by the collector workers
    collector.shutdown()

    del test_env
    del collector


if __name__ == "__main__":
    main()
