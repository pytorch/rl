# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import os
from functools import partial
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from torchrl import merge_ray_runtime_env, torchrl_logger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.weight_update.llm import get_model_metadata

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for async training. Please install ray with `pip install ray`."
    )
import torch
import tqdm

from grpo_utils import (
    add_kl_transforms_to_replay_buffer,
    check_grpo_dependencies,
    compute_device_allocation,
    get_inference_model,
    get_train_model,
    make_env,
    make_weight_sync_scheme,
)
from omegaconf import DictConfig

try:
    from tensordict import set_list_to_stack
except ImportError:
    raise ImportError(
        "TensorDict is required. Please install it with `pip install tensordict`."
    )
from torchrl.collectors.llm import RayLLMCollector
from torchrl.data import LazyStackStorage, ReplayBuffer
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage
from torchrl.trainers.algorithms.grpo import GRPOTrainer


def _finish_wandb_logger(wandb_logger: WandbLogger | None, exit_code: int) -> None:
    """Finish a wandb run if one was created."""
    if wandb_logger is None:
        return
    finish = getattr(wandb_logger.experiment, "finish", None)
    if finish is not None:
        finish(exit_code=exit_code)


def setup_environment() -> None:
    """Setup required environment variables and configurations."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")

    # Set default dtype to float32 for mixed precision training
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")
    set_list_to_stack(True).set()

    # Ensure CUDA is using the correct dtype
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")


def train(
    replay_buffer: ReplayBuffer,
    cfg: DictConfig,
    collectors: list[RayLLMCollector],
    inference_policy,
    devices: list[int] | None = None,
):
    """Main training entry point for GRPO async.

    Data collection and optimization happen concurrently (async mode).
    The training loop is fully managed by :class:`~torchrl.trainers.GRPOTrainer`
    with ``async_collection=True``.

    Args:
        replay_buffer: The replay buffer to store experiences.
        cfg: The configuration object containing training parameters.
        collectors: The list of async collector objects.
        inference_policy: The inference-side LLM wrapper.
        devices: The devices to use for the training model.
    """
    # Setup training model and tokenizer
    policy_training, _train_tokenizer = get_train_model(cfg, devices=devices)
    train_device = torch.device(f"cuda:{devices[0]}" if devices else "cuda:0")

    # Setup loss function
    loss_fn = GRPOLoss(
        actor_network=policy_training,
        kl_to_ref_coeff=cfg.train.kl_to_ref_coeff
        if (cfg.train.kl_coef_in_loss and cfg.train.use_kl_to_ref)
        else 0.0,
        kl_to_inference_coeff=cfg.train.kl_to_inference_coeff,
        entropy_coeff=cfg.train.entropy_coeff,
        masking_strategy="rlhf" if cfg.env.reasoning else "sft",
        device=train_device,
    )
    if cfg.env.reasoning:
        # TODO: this is clunky, we should find a way to do this more naturally
        loss_fn.set_keys(sample_log_prob=("next", "log_probs", "full"))
    if cfg.model.compile:
        loss_fn = torch.compile(loss_fn)

    inference_engine = inference_policy.model

    # Create weight sync scheme for the collectors
    weight_sync_scheme = make_weight_sync_scheme(
        engine=inference_engine, cfg=cfg, device=train_device
    )

    # Set up weight sync scheme for collectors
    torchrl_logger.info("Setting up weight synchronization scheme...")
    sender = weight_sync_scheme.create_sender()
    # Register the HuggingFace model directly (not the TransformersWrapper)
    # so state_dict() keys match vLLM's expected format (e.g., model.layers.0.*)
    sender.register_model(policy_training.model)

    # Initialize collective group
    torchrl_logger.info("Initializing collective group...")
    metadata = get_model_metadata(policy_training.model)
    if getattr(cfg.inference_model, "backend", "vllm") == "sglang":
        sender.init_all_workers_group(metadata)
    else:
        sender.init_all_workers_group(metadata, vllm_engine=inference_engine)

    # First weight update
    torchrl_logger.info("Performing first weight update...")
    sender.update_weights()
    torchrl_logger.info("Completed first update_policy_weights. Starting collectors...")

    for i, collector in enumerate(collectors):
        torchrl_logger.info(f"Starting collector {i}...")
        collector.start()

    # Register collectors with the sender so increment_version() is
    # called automatically after each update_weights().
    if hasattr(sender, "register_collector"):
        for collector in collectors:
            sender.register_collector(collector)

    # The trainer waits for the replay buffer to receive its first write
    # before starting optimization, so no manual wait is needed here.

    # Make optimizer
    optimizer = torch.optim.Adam(
        policy_training.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=getattr(cfg.optimizer, "eps", 1e-8),
        fused=False,
    )

    # Make checkpoint dir
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Make wandb logger
    experiment_name = cfg.logging.experiment_name
    if experiment_name is not None:
        experiment_name = [experiment_name]
    else:
        experiment_name = []
    experiment_name.append(cfg.env.dataset)
    experiment_name.append(cfg.model.name)
    wandb_logger = WandbLogger(
        project="grpo-async",
        exp_name="-".join(["grpo-async"] + experiment_name),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # In async mode the collectors run in the background and write to the
    # replay buffer directly. GRPOTrainer with async_collection=True skips
    # the pre_epoch buffer-extend hook and only samples from it.
    # We pass the first collector as the reference collector for the trainer's
    # progress tracking; the others have already been started above
    # (start() is a no-op on an already-running collector).
    torchrl_logger.info("Building GRPOTrainer...")
    autocast_dtype = getattr(torch, cfg.train_model.torch_dtype)
    trainer = GRPOTrainer(
        collector=collectors[0],
        total_frames=cfg.train.total_dialog_turns,
        frame_skip=1,
        # One optimizer step (gradient_accumulation_steps micro-batches) per
        # trainer iteration, so that weight updates every
        # `weight_update_frequency` optimizer steps interleave with collection
        # as in the reference loop.
        optim_steps_per_batch=cfg.train.gradient_accumulation_steps,
        loss_module=loss_fn,
        optimizer=optimizer,
        weight_sync_sender=sender,
        weight_update_frequency=cfg.train.weight_update_frequency,
        empty_replay_buffer_on_weight_update=False,  # async mode: never flush
        replay_buffer=replay_buffer,
        device=train_device,
        mixed_precision=cfg.train.mixed_precision,
        autocast_dtype=autocast_dtype,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        clip_norm=cfg.optimizer.clip_grad_norm,
        logger=wandb_logger,
        log_interval=cfg.train.logging_frequency,
        num_epochs=1,  # async: one pass per collection step
        async_collection=True,
        log_rewards=True,
        log_kl=cfg.train.use_kl_to_ref,
        log_timings=True,
    )

    torchrl_logger.info("Starting training loop.")
    exit_code = 1
    try:
        trainer.train()
        exit_code = 0
    finally:
        with contextlib.suppress(Exception):
            _finish_wandb_logger(wandb_logger, exit_code)
        for collector in collectors:
            with contextlib.suppress(Exception):
                collector.shutdown()
        shutdown = getattr(sender, "shutdown", None)
        if shutdown is not None:
            with contextlib.suppress(Exception):
                shutdown()


@hydra.main(version_base="1.3", config_path="config", config_name="grpo_gsm8k")
def main(cfg):
    # Check for required GRPO dependencies
    check_grpo_dependencies(getattr(cfg.inference_model, "backend", "vllm"))

    # Force async mode
    if cfg.train.sync:
        raise ValueError(
            "grpo-async.py must run in async mode (`python grpo-async.py mode=async`). "
            "Please use grpo-sync.py for sync mode (`python grpo-sync.py mode=sync`)."
        )

    # Compute device allocation
    device_config = compute_device_allocation(cfg)

    if not ray.is_initialized():
        # Convert OmegaConf to regular dict and filter out unsupported parameters
        ray_init_config = {
            k: dict(v) if isinstance(v, DictConfig) else v
            for k, v in dict(cfg.ray.init_config).items()
            if not k.startswith("_")
        }

        # Add computed GPU configuration and merge with default runtime_env
        ray_init_config["num_gpus"] = device_config["ray_num_gpus"]
        ray_init_config = merge_ray_runtime_env(ray_init_config)
        torchrl_logger.info(f"Ray init config: {ray_init_config=}")
        ray_managed_externally = os.environ.get("RAY_CLUSTER_MANAGED_EXTERNALLY")
        if ray_managed_externally:
            ray.init(address="auto")
        else:
            ray.init(**ray_init_config)

    # Check if num_devices is set
    if cfg.inference_model.num_devices is None:
        raise ValueError(
            "Inference model num_devices must be set via inference_model.num_devices"
        )
    if cfg.train.use_kl_to_ref and cfg.ref_model.num_devices is None:
        raise ValueError(
            "Ref model num_devices must be set via ref_model.num_devices when use_kl_to_ref is True"
        )
    if cfg.train_model.num_devices is None:
        raise ValueError(
            "Train model num_devices must be set via train_model.num_devices"
        )

    # Convert OmegaConf to regular dict for Ray configs
    replay_buffer_config = dict(cfg.ray.replay_buffer_config)
    collector_config = dict(cfg.ray.collector_config)
    train_handler_config = dict(cfg.ray.train_handler_config)

    inference_policy = get_inference_model(
        cfg,
        devices=device_config["inference_model_devices"],
    )
    torchrl_logger.info(f"Inference policy: {inference_policy}")

    torchrl_logger.info(f"Starting replay buffer with {replay_buffer_config=}")
    if cfg.train.optim_batch_size % cfg.train.gradient_accumulation_steps != 0:
        raise ValueError(
            "optim_batch_size must be divisible by gradient_accumulation_steps"
        )
    rb = RayReplayBuffer(
        storage=partial(
            LazyStackStorage,
            cfg.train.buffer_size
            if cfg.train.buffer_size
            else cfg.env.repeats * cfg.env.num_envs,
        ),
        transform_factory=partial(MCAdvantage, grpo_size=cfg.env.repeats),
        batch_size=max(
            1, cfg.train.optim_batch_size // cfg.train.gradient_accumulation_steps
        ),
        remote_config=replay_buffer_config,
    )

    add_kl_transforms_to_replay_buffer(rb, cfg)

    torchrl_logger.info(f"Replay buffer: {rb}")

    collector_config["num_gpus"] = 0
    collector_config["num_cpus"] = 2
    torchrl_logger.info(f"Starting collector with {collector_config=}")

    if cfg.train.sync_iter is not None:
        raise ValueError("sync_iter is not supported in async mode.")
    collectors = []
    for i in tqdm.trange(cfg.env.num_envs, desc="Starting collectors"):
        collector = RayLLMCollector(
            env=partial(make_env, cfg, single_env=True),
            policy=inference_policy,
            dialog_turns_per_batch=cfg.train.dialog_turns_per_batch,
            total_dialog_turns=cfg.train.total_dialog_turns,
            replay_buffer=rb,
            ray_init_config=None,
            weight_updater=None,
            track_policy_version=True,
            remote_config=collector_config,
            yield_only_last_steps=cfg.env.reasoning,
            verbose=False,
        )
        collectors.append(collector)
        if i == 0:
            # wait for the first collector to initialize
            ray.get(collector._collector.is_initialized.remote())
    inits = []
    for collector in tqdm.tqdm(
        collectors[1:], desc="Checking collector initialization"
    ):
        inits.append(collector._collector.is_initialized.remote())
    ray.get(inits)
    torchrl_logger.info("All collectors initialized")

    train_handler_config = {
        "num_cpus": train_handler_config.get("num_cpus", 1),
        "num_gpus": cfg.train_model.num_devices,
    }
    torchrl_logger.info(f"Starting training handler with {train_handler_config=}")
    train_handler = ray.remote(
        **train_handler_config,
    )(train)

    # launch training
    try:
        ray.get(
            train_handler.remote(
                rb,
                cfg,
                collectors,
                inference_policy,
                devices=device_config["train_model_devices"],
            )
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    main()
