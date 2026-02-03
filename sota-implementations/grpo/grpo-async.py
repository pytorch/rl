# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import os
import time
from functools import partial
from pathlib import Path

import hydra

from torchrl import merge_ray_runtime_env, torchrl_logger
from torchrl.data.llm.history import History
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
    log_training_metrics,
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
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchrl._utils import timeit
from torchrl.collectors.llm import RayLLMCollector
from torchrl.data import LazyStackStorage, ReplayBuffer
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage


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
    """Main training loop for GRPO async.

    This function implements asynchronous training where data collection and optimization
    happen concurrently. The total number of steps is determined by the number of epochs,
    samples per epoch, and batches collected.

    Args:
        replay_buffer: The replay buffer to store experiences
        cfg: The configuration object containing training parameters
        collectors: The collectors objects.
        devices: The devices to use for the training model.
    """
    # Setup training model and tokenizer
    policy_training, train_tokenizer = get_train_model(cfg, devices=devices)
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
    weight_sync_scheme = make_weight_sync_scheme(engine=inference_engine, cfg=cfg)

    # Set up weight sync scheme for collectors
    # Note: We need to get the sender after the collectors are created
    # For now, we'll update the collectors to use the scheme
    torchrl_logger.info("Setting up weight synchronization scheme...")

    # We'll need to manually set up the sender since collectors were already created
    # without the scheme. In production, collectors should be created with weight_sync_schemes parameter.
    sender = weight_sync_scheme.create_sender()
    sender.register_model(policy_training)

    # Initialize collective group
    torchrl_logger.info("Initializing collective group...")
    metadata = get_model_metadata(policy_training)
    sender.init_all_workers_group(metadata, vllm_engine=inference_engine)

    # First weight update
    with timeit("update_policy_weights"):
        sender.update_weights()
    torchrl_logger.info("Completed first update_policy_weights. Starting collectors...")
    timeit.print(prefix="First update_policy_weights_ time")
    timeit.reset()

    for i, collector in enumerate(collectors):
        torchrl_logger.info(f"Starting collector {i}...")
        collector.start()

    while not replay_buffer.write_count:
        torchrl_logger.info("Waiting for replay buffer...")
        time.sleep(1)

    # Make optimizer
    optimizer = torch.optim.Adam(
        policy_training.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        fused=False,
    )
    scaler = GradScaler(enabled=cfg.train.mixed_precision)

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
        project="grpo-async", exp_name="-".join(["grpo-async"] + experiment_name)
    )

    # Training loop
    total_steps = (
        -(cfg.train.total_dialog_turns // -cfg.train.optim_batch_size)
        * cfg.train.epochs
    )
    torchrl_logger.info(f"Total steps: {total_steps}")

    pbar = tqdm.tqdm(total=total_steps)
    grad_norm = 0.0  # Initialize grad_norm
    data_read_count = 0
    start_time = time.time()

    for step in range(total_steps):
        if not any(collector.is_running() for collector in collectors):
            torchrl_logger.info("Collectors stopped, stopping training")
            break
        pbar.update(1)
        pbar.set_description(f"Step {step}, writes: {replay_buffer.write_count}")

        with timeit("sampling"):
            # Sample the correct batch size for gradient accumulation
            # The replay buffer is configured with batch_size = optim_batch_size // gradient_accumulation_steps
            # So we should sample that amount per step, not the full optim_batch_size
            batch_size_per_step = (
                cfg.train.optim_batch_size // cfg.train.gradient_accumulation_steps
            )
            batch = replay_buffer.sample(batch_size_per_step).to(train_device)
            history: History = batch.view(-1)[0]["history", "full"]
            history_str: list[str] | str = history.apply_chat_template(
                tokenizer=train_tokenizer
            )
            while not isinstance(history_str, str):
                history_str = "\n".join(history_str)

            data_read_count += batch.numel()

        with timeit("forward_pass"):
            with autocast("cuda", enabled=cfg.train.mixed_precision):
                loss = loss_fn(batch)
                loss_val = (
                    loss.mean(reduce=True) / cfg.train.gradient_accumulation_steps
                )

        with timeit("backward_pass"):
            if cfg.train.mixed_precision and cfg.train_model.torch_dtype == "float16":
                scaler = GradScaler(enabled=True)
                scaler.scale(loss_val).backward()
            else:
                loss_val.backward()

        if (step + 1) % cfg.train.gradient_accumulation_steps == 0:
            with timeit("optim_step"):
                if (
                    cfg.train.mixed_precision
                    and cfg.train_model.torch_dtype == "float16"
                ):
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_training.parameters(),
                    cfg.optimizer.clip_grad_norm,
                )

                if (
                    cfg.train.mixed_precision
                    and cfg.train_model.torch_dtype == "float16"
                ):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if (step % cfg.train.logging_frequency) == 0:
            log_training_metrics(
                wandb_logger=wandb_logger,
                replay_buffer=replay_buffer,
                batch=batch,
                loss=loss,
                grad_norm=grad_norm,
                global_step=step,
                data_read_count=data_read_count,
                collector=collectors[0],
                start_time=start_time,
                gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                history_str=history_str,
                use_kl_to_ref=cfg.train.use_kl_to_ref,
            )

        if step % cfg.train.weight_update_frequency == 0:
            with timeit("update_policy_weights"):
                torchrl_logger.info("Updating policy weights...")
                sender.update_weights()
                # TODO: do we need this? Does it interfere with other processes?
                # torch.cuda.empty_cache()
                gc.collect()

        # Checkpointing disabled to prevent disk space issues
        # if (step + 1) % cfg.train.checkpoint_frequency == 0:
        #     with timeit("save_checkpoint"):
        #         torchrl_logger.info(
        #             f"Saving checkpoint {(step+1) // cfg.train.checkpoint_frequency}..."
        #         )
        #         checkpoint = {
        #             "step": step,
        #             "model_state_dict": policy_training.model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "scaler_state_dict": scaler.state_dict(),
        #             "config": dict(cfg),
        #         }
        #         torch.save(checkpoint, checkpoint_dir / f"checkpoint_{step:04d}.pt")

        if step % cfg.train.weight_update_frequency == 0:
            timeit.print(prefix="timeit")
            for key, val in timeit.todict().items():
                wandb_logger.log_scalar(f"timeit/{key}", val)
            timeit.reset()

        del loss_val
        # TODO: do we need this? Does it interfere with other processes?
        # torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    collector.shutdown()


@hydra.main(version_base=None, config_path="config", config_name="grpo_gsm8k")
def main(cfg):
    # Check for required GRPO dependencies
    check_grpo_dependencies()

    # Force async mode
    if cfg.train.sync:
        raise ValueError(
            "grpo-async.py must run in async mode (`python grpo-async.py mode=async`). Please use grpo-sync.py for sync mode (`python grpo-sync.py mode=sync`)."
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
    ray.get(
        train_handler.remote(
            rb,
            cfg,
            collectors,
            inference_policy,
            devices=device_config["train_model_devices"],
        )
    )


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    main()
