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

from torchrl import torchrl_logger
from torchrl.collectors.llm.weight_update.vllm import vLLMUpdater
from torchrl.data.llm.history import History
from torchrl.record.loggers.wandb import WandbLogger

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for async training. Please install ray with `pip install ray`."
    )
import torch
import tqdm

from ei_utils import (
    compute_device_allocation,
    create_cosine_scheduler_with_warmup,
    get_inference_model,
    get_train_model,
    log_training_metrics,
    make_env,
    make_weight_updater,
    RemoteDataLogger,
)
from omegaconf import DictConfig
from ray.util.queue import Queue

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
from torchrl.data import (
    LazyStackStorage,
    PrioritizedSampler,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.llm.topk import TopKRewardSelector
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.objectives.llm.sft import SFTLoss


def setup_environment() -> None:
    """Setup required environment variables and configurations."""
    if os.getenv("VLLM_USE_V1", "1") != "0":
        raise RuntimeError("VLLM_USE_V1=0 must be set in environment")

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
    collector: RayLLMCollector,
    devices: list[int] | None = None,
):
    """Main training loop for EI async.

    This function implements asynchronous training where data collection and optimization
    happen concurrently. The total number of steps is determined by the number of epochs,
    samples per epoch, and batches collected.

    Args:
        replay_buffer: The replay buffer to store experiences
        cfg: The configuration object containing training parameters
        collector: The collector object.
        devices: The devices to use for the training model.
    """
    # Setup training model and tokenizer
    policy_training, train_tokenizer = get_train_model(
        cfg, devices=devices, chat_template_name="qwen"
    )
    train_device = devices[0]  # Use first device for batch processing

    # Setup loss function
    loss_fn = SFTLoss(
        actor_network=policy_training,
        kl_to_ref_coeff=cfg.train.kl_to_ref_coeff,
        tokenizer=train_tokenizer,
        tokenizer_kwargs={"chat_template_name": "qwen"},
        device=torch.device(f"cuda:{train_device}")
        if train_device is not None
        else None,
        loss_function=cfg.train.loss_function,
        beta=cfg.train.minor_sft_beta,
    )
    if cfg.model.compile:
        loss_fn = torch.compile(loss_fn)

    # Get metadata
    model_metadata = vLLMUpdater.get_model_metadata(policy_training)

    # Create weight updater with remote LLM
    weight_updater: vLLMUpdater = make_weight_updater(
        master_address="localhost",  # Since we're running locally
        master_port=None,  # Will auto-assign an open port
        model_metadata=model_metadata,
        vllm_tp_size=cfg.inference_model.num_devices
        if cfg.inference_model.num_devices is not None
        else len(cfg.inference_model.get("devices", [1])),
    )
    collector.weight_updater = weight_updater

    # Initialize the weight updater
    weight_updater.init(model_metadata=model_metadata)

    # First update the weights
    with timeit("update_policy_weights"):
        weight_updater.push_weights(policy_training)
    timeit.print(prefix="First update_policy_weights_ time")
    timeit.reset()

    # Make optimizer
    torchrl_logger.info("Starting optimizer.")
    optimizer = torch.optim.Adam(
        policy_training.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        fused=False,
    )
    scaler = GradScaler(enabled=cfg.train.mixed_precision)

    # Calculate total optimization steps for scheduler
    # The training loop structure: for each collector iteration, we do cfg.train.epochs epochs
    # Each epoch processes the entire replay buffer, and optimization happens every gradient_accumulation_steps
    # We need to estimate the total number of optimization steps
    # For now, we'll use a conservative estimate based on the total dialog turns
    # This can be refined based on the actual training dynamics
    total_optim_steps = (
        cfg.train.total_dialog_turns
        * cfg.train.epochs
        // cfg.train.gradient_accumulation_steps
    )

    # Create scheduler if enabled
    scheduler = None
    if cfg.optimizer.scheduler.enabled:
        warmup_steps = cfg.optimizer.scheduler.warmup_steps
        num_cycles = cfg.optimizer.scheduler.num_cycles
        torchrl_logger.info(
            f"Creating {cfg.optimizer.scheduler.type} scheduler with {warmup_steps} warmup steps out of {total_optim_steps} total steps"
        )

        scheduler = create_cosine_scheduler_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optim_steps,
            num_cycles=num_cycles,
        )

    # Make checkpoint dir
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Make wandb logger
    torchrl_logger.info("Starting wandb logger.")
    experiment_name = cfg.logging.experiment_name
    if experiment_name is not None:
        experiment_name = [experiment_name]
    else:
        experiment_name = []

    experiment_name.append(cfg.env.dataset)
    experiment_name.append(cfg.model.name)

    # Create local wandb logger for training metrics
    wandb_config = {
        "project": "ei-async",
        "exp_name": "-".join(["ei-async"] + experiment_name),
    }
    wandb_logger = WandbLogger(**wandb_config)

    # Pass the logging actor reference to the collector
    log_queue = Queue(maxsize=1000)
    collector.set_postproc(RemoteDataLogger(log_queue=log_queue))

    # Start collector
    collector.start()

    # Wait for initial data
    while not replay_buffer.write_count:
        time.sleep(1)

    # Training loop
    total_steps = (
        -(cfg.train.total_dialog_turns // -cfg.train.optim_batch_size)
        * cfg.train.epochs
    )
    torchrl_logger.info(f"Total steps: {total_steps}")

    pbar = tqdm.tqdm(total=total_steps)
    grad_norm = 0.0  # Initialize grad_norm
    data_read_count = 0
    optim_step = 0
    start_time = time.time()

    for step in range(total_steps):
        pbar.update(1)
        pbar.set_description(f"Step {step}, writes: {replay_buffer.write_count}")

        with timeit("sampling"):
            # Sample batch and move to device
            batch = replay_buffer.sample(cfg.train.optim_batch_size).to(train_device)

            max_policy_age = (
                batch.view(-1)[0]["next", "policy_version"] - collector.policy_version
            ).max()
            if (
                cfg.train.max_policy_age is not None
                and max_policy_age > cfg.train.max_policy_age
            ):
                # Skip this batch, as it's too old
                torchrl_logger.info(f"Skipping batch with policy age {max_policy_age}")
                continue

            # For logging purposes, we get the last element of the history
            # and convert it to a string
            history: History = batch.view(-1)[0]["next", "history", "prompt"]
            history_str: list[str] | str = history.apply_chat_template(
                tokenizer=train_tokenizer
            )
            while not isinstance(history_str, str):
                history_str = "\n".join(history_str)

            data_read_count += batch.numel()

        with timeit("forward_pass"):
            # Forward pass with mixed precision
            with autocast("cuda", enabled=cfg.train.mixed_precision):
                loss = loss_fn(batch)
                if loss.loss_kl_to_ref is not None:
                    loss_val = loss.loss_sft + loss.loss_kl_to_ref
                else:
                    loss_val = loss.loss_sft
                loss_val = loss_val / cfg.train.gradient_accumulation_steps
        with timeit("backward_pass"):
            # Backward pass
            if cfg.train.mixed_precision and cfg.train_model.torch_dtype == "float16":
                scaler = GradScaler(enabled=True)
                scaler.scale(loss_val).backward()
            else:
                loss_val.backward()

        # Optimization step
        if ((step + 1) % cfg.train.gradient_accumulation_steps) == 0:
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

                # Step the scheduler
                if scheduler is not None:
                    scheduler.step()

                # Increment optimization step counter
                optim_step += 1

        # Update metrics
        if (step % cfg.train.logging_frequency) == 0:
            log_training_metrics(
                wandb_logger=wandb_logger,
                replay_buffer=replay_buffer,
                batch=batch,
                loss=loss,
                grad_norm=grad_norm,
                global_step=step,
                data_read_count=data_read_count,
                collector=collector,
                start_time=start_time,
                gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                history_str=history_str,
            )
            # Log additional metrics
            wandb_logger.log_scalar(
                "learning_rate", float(optimizer.param_groups[0]["lr"]), step=step
            )
            wandb_logger.log_scalar("optim_step", optim_step, step=step)
            while not log_queue.empty():
                logs = log_queue.get()
                for k, v in logs.items():
                    wandb_logger.log_scalar(k, v)

        # Update policy weights
        if step % cfg.train.weight_update_frequency == 0:
            with timeit("update_policy_weights"):
                torchrl_logger.info("Updating policy weights...")
                weight_updater.push_weights(policy_training)
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

        # Clear memory
        del loss_val
        # TODO: do we need this? Does it interfere with other processes?
        # torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    collector.shutdown()


@hydra.main(version_base=None, config_path="config", config_name="ei_gsm8k")
def main(cfg):
    # Force async mode
    if cfg.train.sync:
        raise ValueError(
            "expert-iteration-async.py must run in async mode (`python expert-iteration-async.py mode=async`). Please use expert-iteration-sync.py for sync mode (`python expert-iteration-sync.py mode=sync`)."
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

        # Add computed GPU configuration
        ray_init_config["num_gpus"] = device_config["ray_num_gpus"]
        # Ensure runtime_env and env_vars exist
        if "runtime_env" not in ray_init_config:
            ray_init_config["runtime_env"] = {}
        if not isinstance(ray_init_config["runtime_env"], dict):
            ray_init_config["runtime_env"] = dict(ray_init_config["runtime_env"])
        if "env_vars" not in ray_init_config["runtime_env"]:
            ray_init_config["runtime_env"]["env_vars"] = {}
        if not isinstance(ray_init_config["runtime_env"]["env_vars"], dict):
            ray_init_config["runtime_env"]["env_vars"] = dict(
                ray_init_config["runtime_env"]["env_vars"]
            )
        torchrl_logger.info(f"Ray init config: {ray_init_config=}")
        ray.init(**ray_init_config)

    # Check if num_devices is set
    if cfg.inference_model.num_devices is None:
        raise ValueError(
            "Inference model num_devices must be set via inference_model.num_devices"
        )
    if cfg.ref_model.num_devices is None:
        raise ValueError("Ref model num_devices must be set via ref_model.num_devices")
    if cfg.train_model.num_devices is None:
        raise ValueError(
            "Train model num_devices must be set via train_model.num_devices"
        )

    # Convert OmegaConf to regular dict for Ray configs
    replay_buffer_config = dict(cfg.ray.replay_buffer_config)
    collector_config = dict(cfg.ray.collector_config)
    train_handler_config = dict(cfg.ray.train_handler_config)

    inference_policy = get_inference_model(
        cfg, devices=device_config["inference_model_devices"]
    )
    torchrl_logger.info(f"Inference policy: {inference_policy}")

    torchrl_logger.info(f"Starting replay buffer with {replay_buffer_config=}")
    rb_size = cfg.train.buffer_size
    if rb_size is None:
        # Hardcoded for now
        rb_size = 256
    if cfg.train.prioritized_sampling:
        rb_cls = TensorDictReplayBuffer
        rb_sampler_cls = partial(
            PrioritizedSampler,
            max_capacity=rb_size,
            alpha=cfg.train.prioritized_sampling_alpha,
            beta=cfg.train.prioritized_sampling_beta,
            eps=cfg.train.prioritized_sampling_epsilon,
        )
        kwargs = {"priority_key": ("next", "reward")}
    else:
        rb_cls = ReplayBuffer
        rb_sampler_cls = None
        kwargs = {}
    rb = RayReplayBuffer(
        storage=partial(
            LazyStackStorage,
            rb_size,
            device="cpu",
        ),
        transform_factory=partial(
            TopKRewardSelector,
            total_dialog_turns=cfg.env.repeats,
            topk_size=cfg.train.topk_size,
        ),
        batch_size=cfg.train.optim_batch_size,
        remote_config=replay_buffer_config,
        replay_buffer_cls=rb_cls,
        sampler=rb_sampler_cls,
        **kwargs,
    )
    torchrl_logger.info(f"Replay buffer: {rb}")

    # Create remote collector using RayLLMCollector
    collector_config["num_gpus"] = (
        # The ref model will be instantiated within the collector, so we only need to allocate the number of devices for the inference model
        cfg.ref_model.num_devices
    )
    torchrl_logger.info(f"Starting collector with {collector_config=}")

    dialog_turns_per_batch = cfg.train.dialog_turns_per_batch
    if dialog_turns_per_batch is None:
        # Hardcoded for now
        dialog_turns_per_batch = cfg.env.repeats

    collector = RayLLMCollector(
        env=partial(make_env, cfg, devices=device_config["ref_model_devices"]),
        policy=inference_policy,
        dialog_turns_per_batch=dialog_turns_per_batch,
        total_dialog_turns=cfg.train.total_dialog_turns,
        replay_buffer=rb,
        ray_init_config=None,  # Ray is already initialized
        weight_updater=None,  # We'll create this after getting the remote LLM
        track_policy_version=True,
        remote_config=collector_config,
        verbose=True,
    )
    # Ensure collector is initialized by calling a method that will block until ready
    ray.get(collector._collector.is_initialized.remote())
    torchrl_logger.info(f"Collector: {collector}")

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
        train_handler.remote(rb, cfg, collector, device_config["train_model_devices"])
    )


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    main()
