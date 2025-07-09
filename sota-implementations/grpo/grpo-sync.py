# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import os
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
        "Ray is required for sync training. Please install ray with `pip install ray`."
    )
import time

import torch
import tqdm

from grpo_utils import (
    compute_device_allocation,
    get_inference_model,
    get_train_model,
    log_training_metrics,
    make_env,
    make_weight_updater,
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
from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage


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
    """Main training loop for GRPO sync.

    This function implements synchronous training where data collection and optimization
    happen in separate, consecutive steps. The total number of steps is determined by the number of epochs,
    samples per epoch, and batches collected.

    Args:
        replay_buffer: The replay buffer to store experiences. The sampler will typically be a `SamplerWithoutReplacement`.
        cfg: The configuration object containing training parameters
        collector: The collector object.
        devices: The devices to use for the training model.
    """
    # Setup training model and tokenizer
    policy_training, train_tokenizer = get_train_model(cfg, devices=devices)
    train_device = devices[0]  # Use first device for batch processing

    # Setup loss function
    loss_fn = GRPOLoss(
        actor_network=policy_training,
        kl_to_ref_coeff=cfg.train.kl_to_ref_coeff if cfg.train.kl_coef_in_loss else 0.0,
        kl_to_inference_coeff=cfg.train.kl_to_inference_coeff,
        entropy_coeff=cfg.train.entropy_coeff,
        # use prompt/response masking for regular training, and assistant masking for reasoning
        masking_strategy="rlhf" if cfg.env.reasoning else "sft",
        device=train_device,
    )
    if cfg.env.reasoning:
        # TODO: this is clunky, we should find a way to do this more naturally
        loss_fn.set_keys(sample_log_prob=("next", "log_probs", "full"))
    if cfg.model.compile:
        loss_fn = torch.compile(loss_fn)

    # Get metadata
    model_metadata = vLLMUpdater.get_model_metadata(policy_training)

    # Create weight updater with remote LLM
    ray_managed_externally = os.environ.get("RAY_CLUSTER_MANAGED_EXTERNALLY")
    weight_updater: vLLMUpdater = make_weight_updater(
        master_address="localhost"
        if not ray_managed_externally
        else ray.util.get_node_ip_address(),
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
    wandb_logger = WandbLogger(
        project="grpo-sync", exp_name="-".join(["grpo-sync"] + experiment_name)
    )

    # Training loop
    torchrl_logger.info("Starting training loop.")
    pbar = tqdm.tqdm(collector)
    grad_norm = 0.0  # Initialize grad_norm
    data_read_count = 0

    global_step = 0
    start_time = time.time()
    for data in pbar:
        # Wait for the replay buffer to be filled - when reasoning, we collect trajectories
        #  so the buffer may not be filled straight away
        if not len(replay_buffer):
            torchrl_logger.info("Waiting for replay buffer to be filled")
            continue
        else:
            torchrl_logger.info(f"Replay buffer filled: {len(replay_buffer)}")

        pbar.update(1)

        # data is None as the collector directly writes to the replay buffer
        if data is not None:
            raise ValueError("Data is not None")

        for _ in range(cfg.train.epochs):
            # Iterate over the replay buffer
            for batch in replay_buffer:
                batch = batch.to(train_device)
                global_step += 1
                pbar.set_description(
                    f"Gradient step {global_step}, writes: {replay_buffer.write_count}, batch size: {batch.shape}"
                )
                # For logging purposes, we get the last element of the history
                # and convert it to a string
                history: History = batch.view(-1)[0]["next", "history"].prompt
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
                        loss_val = (
                            loss.mean(reduce=True)
                            / cfg.train.gradient_accumulation_steps
                        )

                with timeit("backward_pass"):
                    # Backward pass
                    if (
                        cfg.train.mixed_precision
                        and cfg.train_model.torch_dtype == "float16"
                    ):
                        scaler = GradScaler(enabled=True)
                        scaler.scale(loss_val).backward()
                    else:
                        loss_val.backward()

                # Optimization step
                if ((global_step + 1) % cfg.train.gradient_accumulation_steps) == 0:
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

                # Clear memory
                del loss_val
                # TODO: do we need this? Does it interfere with other processes?
                # torch.cuda.empty_cache()
                gc.collect()

                # Update metrics
                if (global_step % cfg.train.logging_frequency) == 0:
                    log_training_metrics(
                        wandb_logger=wandb_logger,
                        replay_buffer=replay_buffer,
                        batch=batch,
                        loss=loss,
                        grad_norm=grad_norm,
                        global_step=global_step,
                        data_read_count=data_read_count,
                        collector=collector,
                        start_time=start_time,
                        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                        history_str=history_str,
                    )

                # Checkpointing disabled to prevent disk space issues
                # if (global_step + 1) % cfg.train.checkpoint_frequency == 0:
                #     with timeit("save_checkpoint"):
                #         torchrl_logger.info(
                #             f"Saving checkpoint {(global_step+1) // cfg.train.checkpoint_frequency}..."
                #         )
                #         checkpoint = {
                #             "step": global_step,
                #             "model_state_dict": policy_training.model.state_dict(),
                #             "optimizer_state_dict": optimizer.state_dict(),
                #             "scaler_state_dict": scaler.state_dict(),
                #             "config": dict(cfg),
                #         }
                #         torch.save(checkpoint, checkpoint_dir / f"checkpoint_{global_step:04d}.pt")

        # Update policy weights
        with timeit("update_policy_weights"):
            torchrl_logger.info("Updating policy weights...")
            weight_updater.push_weights(policy_training)
            # TODO: do we need this? Does it interfere with other processes?
            # torch.cuda.empty_cache()
            gc.collect()

        timeit.print(prefix="timeit")
        for key, val in timeit.todict().items():
            wandb_logger.log_scalar(f"timeit/{key}", val)
        timeit.reset()

        if cfg.train.empty_replay_buffer:
            replay_buffer.empty(empty_write_count=False)

    pbar.close()
    collector.shutdown()


@hydra.main(version_base=None, config_path="config", config_name="grpo_gsm8k")
def main(cfg):
    # Force sync mode
    if not cfg.train.sync:
        raise ValueError(
            "grpo-sync.py must run in sync mode (`python grpo-sync.py mode=sync`). Please use grpo-async.py for async mode (`python grpo-async.py mode=async`)."
        )
    if cfg.train.weight_update_frequency is not None:
        raise ValueError("weight_update_frequency must be left empty in sync mode.")

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
    if cfg.train.buffer_size is not None and (
        cfg.train.buffer_size != cfg.train.dialog_turns_per_batch
    ):
        raise ValueError(
            "buffer_size must be equal to dialog_turns_per_batch in sync settings."
        )

    if cfg.train.optim_batch_size % cfg.train.gradient_accumulation_steps != 0:
        raise ValueError(
            "optim_batch_size must be divisible by gradient_accumulation_steps"
        )

    rb = RayReplayBuffer(
        storage=partial(
            LazyStackStorage,
            # Since we cache the values in the queue until we have "repeats" samples,
            # the buffer can be bigger than what the dialog_turns_per_batch (at most repeats * num_envs)
            cfg.train.buffer_size
            if cfg.train.buffer_size
            else cfg.env.repeats * cfg.env.num_envs,
        ),
        sampler=SamplerWithoutReplacement,
        transform_factory=partial(MCAdvantage, grpo_size=cfg.env.repeats, verbose=True),
        batch_size=cfg.train.optim_batch_size // cfg.train.gradient_accumulation_steps,
        remote_config=replay_buffer_config,
    )
    torchrl_logger.info(f"Replay buffer: {rb}")

    # Create remote collector using RayLLMCollector
    collector_config["num_gpus"] = (
        # The ref model will be instantiated within the collector, so we only need to allocate the number of devices for the inference model
        cfg.ref_model.num_devices
    )
    collector_config["num_cpus"] = cfg.ray.collector_config.get("num_cpus", 1)
    torchrl_logger.info(f"Starting collector with {collector_config=}")

    collector = RayLLMCollector(
        env=partial(make_env, cfg, devices=device_config["ref_model_devices"]),
        policy=inference_policy,
        dialog_turns_per_batch=cfg.train.dialog_turns_per_batch,
        total_dialog_turns=cfg.train.total_dialog_turns,
        replay_buffer=rb,
        ray_init_config=None,  # Ray is already initialized
        weight_updater=None,  # We'll create this after getting the remote LLM
        track_policy_version=True,
        remote_config=collector_config,
        sync_iter=cfg.train.sync_iter,
        verbose=False,
        yield_only_last_steps=cfg.env.reasoning,
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
        train_handler.remote(
            rb, cfg, collector, devices=device_config["train_model_devices"]
        )
    )


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    main()
