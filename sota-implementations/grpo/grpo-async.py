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
from torchrl.data.llm.chat import History
from torchrl.record.loggers.wandb import WandbLogger

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for async training. Please install ray with `pip install ray`."
    )
import torch
import tqdm

from grpo_utils import (
    compute_device_allocation,
    get_inference_model,
    get_ref_model,
    get_tokenizer,
    get_train_model,
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
from torchrl.data import LazyStackStorage, ReplayBuffer
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.envs.llm import GSM8KEnv, KLRewardTransform
from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
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


def make_env(cfg: DictConfig, devices: list[int] | None = None):
    """Create the environment with proper device allocation.

    Args:
        cfg: The configuration object

    Returns:
        The configured environment
    """
    # Create reference model with proper device allocation
    # For the collector actor, we want inference_model devices first, then ref_model devices
    train_tokenizer = get_tokenizer(cfg)

    # Get device information
    num_inf_devices = cfg.inference_model.num_devices
    num_ref_devices = cfg.ref_model.num_devices
    num_inf_devices + num_ref_devices

    # Create a new config with adjusted device assignments
    ref_cfg = DictConfig(dict(cfg))
    ref_model = get_ref_model(ref_cfg, train_tokenizer, devices=devices)

    # Setup environment
    if cfg.env.dataset == "gsm8k":
        env = GSM8KEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs,
        )
    else:  # ifeval
        env = IFEvalEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs,
        )

    # Pass device directly to KLRewardTransform - Since, for Ray, the local device is always 0
    # we can just use 0 here.
    device = torch.device("cuda:0")
    env = env.append_transform(
        KLRewardTransform(
            actor=ref_model,
            coef=cfg.train.kl_to_ref_coeff,
            add_to_reward=not cfg.train.kl_coef_in_loss,
            device=device,
        )
    )
    return env


def train(
    replay_buffer: ReplayBuffer,
    cfg: DictConfig,
    collector: RayLLMCollector,
    devices: list[int] | None = None,
):
    """Main training loop for GRPO async.

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
    policy_training, train_tokenizer = get_train_model(cfg, devices=devices)
    train_device = devices[0]  # Use first device for batch processing

    # Setup loss function
    loss_fn = GRPOLoss(
        actor_network=policy_training,
        kl_to_ref_coeff=cfg.train.kl_to_ref_coeff if cfg.train.kl_coef_in_loss else 0.0,
        kl_to_inference_coeff=cfg.train.kl_to_inference_coeff,
        entropy_coeff=cfg.train.entropy_coeff,
        device=train_device,
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

    # Start collector
    collector.start()

    # Wait for initial data
    while not replay_buffer.write_count:
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
    metrics = {}  # Initialize metrics dict
    grad_norm = 0.0  # Initialize grad_norm
    data_read_count = 0
    start_time = time.time()

    for step in range(total_steps):
        pbar.update(1)
        pbar.set_description(f"Step {step}, writes: {replay_buffer.write_count}")

        with timeit("sampling"):
            # Sample batch and move to device
            batch = replay_buffer.sample(cfg.train.optim_batch_size).to(train_device)
            # For logging purposes, we get the last element of the history
            # and convert it to a string
            history: History = batch.view(-1)[0]["next", "history"]
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
                    loss.mean(reduce=True) / cfg.train.gradient_accumulation_steps
                )

        with timeit("backward_pass"):
            # Backward pass
            if cfg.train.mixed_precision and cfg.train_model.torch_dtype == "float16":
                scaler = GradScaler(enabled=True)
                scaler.scale(loss_val).backward()
            else:
                loss_val.backward()

        # Optimization step
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

        # Update metrics
        if (step % cfg.train.logging_frequency) == 0:
            with torch.no_grad():
                rb_content = replay_buffer[:]
                batch_policy_version = batch["next", "policy_version"].view(-1).min()
                batch_policy_age = collector.policy_version - batch_policy_version
                metrics = {
                    "reward from buffer": float(
                        torch.cat(
                            rb_content.get(("next", "reward"), as_list=True)
                        ).mean()
                    ),
                    "kl_penalty (inference to ref) from buffer": float(
                        torch.cat(
                            rb_content.get(("next", "kl_penalty"), as_list=True)
                        ).mean()
                    ),
                    "seq_length from buffer": float(
                        torch.tensor(
                            [
                                t.numel()
                                for t in rb_content.get("tokens_response", as_list=True)
                            ],
                            dtype=torch.float,
                        ).mean()
                    ),
                    "ESS, from loss": float(loss.ESS),
                    "loss_objective, from loss": float(loss.loss_objective),
                    "clip_fraction, from loss": float(loss.clip_fraction),
                    "kl_approx (train to inference), from loss": float(loss.kl_approx),
                    "kl_to_inference (train to inference - differentiable), from loss": float(
                        loss.kl_to_inference.mean()
                    ),
                    "kl_to_ref, from loss": float(loss.kl_to_ref.mean()),
                    "loss_kl_to_inference, from loss": float(
                        loss.loss_kl_to_inference.mean()
                    ),
                    "loss_kl_to_ref, from loss": float(loss.loss_kl_to_ref.mean()),
                    "entropy loss, from loss": float(loss.loss_entropy.mean()),
                    "grad_norm": float(grad_norm)
                    if step % cfg.train.gradient_accumulation_steps == 0
                    else metrics.get("grad_norm", 0.0),
                    "write_count, from buffer": int(replay_buffer.write_count),
                    # how many gradient steps per write
                    "gradient_step_throughput (gradient step per write)": float(
                        step / replay_buffer.write_count
                    ),
                    # how many optim steps per write
                    "optim_step_throughput (optim step per write)": float(
                        (step // cfg.train.gradient_accumulation_steps)
                        / replay_buffer.write_count
                    ),
                    "data_read_count (total)": data_read_count,
                    "current_policy_version (collector)": collector.policy_version,
                    # FIXME: Assume batch is a single trajectory
                    # FIXME: The addition of the transform after the env instantiation + _shuttle creation
                    #  is messed up - we need the next data
                    "batch_policy_version (sampled batch)": batch_policy_version,
                    "batch_policy_age (sampled batch)": batch_policy_age,
                    "throughput (steps per second)": float(
                        step / (time.time() - start_time)
                    ),
                }
                for name, value in metrics.items():
                    wandb_logger.log_scalar(name, value)
                wandb_logger.log_str("history", history_str, step=step)

        # Update policy weights
        if step % cfg.train.weight_update_frequency == 0:
            with timeit("update_policy_weights"):
                torchrl_logger.info("Updating policy weights...")
                weight_updater.push_weights(policy_training)
                torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    collector.shutdown()


@hydra.main(version_base=None, config_path="config", config_name="grpo_gsm8k")
def main(cfg):
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
    rb = RayReplayBuffer(
        storage=partial(
            LazyStackStorage,
            cfg.train.buffer_size
            if cfg.train.buffer_size
            else cfg.train.steps_per_batch,
        ),
        transform_factory=partial(MCAdvantage, grpo_size=cfg.env.repeats),
        batch_size=cfg.train.optim_batch_size,
        remote_config=replay_buffer_config,
    )
    torchrl_logger.info(f"Replay buffer: {rb}")

    # Create remote collector using RayLLMCollector
    collector_config["num_gpus"] = (
        # The ref model will be instantiated within the collector, so we only need to allocate the number of devices for the inference model
        cfg.ref_model.num_devices
    )
    torchrl_logger.info(f"Starting collector with {collector_config=}")

    collector = RayLLMCollector(
        env=partial(make_env, cfg, devices=device_config["ref_model_devices"]),
        policy=inference_policy,
        dialog_turns_per_batch=cfg.train.steps_per_batch,
        total_dialog_turns=cfg.train.total_dialog_turns,
        replay_buffer=rb,
        ray_init_config=None,  # Ray is already initialized
        weight_updater=None,  # We'll create this after getting the remote LLM
        track_policy_version=True,
        remote_config=collector_config,
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
