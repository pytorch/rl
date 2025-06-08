# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO: Generalized Reward-Conditioned Policy Optimization

This module implements GRPO training for language models.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import hydra
import torch
import tqdm
from grpo_utils import get_inference_model, get_ref_model, get_train_model
from omegaconf import DictConfig
from tensordict import set_list_to_stack, TensorDict
from torch.cuda.amp import GradScaler
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.llm import LLMCollector
from torchrl.collectors.llm.weight_update.vllm import vLLMUpdater
from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs.llm import GSM8KEnv, KLRewardTransform
from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage
from torchrl.record import WandbLogger


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

    # Set all loggers to WARNING by default
    logging.getLogger().setLevel(logging.WARNING)
    # But keep torchrl at INFO
    logging.getLogger("torchrl").setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="grpo_gsm8k")
def train(cfg: DictConfig) -> None:
    """Main training loop.

    Args:
        cfg: Hydra configuration object
    """
    import ray

    ray.init()

    # Initialize models using config-based device allocation
    torchrl_logger.info("Initializing models...")
    torchrl_logger.info("Inference model...")
    policy = get_inference_model(cfg)
    torchrl_logger.info("Training model...")
    policy_training, train_tokenizer = get_train_model(cfg)
    torchrl_logger.info("Reference model...")
    ref_model = get_ref_model(cfg, train_tokenizer)
    torchrl_logger.info("Done initializing models")

    # Get reference model device for KL transform
    ref_devices = cfg.ref_model.get("devices", [2])
    ref_device = ref_devices[0]  # Use first device for KL transform

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

    env = env.append_transform(
        KLRewardTransform(
            actor=ref_model,
            coef=cfg.policy.kl_coef,
            device=torch.device(f"cuda:{ref_device}"),
            add_to_reward=not cfg.train.kl_coef_in_loss,
        )
    )

    # Setup replay buffer
    rb = ReplayBuffer(
        storage=LazyStackStorage(cfg.train.steps_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.optim_batch_size,
    )
    rb.append_transform(MCAdvantage(grpo_size=cfg.env.repeats))

    # Setup collector
    model_metadata = {
        k: (v.dtype, v.shape)
        for k, v in policy_training.model.merge_and_unload().state_dict().items()
    }
    updater = vLLMUpdater(
        master_address=None,
        master_port=None,
        model_metadata=model_metadata,
    )

    collector = LLMCollector(
        env,
        policy=policy,
        dialog_turns_per_batch=cfg.train.steps_per_batch,
        total_dialog_turns=cfg.train.total_dialog_turns,
        weight_updater=updater,
    )
    updater.maybe_init_group()

    # Get training device for batch processing
    train_devices = cfg.train_model.get("devices", [0])
    train_device = train_devices[0]  # Use first device for batch processing

    # Setup loss and optimizer
    loss_fn = GRPOLoss(
        actor_network=policy_training,
        kl_to_ref_coeff=cfg.policy.kl_coef if cfg.train.kl_coef_in_loss else 0.0,
    )
    if cfg.model.compile:
        loss_fn = torch.compile(loss_fn)

    optim = getattr(torch.optim, cfg.train.optimizer.name)(
        policy_training.model.parameters(),
        lr=cfg.train.optimizer.lr,
        foreach=len(train_devices) == 1,
    )

    # Only use GradScaler with float16, not with bfloat16
    use_grad_scaling = (
        cfg.train.mixed_precision and cfg.train_model.torch_dtype == "float16"
    )
    scaler = GradScaler(enabled=use_grad_scaling)

    # Setup logging
    experiment_name = (
        cfg.logging.experiment_name
        or f"{cfg.model.name.split('/')[-1]}_{cfg.env.dataset}"
    )
    wandb_logger = WandbLogger(exp_name=experiment_name, config=dict(cfg))

    # Create checkpoint directory
    checkpoint_dir = Path(cfg.logging.checkpoint_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize weights
    torchrl_logger.info("Initializing weights...")
    # Ensure weights are on the first training device for vLLM
    state_dict = TensorDict(policy_training.model.merge_and_unload().state_dict()).to(
        f"cuda:{train_device}"
    )
    collector.update_policy_weights_(state_dict, worker_ids=[0])
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

    # Training loop
    for i, trajs in enumerate(collector):
        torchrl_logger.info(f"Collected batch {i}: {len(trajs)} trajectories")

        # Clear memory after collection
        torch.cuda.empty_cache()
        gc.collect()

        trajs = trajs.reshape(-1)
        rb.extend(trajs)

        # Calculate metrics
        with torch.no_grad():
            reward = torch.cat(rb[:].get(("next", "reward"), as_list=True)).mean()
            kl_penalty = torch.cat(
                rb[:].get(("next", "kl_penalty"), as_list=True)
            ).mean()
            seq_length = torch.tensor(
                [t.numel() for t in rb[:].get("tokens_response", as_list=True)],
                dtype=torch.float,
            ).mean()
            metrics = {
                "reward": float(reward),
                "kl_penalty": float(kl_penalty),
                "seq_length": float(seq_length),
            }

            # Clear memory after metrics calculation
            del trajs
            torch.cuda.empty_cache()
            gc.collect()

        if not reward:
            torchrl_logger.info("No reward - skipping batch")
            torch.cuda.empty_cache()
            continue

        # Training epochs
        for epoch in range(cfg.train.epochs):
            torchrl_logger.info(f"Epoch {epoch}")
            pbar = tqdm.tqdm(total=len(rb) // cfg.train.optim_batch_size)

            for batch_idx, batch in enumerate(rb):
                # Move batch to device and clear CPU memory
                batch = batch.to(train_device)
                torch.cuda.empty_cache()

                pbar.update(1)

                # Forward pass
                with torch.amp.autocast(
                    "cuda",
                    enabled=cfg.train.mixed_precision,
                    dtype=getattr(torch, cfg.train_model.torch_dtype),
                ):
                    loss = loss_fn(batch)
                    loss_val = loss.mean(reduce=True)
                    loss_val = loss_val / cfg.train.gradient_accumulation_steps

                    # Store metrics before clearing memory
                    metrics.update(
                        {
                            "ESS": float(loss.ESS),
                            "loss_objective": float(loss.loss_objective),
                            "clip_fraction": float(loss.clip_fraction),
                            "kl_approx": float(loss.kl_approx),
                            "entropy": float(loss.loss_entropy.mean()),
                            "kl_to_ref": float(loss.kl_to_ref.mean()),
                            "loss_kl_to_ref": float(loss.loss_kl_to_ref.mean()),
                        }
                    )

                # Clear intermediate tensors
                del loss
                torch.cuda.empty_cache()

                # Backward pass with gradient scaling only for float16
                if use_grad_scaling:
                    scaler.scale(loss_val).backward()
                else:
                    loss_val.backward()

                if (batch_idx + 1) % cfg.train.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if use_grad_scaling:
                        scaler.unscale_(optim)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_training.model.parameters(),
                        cfg.train.optimizer.clip_grad_norm,
                    )

                    # Optimizer step with or without scaler
                    if use_grad_scaling:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()

                    optim.zero_grad()

                    # Clear memory after optimization step
                    del loss_val
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Log metrics
                    for name, value in metrics.items():
                        wandb_logger.log_scalar(name, value)
                    wandb_logger.log_scalar("grad_norm", float(grad_norm))

            pbar.close()
            # Clear memory after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        # Update policy weights
        torchrl_logger.info("Updating policy weights...")
        # Ensure weights are on the first training device for vLLM
        state_dict = TensorDict(
            policy_training.model.merge_and_unload().state_dict()
        ).to(f"cuda:{train_device}")
        collector.update_policy_weights_(
            policy_weights=state_dict,
            worker_ids=[0],
        )
        del state_dict

        # Clear memory after weight update
        torch.cuda.empty_cache()
        gc.collect()

        # Save checkpoint
        if (i + 1) % cfg.logging.checkpoint_frequency == 0:
            torchrl_logger.info(
                f"Saving checkpoint {(i+1) // cfg.logging.checkpoint_frequency}..."
            )
            checkpoint = {
                "batch": i,
                "model_state_dict": policy_training.model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": dict(cfg),
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_{i:04d}.pt")


if __name__ == "__main__":
    # Setup environment
    setup_environment()
    train()
