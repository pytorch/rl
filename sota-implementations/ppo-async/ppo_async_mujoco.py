# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Asynchronous PPO (Stale PPO) for MuJoCo environments.

This script demonstrates async PPO where the collector runs continuously in
the background, feeding data into a replay buffer, while the trainer samples
from the buffer and trains. Because the collector's policy may be several
gradient steps behind the trainer, importance sampling corrections are applied
automatically by ClipPPOLoss (via the stored action_log_prob).

Supports three collection/advantage modes (configured via YAML):

  async_mode=iterate (semi-async):
    Uses `for data in collector:` loop. GAE computed on learner with current
    critic. Training is gated on collector output.

  async_mode=start, advantage_on=worker (fully async, worker GAE):
    Uses `collector.start()` for fully decoupled collection. GAE computed on
    collector workers via postproc with a (stale) critic copy. Uses
    SharedMemWeightSyncScheme to sync both actor and critic to workers.

  async_mode=start, advantage_on=learner (fully async, learner TD(0)):
    Uses `collector.start()` for fully decoupled collection. TD(0) advantage
    computed at training time with the current critic. Only the actor runs on
    collector workers.

Key components:
  - MultiaSyncDataCollector for continuous background collection
  - Replay buffer with StalenessAwareSampler for freshness-weighted sampling
  - Policy version tracking for staleness computation
  - IS diagnostics: ESS, max_ratio, mean_ratio, kl_approx
"""
from __future__ import annotations

import hydra
import torch
import torch.optim

from omegaconf import DictConfig
from torchrl._utils import get_available_device
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    RandomSampler,
    SamplerWithoutReplacement,
    StalenessAwareSampler,
)
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger
from train import train_iterate, train_start
from utils_mujoco import make_ppo_models


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision("high")

    device = (
        torch.device(cfg.optim.device) if cfg.optim.device else get_available_device()
    )
    collect_device = (
        torch.device(cfg.optim.collect_device)
        if cfg.optim.get("collect_device")
        else device
    )
    eval_device = (
        torch.device(cfg.optim.eval_device) if cfg.optim.get("eval_device") else device
    )

    # ── Mode selection ──────────────────────────────────────────────────
    async_mode = cfg.collector.get("async_mode", "iterate")
    advantage_on = cfg.loss.get("advantage_on", "worker")
    if async_mode == "iterate":
        advantage_on = "learner"  # always learner in iterate mode

    # ── Models ──────────────────────────────────────────────────────────
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # ── Loss & advantage ────────────────────────────────────────────────
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )

    # ── Optimizers ──────────────────────────────────────────────────────
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    # ── Sampler ─────────────────────────────────────────────────────────
    sampler_type = cfg.buffer.get("sampler", "staleness")
    if sampler_type == "staleness":
        sampler = StalenessAwareSampler(
            max_staleness=cfg.buffer.max_staleness,
            version_key="policy_version",
        )
    elif sampler_type == "staleness_no_gate":
        sampler = StalenessAwareSampler(
            max_staleness=-1,
            version_key="policy_version",
        )
    elif sampler_type == "random":
        sampler = RandomSampler()
    elif sampler_type == "no_replacement":
        sampler = SamplerWithoutReplacement()
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    # ── Replay buffer ───────────────────────────────────────────────────
    # shared_init=True uses CPU-backed shared memory; storage device must
    # match so samplers don't generate GPU indices for CPU tensors.
    shared = async_mode == "start"
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.buffer.size,
            device="cpu" if shared else device,
            shared_init=shared,
        ),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # ── Logger ──────────────────────────────────────────────────────────
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name(
            "PPO-Async", f"{cfg.logger.exp_name}_{cfg.env.env_name}"
        )
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo_async",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    num_eval_envs = cfg.logger.get("num_eval_envs", 4096)

    # ── Config extraction ───────────────────────────────────────────────
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    cfg_buffer_min_fill = cfg.buffer.min_fill
    cfg_loss_gamma = cfg.loss.gamma
    total_frames = cfg.collector.total_frames

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (total_frames // cfg.collector.frames_per_batch)
        * cfg_loss_ppo_epochs
        * num_mini_batches
    )

    # ── Dispatch to training loop ───────────────────────────────────────
    shared_kwargs = {
        "cfg": cfg,
        "actor": actor,
        "critic": critic,
        "adv_module": adv_module,
        "loss_module": loss_module,
        "optim": optim,
        "sampler": sampler,
        "data_buffer": data_buffer,
        "device": device,
        "collect_device": collect_device,
        "logger": logger,
        "eval_device": eval_device,
        "num_eval_envs": num_eval_envs,
        "cfg_optim_anneal_lr": cfg_optim_anneal_lr,
        "cfg_optim_lr": cfg_optim_lr,
        "cfg_loss_anneal_clip_eps": cfg_loss_anneal_clip_eps,
        "cfg_loss_clip_epsilon": cfg_loss_clip_epsilon,
        "cfg_optim_max_grad_norm": cfg_optim_max_grad_norm,
        "cfg_buffer_min_fill": cfg_buffer_min_fill,
        "test_interval": cfg.logger.test_interval,
        "total_frames": total_frames,
        "total_network_updates": total_network_updates,
    }

    if async_mode == "start":
        train_start(
            **shared_kwargs,
            advantage_on=advantage_on,
            cfg_loss_gamma=cfg_loss_gamma,
        )
    else:
        train_iterate(**shared_kwargs, cfg_loss_ppo_epochs=cfg_loss_ppo_epochs)


if __name__ == "__main__":
    main()
