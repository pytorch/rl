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
automatically by ClipPPOLoss (via the stored sample_log_prob).

Supports three collection/advantage modes (configured via YAML):

  async_mode=iterate (semi-async):
    Uses `for data in collector:` loop. GAE computed on learner with current
    critic. Training is gated on collector output.

  async_mode=start, advantage_on=worker (fully async, worker GAE):
    Uses `collector.start()` for fully decoupled collection. GAE computed on
    collector workers via postproc with a (stale) critic copy. Uses
    ActorCriticWrapper so update_policy_weights_ syncs both actor and critic.

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

import multiprocessing

import hydra

import torch
from torchrl._utils import get_available_device


# ---------------------------------------------------------------------------
# Postproc callables (module-level for pickle compatibility with spawn)
# ---------------------------------------------------------------------------


class _ActorWithCritic(torch.nn.Module):
    """Wrapper that holds both actor and critic but only runs the actor.

    This ensures ``update_policy_weights_()`` syncs both modules to workers,
    while keeping per-step collection lightweight (actor only). The critic is
    used by the postproc for batched GAE computation.
    """

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, td):
        return self.actor(td)


class _WorkerGAEPostproc:
    """Postproc for worker GAE mode: critic + GAE, flatten, stamp policy_version."""

    def __init__(self, adv_module, version_counter):
        self.adv_module = adv_module
        self.version_counter = version_counter

    def __call__(self, data):
        with torch.no_grad():
            data = self.adv_module(data)
        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],),
            float(self.version_counter.value),
            device=data_flat.device,
        )
        return data_flat


def _make_eval_env(env_name, device, from_pixels, num_eval_envs):
    """Env factory for the process-based Evaluator (batched ParallelEnv)."""
    from torchrl.envs import ParallelEnv
    from utils_mujoco import make_env

    return ParallelEnv(
        num_eval_envs,
        lambda: make_env(env_name, device=device, from_pixels=from_pixels),
    )


def _make_eval_policy(env_name, device, env):
    """Policy factory for the process-based Evaluator."""
    from utils_mujoco import make_ppo_models

    return make_ppo_models(env_name, device=device)[0]


class _LearnerPostproc:
    """Postproc for start+learner mode: flatten, stamp policy_version."""

    def __init__(self, version_counter):
        self.version_counter = version_counter

    def __call__(self, data):
        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],),
            float(self.version_counter.value),
            device=data_flat.device,
        )
        return data_flat


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    import torch.optim

    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import (
        RandomSampler,
        SamplerWithoutReplacement,
        StalenessAwareSampler,
    )
    from torchrl.objectives import ClipPPOLoss, group_optimizers
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_mujoco import make_ppo_models

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
    # shared_init=True allows workers to initialize the storage schema
    # when using collector.start() mode (local_init_rb=True).
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.buffer.size,
            device=device,
            shared_init=(async_mode == "start"),
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
        logger_video = cfg.logger.video
    else:
        logger_video = False

    # ── Async evaluator (separate process — no GIL contention) ─────────
    from functools import partial

    from torchrl.collectors import Evaluator

    num_eval_envs = cfg.logger.get("num_eval_envs", 16)
    evaluator = Evaluator(
        env=partial(
            _make_eval_env, cfg.env.env_name, eval_device, logger_video, num_eval_envs
        ),
        policy_factory=partial(_make_eval_policy, cfg.env.env_name, eval_device),
        max_steps=10_000,
        logger=logger,
        log_prefix="eval",
        backend="process",
    )

    # ── Config extraction ───────────────────────────────────────────────
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
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

    # ====================================================================
    # Dispatch to the appropriate training loop
    # ====================================================================

    # Shared kwargs for both training loops
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
        "evaluator": evaluator,
        "cfg_loss_ppo_epochs": cfg_loss_ppo_epochs,
        "cfg_optim_anneal_lr": cfg_optim_anneal_lr,
        "cfg_optim_lr": cfg_optim_lr,
        "cfg_loss_anneal_clip_eps": cfg_loss_anneal_clip_eps,
        "cfg_loss_clip_epsilon": cfg_loss_clip_epsilon,
        "cfg_logger_test_interval": cfg_logger_test_interval,
        "cfg_optim_max_grad_norm": cfg_optim_max_grad_norm,
        "cfg_buffer_min_fill": cfg_buffer_min_fill,
        "total_frames": total_frames,
        "total_network_updates": total_network_updates,
    }

    if async_mode == "start":
        _train_start(
            **shared_kwargs,
            advantage_on=advantage_on,
            cfg_loss_gamma=cfg_loss_gamma,
        )
    else:
        _train_iterate(**shared_kwargs)

    evaluator.shutdown()


# ========================================================================
# Training loop: start mode (fully async)
# ========================================================================


def _train_start(
    *,
    cfg,
    actor,
    critic,
    adv_module,
    loss_module,
    optim,
    sampler,
    data_buffer,
    advantage_on,
    device,
    collect_device,
    logger,
    evaluator,
    cfg_loss_ppo_epochs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_logger_test_interval,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    cfg_loss_gamma,
    total_frames,
    total_network_updates,
):
    import time

    import tqdm
    from torchrl.collectors import MultiaSyncDataCollector
    from utils_mujoco import make_env

    # Shared version counter (readable by workers via postproc)
    version_counter = multiprocessing.Value("i", 0)

    # Build collector policy and postproc based on advantage_on mode
    if advantage_on == "worker":
        # Workers compute GAE via postproc; _ActorWithCritic ensures
        # update_policy_weights_() syncs both actor and critic to workers,
        # but only the actor runs per-step during collection.
        collector_policy = _ActorWithCritic(actor, critic)
        postproc = _WorkerGAEPostproc(adv_module, version_counter)
    else:
        # Workers only collect; TD(0) advantage on learner at training time
        collector_policy = actor
        postproc = _LearnerPostproc(version_counter)

    collector = MultiaSyncDataCollector(
        create_env_fn=[make_env(cfg.env.env_name, device)] * cfg.collector.num_workers,
        policy=collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=device,
        max_frames_per_traj=-1,
        replay_buffer=data_buffer,
        postproc=postproc,
        local_init_rb=True,
    )

    # Start collection — workers fill buffer independently
    collector.start()

    policy_version = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()
    last_write_count = 0
    last_test_frames = 0
    last_trained_wc = 0  # gate training on new data arriving

    while True:
        # Track collection progress
        current_wc = data_buffer.write_count
        if current_wc > last_write_count:
            pbar.update(current_wc - last_write_count)
            last_write_count = current_wc
        if current_wc >= total_frames:
            break

        # Wait for new data before training (prevents consumer_version
        # from racing ahead of what's in the buffer)
        if current_wc <= last_trained_wc or len(data_buffer) < cfg_buffer_min_fill:
            time.sleep(0.05)
            continue
        last_trained_wc = current_wc

        metrics_to_log = {}

        # Train ppo_epochs gradient steps, then push weights
        for _epoch in range(cfg_loss_ppo_epochs):
            batch, info = data_buffer.sample(return_info=True)
            batch_staleness = info.get("staleness")

            # Compute TD(0) advantage on learner if needed
            if advantage_on == "learner":
                with torch.no_grad():
                    state_value = critic(batch).get("state_value")
                    next_state_value = critic(batch.get("next")).get("state_value")
                    reward = batch.get(("next", "reward"))
                    done = batch.get(("next", "done")).float()
                    value_target = (
                        reward + cfg_loss_gamma * (1 - done) * next_state_value
                    )
                    advantage = value_target - state_value
                    batch.set("advantage", advantage)
                    batch.set("value_target", value_target)

            # LR annealing based on collection progress
            alpha = 1.0
            if cfg_optim_anneal_lr:
                alpha = 1 - (current_wc / total_frames)
                for group in optim.param_groups:
                    group["lr"] = cfg_optim_lr * alpha
            if cfg_loss_anneal_clip_eps:
                loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

            # Forward / backward
            optim.zero_grad(set_to_none=True)
            loss = loss_module(batch)
            total_loss = (
                loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
            )
            optim.step()
            num_network_updates += 1

        # Push updated weights to workers and bump version
        collector.update_policy_weights_()
        policy_version += 1
        version_counter.value = policy_version
        if hasattr(sampler, "consumer_version"):
            sampler.consumer_version = policy_version

        # Log training metrics
        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
                "train/lr": alpha * cfg_optim_lr.item(),
                "train/clip_epsilon": (alpha * cfg_loss_clip_epsilon)
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
                "train/ESS": loss["ESS"].item(),
                "train/clip_fraction": loss["clip_fraction"].item(),
                "train/kl_approx": loss["kl_approx"].item(),
                "train/max_ratio": loss["max_ratio"].item(),
                "train/mean_ratio": loss["mean_ratio"].item(),
                "staleness/consumer_version": getattr(sampler, "consumer_version", 0),
                "staleness/policy_version": policy_version,
                "staleness/batch_mean": batch_staleness.float().mean().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_max": batch_staleness.max().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_min": batch_staleness.min().item()
                if batch_staleness is not None
                else 0,
                "buffer/size": len(data_buffer),
                "buffer/write_count": current_wc,
                "collector/collected_frames": current_wc,
            }
        )

        # Async eval: trigger at intervals, poll for results (non-blocking)
        if (current_wc // cfg_logger_test_interval) > (
            last_test_frames // cfg_logger_test_interval
        ):
            if not evaluator.pending:
                evaluator.trigger_eval(actor, step=current_wc)
            last_test_frames = current_wc
        eval_metrics = evaluator.poll()
        if eval_metrics is not None:
            metrics_to_log.update(eval_metrics)

        if logger:
            logger.log_metrics(metrics_to_log, current_wc)

    pbar.close()
    collector.shutdown()

    elapsed = time.time() - start_time
    print(  # noqa: T001
        f"Training took {elapsed:.2f} seconds (mode=start, advantage_on={advantage_on})"
    )


# ========================================================================
# Training loop: iterate mode (semi-async, current behaviour)
# ========================================================================


def _train_iterate(
    *,
    cfg,
    actor,
    critic,
    adv_module,
    loss_module,
    optim,
    sampler,
    data_buffer,
    device,
    collect_device,
    logger,
    evaluator,
    cfg_loss_ppo_epochs,
    cfg_optim_anneal_lr,
    cfg_optim_lr,
    cfg_loss_anneal_clip_eps,
    cfg_loss_clip_epsilon,
    cfg_logger_test_interval,
    cfg_optim_max_grad_norm,
    cfg_buffer_min_fill,
    total_frames,
    total_network_updates,
):
    import time

    import tqdm
    from torchrl.collectors import MultiaSyncDataCollector
    from utils_mujoco import make_env

    # Use _ActorWithCritic so update_policy_weights_ syncs both actor and
    # critic to workers. Only the actor runs per env step; the critic runs
    # batched in the postproc for GAE.
    collector_policy = _ActorWithCritic(actor, critic)

    collector = MultiaSyncDataCollector(
        create_env_fn=[make_env(cfg.env.env_name, device)] * cfg.collector.num_workers,
        policy=collector_policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames,
        device=device,
        max_frames_per_traj=-1,
        update_at_each_batch=True,
        postproc=adv_module,
    )

    policy_version = 0
    collected_frames = 0
    num_network_updates = 0
    pbar = tqdm.tqdm(total=total_frames)
    start_time = time.time()

    for i, data in enumerate(collector):

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Train episode rewards
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        # Data already has GAE advantages from postproc; flatten and stamp version
        data_flat = data.reshape(-1)
        data_flat["policy_version"] = torch.full(
            (data_flat.shape[0],), float(policy_version), device=device
        )
        data_buffer.extend(data_flat)

        # Warmup
        if len(data_buffer) < cfg_buffer_min_fill:
            if logger:
                metrics_to_log["buffer/size"] = len(data_buffer)
                logger.log_metrics(metrics_to_log, collected_frames)
            continue

        # Train
        for _epoch in range(cfg_loss_ppo_epochs):
            batch, info = data_buffer.sample(return_info=True)
            batch_staleness = info.get("staleness")

            alpha = 1.0
            if cfg_optim_anneal_lr:
                alpha = 1 - (num_network_updates / total_network_updates)
                for group in optim.param_groups:
                    group["lr"] = cfg_optim_lr * alpha
            if cfg_loss_anneal_clip_eps:
                loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)

            optim.zero_grad(set_to_none=True)
            loss = loss_module(batch)
            total_loss = (
                loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
            )
            optim.step()
            num_network_updates += 1

        # Push weights & bump version
        collector.update_policy_weights_()
        policy_version += 1
        if hasattr(sampler, "consumer_version"):
            sampler.consumer_version = policy_version

        metrics_to_log.update(
            {
                "train/loss_objective": loss["loss_objective"].item(),
                "train/loss_critic": loss["loss_critic"].item(),
                "train/loss_entropy": loss["loss_entropy"].item(),
                "train/lr": alpha * cfg_optim_lr.item(),
                "train/clip_epsilon": (alpha * cfg_loss_clip_epsilon)
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
                "train/ESS": loss["ESS"].item(),
                "train/clip_fraction": loss["clip_fraction"].item(),
                "train/kl_approx": loss["kl_approx"].item(),
                "train/max_ratio": loss["max_ratio"].item(),
                "train/mean_ratio": loss["mean_ratio"].item(),
                "staleness/consumer_version": getattr(sampler, "consumer_version", 0),
                "staleness/policy_version": policy_version,
                "staleness/batch_mean": batch_staleness.float().mean().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_max": batch_staleness.max().item()
                if batch_staleness is not None
                else 0,
                "staleness/batch_min": batch_staleness.min().item()
                if batch_staleness is not None
                else 0,
                "buffer/size": len(data_buffer),
                "collector/collected_frames": collected_frames,
            }
        )

        # Async eval: trigger at intervals, poll for results (non-blocking)
        if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
            i * frames_in_batch
        ) // cfg_logger_test_interval:
            if not evaluator.pending:
                evaluator.trigger_eval(actor, step=collected_frames)
        eval_metrics = evaluator.poll()
        if eval_metrics is not None:
            metrics_to_log.update(eval_metrics)

        if logger:
            logger.log_metrics(metrics_to_log, collected_frames)

    pbar.close()
    collector.shutdown()

    elapsed = time.time() - start_time
    print(f"Training took {elapsed:.2f} seconds (mode=iterate)")  # noqa: T001


if __name__ == "__main__":
    main()
