# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 training with native replay and configurable environments.

Supports vector and image observations and continuous or discrete actions.
The Walker preset reproduces a pinned JAX configuration and reporting axis.

Usage::

    python sota-implementations/dreamer_v3/train.py \\
        collector.total_frames=5000 logger.eval_every=500

    python sota-implementations/dreamer_v3/train.py \\
        --config-name=config_dmc_walker
"""

from __future__ import annotations

import copy
import functools as ft
import math
import signal
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple

import hydra
import torch
from dreamer_v3_agent import (
    build_actor,
    build_continuation_model,
    build_imagination_model,
    build_mb_env,
    build_real_world_actor,
    build_value,
    build_world_model,
    DreamerV3BehaviorPolicySync,
    make_env,
    make_primed_env,
)
from dreamer_v3_replay import collector_action_budget, replay_context_update
from dreamer_v3_utils import (
    append_jsonl,
    eval_episode_reward,
    latent_state_dim,
    LEARNER_RNG_STREAM,
    plot_enabled,
    REPLAY_RNG_STREAM,
    resolve_compile_settings,
    save_run_plot,
    stream_seed,
    training_episode_returns,
)
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.checkpoint import Checkpoint, CheckpointRotation, GlobalRNGState
from torchrl.collectors import AsyncBatchedCollector, Collector
from torchrl.data import (
    LazyTensorStorage,
    OneHot,
    ReplayBufferEnsemble,
    SliceSampler,
    StreamingSliceSampler,
    TensorDictReplayBuffer,
    TensorDictRoundRobinWriter,
)
from torchrl.envs import SelectTransform, SerialEnv
from torchrl.envs.utils import ExplorationType
from torchrl.modules import DreamerV3SeededPolicy
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServerConfig,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3Loss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.record.loggers import get_logger
from torchrl.trainers.algorithms import (
    DreamerV3OptimizationStepper,
    DreamerV3Optimizer,
    DreamerV3UpdateRatio,
)
from torchrl.trainers.algorithms.configs.common import _normalize_hydra_key


class _ElapsedTimer:
    """Include elapsed time from the process that wrote a checkpoint."""

    def __init__(self, timer, offset: float = 0.0):
        self.timer = timer
        self.offset = offset

    def elapsed(self) -> float:
        return self.offset + self.timer.elapsed()


class _ShutdownRequest:
    """Handle termination after completing the current collection/update batch."""

    def __init__(self):
        self.signal_number: int | None = None

    def __call__(self, signal_number: int, _frame) -> None:
        self.signal_number = signal_number


def _resolve_resume_path(requested: str | None) -> Path | None:
    if not requested:
        return None
    candidate = Path(requested).expanduser().resolve()
    if Checkpoint.is_checkpoint(candidate):
        return candidate
    if candidate.is_dir():
        latest = CheckpointRotation(candidate, keep_last=1).latest()
        if latest is not None:
            return latest
    raise FileNotFoundError(f"No checkpoint was found at {candidate}.")


class _RunLogger:
    """Write consistent records to JSONL and a selected TorchRL logger."""

    def __init__(
        self, cfg: DictConfig, jsonl_path: Path | None, state: dict | None = None
    ):
        self.jsonl_path = jsonl_path
        self.milestone_names = list(cfg.env.milestone_names)
        self.backend = cfg.logger.backend
        if cfg.logger.base_url and self.backend != "wandb":
            raise ValueError(
                "logger.base_url is only supported with logger.backend='wandb'."
            )
        kwargs = {}
        if self.backend == "wandb":
            kwargs["wandb_kwargs"] = {
                "project": cfg.logger.project,
                "entity": cfg.logger.entity,
                "group": cfg.logger.group,
                "tags": list(cfg.logger.tags) or None,
                "mode": cfg.logger.mode,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.logger.base_url:
                kwargs["wandb_kwargs"]["base_url"] = cfg.logger.base_url
        self.logger = get_logger(
            self.backend,
            logger_name=cfg.logger.log_dir,
            experiment_name=cfg.logger.exp_name or f"dreamer_v3_{cfg.env.name}",
            state_dict=state or None,
            **kwargs,
        )
        if self.backend == "wandb":
            self.logger.experiment.define_metric("environment_steps")
            self.logger.experiment.define_metric("*", step_metric="environment_steps")

    def log(self, record: dict[str, object]) -> None:
        append_jsonl(self.jsonl_path, record)
        if self.logger is None:
            return
        kind = record.get("type", "run")
        step = record.get("environment_steps", record.get("total_environment_steps"))
        payload = {}
        for key, value in record.items():
            if key in ("type", "environment_steps"):
                continue
            if key == "milestones":
                if len(value) != len(self.milestone_names):
                    raise ValueError(
                        "env.milestone_names must match the milestone vector."
                    )
                for name, flag in zip(self.milestone_names, value):
                    payload[f"{kind}/obtained_{name}"] = float(flag)
            elif isinstance(value, (bool, int, float)):
                payload[f"{kind}/{key}"] = value
        if self.backend == "wandb":
            payload["environment_steps"] = step
            self.logger.experiment.log(payload)
        else:
            for key, value in payload.items():
                self.logger.log_scalar(key, value, step=step)

    def finish(self) -> None:
        if self.logger is not None:
            self.logger.close()


class _Learner(NamedTuple):
    world_model: TensorDictModuleBase
    model_loss: DreamerV3ModelLoss
    actor_loss: DreamerV3ActorLoss
    value_loss: DreamerV3ValueLoss
    value_target_updater: SoftUpdate
    optimizer: DreamerV3Optimizer
    real_world_actor: TensorDictModuleBase


def _make_learner_update(
    cfg: DictConfig,
    device: torch.device,
    learner: _Learner,
    *,
    cudagraph_warmup: int = 5,
) -> DreamerV3OptimizationStepper:
    """Assemble public learner components from the experiment configuration."""
    loss_module = DreamerV3Loss(
        learner.model_loss,
        learner.actor_loss,
        learner.value_loss,
        replay_value_loss_weight=cfg.optimization.replay_value_loss_weight,
        continuation_horizon=cfg.optimization.continuation_horizon,
        lmbda=cfg.optimization.lmbda,
    )
    settings = resolve_compile_settings(cfg, device)
    return DreamerV3OptimizationStepper(
        loss_module,
        learner.optimizer,
        learner.value_target_updater,
        compile_train_step=settings.train_step,
        compile_mode=settings.mode,
        cudagraph=settings.cudagraph,
        # Inside the compiled step only the scan or the explicit loop exist.
        rssm_scan_unroll=settings.scan_unroll if settings.rssm == "scan" else None,
        warmup_steps=cudagraph_warmup if settings.cudagraph else 1,
        mixed_precision=cfg.optimization.mixed_precision and device.type == "cuda",
    )


def _learner_metrics(losses: TensorDictBase) -> torch.Tensor:
    """Select the six loss series recorded by this experiment."""
    return torch.stack(
        (
            losses["loss_model_dynamic"] + losses["loss_model_representation"],
            losses["loss_model_reco"],
            losses["loss_model_reward"],
            losses["loss_actor"],
            losses["loss_value"],
            losses["replay_value"],
        )
    )


def _fake_learner_sample(
    cfg: DictConfig,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
) -> TensorDict:
    env = make_primed_env(cfg, cfg.env.seed + 3, latent_state_dim(cfg), action_dim)
    try:
        sample = env.fake_tensordict().select(
            "action",
            "is_init",
            "state",
            "belief",
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
            *[
                ("next", _normalize_hydra_key(key))
                for key in (cfg.env.vector_key, cfg.env.pixels_key)
                if key is not None
            ],
        )
        return (
            sample.expand(cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len)
            .clone()
            .to(device)
        )
    finally:
        env.close()


def _warm_up_learner(
    cfg: DictConfig,
    device: torch.device,
    learner_update: DreamerV3OptimizationStepper,
    obs_dim: int,
    action_dim: int,
) -> None:
    if not resolve_compile_settings(cfg, device).enabled:
        return
    learner_update.warmup(_fake_learner_sample(cfg, device, obs_dim, action_dim))


def _validated_action_budget(cfg: DictConfig) -> int:
    num_envs = cfg.collector.num_envs
    if num_envs <= 0:
        raise ValueError(f"collector.num_envs must be positive, got {num_envs}.")
    if cfg.collector.backend not in ("sync", "async"):
        raise ValueError(
            "collector.backend must be 'sync' or 'async', got "
            f"{cfg.collector.backend!r}."
        )
    if cfg.collector.backend == "sync" and cfg.collector.frames_per_batch % num_envs:
        raise ValueError(
            "collector.frames_per_batch must be divisible by collector.num_envs, "
            f"got {cfg.collector.frames_per_batch} and {num_envs}."
        )
    max_time = cfg.optimization.max_time
    if max_time is not None and (max_time <= 0 or not math.isfinite(max_time)):
        raise ValueError("optimization.max_time must be positive and finite.")
    if cfg.optimization.collection_warmup_seconds < 0 or not math.isfinite(
        cfg.optimization.collection_warmup_seconds
    ):
        raise ValueError(
            "optimization.collection_warmup_seconds must be non-negative and finite."
        )
    if cfg.collector.total_frames < 0:
        if max_time is None:
            raise ValueError(
                "An unlimited frame budget requires optimization.max_time."
            )
        return -1
    collector_action_frames = (
        collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if cfg.collector.count_reset_records
        else cfg.collector.total_frames
    )
    if (
        cfg.collector.backend == "sync"
        and collector_action_frames % cfg.collector.frames_per_batch
    ):
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    return collector_action_frames


def _build_learner(
    cfg: DictConfig,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
    pixels_shape: tuple[int, int, int] | None = None,
    discrete: bool = False,
) -> _Learner:
    settings = resolve_compile_settings(cfg, device)
    (
        world_model,
        prior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    ) = build_world_model(
        cfg=cfg,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pixels_shape=pixels_shape,
        compile_rollout=not settings.train_step,
        rssm_backend=settings.rssm,
        rssm_scan_unroll=settings.scan_unroll,
    )
    world_model = world_model.to(device)
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
        # The whole-step compile owns the shared modules and traces the prior.
        compile_prior=settings.rssm == "scan" and not settings.train_step,
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim, discrete=discrete).to(
        device
    )
    value_model = build_value(cfg=cfg).to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
    )

    vector = _normalize_hydra_key(cfg.env.vector_key) if obs_dim else None
    pixels = (
        _normalize_hydra_key(cfg.env.pixels_key) if pixels_shape is not None else None
    )
    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_mode="separate",
        lambda_dynamic=cfg.optimization.dynamic_loss_weight,
        lambda_representation=cfg.optimization.representation_loss_weight,
        unimix=cfg.networks.unimix,
        lambda_continue=1.0,
        reco_symlog=[True, False]
        if vector is not None and pixels is not None
        else vector is not None,
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        # The reference adds the event dimensions, then averages batch and time.
        global_average=False,
        detach_output=False,
    ).to(device)
    model_loss.set_keys(
        pixels=[vector, pixels]
        if vector is not None and pixels is not None
        else vector
        if vector is not None
        else pixels,
        reco_pixels=["reco_vector", "reco_pixels"]
        if vector is not None and pixels is not None
        else "reco_pixels",
    )
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        continuation_model=continuation_model,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        return_normalization_rate=cfg.optimization.return_normalization_rate,
        return_normalization_min_scale=cfg.optimization.return_normalization_min_scale,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    actor_loss.to(device)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_critic_regularization=cfg.optimization.slow_critic_regularization,
    ).to(device)
    value_target_updater = SoftUpdate(value_loss, tau=cfg.optimization.slow_critic_tau)

    trainable_parameters = (
        list(world_model.parameters())
        + list(actor_model.parameters())
        + list(value_loss.parameters())
    )
    optimizer = DreamerV3Optimizer(
        trainable_parameters,
        lr=cfg.optimization.lr,
        agc=cfg.optimization.adaptive_grad_clip,
        eps=cfg.optimization.optimizer_eps,
        warmup_steps=cfg.optimization.warmup_steps,
    )

    real_world_actor = build_real_world_actor(
        world_model=world_model,
        actor_model=actor_model,
        mixed_precision=cfg.optimization.mixed_precision,
    )
    return _Learner(
        world_model=world_model,
        model_loss=model_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        value_target_updater=value_target_updater,
        optimizer=optimizer,
        real_world_actor=real_world_actor,
    )


def _build_collection(
    cfg: DictConfig,
    device: torch.device,
    learner: _Learner,
    state_dim: int,
    action_dim: int,
    collector_action_frames: int,
    replay_buffer: ReplayBufferEnsemble,
    post_collect_hook: Callable[[TensorDictBase], None],
) -> tuple[Collector | AsyncBatchedCollector, DreamerV3BehaviorPolicySync | None]:
    num_envs = cfg.collector.num_envs
    collector_backend = cfg.collector.backend
    if collector_backend not in ("sync", "async"):
        raise ValueError(
            "collector.backend must be 'sync' or 'async', got "
            f"{collector_backend!r}."
        )
    real_world_actor = learner.real_world_actor
    if cfg.optimization.deferred_policy_sync and collector_backend == "sync":
        collector_actor = copy.deepcopy(real_world_actor)
        # The decoder cannot act, but the reference syncs both parameter trees.
        behavior_decoder = copy.deepcopy(learner.world_model[2])
        learner_policy_tree = torch.nn.ModuleList(
            [real_world_actor, learner.world_model[2]]
        )
        behavior_policy_tree = torch.nn.ModuleList([collector_actor, behavior_decoder])
        behavior_policy_sync = DreamerV3BehaviorPolicySync(
            learner_policy_tree, behavior_policy_tree
        )
    elif collector_backend == "async":
        # The inference server owns a separate actor that is refreshed through
        # the collector's synchronized policy-update path after each train batch.
        collector_actor = copy.deepcopy(real_world_actor)
        behavior_policy_sync = None
    else:
        collector_actor = real_world_actor
        behavior_policy_sync = None
    collector_policy = (
        DreamerV3SeededPolicy(collector_actor, cfg.env.seed)
        if cfg.optimization.separate_policy_rng
        else collector_actor
    )
    create_env_fns = [
        ft.partial(
            make_primed_env,
            cfg,
            cfg.env.seed + 2 + index if cfg.env.use_seed else None,
            state_dim,
            action_dim,
            env_index=index,
        )
        for index in range(num_envs)
    ]
    observation_keys = [
        ("next", _normalize_hydra_key(key))
        for key in (cfg.env.vector_key, cfg.env.pixels_key, cfg.env.milestone_key)
        if key is not None
    ]
    replay_postproc = SelectTransform(
        "action",
        "is_init",
        "state",
        "belief",
        "env_index",
        *observation_keys,
        ("next", "reward"),
        ("next", "done"),
        ("next", "terminated"),
        ("next", "truncated"),
        keep_rewards=False,
        keep_dones=False,
    )

    collector_kwargs = {
        "frames_per_batch": cfg.collector.frames_per_batch,
        "total_frames": collector_action_frames,
        "postproc": replay_postproc,
        "post_collect_hook": post_collect_hook,
        "replay_buffer": replay_buffer,
        "exploration_type": (
            ExplorationType.RANDOM
            if cfg.collector.exploration == "random"
            else ExplorationType.MODE
        ),
    }
    if collector_backend == "async":
        collector = AsyncBatchedCollector(
            create_env_fns,
            policy=collector_policy,
            env_backend=cfg.collector.async_env_backend,
            env_exchange=cfg.collector.env_exchange,
            envs_per_worker=cfg.collector.envs_per_worker,
            server_config=InferenceServerConfig(
                max_batch_size=cfg.collector.inference_max_batch_size or num_envs,
                min_batch_size=cfg.collector.inference_min_batch_size,
                timeout=cfg.collector.inference_timeout,
                static_batch_size=cfg.collector.inference_static_batch_size,
            ),
            device_config=InferenceDeviceConfig(
                policy_device=cfg.collector.policy_device or device,
                output_device="cpu",
                env_device="cpu",
                storing_device="cpu",
            ),
            policy_version_key=None,
            **collector_kwargs,
        )
    else:
        collector = Collector(
            SerialEnv(num_envs, create_env_fns),
            collector_policy,
            policy_device=device,
            env_device="cpu",
            storing_device="cpu",
            **collector_kwargs,
        )
    if cfg.optimization.separate_policy_rng:
        # The collector's construction-time policy call is not an action.
        collector_policy.reset_counter()
    return collector, behavior_policy_sync


def _build_replay(
    cfg: DictConfig,
    num_envs: int,
    replay_device: torch.device,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> ReplayBufferEnsemble:
    sequence_records = cfg.replay_buffer.seq_len + 1
    base_capacity, remainder = divmod(cfg.replay_buffer.buffer_size, num_envs)
    capacities = [base_capacity + (index < remainder) for index in range(num_envs)]
    if min(capacities) < sequence_records:
        raise ValueError(
            f"replay_buffer.buffer_size={cfg.replay_buffer.buffer_size} split "
            f"across {num_envs} streams cannot hold one {sequence_records}-record "
            "sequence per stream."
        )

    sampler_type = StreamingSliceSampler if cfg.replay_buffer.online else SliceSampler
    members = [
        TensorDictReplayBuffer(
            storage=LazyTensorStorage(capacity, device=replay_device),
            sampler=sampler_type(
                slice_len=sequence_records,
                end_key=("next", "done"),
            ),
            writer=TensorDictRoundRobinWriter(track_generations=True),
        )
        for capacity in capacities
    ]
    if generator is None:
        generator = torch.Generator().manual_seed(
            stream_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM)
        )
    return ReplayBufferEnsemble(
        *members,
        p="sampleable",
        num_buffer_sampled=cfg.replay_buffer.batch_size,
        routing_key="env_index" if cfg.collector.backend == "async" else None,
        routing_dim=0 if cfg.collector.backend == "sync" else None,
        batch_size=cfg.replay_buffer.batch_size * sequence_records,
        generator=generator,
        pin_memory=replay_device.type == "cpu" and device.type == "cuda",
        prefetch=1,
    )


def _log_train_episodes(
    cfg: DictConfig,
    run_logger: _RunLogger,
    completed_episodes: list[tuple[int, int, float]],
    batch_start_action_step: int,
    batch_start_record_step: int,
    batch_reset_prefix: list[int],
    milestones: list[list[bool]] | None = None,
) -> None:
    num_envs = cfg.collector.num_envs
    for episode_index, (time_index, env_index, score) in enumerate(completed_episodes):
        position = (
            time_index
            if cfg.collector.backend == "async"
            else time_index * num_envs + env_index
        )
        episode_step = batch_start_action_step + position + 1
        if cfg.collector.count_reset_records:
            episode_step = (
                batch_start_record_step + position + 1 + batch_reset_prefix[position]
            )
        run_logger.log(
            {
                "type": "train_episode",
                "environment_steps": episode_step,
                "score": score,
                **({"milestones": milestones[episode_index]} if milestones else {}),
            },
        )


def _log_train_window(
    *,
    cfg: DictConfig,
    run_logger: _RunLogger,
    run_timer,
    record_step: int,
    update_step: int,
    loss_window_sum: torch.Tensor,
    loss_window_updates: int,
) -> None:
    run_logger.log(
        {
            "type": "train",
            "environment_steps": record_step,
            "updates": update_step,
            "updates_in_window": loss_window_updates,
            **dict(
                zip(
                    (
                        "loss_dynamic_representation",
                        "loss_reconstruction",
                        "loss_reward",
                        "loss_actor",
                        "loss_value",
                        "loss_replay_value",
                    ),
                    (loss_window_sum / max(loss_window_updates, 1)).cpu().tolist(),
                )
            ),
            "elapsed_seconds": run_timer.elapsed(),
        },
    )


def _evaluate(
    *,
    cfg: DictConfig,
    device: torch.device,
    eval_env,
    real_world_actor: TensorDictModuleBase,
    run_logger: _RunLogger,
    run_timer,
    record_step: int,
    latest_losses: torch.Tensor,
) -> torch.Tensor:
    # Evaluation samples RSSM latents, thus a fork keeps training unchanged.
    with (
        timeit("dreamer_v3/evaluation"),
        torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
    ):
        r = eval_episode_reward(
            eval_env,
            real_world_actor,
            cfg.logger.eval_episodes,
            cfg.env.max_episode_steps,
        )
    torchrl_logger.info(
        "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
        record_step,
        r.item(),
        latest_losses[0].item(),
        latest_losses[1].item(),
        latest_losses[2].item(),
        latest_losses[3].item(),
    )
    run_logger.log(
        {
            "type": "evaluation",
            "environment_steps": record_step,
            "return": r.item(),
            "episodes": cfg.logger.eval_episodes,
            "elapsed_seconds": run_timer.elapsed(),
        },
    )
    return r


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)
    resume_path = _resolve_resume_path(cfg.optimization.resume_from)
    run_state = {}
    if resume_path is not None:
        saved_config = {}
        Checkpoint(run_state=run_state, config=saved_config).load(resume_path)
        for key in ("buffer_size", "batch_size", "seq_len", "online"):
            if cfg.replay_buffer[key] != saved_config["replay_buffer"][key]:
                raise ValueError(f"Resume requires the saved replay_buffer.{key}.")
        if run_state["num_envs"] != cfg.collector.num_envs:
            raise ValueError("Resume requires the saved collector.num_envs.")
        if run_state["logger_backend"] != cfg.logger.backend:
            raise ValueError("Resume requires the saved logger.backend.")
    checkpoint_every = cfg.optimization.checkpoint_every
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("optimization.checkpoint_every must be positive or null.")
    rotation = (
        CheckpointRotation(
            Path(cfg.optimization.checkpoint_dir).expanduser().resolve(),
            keep_last=cfg.optimization.checkpoint_keep_last,
        )
        if cfg.optimization.checkpoint_dir
        else None
    )

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    replay_device = (
        torch.device(cfg.replay_buffer.device) if cfg.replay_buffer.device else device
    )
    use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"
    compile_settings = resolve_compile_settings(cfg, device)
    torchrl_logger.info(
        "DreamerV3 execution: device=%s, replay_device=%s, compile=%s "
        "(train_step=%s, rssm_backend=%s, rssm_scan_unroll=%s, cudagraph=%s), "
        "mixed_precision=%s",
        device,
        replay_device,
        compile_settings.strategy,
        compile_settings.train_step,
        compile_settings.rssm or "eager",
        compile_settings.scan_unroll if compile_settings.rssm == "scan" else "n/a",
        compile_settings.cudagraph,
        use_bfloat16,
    )
    num_envs = cfg.collector.num_envs
    count_reset_records = cfg.collector.count_reset_records
    collector_action_frames = _validated_action_budget(cfg)
    real_env = make_env(cfg, cfg.env.seed)
    vector = _normalize_hydra_key(cfg.env.vector_key)
    pixels = _normalize_hydra_key(cfg.env.pixels_key)
    obs_dim = real_env.observation_spec[vector].shape[-1] if vector is not None else 0
    pixels_shape = (
        tuple(real_env.observation_spec[pixels].shape[-3:])
        if pixels is not None
        else None
    )
    if not obs_dim and pixels_shape is None:
        raise ValueError("Set env.vector_key, env.pixels_key or both.")
    discrete = isinstance(real_env.action_spec, OneHot)
    if len(real_env.action_spec.shape) != 1:
        raise ValueError(
            "DreamerV3 requires a vector action or one-hot discrete action spec."
        )
    action_dim = real_env.action_spec.shape[0]
    real_env.close()
    state_dim = latent_state_dim(cfg)
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if resume_path is not None:
        saved_metrics_path = run_state.get("metrics_jsonl")
        metrics_jsonl_path = Path(saved_metrics_path) if saved_metrics_path else None
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_path is None:
            metrics_jsonl_path.write_text("")
    timeit.reset()
    run_timer = _ElapsedTimer(
        timeit("dreamer_v3/run").start(), run_state.get("elapsed_seconds", 0.0)
    )

    learner = _build_learner(cfg, device, obs_dim, action_dim, pixels_shape, discrete)
    learner_update = _make_learner_update(cfg, device, learner)
    _warm_up_learner(cfg, device, learner_update, obs_dim, action_dim)
    replay_rng = torch.Generator().manual_seed(
        stream_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM)
    )
    rb = _build_replay(cfg, num_envs, replay_device, device, generator=replay_rng)
    checkpoint = Checkpoint(
        learner=learner_update.loss_module,
        learner_update=learner_update,
        replay=rb,
        run_state=run_state,
        rng=GlobalRNGState(),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    replay_restored = False
    if resume_path is not None:
        checkpoint.load(
            resume_path,
            components={"learner", "learner_update"},
            map_location=device,
        )
        if "replay" in Checkpoint.manifest(resume_path)["components"]:
            checkpoint.load(resume_path, components={"replay"})
            rb.end_streams()
            replay_restored = True
        replay_rng.set_state(torch.tensor(run_state["replay_rng"], dtype=torch.uint8))
        rb.set_rng(replay_rng)
    action_offset = int(run_state.get("action_steps", 0))
    action_step = action_offset
    reset_records = int(run_state.get("reset_records", 0)) + num_envs
    record_step = action_step + reset_records if count_reset_records else action_step
    update_step = int(run_state.get("updates", 0))
    running_training_return = torch.zeros(num_envs)
    seen_stream = torch.zeros(num_envs, dtype=torch.bool)
    completed_episodes: list[tuple[int, int, float]] = []
    completed_milestones: list[list[bool]] = []
    milestone_key = _normalize_hydra_key(cfg.env.milestone_key)
    batch_reset_prefix: list[int] = []

    def post_collect_hook(data: TensorDictBase) -> None:
        nonlocal reset_records
        completed_episodes.clear()
        completed_episodes.extend(
            training_episode_returns(data, running_training_return, num_envs)
        )
        completed_milestones.clear()
        if milestone_key is not None:
            flags = data.get(("next", milestone_key))
            for position, env_index, _ in completed_episodes:
                value = (
                    flags[position]
                    if cfg.collector.backend == "async"
                    else flags[env_index, position]
                )
                completed_milestones.append(value.bool().tolist())
        if not count_reset_records:
            return
        is_init = data.get("is_init").reshape(-1).cpu()
        env_index = data.get("env_index", default=None)
        if env_index is None:
            # Report synchronous episodes in time-major, then environment order.
            is_init = is_init.reshape(num_envs, -1).t().reshape(-1)
            env_index = torch.arange(num_envs).repeat(data.numel() // num_envs)
        else:
            env_index = env_index.reshape(-1).cpu()
        batch_reset_prefix.clear()
        batch_resets = 0
        for stream, reset in zip(env_index, is_init):
            if reset:
                stream = int(stream)
                if seen_stream[stream]:
                    reset_records += 1
                    batch_resets += 1
                else:
                    seen_stream[stream] = True
            batch_reset_prefix.append(batch_resets)

    collector, behavior_policy_sync = _build_collection(
        cfg,
        device,
        learner,
        state_dim,
        action_dim,
        max(0, collector_action_frames - action_offset)
        if collector_action_frames >= 0
        else -1,
        rb,
        post_collect_hook,
    )

    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    loss_window_sum = torch.tensor(
        run_state.get("loss_window_sum", [0.0] * 6), device=device
    )
    loss_window_updates = int(run_state.get("loss_window_updates", 0))
    record_loss_history = plot_enabled(cfg)
    next_eval = int(run_state.get("next_eval", 0))
    next_train_log = int(run_state.get("next_train_log", 0))

    eval_env = make_primed_env(cfg, cfg.env.seed + 100, state_dim, action_dim)
    run_logger = _RunLogger(cfg, metrics_jsonl_path, run_state.get("logger"))

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )
    warmup = max(warmup, num_envs * (cfg.replay_buffer.seq_len + 1))

    updates_per_batch = cfg.optimization.updates_per_batch
    update_ratio = (
        DreamerV3UpdateRatio(
            cfg.optimization.train_ratio
            / (cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len)
        )
        if cfg.optimization.train_ratio is not None
        else None
    )

    if cfg.optimization.separate_policy_rng:
        # Keep the learner draws in a range apart from the policy stream.
        torch.manual_seed(stream_seed(cfg.env.seed, 0, LEARNER_RNG_STREAM))

    stateful_components = set()
    if update_ratio is not None:
        checkpoint.register("update_ratio", update_ratio)
        stateful_components.add("update_ratio")
    if isinstance(collector.policy, DreamerV3SeededPolicy):
        checkpoint.register("policy", collector.policy)
        stateful_components.add("policy")
    if resume_path is not None:
        if stateful_components:
            checkpoint.load(
                resume_path, components=stateful_components, map_location=device
            )
        # Construction and compile/capture warm-up may consume RNG and update
        # normalization. Restore training state after warm-up, and RNG last.
        checkpoint.load(resume_path, components={"rng"})
        torchrl_logger.info(
            "Resumed DreamerV3 at action step %s with replay=%s; environments restart.",
            action_step,
            replay_restored,
        )
    next_checkpoint = update_step + checkpoint_every if checkpoint_every else None
    shutdown_request = _ShutdownRequest()
    previous_handlers = {
        signum: signal.signal(signum, shutdown_request)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def save_checkpoint() -> None:
        if rotation is None:
            return
        pause = (
            collector.pause()
            if isinstance(collector, AsyncBatchedCollector)
            else nullcontext()
        )
        with pause:
            rb.synchronize()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            run_state.update(
                action_steps=action_step,
                reset_records=reset_records,
                environment_steps=record_step,
                updates=update_step,
                num_envs=num_envs,
                elapsed_seconds=run_timer.elapsed(),
                next_eval=next_eval,
                next_train_log=next_train_log,
                loss_window_sum=loss_window_sum.cpu().tolist(),
                loss_window_updates=loss_window_updates,
                replay_rng=replay_rng.get_state().cpu().tolist(),
                logger_backend=cfg.logger.backend,
                logger=run_logger.logger.state_dict()
                if run_logger.logger is not None
                else {},
                metrics_jsonl=str(metrics_jsonl_path) if metrics_jsonl_path else None,
            )
            components = set(checkpoint.components)
            if not cfg.optimization.checkpoint_include_replay:
                components.discard("replay")
            path = rotation.save(checkpoint, step=action_step, components=components)
            torchrl_logger.info("Saved DreamerV3 checkpoint to %s", path)

    collection_timer = None
    try:
        for _ in collector:
            if collection_timer is None:
                collection_timer = timeit("dreamer_v3/collection", sync=False).start()
            # The collector writes canonical transitions directly into replay.
            if behavior_policy_sync is not None:
                behavior_policy_sync.apply_after_action()
            batch_start_action_step = action_step
            batch_start_record_step = record_step
            action_step = action_offset + int(collector.stats()["frames"])
            record_step = (
                action_step + reset_records if count_reset_records else action_step
            )
            _log_train_episodes(
                cfg,
                run_logger,
                completed_episodes,
                batch_start_action_step,
                batch_start_record_step,
                batch_reset_prefix,
                completed_milestones,
            )

            if shutdown_request.signal_number is not None:
                break
            if (
                cfg.optimization.max_time is not None
                and run_timer.elapsed() >= cfg.optimization.max_time
            ):
                break
            if collection_timer.elapsed() < cfg.optimization.collection_warmup_seconds:
                continue
            replay_stats = rb.stats()
            if replay_stats["size"] < warmup or not rb.can_sample():
                if (
                    resume_path is not None
                    and not replay_restored
                    and update_ratio is not None
                ):
                    # New replay must warm up without accumulating a learner catch-up burst.
                    update_ratio.reset(record_step)
                continue

            batch_updates = (
                update_ratio(record_step)
                if update_ratio is not None
                else updates_per_batch
            )
            if not batch_updates:
                continue

            if behavior_policy_sync is not None:
                # Stage one time per batch; more updates keep the pending snapshot.
                behavior_policy_sync.stage_before_training()

            batch_losses = torch.empty((batch_updates, 6), device=device)
            for update_index in range(batch_updates):
                with timeit("dreamer_v3/replay_sample"):
                    replay_sample = rb.sample().reshape(
                        cfg.replay_buffer.batch_size,
                        cfg.replay_buffer.seq_len + 1,
                    )
                    sample_info = replay_sample.select("index", "index_generation")
                    sample = replay_sample.exclude("index", "index_generation")
                    sample = sample.to(device, non_blocking=True)[:, :-1]
                with timeit(
                    "dreamer_v3/train_update", sync=cfg.optimization.sync_timers
                ):
                    losses = learner_update.step(None, sample)
                    update_losses = _learner_metrics(losses)
                    refreshed_state = sample["replay_context", "state"]
                    refreshed_belief = sample["replay_context", "belief"]
                    batch_losses[update_index].copy_(update_losses)
                    loss_window_sum += update_losses
                    loss_window_updates += 1
                with timeit("dreamer_v3/replay_submit"):
                    index, generation, patch = replay_context_update(
                        sample_info, refreshed_state, refreshed_belief
                    )
                    rb.submit_update_if_present(
                        index=index,
                        generation=generation,
                        patch=patch,
                    )
                update_step += 1

            if isinstance(collector, AsyncBatchedCollector):
                policy_weights = learner.real_world_actor
                if cfg.optimization.separate_policy_rng:
                    policy_weights = TensorDict(
                        {"module": TensorDict.from_module(policy_weights)}, []
                    )
                collector.update_policy_weights_(policy_weights)

            if record_loss_history:
                loss_history.append(batch_losses.cpu())

            train_log_due = bool(
                (metrics_jsonl_path is not None or cfg.logger.backend)
                and cfg.logger.train_every
                and (
                    record_step >= next_train_log
                    or action_step >= collector_action_frames
                )
            )
            eval_due = bool(cfg.logger.eval_every and record_step >= next_eval)
            latest_losses = batch_losses[-1].cpu() if eval_due else None
            if train_log_due:
                _log_train_window(
                    cfg=cfg,
                    run_logger=run_logger,
                    run_timer=run_timer,
                    record_step=record_step,
                    update_step=update_step,
                    loss_window_sum=loss_window_sum,
                    loss_window_updates=loss_window_updates,
                )
                loss_window_sum.zero_()
                loss_window_updates = 0
                next_train_log = record_step + cfg.logger.train_every

            if eval_due:
                r = _evaluate(
                    cfg=cfg,
                    device=device,
                    eval_env=eval_env,
                    real_world_actor=learner.real_world_actor,
                    run_logger=run_logger,
                    run_timer=run_timer,
                    record_step=record_step,
                    latest_losses=latest_losses,
                )
                history_steps.append(record_step)
                history_eval.append(r)
                next_eval = record_step + cfg.logger.eval_every

            if next_checkpoint is not None and update_step >= next_checkpoint:
                save_checkpoint()
                next_checkpoint = update_step + checkpoint_every

        save_checkpoint()
        run_logger.log(
            {
                "type": "summary",
                "backend": cfg.env.backend,
                "environment": cfg.env.name,
                "task": cfg.env.task,
                "seed": cfg.env.seed,
                "environment_seeded": bool(cfg.env.use_seed),
                "total_environment_steps": record_step,
                "total_action_steps": action_step,
                "updates": update_step,
                "bfloat16": use_bfloat16,
                "elapsed_seconds": run_timer.elapsed(),
                "timings": timeit.todict(percall=False),
            },
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            collector.shutdown()
        finally:
            try:
                eval_env.close()
            finally:
                try:
                    rb.shutdown()
                finally:
                    run_logger.finish()

    if cfg.logger.output_plot:
        save_run_plot(cfg, history_steps, history_eval, loss_history)

    if metrics_jsonl_path is not None:
        torchrl_logger.info("Saved run metrics to %s", metrics_jsonl_path)


if __name__ == "__main__":
    main()
