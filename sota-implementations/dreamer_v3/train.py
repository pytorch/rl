# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 training script that reproduces a pinned JAX configuration.

The script is proprioceptive, not pixel-based, and writes its metrics to a
JSONL file on the same step axis as the author-maintained JAX implementation.

Usage::

    python sota-implementations/dreamer_v3/train.py \\
        collector.total_frames=5000 logger.eval_every=500

    python sota-implementations/dreamer_v3/train.py \\
        --config-name=config_dmc_walker
"""

from __future__ import annotations

import copy
import functools as ft
from collections.abc import Callable
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
    DreamerV3SeededPolicy,
    make_env,
    make_primed_env,
)
from dreamer_v3_replay import (
    collector_action_budget,
    DreamerV3UpdateRatio,
    driver_step_for_action,
    replay_context_update,
)
from dreamer_v3_utils import (
    append_jsonl,
    eval_episode_reward,
    latent_state_dim,
    LEARNER_RNG_STREAM,
    plot_enabled,
    REPLAY_RNG_STREAM,
    save_run_plot,
    stream_seed,
    training_episode_returns,
)
from omegaconf import DictConfig
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl import timeit
from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import AsyncBatchedCollector, Collector
from torchrl.data import (
    LazyTensorStorage,
    ReplayBufferEnsemble,
    SliceSampler,
    StreamingSliceSampler,
    TensorDictReplayBuffer,
    TensorDictRoundRobinWriter,
)
from torchrl.envs import SelectTransform, SerialEnv
from torchrl.envs.utils import ExplorationType
from torchrl.modules.inference_server import InferenceDeviceConfig
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3Loss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.trainers.algorithms import DreamerV3OptimizationStepper, DreamerV3Optimizer


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
    return DreamerV3OptimizationStepper(
        loss_module,
        learner.optimizer,
        learner.value_target_updater,
        compile_train_step=cfg.optimization.compile_train_step,
        compile_mode=cfg.optimization.compile_train_step_mode,
        cudagraph=cfg.optimization.cudagraph_train_step,
        warmup_steps=cudagraph_warmup if cfg.optimization.cudagraph_train_step else 1,
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
    batch_size = (cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len)
    return TensorDict(
        {
            "state": torch.zeros(*batch_size, latent_state_dim(cfg)),
            "belief": torch.zeros(*batch_size, cfg.networks.rnn_hidden_dim),
            "action": torch.zeros(*batch_size, action_dim),
            "is_init": torch.zeros(*batch_size, 1, dtype=torch.bool),
            "next": {
                "observation": torch.zeros(*batch_size, obs_dim),
                "reward": torch.zeros(*batch_size, 1),
                "done": torch.zeros(*batch_size, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch_size, 1, dtype=torch.bool),
            },
        },
        batch_size,
        device=device,
    )


def _warm_up_learner(
    cfg: DictConfig,
    device: torch.device,
    learner_update: DreamerV3OptimizationStepper,
    obs_dim: int,
    action_dim: int,
) -> None:
    if not (
        cfg.optimization.compile_train_step
        or cfg.optimization.compile_rssm
        or cfg.optimization.cudagraph_train_step
    ):
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
    collector_action_frames = (
        collector_action_budget(
            cfg.collector.total_frames,
            num_envs,
            cfg.env.max_episode_steps,
        )
        if cfg.collector.count_reset_records
        else cfg.collector.total_frames
    )
    if collector_action_frames % cfg.collector.frames_per_batch:
        raise ValueError(
            "The action budget derived from collector.total_frames must be "
            "divisible by collector.frames_per_batch, got "
            f"{collector_action_frames} and {cfg.collector.frames_per_batch}."
        )
    return collector_action_frames


def _build_learner(
    cfg: DictConfig, device: torch.device, obs_dim: int, action_dim: int
) -> _Learner:
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
        compile_rollout=not cfg.optimization.compile_train_step,
    )
    world_model = world_model.to(device)
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
        # The whole-step compile owns shared modules and subsumes RSSM compile.
        compile_prior=(
            cfg.optimization.compile_rssm == "scan"
            and not cfg.optimization.compile_train_step
        ),
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim).to(device)
    value_model = build_value(cfg=cfg).to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
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
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        # The reference adds the event dimensions, then averages batch and time.
        global_average=False,
        detach_output=False,
    ).to(device)
    model_loss.set_keys(pixels="observation")
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
        )
        for index in range(num_envs)
    ]
    replay_postproc = SelectTransform(
        "action",
        "is_init",
        "state",
        "belief",
        "env_index",
        ("next", "observation"),
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
    }
    if collector_backend == "async":
        collector = AsyncBatchedCollector(
            create_env_fns,
            policy=collector_policy,
            max_batch_size=num_envs,
            env_backend=cfg.collector.async_env_backend,
            device_config=InferenceDeviceConfig(
                policy_device=device,
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
            exploration_type=(
                ExplorationType.RANDOM
                if cfg.collector.exploration == "random"
                else ExplorationType.MODE
            ),
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
    return ReplayBufferEnsemble(
        *members,
        p="sampleable",
        num_buffer_sampled=cfg.replay_buffer.batch_size,
        routing_key="env_index" if cfg.collector.backend == "async" else None,
        routing_dim=0 if cfg.collector.backend == "sync" else None,
        batch_size=cfg.replay_buffer.batch_size * sequence_records,
        generator=torch.Generator().manual_seed(
            stream_seed(cfg.env.seed, 0, REPLAY_RNG_STREAM)
        ),
        pin_memory=replay_device.type == "cpu" and device.type == "cuda",
        prefetch=1,
    )


def _log_train_episodes(
    cfg: DictConfig,
    metrics_jsonl_path: Path | None,
    completed_episodes: list[tuple[int, int, float]],
    batch_start_action_step: int,
    batch_start_record_step: int,
    batch_reset_prefix: list[int],
) -> None:
    num_envs = cfg.collector.num_envs
    for time_index, env_index, score in completed_episodes:
        if cfg.collector.backend == "async":
            episode_step = batch_start_action_step + time_index + 1
            if cfg.collector.count_reset_records:
                episode_step = (
                    batch_start_record_step
                    + time_index
                    + 1
                    + batch_reset_prefix[time_index]
                )
        elif cfg.collector.count_reset_records:
            action_index = batch_start_action_step // num_envs + time_index + 1
            episode_step = driver_step_for_action(
                action_index,
                env_index,
                num_envs,
                cfg.env.max_episode_steps,
            )
        else:
            episode_step = (
                batch_start_action_step + time_index * num_envs + env_index + 1
            )
        append_jsonl(
            metrics_jsonl_path,
            {
                "type": "train_episode",
                "environment_steps": episode_step,
                "score": score,
            },
        )


def _log_train_window(
    *,
    cfg: DictConfig,
    metrics_jsonl_path: Path | None,
    run_timer,
    record_step: int,
    update_step: int,
    loss_window_sum: torch.Tensor,
    loss_window_updates: int,
) -> None:
    append_jsonl(
        metrics_jsonl_path,
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
    metrics_jsonl_path: Path | None,
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
    append_jsonl(
        metrics_jsonl_path,
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

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    replay_device = (
        torch.device(cfg.replay_buffer.device) if cfg.replay_buffer.device else device
    )
    use_bfloat16 = cfg.optimization.mixed_precision and device.type == "cuda"
    torchrl_logger.info(
        "DreamerV3 execution: device=%s, replay_device=%s, rssm_backend=%s, "
        "rssm_scan_unroll=%s, mixed_precision=%s, compile_train_step=%s, "
        "cudagraph_train_step=%s",
        device,
        replay_device,
        cfg.optimization.compile_rssm or "eager",
        (
            cfg.optimization.rssm_scan_unroll
            if cfg.optimization.compile_rssm == "scan"
            else "n/a"
        ),
        use_bfloat16,
        cfg.optimization.compile_train_step,
        cfg.optimization.cudagraph_train_step,
    )
    num_envs = cfg.collector.num_envs
    count_reset_records = cfg.collector.count_reset_records
    collector_action_frames = _validated_action_budget(cfg)
    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    real_env.close()
    state_dim = latent_state_dim(cfg)
    metrics_jsonl_path = (
        Path(cfg.logger.metrics_jsonl).resolve() if cfg.logger.metrics_jsonl else None
    )
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("")
    timeit.reset()
    run_timer = timeit("dreamer_v3/run").start()

    learner = _build_learner(cfg, device, obs_dim, action_dim)
    learner_update = _make_learner_update(cfg, device, learner)
    _warm_up_learner(cfg, device, learner_update, obs_dim, action_dim)
    rb = _build_replay(cfg, num_envs, replay_device, device)
    action_step = 0
    reset_records = num_envs
    record_step = reset_records if count_reset_records else 0
    update_step = 0
    running_training_return = torch.zeros(num_envs)
    seen_stream = torch.zeros(num_envs, dtype=torch.bool)
    completed_episodes: list[tuple[int, int, float]] = []
    batch_reset_prefix: list[int] = []

    def post_collect_hook(data: TensorDictBase) -> None:
        nonlocal reset_records
        completed_episodes.clear()
        completed_episodes.extend(
            training_episode_returns(data, running_training_return, num_envs)
        )
        if not count_reset_records:
            return
        is_init = data.get("is_init").reshape(-1).cpu()
        env_index = data.get("env_index", default=None)
        if env_index is None:
            env_index = torch.arange(num_envs).reshape(num_envs, 1)
            env_index = env_index.expand(data.batch_size).reshape(-1)
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
        collector_action_frames,
        rb,
        post_collect_hook,
    )

    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    loss_window_sum = torch.zeros(6, device=device)
    loss_window_updates = 0
    record_loss_history = plot_enabled(cfg)
    next_eval = 0
    next_train_log = 0

    eval_env = make_primed_env(cfg, cfg.env.seed + 100, state_dim, action_dim)

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

    try:
        for _ in collector:
            # The collector writes canonical transitions directly into replay.
            if behavior_policy_sync is not None:
                behavior_policy_sync.apply_after_action()
            batch_start_action_step = action_step
            batch_start_record_step = record_step
            action_step = int(collector.stats()["frames"])
            record_step = (
                action_step + reset_records if count_reset_records else action_step
            )
            _log_train_episodes(
                cfg,
                metrics_jsonl_path,
                completed_episodes,
                batch_start_action_step,
                batch_start_record_step,
                batch_reset_prefix,
            )

            replay_stats = rb.stats()
            if replay_stats["size"] < warmup or not rb.can_sample():
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
                metrics_jsonl_path is not None
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
                    metrics_jsonl_path=metrics_jsonl_path,
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
                    metrics_jsonl_path=metrics_jsonl_path,
                    run_timer=run_timer,
                    record_step=record_step,
                    latest_losses=latest_losses,
                )
                history_steps.append(record_step)
                history_eval.append(r)
                next_eval = record_step + cfg.logger.eval_every

    finally:
        try:
            collector.shutdown()
        finally:
            try:
                eval_env.close()
            finally:
                rb.shutdown()

    if cfg.logger.output_plot:
        save_run_plot(cfg, history_steps, history_eval, loss_history)

    append_jsonl(
        metrics_jsonl_path,
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
    if metrics_jsonl_path is not None:
        torchrl_logger.info("Saved run metrics to %s", metrics_jsonl_path)


if __name__ == "__main__":
    main()
