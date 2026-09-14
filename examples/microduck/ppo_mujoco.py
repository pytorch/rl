# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Recurrent PPO on :class:`~torchrl.envs.MicroDuckEnv` with whole-episode replay.

The policy is a GRU backbone shared by the actor and the critic, with a
Gaussian head trained end to end from scratch: it relies on the contact-based
gait terms of the :class:`~torchrl.envs.MicroDuckEnv` reward, the task
library's warm start and a unit exploration scale. ``policy.from_prior=true``
is the quick debugging start: the head then adds a bounded residual to the
closed-form gait from ``heuristic_gait.py``, so the first policy already walks
forward, but that prior only knows forward walking and never learns to
sidestep or hop.

Optimization is owned by :class:`~torchrl.trainers.algorithms.PPOTrainer`.
Data flows through the standard TorchRL pieces: a
:class:`~torchrl.collectors.Collector` writes every finished episode as a
whole, unpadded sequence into a
:class:`~torchrl.data.TensorDictReplayBuffer`; GAE is computed once over the
buffer; :class:`~torchrl.data.SliceSampler` draws whole episodes for the
recurrent PPO updates; the buffer is erased before collecting again with the
updated policy. One :class:`~torchrl.collectors.Evaluator` per velocity
command runs the deterministic evaluations, and checkpoints are unified TorchRL
checkpoints written with :func:`~torchrl.render.save_render_checkpoint`, so
``rlrender`` and ``policy.init_from`` read them directly.

The script is configured with Hydra from ``config.yaml``. Run a short CPU job
from a TorchRL checkout::

    python examples/microduck/ppo_mujoco.py env.download=true smoke=true

and train over a speed range with::

    python examples/microduck/ppo_mujoco.py env.download=true \\
        'env.tasks=[{preset:speed_range_task,low:0.1,high:0.3}]' logger.entity=YOUR_ENTITY

``env.tasks`` is the task library: one :class:`~torchrl.envs.MicroDuckEnv`
preset per entry with its arguments, e.g. ``{preset: sidestep_task, speed: 0.15,
weight: 0.5}``. Every env picks a task at reset and the policy reads the task
index through a learned embedding, so one policy trains on the whole library.

``env.download=true`` fetches the pinned ``microduck_rl`` assets into
``~/.cache/torchrl/microduck``; set ``env.microduck_root`` or
``MICRODUCK_RL_ROOT`` to use an existing checkout instead. ``env.backend=mjx``
or ``env.backend=mujoco-torch env.compile_step=true`` change only the
simulator.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential

from torchrl import timeit, torchrl_logger
from torchrl.checkpoint import Checkpoint, GlobalRNGState
from torchrl.collectors import Collector, Evaluator
from torchrl.data import LazyTensorStorage, SliceSampler, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    EnvBase,
    InitTracker,
    MicroDuckEnv,
    MicroDuckTask,
    MicroDuckTaskSampler,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    get_primers_from_module,
    ProbabilisticActor,
    set_recurrent_mode,
)
from torchrl.objectives import ClipPPOLoss, KLAdaptiveLR
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.render import load_checkpoint, save_render_checkpoint
from torchrl.trainers.algorithms import PPOTrainer

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent))

from torchrl.envs.custom.mujoco._skill_models import (  # noqa: F401 - example re-exports
    make_actor_critic,
    make_tasks,
    MicroDuckGaitActor,
    MicroDuckGaitConfig,
    PolicyHead,
)

# The asset location is machine specific and is never taken from a checkpoint.
ASSET_KEYS = ("microduck_root", "root", "download")


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


def make_env(
    cfg: DictConfig | Mapping[str, Any] | None = None,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    microduck_root: str | Path | None = None,
    root: str | Path | None = None,
    download: bool | str | None = None,
    num_envs: int | None = None,
    parallel: bool | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
) -> TransformedEnv:
    """Build the batched env from the ``env`` section of ``config.yaml``.

    ``cfg`` is that section, as the Hydra ``DictConfig`` or a plain mapping;
    missing entries take the defaults of ``config.yaml``. ``tasks`` becomes
    the task library through :func:`make_tasks`, and ``task_id`` pins one task
    of the library at every reset with a
    :class:`~torchrl.envs.MicroDuckTaskSampler` (the evaluators and
    ``rlrender`` use it). ``rlrender`` passes the training checkpoint, whose
    recorded config (minus the asset location, which is machine specific) sits
    between those defaults and ``cfg``. The keyword arguments override single
    entries so a checkpoint renders with one env from a local asset path; other
    entries go through ``cfg``, e.g.
    ``--env-kwargs '{"cfg": {"backend": "mujoco", "task_id": 2}}'``.

    ``from_pixels`` adds a rendered ``pixels`` observation at the config's
    ``render_width`` and ``render_height`` (see :func:`make_video_env`).

    :class:`~torchrl.envs.InitTracker` marks episode starts. The primer that
    carries the GRU state between steps comes from the policy
    (:func:`~torchrl.modules.get_primers_from_module`) and is appended where
    a policy meets an env: :func:`make_models`, :func:`make_evaluator` and
    :func:`make_render_policy`.
    """
    recorded = checkpoint if isinstance(checkpoint, Mapping) else {}
    recorded_env = {
        key: value
        for key, value in ((recorded.get("config") or {}).get("env") or {}).items()
        if key not in ASSET_KEYS
    }
    # rlrender passes Path and torch.device objects, which OmegaConf rejects.
    overrides = {
        "microduck_root": None if microduck_root is None else str(microduck_root),
        "root": None if root is None else str(root),
        "download": download,
        "num_envs": num_envs,
        "parallel": parallel,
        "device": None if device is None else str(device),
    }
    env_cfg = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(PACKAGE_DIR / "config.yaml").env,
            recorded_env,
            cfg or {},
            {key: value for key, value in overrides.items() if value is not None},
        ),
        resolve=True,
    )
    transforms = [InitTracker()]
    tasks = make_tasks(env_cfg["tasks"])
    if env_cfg["task_id"] is not None:
        task_id = int(env_cfg["task_id"])
        if not 0 <= task_id < len(tasks):
            raise ValueError(
                f"task_id={task_id} does not index the {len(tasks)} tasks of env.tasks."
            )
        # Every reset picks this task of the library; the env's own weighted
        # draw is only replaced, the library (and the ids) stay the same.
        weights = [0.0] * len(tasks)
        weights[task_id] = 1.0
        transforms.append(MicroDuckTaskSampler(weights, seed=env_cfg["seed"]))
    kwargs: dict[str, Any] = {
        "root": env_cfg["root"],
        "download": env_cfg["download"],
        "backend": env_cfg["backend"],
        "tasks": tasks,
        "action_scale": env_cfg["action_scale"],
        "diagnostics": env_cfg["diagnostics"],
        "num_envs": env_cfg["num_envs"],
        # MuJoCo state is float64, which MPS does not support: CUDA or CPU.
        "device": torch.device(
            env_cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "seed": env_cfg["seed"],
        "max_episode_steps": env_cfg["max_episode_steps"],
        "camera_id": env_cfg["camera_id"],
        "render_width": env_cfg["render_width"],
        "render_height": env_cfg["render_height"],
        "from_pixels": from_pixels,
    }
    if env_cfg["backend"] == "mujoco":
        kwargs["parallel"] = env_cfg["parallel"]
    elif env_cfg["backend"] == "mujoco-torch":
        kwargs["compile_step"] = env_cfg["compile_step"]
    return TransformedEnv(
        MicroDuckEnv(env_cfg["microduck_root"], **kwargs), Compose(*transforms)
    )


def make_video_env(
    cfg: DictConfig | Mapping[str, Any],
    task_ids: Sequence[int],
    *,
    recorder: VideoRecorder,
    width: int,
    height: int,
) -> TransformedEnv:
    """Build the env that films one task per tile of the evaluation video.

    A batched native env with one simulator per entry of ``task_ids``, each
    pinned to its task at every reset by
    :meth:`~torchrl.envs.MicroDuckTaskSampler.fixed`, rendering ``pixels`` at
    ``width`` x ``height`` into ``recorder``, whose ``make_grid`` tiles the
    batch (2x2 for four tasks, row-major) into one video.
    """
    env = make_env(
        OmegaConf.merge(
            cfg,
            {
                "backend": "mujoco",
                "task_id": None,
                "render_width": width,
                "render_height": height,
            },
        ),
        num_envs=len(task_ids),
        from_pixels=True,
    )
    env.append_transform(MicroDuckTaskSampler.fixed(task_ids))
    env.append_transform(recorder)
    return env


def record_task_grid(
    env: TransformedEnv,
    recorder: VideoRecorder,
    actor: ProbabilisticActor,
    *,
    steps: int,
    step: int,
) -> None:
    """Roll the deterministic actor out on the video env and log the grid at ``step``."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        env.rollout(steps, actor, break_when_any_done=False)
    recorder.dump(step=step)


def task_labels(tasks: Sequence[MicroDuckTask]) -> list[str]:
    """Metric group of every task of the library: its name, made unique.

    Duplicate names get their library index appended.
    """
    labels = [str(task.name) for task in tasks]
    if len(set(labels)) != len(labels):
        labels = [f"{label}#{index}" for index, label in enumerate(labels)]
    return labels


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


def make_models(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 128,
    policy_head: PolicyHead = "gaussian",
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    residual_scale: float = 0.2,
    initial_policy_scale: float = 0.05,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Create the GRU actor and the critic that shares its backbone.

    Every network is built on ``device``, and the GRU's recurrent-state primer
    is appended to ``env`` so its rollouts carry the state between steps.

    ``policy_head="gaussian"`` (default) is a plain Gaussian head trained from
    scratch; ``policy_head="gait-residual"`` wraps the closed-form gait with a
    learned residual. Returns the actor and the full value network (backbone plus value
    head) expected by :class:`~torchrl.objectives.value.GAE` and
    :class:`~torchrl.objectives.ClipPPOLoss`.
    """
    actor, critic = make_actor_critic(
        env.observation_spec["observation"].shape[-1],
        env.observation_spec["task_id"].n,
        device=device,
        hidden_size=hidden_size,
        policy_head=policy_head,
        gait=gait,
        residual_scale=residual_scale,
        initial_policy_scale=initial_policy_scale,
    )
    env.append_transform(get_primers_from_module(actor))
    return actor, critic


def make_render_policy(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    checkpoint: Mapping[str, Any] | None = None,
    hidden_size: int | None = None,
    policy_head: PolicyHead | None = None,
    gait: Mapping[str, float] | None = None,
    residual_scale: float | None = None,
    initial_policy_scale: float | None = None,
) -> ProbabilisticActor:
    """Build the actor whose weights an ``rlrender`` checkpoint provides.

    Architecture arguments default to the ``policy_kwargs`` recorded in the
    training checkpoint, which ``rlrender`` passes as ``checkpoint``; explicit
    ``--policy-kwargs`` override them.
    """
    recorded = (
        dict(checkpoint.get("policy_kwargs") or {})
        if isinstance(checkpoint, Mapping)
        else {}
    )
    overrides = {
        "hidden_size": hidden_size,
        "policy_head": policy_head,
        "gait": gait,
        "residual_scale": residual_scale,
        "initial_policy_scale": initial_policy_scale,
    }
    recorded.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    actor, _ = make_models(env, device=device, **recorded)
    return actor


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    actor: ProbabilisticActor,
    critic: TensorDictSequential,
    *,
    transitions: int,
    policy_kwargs: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Write a unified TorchRL checkpoint that ``rlrender`` and ``init_from`` read.

    The actor is the checkpoint policy, the Hydra ``config`` and the policy
    kwargs let :func:`make_env` and :func:`make_render_policy` rebuild the
    training setup, and the critic is kept alongside for resuming.
    """
    return save_render_checkpoint(
        path,
        actor,
        env_metadata={"policy_kwargs": dict(policy_kwargs)},
        frames=transitions,
        metrics=dict(metrics),
        config=dict(config),
        extra={"critic_state_dict": critic.state_dict()},
        format="archive",
    )


def load_parameters(
    path: str | Path,
    actor: ProbabilisticActor,
    critic: TensorDictSequential,
    *,
    task_mapping: Sequence[int] | None = None,
) -> int:
    """Load actor parameters and, when saved, critic parameters from a checkpoint.

    ``task_mapping`` optionally gives one source embedding index for each
    destination task, allowing a skill library to grow without changing the
    existing skills' inference weights. Optimizer state is not loaded.

    Returns:
        The number of transitions the checkpoint was trained on.
    """
    payload = load_checkpoint(path)
    if task_mapping is not None:
        for state in (
            payload["model_state_dict"],
            payload.get("critic_state_dict", {}),
        ):
            for key, value in state.items():
                if key.endswith(".task.weight"):
                    indices = torch.as_tensor(
                        task_mapping, dtype=torch.long, device=value.device
                    )
                    state[key] = value.index_select(0, indices)
    try:
        actor.load_state_dict(payload["model_state_dict"])
        if "critic_state_dict" in payload:
            critic.load_state_dict(payload["critic_state_dict"])
    except RuntimeError as err:
        raise RuntimeError(
            f"The checkpoint {path} was trained with policy kwargs "
            f"{payload.get('policy_kwargs')} and env config "
            f"{(payload.get('config') or {}).get('env')}; the current models must "
            "match them."
        ) from err
    return int(payload.get("frames", 0))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def microduck_metrics(
    trajectories: TensorDictBase, *, jumping: bool = False
) -> dict[str, float]:
    """Task metrics of the padded trajectory batch an :class:`Evaluator` collects.

    Speeds are the body-frame velocities read from the observation, so a
    policy that turns is still credited for walking. Means are taken over
    transitions, so an episode that falls after twenty steps does not weigh as
    much as one that walks for five hundred. ``task_score`` is in ``[0, 1]``
    for every task: velocity tracking error relative to the commanded speed
    for walking and sidestepping, stillness for standing, and with
    ``jumping=True`` the fraction of time with both feet off the ground
    (which needs the env's ``diagnostics``).

    With position diagnostics, ``drift_speed`` is net displacement divided
    by elapsed time in m/s; ``displacement_max`` also reports the largest
    excursion in metres, including trajectories that return to the start.
    Heading rates in rad/s are unwrapped per episode, with extrema exposing
    episodes that turn in the wrong direction despite a correct mean.
    ``ground_forward_speed`` and ``ground_lateral_speed`` use horizontal
    position differences and the interval's midpoint heading, so trunk pitch
    does not mix vertical hopping into measured ground speed.
    """
    mask = trajectories["collector", "mask"]
    lengths = mask.sum(-1)
    velocity = trajectories["next", "observation"][..., 6:8]
    command = trajectories["command"]
    error = (velocity - command).norm(dim=-1)
    velocity_score = 1 - (error / command.norm(dim=-1).clamp_min(0.1)).clamp(max=1.0)
    if ("next", "diagnostic_left_foot_contact") in trajectories.keys(True):
        airborne = (
            (trajectories["next", "diagnostic_left_foot_contact"][..., 0] < 0.5)
            & (trajectories["next", "diagnostic_right_foot_contact"][..., 0] < 0.5)
        ).float()
    else:
        airborne = torch.zeros_like(error)
    score = airborne if jumping else velocity_score
    episode_score = (score * mask).sum(-1) / lengths
    last = trajectories["next", "terminated"][..., 0].gather(
        -1, (lengths - 1).unsqueeze(-1)
    )
    metrics = {
        "tracking_error": float(error[mask].mean()),
        "forward_speed": float(velocity[..., 0][mask].mean()),
        "lateral_speed": float(velocity[..., 1][mask].mean()),
        "airborne_fraction": float(airborne[mask].mean()),
        "survival_rate": float((~last).float().mean()),
        "episode_length_min": float(lengths.min()),
        "task_score": float(score[mask].mean()),
        "task_score_min": float(episode_score.min()),
    }
    if ("next", "diagnostic_head_pitch") in trajectories.keys(True):
        for name in ("head_pitch", "head_yaw", "yaw_rate"):
            values = trajectories["next", f"diagnostic_{name}"][..., 0]
            metrics[name] = float(values[mask].mean())
            metrics[f"{name}_abs"] = float(values[mask].abs().mean())
            if name != "yaw_rate":
                metrics[f"{name}_abs_p95"] = float(values[mask].abs().quantile(0.95))
        height = trajectories["next", "diagnostic_height_gain"][..., 0]
        metrics["hop_height_max"] = float(height.masked_fill(~mask, 0).amax(-1).mean())
        metrics["planar_speed"] = float(velocity.norm(dim=-1)[mask].mean())
        pairs = mask[..., 1:] & mask[..., :-1]
        takeoffs = (airborne[..., 1:] > airborne[..., :-1]) & pairs
        landings = (airborne[..., 1:] < airborne[..., :-1]) & pairs
        metrics["takeoffs_per_episode"] = float(takeoffs.sum(-1).float().mean())
        metrics["landings_per_episode"] = float(landings.sum(-1).float().mean())
        metrics["hopping_episode_fraction"] = float(
            ((takeoffs.sum(-1) >= 2) & (landings.sum(-1) >= 2)).float().mean()
        )
    if ("next", "diagnostic_position_x") in trajectories.keys(True):
        # Net drift is distinct from the back-and-forth velocity of a hop.
        displacement = []
        offsets = []
        for name in ("position_x", "position_y", "time"):
            values = trajectories["next", f"diagnostic_{name}"][..., 0]
            if name != "time":
                offsets.append(values - values[..., :1])
            else:
                intervals = (values[..., 1:] - values[..., :-1]).clamp_min(1e-6)
            displacement.append(
                values.gather(-1, (lengths - 1).unsqueeze(-1)).squeeze(-1)
                - values[..., 0]
            )
        elapsed = displacement[2].clamp_min(1e-6)
        drift = torch.stack(displacement[:2], -1).norm(dim=-1) / elapsed
        metrics["drift_speed"] = float(drift.mean())
        metrics["drift_speed_max"] = float(drift.max())
        # Returning to the starting point must not hide a large excursion.
        distance = torch.stack(offsets, -1).norm(dim=-1).masked_fill(~mask, 0)
        metrics["displacement_max"] = float(distance.max())
        heading = trajectories["next", "diagnostic_heading"][..., 0]
        delta = heading[..., 1:] - heading[..., :-1]
        delta = torch.atan2(delta.sin(), delta.cos())
        delta = delta * (mask[..., 1:] & mask[..., :-1])
        heading_rate = delta.sum(-1) / elapsed
        metrics["heading_rate"] = float(heading_rate.mean())
        metrics["heading_rate_min"] = float(heading_rate.min())
        metrics["heading_rate_max"] = float(heading_rate.max())
        position = torch.stack(offsets, -1)
        ground_velocity = (position[..., 1:, :] - position[..., :-1, :]) / intervals[
            ..., None
        ]
        midpoint = heading[..., :-1] + 0.5 * delta
        vx, vy = ground_velocity.unbind(-1)
        pairs = mask[..., 1:] & mask[..., :-1]
        count = pairs.sum().clamp_min(1)
        metrics["ground_forward_speed"] = float(
            (vx * midpoint.cos() + vy * midpoint.sin())[pairs].sum() / count
        )
        metrics["ground_lateral_speed"] = float(
            (-vx * midpoint.sin() + vy * midpoint.cos())[pairs].sum() / count
        )
    return metrics


def make_evaluator(
    env: TransformedEnv,
    actor: ProbabilisticActor,
    *,
    label: str,
    jumping: bool = False,
    num_episodes: int,
    steps: int,
) -> Evaluator:
    """Deterministic evaluator of ``actor`` on an env pinned to one task.

    Metrics are logged under ``evaluation/<label>/``; the evaluator adds
    ``reward`` and ``episode_length`` to :func:`microduck_metrics`, whose
    ``task_score`` measures airborne time when ``jumping`` is set. The actor's
    recurrent-state primer is appended to ``env``.
    """
    env.append_transform(get_primers_from_module(actor))
    return Evaluator(
        env,
        actor,
        num_trajectories=num_episodes,
        max_steps=steps,
        metrics_fn=partial(microduck_metrics, jumping=jumping),
        log_prefix=f"evaluation/{label}",
    )


def evaluation_metrics(results: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Merge the per-task evaluator results and average them over tasks."""
    metrics = {
        key.replace("/custom/", "/"): float(value)
        for result in results
        for key, value in result.items()
        if isinstance(value, (int, float))
    }
    for name in (
        "reward",
        "episode_length",
        "tracking_error",
        "forward_speed",
        "survival_rate",
        "task_score",
    ):
        values = [value for key, value in metrics.items() if key.endswith(f"/{name}")]
        metrics[f"evaluation/{name}"] = sum(values) / len(values)
    return metrics


def evaluation_score(results: Sequence[Mapping[str, float]]) -> tuple[float, ...]:
    """Rank checkpoints by survival, then the worst task, then the mean task score, then return.

    A short forward fall can earn a higher raw return than a full episode of
    balanced walking, so survival and episode length are compared before any
    task score or reward figure.
    """

    def per_command(name: str) -> list[float]:
        return [
            float(value)
            for result in results
            for key, value in result.items()
            if key.endswith(f"/{name}")
        ]

    survived = sum(
        rate * episodes
        for rate, episodes in zip(
            per_command("survival_rate"), per_command("num_episodes")
        )
    )
    return (
        survived,
        min(per_command("episode_length_min")),
        min(per_command("task_score_min")),
        sum(per_command("task_score")) / len(results),
        sum(per_command("reward")) / len(results),
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def _collection_metrics(data: TensorDictBase) -> tuple[dict[str, float], int]:
    rewards = data["next", "reward"].squeeze(-1)
    traj_ids = data["collector", "traj_ids"]
    done = data["next", "done"].squeeze(-1)
    terminated = data["next", "terminated"].squeeze(-1)
    unique_ids, inverse = torch.unique(traj_ids, return_inverse=True)
    returns = torch.zeros(unique_ids.numel()).index_add_(0, inverse, rewards)
    lengths = torch.zeros(unique_ids.numel()).index_add_(
        0, inverse, torch.ones_like(rewards)
    )
    ends = done.nonzero().squeeze(-1)
    tracking_error = (data["observation"][..., 6:8] - data["command"]).norm(dim=-1)
    metrics = {
        "collection/reward_mean": float(rewards.mean()),
        "collection/tracking_error_mean": float(tracking_error.abs().mean()),
        "episode/return_mean": float(returns.mean()),
        "episode/length_mean": float(lengths.mean()),
        "episode/length_min": float(lengths.min()),
        "episode/survival_rate": float((~terminated[ends]).float().mean()),
    }
    return metrics, int(unique_ids.numel())


class _PriorCollector(Collector):
    """Yield the requested transition budget using complete episodes only."""

    transitions_per_update: int

    def __iter__(self):
        iterator = super().__iter__()
        try:
            while True:
                timeit.reset()
                with timeit("collect"):
                    while len(self.replay_buffer) < self.transitions_per_update:
                        next(iterator)
                yield self.replay_buffer[:].refine_names("time")
        finally:
            iterator.close()

    def shutdown(self, timeout=None, close_env=False, raise_on_error=True):
        # The recipe/caller owns the environment, including legacy train_ppo
        # callers that evaluate or start another run with it afterwards.
        return super().shutdown(timeout, close_env, raise_on_error)


class _PriorTrainingHooks:
    """Whole-episode preparation, evaluation and checkpoint state for the prior."""

    def __init__(
        self,
        trainer,
        actor,
        critic,
        replay_buffer,
        advantage,
        *,
        minibatch_trajectories,
        scheduler,
        evaluators,
        evaluation_interval,
        video_recorder,
        video_interval,
        best_checkpoint_path,
        latest_checkpoint_path,
        policy_kwargs,
        config,
        logger,
    ):
        self.trainer = trainer
        self.actor, self.critic = actor, critic
        self.replay_buffer, self.advantage = replay_buffer, advantage
        self.minibatch_trajectories, self.scheduler = minibatch_trajectories, scheduler
        self.evaluators, self.evaluation_interval = evaluators, evaluation_interval
        self.video_recorder, self.video_interval = video_recorder, video_interval
        self.best_checkpoint_path, self.latest_checkpoint_path = (
            best_checkpoint_path,
            latest_checkpoint_path,
        )
        self.policy_kwargs, self.config, self.logger = policy_kwargs, config, logger
        self.iteration = self.evaluations = 0
        self.best_score = self.best_state = None
        self.history, self.updates = [], []
        self.metrics = {}
        self.video_env = None
        self.env = None

    def setup(self):
        if self.evaluation_interval is not None and self.evaluations == 0:
            self.log(self.evaluate())

    @torch.no_grad()
    def prepare(self, data):
        self.iteration += 1
        self.metrics, trajectories = _collection_metrics(data)
        self.metrics["collection/transitions"] = float(data.numel())
        self.metrics["collection/trajectories"] = float(trajectories)
        self.trainer.optim_steps_per_batch = max(
            1, math.ceil(trajectories / self.minibatch_trajectories)
        )
        self.updates = []
        self.trained_transitions = 0
        with timeit("advantage"), set_recurrent_mode(True):
            processed = data.to(next(self.actor.parameters()).device)
            # Every complete episode may need a terminal bootstrap observation.
            # A fixed one-row shifted budget would discard the end of this
            # concatenated batch whenever multiple episodes truncate.
            self.advantage.shifted_budget = trajectories
            self.advantage(processed)
            self.replay_buffer[: data.numel()] = processed.cpu()
        target = processed["value_target"]
        self.metrics["value/explained_variance"] = float(
            1
            - (target - processed["state_value"]).var()
            / target.var().clamp_min(torch.finfo(target.dtype).eps)
        )
        return data

    def sample(self, batch):
        sample = (
            self.replay_buffer.sample()
            .to(next(self.actor.parameters()).device)
            .refine_names("time")
        )
        self.trained_transitions += sample.numel()
        return sample

    def process_loss(self, batch, losses):
        self.updates.append(
            {key: float(value.detach().mean()) for key, value in losses.items()}
        )
        return losses

    def finish_batch(self):
        metrics = self.metrics
        for key in self.updates[0]:
            metrics[f"ppo/{key}"] = sum(row[key] for row in self.updates) / len(
                self.updates
            )
        if self.scheduler is not None:
            self.scheduler.step(metrics["ppo/kl_approx"])
        metrics["ppo/learning_rate"] = self.trainer.optimizer.param_groups[0]["lr"]
        self.replay_buffer.empty()
        # OnPolicyTrainer synchronized collection weights before this hook.
        self.trainer.collector.reset()
        metrics["progress/transitions"] = float(self.trainer.collected_frames)
        timings = timeit.todict(prefix="time")
        metrics.update(timings)
        metrics["throughput/collection_transitions_per_second"] = metrics[
            "collection/transitions"
        ] / max(timings["time/collect"], 1e-9)
        if self.evaluation_interval is not None and (
            self.iteration % self.evaluation_interval == 0
            or self.trainer.collected_frames >= self.trainer.total_frames
        ):
            metrics.update(self.evaluate())
        self.history.append(dict(metrics))
        self.log(metrics)
        torchrl_logger.info(
            "MicroDuck PPO transitions=%d/%d reward=%+.4f lr=%.2e",
            self.trainer.collected_frames,
            self.trainer.total_frames,
            metrics["collection/reward_mean"],
            metrics["ppo/learning_rate"],
        )

    def evaluate(self):
        step = self.trainer.collected_frames
        results = [
            evaluator.evaluate(weights=self.actor, step=step)
            for evaluator in self.evaluators
        ]
        if (
            self.video_recorder is not None
            and self.evaluations % self.video_interval == 0
        ):
            self.video_recorder(step)
        self.evaluations += 1
        metrics, score = evaluation_metrics(results), evaluation_score(results)
        is_best = self.best_score is None or score > self.best_score
        if is_best:
            self.best_score = score
            self.best_state = (
                deepcopy(self.actor.state_dict()),
                deepcopy(self.critic.state_dict()),
            )
        paths = [self.latest_checkpoint_path]
        if is_best:
            paths.append(self.best_checkpoint_path)
        for path in paths:
            if path is not None:
                save_checkpoint(
                    path,
                    self.actor,
                    self.critic,
                    transitions=step,
                    policy_kwargs=self.policy_kwargs,
                    metrics={**metrics, "evaluation_score": list(score)},
                    config=self.config,
                )
        metrics["evaluation/is_best"] = float(is_best)
        return metrics

    def log(self, metrics):
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.log_scalar(key, value, step=self.trainer.collected_frames)

    def close(self):
        for evaluator in self.evaluators or ():
            evaluator.shutdown()
        self.evaluators = []
        if self.video_env is not None and not self.video_env.is_closed:
            self.video_env.close()
        if self.env is not None and not self.env.is_closed:
            self.env.close()
        if self.logger is not None and hasattr(self.logger.experiment, "finish"):
            self.logger.experiment.finish()
            self.logger = None

    def state_dict(self):
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "best_score": self.best_score,
            "best_state": self.best_state,
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }

    def load_state_dict(self, state):
        self.iteration, self.evaluations = state["iteration"], state["evaluations"]
        self.best_score, self.best_state = state["best_score"], state["best_state"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])


def make_trainer(
    env: TransformedEnv,
    actor: ProbabilisticActor,
    critic: TensorDictSequential,
    *,
    total_transitions: int = 10_000_000,
    transitions_per_update: int = 16_384,
    max_episode_steps: int = 500,
    epochs: int = 5,
    minibatch_trajectories: int = 32,
    learning_rate: float = 1e-4,
    target_kl: float | None = 0.01,
    entropy_coeff: float = 0.0,
    critic_coeff: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 1.0,
    per_task_advantage: bool = True,
    evaluators: Sequence[Evaluator] | None = None,
    video_recorder: Callable[[int], None] | None = None,
    video_interval: int | None = None,
    evaluation_interval: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    latest_checkpoint_path: str | Path | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    logger: Logger | None = None,
    loss_kwargs: Mapping[str, Any] | None = None,
    target_net_updater: Callable | None = None,
    save_trainer_file: str | Path | None = None,
) -> PPOTrainer:
    """Build the prior PPOTrainer around complete episodes and recurrent minibatches.

    All optimization is owned by PPOTrainer. Collection waits for at least the
    requested transition count in complete episodes; GAE normalizes per task
    before whole-episode sampling. Each update discards in-flight episodes after
    synchronizing the collector. Best/latest exports are separate from resumable
    optimizer, proximal actor, scheduler, RNG and evaluation-hook state.
    The caller owns the supplied environment and evaluation resources.
    """
    if (
        min(total_transitions, transitions_per_update, epochs, minibatch_trajectories)
        < 1
    ):
        raise ValueError("PPO transition, epoch and minibatch sizes must be positive.")
    if transitions_per_update < max_episode_steps:
        raise ValueError(
            "transitions_per_update must hold at least one full episode "
            f"({max_episode_steps} transitions)."
        )
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when provided.")
    if evaluation_interval is not None and not evaluators:
        raise ValueError("evaluation_interval requires evaluators.")
    if video_recorder is not None and (video_interval is None or video_interval < 1):
        raise ValueError("video_recorder requires a positive video_interval.")
    checkpointing = (
        best_checkpoint_path is not None or latest_checkpoint_path is not None
    )
    if checkpointing and evaluation_interval is None:
        raise ValueError("Checkpoint paths require periodic evaluation.")
    if checkpointing and (config is None or policy_kwargs is None):
        raise ValueError("Checkpoint paths require config and policy_kwargs.")

    device = next(actor.parameters()).device
    num_envs = env.batch_size.numel()
    # Headroom for the episodes that finish while the last poll completes, so
    # the buffer never wraps around before its content is consumed.
    capacity = transitions_per_update + num_envs * max_episode_steps
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity, ndim=1),
        sampler=SliceSampler(
            num_slices=minibatch_trajectories,
            traj_key=("collector", "traj_ids"),
            strict_length=False,
            cache_values=True,
        ),
        batch_size=minibatch_trajectories * max_episode_steps,
    )
    collector = _PriorCollector(
        env,
        actor,
        frames_per_batch=num_envs * min(50, max_episode_steps),
        total_frames=-1,
        replay_buffer=replay_buffer,
        trajs_per_batch=1,
        trajs_per_write=1,
        storing_device="cpu",
    )
    collector.transitions_per_update = transitions_per_update
    # Advantages standardized within each task of the library, so tasks whose
    # rewards vary least keep a learning signal next to the walking tasks.
    advantage = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
        average_gae=per_task_advantage,
        group_key="task_id" if per_task_advantage else None,
        shifted=True,
        deactivate_vmap=True,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
        normalize_advantage=not per_task_advantage,
        **dict(loss_kwargs or {}),
    )
    # Recompute every recurrent distribution from complete sequences, including
    # the distinct proximal actor. Collector inference remains sequential.
    loss_module.forward = set_recurrent_mode(True)(loss_module.forward)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = (
        KLAdaptiveLR(optimizer, target_kl=target_kl) if target_kl is not None else None
    )

    updater = (
        target_net_updater(loss_module) if target_net_updater is not None else None
    )
    trainer = PPOTrainer(
        collector=collector,
        total_frames=total_transitions,
        frame_skip=1,
        optim_steps_per_batch=1,
        num_epochs=epochs,
        loss_module=loss_module,
        optimizer=optimizer,
        target_net_updater=updater,
        clip_norm=max_grad_norm,
        add_gae=False,
        enable_logging=False,
        progress_bar=False,
        auto_log_optim_steps=False,
        checkpoint=Checkpoint(rng=GlobalRNGState()),
        save_trainer_file=save_trainer_file,
        save_trainer_interval=transitions_per_update,
    )
    hooks = _PriorTrainingHooks(
        trainer,
        actor,
        critic,
        replay_buffer,
        advantage,
        minibatch_trajectories=minibatch_trajectories,
        scheduler=scheduler,
        evaluators=evaluators,
        evaluation_interval=evaluation_interval,
        video_recorder=video_recorder,
        video_interval=video_interval,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        policy_kwargs=policy_kwargs,
        config=config,
        logger=logger,
    )
    trainer.prior_hooks = hooks
    trainer.register_module("prior_hooks", hooks)
    trainer.register_op("setup", hooks.setup)
    trainer.register_op("batch_process", hooks.prepare)
    trainer.register_op("process_optim_batch", hooks.sample)
    trainer.register_op("process_loss", hooks.process_loss)
    trainer.register_op("post_steps", hooks.finish_batch)
    return trainer


def train_ppo(
    env: TransformedEnv,
    actor: ProbabilisticActor,
    critic: TensorDictSequential,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Run the prior PPOTrainer, retaining the legacy best-actor return behavior.

    The caller owns the environment and evaluation resources. New resumable
    recipes use :func:`make_trainer` so the live actor stays paired with its
    optimizer; best/latest inference exports remain separate.
    """
    trainer = make_trainer(env, actor, critic, **kwargs)
    try:
        trainer.train()
    finally:
        trainer.collector.shutdown(close_env=False)
    if trainer.prior_hooks.best_state is not None:
        actor.load_state_dict(trainer.prior_hooks.best_state[0])
        critic.load_state_dict(trainer.prior_hooks.best_state[1])
    return trainer.prior_hooks.history


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def make_training(recipe: DictConfig) -> PPOTrainer:
    """Build the configured skill/prior trainer and its owned evaluation resources."""
    cfg = recipe
    if cfg.smoke:
        # One native simulator on CPU: a pipeline check, not a speed test.
        cfg.env.backend = "mujoco"
        cfg.env.device = "cpu"
        cfg.env.num_envs = 1
        cfg.env.max_episode_steps = 50
        cfg.ppo.total_transitions = 200
        cfg.ppo.transitions_per_update = 100
        cfg.ppo.epochs = 1
        cfg.ppo.minibatch_trajectories = 2
        cfg.evaluation.interval = 1
        cfg.evaluation.num_episodes = 1
        cfg.evaluation.steps = 20
        cfg.evaluation.best_checkpoint_path = None
        cfg.evaluation.latest_checkpoint_path = None
        cfg.logger.backend = None
    if cfg.logger.backend == "wandb" and not cfg.logger.entity:
        raise ValueError(
            "W&B logging requires logger.entity so runs do not land in an "
            "unintended default workspace."
        )
    torch.manual_seed(cfg.env.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    tasks = make_tasks(config["env"]["tasks"])
    labels = task_labels(tasks)
    # The closed-form gait follows the clock the tasks expose in the
    # observation; its frequency is the first task's.
    gait = replace(
        MicroDuckGaitConfig(), frequency_hz=float(tasks[0].gait_frequency_hz)
    )
    policy_head = "gait-residual" if cfg.policy.from_prior else "gaussian"
    policy_kwargs = {
        "hidden_size": cfg.policy.hidden_size,
        "policy_head": policy_head,
        "gait": asdict(gait),
        "residual_scale": cfg.policy.residual_scale,
        "initial_policy_scale": cfg.policy.initial_policy_scale
        or (0.05 if cfg.policy.from_prior else 1.0),
    }
    env = make_env(cfg.env)
    evaluators: list[Evaluator] = []
    video_env = None
    logger = None
    try:
        actor, critic = make_models(env, device=env.device, **policy_kwargs)
        if cfg.policy.init_from:
            trained = load_parameters(cfg.policy.init_from, actor, critic)
            torchrl_logger.info(
                "Initialized actor and critic from %s (%d transitions).",
                cfg.policy.init_from,
                trained,
            )
        if cfg.evaluation.interval is not None:
            jump_index = list(MicroDuckEnv.REWARD_TERMS).index("jump")
            for task_id, (task, label) in enumerate(zip(tasks, labels)):
                # One single-env evaluator per task of the library, pinned to
                # it at every reset; diagnostics feed the airborne metric.
                evaluation_cfg = OmegaConf.merge(
                    cfg.env, {"task_id": task_id, "diagnostics": True}
                )
                evaluators.append(
                    make_evaluator(
                        make_env(evaluation_cfg, num_envs=1, parallel=False),
                        actor,
                        label=label,
                        jumping=bool(task.reward_weights[jump_index] > 0),
                        num_episodes=cfg.evaluation.num_episodes,
                        steps=cfg.evaluation.steps,
                    )
                )
        logger = get_logger(
            cfg.logger.backend,
            logger_name="microduck_ppo",
            experiment_name=cfg.logger.exp_name
            or generate_exp_name("microduck", f"{policy_head}-{cfg.env.backend}"),
            wandb_kwargs={
                "project": cfg.logger.project,
                "entity": cfg.logger.entity,
                "mode": cfg.logger.mode,
                "config": config,
            },
        )
        video_callback = None
        if cfg.evaluation.video.interval is not None and logger is not None:
            # A 2x2 grid of four tasks filmed in parallel, logged at every
            # `video.interval`-th evaluation.
            grid_recorder = VideoRecorder(
                logger, tag="evaluation/task_grid", make_grid=True, fps=50
            )
            video_env = make_video_env(
                cfg.env,
                list(cfg.evaluation.video.tasks),
                recorder=grid_recorder,
                width=cfg.evaluation.video.width,
                height=cfg.evaluation.video.height,
            )

            def _video_callback(step: int) -> None:
                record_task_grid(
                    video_env,
                    grid_recorder,
                    actor,
                    steps=cfg.evaluation.video.steps,
                    step=step,
                )

            video_callback = _video_callback

        trainer = make_trainer(
            env,
            actor,
            critic,
            **config["ppo"],
            max_episode_steps=cfg.env.max_episode_steps,
            evaluators=evaluators,
            evaluation_interval=cfg.evaluation.interval,
            video_recorder=video_callback,
            video_interval=cfg.evaluation.video.interval,
            best_checkpoint_path=cfg.evaluation.best_checkpoint_path,
            latest_checkpoint_path=cfg.evaluation.latest_checkpoint_path,
            policy_kwargs=policy_kwargs,
            config=config,
            logger=logger,
            loss_kwargs=config.get("loss"),
            target_net_updater=instantiate(cfg.target_net_updater)
            if cfg.get("target_net_updater")
            else None,
            save_trainer_file=cfg.save_trainer_file,
        )
        trainer.prior_hooks.env = env
        trainer.prior_hooks.video_env = video_env
        trainer.register_op("shutdown", trainer.prior_hooks.close)
        if cfg.resume:
            trainer.load_from_file(cfg.resume)
        return trainer
    except BaseException:
        if logger is not None and hasattr(logger.experiment, "finish"):
            logger.experiment.finish()
        for evaluator in evaluators:
            evaluator.shutdown()
        if video_env is not None and not video_env.is_closed:
            video_env.close()
        if not env.is_closed:
            env.close()
        raise


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Instantiate the prior recipe and run the shared TorchRL trainer."""
    trainer = instantiate(cfg.trainer, recipe=cfg, _recursive_=False)
    try:
        trainer.train()
    finally:
        trainer.prior_hooks.close()
        trainer.collector.shutdown(close_env=False)


if __name__ == "__main__":
    main()
