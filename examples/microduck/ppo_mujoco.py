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
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl import timeit, torchrl_logger
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
    GRUModule,
    ProbabilisticActor,
    set_recurrent_mode,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, KLAdaptiveLR
from torchrl.objectives.value import GAE
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.render import load_checkpoint, save_render_checkpoint

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent))

from examples.microduck.heuristic_gait import (  # noqa: E402
    MicroDuckGaitActor,
    MicroDuckGaitConfig,
)

PolicyHead = Literal["gait-residual", "gaussian"]
RECURRENT_STATE_KEY = "recurrent_state"
TASK_PRESETS = (
    "make_task",
    "tracking_task",
    "standing_task",
    "speed_range_task",
    "sidestep_task",
    "jump_task",
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


def make_tasks(entries: Sequence[Mapping[str, Any]]) -> list[MicroDuckTask]:
    """Build the task library from the ``env.tasks`` list of ``config.yaml``.

    Each entry names a :class:`~torchrl.envs.MicroDuckEnv` preset in
    :data:`TASK_PRESETS` under ``preset`` and passes its other keys to it,
    e.g. ``{preset: tracking_task, speed: 0.2, weight: 2.0}`` or
    ``{preset: jump_task, reward_weights: {jump: 8.0}, name: hop}``.
    """
    if not entries:
        raise ValueError("env.tasks needs at least one task entry.")
    tasks = []
    for entry in entries:
        kwargs = dict(entry)
        preset = kwargs.pop("preset", None)
        if preset not in TASK_PRESETS:
            raise ValueError(
                f"Unknown task preset {preset!r}; env.tasks entries name one of "
                f"{TASK_PRESETS} under `preset`."
            )
        tasks.append(getattr(MicroDuckEnv, preset)(**kwargs))
    return tasks


def make_video_env(
    cfg: DictConfig | Mapping[str, Any],
    task_ids: Sequence[int],
    *,
    width: int,
    height: int,
) -> TransformedEnv:
    """Build the env that films one task per tile of the evaluation video.

    A batched native env with one simulator per entry of ``task_ids``, each
    pinned to its task at every reset by
    :meth:`~torchrl.envs.MicroDuckTaskSampler.fixed`, rendering ``pixels`` at
    ``width`` x ``height``. Four tasks make the 2x2 grid of
    :func:`tile_task_grid`.
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
    return env


def tile_task_grid(pixels: torch.Tensor) -> torch.Tensor:
    """Tile the per-env frames of a rollout into one video, row-major.

    ``pixels`` has shape ``(*batch, T, H, W, 3)`` with a batch of ``n`` envs;
    the result has shape ``(T, 3, rows * H, cols * W)`` with ``cols`` the
    ceiling of ``sqrt(n)`` (2x2 for four envs), in the ``uint8`` layout that
    :meth:`~torchrl.record.loggers.Logger.log_video` expects.
    """
    frames = pixels.reshape(-1, *pixels.shape[-4:])
    n, cols = frames.shape[0], math.ceil(math.sqrt(frames.shape[0]))
    rows = math.ceil(n / cols)
    if rows * cols != n:
        frames = torch.cat(
            (frames, frames.new_zeros(rows * cols - n, *frames.shape[1:]))
        )
    grid = torch.cat(
        [
            torch.cat(list(frames[r * cols : (r + 1) * cols]), dim=-2)
            for r in range(rows)
        ],
        dim=-3,
    )
    return grid.permute(0, 3, 1, 2).contiguous()


def record_task_grid(
    env: TransformedEnv, actor: ProbabilisticActor, steps: int
) -> torch.Tensor:
    """Roll the deterministic actor out on the video env and return the tiled video."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        rollout = env.rollout(steps, actor, break_when_any_done=False)
    return tile_task_grid(rollout["pixels"])


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


class TaskConditionedEncoder(nn.Module):
    """Observation encoder conditioned on the task index.

    The observation carries the command but no other task parameter, so a
    learned embedding of the task index tells the policy which task of the
    library the env is in (for instance jumping, whose command is zero).
    """

    def __init__(
        self,
        observation_dim: int,
        num_tasks: int,
        hidden_size: int,
        *,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.observation = nn.Linear(observation_dim, hidden_size, device=device)
        self.task = nn.Embedding(num_tasks, hidden_size, device=device)

    def forward(self, observation: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self.observation(observation) + self.task(task_id.squeeze(-1))
        )


class GaitResidualHead(nn.Module):
    """Gaussian policy head: closed-form gait plus a bounded learned residual.

    The residual is a zero-initialized linear map from the recurrent features,
    squashed by ``tanh`` and scaled by ``residual_scale``, so the initial policy
    reproduces the gait exactly. The exploration scale is state independent.
    """

    def __init__(
        self,
        hidden_size: int,
        gait: MicroDuckGaitConfig | Mapping[str, float] | None,
        *,
        residual_scale: float,
        initial_policy_scale: float,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.gait = MicroDuckGaitActor(gait)
        self.residual_scale = float(residual_scale)
        self.residual = nn.Linear(hidden_size, MicroDuckEnv.NUM_JOINTS, device=device)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.scale = nn.Parameter(torch.zeros(MicroDuckEnv.NUM_JOINTS, device=device))
        self.param_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{initial_policy_scale}"
        )

    def forward(
        self, features: torch.Tensor, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nominal = self.gait.gait_action(observation)
        mean = nominal + self.residual_scale * torch.tanh(self.residual(features))
        pre_tanh_mean = torch.atanh(mean.clamp(-0.999, 0.999))
        scale = self.scale.expand_as(pre_tanh_mean)
        return self.param_extractor(torch.cat((pre_tanh_mean, scale), dim=-1))


class GaussianHead(nn.Module):
    """Plain Gaussian policy head for training from scratch.

    The mean starts near zero, which is the ``STAND`` pose, and the
    state-independent exploration scale starts at ``initial_policy_scale``.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        initial_policy_scale: float,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.loc = nn.Linear(hidden_size, MicroDuckEnv.NUM_JOINTS, device=device)
        nn.init.orthogonal_(self.loc.weight, gain=0.01)
        nn.init.zeros_(self.loc.bias)
        self.scale = nn.Parameter(torch.zeros(MicroDuckEnv.NUM_JOINTS, device=device))
        self.param_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{initial_policy_scale}"
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc(features)
        return self.param_extractor(torch.cat((loc, self.scale.expand_as(loc)), -1))


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
    if not math.isfinite(initial_policy_scale) or initial_policy_scale <= 0:
        raise ValueError("initial_policy_scale must be finite and positive.")
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be finite and positive.")
    device = torch.device(device)
    observation_dim = env.observation_spec["observation"].shape[-1]
    embed = TensorDictModule(
        TaskConditionedEncoder(
            observation_dim,
            env.observation_spec["task_id"].n,
            hidden_size,
            device=device,
        ),
        in_keys=["observation", "task_id"],
        out_keys=["embed"],
    )
    gru = GRUModule(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        in_keys=["embed", RECURRENT_STATE_KEY, "is_init"],
        out_keys=["features", ("next", RECURRENT_STATE_KEY)],
        device=device,
    )
    backbone = TensorDictSequential(embed, gru)
    if policy_head == "gait-residual":
        actor_head = TensorDictModule(
            GaitResidualHead(
                hidden_size,
                gait,
                residual_scale=residual_scale,
                initial_policy_scale=initial_policy_scale,
                device=device,
            ),
            in_keys=["features", "observation"],
            out_keys=["loc", "scale"],
        )
    elif policy_head == "gaussian":
        actor_head = TensorDictModule(
            GaussianHead(
                hidden_size, initial_policy_scale=initial_policy_scale, device=device
            ),
            in_keys=["features"],
            out_keys=["loc", "scale"],
        )
    else:
        raise ValueError(f"Unknown policy_head {policy_head!r}.")
    actor = ProbabilisticActor(
        module=TensorDictSequential(backbone, actor_head),
        spec=env.action_spec_unbatched,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )
    value_head = ValueOperator(
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, device=device),
        ),
        in_keys=["features"],
    )
    critic = TensorDictSequential(backbone, value_head)
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
    path: str | Path, actor: ProbabilisticActor, critic: TensorDictSequential
) -> int:
    """Load actor and critic parameters from a checkpoint written by :func:`save_checkpoint`.

    Returns:
        The number of transitions the checkpoint was trained on.
    """
    payload = load_checkpoint(path)
    try:
        actor.load_state_dict(payload["model_state_dict"])
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
    return {
        "tracking_error": float(error[mask].mean()),
        "forward_speed": float(velocity[..., 0][mask].mean()),
        "lateral_speed": float(velocity[..., 1][mask].mean()),
        "airborne_fraction": float(airborne[mask].mean()),
        "survival_rate": float((~last).float().mean()),
        "episode_length_min": float(lengths.min()),
        "task_score": float(score[mask].mean()),
        "task_score_min": float(episode_score.min()),
    }


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


def train_ppo(
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
) -> list[dict[str, float]]:
    """Train the recurrent policy with PPO on whole episodes.

    Each iteration collects at least ``transitions_per_update`` transitions of
    complete episodes into the replay buffer, computes GAE once over the buffer
    in recurrent mode, runs ``epochs`` passes of whole-episode minibatches, then
    empties the buffer and drops the collector's in-flight episodes so the next
    collection only contains data from the updated policy.

    With ``per_task_advantage`` the GAE standardizes advantages within each
    task of the library (its ``group_key`` is the ``task_id``) rather than the
    loss over the whole minibatch, so the tasks whose rewards vary least
    (standing, a hop that has not happened yet) keep a learning signal next to
    the walking tasks.

    Every ``video_interval`` evaluations, ``video_recorder`` is called with
    the transition count, for a video logged alongside the metrics (see
    :func:`record_task_grid`); it is off by default because video storage is
    not free.

    Every ``evaluation_interval`` iterations the ``evaluators`` (one per
    task of the library, see :func:`make_evaluator`) run the actor's current
    weights. ``best_checkpoint_path`` receives the best-scoring parameters and
    ``latest_checkpoint_path`` the current ones at every evaluation, so
    training progress can be rendered while the best checkpoint protects
    against regressions. Checkpoints record ``policy_kwargs`` and the Hydra
    ``config`` so ``rlrender`` rebuilds the actor and the env through
    :func:`make_render_policy` and :func:`make_env`.

    Returns:
        One metrics dictionary per iteration. When evaluation is enabled the
        actor and critic end up holding the best-scoring parameters.
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
    collector = Collector(
        env,
        actor,
        frames_per_batch=num_envs * min(50, max_episode_steps),
        total_frames=-1,
        replay_buffer=replay_buffer,
        trajs_per_batch=1,
        trajs_per_write=1,
        storing_device="cpu",
    )
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
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = (
        KLAdaptiveLR(optimizer, target_kl=target_kl) if target_kl is not None else None
    )

    history: list[dict[str, float]] = []
    collected = 0
    iteration = 0
    best_score: tuple[float, ...] | None = None
    best_state: tuple[dict, dict] | None = None

    def checkpoint(path: str | Path | None, step: int, metrics, score) -> None:
        if path is None:
            return
        save_checkpoint(
            path,
            actor,
            critic,
            transitions=step,
            policy_kwargs=policy_kwargs,
            metrics={**metrics, "evaluation_score": list(score)},
            config=config,
        )

    evaluations = 0

    def evaluate(step: int) -> dict[str, float]:
        nonlocal best_score, best_state, evaluations
        results = [
            evaluator.evaluate(weights=actor, step=step) for evaluator in evaluators
        ]
        if video_recorder is not None and evaluations % video_interval == 0:
            with timeit("video"):
                video_recorder(step)
        evaluations += 1
        metrics = evaluation_metrics(results)
        score = evaluation_score(results)
        checkpoint(latest_checkpoint_path, step, metrics, score)
        if best_score is None or score > best_score:
            best_score = score
            best_state = (deepcopy(actor.state_dict()), deepcopy(critic.state_dict()))
            checkpoint(best_checkpoint_path, step, metrics, score)
        metrics["evaluation/is_best"] = float(score == best_score)
        torchrl_logger.info(
            "MicroDuck evaluation transitions=%d survival=%.2f length=%.1f "
            "forward_speed=%+.4f tracking_error=%.4f",
            step,
            metrics["evaluation/survival_rate"],
            metrics["evaluation/episode_length"],
            metrics["evaluation/forward_speed"],
            metrics["evaluation/tracking_error"],
        )
        return metrics

    def log(metrics: Mapping[str, float], step: int) -> None:
        if logger is None:
            return
        for key, value in metrics.items():
            logger.log_scalar(key, value, step=step)

    if evaluation_interval is not None:
        log(evaluate(0), step=0)

    collector_iterator = iter(collector)
    try:
        while collected < total_transitions:
            iteration += 1
            timeit.reset()
            with timeit("collect"):
                while len(replay_buffer) < transitions_per_update:
                    next(collector_iterator)
            data = replay_buffer[:]
            num_transitions = data.numel()
            collected += num_transitions
            metrics, num_trajectories = _collection_metrics(data)

            with timeit("advantage"), torch.no_grad(), set_recurrent_mode(True):
                processed = data.to(device)
                advantage(processed)
                replay_buffer[:num_transitions] = processed.to("cpu")
            value_target = processed["value_target"]
            metrics["value/explained_variance"] = float(
                1.0
                - (value_target - processed["state_value"]).var()
                / value_target.var().clamp_min(torch.finfo(value_target.dtype).eps)
            )

            updates_per_epoch = max(
                1, math.ceil(num_trajectories / minibatch_trajectories)
            )
            updates = []
            trained_transitions = 0
            with timeit("train"):
                for _ in range(epochs):
                    for _ in range(updates_per_epoch):
                        sample = replay_buffer.sample().to(device)
                        trained_transitions += sample.numel()
                        with set_recurrent_mode(True):
                            losses = loss_module(sample)
                        loss = (
                            losses["loss_objective"]
                            + losses["loss_critic"]
                            + losses["loss_entropy"]
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            loss_module.parameters(), max_grad_norm
                        )
                        optimizer.step()
                        updates.append(
                            losses.select(
                                "loss_objective",
                                "loss_critic",
                                "loss_entropy",
                                "entropy",
                                "kl_approx",
                                "clip_fraction",
                                "ESS",
                            )
                            .detach()
                            .set("grad_norm", grad_norm)
                        )
                # Average the per-update loss tensordicts over the epoch passes.
                for key, value in torch.stack(updates).mean(dim=0).items():
                    metrics[f"ppo/{key}"] = float(value)
                if scheduler is not None:
                    scheduler.step(metrics["ppo/kl_approx"])
                metrics["ppo/learning_rate"] = optimizer.param_groups[0]["lr"]
            replay_buffer.empty()
            collector.update_policy_weights_()
            collector.reset()

            timings = timeit.todict(prefix="time")
            metrics.update(timings)
            metrics.update(
                {
                    "collection/transitions": float(num_transitions),
                    "collection/trajectories": float(num_trajectories),
                    "progress/transitions": float(collected),
                    "throughput/collection_transitions_per_second": num_transitions
                    / timings["time/collect"],
                    "throughput/training_transitions_per_second": trained_transitions
                    / timings["time/train"],
                }
            )
            if evaluation_interval is not None and (
                iteration % evaluation_interval == 0 or collected >= total_transitions
            ):
                with timeit("evaluate"):
                    metrics.update(evaluate(collected))
                metrics.update(timeit.todict(prefix="time"))
            history.append(metrics)
            log(metrics, step=collected)
            torchrl_logger.info(
                "MicroDuck PPO transitions=%d/%d trajectories=%d reward=%+.4f "
                "return=%+.2f survival=%.2f collect=%.0f/s train=%.0f/s lr=%.2e",
                collected,
                total_transitions,
                num_trajectories,
                metrics["collection/reward_mean"],
                metrics["episode/return_mean"],
                metrics["episode/survival_rate"],
                metrics["throughput/collection_transitions_per_second"],
                metrics["throughput/training_transitions_per_second"],
                metrics["ppo/learning_rate"],
            )
    finally:
        replay_buffer.empty()
        collector.shutdown(close_env=False)
    if best_state is not None:
        actor.load_state_dict(best_state[0])
        critic.load_state_dict(best_state[1])
    return history


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
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
        video_recorder = None
        if cfg.evaluation.video.interval is not None and logger is not None:
            # A 2x2 grid of four tasks filmed in parallel, logged at every
            # `video.interval`-th evaluation.
            video_env = make_video_env(
                cfg.env,
                list(cfg.evaluation.video.tasks),
                width=cfg.evaluation.video.width,
                height=cfg.evaluation.video.height,
            )

            def video_recorder(step: int) -> None:
                logger.log_video(
                    "evaluation/task_grid",
                    record_task_grid(video_env, actor, cfg.evaluation.video.steps),
                    step=step,
                    fps=50,
                )

        train_ppo(
            env,
            actor,
            critic,
            **config["ppo"],
            max_episode_steps=cfg.env.max_episode_steps,
            evaluators=evaluators,
            evaluation_interval=cfg.evaluation.interval,
            video_recorder=video_recorder,
            video_interval=cfg.evaluation.video.interval,
            best_checkpoint_path=cfg.evaluation.best_checkpoint_path,
            latest_checkpoint_path=cfg.evaluation.latest_checkpoint_path,
            policy_kwargs=policy_kwargs,
            config=config,
            logger=logger,
        )
    finally:
        if logger is not None and hasattr(logger.experiment, "finish"):
            logger.experiment.finish()
        for evaluator in evaluators:
            evaluator.shutdown()
        if video_env is not None and not video_env.is_closed:
            video_env.close()
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main()
