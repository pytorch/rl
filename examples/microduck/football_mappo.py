# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-agent PPO for MicroDuck football, five against five by default.

Both teams share one policy (self-play) and every quantity a duck observes is
expressed in its own frame and in its team's frame, so the same parameters
play both sides. The critic is centralized: it reads every duck's observation
and predicts one value per duck. Data flows through
:class:`~torchrl.envs.MicroDuckFootballEnv`, a
:class:`~torchrl.collectors.Collector`, a
:class:`~torchrl.objectives.ClipPPOLoss` with GAE and a
:class:`~torchrl.data.ReplayBuffer` for the minibatches.

The ducks come with a walker. ``policy.walker_checkpoint`` names a MicroDuck
locomotion policy trained with ``ppo_mujoco.py`` (a local path or a URL,
verified against ``policy.walker_sha256``);
:func:`~torchrl.envs.microduck_skill_env` builds an env in which the football policy picks
one of the walker's tasks per duck (stand, walk forward or backward, sidestep
left or right) every ``policy.decision_period`` control steps, and the frozen
walker drives the joints in between. ``policy.walker_checkpoint=null`` trains
joint-level actions end to end instead, at 50 Hz.

Evaluation runs deterministic matches with a
:class:`~torchrl.collectors.Evaluator` and, on request, films one from the
broadcast camera into the logger. Checkpoints are unified TorchRL checkpoints
written with :func:`~torchrl.render.save_render_checkpoint`; ``rlrender``
rebuilds the match with :func:`make_env` and :func:`make_render_policy`.

The script is configured with Hydra from ``football.yaml``. Run a short CPU
job from a TorchRL checkout::

    python examples/microduck/football_mappo.py env.download=true smoke=true

and a 5-a-side training run with::

    python examples/microduck/football_mappo.py env.download=true \\
        logger.entity=YOUR_ENTITY evaluation.video.interval=2

``env.download=true`` fetches the pinned ``microduck_rl`` assets into
``~/.cache/torchrl/microduck``; set ``env.microduck_root`` or
``MICRODUCK_RL_ROOT`` to use an existing checkout instead.
"""

from __future__ import annotations

import hashlib
import math
import sys
import urllib.request
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl import timeit, torchrl_logger
from torchrl.collectors import Collector, Evaluator
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.data.tensor_specs import Categorical as CategoricalSpec
from torchrl.envs import (
    EnvBase,
    microduck_skill_env,
    MicroDuckEnv,
    MicroDuckFootballEnv,
    MicroDuckTask,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, KLAdaptiveLR, ValueEstimators
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.render import load_checkpoint, save_render_checkpoint

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent))

from examples.microduck.ppo_mujoco import make_actor_critic, make_tasks  # noqa: E402

# The asset location is machine specific and is never taken from a checkpoint.
ASSET_KEYS = ("microduck_root", "root", "download")
CONTROL_PERIOD_S = MicroDuckFootballEnv.FRAME_SKIP * 0.002
OBSERVATION_KEY = ("agents", "observation")
REWARD_KEY = ("agents", "reward")
VALUE_KEY = ("agents", "state_value")


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


def fetch_checkpoint(
    source: str | Path,
    *,
    root: str | Path | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a local path to ``source``, downloading a URL into the cache once.

    ``root`` is the MicroDuck cache directory (``~/.cache/torchrl/microduck``
    by default); URLs land in its ``checkpoints`` folder under their file
    name. The SHA-256 digest of the file is checked against ``sha256`` when
    given, for downloads and local files alike.
    """
    source = str(source)
    if source.startswith(("http://", "https://")):
        cache_root = (
            Path("~/.cache/torchrl/microduck").expanduser()
            if root is None
            else Path(root).expanduser()
        )
        path = cache_root / "checkpoints" / Path(source.split("?")[0]).name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torchrl_logger.info(
                "Downloading the walker checkpoint %s to %s", source, path
            )
            partial = path.with_name(path.name + ".partial")
            urllib.request.urlretrieve(source, partial)
            partial.replace(path)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256 is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256.lower():
            raise ValueError(
                f"The checkpoint {path} has SHA-256 {digest}, expected {sha256}."
            )
    return path


def load_walker(
    source: str | Path,
    *,
    device: torch.device | str = "cpu",
    root: str | Path | None = None,
    sha256: str | None = None,
    action_scale: float | None = None,
) -> tuple[ProbabilisticActor, list[MicroDuckTask]]:
    """Load a ``ppo_mujoco.py`` checkpoint as a frozen controller.

    Returns the actor, built on ``device`` with the ``policy_kwargs`` the
    checkpoint recorded and set to evaluation mode without gradients, and the
    task library it was trained with (``task_id`` order). ``action_scale``
    must match the one the walker was trained with, since the football env
    applies the walker's actions with its own scale.
    """
    payload = load_checkpoint(fetch_checkpoint(source, root=root, sha256=sha256))
    config = payload.get("config") or {}
    tasks = make_tasks((config.get("env") or {})["tasks"])
    trained_scale = float((config.get("env") or {}).get("action_scale", 0.35))
    if action_scale is not None and not math.isclose(trained_scale, action_scale):
        raise ValueError(
            f"The walker was trained with action_scale={trained_scale}; set "
            f"env.action_scale to that value (got {action_scale})."
        )
    walker, _ = make_actor_critic(
        MicroDuckEnv.OBSERVATION_DIM,
        len(tasks),
        device=device,
        **dict(payload["policy_kwargs"]),
    )
    walker.load_state_dict(payload["model_state_dict"])
    walker.requires_grad_(False).eval()
    return walker, tasks


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
    render_width: int | None = None,
    render_height: int | None = None,
) -> TransformedEnv:
    """Build the match env from the ``env`` and ``policy`` sections of ``football.yaml``.

    ``cfg`` is the whole configuration, as the Hydra ``DictConfig`` or a plain
    mapping; missing entries take the defaults of ``football.yaml``.
    ``rlrender`` passes the training checkpoint, whose recorded config (minus
    the asset location, which is machine specific) sits between those defaults
    and ``cfg``. The keyword arguments override single entries so a checkpoint
    renders with one match from a local asset path.

    With ``policy.walker_checkpoint`` set, the joint-level
    :class:`~torchrl.envs.MicroDuckFootballEnv` is wrapped by
    :func:`~torchrl.envs.microduck_skill_env`, driven by the walker, and the
    env's actions are skill indices. ``from_pixels`` adds a rendered
    ``pixels`` observation for the video.
    """
    recorded = checkpoint if isinstance(checkpoint, Mapping) else {}
    recorded_config = dict(recorded.get("config") or {})
    recorded_config["env"] = {
        key: value
        for key, value in (recorded_config.get("env") or {}).items()
        if key not in ASSET_KEYS
    }
    recorded_config = {
        key: recorded_config[key] for key in ("env", "policy") if key in recorded_config
    }
    overrides = {
        "microduck_root": None if microduck_root is None else str(microduck_root),
        "root": None if root is None else str(root),
        "download": download,
        "num_envs": num_envs,
        "parallel": parallel,
        "device": None if device is None else str(device),
        "render_width": render_width,
        "render_height": render_height,
    }
    if cfg is not None and not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))
    merged = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(PACKAGE_DIR / "football.yaml"),
            recorded_config,
            cfg or {},
            {
                "env": {
                    key: value for key, value in overrides.items() if value is not None
                }
            },
        ),
        resolve=True,
    )
    env_cfg = merged["env"]
    policy_cfg = merged["policy"]
    kwargs: dict[str, Any] = {
        "root": env_cfg["root"],
        "download": env_cfg["download"],
        "players_per_team": env_cfg["players_per_team"],
        "pitch": dict(env_cfg["pitch"] or {}),
        "backend": env_cfg["backend"],
        "num_envs": env_cfg["num_envs"],
        "device": torch.device(env_cfg["device"]),
        "seed": env_cfg["seed"],
        "max_episode_steps": env_cfg["max_episode_steps"],
        "action_scale": env_cfg["action_scale"],
        "reward_weights": dict(env_cfg["reward_weights"] or {}),
        "respawn": env_cfg["respawn"],
        "respawn_mode": env_cfg["respawn_mode"],
        "respawn_delay_s": env_cfg["respawn_delay_s"],
        "camera_id": env_cfg["camera_id"],
        "render_width": env_cfg["render_width"],
        "render_height": env_cfg["render_height"],
        "from_pixels": from_pixels,
    }
    if env_cfg["backend"] == "mujoco":
        kwargs["parallel"] = env_cfg["parallel"]
    env: EnvBase = MicroDuckFootballEnv(
        microduck_root=env_cfg["microduck_root"], **kwargs
    )
    if policy_cfg["walker_checkpoint"] is not None:
        walker, tasks = load_walker(
            policy_cfg["walker_checkpoint"],
            device=env.device,
            root=env_cfg["root"],
            sha256=policy_cfg["walker_sha256"],
            action_scale=env_cfg["action_scale"],
        )
        env = microduck_skill_env(
            env,
            walker,
            tasks,
            skills=policy_cfg["skills"],
            steps=policy_cfg["decision_period"],
            control_period_s=CONTROL_PERIOD_S,
        )
    return TransformedEnv(env)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


def make_models(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 256,
    depth: int = 2,
    initial_policy_scale: float = 1.0,
    centralized_critic: bool = False,
) -> tuple[ProbabilisticActor, TensorDictModule]:
    """Create the shared-parameter actor and the critic.

    Both are :class:`~torchrl.modules.MultiAgentMLP` networks built on
    ``device`` and shared by every duck. The actor is decentralized (each duck
    acts on its own observation): a categorical head over the skills when the
    env's action is a skill index, a tanh-squashed Gaussian over the 14 joint
    offsets otherwise, with a state-independent exploration scale starting at
    ``initial_policy_scale``. The critic returns one value per duck from that
    duck's observation, which already describes the whole match in the duck's
    team frame; with ``centralized_critic`` it reads every duck's observation
    instead and returns the same value for all of them, which cannot tell the
    two teams of a zero-sum match apart.
    """
    if not math.isfinite(initial_policy_scale) or initial_policy_scale <= 0:
        raise ValueError("initial_policy_scale must be finite and positive.")
    device = torch.device(device)
    observation = env.observation_spec[OBSERVATION_KEY]
    num_agents, observation_dim = observation.shape[-2:]
    action_spec = env.full_action_spec_unbatched[env.action_key]
    network_kwargs = {
        "n_agents": num_agents,
        "share_params": True,
        "device": device,
        "depth": depth,
        "num_cells": hidden_size,
        "activation_class": nn.Tanh,
    }
    if isinstance(action_spec, CategoricalSpec):
        head = TensorDictModule(
            MultiAgentMLP(
                observation_dim,
                action_spec.space.n,
                centralized=False,
                **network_kwargs,
            ),
            in_keys=[OBSERVATION_KEY],
            out_keys=[("agents", "logits")],
        )
        actor = ProbabilisticActor(
            module=head,
            spec=action_spec,
            in_keys={"logits": ("agents", "logits")},
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
        )
    else:
        head = TensorDictModule(
            nn.Sequential(
                MultiAgentMLP(
                    observation_dim,
                    2 * action_spec.shape[-1],
                    centralized=False,
                    **network_kwargs,
                ),
                NormalParamExtractor(
                    scale_mapping=f"biased_softplus_{initial_policy_scale}"
                ),
            ),
            in_keys=[OBSERVATION_KEY],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        actor = ProbabilisticActor(
            module=head,
            spec=action_spec,
            in_keys={"loc": ("agents", "loc"), "scale": ("agents", "scale")},
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
        )
    critic = TensorDictModule(
        MultiAgentMLP(
            observation_dim, 1, centralized=centralized_critic, **network_kwargs
        ),
        in_keys=[OBSERVATION_KEY],
        out_keys=[VALUE_KEY],
    )
    return actor, critic


def make_render_policy(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    checkpoint: Mapping[str, Any] | None = None,
    hidden_size: int | None = None,
    depth: int | None = None,
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
        "depth": depth,
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
    critic: TensorDictModule,
    *,
    frames: int,
    policy_kwargs: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Write a unified TorchRL checkpoint that ``rlrender`` and ``init_from`` read."""
    return save_render_checkpoint(
        path,
        actor,
        env_metadata={"policy_kwargs": dict(policy_kwargs)},
        frames=frames,
        metrics=dict(metrics),
        config=dict(config),
        extra={"critic_state_dict": critic.state_dict()},
        format="archive",
    )


def load_parameters(
    path: str | Path, actor: ProbabilisticActor, critic: TensorDictModule
) -> int:
    """Load actor and critic parameters from a checkpoint written by :func:`save_checkpoint`.

    Returns:
        The number of env steps the checkpoint was trained on.
    """
    payload = load_checkpoint(path)
    try:
        actor.load_state_dict(payload["model_state_dict"])
        critic.load_state_dict(payload["critic_state_dict"])
    except RuntimeError as err:
        raise RuntimeError(
            f"The checkpoint {path} was trained with policy kwargs "
            f"{payload.get('policy_kwargs')} and config "
            f"{payload.get('config')}; the current models must match them."
        ) from err
    return int(payload.get("frames", 0))


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def football_metrics(trajectories: TensorDictBase) -> dict[str, float]:
    """Match statistics of the padded trajectory batch an :class:`Evaluator` collects.

    Goals are counted per match for each team (both teams play the same
    policy, so a lasting difference between the two is a sign of an asymmetry
    in the env), together with the fraction of matches decided by a goal, the
    match length, the falls per duck and per match, and how far the ball
    traveled along blue's attacking direction.
    """
    mask = trajectories["collector", "mask"]
    lengths = mask.sum(-1)
    last = (lengths - 1).unsqueeze(-1)
    goal = trajectories["next", "goal"].squeeze(-1)
    goal = torch.where(mask, goal, torch.zeros_like(goal))
    fallen = trajectories["next", "agents", "fallen"].squeeze(-1) & mask.unsqueeze(-1)
    ball_x = trajectories["next", "ball_position"][..., 0]
    start_x = trajectories["ball_position"][..., 0, 0]
    end_x = ball_x.gather(-1, last).squeeze(-1)
    return {
        "goals_blue": float((goal == 1).sum(-1).float().mean()),
        "goals_red": float((goal == -1).sum(-1).float().mean()),
        "decided_rate": float((goal != 0).any(-1).float().mean()),
        "match_length": float(lengths.float().mean()),
        "falls_per_duck": float(
            fallen.sum(dim=(1, 2)).float().mean() / fallen.shape[-1]
        ),
        "ball_progress_blue": float((end_x - start_x).mean()),
    }


def _collection_metrics(data: TensorDictBase) -> dict[str, float]:
    reward = data["next", REWARD_KEY].squeeze(-1)
    done = data["next", "done"].squeeze(-1)
    goal = data["next", "goal"].squeeze(-1)
    fallen = data["next", "agents", "fallen"].squeeze(-1)
    players = reward.shape[-1] // 2
    traj_ids = data["collector", "traj_ids"]
    unique_ids, inverse = torch.unique(traj_ids, return_inverse=True)
    lengths = torch.zeros(unique_ids.numel()).index_add_(
        0, inverse.reshape(-1), torch.ones(inverse.numel())
    )
    returns = torch.zeros(unique_ids.numel(), reward.shape[-1]).index_add_(
        0, inverse.reshape(-1), reward.reshape(-1, reward.shape[-1])
    )
    finished = float(done.sum())
    return {
        "collection/reward_mean": float(reward.mean()),
        "collection/reward_blue": float(reward[..., :players].mean()),
        "collection/reward_red": float(reward[..., players:].mean()),
        "collection/falls_per_duck_step": float(fallen.float().mean()),
        "collection/goals_blue": float((goal == 1).sum()),
        "collection/goals_red": float((goal == -1).sum()),
        "episode/finished": finished,
        "episode/decided_rate": float((goal != 0).sum()) / max(finished, 1.0),
        "episode/length_mean": float(lengths.mean()),
        "episode/return_mean": float(returns.mean()),
    }


def evaluation_score(metrics: Mapping[str, float]) -> tuple[float, ...]:
    """Rank checkpoints by goals per match, then ball progress, then fewer falls.

    Both teams play the same policy, so more goals means the policy scores
    faster against itself, which is the quantity self-play improves.
    """
    return (
        metrics["evaluation/goals_blue"] + metrics["evaluation/goals_red"],
        metrics["evaluation/ball_progress_blue"],
        -metrics["evaluation/falls_per_duck"],
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_mappo(
    env: EnvBase,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    *,
    total_frames: int = 5_000_000,
    frames_per_batch: int = 4800,
    epochs: int = 4,
    minibatch_size: int = 1200,
    learning_rate: float = 3e-4,
    target_kl: float | None = 0.02,
    max_learning_rate: float = 1e-2,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 1.0,
    evaluator: Evaluator | None = None,
    evaluation_interval: int | None = None,
    video_recorder: Callable[[int], None] | None = None,
    video_interval: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    latest_checkpoint_path: str | Path | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    logger: Logger | None = None,
) -> list[dict[str, float]]:
    """Train the shared policy with multi-agent PPO and a centralized critic.

    Each iteration collects ``frames_per_batch`` env steps over the batch of
    matches, computes GAE per duck with the team-shared done flags, then runs
    ``epochs`` passes of ``minibatch_size``-step minibatches (every duck of a
    step travels together). :class:`~torchrl.objectives.KLAdaptiveLR` keeps
    the mean policy KL near ``target_kl``.

    Every ``evaluation_interval`` iterations the ``evaluator`` plays
    deterministic matches (see :func:`football_metrics`); every
    ``video_interval`` evaluations ``video_recorder`` films one.
    ``best_checkpoint_path`` receives the parameters that rank best under
    :func:`evaluation_score` and ``latest_checkpoint_path`` the current ones.

    Returns:
        One metrics dictionary per iteration. When evaluation is enabled the
        actor and critic end up holding the best-scoring parameters.
    """
    if min(total_frames, frames_per_batch, epochs, minibatch_size) < 1:
        raise ValueError("PPO frame, epoch and minibatch sizes must be positive.")
    num_envs = env.batch_size.numel()
    if frames_per_batch % num_envs:
        raise ValueError(
            f"frames_per_batch ({frames_per_batch}) must be a multiple of the "
            f"{num_envs} matches simulated in parallel."
        )
    if minibatch_size > frames_per_batch:
        raise ValueError("minibatch_size cannot exceed frames_per_batch.")
    if evaluation_interval is not None and (
        evaluation_interval < 1 or evaluator is None
    ):
        raise ValueError(
            "evaluation_interval requires a positive value and an evaluator."
        )
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
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        storing_device="cpu",
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )
    loss_module.set_keys(
        reward=REWARD_KEY,
        action=env.action_key,
        value=VALUE_KEY,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
    advantage = loss_module.value_estimator
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = (
        KLAdaptiveLR(optimizer, target_kl=target_kl, max_lr=max_learning_rate)
        if target_kl is not None
        else None
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    history: list[dict[str, float]] = []
    collected = 0
    iteration = 0
    evaluations = 0
    best_score: tuple[float, ...] | None = None
    best_state: tuple[dict, dict] | None = None

    def checkpoint(path, step: int, metrics, score) -> None:
        if path is None:
            return
        save_checkpoint(
            path,
            actor,
            critic,
            frames=step,
            policy_kwargs=policy_kwargs,
            metrics={**metrics, "evaluation_score": list(score)},
            config=config,
        )

    def evaluate(step: int) -> dict[str, float]:
        nonlocal best_score, best_state, evaluations
        result = evaluator.evaluate(weights=actor, step=step)
        metrics = {
            key.replace("/custom/", "/"): float(value)
            for key, value in result.items()
            if isinstance(value, (int, float))
        }
        if video_recorder is not None and evaluations % video_interval == 0:
            with timeit("video"):
                video_recorder(step)
        evaluations += 1
        score = evaluation_score(metrics)
        checkpoint(latest_checkpoint_path, step, metrics, score)
        if best_score is None or score > best_score:
            best_score = score
            best_state = (deepcopy(actor.state_dict()), deepcopy(critic.state_dict()))
            checkpoint(best_checkpoint_path, step, metrics, score)
        metrics["evaluation/is_best"] = float(score == best_score)
        torchrl_logger.info(
            "Football evaluation frames=%d goals blue=%.2f red=%.2f decided=%.2f "
            "length=%.1f falls/duck=%.2f",
            step,
            metrics["evaluation/goals_blue"],
            metrics["evaluation/goals_red"],
            metrics["evaluation/decided_rate"],
            metrics["evaluation/match_length"],
            metrics["evaluation/falls_per_duck"],
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
        while collected < total_frames:
            iteration += 1
            timeit.reset()
            with timeit("collect"):
                data = next(collector_iterator)
            num_frames = data.numel()
            collected += num_frames
            metrics = _collection_metrics(data)
            # The value estimator expects the done flags with the reward's shape.
            for key in ("done", "terminated"):
                data.set(
                    ("next", "agents", key),
                    data.get(("next", key))
                    .unsqueeze(-1)
                    .expand(data.get_item_shape(("next", REWARD_KEY))),
                )
            with timeit("advantage"), torch.no_grad():
                processed = advantage(data.to(device))
            value_target = processed["value_target"]
            metrics["value/explained_variance"] = float(
                1.0
                - (value_target - processed[VALUE_KEY]).var()
                / value_target.var().clamp_min(torch.finfo(value_target.dtype).eps)
            )
            replay_buffer.extend(processed.reshape(-1).cpu())
            updates = []
            with timeit("train"):
                for _ in range(epochs):
                    for _ in range(num_frames // minibatch_size):
                        sample = replay_buffer.sample().to(device)
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
                # Entries such as the entropy carry the agent dimension.
                for key, value in torch.stack(updates).mean(dim=0).items():
                    metrics[f"ppo/{key}"] = float(value.mean())
                if scheduler is not None:
                    scheduler.step(metrics["ppo/kl_approx"])
                metrics["ppo/learning_rate"] = optimizer.param_groups[0]["lr"]
            replay_buffer.empty()
            collector.update_policy_weights_()

            timings = timeit.todict(prefix="time")
            metrics.update(timings)
            metrics.update(
                {
                    "collection/frames": float(num_frames),
                    "progress/frames": float(collected),
                    "throughput/collection_frames_per_second": num_frames
                    / timings["time/collect"],
                    "throughput/collection_duck_steps_per_second": num_frames
                    * data.get_item_shape(("next", REWARD_KEY))[-2]
                    / timings["time/collect"],
                }
            )
            if evaluation_interval is not None and (
                iteration % evaluation_interval == 0 or collected >= total_frames
            ):
                with timeit("evaluate"):
                    metrics.update(evaluate(collected))
                metrics.update(timeit.todict(prefix="time"))
            history.append(metrics)
            log(metrics, step=collected)
            torchrl_logger.info(
                "Football MAPPO frames=%d/%d matches=%d goals blue=%d red=%d "
                "reward=%+.4f collect=%.0f frames/s lr=%.2e",
                collected,
                total_frames,
                int(metrics["episode/finished"]),
                int(metrics["collection/goals_blue"]),
                int(metrics["collection/goals_red"]),
                metrics["collection/reward_mean"],
                metrics["throughput/collection_frames_per_second"],
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
# Video
# ----------------------------------------------------------------------


def record_match(
    env: TransformedEnv,
    recorder: VideoRecorder,
    actor: ProbabilisticActor,
    *,
    steps: int,
    step: int,
) -> None:
    """Play one deterministic match on the filmed env and log the clip at ``step``."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        env.rollout(steps, actor, break_when_any_done=True)
    recorder.dump(step=step)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


@hydra.main(config_path="", config_name="football", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.smoke:
        # One match in one process, a few decisions, and a short clip written
        # by a CSV logger: a pipeline check, not a speed test.
        cfg.env.backend = "mujoco"
        cfg.env.parallel = False
        cfg.env.device = "cpu"
        cfg.env.num_envs = 1
        cfg.env.players_per_team = 1
        cfg.env.max_episode_steps = 30
        cfg.policy.hidden_size = 32
        cfg.ppo.total_frames = 40
        cfg.ppo.frames_per_batch = 20
        cfg.ppo.minibatch_size = 10
        cfg.ppo.epochs = 1
        cfg.evaluation.interval = 1
        cfg.evaluation.num_matches = 1
        cfg.evaluation.steps = 6
        cfg.evaluation.video.interval = 1
        cfg.evaluation.video.steps = 6
        cfg.evaluation.video.width = 160
        cfg.evaluation.video.height = 90
        cfg.evaluation.best_checkpoint_path = None
        cfg.evaluation.latest_checkpoint_path = None
        cfg.logger.backend = "csv"
    if cfg.logger.backend == "wandb" and not cfg.logger.entity:
        raise ValueError(
            "W&B logging requires logger.entity so runs do not land in an "
            "unintended default workspace."
        )
    torch.manual_seed(cfg.env.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    policy_kwargs = {
        "hidden_size": cfg.policy.hidden_size,
        "depth": cfg.policy.depth,
        "initial_policy_scale": cfg.policy.initial_policy_scale,
        "centralized_critic": cfg.policy.centralized_critic,
    }
    env = make_env(cfg)
    skills = cfg.policy.walker_checkpoint is not None
    evaluator = None
    video_env = None
    logger = None
    try:
        actor, critic = make_models(env, device=env.device, **policy_kwargs)
        if cfg.policy.init_from:
            trained = load_parameters(cfg.policy.init_from, actor, critic)
            torchrl_logger.info(
                "Initialized actor and critic from %s (%d frames).",
                cfg.policy.init_from,
                trained,
            )
        if cfg.evaluation.interval is not None:
            evaluator = Evaluator(
                make_env(cfg, num_envs=1, parallel=False),
                actor,
                num_trajectories=cfg.evaluation.num_matches,
                max_steps=cfg.evaluation.steps,
                metrics_fn=football_metrics,
                reward_keys=("next", *REWARD_KEY),
                log_prefix="evaluation",
            )
        mode = "skills" if skills else "joints"
        logger = get_logger(
            cfg.logger.backend,
            logger_name="microduck_football",
            experiment_name=cfg.logger.exp_name
            or generate_exp_name("football", f"{mode}-{cfg.env.backend}"),
            wandb_kwargs={
                "project": cfg.logger.project,
                "entity": cfg.logger.entity,
                "mode": cfg.logger.mode,
                "config": config,
            },
        )
        video_callback = None
        if cfg.evaluation.video.interval is not None and logger is not None:
            # One match filmed by the broadcast camera; with skills the
            # recorder sees one frame per decision.
            fps = 1.0 / CONTROL_PERIOD_S
            if skills:
                fps /= cfg.policy.decision_period
            recorder = VideoRecorder(
                logger, tag="evaluation/match", skip=1, fps=int(round(fps))
            )
            video_env = make_env(
                cfg,
                num_envs=1,
                parallel=False,
                from_pixels=True,
                render_width=cfg.evaluation.video.width,
                render_height=cfg.evaluation.video.height,
            )
            video_env.append_transform(recorder)

            def _video_callback(step: int) -> None:
                record_match(
                    video_env,
                    recorder,
                    actor,
                    steps=cfg.evaluation.video.steps,
                    step=step,
                )

            video_callback = _video_callback

        train_mappo(
            env,
            actor,
            critic,
            **config["ppo"],
            evaluator=evaluator,
            evaluation_interval=cfg.evaluation.interval,
            video_recorder=video_callback,
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
        if evaluator is not None:
            evaluator.shutdown()
        if video_env is not None and not video_env.is_closed:
            video_env.close()
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main()
