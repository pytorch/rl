# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Recurrent PPO on :class:`~torchrl.envs.MicroDuckEnv` with whole-episode replay.

The policy is a GRU backbone shared by the actor and the critic. With
``--policy-head gait-residual`` the actor head adds a bounded residual to the
closed-form gait from ``heuristic_gait.py``, so the first policy is already a
walking controller; with ``--policy-head gaussian`` the actor is a plain
Gaussian head trained from scratch, relying on the contact-based gait terms of
the :class:`~torchrl.envs.MicroDuckEnv` reward, a command range, an optional
forward warm start and a larger exploration scale.

Data flows through the standard TorchRL pieces: a
:class:`~torchrl.collectors.Collector` writes every finished episode as a
whole, unpadded sequence into a
:class:`~torchrl.data.TensorDictReplayBuffer`; GAE is computed once over the
buffer; :class:`~torchrl.data.SliceSampler` draws whole episodes for the
recurrent PPO updates; the buffer is erased before collecting again with the
updated policy. The same task runs on the native MuJoCo, MJX and
``mujoco-torch`` backends.

Run a short CPU job from a TorchRL checkout::

    python examples/microduck/ppo_mujoco.py --microduck-root /path/to/microduck_rl --smoke

Pass ``--backend mjx`` or ``--backend mujoco-torch --compile-step`` to change
only the simulator. Set ``MICRODUCK_RL_ROOT`` instead of ``--microduck-root`` to
point every script at the same checkout.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl import timeit, torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import (
    LazyTensorStorage,
    SliceSampler,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.envs import (
    Compose,
    EnvBase,
    ExplorationType,
    InitTracker,
    MicroDuckEnv,
    set_exploration_type,
    TensorDictPrimer,
    TransformedEnv,
)
from torchrl.envs.custom.mujoco._backends import BackendName
from torchrl.modules import (
    GRUModule,
    ProbabilisticActor,
    set_recurrent_mode,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, KLAdaptiveLR
from torchrl.objectives.value import GAE
from torchrl.record import WandbLogger

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent.parent))

from examples.microduck.heuristic_gait import (  # noqa: E402
    gait_action,
    heading_xy,
    MicroDuckGaitConfig,
)

DEFAULT_COMMANDS = (0.03,)
DEFAULT_EVALUATION_SEEDS = tuple(range(8))
PolicyHead = Literal["gait-residual", "gaussian"]
DEFAULT_TRANSITIONS_PER_UPDATE = 16_384
RECURRENT_STATE_KEY = "recurrent_state"


def _as_gait(
    gait: MicroDuckGaitConfig | Mapping[str, float] | None
) -> MicroDuckGaitConfig:
    if isinstance(gait, Mapping):
        return MicroDuckGaitConfig(**gait)
    return MicroDuckGaitConfig() if gait is None else gait


def _checkpoint_policy_kwargs(checkpoint: Any) -> dict[str, Any]:
    """Return the policy kwargs a training checkpoint recorded, if any."""
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("policy_kwargs"), Mapping
    ):
        return dict(checkpoint["policy_kwargs"])
    return {}


def _checkpoint_env_kwargs(checkpoint: Any) -> dict[str, Any]:
    """Return the env kwargs a training checkpoint recorded, if any."""
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("env_kwargs"), Mapping
    ):
        return dict(checkpoint["env_kwargs"])
    return {}


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


def make_env(
    microduck_root: str | Path | None = None,
    *,
    backend: BackendName = "mujoco",
    commanded_x_velocity: float | Sequence[float] = DEFAULT_COMMANDS,
    command_range: Sequence[float] | None = None,
    warm_start_velocity: Sequence[float] | None = None,
    warm_start_fraction: float = 0.0,
    joint_reset_noise_scale: float | None = None,
    action_scale: float | None = None,
    gait_frequency_per_mps: float | None = None,
    observe_lateral_velocity: bool | None = None,
    reward_scales: Mapping[str, float] | None = None,
    num_envs: int = 8,
    device: torch.device | str = "cpu",
    seed: int = 0,
    parallel: bool = False,
    compile_step: bool = False,
    hidden_size: int | None = None,
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    max_episode_steps: int = 500,
    camera_id: int = -1,
    render_width: int = 640,
    render_height: int = 480,
    reset_noise_scale: float = MicroDuckEnv.RESET_NOISE_SCALE,
    checkpoint: Mapping[str, Any] | None = None,
) -> TransformedEnv:
    """Build the batched task with the transforms the recurrent policy needs.

    :class:`~torchrl.envs.InitTracker` marks episode starts and a
    :class:`~torchrl.envs.TensorDictPrimer` carries the GRU state between steps,
    so the same env serves the collector, evaluation rollouts and ``rlrender``.
    ``rlrender`` passes the loaded training checkpoint as ``checkpoint``; its
    recorded ``hidden_size`` sizes the recurrent state and its recorded env
    options (action scale, gait clock, velocity observation) apply when the
    matching arguments are omitted, so a checkpoint renders without repeating
    how it was trained.
    The native backend batches with :class:`~torchrl.envs.SerialEnv` unless
    ``parallel=True``; MJX and ``mujoco-torch`` batch inside their frameworks.
    ``compile_step`` is forwarded to the ``mujoco-torch`` backend only. Only
    environment arguments are accepted so ``rlrender`` can call this factory
    with its own keyword arguments.
    """
    recorded_env = _checkpoint_env_kwargs(checkpoint)
    if gait is None and recorded_env.get("gait") is not None:
        gait = MicroDuckGaitConfig(**recorded_env["gait"])
    gait = _as_gait(gait)
    if hidden_size is None:
        hidden_size = _checkpoint_policy_kwargs(checkpoint).get("hidden_size", 128)
    if action_scale is None:
        action_scale = recorded_env.get("action_scale", 0.35)
    if gait_frequency_per_mps is None:
        gait_frequency_per_mps = recorded_env.get("gait_frequency_per_mps", 0.0)
    if observe_lateral_velocity is None:
        observe_lateral_velocity = recorded_env.get("observe_lateral_velocity", False)
    kwargs: dict[str, Any] = {
        "backend": backend,
        "commanded_x_velocity": commanded_x_velocity,
        "command_range": None if command_range is None else tuple(command_range),
        "warm_start_velocity": (
            None if warm_start_velocity is None else tuple(warm_start_velocity)
        ),
        "warm_start_fraction": warm_start_fraction,
        "joint_reset_noise_scale": joint_reset_noise_scale,
        "action_scale": action_scale,
        "gait_frequency_per_mps": gait_frequency_per_mps,
        "observe_lateral_velocity": observe_lateral_velocity,
        "reward_scales": None if reward_scales is None else dict(reward_scales),
        "num_envs": num_envs,
        "device": torch.device(device),
        "seed": seed,
        "max_episode_steps": max_episode_steps,
        "camera_id": camera_id,
        "render_width": render_width,
        "render_height": render_height,
        "reset_noise_scale": reset_noise_scale,
        **gait.env_kwargs(),
    }
    if backend == "mujoco":
        kwargs["parallel"] = parallel
    elif backend == "mujoco-torch":
        kwargs["compile_step"] = compile_step
    base_env = MicroDuckEnv(microduck_root, **kwargs)
    return TransformedEnv(
        base_env,
        Compose(
            InitTracker(),
            TensorDictPrimer(
                {RECURRENT_STATE_KEY: Unbounded(shape=(1, hidden_size))},
                expand_specs=True,
            ),
        ),
    )


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


class GaitResidualHead(nn.Module):
    """Gaussian policy head: closed-form gait plus a bounded learned residual.

    The residual is a zero-initialized linear map from the recurrent features,
    squashed by ``tanh`` and scaled by ``residual_scale``, so the initial policy
    reproduces the gait exactly. The exploration scale is state independent.
    """

    def __init__(
        self,
        hidden_size: int,
        gait: MicroDuckGaitConfig,
        *,
        residual_scale: float,
        initial_policy_scale: float,
    ):
        super().__init__()
        self.gait = gait
        self.residual_scale = float(residual_scale)
        self.residual = nn.Linear(hidden_size, MicroDuckEnv.NUM_JOINTS)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.scale = nn.Parameter(torch.zeros(MicroDuckEnv.NUM_JOINTS))
        self.param_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{initial_policy_scale}"
        )

    def forward(
        self, features: torch.Tensor, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nominal = gait_action(self.gait, observation)
        mean = nominal + self.residual_scale * torch.tanh(self.residual(features))
        pre_tanh_mean = torch.atanh(mean.clamp(-0.999, 0.999))
        scale = self.scale.expand_as(pre_tanh_mean)
        return self.param_extractor(torch.cat((pre_tanh_mean, scale), dim=-1))


class GaussianHead(nn.Module):
    """Plain Gaussian policy head for training from scratch.

    The mean starts near zero, which is the ``STAND`` pose, and the
    state-independent exploration scale starts at ``initial_policy_scale``.
    """

    def __init__(self, hidden_size: int, *, initial_policy_scale: float):
        super().__init__()
        self.loc = nn.Linear(hidden_size, MicroDuckEnv.NUM_JOINTS)
        nn.init.orthogonal_(self.loc.weight, gain=0.01)
        nn.init.zeros_(self.loc.bias)
        self.scale = nn.Parameter(torch.zeros(MicroDuckEnv.NUM_JOINTS))
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
    policy_head: PolicyHead = "gait-residual",
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    residual_scale: float = 0.2,
    initial_policy_scale: float = 0.05,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Create the GRU actor and the critic that shares its backbone.

    ``policy_head="gait-residual"`` wraps the closed-form gait with a learned
    residual; ``policy_head="gaussian"`` is a plain Gaussian head trained from
    scratch. Returns the actor and the full value network (backbone plus value
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
        nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh()),
        in_keys=["observation"],
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
                _as_gait(gait),
                residual_scale=residual_scale,
                initial_policy_scale=initial_policy_scale,
            ),
            in_keys=["features", "observation"],
            out_keys=["loc", "scale"],
        )
    elif policy_head == "gaussian":
        actor_head = TensorDictModule(
            GaussianHead(hidden_size, initial_policy_scale=initial_policy_scale),
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
    ).to(device)
    value_head = ValueOperator(
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        ),
        in_keys=["features"],
    )
    critic = TensorDictSequential(backbone, value_head).to(device)
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
    recorded = _checkpoint_policy_kwargs(checkpoint)
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
# Evaluation
# ----------------------------------------------------------------------


@torch.no_grad()
def evaluate_policy(
    env: TransformedEnv,
    policy: ProbabilisticActor,
    *,
    commanded_x_velocities: Sequence[float] = DEFAULT_COMMANDS,
    seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    steps: int = 500,
) -> list[dict[str, float]]:
    """Roll the deterministic policy out for every command and seed.

    Results are kept per command so an aggregate cannot hide a policy that
    walks the wrong way for some commands.

    Returns:
        One dictionary per rollout with the return, mean absolute velocity
        tracking error, survival flag, episode length and displacement along
        the initial heading.
    """
    if env.batch_size.numel() != 1:
        raise ValueError("MicroDuck evaluation expects a single environment.")
    if steps < 1 or not commanded_x_velocities or not seeds:
        raise ValueError("Evaluation commands, seeds and steps must be non-empty.")
    was_training = policy.training
    policy.eval()
    results = []
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for command in commanded_x_velocities:
                for seed in seeds:
                    env.set_seed(seed)
                    reset = env.reset(
                        TensorDict(
                            {
                                "commanded_x_velocity": torch.full(
                                    (*env.batch_size, 1), float(command)
                                )
                            },
                            batch_size=env.batch_size,
                        )
                    )
                    start_qpos = env.base_env.get_state()["qpos"].reshape(-1)
                    heading = heading_xy(start_qpos[3:7])
                    rollout = env.rollout(
                        steps,
                        policy=policy,
                        tensordict=reset,
                        auto_reset=False,
                        break_when_any_done=True,
                    )
                    end_qpos = env.base_env.get_state()["qpos"].reshape(-1)
                    measured = rollout["next", "observation"][..., 6]
                    results.append(
                        {
                            "commanded_x_velocity": float(command),
                            "seed": float(seed),
                            "episode_return": float(rollout["next", "reward"].sum()),
                            "tracking_error": float((measured - command).abs().mean()),
                            "survived": float(
                                not rollout["next", "terminated"][..., -1, :].any()
                            ),
                            "episode_length": float(rollout.shape[-1]),
                            "signed_displacement": float(
                                torch.dot(end_qpos[:2] - start_qpos[:2], heading)
                            ),
                        }
                    )
    finally:
        policy.train(was_training)
    return results


def evaluation_metrics(evaluation: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average the per-rollout evaluation fields, overall and per command."""
    fields = (
        "episode_return",
        "tracking_error",
        "survived",
        "episode_length",
        "signed_displacement",
    )
    metrics = {
        f"evaluation/{field}": sum(row[field] for row in evaluation) / len(evaluation)
        for field in fields
    }
    for command in sorted({row["commanded_x_velocity"] for row in evaluation}):
        rows = [row for row in evaluation if row["commanded_x_velocity"] == command]
        name = f"{command:+.3f}".replace("+", "plus_").replace("-", "minus_")
        for field in fields:
            metrics[f"evaluation/{name}/{field}"] = sum(
                row[field] for row in rows
            ) / len(rows)
    return metrics


def evaluation_score(evaluation: Sequence[dict[str, float]]) -> tuple[float, ...]:
    """Rank checkpoints by survival, then direction, then displacement, then return.

    A short forward fall can earn a higher raw return than a full episode of
    balanced walking, so survival and the number of wrong-way rollouts are
    compared before any displacement or reward figure.
    """
    directional = []
    for row in evaluation:
        command = row["commanded_x_velocity"]
        displacement = row["signed_displacement"]
        directional.append(
            math.copysign(1.0, command) * displacement
            if command
            else -abs(displacement)
        )
    return (
        sum(row["survived"] for row in evaluation),
        min(row["episode_length"] for row in evaluation),
        -float(sum(value <= 0.0 for value in directional)),
        min(directional),
        sum(directional) / len(directional),
        sum(row["episode_return"] for row in evaluation) / len(evaluation),
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
    tracking_error = data["observation"][..., 6] - data["commanded_x_velocity"][..., 0]
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
    transitions_per_update: int = DEFAULT_TRANSITIONS_PER_UPDATE,
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
    evaluation_env: TransformedEnv | None = None,
    evaluation_interval: int | None = None,
    evaluation_commands: Sequence[float] = DEFAULT_COMMANDS,
    evaluation_seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    evaluation_steps: int = 500,
    best_checkpoint_path: str | Path | None = None,
    latest_checkpoint_path: str | Path | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    logger: WandbLogger | None = None,
) -> list[dict[str, float]]:
    """Train the recurrent policy with PPO on whole episodes.

    Each iteration collects at least ``transitions_per_update`` transitions of
    complete episodes into the replay buffer, computes GAE once over the buffer
    in recurrent mode, runs ``epochs`` passes of whole-episode minibatches, then
    empties the buffer and drops the collector's in-flight episodes so the next
    collection only contains data from the updated policy.

    ``best_checkpoint_path`` receives the best-scoring parameters and
    ``latest_checkpoint_path`` the current ones at every evaluation, so
    training progress can be rendered while the best checkpoint protects
    against regressions. ``policy_kwargs`` and ``env_kwargs`` are stored in
    both so ``rlrender`` can rebuild the actor and the env through
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
    if evaluation_interval is not None and evaluation_env is None:
        raise ValueError("evaluation_interval requires an evaluation_env.")
    if (
        best_checkpoint_path is not None or latest_checkpoint_path is not None
    ) and evaluation_interval is None:
        raise ValueError("Checkpoint paths require periodic evaluation.")

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
    advantage = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
        average_gae=False,
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
        normalize_advantage=True,
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
    checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path else None
    latest_path = Path(latest_checkpoint_path) if latest_checkpoint_path else None

    def save_checkpoint(path: Path, step: int, evaluation, score) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "transitions": step,
                "evaluation": evaluation,
                "evaluation_score": list(score),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "policy_kwargs": dict(policy_kwargs or {}),
                "env_kwargs": dict(env_kwargs or {}),
            },
            path,
        )

    def evaluate(step: int) -> dict[str, float]:
        nonlocal best_score, best_state
        evaluation = evaluate_policy(
            evaluation_env,
            actor,
            commanded_x_velocities=evaluation_commands,
            seeds=evaluation_seeds,
            steps=evaluation_steps,
        )
        metrics = evaluation_metrics(evaluation)
        score = evaluation_score(evaluation)
        if latest_path is not None:
            save_checkpoint(latest_path, step, evaluation, score)
        if best_score is None or score > best_score:
            best_score = score
            best_state = (deepcopy(actor.state_dict()), deepcopy(critic.state_dict()))
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, step, evaluation, score)
        metrics["evaluation/is_best"] = float(score == best_score)
        torchrl_logger.info(
            "MicroDuck evaluation transitions=%d survival=%.2f length=%.1f "
            "displacement=%+.4f tracking_error=%.4f",
            step,
            metrics["evaluation/survived"],
            metrics["evaluation/episode_length"],
            metrics["evaluation/signed_displacement"],
            metrics["evaluation/tracking_error"],
        )
        return metrics

    if evaluation_interval is not None:
        initial_metrics = evaluate(0)
        if logger is not None:
            logger.log_metrics(initial_metrics, step=0, override_global_step=True)

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
            sums: dict[str, float] = {}
            update_count = 0
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
                        for key in (
                            "loss_objective",
                            "loss_critic",
                            "loss_entropy",
                            "entropy",
                            "kl_approx",
                            "clip_fraction",
                            "ESS",
                        ):
                            sums[f"ppo/{key}"] = sums.get(f"ppo/{key}", 0.0) + float(
                                losses[key].detach()
                            )
                        sums["ppo/grad_norm"] = sums.get("ppo/grad_norm", 0.0) + float(
                            grad_norm
                        )
                        update_count += 1
                metrics.update(
                    {key: value / update_count for key, value in sums.items()}
                )
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
            if logger is not None:
                logger.log_metrics(metrics, step=collected, override_global_step=True)
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
# Command line
# ----------------------------------------------------------------------


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument(
        "--backend", choices=("mujoco", "mjx", "mujoco-torch"), default="mujoco"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Batch the native backend with ParallelEnv instead of SerialEnv.",
    )
    parser.add_argument(
        "--compile-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the mujoco-torch physics step.",
    )
    parser.add_argument("--total-transitions", type=int, default=10_000_000)
    parser.add_argument(
        "--transitions-per-update", type=int, default=DEFAULT_TRANSITIONS_PER_UPDATE
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch-trajectories", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="KL target of the adaptive learning rate; pass a negative value to disable.",
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--critic-coeff", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--policy-head",
        choices=("gait-residual", "gaussian"),
        default="gait-residual",
        help="gait-residual learns around the closed-form gait; gaussian trains "
        "from scratch.",
    )
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument(
        "--initial-policy-scale",
        type=float,
        help="Initial exploration scale; defaults to 0.05 for gait-residual and "
        "0.3 for gaussian.",
    )
    parser.add_argument(
        "--commanded-x-velocity",
        action="append",
        type=float,
        dest="commanded_x_velocities",
        help="Velocity command sampled at reset; repeat for several commands.",
    )
    parser.add_argument(
        "--command-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Sample the command uniformly from this interval instead.",
    )
    parser.add_argument(
        "--warm-start-velocity",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Forward speed interval for the reset warm start.",
    )
    parser.add_argument("--warm-start-fraction", type=float, default=0.0)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.35,
        help="Position-target offset in radians for a unit normalized action.",
    )
    parser.add_argument("--gait-frequency-hz", type=float, default=1.8913)
    parser.add_argument(
        "--gait-frequency-per-mps",
        type=float,
        default=0.0,
        help="Gait clock frequency increase per m/s of commanded speed.",
    )
    parser.add_argument(
        "--observe-lateral-velocity",
        action="store_true",
        help="Add body-frame lateral and vertical velocity to the observation.",
    )
    parser.add_argument(
        "--reward-scale",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a MicroDuckEnv reward attribute, e.g. TRACKING_WEIGHT=4.",
    )
    parser.add_argument(
        "--joint-reset-noise-scale",
        type=float,
        help="Uniform joint-position noise at reset in radians; defaults to the "
        "env's reset noise.",
    )
    parser.add_argument(
        "--evaluation-command",
        action="append",
        type=float,
        dest="evaluation_commands",
        help="Commands for deterministic evaluation; defaults to the training "
        "commands, or to the range bounds and midpoint.",
    )
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument(
        "--best-checkpoint-path", type=Path, default=Path("microduck_ppo_best.pt")
    )
    parser.add_argument(
        "--latest-checkpoint-path",
        type=Path,
        help="Also save the current parameters at every evaluation.",
    )
    parser.add_argument("--wandb-project", default="torchrl-microduck-ppo")
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity; required when logging so no default workspace is used.",
    )
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Tiny pipeline check.")
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    if args.smoke:
        args.num_envs = 1
        args.max_episode_steps = 50
        args.total_transitions = 200
        args.transitions_per_update = 100
        args.epochs = 1
        args.minibatch_trajectories = 2
        args.evaluation_interval = 1
        args.evaluation_steps = 20
        args.best_checkpoint_path = None
        args.latest_checkpoint_path = None
        args.wandb_mode = "disabled"
    if args.wandb_mode != "disabled" and not args.wandb_entity:
        raise ValueError(
            "W&B logging requires --wandb-entity so runs do not land in an "
            "unintended default workspace."
        )
    torch.manual_seed(args.seed)
    commands = tuple(args.commanded_x_velocities or DEFAULT_COMMANDS)
    if args.evaluation_commands:
        evaluation_commands = tuple(args.evaluation_commands)
    elif args.command_range is not None:
        low, high = args.command_range
        evaluation_commands = (low, (low + high) / 2, high)
    else:
        evaluation_commands = commands
    if args.initial_policy_scale is None:
        args.initial_policy_scale = 0.05 if args.policy_head == "gait-residual" else 0.3
    reward_scales = {}
    for item in args.reward_scale:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--reward-scale expects NAME=VALUE, got {item!r}.")
        reward_scales[name.strip()] = float(value)
    gait = replace(MicroDuckGaitConfig(), frequency_hz=args.gait_frequency_hz)
    env_kwargs = {
        "backend": args.backend,
        "commanded_x_velocity": commands,
        "command_range": args.command_range,
        "device": args.device,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "max_episode_steps": args.max_episode_steps,
        "compile_step": args.compile_step,
        "action_scale": args.action_scale,
        "gait": gait,
        "gait_frequency_per_mps": args.gait_frequency_per_mps,
        "observe_lateral_velocity": args.observe_lateral_velocity,
        "reward_scales": reward_scales or None,
    }
    recorded_env_kwargs = {
        "action_scale": args.action_scale,
        "gait": asdict(gait),
        "gait_frequency_per_mps": args.gait_frequency_per_mps,
        "observe_lateral_velocity": args.observe_lateral_velocity,
        "reward_scales": reward_scales,
        "command_range": args.command_range,
    }
    policy_kwargs = {
        "hidden_size": args.hidden_size,
        "policy_head": args.policy_head,
        "residual_scale": args.residual_scale,
        "initial_policy_scale": args.initial_policy_scale,
    }
    env = make_env(
        args.microduck_root,
        num_envs=args.num_envs,
        parallel=args.parallel,
        warm_start_velocity=args.warm_start_velocity,
        warm_start_fraction=args.warm_start_fraction,
        joint_reset_noise_scale=args.joint_reset_noise_scale,
        **env_kwargs,
    )
    evaluation_env = None
    logger = None
    try:
        actor, critic = make_models(env, device=args.device, **policy_kwargs)
        if args.evaluation_interval is not None:
            evaluation_env = make_env(args.microduck_root, num_envs=1, **env_kwargs)
        if args.wandb_mode != "disabled":
            logger = WandbLogger(
                exp_name=args.wandb_name
                or f"microduck-{args.backend}-seed-{args.seed}",
                project=args.wandb_project,
                entity=args.wandb_entity,
                offline=args.wandb_mode == "offline",
            )
            logger.log_hparams(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                }
            )
        train_ppo(
            env,
            actor,
            critic,
            total_transitions=args.total_transitions,
            transitions_per_update=args.transitions_per_update,
            max_episode_steps=args.max_episode_steps,
            epochs=args.epochs,
            minibatch_trajectories=args.minibatch_trajectories,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl if args.target_kl > 0 else None,
            entropy_coeff=args.entropy_coeff,
            critic_coeff=args.critic_coeff,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            max_grad_norm=args.max_grad_norm,
            evaluation_env=evaluation_env,
            evaluation_interval=args.evaluation_interval,
            evaluation_commands=evaluation_commands,
            evaluation_steps=args.evaluation_steps,
            best_checkpoint_path=args.best_checkpoint_path,
            latest_checkpoint_path=args.latest_checkpoint_path,
            policy_kwargs=policy_kwargs,
            env_kwargs=recorded_env_kwargs,
            logger=logger,
        )
    finally:
        if logger is not None:
            logger.experiment.finish()
        if evaluation_env is not None:
            evaluation_env.close()
        if not env.is_closed:
            env.close()


if __name__ == "__main__":
    main(parse_args())
