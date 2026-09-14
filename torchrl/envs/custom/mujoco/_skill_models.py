# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""MicroDuck skill reconstruction shared by training examples and deployed games."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from tensordict import TensorDictBase
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import nn

from torchrl.data.tensor_specs import Bounded
from torchrl.envs.custom.mujoco.microduck import MicroDuckEnv, MicroDuckTask
from torchrl.modules import GRUModule, ProbabilisticActor, TanhNormal, ValueOperator

PolicyHead = Literal["gait-residual", "gaussian"]
RECURRENT_STATE_KEY = "recurrent_state"
TASK_PRESETS = (
    "make_task",
    "tracking_task",
    "standing_task",
    "speed_range_task",
    "sidestep_task",
    "turning_task",
    "jump_task",
)


@dataclass(frozen=True)
class MicroDuckGaitConfig:
    """Parameters of the closed-form MicroDuck gait.

    Amplitudes are normalized actions: an amplitude of one is a full
    ``action_scale`` offset around the MJCF ``STAND`` actuator target. The
    oscillator drives mirrored leg joints half a cycle apart, and pitch
    feedback acts through the hip and ankle targets. The clock parameters are
    forwarded to :class:`~torchrl.envs.MicroDuckEnv` through :meth:`env_kwargs`
    so the gait phase read from the observation matches this configuration.
    """

    frequency_hz: float = 1.8913
    hip_amplitude: float = 0.999
    knee_amplitude: float = 0.9097
    ankle_amplitude: float = 0.0317
    lateral_amplitude: float = 0.9584
    lateral_phase_offset: float = -0.1624
    phase_offset: float = -1.5237
    pitch_kp: float = -9.9495
    pitch_kd: float = -0.6119
    ankle_pitch_kp: float = 10.9934
    ankle_pitch_kd: float = 0.6033
    ramp_duration_s: float = 0.4

    def task_kwargs(self) -> dict[str, float]:
        """Return the :class:`~torchrl.envs.MicroDuckTask` gait-clock overrides.

        The phase offset and the ramp duration are class constants of
        :class:`~torchrl.envs.MicroDuckEnv` and must match the config.
        """
        if (self.phase_offset, self.ramp_duration_s) != (
            MicroDuckEnv.GAIT_PHASE_OFFSET,
            MicroDuckEnv.GAIT_RAMP_DURATION_S,
        ):
            raise ValueError(
                "The gait clock offset and ramp are fixed by MicroDuckEnv: "
                f"{MicroDuckEnv.GAIT_PHASE_OFFSET}, {MicroDuckEnv.GAIT_RAMP_DURATION_S}."
            )
        return {"gait_frequency_hz": self.frequency_hz}


class MicroDuckGaitActor(TensorDictModuleBase):
    """Closed-form MicroDuck walking gait as a TensorDict policy.

    A bilateral phase oscillator drives the hip, knee, ankle and lateral
    targets while proportional-derivative feedback on the torso pitch acts
    through the hip and ankle targets. Everything is read from the
    :class:`~torchrl.envs.MicroDuckEnv` observation: projected gravity, body
    angular velocity, the velocity command and the gait clock. The command sign
    sets the walking direction; a zero command keeps only the balance feedback.

    Args:
        config: gait parameters, as a :class:`MicroDuckGaitConfig` or a mapping
            of its fields. Defaults to the tuned gait.
        in_keys: the observation key. Defaults to ``["observation"]``.
        out_keys: the action key. Defaults to ``["action"]``.

    Examples:
        >>> from torchrl.envs import MicroDuckEnv
        >>> config = MicroDuckGaitConfig()
        >>> env = MicroDuckEnv(download=True, tasks=MicroDuckEnv.tracking_task(0.03, **config.task_kwargs()))
        >>> rollout = env.rollout(100, MicroDuckGaitActor(config))
        >>> rollout["action"].shape
        torch.Size([1, 100, 14])
    """

    def __init__(
        self,
        config: MicroDuckGaitConfig | Mapping[str, float] | None = None,
        *,
        in_keys: Sequence[str] = ("observation",),
        out_keys: Sequence[str] = ("action",),
    ):
        super().__init__()
        if isinstance(config, Mapping):
            config = MicroDuckGaitConfig(**config)
        self.config = MicroDuckGaitConfig() if config is None else config
        self.in_keys = list(in_keys)
        self.out_keys = list(out_keys)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set(
            self.out_keys[0], self.gait_action(tensordict.get(self.in_keys[0]))
        )
        return tensordict

    def gait_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the normalized gait action from a MicroDuck observation.

        Args:
            observation: tensor of shape ``(*, MicroDuckEnv.observation_dim)``.

        Returns:
            A tensor of normalized actions in ``[-1, 1]`` with shape ``(*, 14)``.
        """
        config = self.config
        phase_start = MicroDuckEnv.GAIT_PHASE_START
        pitch = observation[..., 0:1].clamp(-1.0, 1.0).asin()
        pitch_rate = observation[..., 4:5]
        command_start = MicroDuckEnv.COMMAND_START
        direction = observation[..., command_start : command_start + 1].sign()
        gait_sin = direction * observation[..., phase_start : phase_start + 1]
        gait_cos = direction * observation[..., phase_start + 1 : phase_start + 2]
        ramp = observation[..., phase_start + 2 : phase_start + 3]

        pitch_correction = (
            config.pitch_kp * pitch + config.pitch_kd * pitch_rate
        ).clamp(-0.95, 0.95)
        ankle_pitch_correction = (
            config.ankle_pitch_kp * pitch + config.ankle_pitch_kd * pitch_rate
        ).clamp(-1.0, 1.0)
        gait_wave = ramp * gait_sin
        left_swing = ramp * gait_sin.clamp_min(0.0)
        right_swing = ramp * (-gait_sin).clamp_min(0.0)
        lateral_wave = ramp * (
            gait_sin * math.cos(config.lateral_phase_offset)
            + gait_cos * math.sin(config.lateral_phase_offset)
        )

        action = observation.new_zeros(*observation.shape[:-1], MicroDuckEnv.NUM_JOINTS)
        # A common hip-pitch oscillation produces opposite physical leg motion
        # because the left and right joints use opposite sign conventions.
        action[..., 2:3] = -pitch_correction - config.hip_amplitude * gait_wave
        action[..., 11:12] = pitch_correction - config.hip_amplitude * gait_wave
        action[..., 3:4] = config.knee_amplitude * left_swing
        action[..., 12:13] = -config.knee_amplitude * right_swing
        action[..., 4:5] = ankle_pitch_correction - config.ankle_amplitude * left_swing
        action[..., 13:14] = (
            -ankle_pitch_correction + config.ankle_amplitude * right_swing
        )
        action[..., 1:2] = config.lateral_amplitude * lateral_wave
        action[..., 10:11] = config.lateral_amplitude * lateral_wave
        return action.clamp(-1.0, 1.0)


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


def make_actor_critic(
    observation_dim: int,
    num_tasks: int,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 128,
    policy_head: PolicyHead = "gaussian",
    gait: MicroDuckGaitConfig | Mapping[str, float] | None = None,
    residual_scale: float = 0.2,
    initial_policy_scale: float = 0.05,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Create the actor and critic of :func:`make_models` from their sizes alone.

    ``observation_dim`` is the size of the env observation and ``num_tasks``
    the size of the task library the ``task_id`` embedding indexes. Use it to
    rebuild a trained actor without an env, for instance to run a checkpoint
    as a controller inside another env; the recurrent-state primer is then up
    to the caller.
    """
    if not math.isfinite(initial_policy_scale) or initial_policy_scale <= 0:
        raise ValueError("initial_policy_scale must be finite and positive.")
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be finite and positive.")
    device = torch.device(device)
    embed = TensorDictModule(
        TaskConditionedEncoder(
            observation_dim,
            num_tasks,
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
        spec=Bounded(-1.0, 1.0, shape=(MicroDuckEnv.NUM_JOINTS,), device=device),
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
    return actor, TensorDictSequential(backbone, value_head)
