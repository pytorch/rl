# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sim-to-real transforms: sensor noise, latency, actuator dynamics and symmetry."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from tensordict import TensorDictBase
from tensordict.utils import expand_as_right, NestedKey, unravel_key

from torchrl._utils import _replace_last
from torchrl.data.tensor_specs import Bounded, TensorSpec, Unbounded
from torchrl.envs.transforms._base import (
    FORWARD_NOT_IMPLEMENTED,
    ObservationTransform,
    Transform,
)

__all__ = [
    "ActionDelay",
    "ActuatorModel",
    "ObservationDelay",
    "ObservationNoise",
    "SymmetryAugmentation",
]


def _as_key_list(
    value: Sequence[NestedKey] | NestedKey | None,
) -> list[NestedKey] | None:
    if value is None:
        return None
    if isinstance(value, (str, tuple)):
        value = [value]
    return [unravel_key(val) for val in value]


def _last(key: NestedKey) -> str:
    return key if isinstance(key, str) else key[-1]


def _parse_range(
    name: str, value: float | Sequence[float], *, integer: bool
) -> tuple[float, float]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(
                f"{name} must be a number or a (low, high) pair, got {value!r}."
            )
        low, high = value
    else:
        low = high = value
    if integer and not (isinstance(low, int) and isinstance(high, int)):
        raise TypeError(f"{name} must be an integer or a pair of integers.")
    if low > high:
        raise ValueError(f"{name} low bound {low} exceeds high bound {high}.")
    return low, high


def _select_delayed(history: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """Returns ``history[..., -1 - delay]`` with a per-batch-element ``delay``.

    ``history`` stacks entries along its last dimension, newest last, and its
    leading dimensions start with ``delay.shape``.
    """
    index = history.shape[-1] - 1 - delay
    index = index.reshape(*index.shape, *([1] * (history.ndim - 1 - index.ndim)))
    index = index.expand(history.shape[:-1]).unsqueeze(-1)
    return history.gather(-1, index).squeeze(-1)


class _ActionStateTransform(Transform):
    """Base for transforms that act on actions and keep state as observations.

    ``in_keys`` default to the parent's action keys; ``out_keys`` hold the
    state carried across steps and default to the action key with its last
    component replaced by :attr:`_out_suffix`.
    """

    _out_suffix: str
    invertible = False

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is not None:
            return in_keys
        parent = self.parent
        if parent is None:
            return ["action"]
        in_keys = list(parent.action_keys)
        self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value: Sequence[NestedKey] | NestedKey | None) -> None:
        self._in_keys = _as_key_list(value)

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is not None:
            return out_keys
        derived = [_replace_last(key, self._out_suffix) for key in self.in_keys]
        if self.__dict__.get("_in_keys", None) is not None:
            self._out_keys = derived
        return derived

    @out_keys.setter
    def out_keys(self, value: Sequence[NestedKey] | NestedKey | None) -> None:
        self._out_keys = _as_key_list(value)

    def _action_spec(self, in_key: NestedKey) -> TensorSpec:
        parent = self.parent
        if parent is None:
            raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self).__name__))
        try:
            return parent.full_action_spec[in_key]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} in_key {in_key!r} is not in the parent "
                f"action spec {parent.full_action_spec}."
            ) from None

    def _zero_action(
        self, in_key: NestedKey, batch_size: torch.Size, fill: float
    ) -> torch.Tensor:
        spec = self._action_spec(in_key)
        # Unlocked parents can reset with a larger runtime batch than the spec.
        extra_batch = () if self.parent.batch_locked else batch_size
        return spec.zero(extra_batch).fill_(fill)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict

    def _reset_on_native_autoreset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._reset(tensordict, tensordict_reset)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )


class ObservationNoise(ObservationTransform):
    """Adds Gaussian noise to observations, at every step and at reset.

    Each selected observation becomes ``obs + loc + std * eps`` with ``eps``
    drawn from a standard normal at every call, which models sensor noise and
    a fixed calibration offset. Observation specs of ``out_keys`` become
    :class:`~torchrl.data.Unbounded`, as noise can push values outside their
    original bounds.

    The transform also works outside an environment, e.g. as a replay-buffer
    transform that perturbs sampled observations.

    Args:
        std (float or torch.Tensor): standard deviation of the noise. A tensor
            is broadcast against the observation (e.g. one entry per feature).
        in_keys (NestedKey or sequence of NestedKey, optional): observations to
            perturb. Defaults to ``["observation"]``.
        out_keys (NestedKey or sequence of NestedKey, optional): destination
            keys. Defaults to ``in_keys`` (observations are replaced).

    Keyword Args:
        loc (float or torch.Tensor, optional): constant offset added to the
            observation, broadcast like ``std``. Defaults to ``0.0``.

    Examples:
        >>> import torch
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import ObservationNoise
        >>> base_env = GymEnv("Pendulum-v1")
        >>> # per-feature noise on cos(theta), sin(theta), theta_dot
        >>> env = TransformedEnv(
        ...     base_env,
        ...     ObservationNoise(
        ...         std=torch.tensor([0.01, 0.01, 0.1]),
        ...         out_keys=["noisy_observation"],
        ...     ),
        ... )
        >>> rollout = env.rollout(5)
        >>> noise = rollout["noisy_observation"] - rollout["observation"]
        >>> noise.shape
        torch.Size([5, 3])
    """

    def __init__(
        self,
        std: float | torch.Tensor,
        in_keys: Sequence[NestedKey] | NestedKey | None = None,
        out_keys: Sequence[NestedKey] | NestedKey | None = None,
        *,
        loc: float | torch.Tensor = 0.0,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = in_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if isinstance(std, torch.Tensor):
            if (std < 0).any():
                raise ValueError("ObservationNoise std must be non-negative.")
            self.register_buffer("std", std.clone())
        else:
            if std < 0:
                raise ValueError("ObservationNoise std must be non-negative.")
            self.std = std
        if isinstance(loc, torch.Tensor):
            self.register_buffer("loc", loc.clone())
        else:
            self.loc = loc

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if not obs.dtype.is_floating_point:
            raise TypeError(
                f"{type(self).__name__} requires floating-point observations, "
                f"got dtype={obs.dtype}."
            )
        return obs + self.loc + self.std * torch.randn_like(obs)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            spec = observation_spec[in_key]
            observation_spec[out_key] = Unbounded(
                shape=spec.shape, dtype=spec.dtype, device=spec.device
            )
        return observation_spec


class ObservationDelay(Transform):
    """Delivers observations a fixed or random number of steps late.

    At step ``t`` the policy reads the observation produced at step
    ``t - delay``. Right after a reset, when fewer than ``delay`` observations
    exist, the reset observation is repeated. The raw observations needed to
    serve delayed values are kept in the tensordict under ``history_keys``
    (one entry per ``in_key``, stacked along a new last dimension of size
    ``max(delay)``), so the transform works with batched, parallel and
    partially-reset environments.

    A single instance applies one delay to all its ``in_keys``. Sensors with
    distinct latencies are modelled with one instance per sensor in a
    :class:`~torchrl.envs.transforms.Compose`.

    Args:
        delay (int or tuple of two ints): latency in environment steps. A
            ``(low, high)`` pair samples an integer delay uniformly in
            ``[low, high]`` (both included) for each environment at every
            reset; the delay then stays fixed for the episode and is written
            under ``delay_key``. The maximum delay must be at least ``1``.
        in_keys (NestedKey or sequence of NestedKey, optional): observations to
            delay. Defaults to ``["observation"]``.
        out_keys (NestedKey or sequence of NestedKey, optional): keys where the
            delayed observations are written. Defaults to ``in_keys``.

    Keyword Args:
        history_keys (NestedKey or sequence of NestedKey, optional): keys
            holding the raw observation history. Defaults to each ``in_key``
            with ``"_history"`` appended to its last component (e.g.
            ``"pixels"`` -> ``"pixels_history"``).
        delay_key (NestedKey, optional): key of the per-environment delay when
            ``delay`` is a range. Defaults to ``"observation_delay"``.

    Examples:
        >>> from torchrl.envs import Compose, GymEnv, SerialEnv, TransformedEnv
        >>> from torchrl.envs.transforms import ObservationDelay
        >>> base_env = SerialEnv(4, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(
        ...     base_env,
        ...     ObservationDelay(2, out_keys=["delayed_observation"]),
        ... )
        >>> rollout = env.rollout(10)
        >>> (rollout[:, 2:]["delayed_observation"] == rollout[:, :-2]["observation"]).all()
        tensor(True)
        >>> # a random 1-3 step latency, resampled per environment at each reset
        >>> env = TransformedEnv(base_env, ObservationDelay((1, 3)))
        >>> env.reset()["observation_delay"].shape
        torch.Size([4])
        >>> # per-sensor latencies: one instance per sensor
        >>> transform = Compose(
        ...     ObservationDelay(4, in_keys=["joint_pos"]),
        ...     ObservationDelay(1, in_keys=["pixels"]),
        ... )
    """

    invertible = False

    def __init__(
        self,
        delay: int | tuple[int, int],
        in_keys: Sequence[NestedKey] | NestedKey | None = None,
        out_keys: Sequence[NestedKey] | NestedKey | None = None,
        *,
        history_keys: Sequence[NestedKey] | NestedKey | None = None,
        delay_key: NestedKey = "observation_delay",
    ):
        self.min_delay, self.max_delay = _parse_range("delay", delay, integer=True)
        if self.min_delay < 0 or self.max_delay < 1:
            raise ValueError(
                f"delay must be non-negative with a maximum of at least 1, got "
                f"{delay!r}."
            )
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = in_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        history_keys = _as_key_list(history_keys)
        if history_keys is None:
            history_keys = [
                _replace_last(key, f"{_last(key)}_history") for key in self.in_keys
            ]
        if len(history_keys) != len(self.in_keys):
            raise ValueError("history_keys and in_keys must have the same length.")
        self.history_keys = history_keys
        self.delay_key = unravel_key(delay_key)

    @property
    def _random_delay(self) -> bool:
        return self.min_delay != self.max_delay

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        delay = None
        for in_key, out_key, history_key in zip(
            self.in_keys, self.out_keys, self.history_keys, strict=True
        ):
            obs = tensordict_reset.get(in_key)
            history = obs.unsqueeze(-1).expand(*obs.shape, self.max_delay)
            tensordict_reset.set(history_key, history.clone())
            if out_key != in_key:
                tensordict_reset.set(out_key, obs.clone())
            if delay is None and self._random_delay:
                delay = torch.randint(
                    self.min_delay,
                    self.max_delay + 1,
                    tensordict_reset.batch_size,
                    device=obs.device,
                )
                tensordict_reset.set(self.delay_key, delay)
        return tensordict_reset

    def _reset_on_native_autoreset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._reset(tensordict, tensordict_reset)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        delay = tensordict.get(self.delay_key) if self._random_delay else None
        for in_key, out_key, history_key in zip(
            self.in_keys, self.out_keys, self.history_keys, strict=True
        ):
            obs = next_tensordict.get(in_key)
            history = torch.cat(
                [tensordict.get(history_key), obs.unsqueeze(-1)], dim=-1
            )
            if delay is None:
                delayed = history[..., 0]
            else:
                delayed = _select_delayed(history, delay)
            next_tensordict.set(history_key, history[..., 1:])
            next_tensordict.set(out_key, delayed)
        if delay is not None:
            next_tensordict.set(self.delay_key, delay)
        return next_tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key, history_key in zip(
            self.in_keys, self.out_keys, self.history_keys, strict=True
        ):
            spec = observation_spec[in_key]
            observation_spec[history_key] = Unbounded(
                shape=(*spec.shape, self.max_delay),
                dtype=spec.dtype,
                device=spec.device,
            )
            if out_key != in_key:
                observation_spec[out_key] = spec.clone()
        if self._random_delay:
            observation_spec[self.delay_key] = Bounded(
                low=self.min_delay,
                high=self.max_delay,
                shape=observation_spec.shape,
                dtype=torch.int64,
                device=observation_spec.device,
            )
        return observation_spec


class ActionDelay(_ActionStateTransform):
    """Executes each action a fixed or random number of steps after it is issued.

    The policy's action at step ``t`` reaches the base environment at step
    ``t + delay``; in between, the environment executes older actions from a
    queue. The queue starts filled with ``default`` after a reset, so the
    first ``delay`` executed actions are ``default``. The tensordict returned
    by :meth:`~torchrl.envs.EnvBase.step` keeps the action the policy issued,
    which is what a learner should be trained on.

    The queue is stored as an observation under ``out_keys`` (stacked along a
    new last dimension of size ``max(delay)``, oldest first) so the policy can
    condition on the actions in flight, and so the transform works with
    batched, parallel and partially-reset environments.

    In a :class:`~torchrl.envs.transforms.Compose`, inverse transforms run
    last-to-first: an :class:`ActuatorModel` placed *before* this transform
    receives the delayed action (delay on the command bus, then actuator
    dynamics).

    Args:
        delay (int or tuple of two ints): latency in environment steps. A
            ``(low, high)`` pair samples an integer delay uniformly in
            ``[low, high]`` (both included) for each environment at every
            reset; the delay then stays fixed for the episode and is written
            under ``delay_key``. The maximum delay must be at least ``1``.
        in_keys (NestedKey or sequence of NestedKey, optional): actions to
            delay. Defaults to the parent environment's
            :attr:`~torchrl.envs.EnvBase.action_keys`.
        out_keys (NestedKey or sequence of NestedKey, optional): keys holding
            the action queues. Defaults to each ``in_key`` with its last
            component replaced by ``"action_queue"``.

    Keyword Args:
        default (float, optional): value of the actions in the queue after a
            reset. Defaults to ``0.0``.
        delay_key (NestedKey, optional): key of the per-environment delay when
            ``delay`` is a range. Defaults to ``"action_delay"``.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import ActionDelay
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), ActionDelay(2))
        >>> td = env.reset()
        >>> td["action_queue"].shape  # (action_dim, delay)
        torch.Size([1, 2])
        >>> rollout = env.rollout(5)
        >>> # the last queued entry is the action issued at the previous step
        >>> (rollout["next", "action_queue"][..., -1] == rollout["action"]).all()
        tensor(True)
        >>> # 1 to 4 steps of command latency, resampled at each reset
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), ActionDelay((1, 4)))
        >>> env.reset()["action_delay"].shape
        torch.Size([])
    """

    _out_suffix = "action_queue"

    def __init__(
        self,
        delay: int | tuple[int, int],
        in_keys: Sequence[NestedKey] | NestedKey | None = None,
        out_keys: Sequence[NestedKey] | NestedKey | None = None,
        *,
        default: float = 0.0,
        delay_key: NestedKey = "action_delay",
    ):
        self.min_delay, self.max_delay = _parse_range("delay", delay, integer=True)
        if self.min_delay < 0 or self.max_delay < 1:
            raise ValueError(
                f"delay must be non-negative with a maximum of at least 1, got "
                f"{delay!r}."
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.default = default
        self.delay_key = unravel_key(delay_key)

    @property
    def _random_delay(self) -> bool:
        return self.min_delay != self.max_delay

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        delay = None
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            action = self._zero_action(
                in_key, tensordict_reset.batch_size, self.default
            )
            queue = action.unsqueeze(-1).expand(*action.shape, self.max_delay)
            tensordict_reset.set(out_key, queue.clone())
            if delay is None and self._random_delay:
                delay = torch.randint(
                    self.min_delay,
                    self.max_delay + 1,
                    tensordict_reset.batch_size,
                    device=action.device,
                )
                tensordict_reset.set(self.delay_key, delay)
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        delay = tensordict.get(self.delay_key) if self._random_delay else None
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            action = tensordict.get(in_key, None)
            if action is None:
                if not self.missing_tolerance:
                    raise KeyError(
                        f"{self}: '{in_key}' not found in tensordict {tensordict}"
                    )
                continue
            queue = tensordict.get(out_key)
            history = torch.cat([queue, action.unsqueeze(-1)], dim=-1)
            if delay is None:
                executed = history[..., 0]
            else:
                executed = _select_delayed(history, delay)
            tensordict.set(in_key, executed)
            # Entries under "next" written by an inverse transform are merged
            # into the step output by TransformedEnv.
            tensordict.set(unravel_key(("next", out_key)), history[..., 1:])
        if delay is not None:
            tensordict.set(unravel_key(("next", self.delay_key)), delay)
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            spec = self._action_spec(in_key)
            # Reset fills (e.g. zeros on a bounded action) need not be in the
            # action domain.
            observation_spec[out_key] = Unbounded(
                shape=(*spec.shape, self.max_delay),
                dtype=spec.dtype,
                device=spec.device,
            )
        if self._random_delay:
            observation_spec[self.delay_key] = Bounded(
                low=self.min_delay,
                high=self.max_delay,
                shape=observation_spec.shape,
                dtype=torch.int64,
                device=observation_spec.device,
            )
        return observation_spec


class ActuatorModel(_ActionStateTransform):
    """First-order actuator dynamics between the policy and the simulator.

    Real actuators do not reach a commanded set-point instantly. This
    transform replaces each command ``u_t`` by the applied value

    .. math::

        a_t = a_{t-1} + \\mathrm{clip}(k (u_t - a_{t-1}), -\\delta, \\delta)

    where ``k`` is the per-step ``gain`` (``1`` executes commands as-is,
    smaller values give a first-order lag with time constant
    ``-dt / log(1 - k)``) and ``\\delta`` is an optional slew-rate limit
    ``max_delta``. The applied value is stored as an observation under
    ``out_keys`` so the policy can read the actuator state. The tensordict
    returned by :meth:`~torchrl.envs.EnvBase.step` keeps the commanded action.

    For reference, the joints of an SO-101 arm driven by Feetech STS3215 servos
    at 30 Hz follow ``k ~= 0.25``.

    Args:
        gain (float or tuple of two floats, optional): per-step gain ``k`` in
            ``(0, 1]``. A ``(low, high)`` pair samples ``k`` uniformly for each
            environment at every reset and writes it under ``gain_key``.
            Defaults to ``1.0``.
        in_keys (NestedKey or sequence of NestedKey, optional): actions to
            filter. Defaults to the parent environment's
            :attr:`~torchrl.envs.EnvBase.action_keys`.
        out_keys (NestedKey or sequence of NestedKey, optional): keys holding
            the applied actions. Defaults to each ``in_key`` with its last
            component replaced by ``"applied_action"``.

    Keyword Args:
        max_delta (float, optional): maximum change of the applied action per
            step, applied elementwise. Defaults to ``None`` (no limit).
        default (float, optional): applied action after a reset, e.g. the
            resting position of a position-controlled joint. Defaults to
            ``0.0``.
        gain_key (NestedKey, optional): key of the per-environment gain when
            ``gain`` is a range. Defaults to ``"actuator_gain"``.

    Examples:
        >>> from torchrl.envs import Compose, GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import ActionDelay, ActuatorModel
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"), ActuatorModel(0.25, max_delta=0.5)
        ... )
        >>> rollout = env.rollout(5)
        >>> rollout["next", "applied_action"].shape
        torch.Size([5, 1])
        >>> # servo model of a real arm: 4 steps of bus latency, then a lag
        >>> # whose gain is randomized per episode
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     Compose(ActuatorModel((0.2, 0.3)), ActionDelay(4)),
        ... )
        >>> td = env.reset()
        >>> td["actuator_gain"].shape, td["action_queue"].shape
        (torch.Size([]), torch.Size([1, 4]))
    """

    _out_suffix = "applied_action"

    def __init__(
        self,
        gain: float | tuple[float, float] = 1.0,
        in_keys: Sequence[NestedKey] | NestedKey | None = None,
        out_keys: Sequence[NestedKey] | NestedKey | None = None,
        *,
        max_delta: float | None = None,
        default: float = 0.0,
        gain_key: NestedKey = "actuator_gain",
    ):
        self.min_gain, self.max_gain = _parse_range("gain", gain, integer=False)
        if self.min_gain <= 0 or self.max_gain > 1:
            raise ValueError(f"gain must lie in (0, 1], got {gain!r}.")
        if max_delta is not None and max_delta <= 0:
            raise ValueError(f"max_delta must be positive, got {max_delta!r}.")
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.max_delta = max_delta
        self.default = default
        self.gain_key = unravel_key(gain_key)

    @property
    def _random_gain(self) -> bool:
        return self.min_gain != self.max_gain

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        gain = None
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            applied = self._zero_action(
                in_key, tensordict_reset.batch_size, self.default
            )
            tensordict_reset.set(out_key, applied)
            if gain is None and self._random_gain:
                gain = torch.empty(
                    tensordict_reset.batch_size,
                    dtype=applied.dtype,
                    device=applied.device,
                ).uniform_(self.min_gain, self.max_gain)
                tensordict_reset.set(self.gain_key, gain)
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        gain = tensordict.get(self.gain_key) if self._random_gain else self.min_gain
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            command = tensordict.get(in_key, None)
            if command is None:
                if not self.missing_tolerance:
                    raise KeyError(
                        f"{self}: '{in_key}' not found in tensordict {tensordict}"
                    )
                continue
            previous = tensordict.get(out_key)
            step_gain = (
                expand_as_right(gain, command)
                if isinstance(gain, torch.Tensor)
                else gain
            )
            delta = step_gain * (command - previous)
            if self.max_delta is not None:
                delta = delta.clamp(-self.max_delta, self.max_delta)
            applied = previous + delta
            tensordict.set(in_key, applied)
            # Entries under "next" written by an inverse transform are merged
            # into the step output by TransformedEnv.
            tensordict.set(unravel_key(("next", out_key)), applied)
        if self._random_gain:
            tensordict.set(unravel_key(("next", self.gain_key)), gain)
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys, strict=True):
            spec = self._action_spec(in_key)
            if not spec.dtype.is_floating_point:
                raise TypeError(
                    f"{type(self).__name__} requires a floating-point action, "
                    f"got dtype={spec.dtype} for key {in_key!r}."
                )
            observation_spec[out_key] = Unbounded(
                shape=spec.shape, dtype=spec.dtype, device=spec.device
            )
        if self._random_gain:
            observation_spec[self.gain_key] = Bounded(
                low=self.min_gain,
                high=self.max_gain,
                shape=observation_spec.shape,
                dtype=self._action_spec(self.in_keys[0]).dtype,
                device=observation_spec.device,
            )
        return observation_spec


class SymmetryAugmentation(Transform):
    """Augments batches of transitions with their mirror image.

    Many robots are symmetric (e.g. left/right for a legged robot): mirroring
    a valid transition gives another valid transition. Each selected entry is
    mirrored along its last dimension as ``x[..., index] * sign``, with one
    ``index`` and ``sign`` per key. Entries that are not listed (rewards,
    dones, ...) are copied unchanged, so they must be invariant under the
    symmetry. Quantities that depend on the policy that collected the data
    (e.g. log-probabilities used by PPO) are not mirrored and should be
    recomputed or excluded.

    This is a batch transform for replay buffers and training data; it does
    not act on environment steps.

    Reference: Mittal et al., "Symmetry Considerations for Learning Task
    Symmetric Robot Policies", ICRA 2024. https://arxiv.org/abs/2403.04359

    Args:
        in_keys (NestedKey or sequence of NestedKey): entries to mirror. Next
            observations are listed explicitly (e.g. ``("next",
            "observation")``).

    Keyword Args:
        index (sequence of sequences of int or torch.Tensor): for each
            ``in_key``, the permutation of its last dimension, i.e. ``index[j]``
            is the position whose value goes to position ``j``.
        sign (sequence of sequences of float, torch.Tensor or None, optional):
            for each ``in_key``, a multiplier applied after the permutation
            (``-1`` flips a lateral velocity), or ``None`` for no sign change.
            Defaults to ``None`` for every key.
        mode (str, optional): ``"append"`` (default) concatenates the mirrored
            batch after the original along the first dimension, doubling it;
            ``"random"`` mirrors each element of the first dimension with
            probability one half and keeps the batch size.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.envs.transforms import SymmetryAugmentation
        >>> # observation = (left_foot, right_foot, lateral_velocity)
        >>> # action = (left_torque, right_torque)
        >>> obs_index, obs_sign = [1, 0, 2], [1.0, 1.0, -1.0]
        >>> transform = SymmetryAugmentation(
        ...     ["observation", ("next", "observation"), "action"],
        ...     index=[obs_index, obs_index, [1, 0]],
        ...     sign=[obs_sign, obs_sign, None],
        ... )
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100), transform=transform, batch_size=8
        ... )
        >>> data = TensorDict(
        ...     observation=torch.randn(16, 3),
        ...     action=torch.randn(16, 2),
        ...     next=TensorDict(observation=torch.randn(16, 3), batch_size=[16]),
        ...     batch_size=[16],
        ... )
        >>> indices = rb.extend(data)
        >>> rb.sample().shape
        torch.Size([16])
    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | NestedKey,
        *,
        index: Sequence[Sequence[int] | torch.Tensor],
        sign: Sequence[Sequence[float] | torch.Tensor | None] | None = None,
        mode: Literal["append", "random"] = "append",
    ):
        super().__init__(in_keys=in_keys, out_keys=in_keys)
        if mode not in ("append", "random"):
            raise ValueError(f"mode must be 'append' or 'random', got {mode!r}.")
        if sign is None:
            sign = [None] * len(self.in_keys)
        if not (len(index) == len(sign) == len(self.in_keys)):
            raise ValueError("index, sign and in_keys must have the same length.")
        self.mode = mode
        self._has_sign = []
        for i, (idx, sgn) in enumerate(zip(index, sign)):
            idx = torch.as_tensor(idx, dtype=torch.long)
            if idx.ndim != 1:
                raise ValueError(f"index[{i}] must be one-dimensional.")
            self.register_buffer(f"_index_{i}", idx)
            self._has_sign.append(sgn is not None)
            if sgn is not None:
                sgn = torch.as_tensor(sgn)
                if sgn.shape != idx.shape:
                    raise ValueError(
                        f"sign[{i}] has shape {tuple(sgn.shape)}, expected "
                        f"{tuple(idx.shape)} to match index[{i}]."
                    )
                self.register_buffer(f"_sign_{i}", sgn)

    def _mirror(self, i: int, value: torch.Tensor) -> torch.Tensor:
        mirrored = value.index_select(-1, getattr(self, f"_index_{i}"))
        if self._has_sign[i]:
            mirrored = (mirrored * getattr(self, f"_sign_{i}")).to(value.dtype)
        return mirrored

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        mirrored = tensordict.clone(False)
        mask = None
        for i, key in enumerate(self.in_keys):
            value = tensordict.get(key)
            flipped = self._mirror(i, value)
            if self.mode == "random":
                if mask is None:
                    mask = (
                        torch.rand(tensordict.shape[:1], device=value.device) < 0.5
                    )
                flipped = torch.where(expand_as_right(mask, value), flipped, value)
            mirrored.set(key, flipped)
        if self.mode == "random":
            return mirrored
        return torch.cat([tensordict, mirrored], dim=0)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            f"{type(self).__name__} augments batches of transitions; use it as "
            f"a replay buffer transform or call it on a batch, not inside an "
            f"environment."
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be appended to an environment."
        )
