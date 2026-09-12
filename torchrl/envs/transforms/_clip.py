# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import Any, TYPE_CHECKING

import numpy as np

import torch

from tensordict import TensorDictBase, unravel_key
from tensordict.utils import expand_as_right, NestedKey

from torchrl.data.tensor_specs import Bounded, Composite, TensorSpec
from torchrl.envs.transforms.utils import _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import _apply_to_composite, Transform

__all__ = [
    "ClipTransform",
    "DoneTransform",
    "ExpandAs",
]


class ClipTransform(Transform):
    """A transform to clip input (state, action) or output (observation, reward) values.

    This transform can take multiple input or output keys but only one value per
    transform. If multiple clipping values are needed, several transforms should
    be appended one after the other.

    Args:
        in_keys (list of NestedKeys): input entries (read)
        out_keys (list of NestedKeys): input entries (write)
        in_keys_inv (list of NestedKeys): input entries (read) during ``inv`` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during ``inv`` calls.

    Keyword Args:
        low (scalar, optional): the lower bound of the clipped space.
        high (scalar, optional): the higher bound of the clipped space.

    .. note:: Providing just one of the arguments ``low`` or ``high`` is permitted,
        but at least one must be provided.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, ClipTransform(in_keys=['observation'], low=-1, high=0.1))
        >>> r = env.rollout(100)
        >>> assert (r["observation"] <= 0.1).all()
    """

    def __init__(
        self,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
        *,
        low=None,
        high=None,
    ):
        if in_keys is None:
            in_keys = []
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if low is None and high is None:
            raise TypeError("Either one or both of `high` and `low` must be provided.")

        def check_val(val):
            if (isinstance(val, torch.Tensor) and val.numel() > 1) or (
                isinstance(val, np.ndarray) and val.size > 1
            ):
                raise TypeError(
                    f"low and high must be scalars or None. Got low={low} and high={high}."
                )
            if val is None:
                return None, None, torch.finfo(torch.get_default_dtype()).max
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            if not val.dtype.is_floating_point:
                val = val.float()
            eps = torch.finfo(val.dtype).resolution
            ext = torch.finfo(val.dtype).max
            return val, eps, ext

        low, low_eps, low_min = check_val(low)
        high, high_eps, high_max = check_val(high)
        if low is not None and high is not None and low >= high:
            raise ValueError("`low` must be strictly lower than `high`.")
        self.register_buffer("low", low)
        self.low_eps = low_eps
        self.low_min = -low_min
        self.register_buffer("high", high)
        self.high_eps = high_eps
        self.high_max = high_max

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if self.low is None:
            return obs.clamp_max(self.high)
        elif self.high is None:
            return obs.clamp_min(self.low)
        return obs.clamp(self.low, self.high)

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if self.low is None:
            return state.clamp_max(self.high)
        elif self.high is None:
            return state.clamp_min(self.low)
        return state.clamp(self.low, self.high)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return Bounded(
            shape=observation_spec.shape,
            device=observation_spec.device,
            dtype=observation_spec.dtype,
            high=self.high + self.high_eps if self.high is not None else self.high_max,
            low=self.low - self.low_eps if self.low is not None else self.low_min,
        )

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for key in self.in_keys:
            if key in self.parent.reward_keys:
                spec = reward_spec[key]
                reward_spec[key] = Bounded(
                    shape=spec.shape,
                    device=spec.device,
                    dtype=spec.dtype,
                    high=self.high + self.high_eps
                    if self.high is not None
                    else self.high_max,
                    low=self.low - self.low_eps
                    if self.low is not None
                    else self.low_min,
                )
        return reward_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    # No need to transform the input spec since the outside world won't see the difference
    # def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
    #     ...


class ExpandAs(Transform):
    """Expands one entry to the right to match a reference entry shape.

    This is a transform wrapper around :func:`tensordict.utils.expand_as_right`.

    Args:
        in_key (NestedKey): key to expand.
        ref_key (NestedKey): key used as shape reference.
        out_key (NestedKey, optional): output key where the expanded tensor is
            written. Defaults to ``in_key``.

    Examples:
        Expanding an environment-level ``done`` signal to the per-agent reward
        shape in a VMAS environment:

        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.libs.vmas import VmasEnv
        >>> from torchrl.envs.transforms import ExpandAs
        >>> base_env = VmasEnv(
        ...     scenario="navigation",
        ...     num_envs=16,
        ...     continuous_actions=True,
        ...     n_agents=3,
        ... )
        >>> env = TransformedEnv(
        ...     base_env,
        ...     ExpandAs(
        ...         in_key="done",
        ...         ref_key=("agents", "reward"),
        ...     ),
        ... )
        >>> td = env.reset()
        >>> td = env.rand_step(td)
        >>> td["next", "done"].shape == td["next", "agents", "reward"].shape
        True
    """

    def __init__(
        self,
        in_key: NestedKey,
        ref_key: NestedKey,
        out_key: NestedKey | None = None,
    ):
        if out_key is None:
            out_key = in_key
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.in_key = unravel_key(in_key)
        self.ref_key = unravel_key(ref_key)
        self.out_key = unravel_key(out_key)

    @staticmethod
    def _find_key_spec(
        output_spec: Composite, key: NestedKey
    ) -> tuple[str, TensorSpec]:
        for spec_name in (
            "full_observation_spec",
            "full_reward_spec",
            "full_done_spec",
        ):
            if spec_name not in output_spec.keys():
                continue
            spec = output_spec[spec_name]
            if key in spec.keys(True, True):
                return spec_name, spec[key]
        raise KeyError(f"Key {key} was not found in output specs.")

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        ref = next_tensordict.get(self.ref_key, default=None)
        if ref is None:
            if self.missing_tolerance:
                return next_tensordict
            raise KeyError(
                f"{self}: '{self.ref_key}' not found in tensordict {next_tensordict}"
            )
        value = next_tensordict.get(self.in_key, default=None)
        if value is None:
            if self.missing_tolerance:
                return next_tensordict
            raise KeyError(
                f"{self}: '{self.in_key}' not found in tensordict {next_tensordict}"
            )
        next_tensordict.set(self.out_key, expand_as_right(value, ref))
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        if self.out_key in tensordict_reset.keys(True, True):
            return tensordict_reset

        value = tensordict_reset.get(self.in_key, default=None)
        if value is None:
            return tensordict_reset

        ref = tensordict_reset.get(self.ref_key, default=None)
        if ref is None and self.parent is not None:
            try:
                _, ref_spec = self._find_key_spec(self.parent.output_spec, self.ref_key)
            except KeyError:
                ref_spec = None
            if ref_spec is not None:
                ref = torch.empty(
                    ref_spec.shape,
                    dtype=value.dtype,
                    device=value.device,
                )

        if ref is None:
            tensordict_reset.set(self.out_key, value)
        else:
            tensordict_reset.set(self.out_key, expand_as_right(value, ref))
        return tensordict_reset

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_spec = output_spec.clone()
        _, ref_spec = self._find_key_spec(output_spec, self.ref_key)
        in_spec_name, in_spec = self._find_key_spec(output_spec, self.in_key)
        target_spec_name = in_spec_name
        if in_spec_name == "full_done_spec" and self.out_key != self.in_key:
            target_spec_name = "full_observation_spec"

        while len(in_spec.shape) < len(ref_spec.shape):
            in_spec = in_spec.unsqueeze(-1)

        spec = output_spec[target_spec_name]
        spec[self.out_key] = in_spec.expand(ref_spec.shape)
        output_spec[target_spec_name] = spec
        return output_spec


def _as_key_list(
    keys: Sequence[NestedKey] | NestedKey | None,
) -> list[NestedKey] | None:
    if keys is None:
        return None
    if isinstance(keys, (str, tuple)):
        return [unravel_key(keys)]
    return [unravel_key(key) for key in keys]


def _swap_last(source: NestedKey, dest: NestedKey) -> NestedKey:
    """Place the last component of ``dest`` under the group of ``source``."""
    source = unravel_key(source)
    dest = unravel_key(dest)
    leaf = dest[-1] if isinstance(dest, tuple) else dest
    if isinstance(source, str):
        return leaf
    return source[:-1] + (leaf,)


class DoneTransform(Transform):
    """Expands done flags to match the reward shape.

    Multi-agent environments often expose a shared (environment-level) done
    while rewards are per-agent. Value estimators such as GAE expect these
    entries to share a trailing shape. This transform expands each done key
    to the reward shape and writes the result under the reward group
    (for example ``("agents", "done")`` when the reward key is
    ``("agents", "reward")``).

    The transform can be appended to a :class:`~torchrl.envs.TransformedEnv`,
    a collector (as ``postproc``), or a replay buffer. When used as a collector
    or replay-buffer transform, :meth:`forward` expands entries under the
    ``"next"`` sub-tensordict if that key is present.

    Args:
        in_keys (NestedKey or sequence of NestedKey, optional): done keys to
            expand. Defaults to ``("done", "terminated")``. A single NestedKey
            is accepted. Mutually exclusive with ``done_keys``.
        out_keys (NestedKey or sequence of NestedKey, optional): destination
            keys, one per ``in_keys`` entry. Defaults to the last component
            of each input key placed under the reward group (e.g.
            ``("agents", "done")`` if ``reward_key`` is ``("agents", "reward")``
            and the input key ends with ``"done"``).

    Keyword Args:
        reward_key (NestedKey, optional): key of the reward used as the
            expansion target. Defaults to ``"reward"``. The default
            ``out_keys`` are derived from this key's group unless
            ``out_keys`` is provided.
        done_keys (NestedKey or sequence of NestedKey, optional): alias of
            ``in_keys`` kept for compatibility with the historical multi-agent
            helper. Mutually exclusive with ``in_keys``.

    See also :class:`~torchrl.trainers.algorithms.configs.transforms.DoneTransformConfig`.

    Examples:
        Expand shared done flags onto the per-agent reward shape:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import DoneTransform
        >>> n_envs, n_agents = 2, 3
        >>> td = TensorDict(
        ...     {
        ...         "done": torch.tensor([[False], [True]]),
        ...         "terminated": torch.tensor([[False], [True]]),
        ...         "agents": {"reward": torch.zeros(n_envs, n_agents, 1)},
        ...     },
        ...     [n_envs],
        ... )
        >>> transform = DoneTransform(
        ...     in_keys=["done", "terminated"],
        ...     reward_key=("agents", "reward"),
        ... )
        >>> td = transform(td)
        >>> td["agents", "done"].shape
        torch.Size([2, 3, 1])
        >>> bool((td["agents", "done"] == td["done"].unsqueeze(-1)).all())
        True

        As a collector post-processing transform the same keys are expanded
        under ``"next"``:

        >>> collected = TensorDict(
        ...     {
        ...         "next": TensorDict(
        ...             {
        ...                 "done": torch.tensor([[False], [True]]),
        ...                 "terminated": torch.tensor([[False], [True]]),
        ...                 "agents": {"reward": torch.zeros(n_envs, n_agents, 1)},
        ...             },
        ...             [n_envs],
        ...         )
        ...     },
        ...     [n_envs],
        ... )
        >>> collected = DoneTransform(
        ...     reward_key=("agents", "reward"),
        ...     done_keys=["done", "terminated"],
        ... )(collected)
        >>> collected["next", "agents", "done"].shape
        torch.Size([2, 3, 1])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | NestedKey | None = None,
        out_keys: Sequence[NestedKey] | NestedKey | None = None,
        *,
        reward_key: NestedKey | None = None,
        done_keys: Sequence[NestedKey] | NestedKey | None = None,
    ) -> None:
        if in_keys is not None and done_keys is not None:
            raise TypeError("Specify either in_keys or done_keys, not both.")
        if done_keys is not None:
            in_keys = done_keys
        if in_keys is None:
            in_keys = ["done", "terminated"]
        in_keys = _as_key_list(in_keys)
        if reward_key is None:
            reward_key = "reward"
        self.reward_key = unravel_key(reward_key)
        if out_keys is None:
            out_keys = [_swap_last(self.reward_key, key) for key in in_keys]
        else:
            out_keys = _as_key_list(out_keys)
            if len(out_keys) != len(in_keys):
                raise ValueError(
                    "out_keys must have the same length as in_keys, "
                    f"got {len(out_keys)} and {len(in_keys)}."
                )
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _reference_tensor(
        self, tensordict: TensorDictBase, value: torch.Tensor
    ) -> torch.Tensor | None:
        reward = tensordict.get(self.reward_key, default=None)
        if reward is not None:
            return reward
        parent = self.parent
        if parent is None:
            return None
        try:
            _, ref_spec = ExpandAs._find_key_spec(parent.output_spec, self.reward_key)
        except KeyError:
            return None
        return value.new_empty(ref_spec.shape)

    def _expand_in_tensordict(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict.get(in_key, default=None)
            if value is None:
                if self.missing_tolerance:
                    continue
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {tensordict}"
                )
            ref = self._reference_tensor(tensordict, value)
            if ref is None:
                if self.missing_tolerance:
                    continue
                raise KeyError(
                    f"{self}: '{self.reward_key}' not found in tensordict {tensordict}"
                )
            tensordict.set(out_key, expand_as_right(value, ref))
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self._expand_in_tensordict(next_tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        nxt = tensordict.get("next", default=None)
        if nxt is not None:
            self._expand_in_tensordict(nxt)
            return tensordict
        return self._expand_in_tensordict(tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_spec = output_spec.clone()
        try:
            _, ref_spec = ExpandAs._find_key_spec(output_spec, self.reward_key)
        except KeyError:
            return output_spec
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            try:
                in_spec_name, in_spec = ExpandAs._find_key_spec(output_spec, in_key)
            except KeyError:
                continue
            target_spec_name = in_spec_name
            if in_spec_name == "full_done_spec" and out_key != in_key:
                target_spec_name = "full_observation_spec"
            while len(in_spec.shape) < len(ref_spec.shape):
                in_spec = in_spec.unsqueeze(-1)
            spec = output_spec[target_spec_name]
            spec[out_key] = in_spec.expand(ref_spec.shape)
            output_spec[target_spec_name] = spec
        return output_spec

