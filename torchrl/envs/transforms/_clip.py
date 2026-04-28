# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
                spec = self.parent.output_spec["full_reward_spec"][key]
                self.parent.output_spec["full_reward_spec"][key] = Bounded(
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
        return self.parent.output_spec["full_reward_spec"]

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
