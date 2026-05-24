# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import torch

from tensordict import TensorDictBase
from tensordict.utils import NestedKey

from torchrl.data.tensor_specs import TensorSpec, Unbounded

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import FORWARD_NOT_IMPLEMENTED, Transform

__all__ = [
    "Timer",
]


class Timer(Transform):
    """A transform that measures the time intervals between `inv` and `call` operations in an environment.

    The `Timer` transform is used to track the time elapsed between the `inv` call and the `call`,
    and between the `call` and the `inv` call. This is useful for performance monitoring and debugging
    within an environment. The time is measured in seconds and stored as a tensor with the default
    dtype from PyTorch. If the tensordict has a batch size (e.g., in batched environments), the time will be expended
    to the size of the input tensordict.

    Attributes:
        out_keys: The keys of the output tensordict for the inverse transform. Defaults to
            `out_keys = [f"{time_key}_step", f"{time_key}_policy", f"{time_key}_reset"]`, where the first key represents
            the time it takes to make a step in the environment, and the second key represents the
            time it takes to execute the policy, the third the time for the call to `reset`.
        time_key: A prefix for the keys where the time intervals will be stored in the tensordict.
            Defaults to `"time"`.

    .. note:: During a succession of rollouts, the time marks of the reset are written at the root (the `"time_reset"`
        entry or equivalent key is always 0 in the `"next"` tensordict). At the root, the `"time_policy"` and `"time_step"`
        entries will be 0 when there is a reset. they will never be `0` in the `"next"`.

    Examples:
        >>> from torchrl.envs import Timer, GymEnv
        >>>
        >>> env = GymEnv("Pendulum-v1").append_transform(Timer())
        >>> r = env.rollout(10)
        >>> print("time for policy", r["time_policy"])
        time for policy tensor([0.0000, 0.0882, 0.0004, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
                0.0002])
        >>> print("time for step", r["time_step"])
        time for step tensor([9.5797e-04, 1.6289e-03, 9.7990e-05, 8.0824e-05, 9.0837e-05, 7.6056e-05,
                8.2016e-05, 7.6056e-05, 8.1062e-05, 7.7009e-05])


    """

    def __init__(self, out_keys: Sequence[NestedKey] = None, time_key: str = "time"):
        if out_keys is None:
            out_keys = [f"{time_key}_step", f"{time_key}_policy", f"{time_key}_reset"]
        elif len(out_keys) != 3:
            raise TypeError(f"Expected three out_keys. Got out_keys={out_keys}.")
        super().__init__([], out_keys)
        self.time_key = time_key
        self.last_inv_time = None
        self.last_call_time = None
        self.last_reset_time = None
        self.time_step_key = self.out_keys[0]
        self.time_policy_key = self.out_keys[1]
        self.time_reset_key = self.out_keys[2]

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.last_reset_time = self.last_inv_time = time.time()
        return tensordict

    def _maybe_expand_and_set(self, key, time_elapsed, tensordict):
        if isinstance(key, tuple):
            parent_td = tensordict.get(key[:-1])
            key = key[-1]
        else:
            parent_td = tensordict
        batch_size = parent_td.batch_size
        if batch_size:
            # Get the parent shape
            time_elapsed_expand = time_elapsed.expand(parent_td.batch_size)
        else:
            time_elapsed_expand = time_elapsed
        parent_td.set(key, time_elapsed_expand)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        current_time = time.time()
        if self.last_reset_time is not None:
            time_elapsed = torch.tensor(
                current_time - self.last_reset_time, device=tensordict.device
            )
            self._maybe_expand_and_set(
                self.time_reset_key, time_elapsed, tensordict_reset
            )
            self._maybe_expand_and_set(
                self.time_step_key, time_elapsed * 0, tensordict_reset
            )
        self.last_call_time = current_time
        # Placeholder
        self._maybe_expand_and_set(
            self.time_policy_key, time_elapsed * 0, tensordict_reset
        )
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        current_time = time.time()
        if self.last_call_time is not None:
            time_elapsed = torch.tensor(
                current_time - self.last_call_time, device=tensordict.device
            )
            self._maybe_expand_and_set(self.time_policy_key, time_elapsed, tensordict)
        self.last_inv_time = current_time
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        current_time = time.time()
        if self.last_inv_time is not None:
            time_elapsed = torch.tensor(
                current_time - self.last_inv_time, device=tensordict.device
            )
            self._maybe_expand_and_set(
                self.time_step_key, time_elapsed, next_tensordict
            )
            self._maybe_expand_and_set(
                self.time_reset_key, time_elapsed * 0, next_tensordict
            )
        self.last_call_time = current_time
        # presumbly no need to worry about batch size incongruencies here
        next_tensordict.set(self.time_policy_key, tensordict.get(self.time_policy_key))
        return next_tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec[self.time_step_key] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        observation_spec[self.time_policy_key] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        observation_spec[self.time_reset_key] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(FORWARD_NOT_IMPLEMENTED)
