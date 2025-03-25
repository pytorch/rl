# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import uuid
import warnings
from copy import copy

from typing import Any, OrderedDict, Sequence

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase, unravel_key
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded

from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import Compose, ObservationNorm, Transform

from torchrl.envs.transforms.utils import _set_missing_tolerance


class VecNormV2(Transform):
    """A class for normalizing vectorized observations and rewards in reinforcement learning environments.

    `VecNormV2` can operate in either a stateful or stateless mode. In stateful mode, it maintains
    internal statistics (mean and variance) to normalize inputs. In stateless mode, it requires
    external statistics to be provided for normalization.

    .. note:: This class is designed to be an almost drop-in replacement for :class:`~torchrl.envs.transforms.VecNorm`.
        It should not be constructed directly, but rather with the :class:`~torchrl.envs.transforms.VecNorm`
        transform using the `new_api=True` keyword argument. In v0.10, the :class:`~torchrl.envs.transforms.VecNorm`
        transform will be switched to the new api by default.

    Stateful vs. Stateless:
        Stateful Mode (`stateful=True`):

            - Maintains internal statistics (`loc`, `var`, `count`) for normalization.
            - Updates statistics with each call unless frozen.
            - `state_dict` returns the current statistics.
            - `load_state_dict` updates the internal statistics with the provided state.

        Stateless Mode (`stateful=False`):

            - Requires external statistics to be provided for normalization.
            - Does not maintain or update internal statistics.
            - `state_dict` returns an empty dictionary.
            - `load_state_dict` does not affect internal state.

    Args:
        in_keys (Sequence[NestedKey]): The input keys for the data to be normalized.
        out_keys (Sequence[NestedKey] | None): The output keys for the normalized data. Defaults to `in_keys` if
            not provided.
        lock (mp.Lock, optional): A lock for thread safety.
        stateful (bool, optional): Whether the `VecNorm` is stateful. Defaults to `True`.
        decay (float, optional): The decay rate for updating statistics. Defaults to `0.9999`.
        eps (float, optional): A small value to prevent division by zero. Defaults to `1e-4`.
        shapes (list[torch.Size], optional): The shapes of the inputs. Defaults to `None`.
        shared_data (TensorDictBase | None, optional): Shared data for initialization. Defaults to `None`.

    Attributes:
        stateful (bool): Indicates whether the VecNormV2 is stateful or stateless.
        lock (mp.Lock): A multiprocessing lock to ensure thread safety when updating statistics.
        decay (float): The decay rate for updating statistics.
        eps (float): A small value to prevent division by zero during normalization.
        shapes (list[torch.Size]): The shapes of the inputs to be normalized.
        frozen (bool): Indicates whether the VecNormV2 is frozen, preventing updates to statistics.
        _cast_int_to_float (bool): Indicates whether integer inputs should be cast to float.

    Methods:
        freeze(): Freezes the VecNorm, preventing updates to statistics.
        unfreeze(): Unfreezes the VecNorm, allowing updates to statistics.
        frozen_copy(): Returns a frozen copy of the VecNorm.
        clone(): Returns a clone of the VecNorm.
        transform_observation_spec(observation_spec): Transforms the observation specification.
        transform_reward_spec(reward_spec, observation_spec): Transforms the reward specification.
        transform_output_spec(output_spec): Transforms the output specification.
        to_observation_norm(): Converts the VecNorm to an ObservationNorm transform.
        set_extra_state(state): Sets the extra state for the VecNorm.
        get_extra_state(): Gets the extra state of the VecNorm.
        loc: Returns the location (mean) for normalization.
        scale: Returns the scale (standard deviation) for normalization.
        standard_normal: Indicates whether the normalization follows the standard normal distribution.

    State Dict Behavior:

        - In stateful mode, `state_dict` returns a dictionary containing the current `loc`, `var`, and `count`.
          These can be used to share the tensors across processes (this method is automatically triggered by
          :class:`~torchrl.envs.VecNorm` to share the VecNorm states across processes).
        - In stateless mode, `state_dict` returns an empty dictionary as no internal state is maintained.

    Load State Dict Behavior:

        - In stateful mode, `load_state_dict` updates the internal `loc`, `var`, and `count` with the provided state.
        - In stateless mode, `load_state_dict` does not modify any internal state as there is none to update.

    .. seealso:: :class:`~torchrl.envs.transforms.VecNorm` for the first version of this transform.

    """

    # TODO:
    # - test 2 different vecnorms, one for reward one for obs and that they don't collide
    # - test that collision is spotted
    # - customize the vecnorm keys in stateless
    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey] | None = None,
        *,
        lock: mp.Lock = None,
        stateful: bool = True,
        decay: float = 0.9999,
        eps: float = 1e-4,
        shapes: list[torch.Size] = None,
        shared_data: TensorDictBase | None = None,
    ) -> None:
        self.stateful = stateful
        if lock is None:
            lock = mp.Lock()
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps
        self.frozen = False
        self._cast_int_to_float = False
        if self.stateful:
            self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
            if shared_data:
                self._loc = shared_data["loc"]
                self._var = shared_data["var"]
                self._count = shared_data["count"]
            else:
                self._loc = None
                self._var = None
                self._count = None
        else:
            self.initialized = False
            if shared_data:
                # FIXME
                raise NotImplementedError

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        in_keys = self._in_keys
        if not self.stateful:
            in_keys = in_keys + [
                f"{self.prefix}_count",
                f"{self.prefix}_loc",
                f"{self.prefix}_var",
            ]
        return in_keys

    @in_keys.setter
    def in_keys(self, in_keys: Sequence[NestedKey]):
        self._in_keys = in_keys

    def set_container(self, container: Transform | EnvBase) -> None:
        super().set_container(container)
        if self.stateful:
            parent = getattr(self, "parent", None)
            if parent is not None and isinstance(parent, EnvBase):
                if not parent.batch_locked:
                    warnings.warn(
                        f"Support of {type(self).__name__} for unbatched container is experimental and subject to change."
                    )
                if parent.batch_size:
                    warnings.warn(
                        f"Support of {type(self).__name__} for containers with non-empty batch-size is experimental and subject to change."
                    )
                # init
                data = parent.fake_tensordict().get("next")
                self._maybe_stateful_init(data)
        else:
            parent = getattr(self, "parent", None)
            if parent is not None and isinstance(parent, EnvBase):
                self._make_prefix(parent.output_spec)

    def freeze(self) -> VecNormV2:
        """Freezes the VecNorm, avoiding the stats to be updated when called.

        See :meth:`~.unfreeze`.
        """
        self.frozen = True
        return self

    def unfreeze(self) -> VecNormV2:
        """Unfreezes the VecNorm.

        See :meth:`~.freeze`.
        """
        self.frozen = False
        return self

    def frozen_copy(self):
        """Returns a copy of the Transform that keeps track of the stats but does not update them."""
        if not self.stateful:
            raise RuntimeError("Cannot create a frozen copy of a statelss VecNorm.")
        if self._loc is None:
            raise RuntimeError(
                "Make sure the VecNorm has been initialized before creating a frozen copy."
            )
        clone = self.clone()
        if self.stateful:
            # replace values
            clone._var = self._var.clone()
            clone._loc = self._loc.clone()
            clone._count = self._count.clone()
        # freeze
        return clone.freeze()

    def clone(self) -> VecNormV2:
        other = super().clone()
        if self.stateful:
            delattr(other, "initialized")
            other.register_buffer("initialized", self.initialized.clone())
            if self._loc is not None:
                other.initialized.fill_(True)
                other._loc = self._loc.clone()
                other._var = self._var.clone()
                other._count = self._count.clone()
        return other

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: remove this decorator when trackers are in data
        with _set_missing_tolerance(self, True):
            return self._step(tensordict_reset, tensordict_reset)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()
        try:
            if self.stateful:
                self._maybe_stateful_init(next_tensordict)
                next_tensordict_select = next_tensordict.select(
                    *self.in_keys, strict=not self.missing_tolerance
                )
                if self.missing_tolerance and next_tensordict_select.is_empty():
                    return next_tensordict
                next_tensordict_norm = self._stateful_norm(next_tensordict_select)
                self._stateful_update(next_tensordict_select)
            else:
                self._maybe_stateless_init(tensordict)
                next_tensordict_select = next_tensordict.select(
                    *self._in_keys_safe, strict=not self.missing_tolerance
                )
                if self.missing_tolerance and next_tensordict_select.is_empty():
                    return next_tensordict
                loc = tensordict[f"{self.prefix}_loc"]
                var = tensordict[f"{self.prefix}_var"]
                count = tensordict[f"{self.prefix}_count"]

                next_tensordict_norm = self._stateless_norm(
                    next_tensordict_select, loc, var, count
                )
                loc, var, count = self._stateless_update(
                    next_tensordict_select, loc, var, count
                )
                # updates have been done in-place, we're good
                next_tensordict_norm.set(f"{self.prefix}_loc", loc)
                next_tensordict_norm.set(f"{self.prefix}_var", var)
                next_tensordict_norm.set(f"{self.prefix}_count", count)

            next_tensordict.update(next_tensordict_norm)
        finally:
            if self.lock is not None:
                self.lock.release()

        return next_tensordict

    def _maybe_cast_to_float(self, data):
        if self._cast_int_to_float:
            dtype = torch.get_default_dtype()
            data = data.apply(
                lambda x: x.to(dtype) if not x.dtype.is_floating_point else x
            )
        return data

    @staticmethod
    def _maybe_make_float(x):
        if x.dtype.is_floating_point:
            return x
        return x.to(torch.get_default_dtype())

    def _maybe_stateful_init(self, data):
        if not self.initialized:
            self.initialized.copy_(True)
            #  Some keys (specifically rewards) may be missing, but we can use the
            #  specs for them
            try:
                data_select = data.select(*self._in_keys_safe, strict=True)
            except KeyError:
                data_select = self.parent.full_observation_spec.zero().update(
                    self.parent.full_reward_spec.zero()
                )
                data_select = data_select.update(data)
                data_select = data_select.select(*self._in_keys_safe, strict=True)

            # For the count, we must use a TD because some keys (eg Reward) may be missing at some steps (eg, reset)
            #  We use mean() to eliminate all dims - since it's local we don't need to expand the shape
            count = (
                torch.zeros_like(data_select, dtype=torch.float32)
                .mean()
                .to(torch.int64)
            )
            # create loc
            loc = torch.zeros_like(data_select.apply(self._maybe_make_float))
            # create var
            var = torch.zeros_like(data_select.apply(self._maybe_make_float))
            self._loc = loc
            self._var = var
            self._count = count

    @property
    def _in_keys_safe(self):
        if not self.stateful:
            return self.in_keys[:-3]
        return self.in_keys

    def _norm(self, data, loc, var):
        if self.missing_tolerance:
            loc = loc.select(*data.keys(True, True))
            var = var.select(*data.keys(True, True))
            if loc.is_empty():
                return data

        var = var - loc.pow(2)
        scale = var.sqrt().clamp_min(self.eps)

        data_update = (data - loc) / scale
        if self.out_keys[: len(self.in_keys)] != self.in_keys:
            # map names
            for in_key, out_key in _zip_strict(self._in_keys_safe, self.out_keys):
                data_update.rename_key_(in_key, out_key)
        else:
            pass
        return data_update

    def _stateful_norm(self, data):
        return self._norm(data, self._loc, self._var)

    def _stateful_update(self, data):
        if self.frozen:
            return
        if self.missing_tolerance:
            var = self._var.select(*data.keys(True, True))
            loc = self._loc.select(*data.keys(True, True))
            count = self._count.select(*data.keys(True, True))
        else:
            var = self._var
            loc = self._loc
            count = self._count
        count += 1
        data = self._maybe_cast_to_float(data)
        if self.decay < 1.0:
            bias_correction = 1 - (count * math.log(self.decay)).exp()
            bias_correction = bias_correction.apply(lambda x, y: x.to(y.dtype), data)
        else:
            bias_correction = 1
        weight = (1 - self.decay) / bias_correction
        loc.lerp_(end=data, weight=weight)
        var.lerp_(end=data.pow(2), weight=weight)

    def _maybe_stateless_init(self, data):
        if not self.initialized or f"{self.prefix}_loc" not in data.keys():
            self.initialized = True
            # select all except vecnorm
            #  Some keys (specifically rewards) may be missing, but we can use the
            #  specs for them
            try:
                data_select = data.select(*self._in_keys_safe, strict=True)
            except KeyError:
                data_select = self.parent.full_observation_spec.zero().update(
                    self.parent.full_reward_spec.zero()
                )
                data_select = data_select.update(data)
                data_select = data_select.select(*self._in_keys_safe, strict=True)

            data[f"{self.prefix}_count"] = torch.zeros_like(
                data_select, dtype=torch.int64
            )
            # create loc
            loc = torch.zeros_like(data_select.apply(self._maybe_make_float))
            # create var
            var = torch.zeros_like(data_select.apply(self._maybe_make_float))
            data[f"{self.prefix}_loc"] = loc
            data[f"{self.prefix}_var"] = var

    def _stateless_norm(self, data, loc, var, count):
        data = self._norm(data, loc, var)
        return data

    def _stateless_update(self, data, loc, var, count):
        if self.frozen:
            return loc, var, count
        count = count + 1
        data = self._maybe_cast_to_float(data)
        if self.decay < 1.0:
            bias_correction = 1 - (count * math.log(self.decay)).exp()
            bias_correction = bias_correction.apply(lambda x, y: x.to(y.dtype), data)
        else:
            bias_correction = 1
        weight = (1 - self.decay) / bias_correction
        loc = loc.lerp(end=data, weight=weight)
        var = var.lerp(end=data.pow(2), weight=weight)
        return loc, var, count

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        return self._transform_spec(observation_spec)

    def transform_reward_spec(
        self, reward_spec: Composite, observation_spec
    ) -> Composite:
        return self._transform_spec(reward_spec, observation_spec)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        # This is a copy-paste of the parent methd to ensure that we correct the reward spec properly
        output_spec = output_spec.clone()
        observation_spec = self.transform_observation_spec(
            output_spec["full_observation_spec"]
        )
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self.transform_reward_spec(
                output_spec["full_reward_spec"], observation_spec
            )
        output_spec["full_observation_spec"] = observation_spec
        if "full_done_spec" in output_spec.keys():
            output_spec["full_done_spec"] = self.transform_done_spec(
                output_spec["full_done_spec"]
            )
        output_spec_keys = [
            unravel_key(k[1:]) for k in output_spec.keys(True) if isinstance(k, tuple)
        ]
        out_keys = {unravel_key(k) for k in self.out_keys}
        in_keys = {unravel_key(k) for k in self.in_keys}
        for key in out_keys - in_keys:
            if unravel_key(key) not in output_spec_keys:
                warnings.warn(
                    f"The key '{key}' is unaccounted for by the transform (expected keys {output_spec_keys}). "
                    f"Every new entry in the tensordict resulting from a call to a transform must be "
                    f"registered in the specs for torchrl rollouts to be consistently built. "
                    f"Make sure transform_output_spec/transform_observation_spec/... is coded correctly. "
                    "This warning will trigger a KeyError in v0.9, make sure to adapt your code accordingly.",
                    category=FutureWarning,
                )
        return output_spec

    def _maybe_convert_bounded(self, in_spec):
        if isinstance(in_spec, Composite):
            return Composite(
                {
                    key: self._maybe_convert_bounded(value)
                    for key, value in in_spec.items()
                }
            )
        dtype = in_spec.dtype
        if dtype is not None and not dtype.is_floating_point:
            # we need to cast the tensor and spec to a float type
            in_spec = in_spec.clone()
            in_spec.dtype = torch.get_default_dtype()
            self._cast_int_to_float = True

        if isinstance(in_spec, Bounded):
            in_spec = Unbounded(
                shape=in_spec.shape, device=in_spec.device, dtype=in_spec.dtype
            )
        return in_spec

    @property
    def prefix(self):
        prefix = getattr(self, "_prefix", "_vecnorm")
        return prefix

    def _make_prefix(self, output_spec):
        prefix = getattr(self, "_prefix", None)
        if prefix is not None:
            return prefix
        if (
            "_vecnorm_loc" in output_spec["full_observation_spec"].keys()
            or "_vecnorm_loc" in output_spec["full_reward_spec"].keys()
        ):
            prefix = "_vecnorm" + str(uuid.uuid1())
        else:
            prefix = "_vecnorm"
        self._prefix = prefix
        return prefix

    def _proc_count_spec(self, count_spec, parent_shape=None):
        if isinstance(count_spec, Composite):
            for key, spec in count_spec.items():
                spec = self._proc_count_spec(spec, parent_shape=count_spec.shape)
                count_spec[key] = spec
            return count_spec
        if count_spec.dtype:
            count_spec = Unbounded(
                shape=count_spec.shape, dtype=torch.int64, device=count_spec.device
            )
        return count_spec

    def _transform_spec(
        self, spec: Composite, obs_spec: Composite | None = None
    ) -> Composite:
        in_specs = {}
        for in_key, out_key in zip(self._in_keys_safe, self.out_keys):
            if unravel_key(in_key) in spec.keys(True):
                in_spec = spec.get(in_key).clone()
                in_spec = self._maybe_convert_bounded(in_spec)
                spec.set(out_key, in_spec)
                in_specs[in_key] = in_spec
        if not self.stateful and in_specs:
            if obs_spec is None:
                obs_spec = spec
            loc_spec = obs_spec.get(f"{self.prefix}_loc", default=None)
            var_spec = obs_spec.get(f"{self.prefix}_var", default=None)
            count_spec = obs_spec.get(f"{self.prefix}_count", default=None)
            if loc_spec is None:
                loc_spec = Composite(shape=obs_spec.shape, device=obs_spec.device)
                var_spec = Composite(shape=obs_spec.shape, device=obs_spec.device)
                count_spec = Composite(shape=obs_spec.shape, device=obs_spec.device)
            loc_spec.update(in_specs)
            # should we clone?
            var_spec.update(in_specs)
            count_spec = count_spec.update(in_specs)
            count_spec = self._proc_count_spec(count_spec)
            obs_spec[f"{self.prefix}_loc"] = loc_spec
            obs_spec[f"{self.prefix}_var"] = var_spec
            obs_spec[f"{self.prefix}_count"] = count_spec
        return spec

    def to_observation_norm(self) -> Compose | ObservationNorm:
        if not self.stateful:
            # FIXME
            raise NotImplementedError()
        result = []

        loc, scale = self._get_loc_scale()

        for key, key_out in _zip_strict(self.in_keys, self.out_keys):
            local_result = ObservationNorm(
                loc=loc.get(key),
                scale=scale.get(key),
                standard_normal=True,
                in_keys=key,
                out_keys=key_out,
                eps=self.eps,
            )
            result += [local_result]
        if len(self.in_keys) > 1:
            return Compose(*result)
        return local_result

    def _get_loc_scale(self, loc_only: bool = False) -> tuple:
        if self.stateful:
            loc = self._loc
            if loc_only:
                return loc, None
            var = self._var
            var = var - loc.pow(2)
            scale = var.sqrt().clamp_min(self.eps)
            return loc, scale
        else:
            raise RuntimeError("_get_loc_scale() called on stateless vecnorm.")

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        _lock = state.pop("lock", None)
        if _lock is not None:
            state["lock_placeholder"] = None
        return state

    def __setstate__(self, state: dict[str, Any]):
        if "lock_placeholder" in state:
            state.pop("lock_placeholder")
            _lock = mp.Lock()
            state["lock"] = _lock
        super().__setstate__(state)

    SEP = ".-|-."

    def set_extra_state(self, state: OrderedDict) -> None:
        if not self.stateful:
            return
        if not state:
            if self._loc is None:
                # we're good, not init yet
                return
            raise RuntimeError(
                "set_extra_state() called with a void state-dict while the instance is initialized."
            )
        td = TensorDict(state).unflatten_keys(self.SEP)
        if self._loc is None and not all(v.is_shared() for v in td.values(True, True)):
            warnings.warn(
                "VecNorm wasn't initialized and the tensordict is not shared. In single "
                "process settings, this is ok, but if you need to share the statistics "
                "between workers this should require some attention. "
                "Make sure that the content of VecNorm is transmitted to the workers "
                "after calling load_state_dict and not before, as other workers "
                "may not have access to the loaded TensorDict."
            )
            td.share_memory_()
        self._loc = td["loc"]
        self._var = td["var"]
        self._count = td["count"]

    def get_extra_state(self) -> OrderedDict:
        if not self.stateful:
            return {}
        if self._loc is None:
            warnings.warn(
                "Querying state_dict on an uninitialized VecNorm transform will "
                "return a `None` value for the summary statistics. "
                "Loading such a state_dict on an initialized VecNorm will result in "
                "an error."
            )
            return {}
        td = TensorDict(
            loc=self._loc,
            var=self._var,
            count=self._count,
        )
        return td.flatten_keys(self.SEP).to_dict()

    @property
    def loc(self):
        """Returns a TensorDict with the loc to be used for an affine transform."""
        if not self.stateful:
            raise RuntimeError("loc cannot be computed with stateless vecnorm.")
        # We can't cache that value bc the summary stats could be updated by a different process
        loc, _ = self._get_loc_scale(loc_only=True)
        return loc

    @property
    def scale(self):
        """Returns a TensorDict with the scale to be used for an affine transform."""
        if not self.stateful:
            raise RuntimeError("scale cannot be computed with stateless vecnorm.")
        # We can't cache that value bc the summary stats could be updated by a different process
        _, scale = self._get_loc_scale()
        return scale

    @property
    def standard_normal(self):
        """Whether the affine transform given by `loc` and `scale` follows the standard normal equation.

        Similar to :class:`~torchrl.envs.ObservationNorm` standard_normal attribute.

        Always returns ``True``.
        """
        return True
