# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from copy import copy

from typing import OrderedDict, Sequence, Any

import torch
from tensordict import assert_close, NestedKey, TensorDict, TensorDictBase
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded

from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.transforms import Compose, ObservationNorm, Transform

from torchrl.envs.transforms.utils import _set_missing_tolerance

torch.set_default_dtype(torch.double)


class VecNormV2(Transform):

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
        if lock is None:
            lock = mp.Lock()
        if out_keys is None:
            out_keys = copy(in_keys)
        if not stateful:
            in_keys = in_keys + ["_vecnorm_count", "_vecnorm_loc", "_vecnorm_var"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps
        self.frozen = False
        self.stateful = stateful
        if self.stateful:
            if shared_data:
                self._loc = shared_data["loc"]
                self._var = shared_data["var"]
                self._count = shared_data["count"]
            else:
                self._loc = None
                self._var = None
                self._count = None
        elif shared_data:
            # FIXME
            raise NotImplementedError

    def set_container(self, container: Transform | EnvBase) -> None:
        super().set_container(container)
        if self.stateful:
            parent = getattr(self, "parent", None)
            if parent is not None and isinstance(parent, EnvBase):
                # init
                data = parent.fake_tensordict().get("next")
                self._maybe_stateful_init(data.select(*self.in_keys))

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

        if self.stateful:
            next_tensordict_select = next_tensordict.select(
                *self.in_keys, strict=not self.missing_tolerance
            )
            self._maybe_stateful_init(next_tensordict_select)
            next_tensordict_norm = self._stateful_norm(next_tensordict_select)
            self._stateful_update(next_tensordict_select)
        else:
            self._maybe_stateless_init(tensordict)
            next_tensordict_select = next_tensordict.select(
                *self._in_keys_safe, strict=not self.missing_tolerance
            )

            loc = tensordict.get("_vecnorm_loc")
            var = tensordict.get("_vecnorm_var")
            count = tensordict["_vecnorm_count"]

            next_tensordict_norm = self._stateless_norm(
                next_tensordict_select, loc, var, count
            )
            loc, var, count = self._stateless_update(
                next_tensordict_select, loc, var, count
            )
            # updates have been done in-place, we're good
            next_tensordict_norm.set("_vecnorm_loc", loc)
            next_tensordict_norm.set("_vecnorm_var", var)
            next_tensordict_norm.set("_vecnorm_count", count)

        next_tensordict.update(next_tensordict_norm)
        if self.lock is not None:
            self.lock.release()

        return next_tensordict

    def _maybe_stateful_init(self, data):
        if self._loc is None:
            self._count = torch.tensor(0, dtype=torch.int64)
            # create loc
            loc = torch.zeros_like(data)
            # create var
            var = torch.zeros_like(data)
            self._loc = loc
            self._var = var

    @property
    def _in_keys_safe(self):
        if not self.stateful:
            return self.in_keys[:-3]
        return self.in_keys

    def _norm(self, data, loc, var):
        var = var - loc.pow(2)
        scale = var.sqrt().clamp_min(self.eps)

        # print(data, loc, var)
        # data_update = data.sub(loc).div(var.sqrt().clamp_min(self.eps))
        if self.missing_tolerance:
            loc = loc.select(*data.keys(True, True))
            scale = scale.select(*data.keys(True, True))

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
        self._count += 1
        if self.decay < 1.0:
            bias_correction = 1 - self.decay**self._count
        else:
            bias_correction = 1
        weight = (1 - self.decay) / bias_correction
        if self.missing_tolerance:
            var = self._var.select(*data.keys(True, True))
            loc = self._loc.select(*data.keys(True, True))
        else:
            var = self._var
            loc = self._loc
        var.lerp_(end=data.pow(2), weight=weight)
        loc.lerp_(end=data, weight=weight)

    def _maybe_stateless_init(self, data):
        if "_vecnorm_loc" not in data:
            # select all except vecnorm
            data_select = data.select(*self._in_keys_safe)
            data["_vecnorm_count"] = torch.zeros((), dtype=torch.int64)
            # create loc
            loc = torch.zeros_like(data_select)
            # create var
            var = torch.zeros_like(data_select)
            data["_vecnorm_loc"] = loc
            data["_vecnorm_var"] = var

    def _stateless_norm(self, data, loc, var, count):
        return self._norm(data, loc, var)

    def _stateless_update(self, data, loc, var, count):
        if self.frozen:
            return loc, var, count
        count = count + 1
        if self.decay < 1.0:
            bias_correction = 1 - self.decay**count
        else:
            bias_correction = 1
        weight = (1 - self.decay) / bias_correction
        var = var.lerp(end=data.pow(2), weight=weight)
        loc = loc.lerp(end=data, weight=weight)
        return loc, var, count

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        return self._transform_spec(observation_spec)

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        return self._transform_spec(reward_spec)

    def _maybe_convert_bounded(self, in_spec):
        if isinstance(in_spec, Composite):
            return Composite(
                {
                    key: self._maybe_convert_bounded(value)
                    for key, value in in_spec.items()
                }
            )
        if isinstance(in_spec, Bounded):
            in_spec = Unbounded(
                shape=in_spec.shape, device=in_spec.device, dtype=in_spec.dtype
            )
        return in_spec

    def _transform_spec(self, spec: Composite) -> Composite:
        in_specs = {}
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in spec.keys(True, True):
                in_spec = spec.get(in_key).clone()
                in_spec = self._maybe_convert_bounded(in_spec)
                spec.set(out_key, in_spec)
                in_specs[in_key] = in_spec

        if not self.stateful and in_specs:
            loc_spec = spec.get("_vecnorm_loc", default=None)
            var_spec = spec.get("_vecnorm_var", default=None)
            if loc_spec is None:
                loc_spec = Composite(shape=spec.shape, device=spec.device)
                var_spec = Composite(shape=spec.shape, device=spec.device)
            loc_spec.update(in_specs)
            # should we clone?
            var_spec.update(in_specs)
            spec["_vecnorm_loc"] = loc_spec
            spec["_vecnorm_var"] = var_spec
            spec["_vecnorm_count"] = Unbounded(dtype=torch.int64, shape=())

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

