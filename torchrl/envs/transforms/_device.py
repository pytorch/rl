# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from collections.abc import Sequence
from copy import copy
from textwrap import indent
from typing import Any, TYPE_CHECKING

import torch

from tensordict import TensorDictBase, unravel_key
from tensordict.nn import dispatch
from tensordict.utils import _zip_strict, NestedKey

from torchrl._utils import _make_ordinal_device

from torchrl.data.tensor_specs import Composite, ContinuousBox, TensorSpec
from torchrl.envs.common import _do_nothing, EnvBase
from torchrl.envs.transforms.utils import _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import Transform

__all__ = [
    "DTypeCastTransform",
    "DeviceCastTransform",
    "DoubleToFloat",
]


class DTypeCastTransform(Transform):
    """Casts one dtype to another for selected keys.

    Depending on whether the ``in_keys`` or ``in_keys_inv`` are provided
    during construction, the class behavior will change:

      * If the keys are provided, those entries and those entries only will be
        transformed from ``dtype_in`` to ``dtype_out`` entries;
      * If the keys are not provided and the object is within an environment
        register of transforms, the input and output specs that have a dtype
        set to ``dtype_in`` will be used as in_keys_inv / in_keys respectively.
      * If the keys are not provided and the object is used without an
        environment, the ``forward`` / ``inverse`` pass will scan through the
        input tensordict for all ``dtype_in`` values and map them to a ``dtype_out``
        tensor. For large data structures, this can impact performance as this
        scanning doesn't come for free. The keys to be
        transformed will not be cached.
        Note that, in this case, the out_keys (resp.
        out_keys_inv) cannot be passed as the order on which the keys are processed
        cannot be anticipated precisely.

    Args:
        dtype_in (torch.dtype): the input dtype (from the env).
        dtype_out (torch.dtype): the output dtype (for model training).
        in_keys (sequence of NestedKey, optional): list of ``dtype_in`` keys to be converted to
            ``dtype_out`` before being exposed to external objects and functions.
        out_keys (sequence of NestedKey, optional): list of destination keys.
            Defaults to ``in_keys`` if not provided.
        in_keys_inv (sequence of NestedKey, optional): list of ``dtype_out`` keys to be converted to
            ``dtype_in`` before being passed to the contained base_env or storage.
        out_keys_inv (sequence of NestedKey, optional): list of destination keys for inverse
            transform.
            Defaults to ``in_keys_inv`` if not provided.

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double),
        ...     'not_transformed': torch.ones(1, dtype=torch.double),
        ... }, [])
        >>> transform = DTypeCastTransform(torch.double, torch.float, in_keys=["obs"])
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32
        >>> print(td.get("not_transformed").dtype)
        torch.float64

    In "automatic" mode, all float64 entries are transformed:

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double),
        ...     'not_transformed': torch.ones(1, dtype=torch.double),
        ... }, [])
        >>> transform = DTypeCastTransform(torch.double, torch.float)
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32
        >>> print(td.get("not_transformed").dtype)
        torch.float32

    The same behavior is the rule when environments are constructed without
    specifying the transform keys:

    Examples:
        >>> class MyEnv(EnvBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = Composite(obs=Unbounded((), dtype=torch.float64))
        ...         self.action_spec = Unbounded((), dtype=torch.float64)
        ...         self.reward_spec = Unbounded((1,), dtype=torch.float64)
        ...         self.done_spec = Unbounded((1,), dtype=torch.bool)
        ...     def _reset(self, data=None):
        ...         return TensorDict({"done": torch.zeros((1,), dtype=torch.bool), **self.observation_spec.rand()}, [])
        ...     def _step(self, data):
        ...         assert data["action"].dtype == torch.float64
        ...         reward = self.reward_spec.rand()
        ...         done = torch.zeros((1,), dtype=torch.bool)
        ...         obs = self.observation_spec.rand()
        ...         assert reward.dtype == torch.float64
        ...         assert obs["obs"].dtype == torch.float64
        ...         return obs.empty().set("next", obs.update({"reward": reward, "done": done}))
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        >>> env = TransformedEnv(MyEnv(), DTypeCastTransform(torch.double, torch.float))
        >>> assert env.action_spec.dtype == torch.float32
        >>> assert env.observation_spec["obs"].dtype == torch.float32
        >>> assert env.reward_spec.dtype == torch.float32, env.reward_spec.dtype
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> assert env.transform.in_keys == ["obs", "reward"]
        >>> assert env.transform.in_keys_inv == ["action"]

    """

    invertible = True

    def __init__(
        self,
        dtype_in: torch.dtype,
        dtype_out: torch.dtype,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        if in_keys is not None and in_keys_inv is None:
            in_keys_inv = []

        self.dtype_in = dtype_in
        self.dtype_out = dtype_out
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )

    @property
    def in_keys(self) -> Sequence[NestedKey] | None:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            parent = self.parent
            if parent is None:
                # in_keys=None means all entries of dtype_in will be mapped to dtype_out
                return None
            in_keys = []
            for key, spec in parent.observation_spec.items(True, True):
                if spec.dtype == self.dtype_in:
                    in_keys.append(unravel_key(key))
            for key, spec in parent.full_reward_spec.items(True, True):
                if spec.dtype == self.dtype_in:
                    in_keys.append(unravel_key(key))
            self._in_keys = in_keys
            if self.__dict__.get("_out_keys", None) is None:
                self.out_keys = copy(in_keys)
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self) -> Sequence[NestedKey] | None:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            out_keys = self._out_keys = copy(self.in_keys)
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def in_keys_inv(self) -> Sequence[NestedKey] | None:
        in_keys_inv = self.__dict__.get("_in_keys_inv", None)
        if in_keys_inv is None:
            parent = self.parent
            if parent is None:
                # in_keys_inv=None means all entries of dtype_out will be mapped to dtype_in
                return None
            in_keys_inv = []
            for key, spec in parent.full_action_spec.items(True, True):
                if spec.dtype == self.dtype_in:
                    in_keys_inv.append(unravel_key(key))
            for key, spec in parent.full_state_spec.items(True, True):
                if spec.dtype == self.dtype_in:
                    in_keys_inv.append(unravel_key(key))
            self._in_keys_inv = in_keys_inv
            if self.__dict__.get("_out_keys_inv", None) is None:
                self.out_keys_inv = copy(in_keys_inv)
        return in_keys_inv

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys_inv = value

    @property
    def out_keys_inv(self) -> Sequence[NestedKey] | None:
        out_keys_inv = self.__dict__.get("_out_keys_inv", None)
        if out_keys_inv is None:
            out_keys_inv = self._out_keys_inv = copy(self.in_keys_inv)
        return out_keys_inv

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys_inv = value

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        in_keys = self.in_keys
        out_keys = self.out_keys
        if in_keys is None:
            if out_keys is not None:
                raise ValueError(
                    "in_keys wasn't provided and couldn't be retrieved. However, "
                    "out_keys was passed to the constructor. Since the order of the "
                    "entries mapped from dtype_in to dtype_out cannot be guaranteed, "
                    "this functionality is not covered. Consider passing the in_keys "
                    "or not passing any out_keys."
                )

            def func(name, item):
                if item.dtype == self.dtype_in:
                    item = self._apply_transform(item)
                    tensordict.set(name, item)

            tensordict._fast_apply(
                func, named=True, nested_keys=True, filter_empty=True
            )
            return tensordict
        else:
            # we made sure that if in_keys is not None, out_keys is not None either
            for in_key, out_key in _zip_strict(in_keys, out_keys):
                item = self._apply_transform(tensordict.get(in_key))
                tensordict.set(out_key, item)
            return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        in_keys_inv = self.in_keys_inv
        out_keys_inv = self.out_keys_inv
        if in_keys_inv is None:
            if out_keys_inv is not None:
                raise ValueError(
                    "in_keys_inv wasn't provided and couldn't be retrieved. However, "
                    "out_keys_inv was passed to the constructor. Since the order of the "
                    "entries mapped from dtype_in to dtype_out cannot be guaranteed, "
                    "this functionality is not covered. Consider passing the in_keys_inv "
                    "or not passing any out_keys_inv."
                )
            for in_key_inv, item in list(tensordict.items(True, True)):
                if item.dtype == self.dtype_out:
                    item = self._inv_apply_transform(item)
                    tensordict.set(in_key_inv, item)
            return tensordict
        else:
            return super()._inv_call(tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(self.dtype_out)

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return state.to(self.dtype_in)

    def _transform_spec(self, spec: TensorSpec) -> None:
        if isinstance(spec, Composite):
            for key in spec:
                self._transform_spec(spec[key])
        else:
            spec = spec.clone()
            spec.dtype = self.dtype_out
            space = spec.space
            if isinstance(space, ContinuousBox):
                space.low = space.low.to(self.dtype_out)
                space.high = space.high.to(self.dtype_out)
        return spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        full_action_spec = input_spec["full_action_spec"]
        full_state_spec = input_spec["full_state_spec"]
        # if this method is called, then it must have a parent and in_keys_inv will be defined
        if self.in_keys_inv is None:
            raise NotImplementedError(
                f"Calling transform_input_spec without a parent environment isn't supported yet for {type(self)}."
            )
        for in_key_inv, out_key_inv in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            if in_key_inv in full_action_spec.keys(True):
                _spec = full_action_spec[in_key_inv]
                target = "action"
            elif in_key_inv in full_state_spec.keys(True):
                _spec = full_state_spec[in_key_inv]
                target = "state"
            else:
                raise KeyError(
                    f"Key {in_key_inv} not found in state_spec and action_spec."
                )
            if _spec.dtype != self.dtype_in:
                raise TypeError(
                    f"input_spec[{in_key_inv}].dtype is not {self.dtype_in}: {in_key_inv.dtype}"
                )
            _spec = self._transform_spec(_spec)
            if target == "action":
                full_action_spec[out_key_inv] = _spec
            elif target == "state":
                full_state_spec[out_key_inv] = _spec
            else:
                # unreachable
                raise RuntimeError
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.in_keys is None:
            raise NotImplementedError(
                f"Calling transform_reward_spec without a parent environment isn't supported yet for {type(self)}."
            )
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        for reward_key, reward_spec in list(full_reward_spec.items(True, True)):
            # find out_key that match the in_key
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                if reward_key == in_key:
                    if reward_spec.dtype != self.dtype_in:
                        raise TypeError(f"reward_spec.dtype is not {self.dtype_in}")
                    full_reward_spec[out_key] = self._transform_spec(reward_spec)
        output_spec["full_observation_spec"] = self.transform_observation_spec(
            full_observation_spec
        )
        return output_spec

    def transform_observation_spec(self, observation_spec):
        full_observation_spec = observation_spec
        for observation_key, observation_spec in list(
            full_observation_spec.items(True, True)
        ):
            # find out_key that match the in_key
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                if observation_key == in_key:
                    if observation_spec.dtype != self.dtype_in:
                        raise TypeError(
                            f"observation_spec.dtype is not {self.dtype_in}"
                        )
                    full_observation_spec[out_key] = self._transform_spec(
                        observation_spec
                    )
        return full_observation_spec

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_keys={self.out_keys}, "
            f"in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
        return s


class DoubleToFloat(DTypeCastTransform):
    """Casts one dtype to another for selected keys.

    Depending on whether the ``in_keys`` or ``in_keys_inv`` are provided
    during construction, the class behavior will change:

      * If the keys are provided, those entries and those entries only will be
        transformed from ``float64`` to ``float32`` entries;
      * If the keys are not provided and the object is within an environment
        register of transforms, the input and output specs that have a dtype
        set to ``float64`` will be used as in_keys_inv / in_keys respectively.
      * If the keys are not provided and the object is used without an
        environment, the ``forward`` / ``inverse`` pass will scan through the
        input tensordict for all float64 values and map them to a float32
        tensor. For large data structures, this can impact performance as this
        scanning doesn't come for free. The keys to be
        transformed will not be cached.
        Note that, in this case, the out_keys (resp.
        out_keys_inv) cannot be passed as the order on which the keys are processed
        cannot be anticipated precisely.

    Args:
        in_keys (sequence of NestedKey, optional): list of double keys to be converted to
            float before being exposed to external objects and functions.
        out_keys (sequence of NestedKey, optional): list of destination keys.
            Defaults to ``in_keys`` if not provided.
        in_keys_inv (sequence of NestedKey, optional): list of float keys to be converted to
            double before being passed to the contained base_env or storage.
        out_keys_inv (sequence of NestedKey, optional): list of destination keys for inverse
            transform.
            Defaults to ``in_keys_inv`` if not provided.

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double),
        ...     'not_transformed': torch.ones(1, dtype=torch.double),
        ... }, [])
        >>> transform = DoubleToFloat(in_keys=["obs"])
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32
        >>> print(td.get("not_transformed").dtype)
        torch.float64

    In "automatic" mode, all float64 entries are transformed:

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double),
        ...     'not_transformed': torch.ones(1, dtype=torch.double),
        ... }, [])
        >>> transform = DoubleToFloat()
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32
        >>> print(td.get("not_transformed").dtype)
        torch.float32

    The same behavior is the rule when environments are constructed without
    specifying the transform keys:

    Examples:
        >>> class MyEnv(EnvBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = Composite(obs=Unbounded((), dtype=torch.float64))
        ...         self.action_spec = Unbounded((), dtype=torch.float64)
        ...         self.reward_spec = Unbounded((1,), dtype=torch.float64)
        ...         self.done_spec = Unbounded((1,), dtype=torch.bool)
        ...     def _reset(self, data=None):
        ...         return TensorDict({"done": torch.zeros((1,), dtype=torch.bool), **self.observation_spec.rand()}, [])
        ...     def _step(self, data):
        ...         assert data["action"].dtype == torch.float64
        ...         reward = self.reward_spec.rand()
        ...         done = torch.zeros((1,), dtype=torch.bool)
        ...         obs = self.observation_spec.rand()
        ...         assert reward.dtype == torch.float64
        ...         assert obs["obs"].dtype == torch.float64
        ...         return obs.empty().set("next", obs.update({"reward": reward, "done": done}))
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        >>> env = TransformedEnv(MyEnv(), DoubleToFloat())
        >>> assert env.action_spec.dtype == torch.float32
        >>> assert env.observation_spec["obs"].dtype == torch.float32
        >>> assert env.reward_spec.dtype == torch.float32, env.reward_spec.dtype
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> assert env.transform.in_keys == ["obs", "reward"]
        >>> assert env.transform.in_keys_inv == ["action"]

    """

    invertible = True

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__(
            dtype_in=torch.double,
            dtype_out=torch.float,
            in_keys=in_keys,
            in_keys_inv=in_keys_inv,
            out_keys=out_keys,
            out_keys_inv=out_keys_inv,
        )


class DeviceCastTransform(Transform):
    """Moves data from one device to another.

    Args:
        device (torch.device or equivalent): the destination device (outside the environment or buffer).
        orig_device (torch.device or equivalent): the origin device (inside the environment or buffer).
            If not specified and a parent environment exists, it it retrieved from it. In all other cases,
            it remains unspecified.

    Keyword Args:
        in_keys (list of NestedKey): the list of entries to map to a different device.
            Defaults to ``None``.
        out_keys (list of NestedKey): the output names of the entries mapped onto a device.
            Defaults to the values of ``in_keys``.
        in_keys_inv (list of NestedKey): the list of entries to map to a different device.
            ``in_keys_inv`` are the names expected by the base environment.
            Defaults to ``None``.
        out_keys_inv (list of NestedKey): the output names of the entries mapped onto a device.
            ``out_keys_inv`` are the names of the keys as seen from outside the transformed env.
            Defaults to the values of ``in_keys_inv``.


    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double),
        ... }, [], device="cpu:0")
        >>> transform = DeviceCastTransform(device=torch.device("cpu:2"))
        >>> td = transform(td)
        >>> print(td.device)
        cpu:2

    """

    invertible = True

    def __init__(
        self,
        device,
        orig_device=None,
        *,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
    ):
        device = self.device = _make_ordinal_device(torch.device(device))
        self.orig_device = (
            torch.device(orig_device) if orig_device is not None else orig_device
        )
        if out_keys is None:
            out_keys = copy(in_keys)
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self._map_env_device = not self.in_keys and not self.in_keys_inv

        self._rename_keys = self.in_keys != self.out_keys
        self._rename_keys_inv = self.in_keys_inv != self.out_keys_inv

        if device.type != "cuda":
            if torch.cuda.is_available():
                self._sync_device = torch.cuda.synchronize
            elif torch.backends.mps.is_available():
                self._sync_device = torch.mps.synchronize
            elif device.type == "cpu":
                self._sync_device = _do_nothing
        else:
            self._sync_device = _do_nothing

    def set_container(self, container: Transform | EnvBase) -> None:
        if self.orig_device is None:
            if isinstance(container, EnvBase):
                device = container.device
            else:
                parent = container.parent
                if parent is not None:
                    device = parent.device
                else:
                    device = torch.device("cpu")
            self.orig_device = device
        return super().set_container(container)

    def _to(self, name, tensor):
        if name in self.in_keys:
            return tensor.to(self.device, non_blocking=True)
        return tensor

    def _to_inv(self, name, tensor, device):
        if name in self.in_keys_inv:
            return tensor.to(device, non_blocking=True)
        return tensor

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._map_env_device:
            result = tensordict.to(self.device, non_blocking=True)
            self._sync_device()
            return result
        tensordict_t = tensordict.named_apply(self._to, nested_keys=True, device=None)
        if self._rename_keys:
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                if out_key != in_key:
                    tensordict_t.rename_key_(in_key, out_key)
                    tensordict_t.set(in_key, tensordict.get(in_key))
        self._sync_device()
        return tensordict_t

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self._map_env_device:
            result = next_tensordict.to(self.device, non_blocking=True)
            self._sync_device()
            return result
        tensordict_t = next_tensordict.named_apply(
            self._to, nested_keys=True, device=None
        )
        if self._rename_keys:
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                if out_key != in_key:
                    tensordict_t.rename_key_(in_key, out_key)
                    tensordict_t.set(in_key, next_tensordict.get(in_key))
        self._sync_device()
        return tensordict_t

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        parent = self.parent
        device = self.orig_device if parent is None else parent.device
        if device is None:
            return tensordict
        if self._map_env_device:
            result = tensordict.to(device, non_blocking=True)
            self._sync_orig_device()
            return result
        tensordict_t = tensordict.named_apply(
            functools.partial(self._to_inv, device=device),
            nested_keys=True,
            device=None,
        )
        if self._rename_keys_inv:
            for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
                if out_key != in_key:
                    tensordict_t.rename_key_(in_key, out_key)
                    tensordict_t.set(in_key, tensordict.get(in_key))
        self._sync_orig_device()
        return tensordict_t

    @property
    def _sync_orig_device(self):
        sync_func = self.__dict__.get("_sync_orig_device_val", None)
        if sync_func is None:
            parent = self.parent
            device = self.orig_device if parent is None else parent.device
            if device.type != "cuda":
                if torch.cuda.is_available():
                    self._sync_orig_device_val = torch.cuda.synchronize
                elif torch.backends.mps.is_available():
                    self._sync_orig_device_val = torch.mps.synchronize
                elif device.type == "cpu":
                    self._sync_orig_device_val = _do_nothing
            else:
                self._sync_orig_device_val = _do_nothing
            return self._sync_orig_device
        return sync_func

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if self._map_env_device:
            return input_spec.to(self.device)
        else:
            input_spec.clear_device_()
            return super().transform_input_spec(input_spec)

    def transform_action_spec(self, full_action_spec: Composite) -> Composite:
        full_action_spec = full_action_spec.clear_device_()
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            local_action_spec = full_action_spec.get(in_key, None)
            if local_action_spec is not None:
                full_action_spec[out_key] = local_action_spec.to(self.device)
        return full_action_spec

    def transform_state_spec(self, full_state_spec: Composite) -> Composite:
        full_state_spec = full_state_spec.clear_device_()
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            local_state_spec = full_state_spec.get(in_key, None)
            if local_state_spec is not None:
                full_state_spec[out_key] = local_state_spec.to(self.device)
        return full_state_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self._map_env_device:
            return output_spec.to(self.device)
        else:
            output_spec.clear_device_()
            return super().transform_output_spec(output_spec)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clear_device_()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            local_obs_spec = observation_spec.get(in_key, None)
            if local_obs_spec is not None:
                observation_spec[out_key] = local_obs_spec.to(self.device)
        return observation_spec

    def transform_done_spec(self, full_done_spec: Composite) -> Composite:
        full_done_spec = full_done_spec.clear_device_()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            local_done_spec = full_done_spec.get(in_key, None)
            if local_done_spec is not None:
                full_done_spec[out_key] = local_done_spec.to(self.device)
        return full_done_spec

    def transform_reward_spec(self, full_reward_spec: Composite) -> Composite:
        full_reward_spec = full_reward_spec.clear_device_()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            local_reward_spec = full_reward_spec.get(in_key, None)
            if local_reward_spec is not None:
                full_reward_spec[out_key] = local_reward_spec.to(self.device)
        return full_reward_spec

    def transform_env_device(self, device):
        if self._map_env_device:
            return self.device
        # In all other cases the device is not defined
        return None

    def __repr__(self) -> str:
        if self._map_env_device:
            return f"{self.__class__.__name__}(device={self.device}, orig_device={self.orig_device})"
        device = indent(4 * " ", f"device={self.device}")
        orig_device = indent(4 * " ", f"orig_device={self.orig_device}")
        in_keys = indent(4 * " ", f"in_keys={self.in_keys}")
        out_keys = indent(4 * " ", f"out_keys={self.out_keys}")
        in_keys_inv = indent(4 * " ", f"in_keys_inv={self.in_keys_inv}")
        out_keys_inv = indent(4 * " ", f"out_keys_inv={self.out_keys_inv}")
        return f"{self.__class__.__name__}(\n{device},\n{orig_device},\n{in_keys},\n{out_keys},\n{in_keys_inv},\n{out_keys_inv})"
