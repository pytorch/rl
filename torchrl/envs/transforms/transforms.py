# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc

import functools
import hashlib
import importlib.util
import multiprocessing as mp
import time
import warnings
import weakref
from copy import copy
from enum import IntEnum
from functools import wraps
from textwrap import indent
from typing import Any, Callable, Mapping, OrderedDict, Sequence, TypeVar, Union

import numpy as np

import torch

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    set_lazy_legacy,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict.base import _is_leaf_nontensor
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    expand_as_right,
    expand_right,
    NestedKey,
)
from torch import nn, Tensor
from torch.utils._pytree import tree_map

from torchrl._utils import (
    _append_last,
    _ends_with,
    _make_ordinal_device,
    _replace_last,
    auto_unwrap_transformed_env,
    logger as torchrl_logger,
)

from torchrl.data.tensor_specs import (
    Binary,
    Bounded,
    BoundedContinuous,
    Categorical,
    Composite,
    ContinuousBox,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    TensorSpec,
    Unbounded,
    UnboundedContinuous,
)
from torchrl.envs.common import (
    _do_nothing,
    _EnvPostInit,
    _maybe_unlock,
    EnvBase,
    make_tensordict,
)
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import (
    _sort_keys,
    _terminated_or_truncated,
    _update_during_reset,
    make_composite_from_td,
    step_mdp,
)

_has_tv = importlib.util.find_spec("torchvision", None) is not None

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."

T = TypeVar("T", bound="Transform")


def _apply_to_composite(function):
    @wraps(function)
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, Composite):
            _specs = observation_spec._specs
            in_keys = self.in_keys
            out_keys = self.out_keys
            for in_key, out_key in _zip_strict(in_keys, out_keys):
                if in_key in observation_spec.keys(True, True):
                    _specs[out_key] = function(self, observation_spec[in_key].clone())
            return Composite(
                _specs, shape=observation_spec.shape, device=observation_spec.device
            )
        else:
            return function(self, observation_spec)

    return new_fun


def _apply_to_composite_inv(function):
    # Changes the input_spec following a transform function.
    # The usage is: if an env expects a certain input (e.g. a double tensor)
    # but the input has to be transformed (e.g. it is float), this function will
    # modify the spec to get a spec that from the outside matches what is given
    # (ie a float).
    # Now since EnvBase.step ignores new inputs (ie the root level of the
    # tensor is not updated) an out_key that does not match the in_key has
    # no effect on the spec.
    @wraps(function)
    def new_fun(self, input_spec):
        if "full_action_spec" in input_spec.keys():
            skip = False
            action_spec = input_spec["full_action_spec"].clone()
            state_spec = input_spec["full_state_spec"]
            if state_spec is None:
                state_spec = Composite(shape=input_spec.shape, device=input_spec.device)
            else:
                state_spec = state_spec.clone()
        else:
            skip = True
            # In case we pass full_action_spec or full_state_spec directly
            action_spec = state_spec = Composite()
        in_keys_inv = self.in_keys_inv
        out_keys_inv = self.out_keys_inv
        for in_key, out_key in _zip_strict(in_keys_inv, out_keys_inv):
            in_key = unravel_key(in_key)
            out_key = unravel_key(out_key)
            # if in_key != out_key:
            #     # we only change the input spec if the key is the same
            #     continue
            if in_key in action_spec.keys(True, True):
                action_spec[out_key] = function(self, action_spec[in_key].clone())
                if in_key != out_key:
                    del action_spec[in_key]
            elif in_key in state_spec.keys(True, True):
                state_spec[out_key] = function(self, state_spec[in_key].clone())
                if in_key != out_key:
                    del state_spec[in_key]
            elif in_key in input_spec.keys(False, True):
                input_spec[out_key] = function(self, input_spec[in_key].clone())
                if in_key != out_key:
                    del input_spec[in_key]
        if skip:
            return input_spec
        return Composite(
            full_state_spec=state_spec,
            full_action_spec=action_spec,
            shape=input_spec.shape,
            device=input_spec.device,
        )

    return new_fun


class Transform(nn.Module):
    """Base class for environment transforms, which modify or create new data in a tensordict.

    Transforms are used to manipulate the input and output data of an environment. They can be used to preprocess
    observations, modify rewards, or transform actions. Transforms can be composed together to create more complex
    transformations.

    A transform receives a tensordict as input and returns (the same or another) tensordict as output, where a series
    of values have been modified or created with a new key.

    Attributes:
        parent: The parent environment of the transform.
        container: The container that holds the transform.
        in_keys: The keys of the input tensordict that the transform will read from.
        out_keys: The keys of the output tensordict that the transform will write to.

    .. seealso:: :ref:`TorchRL transforms <transforms>`.

    Subclassing `Transform`:

        There are various ways of subclassing a transform. The things to take into considerations are:

        - Is the transform identical for each tensor / item being transformed? Use
          :meth:`~torchrl.envs.Transform._apply_transform` and :meth:`~torchrl.envs.Transform._inv_apply_transform`.
        - The transform needs access to the input data to env.step as well as output? Rewrite
          :meth:`~torchrl.envs.Transform._step`.
          Otherwise, rewrite :meth:`~torchrl.envs.Transform._call` (or :meth:`~torchrl.envs.Transform._inv_call`).
        - Is the transform to be used within a replay buffer? Overwrite :meth:`~torchrl.envs.Transform.forward`,
          :meth:`~torchrl.envs.Transform.inv`, :meth:`~torchrl.envs.Transform._apply_transform` or
          :meth:`~torchrl.envs.Transform._inv_apply_transform`.
        - Within a transform, you can access (and make calls to) the parent environment using
          :attr:`~torchrl.envs.Transform.parent` (the base env + all transforms till this one) or
          :meth:`~torchrl.envs.Transform.container` (The object that encapsulates the transform).
        - Don't forget to edits the specs if needed: top level: :meth:`~torchrl.envs.Transform.transform_output_spec`,
          :meth:`~torchrl.envs.Transform.transform_input_spec`.
          Leaf level: :meth:`~torchrl.envs.Transform.transform_observation_spec`,
          :meth:`~torchrl.envs.Transform.transform_action_spec`, :meth:`~torchrl.envs.Transform.transform_state_spec`,
          :meth:`~torchrl.envs.Transform.transform_reward_spec` and
          :meth:`~torchrl.envs.Transform.transform_reward_spec`.

        For practical examples, see the methods listed above.

    Methods:
        clone: creates a copy of the tensordict, without parent (a transform object can only have one parent).
        set_container: Sets the container for the transform, and in turn the parent if the container is or has one
            an environment within.
        reset_parent: resets the parent and container caches.

    """

    invertible = False
    enable_inv_on_reset = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.in_keys_inv = in_keys_inv
        self.out_keys_inv = out_keys_inv
        self._missing_tolerance = False
        # we use __dict__ to avoid having nn.Module placing these objects in the module list
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def close(self):
        """Close the transform."""

    @property
    def in_keys(self):
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            return []
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self):
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            return []
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def in_keys_inv(self):
        in_keys_inv = self.__dict__.get("_in_keys_inv", None)
        if in_keys_inv is None:
            return []
        return in_keys_inv

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys_inv = value

    @property
    def out_keys_inv(self):
        out_keys_inv = self.__dict__.get("_out_keys_inv", None)
        if out_keys_inv is None:
            return []
        return out_keys_inv

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys_inv = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets a transform if it is stateful."""
        return tensordict_reset

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Inverts the input to :meth:`TransformedEnv._reset`, if needed."""
        if self.enable_inv_on_reset and tensordict is not None:
            with _set_missing_tolerance(self, True):
                tensordict = self._inv_call(tensordict)
        return tensordict

    def init(self, tensordict) -> None:
        """Runs init steps for the transform."""

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor or a leaf.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform) for each entry in ``self.in_keys``
        after the `TransformedEnv().base_env.step` is undertaken.

        Examples:
            >>> class AddOneToObs(Transform):
            ...     '''A transform that adds 1 to the observation tensor.'''
            ...     def __init__(self):
            ...         super().__init__(in_keys=["observation"], out_keys=["observation"])
            ...
            ...     def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
            ...         return obs + 1

        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._apply_transform is not coded. If the transform is coded in "
            "transform._call, make sure that this method is called instead of"
            "transform.forward, which is reserved for usage inside nn.Modules"
            "or appended to a replay buffer."
        )

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """The parent method of a transform during the ``env.step`` execution.

        This method should be overwritten whenever the :meth:`_step` needs to be
        adapted. Unlike :meth:`_call`, it is assumed that :meth:`_step`
        will execute some operation with the parent env or that it requires
        access to the content of the tensordict at time ``t`` and not only
        ``t+1`` (the ``"next"`` entry in the input tensordict).

        :meth:`_step` will only be called by :meth:`TransformedEnv.step` and
        not by :meth:`TransformedEnv.reset`.

        Args:
            tensordict (TensorDictBase): data at time t
            next_tensordict (TensorDictBase): data at time t+1

        Returns: the data at t+1

        Examples:
            >>> class AddActionToObservation(Transform):
            ...     '''A transform that adds the action to the observation tensor.'''
            ...     def _step(
            ...         self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
            ...     ) -> TensorDictBase:
            ...         # This can only be done if we have access to the 'root' tensordict
            ...         next_tensordict["observation"] += tensordict["action"]
            ...         return next_tensordict

        """
        next_tensordict = self._call(next_tensordict)
        return next_tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform.

        ``_call`` can be re-written whenever a modification of the output of env.step needs to be modified independently
        of the data collected in the previous step (including actions and states).

        For any operation that relates exclusively to the parent env (e.g. ``FrameSkip``),
        modify the :meth:`~torchrl.envs.Transform._step` method instead.
        :meth:`_call` should only be overwritten if a modification of the input tensordict is needed.

        :meth:`_call` will be called by :meth:`~torchrl.envs.TransformedEnv.step` and
        :meth:`~torchrl.envs.TransformedEnv.reset` but not during :meth:`~torchrl.envs.Transform.forward`.

        """
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            value = next_tensordict.get(in_key, default=None)
            if value is not None:
                observation = self._apply_transform(value)
                next_tensordict.set(
                    out_key,
                    observation,
                )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {next_tensordict}"
                )
        return next_tensordict

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform.

        By default, this method:

        - calls directly :meth:`~torchrl.envs.Transform._apply_transform`.
        - does not call :meth:`~torchrl.envs.Transform._step` or :meth:`~torchrl.envs.Transform._call`.

        This method is not called within `env.step` at any point. However, is is called within
        :meth:`~torchrl.data.ReplayBuffer.sample`.

        .. note:: ``forward`` also works with regular keyword arguments using :class:`~tensordict.nn.dispatch` to cast the args
            names to the keys.

        Examples:
            >>> class TransformThatMeasuresBytes(Transform):
            ...     '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
            ...     def __init__(self):
            ...         super().__init__(in_keys=[], out_keys=["bytes"])
            ...
            ...     def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
            ...         bytes_in_td = tensordict.bytes()
            ...         tensordict["bytes"] = bytes
            ...         return tensordict
            >>> t = TransformThatMeasuresBytes()
            >>> env = env.append_transform(t) # works within envs
            >>> t(TensorDict(a=0))  # Works offline too.

        """
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            data = tensordict.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                tensordict.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        """Applies the inverse transform to a tensor or a leaf.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform) for each entry in ``self.in_keys_inv``
        before the `TransformedEnv().base_env.step` is undertaken.

        Examples:
            >>> class AddOneToAction(Transform):
            ...     '''A transform that adds 1 to the action tensor.'''
            ...     def __init__(self):
            ...         super().__init__(in_keys=[], out_keys=[], in_keys_inv=["action"], out_keys_inv=["action"])
            ...
            ...     def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
            ...         return action + 1

        """
        if self.invertible:
            raise NotImplementedError
        else:
            return state

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads and possibly modify the input tensordict before it is passed to :meth:`~torchrl.envs.EnvBase.step`.

        Examples:
            >>> class AddOneToAllTensorDictBeforeStep(Transform):
            ...     '''Adds 1 to the whole content of the input to the env before the step is taken.'''
            ...
            ...     def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
            ...         return tensordict + 1

        """
        if not self.in_keys_inv:
            return tensordict
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            data = tensordict.get(out_key, None)
            if data is not None:
                item = self._inv_apply_transform(data)
                tensordict.set(in_key, item)
            elif not self.missing_tolerance:
                raise KeyError(f"'{out_key}' not found in tensordict {tensordict}")
        return tensordict

    @dispatch(source="in_keys_inv", dest="out_keys_inv")
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the inverse transform.

        By default, this method:

        - calls directly :meth:`~torchrl.envs.Transform._inv_apply_transform`.
        - does not call :meth:`~torchrl.envs.Transform._inv_call`.

        .. note:: ``inv`` also works with regular keyword arguments using :class:`~tensordict.nn.dispatch` to cast the args
            names to the keys.

        .. note:: ``inv`` is called by :meth:`~torchrl.data.ReplayBuffer.extend`.

        """

        def clone(data):
            try:
                # we privilege speed for tensordicts
                return data.clone(recurse=False)
            except AttributeError:
                return tree_map(lambda x: x, data)
            except TypeError:
                return tree_map(lambda x: x, data)

        out = self._inv_call(clone(tensordict))
        return out

    def transform_env_device(self, device: torch.device):
        """Transforms the device of the parent env."""
        return device

    def transform_env_batch_size(self, batch_size: torch.Size):
        """Transforms the batch-size of the parent env."""
        return batch_size

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        """Transforms the output spec such that the resulting spec matches transform mapping.

        This method should generally be left untouched. Changes should be implemented using
        :meth:`transform_observation_spec`, :meth:`transform_reward_spec` and :meth:`transform_full_done_spec`.
        Args:
            output_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        output_spec = output_spec.clone()
        output_spec["full_observation_spec"] = self.transform_observation_spec(
            output_spec["full_observation_spec"]
        )
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self.transform_reward_spec(
                output_spec["full_reward_spec"]
            )
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

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        """Transforms the input spec such that the resulting spec matches transform mapping.

        Args:
            input_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        input_spec = input_spec.clone()
        input_spec["full_state_spec"] = self.transform_state_spec(
            input_spec["full_state_spec"]
        )
        input_spec["full_action_spec"] = self.transform_action_spec(
            input_spec["full_action_spec"]
        )
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec such that the resulting spec matches transform mapping.

        Args:
            observation_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        """Transforms the reward spec such that the resulting spec matches transform mapping.

        Args:
            reward_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return reward_spec

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        """Transforms the done spec such that the resulting spec matches transform mapping.

        Args:
            done_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return done_spec

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        """Transforms the action spec such that the resulting spec matches transform mapping.

        Args:
            action_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return action_spec

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        """Transforms the state spec such that the resulting spec matches transform mapping.

        Args:
            state_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return state_spec

    def dump(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.in_keys})"

    def set_container(self, container: Transform | EnvBase) -> None:
        if self.parent is not None:
            raise AttributeError(
                f"parent of transform {type(self)} already set. "
                "Call `transform.clone()` to get a similar transform with no parent set."
            )
        self.__dict__["_container"] = (
            weakref.ref(container) if container is not None else None
        )
        self.__dict__["_parent"] = None

    def reset_parent(self) -> None:
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def clone(self) -> T:
        self_copy = copy(self)
        state = copy(self.__dict__)
        # modules, params, buffers
        buffers = state.pop("_buffers")
        modules = state.pop("_modules")
        parameters = state.pop("_parameters")
        state["_parameters"] = copy(parameters)
        state["_modules"] = copy(modules)
        state["_buffers"] = copy(buffers)

        state["_container"] = None
        state["_parent"] = None
        self_copy.__dict__.update(state)
        return self_copy

    @property
    def container(self):
        """Returns the env containing the transform.

        Examples:
            >>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
            >>> env.transform[0].container is env
            True
        """
        if "_container" not in self.__dict__:
            raise AttributeError("transform parent uninitialized")
        container_weakref = self.__dict__["_container"]
        if container_weakref is not None:
            container = container_weakref()
        else:
            container = container_weakref
        if container is None:
            return container
        while not isinstance(container, EnvBase):
            # if it's not an env, it should be a Compose transform
            if not isinstance(container, Compose):
                raise ValueError(
                    "A transform parent must be either another Compose transform or an environment object."
                )
            compose = container
            container_weakref = compose.__dict__.get("_container")
            if container_weakref is not None:
                # container is a weakref
                container = container_weakref()
            else:
                container = container_weakref
        return container

    def __getstate__(self):
        result = self.__dict__.copy()
        container = result["_container"]
        if container is not None:
            container = container()
        result["_container"] = container
        return result

    def __setstate__(self, state):
        state["_container"] = (
            weakref.ref(state["_container"])
            if state["_container"] is not None
            else None
        )
        self.__dict__.update(state)

    @property
    def parent(self) -> EnvBase | None:
        """Returns the parent env of the transform.

        The parent env is the env that contains all the transforms up until the current one.

        Examples:
            >>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
            >>> env.transform[1].parent
            TransformedEnv(
                env=GymEnv(env=Pendulum-v1, batch_size=torch.Size([]), device=cpu),
                transform=Compose(
                        RewardSum(keys=['reward'])))

        """
        # TODO: ideally parent should be a weakref, like container, to avoid keeping track of a parent that
        #  is de facto out of scope.
        parent = self.__dict__.get("_parent")
        if parent is None:
            if "_container" not in self.__dict__:
                raise AttributeError("transform parent uninitialized")
            container_weakref = self.__dict__["_container"]
            if container_weakref is None:
                return container_weakref
            container = container_weakref()
            if container is None:
                torchrl_logger.info(
                    "transform container out of scope. Returning None for parent."
                )
                return container
            parent = None
            if not isinstance(container, EnvBase):
                # if it's not an env, it should be a Compose transform
                if not isinstance(container, Compose):
                    raise ValueError(
                        "A transform parent must be either another Compose transform or an environment object."
                    )
                parent, _ = container._rebuild_up_to(self)
            elif isinstance(container, TransformedEnv):
                parent = TransformedEnv(container.base_env, auto_unwrap=False)
            else:
                raise ValueError(f"container is of type {type(container)}")
            self.__dict__["_parent"] = parent
        return parent

    def empty_cache(self):
        self.__dict__["_parent"] = None

    def set_missing_tolerance(self, mode=False):
        self._missing_tolerance = mode

    @property
    def missing_tolerance(self):
        return self._missing_tolerance

    def to(self, *args, **kwargs):
        # remove the parent, because it could have the wrong device associated
        self.empty_cache()
        return super().to(*args, **kwargs)


class _TEnvPostInit(_EnvPostInit):
    def __call__(self, *args, **kwargs):
        instance: EnvBase = super(_EnvPostInit, self).__call__(*args, **kwargs)
        # we skip the materialization of the specs, because this can't be done with lazy
        # transforms such as ObservationNorm.
        return instance


class TransformedEnv(EnvBase, metaclass=_TEnvPostInit):
    """A transformed_in environment.

    Args:
        env (EnvBase): original environment to be transformed_in.
        transform (Transform or callable, optional): transform to apply to the tensordict resulting
            from :obj:`env.step(td)`. If none is provided, an empty Compose
            placeholder in an eval mode is used.

            .. note:: If ``transform`` is a callable, it must receive as input a single tensordict
              and output a tensordict as well. The callable will be called at ``step``
              and ``reset`` time: if it acts on the reward (which is absent at
              reset time), a check needs to be implemented to ensure that
              the transform will run smoothly:

                >>> def add_1(data):
                ...     if "reward" in data.keys():
                ...         return data.set("reward", data.get("reward") + 1)
                ...     return data
                >>> env = TransformedEnv(base_env, add_1)

        cache_specs (bool, optional): if ``True``, the specs will be cached once
            and for all after the first call (i.e. the specs will be
            transformed_in only once). If the transform changes during
            training, the original spec transform may not be valid anymore,
            in which case this value should be set  to `False`. Default is
            `True`.

    Keyword Args:
        auto_unwrap (bool, optional): if ``True``, wrapping a transformed env in  transformed env
            unwraps the transforms of the inner TransformedEnv in the outer one (the new instance).
            Defaults to ``True``.

            .. note:: This behavior will switch to ``False`` in v0.9.

            .. seealso:: :class:`~torchrl.set_auto_unwrap_transformed_env`

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transform = RewardScaling(0.0, 1.0)
        >>> transformed_env = TransformedEnv(env, transform)
        >>> # check auto-unwrap
        >>> transformed_env = TransformedEnv(transformed_env, StepCounter())
        >>> # The inner env has been unwrapped
        >>> assert isinstance(transformed_env.base_env, GymEnv)

    """

    def __init__(
        self,
        env: EnvBase,
        transform: Transform | None = None,
        cache_specs: bool = True,
        *,
        auto_unwrap: bool | None = None,
        **kwargs,
    ):
        self._transform = None
        device = kwargs.pop("device", None)
        if device is not None:
            env = env.to(device)
        else:
            device = env.device
        super().__init__(device=None, allow_done_after_reset=None, **kwargs)

        # Type matching must be exact here, because subtyping could introduce differences in behavior that must
        # be contained within the subclass.
        if type(env) is TransformedEnv and type(self) is TransformedEnv:
            if auto_unwrap is None:
                auto_unwrap = auto_unwrap_transformed_env(allow_none=True)
                if auto_unwrap is None:
                    warnings.warn(
                        "The default behavior of TransformedEnv will change in version 0.9. "
                        "Nested TransformedEnvs will no longer be automatically unwrapped by default. "
                        "To prepare for this change, use set_auto_unwrap_transformed_env(val: bool) "
                        "as a decorator or context manager, or set the environment variable "
                        "AUTO_UNWRAP_TRANSFORMED_ENV to 'False'.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    auto_unwrap = True
        else:
            auto_unwrap = False

        if auto_unwrap:
            self._set_env(env.base_env, device)
            if type(transform) is not Compose:
                # we don't use isinstance as some transforms may be subclassed from
                # Compose but with other features that we don't want to lose.
                if not isinstance(transform, Transform):
                    if callable(transform):
                        transform = _CallableTransform(transform)
                    else:
                        raise ValueError(
                            "Invalid transform type, expected a Transform instance or a callable "
                            f"but got an object of type {type(transform)}."
                        )
                if transform is not None:
                    transform = [transform]
                else:
                    transform = []
            else:
                for t in transform:
                    t.reset_parent()
            env_transform = env.transform.clone()
            if type(env_transform) is not Compose:
                env_transform = [env_transform]
            else:
                for t in env_transform:
                    t.reset_parent()
            transform = Compose(*env_transform, *transform).to(device)
        else:
            self._set_env(env, device)
            if transform is None:
                transform = Compose()

        self.transform = transform

        self._last_obs = None
        self.cache_specs = cache_specs
        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None

    @property
    def batch_size(self) -> torch.Size:
        try:
            if self.transform is not None:
                return self.transform.transform_env_batch_size(self.base_env.batch_size)
            return self.base_env.batch_size
        except AttributeError:
            # during init, the base_env is not yet defined
            return torch.Size([])

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        raise RuntimeError(
            "Cannot modify the batch-size of a transformed env. Change the batch size of the base_env instead."
        )

    def add_truncated_keys(self) -> TransformedEnv:
        self.base_env.add_truncated_keys()
        self.empty_cache()
        return self

    def _set_env(self, env: EnvBase, device) -> None:
        if device != env.device:
            env = env.to(device)
        self.base_env = env
        # updates need not be inplace, as transforms may modify values out-place
        self.base_env._inplace_update = False

    @property
    def transform(self) -> Transform:
        return getattr(self, "_transform", None)

    @transform.setter
    def transform(self, transform: Transform):
        if not isinstance(transform, Transform):
            if callable(transform):
                transform = _CallableTransform(transform)
            else:
                raise ValueError(
                    f"""Expected a transform of type torchrl.envs.transforms.Transform or a callable,
but got an object of type {type(transform)}."""
                )
        prev_transform = getattr(self, "_transform", None)
        if prev_transform is not None:
            prev_transform.empty_cache()
            prev_transform.reset_parent()
        transform = transform.to(self.device)
        transform.set_container(self)
        transform.eval()
        self._transform = transform

    @property
    def device(self) -> bool:
        device = self.base_env.device
        if self.transform is None:
            # during init, the device is checked
            return device
        return self.transform.transform_env_device(device)

    @device.setter
    def device(self, value):
        raise RuntimeError("device is a read-only property")

    @property
    def batch_locked(self) -> bool:
        return self.base_env.batch_locked

    @batch_locked.setter
    def batch_locked(self, value):
        raise RuntimeError("batch_locked is a read-only property")

    @property
    def run_type_checks(self) -> bool:
        return self.base_env.run_type_checks

    @run_type_checks.setter
    def run_type_checks(self, value):
        raise RuntimeError(
            "run_type_checks is a read-only property for TransformedEnvs"
        )

    @property
    def _allow_done_after_reset(self) -> bool:
        return self.base_env._allow_done_after_reset

    @_allow_done_after_reset.setter
    def _allow_done_after_reset(self, value):
        if value is None:
            return
        raise RuntimeError(
            "_allow_done_after_reset is a read-only property for TransformedEnvs"
        )

    @property
    def _inplace_update(self):
        return self.base_env._inplace_update

    @property
    def output_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment."""
        if self.cache_specs:
            output_spec = self.__dict__.get("_output_spec")
            if output_spec is not None:
                return output_spec
        output_spec = self._make_output_spec()
        return output_spec

    @_maybe_unlock
    def _make_output_spec(self):
        output_spec = self.base_env.output_spec.clone()

        # remove cached key values, but not _input_spec
        super().empty_cache()
        output_spec = self.transform.transform_output_spec(output_spec)
        if self.cache_specs:
            self.__dict__["_output_spec"] = output_spec
        return output_spec

    @property
    def input_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment."""
        if self.cache_specs:
            input_spec = self.__dict__.get("_input_spec")
            if input_spec is not None:
                return input_spec
        input_spec = self._make_input_spec()
        return input_spec

    @_maybe_unlock
    def _make_input_spec(self):
        input_spec = self.base_env.input_spec.clone()

        # remove cached key values, but not _input_spec
        super().empty_cache()
        input_spec = self.transform.transform_input_spec(input_spec)
        if self.cache_specs:
            self.__dict__["_input_spec"] = input_spec
        return input_spec

    def rand_action(self, tensordict: TensorDictBase | None = None) -> TensorDict:
        if type(self.base_env).rand_action is not EnvBase.rand_action:
            # TODO: this will fail if the transform modifies the input.
            #  For instance, if an env overrides rand_action and we build a
            #  env = PendulumEnv().append_transform(ActionDiscretizer(num_intervals=4))
            #  env.rand_action will NOT have a discrete action!
            #  Getting a discrete action would require coding the inverse transform of an action within
            #  ActionDiscretizer (ie, float->int, not int->float).
            #  We can loosely check that the action_spec isn't altered - that doesn't mean the action is
            #  intact but it covers part of these alterations.
            #
            # The following check may be expensive to run and could be cached.
            if self.full_action_spec != self.base_env.full_action_spec:
                raise RuntimeError(
                    f"The rand_action method from the base env {self.base_env.__class__.__name__} "
                    "has been overwritten, but the transforms appended to the environment modify "
                    "the action. To call the base env rand_action method, we should then invert the "
                    "action transform, which is (in general) not doable. "
                    f"The full action spec of the base env is: {self.base_env.full_action_spec}, \n"
                    f"the full action spec of the transformed env is {self.full_action_spec}."
                )
            return self.base_env.rand_action(tensordict)
        return super().rand_action(tensordict)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # No need to clone here because inv does it already
        # tensordict = tensordict.clone(False)
        next_preset = tensordict.get("next", None)
        tensordict_in = self.transform.inv(tensordict)

        # It could be that the step must be skipped
        partial_steps = tensordict_in.pop("_step", None)
        next_tensordict = None
        tensordict_batch_size = None
        if partial_steps is not None:
            if not self.batch_locked:
                # Batched envs have their own way of dealing with this - batched envs that are not batched-locked may fail here
                if partial_steps.all():
                    partial_steps = None
                else:
                    tensordict_batch_size = tensordict_in.batch_size
                    partial_steps = partial_steps.view(tensordict_batch_size)
                    tensordict_in_save = tensordict_in[~partial_steps]
                    tensordict_in = tensordict_in[partial_steps]
            else:
                if not partial_steps.any():
                    next_tensordict = self._skip_tensordict(tensordict_in)
                    # No need to copy anything
                    partial_steps = None
                elif not partial_steps.all():
                    # trust that the _step can handle this!
                    tensordict_in.set("_step", partial_steps)
                    # The filling should be handled by the sub-env
                    partial_steps = None
                else:
                    partial_steps = None
            if tensordict_batch_size is None:
                tensordict_batch_size = self.batch_size

        if next_tensordict is None:
            next_tensordict = self.base_env._step(tensordict_in)
            if next_preset is not None:
                # tensordict could already have a "next" key
                # this could be done more efficiently by not excluding but just passing
                # the necessary keys
                next_tensordict.update(
                    next_preset.exclude(*next_tensordict.keys(True, True))
                )
            self.base_env._complete_done(self.base_env.full_done_spec, next_tensordict)
            # we want the input entries to remain unchanged
            next_tensordict = self.transform._step(tensordict_in, next_tensordict)

        if partial_steps is not None:
            result = next_tensordict.new_zeros(tensordict_batch_size)

            def select_and_clone(x, y):
                if y is not None:
                    if x.device == y.device:
                        return y.clone()
                    return y.to(y.device)

            if not partial_steps.all():
                result[~partial_steps] = tensordict_in_save._fast_apply(
                    select_and_clone,
                    tensordict_in_save,
                    device=result.device,
                    filter_empty=True,
                    default=None,
                    is_leaf=_is_leaf_nontensor,
                )
            if partial_steps.any():
                result[partial_steps] = next_tensordict
            next_tensordict = result
        return next_tensordict

    def set_seed(
        self, seed: int | None = None, static_seed: bool = False
    ) -> int | None:
        """Set the seeds of the environment."""
        return self.base_env.set_seed(seed, static_seed=static_seed)

    def _set_seed(self, seed: int | None) -> None:
        """This method is not used in transformed envs."""

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        if tensordict is not None:
            # We must avoid modifying the original tensordict so a shallow copy is necessary.
            # We just select the input data and reset signal, which is all we need.
            tensordict = tensordict.select(
                *self.reset_keys, *self.state_spec.keys(True, True), strict=False
            )
        # We always call _reset_env_preprocess, even if tensordict is None - that way one can augment that
        # method to do any pre-reset operation.
        # By default, within _reset_env_preprocess we will skip the inv call when tensordict is None.
        tensordict = self.transform._reset_env_preprocess(tensordict)
        tensordict_reset = self.base_env._reset(tensordict, **kwargs)
        if tensordict is None:
            # make sure all transforms see a source tensordict
            tensordict = tensordict_reset.empty()
        self.base_env._complete_done(self.base_env.full_done_spec, tensordict_reset)
        tensordict_reset = self.transform._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def _reset_proc_data(self, tensordict, tensordict_reset):
        # self._complete_done(self.full_done_spec, reset)
        self._reset_check_done(tensordict, tensordict_reset)
        if tensordict is not None:
            tensordict_reset = _update_during_reset(
                tensordict_reset, tensordict, self.reset_keys
            )
        # # we need to call `_call` as some transforms don't do the work in reset
        # # eg: CatTensor has only a _call method, no need for a reset since reset
        # # doesn't do anything special
        # mt_mode = self.transform.missing_tolerance
        # self.set_missing_tolerance(True)
        # reset = self.transform._call(reset)
        # self.set_missing_tolerance(mt_mode)
        return tensordict_reset

    def _complete_done(
        cls, done_spec: Composite, data: TensorDictBase
    ) -> TensorDictBase:
        # This step has already been completed. We assume the transform module do their job correctly.
        return data

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        state_dict = self.transform.state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        self.transform.load_state_dict(state_dict, **kwargs)

    def eval(self) -> TransformedEnv:
        if "transform" in self.__dir__():
            # when calling __init__, eval() is called but transforms are not set
            # yet.
            self.transform.eval()
        return self

    def train(self, mode: bool = True) -> TransformedEnv:
        self.transform.train(mode)
        return self

    @property
    def is_closed(self) -> bool:
        return self.base_env.is_closed

    @is_closed.setter
    def is_closed(self, value: bool):
        self.base_env.is_closed = value

    def close(self, *, raise_if_closed: bool = True):
        self.base_env.close(raise_if_closed=raise_if_closed)
        self.transform.close()
        self.is_closed = True

    def empty_cache(self):
        self.__dict__["_output_spec"] = None
        self.__dict__["_input_spec"] = None
        super().empty_cache()

    def append_transform(
        self, transform: Transform | Callable[[TensorDictBase], TensorDictBase]
    ) -> TransformedEnv:
        """Appends a transform to the env.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        self.empty_cache()
        if not isinstance(transform, Transform):
            if callable(transform):
                transform = _CallableTransform(transform)
            else:
                raise ValueError(
                    "TransformedEnv.append_transform expected a transform or a callable, "
                    f"but received an object of type {type(transform)} instead."
                )
        transform = transform.to(self.device)
        if not isinstance(self.transform, Compose):
            prev_transform = self.transform
            prev_transform.reset_parent()
            self.transform = Compose()
            self.transform.append(prev_transform)

        self.transform.append(transform)
        return self

    def insert_transform(self, index: int, transform: Transform) -> TransformedEnv:
        """Inserts a transform to the env at the desired index.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        self.empty_cache()
        if not isinstance(transform, Transform):
            if callable(transform):
                transform = _CallableTransform(transform)
            else:
                raise ValueError(
                    "TransformedEnv.insert_transform expected a transform or a callable, "
                    f"but received an object of type {type(transform)} instead."
                )
        transform = transform.to(self.device)
        if not isinstance(self.transform, Compose):
            compose = Compose(self.transform.clone())
            self.transform = compose  # parent set automatically

        self.transform.insert(index, transform)
        return self

    def __getattr__(self, attr: str) -> Any:
        try:
            return super().__getattr__(
                attr
            )  # make sure that appropriate exceptions are raised
        except AttributeError as err:
            if attr in (
                "action_spec",
                "done_spec",
                "full_action_spec",
                "full_done_spec",
                "full_observation_spec",
                "full_reward_spec",
                "full_state_spec",
                "input_spec",
                "observation_spec",
                "output_spec",
                "reward_spec",
                "state_spec",
            ):
                raise AttributeError(
                    f"Could not get {attr} because an internal error was raised. To find what this error "
                    f"is, call env.transform.transform_<placeholder>_spec(env.base_env.spec)."
                )
            if attr.startswith("__"):
                raise AttributeError(
                    "passing built-in private methods is "
                    f"not permitted with type {type(self)}. "
                    f"Got attribute {attr}."
                )
            elif "base_env" in self.__dir__():
                base_env = self.__getattr__("base_env")
                return getattr(base_env, attr)
            raise AttributeError(
                f"env not set in {self.__class__.__name__}, cannot access {attr}"
            ) from err

    def __repr__(self) -> str:
        env_str = indent(f"env={self.base_env}", 4 * " ")
        t_str = indent(f"transform={self.transform}", 4 * " ")
        return f"TransformedEnv(\n{env_str},\n{t_str})"

    def to(self, *args, **kwargs) -> TransformedEnv:
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.base_env = self.base_env.to(device)
            self._transform = self._transform.to(device)
            self.empty_cache()
        return super().to(*args, **kwargs)

    def __setattr__(self, key, value):
        propobj = getattr(self.__class__, key, None)

        if isinstance(propobj, property):
            ancestors = list(__class__.__mro__)[::-1]
            while isinstance(propobj, property):
                if propobj.fset is not None:
                    return propobj.fset(self, value)
                propobj = getattr(ancestors.pop(), key, None)
            else:
                raise AttributeError(f"can't set attribute {key}")
        else:
            return super().__setattr__(key, value)

    def __del__(self):
        # we may delete a TransformedEnv that contains an env contained by another
        # transformed env and that we don't want to close
        pass

    def set_missing_tolerance(self, mode=False):
        """Indicates if an KeyError should be raised whenever an in_key is missing from the input tensordict."""
        self.transform.set_missing_tolerance(mode)


class ObservationTransform(Transform):
    """Abstract class for transformations of the observations."""

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = [
                "observation",
                "pixels",
            ]
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )


class Compose(Transform):
    """Composes a chain of transforms.

    :class:`~torchrl.envs.transforms.Transform` or ``callable``s are accepted.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transforms = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(*transforms)
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    def __init__(self, *transforms: Transform):
        super().__init__()

        def map_transform(trsf):
            if isinstance(trsf, Transform):
                return trsf
            if callable(trsf):
                return _CallableTransform(trsf)
            raise ValueError(
                f"Transform list must contain only transforms or "
                f"callable. Got a element of type {type(trsf)}."
            )

        transforms = [map_transform(trsf) for trsf in transforms]
        self.transforms = nn.ModuleList(transforms)
        for t in transforms:
            t.set_container(self)

    def close(self):
        """Close the transform."""
        for t in self.transforms:
            t.close()

    def to(self, *args, **kwargs):
        # because Module.to(...) does not call to(...) on sub-modules, we have
        # manually call it:
        self.transforms = nn.ModuleList(
            [t.to(*args, **kwargs) for t in self.transforms]
        )
        return super().to(*args, **kwargs)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            next_tensordict = t._call(next_tensordict)
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for t in self.transforms:
            next_tensordict = t._step(tensordict, next_tensordict)
        return next_tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in reversed(self.transforms):
            tensordict = t._inv_call(tensordict)
        return tensordict

    def transform_env_device(self, device: torch.device):
        for t in self.transforms:
            device = t.transform_env_device(device)
        return device

    def transform_env_batch_size(self, batch_size: torch.batch_size):
        for t in self.transforms:
            batch_size = t.transform_env_batch_size(batch_size)
        return batch_size

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        # Input, action and state specs do NOT need to be reversed
        # although applying these specs requires them to be called backward.
        # To prove this, imagine we have 2 action transforms: t0 is an ActionDiscretizer, it maps float actions
        # from the env to int actions for the policy. We add one more transform t1 that, if a == a_action_max,
        # reduces its value by 1 (ie, the policy can sample actions from 0 to N + 1, and ActionDiscretizer
        # has top N values).
        # To apply this transform given an int action from the policy, we first call t1 to clamp the action to
        # N (from N+1), then call t0 to map it to a float.
        # We build this from TEnv(env, Compose(ActionDiscretizer, ActionClamp)) and call them starting with the
        # last then the first.
        # To know what the action spec is to the 'outside world' (ie, to the policy) we must take
        # the action spec from the env, map it using t0 then t1 (going from in to out).
        for t in self.transforms:
            input_spec = t.transform_input_spec(input_spec)
            if not isinstance(input_spec, Composite):
                raise TypeError(
                    f"Expected Compose but got {type(input_spec)} with transform {t}"
                )
        return input_spec

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        # To understand why we don't invert, look up at transform_input_spec
        for t in self.transforms:
            action_spec = t.transform_action_spec(action_spec)
            if not isinstance(action_spec, TensorSpec):
                raise TypeError(
                    f"Expected TensorSpec but got {type(action_spec)} with transform {t}"
                )
        return action_spec

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        # To understand why we don't invert, look up at transform_input_spec
        for t in self.transforms:
            state_spec = t.transform_state_spec(state_spec)
            if not isinstance(state_spec, Composite):
                raise TypeError(
                    f"Expected Compose but got {type(state_spec)} with transform {t}"
                )
        return state_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
            if not isinstance(observation_spec, TensorSpec):
                raise TypeError(
                    f"Expected TensorSpec but got {type(observation_spec)} with transform {t}"
                )
        return observation_spec

    def transform_output_spec(self, output_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            output_spec = t.transform_output_spec(output_spec)
            if not isinstance(output_spec, Composite):
                raise TypeError(
                    f"Expected Compose but got {type(output_spec)} with transform {t}"
                )
        return output_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
            if not isinstance(reward_spec, TensorSpec):
                raise TypeError(
                    f"Expected TensorSpec but got {type(reward_spec)} with transform {t}"
                )
        return reward_spec

    def __getitem__(self, item: int | slice | list) -> Union:
        transform = self.transforms
        transform = transform[item]
        if not isinstance(transform, Transform):
            out = Compose(*(t.clone() for t in self.transforms[item]))
            out.set_container(self.parent)
            return out
        return transform

    def dump(self, **kwargs) -> None:
        for t in self:
            t.dump(**kwargs)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        for t in self.transforms:
            tensordict_reset = t._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in reversed(self.transforms):
            tensordict = t._reset_env_preprocess(tensordict)
        return tensordict

    def init(self, tensordict: TensorDictBase) -> None:
        for t in self.transforms:
            t.init(tensordict)

    def append(
        self, transform: Transform | Callable[[TensorDictBase], TensorDictBase]
    ) -> None:
        """Appends a transform in the chain.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        self.empty_cache()
        if not isinstance(transform, Transform):
            if callable(transform):
                transform = _CallableTransform(transform)
            else:
                raise ValueError(
                    "Compose.append expected a transform or a callable, "
                    f"but received an object of type {type(transform)} instead."
                )
        transform.eval()
        if type(self) == type(transform) == Compose:
            for t in transform:
                self.append(t)
        else:
            self.transforms.append(transform)
        transform.set_container(self)

    def set_container(self, container: Transform | EnvBase) -> None:
        self.reset_parent()
        super().set_container(container)
        for t in self.transforms:
            t.set_container(self)

    def insert(
        self,
        index: int,
        transform: Transform | Callable[[TensorDictBase], TensorDictBase],
    ) -> None:
        """Inserts a transform in the chain at the desired index.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        if not isinstance(transform, Transform):
            if callable(transform):
                transform = _CallableTransform(transform)
            else:
                raise ValueError(
                    "Compose.append expected a transform or a callable, "
                    f"but received an object of type {type(transform)} instead."
                )

        if abs(index) > len(self.transforms):
            raise ValueError(
                f"Index expected to be between [-{len(self.transforms)}, {len(self.transforms)}] got index={index}"
            )

        # empty cache of all transforms to reset parents and specs
        self.empty_cache()
        if index < 0:
            index = index + len(self.transforms)
        transform.eval()
        self.transforms.insert(index, transform)
        transform.set_container(self)

    def __iter__(self):
        yield from self.transforms

    def __len__(self):
        return len(self.transforms)

    def __repr__(self) -> str:
        if len(self.transforms):
            layers_str = ",\n".join(
                [indent(str(trsf), 4 * " ") for trsf in self.transforms]
            )
            layers_str = f"\n{indent(layers_str, 4 * ' ')}"
        else:
            layers_str = ""
        return f"{self.__class__.__name__}({layers_str})"

    def empty_cache(self):
        for t in self.transforms:
            t.empty_cache()
        super().empty_cache()

    def reset_parent(self):
        for t in self.transforms:
            t.reset_parent()
        super().reset_parent()

    def clone(self) -> T:
        transforms = []
        for t in self.transforms:
            transforms.append(t.clone())
        return Compose(*transforms)

    def set_missing_tolerance(self, mode=False):
        for t in self.transforms:
            t.set_missing_tolerance(mode)
        super().set_missing_tolerance(mode)

    def _rebuild_up_to(self, final_transform):
        container_weakref = self.__dict__["_container"]
        if container_weakref is not None:
            container = container_weakref()
        else:
            container = container_weakref
        if isinstance(container, Compose):
            out, parent_compose = container._rebuild_up_to(self)
            if out is None:
                # returns None if there is no parent env
                return None, None
        elif isinstance(container, TransformedEnv):
            out = TransformedEnv(container.base_env, auto_unwrap=False)
        elif container is None:
            # returns None if there is no parent env
            return None, None
        else:
            raise ValueError(f"Container of type {type(container)} isn't supported.")

        if final_transform not in self.transforms:
            raise ValueError(f"Cannot rebuild with transform {final_transform}.")
        list_of_transforms = []
        for orig_trans in self.transforms:
            if orig_trans is final_transform:
                break
            transform = orig_trans.clone()
            transform.reset_parent()
            list_of_transforms.append(transform)
        if isinstance(container, Compose):
            parent_compose.append(Compose(*list_of_transforms))
            return out, parent_compose[-1]
        elif isinstance(container, TransformedEnv):
            for t in list_of_transforms:
                out.append_transform(t)
            return out, out.transform


class ToTensorImage(ObservationTransform):
    """Transforms a numpy-like image (W x H x C) to a pytorch image (C x W x H).

    Transforms an observation image from a (... x W x H x C) tensor to a
    (... x C x W x H) tensor. Optionally, scales the input tensor from the range
    [0, 255] to the range [0.0, 1.0] (see ``from_int`` for more details).

    In the other cases, tensors are returned without scaling.

    Args:
        from_int (bool, optional): if ``True``, the tensor will be scaled from
            the range [0, 255] to the range [0.0, 1.0]. if `False``, the tensor
            will not be scaled. if `None`, the tensor will be scaled if
            it's not a floating-point tensor. default=None.
        unsqueeze (bool): if ``True``, the observation tensor is unsqueezed
            along the first dimension. default=False.
        dtype (torch.dtype, optional): dtype to use for the resulting
            observations.

    Keyword arguments:
        in_keys (list of NestedKeys): keys to process.
        out_keys (list of NestedKeys): keys to write.
        shape_tolerant (bool, optional): if ``True``, the shape of the input
            images will be check. If the last channel is not `3`, the permutation
            will be ignored. Defaults to ``False``.

    Examples:
        >>> transform = ToTensorImage(in_keys=["pixels"])
        >>> ri = torch.randint(0, 255, (1 , 1, 10, 11, 3), dtype=torch.uint8)
        >>> td = TensorDict(
        ...     {"pixels": ri},
        ...     [1, 1])
        >>> _ = transform(td)
        >>> obs = td.get("pixels")
        >>> print(obs.shape, obs.dtype)
        torch.Size([1, 1, 3, 10, 11]) torch.float32
    """

    def __init__(
        self,
        from_int: bool | None = None,
        unsqueeze: bool = False,
        dtype: torch.device | None = None,
        *,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shape_tolerant: bool = False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.from_int = from_int
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.shape_tolerant = shape_tolerant

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
        if not self.shape_tolerant or observation.shape[-1] == 3:
            observation = observation.permute(
                *list(range(observation.ndimension() - 3)), -1, -3, -2
            )
        if self.from_int or (
            self.from_int is None and not torch.is_floating_point(observation)
        ):
            observation = observation.div(255)
        observation = observation.to(self.dtype)
        if self._should_unsqueeze(observation):
            observation = observation.unsqueeze(0)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._pixel_observation(observation_spec)
        dim = [1] if self._should_unsqueeze(observation_spec) else []
        if not self.shape_tolerant or observation_spec.shape[-1] == 3:
            observation_spec.shape = torch.Size(
                [
                    *dim,
                    *observation_spec.shape[:-3],
                    observation_spec.shape[-1],
                    observation_spec.shape[-3],
                    observation_spec.shape[-2],
                ]
            )
        observation_spec.dtype = self.dtype
        return observation_spec

    def _should_unsqueeze(self, observation_like: torch.FloatTensor | TensorSpec):
        if isinstance(observation_like, torch.FloatTensor):
            has_3_dimensions = observation_like.ndimension() == 3
        else:
            has_3_dimensions = len(observation_like.shape) == 3
        return has_3_dimensions and self.unsqueeze

    def _pixel_observation(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._apply_transform(spec.space.high)
            spec.space.low = self._apply_transform(spec.space.low)
        return spec


class ClipTransform(Transform):
    """A transform to clip input (state, action) or output (observation, reward) values.

    This transform can take multiple input or output keys but only one value per
    transform. If multiple clipping values are needed, several transforms should
    be appended one after the other.

    Args:
        in_keys (list of NestedKeys): input entries (read)
        out_keys (list of NestedKeys): input entries (write)
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`inv` calls.

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

    def _apply_transform(self, obs: torch.Tensor) -> None:
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


class TargetReturn(Transform):
    """Sets a target return for the agent to achieve in the environment.

    In goal-conditioned RL, the :class:`~.TargetReturn` is defined as the
    expected cumulative reward obtained from the current state to the goal state
    or the end of the episode. It is used as input for the policy to guide its behavior.
    For a trained policy typically the maximum return in the environment is
    chosen as the target return.
    However, as it is used as input to the policy module, it should be scaled
    accordingly.
    With the :class:`~.TargetReturn` transform, the tensordict can be updated
    to include the user-specified target return.
    The ``mode`` parameter can be used to specify
    whether the target return gets updated at every step by subtracting the
    reward achieved at each step or remains constant.

    Args:
        target_return (:obj:`float`): target return to be achieved by the agent.
        mode (str): mode to be used to update the target return. Can be either "reduce" or "constant". Default: "reduce".
        in_keys (sequence of NestedKey, optional): keys pointing to the reward
            entries. Defaults to the reward keys of the parent env.
        out_keys (sequence of NestedKey, optional): keys pointing to the
            target keys. Defaults to a copy of in_keys where the last element
            has been substituted by ``"target_return"``, and raises an exception
            if these keys aren't unique.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("CartPole-v1"),
        ...     TargetReturn(10.0, mode="reduce"))
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> env.rollout(20)['target_return'].squeeze()
        tensor([10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0., -1., -2., -3.])

    """

    MODES = ["reduce", "constant"]
    MODE_ERR = "Mode can only be 'reduce' or 'constant'."

    def __init__(
        self,
        target_return: float,
        mode: str = "reduce",
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reset_key: NestedKey | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(self.MODE_ERR)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.target_return = target_return
        self.mode = mode
        self.reset_key = reset_key

    @property
    def reset_key(self):
        reset_key = getattr(self, "_reset_key", None)
        if reset_key is not None:
            return reset_key
        reset_keys = self.parent.reset_keys
        if len(reset_keys) > 1:
            raise RuntimeError(
                f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
            )
        reset_key = reset_keys[0]
        self._reset_key = reset_key
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    @property
    def in_keys(self):
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            in_keys = self.parent.reward_keys
            self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value

    @property
    def out_keys(self):
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            out_keys = [
                _replace_last(in_key, "target_return") for in_key in self.in_keys
            ]
            if len(set(out_keys)) < len(out_keys):
                raise ValueError(
                    "Could not infer the target_return because multiple rewards are located at the same level."
                )
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def _reset(self, tensordict: TensorDict, tensordict_reset: TensorDictBase):
        _reset = _get_reset(self.reset_key, tensordict)
        for out_key in self.out_keys:
            target_return = tensordict.get(out_key, None)
            if target_return is None:
                target_return = torch.full(
                    size=(*tensordict.batch_size, 1),
                    fill_value=self.target_return,
                    dtype=torch.float32,
                    device=tensordict.device,
                )
            else:
                target_return = torch.where(
                    expand_as_right(~_reset, target_return),
                    target_return,
                    self.target_return,
                )
            tensordict_reset.set(
                out_key,
                target_return,
            )
        return tensordict_reset

    def _call(self, next_tensordict: TensorDict) -> TensorDict:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            val_in = next_tensordict.get(in_key, None)
            val_out = next_tensordict.get(out_key, None)
            if val_in is not None:
                target_return = self._apply_transform(
                    val_in,
                    val_out,
                )
                next_tensordict.set(out_key, target_return)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {next_tensordict}")
        return next_tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for out_key in self.out_keys:
            next_tensordict.set(out_key, tensordict.get(out_key))
        return super()._step(tensordict, next_tensordict)

    def _apply_transform(
        self, reward: torch.Tensor, target_return: torch.Tensor
    ) -> torch.Tensor:
        if target_return.shape != reward.shape:
            raise ValueError(
                f"The shape of the reward ({reward.shape}) and target return ({target_return.shape}) must match."
            )
        if self.mode == "reduce":
            target_return = target_return - reward
            return target_return
        elif self.mode == "constant":
            target_return = target_return
            return target_return
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in self.parent.full_observation_spec.keys(True):
                target = self.parent.full_observation_spec[in_key]
            elif in_key in self.parent.full_reward_spec.keys(True):
                target = self.parent.full_reward_spec[in_key]
            elif in_key in self.parent.full_done_spec.keys(True):
                # we account for this for completeness but it should never be the case
                target = self.parent.full_done_spec[in_key]
            else:
                raise RuntimeError(f"in_key {in_key} not found in output_spec.")
            target_return_spec = Unbounded(
                shape=target.shape,
                dtype=target.dtype,
                device=target.device,
            )
            # because all reward keys are discarded from the data during calls
            # to step_mdp, we must put this in observation_spec
            observation_spec[out_key] = target_return_spec
        return observation_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        # we must add the target return to the input spec
        input_spec["full_state_spec"] = self.transform_observation_spec(
            input_spec["full_state_spec"]
        )
        return input_spec


class RewardClipping(Transform):
    """Clips the reward between `clamp_min` and `clamp_max`.

    Args:
        clip_min (scalar): minimum value of the resulting reward.
        clip_max (scalar): maximum value of the resulting reward.

    """

    def __init__(
        self,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        clamp_min_tensor = (
            clamp_min if isinstance(clamp_min, Tensor) else torch.as_tensor(clamp_min)
        )
        clamp_max_tensor = (
            clamp_max if isinstance(clamp_max, Tensor) else torch.as_tensor(clamp_max)
        )
        self.register_buffer("clamp_min", clamp_min_tensor)
        self.register_buffer("clamp_max", clamp_max_tensor)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if self.clamp_max is not None and self.clamp_min is not None:
            reward = reward.clamp(self.clamp_min, self.clamp_max)
        elif self.clamp_min is not None:
            reward = reward.clamp_min(self.clamp_min)
        elif self.clamp_max is not None:
            reward = reward.clamp_max(self.clamp_max)
        return reward

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, Unbounded):
            return Bounded(
                self.clamp_min,
                self.clamp_max,
                shape=reward_spec.shape,
                device=reward_spec.device,
                dtype=reward_spec.dtype,
            )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clamp_min={float(self.clamp_min):4.4f}, clamp_max"
            f"={float(self.clamp_max):4.4f}, keys={self.in_keys})"
        )


class BinarizeReward(Transform):
    """Maps the reward to a binary value (0 or 1) if the reward is null or non-null, respectively.

    Args:
        in_keys (List[NestedKey]): input keys
        out_keys (List[NestedKey], optional): output keys. Defaults to value
            of ``in_keys``.
        dtype (torch.dtype, optional): the dtype of the binerized reward.
            Defaults to ``torch.int8``.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if not reward.shape or reward.shape[-1] != 1:
            raise RuntimeError(
                f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
            )
        return (reward > 0.0).to(torch.int8)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return Binary(
            n=1,
            device=reward_spec.device,
            shape=reward_spec.shape,
        )


class Resize(ObservationTransform):
    """Resizes a pixel observation.

    Args:
        w (int): resulting width.
        h (int, optional): resulting height. If not provided, the value of `w`
            is taken.
        interpolation (str): interpolation method

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> t = Resize(64, 84)
        >>> base_env = GymEnv("HalfCheetah-v4", from_pixels=True)
        >>> env = TransformedEnv(base_env, Compose(ToTensorImage(), t))
    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        interpolation: str = "bilinear",
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        # we also allow lists or tuples
        if isinstance(w, (list, tuple)):
            w, h = w
        if h is None:
            h = w
        if not _has_tv:
            raise ImportError(
                "Torchvision not found. The Resize transform relies on "
                "torchvision implementation. "
                "Consider installing this dependency."
            )
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = int(w)
        self.h = int(h)

        try:
            from torchvision.transforms.functional import InterpolationMode

            def interpolation_fn(interpolation):  # noqa: D103
                return InterpolationMode(interpolation)

        except ImportError:

            def interpolation_fn(interpolation):  # noqa: D103
                return interpolation

        self.interpolation = interpolation_fn(interpolation)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        # flatten if necessary
        if observation.shape[-2:] == torch.Size([self.w, self.h]):
            return observation
        ndim = observation.ndimension()
        if ndim > 4:
            sizes = observation.shape[:-3]
            observation = torch.flatten(observation, 0, ndim - 4)
        try:
            from torchvision.transforms.functional import resize
        except ImportError:
            from torchvision.transforms.functional_tensor import resize
        observation = resize(
            observation,
            [self.w, self.h],
            interpolation=self.interpolation,
            antialias=True,
        )
        if ndim > 4:
            observation = observation.unflatten(0, sizes)

        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape

        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={int(self.w)}, h={int(self.h)}, "
            f"interpolation={self.interpolation}, keys={self.in_keys})"
        )

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class Crop(ObservationTransform):
    """Crops the input image at the specified location and output size.

    Args:
        w (int): resulting width
        h (int, optional): resulting height. If None, then w is used (square crop).
        top (int, optional): top pixel coordinate to start cropping. Default is 0, i.e. top of the image.
        left (int, optional): left pixel coordinate to start cropping. Default is 0, i.e. left of the image.
        in_keys (sequence of NestedKey, optional): the entries to crop. If none is provided,
            ``["pixels"]`` is assumed.
        out_keys (sequence of NestedKey, optional): the cropped images keys. If none is
            provided, ``in_keys`` is assumed.

    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        top: int = 0,
        left: int = 0,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = w
        self.h = h if h else w
        self.top = top
        self.left = left

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import crop

        observation = crop(observation, self.top, self.left, self.w, self.h)
        return observation

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={float(self.w):4.4f}, h={float(self.h):4.4f}, top={float(self.top):4.4f}, left={float(self.left):4.4f}, "
        )


class CenterCrop(ObservationTransform):
    """Crops the center of an image.

    Args:
        w (int): resulting width
        h (int, optional): resulting height. If None, then w is used (square crop).
        in_keys (sequence of NestedKey, optional): the entries to crop. If none is provided,
            :obj:`["pixels"]` is assumed.
        out_keys (sequence of NestedKey, optional): the cropped images keys. If none is
            provided, :obj:`in_keys` is assumed.

    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = w
        self.h = h if h else w

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import center_crop

        observation = center_crop(observation, [self.w, self.h])
        return observation

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={float(self.w):4.4f}, h={float(self.h):4.4f}, "
        )


class FlattenObservation(ObservationTransform):
    """Flatten adjacent dimensions of a tensor.

    Args:
        first_dim (int): first dimension of the dimensions to flatten.
        last_dim (int): last dimension of the dimensions to flatten.
        in_keys (sequence of NestedKey, optional): the entries to flatten. If none is provided,
            :obj:`["pixels"]` is assumed.
        out_keys (sequence of NestedKey, optional): the flatten observation keys. If none is
            provided, :obj:`in_keys` is assumed.
        allow_positive_dim (bool, optional): if ``True``, positive dimensions are accepted.
            :obj:`FlattenObservation` will map these to the n^th feature dimension
            (ie n^th dimension after batch size of parent env) of the input tensor.
            Defaults to False, ie. non-negative dimensions are not permitted.
    """

    def __init__(
        self,
        first_dim: int,
        last_dim: int,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        allow_positive_dim: bool = False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not allow_positive_dim and first_dim >= 0:
            raise ValueError(
                "first_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes."
            )
        if not allow_positive_dim and last_dim >= 0:
            raise ValueError(
                "last_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes."
            )
        self._first_dim = first_dim
        self._last_dim = last_dim

    @property
    def first_dim(self):
        if self._first_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._first_dim
        return self._first_dim

    @property
    def last_dim(self):
        if self._last_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._last_dim
        return self._last_dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.flatten(observation, self.first_dim, self.last_dim)
        return observation

    forward = ObservationTransform._call

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space

        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"first_dim={int(self.first_dim)}, last_dim={int(self.last_dim)}, in_keys={self.in_keys}, out_keys={self.out_keys})"
        )


class UnsqueezeTransform(Transform):
    """Inserts a dimension of size one at the specified position.

    Args:
        dim (int): dimension to unsqueeze. Must be negative (or allow_positive_dim
            must be turned on).

    Keyword Args:
        allow_positive_dim (bool, optional): if ``True``, positive dimensions are accepted.
            `UnsqueezeTransform`` will map these to the n^th feature dimension
            (ie n^th dimension after batch size of parent env) of the input tensor,
            independently of the tensordict batch size (ie positive dims may be
            dangerous in contexts where tensordict of different batch dimension
            are passed).
            Defaults to False, ie. non-negative dimensions are not permitted.
        in_keys (list of NestedKeys): input entries (read).
        out_keys (list of NestedKeys): input entries (write). Defaults to ``in_keys`` if
            not provided.
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`~.inv` calls.
            Defaults to ``in_keys_in`` if not provided.
    """

    invertible = True

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._dim = None
        return super().__new__(cls)

    def __init__(
        self,
        dim: int | None = None,
        *,
        allow_positive_dim: bool = False,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = []  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []  # default
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self.allow_positive_dim = allow_positive_dim
        if dim >= 0 and not allow_positive_dim:
            raise RuntimeError(
                "dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes. Turn allow_positive_dim to accommodate "
                "for positive dim."
            )
        self._dim = dim

    @property
    def unsqueeze_dim(self):
        return self.dim

    @property
    def dim(self):
        if self._dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._dim
        return self._dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.unsqueeze(self.dim)
        return observation

    def _inv_apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.squeeze(self.dim)
        return observation

    def _transform_spec(self, spec: TensorSpec):
        space = spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            spec.shape = space.low.shape
        else:
            spec.shape = self._apply_transform(torch.zeros(spec.shape)).shape
        return spec

    # To map the specs, we actually use the forward call, not the inv
    _inv_transform_spec = _transform_spec

    @_apply_to_composite_inv
    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        return self._inv_transform_spec(action_spec)

    @_apply_to_composite_inv
    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        return self._inv_transform_spec(state_spec)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_key = self.parent.reward_key if self.parent is not None else "reward"
        if reward_key in self.in_keys:
            reward_spec = self._transform_spec(reward_spec)
        return reward_spec

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(observation_spec)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(dim={self.dim}, in_keys={self.in_keys}, out_keys={self.out_keys},"
            f" in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
        return s


class SqueezeTransform(UnsqueezeTransform):
    """Removes a dimension of size one at the specified position.

    Args:
        dim (int): dimension to squeeze.
    """

    invertible = True

    def __init__(
        self,
        dim: int | None = None,
        *args,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str] | None = None,
        in_keys_inv: Sequence[str] | None = None,
        out_keys_inv: Sequence[str] | None = None,
        **kwargs,
    ):
        if dim is None:
            if "squeeze_dim" in kwargs:
                warnings.warn(
                    f"squeeze_dim will be deprecated in favor of dim arg in {type(self).__name__}."
                )
                dim = kwargs.pop("squeeze_dim")
            else:
                raise TypeError(
                    f"dim must be passed to {type(self).__name__} constructor."
                )

        super().__init__(
            dim,
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            **kwargs,
        )

    @property
    def squeeze_dim(self):
        return super().dim

    _apply_transform = UnsqueezeTransform._inv_apply_transform
    _inv_apply_transform = UnsqueezeTransform._apply_transform


class PermuteTransform(Transform):
    """Permutation transform.

    Permutes input tensors along the desired dimensions. The permutations
    must be provided along the feature dimension (not batch dimension).

    Args:
        dims (list of int): the permuted order of the dimensions. Must be a reordering
            of the dims ``[-(len(dims)), ..., -1]``.
        in_keys (list of NestedKeys): input entries (read).
        out_keys (list of NestedKeys): input entries (write). Defaults to ``in_keys`` if
            not provided.
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`~.inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`~.inv` calls. Defaults to ``in_keys_in`` if
            not provided.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> base_env = GymEnv("ALE/Pong-v5")
        >>> base_env.rollout(2)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> env = TransformedEnv(base_env, PermuteTransform((-1, -3, -2), in_keys=["pixels"]))
        >>> env.rollout(2)  # channels are at the end
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        dims,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
    ):
        if in_keys is None:
            in_keys = []
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)

        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        # check dims
        self.dims = dims
        if sorted(dims) != list(range(-len(dims), 0)):
            raise ValueError(
                f"Only tailing dims with negative indices are supported by {self.__class__.__name__}. Got {dims} instead."
            )

    @staticmethod
    def _invert_permute(p):
        def _find_inv(i):
            for j, _p in enumerate(p):
                if _p < 0:
                    inv = True
                    _p = len(p) + _p
                else:
                    inv = False
                if i == _p:
                    if inv:
                        return j - len(p)
                    else:
                        return j
            else:
                # unreachable
                raise RuntimeError

        return [_find_inv(i) for i in range(len(p))]

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
        observation = observation.permute(
            *list(range(observation.ndimension() - len(self.dims))), *self.dims
        )
        return observation

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        permuted_dims = self._invert_permute(self.dims)
        state = state.permute(
            *list(range(state.ndimension() - len(self.dims))), *permuted_dims
        )
        return state

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._edit_space(observation_spec)
        observation_spec.shape = torch.Size(
            [
                *observation_spec.shape[: -len(self.dims)],
                *[observation_spec.shape[dim] for dim in self.dims],
            ]
        )
        return observation_spec

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        permuted_dims = self._invert_permute(self.dims)
        input_spec = self._edit_space_inv(input_spec)
        input_spec.shape = torch.Size(
            [
                *input_spec.shape[: -len(permuted_dims)],
                *[input_spec.shape[dim] for dim in permuted_dims],
            ]
        )
        return input_spec

    def _edit_space(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._apply_transform(spec.space.high)
            spec.space.low = self._apply_transform(spec.space.low)
        return spec

    def _edit_space_inv(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._inv_apply_transform(spec.space.high)
            spec.space.low = self._inv_apply_transform(spec.space.low)
        return spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class GrayScale(ObservationTransform):
    """Turns a pixel observation to grayscale."""

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = F.rgb_to_grayscale(observation)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class ObservationNorm(ObservationTransform):
    """Observation affine transformation layer.

    Normalizes an observation according to

    .. math::
        obs = obs * scale + loc

    Args:
        loc (number or tensor): location of the affine transform
        scale (number or tensor): scale of the affine transform
        in_keys (sequence of NestedKey, optional): entries to be normalized. Defaults to ["observation", "pixels"].
            All entries will be normalized with the same values: if a different behavior is desired
            (e.g. a different normalization for pixels and states) different :obj:`ObservationNorm`
            objects should be used.
        out_keys (sequence of NestedKey, optional): output entries. Defaults to the value of `in_keys`.
        in_keys_inv (sequence of NestedKey, optional): ObservationNorm also supports inverse transforms. This will
            only occur if a list of keys is provided to :obj:`in_keys_inv`. If none is provided,
            only the forward transform will be called.
        out_keys_inv (sequence of NestedKey, optional): output entries for the inverse transform.
            Defaults to the value of `in_keys_inv`.
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                obs = (obs-loc)/scale

            as it is done for standardization. Default is `False`.

        eps (:obj:`float`, optional): epsilon increment for the scale in the ``standard_normal`` case.
            Defaults to ``1e-6`` if not recoverable directly from the scale dtype.

    Examples:
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> r = torch.randn(100, 3)*torch.randn(3) + torch.randn(3)
        >>> td = TensorDict({'obs': r}, [100])
        >>> transform = ObservationNorm(
        ...     loc = td.get('obs').mean(0),
        ...     scale = td.get('obs').std(0),
        ...     in_keys=["obs"],
        ...     standard_normal=True)
        >>> _ = transform(td)
        >>> print(torch.isclose(td.get('obs').mean(0),
        ...     torch.zeros(3)).all())
        tensor(True)
        >>> print(torch.isclose(td.get('next_obs').std(0),
        ...     torch.ones(3)).all())
        tensor(True)

    The normalization stats can be automatically computed:
    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> torch.manual_seed(0)
        >>> env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(env, ObservationNorm(in_keys=["observation"]))
        >>> env.set_seed(0)
        >>> env.transform.init_stats(100)
        >>> print(env.transform.loc, env.transform.scale)
        tensor([-1.3752e+01, -6.5087e-03,  2.9294e-03], dtype=torch.float32) tensor([14.9636,  2.5608,  0.6408], dtype=torch.float32)

    """

    _ERR_INIT_MSG = "Cannot have an mixed initialized and uninitialized loc and scale"

    def __init__(
        self,
        loc: float | torch.Tensor | None = None,
        scale: float | torch.Tensor | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
        eps: float | None = None,
    ):
        if in_keys is None:
            raise RuntimeError(
                "Not passing in_keys to ObservationNorm is a deprecated behavior."
            )

        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)

        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        if not isinstance(standard_normal, torch.Tensor):
            standard_normal = torch.as_tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)
        self.eps = (
            eps
            if eps is not None
            else torch.finfo(scale.dtype).eps
            if isinstance(scale, torch.Tensor) and scale.dtype.is_floating_point
            else 1e-6
        )

        if loc is not None and not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.get_default_dtype())
        elif loc is None:
            if scale is not None:
                raise ValueError(self._ERR_INIT_MSG)
            loc = nn.UninitializedBuffer()

        if scale is not None and not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.get_default_dtype())
            scale = scale.clamp_min(self.eps)
        elif scale is None:
            # check that loc is None too
            if not isinstance(loc, nn.UninitializedBuffer):
                raise ValueError(self._ERR_INIT_MSG)
            scale = nn.UninitializedBuffer()

        # self.observation_spec_key = observation_spec_key
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def initialized(self):
        return not isinstance(self.loc, nn.UninitializedBuffer)

    def init_stats(
        self,
        num_iter: int,
        reduce_dim: int | tuple[int] = 0,
        cat_dim: int | None = None,
        key: NestedKey | None = None,
        keep_dims: tuple[int] | None = None,
    ) -> None:
        """Initializes the loc and scale stats of the parent environment.

        Normalization constant should ideally make the observation statistics approach
        those of a standard Gaussian distribution. This method computes a location
        and scale tensor that will empirically compute the mean and standard
        deviation of a Gaussian distribution fitted on data generated randomly with
        the parent environment for a given number of steps.

        Args:
            num_iter (int): number of random iterations to run in the environment.
            reduce_dim (int or tuple of int, optional): dimension to compute the mean and std over.
                Defaults to 0.
            cat_dim (int, optional): dimension along which the batches collected will be concatenated.
                It must be part equal to reduce_dim (if integer) or part of the reduce_dim tuple.
                Defaults to the same value as reduce_dim.
            key (NestedKey, optional): if provided, the summary statistics will be
                retrieved from that key in the resulting tensordicts.
                Otherwise, the first key in :obj:`ObservationNorm.in_keys` will be used.
            keep_dims (tuple of int, optional): the dimensions to keep in the loc and scale.
                For instance, one may want the location and scale to have shape [C, 1, 1]
                when normalizing a 3D tensor over the last two dimensions, but not the
                third. Defaults to None.

        """
        if cat_dim is None:
            cat_dim = reduce_dim
            if not isinstance(cat_dim, int):
                raise ValueError(
                    "cat_dim must be specified if reduce_dim is not an integer."
                )
        if (isinstance(reduce_dim, tuple) and cat_dim not in reduce_dim) or (
            isinstance(reduce_dim, int) and cat_dim != reduce_dim
        ):
            raise ValueError("cat_dim must be part of or equal to reduce_dim.")
        if self.initialized:
            raise RuntimeError(
                f"Loc/Scale are already initialized: ({self.loc}, {self.scale})"
            )

        if len(self.in_keys) > 1 and key is None:
            raise RuntimeError(
                "Transform has multiple in_keys but no specific key was passed as an argument"
            )
        key = self.in_keys[0] if key is None else key

        def raise_initialization_exception(module):
            if isinstance(module, ObservationNorm) and not module.initialized:
                raise RuntimeError(
                    "ObservationNorms need to be initialized in the right order."
                    "Trying to initialize an ObservationNorm "
                    "while a parent ObservationNorm transform is still uninitialized"
                )

        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "Cannot initialize the transform if parent env is not defined."
            )
        parent.apply(raise_initialization_exception)

        collected_frames = 0
        data = []
        while collected_frames < num_iter:
            tensordict = parent.rollout(max_steps=num_iter)
            collected_frames += tensordict.numel()
            data.append(tensordict.get(key))

        data = torch.cat(data, cat_dim)
        if isinstance(reduce_dim, int):
            reduce_dim = [reduce_dim]
        # make all reduce_dim and keep_dims negative
        reduce_dim = sorted(dim if dim < 0 else dim - data.ndim for dim in reduce_dim)

        if keep_dims is not None:
            keep_dims = sorted(dim if dim < 0 else dim - data.ndim for dim in keep_dims)
            if not all(k in reduce_dim for k in keep_dims):
                raise ValueError("keep_dim elements must be part of reduce_dim list.")
        else:
            keep_dims = []
        loc = data.mean(reduce_dim, keepdim=True)
        scale = data.std(reduce_dim, keepdim=True)
        for r in reduce_dim:
            if r not in keep_dims:
                loc = loc.squeeze(r)
                scale = scale.squeeze(r)

        if not self.standard_normal:
            scale = 1 / scale.clamp_min(self.eps)
            loc = -loc * scale

        if not torch.isfinite(loc).all():
            raise RuntimeError("Non-finite values found in loc")
        if not torch.isfinite(scale).all():
            raise RuntimeError("Non-finite values found in scale")
        self.loc.materialize(shape=loc.shape, dtype=loc.dtype)
        self.loc.copy_(loc)
        self.scale.materialize(shape=scale.shape, dtype=scale.dtype)
        self.scale.copy_(scale.clamp_min(self.eps))

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError(
                "Loc/Scale have not been initialized. Either pass in values in the constructor "
                "or call the init_stats method"
            )
        if self.standard_normal:
            loc = self.loc
            scale = self.scale
            return (obs - loc) / scale
        else:
            scale = self.scale
            loc = self.loc
            return obs * scale + loc

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if self.loc is None or self.scale is None:
            raise RuntimeError(
                "Loc/Scale have not been initialized. Either pass in values in the constructor "
                "or call the init_stats method"
            )
        if not self.standard_normal:
            loc = self.loc
            scale = self.scale
            return (state - loc) / scale
        else:
            scale = self.scale
            loc = self.loc
            return state * scale + loc

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return observation_spec

    # @_apply_to_composite_inv
    # def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
    #     space = input_spec.space
    #     if isinstance(space, ContinuousBox):
    #         space.low = self._apply_transform(space.low)
    #         space.high = self._apply_transform(space.high)
    #     return input_spec

    @_apply_to_composite_inv
    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        space = action_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return action_spec

    @_apply_to_composite_inv
    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        space = state_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return state_spec

    def __repr__(self) -> str:
        if self.initialized and (self.loc.numel() == 1 and self.scale.numel() == 1):
            return (
                f"{self.__class__.__name__}("
                f"loc={float(self.loc):4.4f}, scale"
                f"={float(self.scale):4.4f}, keys={self.in_keys})"
            )
        else:
            return super().__repr__()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class CatFrames(ObservationTransform):
    """Concatenates successive observation frames into a single tensor.

    This transform is useful for creating a sense of movement or velocity in the observed features.
    It can also be used with models that require access to past observations such as transformers and the like.
    It was first proposed in "Playing Atari with Deep Reinforcement Learning" (https://arxiv.org/pdf/1312.5602.pdf).

    When used within a transformed environment,
    :class:`CatFrames` is a stateful class, and it can be reset to its native state by
    calling the :meth:`~.reset` method. This method accepts tensordicts with a
    ``"_reset"`` entry that indicates which buffer to reset.

    Args:
        N (int): number of observation to concatenate.
        dim (int): dimension along which concatenate the
            observations. Should be negative, to ensure that it is compatible
            with environments of different batch_size.
        in_keys (sequence of NestedKey, optional): keys pointing to the frames that have
            to be concatenated. Defaults to ["pixels"].
        out_keys (sequence of NestedKey, optional): keys pointing to where the output
            has to be written. Defaults to the value of `in_keys`.
        padding (str, optional): the padding method. One of ``"same"`` or ``"constant"``.
            Defaults to ``"same"``, ie. the first value is used for padding.
        padding_value (:obj:`float`, optional): the value to use for padding if ``padding="constant"``.
            Defaults to 0.
        as_inverse (bool, optional): if ``True``, the transform is applied as an inverse transform. Defaults to ``False``.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.
        done_key (NestedKey, optional): the done key to be used as partial
            done indicator. Must be unique. If not provided, defaults to ``"done"``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv('Pendulum-v1'),
        ...     Compose(
        ...         UnsqueezeTransform(-1, in_keys=["observation"]),
        ...         CatFrames(N=4, dim=-1, in_keys=["observation"]),
        ...     )
        ... )
        >>> print(env.rollout(3))

    The :class:`CatFrames` transform can also be used offline to reproduce the
    effect of the online frame concatenation at a different scale (or for the
    purpose of limiting the memory consumption). The following example
    gives the complete picture, together with the usage of a :class:`torchrl.data.ReplayBuffer`:

    Examples:
        >>> from torchrl.envs.utils import RandomPolicy        >>> from torchrl.envs import UnsqueezeTransform, CatFrames
        >>> from torchrl.collectors import SyncDataCollector
        >>> # Create a transformed environment with CatFrames: notice the usage of UnsqueezeTransform to create an extra dimension
        >>> env = TransformedEnv(
        ...     GymEnv("CartPole-v1", from_pixels=True),
        ...     Compose(
        ...         ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        ...         Resize(in_keys=["pixels_trsf"], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf"]),
        ...         UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
        ...     )
        ... )
        >>> # we design a collector
        >>> collector = SyncDataCollector(
        ...     env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=10,
        ...     total_frames=1000,
        ... )
        >>> for data in collector:
        ...     print(data)
        ...     break
        >>> # now let's create a transform for the replay buffer. We don't need to unsqueeze the data here.
        >>> # however, we need to point to both the pixel entry at the root and at the next levels:
        >>> t = Compose(
        ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ... )
        >>> from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
        >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
        >>> data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
        >>> rb.add(data_exclude)
        >>> s = rb.sample(1) # the buffer has only one element
        >>> # let's check that our sample is the same as the batch collected during inference
        >>> assert (data.exclude("collector")==s.squeeze(0).exclude("index", "collector")).all()

    .. note:: :class:`~CatFrames` currently only supports ``"done"``
        signal at the root. Nested ``done``,
        such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    .. note:: Storing stacks of frames in the replay buffer can significantly increase memory consumption (by N times).
        To mitigate this, you can store trajectories directly in the replay buffer and apply :class:`CatFrames` at sampling time.
        This approach involves sampling slices of the stored trajectories and then applying the frame stacking transform.
        For convenience, :class:`CatFrames` provides a :meth:`~.make_rb_transform_and_sampler` method that creates:

        - A modified version of the transform suitable for use in replay buffers
        - A corresponding :class:`SliceSampler` to use with the buffer

    """

    inplace = False
    _CAT_DIM_ERR = (
        "dim must be < 0 to accommodate for tensordict of "
        "different batch-sizes (since negative dims are batch invariant)."
    )
    ACCEPTED_PADDING = {"same", "constant", "zeros"}

    def __init__(
        self,
        N: int,
        dim: int,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        padding="same",
        padding_value=0,
        as_inverse=False,
        reset_key: NestedKey | None = None,
        done_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.N = N
        if dim >= 0:
            raise ValueError(self._CAT_DIM_ERR)
        self.dim = dim
        if padding not in self.ACCEPTED_PADDING:
            raise ValueError(f"padding must be one of {self.ACCEPTED_PADDING}")
        if padding == "zeros":
            raise RuntimeError("Padding option 'zeros' will is deprecated")
        self.padding = padding
        self.padding_value = padding_value
        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            self.register_buffer(
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        # keeps track of calls to _reset since it's only _call that will populate the buffer
        self.as_inverse = as_inverse
        self.reset_key = reset_key
        self.done_key = done_key

    def make_rb_transform_and_sampler(
        self, batch_size: int, **sampler_kwargs
    ) -> tuple[Transform, torchrl.data.replay_buffers.SliceSampler]:  # noqa: F821
        """Creates a transform and sampler to be used with a replay buffer when storing frame-stacked data.

        This method helps reduce redundancy in stored data by avoiding the need to
        store the entire stack of frames in the buffer. Instead, it creates a
        transform that stacks frames on-the-fly during sampling, and a sampler that
        ensures the correct sequence length is maintained.

        Args:
            batch_size (int): The batch size to use for the sampler.
            **sampler_kwargs: Additional keyword arguments to pass to the
                :class:`~torchrl.data.replay_buffers.SliceSampler` constructor.

        Returns:
            A tuple containing:

                - transform (Transform): A transform that stacks frames on-the-fly during sampling.
                - sampler (SliceSampler): A sampler that ensures the correct sequence length is maintained.

        Example:
            >>> env = TransformedEnv(...)
            >>> catframes = CatFrames(N=4, ...)
            >>> transform, sampler = catframes.make_rb_transform_and_sampler(batch_size=32)
            >>> rb = ReplayBuffer(..., sampler=sampler, transform=transform)

        .. note:: When working with images, it's recommended to use distinct ``in_keys`` and ``out_keys`` in the preceding
            :class:`~torchrl.envs.ToTensorImage` transform. This ensures that the tensors stored in the buffer are separate
            from their processed counterparts, which we don't want to store.
            For non-image data, consider inserting a :class:`~torchrl.envs.RenameTransform` before :class:`CatFrames` to create
            a copy of the data that will be stored in the buffer.

        .. note:: When adding the transform to the replay buffer, one should pay attention to also pass the transforms
            that precede CatFrames, such as :class:`~torchrl.envs.ToTensorImage` or :class:`~torchrl.envs.UnsqueezeTransform`
            in such a way that the :class:`~torchrl.envs.CatFrames` transforms sees data formatted as it was during data
            collection.

        .. note:: For a more complete example, refer to torchrl's github repo `examples` folder:
            https://github.com/pytorch/rl/tree/main/examples/replay-buffers/catframes-in-buffer.py

        """
        from torchrl.data.replay_buffers import SliceSampler

        in_keys = self.in_keys
        in_keys = in_keys + [unravel_key(("next", key)) for key in in_keys]
        out_keys = self.out_keys
        out_keys = out_keys + [unravel_key(("next", key)) for key in out_keys]
        catframes = type(self)(
            N=self.N,
            in_keys=in_keys,
            out_keys=out_keys,
            dim=self.dim,
            padding=self.padding,
            padding_value=self.padding_value,
            as_inverse=False,
            reset_key=self.reset_key,
            done_key=self.done_key,
        )
        sampler = SliceSampler(slice_len=self.N, **sampler_kwargs)
        sampler._batch_size_multiplier = self.N
        transform = Compose(
            lambda td: td.reshape(-1, self.N),
            catframes,
            lambda td: td[:, -1],
            # We only store "pixels" to the replay buffer to save memory
            ExcludeTransform(*out_keys, inverse=True),
        )
        return transform, sampler

    @property
    def done_key(self):
        done_key = self.__dict__.get("_done_key", None)
        if done_key is None:
            done_key = "done"
            self._done_key = done_key
        return done_key

    @done_key.setter
    def done_key(self, value):
        self._done_key = value

    @property
    def reset_key(self):
        reset_key = getattr(self, "_reset_key", None)
        if reset_key is not None:
            return reset_key
        reset_keys = self.parent.reset_keys
        if len(reset_keys) > 1:
            raise RuntimeError(
                f"Got more than one reset key in env {self.container}, cannot infer which one to use. "
                f"Consider providing the reset key in the {type(self)} constructor."
            )
        reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets _buffers."""
        _reset = _get_reset(self.reset_key, tensordict)
        if self.as_inverse and self.parent is not None:
            raise Exception(
                "CatFrames as inverse is not supported as a transform for environments, only for replay buffers."
            )

        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset, _reset=_reset)

        return tensordict_reset

    def _make_missing_buffer(self, data, buffer_name):
        shape = list(data.shape)
        d = shape[self.dim]
        shape[self.dim] = d * self.N
        shape = torch.Size(shape)
        getattr(self, buffer_name).materialize(shape)
        buffer = (
            getattr(self, buffer_name)
            .to(dtype=data.dtype, device=data.device)
            .fill_(self.padding_value)
        )
        setattr(self, buffer_name, buffer)
        return buffer

    def _inv_call(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.as_inverse:
            return self.unfolding(tensordict)
        else:
            return tensordict

    def _call(self, next_tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        _just_reset = _reset is not None
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_cat_buffers_{in_key}"
            data = next_tensordict.get(in_key)
            d = data.size(self.dim)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(data, buffer_name)
            # shift obs 1 position to the right
            if _just_reset:
                if _reset.all():
                    _all = True
                    data_reset = data
                    buffer_reset = buffer
                    dim = self.dim
                else:
                    _all = False
                    data_reset = data[_reset]
                    buffer_reset = buffer[_reset]
                    dim = self.dim - _reset.ndim + 1
                shape = [1 for _ in buffer_reset.shape]
                if _all:
                    shape[dim] = self.N
                else:
                    shape[dim] = self.N

                if self.padding == "same":
                    if _all:
                        buffer.copy_(data_reset.repeat(shape).clone())
                    else:
                        buffer[_reset] = data_reset.repeat(shape).clone()
                elif self.padding == "constant":
                    if _all:
                        buffer.fill_(self.padding_value)
                    else:
                        buffer[_reset] = self.padding_value
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError

                if self.dim < 0:
                    n = buffer_reset.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = tuple([slice(None, None) for _ in range(n)] + [slice(-d, None)])
                if not _all:
                    buffer_reset = buffer[_reset]
                buffer_reset[idx] = data_reset
                if not _all:
                    buffer[_reset] = buffer_reset
            else:
                buffer.copy_(torch.roll(buffer, shifts=-d, dims=self.dim))
                # add new obs
                if self.dim < 0:
                    n = buffer.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = tuple([slice(None, None) for _ in range(n)] + [slice(-d, None)])
                buffer[idx] = buffer[idx].copy_(data)
            # add to tensordict
            next_tensordict.set(out_key, buffer.clone())
        return next_tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = torch.cat([space.low] * self.N, self.dim)
            space.high = torch.cat([space.high] * self.N, self.dim)
            observation_spec.shape = space.low.shape
        else:
            shape = list(observation_spec.shape)
            shape[self.dim] = self.N * shape[self.dim]
            observation_spec.shape = torch.Size(shape)
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.as_inverse:
            return tensordict
        else:
            return self.unfolding(tensordict)

    def _apply_same_padding(self, dim, data, done_mask):
        d = data.ndim + dim - 1
        res = data.clone()
        num_repeats_per_sample = done_mask.sum(dim=-1)

        if num_repeats_per_sample.dim() > 2:
            extra_dims = num_repeats_per_sample.dim() - 2
            num_repeats_per_sample = num_repeats_per_sample.flatten(0, extra_dims)
            res_flat_series = res.flatten(0, extra_dims)
        else:
            extra_dims = 0
            res_flat_series = res

        if d - 1 > extra_dims:
            res_flat_series_flat_batch = res_flat_series.flatten(1, d - 1)
        else:
            res_flat_series_flat_batch = res_flat_series[:, None]

        for sample_idx, num_repeats in enumerate(num_repeats_per_sample):
            if num_repeats > 0:
                res_slice = res_flat_series_flat_batch[sample_idx]
                res_slice[:, :num_repeats] = res_slice[:, num_repeats : num_repeats + 1]

        return res

    @set_lazy_legacy(False)
    def unfolding(self, tensordict: TensorDictBase) -> TensorDictBase:
        # it is assumed that the last dimension of the tensordict is the time dimension
        if not tensordict.ndim:
            raise ValueError(
                "CatFrames cannot process unbatched tensordict instances. "
                "Make sure your input has more than one dimension and "
                "the time dimension is marked as 'time', e.g., "
                "`tensordict.refine_names(None, 'time', None)`."
            )
        i = 0
        for i, name in enumerate(tensordict.names):  # noqa: B007
            if name == "time":
                break
        else:
            warnings.warn(
                "The last dimension of the tensordict should be marked as 'time'. "
                "CatFrames will unfold the data along the time dimension assuming that "
                "the time dimension is the last dimension of the input tensordict. "
                "Define a 'time' dimension name (e.g., `tensordict.refine_names(..., 'time')`) to skip this warning. ",
                category=UserWarning,
            )
        tensordict_orig = tensordict
        if i != tensordict.ndim - 1:
            tensordict = tensordict.transpose(tensordict.ndim - 1, i)
        # first sort the in_keys with strings and non-strings
        keys = [
            (in_key, out_key)
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys)
            if isinstance(in_key, str)
        ]
        keys += [
            (in_key, out_key)
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys)
            if not isinstance(in_key, str)
        ]

        def unfold_done(done, N):
            prefix = (slice(None),) * (tensordict.ndim - 1)
            reset = torch.cat(
                [
                    torch.zeros_like(done[prefix + (slice(self.N - 1),)]),
                    torch.ones_like(done[prefix + (slice(1),)]),
                    done[prefix + (slice(None, -1),)],
                ],
                tensordict.ndim - 1,
            )
            reset_unfold = reset.unfold(tensordict.ndim - 1, self.N, 1)
            reset_unfold_slice = reset_unfold[..., -1]
            reset_unfold_list = [torch.zeros_like(reset_unfold_slice)]
            for r in reversed(reset_unfold.unbind(-1)):
                reset_unfold_list.append(r | reset_unfold_list[-1])
                # reset_unfold_slice = reset_unfold_list[-1]
            reset_unfold = torch.stack(list(reversed(reset_unfold_list))[1:], -1)
            reset = reset[prefix + (slice(self.N - 1, None),)]
            reset[prefix + (0,)] = 1
            return reset_unfold, reset

        done = tensordict.get(("next", self.done_key))
        done_mask, reset = unfold_done(done, self.N)

        for in_key, out_key in keys:
            # check if we have an obs in "next" that has already been processed.
            # If so, we must add an offset
            data_orig = data = tensordict.get(in_key)
            n_feat = data_orig.shape[data.ndim + self.dim]
            first_val = None
            if isinstance(in_key, tuple) and in_key[0] == "next":
                # let's get the out_key we have already processed
                prev_out_key = dict(_zip_strict(self.in_keys, self.out_keys)).get(
                    in_key[1], None
                )
                if prev_out_key is not None:
                    prev_val = tensordict.get(prev_out_key)
                    # n_feat = prev_val.shape[data.ndim + self.dim] // self.N
                    first_val = prev_val.unflatten(
                        data.ndim + self.dim, (self.N, n_feat)
                    )

            idx = [slice(None)] * (tensordict.ndim - 1) + [0]
            data0 = [
                torch.full_like(data[tuple(idx)], self.padding_value).unsqueeze(
                    tensordict.ndim - 1
                )
            ] * (self.N - 1)

            data = torch.cat(data0 + [data], tensordict.ndim - 1)

            data = data.unfold(tensordict.ndim - 1, self.N, 1)

            # Place -1 dim at self.dim place before squashing
            done_mask_expand = done_mask.view(
                *done_mask.shape[: tensordict.ndim],
                *(1,) * (data.ndim - 1 - tensordict.ndim),
                done_mask.shape[-1],
            )
            done_mask_expand = expand_as_right(done_mask_expand, data)
            data = data.permute(
                *range(0, data.ndim + self.dim - 1),
                -1,
                *range(data.ndim + self.dim - 1, data.ndim - 1),
            )
            done_mask_expand = done_mask_expand.permute(
                *range(0, done_mask_expand.ndim + self.dim - 1),
                -1,
                *range(done_mask_expand.ndim + self.dim - 1, done_mask_expand.ndim - 1),
            )
            if self.padding != "same":
                data = torch.where(done_mask_expand, self.padding_value, data)
            else:
                data = self._apply_same_padding(self.dim, data, done_mask)

            if first_val is not None:
                # Aggregate reset along last dim
                reset_any = reset.any(-1, False)
                rexp = expand_right(
                    reset_any, (*reset_any.shape, *data.shape[data.ndim + self.dim :])
                )
                rexp = torch.cat(
                    [
                        torch.zeros_like(
                            data0[0].repeat_interleave(
                                len(data0), dim=tensordict.ndim - 1
                            ),
                            dtype=torch.bool,
                        ),
                        rexp,
                    ],
                    tensordict.ndim - 1,
                )
                rexp = rexp.unfold(tensordict.ndim - 1, self.N, 1)
                rexp_orig = rexp
                rexp = torch.cat([rexp[..., 1:], torch.zeros_like(rexp[..., -1:])], -1)
                if self.padding == "same":
                    rexp_orig = rexp_orig.flip(-1).cumsum(-1).flip(-1).bool()
                    rexp = rexp.flip(-1).cumsum(-1).flip(-1).bool()
                rexp_orig = torch.cat(
                    [torch.zeros_like(rexp_orig[..., -1:]), rexp_orig[..., 1:]], -1
                )
                rexp = rexp.permute(
                    *range(0, rexp.ndim + self.dim - 1),
                    -1,
                    *range(rexp.ndim + self.dim - 1, rexp.ndim - 1),
                )
                rexp_orig = rexp_orig.permute(
                    *range(0, rexp_orig.ndim + self.dim - 1),
                    -1,
                    *range(rexp_orig.ndim + self.dim - 1, rexp_orig.ndim - 1),
                )
                data[rexp] = first_val[rexp_orig]
            data = data.flatten(data.ndim + self.dim - 1, data.ndim + self.dim)
            tensordict.set(out_key, data)
        if tensordict_orig is not tensordict:
            tensordict_orig = tensordict.transpose(tensordict.ndim - 1, i)
        return tensordict_orig

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={self.N}, dim"
            f"={self.dim}, keys={self.in_keys})"
        )


class RewardScaling(Transform):
    """Affine transform of the reward.

     The reward is transformed according to:

    .. math::
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                reward = (reward-loc)/scale

            as it is done for standardization. Default is `False`.
    """

    def __init__(
        self,
        loc: float | torch.Tensor,
        scale: float | torch.Tensor,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not isinstance(standard_normal, torch.Tensor):
            standard_normal = torch.tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)

        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale.clamp_min(1e-6))

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if self.standard_normal:
            loc = self.loc
            scale = self.scale
            reward = (reward - loc) / scale
            return reward
        else:
            scale = self.scale
            loc = self.loc
            reward = reward * scale + loc
            return reward

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, Unbounded):
            return reward_spec
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loc={self.loc.item():4.4f}, scale={self.scale.item():4.4f}, "
            f"keys={self.in_keys})"
        )


class FiniteTensorDictCheck(Transform):
    """This transform will check that all the items of the tensordict are finite, and raise an exception if they are not."""

    def __init__(self):
        super().__init__(in_keys=[])

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict.apply(check_finite, filter_empty=True)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    forward = _call


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
    def in_keys(self):
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
    def out_keys(self):
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
    def in_keys_inv(self):
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
    def out_keys_inv(self):
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
        device (torch.device or equivalent): the destination device.
        orig_device (torch.device or equivalent): the origin device. If not specified and
            a parent environment exists, it it retrieved from it. In all other cases,
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


class CatTensors(Transform):
    """Concatenates several keys in a single tensor.

    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        in_keys (sequence of NestedKey): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behavior will only work if a parent is set.
        out_key (NestedKey): key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is ``-1``.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted after
            concatenation. Default is ``True``.
        unsqueeze_if_oor (bool, optional): if ``True``, CatTensor will check that
            the indicated dimension exists for the tensors to concatenate. If not,
            the tensors will be unsqueezed along that dimension.
            Default is ``False``.
        sort (bool, optional): if ``True``, the keys will be sorted in the
            transform. Otherwise, the order provided by the user will prevail.
            Defaults to ``True``.

    Examples:
        >>> transform = CatTensors(in_keys=["key1", "key2"])
        >>> td = TensorDict({"key1": torch.zeros(1, 1),
        ...     "key2": torch.ones(1, 1)}, [1])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector"))
        tensor([[0., 1.]])
        >>> transform = CatTensors(in_keys=["key1", "key2"], dim=-2, unsqueeze_if_oor=True)
        >>> td = TensorDict({"key1": torch.zeros(1),
        ...     "key2": torch.ones(1)}, [])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector").shape)
        torch.Size([2, 1])

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_key: NestedKey = "observation_vector",
        dim: int = -1,
        *,
        del_keys: bool = True,
        unsqueeze_if_oor: bool = False,
        sort: bool = True,
    ):
        self._initialized = in_keys is not None
        if not self._initialized:
            if dim != -1:
                raise ValueError(
                    "Lazy call to CatTensors is only supported when `dim=-1`."
                )
        elif sort:
            in_keys = sorted(in_keys, key=_sort_keys)
        if not isinstance(out_key, (str, tuple)):
            raise Exception("CatTensors requires out_key to be of type NestedKey")
        super().__init__(in_keys=in_keys, out_keys=[out_key])
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None
        self.unsqueeze_if_oor = unsqueeze_if_oor

    @property
    def keys_to_exclude(self):
        if self._keys_to_exclude is None:
            self._keys_to_exclude = [
                key for key in self.in_keys if key != self.out_keys[0]
            ]
        return self._keys_to_exclude

    def _find_in_keys(self):
        """Gathers all the entries from observation spec which shape is 1d."""
        parent = self.parent
        obs_spec = parent.observation_spec
        in_keys = []
        for key, value in obs_spec.items(True, True):
            if len(value.shape) == 1:
                in_keys.append(key)
        return sorted(in_keys, key=_sort_keys)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        values = [next_tensordict.get(key, None) for key in self.in_keys]
        if any(value is None for value in values):
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(self.in_keys, key=_sort_keys)} but got a TensorDict with keys"
                f" {sorted(next_tensordict.keys(include_nested=True), key=_sort_keys)}"
            )
        if self.unsqueeze_if_oor:
            pos_idx = self.dim > 0
            abs_idx = self.dim if pos_idx else -self.dim - 1
            values = [
                v
                if abs_idx < v.ndimension()
                else v.unsqueeze(0)
                if not pos_idx
                else v.unsqueeze(-1)
                for v in values
            ]

        out_tensor = torch.cat(values, dim=self.dim)
        next_tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            next_tensordict.exclude(*self.keys_to_exclude, inplace=True)
        return next_tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, Composite):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, Composite) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, Composite):
            # by def, there must be only one key
            return observation_spec

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]

        sum_shape = sum(
            [
                observation_spec[key].shape[self.dim]
                if observation_spec[key].shape
                else 1
                for key in keys
            ]
        )
        spec0 = observation_spec[keys[0]]
        out_key = self.out_keys[0]
        shape = list(spec0.shape)
        device = spec0.device
        shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = Unbounded(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        if self._del_keys:
            for key in self.keys_to_exclude:
                if key in observation_spec.keys(True):
                    del observation_spec[key]
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_key"
            f"={self.out_keys[0]})"
        )


class UnaryTransform(Transform):
    r"""Applies a unary operation on the specified inputs.

    Args:
        in_keys (sequence of NestedKey): the keys of inputs to the unary operation.
        out_keys (sequence of NestedKey): the keys of the outputs of the unary operation.
        in_keys_inv (sequence of NestedKey, optional): the keys of inputs to the unary operation during inverse call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the outputs of the unary operation durin inverse call.

    Keyword Args:
        fn (Callable[[Any], Tensor | TensorDictBase]): the function to use as the unary operation. If it accepts
            a non-tensor input, it must also accept ``None``.
        inv_fn (Callable[[Any], Any], optional): the function to use as the unary operation during inverse calls.
            If it accepts a non-tensor input, it must also accept ``None``.
            Can be ommitted, in which case :attr:`fn` will be used for inverse maps.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before ``fn`` is called
            on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to ``fn``, which must support those
            inputs. Default is ``False``.

    Example:
        >>> from torchrl.envs import GymEnv, UnaryTransform
        >>> env = GymEnv("Pendulum-v1")
        >>> env = env.append_transform(
        ...     UnaryTransform(
        ...         in_keys=["observation"],
        ...         out_keys=["observation_trsf"],
        ...             fn=lambda tensor: str(tensor.numpy().tobytes())))
        >>> env.observation_spec
        Composite(
            observation: BoundedContinuous(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            observation_trsf: NonTensor(
                shape=torch.Size([]),
                space=None,
                device=cpu,
                dtype=None,
                domain=None),
            device=None,
            shape=torch.Size([]))
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        observation_trsf: NonTensorStack(
                            ["b'\\xbe\\xbc\\x7f?8\\x859=/\\x81\\xbe;'", "b'\\x...,
                            batch_size=torch.Size([3]),
                            device=None),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation_trsf: NonTensorStack(
                    ["b'\\x9a\\xbd\\x7f?\\xb8T8=8.c>'", "b'\\xbe\\xbc\...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> env.check_env_specs()
        [torchrl][INFO] check_env_specs succeeded!

    """
    enable_inv_on_reset = True

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey],
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        *,
        fn: Callable[[Any], Tensor | TensorDictBase],
        inv_fn: Callable[[Any], Any] | None = None,
        use_raw_nontensor: bool = False,
    ):
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self._fn = fn
        self._inv_fn = inv_fn
        self._use_raw_nontensor = use_raw_nontensor

    def _apply_transform(self, value):
        if not self._use_raw_nontensor:
            if isinstance(value, NonTensorData):
                if value.dim() == 0:
                    value = value.get("data")
                else:
                    value = value.tolist()
            elif isinstance(value, NonTensorStack):
                value = value.tolist()
        return self._fn(value)

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if not self._use_raw_nontensor:
            if isinstance(state, NonTensorData):
                if state.dim() == 0:
                    state = state.get("data")
                else:
                    state = state.tolist()
            elif isinstance(state, NonTensorStack):
                state = state.tolist()
        if self._inv_fn is not None:
            return self._inv_fn(state)
        return self._fn(state)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        input_spec = input_spec.clone()

        # Make a generic input from the spec, call the transform with that
        # input, and then generate the output spec from the output.
        zero_input_ = input_spec.zero()
        test_input = zero_input_["full_action_spec"].update(
            zero_input_["full_state_spec"]
        )
        # We use forward and not inv because the spec comes from the base env and
        # we are trying to infer what the spec looks like from the outside.
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            data = test_input.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                test_input.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {test_input}")
        test_output = test_input
        # test_output = self.inv(test_input)
        test_input_spec = make_composite_from_td(
            test_output, unsqueeze_null_shapes=False
        )

        input_spec["full_action_spec"] = self.transform_action_spec(
            input_spec["full_action_spec"],
            test_input_spec,
        )
        if "full_state_spec" in input_spec.keys():
            input_spec["full_state_spec"] = self.transform_state_spec(
                input_spec["full_state_spec"],
                test_input_spec,
            )
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_spec = output_spec.clone()

        # Make a generic input from the spec, call the transform with that
        # input, and then generate the output spec from the output.
        zero_input_ = output_spec.zero()
        test_input = (
            zero_input_["full_observation_spec"]
            .update(zero_input_["full_reward_spec"])
            .update(zero_input_["full_done_spec"])
        )
        test_output = self.forward(test_input)
        test_output_spec = make_composite_from_td(
            test_output, unsqueeze_null_shapes=False
        )

        output_spec["full_observation_spec"] = self.transform_observation_spec(
            output_spec["full_observation_spec"],
            test_output_spec,
        )
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self.transform_reward_spec(
                output_spec["full_reward_spec"],
                test_output_spec,
            )
        if "full_done_spec" in output_spec.keys():
            output_spec["full_done_spec"] = self.transform_done_spec(
                output_spec["full_done_spec"],
                test_output_spec,
            )
        return output_spec

    def _transform_spec(
        self, spec: TensorSpec, test_output_spec: TensorSpec, inverse: bool = False
    ) -> TensorSpec:
        if not isinstance(spec, Composite):
            raise TypeError(f"{self}: Only specs of type Composite can be transformed")

        spec_keys = set(spec.keys(include_nested=True))

        iterator = (
            zip(self.in_keys, self.out_keys)
            if not inverse
            else zip(self.in_keys_inv, self.out_keys_inv)
        )
        for in_key, out_key in iterator:
            if in_key in spec_keys:
                spec.set(out_key, test_output_spec[out_key])
        return spec

    def transform_observation_spec(
        self, observation_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(observation_spec, test_output_spec)

    def transform_reward_spec(
        self, reward_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(reward_spec, test_output_spec)

    def transform_done_spec(
        self, done_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(done_spec, test_output_spec)

    def transform_action_spec(
        self, action_spec: TensorSpec, test_input_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(action_spec, test_input_spec, inverse=True)

    def transform_state_spec(
        self, state_spec: TensorSpec, test_input_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(state_spec, test_input_spec, inverse=True)


class Hash(UnaryTransform):
    r"""Adds a hash value to a tensordict.

    Args:
        in_keys (sequence of NestedKey): the keys of the values to hash.
        out_keys (sequence of NestedKey): the keys of the resulting hashes.
        in_keys_inv (sequence of NestedKey, optional): the keys of the values to hash during inv call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the resulting hashes during inv call.

    Keyword Args:
        hash_fn (Callable, optional): the hash function to use. The function
            signature must be
            ``(input: Any, seed: Any | None) -> torch.Tensor``.
            ``seed`` is only used if this transform is initialized with the
            ``seed`` argument.  Default is ``Hash.reproducible_hash``.
        seed (optional): seed to use for the hash function, if it requires one.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before ``fn`` is called
            on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to ``fn``, which must support those
            inputs. Default is ``False``.
        repertoire (Dict[Tuple[int], Any], optional): If given, this dict stores
            the inverse mappings from hashes to inputs. This repertoire isn't
            copied, so it can be modified in the same workspace after the
            transform instantiation and these modifications will be reflected in
            the map. Missing hashes will be mapped to ``None``. Default: ``None``

        >>> from torchrl.envs import GymEnv, UnaryTransform, Hash
        >>> env = GymEnv("Pendulum-v1")
        >>> # Add a string output
        >>> env = env.append_transform(
        ...     UnaryTransform(
        ...         in_keys=["observation"],
        ...         out_keys=["observation_str"],
        ...             fn=lambda tensor: str(tensor.numpy().tobytes())))
        >>> # process the string output
        >>> env = env.append_transform(
        ...     Hash(
        ...         in_keys=["observation_str"],
        ...         out_keys=["observation_hash"],)
        ... )
        >>> env.observation_spec
        Composite(
            observation: BoundedContinuous(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            observation_str: NonTensor(
                shape=torch.Size([]),
                space=None,
                device=cpu,
                dtype=None,
                domain=None),
            observation_hash: UnboundedDiscrete(
                shape=torch.Size([32]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True),
                    high=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True)),
                device=cpu,
                dtype=torch.uint8,
                domain=discrete),
            device=None,
            shape=torch.Size([]))
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
                        observation_str: NonTensorStack(
                            ["b'g\\x08\\x8b\\xbexav\\xbf\\x00\\xee(>'", "b'\\x...,
                            batch_size=torch.Size([3]),
                            device=None),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
                observation_str: NonTensorStack(
                    ["b'\\xb5\\x17\\x8f\\xbe\\x88\\xccu\\xbf\\xc0Vr?'"...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> env.check_env_specs()
        [torchrl][INFO] check_env_specs succeeded!
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey],
        in_keys_inv: Sequence[NestedKey] = None,
        out_keys_inv: Sequence[NestedKey] = None,
        *,
        hash_fn: Callable = None,
        seed: Any | None = None,
        use_raw_nontensor: bool = False,
        repertoire: tuple[tuple[int], Any] = None,
    ):
        if hash_fn is None:
            hash_fn = Hash.reproducible_hash

        if repertoire is None and in_keys_inv is not None and len(in_keys_inv) > 0:
            self._repertoire = {}
        else:
            self._repertoire = repertoire

        self._seed = seed
        self._hash_fn = hash_fn
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            fn=self.call_hash_fn,
            inv_fn=self.get_input_from_hash,
            use_raw_nontensor=use_raw_nontensor,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {"_repertoire": self._repertoire}

    @classmethod
    def hash_to_repertoire_key(cls, hash_tensor):
        if isinstance(hash_tensor, torch.Tensor):
            if hash_tensor.dim() == 0:
                return hash_tensor.tolist()
            return tuple(cls.hash_to_repertoire_key(t) for t in hash_tensor.tolist())
        elif isinstance(hash_tensor, list):
            return tuple(cls.hash_to_repertoire_key(t) for t in hash_tensor)
        else:
            return hash_tensor

    def get_input_from_hash(self, hash_tensor):
        """Look up the input that was given for a particular hash output.

        This feature is only available if, during initialization, either the
        :arg:`repertoire` argument was given or both the :arg:`in_keys_inv` and
        :arg:`out_keys_inv` arguments were given.

        Args:
            hash_tensor (Tensor): The hash output.

        Returns:
            Any: The input that the hash was generated from.
        """
        if self._repertoire is None:
            raise RuntimeError(
                "An inverse transform was queried but the repertoire is None."
            )
        return self._repertoire[self.hash_to_repertoire_key(hash_tensor)]

    def call_hash_fn(self, value):
        if self._seed is None:
            hash_tensor = self._hash_fn(value)
        else:
            hash_tensor = self._hash_fn(value, self._seed)
        if not torch.is_tensor(hash_tensor):
            raise ValueError(
                f"Hash function must return a tensor, but got {type(hash_tensor)}"
            )
        if self._repertoire is not None:
            self._repertoire[self.hash_to_repertoire_key(hash_tensor)] = copy(value)
        return hash_tensor

    @classmethod
    def reproducible_hash(cls, string, seed=None):
        """Creates a reproducible 256-bit hash from a string using a seed.

        Args:
            string (str or None): The input string. If ``None``, null string ``""`` is used.
            seed (str, optional): The seed value. Default is ``None``.

        Returns:
            Tensor: Shape ``(32,)`` with dtype ``torch.uint8``.
        """
        if string is None:
            string = ""

        # Prepend the seed to the string
        if seed is not None:
            seeded_string = seed + string
        else:
            seeded_string = str(string)

        # Create a new SHA-256 hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the seeded string
        hash_object.update(seeded_string.encode("utf-8"))

        # Get the hash value as bytes
        hash_bytes = bytearray(hash_object.digest())

        return torch.frombuffer(hash_bytes, dtype=torch.uint8)


class Tokenizer(UnaryTransform):
    r"""Applies a tokenization operation on the specified inputs.

    Args:
        in_keys (sequence of NestedKey): the keys of inputs to the tokenization operation.
        out_keys (sequence of NestedKey): the keys of the outputs of the tokenization operation.
        in_keys_inv (sequence of NestedKey, optional): the keys of inputs to the tokenization operation during inverse call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the outputs of the tokenization operation during inverse call.

    Keyword Args:
        tokenizer (transformers.PretrainedTokenizerBase or str, optional): the tokenizer to use. If ``None``,
            "bert-base-uncased" will be used by default. If a string is provided, it should be the name of a
            pre-trained tokenizer.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before the tokenization
            function is called on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to the tokenization function, which must support those inputs. Default is ``False``.
        additional_tokens (List[str], optional): list of additional tokens to add to the tokenizer's vocabulary.

    .. note:: This transform can be used both to transform output strings into tokens and to transform back tokenized
        actions or states into strings. If the environment has a string state-spec, the transformed version will have
        a tokenized state-spec. If it is a string action spec, it will result in a tokenized action spec.

    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        *,
        tokenizer: transformers.PretrainedTokenizerBase = None,  # noqa: F821
        use_raw_nontensor: bool = False,
        additional_tokens: list[str] | None = None,
        skip_special_tokens: bool = True,
        add_special_tokens: bool = False,
        padding: bool = True,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        missing_tolerance: bool = True,
        call_before_reset: bool = False,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        elif isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.skip_special_tokens = skip_special_tokens
        self.padding = padding
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
        self.call_before_reset = call_before_reset
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            fn=self.call_tokenizer_fn,
            inv_fn=self.call_tokenizer_inv_fn,
            use_raw_nontensor=use_raw_nontensor,
        )
        self._missing_tolerance = missing_tolerance

    @property
    def device(self):
        if "_device" in self.__dict__:
            return self._device
        parent = self.parent
        if parent is None:
            return None
        device = parent.device
        self._device = device
        return device

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # Specialized for attention mask
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            value = next_tensordict.get(in_key, default=None)
            if value is not None:
                observation = self._apply_transform(value)
                if self.return_attention_mask:
                    observation, attention_mask = observation
                    next_tensordict.set(
                        _replace_last(out_key, "attention_mask"),
                        attention_mask,
                    )
                next_tensordict.set(
                    out_key,
                    observation,
                )
            elif (
                self.missing_tolerance
                and self.return_attention_mask
                and out_key in next_tensordict.keys(True)
            ):
                attention_key = _replace_last(out_key, "attention_mask")
                if attention_key not in next_tensordict:
                    next_tensordict[attention_key] = torch.ones_like(
                        next_tensordict.get(out_key)
                    )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {next_tensordict}"
                )
        return next_tensordict

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            data = tensordict.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                if self.return_attention_mask:
                    data, attention_mask = data
                    tensordict.set(
                        _replace_last(out_key, "attention_mask"),
                        attention_mask,
                    )
                tensordict.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.call_before_reset:
            with _set_missing_tolerance(self, True):
                tensordict = self._call(tensordict)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.call_before_reset:
            return tensordict_reset
        return super()._reset(tensordict, tensordict_reset)

    def call_tokenizer_fn(self, value: str | list[str]):
        device = self.device
        kwargs = {"add_special_tokens": self.add_special_tokens}
        if self.max_length is not None:
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = self.max_length
        if isinstance(value, str):
            out = self.tokenizer.encode(value, return_tensors="pt", **kwargs)[0]
            # TODO: incorporate attention mask
            if self.return_attention_mask:
                attention_mask = torch.ones_like(out, dtype=torch.int64)
        else:
            kwargs["padding"] = (
                self.padding if self.max_length is None else "max_length"
            )
            kwargs["return_attention_mask"] = self.return_attention_mask
            # kwargs["return_token_type_ids"] = False
            out = self.tokenizer.batch_encode_plus(value, return_tensors="pt", **kwargs)
            if self.return_attention_mask:
                attention_mask = out["attention_mask"]
            out = out["input_ids"]

        if device is not None and out.device != device:
            out = out.to(device)
            if self.return_attention_mask:
                attention_mask = attention_mask.to(device)
        if self.return_attention_mask:
            return out, attention_mask
        return out

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Override _inv_call to account for ragged dims
        if not self.in_keys_inv:
            return tensordict
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            data = tensordict.get(out_key, None, as_padded_tensor=True)
            if data is not None:
                item = self._inv_apply_transform(data)
                tensordict.set(in_key, item)
            elif not self.missing_tolerance:
                raise KeyError(f"'{out_key}' not found in tensordict {tensordict}")
        return tensordict

    def call_tokenizer_inv_fn(self, value: Tensor):
        if value.ndim == 1:
            out = self.tokenizer.decode(
                value.int(), skip_special_tokens=self.skip_special_tokens
            )
        else:
            out = self.tokenizer.batch_decode(
                value.int(), skip_special_tokens=self.skip_special_tokens
            )
        device = self._str_device
        if isinstance(out, list):
            result = NonTensorStack(*out)
            if device:
                result = result.to(device)
            return result
        return NonTensorData(out, device=device)

    @property
    def _str_device(self):
        parent = self.parent
        if parent is None:
            return None
        if self.in_keys:
            in_key = self.in_keys[0]
        elif self.in_keys_inv:
            in_key = self.in_keys_inv[0]
        else:
            return None
        if in_key in parent.observation_keys:
            return parent.full_observation_spec[in_key].device
        if in_key in parent.action_keys:
            return parent.full_action_spec[in_key].device
        if in_key in parent.state_keys:
            return parent.full_state_spec[in_key].device
        return None

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        # We need to cap the spec to generate valid random strings
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            if in_key in input_spec["full_state_spec"].keys(True, True):
                spec = input_spec["full_state_spec"]
            elif in_key in input_spec["full_action_spec"].keys(False, True):
                spec = input_spec["full_action_spec"]
            else:
                raise KeyError(
                    f"The input keys {in_key} wasn't found in the env input specs."
                )
            local_spec = spec.pop(in_key)
            local_dtype = local_spec.dtype
            if local_dtype is None or local_dtype.is_floating_point:
                local_dtype = torch.int64
            new_shape = spec.shape
            if self.max_length is None:
                # Then we can't tell what the shape will be
                new_shape = new_shape + torch.Size((-1,))
            else:
                new_shape = new_shape + torch.Size((self.max_length,))
            spec[out_key] = Bounded(
                0,
                self.tokenizer.vocab_size,
                shape=new_shape,
                device=local_spec.device,
                dtype=local_dtype,
            )
        return input_spec

    transform_output_spec = Transform.transform_output_spec
    transform_reward_spec = Transform.transform_reward_spec
    transform_done_spec = Transform.transform_done_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        attention_mask_keys = set()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            new_shape = observation_spec.shape + torch.Size((-1,))
            try:
                in_spec = observation_spec[in_key]
                obs_dtype = in_spec.dtype
                device = in_spec.device
            except KeyError:
                # In some cases (eg, the tokenizer is applied during reset on data that
                #  originates from a dataloader) we don't have an in_spec
                in_spec = None
                obs_dtype = None
                device = observation_spec.device
            if obs_dtype is None or obs_dtype.is_floating_point:
                obs_dtype = torch.int64
            observation_spec[out_key] = Bounded(
                0,
                self.tokenizer.vocab_size,
                shape=new_shape,
                device=device,
                dtype=obs_dtype,
            )
            if self.return_attention_mask:
                attention_mask_key = _replace_last(out_key, "attention_mask")
                if attention_mask_key in attention_mask_keys:
                    raise KeyError(
                        "Conflicting attention_mask keys. Make sure the token tensors are "
                        "nested at different places in the tensordict such that `(*root, 'attention_mask')` "
                        "entries are unique."
                    )
                attention_mask_keys.add(attention_mask_key)
                attention_dtype = obs_dtype
                if attention_dtype is None or attention_dtype.is_floating_point:
                    attention_dtype = torch.int64
                observation_spec[attention_mask_key] = Bounded(
                    0,
                    2,
                    shape=new_shape,
                    device=device,
                    dtype=attention_dtype,
                )
        return observation_spec


class Stack(Transform):
    """Stacks tensors and tensordicts.

    Concatenates a sequence of tensors or tensordicts along a new dimension.
    The tensordicts or tensors under ``in_keys`` must all have the same shapes.

    This transform only stacks the inputs into one output key. Stacking multiple
    groups of input keys into different output keys requires multiple
    transforms.

    This transform can be useful for environments that have multiple agents with
    identical specs under different keys. The specs and tensordicts for the
    agents can be stacked together under a shared key, in order to run MARL
    algorithms that expect the tensors for observations, rewards, etc. to
    contain batched data for all the agents.

    Args:
        in_keys (sequence of NestedKey): keys to be stacked.
        out_key (NestedKey): key of the resulting stacked entry.
        in_key_inv (NestedKey, optional): key to unstack during :meth:`~.inv`
            calls. Default is ``None``.
        out_keys_inv (sequence of NestedKey, optional): keys of the resulting
            unstacked entries after :meth:`~.inv` calls. Default is ``None``.
        dim (int, optional): dimension to insert. Default is ``-1``.
        allow_positive_dim (bool, optional): if ``True``, positive dimensions
            are accepted.  Defaults to ``False``, ie. non-negative dimensions are
            not permitted.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted
            after stacking. Default is ``True``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import Stack
        >>> td = TensorDict({"key1": torch.zeros(3), "key2": torch.ones(3)}, [])
        >>> td
        TensorDict(
            fields={
                key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> transform = Stack(in_keys=["key1", "key2"], out_key="out", dim=-2)
        >>> transform(td)
        TensorDict(
            fields={
                out: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> td["out"]
        tensor([[0., 0., 0.],
                [1., 1., 1.]])

        >>> agent_0 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
        >>> agent_1 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
        >>> td = TensorDict({"agent_0": agent_0, "agent_1": agent_1})
        >>> transform = Stack(in_keys=["agent_0", "agent_1"], out_key="agents")
        >>> transform(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        obs: Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """

    invertible = True

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_key: NestedKey,
        in_key_inv: NestedKey | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        dim: int = -1,
        allow_positive_dim: bool = False,
        *,
        del_keys: bool = True,
    ):
        if not allow_positive_dim and dim >= 0:
            raise ValueError(
                "dim should be negative to accommodate for envs of different "
                "batch_sizes. If you need dim to be positive, set "
                "allow_positive_dim=True."
            )

        if in_key_inv is None and out_keys_inv is not None:
            raise ValueError("out_keys_inv was specified, but in_key_inv was not")
        elif in_key_inv is not None and out_keys_inv is None:
            raise ValueError("in_key_inv was specified, but out_keys_inv was not")

        super().__init__(
            in_keys=in_keys,
            out_keys=[out_key],
            in_keys_inv=None if in_key_inv is None else [in_key_inv],
            out_keys_inv=out_keys_inv,
        )

        for in_key in self.in_keys:
            if len(in_key) == len(self.out_keys[0]):
                if all(k1 == k2 for k1, k2 in zip(in_key, self.out_keys[0])):
                    raise ValueError(f"{self}: out_key cannot be in in_keys")
        parent_keys = []
        for key in self.in_keys:
            if isinstance(key, (list, tuple)):
                for parent_level in range(1, len(key)):
                    parent_key = tuple(key[:-parent_level])
                    if parent_key not in parent_keys:
                        parent_keys.append(parent_key)
        self._maybe_del_parent_keys = sorted(parent_keys, key=len, reverse=True)
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        values = []
        for in_key in self.in_keys:
            value = next_tensordict.get(in_key, default=None)
            if value is not None:
                values.append(value)
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {next_tensordict}"
                )

        out_tensor = torch.stack(values, dim=self.dim)
        next_tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            next_tensordict.exclude(*self.in_keys, inplace=True)
            for parent_key in self._maybe_del_parent_keys:
                if len(next_tensordict[parent_key].keys()) == 0:
                    next_tensordict.exclude(parent_key, inplace=True)
        return next_tensordict

    forward = _call

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if len(self.in_keys_inv) == 0:
            return tensordict

        if self.in_keys_inv[0] not in tensordict.keys(include_nested=True):
            return tensordict
        values = torch.unbind(tensordict[self.in_keys_inv[0]], dim=self.dim)
        for value, out_key_inv in _zip_strict(values, self.out_keys_inv):
            tensordict = tensordict.set(out_key_inv, value)
        return tensordict.exclude(self.in_keys_inv[0])

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _transform_spec(self, spec: TensorSpec) -> TensorSpec:
        if not isinstance(spec, Composite):
            raise TypeError(f"{self}: Only specs of type Composite can be transformed")

        spec_keys = spec.keys(include_nested=True)
        keys_to_stack = [key for key in spec_keys if key in self.in_keys]
        specs_to_stack = [spec[key] for key in keys_to_stack]

        if len(specs_to_stack) == 0:
            return spec

        stacked_specs = torch.stack(specs_to_stack, dim=self.dim)
        spec.set(self.out_keys[0], stacked_specs)

        if self._del_keys:
            for key in keys_to_stack:
                del spec[key]
            for parent_key in self._maybe_del_parent_keys:
                if len(spec[parent_key]) == 0:
                    del spec[parent_key]

        return spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        self._transform_spec(input_spec["full_state_spec"])
        self._transform_spec(input_spec["full_action_spec"])
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(observation_spec)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(reward_spec)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(done_spec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_keys={self.in_keys}, "
            f"out_key={self.out_keys[0]}, "
            f"dim={self.dim}"
            ")"
        )


class DiscreteActionProjection(Transform):
    """Projects discrete actions from a high dimensional space to a low dimensional space.

    Given a discrete action (from 1 to N) encoded as a one-hot vector and a
    maximum action index num_actions (with num_actions < N), transforms the action such that
    action_out is at most num_actions.

    If the input action is > num_actions, it is being replaced by a random value
    between 0 and num_actions-1. Otherwise the same action is kept.
    This is intended to be used with policies applied over multiple discrete
    control environments with different action space.

    A call to DiscreteActionProjection.forward (eg from a replay buffer or in a
    sequence of nn.Modules) will call the transform num_actions_effective -> max_actions
    on the :obj:`"in_keys"`, whereas a call to _call will be ignored. Indeed,
    transformed envs are instructed to update the input keys only for the inner
    base_env, but the original input keys will remain unchanged.

    Args:
        num_actions_effective (int): max number of action considered.
        max_actions (int): maximum number of actions that this module can read.
        action_key (NestedKey, optional): key name of the action. Defaults to "action".
        include_forward (bool, optional): if ``True``, a call to forward will also
            map the action from one domain to the other when the module is called
            by a replay buffer or an nn.Module chain. Defaults to `True`.

    Examples:
        >>> torch.manual_seed(0)
        >>> N = 3
        >>> M = 2
        >>> action = torch.zeros(N, dtype=torch.long)
        >>> action[-1] = 1
        >>> td = TensorDict({"action": action}, [])
        >>> transform = DiscreteActionProjection(num_actions_effective=M, max_actions=N)
        >>> _ = transform.inv(td)
        >>> print(td.get("action"))
        tensor([1])
    """

    def __init__(
        self,
        num_actions_effective: int,
        max_actions: int,
        action_key: NestedKey = "action",
        include_forward: bool = True,
    ):
        in_keys_inv = [action_key]
        if include_forward:
            in_keys = in_keys_inv
        else:
            in_keys = []
        if in_keys_inv is None:
            in_keys_inv = []
        super().__init__(
            in_keys=in_keys,
            out_keys=copy(in_keys),
            in_keys_inv=in_keys_inv,
            out_keys_inv=copy(in_keys_inv),
        )
        self.num_actions_effective = num_actions_effective
        self.max_actions = max_actions
        if max_actions < num_actions_effective:
            raise RuntimeError(
                "The `max_actions` int must be greater or equal to `num_actions_effective`."
            )

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # We don't do anything here because the action is modified by the inv
        # method but we don't need to map it back as it won't be updated in the original
        # tensordict
        return next_tensordict

    def _apply_transform(self, action: torch.Tensor) -> None:
        # We still need to code the forward transform for replay buffers and models
        action = action.argmax(-1)  # bool to int
        action = nn.functional.one_hot(action, self.max_actions)
        return action

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.max_actions:
            raise RuntimeError(
                f"action.shape[-1]={action.shape[-1]} must match self.max_actions={self.max_actions}."
            )
        action = action.long().argmax(-1)  # bool to int
        idx = action >= self.num_actions_effective
        if idx.any():
            action[idx] = torch.randint(self.num_actions_effective, (idx.sum(),))
        action = nn.functional.one_hot(action, self.num_actions_effective)
        return action

    def transform_input_spec(self, input_spec: Composite):
        input_spec = input_spec.clone()
        for key in input_spec["full_action_spec"].keys(True, True):
            key = ("full_action_spec", key)
            break
        else:
            raise KeyError("key not found in action_spec.")
        input_spec[key] = OneHot(
            self.max_actions,
            shape=(*input_spec[key].shape[:-1], self.max_actions),
            device=input_spec.device,
            dtype=input_spec[key].dtype,
        )
        return input_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_actions_effective={self.num_actions_effective}, max_actions={self.max_actions}, "
            f"in_keys_inv={self.in_keys_inv})"
        )


class FrameSkipTransform(Transform):
    """A frame-skip transform.

    This transform applies the same action repeatedly in the parent environment,
    which improves stability on certain training sota-implementations.

    Args:
        frame_skip (int, optional): a positive integer representing the number
            of frames during which the same action must be applied.

    """

    def __init__(self, frame_skip: int = 1):
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip should have a value greater or equal to one.")
        self.frame_skip = frame_skip

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError("parent not found for FrameSkipTransform")
        reward_key = parent.reward_key
        reward = next_tensordict.get(reward_key)
        for _ in range(self.frame_skip - 1):
            next_tensordict = parent._step(tensordict)
            reward = reward + next_tensordict.get(reward_key)
        return next_tensordict.set(reward_key, reward)

    def forward(self, tensordict):
        raise RuntimeError(
            "FrameSkipTransform can only be used when appended to a transformed env."
        )


class NoopResetEnv(Transform):
    """Runs a series of random actions when an environment is reset.

    Args:
        env (EnvBase): env on which the random actions have to be
            performed. Can be the same env as the one provided to the
            TransformedEnv class
        noops (int, optional): upper-bound on the number of actions
            performed after reset. Default is `30`.
            If noops is too high such that it results in the env being
            done or truncated before the all the noops are applied,
            in multiple trials, the transform raises a RuntimeError.
        random (bool, optional): if False, the number of random ops will
            always be equal to the noops value. If True, the number of
            random actions will be randomly selected between 0 and noops.
            Default is `True`.

    """

    def __init__(self, noops: int = 30, random: bool = True):
        """Sample initial states by taking random number of no-ops on reset."""
        super().__init__()
        self.noops = noops
        self.random = random

    @property
    def base_env(self):
        return self.parent

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Do no-op action for a number of steps in [1, noop_max]."""
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "NoopResetEnv.parent not found. Make sure that the parent is set."
            )
        # Merge the two tensordicts
        tensordict = parent._reset_proc_data(tensordict.clone(False), tensordict_reset)
        # check that there is a single done state -- behavior is undefined for multiple dones
        done_keys = parent.done_keys
        reward_key = parent.reward_key
        if parent.batch_size.numel() > 1:
            raise ValueError(
                "The parent environment batch-size is non-null. "
                "NoopResetEnv is designed to work on single env instances, as partial reset "
                "is currently not supported. If you feel like this is a missing feature, submit "
                "an issue on TorchRL github repo. "
                "In case you are trying to use NoopResetEnv over a batch of environments, know "
                "that you can have a transformed batch of transformed envs, such as: "
                "`TransformedEnv(ParallelEnv(3, lambda: TransformedEnv(MyEnv(), NoopResetEnv(3))), OtherTransform())`."
            )

        noops = (
            self.noops if not self.random else torch.randint(self.noops, (1,)).item()
        )

        trial = 0
        while trial <= _MAX_NOOPS_TRIALS:
            i = 0

            while i < noops:
                i += 1
                tensordict = parent.rand_step(tensordict)
                reset = False
                # if any of the done_keys is True, we break
                for done_key in done_keys:
                    done = tensordict.get(("next", done_key))
                    if done.numel() > 1:
                        raise ValueError(
                            f"{type(self)} only supports scalar done states."
                        )
                    if done:
                        reset = True
                        break
                tensordict = step_mdp(tensordict, exclude_done=False)
                if reset:
                    tensordict = parent.reset(tensordict.clone(False))
                    break
            else:
                break

            trial += 1

        else:
            raise RuntimeError(
                f"Parent env was repeatedly done or truncated"
                f" before the sampled number of noops (={noops}) could be applied. "
            )
        tensordict_reset = tensordict
        return tensordict_reset.exclude(reward_key, inplace=True)

    def __repr__(self) -> str:
        random = self.random
        noops = self.noops
        class_name = self.__class__.__name__
        return f"{class_name}(noops={noops}, random={random})"


class TensorDictPrimer(Transform):
    """A primer for TensorDict initialization at reset time.

    This transform will populate the tensordict at reset with values drawn from
    the relative tensorspecs provided at initialization.
    If the transform is used out of the env context (e.g. as an nn.Module or
    appended to a replay buffer), a call to `forward` will also populate the
    tensordict with the desired features.

    Args:
        primers (dict or Composite, optional): a dictionary containing
            key-spec pairs which will be used to populate the input tensordict.
            :class:`~torchrl.data.Composite` instances are supported too.
        random (bool, optional): if ``True``, the values will be drawn randomly from
            the TensorSpec domain (or a unit Gaussian if unbounded). Otherwise a fixed value will be assumed.
            Defaults to `False`.
        default_value (:obj:`float`, Callable, Dict[NestedKey, float], Dict[NestedKey, Callable], optional): If non-random
            filling is chosen, `default_value` will be used to populate the tensors.

            - If `default_value` is a float or any other scala, all elements of the tensors will be set to that value.
            - If it is a callable and `single_default_value=False` (default), this callable is expected to return a tensor
              fitting the specs (ie, ``default_value()`` will be called independently for each leaf spec).
            - If it is a callable and ``single_default_value=True``, then the callable will be called just once and it is expected
              that the structure of its returned TensorDict instance or equivalent will match the provided specs.
              The ``default_value`` must accept an optional `reset` keyword argument indicating which envs are to be reset.
              The returned `TensorDict` must have as many elements as the number of envs to reset.

              .. seealso:: :class:`~torchrl.envs.DataLoadingPrimer`

            - Finally, if `default_value` is a dictionary of tensors or a dictionary of callables with keys matching
              those of the specs, these will be used to generate the corresponding tensors. Defaults to `0.0`.

        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.
        single_default_value (bool, optional): if ``True`` and `default_value` is a callable, it will be expected that
            ``default_value`` returns a single tensordict matching the specs. If `False`, `default_value()` will be
            called independently for each leaf. Defaults to ``False``.
        call_before_env_reset (bool, optional): if ``True``, the tensordict is populated before `env.reset` is called.
            Defaults to ``False``.
        **kwargs: each keyword argument corresponds to a key in the tensordict.
            The corresponding value has to be a TensorSpec instance indicating
            what the value must be.

    When used in a `TransformedEnv`, the spec shapes must match the environment's shape if
    the parent environment is batch-locked (`env.batch_locked=True`). If the spec shapes and
    parent shapes do not match, the spec shapes are modified in-place to match the leading
    dimensions of the parent's batch size. This adjustment is made for cases where the parent
    batch size dimension is not known during instantiation.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import SerialEnv
        >>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(base_env)
        >>> # the env is batch-locked, so the leading dims of the spec must match those of the env
        >>> env.append_transform(TensorDictPrimer(mykey=Unbounded([2, 3])))
        >>> td = env.reset()
        >>> print(td)
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                mykey: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> # the entry is populated with 0s
        >>> print(td.get("mykey"))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

    When calling ``env.step()``, the current value of the key will be carried
    in the ``"next"`` tensordict __unless it already exists__.

    Examples:
        >>> td = env.rand_step(td)
        >>> print(td.get(("next", "mykey")))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
        >>> # with another value for "mykey", the previous value is not carried on
        >>> td = env.reset()
        >>> td = td.set(("next", "mykey"), torch.ones(2, 3))
        >>> td = env.rand_step(td)
        >>> print(td.get(("next", "mykey")))
        tensor([[1., 1., 1.],
                [1., 1., 1.]])

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import SerialEnv, TransformedEnv
        >>> from torchrl.modules.utils import get_primers_from_module
        >>> from torchrl.modules import GRUModule
        >>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(base_env)
        >>> model = GRUModule(input_size=2, hidden_size=2, in_key="observation", out_key="action")
        >>> primers = get_primers_from_module(model)
        >>> print(primers) # Primers shape is independent of the env batch size
        TensorDictPrimer(primers=Composite(
            recurrent_state: UnboundedContinuous(
                shape=torch.Size([1, 2]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            device=None,
            shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)
        >>> env.append_transform(primers)
        >>> print(env.reset()) # The primers are automatically expanded to match the env batch size
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                recurrent_state: Tensor(shape=torch.Size([2, 1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)

    .. note:: Some TorchRL modules rely on specific keys being present in the environment TensorDicts,
        like :class:`~torchrl.modules.models.LSTM` or :class:`~torchrl.modules.models.GRU`.
        To facilitate this process, the method :func:`~torchrl.modules.utils.get_primers_from_module`
        automatically checks for required primer transforms in a module and its submodules and
        generates them.
    """

    def __init__(
        self,
        primers: dict | Composite | None = None,
        random: bool | None = None,
        default_value: float
        | Callable
        | dict[NestedKey, float]
        | dict[NestedKey, Callable]
        | None = None,
        reset_key: NestedKey | None = None,
        expand_specs: bool | None = None,
        single_default_value: bool = False,
        call_before_env_reset: bool = False,
        **kwargs,
    ):
        self.device = kwargs.pop("device", None)
        if primers is not None:
            if kwargs:
                raise RuntimeError(
                    f"providing the primers as a dictionary is incompatible with extra keys "
                    f"'{kwargs.keys()}' provided as kwargs."
                )
            kwargs = primers
        if not isinstance(kwargs, Composite):
            shape = kwargs.pop("shape", None)
            device = self.device
            if "batch_size" in kwargs.keys():
                extra_kwargs = {"batch_size": kwargs.pop("batch_size")}
            else:
                extra_kwargs = {}
            primers = Composite(kwargs, device=device, shape=shape, **extra_kwargs)
        self.primers = primers
        self.expand_specs = expand_specs
        self.call_before_env_reset = call_before_env_reset

        if random and default_value:
            raise ValueError(
                "Setting random to True and providing a default_value are incompatible."
            )
        default_value = (
            default_value or 0.0
        )  # if not random and no default value, use 0.0
        self.random = random
        if isinstance(default_value, dict):
            default_value = TensorDict(default_value, [])
            default_value_keys = default_value.keys(
                True,
                True,
                is_leaf=lambda x: issubclass(x, (NonTensorData, torch.Tensor)),
            )
            if set(default_value_keys) != set(self.primers.keys(True, True)):
                raise ValueError(
                    "If a default_value dictionary is provided, it must match the primers keys."
                )
        elif single_default_value:
            pass
        else:
            default_value = {
                key: default_value for key in self.primers.keys(True, True)
            }
        self.single_default_value = single_default_value
        self.default_value = default_value
        self._validated = False
        self.reset_key = reset_key

        # sanity check
        for spec in self.primers.values(True, True):
            if not isinstance(spec, TensorSpec):
                raise ValueError(
                    "The values of the primers must be a subtype of the TensorSpec class. "
                    f"Got {type(spec)} instead."
                )
        super().__init__()

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key")
        if reset_key is None:
            if self.parent is None:
                raise RuntimeError(
                    "Missing parent, cannot infer reset_key automatically."
                )
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. "
                    f"Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    @property
    def device(self):
        device = self._device
        if device is None and hasattr(self, "parent") and self.parent is not None:
            device = self.parent.device
            self._device = device
        return device

    @device.setter
    def device(self, value):
        if value is None:
            self._device = None
            return
        self._device = torch.device(value)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
            self.empty_cache()
            self.primers = self.primers.to(device)
        return super().to(*args, **kwargs)

    def _expand_shape(self, spec):
        return spec.expand((*self.parent.batch_size, *spec.shape))

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )

        if self.primers.shape[: observation_spec.ndim] != observation_spec.shape:
            if self.expand_specs:
                self.primers = self._expand_shape(self.primers)
            elif self.expand_specs is None:
                raise RuntimeError(
                    f"expand_specs wasn't specified in the {type(self).__name__} constructor, and the shape of the primers "
                    f"and observation specs mismatch ({self.primers.shape=} and {observation_spec.shape=}) - indicating a batch-size incongruency. Make sure the expand_specs arg "
                    f"is properly set or that the primer shape matches the environment batch-size."
                )
            else:
                self.primers.shape = observation_spec.shape

        device = observation_spec.device
        observation_spec.update(self.primers.clone().to(device))
        return observation_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        new_state_spec = self.transform_observation_spec(input_spec["full_state_spec"])
        for action_key in list(input_spec["full_action_spec"].keys(True, True)):
            if action_key in new_state_spec.keys(True, True):
                input_spec["full_action_spec", action_key] = new_state_spec[action_key]
                del new_state_spec[action_key]
        input_spec["full_state_spec"] = new_state_spec
        return input_spec

    @property
    def _batch_size(self):
        return self.parent.batch_size

    def _validate_value_tensor(self, value, spec):
        if not spec.is_in(value):
            raise RuntimeError(f"Value ({value}) is not in the spec domain ({spec}).")
        return True

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.single_default_value and callable(self.default_value):
            tensordict.update(self.default_value())
            for key, spec in self.primers.items(True, True):
                if not self._validated:
                    self._validate_value_tensor(tensordict.get(key), spec)
            if not self._validated:
                self._validated = True
            return tensordict
        for key, spec in self.primers.items(True, True):
            if spec.shape[: len(tensordict.shape)] != tensordict.shape:
                raise RuntimeError(
                    "The leading shape of the spec must match the tensordict's, "
                    "but it does not: got "
                    f"tensordict.shape={tensordict.shape} whereas {key} spec's shape is "
                    f"{spec.shape}."
                )
            if self.random:
                value = spec.rand()
            else:
                value = self.default_value[key]
                if callable(value):
                    value = value()
                    if not self._validated:
                        self._validate_value_tensor(value, spec)
                else:
                    value = torch.full(
                        spec.shape,
                        value,
                        device=spec.device,
                    )

            tensordict.set(key, value)
        if not self._validated:
            self._validated = True
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for key in self.primers.keys(True, True):
            # We relax a bit the condition here, allowing nested but not leaf values to
            #  be checked against
            if key not in next_tensordict.keys(True, is_leaf=_is_leaf_nontensor):
                prev_val = tensordict.get(key)
                next_tensordict.set(key, prev_val)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Sets the default values in the input tensordict.

        If the parent is batch-locked, we make sure the specs have the appropriate leading
        shape. We allow for execution when the parent is missing, in which case the
        spec shape is assumed to match the tensordict's.
        """
        if self.call_before_env_reset:
            return tensordict_reset
        return self._reset_func(tensordict, tensordict_reset)

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.call_before_env_reset:
            return tensordict
        if tensordict is None:
            parent = self.parent
            if parent is not None:
                device = parent.device
                batch_size = parent.batch_size
            else:
                device = None
                batch_size = ()
            tensordict = TensorDict(device=device, batch_size=batch_size)
        return self._reset_func(tensordict, tensordict)

    def _reset_func(
        self, tensordict, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        _reset = _get_reset(self.reset_key, tensordict)
        if (
            self.parent
            and self.parent.batch_locked
            and self.primers.shape[: len(self.parent.shape)] != self.parent.batch_size
        ):
            self.primers = self._expand_shape(self.primers)
        if _reset.any():
            if self.single_default_value and callable(self.default_value):
                if not _reset.all():
                    # FIXME: use masked op
                    tensordict_reset = tensordict_reset.clone()
                    reset_val = self.default_value(reset=_reset)
                    # This is safe because env.reset calls _update_during_reset which will discard the new data
                    tensordict_reset = (
                        self.container.full_observation_spec.zero().select(
                            *reset_val.keys(True)
                        )
                    )
                    tensordict_reset[_reset] = reset_val
                else:
                    resets = self.default_value(reset=_reset)
                    tensordict_reset.update(resets)

                for key, spec in self.primers.items(True, True):
                    if not self._validated:
                        self._validate_value_tensor(tensordict_reset.get(key), spec)
                self._validated = True
                return tensordict_reset

            for key, spec in self.primers.items(True, True):
                if self.random:
                    shape = (
                        ()
                        if (not self.parent or self.parent.batch_locked)
                        else tensordict.batch_size
                    )
                    value = spec.rand(shape)
                else:
                    value = self.default_value[key]
                    if callable(value):
                        value = value()
                        if not self._validated:
                            self._validate_value_tensor(value, spec)
                    else:
                        value = torch.full(
                            spec.shape,
                            value,
                            device=spec.device,
                        )
                        prev_val = tensordict.get(key, 0.0)
                        value = torch.where(
                            expand_as_right(_reset, value), value, prev_val
                        )
                tensordict_reset.set(key, value)
            self._validated = True
        return tensordict_reset

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if callable(self.default_value):
            default_value = self.default_value
        else:
            default_value = {
                key: value if isinstance(value, float) else "Callable"
                for key, value in self.default_value.items()
            }
        return f"{class_name}(primers={self.primers}, default_value={default_value}, random={self.random})"


class PinMemoryTransform(Transform):
    """Calls pin_memory on the tensordict to facilitate writing on CUDA devices."""

    def __init__(self):
        super().__init__()

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict.pin_memory()

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


def _sum_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.sum(0)
    return val


class gSDENoise(TensorDictPrimer):
    """A gSDE noise initializer.

    See the :func:`~torchrl.modules.models.exploration.gSDEModule' for more info.
    """

    def __init__(
        self,
        state_dim=None,
        action_dim=None,
        shape=None,
        **kwargs,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        if shape is None:
            shape = ()
        tail_dim = (
            (1,) if state_dim is None or action_dim is None else (action_dim, state_dim)
        )
        random = state_dim is not None and action_dim is not None
        feat_shape = tuple(shape) + tail_dim
        primers = Composite({"_eps_gSDE": Unbounded(shape=feat_shape)}, shape=shape)
        super().__init__(primers=primers, random=random, **kwargs)


class _VecNormMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        new_api = kwargs.pop("new_api", None)
        if new_api is None:
            warnings.warn(
                "The VecNorm class is to be deprecated in favor of `torchrl.envs.VecNormV2` and will be replaced by "
                "that class in v0.10. You can adapt to these changes by using the `new_api` argument or importing "
                "the `VecNormV2` class from `torchrl.envs`.",
                category=FutureWarning,
            )
            new_api = False
        if new_api:
            from torchrl.envs import VecNormV2

            return VecNormV2(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class VecNorm(Transform, metaclass=_VecNormMeta):
    """Moving average normalization layer for torchrl environments.

    .. warning:: This class is to be deprecated in favor of :class:`~torchrl.envs.VecNormV2` and will be replaced by
        that class in v0.10. You can adapt to these changes by using the `new_api` argument or importing the
        `VecNormV2` class from `torchrl.envs`.

    VecNorm keeps track of the summary statistics of a dataset to standardize
    it on-the-fly. If the transform is in 'eval' mode, the running
    statistics are not updated.

    If multiple processes are running a similar environment, one can pass a
    TensorDictBase instance that is placed in shared memory: if so, every time
    the normalization layer is queried it will update the values for all
    processes that share the same reference.

    To use VecNorm at inference time and avoid updating the values with the new
    observations, one should substitute this layer by :meth:`~.to_observation_norm`.
    This will provide a static version of `VecNorm` which will not be updated
    when the source transform is updated.
    To get a frozen copy of the VecNorm layer, see :meth:`~.frozen_copy`.

    Args:
        in_keys (sequence of NestedKey, optional): keys to be updated.
            default: ["observation", "reward"]
        out_keys (sequence of NestedKey, optional): destination keys.
            Defaults to ``in_keys``.
        shared_td (TensorDictBase, optional): A shared tensordict containing the
            keys of the transform.
        lock (mp.Lock): a lock to prevent race conditions between processes.
            Defaults to None (lock created during init).
        decay (number, optional): decay rate of the moving average.
            default: 0.99
        eps (number, optional): lower bound of the running standard
            deviation (for numerical underflow). Default is 1e-4.
        shapes (List[torch.Size], optional): if provided, represents the shape
            of each in_keys. Its length must match the one of ``in_keys``.
            Each shape must match the trailing dimension of the corresponding
            entry.
            If not, the feature dimensions of the entry (ie all dims that do
            not belong to the tensordict batch-size) will be considered as
            feature dimension.
        new_api (bool or None, optional): if ``True``, an instance of VecNormV2 will be returned.
            If not passed, a warning will be raised.
            Defaults to ``False``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> t = VecNorm(decay=0.9)
        >>> env = GymEnv("Pendulum-v0")
        >>> env = TransformedEnv(env, t)
        >>> tds = []
        >>> for _ in range(1000):
        ...     td = env.rand_step()
        ...     if td.get("done"):
        ...         _ = env.reset()
        ...     tds += [td]
        >>> tds = torch.stack(tds, 0)
        >>> print((abs(tds.get(("next", "observation")).mean(0))<0.2).all())
        tensor(True)
        >>> print((abs(tds.get(("next", "observation")).std(0)-1)<0.2).all())
        tensor(True)

    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shared_td: TensorDictBase | None = None,
        lock: mp.Lock = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
        shapes: list[torch.Size] = None,
        new_api: bool | None = None,
    ) -> None:

        warnings.warn(
            "This class is to be deprecated in favor of :class:`~torchrl.envs.VecNormV2`.",
            category=FutureWarning,
        )

        if lock is None:
            lock = mp.Lock()
        if in_keys is None:
            in_keys = ["observation", "reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._td = shared_td
        if shared_td is not None and not (
            shared_td.is_shared() or shared_td.is_memmap()
        ):
            raise RuntimeError(
                "shared_td must be either in shared memory or a memmap " "tensordict."
            )
        if shared_td is not None:
            for key in in_keys:
                if (
                    (_append_last(key, "_sum") not in shared_td.keys())
                    or (_append_last(key, "_ssq") not in shared_td.keys())
                    or (_append_last(key, "_count") not in shared_td.keys())
                ):
                    raise KeyError(
                        f"key {key} not present in the shared tensordict "
                        f"with keys {shared_td.keys()}"
                    )

        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps
        self.frozen = False

    def freeze(self) -> VecNorm:
        """Freezes the VecNorm, avoiding the stats to be updated when called.

        See :meth:`~.unfreeze`.
        """
        self.frozen = True
        return self

    def unfreeze(self) -> VecNorm:
        """Unfreezes the VecNorm.

        See :meth:`~.freeze`.
        """
        self.frozen = False
        return self

    def frozen_copy(self):
        """Returns a copy of the Transform that keeps track of the stats but does not update them."""
        if self._td is None:
            raise RuntimeError(
                "Make sure the VecNorm has been initialized before creating a frozen copy."
            )
        clone = self.clone()
        # replace values
        clone._td = self._td.copy()
        # freeze
        return clone.freeze()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: remove this decorator when trackers are in data
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)
        return tensordict_reset

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key, key_out in _zip_strict(self.in_keys, self.out_keys):
            if key not in next_tensordict.keys(include_nested=True):
                # TODO: init missing rewards with this
                # for key_suffix in [_append_last(key, suffix) for suffix in ("_sum", "_ssq", "_count")]:
                #     tensordict.set(key_suffix, self.container.observation_spec[key_suffix].zero())
                continue
            self._init(next_tensordict, key)
            # update and standardize
            new_val = self._update(
                key, next_tensordict.get(key), N=max(1, next_tensordict.numel())
            )

            next_tensordict.set(key_out, new_val)

        if self.lock is not None:
            self.lock.release()

        return next_tensordict

    forward = _call

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        if self._td is None or _append_last(key, "_sum") not in self._td.keys(True):
            if key is not key and key in tensordict.keys():
                raise RuntimeError(
                    f"Conflicting key names: {key} from VecNorm and input tensordict keys."
                )
            if self.shapes is None:
                td_view = tensordict.view(-1)
                td_select = td_view[0]
                item = td_select.get(key)
                d = {_append_last(key, "_sum"): torch.zeros_like(item)}
                d.update({_append_last(key, "_ssq"): torch.zeros_like(item)})
            else:
                idx = 0
                for in_key in self.in_keys:
                    if in_key != key:
                        idx += 1
                    else:
                        break
                shape = self.shapes[idx]
                item = tensordict.get(key)
                d = {
                    _append_last(key, "_sum"): torch.zeros(
                        shape, device=item.device, dtype=item.dtype
                    )
                }
                d.update(
                    {
                        _append_last(key, "_ssq"): torch.zeros(
                            shape, device=item.device, dtype=item.dtype
                        )
                    }
                )

            d.update(
                {
                    _append_last(key, "_count"): torch.zeros(
                        1, device=item.device, dtype=torch.float
                    )
                }
            )
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        # TODO: we should revert this and have _td be like: TensorDict{"sum": ..., "ssq": ..., "count"...})
        #  to facilitate the computation of the stats using TD internals.
        #  Moreover, _td can be locked so these ops will be very fast on CUDA.
        _sum = self._td.get(_append_last(key, "_sum"))
        _ssq = self._td.get(_append_last(key, "_ssq"))
        _count = self._td.get(_append_last(key, "_count"))

        value_sum = _sum_left(value, _sum)

        if not self.frozen:
            _sum *= self.decay
            _sum += value_sum
            self._td.set_(
                _append_last(key, "_sum"),
                _sum,
            )

        _ssq = self._td.get(_append_last(key, "_ssq"))
        value_ssq = _sum_left(value.pow(2), _ssq)
        if not self.frozen:
            _ssq *= self.decay
            _ssq += value_ssq
            self._td.set_(
                _append_last(key, "_ssq"),
                _ssq,
            )

        _count = self._td.get(_append_last(key, "_count"))
        if not self.frozen:
            _count *= self.decay
            _count += N
            self._td.set_(
                _append_last(key, "_count"),
                _count,
            )

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)

    def to_observation_norm(self) -> Compose | ObservationNorm:
        """Converts VecNorm into an ObservationNorm class that can be used at inference time.

        The :class:`~torchrl.envs.ObservationNorm` layer can be updated using the :meth:`~torch.nn.Module.state_dict`
        API.

        Examples:
            >>> from torchrl.envs import GymEnv, VecNorm
            >>> vecnorm = VecNorm(in_keys=["observation"])
            >>> train_env = GymEnv("CartPole-v1", device=None).append_transform(
            ...     vecnorm)
            >>>
            >>> r = train_env.rollout(4)
            >>>
            >>> eval_env = GymEnv("CartPole-v1").append_transform(
            ...     vecnorm.to_observation_norm())
            >>> print(eval_env.transform.loc, eval_env.transform.scale)
            >>>
            >>> r = train_env.rollout(4)
            >>> # Update entries with state_dict
            >>> eval_env.transform.load_state_dict(
            ...     vecnorm.to_observation_norm().state_dict())
            >>> print(eval_env.transform.loc, eval_env.transform.scale)

        """
        out = []
        loc = self.loc
        scale = self.scale
        for key, key_out in _zip_strict(self.in_keys, self.out_keys):
            _out = ObservationNorm(
                loc=loc.get(key),
                scale=scale.get(key),
                standard_normal=True,
                in_keys=key,
                out_keys=key_out,
            )
            out += [_out]
        if len(self.in_keys) > 1:
            return Compose(*out)
        return _out

    def _get_loc_scale(self, loc_only=False, scale_only=False):
        loc = {}
        scale = {}
        for key in self.in_keys:
            _sum = self._td.get(_append_last(key, "_sum"))
            _ssq = self._td.get(_append_last(key, "_ssq"))
            _count = self._td.get(_append_last(key, "_count"))
            loc[key] = _sum / _count
            scale[key] = (_ssq / _count - loc[key].pow(2)).clamp_min(self.eps).sqrt()
        if not scale_only:
            loc = TensorDict(loc)
        else:
            loc = None
        if not loc_only:
            scale = TensorDict(scale)
        else:
            scale = None
        return loc, scale

    @property
    def standard_normal(self):
        """Whether the affine transform given by `loc` and `scale` follows the standard normal equation.

        Similar to :class:`~torchrl.envs.ObservationNorm` standard_normal attribute.

        Always returns ``True``.
        """
        return True

    @property
    def loc(self):
        """Returns a TensorDict with the loc to be used for an affine transform."""
        # We can't cache that value bc the summary stats could be updated by a different process
        loc, _ = self._get_loc_scale(loc_only=True)
        return loc

    @property
    def scale(self):
        """Returns a TensorDict with the scale to be used for an affine transform."""
        # We can't cache that value bc the summary stats could be updated by a different process
        _, scale = self._get_loc_scale(scale_only=True)
        return scale

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: EnvBase,
        keys: Sequence[str] | None = None,
        memmap: bool = False,
    ) -> TensorDictBase:
        """Creates a shared tensordict for normalization across processes.

        Args:
            env (EnvBase): example environment to be used to create the
                tensordict
            keys (sequence of NestedKey, optional): keys that
                have to be normalized. Default is `["next", "reward"]`
            memmap (bool): if ``True``, the resulting tensordict will be cast into
                memmory map (using `memmap_()`). Otherwise, the tensordict
                will be placed in shared memory.

        Returns:
            A memory in shared memory to be sent to each process.

        Examples:
            >>> from torch import multiprocessing as mp
            >>> queue = mp.Queue()
            >>> env = make_env()
            >>> td_shared = VecNorm.build_td_for_shared_vecnorm(env,
            ...     ["next", "reward"])
            >>> assert td_shared.is_shared()
            >>> queue.put(td_shared)
            >>> # on workers
            >>> v = VecNorm(shared_td=queue.get())
            >>> env = TransformedEnv(make_env(), v)

        """
        raise NotImplementedError("this feature is currently put on hold.")
        sep = ".-|-."
        if keys is None:
            keys = ["next", "reward"]
        td = make_tensordict(env)
        keys = {key for key in td.keys() if key in keys}
        td_select = td.select(*keys)
        td_select = td_select.flatten_keys(sep)
        if td.batch_dims:
            raise RuntimeError(
                f"VecNorm should be used with non-batched environments. "
                f"Got batch_size={td.batch_size}"
            )
        keys = list(td_select.keys())
        for key in keys:
            td_select.set(_append_last(key, "_ssq"), td_select.get(key).clone())
            td_select.set(
                _append_last(key, "_count"),
                torch.zeros(
                    *td.batch_size,
                    1,
                    device=td_select.device,
                    dtype=torch.float,
                ),
            )
            td_select.rename_key_(key, _append_last(key, "_sum"))
        td_select.exclude(*keys).zero_()
        td_select = td_select.unflatten_keys(sep)
        if memmap:
            return td_select.memmap_()
        return td_select.share_memory_()

    # We use a different separator to ensure that keys can have points within them.
    SEP = "-<.>-"

    def get_extra_state(self) -> OrderedDict:
        if self._td is None:
            warnings.warn(
                "Querying state_dict on an uninitialized VecNorm transform will "
                "return a `None` value for the summary statistics. "
                "Loading such a state_dict on an initialized VecNorm will result in "
                "an error."
            )
            return
        return self._td.flatten_keys(self.SEP).to_dict()

    def set_extra_state(self, state: OrderedDict) -> None:
        if state is not None:
            td = TensorDict(state).unflatten_keys(self.SEP)
            if self._td is None and not td.is_shared():
                warnings.warn(
                    "VecNorm wasn't initialized and the tensordict is not shared. In single "
                    "process settings, this is ok, but if you need to share the statistics "
                    "between workers this should require some attention. "
                    "Make sure that the content of VecNorm is transmitted to the workers "
                    "after calling load_state_dict and not before, as other workers "
                    "may not have access to the loaded TensorDict."
                )
                td.share_memory_()
            if self._td is not None:
                self._td.update_(td)
            else:
                self._td = td
        elif self._td is not None:
            raise KeyError("Could not find a tensordict in the state_dict.")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(decay={self.decay:4.4f},"
            f"eps={self.eps:4.4f}, in_keys={self.in_keys}, out_keys={self.out_keys})"
        )

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

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, Bounded):
            return Unbounded(
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )
        return observation_spec

    # TODO: incorporate this when trackers are part of the data
    # def transform_output_spec(self, output_spec: TensorSpec) -> TensorSpec:
    #     observation_spec = output_spec["full_observation_spec"]
    #     reward_spec = output_spec["full_reward_spec"]
    #     for key in list(observation_spec.keys(True, True)):
    #         if key in self.in_keys:
    #             observation_spec[_append_last(key, "_sum")] = observation_spec[key].clone()
    #             observation_spec[_append_last(key, "_ssq")] = observation_spec[key].clone()
    #             observation_spec[_append_last(key, "_count")] = observation_spec[key].clone()
    #     for key in list(reward_spec.keys(True, True)):
    #         if key in self.in_keys:
    #             observation_spec[_append_last(key, "_sum")] = reward_spec[key].clone()
    #             observation_spec[_append_last(key, "_ssq")] = reward_spec[key].clone()
    #             observation_spec[_append_last(key, "_count")] = reward_spec[key].clone()
    #     return output_spec


class RewardSum(Transform):
    """Tracks episode cumulative rewards.

    This transform accepts a list of tensordict reward keys (i.e. ´in_keys´) and tracks their cumulative
    value along the time dimension for each episode.

    When called, the transform writes a new tensordict entry for each ``in_key`` named
    ``episode_{in_key}`` where the cumulative values are written.

    Args:
        in_keys (list of NestedKeys, optional): Input reward keys.
            All ´in_keys´ should be part of the environment reward_spec.
            If no ``in_keys`` are specified, this transform assumes ``"reward"`` to be the input key.
            However, multiple rewards (e.g. ``"reward1"`` and ``"reward2""``) can also be specified.
        out_keys (list of NestedKeys, optional): The output sum keys, should be one per each input key.
        reset_keys (list of NestedKeys, optional): the list of reset_keys to be
            used, if the parent environment cannot be found. If provided, this
            value will prevail over the environment ``reset_keys``.

    Keyword Args:
        reward_spec (bool, optional): if ``True``, the new reward entry will be registered in the
            reward specs. Defaults to ``False`` (registered in ``observation_specs``).

    Examples:
        >>> from torchrl.envs.transforms import RewardSum, TransformedEnv
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), RewardSum())
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> td = env.reset()
        >>> print(td["episode_reward"])
        tensor([0.])
        >>> td = env.rollout(3)
        >>> print(td["next", "episode_reward"])
        tensor([[1.],
                [2.],
                [3.]])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reset_keys: Sequence[NestedKey] | None = None,
        *,
        reward_spec: bool = False,
    ):
        """Initialises the transform. Filters out non-reward input keys and defines output keys."""
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys
        self._keys_checked = False
        self.reward_spec = reward_spec

    @property
    def in_keys(self):
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys in (None, []):
            # retrieve rewards from parent env
            parent = self.parent
            if parent is None:
                in_keys = ["reward"]
            else:
                in_keys = copy(parent.reward_keys)
            self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self):
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys in (None, []):
            out_keys = [
                _replace_last(in_key, f"episode_{_unravel_key_to_tuple(in_key)[-1]}")
                for in_key in self.in_keys
            ]
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        # we must access the private attribute because this check occurs before
        # the parent env is defined
        if value is not None and len(self._in_keys) != len(value):
            raise ValueError(
                "RewardSum expects the same number of input and output keys"
            )
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def reset_keys(self):
        reset_keys = self.__dict__.get("_reset_keys", None)
        if reset_keys is None:
            parent = self.parent
            if parent is None:
                raise TypeError(
                    "reset_keys not provided but parent env not found. "
                    "Make sure that the reset_keys are provided during "
                    "construction if the transform does not have a container env."
                )
            # let's try to match the reset keys with the in_keys.
            # We take the filtered reset keys, which are the only keys that really
            # matter when calling reset, and check that they match the in_keys root.
            reset_keys = parent._filtered_reset_keys
            if len(reset_keys) == 1:
                reset_keys = list(reset_keys) * len(self.in_keys)

            def _check_match(reset_keys, in_keys):
                # if this is called, the length of reset_keys and in_keys must match
                for reset_key, in_key in _zip_strict(reset_keys, in_keys):
                    # having _reset at the root and the reward_key ("agent", "reward") is allowed
                    # but having ("agent", "_reset") and "reward" isn't
                    if isinstance(reset_key, tuple) and isinstance(in_key, str):
                        return False
                    if (
                        isinstance(reset_key, tuple)
                        and isinstance(in_key, tuple)
                        and in_key[: (len(reset_key) - 1)] != reset_key[:-1]
                    ):
                        return False
                return True

            if not _check_match(reset_keys, self.in_keys):
                raise ValueError(
                    f"Could not match the env reset_keys {reset_keys} with the {type(self)} in_keys {self.in_keys}. "
                    f"Please provide the reset_keys manually. Reset entries can be "
                    f"non-unique and must be right-expandable to the shape of "
                    f"the input entries."
                )
            reset_keys = copy(reset_keys)
            self._reset_keys = reset_keys

        if not self._keys_checked and len(reset_keys) != len(self.in_keys):
            raise ValueError(
                f"Could not match the env reset_keys {reset_keys} with the in_keys {self.in_keys}. "
                "Please make sure that these have the same length."
            )
        self._keys_checked = True

        return reset_keys

    @reset_keys.setter
    def reset_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._reset_keys = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets episode rewards."""
        for in_key, reset_key, out_key in _zip_strict(
            self.in_keys, self.reset_keys, self.out_keys
        ):
            _reset = _get_reset(reset_key, tensordict)
            value = tensordict.get(out_key, default=None)
            if value is None:
                value = self.parent.full_reward_spec[in_key].zero()
            else:
                value = torch.where(expand_as_right(~_reset, value), value, 0.0)
            tensordict_reset.set(out_key, value)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Updates the episode rewards with the step rewards."""
        # Update episode rewards
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in next_tensordict.keys(include_nested=True):
                reward = next_tensordict.get(in_key)
                prev_reward = tensordict.get(out_key, 0.0)
                next_tensordict.set(out_key, prev_reward + reward)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return next_tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = Composite(shape=input_spec.shape, device=input_spec.device)
        state_spec.update(self._generate_episode_reward_spec())
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _generate_episode_reward_spec(self) -> Composite:
        episode_reward_spec = Composite()
        reward_spec = self.parent.full_reward_spec
        reward_spec_keys = self.parent.reward_keys
        # Define episode specs for all out_keys
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if (
                in_key in reward_spec_keys
            ):  # if this out_key has a corresponding key in reward_spec
                out_key = _unravel_key_to_tuple(out_key)
                temp_episode_reward_spec = episode_reward_spec
                temp_rew_spec = reward_spec
                for sub_key in out_key[:-1]:
                    if (
                        not isinstance(temp_rew_spec, Composite)
                        or sub_key not in temp_rew_spec.keys()
                    ):
                        break
                    if sub_key not in temp_episode_reward_spec.keys():
                        temp_episode_reward_spec[sub_key] = temp_rew_spec[
                            sub_key
                        ].empty()
                    temp_rew_spec = temp_rew_spec[sub_key]
                    temp_episode_reward_spec = temp_episode_reward_spec[sub_key]
                episode_reward_spec[out_key] = reward_spec[in_key].clone()
            else:
                raise ValueError(
                    f"The in_key: {in_key} is not present in the reward spec {reward_spec}."
                )
        return episode_reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec, adding the new keys generated by RewardSum."""
        if self.reward_spec:
            return observation_spec
        if not isinstance(observation_spec, Composite):
            observation_spec = Composite(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(self._generate_episode_reward_spec())
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if not self.reward_spec:
            return reward_spec
        reward_spec.update(self._generate_episode_reward_spec())
        return reward_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        time_dim = [i for i, name in enumerate(tensordict.names) if name == "time"]
        if not time_dim:
            raise ValueError(
                "At least one dimension of the tensordict must be named 'time' in offline mode"
            )
        time_dim = time_dim[0] - 1
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            reward = tensordict[in_key]
            cumsum = reward.cumsum(time_dim)
            tensordict.set(out_key, cumsum)
        return tensordict


class StepCounter(Transform):
    """Counts the steps from a reset and optionally sets the truncated state to ``True`` after a certain number of steps.

    The ``"done"`` state is also adapted accordingly (as done is the disjunction
    of task completion and early truncation).

    Args:
        max_steps (int, optional): a positive integer that indicates the
            maximum number of steps to take before setting the ``truncated_key``
            entry to ``True``.
        truncated_key (str, optional): the key where the truncated entries
            should be written. Defaults to ``"truncated"``, which is recognised by
            data collectors as a reset signal.
            This argument can only be a string (not a nested key) as it will be
            matched to each of the leaf done key in the parent environment
            (eg, a ``("agent", "done")`` key will be accompanied by a
            ``("agent", "truncated")`` if the ``"truncated"`` key name is used).
        step_count_key (str, optional): the key where the step count entries
            should be written. Defaults to ``"step_count"``.
            This argument can only be a string (not a nested key) as it will be
            matched to each of the leaf done key in the parent environment
            (eg, a ``("agent", "done")`` key will be accompanied by a
            ``("agent", "step_count")`` if the ``"step_count"`` key name is used).
        update_done (bool, optional): if ``True``, the ``"done"`` boolean tensor
            at the level of ``"truncated"``
            will be updated.
            This signal indicates that the trajectory has reached its ends,
            either because the task is completed (``"completed"`` entry is
            ``True``) or because it has been truncated (``"truncated"`` entry
            is ``True``).
            Defaults to ``True``.

    .. note:: To ensure compatibility with environments that have multiple
        done_key(s), this transform will write a step_count entry for
        every done entry within the tensordict.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
        >>> env = TransformedEnv(base_env,
        ...     StepCounter(max_steps=5))
        >>> rollout = env.rollout(100)
        >>> print(rollout)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print(rollout["next", "step_count"])
        tensor([[1],
                [2],
                [3],
                [4],
                [5]])

    """

    invertible = False

    def __init__(
        self,
        max_steps: int | None = None,
        truncated_key: str | None = "truncated",
        step_count_key: str | None = "step_count",
        update_done: bool = True,
    ):
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps should have a value greater or equal to one.")
        if not isinstance(truncated_key, str):
            raise ValueError("truncated_key must be a string.")
        if not isinstance(step_count_key, str):
            raise ValueError("step_count_key must be a string.")
        self.max_steps = max_steps
        self.truncated_key = truncated_key
        self.step_count_key = step_count_key
        self.update_done = update_done
        super().__init__()

    @property
    def truncated_keys(self):
        truncated_keys = self.__dict__.get("_truncated_keys", None)
        if truncated_keys is None:
            # make the default truncated keys
            truncated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = self.truncated_key
                else:
                    key = (*reset_key[:-1], self.truncated_key)
                truncated_keys.append(key)
        self._truncated_keys = truncated_keys
        return truncated_keys

    @property
    def done_keys(self):
        done_keys = self.__dict__.get("_done_keys", None)
        if done_keys is None:
            # make the default done keys
            done_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "done"
                else:
                    key = (*reset_key[:-1], "done")
                done_keys.append(key)
        self.__dict__["_done_keys"] = done_keys
        return done_keys

    @property
    def terminated_keys(self):
        terminated_keys = self.__dict__.get("_terminated_keys", None)
        if terminated_keys is None:
            # make the default terminated keys
            terminated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "terminated"
                else:
                    key = (*reset_key[:-1], "terminated")
                terminated_keys.append(key)
        self.__dict__["_terminated_keys"] = terminated_keys
        return terminated_keys

    @property
    def step_count_keys(self):
        step_count_keys = self.__dict__.get("_step_count_keys", None)
        if step_count_keys is None:
            # make the default step_count keys
            step_count_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = self.step_count_key
                else:
                    key = (*reset_key[:-1], self.step_count_key)
                step_count_keys.append(key)
        self.__dict__["_step_count_keys"] = step_count_keys
        return step_count_keys

    @property
    def reset_keys(self):
        if self.parent is not None:
            return self.parent._filtered_reset_keys
        # fallback on default "_reset"
        return ["_reset"]

    @property
    def full_done_spec(self):
        return self.parent.output_spec["full_done_spec"] if self.parent else None

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # get reset signal
        for (
            step_count_key,
            truncated_key,
            terminated_key,
            reset_key,
            done_key,
        ) in _zip_strict(
            self.step_count_keys,
            self.truncated_keys,
            self.terminated_keys,
            self.reset_keys,
            self.done_keys,
        ):
            reset = tensordict.get(reset_key, default=None)
            if reset is None:
                # get done status, just to inform the reset shape, dtype and device
                for entry_name in (terminated_key, truncated_key, done_key):
                    done = tensordict.get(entry_name, default=None)
                    if done is not None:
                        break
                else:
                    # It may be the case that reset did not provide a done state, in which case
                    # we fall back on the spec
                    done = self.parent.output_spec_unbatched[
                        "full_done_spec", entry_name
                    ].zero(tensordict_reset.shape)
                reset = torch.ones_like(done)

            step_count = tensordict.get(step_count_key, default=None)
            if step_count is None:
                step_count = self.container.observation_spec[step_count_key].zero()
                if step_count.device != reset.device:
                    step_count = step_count.to(reset.device, non_blocking=True)

            # zero the step count if reset is needed
            step_count = torch.where(~reset, step_count.expand_as(reset), 0)
            tensordict_reset.set(step_count_key, step_count)
            if self.max_steps is not None:
                truncated = step_count >= self.max_steps
                truncated = truncated | tensordict_reset.get(truncated_key, False)
                if self.update_done:
                    # we assume no done after reset
                    tensordict_reset.set(done_key, truncated)
                tensordict_reset.set(truncated_key, truncated)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for step_count_key, truncated_key, done_key in _zip_strict(
            self.step_count_keys, self.truncated_keys, self.done_keys
        ):
            step_count = tensordict.get(step_count_key)
            next_step_count = step_count + 1
            next_tensordict.set(step_count_key, next_step_count)

            if self.max_steps is not None:
                truncated = next_step_count >= self.max_steps
                truncated = truncated | next_tensordict.get(truncated_key, False)
                if self.update_done:
                    done = next_tensordict.get(done_key, None)

                    # we can have terminated and truncated
                    # terminated = next_tensordict.get(terminated_key, None)
                    # if terminated is not None:
                    #     truncated = truncated & ~terminated

                    done = truncated | done  # we assume no done after reset
                    next_tensordict.set(done_key, done)
                next_tensordict.set(truncated_key, truncated)
        return next_tensordict

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )
        full_done_spec = self.parent.output_spec["full_done_spec"]
        for step_count_key in self.step_count_keys:
            step_count_key = unravel_key(step_count_key)
            # find a matching done key (there might be more than one)
            for done_key in self.done_keys:
                # check root
                if type(done_key) != type(step_count_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == step_count_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break

            else:
                raise KeyError(
                    f"Could not find root of step_count_key {step_count_key} in done keys {self.done_keys}."
                )
            observation_spec[step_count_key] = Bounded(
                shape=shape,
                dtype=torch.int64,
                device=observation_spec.device,
                low=0,
                high=torch.iinfo(torch.int64).max,
            )
        return super().transform_observation_spec(observation_spec)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.max_steps:
            full_done_spec = self.parent.output_spec["full_done_spec"]
            for truncated_key in self.truncated_keys:
                truncated_key = unravel_key(truncated_key)
                # find a matching done key (there might be more than one)
                for done_key in self.done_keys:
                    # check root
                    if type(done_key) != type(truncated_key):
                        continue
                    if isinstance(done_key, tuple):
                        if done_key[:-1] == truncated_key[:-1]:
                            shape = full_done_spec[done_key].shape
                            break
                    if isinstance(done_key, str):
                        shape = full_done_spec[done_key].shape
                        break

                else:
                    raise KeyError(
                        f"Could not find root of truncated_key {truncated_key} in done keys {self.done_keys}."
                    )
                full_done_spec[truncated_key] = Categorical(
                    2, dtype=torch.bool, device=output_spec.device, shape=shape
                )
            if self.update_done:
                for done_key in self.done_keys:
                    done_key = unravel_key(done_key)
                    # find a matching done key (there might be more than one)
                    for done_key in self.done_keys:
                        # check root
                        if type(done_key) != type(done_key):
                            continue
                        if isinstance(done_key, tuple):
                            if done_key[:-1] == done_key[:-1]:
                                shape = full_done_spec[done_key].shape
                                break
                        if isinstance(done_key, str):
                            shape = full_done_spec[done_key].shape
                            break

                    else:
                        raise KeyError(
                            f"Could not find root of stop_key {done_key} in done keys {self.done_keys}."
                        )
                    full_done_spec[done_key] = Categorical(
                        2, dtype=torch.bool, device=output_spec.device, shape=shape
                    )
            output_spec["full_done_spec"] = full_done_spec
        return super().transform_output_spec(output_spec)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if not isinstance(input_spec, Composite):
            raise ValueError(
                f"input_spec was expected to be of type Composite. Got {type(input_spec)} instead."
            )
        if input_spec["full_state_spec"] is None:
            input_spec["full_state_spec"] = Composite(
                shape=input_spec.shape, device=input_spec.device
            )

        full_done_spec = self.parent.output_spec["full_done_spec"]
        for step_count_key in self.step_count_keys:
            step_count_key = unravel_key(step_count_key)
            # find a matching done key (there might be more than one)
            for done_key in self.done_keys:
                # check root
                if type(done_key) != type(step_count_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == step_count_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break

            else:
                raise KeyError(
                    f"Could not find root of step_count_key {step_count_key} in done keys {self.done_keys}."
                )

            input_spec[unravel_key(("full_state_spec", step_count_key))] = Bounded(
                shape=shape,
                dtype=torch.int64,
                device=input_spec.device,
                low=0,
                high=torch.iinfo(torch.int64).max,
            )

        return input_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "StepCounter cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "StepCounter with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need StepCounter to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )


class ExcludeTransform(Transform):
    """Excludes keys from the data.

    Args:
        *excluded_keys (iterable of NestedKey): The name of the keys to exclude. If the key is
            not present, it is simply ignored.
        inverse (bool, optional): if ``True``, the exclusion will occur during the ``inv`` call.
            Defaults to ``False``.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> env = TransformedEnv(
        ...     GymWrapper(gymnasium.make("Pendulum-v1")),
        ...     ExcludeTransform("truncated")
        ... )
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(self, *excluded_keys, inverse: bool = False):
        super().__init__()
        try:
            excluded_keys = unravel_key_list(excluded_keys)
        except TypeError:
            raise TypeError(
                "excluded keys must be a list or tuple of strings or tuples of strings."
            )
        self.excluded_keys = excluded_keys
        self.inverse = inverse

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if not self.inverse:
            return next_tensordict.exclude(*self.excluded_keys)
        return next_tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.inverse:
            return tensordict.exclude(*self.excluded_keys)
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if not self.inverse:
            return tensordict_reset.exclude(*self.excluded_keys)
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if not self.inverse:
            full_done_spec = output_spec["full_done_spec"]
            full_reward_spec = output_spec["full_reward_spec"]
            full_observation_spec = output_spec["full_observation_spec"]
            for key in self.excluded_keys:
                # done_spec
                if unravel_key(key) in list(full_done_spec.keys(True, True)):
                    del full_done_spec[key]
                    continue
                # reward_spec
                if unravel_key(key) in list(full_reward_spec.keys(True, True)):
                    del full_reward_spec[key]
                    continue
                # observation_spec
                if unravel_key(key) in list(full_observation_spec.keys(True, True)):
                    del full_observation_spec[key]
                    continue
                raise KeyError(f"Key {key} not found in the environment outputs.")
        return output_spec


class SelectTransform(Transform):
    """Select keys from the input tensordict.

    In general, the :obj:`ExcludeTransform` should be preferred: this transforms also
        selects the "action" (or other keys from input_spec), "done" and "reward"
        keys but other may be necessary.

    Args:
        *selected_keys (iterable of NestedKey): The name of the keys to select. If the key is
            not present, it is simply ignored.

    Keyword Args:
        keep_rewards (bool, optional): if ``False``, the reward keys must be provided
            if they should be kept. Defaults to ``True``.
        keep_dones (bool, optional): if ``False``, the done keys must be provided
            if they should be kept. Defaults to ``True``.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> env = TransformedEnv(
        ...     GymWrapper(gymnasium.make("Pendulum-v1")),
        ...     SelectTransform("observation", "reward", "done", keep_dones=False), # we leave done behind
        ... )
        >>> env.rollout(3)  # the truncated key is now absent
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        *selected_keys: NestedKey,
        keep_rewards: bool = True,
        keep_dones: bool = True,
    ):
        super().__init__()
        try:
            selected_keys = unravel_key_list(selected_keys)
        except TypeError:
            raise TypeError(
                "selected keys must be a list or tuple of strings or tuples of strings."
            )
        self.selected_keys = selected_keys
        self.keep_done_keys = keep_dones
        self.keep_reward_keys = keep_rewards

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.state_spec.keys(True, True)
        else:
            input_keys = []
        if self.keep_reward_keys:
            reward_keys = self.parent.reward_keys if self.parent else ["reward"]
        else:
            reward_keys = []
        if self.keep_done_keys:
            done_keys = self.parent.done_keys if self.parent else ["done"]
        else:
            done_keys = []
        return next_tensordict.select(
            *self.selected_keys, *reward_keys, *done_keys, *input_keys, strict=False
        )

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.state_spec.keys(True, True)
        else:
            input_keys = []
        if self.keep_reward_keys:
            reward_keys = self.parent.reward_keys if self.parent else ["reward"]
        else:
            reward_keys = []
        if self.keep_done_keys:
            done_keys = self.parent.done_keys if self.parent else ["done"]
        else:
            done_keys = []
        return tensordict_reset.select(
            *self.selected_keys, *reward_keys, *done_keys, *input_keys, strict=False
        )

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        full_done_spec = output_spec["full_done_spec"]
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        if not self.keep_done_keys:
            for key in list(full_done_spec.keys(True, True)):
                if unravel_key(key) not in self.selected_keys:
                    del full_done_spec[key]

        for key in list(full_observation_spec.keys(True, True)):
            if unravel_key(key) not in self.selected_keys:
                del full_observation_spec[key]

        if not self.keep_reward_keys:
            for key in list(full_reward_spec.keys(True, True)):
                if unravel_key(key) not in self.selected_keys:
                    del full_reward_spec[key]

        return output_spec


class TimeMaxPool(Transform):
    """Take the maximum value in each position over the last T observations.

    This transform take the maximum value in each position for all in_keys tensors over the last T time steps.

    Args:
        in_keys (sequence of NestedKey, optional): input keys on which the max pool will be applied. Defaults to "observation" if left empty.
        out_keys (sequence of NestedKey, optional): output keys where the output will be written. Defaults to `in_keys` if left empty.
        T (int, optional): Number of time steps over which to apply max pooling.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, TimeMaxPool(in_keys=["observation"], T=10))
        >>> torch.manual_seed(0)
        >>> env.set_seed(0)
        >>> rollout = env.rollout(10)
        >>> print(rollout["observation"])  # values should be increasing up until the 10th step
        tensor([[ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0216,  0.0000],
                [ 0.0000,  0.1149,  0.0000],
                [ 0.0000,  0.1990,  0.0000],
                [ 0.0000,  0.2749,  0.0000],
                [ 0.0000,  0.3281,  0.0000],
                [-0.9290,  0.3702, -0.8978]])

    .. note:: :class:`~TimeMaxPool` currently only supports ``done`` signal at the root.
        Nested ``done``, such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        T: int = 1,
        reset_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if T < 1:
            raise ValueError(
                "TimeMaxPoolTransform T parameter should have a value greater or equal to one."
            )
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                "TimeMaxPoolTransform in_keys and out_keys don't have the same number of elements"
            )
        self.buffer_size = T
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            setattr(
                self,
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        self.reset_key = reset_key

    @staticmethod
    def _buffer_name(in_key):
        in_key_str = "_".join(in_key) if isinstance(in_key, tuple) else in_key
        buffer_name = f"_maxpool_buffer_{in_key_str}"
        return buffer_name

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:

        _reset = _get_reset(self.reset_key, tensordict)
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            if not _reset.all():
                _reset_exp = _reset.expand(buffer.shape[0], *_reset.shape)
                buffer[_reset_exp] = 0.0
            else:
                buffer.fill_(0.0)
        with _set_missing_tolerance(self, True):
            for in_key in self.in_keys:
                val_reset = tensordict_reset.get(in_key, None)
                val_prev = tensordict.get(in_key, None)
                # if an in_key is missing, we try to copy it from the previous step
                if val_reset is None and val_prev is not None:
                    tensordict_reset.set(in_key, val_prev)
                elif val_prev is None and val_reset is None:
                    raise KeyError(f"Could not find {in_key} in the reset data.")
            return self._call(tensordict_reset, _reset=_reset)

    def _make_missing_buffer(self, tensordict, in_key, buffer_name):
        buffer = getattr(self, buffer_name)
        data = tensordict.get(in_key)
        size = list(data.shape)
        size.insert(0, self.buffer_size)
        buffer.materialize(size)
        buffer = buffer.to(dtype=data.dtype, device=data.device).zero_()
        setattr(self, buffer_name, buffer)
        return buffer

    def _call(self, next_tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(next_tensordict, in_key, buffer_name)
            if _reset is not None:
                # we must use only the reset data
                buffer[:, _reset] = torch.roll(buffer[:, _reset], shifts=1, dims=0)
                # add new obs
                data = next_tensordict.get(in_key)
                buffer[0, _reset] = data[_reset]
                # apply max pooling
                pooled_tensor, _ = buffer[:, _reset].max(dim=0)
                pooled_tensor = torch.zeros_like(data).masked_scatter_(
                    expand_as_right(_reset, data), pooled_tensor
                )
                # add to tensordict
                next_tensordict.set(out_key, pooled_tensor)
                continue
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(next_tensordict.get(in_key))
            # apply max pooling
            pooled_tensor, _ = buffer.max(dim=0)
            # add to tensordict
            next_tensordict.set(out_key, pooled_tensor)

        return next_tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "TimeMaxPool cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "TimeMaxPool with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need TimeMaxPool to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )


class RandomCropTensorDict(Transform):
    """A trajectory sub-sampler for ReplayBuffer and modules.

    Gathers a sub-sequence of a defined length along the last dimension of the input
    tensordict.
    This can be used to get cropped trajectories from trajectories sampled
    from a ReplayBuffer.

    This transform is primarily designed to be used with replay buffers and modules.
    Currently, it cannot be used as an environment transform.
    Do not hesitate to request for this behavior through an issue if this is
    desired.

    Args:
        sub_seq_len (int): the length of the sub-trajectory to sample
        sample_dim (int, optional): the dimension along which the cropping
            should occur. Negative dimensions should be preferred to make
            the transform robust to tensordicts of varying batch dimensions.
            Defaults to -1 (the default time dimension in TorchRL).
        mask_key (NestedKey): If provided, this represents the mask key to be looked
            for when doing the sampling. If provided, it only valid elements will
            be returned. It is assumed that the mask is a boolean tensor with
            first True values and then False values, not mixed together.
            :class:`RandomCropTensorDict` will NOT check that this is respected
            hence any error caused by an improper mask risks to go unnoticed.
            Defaults: None (no mask key).
    """

    def __init__(
        self,
        sub_seq_len: int,
        sample_dim: int = -1,
        mask_key: NestedKey | None = None,
    ):
        self.sub_seq_len = sub_seq_len
        if sample_dim > 0:
            warnings.warn(
                "A positive shape has been passed to the RandomCropTensorDict "
                "constructor. This may have unexpected behaviors when the "
                "passed tensordicts have inconsistent batch dimensions. "
                "For context, by convention, TorchRL concatenates time steps "
                "along the last dimension of the tensordict."
            )
        self.sample_dim = sample_dim
        self.mask_key = mask_key
        super().__init__()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = tensordict.shape
        dim = self.sample_dim
        # shape must have at least one dimension
        if not len(shape):
            raise RuntimeError(
                "Cannot sub-sample from a tensordict with an empty shape."
            )
        if shape[dim] < self.sub_seq_len:
            raise RuntimeError(
                f"Cannot sample trajectories of length {self.sub_seq_len} along"
                f" dimension {dim} given a tensordict of shape "
                f"{tensordict.shape}. Consider reducing the sub_seq_len "
                f"parameter or increase sample length."
            )
        max_idx_0 = shape[dim] - self.sub_seq_len
        idx_shape = list(tensordict.shape)
        idx_shape[dim] = 1
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        if self.mask_key is None or self.mask_key not in tensordict.keys(
            isinstance(self.mask_key, tuple)
        ):
            idx_0 = torch.randint(max_idx_0, idx_shape, device=device)
        else:
            # get the traj length for each entry
            mask = tensordict.get(self.mask_key)
            if mask.shape != tensordict.shape:
                raise ValueError(
                    "Expected a mask of the same shape as the tensordict. Got "
                    f"mask.shape={mask.shape} and tensordict.shape="
                    f"{tensordict.shape} instead."
                )
            traj_lengths = mask.cumsum(self.sample_dim).max(self.sample_dim, True)[0]
            if (traj_lengths < self.sub_seq_len).any():
                raise RuntimeError(
                    f"Cannot sample trajectories of length {self.sub_seq_len} when the minimum "
                    f"trajectory length is {traj_lengths.min()}."
                )
            # take a random number between 0 and traj_lengths - self.sub_seq_len
            idx_0 = (
                torch.rand(idx_shape, device=device) * (traj_lengths - self.sub_seq_len)
            ).to(torch.long)
        arange = torch.arange(self.sub_seq_len, device=idx_0.device)
        arange_shape = [1 for _ in range(tensordict.ndimension())]
        arange_shape[dim] = len(arange)
        arange = arange.view(arange_shape)
        idx = idx_0 + arange
        return tensordict.gather(dim=self.sample_dim, index=idx)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self.forward(tensordict_reset)
        return tensordict_reset


class InitTracker(Transform):
    """Reset tracker.

    This transform populates the step/reset tensordict with a reset tracker entry
    that is set to ``True`` whenever :meth:`~.reset` is called.

    Args:
         init_key (NestedKey, optional): the key to be used for the tracker entry.
            In case of multiple _reset flags, this key is used as the leaf replacement for each.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> td = env.reset()
        >>> print(td["is_init"])
        tensor(True)
        >>> td = env.rand_step(td)
        >>> print(td["next", "is_init"])
        tensor(False)

    """

    def __init__(self, init_key: str = "is_init"):
        if not isinstance(init_key, str):
            raise ValueError(
                "init_key can only be of type str as it will be the leaf key associated to each reset flag."
            )
        self.init_key = init_key
        super().__init__()

    def set_container(self, container: Transform | EnvBase) -> None:
        self._init_keys = None
        return super().set_container(container)

    @property
    def out_keys(self):
        return self.init_keys

    @out_keys.setter
    def out_keys(self, value):
        if value in (None, []):
            return
        raise ValueError(
            "Cannot set non-empty out-keys when out-keys are defined by the init_key value."
        )

    @property
    def init_keys(self):
        init_keys = self.__dict__.get("_init_keys", None)
        if init_keys is not None:
            return init_keys
        init_keys = []
        if self.parent is None:
            raise NotImplementedError(
                FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
            )
        for reset_key in self.parent._filtered_reset_keys:
            if isinstance(reset_key, str):
                init_key = self.init_key
            else:
                init_key = unravel_key((reset_key[:-1], self.init_key))
            init_keys.append(init_key)
        self._init_keys = init_keys
        return self._init_keys

    @property
    def reset_keys(self):
        return self.parent._filtered_reset_keys

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        for init_key in self.init_keys:
            done_key = _replace_last(init_key, "done")
            if init_key not in next_tensordict.keys(True, True):
                device = next_tensordict.device
                if device is None:
                    device = torch.device("cpu")
                shape = self.parent.full_done_spec[done_key].shape
                next_tensordict.set(
                    init_key,
                    torch.zeros(shape, device=device, dtype=torch.bool),
                )
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        for reset_key, init_key in _zip_strict(self.reset_keys, self.init_keys):
            _reset = tensordict.get(reset_key, None)
            if _reset is None:
                done_key = _replace_last(init_key, "done")
                shape = self.parent.full_done_spec[done_key]._safe_shape
                tensordict_reset.set(
                    init_key,
                    torch.ones(
                        shape,
                        device=device,
                        dtype=torch.bool,
                    ),
                )
            else:
                init_val = _reset.clone()
                parent_td = (
                    tensordict_reset
                    if isinstance(init_key, str)
                    else tensordict_reset.get(init_key[:-1])
                )
                if init_val.ndim == parent_td.ndim:
                    # unsqueeze, to match the done shape
                    init_val = init_val.unsqueeze(-1)
                tensordict_reset.set(init_key, init_val)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        full_done_spec = self.parent.output_spec["full_done_spec"]
        for init_key in self.init_keys:
            for done_key in self.parent.done_keys:
                # check root
                if type(done_key) != type(init_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == init_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break
            else:
                raise KeyError(
                    f"Could not find root of init_key {init_key} within done_keys {self.parent.done_keys}."
                )
            observation_spec[init_key] = Categorical(
                2,
                dtype=torch.bool,
                device=self.parent.device,
                shape=shape,
            )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )


class RenameTransform(Transform):
    """A transform to rename entries in the output tensordict (or input tensordict via the inverse keys).

    Args:
        in_keys (sequence of NestedKey): the entries to rename.
        out_keys (sequence of NestedKey): the name of the entries after renaming.
        in_keys_inv (sequence of NestedKey, optional): the entries to rename
            in the input tensordict, which will be passed to :meth:`EnvBase._step`.
        out_keys_inv (sequence of NestedKey, optional): the names of the entries
            in the input tensordict after renaming.
        create_copy (bool, optional): if ``True``, the entries will be copied
            with a different name rather than being renamed. This allows for
            renaming immutable entries such as ``"reward"`` and ``"done"``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     RenameTransform(["observation", ], ["stuff",], create_copy=False),
        ... )
        >>> tensordict = env.rollout(3)
        >>> print(tensordict)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)
        >>> # if the output is also an input, we need to rename if both ways:
        >>> from torchrl.envs.libs.brax import BraxEnv
        >>> env = TransformedEnv(
        ...     BraxEnv("fast"),
        ...     RenameTransform(["state"], ["newname"], ["state"], ["newname"])
        ... )
        >>> _ = env.set_seed(1)
        >>> tensordict = env.rollout(3)
        >>> assert "newname" in tensordict.keys()
        >>> assert "state" not in tensordict.keys()

    """

    def __init__(
        self, in_keys, out_keys, in_keys_inv=None, out_keys_inv=None, create_copy=False
    ):
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        self.create_copy = create_copy
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should match the number of out_keys ({len(self.out_keys)})."
            )
        if len(self.in_keys_inv) != len(self.out_keys_inv):
            raise ValueError(
                f"The number of in_keys_inv ({len(self.in_keys_inv)}) should match the number of out_keys_inv ({len(self.out_keys)})."
            )
        if len(set(out_keys).intersection(in_keys)):
            raise ValueError(
                f"Cannot have matching in and out_keys because order is unclear. "
                f"Please use separated transforms. "
                f"Got in_keys={in_keys} and out_keys={out_keys}."
            )

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.create_copy:
            out = next_tensordict.select(
                *self.in_keys, strict=not self._missing_tolerance
            )
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                try:
                    out.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
            next_tensordict = next_tensordict.update(out)
        else:
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                try:
                    next_tensordict.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return next_tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # no in-place modif
        if self.create_copy:
            out = tensordict.select(
                *self.out_keys_inv, strict=not self._missing_tolerance
            )
            for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
                try:
                    out.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise

            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
                try:
                    tensordict.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        for done_key in self.parent.done_keys:
            if done_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == done_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_done_spec"][out_key] = output_spec["full_done_spec"][
                    done_key
                ].clone()
                if not self.create_copy:
                    del output_spec["full_done_spec"][done_key]
        for reward_key in self.parent.reward_keys:
            if reward_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == reward_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_reward_spec"][out_key] = output_spec[
                    "full_reward_spec"
                ][reward_key].clone()
                if not self.create_copy:
                    del output_spec["full_reward_spec"][reward_key]
        for observation_key in self.parent.full_observation_spec.keys(True):
            if observation_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == observation_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_observation_spec"][out_key] = output_spec[
                    "full_observation_spec"
                ][observation_key].clone()
                if not self.create_copy:
                    del output_spec["full_observation_spec"][observation_key]
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        for action_key in self.parent.action_keys:
            if action_key in self.in_keys_inv:
                for i, out_key in enumerate(self.out_keys_inv):  # noqa: B007
                    if self.in_keys_inv[i] == action_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_action_spec"][out_key] = input_spec[
                    "full_action_spec"
                ][action_key].clone()
        if not self.create_copy:
            for action_key in self.parent.action_keys:
                if action_key in self.in_keys_inv:
                    del input_spec["full_action_spec"][action_key]
        for state_key in self.parent.full_state_spec.keys(True, True):
            if state_key in self.in_keys_inv:
                for i, out_key in enumerate(self.out_keys_inv):  # noqa: B007
                    if self.in_keys_inv[i] == state_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_state_spec"][out_key] = input_spec["full_state_spec"][
                    state_key
                ].clone()
        if not self.create_copy:
            for state_key in self.parent.full_state_spec.keys(True, True):
                if state_key in self.in_keys_inv:
                    del input_spec["full_state_spec"][state_key]
        return input_spec


class Reward2GoTransform(Transform):
    """Calculates the reward to go based on the episode reward and a discount factor.

    As the :class:`~.Reward2GoTransform` is only an inverse transform the ``in_keys`` will be directly used for the ``in_keys_inv``.
    The reward-to-go can be only calculated once the episode is finished. Therefore, the transform should be applied to the replay buffer
    and not to the collector or within an environment.

    Args:
        gamma (:obj:`float` or torch.Tensor): the discount factor. Defaults to 1.0.
        in_keys (sequence of NestedKey): the entries to rename. Defaults to
            ``("next", "reward")`` if none is provided.
        out_keys (sequence of NestedKey): the entries to rename. Defaults to
            the values of ``in_keys`` if none is provided.
        done_key (NestedKey): the done entry. Defaults to ``"done"``.
        truncated_key (NestedKey): the truncated entry. Defaults to ``"truncated"``.
            If no truncated entry is found, only the ``"done"`` will be used.

    Examples:
        >>> # Using this transform as part of a replay buffer
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> torch.manual_seed(0)
        >>> r2g = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100), transform=r2g)
        >>> batch, timesteps = 4, 5
        >>> done = torch.zeros(batch, timesteps, 1, dtype=torch.bool)
        >>> for i in range(batch):
        ...     while not done[i].any():
        ...         done[i] = done[i].bernoulli_(0.1)
        >>> reward = torch.ones(batch, timesteps, 1)
        >>> td = TensorDict(
        ...     {"next": {"done": done, "reward": reward}},
        ...     [batch, timesteps],
        ... )
        >>> rb.extend(td)
        >>> sample = rb.sample(1)
        >>> print(sample["next", "reward"])
        tensor([[[1.],
                 [1.],
                 [1.],
                 [1.],
                 [1.]]])
        >>> print(sample["reward_to_go"])
        tensor([[[4.9010],
                 [3.9404],
                 [2.9701],
                 [1.9900],
                 [1.0000]]])

    One can also use this transform directly with a collector: make sure to
    append the `inv` method of the transform.

    Examples:
        >>> from torchrl.envs.utils import RandomPolicy        >>> from torchrl.collectors import SyncDataCollector
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> t = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
        >>> env = GymEnv("Pendulum-v1")
        >>> collector = SyncDataCollector(
        ...     env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=200,
        ...     total_frames=-1,
        ...     postproc=t.inv
        ... )
        >>> for data in collector:
        ...     break
        >>> print(data)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                reward_to_go: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)

    Using this transform as part of an env will raise an exception

    Examples:
        >>> t = Reward2GoTransform(gamma=0.99)
        >>> TransformedEnv(GymEnv("Pendulum-v1"), t)  # crashes

    .. note:: In settings where multiple done entries are present, one should build
        a single :class:`~Reward2GoTransform` for each done-reward pair.

    """

    ENV_ERR = (
        "The Reward2GoTransform is only an inverse transform and can "
        "only be applied to the replay buffer."
    )

    def __init__(
        self,
        gamma: float | torch.Tensor | None = 1.0,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        done_key: NestedKey | None = "done",
    ):
        if in_keys is None:
            in_keys = [("next", "reward")]
        if out_keys is None:
            out_keys = copy(in_keys)
        # out_keys = ["reward_to_go"]
        super().__init__(
            in_keys=in_keys,
            in_keys_inv=in_keys,
            out_keys_inv=out_keys,
        )
        self.done_key = done_key

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma)

        self.register_buffer("gamma", gamma)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is not None:
            raise ValueError(self.ENV_ERR)
        done = tensordict.get(("next", self.done_key))

        if not done.any(-2).all():
            raise RuntimeError(
                "No episode ends found to calculate the reward to go. Make sure that the number of frames_per_batch is larger than number of steps per episode."
            )
        found = False
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                found = True
                item = self._inv_apply_transform(tensordict.get(in_key), done)
                tensordict.set(out_key, item)
        if not found:
            raise KeyError(f"Could not find any of the input keys {self.in_keys}.")
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def _inv_apply_transform(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        from torchrl.objectives.value.functional import reward2go

        return reward2go(reward, done, self.gamma)

    def set_container(self, container):
        if isinstance(container, EnvBase) or container.parent is not None:
            raise ValueError(self.ENV_ERR)


class ActionMask(Transform):
    """An adaptive action masker.

    This transform is useful to ensure that randomly generated actions
    respect legal actions, by masking the action specs.
    It reads the mask from the input tensordict after the step is executed,
    and adapts the mask of the finite action spec.

      .. note:: This transform will fail when used without an environment.

    Args:
        action_key (NestedKey, optional): the key where the action tensor can be found.
            Defaults to ``"action"``.
        mask_key (NestedKey, optional): the key where the action mask can be found.
            Defaults to ``"action_mask"``.

    Examples:
        >>> import torch
        >>> from torchrl.data.tensor_specs import Categorical, Binary, Unbounded, Composite
        >>> from torchrl.envs.transforms import ActionMask, TransformedEnv
        >>> from torchrl.envs.common import EnvBase
        >>> class MaskedEnv(EnvBase):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.action_spec = Categorical(4)
        ...         self.state_spec = Composite(action_mask=Binary(4, dtype=torch.bool))
        ...         self.observation_spec = Composite(obs=Unbounded(3))
        ...         self.reward_spec = Unbounded(1)
        ...
        ...     def _reset(self, tensordict=None):
        ...         td = self.observation_spec.rand()
        ...         td.update(torch.ones_like(self.state_spec.rand()))
        ...         return td
        ...
        ...     def _step(self, data):
        ...         td = self.observation_spec.rand()
        ...         mask = data.get("action_mask")
        ...         action = data.get("action")
        ...         mask = mask.scatter(-1, action.unsqueeze(-1), 0)
        ...
        ...         td.set("action_mask", mask)
        ...         td.set("reward", self.reward_spec.rand())
        ...         td.set("done", ~mask.any().view(1))
        ...         return td
        ...
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        ...
        >>> torch.manual_seed(0)
        >>> base_env = MaskedEnv()
        >>> env = TransformedEnv(base_env, ActionMask())
        >>> r = env.rollout(10)
        >>> r["action_mask"]
        tensor([[ True,  True,  True,  True],
                [ True,  True, False,  True],
                [ True,  True, False, False],
                [ True, False, False, False]])

    """

    ACCEPTED_SPECS = (
        OneHot,
        Categorical,
        MultiOneHot,
        MultiCategorical,
    )
    SPEC_TYPE_ERROR = "The action spec must be one of {}. Got {} instead."

    def __init__(
        self, action_key: NestedKey = "action", mask_key: NestedKey = "action_mask"
    ):
        if not isinstance(action_key, (tuple, str)):
            raise ValueError(
                f"The action key must be a nested key. Got {type(action_key)} instead."
            )
        if not isinstance(mask_key, (tuple, str)):
            raise ValueError(
                f"The mask key must be a nested key. Got {type(mask_key)} instead."
            )
        super().__init__(
            in_keys=[action_key, mask_key], out_keys=[], in_keys_inv=[], out_keys_inv=[]
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self)))

    @property
    def action_spec(self) -> TensorSpec:
        action_spec = self.container.full_action_spec[self.in_keys[0]]
        if not isinstance(action_spec, self.ACCEPTED_SPECS):
            raise ValueError(
                self.SPEC_TYPE_ERROR.format(self.ACCEPTED_SPECS, type(action_spec))
            )
        return action_spec

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is None:
            raise RuntimeError(
                f"{type(self)}.parent cannot be None: make sure this transform is executed within an environment."
            )

        mask = next_tensordict.get(self.in_keys[1])
        self.action_spec.update_mask(mask.to(self.action_spec.device))

        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


class VecGymEnvTransform(Transform):
    """A transform for GymWrapper subclasses that handles the auto-reset in a consistent way.

    Gym, gymnasium and SB3 provide vectorized (read, parallel or batched) environments
    that are automatically reset. When this occurs, the actual observation resulting
    from the action is saved within a key in the info.
    The class :class:`torchrl.envs.libs.gym.terminal_obs_reader` reads that observation
    and stores it in a ``"final"`` key within the output tensordict.
    In turn, this transform reads that final data, swaps it with the observation
    written in its place that results from the actual reset, and saves the
    reset output in a private container. The resulting data truly reflects
    the output of the step.

    This class works from gym 0.13 till the most recent gymnasium version.

    .. note:: Gym versions < 0.22 did not return the final observations. For these,
        we simply fill the next observations with NaN (because it is lost) and
        do the swap at the next step.

    Then, when calling `env.reset`, the saved data is written back where it belongs
    (and the `reset` is a no-op).

    This transform is automatically appended to the gym env whenever the wrapper
    is created with an async env.

    Args:
        final_name (str, optional): the name of the final observation in the dict.
            Defaults to `"final"`.
        missing_obs_value (Any, optional): default value to use as placeholder for missing
            last observations. Defaults to `np.nan`.

    .. note:: In general, this class should not be handled directly. It is
        created whenever a vectorized environment is placed within a :class:`GymWrapper`.

    """

    def __init__(self, final_name: str = "final", missing_obs_value: Any = np.nan):
        self.final_name = final_name
        super().__init__()
        self._memo = {}
        if not isinstance(missing_obs_value, torch.Tensor):
            missing_obs_value = torch.tensor(missing_obs_value)
        self.missing_obs_value = missing_obs_value

    def set_container(self, container: Transform | EnvBase) -> None:
        out = super().set_container(container)
        self._done_keys = None
        self._obs_keys = None
        return out

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # save the final info
        done = False
        for done_key in self.done_keys:
            # we assume dones can be broadcast
            done = done | next_tensordict.get(done_key)
        if done is False:
            raise RuntimeError(
                f"Could not find any done signal in tensordict:\n{tensordict}"
            )
        self._memo["done"] = done
        final = next_tensordict.pop(self.final_name, None)
        # if anything's done, we need to swap the final obs
        if done.any():
            done = done.squeeze(-1)
            if final is not None:
                saved_next = next_tensordict.select(*final.keys(True, True)).clone()
                next_tensordict[done] = final[done]
            else:
                saved_next = next_tensordict.select(*self.obs_keys).clone()
                for obs_key in self.obs_keys:
                    next_tensordict[obs_key][done] = self.missing_obs_value

            self._memo["saved_next"] = saved_next
        else:
            self._memo["saved_next"] = None
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        done = self._memo.get("done", None)
        reset = tensordict.get("_reset", done)
        if done is not None:
            done = done.view_as(reset)
        if (
            reset is not done
            and (reset != done).any()
            # it can happen that all are reset, in which case
            # it's fine (doesn't need to match done)
            and not reset.all()
        ):
            raise RuntimeError(
                "Cannot partially reset a gym(nasium) async env with a "
                "reset mask that does not match the done mask. "
                f"Got reset={reset}\nand done={done}"
            )
        # if not reset.any(), we don't need to do anything.
        # if reset.all(), we don't either (bc GymWrapper will call a plain reset).
        if reset is not None and reset.any():
            if reset.all():
                # We're fine: this means that a full reset was passed and the
                # env was manually reset
                tensordict_reset.pop(self.final_name, None)
                return tensordict_reset
            saved_next = self._memo["saved_next"]
            if saved_next is None:
                raise RuntimeError(
                    "Did not find a saved tensordict while the reset mask was "
                    f"not empty: reset={reset}. Done was {done}."
                )
            # reset = reset.view(tensordict.shape)
            # we have a data container from the previous call to step
            # that contains part of the observation we need.
            # We can safely place them back in the reset result tensordict:
            # in env.rollout(), the result of reset() is assumed to be just
            # the td from previous step with updated values from reset.
            # In our case, it will always be the case that all these values
            # are properly set.
            # collectors even take care of doing an extra masking so it's even
            # safer.
            tensordict_reset.update(saved_next)
            for done_key in self.done_keys:
                # Make sure that all done are False
                done = tensordict.get(done_key, None)
                if done is not None:
                    done = done.clone().fill_(0)
                else:
                    done = torch.zeros(
                        (*tensordict.batch_size, 1),
                        device=tensordict.device,
                        dtype=torch.bool,
                    )
                tensordict.set(done_key, done)
        tensordict_reset.pop(self.final_name, None)
        return tensordict_reset

    @property
    def done_keys(self) -> list[NestedKey]:
        keys = self.__dict__.get("_done_keys", None)
        if keys is None:
            keys = self.parent.done_keys
            # we just want the "done" key
            _done_keys = []
            for key in keys:
                if not isinstance(key, tuple):
                    key = (key,)
                if key[-1] == "done":
                    _done_keys.append(unravel_key(key))
            if not len(_done_keys):
                raise RuntimeError("Could not find a 'done' key in the env specs.")
            self._done_keys = _done_keys
        return keys

    @property
    def obs_keys(self) -> list[NestedKey]:
        keys = self.__dict__.get("_obs_keys", None)
        if keys is None:
            keys = list(self.parent.observation_spec.keys(True, True))
            self._obs_keys = keys
        return keys

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if self.final_name in observation_spec.keys(True):
            del observation_spec[self.final_name]
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self)))


class BurnInTransform(Transform):
    """Transform to partially burn-in data sequences.

    This transform is useful to obtain up-to-date recurrent states when
    they are not available. It burns-in a number of steps along the time dimension
    from sampled sequential data slices and returns the remaining data sequence with
    the burnt-in data in its initial time step. This transform is intended to be used as a
    replay buffer transform, not as an environment transform.

    Args:
        modules (sequence of TensorDictModule): A list of modules used to burn-in data sequences.
        burn_in (int): The number of time steps to burn in.
        out_keys (sequence of NestedKey, optional): destination keys. Defaults to
        all the modules `out_keys` that point to the next time step (e.g. `"hidden"` if `
        ("next", "hidden")` is part of the `out_keys` of a module).

    .. note::
        This transform expects as inputs TensorDicts with its last dimension being the
        time dimension. It also assumes that all provided modules can process
        sequential data.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import BurnInTransform
        >>> from torchrl.modules import GRUModule
        >>> gru_module = GRUModule(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     in_keys=["observation", "hidden"],
        ...     out_keys=["intermediate", ("next", "hidden")],
        ...     default_recurrent_mode=True,
        ... )
        >>> burn_in_transform = BurnInTransform(
        ...     modules=[gru_module],
        ...     burn_in=5,
        ... )
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> td = burn_in_transform(td)
        >>> td.shape
        torch.Size([2, 5])
        >>> td.get("hidden").abs().sum()
        tensor(86.3008)

        >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
        >>> buffer = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(2),
        ...     batch_size=1,
        ... )
        >>> buffer.append_transform(burn_in_transform)
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> buffer.extend(td)
        >>> td = buffer.sample(1)
        >>> td.shape
        torch.Size([1, 5])
        >>> td.get("hidden").abs().sum()
        tensor(37.0344)
    """

    invertible = False

    def __init__(
        self,
        modules: Sequence[TensorDictModuleBase],
        burn_in: int,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if not isinstance(modules, Sequence):
            modules = [modules]

        for module in modules:
            if not isinstance(module, TensorDictModuleBase):
                raise ValueError(
                    f"All modules must be TensorDictModules, but a {type(module)} was provided."
                )

        in_keys = set()
        for module in modules:
            in_keys.update(module.in_keys)

        if out_keys is None:
            out_keys = set()
            for module in modules:
                for key in module.out_keys:
                    if key[0] == "next":
                        out_keys.add(key[1])
        else:
            out_keys_ = set()
            for key in out_keys:
                if isinstance(key, tuple) and key[0] == "next":
                    key = key[1]
                    warnings.warn(
                        f"The 'next' key is not needed in the BurnInTransform `out_key` {key} and "
                        f"will be ignored. This transform already assumes that `out_keys` will be "
                        f"retrieved from the next time step of the burnt-in data."
                    )
                out_keys_.add(key)
            out_keys = out_keys_

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.modules = modules
        self.burn_in = burn_in

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("BurnInTransform can only be appended to a ReplayBuffer")

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise RuntimeError("BurnInTransform can only be appended to a ReplayBuffer.")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        if self.burn_in == 0:
            return tensordict

        td_device = tensordict.device
        B, T, *extra_dims = tensordict.batch_size

        # Split the tensor dict into burn-in data and the rest.
        td_burn_in = tensordict[..., : self.burn_in]
        td_out = tensordict[..., self.burn_in :]

        # Burn in the recurrent state.
        with torch.no_grad():
            for module in self.modules:
                module_device = next(module.parameters()).device or None
                td_burn_in = td_burn_in.to(module_device)
                td_burn_in = module(td_burn_in)
        td_burn_in = td_burn_in.to(td_device)

        # Update out TensorDict with the burnt-in data.
        for out_key in self.out_keys:
            if out_key not in td_out.keys(include_nested=True):
                td_out.set(
                    out_key,
                    torch.zeros(
                        B, T - self.burn_in, *tensordict.get(out_key).shape[2:]
                    ),
                )
            td_out[..., 0][out_key].copy_(td_burn_in["next"][..., -1][out_key])

        return td_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(burn_in={self.burn_in}, in_keys={self.in_keys}, out_keys={self.out_keys})"


class SignTransform(Transform):
    """A transform to compute the signs of TensorDict values.

    This transform reads the tensors in ``in_keys`` and ``in_keys_inv``, computes the
    signs of their elements and writes the resulting sign tensors to ``out_keys`` and
    ``out_keys_inv`` respectively.

    Args:
        in_keys (list of NestedKeys): input entries (read)
        out_keys (list of NestedKeys): input entries (write)
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`~.inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`~.inv` calls.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, SignTransform
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, SignTransform(in_keys=['observation']))
        >>> r = env.rollout(100)
        >>> obs = r["observation"]
        >>> assert (torch.logical_or(torch.logical_or(obs == -1, obs == 1), obs == 0.0)).all()
    """

    def __init__(
        self,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
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

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.sign()

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return state.sign()

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return Bounded(
            shape=observation_spec.shape,
            device=observation_spec.device,
            dtype=observation_spec.dtype,
            high=1.0,
            low=-1.0,
        )

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for key in self.in_keys:
            if key in self.parent.reward_keys:
                spec = self.parent.output_spec["full_reward_spec"][key]
                self.parent.output_spec["full_reward_spec"][key] = Bounded(
                    shape=spec.shape,
                    device=spec.device,
                    dtype=spec.dtype,
                    high=1.0,
                    low=-1.0,
                )
        return self.parent.output_spec["full_reward_spec"]

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class RemoveEmptySpecs(Transform):
    """Removes empty specs and content from an environment.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded, Composite, \
        ...     Categorical
        >>> from torchrl.envs import EnvBase, TransformedEnv, RemoveEmptySpecs
        >>>
        >>>
        >>> class DummyEnv(EnvBase):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.observation_spec = Composite(
        ...             observation=UnboundedContinuous((*self.batch_size, 3)),
        ...             other=Composite(
        ...                 another_other=Composite(shape=self.batch_size),
        ...                 shape=self.batch_size,
        ...             ),
        ...             shape=self.batch_size,
        ...         )
        ...         self.action_spec = UnboundedContinuous((*self.batch_size, 3))
        ...         self.done_spec = Categorical(
        ...             2, (*self.batch_size, 1), dtype=torch.bool
        ...         )
        ...         self.full_done_spec["truncated"] = self.full_done_spec[
        ...             "terminated"].clone()
        ...         self.reward_spec = Composite(
        ...             reward=UnboundedContinuous(*self.batch_size, 1),
        ...             other_reward=Composite(shape=self.batch_size),
        ...             shape=self.batch_size
        ...             )
        ...
        ...     def _reset(self, tensordict):
        ...         return self.observation_spec.rand().update(self.full_done_spec.zero())
        ...
        ...     def _step(self, tensordict):
        ...         return TensorDict(
        ...             {},
        ...             batch_size=[]
        ...         ).update(self.observation_spec.rand()).update(
        ...             self.full_done_spec.zero()
        ...             ).update(self.full_reward_spec.rand())
        ...
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        >>>
        >>>
        >>> base_env = DummyEnv()
        >>> print(base_env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        other: TensorDict(
                            fields={
                                another_other: TensorDict(
                                    fields={
                                    },
                                    batch_size=torch.Size([2]),
                                    device=cpu,
                                    is_shared=False)},
                            batch_size=torch.Size([2]),
                            device=cpu,
                            is_shared=False),
                        other_reward: TensorDict(
                            fields={
                            },
                            batch_size=torch.Size([2]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> check_env_specs(base_env)
        >>> env = TransformedEnv(base_env, RemoveEmptySpecs())
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        check_env_specs(env)
    """

    _has_empty_input = True

    @staticmethod
    def _sorter(key_val):
        key, _ = key_val
        if isinstance(key, str):
            return 0
        return len(key)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        full_done_spec = output_spec["full_done_spec"]
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        # we reverse things to make sure we delete things from the back
        for key, spec in sorted(
            full_done_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_done_spec[key]

        for key, spec in sorted(
            full_observation_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_observation_spec[key]

        for key, spec in sorted(
            full_reward_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_reward_spec[key]
        return output_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        full_action_spec = input_spec["full_action_spec"]
        full_state_spec = input_spec["full_state_spec"]
        # we reverse things to make sure we delete things from the back

        self._has_empty_input = False
        for key, spec in sorted(
            full_action_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                self._has_empty_input = True
                del full_action_spec[key]

        for key, spec in sorted(
            full_state_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                self._has_empty_input = True
                del full_state_spec[key]
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._has_empty_input:
            input_spec = getattr(self.parent, "input_spec", None)
            if input_spec is None:
                return tensordict

            full_action_spec = input_spec["full_action_spec"]
            full_state_spec = input_spec["full_state_spec"]
            # we reverse things to make sure we delete things from the back

            for key, spec in sorted(
                full_action_spec.items(True), key=self._sorter, reverse=True
            ):
                if (
                    isinstance(spec, Composite)
                    and spec.is_empty()
                    and key not in tensordict.keys(True)
                ):
                    tensordict.create_nested(key)

            for key, spec in sorted(
                full_state_spec.items(True), key=self._sorter, reverse=True
            ):
                if (
                    isinstance(spec, Composite)
                    and spec.is_empty()
                    and key not in tensordict.keys(True)
                ):
                    tensordict.create_nested(key)
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        for key, value in sorted(
            next_tensordict.items(True), key=self._sorter, reverse=True
        ):
            if (
                is_tensor_collection(value)
                and not isinstance(value, NonTensorData)
                and value.is_empty()
            ):
                del next_tensordict[key]
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets a transform if it is stateful."""
        return self._call(tensordict_reset)

    forward = _call


class _InvertTransform(Transform):
    _MISSING_TRANSFORM_ERROR = (
        "There is not generic rule to invert a spec transform. "
        "Please file an issue on github to get help."
    )

    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform

    @property
    def in_keys(self):
        return self.transform.in_keys_inv

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in in_keys.")

    @property
    def in_keys_inv(self):
        return self.transform.in_keys

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in in_keys_inv.")

    @property
    def out_keys(self):
        return self.transform.out_keys_inv

    @out_keys.setter
    def out_keys(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in out_keys.")

    @property
    def out_keys_inv(self):
        return self.transform.out_keys

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in out_keys_inv.")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform.inv(tensordict)

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform.forward(tensordict)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform._inv_call(next_tensordict)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform._call(tensordict)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)


class _CallableTransform(Transform):
    # A wrapper around a custom callable to make it possible to transform any data type
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _call(self, next_tensordict: TensorDictBase):
        return self.func(next_tensordict)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


class BatchSizeTransform(Transform):
    """A transform to modify the batch-size of an environment.

    This transform has two distinct usages: it can be used to set the
    batch-size for non-batch-locked (e.g. stateless) environments to
    enable data collection using data collectors. It can also be used
    to modify the batch-size of an environment (e.g. squeeze, unsqueeze or
    reshape).

    This transform modifies the environment batch-size to match the one provided.
    It expects the parent environment batch-size to be expandable to the
    provided one.

    Keyword Args:
        batch_size (torch.Size or equivalent, optional): the new batch-size of the environment.
            Exclusive with ``reshape_fn``.
        reshape_fn (callable, optional): a callable to modify the environment batch-size.
            Exclusive with ``batch_size``.

            .. note:: Currently, transformations involving
                ``reshape``, ``flatten``, ``unflatten``, ``squeeze`` and ``unsqueeze``
                are supported. If another reshape operation is required, please submit
                a feature request on TorchRL github.

        reset_func (callable, optional): a function that produces a reset tensordict.
            The signature must match ``Callable[[TensorDictBase, TensorDictBase], TensorDictBase]``
            where the first input argument is the optional tensordict passed to the
            environment during the call to :meth:`~EnvBase.reset` and the second
            is the output of ``TransformedEnv.base_env.reset``. It can also support an
            optional ``env`` keyword argument if ``env_kwarg=True``.
        env_kwarg (bool, optional): if ``True``, ``reset_func`` must support a
            ``env`` keyword argument. Defaults to ``False``. The env passed will
            be the env accompanied by its transform.

    Example:
        >>> # Changing the batch-size with a function
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("CartPole-v1")
        >>> env = TransformedEnv(base_env, BatchSizeTransform(reshape_fn=lambda data: data.reshape(1, 1)))
        >>> env.rollout(4)
        >>> # Setting the shape of a stateless environment
        >>> class MyEnv(EnvBase):
        ...     batch_locked = False
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = Composite(observation=Unbounded(3))
        ...         self.reward_spec = Unbounded(1)
        ...         self.action_spec = Unbounded(1)
        ...
        ...     def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        ...         tensordict_batch_size = tensordict.batch_size if tensordict is not None else torch.Size([])
        ...         result = self.observation_spec.rand(tensordict_batch_size)
        ...         result.update(self.full_done_spec.zero(tensordict_batch_size))
        ...         return result
        ...
        ...     def _step(
        ...         self,
        ...         tensordict: TensorDictBase,
        ...     ) -> TensorDictBase:
        ...         result = self.observation_spec.rand(tensordict.batch_size)
        ...         result.update(self.full_done_spec.zero(tensordict.batch_size))
        ...         result.update(self.full_reward_spec.zero(tensordict.batch_size))
        ...         return result
        ...
        ...     def _set_seed(self, seed: Optional[int]) -> None:
        ...         pass
        ...
        >>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5]))
        >>> assert env.batch_size == torch.Size([5])
        >>> assert env.rollout(10).shape == torch.Size([5, 10])

    The ``reset_func`` can create a tensordict with the desired batch-size, allowing for
    a fine-grained reset call:

        >>> def reset_func(tensordict, tensordict_reset, env):
        ...     result = env.observation_spec.rand()
        ...     result.update(env.full_done_spec.zero())
        ...     assert result.batch_size != torch.Size([])
        ...     return result
        >>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5], reset_func=reset_func, env_kwarg=True))
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5, 2]),
            device=None,
            is_shared=False)

    This transform can be used to deploy non-batch-locked environments within data
    collectors:

        >>> from torchrl.collectors import SyncDataCollector
        >>> collector = SyncDataCollector(env, lambda td: env.rand_action(td), frames_per_batch=10, total_frames=-1)
        >>> for data in collector:
        ...     print(data)
        ...     break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([5, 2]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5, 2]),
            device=None,
            is_shared=False)
        >>> collector.shutdown()
    """

    _ENV_ERR = "BatchSizeTransform.{} requires a parent env."

    def __init__(
        self,
        *,
        batch_size: torch.Size | None = None,
        reshape_fn: Callable[[TensorDictBase], TensorDictBase] | None = None,
        reset_func: Callable[[TensorDictBase, TensorDictBase], TensorDictBase]
        | None = None,
        env_kwarg: bool = False,
    ):
        super().__init__()
        if not ((batch_size is None) ^ (reshape_fn is None)):
            raise ValueError(
                "One and only one of batch_size OR reshape_fn must be provided."
            )
        if batch_size is not None:
            self.batch_size = torch.Size(batch_size)
            self.reshape_fn = None
        else:
            self.reshape_fn = reshape_fn
            self.batch_size = None
        self.reshape_fn = reshape_fn
        self.reset_func = reset_func
        self.env_kwarg = env_kwarg

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.reset_func is not None:
            if self.env_kwarg:
                tensordict_reset = self.reset_func(
                    tensordict, tensordict_reset, env=self.container
                )
            else:
                tensordict_reset = self.reset_func(tensordict, tensordict_reset)
        if self.batch_size is not None:
            return tensordict_reset.expand(self.batch_size)
        return self.reshape_fn(tensordict_reset)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.reshape_fn is not None:
            next_tensordict = self.reshape_fn(next_tensordict)
        return next_tensordict

    forward = _call

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.reshape_fn is not None:
            parent = self.parent
            if parent is not None:
                parent_batch_size = parent.batch_size
                tensordict = tensordict.reshape(parent_batch_size)
        return tensordict

    def transform_env_batch_size(self, batch_size: torch.Size):
        if self.batch_size is not None:
            return self.batch_size
        return self.reshape_fn(torch.zeros(batch_size, device="meta")).shape

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.batch_size is not None:
            return output_spec.expand(self.batch_size)
        return self.reshape_fn(output_spec)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if self.batch_size is not None:
            return input_spec.expand(self.batch_size)
        return self.reshape_fn(input_spec)


class AutoResetEnv(TransformedEnv):
    """A subclass for auto-resetting envs."""

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        if tensordict is not None:
            # We must avoid modifying the original tensordict so a shallow copy is necessary.
            # We just select the input data and reset signal, which is all we need.
            tensordict = tensordict.select(
                *self.reset_keys, *self.state_spec.keys(True, True), strict=False
            )
        for reset_key in self.base_env.reset_keys:
            if tensordict is not None and reset_key in tensordict.keys(True):
                tensordict_reset = tensordict.exclude(*self.base_env.reset_keys)
            else:
                tensordict_reset = self.base_env._reset(tensordict, **kwargs)
            break
        if tensordict is None:
            # make sure all transforms see a source tensordict
            tensordict = tensordict_reset.empty()
        self.base_env._complete_done(self.base_env.full_done_spec, tensordict_reset)
        tensordict_reset = self.transform._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def insert_transform(self, index: int, transform: Transform) -> None:
        raise RuntimeError(f"Cannot insert a transform in {self.__class_.__name__}.")


class AutoResetTransform(Transform):
    """A transform for auto-resetting environments.

    This transform can be appended to any auto-resetting environment, or automatically
    appended using ``env = SomeEnvClass(..., auto_reset=True)``. If the transform is explicitly
    appended to an env, a :class:`~torchrl.envs.transforms.AutoResetEnv` must be used.

    An auto-reset environment must have the following properties (differences from this
    description should be accounted for by subclassing this class):

      - the reset function can be called once at the beginning (after instantiation) with
        or without effect. Whether calls to `reset` are allowed after that is up to the
        environment itself.
      - During a rollout, any ``done`` state will result in a reset and produce an observation
        that isn't the last observation of the current episode, but the first observation
        of the next episode (this transform will extract and cache this observation
        and fill the obs with some arbitrary value).

    Keyword Args:
        replace (bool, optional): if ``False``, values are just placed as they are in the
            ``"next"`` entry even if they are not valid. Defaults to ``True``. A value of
            ``False`` overrides any subsequent filling keyword argument.
            This argument can also be passed with the constructor method by passing a
            ``auto_reset_replace`` argument: ``env = FooEnv(..., auto_reset=True, auto_reset_replace=False)``.
        fill_float (:obj:`float` or str, optional): The filling value for floating point tensors
            that terminate an episode. A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).
        fill_int (int, optional): The filling value for signed integer tensors
            that terminate an episode.  A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).
        fill_bool (bool, optional): The filling value for boolean tensors
            that terminate an episode.  A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).

    Arguments are only available when the transform is explicitly instantiated (not through `EnvType(..., auto_reset=True)`).

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.envs import set_gym_backend
        >>> import torch
        >>> torch.manual_seed(0)
        >>>
        >>> class AutoResettingGymEnv(GymEnv):
        ...     def _step(self, tensordict):
        ...         tensordict = super()._step(tensordict)
        ...         if tensordict["done"].any():
        ...             td_reset = super().reset()
        ...             tensordict.update(td_reset.exclude(*self.done_keys))
        ...         return tensordict
        ...
        ...     def _reset(self, tensordict=None):
        ...         if tensordict is not None and "_reset" in tensordict:
        ...             return tensordict.copy()
        ...         return super()._reset(tensordict)
        >>>
        >>> with set_gym_backend("gym"):
        ...     env = AutoResettingGymEnv("CartPole-v1", auto_reset=True, auto_reset_replace=True)
        ...     env.set_seed(0)
        ...     r = env.rollout(30, break_when_any_done=False)
        >>> print(r["next", "done"].squeeze())
        tensor([False, False, False, False, False, False, False, False, False, False,
                False, False, False,  True, False, False, False, False, False, False,
                False, False, False, False, False,  True, False, False, False, False])
        >>> print("observation after reset are set as nan", r["next", "observation"])
        observation after reset are set as nan tensor([[-4.3633e-02, -1.4877e-01,  1.2849e-02,  2.7584e-01],
                [-4.6609e-02,  4.6166e-02,  1.8366e-02, -1.2761e-02],
                [-4.5685e-02,  2.4102e-01,  1.8111e-02, -2.9959e-01],
                [-4.0865e-02,  4.5644e-02,  1.2119e-02, -1.2542e-03],
                [-3.9952e-02,  2.4059e-01,  1.2094e-02, -2.9009e-01],
                [-3.5140e-02,  4.3554e-01,  6.2920e-03, -5.7893e-01],
                [-2.6429e-02,  6.3057e-01, -5.2867e-03, -8.6963e-01],
                [-1.3818e-02,  8.2576e-01, -2.2679e-02, -1.1640e+00],
                [ 2.6972e-03,  1.0212e+00, -4.5959e-02, -1.4637e+00],
                [ 2.3121e-02,  1.2168e+00, -7.5232e-02, -1.7704e+00],
                [ 4.7457e-02,  1.4127e+00, -1.1064e-01, -2.0854e+00],
                [ 7.5712e-02,  1.2189e+00, -1.5235e-01, -1.8289e+00],
                [ 1.0009e-01,  1.0257e+00, -1.8893e-01, -1.5872e+00],
                [        nan,         nan,         nan,         nan],
                [-3.9405e-02, -1.7766e-01, -1.0403e-02,  3.0626e-01],
                [-4.2959e-02, -3.7263e-01, -4.2775e-03,  5.9564e-01],
                [-5.0411e-02, -5.6769e-01,  7.6354e-03,  8.8698e-01],
                [-6.1765e-02, -7.6292e-01,  2.5375e-02,  1.1820e+00],
                [-7.7023e-02, -9.5836e-01,  4.9016e-02,  1.4826e+00],
                [-9.6191e-02, -7.6387e-01,  7.8667e-02,  1.2056e+00],
                [-1.1147e-01, -9.5991e-01,  1.0278e-01,  1.5219e+00],
                [-1.3067e-01, -7.6617e-01,  1.3322e-01,  1.2629e+00],
                [-1.4599e-01, -5.7298e-01,  1.5848e-01,  1.0148e+00],
                [-1.5745e-01, -7.6982e-01,  1.7877e-01,  1.3527e+00],
                [-1.7285e-01, -9.6668e-01,  2.0583e-01,  1.6956e+00],
                [        nan,         nan,         nan,         nan],
                [-4.3962e-02,  1.9845e-01, -4.5015e-02, -2.5903e-01],
                [-3.9993e-02,  3.9418e-01, -5.0196e-02, -5.6557e-01],
                [-3.2109e-02,  5.8997e-01, -6.1507e-02, -8.7363e-01],
                [-2.0310e-02,  3.9574e-01, -7.8980e-02, -6.0090e-01]])

    """

    def __init__(
        self,
        *,
        replace: bool | None = None,
        fill_float="nan",
        fill_int=-1,
        fill_bool=False,
    ):
        super().__init__()
        if replace is False:
            fill_float = fill_int = fill_bool = None
        if fill_float == "nan":
            fill_float = float("nan")
        self.fill_float = fill_float
        self.fill_int = fill_int
        self.fill_bool = fill_bool
        self._validated = False

    def _validate_container(self):
        if self._validated:
            return
        if type(self.container) is not AutoResetEnv:
            raise RuntimeError(
                f"The {self.__class__.__name__} container must be of type AutoResetEnv."
            )
        self._validated = True

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self._validate_container()
        return self._replace_auto_reset_vals(tensordict_reset=tensordict_reset)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return self._correct_auto_reset_vals(next_tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError

    @property
    def _simple_done(self):
        return self.parent._simple_done

    def _correct_auto_reset_vals(self, tensordict):
        # we need to move the data from tensordict to tensordict_
        def replace_and_set(key, val, mask, saved_td_autoreset, agent=tensordict):
            saved_td_autoreset.set(key, val)
            if val.dtype.is_floating_point:
                if self.fill_float is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_float),
                        val,
                    )
            elif val.dtype.is_signed:
                if self.fill_int is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_int),
                        val,
                    )
            else:
                if self.fill_bool is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_bool),
                        val,
                    )
            agent.set(key, val_set_nan)

        if self._simple_done:
            done = tensordict.get("done")
            if done.any():
                mask = done.squeeze(-1)
                self._saved_td_autorest = TensorDict()
                for key in self.parent.full_observation_spec.keys(True, True):
                    val = tensordict.get(key)
                    replace_and_set(
                        key, val, mask, saved_td_autoreset=self._saved_td_autorest
                    )
        else:
            parents = []
            # Go through each "done" key and get the corresponding agent.
            _saved_td_autorest = None
            obs_keys = list(self.parent.full_observation_spec.keys(True, True))
            for done_key in self.parent.done_keys:
                if _ends_with(done_key, "done"):
                    if isinstance(done_key, str):
                        raise TypeError(
                            "A 'done' key was a string but a tuple was expected."
                        )
                    agent_key = done_key[:-1]
                    done = tensordict.get(done_key)
                    mask = done.squeeze(-1)
                    if done.any():
                        if _saved_td_autorest is None:
                            _saved_td_autorest = TensorDict()
                        agent = tensordict.get(agent_key)
                        if isinstance(agent, LazyStackedTensorDict):
                            agents = agent.tensordicts
                            masks = mask.unbind(agent.stack_dim)
                            saved_td_autorest_agent = LazyStackedTensorDict(
                                *[td.empty() for td in agents],
                                stack_dim=agent.stack_dim,
                            )
                            saved_td_autorest_agents = (
                                saved_td_autorest_agent.tensordicts
                            )
                        else:
                            agents = [agent]
                            masks = [mask]
                            saved_td_autorest_agent = _saved_td_autorest.setdefault(
                                agent_key, agent.empty()
                            )
                            saved_td_autorest_agents = [saved_td_autorest_agent]
                        for key in obs_keys:
                            if (
                                isinstance(key, tuple)
                                and key[: len(agent_key)] == agent_key
                            ):
                                for _agent, _mask, _saved_td_autorest_agent in zip(
                                    agents, masks, saved_td_autorest_agents
                                ):
                                    val = _agent.get(key[len(agent_key) :])
                                    replace_and_set(
                                        key[len(agent_key) :],
                                        val,
                                        _mask,
                                        saved_td_autoreset=_saved_td_autorest_agent,
                                        agent=_agent,
                                    )
                    parents.append(done_key[:-1])
            if _saved_td_autorest is not None:
                self.__dict__["_saved_td_autorest"] = _saved_td_autorest

        return tensordict

    def _replace_auto_reset_vals(self, *, tensordict_reset):
        _saved_td_autorest = self.__dict__.get("_saved_td_autorest", None)
        if _saved_td_autorest is None:
            return tensordict_reset
        if self._simple_done:
            for key, val in self._saved_td_autorest.items(True, True):
                if _ends_with(key, "_reset"):
                    continue
                val_set_reg = val
                tensordict_reset.set(key, val_set_reg)
        else:
            for done_key in self.parent.done_keys:
                if _ends_with(done_key, "done"):
                    agent_key = done_key[:-1]
                    mask = self._saved_td_autorest.pop(
                        _replace_last(done_key, "__mask__"), None
                    )
                    if mask is not None:
                        agent = self._saved_td_autorest.get(agent_key)

                        if isinstance(agent, LazyStackedTensorDict):
                            agents = agent.tensordicts
                            masks = mask.unbind(agent.stack_dim)
                            dests = tensordict_reset.setdefault(
                                agent_key,
                                LazyStackedTensorDict(
                                    *[td.empty() for td in agents],
                                    stack_dim=agent.stack_dim,
                                ),
                            )
                        else:
                            agents = [agent]
                            masks = [mask]
                            dests = [
                                tensordict_reset.setdefault(agent_key, agent.empty())
                            ]
                        for _agent, _mask, _dest in zip(agents, masks, dests):
                            for key, val in _agent.items(True, True):
                                if _ends_with(key, "_reset"):
                                    continue
                                if not _mask.all():
                                    val_not_reset = _dest.get(key)
                                    val_set_reg = torch.where(
                                        expand_as_right(mask, val), val, val_not_reset
                                    )
                                else:
                                    val_set_reg = val
                                _dest.set(key, val_set_reg)
        delattr(self, "_saved_td_autorest")
        return tensordict_reset


class ActionDiscretizer(Transform):
    """A transform to discretize a continuous action space.

    This transform makes it possible to use an algorithm designed for discrete
    action spaces such as DQN over environments with a continuous action space.

    Args:
        num_intervals (int or torch.Tensor): the number of discrete values
            for each element of the action space. If a single integer is provided,
            all action items are sliced with the same number of elements.
            If a tensor is provided, it must have the same number of elements
            as the action space (ie, the length of the ``num_intervals`` tensor
            must match the last dimension of the action space).
        action_key (NestedKey, optional): the action key to use. Points to
            the action of the parent env (the floating point action).
            Defaults to ``"action"``.
        out_action_key (NestedKey, optional): the key where the discrete
            action should be written. If ``None`` is provided, it defaults to
            the value of ``action_key``. If both keys do not match, the
            continuous action_spec is moved from the ``full_action_spec``
            environment attribute to the ``full_state_spec`` container,
            as only the discrete action should be sampled for an action to
            be taken. Providing ``out_action_key`` can ensure that the
            floating point action is available to be recorded.
        sampling (ActionDiscretizer.SamplingStrategy, optinoal): an element
            of the ``ActionDiscretizer.SamplingStrategy`` ``IntEnum`` object
            (``MEDIAN``, ``LOW``, ``HIGH`` or ``RANDOM``). Indicates how the
            continuous action should be sampled in the provided interval.
        categorical (bool, optional): if ``False``, one-hot encoding is used.
            Defaults to ``True``.

    Examples:
        >>> from torchrl.envs import GymEnv, check_env_specs
        >>> import torch
        >>> base_env = GymEnv("HalfCheetah-v4")
        >>> num_intervals = torch.arange(5, 11)
        >>> categorical = True
        >>> sampling = ActionDiscretizer.SamplingStrategy.MEDIAN
        >>> t = ActionDiscretizer(
        ...     num_intervals=num_intervals,
        ...     categorical=categorical,
        ...     sampling=sampling,
        ...     out_action_key="action_disc",
        ... )
        >>> env = base_env.append_transform(t)
        TransformedEnv(
            env=GymEnv(env=HalfCheetah-v4, batch_size=torch.Size([]), device=cpu),
            transform=ActionDiscretizer(
                num_intervals=tensor([ 5,  6,  7,  8,  9, 10]),
                action_key=action,
                out_action_key=action_disc,,
                sampling=0,
                categorical=True))
        >>> check_env_specs(env)
        >>> # Produce a rollout
        >>> r = env.rollout(4)
        >>> print(r)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                action_disc: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4]),
            device=cpu,
            is_shared=False)
        >>> assert r["action"].dtype == torch.float
        >>> assert r["action_disc"].dtype == torch.int64
        >>> assert (r["action"] < base_env.action_spec.high).all()
        >>> assert (r["action"] > base_env.action_spec.low).all()

    """

    class SamplingStrategy(IntEnum):
        """The sampling strategies for ActionDiscretizer."""

        MEDIAN = 0
        LOW = 1
        HIGH = 2
        RANDOM = 3

    def __init__(
        self,
        num_intervals: int | torch.Tensor,
        action_key: NestedKey = "action",
        out_action_key: NestedKey = None,
        sampling=None,
        categorical: bool = True,
    ):
        if out_action_key is None:
            out_action_key = action_key
        super().__init__(in_keys_inv=[action_key], out_keys_inv=[out_action_key])
        self.action_key = action_key
        self.out_action_key = out_action_key
        if not isinstance(num_intervals, torch.Tensor):
            self.num_intervals = num_intervals
        else:
            self.register_buffer("num_intervals", num_intervals)
        if sampling is None:
            sampling = self.SamplingStrategy.MEDIAN
        self.sampling = sampling
        self.categorical = categorical

    def __repr__(self):
        def _indent(s):
            return indent(s, 4 * " ")

        num_intervals = f"num_intervals={self.num_intervals}"
        action_key = f"action_key={self.action_key}"
        out_action_key = f"out_action_key={self.out_action_key}"
        sampling = f"sampling={self.sampling}"
        categorical = f"categorical={self.categorical}"
        return (
            f"{type(self).__name__}(\n{_indent(num_intervals)},\n{_indent(action_key)},"
            f"\n{_indent(out_action_key)},\n{_indent(sampling)},\n{_indent(categorical)})"
        )

    def _custom_arange(self, nint, device):
        result = torch.arange(
            start=0.0,
            end=1.0,
            step=1 / nint,
            dtype=self.dtype,
            device=device,
        )
        result_ = result
        if self.sampling in (
            self.SamplingStrategy.HIGH,
            self.SamplingStrategy.MEDIAN,
        ):
            result_ = (1 - result).flip(0)
        if self.sampling == self.SamplingStrategy.MEDIAN:
            result = (result + result_) / 2
        else:
            result = result_
        return result

    def transform_input_spec(self, input_spec):
        try:
            action_spec = self.parent.full_action_spec_unbatched[self.in_keys_inv[0]]
            if not isinstance(action_spec, Bounded):
                raise TypeError(
                    f"action spec type {type(action_spec)} is not supported. The action spec type must be Bounded."
                )

            n_act = action_spec.shape
            if not n_act:
                n_act = ()
                empty_shape = True
            else:
                n_act = (n_act[-1],)
                empty_shape = False
            self.n_act = n_act

            self.dtype = action_spec.dtype
            interval = action_spec.high - action_spec.low

            num_intervals = self.num_intervals

            if not empty_shape:
                interval = interval.unsqueeze(-1)
            elif isinstance(num_intervals, torch.Tensor):
                num_intervals = int(num_intervals.squeeze())
                self.num_intervals = torch.as_tensor(num_intervals)

            if isinstance(num_intervals, int):
                arange = (
                    self._custom_arange(num_intervals, action_spec.device).expand(
                        (*n_act, num_intervals)
                    )
                    * interval
                )
                low = action_spec.low
                if not empty_shape:
                    low = low.unsqueeze(-1)
                self.register_buffer("intervals", low + arange)
            else:
                arange = [
                    self._custom_arange(_num_intervals, action_spec.device) * interval
                    for _num_intervals, interval in zip(
                        num_intervals.tolist(), interval.unbind(-2)
                    )
                ]
                self.intervals = [
                    low + arange
                    for low, arange in zip(
                        action_spec.low.unsqueeze(-1).unbind(-2), arange
                    )
                ]

            if not isinstance(num_intervals, torch.Tensor):
                nvec = torch.as_tensor(num_intervals, device=action_spec.device)
            else:
                nvec = num_intervals
            if nvec.ndim > 1:
                raise RuntimeError(f"Cannot use num_intervals with shape {nvec.shape}")
            if nvec.ndim == 0 or nvec.numel() == 1:
                if not empty_shape:
                    nvec = nvec.expand(action_spec.shape[-1])
                else:
                    nvec = nvec.squeeze()
            self.register_buffer("nvec", nvec)
            if self.sampling == self.SamplingStrategy.RANDOM:
                # compute jitters
                self.jitters = interval.squeeze(-1) / nvec
            shape = (
                action_spec.shape
                if self.categorical
                else (*action_spec.shape[:-1], nvec.sum())
            )

            if not empty_shape:
                cls = (
                    functools.partial(MultiCategorical, remove_singleton=False)
                    if self.categorical
                    else MultiOneHot
                )
                action_spec = cls(nvec=nvec, shape=shape, device=action_spec.device)

            else:
                cls = Categorical if self.categorical else OneHot
                action_spec = cls(n=int(nvec), shape=shape, device=action_spec.device)

            batch_size = self.parent.batch_size
            if batch_size:
                action_spec = action_spec.expand(batch_size + action_spec.shape)
            input_spec["full_action_spec", self.out_keys_inv[0]] = action_spec

            if self.out_keys_inv[0] != self.in_keys_inv[0]:
                input_spec["full_state_spec", self.in_keys_inv[0]] = input_spec[
                    "full_action_spec", self.in_keys_inv[0]
                ].clone()
                del input_spec["full_action_spec", self.in_keys_inv[0]]
            return input_spec
        except AttributeError as err:
            # To avoid silent AttributeErrors
            raise RuntimeError(str(err))

    def _init(self):
        # We just need to access the action spec for everything to be initialized
        try:
            _ = self.container.full_action_spec
        except AttributeError:
            raise RuntimeError(
                f"Cannot execute transform {type(self).__name__} without a parent env."
            )

    def inv(self, tensordict):
        if self.out_keys_inv[0] == self.in_keys_inv[0]:
            return super().inv(tensordict)
        # We re-write this because we don't want to clone the TD here
        return self._inv_call(tensordict)

    def _inv_call(self, tensordict):
        # action is categorical, map it to desired dtype
        intervals = getattr(self, "intervals", None)
        if intervals is None:
            self._init()
            return self._inv_call(tensordict)
        action = tensordict.get(self.out_keys_inv[0])
        if self.categorical:
            action = action.unsqueeze(-1)
            if isinstance(intervals, torch.Tensor):
                shape = action.shape[: -intervals.ndim]
                intervals = intervals.expand(shape + intervals.shape)
                action = intervals.gather(index=action, dim=-1).squeeze(-1)
            else:
                action = torch.stack(
                    [
                        interval.gather(index=action, dim=-1).squeeze(-1)
                        for interval, action in zip(intervals, action.unbind(-2))
                    ],
                    -1,
                )
        else:
            nvec = self.nvec
            empty_shape = not nvec.ndim
            if not empty_shape:
                nvec = nvec.tolist()
                if isinstance(intervals, torch.Tensor):
                    shape = action.shape[: (-intervals.ndim + 1)]
                    intervals = intervals.expand(shape + intervals.shape)
                    intervals = intervals.unbind(-2)
                action = action.split(nvec, dim=-1)
                action = torch.stack(
                    [
                        intervals[action].view(action.shape[:-1])
                        for (intervals, action) in zip(intervals, action)
                    ],
                    -1,
                )
            else:
                shape = action.shape[: -intervals.ndim]
                intervals = intervals.expand(shape + intervals.shape)
                action = intervals[action].squeeze(-1)

        if self.sampling == self.SamplingStrategy.RANDOM:
            action = action + self.jitters * torch.rand_like(self.jitters)
        return tensordict.set(self.in_keys_inv[0], action)


class TrajCounter(Transform):
    """Global trajectory counter transform.

    TrajCounter can be used to count the number of trajectories (i.e., the number of times `reset` is called) in any
    TorchRL environment.
    This transform will work within a single node across multiple processes (see note below).
    A single transform can only count the trajectories associated with a single done state, but nested done states are
    accepted as long as their prefix matches the prefix of the counter key.

    Args:
        out_key (NestedKey, optional): The entry name of the trajectory counter. Defaults to ``"traj_count"``.

    Examples:
        >>> from torchrl.envs import GymEnv, StepCounter, TrajCounter
        >>> env = GymEnv("Pendulum-v1").append_transform(StepCounter(6))
        >>> env = env.append_transform(TrajCounter())
        >>> r = env.rollout(18, break_when_any_done=False)  # 18 // 6 = 3 trajectories
        >>> r["next", "traj_count"]
        tensor([[0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
                [2]])

    .. note::
        Sharing a trajectory counter among workers can be done in multiple ways, but it will usually involve wrapping the environment in a :class:`~torchrl.envs.EnvCreator`. Not doing so may result in an error during serialization of the transform. The counter will be shared among the workers, meaning that at any point in time, it is guaranteed that there will not be two environments that will share the same trajectory count (and each (step-count, traj-count) pair will be unique).
        Here are examples of valid ways of sharing a ``TrajCounter`` object between processes:

            >>> # Option 1: Create the trajectory counter outside the environment.
            >>> #  This requires the counter to be cloned within the transformed env, as a single transform object cannot have two parents.
            >>> t = TrajCounter()
            >>> def make_env(max_steps=4, t=t):
            ...     # See CountingEnv in torchrl.test.mocking_classes
            ...     env = TransformedEnv(CountingEnv(max_steps=max_steps), t.clone())
            ...     env.transform.transform_observation_spec(env.base_env.observation_spec)
            ...     return env
            >>> penv = ParallelEnv(
            ...     2,
            ...     [EnvCreator(make_env, max_steps=4), EnvCreator(make_env, max_steps=5)],
            ...     mp_start_method="spawn",
            ... )
            >>> # Option 2: Create the transform within the constructor.
            >>> #  In this scenario, we still need to tell each sub-env what kwarg has to be used.
            >>> #  Both EnvCreator and ParallelEnv offer that possibility.
            >>> def make_env(max_steps=4):
            ...     t = TrajCounter()
            ...     env = TransformedEnv(CountingEnv(max_steps=max_steps), t)
            ...     env.transform.transform_observation_spec(env.base_env.observation_spec)
            ...     return env
            >>> make_env_c0 = EnvCreator(make_env)
            >>> # Create a variant of the env with different kwargs
            >>> make_env_c1 = make_env_c0.make_variant(max_steps=5)
            >>> penv = ParallelEnv(
            ...     2,
            ...     [make_env_c0, make_env_c1],
            ...     mp_start_method="spawn",
            ... )
            >>> # Alternatively, pass the kwargs to the ParallelEnv
            >>> penv = ParallelEnv(
            ...     2,
            ...     [make_env_c0, make_env_c0],
            ...     create_env_kwargs=[{"max_steps": 5}, {"max_steps": 4}],
            ...     mp_start_method="spawn",
            ... )

    """

    def __init__(
        self, out_key: NestedKey = "traj_count", *, repeats: int | None = None
    ):
        super().__init__(in_keys=[], out_keys=[out_key])
        self._make_shared_value()
        self._initialized = False
        if repeats is None:
            repeats = 0
        self.repeats = repeats

    def _make_shared_value(self):
        self._traj_count = mp.Value("i", 0)

    def __getstate__(self):
        state = super().__getstate__()
        state["_traj_count"] = None
        return state

    def clone(self):
        clone = super().clone()
        # All clones share the same _traj_count and lock
        clone._traj_count = self._traj_count
        return clone

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            self._initialized = True
        rk = self.parent.reset_keys
        traj_count_key = self.out_keys[0]
        is_str = isinstance(traj_count_key, str)
        for _rk in rk:
            if is_str and isinstance(_rk, str):
                rk = _rk
                break
            elif (
                not is_str
                and isinstance(_rk, tuple)
                and _rk[:-1] == traj_count_key[:-1]
            ):
                rk = _rk
                break
        else:
            raise RuntimeError(
                f"Did not find reset key that matched the prefix of the traj counter key. Reset keys: {rk}, traj count: {traj_count_key}"
            )
        reset = None
        if tensordict is not None:
            reset = tensordict.get(rk, default=None)
        if reset is None:
            reset = torch.ones(
                self.container.observation_spec[self.out_keys[0]].shape,
                device=tensordict_reset.device,
                dtype=torch.bool,
            )
        with (self._traj_count):
            tc = int(self._traj_count.value)
            self._traj_count.value = self._traj_count.value + reset.sum().item()
            episodes = torch.arange(tc, tc + reset.sum(), device=self.parent.device)
            episodes = torch.masked_scatter(
                torch.zeros_like(reset, dtype=episodes.dtype), reset, episodes
            )
            tensordict_reset.set(traj_count_key, episodes)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            raise RuntimeError("_step was called before _reset was called.")
        next_tensordict.set(self.out_keys[0], tensordict.get(self.out_keys[0]))
        return next_tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            f"{type(self).__name__} can only be called within an environment step or reset."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            f"{type(self).__name__} can only be called within an environment step or reset."
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {
            "traj_count": int(self._traj_count.value),
        }

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self._traj_count.value *= 0
        self._traj_count.value += state_dict["traj_count"]

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )
        full_done_spec = self.parent.output_spec["full_done_spec"]
        traj_count_key = self.out_keys[0]
        # find a matching done key (there might be more than one)
        for done_key in self.parent.done_keys:
            # check root
            if type(done_key) != type(traj_count_key):
                continue
            if isinstance(done_key, tuple):
                if done_key[:-1] == traj_count_key[:-1]:
                    shape = full_done_spec[done_key].shape
                    break
            if isinstance(done_key, str):
                shape = full_done_spec[done_key].shape
                break

        else:
            raise KeyError(
                f"Could not find root of traj_count key {traj_count_key} in done keys {self.done_keys}."
            )
        observation_spec[traj_count_key] = Bounded(
            shape=shape,
            dtype=torch.int64,
            device=observation_spec.device,
            low=0,
            high=torch.iinfo(torch.int64).max,
        )
        return super().transform_observation_spec(observation_spec)


class LineariseRewards(Transform):
    """Transforms a multi-objective reward signal to a single-objective one via a weighted sum.

    Args:
        in_keys (List[NestedKey]): The keys under which the multi-objective rewards are found.
        out_keys (List[NestedKey], optional): The keys under which single-objective rewards should be written. Defaults to :attr:`in_keys`.
        weights (List[float], Tensor, optional): Dictates how to weight each reward when summing them. Defaults to `[1.0, 1.0, ...]`.

    .. warning::
        If a sequence of `in_keys` of length strictly greater than one is passed (e.g. one group for each agent in a
        multi-agent set-up), the same weights will be applied for each entry. If you need to aggregate rewards
        differently for each group, use several :class:`~torchrl.envs.LineariseRewards` in a row.

    Example:
        >>> import mo_gymnasium as mo_gym
        >>> from torchrl.envs import MOGymWrapper
        >>> mo_env = MOGymWrapper(mo_gym.make("deep-sea-treasure-v0"))
        >>> mo_env.reward_spec
        BoundedContinuous(
            shape=torch.Size([2]),
            space=ContinuousBox(
            low=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True),
            high=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True)),
            ...)
        >>> so_env = TransformedEnv(mo_env, LineariseRewards(in_keys=("reward",)))
        >>> so_env.reward_spec
        BoundedContinuous(
            shape=torch.Size([1]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
            ...)
        >>> td = so_env.rollout(5)
        >>> td["next", "reward"].shape
        torch.Size([5, 1])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey] | None = None,
        *,
        weights: Sequence[float] | Tensor | None = None,
    ) -> None:
        out_keys = in_keys if out_keys is None else out_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        if weights is not None:
            weights = weights if isinstance(weights, Tensor) else torch.tensor(weights)

            # This transform should only receive vectorial weights (all batch dimensions will be aggregated similarly).
            if weights.ndim >= 2:
                raise ValueError(
                    f"Expected weights to be a unidimensional tensor. Got {weights.ndim} dimension."
                )

            # Avoids switching from reward to costs.
            if (weights < 0).any():
                raise ValueError(f"Expected all weights to be >0. Got {weights}.")

            self.register_buffer("weights", weights)
        else:
            self.weights = None

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if not reward_spec.domain == "continuous":
            raise NotImplementedError(
                "Aggregation of rewards that take discrete values is not supported."
            )

        *batch_size, num_rewards = reward_spec.shape
        weights = (
            torch.ones(num_rewards, device=reward_spec.device, dtype=reward_spec.dtype)
            if self.weights is None
            else self.weights
        )

        num_weights = torch.numel(weights)
        if num_weights != num_rewards:
            raise ValueError(
                "The number of rewards and weights should match. "
                f"Got: {num_rewards} and {num_weights}"
            )

        if isinstance(reward_spec, UnboundedContinuous):
            reward_spec.shape = torch.Size([*batch_size, 1])
            return reward_spec

        # The lines below are correct only if all weights are positive.
        low = (weights * reward_spec.space.low).sum(dim=-1, keepdim=True)
        high = (weights * reward_spec.space.high).sum(dim=-1, keepdim=True)

        return BoundedContinuous(
            low=low,
            high=high,
            device=reward_spec.device,
            dtype=reward_spec.dtype,
        )

    def _apply_transform(self, reward: Tensor) -> TensorDictBase:
        if self.weights is None:
            return reward.sum(dim=-1)

        *batch_size, num_rewards = reward.shape
        num_weights = torch.numel(self.weights)
        if num_weights != num_rewards:
            raise ValueError(
                "The number of rewards and weights should match. "
                f"Got: {num_rewards} and {num_weights}."
            )

        return (self.weights * reward).sum(dim=-1)


class ConditionalSkip(Transform):
    """A transform that skips steps in the env if certain conditions are met.

    This transform writes the result of `cond(tensordict)` in the `"_step"` entry of the
    tensordict passed as input to the `TransformedEnv.base_env._step` method.
    If the `base_env` is not batch-locked (generally speaking, it is stateless), the tensordict is
    reduced to its element that need to go through the step.
    If it is batch-locked (generally speaking, it is stateful), the step is skipped altogether if no
    value in `"_step"` is ``True``. Otherwise, it is trusted that the environment will account for the
    `"_step"` signal accordingly.

    .. note:: The skip will affect transforms that modify the environment output too, i.e., any transform
        that is to be exectued on the tensordict returned by :meth:`~torchrl.envs.EnvBase.step` will be
        skipped if the condition is met. To palliate this effect if it is not desirable, one can wrap
        the transformed env in another transformed env, since the skip only affects the first-degree parent
        of the ``ConditionalSkip`` transform. See example below.

    Args:
        cond (Callable[[TensorDictBase], bool | torch.Tensor]): a callable for the tensordict input
            that checks whether the next env step must be skipped (`True` = skipped, `False` = execute
            env.step).

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.envs.transforms.transforms import ConditionalSkip, StepCounter, TransformedEnv, Compose
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> base_env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     StepCounter(step_count_key="inner_count"),
        ... )
        >>> middle_env = TransformedEnv(
        ...     base_env,
        ...     Compose(
        ...         StepCounter(step_count_key="middle_count"),
        ...         ConditionalSkip(cond=lambda td: td["step_count"] % 2 == 1),
        ...     ),
        ...     auto_unwrap=False)  # makes sure that transformed envs are properly wrapped
        >>> env = TransformedEnv(
        ...     middle_env,
        ...     StepCounter(step_count_key="step_count"),
        ...     auto_unwrap=False)
        >>> env.set_seed(0)
        >>>
        >>> r = env.rollout(10)
        >>> print(r["observation"])
        tensor([[-0.9670, -0.2546, -0.9669],
                [-0.9802, -0.1981, -1.1601],
                [-0.9802, -0.1981, -1.1601],
                [-0.9926, -0.1214, -1.5556],
                [-0.9926, -0.1214, -1.5556],
                [-0.9994, -0.0335, -1.7622],
                [-0.9994, -0.0335, -1.7622],
                [-0.9984,  0.0561, -1.7933],
                [-0.9984,  0.0561, -1.7933],
                [-0.9895,  0.1445, -1.7779]])
        >>> print(r["inner_count"])
        tensor([[0],
                [1],
                [1],
                [2],
                [2],
                [3],
                [3],
                [4],
                [4],
                [5]])
        >>> print(r["middle_count"])
        tensor([[0],
                [1],
                [1],
                [2],
                [2],
                [3],
                [3],
                [4],
                [4],
                [5]])
        >>> print(r["step_count"])
        tensor([[0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]])


    """

    def __init__(self, cond: Callable[[TensorDict], bool | torch.Tensor]):
        super().__init__()
        self.cond = cond

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Run cond
        cond = self.cond(tensordict)
        # Write result in step
        tensordict["_step"] = tensordict.get("_step", True) & ~cond
        if tensordict["_step"].shape != tensordict.batch_size:
            tensordict["_step"] = tensordict["_step"].view(tensordict.batch_size)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )


class MultiAction(Transform):
    """A transform to execute multiple actions in the parent environment.

    This transform unbinds the actions along a specific dimension and passes each action independently.
    The returned transform can be either a stack of the observations gathered during the steps or only the
    last observation (and similarly for the rewards, see args below).

    By default, the actions must be stacked along the first dimension after the root tensordict batch-dims, i.e.

        >>> td = policy(td)
        >>> actions = td.select(*env.action_keys)
        >>> # Adapt the batch-size
        >>> actions = actions.auto_batch_size_(td.ndim + 1)
        >>> # Step-wise actions
        >>> actions = actions.unbind(-1)

    If a `"done"` entry is encountered, the next steps are skipped for the env that has reached that state.

    .. note:: If a transform is appended before the MultiAction, it will be called multiple times. If it is appended
        after, it will be called once per macro-step.

    Keyword Args:
        dim (int, optional): the stack dimension with respect to the tensordict ``ndim`` attribute.
            Must be greater than 0. Defaults to ``1`` (the first dimension after the batch-dims).
        stack_rewards (bool, optional): if ``True``, each step's reward will be stack in the output tensordict.
            If ``False``, only the last reward will be returned. The reward spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``True``.
        stack_observations (bool, optional): if ``True``, each step's observation will be stack in the output tensordict.
            If ``False``, only the last observation will be returned. The observation spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``False``.

    """

    def __init__(
        self,
        *,
        dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
    ):
        super().__init__()
        self.stack_rewards = stack_rewards
        self.stack_observations = stack_observations
        self.dim = dim

    def _stack_tds(self, td_list, next_tensordict, keys):
        td = torch.stack(td_list + [next_tensordict.select(*keys)], -1)
        if self.dim != 1:
            d = td.ndim - 1
            td.auto_batch_size_(d + self.dim)
            td = td.transpose(d, d + self.dim)
        return td

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # Collect the stacks if needed
        if self.stack_rewards:
            reward_td = self.rewards
            reward_td = self._stack_tds(
                reward_td, next_tensordict, self.parent.reward_keys
            )
            next_tensordict.update(reward_td)
        if self.stack_observations:
            obs_td = self.obs
            obs_td = self._stack_tds(
                obs_td, next_tensordict, self.parent.observation_keys
            )
            next_tensordict.update(obs_td)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Get the actions
        parent = self.parent
        action_keys = parent.action_keys
        actions = tensordict.select(*action_keys)
        actions = actions.auto_batch_size_(batch_dims=tensordict.ndim + self.dim)
        actions = actions.unbind(-1)
        td = tensordict
        idx = None
        global_idx = None
        reset = False
        if self.stack_rewards:
            self.rewards = rewards = []
        if self.stack_observations:
            self.obs = obs = []
        for a in actions[:-1]:
            if global_idx is not None:
                a = a[global_idx]
            td = td.replace(a)
            td = parent.step(td)

            # Save rewards and done states
            if self.stack_rewards:
                reward_td = td["next"].select(*self.parent.reward_keys)
                if global_idx is not None:
                    reward_td_expand = reward_td.new_zeros(
                        global_idx.shape + reward_td.shape[global_idx.ndim :]
                    )
                    reward_td_expand[global_idx] = reward_td
                else:
                    reward_td_expand = reward_td

                rewards.append(reward_td_expand)
            if self.stack_observations:
                obs_td = td["next"].select(*self.parent.observation_keys)
                # obs_td = td.select("next", *self.parent.observation_keys).set("next", obs_td)
                if global_idx is not None:
                    obs_td = torch.where(global_idx, obs_td, 0)
                obs.append(obs_td)

            td = parent.step_mdp(td)
            if self.stack_rewards:
                td.update(reward_td)

            any_done = parent.any_done(td)
            if any_done:
                # Intersect the resets to avoid making any step after reset has been called
                reset = reset | td.pop("_reset").view(td.shape)
                if reset.all():
                    # Skip step for all
                    td["_step"] = ~reset
                    break
                elif parent.batch_locked:
                    td["_step"] = ~reset
                else:
                    # we can simply index the tensordict
                    idx = ~reset.view(td.shape)
                    if global_idx is None:
                        global_idx = idx.clone()
                        td_out = td
                    else:
                        td_out[global_idx] = td
                        global_idx = torch.masked_scatter(global_idx, global_idx, idx)
                    td = td[idx]
                    reset = reset[idx]  # Should be all False

        if global_idx is None:
            td_out = td.replace(actions[-1])
            if (self.stack_rewards or self.stack_observations) and not td_out.get(
                "_step", torch.ones((), dtype=torch.bool)
            ).any():
                td_out = self._step(None, td_out)
        else:
            td_out[global_idx] = td.replace(actions[-1][global_idx])
            if self.stack_rewards or self.stack_observations:
                td_out = self._step(None, td_out)
                if self.stack_rewards:
                    self.rewards = list(
                        torch.stack(self.rewards, -1)[global_idx].unbind(-1)
                    )
                if self.stack_observations:
                    self.obs = list(torch.stack(self.obs, -1)[global_idx].unbind(-1))

            td_out["_step"] = global_idx

        return td_out

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        try:
            action_spec = input_spec["full_action_spec"]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} requires an action spec to be present."
            )
        for _ in range(self.dim):
            action_spec = action_spec.unsqueeze(input_spec.ndim)
        # Make the dim dynamic
        action_spec = action_spec.expand(
            tuple(
                d if i != (input_spec.ndim + self.dim - 1) else -1
                for i, d in enumerate(action_spec.shape)
            )
        )
        input_spec["full_action_spec"] = action_spec
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self._transform_reward_spec(
                output_spec["full_reward_spec"], output_spec.ndim
            )
        if "full_observation_spec" in output_spec.keys():
            output_spec["full_observation_spec"] = self._transform_observation_spec(
                output_spec["full_observation_spec"], output_spec.ndim
            )
        return output_spec

    def _transform_reward_spec(self, reward_spec: TensorSpec, ndim) -> TensorSpec:
        if not self.stack_rewards:
            return reward_spec
        for _ in range(self.dim):
            reward_spec = reward_spec.unsqueeze(ndim)
        # Make the dim dynamic
        reward_spec = reward_spec.expand(
            tuple(
                d if i != (ndim + self.dim - 1) else -1
                for i, d in enumerate(reward_spec.shape)
            )
        )
        return reward_spec

    def _transform_observation_spec(
        self, observation_spec: TensorSpec, ndim
    ) -> TensorSpec:
        if not self.stack_observations:
            return observation_spec
        for _ in range(self.dim):
            observation_spec = observation_spec.unsqueeze(ndim)
        # Make the dim dynamic
        observation_spec = observation_spec.expand(
            tuple(
                d if i != (ndim + self.dim - 1) else -1
                for i, d in enumerate(observation_spec.shape)
            )
        )
        return observation_spec


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
            self._maybe_expand_and_set(self.out_keys[2], time_elapsed, tensordict_reset)
            self._maybe_expand_and_set(
                self.out_keys[0], time_elapsed * 0, tensordict_reset
            )
        self.last_call_time = current_time
        # Placeholder
        self._maybe_expand_and_set(self.out_keys[1], time_elapsed * 0, tensordict_reset)
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        current_time = time.time()
        if self.last_call_time is not None:
            time_elapsed = torch.tensor(
                current_time - self.last_call_time, device=tensordict.device
            )
            self._maybe_expand_and_set(self.out_keys[1], time_elapsed, tensordict)
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
            self._maybe_expand_and_set(self.out_keys[0], time_elapsed, next_tensordict)
            self._maybe_expand_and_set(
                self.out_keys[2], time_elapsed * 0, next_tensordict
            )
        self.last_call_time = current_time
        # presumbly no need to worry about batch size incongruencies here
        next_tensordict.set(self.out_keys[1], tensordict.get(self.out_keys[1]))
        return next_tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec[self.out_keys[0]] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        observation_spec[self.out_keys[1]] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        observation_spec[self.out_keys[2]] = Unbounded(
            shape=observation_spec.shape, device=observation_spec.device
        )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(FORWARD_NOT_IMPLEMENTED)


class ConditionalPolicySwitch(Transform):
    """A transform that conditionally switches between policies based on a specified condition.

    This transform evaluates a condition on the data returned by the environment's `step` method.
    If the condition is met, it applies a specified policy to the data. Otherwise, the data is
    returned unaltered. This is useful for scenarios where different policies need to be applied
    based on certain criteria, such as alternating turns in a game.

    Args:
        policy (Callable[[TensorDictBase], TensorDictBase]):
            The policy to be applied when the condition is met. This should be a callable that
            takes a `TensorDictBase` and returns a `TensorDictBase`.
        condition (Callable[[TensorDictBase], bool]):
            A callable that takes a `TensorDictBase` and returns a boolean or a tensor indicating
            whether the policy should be applied.

    .. warning:: This transform must have a parent environment.

    .. note:: Ideally, it should be the last transform  in the stack. If the policy requires transformed
        data (e.g., images), and this transform  is applied before those transformations, the policy will
        not receive the transformed data.

    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule as Mod
        >>>
        >>> from torchrl.envs import GymEnv, ConditionalPolicySwitch, Compose, StepCounter
        >>> # Create a CartPole environment. We'll be looking at the obs: if the first element of the obs is greater than
        >>> # 0 (left position) we do a right action (action=0) using the switch policy. Otherwise, we use our main
        >>> # policy which does a left action.
        >>> base_env = GymEnv("CartPole-v1", categorical_action_encoding=True)
        >>>
        >>> policy = Mod(lambda: torch.ones((), dtype=torch.int64), in_keys=[], out_keys=["action"])
        >>> policy_switch = Mod(lambda: torch.zeros((), dtype=torch.int64), in_keys=[], out_keys=["action"])
        >>>
        >>> cond = lambda td: td.get("observation")[..., 0] >= 0
        >>>
        >>> env = base_env.append_transform(
        ...     Compose(
        ...         # We use two step counters to show that one counts the global steps, whereas the other
        ...         # only counts the steps where the main policy is executed
        ...         StepCounter(step_count_key="step_count_total"),
        ...         ConditionalPolicySwitch(condition=cond, policy=policy_switch),
        ...         StepCounter(step_count_key="step_count_main"),
        ...     )
        ... )
        >>>
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>>
        >>> r = env.rollout(100, policy=policy)
        >>> print("action", r["action"])
        action tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> print("obs", r["observation"])
        obs tensor([[ 0.0322, -0.1540,  0.0111,  0.3190],
                [ 0.0299, -0.1544,  0.0181,  0.3280],
                [ 0.0276, -0.1550,  0.0255,  0.3414],
                [ 0.0253, -0.1558,  0.0334,  0.3596],
                [ 0.0230, -0.1569,  0.0422,  0.3828],
                [ 0.0206, -0.1582,  0.0519,  0.4117],
                [ 0.0181, -0.1598,  0.0629,  0.4469],
                [ 0.0156, -0.1617,  0.0753,  0.4891],
                [ 0.0130, -0.1639,  0.0895,  0.5394],
                [ 0.0104, -0.1665,  0.1058,  0.5987],
                [ 0.0076, -0.1696,  0.1246,  0.6685],
                [ 0.0047, -0.1732,  0.1463,  0.7504],
                [ 0.0016, -0.1774,  0.1715,  0.8459],
                [-0.0020,  0.0150,  0.1884,  0.6117],
                [-0.0017,  0.2071,  0.2006,  0.3838]])
        >>> print("obs'", r["next", "observation"])
        obs' tensor([[ 0.0299, -0.1544,  0.0181,  0.3280],
                [ 0.0276, -0.1550,  0.0255,  0.3414],
                [ 0.0253, -0.1558,  0.0334,  0.3596],
                [ 0.0230, -0.1569,  0.0422,  0.3828],
                [ 0.0206, -0.1582,  0.0519,  0.4117],
                [ 0.0181, -0.1598,  0.0629,  0.4469],
                [ 0.0156, -0.1617,  0.0753,  0.4891],
                [ 0.0130, -0.1639,  0.0895,  0.5394],
                [ 0.0104, -0.1665,  0.1058,  0.5987],
                [ 0.0076, -0.1696,  0.1246,  0.6685],
                [ 0.0047, -0.1732,  0.1463,  0.7504],
                [ 0.0016, -0.1774,  0.1715,  0.8459],
                [-0.0020,  0.0150,  0.1884,  0.6117],
                [-0.0017,  0.2071,  0.2006,  0.3838],
                [ 0.0105,  0.2015,  0.2115,  0.5110]])
        >>> print("total step count", r["step_count_total"].squeeze())
        total step count tensor([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27])
        >>> print("total step with main policy", r["step_count_main"].squeeze())
        total step with main policy tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

    """

    def __init__(
        self,
        policy: Callable[[TensorDictBase], TensorDictBase],
        condition: Callable[[TensorDictBase], bool],
    ):
        super().__init__([], [])
        self.__dict__["policy"] = policy
        self.condition = condition

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        cond = self.condition(next_tensordict)
        if not isinstance(cond, (bool, torch.Tensor)):
            raise RuntimeError(
                "Calling the condition function should return a boolean or a tensor."
            )
        elif isinstance(cond, (torch.Tensor,)):
            if tuple(cond.shape) not in ((1,), (), tuple(tensordict.shape)):
                raise RuntimeError(
                    "Tensor outputs must have the shape of the tensordict, or contain a single element."
                )
        else:
            cond = torch.tensor(cond, device=tensordict.device)

        if cond.any():
            step = tensordict.get("_step", cond)
            if step.shape != cond.shape:
                step = step.view_as(cond)
            cond = cond & step

            parent: TransformedEnv = self.parent
            any_done, done = self._check_done(next_tensordict)
            next_td_save = None
            if any_done:
                if next_tensordict.numel() == 1 or done.all():
                    return next_tensordict
                if parent.base_env.batch_locked:
                    raise RuntimeError(
                        "Cannot run partial steps in a batched locked environment. "
                        "Hint: Parallel and Serial envs can be unlocked through a keyword argument in "
                        "the constructor."
                    )
                done = done.view(next_tensordict.shape)
                cond = cond & ~done
            if not cond.all():
                if parent.base_env.batch_locked:
                    raise RuntimeError(
                        "Cannot run partial steps in a batched locked environment. "
                        "Hint: Parallel and Serial envs can be unlocked through a keyword argument in "
                        "the constructor."
                    )
                next_td_save = next_tensordict
                next_tensordict = next_tensordict[cond]
                tensordict = tensordict[cond]

            # policy may be expensive or raise an exception when executed with unadequate data so
            # we index the td first
            td = self.policy(
                parent.step_mdp(tensordict.copy().set("next", next_tensordict))
            )
            # Mark the partial steps if needed
            if next_td_save is not None:
                td_new = td.new_zeros(cond.shape)
                # TODO: swap with masked_scatter when avail
                td_new[cond] = td
                td = td_new
                td.set("_step", cond)
            next_tensordict = parent._step(td)
            if next_td_save is not None:
                return torch.where(cond, next_tensordict, next_td_save)
            return next_tensordict
        return next_tensordict

    def _check_done(self, tensordict):
        env = self.parent
        if env._simple_done:
            done = tensordict._get_str("done", default=None)
            if done is not None:
                any_done = done.any()
            else:
                any_done = False
        else:
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=env.output_spec["full_done_spec"],
                key="_reset",
            )
            done = tensordict.pop("_reset")
        return any_done, done

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        cond = self.condition(tensordict_reset)
        # TODO: move to validate
        if not isinstance(cond, (bool, torch.Tensor)):
            raise RuntimeError(
                "Calling the condition function should return a boolean or a tensor."
            )
        elif isinstance(cond, (torch.Tensor,)):
            if tuple(cond.shape) not in ((1,), (), tuple(tensordict.shape)):
                raise RuntimeError(
                    "Tensor outputs must have the shape of the tensordict, or contain a single element."
                )
        else:
            cond = torch.tensor(cond, device=tensordict.device)

        if cond.any():
            reset = tensordict.get("_reset", cond)
            if reset.shape != cond.shape:
                reset = reset.view_as(cond)
            cond = cond & reset

            parent: TransformedEnv = self.parent
            reset_td_save = None
            if not cond.all():
                reset_td_save = tensordict_reset.copy()
                tensordict_reset = tensordict_reset[cond]
                tensordict = tensordict[cond]

            td = self.policy(tensordict_reset)
            # Mark the partial steps if needed
            if reset_td_save is not None:
                td_new = td.new_zeros(cond.shape)
                # TODO: swap with masked_scatter when avail
                td_new[cond] = td
                td = td_new
                td.set("_step", cond)
            tensordict_reset = parent._step(td).exclude(*parent.reward_keys)
            if reset_td_save is not None:
                return torch.where(cond, tensordict_reset, reset_td_save)
            return tensordict_reset

        return tensordict_reset

    def forward(self, tensordict: TensorDictBase) -> Any:
        raise RuntimeError(
            "ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional."
        )
