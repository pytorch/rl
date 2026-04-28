# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from copy import copy
from functools import wraps
from textwrap import indent
from typing import Any, overload, TYPE_CHECKING, TypeVar, Union

import torch

from tensordict import TensorDict, TensorDictBase, unravel_key
from tensordict.base import _is_leaf_nontensor
from tensordict.nn import dispatch
from tensordict.utils import _zip_strict, NestedKey
from torch import nn
from torch.utils._pytree import tree_map

from torchrl._utils import auto_unwrap_transformed_env, logger as torchrl_logger

from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.envs.common import _EnvPostInit, _maybe_unlock, EnvBase
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.envs.utils import _update_during_reset

if TYPE_CHECKING:
    pass


_has_tv = importlib.util.find_spec("torchvision", None) is not None

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."

T = TypeVar("T", bound="Transform")

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

__all__ = [
    "AutoResetEnv",
    "Compose",
    "ObservationTransform",
    "Transform",
    "TransformedEnv",
]


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
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__()
        if in_keys is not None:
            self.in_keys = in_keys
        if out_keys is not None:
            self.out_keys = out_keys
        if in_keys_inv is not None:
            self.in_keys_inv = in_keys_inv
        if out_keys_inv is not None:
            self.out_keys_inv = out_keys_inv
        self._missing_tolerance = False
        # we use __dict__ to avoid having nn.Module placing these objects in the module list
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def _getattr(self, val, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError(
                    f"Expected at most 1 positional argument, got {len(args)}"
                )
            default = args[0]
            return getattr(self, val, default)
        if kwargs:
            try:
                default = kwargs.pop("default")
            except KeyError:
                raise TypeError("Only 'default' keyword argument is supported")
            if args:
                raise TypeError("Got two values for keyword argument 'default'")
            return getattr(self, val, default)
        return getattr(self, val)

    def _ready(self):
        # Used to block ray until the actor is ready, see RayTransform
        return True

    def close(self):
        """Close the transform."""

    @property
    def in_keys(self) -> Sequence[NestedKey]:
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
    def out_keys(self) -> Sequence[NestedKey]:
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
    def in_keys_inv(self) -> Sequence[NestedKey]:
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
    def out_keys_inv(self) -> Sequence[NestedKey]:
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

    @property
    def collector(self) -> BaseCollector | None:  # noqa: F821 # type: ignore
        """Returns the collector associated with the container, if it exists.

        This can be used whenever the transform needs to be made aware of the collector or the policy associated with it.

        Make sure to call this property only on transforms that are not nested in sub-processes.
        The collector reference will not be passed to the workers of a :class:`~torchrl.envs.ParallelEnv` or
        similar batched environments.

        """
        return self.container.collector

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

    def _set_attr(self, name, value):
        """Set attribute on the remote actor or locally."""
        setattr(self, name, value)

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
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

    def transform_env_device(self, device: torch.device) -> torch.device:
        """Transforms the device of the parent env."""
        return device

    def transform_env_batch_size(self, batch_size: torch.Size) -> torch.Size:
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
                raise KeyError(
                    f"The key '{key}' is unaccounted for by the transform (expected keys {output_spec_keys}). "
                    f"Every new entry in the tensordict resulting from a call to a transform must be "
                    f"registered in the specs for torchrl rollouts to be consistently built. "
                    f"Make sure transform_output_spec/transform_observation_spec/... is coded correctly."
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

    def clone(self) -> Self:
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
    def container(self) -> EnvBase | None:
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
    def parent(self) -> TransformedEnv | None:
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

    def empty_cache(self) -> None:
        self.__dict__["_parent"] = None

    def set_missing_tolerance(self, mode=False) -> None:
        self._missing_tolerance = mode

    @property
    def missing_tolerance(self) -> bool:
        return self._missing_tolerance

    def to(self, *args, **kwargs) -> Transform:
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
    """A transformed environment.

    Args:
        base_env (EnvBase): original environment to be transformed.
        transform (Transform or callable, optional): transform to apply to the tensordict resulting
            from :obj:`base_env.step(td)`. If none is provided, an empty Compose
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
            transformed only once). If the transform changes during
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

    @overload
    def __init__(
        self,
        base_env: EnvBase,
        transform: Transform | None = None,
        cache_specs: bool = True,
        *,
        auto_unwrap: bool | None = None,
        **kwargs,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        base_env: EnvBase,
        transform: Transform | None = None,
        cache_specs: bool = True,
        auto_unwrap: bool | None = None,
        **kwargs,
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Backward compatibility: handle both old and new syntax
        if len(args) > 0:
            # New syntax: TransformedEnv(base_env, transform, ...)
            base_env = args[0]
            transform = args[1] if len(args) > 1 else kwargs.pop("transform", None)
            cache_specs = args[2] if len(args) > 2 else kwargs.pop("cache_specs", True)
            auto_unwrap = kwargs.pop("auto_unwrap", None)
        elif "env" in kwargs:
            raise TypeError(
                "The 'env' argument has been removed. Use 'base_env' instead."
            )
        elif "base_env" in kwargs:
            # New syntax with keyword arguments: TransformedEnv(base_env=..., transform=...)
            base_env = kwargs.pop("base_env")
            transform = kwargs.pop("transform", None)
            cache_specs = kwargs.pop("cache_specs", True)
            auto_unwrap = kwargs.pop("auto_unwrap", None)
        else:
            raise TypeError("TransformedEnv requires a base_env argument")

        self._transform = None
        device = kwargs.pop("device", None)
        if device is not None:
            base_env = base_env.to(device)
        else:
            device = base_env.device
        super().__init__(device=None, allow_done_after_reset=None, **kwargs)

        # Type matching must be exact here, because subtyping could introduce differences in behavior that must
        # be contained within the subclass.
        if type(base_env) is TransformedEnv and type(self) is TransformedEnv:
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
            self._set_env(base_env.base_env, device)
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
            env_transform = base_env.transform.clone()
            if type(env_transform) is not Compose:
                env_transform = [env_transform]
            else:
                for t in env_transform:
                    t.reset_parent()
            transform = Compose(*env_transform, *transform).to(device)
        else:
            self._set_env(base_env, device)
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

    # def _post_step_mdp_hooks(self, tensordict: TensorDictBase) -> TensorDictBase:
    # """Allows modification of the tensordict after the step_mdp."""
    # if type(self.base_env)._post_step_mdp_hooks is not None:
    # If the base env has a _post_step_mdp_hooks, we call it
    # tensordict = self.base_env._post_step_mdp_hooks(tensordict)
    # return tensordict

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
    def _inplace_update(self) -> bool:
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
        self.transform.empty_cache()
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

    The class can be instantiated in several ways:

    Args:
        *transforms (Transform): Variable number of transforms to compose.
        transforms (list[Transform], optional): A list of transforms to compose.
            This can be passed as a keyword argument.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>>
        >>> # Method 1: Using positional arguments
        >>> transforms = Compose(RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0))
        >>> transformed_env = TransformedEnv(env, transforms)
        >>>
        >>> # Method 2: Using a list with positional argument
        >>> transform_list = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(transform_list)
        >>> transformed_env = TransformedEnv(env, transforms)
        >>>
        >>> # Method 3: Using keyword argument
        >>> transforms = Compose(transforms=[RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)])
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    @overload
    def __init__(self, transforms: list[Transform]):
        ...

    def __init__(self, *trsfs: Transform, **kwargs):
        if len(trsfs) == 0 and "transforms" in kwargs:
            transforms = kwargs.pop("transforms")
        elif len(trsfs) == 1 and isinstance(trsfs[0], list):
            transforms = trsfs[0]
        else:
            transforms = trsfs
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
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

    def pop(self, index: int | None = None) -> Transform:
        """Pop a transform from the chain.

        Args:
            index (int, optional): The index of the transform to pop. If None, the last transform is popped.

        Returns:
            The popped transform.
        """
        if index is None:
            index = len(self.transforms) - 1
        result = self.transforms.pop(index)
        parent = self.parent
        self.empty_cache()
        if parent is not None:
            parent.empty_cache()
        return result

    def __delitem__(self, index: int | slice | list):
        """Delete a transform in the chain.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        del self.transforms[index]
        parent = self.parent
        self.empty_cache()
        if parent is not None:
            parent.empty_cache()

    def __setitem__(
        self,
        index: int | slice | list,
        value: Transform | Callable[[TensorDictBase], TensorDictBase],
    ):
        """Set a transform in the chain.

        :class:`~torchrl.envs.transforms.Transform` or callable are accepted.
        """
        self.transforms[index] = value
        parent = self.parent
        self.empty_cache()
        if parent is not None:
            parent.empty_cache()

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
        if type(self) is type(transform) is Compose:
            for t in transform:
                self.append(t)
        else:
            self.transforms.append(transform)
        transform.set_container(self)
        parent = self.parent
        if parent is not None:
            parent.empty_cache()

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
        parent = self.parent
        if parent is not None:
            parent.empty_cache()
        if index < 0:
            index = index + len(self.transforms)
        transform.eval()
        self.transforms.insert(index, transform)
        transform.set_container(self)

    def __iter__(self) -> Iterator[Transform]:
        yield from self.transforms

    def __len__(self) -> int:
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

    def empty_cache(self) -> None:
        for t in self.transforms:
            t.empty_cache()
        super().empty_cache()

    def reset_parent(self) -> None:
        for t in self.transforms:
            t.reset_parent()
        super().reset_parent()

    def clone(self) -> Self:
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


class _CallableTransform(Transform):
    # A wrapper around a custom callable to make it possible to transform any data type
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs) -> TensorDictBase:
        return self.func(*args, **kwargs)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self.func(next_tensordict)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


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
