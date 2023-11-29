# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import multiprocessing as mp
import warnings
from copy import copy
from functools import wraps
from textwrap import indent
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np

import torch

from tensordict import unravel_key, unravel_key_list
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.nn import dispatch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torch import nn, Tensor

from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    ContinuousBox,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvPostInit, EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import _replace_last, _sort_keys, _update_during_reset, step_mdp
from torchrl.objectives.value.functional import reward2go

try:
    from torchvision.transforms.functional import center_crop

    try:
        from torchvision.transforms.functional import InterpolationMode, resize

        def interpolation_fn(interpolation):  # noqa: D103
            return InterpolationMode(interpolation)

    except ImportError:

        def interpolation_fn(interpolation):  # noqa: D103
            return interpolation

        from torchvision.transforms.functional_tensor import resize

    _has_tv = True
except ImportError:
    _has_tv = False

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."


def _apply_to_composite(function):
    @wraps(function)
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, CompositeSpec):
            _specs = observation_spec._specs
            in_keys = self.in_keys
            out_keys = self.out_keys
            for in_key, out_key in zip(in_keys, out_keys):
                if in_key in observation_spec.keys(True, True):
                    _specs[out_key] = function(self, observation_spec[in_key].clone())
            return CompositeSpec(
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
    def new_fun(self, input_spec):
        action_spec = input_spec["full_action_spec"].clone()
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(shape=input_spec.shape, device=input_spec.device)
        else:
            state_spec = state_spec.clone()
        in_keys_inv = self.in_keys_inv
        out_keys_inv = self.out_keys_inv
        for in_key, out_key in zip(in_keys_inv, out_keys_inv):
            if in_key != out_key:
                # we only change the input spec if the key is the same
                continue
            if in_key in action_spec.keys(True, True):
                action_spec[out_key] = function(self, action_spec[in_key].clone())
            elif in_key in state_spec.keys(True, True):
                state_spec[out_key] = function(self, state_spec[in_key].clone())
        return CompositeSpec(
            full_state_spec=state_spec,
            full_action_spec=action_spec,
            shape=input_spec.shape,
            device=input_spec.device,
        )

    return new_fun


class Transform(nn.Module):
    """Environment transform parent class.

    In principle, a transform receives a tensordict as input and returns (
    the same or another) tensordict as output, where a series of values have
    been modified or created with a new key. When instantiating a new
    transform, the keys that are to be read from are passed to the
    constructor via the :obj:`keys` argument.

    Transforms are to be combined with their target environments with the
    TransformedEnv class, which takes as arguments an :obj:`EnvBase` instance
    and a transform. If multiple transforms are to be used, they can be
    concatenated using the :obj:`Compose` class.
    A transform can be stateless or stateful (e.g. CatTransform). Because of
    this, Transforms support the :obj:`reset` operation, which should reset the
    transform to its initial state (such that successive trajectories are kept
    independent).

    Notably, :obj:`Transform` subclasses take care of transforming the affected
    specs from an environment: when querying
    `transformed_env.observation_spec`, the resulting objects will describe
    the specs of the transformed_in tensors.

    """

    invertible = False

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

    def reset(self, tensordict):
        warnings.warn("Transform.reset public method will be derpecated in v0.4.0.")
        return self._reset(tensordict, tensordict_reset=tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets a transform if it is stateful."""
        return tensordict_reset

    def init(self, tensordict) -> None:
        pass

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform).

        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._apply_transform is not coded. If the transform is coded in "
            "transform._call, make sure that this method is called instead of"
            "transform.forward, which is reserved for usage inside nn.Modules"
            "or appended to a replay buffer."
        )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform.

        For any operation that relates exclusively to the parent env (e.g. FrameSkip),
        modify the _step method instead. :meth:`~._call` should only be overwritten
        if a modification of the input tensordict is needed.

        :meth:`~._call` will be called by :meth:`TransformedEnv.step` and
        :meth:`TransformedEnv.reset`.

        """
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict.get(in_key, default=None)
            if value is not None:
                observation = self._apply_transform(value)
                tensordict.set(
                    out_key,
                    observation,
                )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {tensordict}"
                )
        return tensordict

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            data = tensordict.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                tensordict.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """The parent method of a transform during the ``env.step`` execution.

        This method should be overwritten whenever the :meth:`~._step` needs to be
        adapted. Unlike :meth:`~._call`, it is assumed that :meth:`~._step`
        will execute some operation with the parent env or that it requires
        access to the content of the tensordict at time ``t`` and not only
        ``t+1`` (the ``"next"`` entry in the input tensordict).

        :meth:`~._step` will only be called by :meth:`TransformedEnv.step` and
        not by :meth:`TransformedEnv.reset`.

        Args:
            tensordict (TensorDictBase): data at time t
            next_tensordict (TensorDictBase): data at time t+1

        Returns: the data at t+1
        """
        next_tensordict = self._call(next_tensordict)
        return next_tensordict

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return state

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # # We create a shallow copy of the tensordict to avoid that changes are
        # # exposed to the user: we'd like that the input keys remain unchanged
        # # in the originating script if they're being transformed.
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            data = tensordict.get(in_key, None)
            if data is not None:
                item = self._inv_apply_transform(data)
                tensordict.set(out_key, item)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")

        return tensordict

    @dispatch(source="in_keys_inv", dest="out_keys_inv")
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = self._inv_call(tensordict.clone(False))
        return out

    def transform_env_device(self, device: torch.device):
        """Transforms the device of the parent env."""
        return device

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        """Transforms the output spec such that the resulting spec matches transform mapping.

        This method should generally be left untouched. Changes should be implemented using
        :meth:`~.transform_observation_spec`, :meth:`~.transform_reward_spec` and :meth:`~.transformfull_done_spec`.
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
        return output_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        """Transforms the input spec such that the resulting spec matches transform mapping.

        Args:
            input_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
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

    def dump(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.in_keys})"

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        if self.parent is not None:
            raise AttributeError(
                f"parent of transform {type(self)} already set. "
                "Call `transform.clone()` to get a similar transform with no parent set."
            )
        self.__dict__["_container"] = container
        self.__dict__["_parent"] = None

    def reset_parent(self) -> None:
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def clone(self):
        self_copy = copy(self)
        state = copy(self.__dict__)
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
        container = self.__dict__["_container"]
        if container is None:
            return container
        while not isinstance(container, EnvBase):
            # if it's not an env, it should be a Compose transform
            if not isinstance(container, Compose):
                raise ValueError(
                    "A transform parent must be either another Compose transform or an environment object."
                )
            compose = container
            container = compose.__dict__.get("_container", None)
        return container

    @property
    def parent(self) -> Optional[EnvBase]:
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
        if self.__dict__.get("_parent", None) is None:
            if "_container" not in self.__dict__:
                raise AttributeError("transform parent uninitialized")
            container = self.__dict__["_container"]
            if container is None:
                return container
            out = None
            if not isinstance(container, EnvBase):
                # if it's not an env, it should be a Compose transform
                if not isinstance(container, Compose):
                    raise ValueError(
                        "A transform parent must be either another Compose transform or an environment object."
                    )
                out, _ = container._rebuild_up_to(self)
            elif isinstance(container, TransformedEnv):
                out = TransformedEnv(container.base_env)
            else:
                raise ValueError(f"container is of type {type(container)}")
            self.__dict__["_parent"] = out
        return self.__dict__["_parent"]

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
        transform (Transform, optional): transform to apply to the tensordict resulting
            from :obj:`env.step(td)`. If none is provided, an empty Compose
            placeholder in an eval mode is used.
        cache_specs (bool, optional): if ``True``, the specs will be cached once
            and for all after the first call (i.e. the specs will be
            transformed_in only once). If the transform changes during
            training, the original spec transform may not be valid anymore,
            in which case this value should be set  to `False`. Default is
            `True`.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transform = RewardScaling(0.0, 1.0)
        >>> transformed_env = TransformedEnv(env, transform)

    """

    def __init__(
        self,
        env: EnvBase,
        transform: Optional[Transform] = None,
        cache_specs: bool = True,
        **kwargs,
    ):
        self._transform = None
        device = kwargs.pop("device", None)
        if device is not None:
            env = env.to(device)
        else:
            device = env.device
        super().__init__(device=None, **kwargs)

        if isinstance(env, TransformedEnv):
            self._set_env(env.base_env, device)
            if type(transform) is not Compose:
                # we don't use isinstance as some transforms may be subclassed from
                # Compose but with other features that we don't want to loose.
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
            return self.base_env.batch_size
        except AttributeError:
            # during init, the base_env is not yet defined
            return torch.Size([])

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        raise RuntimeError(
            "Cannot modify the batch-size of a transformed env. Change the batch size of the base_env instead."
        )

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
            raise ValueError(
                f"""Expected a transform of type torchrl.envs.transforms.Transform,
but got an object of type {type(transform)}."""
            )
        prev_transform = getattr(self, "_transform", None)
        if prev_transform is not None:
            prev_transform.empty_cache()
            prev_transform.reset_parent()
        if not isinstance(transform, Transform):
            raise ValueError(
                f"Transforms passed to {type(self)} must be instances of a `torch.envs.Transform` subclass. Got {type(transform)}."
            )
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
    def _inplace_update(self):
        return self.base_env._inplace_update

    @property
    def output_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment."""
        if not self.cache_specs or self.__dict__.get("_output_spec", None) is None:
            output_spec = self.base_env.output_spec.clone()

            # remove cached key values, but not _input_spec
            super().empty_cache()

            output_spec = output_spec.unlock_()
            output_spec = self.transform.transform_output_spec(output_spec)
            output_spec.lock_()
            if self.cache_specs:
                self.__dict__["_output_spec"] = output_spec
        else:
            output_spec = self.__dict__.get("_output_spec", None)
        return output_spec

    @property
    def input_spec(self) -> TensorSpec:
        """Action spec of the transformed environment."""
        if self.__dict__.get("_input_spec", None) is None or not self.cache_specs:
            input_spec = self.base_env.input_spec.clone()

            # remove cached key values but not _output_spec
            super().empty_cache()

            input_spec.unlock_()
            input_spec = self.transform.transform_input_spec(input_spec)
            input_spec.lock_()
            if self.cache_specs:
                self.__dict__["_input_spec"] = input_spec
        else:
            input_spec = self.__dict__.get("_input_spec", None)
        return input_spec

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        next_preset = tensordict.get("next", None)
        tensordict_in = self.transform.inv(tensordict)
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
        next_tensordict = self.transform._step(tensordict, next_tensordict)
        return next_tensordict

    def set_seed(
        self, seed: Optional[int] = None, static_seed: bool = False
    ) -> Optional[int]:
        """Set the seeds of the environment."""
        return self.base_env.set_seed(seed, static_seed=static_seed)

    def _set_seed(self, seed: Optional[int]):
        """This method is not used in transformed envs."""
        pass

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs):
        if tensordict is not None:
            # We must avoid modifying the original tensordict so a shallow copy is necessary.
            # We just select the input data and reset signal, which is all we need.
            tensordict = tensordict.select(
                *self.reset_keys, *self.state_spec.keys(True, True), strict=False
            )
        tensordict_reset = self.base_env._reset(tensordict=tensordict, **kwargs)
        if tensordict is None:
            # make sure all transforms see a source tensordict
            tensordict = tensordict_reset.empty()
        self.base_env._complete_done(self.base_env.full_done_spec, tensordict_reset)
        tensordict_reset = self.transform._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def _reset_proc_data(self, tensordict, tensordict_reset):
        # self._complete_done(self.full_done_spec, tensordict_reset)
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
        # tensordict_reset = self.transform._call(tensordict_reset)
        # self.set_missing_tolerance(mt_mode)
        return tensordict_reset

    def _complete_done(
        cls, done_spec: CompositeSpec, data: TensorDictBase
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

    def close(self):
        self.base_env.close()
        self.is_closed = True

    def empty_cache(self):
        self.__dict__["_output_spec"] = None
        self.__dict__["_input_spec"] = None
        super().empty_cache()

    def append_transform(self, transform: Transform) -> None:
        self.empty_cache()
        if not isinstance(transform, Transform):
            raise ValueError(
                "TransformedEnv.append_transform expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        transform = transform.to(self.device)
        if not isinstance(self.transform, Compose):
            prev_transform = self.transform
            prev_transform.reset_parent()
            self.transform = Compose()
            self.transform.append(prev_transform)

        self.transform.append(transform)

    def insert_transform(self, index: int, transform: Transform) -> None:
        self.empty_cache()
        if not isinstance(transform, Transform):
            raise ValueError(
                "TransformedEnv.insert_transform expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        transform = transform.to(self.device)
        if not isinstance(self.transform, Compose):
            compose = Compose(self.transform.clone())
            self.transform = compose  # parent set automatically

        self.transform.insert(index, transform)

    def __getattr__(self, attr: str) -> Any:
        try:
            return super().__getattr__(
                attr
            )  # make sure that appropriate exceptions are raised
        except Exception as err:
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
        super(ObservationTransform, self).__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )


class Compose(Transform):
    """Composes a chain of transforms.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transforms = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(*transforms)
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    def __init__(self, *transforms: Transform):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        for t in transforms:
            t.set_container(self)

    def to(self, *args, **kwargs):
        # because Module.to(...) does not call to(...) on sub-modules, we have
        # manually call it:
        self.transforms = nn.ModuleList(
            [t.to(*args, **kwargs) for t in self.transforms]
        )
        return super().to(*args, **kwargs)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t._call(tensordict)
        return tensordict

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

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms[::-1]:
            input_spec = t.transform_input_spec(input_spec)
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
        return observation_spec

    def transform_output_spec(self, output_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            output_spec = t.transform_output_spec(output_spec)
        return output_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
        return reward_spec

    def __getitem__(self, item: Union[int, slice, List]) -> Union:
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

    def init(self, tensordict: TensorDictBase) -> None:
        for t in self.transforms:
            t.init(tensordict)

    def append(self, transform):
        self.empty_cache()
        if not isinstance(transform, Transform):
            raise ValueError(
                "Compose.append expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        transform.eval()
        if type(self) == type(transform) == Compose:
            for t in transform:
                self.append(t)
        else:
            self.transforms.append(transform)
        transform.set_container(self)

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        self.reset_parent()
        super().set_container(container)
        for t in self.transforms:
            t.set_container(self)

    def insert(self, index: int, transform: Transform) -> None:
        if not isinstance(transform, Transform):
            raise ValueError(
                "Compose.append expected a transform but received an object of "
                f"type {type(transform)} instead."
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
        layers_str = ",\n".join(
            [indent(str(trsf), 4 * " ") for trsf in self.transforms]
        )
        return f"{self.__class__.__name__}(\n{indent(layers_str, 4 * ' ')})"

    def empty_cache(self):
        for t in self.transforms:
            t.empty_cache()
        super().empty_cache()

    def reset_parent(self):
        for t in self.transforms:
            t.reset_parent()
        super().reset_parent()

    def clone(self):
        transforms = []
        for t in self.transforms:
            transforms.append(t.clone())
        return Compose(*transforms)

    def set_missing_tolerance(self, mode=False):
        for t in self.transforms:
            t.set_missing_tolerance(mode)
        super().set_missing_tolerance(mode)

    def _rebuild_up_to(self, final_transform):
        container = self.__dict__["_container"]

        if isinstance(container, Compose):
            out, parent_compose = container._rebuild_up_to(self)
            if out is None:
                # returns None if there is no parent env
                return None, None
        elif isinstance(container, TransformedEnv):
            out = TransformedEnv(container.base_env)
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
            it's a floating-point tensor. default=None.
        unsqueeze (bool): if ``True``, the observation tensor is unsqueezed
            along the first dimension. default=False.
        dtype (torch.dtype, optional): dtype to use for the resulting
            observations.

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
        from_int: Optional[bool] = None,
        unsqueeze: bool = False,
        dtype: Optional[torch.device] = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.from_int = from_int
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
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
        unsqueeze_dim = [1] if self._should_unsqueeze(observation_spec) else []
        observation_spec.shape = torch.Size(
            [
                *unsqueeze_dim,
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
        in_keys_inv (list of NestedKeys): input entries (read) during :meth:`~.inv` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during :meth:`~.inv` calls.

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
                val = torch.tensor(val)
            if not val.dtype.is_floating_point:
                val = val.float()
            eps = torch.finfo(val.dtype).resolution
            ext = torch.finfo(val.dtype).max
            return val, eps, ext

        low, low_eps, low_min = check_val(low)
        high, high_eps, high_max = check_val(high)
        if low is not None and high is not None and low >= high:
            raise ValueError("`low` must be stricly lower than `high`.")
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
        return BoundedTensorSpec(
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
                self.parent.output_spec["full_reward_spec"][key] = BoundedTensorSpec(
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
    or the end of the episode. It is used as input for the policy to guide its behaviour.
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
        target_return (float): target return to be achieved by the agent.
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

    def _call(self, tensordict: TensorDict) -> TensorDict:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            val_in = tensordict.get(in_key, None)
            val_out = tensordict.get(out_key, None)
            if val_in is not None:
                target_return = self._apply_transform(
                    val_in,
                    val_out,
                )
                tensordict.set(out_key, target_return)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

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
            raise ValueError("Unknown mode: {}".format(self.mode))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in self.parent.full_observation_spec.keys(True):
                target = self.parent.full_observation_spec[in_key]
            elif in_key in self.parent.full_reward_spec.keys(True):
                target = self.parent.full_reward_spec[in_key]
            elif in_key in self.parent.full_done_spec.keys(True):
                # we account for this for completeness but it should never be the case
                target = self.parent.full_done_spec[in_key]
            else:
                raise RuntimeError(f"in_key {in_key} not found in output_spec.")
            target_return_spec = UnboundedContinuousTensorSpec(
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
        clamp_min: float = None,
        clamp_max: float = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        clamp_min_tensor = (
            clamp_min if isinstance(clamp_min, Tensor) else torch.tensor(clamp_min)
        )
        clamp_max_tensor = (
            clamp_max if isinstance(clamp_max, Tensor) else torch.tensor(clamp_max)
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
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return BoundedTensorSpec(
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
    """Maps the reward to a binary value (0 or 1) if the reward is null or non-null, respectively."""

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
        return (reward > 0.0).to(torch.long)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return BinaryDiscreteTensorSpec(
            n=1, device=reward_spec.device, shape=reward_spec.shape
        )


class Resize(ObservationTransform):
    """Resizes a pixel observation.

    Args:
        w (int): resulting width
        h (int): resulting height
        interpolation (str): interpolation method
    """

    def __init__(
        self,
        w: int,
        h: int,
        interpolation: str = "bilinear",
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
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
        self.interpolation = interpolation_fn(interpolation)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        # flatten if necessary
        if observation.shape[-2:] == torch.Size([self.w, self.h]):
            return observation
        ndim = observation.ndimension()
        if ndim > 4:
            sizes = observation.shape[:-3]
            observation = torch.flatten(observation, 0, ndim - 4)
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
        h: int = None,
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
                "first_dim should be smaller than 0 to accomodate for "
                "envs of different batch_sizes."
            )
        if not allow_positive_dim and last_dim >= 0:
            raise ValueError(
                "last_dim should be smaller than 0 to accomodate for "
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
        unsqueeze_dim (int): dimension to unsqueeze. Must be negative (or allow_positive_dim
            must be turned on).
        allow_positive_dim (bool, optional): if ``True``, positive dimensions are accepted.
            :obj:`UnsqueezeTransform` will map these to the n^th feature dimension
            (ie n^th dimension after batch size of parent env) of the input tensor,
            independently from the tensordict batch size (ie positive dims may be
            dangerous in contexts where tensordict of different batch dimension
            are passed).
            Defaults to False, ie. non-negative dimensions are not permitted.
    """

    invertible = True

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._unsqueeze_dim = None
        return super().__new__(cls)

    def __init__(
        self,
        unsqueeze_dim: int,
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
        if unsqueeze_dim >= 0 and not allow_positive_dim:
            raise RuntimeError(
                "unsqueeze_dim should be smaller than 0 to accomodate for "
                "envs of different batch_sizes. Turn allow_positive_dim to accomodate "
                "for positive unsqueeze_dim."
            )
        self._unsqueeze_dim = unsqueeze_dim

    @property
    def unsqueeze_dim(self):
        if self._unsqueeze_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._unsqueeze_dim
        return self._unsqueeze_dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.unsqueeze(self.unsqueeze_dim)
        return observation

    def _inv_apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.squeeze(self.unsqueeze_dim)
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

    def _inv_transform_spec(self, spec: TensorSpec) -> None:
        space = spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._inv_apply_transform(space.low)
            space.high = self._inv_apply_transform(space.high)
            spec.shape = space.low.shape
        else:
            spec.shape = self._inv_apply_transform(torch.zeros(spec.shape)).shape
        return spec

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        return self._inv_transform_spec(input_spec)

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
            f"{self.__class__.__name__}(unsqueeze_dim={self.unsqueeze_dim}, in_keys={self.in_keys}, out_keys={self.out_keys},"
            f" in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
        return s


class SqueezeTransform(UnsqueezeTransform):
    """Removes a dimension of size one at the specified position.

    Args:
        squeeze_dim (int): dimension to squeeze.
    """

    invertible = True

    def __init__(
        self,
        squeeze_dim: int,
        *args,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        in_keys_inv: Optional[Sequence[str]] = None,
        out_keys_inv: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            squeeze_dim,
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            **kwargs,
        )

    @property
    def squeeze_dim(self):
        return super().unsqueeze_dim

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
            All entries will be normalized with the same values: if a different behaviour is desired
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
        loc: Optional[float, torch.Tensor] = None,
        scale: Optional[float, torch.Tensor] = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            warnings.warn(
                "Not passing in_keys to ObservationNorm will soon be deprecated. "
                "Ensure you specify the entries to be normalized",
                category=DeprecationWarning,
            )
            in_keys = [
                "observation",
                "pixels",
            ]

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
            standard_normal = torch.tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)
        self.eps = 1e-6

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
        reduce_dim: Union[int, Tuple[int]] = 0,
        cat_dim: Optional[int] = None,
        key: Optional[NestedKey] = None,
        keep_dims: Optional[Tuple[int]] = None,
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

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        space = input_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return input_spec

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

    This can, for instance, account for movement/velocity of the observed
    feature. Proposed in "Playing Atari with Deep Reinforcement Learning" (
    https://arxiv.org/pdf/1312.5602.pdf).

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
        padding_value (float, optional): the value to use for padding if ``padding="constant"``.
            Defaults to 0.
        as_inverse (bool, optional): if ``True``, the transform is applied as an inverse transform. Defaults to ``False``.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

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
    purpose of limiting the memory consumption). The followin example
    gives the complete picture, together with the usage of a :class:`torchrl.data.ReplayBuffer`:

    Examples:
        >>> from torchrl.envs import UnsqueezeTransform, CatFrames
        >>> from torchrl.collectors import SyncDataCollector, RandomPolicy
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

    """

    inplace = False
    _CAT_DIM_ERR = (
        "dim must be < 0 to accomodate for tensordict of "
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
            warnings.warn(
                "Padding option 'zeros' will be deprecated in the future. "
                "Please use 'constant' padding with padding_value 0 instead.",
                category=DeprecationWarning,
            )
            padding = "constant"
            padding_value = 0
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

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
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

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        _just_reset = _reset is not None
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_cat_buffers_{in_key}"
            data = tensordict.get(in_key)
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

                # # this duplicates the code below, but only for _reset values
                # if _all:
                #     buffer.copy_(torch.roll(buffer_reset, shifts=-d, dims=dim))
                #     buffer_reset = buffer
                # else:
                #     buffer_reset = buffer[_reset] = torch.roll(
                #         buffer_reset, shifts=-d, dims=dim
                #     )
                # add new obs
                if self.dim < 0:
                    n = buffer_reset.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = [slice(None, None) for _ in range(n)] + [slice(-d, None)]
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
                idx = [slice(None, None) for _ in range(n)] + [slice(-d, None)]
                buffer[idx] = buffer[idx].copy_(data)
            # add to tensordict
            tensordict.set(out_key, buffer.clone())
        return tensordict

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
        in_keys = list(
            zip(
                (in_key, out_key)
                for in_key, out_key in zip(self.in_keys, self.out_keys)
                if isinstance(in_key, str) or len(in_key) == 1
            )
        )
        in_keys += list(
            zip(
                (in_key, out_key)
                for in_key, out_key in zip(self.in_keys, self.out_keys)
                if not isinstance(in_key, str) and not len(in_key) == 1
            )
        )
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # check if we have an obs in "next" that has already been processed.
            # If so, we must add an offset
            data = tensordict.get(in_key)
            if isinstance(in_key, tuple) and in_key[0] == "next":
                # let's get the out_key we have already processed
                prev_out_key = dict(zip(self.in_keys, self.out_keys))[in_key[1]]
                prev_val = tensordict.get(prev_out_key)
                # the first item is located along `dim+1` at the last index of the
                # first time index
                idx = (
                    [slice(None)] * (tensordict.ndim - 1)
                    + [0]
                    + [..., -1]
                    + [slice(None)] * (abs(self.dim) - 1)
                )
                first_val = prev_val[tuple(idx)].unsqueeze(tensordict.ndim - 1)
                data0 = [first_val] * (self.N - 1)
                if self.padding == "constant":
                    data0 = [
                        torch.full_like(elt, self.padding_value) for elt in data0[:-1]
                    ] + data0[-1:]
                elif self.padding == "same":
                    pass
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError
            elif self.padding == "same":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [data[tuple(idx)].unsqueeze(tensordict.ndim - 1)] * (self.N - 1)
            elif self.padding == "constant":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [
                    torch.full_like(data[tuple(idx)], self.padding_value).unsqueeze(
                        tensordict.ndim - 1
                    )
                ] * (self.N - 1)
            else:
                # make linter happy. An exception has already been raised
                raise NotImplementedError

            data = torch.cat(data0 + [data], tensordict.ndim - 1)

            data = data.unfold(tensordict.ndim - 1, self.N, 1)
            data = data.permute(
                *range(0, data.ndim + self.dim),
                -1,
                *range(data.ndim + self.dim, data.ndim - 1),
            )
            tensordict.set(out_key, data)
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
        loc: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor],
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
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.apply(check_finite)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    forward = _call


class DTypeCastTransform(Transform):
    """Casts one dtype to another for selected keys.

    Depending on whether the ``in_keys`` or ``in_keys_inv`` are provided
    during construction, the class behaviour will change:

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

    The same behaviour is the rule when environments are constructedw without
    specifying the transform keys:

    Examples:
        >>> class MyEnv(EnvBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = CompositeSpec(obs=UnboundedContinuousTensorSpec((), dtype=torch.float64))
        ...         self.action_spec = UnboundedContinuousTensorSpec((), dtype=torch.float64)
        ...         self.reward_spec = UnboundedContinuousTensorSpec((1,), dtype=torch.float64)
        ...         self.done_spec = UnboundedContinuousTensorSpec((1,), dtype=torch.bool)
        ...     def _reset(self, data=None):
        ...         return TensorDict({"done": torch.zeros((1,), dtype=torch.bool), **self.observation_spec.rand()}, [])
        ...     def _step(self, data):
        ...         assert data["action"].dtype == torch.float64
        ...         reward = self.reward_spec.rand()
        ...         done = torch.zeros((1,), dtype=torch.bool)
        ...         obs = self.observation_spec.rand()
        ...         assert reward.dtype == torch.float64
        ...         assert obs["obs"].dtype == torch.float64
        ...         return obs.select().set("next", obs.update({"reward": reward, "done": done}))
        ...     def _set_seed(self, seed):
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
            for in_key, item in list(tensordict.items(True, True)):
                if item.dtype == self.dtype_in:
                    item = self._apply_transform(item)
                    tensordict.set(in_key, item)
        else:
            # we made sure that if in_keys is not None, out_keys is not None either
            for in_key, out_key in zip(in_keys, out_keys):
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
        if isinstance(spec, CompositeSpec):
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
        for in_key_inv, out_key_inv in zip(self.in_keys_inv, self.out_keys_inv):
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

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        if self.in_keys is None:
            raise NotImplementedError(
                f"Calling transform_reward_spec without a parent environment isn't supported yet for {type(self)}."
            )
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        for reward_key, reward_spec in list(full_reward_spec.items(True, True)):
            # find out_key that match the in_key
            for in_key, out_key in zip(self.in_keys, self.out_keys):
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
            for in_key, out_key in zip(self.in_keys, self.out_keys):
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
    during construction, the class behaviour will change:

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

    The same behaviour is the rule when environments are constructedw without
    specifying the transform keys:

    Examples:
        >>> class MyEnv(EnvBase):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = CompositeSpec(obs=UnboundedContinuousTensorSpec((), dtype=torch.float64))
        ...         self.action_spec = UnboundedContinuousTensorSpec((), dtype=torch.float64)
        ...         self.reward_spec = UnboundedContinuousTensorSpec((1,), dtype=torch.float64)
        ...         self.done_spec = UnboundedContinuousTensorSpec((1,), dtype=torch.bool)
        ...     def _reset(self, data=None):
        ...         return TensorDict({"done": torch.zeros((1,), dtype=torch.bool), **self.observation_spec.rand()}, [])
        ...     def _step(self, data):
        ...         assert data["action"].dtype == torch.float64
        ...         reward = self.reward_spec.rand()
        ...         done = torch.zeros((1,), dtype=torch.bool)
        ...         obs = self.observation_spec.rand()
        ...         assert reward.dtype == torch.float64
        ...         assert obs["obs"].dtype == torch.float64
        ...         return obs.select().set("next", obs.update({"reward": reward, "done": done}))
        ...     def _set_seed(self, seed):
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
    ):
        self.device = torch.device(device)
        self.orig_device = (
            torch.device(orig_device) if orig_device is not None else orig_device
        )
        super().__init__()

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
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

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.to(self.device, non_blocking=True)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.to(self.device, non_blocking=True)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            if self.orig_device is None:
                return tensordict
            return tensordict.to(self.orig_device, non_blocking=True)
        return tensordict.to(parent.device, non_blocking=True)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        return input_spec.to(self.device)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return reward_spec.to(self.device)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec.to(self.device)

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        return output_spec.to(self.device)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        return done_spec.to(self.device)

    def transform_env_device(self, device):
        return self.device

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(device={self.device}, orig_device={self.orig_device})"
        return s


class CatTensors(Transform):
    """Concatenates several keys in a single tensor.

    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        in_keys (sequence of NestedKey): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behaviour will only work if a parent is set.
        out_key (NestedKey): key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is -1.
        del_keys (bool, optional): if ``True``, the input values will be deleted after
            concatenation. Default is True.
        unsqueeze_if_oor (bool, optional): if ``True``, CatTensor will check that
            the dimension indicated exist for the tensors to concatenate. If not,
            the tensors will be unsqueezed along that dimension.
            Default is ``False``.

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
        del_keys: bool = True,
        unsqueeze_if_oor: bool = False,
    ):
        self._initialized = in_keys is not None
        if not self._initialized:
            if dim != -1:
                raise ValueError(
                    "Lazy call to CatTensors is only supported when `dim=-1`."
                )
        else:
            in_keys = sorted(in_keys, key=_sort_keys)
        if not isinstance(out_key, (str, tuple)):
            raise Exception("CatTensors requires out_key to be of type NestedKey")
        super(CatTensors, self).__init__(in_keys=in_keys, out_keys=[out_key])
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        if all(key in tensordict.keys(include_nested=True) for key in self.in_keys):
            values = [tensordict.get(key) for key in self.in_keys]
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
            tensordict.set(self.out_keys[0], out_tensor)
            if self._del_keys:
                tensordict.exclude(*self.keys_to_exclude, inplace=True)
        else:
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(self.in_keys, key=_sort_keys)} but got a TensorDict with keys"
                f" {sorted(tensordict.keys(include_nested=True), key=_sort_keys)}"
            )
        return tensordict

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
        if len(self.in_keys) > 1 and not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, CompositeSpec) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, CompositeSpec):
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
        observation_spec[out_key] = UnboundedContinuousTensorSpec(
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
            by a replay buffer or an nn.Module chain. Defaults to True.

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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # We don't do anything here because the action is modified by the inv
        # method but we don't need to map it back as it won't be updated in the original
        # tensordict
        return tensordict

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

    def transform_input_spec(self, input_spec: CompositeSpec):
        input_spec = input_spec.clone()
        for key in input_spec["full_action_spec"].keys(True, True):
            key = ("full_action_spec", key)
            break
        else:
            raise KeyError("key not found in action_spec.")
        input_spec[key] = OneHotDiscreteTensorSpec(
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
    which improves stability on certain training algorithms.

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
        # check that there is a single done state -- behaviour is undefined for multiple dones
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
        primers (dict or CompositeSpec, optional): a dictionary containing
            key-spec pairs which will be used to populate the input tensordict.
            :class:`~torchrl.data.CompositeSpec` instances are supported too.
        random (bool, optional): if ``True``, the values will be drawn randomly from
            the TensorSpec domain (or a unit Gaussian if unbounded). Otherwise a fixed value will be assumed.
            Defaults to `False`.
        default_value (float, optional): if non-random filling is chosen, this
            value will be used to populate the tensors. Defaults to `0.0`.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.
        **kwargs: each keyword argument corresponds to a key in the tensordict.
            The corresponding value has to be a TensorSpec instance indicating
            what the value must be.

    When used in a TransfomedEnv, the spec shapes must match the envs shape if
    the parent env is batch-locked (:obj:`env.batch_locked=True`).
    If the env is not batch-locked (e.g. model-based envs), it is assumed that the batch is
    given by the input tensordict instead.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import SerialEnv
        >>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(base_env)
        >>> # the env is batch-locked, so the leading dims of the spec must match those of the env
        >>> env.append_transform(TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([2, 3])))
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

    """

    def __init__(
        self,
        primers: dict | CompositeSpec = None,
        random: bool = False,
        default_value: float = 0.0,
        reset_key: NestedKey | None = None,
        **kwargs,
    ):
        self.device = kwargs.pop("device", None)
        if primers is not None:
            if kwargs:
                raise RuntimeError(
                    "providing the primers as a dictionary is incompatible with extra keys provided "
                    "as kwargs."
                )
            kwargs = primers
        if not isinstance(kwargs, CompositeSpec):
            kwargs = CompositeSpec(kwargs)
        self.primers = kwargs
        self.random = random
        self.default_value = default_value
        self.reset_key = reset_key

        # sanity check
        for spec in self.primers.values():
            if not isinstance(spec, TensorSpec):
                raise ValueError(
                    "The values of the primers must be a subtype of the TensorSpec class. "
                    f"Got {type(spec)} instead."
                )
        super().__init__()

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

    @property
    def device(self):
        device = self._device
        if device is None and self.parent is not None:
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

    def transform_observation_spec(
        self, observation_spec: CompositeSpec
    ) -> CompositeSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                f"observation_spec was expected to be of type CompositeSpec. Got {type(observation_spec)} instead."
            )
        for key, spec in self.primers.items():
            if spec.shape[: len(observation_spec.shape)] != observation_spec.shape:
                raise RuntimeError(
                    f"The leading shape of the primer specs ({self.__class__}) should match the one of the parent env. "
                    f"Got observation_spec.shape={observation_spec.shape} but the '{key}' entry's shape is {spec.shape}."
                )
            try:
                device = observation_spec.device
            except RuntimeError:
                device = self.device
            observation_spec[key] = spec.to(device)
        return observation_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        input_spec["full_state_spec"] = self.transform_observation_spec(
            input_spec["full_state_spec"]
        )
        return input_spec

    @property
    def _batch_size(self):
        return self.parent.batch_size

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for key, spec in self.primers.items():
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
                value = torch.full_like(
                    spec.zero(),
                    self.default_value,
                )
            tensordict.set(key, value)
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for key in self.primers.keys():
            if key not in next_tensordict.keys(True):
                prev_val = tensordict.get(key)
                next_tensordict.set(key, prev_val)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Sets the default values in the input tensordict.

        If the parent is batch-locked, we assume that the specs have the appropriate leading
        shape. We allow for execution when the parent is missing, in which case the
        spec shape is assumed to match the tensordict's.

        """
        shape = (
            ()
            if (not self.parent or self.parent.batch_locked)
            else tensordict.batch_size
        )
        _reset = _get_reset(self.reset_key, tensordict)
        if _reset.any():
            for key, spec in self.primers.items():
                if self.random:
                    value = spec.rand(shape)
                else:
                    value = torch.full_like(
                        spec.zero(shape),
                        self.default_value,
                    )
                prev_val = tensordict.get(key, 0.0)
                value = torch.where(expand_as_right(_reset, value), value, prev_val)
                tensordict_reset.set(key, value)
        return tensordict_reset

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(primers={self.primers}, default_value={self.default_value}, random={self.random})"


class PinMemoryTransform(Transform):
    """Calls pin_memory on the tensordict to facilitate writing on CUDA devices."""

    def __init__(self):
        super().__init__()

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.pin_memory()

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
        shape = tuple(shape) + tail_dim
        primers = {"_eps_gSDE": UnboundedContinuousTensorSpec(shape=shape)}
        super().__init__(primers=primers, random=random, **kwargs)


class VecNorm(Transform):
    """Moving average normalization layer for torchrl environments.

    VecNorm keeps track of the summary statistics of a dataset to standardize
    it on-the-fly. If the transform is in 'eval' mode, the running
    statistics are not updated.

    If multiple processes are running a similar environment, one can pass a
    TensorDictBase instance that is placed in shared memory: if so, every time
    the normalization layer is queried it will update the values for all
    processes that share the same reference.

    To use VecNorm at inference time and avoid updating the values with the new
    observations, one should substitute this layer by `vecnorm.to_observation_norm()`.

    Args:
        in_keys (sequence of NestedKey, optional): keys to be updated.
            default: ["observation", "reward"]
        out_keys (sequence of NestedKey, optional): destination keys.
            Defaults to ``in_keys``.
        shared_td (TensorDictBase, optional): A shared tensordict containing the
            keys of the transform.
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
        shared_td: Optional[TensorDictBase] = None,
        lock: mp.Lock = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
        shapes: List[torch.Size] = None,
    ) -> None:
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
                    (key + "_sum" not in shared_td.keys())
                    or (key + "_ssq" not in shared_td.keys())
                    or (key + "_count" not in shared_td.keys())
                ):
                    raise KeyError(
                        f"key {key} not present in the shared tensordict "
                        f"with keys {shared_td.keys()}"
                    )

        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps

    def _key_str(self, key):
        if not isinstance(key, str):
            key = "_".join(key)
        return key

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key in self.in_keys:
            if key not in tensordict.keys(include_nested=True):
                continue
            self._init(tensordict, key)
            # update and standardize
            new_val = self._update(
                key, tensordict.get(key), N=max(1, tensordict.numel())
            )

            tensordict.set(key, new_val)

        if self.lock is not None:
            self.lock.release()

        return tensordict

    forward = _call

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        key_str = self._key_str(key)
        if self._td is None or key_str + "_sum" not in self._td.keys():
            if key is not key_str and key_str in tensordict.keys():
                raise RuntimeError(
                    f"Conflicting key names: {key_str} from VecNorm and input tensordict keys."
                )
            if self.shapes is None:
                td_view = tensordict.view(-1)
                td_select = td_view[0]
                item = td_select.get(key)
                d = {key_str + "_sum": torch.zeros_like(item)}
                d.update({key_str + "_ssq": torch.zeros_like(item)})
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
                    key_str
                    + "_sum": torch.zeros(shape, device=item.device, dtype=item.dtype)
                }
                d.update(
                    {
                        key_str
                        + "_ssq": torch.zeros(
                            shape, device=item.device, dtype=item.dtype
                        )
                    }
                )

            d.update(
                {
                    key_str
                    + "_count": torch.zeros(1, device=item.device, dtype=torch.float)
                }
            )
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        key = self._key_str(key)
        _sum = self._td.get(key + "_sum")
        _ssq = self._td.get(key + "_ssq")
        _count = self._td.get(key + "_count")

        _sum = self._td.get(key + "_sum")
        value_sum = _sum_left(value, _sum)
        _sum *= self.decay
        _sum += value_sum
        self._td.set_(
            key + "_sum",
            _sum,
        )

        _ssq = self._td.get(key + "_ssq")
        value_ssq = _sum_left(value.pow(2), _ssq)
        _ssq *= self.decay
        _ssq += value_ssq
        self._td.set_(
            key + "_ssq",
            _ssq,
        )

        _count = self._td.get(key + "_count")
        _count *= self.decay
        _count += N
        self._td.set_(
            key + "_count",
            _count,
        )

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)

    def to_observation_norm(self) -> Union[Compose, ObservationNorm]:
        """Converts VecNorm into an ObservationNorm class that can be used at inference time."""
        out = []
        for key in self.in_keys:
            _sum = self._td.get(key + "_sum")
            _ssq = self._td.get(key + "_ssq")
            _count = self._td.get(key + "_count")
            mean = _sum / _count
            std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()

            _out = ObservationNorm(
                loc=mean,
                scale=std,
                standard_normal=True,
                in_keys=self.in_keys,
            )
            if len(self.in_keys) == 1:
                return _out
            else:
                out += ObservationNorm
        return Compose(*out)

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: EnvBase,
        keys: Optional[Sequence[str]] = None,
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
            td_select.set(key + "_ssq", td_select.get(key).clone())
            td_select.set(
                key + "_count",
                torch.zeros(
                    *td.batch_size,
                    1,
                    device=td_select.device,
                    dtype=torch.float,
                ),
            )
            td_select.rename_key_(key, key + "_sum")
        td_select.exclude(*keys).zero_()
        td_select = td_select.unflatten_keys(sep)
        if memmap:
            return td_select.memmap_()
        return td_select.share_memory_()

    def get_extra_state(self) -> OrderedDict:
        return collections.OrderedDict({"lock": self.lock, "td": self._td})

    def set_extra_state(self, state: OrderedDict) -> None:
        lock = state["lock"]
        if lock is not None:
            """
            since locks can't be serialized, we have use cases for stripping them
            for example in ParallelEnv, in which case keep the lock we already have
            to avoid an updated tensor dict being sent between processes to erase locks
            """
            self.lock = lock
        td = state["td"]
        if td is not None and not td.is_shared():
            raise RuntimeError(
                "Only shared tensordicts can be set in VecNorm transforms"
            )
        self._td = td

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(decay={self.decay:4.4f},"
            f"eps={self.eps:4.4f}, keys={self.in_keys})"
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _lock = state.pop("lock", None)
        if _lock is not None:
            state["lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "lock_placeholder" in state:
            state.pop("lock_placeholder")
            _lock = mp.Lock()
            state["lock"] = _lock
        self.__dict__.update(state)


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
    ):
        """Initialises the transform. Filters out non-reward input keys and defines output keys."""
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys

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

            def _check_match(reset_keys, in_keys):
                # if this is called, the length of reset_keys and in_keys must match
                for reset_key, in_key in zip(reset_keys, in_keys):
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

            if len(reset_keys) != len(self.in_keys) or not _check_match(
                reset_keys, self.in_keys
            ):
                raise ValueError(
                    f"Could not match the env reset_keys {reset_keys} with the {type(self)} in_keys {self.in_keys}. "
                    f"Please provide the reset_keys manually. Reset entries can be "
                    f"non-unique and must be right-expandable to the shape of "
                    f"the input entries."
                )
            reset_keys = copy(reset_keys)
            self._reset_keys = reset_keys
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
        for in_key, reset_key, out_key in zip(
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
        for in_key, out_key in zip(self.in_keys, self.out_keys):
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
            state_spec = CompositeSpec(shape=input_spec.shape, device=input_spec.device)
        state_spec.update(self._generate_episode_reward_spec())
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _generate_episode_reward_spec(self) -> CompositeSpec:
        episode_reward_spec = CompositeSpec()
        reward_spec = self.parent.full_reward_spec
        reward_spec_keys = self.parent.reward_keys
        # Define episode specs for all out_keys
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if (
                in_key in reward_spec_keys
            ):  # if this out_key has a corresponding key in reward_spec
                out_key = _unravel_key_to_tuple(out_key)
                temp_episode_reward_spec = episode_reward_spec
                temp_rew_spec = reward_spec
                for sub_key in out_key[:-1]:
                    if (
                        not isinstance(temp_rew_spec, CompositeSpec)
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
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(self._generate_episode_reward_spec())
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        time_dim = [i for i, name in enumerate(tensordict.names) if name == "time"]
        if not time_dim:
            raise ValueError(
                "At least one dimension of the tensordict must be named 'time' in offline mode"
            )
        time_dim = time_dim[0] - 1
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            reward = tensordict.get(in_key)
            cumsum = reward.cumsum(time_dim)
            tensordict.set(out_key, cumsum)
        return tensordict


class StepCounter(Transform):
    """Counts the steps from a reset and optionally sets the truncated state to ``True`` after a certain number of steps.

    The ``"done"`` state is also adaptec accordingly (as done is the intersection
    of task completetion and early truncation).

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
        max_steps: Optional[int] = None,
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
    def completed_keys(self):
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
        for step_count_key, truncated_key, terminated_key, reset_key, done_key in zip(
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
                    done = self.parent.output_spec["full_done_spec", entry_name].zero()
                reset = torch.ones_like(done)

            step_count = tensordict.get(step_count_key, default=None)
            if step_count is None:
                step_count = self.container.observation_spec[step_count_key].zero()

            # zero the step count if reset is needed
            step_count = torch.where(~expand_as_right(reset, step_count), step_count, 0)
            tensordict_reset.set(step_count_key, step_count)
            if self.max_steps is not None:
                truncated = step_count >= self.max_steps
                if self.update_done:
                    # we assume no done after reset
                    tensordict_reset.set(done_key, truncated)
                tensordict_reset.set(truncated_key, truncated)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for step_count_key, truncated_key, done_key, terminated_key in zip(
            self.step_count_keys,
            self.truncated_keys,
            self.done_keys,
            self.terminated_keys,
        ):
            step_count = tensordict.get(step_count_key)
            next_step_count = step_count + 1
            next_tensordict.set(step_count_key, next_step_count)
            if self.max_steps is not None:
                truncated = next_step_count >= self.max_steps
                if self.update_done:
                    done = next_tensordict.get(done_key, None)
                    terminated = next_tensordict.get(terminated_key, None)
                    if terminated is not None:
                        truncated = truncated & ~terminated
                    done = truncated | done  # we assume no done after reset
                    next_tensordict.set(done_key, done)
                next_tensordict.set(truncated_key, truncated)
        return next_tensordict

    def transform_observation_spec(
        self, observation_spec: CompositeSpec
    ) -> CompositeSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                f"observation_spec was expected to be of type CompositeSpec. Got {type(observation_spec)} instead."
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
            observation_spec[step_count_key] = BoundedTensorSpec(
                shape=shape,
                dtype=torch.int64,
                device=observation_spec.device,
                low=0,
                high=torch.iinfo(torch.int64).max,
            )
        return super().transform_observation_spec(observation_spec)

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
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
                full_done_spec[truncated_key] = DiscreteTensorSpec(
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
                    full_done_spec[done_key] = DiscreteTensorSpec(
                        2, dtype=torch.bool, device=output_spec.device, shape=shape
                    )
            output_spec["full_done_spec"] = full_done_spec
        return super().transform_output_spec(output_spec)

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        if not isinstance(input_spec, CompositeSpec):
            raise ValueError(
                f"input_spec was expected to be of type CompositeSpec. Got {type(input_spec)} instead."
            )
        if input_spec["full_state_spec"] is None:
            input_spec["full_state_spec"] = CompositeSpec(
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

            input_spec[
                unravel_key(("full_state_spec", step_count_key))
            ] = BoundedTensorSpec(
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

    def __init__(self, *excluded_keys):
        super().__init__()
        try:
            excluded_keys = unravel_key_list(excluded_keys)
        except TypeError:
            raise TypeError(
                "excluded keys must be a list or tuple of strings or tuples of strings."
            )
        self.excluded_keys = excluded_keys

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.exclude(*self.excluded_keys)

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset.exclude(*self.excluded_keys)

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
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

    def __init__(self, *selected_keys, keep_rewards=True, keep_dones=True):
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.input_spec.keys(True, True)
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
        return tensordict.select(
            *self.selected_keys, *reward_keys, *done_keys, *input_keys, strict=False
        )

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.input_spec.keys(True, True)
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

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
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

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(tensordict, in_key, buffer_name)
            if _reset is not None:
                # we must use only the reset data
                buffer[:, _reset] = torch.roll(buffer[:, _reset], shifts=1, dims=0)
                # add new obs
                data = tensordict.get(in_key)
                buffer[0, _reset] = data[_reset]
                # apply max pooling
                pooled_tensor, _ = buffer[:, _reset].max(dim=0)
                pooled_tensor = torch.zeros_like(data).masked_scatter_(
                    expand_as_right(_reset, data), pooled_tensor
                )
                # add to tensordict
                tensordict.set(out_key, pooled_tensor)
                continue
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(tensordict.get(in_key))
            # apply max pooling
            pooled_tensor, _ = buffer.max(dim=0)
            # add to tensordict
            tensordict.set(out_key, pooled_tensor)

        return tensordict

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
    Do not hesitate to request for this behaviour through an issue if this is
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
        mask_key: Optional[NestedKey] = None,
    ):
        self.sub_seq_len = sub_seq_len
        if sample_dim > 0:
            warnings.warn(
                "A positive shape has been passed to the RandomCropTensorDict "
                "constructor. This may have unexpected behaviours when the "
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

    def __init__(self, init_key: NestedKey = "is_init"):
        if not isinstance(init_key, str):
            raise ValueError("init_key can only be of type str.")
        self.init_key = init_key
        self.reset_key = "_reset"
        super().__init__()

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for init_key in self.init_keys:
            done_key = _replace_last(init_key, "done")
            if init_key not in tensordict.keys(True, True):
                device = tensordict.device
                if device is None:
                    device = torch.device("cpu")
                shape = self.parent.full_done_spec[done_key].shape
                tensordict.set(
                    init_key,
                    torch.zeros(shape, device=device, dtype=torch.bool),
                )
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        for reset_key, init_key in zip(self.reset_keys, self.init_keys):
            _reset = tensordict.get(reset_key, None)
            if _reset is None:
                done_key = _replace_last(init_key, "done")
                shape = self.parent.full_done_spec[done_key].shape
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
            observation_spec[init_key] = DiscreteTensorSpec(
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
    """A transform to rename entries in the output tensordict.

    Args:
        in_keys (sequence of NestedKey): the entries to rename
        out_keys (sequence of NestedKey): the name of the entries after renaming.
        in_keys_inv (sequence of NestedKey, optional): the entries to rename before
            passing the input tensordict to :meth:`EnvBase._step`.
        out_keys_inv (sequence of NestedKey, optional): the names of the renamed
            entries passed to :meth:`EnvBase._step`.
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.create_copy:
            out = tensordict.select(*self.in_keys, strict=not self._missing_tolerance)
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                try:
                    tensordict.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                try:
                    tensordict.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return tensordict

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
            for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
                try:
                    out.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise

            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
                try:
                    tensordict.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return tensordict

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
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

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        for action_key in self.parent.action_keys:
            if action_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == action_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_action_spec"][out_key] = input_spec[
                    "full_action_spec"
                ][action_key].clone()
                if not self.create_copy:
                    del input_spec["full_action_spec"][action_key]
        for state_key in self.parent.full_state_spec.keys(True):
            if state_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == state_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_state_spec"][out_key] = input_spec["full_state_spec"][
                    state_key
                ].clone()
                if not self.create_copy:
                    del input_spec["full_state_spec"][state_key]
        return input_spec


class Reward2GoTransform(Transform):
    """Calculates the reward to go based on the episode reward and a discount factor.

    As the :class:`~.Reward2GoTransform` is only an inverse transform the ``in_keys`` will be directly used for the ``in_keys_inv``.
    The reward-to-go can be only calculated once the episode is finished. Therefore, the transform should be applied to the replay buffer
    and not to the collector.

    Args:
        gamma (float or torch.Tensor): the discount factor. Defaults to 1.0.
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
        >>> from torchrl.collectors import SyncDataCollector, RandomPolicy
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
        "only be applied to the replay buffer and not to the collector or the environment."
    )

    def __init__(
        self,
        gamma: Optional[Union[float, torch.Tensor]] = 1.0,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        done_key: Optional[NestedKey] = "done",
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
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                found = True
                item = self._inv_apply_transform(tensordict.get(in_key), done)
                tensordict.set(out_key, item)
        if not found:
            raise KeyError(f"Could not find any of the input keys {self.in_keys}.")
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def _inv_apply_transform(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        return reward2go(reward, done, self.gamma)

    def set_container(self, container):
        if isinstance(container, EnvBase) or container.parent is not None:
            raise ValueError(self.ENV_ERR)


class ActionMask(Transform):
    """An adaptive action masker.

    This transform reads the mask from the input tensordict after the step is executed,
    and adapts the mask of the one-hot / categorical action spec.

      .. note:: This transform will fail when used without an environment.

    Args:
        action_key (NestedKey, optional): the key where the action tensor can be found.
            Defaults to ``"action"``.
        mask_key (NestedKey, optional): the key where the action mask can be found.
            Defaults to ``"action_mask"``.

    Examples:
        >>> import torch
        >>> from torchrl.data.tensor_specs import DiscreteTensorSpec, BinaryDiscreteTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
        >>> from torchrl.envs.transforms import ActionMask, TransformedEnv
        >>> from torchrl.envs.common import EnvBase
        >>> class MaskedEnv(EnvBase):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.action_spec = DiscreteTensorSpec(4)
        ...         self.state_spec = CompositeSpec(action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool))
        ...         self.observation_spec = CompositeSpec(obs=UnboundedContinuousTensorSpec(3))
        ...         self.reward_spec = UnboundedContinuousTensorSpec(1)
        ...
        ...     def _reset(self, data):
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
        ...     def _set_seed(self, seed):
        ...         return seed
        ...
        >>> torch.manual_seed(0)
        >>> base_env = MaskedEnv()
        >>> env = TransformedEnv(base_env, ActionMask())
        >>> r = env.rollout(10)
        >>> env = TransformedEnv(base_env, ActionMask())
        >>> r = env.rollout(10)
        >>> r["action_mask"]
        tensor([[ True,  True,  True,  True],
                [ True,  True, False,  True],
                [ True,  True, False, False],
                [ True, False, False, False]])

    """

    ACCEPTED_SPECS = (
        OneHotDiscreteTensorSpec,
        DiscreteTensorSpec,
        MultiOneHotDiscreteTensorSpec,
        MultiDiscreteTensorSpec,
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

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                f"{type(self)}.parent cannot be None: make sure this transform is executed within an environment."
            )
        mask = tensordict.get(self.in_keys[1])
        action_spec = self.container.action_spec
        if not isinstance(action_spec, self.ACCEPTED_SPECS):
            raise ValueError(
                self.SPEC_TYPE_ERROR.format(self.ACCEPTED_SPECS, type(action_spec))
            )
        action_spec.update_mask(mask)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        action_spec = self.container.action_spec
        if not isinstance(action_spec, self.ACCEPTED_SPECS):
            raise ValueError(
                self.SPEC_TYPE_ERROR.format(self.ACCEPTED_SPECS, type(action_spec))
            )
        action_spec.update_mask(tensordict.get(self.in_keys[1], None))

        # TODO: Check that this makes sense
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


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

    .. note:: In general, this class should not be handled directly. It is
        created whenever a vectorized environment is placed within a :class:`GymWrapper`.

    """

    def __init__(self, final_name="final"):
        self.final_name = final_name
        super().__init__()
        self._memo = {}

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
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
                    next_tensordict[obs_key][done] = torch.tensor(np.nan)

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
            and (not reset.all() or not reset.any())
        ):
            raise RuntimeError(
                "Cannot partially reset a gym(nasium) async env with a reset mask that does not match the done mask. "
                f"Got reset={reset}\nand done={done}"
            )
        # if not reset.any(), we don't need to do anything.
        # if reset.all(), we don't either (bc GymWrapper will call a plain reset).
        if reset is not None and reset.any() and not reset.all():
            saved_next = self._memo["saved_next"]
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
    def done_keys(self) -> List[NestedKey]:
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
    def obs_keys(self) -> List[NestedKey]:
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
