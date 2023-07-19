# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import multiprocessing as mp
import warnings
from copy import copy, deepcopy
from textwrap import indent
from typing import Any, List, Optional, OrderedDict, Sequence, Tuple, Union

import torch

from tensordict import unravel_key, unravel_key_list
from tensordict.nn import dispatch
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, NestedKey
from torch import nn, Tensor

from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    ContinuousBox,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs.common import EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import check_finite
from torchrl.envs.utils import _sort_keys, step_mdp
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

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent" "environment."


def _apply_to_composite(function):
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, CompositeSpec):
            d = observation_spec._specs
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                if in_key in observation_spec.keys(True, True):
                    d[out_key] = function(self, observation_spec[in_key].clone())
            return CompositeSpec(
                d, shape=observation_spec.shape, device=observation_spec.device
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
        action_spec = input_spec["_action_spec"].clone()
        state_spec = input_spec["_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(shape=input_spec.shape, device=input_spec.device)
        else:
            state_spec = state_spec.clone()
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key != out_key:
                # we only change the input spec if the key is the same
                continue
            if in_key in action_spec.keys(True, True):
                action_spec[out_key] = function(self, action_spec[in_key].clone())
            elif in_key in state_spec.keys(True, True):
                state_spec[out_key] = function(self, state_spec[in_key].clone())
        return CompositeSpec(
            _state_spec=state_spec,
            _action_spec=action_spec,
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
        in_keys: Sequence[NestedKey],
        out_keys: Optional[Sequence[NestedKey]] = None,
        in_keys_inv: Optional[Sequence[NestedKey]] = None,
        out_keys_inv: Optional[Sequence[NestedKey]] = None,
    ):
        super().__init__()
        if isinstance(in_keys, str):
            in_keys = [in_keys]

        self.in_keys = in_keys
        if out_keys is None:
            out_keys = copy(self.in_keys)
        self.out_keys = out_keys
        if in_keys_inv is None:
            in_keys_inv = []
        self.in_keys_inv = in_keys_inv
        if out_keys_inv is None:
            out_keys_inv = copy(self.in_keys_inv)
        self.out_keys_inv = out_keys_inv
        self._missing_tolerance = False
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets a tranform if it is stateful."""
        return tensordict

    def init(self, tensordict) -> None:
        pass

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform).

        """
        raise NotImplementedError(
            f"{self.__class__.__name__}_apply_transform is not coded. If the transform is coded in "
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
            if in_key in tensordict.keys(include_nested=True):
                observation = self._apply_transform(tensordict.get(in_key))
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
            if in_key in tensordict.keys(include_nested=True):
                observation = self._apply_transform(tensordict.get(in_key))
                tensordict.set(
                    out_key,
                    observation,
                )
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """The parent method of a transform during the ``env.step`` execution.

        This method should be overwritten whenever the :meth:`~._step` needs to be
        adapted. Unlike :meth:`~._call`, it is assumed that :meth:`~._step`
        will execute some operation with the parent env or that it requires
        access to the content of the tensordict at time ``t`` and not only
        ``t+1`` (the ``"next"`` entry in the input tensordict).

        :meth:`~._step` will only be called by :meth:`TransformedEnv.step` and
        not by :meth:`TransformedEnv.reset`.

        """
        next_tensordict = tensordict.get("next")
        next_tensordict = self._call(next_tensordict)
        tensordict.set("next", next_tensordict)
        return tensordict

    def _inv_apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return obs

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # # We create a shallow copy of the tensordict to avoid that changes are
        # # exposed to the user: we'd like that the input keys remain unchanged
        # # in the originating script if they're being transformed.
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                item = self._inv_apply_transform(tensordict.get(in_key))
                tensordict.set(
                    out_key,
                    item,
                )
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")

        return tensordict

    @dispatch(source="in_keys_inv", dest="out_keys_inv")
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = self._inv_call(tensordict.clone(False))
        return out

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        """Transforms the output spec such that the resulting spec matches transform mapping.

        This method should generally be left untouched. Changes should be implemented using
        :meth:`~.transform_observation_spec`, :meth:`~.transform_reward_spec` and :meth:`~.transform_done_spec`.
        Args:
            output_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        output_spec = output_spec.clone()
        output_spec["_observation_spec"] = self.transform_observation_spec(
            output_spec["_observation_spec"]
        )
        if "_reward_spec" in output_spec.keys():
            output_spec["_reward_spec"] = self.transform_reward_spec(
                output_spec["_reward_spec"]
            )
        if "_done_spec" in output_spec.keys():
            output_spec["_done_spec"] = self.transform_done_spec(
                output_spec["_done_spec"]
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
    def parent(self) -> Optional[EnvBase]:
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
                compose = container
                if compose.__dict__["_container"]:
                    # the parent of the compose must be a TransformedEnv
                    compose_parent = TransformedEnv(
                        compose.__dict__["_container"].base_env
                    )
                    if compose_parent.transform is not compose:
                        comp_parent_trans = compose_parent.transform.clone()
                    else:
                        comp_parent_trans = None
                    out = TransformedEnv(
                        compose_parent.base_env,
                        transform=comp_parent_trans,
                    )
                    for orig_trans in compose.transforms:
                        if orig_trans is self:
                            break
                        transform = orig_trans.clone()
                        transform.reset_parent()
                        out.append_transform(transform)
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
        self.empty_cache()
        return super().to(*args, **kwargs)


class TransformedEnv(EnvBase):
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
            else:
                transform = transform.to(device)
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
        return self._transform

    @transform.setter
    def transform(self, transform: Transform):
        if not isinstance(transform, Transform):
            raise ValueError(
                f"""Expected a transform of type torchrl.envs.transforms.Transform,
but got an object of type {type(transform)}."""
            )
        prev_transform = self.transform
        if prev_transform is not None:
            prev_transform.empty_cache()
            prev_transform.__dict__["_container"] = None
        transform.set_container(self)
        transform.eval()
        self._transform = transform

    @property
    def device(self) -> bool:
        return self.base_env.device

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
        if self.__dict__.get("_output_spec", None) is None or not self.cache_specs:
            output_spec = self.base_env.output_spec.clone()
            output_spec.unlock_()
            output_spec = self.transform.transform_output_spec(output_spec)
            output_spec.lock_()
            if self.cache_specs:
                self.__dict__["_output_spec"] = output_spec
        else:
            output_spec = self.__dict__.get("_output_spec", None)
        return output_spec

    @property
    def action_spec(self) -> TensorSpec:
        """Action spec of the transformed environment."""
        return self.input_spec[("_action_spec", *self.action_key)]

    @property
    def input_spec(self) -> TensorSpec:
        """Action spec of the transformed environment."""
        if self.__dict__.get("_input_spec", None) is None or not self.cache_specs:
            input_spec = self.base_env.input_spec.clone()
            input_spec.unlock_()
            input_spec = self.transform.transform_input_spec(input_spec)
            input_spec.lock_()
            if self.cache_specs:
                self.__dict__["_input_spec"] = input_spec
        else:
            input_spec = self.__dict__.get("_input_spec", None)
        return input_spec

    @property
    def reward_spec(self) -> TensorSpec:
        """Reward spec of the transformed environment."""
        return self.output_spec[("_reward_spec", *self.reward_key)]

    @property
    def observation_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment."""
        observation_spec = self.output_spec["_observation_spec"]
        if observation_spec is None:
            observation_spec = CompositeSpec(device=self.device, shape=self.batch_size)
        return observation_spec

    @property
    def state_spec(self) -> TensorSpec:
        """State spec of the transformed environment."""
        state_spec = self.input_spec["_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(device=self.device, shape=self.batch_size)
        return state_spec

    @property
    def done_spec(self) -> TensorSpec:
        """Done spec of the transformed environment."""
        return self.output_spec[("_done_spec", *self.done_key)]

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        tensordict_in = self.transform.inv(tensordict)
        tensordict_out = self.base_env._step(tensordict_in)
        # we want the input entries to remain unchanged
        tensordict_out = tensordict.update(tensordict_out)
        tensordict_out = self.transform._step(tensordict_out)
        return tensordict_out

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
            tensordict = tensordict.clone(recurse=False)
        out_tensordict = self.base_env.reset(tensordict=tensordict, **kwargs)
        out_tensordict = self.transform.reset(out_tensordict)

        mt_mode = self.transform.missing_tolerance
        self.set_missing_tolerance(True)
        out_tensordict = self.transform._call(out_tensordict)
        self.set_missing_tolerance(mt_mode)
        return out_tensordict

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
        self.__dict__["_cache_in_keys"] = None

    def append_transform(self, transform: Transform) -> None:
        self._erase_metadata()
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
        self._erase_metadata()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return super().__getattr__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
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
        )

    def __repr__(self) -> str:
        env_str = indent(f"env={self.base_env}", 4 * " ")
        t_str = indent(f"transform={self.transform}", 4 * " ")
        return f"TransformedEnv(\n{env_str},\n{t_str})"

    def _erase_metadata(self):
        if self.cache_specs:
            self.__dict__["_input_spec"] = None
            self.__dict__["_output_spec"] = None
            self.__dict__["_cache_in_keys"] = None

    def to(self, device: DEVICE_TYPING) -> TransformedEnv:
        self.base_env.to(device)
        self.transform = self.transform.to(device)

        if self.cache_specs:
            self.__dict__["_input_spec"] = None
            self.__dict__["_output_spec"] = None
        return self

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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        in_keys_inv: Optional[Sequence[NestedKey]] = None,
        out_keys_inv: Optional[Sequence[NestedKey]] = None,
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
        super().__init__(in_keys=[])
        self.transforms = nn.ModuleList(transforms)
        for t in transforms:
            t.set_container(self)

    def to(self, *args, **kwargs):
        # because Module.to(...) does not call to(...) on sub-modules, we have
        # manually call it:
        for t in self.transforms:
            t.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t._call(tensordict)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t._step(tensordict)
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in reversed(self.transforms):
            tensordict = t._inv_call(tensordict)
        return tensordict

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

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t.reset(tensordict)
        return tensordict

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
        self.transforms.append(transform)
        transform.set_container(self)

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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.from_int = from_int
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

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
            spec.space.maximum = self._apply_transform(spec.space.maximum)
            spec.space.minimum = self._apply_transform(spec.space.minimum)
        return spec


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
    to include the
    user-specified target return. The mode parameter can be used to specify
    whether the target return gets updated at every step by subtracting the
    reward achieved at each step or remains constant.
    :class:`~.TargetReturn` should be only used during inference when
    interacting with the environment as the actual
    return received by the environment might be different from the target
    return. Therefore, to have the correct
    return labels for training the policy, the :class:`~.TargetReturn`
    transform should be used in conjunction with
    for example hindsight return relabeling like the
    :class:`~.Reward2GoTransform` to update the return label for the
    actually achieved return.

    Args:
        target_return (float): target return to be achieved by the agent.
        mode (str): mode to be used to update the target return. Can be either "reduce" or "constant". Default: "reduce".

    Examples:
        >>> transform = TargetReturn(10.0, mode="reduce")
        >>> td = TensorDict({}, [10])
        >>> td = transform.reset(td)
        >>> td["target_return"]
        tensor([[10.],
                [10.],
                [10.],
                [10.],
                [10.],
                [10.],
                [10.],
                [10.],
                [10.],
                [10.]])
        >>> # take a step with mode "reduce"
        >>> # target return is updated by subtracting the reward
        >>> reward = torch.ones((10,1))
        >>> td.set(("next", "reward"), reward)
        >>> td = transform._step(td)
        >>> td["next", "target_return"]
        tensor([[9.],
                [9.],
                [9.],
                [9.],
                [9.],
                [9.],
                [9.],
                [9.],
                [9.],
                [9.]])

    """

    MODES = ["reduce", "constant"]
    MODE_ERR = "Mode can only be 'reduce' or 'constant'."

    def __init__(
        self,
        target_return: float,
        mode: str = "reduce",
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = ["target_return"]
        if mode not in self.MODES:
            raise ValueError(self.MODE_ERR)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.target_return = target_return
        self.mode = mode

    def reset(self, tensordict: TensorDict):
        init_target_return = torch.full(
            size=(*tensordict.batch_size, 1),
            fill_value=self.target_return,
            dtype=torch.float32,
            device=tensordict.device,
        )

        for out_key in self.out_keys:
            target_return = tensordict.get(out_key, default=None)

            if target_return is None:
                target_return = init_target_return

            tensordict.set(
                out_key,
                target_return,
            )
        return tensordict

    def _call(self, tensordict: TensorDict) -> TensorDict:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in tensordict.keys(include_nested=True):
                target_return = self._apply_transform(
                    tensordict.get(in_key), tensordict.get(out_key)
                )
                tensordict.set(out_key, target_return)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for out_key in self.out_keys:
            tensordict.set(("next", out_key), tensordict.get(out_key))
        return super()._step(tensordict)

    def _apply_transform(
        self, reward: torch.Tensor, target_return: torch.Tensor
    ) -> torch.Tensor:
        if self.mode == "reduce":
            target_return = target_return - reward
            return target_return
        elif self.mode == "constant":
            return target_return
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )

    def transform_observation_spec(
        self, observation_spec: CompositeSpec
    ) -> CompositeSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                f"observation_spec was expected to be of type CompositeSpec. Got {type(observation_spec)} instead."
            )

        target_return_spec = BoundedTensorSpec(
            minimum=-float("inf"),
            maximum=self.target_return,
            shape=self.parent.reward_spec.shape,
            dtype=self.parent.reward_spec.dtype,
            device=self.parent.reward_spec.device,
        )
        observation_spec[self.out_keys[0]] = target_return_spec

        return observation_spec


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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if not _has_tv:
            raise ImportError(
                "Torchvision not found. The Resize transform relies on "
                "torchvision implementation. "
                "Consider installing this dependency."
            )
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
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
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
            observation_spec.shape = space.minimum.shape
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = w
        self.h = h if h else w

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = center_crop(observation, [self.w, self.h])
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
            observation_spec.shape = space.minimum.shape
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        allow_positive_dim: bool = False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
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
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
            observation_spec.shape = space.minimum.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        in_keys_inv: Optional[Sequence[NestedKey]] = None,
        out_keys_inv: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = []  # default
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
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
            spec.shape = space.minimum.shape
        else:
            spec.shape = self._apply_transform(torch.zeros(spec.shape)).shape
        return spec

    def _inv_transform_spec(self, spec: TensorSpec) -> None:
        space = spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._inv_apply_transform(space.minimum)
            space.maximum = self._inv_apply_transform(space.maximum)
            spec.shape = space.minimum.shape
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


class GrayScale(ObservationTransform):
    """Turns a pixel observation to grayscale."""

    def __init__(
        self,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        super(GrayScale, self).__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = F.rgb_to_grayscale(observation)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
            observation_spec.shape = space.minimum.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec


class ObservationNorm(ObservationTransform):
    """Observation affine transformation layer.

    Normalizes an observation according to

    .. math::
        obs = obs * scale + loc

    Args:
        loc (number or tensor): location of the affine transform
        scale (number or tensor): scale of the affine transform
        in_keys (seuqence of NestedKey, optional): entries to be normalized. Defaults to ["observation", "pixels"].
            All entries will be normalized with the same values: if a different behaviour is desired
            (e.g. a different normalization for pixels and states) different :obj:`ObservationNorm`
            objects should be used.
        out_keys (seuqence of NestedKey, optional): output entries. Defaults to the value of `in_keys`.
        in_keys_inv (seuqence of NestedKey, optional): ObservationNorm also supports inverse transforms. This will
            only occur if a list of keys is provided to :obj:`in_keys_inv`. If none is provided,
            only the forward transform will be called.
        out_keys_inv (seuqence of NestedKey, optional): output entries for the inverse transform.
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        in_keys_inv: Optional[Sequence[NestedKey]] = None,
        out_keys_inv: Optional[Sequence[NestedKey]] = None,
        standard_normal: bool = False,
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

    def _inv_apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if self.loc is None or self.scale is None:
            raise RuntimeError(
                "Loc/Scale have not been initialized. Either pass in values in the constructor "
                "or call the init_stats method"
            )
        if not self.standard_normal:
            loc = self.loc
            scale = self.scale
            return (obs - loc) / scale
        else:
            scale = self.scale
            loc = self.loc
            return obs * scale + loc

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
        return observation_spec

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        space = input_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
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
        in_keys (seuqence of NestedKey, optional): keys pointing to the frames that have
            to be concatenated. Defaults to ["pixels"].
        out_keys (seuqence of NestedKey, optional): keys pointing to where the output
            has to be written. Defaults to the value of `in_keys`.
        padding (str, optional): the padding method. One of ``"same"`` or ``"zeros"``.
            Defaults to ``"same"``, ie. the first value is uesd for padding.
        as_inverse (bool, optional): if ``True``, the transform is applied as an inverse transform. Defaults to ``False``.

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

    """

    inplace = False
    _CAT_DIM_ERR = (
        "dim must be > 0 to accomodate for tensordict of "
        "different batch-sizes (since negative dims are batch invariant)."
    )
    ACCEPTED_PADDING = {"same", "zeros"}

    def __init__(
        self,
        N: int,
        dim: int,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        padding="same",
        as_inverse=False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.N = N
        if dim > 0:
            raise ValueError(self._CAT_DIM_ERR)
        self.dim = dim
        if padding not in self.ACCEPTED_PADDING:
            raise ValueError(f"padding must be one of {self.ACCEPTED_PADDING}")
        self.padding = padding
        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            self.register_buffer(
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        # keeps track of calls to _reset since it's only _call that will populate the buffer
        self._just_reset = False
        self.as_inverse = as_inverse

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets _buffers."""
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            parent = self.parent
            if parent is not None:
                parent_device = parent.device
                if self.as_inverse:
                    raise Exception(
                        "CatFrames as inverse is not supported as a transform for environments, only for replay buffers."
                    )
            else:
                parent_device = None
            _reset = torch.ones(
                self.parent.done_spec.shape if self.parent else tensordict.batch_size,
                dtype=torch.bool,
                device=parent_device,
            )
        _reset = _reset.sum(
            tuple(range(tensordict.batch_dims, _reset.ndim)), dtype=torch.bool
        )

        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            buffer[_reset] = 0

        self._just_reset = True
        return tensordict

    def _make_missing_buffer(self, data, buffer_name):
        shape = list(data.shape)
        d = shape[self.dim]
        shape[self.dim] = d * self.N
        shape = torch.Size(shape)
        getattr(self, buffer_name).materialize(shape)
        buffer = getattr(self, buffer_name).to(data.dtype).to(data.device).zero_()
        setattr(self, buffer_name, buffer)
        return buffer

    def _inv_call(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.as_inverse:
            return self.unfolding(tensordict)
        else:
            return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        _reset = tensordict.get("_reset", None)
        if _reset is not None:
            _reset = _reset.sum(
                tuple(range(tensordict.batch_dims, _reset.ndim)), dtype=torch.bool
            )

        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_cat_buffers_{in_key}"
            data = tensordict[in_key]
            d = data.size(self.dim)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(data, buffer_name)
            # shift obs 1 position to the right
            if self._just_reset or (_reset is not None and _reset.any()):
                data_in = buffer[_reset]
                shape = [1 for _ in data_in.shape]
                shape[self.dim] = self.N
                if self.padding == "same":
                    buffer[_reset] = buffer[_reset].copy_(
                        data[_reset].repeat(shape).clone()
                    )
                elif self.padding == "zeros":
                    buffer[_reset] = 0
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError
            buffer.copy_(torch.roll(buffer, shifts=-d, dims=self.dim))
            # add new obs
            idx = self.dim
            if idx < 0:
                idx = buffer.ndimension() + idx
            else:
                raise ValueError(self._CAT_DIM_ERR)
            idx = [slice(None, None) for _ in range(idx)] + [slice(-d, None)]
            buffer[idx].copy_(data)
            # add to tensordict
            tensordict.set(out_key, buffer.clone())
        self._just_reset = False
        return tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = torch.cat([space.minimum] * self.N, self.dim)
            space.maximum = torch.cat([space.maximum] * self.N, self.dim)
            observation_spec.shape = space.minimum.shape
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
                if self.padding == "zeros":
                    data0 = [torch.zeros_like(elt) for elt in data0[:-1]] + data0[-1:]
                elif self.padding == "same":
                    pass
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError
            elif self.padding == "same":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [data[tuple(idx)].unsqueeze(tensordict.ndim - 1)] * (self.N - 1)
            elif self.padding == "zeros":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [
                    torch.zeros_like(data[tuple(idx)]).unsqueeze(tensordict.ndim - 1)
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
        in_keys: Optional[Sequence[NestedKey]] = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        super().__init__(in_keys=in_keys)
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

    forward = _call


class DoubleToFloat(Transform):
    """Maps actions float to double before they are called on the environment.

    Args:
        in_keys (sequence of NestedKey, optional): list of double keys to be converted to
            float before being exposed to external objects and functions.
        in_keys_inv (sequence of NestedKey, optional): list of float keys to be converted to
            double before being passed to the contained base_env or storage.

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double)}, [])
        >>> transform = DoubleToFloat(in_keys=["obs"])
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32

    """

    invertible = True

    def __init__(
        self,
        in_keys: Optional[Sequence[NestedKey]] = None,
        in_keys_inv: Optional[Sequence[NestedKey]] = None,
    ):
        super().__init__(in_keys=in_keys, in_keys_inv=in_keys_inv)

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float)

    def _inv_apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.double)

    def _transform_spec(self, spec: TensorSpec) -> None:
        if isinstance(spec, CompositeSpec):
            for key in spec:
                self._transform_spec(spec[key])
        else:
            spec.dtype = torch.float
            space = spec.space
            if isinstance(space, ContinuousBox):
                space.minimum = space.minimum.to(torch.float)
                space.maximum = space.maximum.to(torch.float)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec["_action_spec"]
        state_spec = input_spec["_state_spec"]
        for key in self.in_keys_inv:
            if key in action_spec.keys(True):
                _spec = action_spec
            elif state_spec is not None and key in state_spec.keys(True):
                _spec = state_spec
            else:
                raise KeyError(f"Key {key} not found in state_spec and action_spec.")
            if _spec[key].dtype is not torch.double:
                raise TypeError(
                    f"input_spec[{key}].dtype is not double: {input_spec[key].dtype}"
                )
            self._transform_spec(_spec[key])
        return input_spec

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_key = self.parent.reward_key if self.parent is not None else "reward"
        if unravel_key(reward_key) in self.in_keys:
            if reward_spec.dtype is not torch.double:
                raise TypeError("reward_spec.dtype is not double")

            self._transform_spec(reward_spec)
        return reward_spec

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        self._transform_spec(observation_spec)
        return observation_spec

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_keys={self.out_keys}, "
            f"in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
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
            Default is False.

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
        in_keys: Optional[Sequence[NestedKey]] = None,
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
        # super().__init__(in_keys=in_keys)
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
        parent = self.parent
        obs_spec = parent.observation_spec
        in_keys = []
        for key, value in obs_spec.items():
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

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
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
        super().__init__(
            in_keys=in_keys,
            out_keys=in_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=in_keys_inv,
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
        action = action.argmax(-1)  # bool to int
        idx = action >= self.num_actions_effective
        if idx.any():
            action[idx] = torch.randint(self.num_actions_effective, (idx.sum(),))
        action = nn.functional.one_hot(action, self.num_actions_effective)
        return action

    def transform_input_spec(self, input_spec: CompositeSpec):
        input_spec = input_spec.clone()
        for key in input_spec["_action_spec"].keys(True, True):
            key = ("_action_spec", key)
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
        super().__init__([])
        if frame_skip < 1:
            raise ValueError("frame_skip should have a value greater or equal to one.")
        self.frame_skip = frame_skip

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError("parent not found for FrameSkipTransform")
        reward_key = parent.reward_key
        reward = tensordict.get(("next", reward_key))
        for _ in range(self.frame_skip - 1):
            tensordict = parent._step(tensordict)
            reward = reward + tensordict.get(("next", reward_key))
        return tensordict.set(("next", reward_key), reward)

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
        noops (int, optional): number of actions performed after reset.
            Default is `30`.
        random (bool, optional): if False, the number of random ops will
            always be equal to the noops value. If True, the number of
            random actions will be randomly selected between 0 and noops.
            Default is `True`.

    """

    def __init__(self, noops: int = 30, random: bool = True):
        """Sample initial states by taking random number of no-ops on reset.

        No-op is assumed to be action 0.
        """
        super().__init__([])
        self.noops = noops
        self.random = random

    @property
    def base_env(self):
        return self.parent

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Do no-op action for a number of steps in [1, noop_max]."""
        td_reset = tensordict.clone(False)
        tensordict = tensordict.clone(False)
        # check that there is a single done state -- behaviour is undefined for multiple dones
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "NoopResetEnv.parent not found. Make sure that the parent is set."
            )
        done_key = parent.done_key
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

        while True:
            i = 0
            while i < noops:
                i += 1
                tensordict = parent.rand_step(tensordict)
                tensordict = step_mdp(tensordict, exclude_done=False)
                if tensordict.get(done_key):
                    tensordict = parent.reset(td_reset.clone(False))
                    break
            else:
                break

            trial += 1
            if trial > _MAX_NOOPS_TRIALS:
                tensordict = parent.rand_step(tensordict)
                if tensordict.get(("next", done_key)):
                    raise RuntimeError(
                        f"parent is still done after a single random step (i={i})."
                    )
                break

        if tensordict.get(done_key):
            raise RuntimeError("NoopResetEnv concluded with done environment")
        return tensordict.exclude(reward_key, inplace=True)

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
        primers (dict, optional): a dictionary containing key-spec pairs which will
            be used to populate the input tensordict.
        random (bool, optional): if ``True``, the values will be drawn randomly from
            the TensorSpec domain (or a unit Gaussian if unbounded). Otherwise a fixed value will be assumed.
            Defaults to `False`.
        default_value (float, optional): if non-random filling is chosen, this
            value will be used to populate the tensors. Defaults to `0.0`.
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

    def __init__(self, primers: dict = None, random=False, default_value=0.0, **kwargs):
        self.device = kwargs.pop("device", None)
        if primers is not None:
            if kwargs:
                raise RuntimeError(
                    "providing the primers as a dictionary is incompatible with extra keys provided "
                    "as kwargs."
                )
            kwargs = primers
        self.primers = kwargs
        self.random = random
        self.default_value = default_value

        # sanity check
        for spec in self.primers.values():
            if not isinstance(spec, TensorSpec):
                raise ValueError(
                    "The values of the primers must be a subtype of the TensorSpec class. "
                    f"Got {type(spec)} instead."
                )
        super().__init__([])

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

    def to(self, dtype_or_device):
        if not isinstance(dtype_or_device, torch.dtype):
            self.device = dtype_or_device
        return super().to(dtype_or_device)

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

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for key in self.primers.keys():
            tensordict.setdefault(("next", key), tensordict.get(key, default=None))
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
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
        for key, spec in self.primers.items():
            if self.random:
                value = spec.rand(shape)
            else:
                value = torch.full_like(
                    spec.zero(shape),
                    self.default_value,
                )
            tensordict.set(key, value)
        return tensordict

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(primers={self.primers}, default_value={self.default_value}, random={self.random})"


class PinMemoryTransform(Transform):
    """Calls pin_memory on the tensordict to facilitate writing on CUDA devices."""

    def __init__(self):
        super().__init__([])

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.pin_memory()

    forward = _call


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
        super().__init__(primers=primers, random=random)


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
        in_keys: Optional[Sequence[NestedKey]] = None,
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
        super().__init__(in_keys)
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


class RewardSum(Transform):
    """Tracks episode cumulative rewards.

    This transform accepts a list of tensordict reward keys (i.e. ´in_keys´) and tracks their cumulative
    value along each episode. When called, the transform creates a new tensordict key for each in_key named
    ´episode_{in_key}´ where  the cumulative values are written. All ´in_keys´ should be part of the env
    reward and be present in the env reward_spec.

    If no in_keys are specified, this transform assumes ´reward´ to be the input key. However, multiple rewards
    (e.g. reward1 and reward2) can also be specified. If ´in_keys´ are not present in the provided tensordict,
    this transform hos no effect.
    """

    def __init__(
        self,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        """Initialises the transform. Filters out non-reward input keys and defines output keys."""
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None and in_keys == ["reward"]:
            out_keys = ["episode_reward"]
        elif out_keys is None:
            raise RuntimeError(
                "the out_keys must be specified for non-conventional in-keys in RewardSum."
            )

        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets episode rewards."""
        # Non-batched environments
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(
                self.parent.done_spec.shape if self.parent else tensordict.batch_size,
                dtype=torch.bool,
                device=tensordict.device,
            )
        if _reset.any():
            reward_key = self.parent.reward_key if self.parent else "reward"
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                if out_key in tensordict.keys(True, True):
                    value = tensordict[out_key]
                    tensordict[out_key] = value.masked_fill(
                        expand_as_right(_reset, value), 0.0
                    )
                elif unravel_key(in_key) == unravel_key(reward_key):
                    # Since the episode reward is not in the tensordict, we need to allocate it
                    # with zeros entirely (regardless of the _reset mask)
                    tensordict[out_key] = self.parent.reward_spec.zero()
                else:
                    try:
                        tensordict[out_key] = self.parent.observation_spec[
                            in_key
                        ].zero()
                    except KeyError as err:
                        raise KeyError(
                            f"The key {in_key} was not found in the parent "
                            f"observation_spec with keys "
                            f"{list(self.parent.observation_spec.keys(True))}. "
                        ) from err
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Updates the episode rewards with the step rewards."""
        # Update episode rewards
        next_tensordict = tensordict.get("next")
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in next_tensordict.keys(include_nested=True):
                reward = next_tensordict.get(in_key)
                if out_key not in tensordict.keys(True):
                    tensordict.set(out_key, torch.zeros_like(reward))
                next_tensordict.set(out_key, tensordict.get(out_key) + reward)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        tensordict.set("next", next_tensordict)
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec, adding the new keys generated by RewardSum."""
        # Retrieve parent reward spec
        reward_spec = self.parent.reward_spec
        reward_key = self.parent.reward_key if self.parent else "reward"

        episode_specs = {}
        if isinstance(reward_spec, CompositeSpec):
            # If reward_spec is a CompositeSpec, all in_keys should be keys of reward_spec
            if not all(k in reward_spec.keys(True, True) for k in self.in_keys):
                raise KeyError("Not all in_keys are present in ´reward_spec´")

            # Define episode specs for all out_keys
            for out_key in self.out_keys:
                episode_spec = UnboundedContinuousTensorSpec(
                    shape=reward_spec.shape,
                    device=reward_spec.device,
                    dtype=reward_spec.dtype,
                )
                episode_specs.update({out_key: episode_spec})

        else:
            # If reward_spec is not a CompositeSpec, the only in_key should be ´reward´
            if set(unravel_key_list(self.in_keys)) != {unravel_key(reward_key)}:
                raise KeyError(
                    "reward_spec is not a CompositeSpec class, in_keys should only include ´reward´"
                )

            # Define episode spec
            episode_spec = UnboundedContinuousTensorSpec(
                device=reward_spec.device,
                dtype=reward_spec.dtype,
                shape=reward_spec.shape,
            )
            episode_specs.update({"episode_reward": episode_spec})

        # Update observation_spec with episode_specs
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(episode_specs)
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
    """Counts the steps from a reset and sets the done state to True after a certain number of steps.

    Args:
        max_steps (int, optional): a positive integer that indicates the
            maximum number of steps to take before setting the ``truncated_key``
            entry to ``True``.
            However, the step count will still be
            incremented on each call to step() into the `step_count` attribute.
        truncated_key (NestedKey, optional): the key where the truncated key should
            be written. Defaults to ``"truncated"``, which is recognised by
            data collectors as a reset signal.
        step_count_key (NestedKey, optional): the key where the step_count key should
            be written. Defaults to ``"step_count"``, which is recognised by
            data collectors.
    """

    invertible = False

    def __init__(
        self,
        max_steps: Optional[int] = None,
        truncated_key: Optional[NestedKey] = "truncated",
        step_count_key: Optional[NestedKey] = "step_count",
    ):
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps should have a value greater or equal to one.")
        self.max_steps = max_steps
        self.truncated_key = truncated_key
        self.step_count_key = step_count_key
        super().__init__([])

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        done_key = self.parent.done_key if self.parent else "done"
        done = tensordict.get(done_key, None)
        if done is None:
            done = torch.ones(
                self.parent.done_spec.shape,
                dtype=self.parent.done_spec.dtype,
                device=self.parent.done_spec.device,
            )
        _reset = tensordict.get(
            "_reset",
            # TODO: decide if using done here, or using a default `True` tensor
            default=None,
        )
        if _reset is None:
            _reset = torch.ones_like(done)
        step_count = tensordict.get(
            self.step_count_key,
            default=None,
        )
        if step_count is None:
            step_count = torch.zeros_like(done, dtype=torch.int64)

        step_count[_reset] = 0
        tensordict.set(
            self.step_count_key,
            step_count,
        )
        if self.max_steps is not None:
            truncated = step_count >= self.max_steps
            tensordict.set(self.truncated_key, truncated)
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        step_count = tensordict.get(
            self.step_count_key,
        )
        next_step_count = step_count + 1
        tensordict.set(("next", self.step_count_key), next_step_count)
        if self.max_steps is not None:
            truncated = next_step_count >= self.max_steps
            tensordict.set(("next", self.truncated_key), truncated)
        return tensordict

    def transform_observation_spec(
        self, observation_spec: CompositeSpec
    ) -> CompositeSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                f"observation_spec was expected to be of type CompositeSpec. Got {type(observation_spec)} instead."
            )
        observation_spec[self.step_count_key] = UnboundedDiscreteTensorSpec(
            shape=self.parent.done_spec.shape
            if self.parent
            else observation_spec.shape,
            dtype=torch.int64,
            device=observation_spec.device,
        )
        observation_spec[self.step_count_key].space.minimum = (
            observation_spec[self.step_count_key].space.minimum * 0
        )
        if self.max_steps is not None and self.truncated_key != self.parent.done_key:
            observation_spec[self.truncated_key] = self.parent.done_spec.clone()
        return observation_spec

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        if not isinstance(input_spec, CompositeSpec):
            raise ValueError(
                f"input_spec was expected to be of type CompositeSpec. Got {type(input_spec)} instead."
            )
        if input_spec["_state_spec"] is None:
            input_spec["_state_spec"] = CompositeSpec(
                shape=input_spec.shape, device=input_spec.device
            )
        step_spec = UnboundedDiscreteTensorSpec(
            shape=self.parent.done_spec.shape if self.parent else input_spec.shape,
            dtype=torch.int64,
            device=input_spec.device,
        )
        step_spec.space.minimum *= 0
        input_spec["_state_spec", self.step_count_key] = step_spec

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
    """Excludes keys from the input tensordict.

    Args:
        *excluded_keys (iterable of NestedKey): The name of the keys to exclude. If the key is
            not present, it is simply ignored.

    """

    def __init__(self, *excluded_keys):
        super().__init__(in_keys=[], in_keys_inv=[], out_keys=[], out_keys_inv=[])
        try:
            excluded_keys = unravel_key_list(excluded_keys)
        except TypeError:
            raise TypeError(
                "excluded keys must be a list or tuple of strings or tuples of strings."
            )
        self.excluded_keys = excluded_keys
        if "reward" in excluded_keys:
            raise RuntimeError("'reward' cannot be excluded from the keys.")

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.exclude(*self.excluded_keys)

    forward = _call

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.exclude(*self.excluded_keys)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if any(key in observation_spec.keys(True, True) for key in self.excluded_keys):
            return CompositeSpec(
                {
                    key: value
                    for key, value in observation_spec.items()
                    if unravel_key(key) not in self.excluded_keys
                },
                shape=observation_spec.shape,
            )
        return observation_spec


class SelectTransform(Transform):
    """Select keys from the input tensordict.

    In general, the :obj:`ExcludeTransform` should be preferred: this transforms also
        selects the "action" (or other keys from input_spec), "done" and "reward"
        keys but other may be necessary.

    Args:
        *selected_keys (iterable of NestedKey): The name of the keys to select. If the key is
            not present, it is simply ignored.

    """

    def __init__(self, *selected_keys):
        super().__init__(in_keys=[], in_keys_inv=[], out_keys=[], out_keys_inv=[])
        try:
            selected_keys = unravel_key_list(selected_keys)
        except TypeError:
            raise TypeError(
                "selected keys must be a list or tuple of strings or tuples of strings."
            )
        self.selected_keys = selected_keys

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent:
            input_keys = self.parent.input_spec.keys(True, True)
        else:
            input_keys = []
        reward_key = self.parent.reward_key if self.parent else "reward"
        done_key = self.parent.done_key if self.parent else "done"
        return tensordict.select(
            *self.selected_keys, reward_key, done_key, *input_keys, strict=False
        )

    forward = _call

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent:
            input_keys = self.parent.input_spec.keys(True, True)
        else:
            input_keys = []
        reward_key = self.parent.reward_key if self.parent else "reward"
        done_key = self.parent.done_key if self.parent else "done"
        return tensordict.select(
            *self.selected_keys, reward_key, done_key, *input_keys, strict=False
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return CompositeSpec(
            {
                key: value
                for key, value in observation_spec.items()
                if unravel_key(key) in self.selected_keys
            },
            shape=observation_spec.shape,
        )


class TimeMaxPool(Transform):
    """Take the maximum value in each position over the last T observations.

    This transform take the maximum value in each position for all in_keys tensors over the last T time steps.

    Args:
        in_keys (sequence of NestedKey, optional): input keys on which the max pool will be applied. Defaults to "observation" if left empty.
        out_keys (sequence of NestedKey, optional): output keys where the output will be written. Defaults to `in_keys` if left empty.
        T (int, optional): Number of time steps over which to apply max pooling.
    """

    invertible = False

    def __init__(
        self,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        T: int = 1,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if T < 1:
            raise ValueError(
                "TimeMaxPoolTranform T parameter should have a value greater or equal to one."
            )
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                "TimeMaxPoolTranform in_keys and out_keys don't have the same number of elements"
            )
        self.buffer_size = T
        for in_key in self.in_keys:
            buffer_name = f"_maxpool_buffer_{in_key}"
            setattr(
                self,
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets _buffers."""
        # Non-batched environments
        if len(tensordict.batch_size) < 1 or tensordict.batch_size[0] == 1:
            for in_key in self.in_keys:
                buffer_name = f"_maxpool_buffer_{in_key}"
                buffer = getattr(self, buffer_name)
                if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                    continue
                buffer.fill_(0.0)

        # Batched environments
        else:
            _reset = tensordict.get(
                "_reset",
                torch.ones(
                    self.parent.done_spec.shape
                    if self.parent
                    else tensordict.batch_size,
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
            )
            for in_key in self.in_keys:
                buffer_name = f"_maxpool_buffer_{in_key}"
                buffer = getattr(self, buffer_name)
                if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                    continue
                _reset = _reset.sum(
                    tuple(range(tensordict.batch_dims, _reset.ndim)), dtype=torch.bool
                )
                buffer[:, _reset] = 0.0

        return tensordict

    def _make_missing_buffer(self, data, buffer_name):
        buffer = getattr(self, buffer_name)
        buffer.materialize((self.buffer_size,) + data.shape)
        buffer = buffer.to(data.dtype).to(data.device).zero_()
        setattr(self, buffer_name, buffer)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_maxpool_buffer_{in_key}"
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                data = tensordict[in_key]
                self._make_missing_buffer(data, buffer_name)
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(tensordict[in_key])
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
        super().__init__([])

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
        super().__init__(in_keys=[], out_keys=[init_key])

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.out_keys[0] not in tensordict.keys(True, True):
            device = tensordict.device
            if device is None:
                device = torch.device("cpu")
            tensordict.set(
                self.out_keys[0],
                torch.zeros(
                    self.parent.done_spec.shape, device=device, dtype=torch.bool
                ),
            )
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            tensordict.set(
                self.out_keys[0],
                torch.ones(
                    self.parent.done_spec.shape,
                    device=device,
                    dtype=torch.bool,
                ),
            )
        else:
            tensordict.set(self.out_keys[0], _reset.clone())
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec[self.out_keys[0]] = DiscreteTensorSpec(
            2,
            dtype=torch.bool,
            device=self.parent.device,
            shape=self.parent.done_spec.shape,
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
        if "done" in in_keys and not create_copy:
            raise ValueError(
                "Renaming 'done' is not allowed. Set `create_copy` to `True` "
                "to create a copy of the done state."
            )
        if "reward" in in_keys and not create_copy:
            raise ValueError(
                "Renaming 'reward' is not allowed. Set `create_copy` to `True` "
                "to create a copy of the reward entry."
            )
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
            out = tensordict.select(*self.in_keys)
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                out.rename_key_(in_key, out_key)
            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                tensordict.rename_key_(in_key, out_key)
        return tensordict

    forward = _call

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # no in-place modif
        if self.create_copy:
            out = tensordict.select(*self.out_keys_inv)
            for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
                out.rename_key_(out_key, in_key)
            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
                tensordict.rename_key_(out_key, in_key)
        return tensordict

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        # we need to check whether there are special keys
        output_spec = output_spec.clone()
        if "done" in self.in_keys:
            for i, out_key in enumerate(self.out_keys):  # noqa: B007
                if self.in_keys[i] == "done":
                    break
            else:
                raise RuntimeError("Expected one key to be 'done'")
            output_spec["_observation_spec"][out_key] = output_spec[
                "_done_spec"
            ].clone()
        if "reward" in self.in_keys:
            for i, out_key in enumerate(self.out_keys):  # noqa: B007
                if self.in_keys[i] == "reward":
                    break
            else:
                raise RuntimeError("Expected one key to be 'reward'")
            output_spec["_observation_spec"][out_key] = output_spec[
                "_reward_spec"
            ].clone()
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in ("reward", "done"):
                continue
            if out_key in ("done", "reward"):
                output_spec[out_key] = output_spec["_observation_spec"][in_key].clone()
            else:
                output_spec["_observation_spec"][out_key] = output_spec[
                    "_observation_spec"
                ][in_key].clone()
            if not self.create_copy:
                del output_spec["_observation_spec"][in_key]
        return output_spec

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        # we need to check whether there are special keys
        input_spec = input_spec.clone()
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            in_key = (in_key,) if not isinstance(in_key, tuple) else in_key
            out_key = (out_key,) if not isinstance(out_key, tuple) else out_key
            input_spec[("_state_spec", *out_key)] = input_spec[
                ("_state_spec", *in_key)
            ].clone()
            if not self.create_copy:
                del input_spec[("_state_spec", *in_key)]
        return input_spec


class Reward2GoTransform(Transform):
    """Calculates the reward to go based on the episode reward and a discount factor.

    As the :class:`~.Reward2GoTransform` is only an inverse transform the ``in_keys`` will be directly used for the ``in_keys_inv``.
    The reward-to-go can be only calculated once the episode is finished. Therefore, the transform should be applied to the replay buffer
    and not to the collector.

    Args:
        in_keys (sequence of NestedKey): the entries to rename. Defaults to
            ``("next", "reward")`` if none is provided.
        out_keys (sequence of NestedKey): the entries to rename. Defaults to
            the values of ``in_keys`` if none is provided.
        gamma (float or torch.Tensor): the discount factor. Defaults to 1.0.

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

    """

    ENV_ERR = (
        "The Reward2GoTransform is only an inverse transform and can "
        "only be applied to the replay buffer and not to the collector or the environment."
    )

    def __init__(
        self,
        gamma: Optional[Union[float, torch.Tensor]] = 1.0,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if in_keys is None:
            in_keys = [("next", "reward")]
        if out_keys is None:
            out_keys = deepcopy(in_keys)
        # out_keys = ["reward_to_go"]
        super().__init__(
            in_keys=in_keys,
            in_keys_inv=in_keys,
            out_keys_inv=out_keys,
        )

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma)

        self.register_buffer("gamma", gamma)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        done_key = self.parent.done_key if self.parent else "done"
        done = tensordict.get(("next", done_key))
        truncated = tensordict.get(("next", "truncated"), None)
        if truncated is not None:
            done_or_truncated = done | truncated
        else:
            done_or_truncated = done
        if not done_or_truncated.any(-2).all():
            raise RuntimeError(
                "No episode ends found to calculate the reward to go. Make sure that the number of frames_per_batch is larger than number of steps per episode."
            )
        found = False
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                found = True
                item = self._inv_apply_transform(
                    tensordict.get(in_key), done_or_truncated
                )
                tensordict.set(
                    out_key,
                    item,
                )
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
