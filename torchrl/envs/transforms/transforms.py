# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import multiprocessing as mp
from copy import copy, deepcopy
from textwrap import indent
from typing import Any, List, Optional, OrderedDict, Sequence, Tuple, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn, Tensor

from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    ContinuousBox,
    DEVICE_TYPING,
    NdUnboundedContinuousTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import check_finite
from torchrl.envs.utils import step_mdp


try:
    from torchvision.transforms.functional import center_crop
    from torchvision.transforms.functional_tensor import (
        resize,
    )  # as of now resize is imported from torchvision

    _has_tv = True
except ImportError:
    _has_tv = False

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10


def _apply_to_composite(function):
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, CompositeSpec):
            d = observation_spec._specs
            for in_key, out_key in zip(self.in_keys, self.out_keys):
                if in_key in observation_spec.keys():
                    d[out_key] = function(self, observation_spec[in_key])
            return CompositeSpec(d)
        else:
            return function(self, observation_spec)

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
        in_keys: Sequence[str],
        out_keys: Optional[Sequence[str]] = None,
        in_keys_inv: Optional[Sequence[str]] = None,
        out_keys_inv: Optional[Sequence[str]] = None,
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
        self.__dict__["_container"] = None
        self.__dict__["_parent"] = None

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets a tranform if it is stateful."""
        return tensordict

    def _check_inplace(self) -> None:
        if not hasattr(self, "inplace"):
            raise AttributeError(
                f"Transform of class {self.__class__.__name__} has no "
                f"attribute inplace, consider implementing it."
            )

    def init(self, tensordict) -> None:
        pass

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor.

        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform).

        """
        raise NotImplementedError

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reads the input tensordict, and for the selected keys, applies the transform."""
        self._check_inplace()
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in tensordict.keys(include_nested=True):
                observation = self._apply_transform(tensordict.get(in_key))
                tensordict.set(out_key, observation, inplace=self.inplace)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self._call(tensordict)
        return tensordict
        # raise NotImplementedError("""`Transform.forward` is currently not implemented (reserved for usage beyond envs). Use `Transform._step` instead.""")

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # placeholder when we'll move to tensordict['next']
        # tensordict["next"] = self._call(tensordict.get("next"))
        out = self._call(tensordict)
        # print(out, tensordict, out is tensordict, (out==tensordict).all())
        return out

    def _inv_apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return obs

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._check_inplace()
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                observation = self._inv_apply_transform(tensordict.get(in_key))
                tensordict.set(out_key, observation, inplace=self.inplace)
        return tensordict

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._inv_call(tensordict)
        return tensordict

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

    def dump(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.in_keys})"

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        if self.__dict__["_container"] is not None:
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
        self_copy.reset_parent()
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


class TransformedEnv(EnvBase):
    """A transformed_in environment.

    Args:
        env (EnvBase): original environment to be transformed_in.
        transform (Transform, optional): transform to apply to the tensordict resulting
            from :obj:`env.step(td)`. If none is provided, an empty Compose
            placeholder in an eval mode is used.
        cache_specs (bool, optional): if True, the specs will be cached once
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
        device = kwargs.pop("device", env.device)
        env = env.to(device)
        super().__init__(device=None, **kwargs)

        if isinstance(env, TransformedEnv):
            self._set_env(env.base_env, device)
            if type(transform) is not Compose:
                # we don't use isinstance as some transforms may be subclassed from
                # Compose but with other features that we don't want to loose.
                transform = [transform]
            else:
                for t in transform:
                    t.reset_parent()
            env_transform = env.transform
            if type(env_transform) is not Compose:
                env_transform.reset_parent()
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
        self.__dict__["_reward_spec"] = None
        self.__dict__["_input_spec"] = None
        self.__dict__["_observation_spec"] = None
        self.batch_size = self.base_env.batch_size

    def _set_env(self, env: EnvBase, device) -> None:
        self.base_env = env.to(device)
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
    def observation_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment."""
        if self._observation_spec is None or not self.cache_specs:
            observation_spec = self.transform.transform_observation_spec(
                deepcopy(self.base_env.observation_spec)
            )
            if self.cache_specs:
                self.__dict__["_observation_spec"] = observation_spec
        else:
            observation_spec = self._observation_spec
        return observation_spec

    @property
    def action_spec(self) -> TensorSpec:
        """Action spec of the transformed environment."""
        return self.input_spec["action"]

    @property
    def input_spec(self) -> TensorSpec:
        """Action spec of the transformed environment."""
        if self._input_spec is None or not self.cache_specs:
            input_spec = self.transform.transform_input_spec(
                deepcopy(self.base_env.input_spec)
            )
            if self.cache_specs:
                self.__dict__["_input_spec"] = input_spec
        else:
            input_spec = self._input_spec
        return input_spec

    @property
    def reward_spec(self) -> TensorSpec:
        """Reward spec of the transformed environment."""
        if self._reward_spec is None or not self.cache_specs:
            reward_spec = self.transform.transform_reward_spec(
                deepcopy(self.base_env.reward_spec)
            )
            if self.cache_specs:
                self.__dict__["_reward_spec"] = reward_spec
        else:
            reward_spec = self._reward_spec
        return reward_spec

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        tensordict_in = self.transform.inv(tensordict)
        tensordict_out = self.base_env._step(tensordict_in)
        tensordict_out = tensordict_out.update(
            tensordict.exclude(*tensordict_out.keys())
        )
        next_tensordict = self.transform._step(tensordict_out)
        tensordict_out.update(next_tensordict, inplace=False)

        return tensordict_out

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Set the seeds of the environment."""
        return self.base_env.set_seed(seed, static_seed=static_seed)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs):
        if tensordict is not None:
            tensordict = tensordict.clone(recurse=False)
        out_tensordict = self.base_env.reset(tensordict=tensordict, **kwargs)
        out_tensordict = self.transform.reset(out_tensordict)
        out_tensordict = self.transform(out_tensordict)
        return out_tensordict

    def state_dict(self) -> OrderedDict:
        state_dict = self.transform.state_dict()
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
        self.__dict__["_observation_spec"] = None
        self.__dict__["_input_spec"] = None
        self.__dict__["_reward_spec"] = None

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
            self.__dict__["_observation_spec"] = None
            self.__dict__["_reward_spec"] = None

    def to(self, device: DEVICE_TYPING) -> TransformedEnv:
        self.base_env.to(device)
        self.transform.to(device)

        if self.cache_specs:
            self.__dict__["_input_spec"] = None
            self.__dict__["_observation_spec"] = None
            self.__dict__["_reward_spec"] = None
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


class ObservationTransform(Transform):
    """Abstract class for transformations of the observations."""

    inplace = False

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = [
                "observation",
                "pixels",
                "observation_state",
            ]
        super(ObservationTransform, self).__init__(in_keys=in_keys, out_keys=out_keys)


class Compose(Transform):
    """Composes a chain of transforms.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transforms = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(*transforms)
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    inplace = False

    def __init__(self, *transforms: Transform):
        super().__init__(in_keys=[])
        self.transforms = nn.ModuleList(transforms)
        for t in self.transforms:
            t.set_container(self)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms:
            tensordict = t._step(tensordict)
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for t in self.transforms[::-1]:
            tensordict = t.inv(tensordict)
        return tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms[::-1]:
            input_spec = t.transform_input_spec(input_spec)
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
        return reward_spec

    def __getitem__(self, item: Union[int, slice, List]) -> Union:
        transform = self.transforms
        transform = transform[item]
        if not isinstance(transform, Transform):
            out = Compose(*self.transforms[item])
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

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> Compose:
        for t in self.transforms:
            t.to(dest)
        return super().to(dest)

    def __iter__(self):
        return iter(self.transforms)

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


class ToTensorImage(ObservationTransform):
    """Transforms a numpy-like image (3 x W x H) to a pytorch image (3 x W x H).

    Transforms an observation image from a (... x W x H x 3) 0..255 uint8
    tensor to a single/double precision floating point (3 x W x H) tensor
    with values between 0 and 1.

    Args:
        unsqueeze (bool): if True, the observation tensor is unsqueezed
            along the first dimension. default=False.
        dtype (torch.dtype, optional): dtype to use for the resulting
            observations.

    Examples:
        >>> transform = ToTensorImage(in_keys=["pixels"])
        >>> ri = torch.randint(0, 255, (1,1,10,11,3), dtype=torch.uint8)
        >>> td = TensorDict(
        ...     {"pixels": ri},
        ...     [1, 1])
        >>> _ = transform(td)
        >>> obs = td.get("pixels")
        >>> print(obs.shape, obs.dtype)
        torch.Size([1, 1, 3, 10, 11]) torch.float32
    """

    inplace = False

    def __init__(
        self,
        unsqueeze: bool = False,
        dtype: Optional[torch.device] = None,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
        observation = observation.div(255).to(self.dtype)
        observation = observation.permute(
            *list(range(observation.ndimension() - 3)), -1, -3, -2
        )
        if observation.ndimension() == 3 and self.unsqueeze:
            observation = observation.unsqueeze(0)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._pixel_observation(observation_spec)
        observation_spec.shape = torch.Size(
            [
                *observation_spec.shape[:-3],
                observation_spec.shape[-1],
                observation_spec.shape[-3],
                observation_spec.shape[-2],
            ]
        )
        observation_spec.dtype = self.dtype
        return observation_spec

    def _pixel_observation(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.maximum = self._apply_transform(spec.space.maximum)
            spec.space.minimum = self._apply_transform(spec.space.minimum)
        return spec


class RewardClipping(Transform):
    """Clips the reward between `clamp_min` and `clamp_max`.

    Args:
        clip_min (scalar): minimum value of the resulting reward.
        clip_max (scalar): maximum value of the resulting reward.

    """

    inplace = True

    def __init__(
        self,
        clamp_min: float = None,
        clamp_max: float = None,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
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
            reward = reward.clamp_(self.clamp_min, self.clamp_max)
        elif self.clamp_min is not None:
            reward = reward.clamp_min_(self.clamp_min)
        elif self.clamp_max is not None:
            reward = reward.clamp_max_(self.clamp_max)
        return reward

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return BoundedTensorSpec(
                self.clamp_min,
                self.clamp_max,
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

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
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

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return BinaryDiscreteTensorSpec(n=1, device=reward_spec.device)


class Resize(ObservationTransform):
    """Resizes an pixel observation.

    Args:
        w (int): resulting width
        h (int): resulting height
        interpolation (str): interpolation method
    """

    inplace = False

    def __init__(
        self,
        w: int,
        h: int,
        interpolation: str = "bilinear",
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
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
        self.interpolation = interpolation

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        # flatten if necessary
        if observation.shape[-2:] == torch.Size([self.w, self.h]):
            return observation
        ndim = observation.ndimension()
        if ndim > 4:
            sizes = observation.shape[:-3]
            observation = torch.flatten(observation, 0, ndim - 4)
        observation = resize(
            observation, [self.w, self.h], interpolation=self.interpolation
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
    """

    inplace = False

    def __init__(
        self,
        w: int,
        h: int = None,
        in_keys: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(in_keys=in_keys)
        self.w = w
        self.h = h if h else w

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = center_crop(observation, [self.w, self.h])
        return observation

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            return CompositeSpec(
                **{
                    key: self.transform_observation_spec(_obs_spec)
                    if key in self.in_keys
                    else _obs_spec
                    for key, _obs_spec in observation_spec._specs.items()
                }
            )

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
        first_dim (int, optional): first dimension of the dimensions to flatten.
            Default is 0.
        last_dim (int, optional): last dimension of the dimensions to flatten.
            Default is -3.
    """

    inplace = False

    def __init__(
        self,
        first_dim: int,
        last_dim: int = -3,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.first_dim = first_dim
        self.last_dim = last_dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.flatten(observation, self.first_dim, self.last_dim)
        return observation

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        out = super().set_container(container)
        try:
            observation_spec = self.parent.observation_spec
            for key in self.in_keys:
                if key in observation_spec:
                    observation_spec = observation_spec[key]
                    if self.first_dim >= 0:
                        self.first_dim = self.first_dim - len(observation_spec.shape)
                    if self.last_dim >= 0:
                        self.last_dim = self.last_dim - len(observation_spec.shape)
                    break
        except AttributeError:
            if self.first_dim >= 0 or self.last_dim >= 0:
                raise ValueError(
                    f"FlattenObservation got first and last dim {self.first_dim} amd {self.last_dim}. "
                    f"Those values assume that the observation spec is known, which requires the "
                    f"parent environment to be set. "
                    f"Consider setting the parent environment beforehand (ie passing the transform "
                    f"to `TransformedEnv.append_transform()`) or setting strictly negative "
                    f"flatten dimensions to the transform."
                )
        return out

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
            f"first_dim={int(self.first_dim)}, last_dim={int(self.last_dim)})"
        )


class UnsqueezeTransform(Transform):
    """Inserts a dimension of size one at the specified position.

    Args:
        unsqueeze_dim (int): dimension to unsqueeze.
    """

    invertible = True
    inplace = False

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._unsqueeze_dim = None
        return super().__new__(cls)

    def __init__(
        self,
        unsqueeze_dim: int,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        in_keys_inv: Optional[Sequence[str]] = None,
        out_keys_inv: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self._unsqueeze_dim_orig = unsqueeze_dim

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        if self._unsqueeze_dim_orig < 0:
            self._unsqueeze_dim = self._unsqueeze_dim_orig
        else:
            container = self.parent
            try:
                batch_size = container.batch_size
            except AttributeError:
                raise ValueError(
                    f"Got the unsqueeze dimension {self._unsqueeze_dim_orig} which is greater or equal to zero. "
                    f"However this requires to know what the parent environment is, but it has not been provided. "
                    f"Consider providing a negative dimension or setting the transform using the "
                    f"`TransformedEnv.append_transform()` method."
                )
            self._unsqueeze_dim = self._unsqueeze_dim_orig + len(batch_size)
        return super().set_container(container)

    @property
    def unsqueeze_dim(self):
        if self._unsqueeze_dim is None:
            return self._unsqueeze_dim_orig
        return self._unsqueeze_dim

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._unsqueeze_dim_orig >= 0:
            self._unsqueeze_dim = self._unsqueeze_dim_orig + tensordict.ndimension()
        return super().forward(tensordict)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._unsqueeze_dim_orig >= 0:
            self._unsqueeze_dim = self._unsqueeze_dim_orig + tensordict.ndimension()
        return super()._step(tensordict)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.unsqueeze(self.unsqueeze_dim)
        return observation

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._unsqueeze_dim_orig >= 0:
            self._unsqueeze_dim = self._unsqueeze_dim_orig + tensordict.ndimension()
        return super().inv(tensordict)

    def _inv_apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.squeeze(self.unsqueeze_dim)
        return observation

    def _transform_spec(self, spec: TensorSpec) -> None:
        if isinstance(spec, CompositeSpec):
            for key in spec:
                self._transform_spec(spec[key])
        else:
            self._unsqueeze_dim = self._unsqueeze_dim_orig
            space = spec.space
            if isinstance(space, ContinuousBox):
                space.minimum = self._apply_transform(space.minimum)
                space.maximum = self._apply_transform(space.maximum)
                spec.shape = space.minimum.shape
            else:
                spec.shape = self._apply_transform(torch.zeros(spec.shape)).shape
        return spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        for key in self.in_keys_inv:
            input_spec = self._transform_spec(input_spec[key])
        return input_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if "reward" in self.in_keys:
            reward_spec = self._transform_spec(reward_spec)
        return reward_spec

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._transform_spec(observation_spec)
        return observation_spec

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_keys={self.out_keys},"
            f" in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
        return s


class SqueezeTransform(UnsqueezeTransform):
    """Removes a dimension of size one at the specified position.

    Args:
        squeeze_dim (int): dimension to squeeze.
    """

    invertible = True
    inplace = False

    def __init__(
        self,
        squeeze_dim: int,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        in_keys_inv: Optional[Sequence[str]] = None,
        out_keys_inv: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            unsqueeze_dim=squeeze_dim,
            in_keys=in_keys_inv,
            out_keys=out_keys,
            in_keys_inv=in_keys,
            out_keys_inv=out_keys_inv,
        )

    @property
    def squeeze_dim(self):
        return super().unsqueeze_dim

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super().inv(tensordict)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # placeholder for when we'll move to 'next' indexing for steps
        # return super().inv(tensordict["next"])
        return super().inv(tensordict)

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super().forward(tensordict)


class GrayScale(ObservationTransform):
    """Turns a pixel observation to grayscale."""

    inplace = False

    def __init__(self, in_keys: Optional[Sequence[str]] = None):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        super(GrayScale, self).__init__(in_keys=in_keys)

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
        standard_normal (bool, optional): if True, the transform will be

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

    The normalisation stats can be automatically computed:
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

    inplace = True

    def __init__(
        self,
        loc: Optional[float, torch.Tensor] = None,
        scale: Optional[float, torch.Tensor] = None,
        in_keys: Optional[Sequence[str]] = None,
        # observation_spec_key: =None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            in_keys = [
                "observation",
                "pixels",
                "observation_state",
            ]
        super().__init__(in_keys=in_keys)
        self.standard_normal = standard_normal
        self.eps = 1e-6

        if loc is not None and not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float)

        if scale is not None and not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
            scale = scale.clamp_min(self.eps)

        # self.observation_spec_key = observation_spec_key
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def init_stats(
        self,
        num_iter: int,
        reduce_dim: Union[int, Tuple[int]] = 0,
        cat_dim: Optional[int] = None,
        key: Optional[str] = None,
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
            key (str, optional): if provided, the summary statistics will be
                retrieved from that key in the resulting tensordicts.
                Otherwise, the first key in :obj:`ObservationNorm.in_keys` will be used.

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
        if self.loc is not None or self.scale is not None:
            raise RuntimeError(
                f"Loc/Scale are already initialized: ({self.loc}, {self.scale})"
            )

        if len(self.in_keys) > 1 and key is None:
            raise RuntimeError(
                "Transform has multiple in_keys but no specific key was passed as an argument"
            )
        key = self.in_keys[0] if key is None else key

        def raise_initialization_exception(module):
            if (
                isinstance(module, ObservationNorm)
                and module.scale is None
                and module.loc is None
            ):
                raise RuntimeError(
                    "ObservationNorms need to be initialized in the right order."
                    "Trying to initialize an ObservationNorm "
                    "while a parent ObservationNorm transform is still uninitialized"
                )

        parent = self.parent
        parent.apply(raise_initialization_exception)

        collected_frames = 0
        data = []
        while collected_frames < num_iter:
            tensordict = parent.rollout(max_steps=num_iter)
            collected_frames += tensordict.numel()
            data.append(tensordict.get(key))

        data = torch.cat(data, cat_dim)
        loc = data.mean(reduce_dim)
        scale = data.std(reduce_dim)

        if not self.standard_normal:
            loc = loc / scale
            scale = 1 / scale

        if not torch.isfinite(loc).all():
            raise RuntimeError("Non-finite values found in loc")
        if not torch.isfinite(scale).all():
            raise RuntimeError("Non-finite values found in scale")

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale.clamp_min(self.eps))

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if self.loc is None or self.scale is None:
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

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply_transform(space.minimum)
            space.maximum = self._apply_transform(space.maximum)
        return observation_spec

    def __repr__(self) -> str:
        if self.loc.numel() == 1 and self.scale.numel() == 1:
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

    CatFrames is a stateful class and it can be reset to its native state by
    calling the `reset()` method.

    Args:
        N (int, optional): number of observation to concatenate.
            Default is `4`.
        cat_dim (int, optional): dimension along which concatenate the
            observations. Default is `cat_dim=-3`.
        in_keys (list of int, optional): keys pointing to the frames that have
            to be concatenated.

    """

    inplace = False

    def __init__(
        self,
        N: int = 4,
        cat_dim: int = -3,
        in_keys: Optional[Sequence[str]] = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        super().__init__(in_keys=in_keys)
        self.N = N
        self.cat_dim = cat_dim
        self.buffer = []

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.buffer = []
        return tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = torch.cat([space.minimum] * self.N, self.cat_dim)
            space.maximum = torch.cat([space.maximum] * self.N, self.cat_dim)
            observation_spec.shape = space.minimum.shape
        else:
            shape = list(observation_spec.shape)
            shape[self.cat_dim] = self.N * shape[self.cat_dim]
            observation_spec.shape = torch.Size(shape)
        return observation_spec

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        self.buffer.append(obs)
        self.buffer = self.buffer[-self.N :]
        buffer = list(reversed(self.buffer))
        buffer = [buffer[0]] * (self.N - len(buffer)) + buffer
        if len(buffer) != self.N:
            raise RuntimeError(
                f"actual buffer length ({buffer}) differs from expected (" f"{self.N})"
            )
        return torch.cat(buffer, self.cat_dim)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={self.N}, cat_dim"
            f"={self.cat_dim}, keys={self.in_keys})"
        )


class RewardScaling(Transform):
    """Affine transform of the reward.

     The reward is transformed according to:

    .. math::
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
        standard_normal (bool, optional): if True, the transform will be

            .. math::
                reward = (reward-loc)/scale

            as it is done for standardization. Default is `False`.
    """

    inplace = True

    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor],
        in_keys: Optional[Sequence[str]] = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        super().__init__(in_keys=in_keys)
        self.standard_normal = standard_normal

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

    inplace = False

    def __init__(self):
        super().__init__(in_keys=[])

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.apply(check_finite)
        return tensordict


class DoubleToFloat(Transform):
    """Maps actions float to double before they are called on the environment.

    Examples:
        >>> td = TensorDict(
        ...     {'obs': torch.ones(1, dtype=torch.double)}, [])
        >>> transform = DoubleToFloat(in_keys=["obs"])
        >>> _ = transform(td)
        >>> print(td.get("obs").dtype)
        torch.float32

    """

    invertible = True
    inplace = False

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        in_keys_inv: Optional[Sequence[str]] = None,
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
        for key in self.in_keys_inv:
            if input_spec[key].dtype is not torch.double:
                raise TypeError(
                    f"input_spec[{key}].dtype is not double: {input_spec[key].dtype}"
                )
            self._transform_spec(input_spec[key])
        return input_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if "reward" in self.in_keys:
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
        in_keys (Sequence of str): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behaviour will only work if a parent is set.
        out_key: key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is -1.
        del_keys (bool, optional): if True, the input values will be deleted after
            concatenation. Default is True.
        unsqueeze_if_oor (bool, optional): if True, CatTensor will check that
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
    inplace = False

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        out_key: str = "observation_vector",
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
            in_keys = sorted(in_keys)
        if type(out_key) != str:
            raise Exception("CatTensors requires out_key to be of type string")
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
        return sorted(in_keys)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        if all([key in tensordict.keys(include_nested=True) for key in self.in_keys]):
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
                f" {sorted(self.in_keys)} but got a TensorDict with keys"
                f" {sorted(tensordict.keys(include_nested=True))}"
            )
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, CompositeSpec) and len(
            [key for key in self.in_keys if key not in observation_spec]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, CompositeSpec):
            # by def, there must be only one key
            return observation_spec

        keys = [key for key in observation_spec._specs.keys() if key in self.in_keys]

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
        observation_spec[out_key] = NdUnboundedContinuousTensorSpec(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        if self._del_keys:
            for key in self.keys_to_exclude:
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
    maximum action index M (with M < N), transforms the action such that
    action_out is at most M.

    If the input action is > M, it is being replaced by a random value
    between N and M. Otherwise the same action is kept.
    This is intended to be used with policies applied over multiple discrete
    control environments with different action space.

    Args:
        max_n (int): max number of action considered.
        m (int): resulting number of actions.

    Examples:
        >>> torch.manual_seed(0)
        >>> N = 2
        >>> M = 1
        >>> action = torch.zeros(N, dtype=torch.long)
        >>> action[-1] = 1
        >>> td = TensorDict({"action": action}, [])
        >>> transform = DiscreteActionProjection(N, M)
        >>> _ = transform.inv(td)
        >>> print(td.get("action"))
        tensor([1])
    """

    inplace = False

    def __init__(self, max_n: int, m: int, action_key: str = "action"):
        super().__init__([action_key])
        self.max_n = max_n
        self.m = m

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < self.m:
            raise RuntimeError(
                f"action.shape[-1]={action.shape[-1]} is smaller than "
                f"DiscreteActionProjection.M={self.m}"
            )
        action = action.argmax(-1)  # bool to int
        idx = action >= self.m
        if idx.any():
            action[idx] = torch.randint(self.m, (idx.sum(),))
        action = nn.functional.one_hot(action, self.m)
        return action

    def tranform_input_spec(self, input_spec: CompositeSpec):
        input_spec["action"] = self.transform_action_spec(input_spec["action"])
        return input_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_N={self.max_n}, M={self.m}, "
            f"keys={self.in_keys})"
        )


class FrameSkipTransform(Transform):
    """A frame-skip transform.

    This transform applies the same action repeatedly in the parent environment,
    which improves stability on certain training algorithms.

    Args:
        frame_skip (int, optional): a positive integer representing the number
            of frames during which the same action must be applied.

    """

    inplace = False

    def __init__(self, frame_skip: int = 1):
        super().__init__([])
        if frame_skip < 1:
            raise ValueError("frame_skip should have a value greater or equal to one.")
        self.frame_skip = frame_skip

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        parent = self.parent
        reward = tensordict.get("reward")
        for _ in range(self.frame_skip - 1):
            tensordict = parent._step(tensordict)
            reward = reward + tensordict.get("reward")
        tensordict.set("reward", reward)
        return tensordict


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

    inplace = True

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
        if tensordict.get("done").numel() > 1:
            raise ValueError(
                "there is more than one done state in the parent environment. "
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
                if tensordict.get("done"):
                    tensordict = parent.reset(td_reset.clone(False))
                    break
            else:
                break

            trial += 1
            if trial > _MAX_NOOPS_TRIALS:
                tensordict = parent.rand_step(tensordict)
                if tensordict.get("done"):
                    raise RuntimeError(
                        f"parent is still done after a single random step (i={i})."
                    )
                break

        if tensordict.get("done"):
            raise RuntimeError("NoopResetEnv concluded with done environment")
        return tensordict

    def __repr__(self) -> str:
        random = self.random
        noops = self.noops
        class_name = self.__class__.__name__
        return f"{class_name}(noops={noops}, random={random})"


class TensorDictPrimer(Transform):
    """A primer for TensorDict initialization at reset time.

    This transform will populate the tensordict at reset with values drawn from
    the relative tensorspecs provided at initialization.

    Args:
        random (bool, optional): if True, the values will be drawn randomly from
            the TensorSpec domain. Otherwise a fixed value will be assumed.
            Defaults to `False`.
        default_value (float, optional): if non-random filling is chosen, this
            value will be used to populate the tensors. Defaults to `0.0`.
        **kwargs: each keyword argument corresponds to a key in the tensordict.
            The corresponding value has to be a TensorSpec instance indicating
            what the value must be.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env)
        >>> env.append_transform(TensorDictPrimer(mykey=NdUnboundedContinuousTensorSpec([3])))
        >>> print(env.reset())
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                mykey: Tensor(torch.Size([3]), dtype=torch.float32),
                observation: Tensor(torch.Size([3]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
    """

    inplace = False

    def __init__(self, random=False, default_value=0.0, **kwargs):
        self.primers = kwargs
        self.random = random
        self.default_value = default_value
        self.device = kwargs.get("device", torch.device("cpu"))
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
        return self._device

    @device.setter
    def device(self, value):
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
            # deprecating this with the new "next_" logic where we expect keys to collide
            # if key in observation_spec:
            #     raise RuntimeError(
            #         f"The key {key} is already in the observation_spec. This means "
            #         f"that the value reset by TensorDictPrimer will confict with the "
            #         f"value obtained through the call to `env.reset()`. Consider renaming "
            #         f"the {key} key."
            #     )
            observation_spec[key] = spec.to(self.device)
        return observation_spec

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        super().set_container(container)

    @property
    def _batch_size(self):
        return self.parent.batch_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        for key, spec in self.primers.items():
            if self.random:
                value = spec.rand(tensordict.batch_size)
            else:
                value = torch.full_like(
                    spec.rand(tensordict.batch_size),
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


def _sum_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.sum(0)
    return val


class gSDENoise(Transform):
    """A gSDE noise initializer.

    See the :func:`~torchrl.modules.models.exploration.gSDEModule' for more info.
    """

    inplace = False

    def __init__(
        self,
        state_dim=None,
        action_dim=None,
    ) -> None:
        super().__init__(in_keys=[])
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = super().reset(tensordict=tensordict)
        if self.state_dim is None or self.action_dim is None:
            tensordict.set(
                "_eps_gSDE",
                torch.zeros(
                    *tensordict.batch_size,
                    1,
                    device=tensordict.device,
                ),
            )
        else:
            tensordict.set(
                "_eps_gSDE",
                torch.randn(
                    *tensordict.batch_size,
                    self.action_dim,
                    self.state_dim,
                    device=tensordict.device,
                ),
            )

        return tensordict


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
        in_keys (iterable of str, optional): keys to be updated.
            default: ["observation", "reward"]
        shared_td (TensorDictBase, optional): A shared tensordict containing the
            keys of the transform.
        decay (number, optional): decay rate of the moving average.
            default: 0.99
        eps (number, optional): lower bound of the running standard
            deviation (for numerical underflow). Default is 1e-4.

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

    inplace = True

    def __init__(
        self,
        in_keys: Optional[Sequence[str]] = None,
        shared_td: Optional[TensorDictBase] = None,
        lock: mp.Lock = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
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
        self.eps = eps

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

            tensordict.set_(key, new_val)

        if self.lock is not None:
            self.lock.release()

        return tensordict

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        if self._td is None or key + "_sum" not in self._td.keys():
            td_view = tensordict.view(-1)
            td_select = td_view[0]
            d = {key + "_sum": torch.zeros_like(td_select.get(key))}
            d.update({key + "_ssq": torch.zeros_like(td_select.get(key))})
            d.update(
                {
                    key
                    + "_count": torch.zeros(
                        1, device=td_select.get(key).device, dtype=torch.float
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
        _sum = self._td.get(key + "_sum")
        _ssq = self._td.get(key + "_ssq")
        _count = self._td.get(key + "_count")

        _sum = self._td.get(key + "_sum")
        value_sum = _sum_left(value, _sum)
        _sum *= self.decay
        _sum += value_sum
        self._td.set_(key + "_sum", _sum, no_check=True)

        _ssq = self._td.get(key + "_ssq")
        value_ssq = _sum_left(value.pow(2), _ssq)
        _ssq *= self.decay
        _ssq += value_ssq
        self._td.set_(key + "_ssq", _ssq, no_check=True)

        _count = self._td.get(key + "_count")
        _count *= self.decay
        _count += N
        self._td.set_(key + "_count", _count, no_check=True)

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
            keys (iterable of str, optional): keys that
                have to be normalized. Default is `["next", "reward"]`
            memmap (bool): if True, the resulting tensordict will be cast into
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
            td_select.rename_key(key, key + "_sum")
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
