from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torchvision.transforms.functional_tensor import (
    resize,
)  # as of now resize is imported from torchvision

from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec, ContinuousBox, \
    NdUnboundedContinuousTensorSpec
from torchrl.data.tensordict.tensordict import TensorDict, _TensorDict
from torchrl.envs.common import _EnvClass
from . import functional as F
from .utils import FiniteTensor

__all__ = [
    "TransformedEnv",
    "RewardClipping",
    "Resize",
    "GrayScale",
    "Compose",
    "ToTensorImage",
    "ObservationNorm",
    "DataDependentObservationNorm",
    "RewardScaling",
    "ObservationTransform",
    "Transform",
    "CatFrames",
    "FiniteTensorDictCheck",
    "DoubleToFloat",
    "CatTensors",
    "NoopResetEnv",
]

from ...envs.utils import step_tensor_dict

IMAGE_KEYS = ["next_observation", "next_observation_pixels"]


class Transform(nn.Module):
    invertible = False

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        return tensordict

    def _check_inplace(self):
        if not hasattr(self, "inplace"):
            raise AttributeError(f"Transform of class {self.__class__.__name__} has no attribute inplace, consider "
                                 f"implementing it.")

    def init(self, tensordict):
        pass

    def _apply(self, obs):
        raise NotImplementedError

    def _call(self, tensordict):
        self._check_inplace()
        for _obs_key in tensordict.keys():
            if _obs_key in self.keys:
                observation = self._apply(tensordict.get(_obs_key))
                tensordict.set(_obs_key, observation, inplace=self.inplace)
        return tensordict

    def forward(self, tensordict):
        self._call(tensordict)
        return tensordict

    def _inv_apply(self, obs):
        if self.invertible:
            raise NotImplementedError
        else:
            return obs

    def _inv_call(self, tensordict):
        self._check_inplace()
        for _obs_key in tensordict.keys():
            if _obs_key in self.keys:
                observation = self._inv_apply(tensordict.get(_obs_key))
                tensordict.set(_obs_key, observation, inplace=self.inplace)
        return tensordict

    def inv(self, tensordict):
        self._inv_call(tensordict)
        return tensordict

    def transform_action_spec(self, action_spec):
        return action_spec

    def transform_observation_spec(self, observation_spec):
        return observation_spec

    def transform_reward_spec(self, reward_spec):
        return reward_spec

    def dump(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TransformedEnv(_EnvClass):
    def __init__(self, env: _EnvClass, transform: Transform, cache_specs: bool = True, **kwargs):
        self.env = env
        self.transform = transform
        self._last_obs = None
        self.cache_specs = cache_specs

        self._action_spec = None
        self._reward_spec = None
        self._observation_spec = None
        self.batch_size = self.env.batch_size

        super().__init__(**kwargs)

    @property
    def observation_spec(self):
        if self._observation_spec is None or not self.cache_specs:
            observation_spec = self.transform.transform_observation_spec(
                deepcopy(self.env.observation_spec)
            )
            if self.cache_specs:
                self._observation_spec = observation_spec
        else:
            observation_spec = self._observation_spec
        return observation_spec

    @property
    def action_spec(self):
        if self._action_spec is None or not self.cache_specs:
            action_spec = self.transform.transform_action_spec(deepcopy(self.env.action_spec))
            if self.cache_specs:
                self._action_spec = action_spec
        else:
            action_spec = self._action_spec
        return action_spec

    @property
    def reward_spec(self):
        if self._reward_spec is None or not self.cache_specs:
            reward_spec = self.transform.transform_reward_spec(deepcopy(self.env.reward_spec))
            if self.cache_specs:
                self._reward_spec = reward_spec
        else:
            reward_spec = self._reward_spec
        return reward_spec

    def _step(self, tensordict: TensorDict) -> _TensorDict:
        selected_keys = [key for key in tensordict.keys() if "action" in key]
        tensordict_in = tensordict.select(*selected_keys)
        tensordict_in = self.transform.inv(tensordict_in)
        tensordict_out = self.env._step(tensordict_in).to(self.device)
        # tensordict should already have been processed by the transforms
        # for logging purposes
        tensordict_out = self.transform(tensordict_out.clone())
        return tensordict_out

    def set_seed(self, seed=None):
        if seed is not None:
            self.env.set_seed(seed)

    def _reset(self, tensordict: Optional[_TensorDict] = None):
        out_tensordict = self.env.reset().to(self.device)
        out_tensordict = self.transform.reset(out_tensordict)

        # Transforms are made for "next_observations" and alike. We convert all the observations in next_observations,
        # then map them back to their original key name
        keys = list(out_tensordict.keys())
        for key in keys:
            if key.startswith("observation"):
                out_tensordict.rename_key(key, "next_" + key, safe=True)

        out_tensordict = self.transform(out_tensordict)
        keys = list(out_tensordict.keys())
        for key in keys:
            if key.startswith("next_observation"):
                out_tensordict.rename_key(key, key[5:], safe=True)
        return out_tensordict

    def __getattr__(self, attr):
        if attr in self.__dir__():
            return self.__getattribute__(attr)  # make sure that appropriate exceptions are raised

        try:
            env = self.__getattribute__("env")
        except:
            raise Exception(
                f"env not set in {self.__class__.__name__}, cannot access {attr}"
            )
        return getattr(env, attr)

    def __repr__(self):
        return f"TransformedEnv(env={self.env}, transform={self.transform})"


class ObservationTransform(Transform):
    inplace = False

    def __init__(self, keys=None):
        if keys is None:
            keys = [
                "next_observation",
                "next_observation_pixels",
                "next_observation_state"
            ]
        super(ObservationTransform, self).__init__(keys=keys)


class Compose(Transform):
    inplace = False

    def __init__(self, *transforms):
        super().__init__(keys=[])
        self.transforms = transforms

    def _call(self, tensordict):
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def transform_action_spec(self, action_spec):
        for t in self.transforms:
            action_spec = t.transform_action_spec(action_spec)
        return action_spec

    def transform_observation_spec(self, observation_spec):
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
        return observation_spec

    def transform_reward_spec(self, reward_spec):
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
        return reward_spec

    def __getitem__(self, item):
        return self.transforms[item]

    def dump(self):
        for t in self:
            t.dump()

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        for t in self.transforms:
            tensordict = t.reset(tensordict)
        return tensordict

    def init(self, tensordict):
        for t in self.transforms:
            t.init(tensordict)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([str(l) for l in self.transforms])})"


class ToTensorImage(ObservationTransform):
    inplace = False

    def __init__(self, unsqueeze=False, dtype=None, keys=None):
        if keys is None:
            keys = IMAGE_KEYS  # default
        super().__init__(keys=keys)
        self.unsqueeze = unsqueeze
        self.dtype = dtype

    def _apply(self, observation: torch.FloatTensor):
        observation = observation.div(255)
        observation = observation.permute(
            *list(range(observation.ndimension() - 3)), -1, -3, -2
        )
        if observation.ndimension() == 3 and self.unsqueeze:
            observation = observation.unsqueeze(0)
        return observation

    def transform_observation_spec(self, observation_spec):
        _observation_spec = observation_spec["pixels"]
        self._pixel_observation(_observation_spec)
        _observation_spec.shape = torch.Size(
            [
                *_observation_spec.shape[:-3],
                _observation_spec.shape[-1],
                _observation_spec.shape[-3],
                _observation_spec.shape[-2],
            ]
        )
        _observation_spec.dtype = self.dtype if self.dtype is not None else torch.float32
        observation_spec["pixels"] = _observation_spec
        return observation_spec

    def _pixel_observation(self, spec):
        if isinstance(spec, BoundedTensorSpec):
            spec.space.maximum = self._apply(spec.space.maximum)
            spec.space.minimum = self._apply(spec.space.minimum)


class RewardClipping(Transform):
    inplace = True

    def __init__(self, clamp_min=None, clamp_max=None, keys=None):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _apply(self, reward):
        if self.clamp_max is not None and self.clamp_min is not None:
            reward = reward.clamp_(self.clamp_min, self.clamp_max)
        elif self.clamp_min is not None:
            reward = reward.clamp_min_(self.clamp_min)
        elif self.clamp_max is not None:
            reward = reward.clamp_max_(self.clamp_max)
        return reward

    def transform_reward_spec(self, reward_spec):
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return BoundedTensorSpec(self.clamp_min, self.clamp_max, device=reward_spec.device, dtype=reward_spec.dtype)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not implemented for tensor spec of type {type(reward_spec).__name__}"
            )


class Resize(ObservationTransform):
    inplace = False

    def __init__(self, w, h, interpolation="bilinear", keys=None):
        if keys is None:
            keys = IMAGE_KEYS  # default
        super().__init__(keys=keys)
        self.w = w
        self.h = h
        self.interpolation = interpolation

    def _apply(self, observation):
        observation = resize(
            observation, (self.w, self.h), interpolation=self.interpolation
        )

        return observation

    def transform_observation_spec(self, observation_spec):
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(torch.zeros(_observation_spec.shape)).shape
        observation_spec["pixels"] = _observation_spec
        return observation_spec


class GrayScale(ObservationTransform):
    inplace = False

    def __init__(self, keys=None):
        if keys is None:
            keys = IMAGE_KEYS
        super(GrayScale, self).__init__(keys=keys)

    def _apply(self, observation):
        observation = F.rgb_to_grayscale(observation)
        return observation

    def transform_observation_spec(self, observation_spec):
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(torch.zeros(_observation_spec.shape)).shape
        observation_spec["pixels"] = _observation_spec
        return observation_spec


class ObservationNorm(ObservationTransform):
    inplace = True

    def __init__(
            self, loc, scale, keys=None, observation_spec_key=None, standard_normal=False,
    ):
        if keys is None:
            keys = ["next_observation", "next_observation_pixels", "next_observation_state"]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)

        self.observation_spec_key = observation_spec_key
        self.standard_normal = standard_normal
        self.loc = loc
        eps = 1e-6
        self.scale = scale.clamp_min(eps)

        if self.standard_normal:
            # converts the transform (x-m)/sqrt(v) to x * s + loc
            self.scale = self.scale.reciprocal()
            self.loc = -self.loc * self.scale

    def _apply(self, obs):
        return obs * self.scale + self.loc

    def transform_observation_spec(self, observation_spec):
        if isinstance(observation_spec, CompositeSpec):
            # assert self.observation_spec_key is not None, f"Class {self.__class__.__name__} requires " \
            #                                               f"observation_spec_key to be set when observation_spec is of type Composite." \
            #                                               f"Choose one of {list(observation_spec._specs.keys())}"
            key = [key.split("observation_")[-1] for key in self.keys]
            assert len(set(key)) == 1
            key = key[0]
            _observation_spec = observation_spec[key]
        else:
            _observation_spec = observation_spec
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
        return observation_spec


class DataDependentObservationNorm(ObservationNorm):
    inplace = True

    def __init__(self, dims=(-1, -2), keys=None,
                 ):
        if keys is None:
            keys = ["next_observation", "next_observation_pixels", "next_observation_state"]
        super().__init__(keys=keys)
        self.dims = dims
        self.initialized = False

    def _init(self, obs):
        assert torch.isfinite(obs).all()
        loc = obs.mean(self.dims, True)
        scale = obs.std(self.dims, True).clamp_min(1e-6)
        loc = -loc / scale
        scale = scale.reciprocal()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)
        self.initialized = True

    def init(self, tensordict):
        for key in self.keys:
            if key in tensordict.keys():
                self._init(tensordict.get(key))
                break

    def _call(self, tensordict):
        if not self.initialized:
            return tensordict
        return super()._call(tensordict)

    def transform_observation_spec(self, observation_spec):
        if not self.initialized:
            return observation_spec
        return super().transform_observation_spec(observation_spec)


class CatFrames(ObservationTransform):
    inplace = False

    def __init__(self, N=4, cat_dim=-3, keys=None):
        if keys is None:
            keys = IMAGE_KEYS
        super().__init__(keys=keys)
        self.N = N
        self.cat_dim = cat_dim
        self.buffer = []

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        self.buffer = []
        return tensordict

    def transform_observation_spec(self, observation_spec):
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = torch.cat([space.minimum] * self.N, 0)
            space.maximum = torch.cat([space.maximum] * self.N, 0)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = torch.Size([self.N, *_observation_spec.shape])
        observation_spec["pixels"] = _observation_spec
        return observation_spec

    def _apply(self, obs: torch.Tensor):
        self.buffer.append(obs)
        self.buffer = self.buffer[-self.N:]
        buffer = list(reversed(self.buffer))
        buffer = [buffer[0]] * (self.N - len(buffer)) + buffer
        assert len(buffer) == self.N
        return torch.cat(buffer, self.cat_dim)


class RewardScaling(Transform):
    inplace = True

    def __init__(self, loc, scale, keys=None):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.loc = loc
        self.scale = scale.clamp_min(1e-6)

    def _apply(self, reward):
        reward.mul_(self.scale).add_(self.loc)
        return reward

    def transform_reward_spec(self, reward_spec):
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return reward_spec
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not implemented for tensor spec of type {type(reward_spec).__name__}"
            )


class FiniteTensorDictCheck(Transform):
    inplace = False

    def __init__(self):
        super().__init__(keys=[])

    def _call(self, tensordict):
        source = {}
        for key, item in tensordict.items():
            try:
                source[key] = FiniteTensor(item)
            except AssertionError:
                raise Exception(f"Found non-finite elements in {key}")

        finite_tensordict = TensorDict(batch_size=tensordict.batch_size, source=source)
        return finite_tensordict


class DoubleToFloat(Transform):
    invertible = True
    inplace = False

    def __init__(self, keys=None):
        if keys is None:
            keys = ["action"]
        super().__init__(keys=keys)

    def _apply(self, obs):
        return obs.to(torch.float)

    def _inv_apply(self, obs):
        return obs.to(torch.double)

    def _transform_spec(self, spec):
        spec.dtype = torch.float
        space = spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = space.minimum.to(torch.float)
            space.maximum = space.maximum.to(torch.float)

    def transform_action_spec(self, action_spec):
        if "action" in self.keys:
            assert action_spec.dtype is torch.double
            self._transform_spec(action_spec)
            return action_spec

    def transform_reward_spec(self, reward_spec):
        if "reward" in self.keys:
            assert reward_spec.dtype is torch.double
            self._transform_spec(reward_spec)
            return reward_spec

    def transform_observation_spec(self, observation_spec):
        keys = [key for key in self.keys if "observation" in key]
        if keys:
            keys = [key.split("observation_")[-1] for key in keys]
            for key in keys:
                self._transform_spec(observation_spec[key])
        return observation_spec


class CatTensors(Transform):
    invertible = False
    inplace = False

    def __init__(self, keys=None, out_key="observation_vector"):
        if keys is None:
            raise Exception("CatTensors requires keys to be non-empty")
        super().__init__(keys=keys)
        assert "observation_" in out_key, "CatTensors is currently restricted to observation_* keys"
        self.out_key = out_key
        self.keys = sorted(list(self.keys))

    def _call(self, tensordict):
        if all([key in tensordict.keys() for key in self.keys]):
            out_tensor = torch.cat([tensordict.get(key) for key in self.keys], -1)
            tensordict.set(self.out_key, out_tensor)
            for key in self.keys:
                tensordict.del_(key)
        else:
            raise Exception(f"CatTensor failed, as it expected input keys = {sorted(list(self.keys))} but got a "
                            f"TensorDict with keys {sorted(list(tensordict.keys()))}")
        return tensordict

    def transform_observation_spec(self, observation_spec):
        assert isinstance(observation_spec, CompositeSpec)
        keys = [key.split("observation_")[-1] for key in self.keys]

        if all([key in observation_spec for key in keys]):
            sum_shape = sum([
                observation_spec[key].shape[-1] if observation_spec[key].shape else 1
                for key in keys])
            spec0 = observation_spec[keys[0]]
            out_key = self.out_key.split("observation_")[-1]
            observation_spec[out_key] = NdUnboundedContinuousTensorSpec(
                shape=torch.Size([*spec0.shape[:-1], sum_shape]),
                dtype=spec0.dtype)
            for key in keys:
                observation_spec.del_(key)
        return observation_spec


class DiscreteActionProjection(Transform):
    inplace = False

    def __init__(self, n_in, n_out, action_key="action"):
        super().__init__([action_key])
        self.n_in = n_in
        self.n_out = n_out

    def _inv_apply(self, action):
        assert action.shape[-1] >= self.n_out
        action = action.argmax(-1)  # bool to int
        idx = action >= self.n_out
        if idx.any():
            action[idx] = torch.randint(self.n_out, (idx.sum(),))
        action = nn.functional.one_hot(action, self.n_out)
        return action

    def transform_action_spec(self, action_spec):
        shape = action_spec.shape
        shape = torch.Size([*shape[:-1], self.n_in])
        action_spec.shape = shape
        action_spec.space.n = self.n_in
        return action_spec


class NoopResetEnv(Transform):
    inplace = True

    def __init__(self, env: _EnvClass, noops: int = 30, random: bool = True):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__([])
        self.env = env
        self.noops = noops
        self.random = random

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        """ Do no-op action for a number of steps in [1, noop_max]."""
        noops = self.noops if not self.random else torch.randint(self.noops, (1,)).item()
        for _ in range(noops):
            tensordict = self.env.rand_step()

        return step_tensor_dict(tensordict)
