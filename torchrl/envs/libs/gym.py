# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from types import ModuleType
from typing import List, Dict
from warnings import warn

import torch
from packaging import version

from torchrl.data import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from ...data.utils import numpy_to_torch_dtype_dict
from ..gym_like import GymLikeEnv, default_info_dict_reader
from ..utils import _classproperty

try:
    import gym

    _has_gym = True

except ImportError:
    _has_gym = False


if _has_gym:
    try:
        from gym.wrappers.pixel_observation import PixelObservationWrapper

        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as LegacyPixelObservationWrapper,
        )
    except ModuleNotFoundError:
        warnings.warn(
            f"gym {gym.__version__} does not provide the PixelObservationWrapper"
            f"used by torchrl, which will be using a patched version. "
            f"Consider updating gym to a newer version."
        )
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )
    gym_version = version.parse(gym.__version__)
    if gym_version >= version.parse("0.26.0"):
        from gym.wrappers.compatibility import EnvCompatibility

__all__ = ["GymWrapper", "GymEnv"]


def _gym_to_torchrl_spec_transform(
    spec, dtype=None, device="cpu", categorical_action_encoding=False
) -> TensorSpec:
    if isinstance(spec, gym.spaces.tuple.Tuple):
        raise NotImplementedError("gym.spaces.tuple.Tuple mapping not yet implemented")
    if isinstance(spec, gym.spaces.discrete.Discrete):
        action_space_cls = (
            DiscreteTensorSpec
            if categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        return action_space_cls(spec.n, device=device)
    elif isinstance(spec, gym.spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(spec.n, device=device)
    elif isinstance(spec, gym.spaces.multi_discrete.MultiDiscrete):
        return MultOneHotDiscreteTensorSpec(spec.nvec, device=device)
    elif isinstance(spec, gym.spaces.Box):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            torch.tensor(spec.low, device=device, dtype=dtype),
            torch.tensor(spec.high, device=device, dtype=dtype),
            torch.Size(spec.shape),
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, (Dict,)):
        spec_out = {}
        for k in spec.keys():
            spec_out["next_" + k] = _gym_to_torchrl_spec_transform(
                spec[k],
                device=device,
                categorical_action_encoding=categorical_action_encoding,
            )
        return CompositeSpec(**spec_out)
    elif isinstance(spec, gym.spaces.dict.Dict):
        return _gym_to_torchrl_spec_transform(
            spec.spaces,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
        )
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> List:
    if gym_version < version.parse("0.26.0"):
        envs = gym.envs.registration.registry.env_specs.keys()
    else:
        envs = gym.envs.registration.registry.keys()

    envs = list(envs)
    envs = sorted(envs)
    return envs


def _get_gym():
    if _has_gym:
        return gym
    else:
        return None


def _is_from_pixels(env):
    observation_spec = env.observation_space
    if isinstance(observation_spec, (Dict,)):
        if "pixels" in set(observation_spec.keys()):
            return True
    if isinstance(observation_spec, (gym.spaces.dict.Dict,)):
        if "pixels" in set(observation_spec.spaces.keys()):
            return True
    elif (
        isinstance(observation_spec, gym.spaces.Box)
        and (observation_spec.low == 0).all()
        and (observation_spec.high == 255).all()
        and observation_spec.low.shape[-1] == 3
        and observation_spec.low.ndim == 3
    ):
        return True
    elif isinstance(env, PixelObservationWrapper):
        return True
    return False


class GymWrapper(GymLikeEnv):
    """OpenAI Gym environment wrapper.

    Examples:
        >>> env = gym.make("Pendulum-v0")
        >>> env = GymWrapper(env)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/openai/gym"
    libname = "gym"

    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not (hasattr(env, "action_space") and hasattr(env, "observation_space")):
            raise TypeError("env is not of type 'gym.Env'.")

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> "gym.core.Env":
        env_from_pixels = _is_from_pixels(env)
        from_pixels = from_pixels or env_from_pixels
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        if from_pixels and not env_from_pixels:
            if isinstance(env, PixelObservationWrapper):
                raise TypeError(
                    "PixelObservationWrapper cannot be used to wrap an environment"
                    "that is already a PixelObservationWrapper instance."
                )
            if not env.render_mode and gym_version >= version.parse("0.26.0"):
                warnings.warn(
                    "Environments provided to GymWrapper that need to be wrapped in PixelObservationWrapper "
                    "should be created with `gym.make(env_name, render_mode=mode)` where possible,"
                    'where mode is either "rgb_array" or any other supported mode.'
                )
                # resetting as 0.26 comes with a very 'nice' OrderEnforcing wrapper
                env = EnvCompatibility(env)
                env.reset()
                env = LegacyPixelObservationWrapper(env, pixels_only=pixels_only)
            else:
                env = PixelObservationWrapper(env, pixels_only=pixels_only)
        return env

    @_classproperty
    def available_envs(cls) -> List[str]:
        return _get_envs()

    @property
    def lib(self) -> ModuleType:
        return gym

    def _set_seed(self, seed: int) -> int:
        skip = False
        if self._seed_calls_reset is None:
            if gym_version < version.parse("0.19.0"):
                self._seed_calls_reset = False
                self._env.seed(seed=seed)
            else:
                try:
                    self.reset(seed=seed)
                    skip = True
                    self._seed_calls_reset = True
                except TypeError as err:
                    warnings.warn(
                        f"reset with seed kwarg returned an exception: {err}.\n"
                        f"Calling env.seed from now on."
                    )
                    self._seed_calls_reset = False
        if self._seed_calls_reset and not skip:
            self.reset(seed=seed)
        elif not self._seed_calls_reset:
            self._env.seed(seed=seed)
        return seed

    def _make_specs(self, env: "gym.Env") -> None:
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        self.observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if not isinstance(self.observation_spec, CompositeSpec):
            if self.from_pixels:
                self.observation_spec = CompositeSpec(next_pixels=self.observation_spec)
            else:
                self.observation_spec = CompositeSpec(
                    next_observation=self.observation_spec
                )
        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )

    def _init_env(self):
        self.reset()  # make sure that _is_done is populated

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    def rebuild_with_kwargs(self, **new_kwargs):
        self._constructor_kwargs.update(new_kwargs)
        self._env = self._build_env(**self._constructor_kwargs)
        self._make_specs(self._env)

    @property
    def info_dict_reader(self):
        if self._info_dict_reader is None:
            self._info_dict_reader = default_info_dict_reader()
        return self._info_dict_reader

    @info_dict_reader.setter
    def info_dict_reader(self, value: callable):
        self._info_dict_reader = value


ACCEPTED_TYPE_ERRORS = {
    "render_mode": "__init__() got an unexpected keyword argument 'render_mode'",
    "frame_skip": "unexpected keyword argument 'frameskip'",
}


class GymEnv(GymWrapper):
    """OpenAI Gym environment wrapper.

    Examples:
        >>> env = GymEnv(env_name="Pendulum-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    def __init__(self, env_name, disable_env_checker=True, **kwargs):
        kwargs["env_name"] = env_name
        kwargs["disable_env_checker"] = disable_env_checker
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "gym.core.Env":
        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        from_pixels = kwargs.get("from_pixels", False)
        if from_pixels and gym_version > version.parse("0.25.0"):
            kwargs.setdefault("render_mode", "rgb_array")
        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]
        pixels_only = kwargs.get("pixels_only", True)
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]
        made_env = False
        kwargs["frameskip"] = self.frame_skip
        self.wrapper_frame_skip = 1
        while not made_env:
            # env.__init__ may not be compatible with all the kwargs that
            # have been preset. We iterate through the various solutions
            # to find the config that works.
            try:
                with warnings.catch_warnings(record=True) as w:
                    # we catch warnings as they may cause silent bugs
                    env = self.lib.make(env_name, **kwargs)
                    if len(w) and "frameskip" in str(w[-1].message):
                        raise TypeError("unexpected keyword argument 'frameskip'")
                made_env = True
            except TypeError as err:
                if ACCEPTED_TYPE_ERRORS["frame_skip"] in str(err):
                    warn(
                        "Discarding frameskip arg. This will be taken care of by TorchRL env wrapper."
                    )
                    self.wrapper_frame_skip = kwargs.pop("frameskip")
                elif ACCEPTED_TYPE_ERRORS["render_mode"] in str(err):
                    warn("Discarding render_mode from the env constructor.")
                    kwargs.pop("render_mode")
                else:
                    raise err
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"
