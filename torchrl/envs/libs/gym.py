from types import ModuleType
from typing import List, Iterable, Optional

import torch

from torchrl.data import (
    TensorSpec,
    OneHotDiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec,
)
from ..common import GymLikeEnv
from ...data.utils import numpy_to_torch_dtype_dict

try:
    import gym
except ImportError:
    _has_gym = False
else:
    _has_gym = True
    from gym.wrappers.pixel_observation import PixelObservationWrapper
try:
    import retro
except ImportError:
    _has_retro = False
else:
    _has_retro = True

__all__ = ["GymEnv", "RetroEnv"]


def _gym_to_torchrl_spec_transform(spec, dtype=None, device="cpu") -> TensorSpec:
    if isinstance(spec, gym.spaces.tuple.Tuple):
        raise NotImplementedError(f"gym.spaces.tuple.Tuple mapping not yet implemented")
    if isinstance(spec, gym.spaces.discrete.Discrete):
        return OneHotDiscreteTensorSpec(spec.n)
    elif isinstance(spec, gym.spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(spec.n)
    elif isinstance(spec, gym.spaces.multi_discrete.MultiDiscrete):
        return MultOneHotDiscreteTensorSpec(spec.nvec)
    elif isinstance(spec, gym.spaces.Box):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            torch.tensor(spec.low, device=device, dtype=dtype),
            torch.tensor(spec.high, device=device, dtype=dtype),
            torch.Size(spec.shape),
            dtype=dtype,
        )
    elif isinstance(spec, (dict, gym.spaces.dict.Dict)):
        spec = {k: _gym_to_torchrl_spec_transform(spec[k]) for k in spec}
        return CompositeSpec(**spec)
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> List:
    envs = gym.envs.registration.registry.env_specs.keys()
    envs = list(envs)
    envs = sorted(envs)
    return envs


def _get_gym():
    if _has_gym:
        return gym
    else:
        return None


def _is_from_pixels(observation_space):
    return (
        isinstance(observation_space, gym.spaces.Box)
        and (observation_space.low == 0).all()
        and (observation_space.high == 255).all()
        and observation_space.low.shape[-1] == 3
        and observation_space.low.ndim == 3
    )


class GymEnv(GymLikeEnv):
    """
    OpenAI Gym environment wrapper.

    Examples:
        >>> env = GymEnv(envname="Pendulum-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)
    """

    git_url = "https://github.com/openai/gym"
    libname = "gym"

    @property
    def available_envs(self) -> List:
        return _get_envs()

    @property
    def lib(self) -> ModuleType:
        return gym

    def _set_seed(self, seed: int) -> int:
        self._env.seed(seed)
        return seed

    def _build_env(
        self,
        envname: str,
        taskname: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> gym.core.Env:
        self.pixels_only = pixels_only
        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {envname}. "
                f"Consider downloading and installing dm_control from {self.git_url}"
            )
        if not ((taskname == "") or (taskname is None)):
            raise ValueError(
                f"gym does not support taskname, received {taskname} instead."
            )
        try:
            env = self.lib.make(envname, frameskip=self.frame_skip)
            self.wrapper_frame_skip = 1
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise TypeError(err)
            env = self.lib.make(envname)
            self.wrapper_frame_skip = self.frame_skip
        self._env = env

        from_pixels = from_pixels or _is_from_pixels(self._env.observation_space)
        self.from_pixels = from_pixels
        if from_pixels:
            self._env.reset()
            self._env = PixelObservationWrapper(self._env, pixels_only)

        self.action_spec = _gym_to_torchrl_spec_transform(self._env.action_space)
        self.observation_spec = _gym_to_torchrl_spec_transform(
            self._env.observation_space
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )  # default

    def _init_env(self, seed: Optional[int] = None) -> Optional[int]:
        if seed is not None:
            seed = self.set_seed(seed)
        self.reset()  # make sure that _current_observation and _is_done are populated
        return seed


def _get_retro_envs() -> Iterable:
    if not _has_retro:
        return tuple()
    else:
        return retro.data.list_games()


def _get_retro() -> Optional[ModuleType]:
    if _has_retro:
        return retro
    else:
        return None


class RetroEnv(GymEnv):
    available_envs = _get_retro_envs()
    lib = "retro"
    lib = _get_retro()
