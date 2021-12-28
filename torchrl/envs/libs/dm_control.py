from typing import Tuple

import numpy as np
import torch

from torchrl.data import TensorSpec, CompositeSpec, NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
from ..common import GymLikeEnv
from ...data.utils import numpy_to_torch_dtype_dict

try:
    import dm_control
except ImportError:
    _has_dmc = False
else:
    from dm_control.suite.wrappers import pixels
    from dm_control import suite
    import dm_env
    import collections

    _has_dmc = True

__all__ = ["DMControlEnv"]


def _dmcontrol_to_torchrl_spec_transform(spec, dtype=None) -> TensorSpec:
    if isinstance(spec, collections.OrderedDict):
        spec = {k: _dmcontrol_to_torchrl_spec_transform(item) for k, item in spec.items()}
        return CompositeSpec(**spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(shape=spec.shape, minimum=spec.minimum, maximum=spec.maximum, dtype=dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdUnboundedContinuousTensorSpec(shape=spec.shape, dtype=dtype)
    else:
        raise NotImplementedError


def _get_envs(to_dict=True):
    if not _has_dmc:
        return dict()
    if not to_dict:
        return tuple(suite.BENCHMARKING) + tuple(suite.EXTRA)
    d = dict()
    for tup in suite.BENCHMARKING:
        envname = tup[0]
        d.setdefault(envname, []).append(tup[1])
    for tup in suite.EXTRA:
        envname = tup[0]
        d.setdefault(envname, []).append(tup[1])
    return d


def _robust_to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.tensor(array.copy())
    else:
        return torch.tensor(array)


class DMControlEnv(GymLikeEnv):
    git_url = "https://github.com/deepmind/dm_control"
    libname = "dm_control"
    available_envs = _get_envs()

    def _build_env(self, envname, taskname, seed=None, from_pixels=False, render_kwargs=None):
        assert (
            _has_dmc
        ), f"dm_control not found, unable to create {envname}: {taskname}. \
            Consider downloading and installing dm_control from {self.git_url}"
        self.from_pixels = from_pixels
        kwargs = dict()
        if seed is not None:
            random_state = np.random.RandomState(seed)
            kwargs = {"random": random_state}
        env = suite.load(envname, taskname, task_kwargs=kwargs)
        if from_pixels:
            self.render_kwargs = {'camera_id': 0}
            if render_kwargs is not None:
                self.render_kwargs.update(render_kwargs)
            env = pixels.Wrapper(env, pixels_only=False, render_kwargs=self.render_kwargs)
        self.env = env
        observations, *_ = self._output_transform((env.reset(),))
        self._last_obs_dict = self._read_obs(observations)
        self._is_done = torch.zeros(1, dtype=torch.bool)
        return env

    def _output_transform(self, timestep_tuple: Tuple[dm_env._environment.TimeStep]):
        if type(timestep_tuple) is not tuple:
            timestep_tuple = (timestep_tuple, )
        reward = timestep_tuple[0].reward

        done = False # dm_control envs are non-terminating
        observation = timestep_tuple[0].observation
        return observation, reward, done

    def set_seed(self, seed: int) -> int:
        self.env = self._build_env(
            self.envname, self.taskname, seed=seed, **self.constructor_kwargs
        )
        return seed

    @property
    def action_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self.env.action_spec())

    @property
    def observation_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self.env.observation_spec())

    @property
    def reward_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self.env.reward_spec())
