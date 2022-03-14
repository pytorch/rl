from numbers import Number
from typing import Tuple, Optional, Union

import numpy as np
import torch

from torchrl.data import (
    TensorSpec,
    CompositeSpec,
    NdUnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
)
from ..common import GymLikeEnv
from ...data.utils import numpy_to_torch_dtype_dict

try:
    import dm_control
    from dm_control.suite.wrappers import pixels
    from dm_control import suite
    import dm_env
    import collections
    _has_dmc = True
    __all__ = ["DMControlEnv"]

except:
    _has_dmc = False
    __all__ = []



def _dmcontrol_to_torchrl_spec_transform(
    spec, dtype: Optional[torch.dtype] = None
) -> TensorSpec:
    if isinstance(spec, collections.OrderedDict):
        spec = {
            k: _dmcontrol_to_torchrl_spec_transform(item) for k, item in spec.items()
        }
        return CompositeSpec(**spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            shape=spec.shape, minimum=spec.minimum, maximum=spec.maximum, dtype=dtype
        )
    elif isinstance(spec, dm_env.specs.Array):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdUnboundedContinuousTensorSpec(shape=spec.shape, dtype=dtype)
    else:
        raise NotImplementedError


def _get_envs(to_dict: bool = True) -> dict:
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


def _robust_to_tensor(array: Union[float, np.ndarray]) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.tensor(array.copy())
    else:
        return torch.tensor(array)


class DMControlEnv(GymLikeEnv):
    """
    DeepMind Control lab environment wrapper.

    Args:
        envname (str): name of the environment
        taskname (str): name of the task
        seed (int, optional): seed to use for the environment
        from_pixels (bool): if True, the observation

    Examples:
        >>> env = DMControlEnv(envname="cheetah", taskname="run", from_pixels=True, frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)
    """

    git_url = "https://github.com/deepmind/dm_control"
    libname = "dm_control"
    available_envs = _get_envs()

    def _build_env(
        self,
        envname: str,
        taskname: str,
        _seed: Optional[int] = None,
        from_pixels: bool = False,
        render_kwargs: Optional[dict] = None,
        pixels_only: bool = False,
        **kwargs,
    ):
        if not _has_dmc:
            raise RuntimeError(
                f"dm_control not found, unable to create {envname}: {taskname}. \
            Consider downloading and installing dm_control from {self.git_url}"
            )
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if _seed is not None:
            random_state = np.random.RandomState(_seed)
            kwargs = {"random": random_state}
        env = suite.load(envname, taskname, task_kwargs=kwargs)
        if from_pixels:
            self.render_kwargs = {"camera_id": 0}
            if render_kwargs is not None:
                self.render_kwargs.update(render_kwargs)
            env = pixels.Wrapper(
                env, pixels_only=self.pixels_only, render_kwargs=self.render_kwargs
            )
        self._env = env
        return env

    def _init_env(self, seed: Optional[int] = None) -> Optional[int]:
        seed = self.set_seed(seed)
        return seed

    def _set_seed(self, _seed: Optional[int]) -> Optional[int]:
        self._env = self._build_env(
            self.envname, self.taskname, _seed=_seed, **self.constructor_kwargs
        )
        self.reset()
        return _seed

    def _output_transform(
        self, timestep_tuple: Tuple["TimeStep"]
    ) -> Tuple[np.ndarray, float, bool]:
        if type(timestep_tuple) is not tuple:
            timestep_tuple = (timestep_tuple,)
        reward = timestep_tuple[0].reward

        done = False  # dm_control envs are non-terminating
        observation = timestep_tuple[0].observation
        return observation, reward, done

    @property
    def action_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self._env.action_spec())

    @property
    def observation_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self._env.observation_spec())

    @property
    def reward_spec(self) -> TensorSpec:
        return _dmcontrol_to_torchrl_spec_transform(self._env.reward_spec())
