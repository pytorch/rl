# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch

from torchrl.data import (
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    TensorSpec,
)
from ...data.utils import numpy_to_torch_dtype_dict, DEVICE_TYPING
from ..common import GymLikeEnv

__all__ = ["DMControlEnv"]

import collections

try:

    import dm_env

    _has_dmc = True

except ImportError:
    _has_dmc = False

if _has_dmc:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels


def _dmcontrol_to_torchrl_spec_transform(
    spec, dtype: Optional[torch.dtype] = None
) -> TensorSpec:
    if isinstance(spec, collections.OrderedDict):
        spec = {
            "next_" + k: _dmcontrol_to_torchrl_spec_transform(item)
            for k, item in spec.items()
        }
        return CompositeSpec(**spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            shape=spec.shape,
            minimum=spec.minimum,
            maximum=spec.maximum,
            dtype=dtype,
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
        >>> env = DMControlEnv(envname="cheetah", taskname="run",
        ...    from_pixels=True, frame_skip=4)
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
                f"dm_control not found, unable to create {envname}:"
                f" {taskname}. Consider downloading and installing "
                f"dm_control from {self.git_url}"
            )
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if _seed is not None:
            random_state = np.random.RandomState(_seed)
            kwargs = {"random": random_state}
        env = suite.load(envname, taskname, task_kwargs=kwargs)
        if from_pixels:
            self._set_egl_device(self.device)
            self.render_kwargs = {"camera_id": 0}
            if render_kwargs is not None:
                self.render_kwargs.update(render_kwargs)
            env = pixels.Wrapper(
                env,
                pixels_only=self.pixels_only,
                render_kwargs=self.render_kwargs,
            )
        self._env = env
        return env

    def _set_egl_device(self, device: DEVICE_TYPING):
        if device != torch.device("cpu"):
            device_id = str(device).split(":")[-1]
            if (
                "EGL_DEVICE_ID" in os.environ
                and os.environ["EGL_DEVICE_ID"] != device_id
            ):
                raise RuntimeError(
                    f"Conflicting devices for DMControl env pixel rendering: "
                    f"got {device_id} but device already set to {os.environ['EGL_DEVICE_ID']}"
                )
            print(f"rendering on device {device_id}")
            os.environ["EGL_DEVICE_ID"] = device_id

    def to(self, device: DEVICE_TYPING) -> DMControlEnv:
        super().to(device)
        self._set_egl_device(self.device)
        return self

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
