# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from torchrl.data import (
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
    TensorSpec,
)

from ...data.utils import DEVICE_TYPING, numpy_to_torch_dtype_dict
from ..gym_like import GymLikeEnv

if torch.has_cuda and torch.cuda.device_count() > 1:
    n = torch.cuda.device_count() - 1
    os.environ["EGL_DEVICE_ID"] = str(1 + (os.getpid() % n))
    print("EGL_DEVICE_ID: ", os.environ["EGL_DEVICE_ID"])

try:

    import dm_control
    import dm_env
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

    _has_dmc = True

except ImportError as err:
    _has_dmc = False
    IMPORT_ERR = str(err)

__all__ = ["DMControlEnv", "DMControlWrapper"]


def _dmcontrol_to_torchrl_spec_transform(
    spec,
    dtype: Optional[torch.dtype] = None,
    device: DEVICE_TYPING = None,
) -> TensorSpec:
    if isinstance(spec, collections.OrderedDict):
        spec = {
            k: _dmcontrol_to_torchrl_spec_transform(item, device=device)
            for k, item in spec.items()
        }
        return CompositeSpec(**spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        return NdBoundedTensorSpec(
            shape=shape,
            minimum=spec.minimum,
            maximum=spec.maximum,
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, dm_env.specs.Array):
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        if dtype in (torch.float, torch.double, torch.half):
            return NdUnboundedContinuousTensorSpec(
                shape=shape, dtype=dtype, device=device
            )
        else:
            return NdUnboundedDiscreteTensorSpec(
                shape=shape, dtype=dtype, device=device
            )

    else:
        raise NotImplementedError(type(spec))


def _get_envs(to_dict: bool = True) -> Dict[str, Any]:
    if not _has_dmc:
        return {}
    if not to_dict:
        return tuple(suite.BENCHMARKING) + tuple(suite.EXTRA)
    d = {}
    for tup in suite.BENCHMARKING:
        env_name = tup[0]
        d.setdefault(env_name, []).append(tup[1])
    for tup in suite.EXTRA:
        env_name = tup[0]
        d.setdefault(env_name, []).append(tup[1])
    return d


def _robust_to_tensor(array: Union[float, np.ndarray]) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.tensor(array.copy())
    else:
        return torch.tensor(array)


class DMControlWrapper(GymLikeEnv):
    """DeepMind Control lab environment wrapper.

    Args:
        env (dm_control.suite env): environment instance
        from_pixels (bool): if True, the observation

    Examples:
        >>> env = dm_control.suite.load("cheetah", "run")
        >>> env = DMControlWrapper(env,
        ...    from_pixels=True, frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/deepmind/dm_control"
    libname = "dm_control"
    available_envs = _get_envs()

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)

    def _build_env(
        self,
        env,
        _seed: Optional[int] = None,
        from_pixels: bool = False,
        render_kwargs: Optional[dict] = None,
        pixels_only: bool = False,
        camera_id: Union[int, str] = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if from_pixels:
            self._set_egl_device(self.device)
            self.render_kwargs = {"camera_id": camera_id}
            if render_kwargs is not None:
                self.render_kwargs.update(render_kwargs)
            env = pixels.Wrapper(
                env,
                pixels_only=self.pixels_only,
                render_kwargs=self.render_kwargs,
            )
        return env

    def _make_specs(self, env: "gym.Env") -> None:  # noqa: F821
        # specs are defined when first called
        return

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (dm_control.rl.control.Environment, pixels.Wrapper)):
            raise TypeError(
                "env is not of type 'dm_control.rl.control.Environment' or `dm_control.suite.wrappers.pixels.Wrapper`."
            )

    def _set_egl_device(self, device: DEVICE_TYPING):
        # Deprecated as lead to unreliable rendering
        # egl device needs to be set before importing mujoco bindings: in
        # distributed settings, it'll be easy to tell which cuda device to use.
        # In mp settings, we'll need to use mp.Pool with a specific init function
        # that defines the EGL device before importing libraries. For now, we'll
        # just use a common EGL_DEVICE_ID environment variable for all processes.
        return

    def to(self, device: DEVICE_TYPING) -> DMControlEnv:
        super().to(device)
        self._set_egl_device(self.device)
        return self

    def _init_env(self, seed: Optional[int] = None) -> Optional[int]:
        seed = self.set_seed(seed)
        return seed

    def _set_seed(self, _seed: Optional[int]) -> Optional[int]:
        if _seed is None:
            return None
        random_state = np.random.RandomState(_seed)
        if isinstance(self._env, pixels.Wrapper):
            if not hasattr(self._env._env.task, "_random"):
                raise RuntimeError("self._env._env.task._random does not exist")
            self._env._env.task._random = random_state
        else:
            if not hasattr(self._env.task, "_random"):
                raise RuntimeError("self._env._env.task._random does not exist")
            self._env.task._random = random_state
        self.reset()
        return _seed

    def _output_transform(
        self, timestep_tuple: Tuple["TimeStep"]  # noqa: F821
    ) -> Tuple[np.ndarray, float, bool]:
        if type(timestep_tuple) is not tuple:
            timestep_tuple = (timestep_tuple,)
        reward = timestep_tuple[0].reward

        done = False  # dm_control envs are non-terminating
        observation = timestep_tuple[0].observation
        return observation, reward, done

    @property
    def input_spec(self) -> TensorSpec:
        if self._input_spec is None:
            self.__dict__["_input_spec"] = CompositeSpec(
                action=_dmcontrol_to_torchrl_spec_transform(
                    self._env.action_spec(), device=self.device
                )
            )
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        if not isinstance(value, CompositeSpec):
            raise TypeError("The type of an input_spec must be Composite.")
        self.__dict__["_input_spec"] = value

    @property
    def observation_spec(self) -> TensorSpec:
        if self._observation_spec is None:
            self.__dict__["_observation_spec"] = _dmcontrol_to_torchrl_spec_transform(
                self._env.observation_spec(), device=self.device
            )
        return self._observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        if not isinstance(value, CompositeSpec):
            raise TypeError("The type of an observation_spec must be Composite.")
        self.__dict__["_observation_spec"] = value

    @property
    def reward_spec(self) -> TensorSpec:
        if self._reward_spec is None:
            reward_spec = _dmcontrol_to_torchrl_spec_transform(
                self._env.reward_spec(), device=self.device
            )
            if len(reward_spec.shape) == 0:
                reward_spec.shape = torch.Size([1])
            self.__dict__["_reward_spec"] = reward_spec
        return self._reward_spec

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        if not hasattr(value, "shape"):
            raise TypeError(
                f"reward_spec of type {type(value)} do not have a shape " f"attribute."
            )
        if len(value.shape) == 0:
            raise RuntimeError(
                "the reward_spec shape cannot be empty (this error"
                " usually comes from trying to set a reward_spec"
                " with a null number of dimensions. Try using a multidimensional"
                " spec instead, for instance with a singleton dimension at the tail)."
            )
        self.__dict__["_reward_spec"] = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )


class DMControlEnv(DMControlWrapper):
    """DeepMind Control lab environment wrapper.

    Args:
        env_name (str): name of the environment
        task_name (str): name of the task
        seed (int, optional): seed to use for the environment
        from_pixels (bool, optional): if True, the observation will be returned
            as an image.
            Default is False.

    Examples:
        >>> env = DMControlEnv(env_name="cheetah", task_name="run",
        ...    from_pixels=True, frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    def __init__(self, env_name, task_name, **kwargs):
        if not _has_dmc:
            raise ImportError(
                f"""dm_control python package was not found. Please install this dependency.
(Got the error message: {IMPORT_ERR}).
"""
            )
        kwargs["env_name"] = env_name
        kwargs["task_name"] = task_name
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        task_name: str,
        _seed: Optional[int] = None,
        **kwargs,
    ):
        self.env_name = env_name
        self.task_name = task_name

        from_pixels = kwargs.get("from_pixels")
        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]
        pixels_only = kwargs.get("pixels_only")
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]

        if not _has_dmc:
            raise RuntimeError(
                f"dm_control not found, unable to create {env_name}:"
                f" {task_name}. Consider downloading and installing "
                f"dm_control from {self.git_url}"
            )

        if _seed is not None:
            random_state = np.random.RandomState(_seed)
            kwargs = {"random": random_state}
        camera_id = kwargs.pop("camera_id", 0)
        env = suite.load(env_name, task_name, task_kwargs=kwargs)
        return super()._build_env(
            env,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            camera_id=camera_id,
            **kwargs,
        )

    def rebuild_with_kwargs(self, **new_kwargs):
        self._constructor_kwargs.update(new_kwargs)
        self._env = self._build_env()
        self._make_specs(self._env)

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" in kwargs:
            env_name = kwargs["env_name"]
            if "task_name" in kwargs:
                task_name = kwargs["task_name"]
                if (
                    env_name not in self.available_envs
                    or task_name not in self.available_envs[env_name]
                ):
                    raise RuntimeError(
                        f"{env_name} with task {task_name} is unknown in {self.libname}"
                    )
            else:
                raise TypeError("dm_control requires task_name to be specified")
        else:
            raise TypeError("dm_control requires env_name to be specified")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, task={self.task_name}, batch_size={self.batch_size})"
