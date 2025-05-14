# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import importlib
import os
from typing import Any, Dict

import numpy as np
import torch

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import DEVICE_TYPING, numpy_to_torch_dtype_dict
from torchrl.envs.gym_like import GymLikeEnv
from torchrl.envs.utils import _classproperty

if torch.cuda.device_count() > 1:
    n = torch.cuda.device_count() - 1
    os.environ["EGL_DEVICE_ID"] = str(1 + (os.getpid() % n))
    if VERBOSE:
        torchrl_logger.info(f"EGL_DEVICE_ID: {os.environ['EGL_DEVICE_ID']}")

_has_dmc = _has_dm_control = importlib.util.find_spec("dm_control") is not None

__all__ = ["DMControlEnv", "DMControlWrapper"]


def _dmcontrol_to_torchrl_spec_transform(
    spec,
    dtype: torch.dtype | None = None,
    device: DEVICE_TYPING = None,
    categorical_discrete_encoding: bool = False,
) -> TensorSpec:
    import dm_env

    if isinstance(spec, collections.OrderedDict) or isinstance(spec, Dict):
        spec = {
            k: _dmcontrol_to_torchrl_spec_transform(
                item,
                device=device,
                categorical_discrete_encoding=categorical_discrete_encoding,
            )
            for k, item in spec.items()
        }
        return Composite(**spec)
    elif isinstance(spec, dm_env.specs.DiscreteArray):
        # DiscreteArray is a type of BoundedArray so this block needs to go first
        action_space_cls = Categorical if categorical_discrete_encoding else OneHot
        if dtype is None:
            dtype = (
                numpy_to_torch_dtype_dict[spec.dtype]
                if categorical_discrete_encoding
                else torch.long
            )
        return action_space_cls(spec.num_values, device=device, dtype=dtype)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        return Bounded(
            shape=shape,
            low=spec.minimum,
            high=spec.maximum,
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
            return Unbounded(shape=shape, dtype=dtype, device=device)
        else:
            return Unbounded(shape=shape, dtype=dtype, device=device)
    else:
        raise NotImplementedError(type(spec))


def _get_envs(to_dict: bool = True) -> dict[str, Any]:
    if not _has_dm_control:
        raise ImportError("Cannot find dm_control in virtual environment.")
    from dm_control import suite

    if not to_dict:
        return tuple(suite.BENCHMARKING) + tuple(suite.EXTRA)
    d = {}
    for tup in suite.BENCHMARKING:
        env_name = tup[0]
        d.setdefault(env_name, []).append(tup[1])
    for tup in suite.EXTRA:
        env_name = tup[0]
        d.setdefault(env_name, []).append(tup[1])
    return d.items()


def _robust_to_tensor(array: float | np.ndarray) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.as_tensor(array.copy())
    else:
        return torch.as_tensor(array)


class DMControlWrapper(GymLikeEnv):
    """DeepMind Control lab environment wrapper.

    The DeepMind control library can be found here: https://github.com/deepmind/dm_control.

    Paper: https://arxiv.org/abs/2006.12983

    Args:
        env (dm_control.suite env): :class:`~dm_control.suite.base.Task`
            environment instance.

    Keyword Args:
        from_pixels (bool, optional): if ``True``, an attempt to return the pixel
            observations from the env will be performed.
            By default, these observations
            will be written under the ``"pixels"`` entry.
            Defaults to ``False``.
        pixels_only (bool, optional): if ``True``, only the pixel observations will
            be returned (by default under the ``"pixels"`` entry in the output tensordict).
            If ``False``, observations (eg, states) and pixels will be returned
            whenever ``from_pixels=True``. Defaults to ``True``.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs (list): a list of ``Tuple[str, List[str]]`` representing the
            environment / task pairs available.

    Examples:
        >>> from dm_control import suite
        >>> from torchrl.envs import DMControlWrapper
        >>> env = suite.load("cheetah", "run")
        >>> env = DMControlWrapper(env,
        ...    from_pixels=True, frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([6]), device=cpu, dtype=torch.float64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([240, 320, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        position: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        velocity: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        [('acrobot', ['swingup', 'swingup_sparse']), ('ball_in_cup', ['catch']), ('cartpole', ['balance', 'balance_sparse', 'swingup', 'swingup_sparse', 'three_poles', 'two_poles']), ('cheetah', ['run']), ('finger', ['spin', 'turn_easy', 'turn_hard']), ('fish', ['upright', 'swim']), ('hopper', ['stand', 'hop']), ('humanoid', ['stand', 'walk', 'run', 'run_pure_state']), ('manipulator', ['bring_ball', 'bring_peg', 'insert_ball', 'insert_peg']), ('pendulum', ['swingup']), ('point_mass', ['easy', 'hard']), ('reacher', ['easy', 'hard']), ('swimmer', ['swimmer6', 'swimmer15']), ('walker', ['stand', 'walk', 'run']), ('dog', ['fetch', 'run', 'stand', 'trot', 'walk']), ('humanoid_CMU', ['run', 'stand', 'walk']), ('lqr', ['lqr_2_1', 'lqr_6_2']), ('quadruped', ['escape', 'fetch', 'run', 'walk']), ('stacker', ['stack_2', 'stack_4'])]

    """

    git_url = "https://github.com/deepmind/dm_control"
    libname = "dm_control"

    @_classproperty
    def available_envs(cls):
        if not _has_dm_control:
            return []
        return list(_get_envs())

    @property
    def lib(self):
        import dm_control

        return dm_control

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)

    def _build_env(
        self,
        env,
        _seed: int | None = None,
        from_pixels: bool = False,
        render_kwargs: dict | None = None,
        pixels_only: bool = False,
        camera_id: int | str = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if from_pixels:
            from dm_control.suite.wrappers import pixels

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

    def _make_specs(self, env: gym.Env) -> None:  # noqa: F821
        # specs are defined when first called
        self.observation_spec = _dmcontrol_to_torchrl_spec_transform(
            self._env.observation_spec(), device=self.device
        )
        reward_spec = _dmcontrol_to_torchrl_spec_transform(
            self._env.reward_spec(), device=self.device
        )
        if len(reward_spec.shape) == 0:
            reward_spec.shape = torch.Size([1])
        self.reward_spec = reward_spec
        # populate default done spec
        done_spec = Categorical(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_spec.clone(),
            truncated=done_spec.clone(),
            terminated=done_spec.clone(),
            device=self.device,
        )
        self.action_spec = _dmcontrol_to_torchrl_spec_transform(
            self._env.action_spec(), device=self.device
        )

    def _check_kwargs(self, kwargs: dict):
        dm_control = self.lib
        from dm_control.suite.wrappers import pixels

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

    def _init_env(self, seed: int | None = None) -> int | None:
        seed = self.set_seed(seed)
        return seed

    def _set_seed(self, _seed: int | None) -> None:
        from dm_control.suite.wrappers import pixels

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

    def _output_transform(
        self, timestep_tuple: tuple[TimeStep]  # noqa: F821
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        from dm_env import StepType

        if type(timestep_tuple) is not tuple:
            timestep_tuple = (timestep_tuple,)
        reward = timestep_tuple[0].reward

        truncated = terminated = False
        if timestep_tuple[0].step_type == StepType.LAST:
            if np.isclose(timestep_tuple[0].discount, 1):
                truncated = True
            else:
                terminated = True
        done = truncated or terminated

        observation = timestep_tuple[0].observation
        info = {}

        return observation, reward, terminated, truncated, done, info

    def _reset_output_transform(self, reset_data):
        (
            observation,
            reward,
            terminated,
            truncated,
            done,
            info,
        ) = self._output_transform(reset_data)
        return observation, info

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )


class DMControlEnv(DMControlWrapper):
    """DeepMind Control lab environment wrapper.

    The DeepMind control library can be found here: https://github.com/deepmind/dm_control.

    Paper: https://arxiv.org/abs/2006.12983

    Args:
        env_name (str): name of the environment.
        task_name (str): name of the task.

    Keyword Args:
        from_pixels (bool, optional): if ``True``, an attempt to return the pixel
            observations from the env will be performed.
            By default, these observations
            will be written under the ``"pixels"`` entry.
            Defaults to ``False``.
        pixels_only (bool, optional): if ``True``, only the pixel observations will
            be returned (by default under the ``"pixels"`` entry in the output tensordict).
            If ``False``, observations (eg, states) and pixels will be returned
            whenever ``from_pixels=True``. Defaults to ``True``.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Should match the leading dimensions of all observations, done states,
            rewards, actions and infos.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs (list): a list of ``Tuple[str, List[str]]`` representing the
            environment / task pairs available.

    Examples:
        >>> from torchrl.envs import DMControlEnv
        >>> env = DMControlEnv(env_name="cheetah", task_name="run",
        ...    from_pixels=True, frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([6]), device=cpu, dtype=torch.float64, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([240, 320, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        position: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float64, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        velocity: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        [('acrobot', ['swingup', 'swingup_sparse']), ('ball_in_cup', ['catch']), ('cartpole', ['balance', 'balance_sparse', 'swingup', 'swingup_sparse', 'three_poles', 'two_poles']), ('cheetah', ['run']), ('finger', ['spin', 'turn_easy', 'turn_hard']), ('fish', ['upright', 'swim']), ('hopper', ['stand', 'hop']), ('humanoid', ['stand', 'walk', 'run', 'run_pure_state']), ('manipulator', ['bring_ball', 'bring_peg', 'insert_ball', 'insert_peg']), ('pendulum', ['swingup']), ('point_mass', ['easy', 'hard']), ('reacher', ['easy', 'hard']), ('swimmer', ['swimmer6', 'swimmer15']), ('walker', ['stand', 'walk', 'run']), ('dog', ['fetch', 'run', 'stand', 'trot', 'walk']), ('humanoid_CMU', ['run', 'stand', 'walk']), ('lqr', ['lqr_2_1', 'lqr_6_2']), ('quadruped', ['escape', 'fetch', 'run', 'walk']), ('stacker', ['stack_2', 'stack_4'])]

    """

    def __init__(self, env_name, task_name, **kwargs):
        if not _has_dmc:
            raise ImportError(
                "dm_control python package was not found. Please install this dependency."
            )
        kwargs["env_name"] = env_name
        kwargs["task_name"] = task_name
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        task_name: str,
        _seed: int | None = None,
        **kwargs,
    ):
        from dm_control import suite

        self.env_name = env_name
        self.task_name = task_name

        from_pixels = kwargs.get("from_pixels")
        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]
        pixels_only = kwargs.get("pixels_only")
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]

        if not _has_dmc:
            raise ImportError(
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

    def _check_kwargs(self, kwargs: dict):
        if "env_name" in kwargs:
            env_name = kwargs["env_name"]
            if "task_name" in kwargs:
                task_name = kwargs["task_name"]
                available_envs = dict(self.available_envs)
                if (
                    env_name not in available_envs
                    or task_name not in available_envs[env_name]
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
