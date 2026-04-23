# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data.tensor_specs import (
    Bounded,
    Composite,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import _EnvPostInit
from torchrl.envs.gym_like import GymLikeEnv
from torchrl.envs.utils import _classproperty

_has_genesis = importlib.util.find_spec("genesis") is not None

__all__ = ["GenesisEnv", "GenesisWrapper"]


def _genesis_cleanup():
    """Clean up Genesis resources to free memory.

    Call this function to force cleanup of Genesis cached kernels and free up memory.
    This is useful when creating multiple Genesis environments in a long-running process.
    """
    if not _has_genesis:
        return
    import genesis as gs

    gs.destroy()


def _get_obs_func(
    scene,
    observation_type: str = "joints",
) -> dict[str, np.ndarray]:
    """Extract observation from Genesis scene.

    Args:
        scene: Genesis scene object
        observation_type: Type of observation to extract.
            Can be 'joints' (default), 'end_effector', or a custom callable.

    Returns:
        Dictionary of observations
    """
    obs = {}
    if observation_type == "joints":
        for entity in scene.entities:
            if hasattr(entity, "n_dofs") and entity.n_dofs > 0:
                try:
                    joints = entity.joints
                    if joints is not None and len(joints) > 0:
                        qpos = np.array([joint.q for joint in joints])
                        qvel = np.array([joint.dq_dt for joint in joints])
                        obs[f"{entity.name}_qpos"] = qpos
                        obs[f"{entity.name}_qvel"] = qvel
                except Exception:
                    pass
    elif observation_type == "end_effector":
        for entity in scene.entities:
            if hasattr(entity, "end_effector_pos"):
                obs[f"{entity.name}_ee_pos"] = np.array(entity.end_effector_pos)
    elif callable(observation_type):
        return observation_type(scene)

    if not obs:
        obs["empty_obs"] = np.zeros(1)
    return obs


def _default_reward_func(scene) -> float:
    """Default reward function - returns 0."""
    return 0.0


def _default_done_func(scene, max_steps: int, current_step: int) -> bool:
    """Default done function - done when max steps reached."""
    return current_step >= max_steps


class GenesisWrapper(GymLikeEnv):
    """Genesis physics simulation environment wrapper.

    Genesis is a universal physics engine designed for general-purpose
    robotics and embodied AI applications. It provides ultra-fast physics
    simulation and photo-realistic rendering.

    Args:
        scene (gs.Scene): Genesis scene instance.
        observation_func (callable, optional): A callable that takes the scene
            as input and returns a dictionary of observations.
            Defaults to extracting joint positions and velocities.
        reward_func (callable, optional): A callable that takes the scene
            as input and returns a float reward.
            Defaults to returning 0.
        done_func (callable, optional): A callable that takes the scene,
            max_steps, and current_step as inputs and returns a boolean
            indicating if the episode is done.
            Defaults to done when max_steps is reached.
        max_steps (int, optional): Maximum number of steps per episode.
            Defaults to 1000.
        from_pixels (bool, optional): If ``True``, pixel observations
            will be returned. Defaults to ``False``.
        pixels_only (bool, optional): If ``True``, only pixel observations
            will be returned. Defaults to ``True``.
        frame_skip (int, optional): Number of times to repeat each action.
            Defaults to 1.
        device (torch.device, optional): Device for tensor operations.
            Defaults to CPU.
        batch_size (torch.Size, optional): Batch size for vectorized envs.
            Defaults to empty batch size.
        allow_done_after_reset (bool, optional): If ``True``, allows done
            states immediately after reset. Defaults to ``False``.

    Examples:
        >>> import genesis as gs
        >>> from torchrl.envs import GenesisWrapper
        >>> # Initialize Genesis
        >>> gs.init(backend=gs.cpu)
        >>> # Create scene
        >>> scene = gs.Scene()
        >>> plane = scene.add_entity(gs.morphs.Plane())
        >>> franka = scene.add_entity(
        ...     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        ... )
        >>> scene.build()
        >>> # Wrap with TorchRL
        >>> env = GenesisWrapper(scene)
        >>> td = env.rand_step()
        >>> print(td)
        >>> # Or use with custom observation/reward functions:
        >>> def custom_obs(scene):
        ...     return {"joint_pos": franka.joint_pos}
        >>> def custom_reward(scene):
        ...     return -np.linalg.norm(franka.joint_pos)
        >>> env = GenesisWrapper(scene, observation_func=custom_obs,
        ...                      reward_func=custom_reward)
    """

    git_url = "https://github.com/Genesis-Embodied-AI/Genesis"
    libname = "genesis"

    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib
        import genesis

        cls._lib = genesis
        return genesis

    def __init__(
        self,
        scene=None,
        observation_func: Callable | None = None,
        reward_func: Callable | None = None,
        done_func: Callable | None = None,
        max_steps: int = 1000,
        from_pixels: bool = False,
        pixels_only: bool = True,
        frame_skip: int = 1,
        device: DEVICE_TYPING = "cpu",
        batch_size: torch.Size | None = None,
        allow_done_after_reset: bool = False,
        **kwargs,
    ):
        if scene is not None:
            kwargs["scene"] = scene
        if observation_func is not None:
            kwargs["observation_func"] = observation_func
        if reward_func is not None:
            kwargs["reward_func"] = reward_func
        if done_func is not None:
            kwargs["done_func"] = done_func
        kwargs["max_steps"] = max_steps
        kwargs["from_pixels"] = from_pixels
        kwargs["pixels_only"] = pixels_only
        super().__init__(
            device=device,
            batch_size=batch_size,
            frame_skip=frame_skip,
            allow_done_after_reset=allow_done_after_reset,
            **kwargs,
        )

    @_classproperty
    def available_envs(cls):
        if not _has_genesis:
            return []
        return ["custom_scene"]

    def _build_env(
        self,
        scene,
        observation_func: Callable | None = None,
        reward_func: Callable | None = None,
        done_func: Callable | None = None,
        max_steps: int = 1000,
        from_pixels: bool = False,
        pixels_only: bool = True,
        **kwargs,
    ):
        self._scene = scene
        self._observation_func = observation_func or _get_obs_func
        self._reward_func = reward_func or _default_reward_func
        self._done_func = done_func or _default_done_func
        self._max_steps = max_steps
        self._current_step = 0
        self._from_pixels = from_pixels
        self._pixels_only = pixels_only
        return scene

    def _make_specs(self, env) -> None:
        dummy_obs = self._observation_func(self._scene)
        if isinstance(dummy_obs, dict):
            self.observation_spec = Composite(
                **{
                    k: Unbounded(shape=v.shape, dtype=torch.float32)
                    for k, v in dummy_obs.items()
                },
                shape=self.batch_size,
                device=self.device,
            )
        else:
            self.observation_spec = Composite(
                observation=Unbounded(shape=dummy_obs.shape, dtype=torch.float32),
                shape=self.batch_size,
                device=self.device,
            )

        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device,
        )

        done_spec = torch.zeros(
            (*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_spec.clone(),
            truncated=done_spec.clone(),
            terminated=done_spec.clone(),
            device=self.device,
        )

        action_dim = 7
        for entity in self._scene.entities:
            if hasattr(entity, "n_dofs"):
                action_dim = max(action_dim, entity.n_dofs)
            elif hasattr(entity, "qpos"):
                try:
                    action_dim = max(action_dim, len(entity.qpos))
                except TypeError:
                    pass

        self.action_spec = Unbounded(
            shape=(*self.batch_size, action_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def _check_kwargs(self, kwargs: dict):
        if "scene" not in kwargs:
            raise TypeError("Could not find scene key 'scene' in kwargs.")
        scene = kwargs["scene"]
        if not hasattr(scene, "step"):
            raise TypeError("scene does not have a 'step' method.")

    def _init_env(self, seed: int | None = None) -> int | None:
        self._current_step = 0
        return seed

    def _set_seed(self, _seed: int | None) -> None:
        if _seed is None:
            return
        self._current_step = 0

    def _step(self, tensordict):
        if len(self.action_keys) == 1:
            action = tensordict[self.action_key]
        else:
            action = tensordict.select(*self.action_keys).to_dict()

        if self._convert_actions_to_numpy:
            action = self.read_action(action)

        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        elif isinstance(action, dict):
            action_np = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in action.items()
            }
        else:
            action_np = action

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            self._scene.step()
            self._current_step += 1
            r = self._reward_func(self._scene)
            if r is not None:
                reward = reward + r

        obs = self._observation_func(self._scene)
        terminated = False
        truncated = self._done_func(self._scene, self._max_steps, self._current_step)
        done = terminated or truncated

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward

        if isinstance(done, torch.Tensor):
            terminated = done.clone()
        else:
            terminated = torch.tensor(terminated)
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.clone()
        else:
            truncated = torch.tensor(truncated)
        if isinstance(done, torch.Tensor):
            done = done.clone()
        else:
            done = torch.tensor(done)
        obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated

        tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)

        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)

        return tensordict_out

    def _reset(self, tensordict=None, **kwargs):
        self._current_step = 0
        obs = self._observation_func(self._scene)
        source = self.read_obs(obs)
        tensordict_out = TensorDict(source=source, batch_size=self.batch_size)
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)
        return tensordict_out

    def _output_transform(self, step_outputs_tuple) -> tuple:
        obs, reward, terminated, truncated, done, info = step_outputs_tuple
        return obs, reward, terminated, truncated, done, info

    def _reset_output_transform(self, reset_data):
        return reset_data, {}

    def close(self, *, raise_if_closed: bool = True) -> None:
        if hasattr(self, "_scene") and self._scene is not None:
            try:
                self._scene.destroy()
            except Exception:
                pass
        super().close(raise_if_closed=raise_if_closed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scene={type(self._scene).__name__}, batch_size={self.batch_size})"


class _GenesisEnvMeta(_EnvPostInit):
    """Metaclass for GenesisEnv that returns a lazy ParallelEnv when num_workers > 1."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        if num_workers is None:
            num_workers = kwargs.pop("num_workers", 1)
        else:
            kwargs.pop("num_workers", None)

        num_workers = int(num_workers) if num_workers is not None else 1
        if cls.__name__ == "GenesisEnv" and num_workers > 1:
            from torchrl.envs import ParallelEnv

            env_name = args[0] if len(args) >= 1 else kwargs.get("env_name")
            task_name = args[1] if len(args) >= 2 else kwargs.get("task_name")

            env_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("env_name", "task_name")
            }

            def make_env(_env_name=env_name, _task_name=task_name, _kwargs=env_kwargs):
                return cls(_env_name, _task_name, num_workers=1, **_kwargs)

            return ParallelEnv(num_workers, make_env)

        return super().__call__(*args, **kwargs)


class GenesisEnv(GenesisWrapper, metaclass=_GenesisEnvMeta):
    """Genesis physics simulation environment.

    Creates a Genesis scene from configuration and wraps it with TorchRL.

    Args:
        env_name (str): Name of the environment configuration.
            Currently supports 'franka_reach', 'franka_grab', etc.
        task_name (str, optional): Task name for the environment.
        num_workers (int, optional): Number of parallel environments.
            When > 1, returns a lazy ParallelEnv. Defaults to 1.
        observation_func (callable, optional): Custom observation function.
        reward_func (callable, optional): Custom reward function.
        done_func (callable, optional): Custom done function.
        max_steps (int, optional): Max steps per episode. Defaults to 1000.
        from_pixels (bool, optional): Return pixel observations.
            Defaults to `False`.
        pixels_only (bool, optional): Only return pixels if True.
            Defaults to `True`.
        frame_skip (int, optional): Frame skip value. Defaults to 1.
        device (torch.device, optional): Device for tensors.
            Defaults to CPU.
        batch_size (torch.Size, optional): Batch size. Defaults to ().
        allow_done_after_reset (bool, optional): Allow done after reset.
            Defaults to False.

    Examples:
        >>> from torchrl.envs import GenesisEnv
        >>> env = GenesisEnv(env_name="franka_reach")
        >>> td = env.rand_step()
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.float32, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        empty_obs: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
    """

    _ENV_CONFIGS = {
        "franka_reach": {
            "description": "Franka robot reaching task",
            "default_obs": "joints",
        },
        "franka_grab": {
            "description": "Franka robot grasping task",
            "default_obs": "joints",
        },
    }

    def __init__(
        self,
        env_name: str,
        task_name: str | None = None,
        num_workers: int | None = None,
        observation_func: Callable | None = None,
        reward_func: Callable | None = None,
        done_func: Callable | None = None,
        max_steps: int = 1000,
        from_pixels: bool = False,
        pixels_only: bool = True,
        frame_skip: int = 1,
        device: DEVICE_TYPING = "cpu",
        batch_size: torch.Size | None = None,
        allow_done_after_reset: bool = False,
        **kwargs,
    ):
        if not _has_genesis:
            raise ImportError(
                "genesis python package was not found. Please install it with: pip install genesis-world"
            )

        self._env_name = env_name
        self._task_name = task_name

        scene = self._create_scene(env_name, task_name, **kwargs)

        super().__init__(
            scene=scene,
            observation_func=observation_func,
            reward_func=reward_func,
            done_func=done_func,
            max_steps=max_steps,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            frame_skip=frame_skip,
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
            **kwargs,
        )

    def _create_scene(self, env_name: str, task_name: str | None, **kwargs):
        """Create a Genesis scene based on environment configuration."""
        gs = self.lib

        try:
            gs.init(backend=gs.cpu)
        except Exception:
            pass  # Already initialized

        scene = gs.Scene()

        if env_name == "franka_reach":
            plane = scene.add_entity(gs.morphs.Plane())
            franka = scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            )
            scene.build()
        elif env_name == "franka_grab":
            plane = scene.add_entity(gs.morphs.Plane())
            franka = scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            )
            box = scene.add_entity(
                gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.05))
            )
            scene.build()
        else:
            raise ValueError(
                f"Unknown environment: {env_name}. "
                f"Available environments: {list(self._ENV_CONFIGS.keys())}"
            )

        return scene

    def _build_env(
        self,
        env_name: str | None = None,
        task_name: str | None = None,
        _seed: int | None = None,
        **kwargs,
    ):
        # Get env_name from instance if not provided
        if env_name is None:
            env_name = getattr(self, "env_name", None) or getattr(
                self, "_env_name", None
            )
        if task_name is None:
            task_name = getattr(self, "task_name", None) or getattr(
                self, "_task_name", None
            )

        from_pixels = kwargs.get("from_pixels", False)
        pixels_only = kwargs.get("pixels_only", True)

        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]

        if env_name is None:
            raise TypeError("GenesisEnv requires env_name to be specified")

        self.env_name = env_name
        self.task_name = task_name

        if not _has_genesis:
            raise ImportError(
                f"genesis not found, unable to create {env_name}:"
                f" {task_name}. Consider installing from {self.git_url}"
            )

        scene = self._create_scene(env_name, task_name)
        kwargs["scene"] = scene
        return super()._build_env(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        # env_name can be in kwargs or already set as instance attribute
        if "env_name" not in kwargs and not hasattr(self, "_env_name"):
            raise TypeError("GenesisEnv requires env_name to be specified")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, task={self.task_name}, batch_size={self.batch_size})"
