# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.transforms.transforms import DoubleToFloat
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class EnvConfig(ConfigBase):
    _partial_: bool = False

    # def __post_init__(self):
    #     self._partial_ = False


@dataclass
class GymEnvConfig(EnvConfig):
    env_name: Any = None
    backend: str = "gymnasium"  # Changed from Literal to str
    from_pixels: bool = False
    double_to_float: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_env"

    @classmethod
    def default_config(cls, **kwargs) -> "GymEnvConfig":
        """Creates a default Gym environment configuration.
        
        Args:
            **kwargs: Override default values
            
        Returns:
            GymEnvConfig with default values, overridden by kwargs
        """
        defaults = {
            "env_name": "Pendulum-v1",
            "backend": "gymnasium",
            "from_pixels": False,
            "double_to_float": False,
            "_partial_": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class BatchedEnvConfig(EnvConfig):
    create_env_fn: EnvConfig | None = None
    num_workers: int | None = None
    batched_env_type: str = "parallel"
    # batched_env_type: Literal["parallel", "serial", "async"] = "parallel"
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_batched_env"

    def __post_init__(self):
        if self.create_env_fn is not None:
            self.create_env_fn._partial_ = True

    @classmethod
    def default_config(cls, **kwargs) -> "BatchedEnvConfig":
        """Creates a default batched environment configuration.
        
        Args:
            **kwargs: Override default values. Supports nested overrides using double underscore notation
                     (e.g., "create_env_fn__env_name": "CartPole-v1")
            
        Returns:
            BatchedEnvConfig with default values, overridden by kwargs
        """
        from tensordict import TensorDict

        # Unflatten the kwargs using TensorDict to understand what the user wants
        kwargs_td = TensorDict(kwargs)
        unflattened_kwargs = kwargs_td.unflatten_keys("__").to_dict()

        # Create configs with nested overrides applied
        env_overrides = unflattened_kwargs.get("create_env_fn", {})
        env_cfg = GymEnvConfig.default_config(**env_overrides)

        defaults = {
            "create_env_fn": env_cfg,
            "num_workers": unflattened_kwargs.get("num_workers", 4),
            "batched_env_type": unflattened_kwargs.get("batched_env_type", "parallel"),
            "_partial_": True,
        }
        
        return cls(**defaults)


def make_env(*args, **kwargs):
    from torchrl.envs.libs.gym import GymEnv

    backend = kwargs.pop("backend", None)
    double_to_float = kwargs.pop("double_to_float", False)
    with set_gym_backend(backend) if backend is not None else nullcontext():
        env = GymEnv(*args, **kwargs)
    if double_to_float:
        env = env.append_transform(DoubleToFloat(env))
    return env


def make_batched_env(*args, **kwargs):
    from torchrl.envs import AsyncEnvPool, ParallelEnv, SerialEnv

    batched_env_type = kwargs.pop("batched_env_type", "parallel")
    if batched_env_type == "parallel":
        return ParallelEnv(*args, **kwargs)
    elif batched_env_type == "serial":
        return SerialEnv(*args, **kwargs)
    elif batched_env_type == "async":
        kwargs["env_makers"] = [kwargs.pop("create_env_fn")] * kwargs.pop("num_workers")
        return AsyncEnvPool(*args, **kwargs)
