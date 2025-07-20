# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.transforms.transforms import DoubleToFloat
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class EnvConfig(ConfigBase):
    _partial_: bool = False

    def __post_init__(self):
        self._partial_ = False


@dataclass
class GymEnvConfig(EnvConfig):
    env_name: Any = None
    backend: str = "gymnasium"  # Changed from Literal to str
    from_pixels: bool = False
    double_to_float: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_env"


@dataclass
class BatchedEnvConfig(EnvConfig):
    create_env_fn: EnvConfig | None = None
    num_workers: int | None = None
    batched_env_type: str = "parallel"
    # batched_env_type: Literal["parallel", "serial", "async"] = "parallel"
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_batched_env"

    def __post_init__(self):
        self.create_env_fn._partial_ = True


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
