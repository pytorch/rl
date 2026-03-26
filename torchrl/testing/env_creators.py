# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Environment creation utilities for TorchRL tests."""

from __future__ import annotations

import torch

from torchrl.envs import MultiThreadedEnv, ObservationNorm
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.libs.envpool import _has_envpool
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    Compose,
    RewardClipping,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.testing.gym_helpers import HALFCHEETAH_VERSIONED, PONG_VERSIONED
from torchrl.testing.utils import mp_ctx

__all__ = [
    "get_transform_out",
    "make_envs",
    "make_multithreaded_env",
]


def make_envs(
    env_name,
    frame_skip,
    transformed_in,
    transformed_out,
    N,
    device="cpu",
    kwargs=None,
    local_mp_ctx=mp_ctx,
):
    """Create parallel, serial, multithreaded, and single environment instances.

    This helper creates environments suitable for testing batched environment behavior.

    Args:
        env_name: The gym environment name.
        frame_skip: Number of frames to skip.
        transformed_in: Whether to apply transforms inside the base env.
        transformed_out: Whether to apply transforms outside the batched env.
        N: Number of environments in the batch.
        device: Device for the environments.
        kwargs: Additional keyword arguments for environment creation.
        local_mp_ctx: Multiprocessing context ('fork' or 'spawn').

    Returns:
        Tuple of (env_parallel, env_serial, env_multithread, env0).
    """
    torch.manual_seed(0)
    if not transformed_in:

        def create_env_fn():
            return GymEnv(env_name, frame_skip=frame_skip, device=device)

    else:
        if env_name == PONG_VERSIONED():

            def create_env_fn():
                base_env = GymEnv(env_name, frame_skip=frame_skip, device=device)
                in_keys = list(base_env.observation_spec.keys(True, True))[:1]
                return TransformedEnv(
                    base_env,
                    Compose(*[ToTensorImage(in_keys=in_keys), RewardClipping(0, 0.1)]),
                )

        else:

            def create_env_fn():

                base_env = GymEnv(env_name, frame_skip=frame_skip, device=device)
                in_keys = list(base_env.observation_spec.keys(True, True))[:1]

                return TransformedEnv(
                    base_env,
                    Compose(
                        ObservationNorm(in_keys=in_keys, loc=0.5, scale=1.1),
                        RewardClipping(0, 0.1),
                    ),
                )

    env0 = create_env_fn()
    env_parallel = ParallelEnv(
        N, create_env_fn, create_env_kwargs=kwargs, mp_start_method=local_mp_ctx
    )
    env_serial = SerialEnv(N, create_env_fn, create_env_kwargs=kwargs)

    for key in env0.observation_spec.keys(True, True):
        obs_key = key
        break
    else:
        obs_key = None

    if transformed_out:
        t_out = get_transform_out(env_name, transformed_in, obs_key=obs_key)

        env0 = TransformedEnv(
            env0,
            t_out(),
        )
        env_parallel = TransformedEnv(
            env_parallel,
            t_out(),
        )
        env_serial = TransformedEnv(
            env_serial,
            t_out(),
        )
    else:
        t_out = None

    if _has_envpool:
        env_multithread = make_multithreaded_env(
            env_name,
            frame_skip,
            t_out,
            N,
            device="cpu",
            kwargs=None,
        )
    else:
        env_multithread = None

    return env_parallel, env_serial, env_multithread, env0


def make_multithreaded_env(
    env_name,
    frame_skip,
    transformed_out,
    N,
    device="cpu",
    kwargs=None,
):
    """Create a multithreaded environment using envpool.

    Args:
        env_name: The gym environment name.
        frame_skip: Number of frames to skip.
        transformed_out: Transform factory to apply, or None.
        N: Number of environments in the batch.
        device: Device for the environment.
        kwargs: Additional keyword arguments (unused, for API compatibility).

    Returns:
        A MultiThreadedEnv instance, optionally wrapped with transforms.
    """
    torch.manual_seed(0)
    multithreaded_kwargs = (
        {"frame_skip": frame_skip} if env_name == PONG_VERSIONED() else {}
    )
    env_multithread = MultiThreadedEnv(
        N,
        env_name,
        create_env_kwargs=multithreaded_kwargs,
        device=device,
    )

    if transformed_out:
        for key in env_multithread.observation_spec.keys(True, True):
            obs_key = key
            break
        else:
            obs_key = None
        env_multithread = TransformedEnv(
            env_multithread,
            get_transform_out(env_name, transformed_in=False, obs_key=obs_key)(),
        )
    return env_multithread


def get_transform_out(env_name, transformed_in, obs_key=None):
    """Create a transform factory for output transforms based on environment type.

    Args:
        env_name: The gym environment name.
        transformed_in: Whether transforms were already applied inside.
        obs_key: The observation key to transform.

    Returns:
        A callable that returns a Compose transform.
    """
    if env_name == PONG_VERSIONED():
        if obs_key is None:
            obs_key = "pixels"

        def t_out():
            return (
                Compose(*[ToTensorImage(in_keys=[obs_key]), RewardClipping(0, 0.1)])
                if not transformed_in
                else Compose(*[ObservationNorm(in_keys=[obs_key], loc=0, scale=1)])
            )

    elif env_name == HALFCHEETAH_VERSIONED:
        if obs_key is None:
            obs_key = ("observation", "velocity")

        def t_out():
            return Compose(
                ObservationNorm(in_keys=[obs_key], loc=0.5, scale=1.1),
                RewardClipping(0, 0.1),
            )

    else:
        if obs_key is None:
            obs_key = "observation"

        def t_out():
            return (
                Compose(
                    ObservationNorm(in_keys=[obs_key], loc=0.5, scale=1.1),
                    RewardClipping(0, 0.1),
                )
                if not transformed_in
                else Compose(ObservationNorm(in_keys=[obs_key], loc=1.0, scale=1.0))
            )

    return t_out
