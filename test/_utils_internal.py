# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os

import os.path
import time
from functools import wraps

# Get relative file path
# this returns relative path from current file.

import pytest
import torch
import torch.cuda

from tensordict import tensorclass
from torchrl._utils import implement_for, seed_generator

from torchrl.envs import ObservationNorm
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import (
    Compose,
    RewardClipping,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.vec_env import _has_envpool, MultiThreadedEnv, ParallelEnv, SerialEnv

# Specified for test_utils.py
__version__ = "0.3"

# Default versions of the environments.
CARTPOLE_VERSIONED = "CartPole-v1"
HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
PENDULUM_VERSIONED = "Pendulum-v1"
PONG_VERSIONED = "ALE/Pong-v5"


@implement_for("gym", None, "0.21.0")
def _set_gym_environments():  # noqa: F811
    global CARTPOLE_VERSIONED, HALFCHEETAH_VERSIONED, PENDULUM_VERSIONED, PONG_VERSIONED

    CARTPOLE_VERSIONED = "CartPole-v0"
    HALFCHEETAH_VERSIONED = "HalfCheetah-v2"
    PENDULUM_VERSIONED = "Pendulum-v0"
    PONG_VERSIONED = "Pong-v4"


@implement_for("gym", "0.21.0", None)
def _set_gym_environments():  # noqa: F811
    global CARTPOLE_VERSIONED, HALFCHEETAH_VERSIONED, PENDULUM_VERSIONED, PONG_VERSIONED

    CARTPOLE_VERSIONED = "CartPole-v1"
    HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    PENDULUM_VERSIONED = "Pendulum-v1"
    PONG_VERSIONED = "ALE/Pong-v5"


@implement_for("gymnasium", "0.27.0", None)
def _set_gym_environments():  # noqa: F811
    global CARTPOLE_VERSIONED, HALFCHEETAH_VERSIONED, PENDULUM_VERSIONED, PONG_VERSIONED

    CARTPOLE_VERSIONED = "CartPole-v1"
    HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    PENDULUM_VERSIONED = "Pendulum-v1"
    PONG_VERSIONED = "ALE/Pong-v5"


if _has_gym:
    _set_gym_environments()


def get_relative_path(curr_file, *path_components):
    return os.path.join(os.path.dirname(curr_file), *path_components)


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


def generate_seeds(seed, repeat):
    seeds = [seed]
    for _ in range(repeat - 1):
        seed = seed_generator(seed)
        seeds.append(seed)
    return seeds


# Decorator to retry upon certain Exceptions.
def retry(ExceptionToCheck, tries=3, delay=3, skip_after_retries=False):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                if skip_after_retries:
                    raise pytest.skip(
                        f"Skipping after {tries} consecutive {str(e)}"
                    ) from e
                else:
                    raise e

        return f_retry  # true decorator

    return deco_retry


@pytest.fixture
def dtype_fixture():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield dtype
    torch.set_default_dtype(dtype)


@contextlib.contextmanager
def set_global_var(module, var_name, value):
    old_value = getattr(module, var_name)
    setattr(module, var_name, value)
    try:
        yield
    finally:
        setattr(module, var_name, old_value)


def _make_envs(
    env_name,
    frame_skip,
    transformed_in,
    transformed_out,
    N,
    device="cpu",
    kwargs=None,
):
    torch.manual_seed(0)
    if not transformed_in:

        def create_env_fn():
            return GymEnv(env_name, frame_skip=frame_skip, device=device)

    else:
        if env_name == PONG_VERSIONED:

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
    env_parallel = ParallelEnv(N, create_env_fn, create_env_kwargs=kwargs)
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
        env_multithread = _make_multithreaded_env(
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


def _make_multithreaded_env(
    env_name,
    frame_skip,
    transformed_out,
    N,
    device="cpu",
    kwargs=None,
):

    torch.manual_seed(0)
    multithreaded_kwargs = (
        {"frame_skip": frame_skip} if env_name == PONG_VERSIONED else {}
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

    if env_name == PONG_VERSIONED:
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


def make_tc(td):
    """Makes a tensorclass from a tensordict instance."""

    class MyClass:
        pass

    MyClass.__annotations__ = {}
    for key in td.keys():
        MyClass.__annotations__[key] = torch.Tensor
    return tensorclass(MyClass)
