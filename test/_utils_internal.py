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

from tensordict import tensorclass, TensorDict
from torchrl._utils import implement_for, logger as torchrl_logger, seed_generator
from torchrl.data.utils import CloudpickleWrapper

from torchrl.envs import MultiThreadedEnv, ObservationNorm
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.libs.envpool import _has_envpool
from torchrl.envs.libs.gym import _has_gym, gym_backend, GymEnv
from torchrl.envs.transforms import (
    Compose,
    RewardClipping,
    ToTensorImage,
    TransformedEnv,
)

# Specified for test_utils.py
__version__ = "0.3"


def CARTPOLE_VERSIONED():
    # load gym
    if gym_backend() is not None:
        _set_gym_environments()
        return _CARTPOLE_VERSIONED


def HALFCHEETAH_VERSIONED():
    # load gym
    if gym_backend() is not None:
        _set_gym_environments()
        return _HALFCHEETAH_VERSIONED


def PONG_VERSIONED():
    # load gym
    if gym_backend() is not None:
        _set_gym_environments()
        return _PONG_VERSIONED


def PENDULUM_VERSIONED():
    # load gym
    if gym_backend() is not None:
        _set_gym_environments()
        return _PENDULUM_VERSIONED


def _set_gym_environments():
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED

    _CARTPOLE_VERSIONED = None
    _HALFCHEETAH_VERSIONED = None
    _PENDULUM_VERSIONED = None
    _PONG_VERSIONED = None


@implement_for("gym", None, "0.21.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v0"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v2"
    _PENDULUM_VERSIONED = "Pendulum-v0"
    _PONG_VERSIONED = "Pong-v4"


@implement_for("gym", "0.21.0", None)
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"


@implement_for("gymnasium")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"


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


def get_default_devices():
    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        return [torch.device("cpu")]
    elif num_cuda == 1:
        return [torch.device("cuda:0")]
    else:
        # then run on all devices
        return get_available_devices()


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
                    torchrl_logger.info(msg)
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


def make_tc(td):
    """Makes a tensorclass from a tensordict instance."""

    class MyClass:
        pass

    MyClass.__annotations__ = {}
    for key in td.keys():
        MyClass.__annotations__[key] = torch.Tensor
    return tensorclass(MyClass)


def rollout_consistency_assertion(
    rollout, *, done_key="done", observation_key="observation", done_strict=False
):
    """Tests that observations in "next" match observations in the next root tensordict when done is False, and don't match otherwise."""

    done = rollout[..., :-1]["next", done_key].squeeze(-1)
    # data resulting from step, when it's not done
    r_not_done = rollout[..., :-1]["next"][~done]
    # data resulting from step, when it's not done, after step_mdp
    r_not_done_tp1 = rollout[:, 1:][~done]
    torch.testing.assert_close(
        r_not_done[observation_key],
        r_not_done_tp1[observation_key],
        msg=f"Key {observation_key} did not match",
    )

    if done_strict and not done.any():
        raise RuntimeError("No done detected, test could not complete.")
    if done.any():
        # data resulting from step, when it's done
        r_done = rollout[..., :-1]["next"][done]
        # data resulting from step, when it's done, after step_mdp and reset
        r_done_tp1 = rollout[..., 1:][done]
        # check that at least one obs after reset does not match the version before reset
        assert not torch.isclose(
            r_done[observation_key], r_done_tp1[observation_key]
        ).all()


def rand_reset(env):
    """Generates a tensordict with reset keys that mimic the done spec.

    Values are drawn at random until at least one reset is present.

    """
    full_done_spec = env.full_done_spec
    result = {}
    for reset_key, list_of_done in zip(env.reset_keys, env.done_keys_groups):
        val = full_done_spec[list_of_done[0]].rand()
        while not val.any():
            val = full_done_spec[list_of_done[0]].rand()
        result[reset_key] = val
    # create a data structure that keeps the batch size of the nested specs
    result = (
        full_done_spec.zero().update(result).exclude(*full_done_spec.keys(True, True))
    )
    return result


def check_rollout_consistency_multikey_env(td: TensorDict, max_steps: int):
    index_batch_size = (0,) * (len(td.batch_size) - 1)

    # Check done and reset for root
    observation_is_max = td["next", "observation"][..., 0, 0, 0] == max_steps + 1
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "done"][observation_is_max]).all()
    assert (~td["next", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (td["observation"][index_batch_size][1:][next_is_done] == 0).all()
    # Obs after not done is previous obs
    assert (
        td["observation"][index_batch_size][1:][~next_is_done]
        == td["next", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for root
    action_is_count = td["action"].long().argmax(-1).to(torch.bool)
    assert (
        td["next", "observation"][action_is_count]
        == td["observation"][action_is_count] + 1
    ).all()
    assert (td["next", "reward"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for root
    assert (
        td["next", "observation"][~action_is_count]
        == td["observation"][~action_is_count]
    ).all()
    assert (td["next", "reward"][~action_is_count] == 0).all()

    # Check done and reset for nested_1
    observation_is_max = td["next", "nested_1", "observation"][..., 0] == max_steps + 1
    # done at the root always prevail
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "nested_1", "done"][observation_is_max]).all()
    assert (~td["next", "nested_1", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (
        td["nested_1", "observation"][index_batch_size][1:][next_is_done] == 0
    ).all()
    # Obs after not done is previous obs
    assert (
        td["nested_1", "observation"][index_batch_size][1:][~next_is_done]
        == td["next", "nested_1", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for nested_1
    action_is_count = td["nested_1"]["action"].to(torch.bool)
    assert (
        td["next", "nested_1", "observation"][action_is_count]
        == td["nested_1", "observation"][action_is_count] + 1
    ).all()
    assert (td["next", "nested_1", "gift"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for nested_1
    assert (
        td["next", "nested_1", "observation"][~action_is_count]
        == td["nested_1", "observation"][~action_is_count]
    ).all()
    assert (td["next", "nested_1", "gift"][~action_is_count] == 0).all()

    # Check done and reset for nested_2
    observation_is_max = td["next", "nested_2", "observation"][..., 0] == max_steps + 1
    # done at the root always prevail
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "nested_2", "done"][observation_is_max]).all()
    assert (~td["next", "nested_2", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (
        td["nested_2", "observation"][index_batch_size][1:][next_is_done] == 0
    ).all()
    # Obs after not done is previous obs
    assert (
        td["nested_2", "observation"][index_batch_size][1:][~next_is_done]
        == td["next", "nested_2", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for nested_2
    action_is_count = td["nested_2"]["azione"].squeeze(-1).to(torch.bool)
    assert (
        td["next", "nested_2", "observation"][action_is_count]
        == td["nested_2", "observation"][action_is_count] + 1
    ).all()
    assert (td["next", "nested_2", "reward"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for nested_2
    assert (
        td["next", "nested_2", "observation"][~action_is_count]
        == td["nested_2", "observation"][~action_is_count]
    ).all()
    assert (td["next", "nested_2", "reward"][~action_is_count] == 0).all()


def decorate_thread_sub_func(func, num_threads):
    def new_func(*args, **kwargs):
        assert torch.get_num_threads() == num_threads
        return func(*args, **kwargs)

    return CloudpickleWrapper(new_func)
