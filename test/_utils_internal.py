# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import logging
import os
import os.path
import sys
import time
import unittest
import warnings
from functools import wraps

import pytest
import torch
import torch.cuda
from tensordict import NestedKey, tensorclass, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch import nn, vmap

from torchrl._utils import (
    implement_for,
    logger as torchrl_logger,
    RL_WARNINGS,
    seed_generator,
)
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
from torchrl.modules import MLP
from torchrl.objectives.value.advantages import _vmap_func

# Get relative file path
# this returns relative path from current file.

# Specified for test_utils.py
__version__ = "0.3"

IS_WIN = sys.platform == "win32"
if IS_WIN:
    mp_ctx = "spawn"
else:
    mp_ctx = "fork"


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
    # Gymnasium says that the ale_py behavior changes from 1.0
    # but with python 3.12 it is already the case with 0.29.1
    try:
        import ale_py  # noqa
    except ImportError:
        pass

    if gym_backend() is not None:
        _set_gym_environments()
        return _PONG_VERSIONED


def BREAKOUT_VERSIONED():
    # load gym
    # Gymnasium says that the ale_py behavior changes from 1.0
    # but with python 3.12 it is already the case with 0.29.1
    try:
        import ale_py  # noqa
    except ImportError:
        pass

    if gym_backend() is not None:
        _set_gym_environments()
        return _BREAKOUT_VERSIONED


def PENDULUM_VERSIONED():
    # load gym
    if gym_backend() is not None:
        _set_gym_environments()
        return _PENDULUM_VERSIONED


def _set_gym_environments():
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED

    _CARTPOLE_VERSIONED = None
    _HALFCHEETAH_VERSIONED = None
    _PENDULUM_VERSIONED = None
    _PONG_VERSIONED = None
    _BREAKOUT_VERSIONED = None


@implement_for("gym", None, "0.21.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v0"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v2"
    _PENDULUM_VERSIONED = "Pendulum-v0"
    _PONG_VERSIONED = "Pong-v4"
    _BREAKOUT_VERSIONED = "Breakout-v4"


@implement_for("gym", "0.21.0", None)
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"


@implement_for("gymnasium", None, "1.0.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"


@implement_for("gymnasium", "1.0.0", "1.1.0")
def _set_gym_environments():  # noqa: F811
    raise ImportError


@implement_for("gymnasium", "1.1.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v5"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"


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
        # if torch.mps.is_available():
        #     return [torch.device("mps:0")]
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


# After calling this function, any log record whose name contains 'record_name'
# and is emitted from the logger that has qualified name 'logger_qname' is
# appended to the 'records' list.
# NOTE: This function is based on testing utilities for 'torch._logging'
def capture_log_records(records, logger_qname, record_name):
    assert isinstance(records, list)
    logger = logging.getLogger(logger_qname)

    class EmitWrapper:
        def __init__(self, old_emit):
            self.old_emit = old_emit

        def __call__(self, record):
            nonlocal records
            self.old_emit(record)
            if record_name in record.name:
                records.append(record)

    for handler in logger.handlers:
        new_emit = EmitWrapper(handler.emit)
        contextlib.ExitStack().enter_context(
            unittest.mock.patch.object(handler, "emit", new_emit)
        )


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
    local_mp_ctx=mp_ctx,
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


class LSTMNet(nn.Module):
    """An embedder for an LSTM preceded by an MLP.

    The forward method returns the hidden states of the current state
    (input hidden states) and the output, as
    the environment returns the 'observation' and 'next_observation'.

    Because the LSTM kernel only returns the last hidden state, hidden states
    are padded with zeros such that they have the right size to be stored in a
    TensorDict of size [batch x time_steps].

    If a 2D tensor is provided as input, it is assumed that it is a batch of data
    with only one time step. This means that we explicitly assume that users will
    unsqueeze inputs of a single batch with multiple time steps.

    Args:
        out_features (int): number of output features.
        lstm_kwargs (dict): the keyword arguments for the
            :class:`~torch.nn.LSTM` layer.
        mlp_kwargs (dict): the keyword arguments for the
            :class:`~torchrl.modules.MLP` layer.
        device (torch.device, optional): the device where the module should
            be instantiated.

    Keyword Args:
        lstm_backend (str, optional): one of ``"torchrl"`` or ``"torch"`` that
            indeicates where the LSTM class is to be retrieved. The ``"torchrl"``
            backend (:class:`~torchrl.modules.LSTM`) is slower but works with
            :func:`~torch.vmap` and should work with :func:`~torch.compile`.
            Defaults to ``"torch"``.

    Examples:
        >>> batch = 7
        >>> time_steps = 6
        >>> in_features = 4
        >>> out_features = 10
        >>> hidden_size = 5
        >>> net = LSTMNet(
        ...     out_features,
        ...     {"input_size": hidden_size, "hidden_size": hidden_size},
        ...     {"out_features": hidden_size},
        ... )
        >>> # test single step vs multi-step
        >>> x = torch.randn(batch, time_steps, in_features)  # >3 dims = multi-step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)
        >>> x = torch.randn(batch, in_features)  # 2 dims = single step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)

    """

    def __init__(
        self,
        out_features: int,
        lstm_kwargs,
        mlp_kwargs,
        device=None,
        *,
        lstm_backend: str | None = None,
    ) -> None:
        super().__init__()
        lstm_kwargs.update({"batch_first": True})
        self.mlp = MLP(device=device, **mlp_kwargs)
        if lstm_backend is None:
            lstm_backend = "torch"
        self.lstm_backend = lstm_backend
        if self.lstm_backend == "torch":
            LSTM = nn.LSTM
        else:
            from torchrl.modules.tensordict_module.rnn import LSTM
        self.lstm = LSTM(device=device, **lstm_kwargs)
        self.linear = nn.LazyLinear(out_features, device=device)

    def _lstm(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor | None = None,
        hidden1_in: torch.Tensor | None = None,
    ):
        squeeze0 = False
        squeeze1 = False
        if input.ndimension() == 1:
            squeeze0 = True
            input = input.unsqueeze(0).contiguous()

        if input.ndimension() == 2:
            squeeze1 = True
            input = input.unsqueeze(1).contiguous()
        batch, steps = input.shape[:2]

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps) if not squeeze1 else (batch,)
            hidden0_in, hidden1_in = (
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=input.device,
                    dtype=input.dtype,
                )
                for _ in range(2)
            )
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )
        elif squeeze0:
            hidden0_in = hidden0_in.unsqueeze(0)
            hidden1_in = hidden1_in.unsqueeze(0)

        # we only need the first hidden state
        if not squeeze1:
            _hidden0_in = hidden0_in[:, 0]
            _hidden1_in = hidden1_in[:, 0]
        else:
            _hidden0_in = hidden0_in
            _hidden1_in = hidden1_in
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        y0, hidden = self.lstm(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)
        y = self.linear(y0)

        out = [y, hidden0_in, hidden1_in, *hidden]
        if squeeze1:
            # squeezes time
            out[0] = out[0].squeeze(1)
        if not squeeze1:
            # we pad the hidden states with zero to make tensordict happy
            for i in range(3, 5):
                out[i] = torch.stack(
                    [torch.zeros_like(out[i]) for _ in range(input.shape[1] - 1)]
                    + [out[i]],
                    1,
                )
        if squeeze0:
            out = [_out.squeeze(0) for _out in out]
        return tuple(out)

    def forward(
        self,
        input: torch.Tensor,
        hidden0_in: torch.Tensor | None = None,
        hidden1_in: torch.Tensor | None = None,
    ):
        input = self.mlp(input)
        return self._lstm(input, hidden0_in, hidden1_in)


def _call_value_nets(
    value_net: TensorDictModuleBase,
    data: TensorDictBase,
    params: TensorDictBase,
    next_params: TensorDictBase,
    single_call: bool,
    value_key: NestedKey,
    detach_next: bool,
    vmap_randomness: str = "error",
):
    in_keys = value_net.in_keys
    if single_call:
        for i, name in enumerate(data.names):
            if name == "time":
                ndim = i + 1
                break
        else:
            ndim = None
        if ndim is not None:
            # get data at t and last of t+1
            idx0 = (slice(None),) * (ndim - 1) + (slice(-1, None),)
            idx = (slice(None),) * (ndim - 1) + (slice(None, -1),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(1, None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False)[idx0],
                ],
                ndim - 1,
            )
        else:
            if RL_WARNINGS:
                warnings.warn(
                    "Got a tensordict without a time-marked dimension, assuming time is along the last dimension. "
                    "This warning can be turned off by setting the environment variable RL_WARNINGS to False."
                )
            ndim = data.ndim
            idx = (slice(None),) * (ndim - 1) + (slice(None, data.shape[ndim - 1]),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(data.shape[ndim - 1], None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False),
                ],
                ndim - 1,
            )

        # next_params should be None or be identical to params
        if next_params is not None and next_params is not params:
            raise ValueError(
                "the value at t and t+1 cannot be retrieved in a single call without recurring to vmap when both params and next params are passed."
            )
        if params is not None:
            with params.to_module(value_net):
                value_est = value_net(data_in).get(value_key)
        else:
            value_est = value_net(data_in).get(value_key)
        value, value_ = value_est[idx], value_est[idx_]
    else:
        data_in = torch.stack(
            [
                data.select(*in_keys, value_key, strict=False),
                data.get("next").select(*in_keys, value_key, strict=False),
            ],
            0,
        )
        if (params is not None) ^ (next_params is not None):
            raise ValueError(
                "params and next_params must be either both provided or not."
            )
        elif params is not None:
            params_stack = torch.stack([params, next_params], 0).contiguous()
            data_out = _vmap_func(value_net, (0, 0), randomness=vmap_randomness)(
                data_in, params_stack
            )
        else:
            data_out = vmap(value_net, (0,), randomness=vmap_randomness)(data_in)
        value_est = data_out.get(value_key)
        value, value_ = value_est[0], value_est[1]
    data.set(value_key, value)
    data.set(("next", value_key), value_)
    if detach_next:
        value_ = value_.detach()
    return value, value_
