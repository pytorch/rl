# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import importlib.util
from sys import platform

import torch
from packaging import version

_has_ray = importlib.util.find_spec("ray") is not None
_has_ale = importlib.util.find_spec("ale_py") is not None
_has_mujoco = importlib.util.find_spec("mujoco") is not None

IS_WIN = platform == "win32"
if IS_WIN:
    mp_ctx = "spawn"
else:
    mp_ctx = "fork"

TIMEOUT = 100.0
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None


class TransformBase:
    """A base class for transform test.

    We ask for every new transform tests to be coded following this minimum requirement class.

    Of course, specific behaviors can also be tested separately.

    If your transform identifies an issue with the EnvBase or _BatchedEnv abstraction(s),
    this needs to be corrected independently.

    """

    @abc.abstractmethod
    def test_single_trans_env_check(self):
        """tests that a transformed env passes the check_env_specs test.

        If your transform can overwrite a key or create a new entry in the tensordict,
        it is worth trying both options here.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def test_serial_trans_env_check(self):
        """tests that a serial transformed env (SerialEnv(N, lambda: TransformedEnv(env, transform))) passes the check_env_specs test."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_parallel_trans_env_check(self):
        """tests that a parallel transformed env (ParallelEnv(N, lambda: TransformedEnv(env, transform))) passes the check_env_specs test."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_trans_serial_env_check(self):
        """tests that a transformed serial env (TransformedEnv(SerialEnv(N, lambda: env()), transform)) passes the check_env_specs test."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_trans_parallel_env_check(self):
        """tests that a transformed paprallel env (TransformedEnv(ParallelEnv(N, lambda: env()), transform)) passes the check_env_specs test."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_no_env(self):
        """tests the transform on dummy data, without an env."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_compose(self):
        """tests the transform on dummy data, without an env but inside a Compose."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_env(self):
        """tests the transform on a real env.

        If possible, do not use a mock env, as bugs may go unnoticed if the dynamic is too
        simplistic. A call to reset() and step() should be tested independently, ie
        a check that reset produces the desired output and that step() does too.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_model(self):
        """tests the transform before an nn.Module that reads the output."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_rb(self):
        """tests the transform when used with a replay buffer.

        If your transform is not supposed to work with a replay buffer, test that
        an error will be raised when called or appended to a RB.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def test_transform_inverse(self):
        """tests the inverse transform. If not applicable, simply skip this test.

        If your transform is not supposed to work offline, test that
        an error will be raised when called in a nn.Module.
        """
        raise NotImplementedError
