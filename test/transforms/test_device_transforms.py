# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import torch

from _transforms_common import mp_ctx, TransformBase
from tensordict import TensorDict
from torch import nn

from torchrl.data import (
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorStorage,
)
from torchrl.envs import (
    Compose,
    DeviceCastTransform,
    ParallelEnv,
    SerialEnv,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv


class TestDeviceCastTransformPart(TransformBase):
    @pytest.fixture(scope="class")
    def _cast_device(self):
        if torch.cuda.is_available():
            yield torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            yield torch.device("mps:0")
        else:
            yield torch.device("cpu:1")

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_single_trans_env_check(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        env = ContinuousActionVecMockEnv(device="cpu:0")
        env = TransformedEnv(
            env,
            DeviceCastTransform(
                _cast_device,
                in_keys=in_keys,
                out_keys=out_keys,
                in_keys_inv=in_keys_inv,
                out_keys_inv=out_keys_inv,
            ),
        )
        assert env.device is None
        check_env_specs(env)

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_serial_trans_env_check(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(device="cpu:0"),
                DeviceCastTransform(
                    _cast_device,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    in_keys_inv=in_keys_inv,
                    out_keys_inv=out_keys_inv,
                ),
            )

        env = SerialEnv(2, make_env)
        assert env.device is None
        check_env_specs(env)

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_parallel_trans_env_check(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(device="cpu:0"),
                DeviceCastTransform(
                    _cast_device,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    in_keys_inv=in_keys_inv,
                    out_keys_inv=out_keys_inv,
                ),
            )

        env = ParallelEnv(
            2,
            make_env,
            mp_start_method=mp_ctx if not torch.cuda.is_available() else "spawn",
        )
        assert env.device is None
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_trans_serial_env_check(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(
            SerialEnv(2, make_env),
            DeviceCastTransform(
                _cast_device,
                in_keys=in_keys,
                out_keys=out_keys,
                in_keys_inv=in_keys_inv,
                out_keys_inv=out_keys_inv,
            ),
        )
        assert env.device is None
        check_env_specs(env)

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_trans_parallel_env_check(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(
            ParallelEnv(
                2,
                make_env,
                mp_start_method=mp_ctx if not torch.cuda.is_available() else "spawn",
            ),
            DeviceCastTransform(
                _cast_device,
                in_keys=in_keys,
                out_keys=out_keys,
                in_keys_inv=in_keys_inv,
                out_keys_inv=out_keys_inv,
            ),
        )
        assert env.device is None
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self, _cast_device):
        t = DeviceCastTransform(_cast_device, "cpu:0", in_keys=["a"], out_keys=["b"])
        td = TensorDict({"a": torch.randn((), device="cpu:0")}, [], device="cpu:0")
        tdt = t._call(td)
        assert tdt.device is None

    @pytest.mark.parametrize("in_keys", ["observation"])
    @pytest.mark.parametrize("out_keys", [None, ["obs_device"]])
    @pytest.mark.parametrize("in_keys_inv", ["action"])
    @pytest.mark.parametrize("out_keys_inv", [None, ["action_device"]])
    def test_transform_env(
        self, in_keys, out_keys, in_keys_inv, out_keys_inv, _cast_device
    ):
        env = ContinuousActionVecMockEnv(device="cpu:0")
        env = TransformedEnv(
            env,
            DeviceCastTransform(
                _cast_device,
                in_keys=in_keys,
                out_keys=out_keys,
                in_keys_inv=in_keys_inv,
                out_keys_inv=out_keys_inv,
            ),
        )
        assert env.device is None
        assert env.transform.device == _cast_device
        assert env.transform.orig_device == torch.device("cpu:0")

    def test_transform_compose(self, _cast_device):
        t = Compose(
            DeviceCastTransform(
                _cast_device,
                "cpu:0",
                in_keys=["a"],
                out_keys=["b"],
                in_keys_inv=["c"],
                out_keys_inv=["d"],
            )
        )

        td = TensorDict(
            {
                "a": torch.randn((), device="cpu:0"),
                "c": torch.randn((), device=_cast_device),
            },
            [],
            device="cpu:0",
        )
        tdt = t._call(td)
        tdit = t._inv_call(td)

        assert tdt.device is None
        assert tdit.device is None

    def test_transform_model(self, _cast_device):
        t = nn.Sequential(
            Compose(
                DeviceCastTransform(
                    _cast_device,
                    "cpu:0",
                    in_keys=["a"],
                    out_keys=["b"],
                    in_keys_inv=["c"],
                    out_keys_inv=["d"],
                )
            )
        )
        td = TensorDict(
            {
                "a": torch.randn((), device="cpu:0"),
                "c": torch.randn((), device="cpu:1"),
            },
            [],
            device="cpu:0",
        )
        tdt = t(td)

        assert tdt.device is None

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("storage", [LazyTensorStorage])
    def test_transform_rb(self, rbclass, storage, _cast_device):
        # we don't test casting to cuda on Memmap tensor storage since it's discouraged
        t = Compose(
            DeviceCastTransform(
                _cast_device,
                "cpu:0",
                in_keys=["a"],
                out_keys=["b"],
                in_keys_inv=["c"],
                out_keys_inv=["d"],
            )
        )
        rb = rbclass(storage=storage(max_size=20, device="auto"))
        rb.append_transform(t)
        td = TensorDict(
            {
                "a": torch.randn((), device="cpu:0"),
                "c": torch.randn((), device=_cast_device),
            },
            [],
            device="cpu:0",
        )
        rb.add(td)
        assert rb._storage._storage.device is None
        assert rb.sample(4).device is None

    def test_transform_inverse(self):
        # Tested before
        return


class TestDeviceCastTransformWhole(TransformBase):
    def test_single_trans_env_check(self):
        env = ContinuousActionVecMockEnv(device="cpu:0")
        env = TransformedEnv(env, DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(device="cpu:0"), DeviceCastTransform("cpu:1")
            )

        env = SerialEnv(2, make_env)
        assert env.device == torch.device("cpu:1")
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(device="cpu:0"), DeviceCastTransform("cpu:1")
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        assert env.device == torch.device("cpu:1")
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(SerialEnv(2, make_env), DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, make_env), DeviceCastTransform("cpu:1")
        )
        assert env.device == torch.device("cpu:1")
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        t = DeviceCastTransform("cpu:1", "cpu:0")
        assert t._call(TensorDict(device="cpu:0")).device == torch.device("cpu:1")

    def test_transform_compose(self):
        t = Compose(DeviceCastTransform("cpu:1", "cpu:0"))
        assert t._call(TensorDict(device="cpu:0")).device == torch.device("cpu:1")
        assert t._inv_call(TensorDict(device="cpu:1")).device == torch.device("cpu:0")

    def test_transform_env(self):
        env = ContinuousActionVecMockEnv(device="cpu:0")
        assert env.device == torch.device("cpu:0")
        env = TransformedEnv(env, DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        assert env.transform.device == torch.device("cpu:1")
        assert env.transform.orig_device == torch.device("cpu:0")

    def test_transform_model(self):
        t = Compose(DeviceCastTransform("cpu:1", "cpu:0"))
        nn.Sequential(t)
        assert t(TensorDict(device="cpu:0")).device == torch.device("cpu:1")

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("storage", [TensorStorage, LazyTensorStorage])
    def test_transform_rb(self, rbclass, storage):
        # we don't test casting to cuda on Memmap tensor storage since it's discouraged
        t = Compose(DeviceCastTransform("cpu:1", "cpu:0"))
        storage_kwargs = (
            {
                "storage": TensorDict(
                    {"a": torch.zeros(20, 1, device="cpu:0")}, [20], device="cpu:0"
                )
            }
            if storage is TensorStorage
            else {}
        )
        rb = rbclass(storage=storage(max_size=20, device="auto", **storage_kwargs))
        rb.append_transform(t)
        rb.add(TensorDict({"a": [1]}, [], device="cpu:1"))
        assert rb._storage._storage.device == torch.device("cpu:0")
        assert rb.sample(4).device == torch.device("cpu:1")

    def test_transform_inverse(self):
        t = DeviceCastTransform("cpu:1", "cpu:0")
        assert t._inv_call(TensorDict(device="cpu:1")).device == torch.device("cpu:0")
