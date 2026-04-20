# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import torch

from _transforms_common import _has_mujoco, TORCH_VERSION, TransformBase
from packaging import version
from tensordict import TensorDict
from tensordict.utils import assert_allclose_td
from torch import nn

from torchrl.data import (
    Bounded,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    SerialEnv,
    SqueezeTransform,
    Stack,
    TransformedEnv,
    UnityMLAgentsEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.libs.unity_mlagents import _has_unity_mlagents
from torchrl.envs.transforms.transforms import _has_tv
from torchrl.envs.utils import check_env_specs, MarlGroupMapType

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
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    MultiAgentCountingEnv,
)


class TestStack(TransformBase):
    def test_single_trans_env_check(self):
        t = Stack(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            t = Stack(
                in_keys=["observation", "observation_orig"],
                out_key="observation_out",
                dim=-1,
                del_keys=False,
            )
            return TransformedEnv(ContinuousActionVecMockEnv(), t)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            t = Stack(
                in_keys=["observation", "observation_orig"],
                out_key="observation_out",
                dim=-1,
                del_keys=False,
            )
            return TransformedEnv(ContinuousActionVecMockEnv(), t)

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        t = Stack(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-2,
            del_keys=False,
        )

        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), t)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        t = Stack(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-2,
            del_keys=False,
        )

        env = TransformedEnv(maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), t)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("del_keys", [True, False])
    def test_transform_del_keys(self, del_keys):
        td_orig = TensorDict(
            {
                "group_0": TensorDict(
                    {
                        "agent_0": TensorDict({"obs": torch.randn(10)}),
                        "agent_1": TensorDict({"obs": torch.randn(10)}),
                    }
                ),
                "group_1": TensorDict(
                    {
                        "agent_2": TensorDict({"obs": torch.randn(10)}),
                        "agent_3": TensorDict({"obs": torch.randn(10)}),
                    }
                ),
            }
        )
        t = Stack(
            in_keys=[
                ("group_0", "agent_0", "obs"),
                ("group_0", "agent_1", "obs"),
                ("group_1", "agent_2", "obs"),
                ("group_1", "agent_3", "obs"),
            ],
            out_key="observations",
            del_keys=del_keys,
        )
        td = td_orig.clone()
        t(td)
        keys = td.keys(include_nested=True)
        if del_keys:
            assert ("group_0",) not in keys
            assert ("group_0", "agent_0", "obs") not in keys
            assert ("group_0", "agent_1", "obs") not in keys
            assert ("group_1", "agent_2", "obs") not in keys
            assert ("group_1", "agent_3", "obs") not in keys
        else:
            assert ("group_0", "agent_0", "obs") in keys
            assert ("group_0", "agent_1", "obs") in keys
            assert ("group_1", "agent_2", "obs") in keys
            assert ("group_1", "agent_3", "obs") in keys

        assert ("observations",) in keys

    def _test_transform_no_env_tensor(self, compose=False):
        td_orig = TensorDict(
            {
                "key1": torch.rand(1, 3),
                "key2": torch.rand(1, 3),
                "key3": torch.rand(1, 3),
            },
            [1],
        )
        td = td_orig.clone()
        t = Stack(
            in_keys=[("key1",), ("key2",)],
            out_key=("stacked",),
            in_key_inv=("stacked",),
            out_keys_inv=[("key1",), ("key2",)],
            dim=-2,
        )
        if compose:
            t = Compose(t)

        td = t(td)

        assert ("key1",) not in td.keys()
        assert ("key2",) not in td.keys()
        assert ("key3",) in td.keys()
        assert ("stacked",) in td.keys()

        assert td["stacked"].shape == torch.Size([1, 2, 3])
        assert (td["stacked"][:, 0] == td_orig["key1"]).all()
        assert (td["stacked"][:, 1] == td_orig["key2"]).all()

        td = t.inv(td)
        assert (td == td_orig).all()

    def _test_transform_no_env_tensordict(self, compose=False):
        def gen_value():
            return TensorDict(
                {
                    "a": torch.rand(3),
                    "b": torch.rand(2, 4),
                }
            )

        td_orig = TensorDict(
            {
                "key1": gen_value(),
                "key2": gen_value(),
                "key3": gen_value(),
            },
            [],
        )
        td = td_orig.clone()
        t = Stack(
            in_keys=[("key1",), ("key2",)],
            out_key=("stacked",),
            in_key_inv=("stacked",),
            out_keys_inv=[("key1",), ("key2",)],
            dim=0,
            allow_positive_dim=True,
        )
        if compose:
            t = Compose(t)
        td = t(td)

        assert ("key1",) not in td.keys()
        assert ("key2",) not in td.keys()
        assert ("stacked", "a") in td.keys(include_nested=True)
        assert ("stacked", "b") in td.keys(include_nested=True)
        assert ("key3",) in td.keys()

        assert td["stacked", "a"].shape == torch.Size([2, 3])
        assert td["stacked", "b"].shape == torch.Size([2, 2, 4])
        assert (td["stacked"][0] == td_orig["key1"]).all()
        assert (td["stacked"][1] == td_orig["key2"]).all()
        assert (td["key3"] == td_orig["key3"]).all()

        td = t.inv(td)
        assert (td == td_orig).all()

    @pytest.mark.parametrize("datatype", ["tensor", "tensordict"])
    def test_transform_no_env(self, datatype):
        if datatype == "tensor":
            self._test_transform_no_env_tensor()

        elif datatype == "tensordict":
            self._test_transform_no_env_tensordict()

        else:
            raise RuntimeError(f"please add a test case for datatype {datatype}")

    @pytest.mark.parametrize("datatype", ["tensor", "tensordict"])
    def test_transform_compose(self, datatype):
        if datatype == "tensor":
            self._test_transform_no_env_tensor(compose=True)

        elif datatype == "tensordict":
            self._test_transform_no_env_tensordict(compose=True)

        else:
            raise RuntimeError(f"please add a test case for datatype {datatype}")

    @pytest.mark.parametrize("envtype", ["mock", "unity"])
    def test_transform_env(self, envtype):
        if envtype == "mock":
            base_env = MultiAgentCountingEnv(
                n_agents=5,
            )
            rollout_len = 6
            t = Stack(
                in_keys=[
                    ("agents", "agent_0"),
                    ("agents", "agent_2"),
                    ("agents", "agent_3"),
                ],
                out_key="stacked_agents",
                in_key_inv="stacked_agents",
                out_keys_inv=[
                    ("agents", "agent_0"),
                    ("agents", "agent_2"),
                    ("agents", "agent_3"),
                ],
            )

        elif envtype == "unity":
            if not _has_unity_mlagents:
                raise pytest.skip("mlagents not installed")
            base_env = UnityMLAgentsEnv(
                registered_name="3DBall",
                no_graphics=True,
                group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
            )
            rollout_len = 200
            t = Stack(
                in_keys=[("agents", f"agent_{idx}") for idx in range(12)],
                out_key="stacked_agents",
                in_key_inv="stacked_agents",
                out_keys_inv=[("agents", f"agent_{idx}") for idx in range(12)],
            )

        try:
            env = TransformedEnv(base_env, t)
            check_env_specs(env)

            if envtype == "mock":
                base_env.set_seed(123)
            td_orig = base_env.reset()
            if envtype == "mock":
                env.set_seed(123)
            td = env.reset()

            td_keys = td.keys(include_nested=True)

            if envtype == "mock":
                assert ("agents", "agent_0") not in td_keys
                assert ("agents", "agent_2") not in td_keys
                assert ("agents", "agent_3") not in td_keys
                assert ("agents", "agent_1") in td_keys
                assert ("agents", "agent_4") in td_keys
                assert ("stacked_agents",) in td_keys

                assert (td["stacked_agents"][0] == td_orig["agents", "agent_0"]).all()
                assert (td["stacked_agents"][1] == td_orig["agents", "agent_2"]).all()
                assert (td["stacked_agents"][2] == td_orig["agents", "agent_3"]).all()
                assert (td["agents", "agent_1"] == td_orig["agents", "agent_1"]).all()
                assert (td["agents", "agent_4"] == td_orig["agents", "agent_4"]).all()
            else:
                assert ("agents",) not in td_keys
                assert ("stacked_agents",) in td_keys
                assert td["stacked_agents"].shape[0] == 12

                assert ("agents",) not in env.full_action_spec.keys(include_nested=True)
                assert ("stacked_agents",) in env.full_action_spec.keys(
                    include_nested=True
                )

            td = env.step(env.full_action_spec.rand())
            td = env.rollout(rollout_len)

            if envtype == "mock":
                assert td["next", "stacked_agents", "done"].shape == torch.Size(
                    [6, 3, 1]
                )
                assert not (td["next", "stacked_agents", "done"][:-1]).any()
                assert (td["next", "stacked_agents", "done"][-1]).all()
        finally:
            base_env.close()

    def test_transform_model(self):
        t = Stack(
            in_keys=[("next", "observation"), ("observation",)],
            out_key="observation_out",
            dim=-2,
            del_keys=True,
        )
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {("next", "observation"): torch.randn(3), "observation": torch.randn(3)}, []
        )
        td = model(td)
        assert "observation_out" in td.keys()
        assert "observation" not in td.keys()
        assert ("next", "observation") not in td.keys(True)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Stack(
            in_keys=[("next", "observation"), "observation"],
            out_key="observation_out",
            dim=-2,
            del_keys=True,
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {
                "observation": TensorDict({"stuff": torch.randn(3, 4)}, [3, 4]),
                "next": TensorDict(
                    {"observation": TensorDict({"stuff": torch.randn(3, 4)}, [3, 4])},
                    [],
                ),
            },
            [],
        ).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        assert "observation_out" in td.keys()
        assert "observation" not in td.keys()
        assert ("next", "observation") not in td.keys(True)

    def test_transform_inverse(self):
        td_orig = TensorDict(
            {
                "stacked": torch.rand(1, 2, 3),
                "key3": torch.rand(1, 3),
            },
            [1],
        )
        td = td_orig.clone()
        t = Stack(
            in_keys=[("key1",), ("key2",)],
            out_key=("stacked",),
            in_key_inv=("stacked",),
            out_keys_inv=[("key1",), ("key2",)],
            dim=1,
            allow_positive_dim=True,
        )

        td = t.inv(td)

        assert ("key1",) in td.keys()
        assert ("key2",) in td.keys()
        assert ("key3",) in td.keys()
        assert ("stacked",) not in td.keys()
        assert (td["key1"] == td_orig["stacked"][:, 0]).all()
        assert (td["key2"] == td_orig["stacked"][:, 1]).all()

        td = t(td)
        assert (td == td_orig).all()

        # Check that if `out_key` is not in the tensordict,
        # then the inverse transform does nothing.
        t = Stack(
            in_keys=[("key1",), ("key2",)],
            out_key=("sacked",),
            dim=1,
            allow_positive_dim=True,
        )
        td = t.inv(td)
        assert (td == td_orig).all()


class TestCatTensors(TransformBase):
    @pytest.mark.parametrize("append", [True, False])
    def test_cattensors_empty(self, append):
        ct = CatTensors(out_key="observation_out", dim=-1, del_keys=False)
        if append:
            mock_env = TransformedEnv(ContinuousActionVecMockEnv())
            mock_env.append_transform(ct)
        else:
            mock_env = TransformedEnv(ContinuousActionVecMockEnv(), ct)
        tensordict = mock_env.rollout(3)
        assert all(key in tensordict.keys() for key in ["observation_out"])
        # assert not any(key in tensordict.keys() for key in mock_env.base_env.observation_spec)

    def test_single_trans_env_check(self):
        ct = CatTensors(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )
        env = TransformedEnv(ContinuousActionVecMockEnv(), ct)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            ct = CatTensors(
                in_keys=["observation", "observation_orig"],
                out_key="observation_out",
                dim=-1,
                del_keys=False,
            )
            return TransformedEnv(ContinuousActionVecMockEnv(), ct)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            ct = CatTensors(
                in_keys=["observation", "observation_orig"],
                out_key="observation_out",
                dim=-1,
                del_keys=False,
            )
            return TransformedEnv(ContinuousActionVecMockEnv(), ct)

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        ct = CatTensors(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )

        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        ct = CatTensors(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )

        env = TransformedEnv(maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), ct)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", ("some", "other")],
            ["observation_pixels"],
        ],
    )
    @pytest.mark.parametrize("out_key", ["observation_out", ("some", "nested")])
    def test_transform_no_env(self, keys, device, out_key):
        cattensors = CatTensors(in_keys=keys, out_key=out_key, dim=-2)

        dont_touch = torch.randn(1, 3, 3, dtype=torch.double, device=device)
        td = TensorDict(
            {
                key: torch.full(
                    (1, 4, 32),
                    value,
                    dtype=torch.float,
                    device=device,
                )
                for value, key in enumerate(keys)
            },
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())

        tdc = cattensors(td.clone())
        assert tdc.get(out_key).shape[-2] == len(keys) * 4
        assert tdc.get("dont touch").shape == dont_touch.shape

        tdc = cattensors._call(td.clone())
        assert tdc.get(out_key).shape[-2] == len(keys) * 4
        assert tdc.get("dont touch").shape == dont_touch.shape

        if len(keys) == 1:
            observation_spec = Bounded(0, 1, (1, 4, 32))
            observation_spec = cattensors.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([1, len(keys) * 4, 32])
        else:
            observation_spec = Composite(
                {key: Bounded(0, 1, (1, 4, 32)) for key in keys}
            )
            observation_spec = cattensors.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec[out_key].shape == torch.Size([1, len(keys) * 4, 32])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", "observation_other"],
            ["observation_pixels"],
        ],
    )
    def test_transform_compose(self, keys, device):
        cattensors = Compose(
            CatTensors(in_keys=keys, out_key="observation_out", dim=-2)
        )

        dont_touch = torch.randn(1, 3, 3, dtype=torch.double, device=device)
        td = TensorDict(
            {
                key: torch.full(
                    (
                        1,
                        4,
                        32,
                    ),
                    value,
                    dtype=torch.float,
                    device=device,
                )
                for value, key in enumerate(keys)
            },
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())

        tdc = cattensors(td.clone())
        assert tdc.get("observation_out").shape[-2] == len(keys) * 4
        assert tdc.get("dont touch").shape == dont_touch.shape

        tdc = cattensors._call(td.clone())
        assert tdc.get("observation_out").shape[-2] == len(keys) * 4
        assert tdc.get("dont touch").shape == dont_touch.shape

    @pytest.mark.parametrize("del_keys", [True, False])
    @pytest.mark.skipif(not _has_gym, reason="Gym not found")
    @pytest.mark.parametrize("out_key", ["observation_out", ("some", "nested")])
    def test_transform_env(self, del_keys, out_key):
        ct = CatTensors(
            in_keys=[
                "observation",
            ],
            out_key=out_key,
            dim=-1,
            del_keys=del_keys,
        )
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), ct)
        assert env.observation_spec[out_key]
        if del_keys:
            assert "observation" not in env.observation_spec
        else:
            assert "observation" in env.observation_spec

        assert "observation" in env.base_env.observation_spec
        check_env_specs(env)

    def test_transform_model(self):
        ct = CatTensors(
            in_keys=[("next", "observation"), "action"],
            out_key="observation_out",
            dim=-1,
            del_keys=True,
        )
        model = nn.Sequential(ct, nn.Identity())
        td = TensorDict(
            {("next", "observation"): torch.randn(3), "action": torch.randn(2)}, []
        )
        td = model(td)
        assert "observation_out" in td.keys()
        assert "action" not in td.keys()
        assert ("next", "observation") not in td.keys(True)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        ct = CatTensors(
            in_keys=[("next", "observation"), "action"],
            out_key="observation_out",
            dim=-1,
            del_keys=True,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(ct)
        td = (
            TensorDict(
                {("next", "observation"): torch.randn(3), "action": torch.randn(2)}, []
            )
            .expand(10)
            .contiguous()
        )
        rb.extend(td)
        td = rb.sample(10)
        assert "observation_out" in td.keys()
        assert "action" not in td.keys()
        assert ("next", "observation") not in td.keys(True)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for CatTensors")


@pytest.mark.skipif(not _has_tv, reason="no torchvision")
class TestUnsqueezeTransform(TransformBase):
    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, size, nchannels, batch, device, dim):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = UnsqueezeTransform(dim, in_keys=keys, allow_positive_dim=True)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        if dim >= 0 and dim < len(batch):
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                unsqueeze(td)
            return
        unsqueeze(td)
        expected_size = [*batch, *size, nchannels, 16, 16]
        if dim < 0:
            expected_size.insert(len(expected_size) + dim + 1, 1)
        else:
            expected_size.insert(dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys:
            assert td.get(key).shape == expected_size, (
                batch,
                size,
                nchannels,
                dim,
            )
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (*batch, *size, nchannels, 16, 16))
            observation_spec = unsqueeze.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == expected_size
        else:
            observation_spec = Composite(
                {
                    key: Bounded(-1, 1, (*batch, *size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = unsqueeze.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == expected_size

    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys_inv",
        [
            [],
            ["action", ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    def test_unsqueeze_inv(self, keys, keys_inv, size, nchannels, batch, device, dim):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        unsqueeze = UnsqueezeTransform(
            dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )

        td_modif = unsqueeze.inv(td)

        expected_size = [*batch, *size, nchannels, 16, 16]
        for key in keys_total.difference(keys_inv):
            assert td.get(key).shape == torch.Size(expected_size)

        if expected_size[dim] == 1:
            del expected_size[dim]
        for key in keys_inv:
            assert td_modif.get(key).shape == torch.Size(expected_size)
        # for key in keys_inv:
        #     assert td.get(key).shape != torch.Size(expected_size)

    def test_single_trans_env_check(self):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            UnsqueezeTransform(-1, in_keys=["observation"]),
        )
        check_env_specs(env)
        assert "observation" in env.observation_spec.keys()

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                UnsqueezeTransform(-1, in_keys=["observation"]),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                UnsqueezeTransform(-1, in_keys=["observation"]),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            UnsqueezeTransform(-1, in_keys=["observation"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            UnsqueezeTransform(-1, in_keys=["observation"]),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, size, nchannels, batch, device, dim):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = Compose(
            UnsqueezeTransform(dim, in_keys=keys, allow_positive_dim=True)
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        if dim >= 0 and dim < len(batch):
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                unsqueeze(td)
            return
        unsqueeze(td)
        expected_size = [*batch, *size, nchannels, 16, 16]
        if dim < 0:
            expected_size.insert(len(expected_size) + dim + 1, 1)
        else:
            expected_size.insert(dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys:
            assert td.get(key).shape == expected_size, (
                batch,
                size,
                nchannels,
                dim,
            )
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(-1, 1, (*batch, *size, nchannels, 16, 16))
            observation_spec = unsqueeze.transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == expected_size
        else:
            observation_spec = Composite(
                {
                    key: Bounded(-1, 1, (*batch, *size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = unsqueeze.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == expected_size

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            UnsqueezeTransform(-1, in_keys=["observation"], out_keys=out_keys),
        )
        assert "observation" in env.observation_spec.keys()
        if out_keys:
            assert out_keys[0] in env.observation_spec.keys()
            obsshape = list(env.observation_spec["observation"].shape)
            obsshape.insert(len(obsshape), 1)
            assert (
                torch.Size(obsshape) == env.observation_spec[out_keys[0]].rand().shape
            )
        check_env_specs(env)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    @pytest.mark.parametrize("dim", [-1, 1])
    def test_transform_model(self, out_keys, dim):
        t = UnsqueezeTransform(
            dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        td = TensorDict(
            {"observation": TensorDict({"stuff": torch.randn(3, 4)}, [3, 4])}, []
        )
        t(td)
        expected_shape = [3, 4]
        if dim >= 0:
            expected_shape.insert(dim, 1)
        else:
            expected_shape.insert(len(expected_shape) + dim + 1, 1)
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    @pytest.mark.parametrize("dim", [-1, 1])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass, out_keys, dim):
        t = UnsqueezeTransform(
            dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {"observation": TensorDict({"stuff": torch.randn(3, 4)}, [3, 4])}, []
        ).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        expected_shape = [2, 3, 4]
        if dim >= 0:
            expected_shape.insert(dim, 1)
        else:
            expected_shape.insert(len(expected_shape) + dim + 1, 1)
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    @pytest.mark.skipif(not _has_gym, reason="No gym")
    def test_transform_inverse(self):
        if not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        env = TransformedEnv(
            GymEnv(HALFCHEETAH_VERSIONED()),
            # the order is inverted
            Compose(
                UnsqueezeTransform(
                    -1, in_keys_inv=["action"], out_keys_inv=["action_t"]
                ),
                SqueezeTransform(-1, in_keys_inv=["action_t"], out_keys_inv=["action"]),
            ),
        )
        td = env.rollout(3)
        assert env.full_action_spec["action"].shape[-1] == 6
        assert td["action"].shape[-1] == 6


class TestSqueezeTransform(TransformBase):
    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys_inv",
        [
            [],
            ["action", ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    def test_transform_no_env(
        self, keys, keys_inv, size, nchannels, batch, device, dim
    ):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(
            dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )
        squeeze(td)

        expected_size = [*batch, *size, nchannels, 16, 16]
        for key in keys_total.difference(keys):
            assert td.get(key).shape == torch.Size(expected_size)

        if expected_size[dim] == 1:
            del expected_size[dim]
        for key in keys:
            assert td.get(key).shape == torch.Size(expected_size)

    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys_inv",
        [
            [],
            ["action", ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    def test_squeeze_inv(self, keys, keys_inv, size, nchannels, batch, device, dim):
        torch.manual_seed(0)
        if dim >= 0:
            dim = dim + len(batch)
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(
            dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )
        td = squeeze.inv(td)

        expected_size = [*batch, *size, nchannels, 16, 16]
        for key in keys_total.difference(keys_inv):
            assert td.get(key).shape == torch.Size(expected_size)

        if dim < 0:
            expected_size.insert(len(expected_size) + dim + 1, 1)
        else:
            expected_size.insert(dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys_inv:
            assert td.get(key).shape == torch.Size(expected_size), dim

    @property
    def _circular_transform(self):
        return Compose(
            UnsqueezeTransform(
                -1, in_keys=["observation"], out_keys=["observation_un"]
            ),
            SqueezeTransform(
                -1, in_keys=["observation_un"], out_keys=["observation_sq"]
            ),
        )

    @property
    def _inv_circular_transform(self):
        return Compose(
            # The env wants a squeezed action - the inv of unsqueeze
            UnsqueezeTransform(-1, in_keys_inv=["action"], out_keys_inv=["action_un"]),
            # The outsize world has an squeezed action that we unsqueeze - the inv of squeeze
            SqueezeTransform(-1, in_keys_inv=["action_un"], out_keys_inv=["action"]),
        )

    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), self._circular_transform)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), self._circular_transform
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), self._circular_transform
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), self._circular_transform
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            self._circular_transform,
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys_inv", [[], ["action", "some_other_key"], [("next", "observation_pixels")]]
    )
    def test_transform_compose(
        self, keys, keys_inv, size, nchannels, batch, device, dim
    ):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        squeeze = Compose(
            SqueezeTransform(
                dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
            )
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )
        squeeze(td)

        expected_size = [*batch, *size, nchannels, 16, 16]
        for key in keys_total.difference(keys):
            assert td.get(key).shape == torch.Size(expected_size)

        if expected_size[dim] == 1:
            del expected_size[dim]
        for key in keys:
            assert td.get(key).shape == torch.Size(expected_size)

    @pytest.mark.parametrize(
        "keys_inv", [[], ["action", "some_other_key"], [("next", "observation_pixels")]]
    )
    def test_transform_env(self, keys_inv):
        env = TransformedEnv(ContinuousActionVecMockEnv(), self._circular_transform)
        r = env.rollout(3)
        assert "observation" in r.keys()
        assert "observation_un" in r.keys()
        assert "observation_sq" in r.keys()
        assert (r["observation_sq"] == r["observation"]).all()

    @pytest.mark.parametrize("out_keys", [None, ["obs_sq"]])
    def test_transform_model(self, out_keys):
        dim = 1
        t = SqueezeTransform(
            dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {"observation": TensorDict({"stuff": torch.randn(3, 1, 4)}, [3, 1, 4])}, []
        )
        model(td)
        expected_shape = [3, 4]
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.parametrize("out_keys", [None, ["obs_sq"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, out_keys, rbclass):
        dim = -2
        t = SqueezeTransform(
            dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {"observation": TensorDict({"stuff": torch.randn(3, 1, 4)}, [3, 1, 4])}, []
        ).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        expected_shape = [2, 3, 4]
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    def test_transform_inverse(self):
        if not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        env = TransformedEnv(
            GymEnv(HALFCHEETAH_VERSIONED()), self._inv_circular_transform
        )
        check_env_specs(env)
        r = env.rollout(3)
        r2 = GymEnv(HALFCHEETAH_VERSIONED()).rollout(3)
        assert_allclose_td(r.zero_(), r2.zero_(), intersection=True)


class TestDoubleToFloat(TransformBase):
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", ("some_other", "nested_key")],
            ["observation_pixels"],
            ["action"],
        ],
    )
    @pytest.mark.parametrize(
        "keys_inv",
        [
            ["action", ("some_other", "nested_key")],
            ["action"],
            [],
        ],
    )
    def test_double2float(self, keys, keys_inv, device):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        double2float = DoubleToFloat(in_keys=keys, in_keys_inv=keys_inv)
        dont_touch = torch.randn(1, 3, 3, dtype=torch.double, device=device)
        td = TensorDict(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.double, device=device)
                for key in keys_total
            },
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        # check that the transform does change the dtype in forward
        double2float(td)
        for key in keys:
            assert td.get(key).dtype == torch.float
        assert td.get("dont touch").dtype == torch.double

        # check that inv does not affect the tensordict in-place
        td = td.apply(lambda x: x.float())
        td_modif = double2float.inv(td)
        for key in keys_inv:
            assert td.get(key).dtype != torch.double
            assert td_modif.get(key).dtype == torch.double
        assert td.get("dont touch").dtype != torch.double

        if len(keys_total) == 1 and len(keys_inv) and keys[0] == "action":
            action_spec = Bounded(0, 1, (1, 3, 3), dtype=torch.double)
            input_spec = Composite(
                full_action_spec=Composite(action=action_spec), full_state_spec=None
            )
            action_spec = double2float.transform_input_spec(input_spec)
            assert action_spec.dtype == torch.float
        else:
            observation_spec = Composite(
                {key: Bounded(0, 1, (1, 3, 3), dtype=torch.double) for key in keys}
            )
            observation_spec = double2float.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].dtype == torch.float, key

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", ("some_other", "nested_key")],
            ["observation_pixels"],
            ["action"],
        ],
    )
    @pytest.mark.parametrize(
        "keys_inv",
        [
            ["action", ("some_other", "nested_key")],
            ["action"],
            [],
        ],
    )
    def test_double2float_auto(self, keys, keys_inv, device):
        torch.manual_seed(0)
        double2float = DoubleToFloat()
        d = {
            key: torch.zeros(1, 3, 3, dtype=torch.double, device=device) for key in keys
        }
        d.update(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.float32, device=device)
                for key in keys_inv
            }
        )
        td = TensorDict(d, [1], device=device)
        # check that the transform does change the dtype in forward
        double2float(td)
        for key in keys:
            assert td.get(key).dtype == torch.float

        # check that inv does not affect the tensordict in-place
        td = td.apply(lambda x: x.float())
        td_modif = double2float.inv(td)
        for key in keys_inv:
            assert td.get(key).dtype != torch.double
            assert td_modif.get(key).dtype == torch.double

    def test_single_env_no_inkeys(self):
        base_env = ContinuousActionVecMockEnv(spec_locked=False)
        for key, spec in list(base_env.observation_spec.items(True, True)):
            base_env.observation_spec[key] = spec.to(torch.float64)
        for key, spec in list(base_env.state_spec.items(True, True)):
            base_env.state_spec[key] = spec.to(torch.float64)
        if base_env.action_spec.dtype == torch.float32:
            base_env.action_spec = base_env.action_spec.to(torch.float64)
        check_env_specs(base_env)
        env = TransformedEnv(
            base_env,
            DoubleToFloat(),
            spec_locked=False,
        )
        for spec in env.observation_spec.values(True, True):
            assert spec.dtype == torch.float32
        for spec in env.state_spec.values(True, True):
            assert spec.dtype == torch.float32
        assert env.action_spec.dtype != torch.float64
        assert env.transform.in_keys == env.transform.out_keys
        assert env.transform.in_keys_inv == env.transform.out_keys_inv
        check_env_specs(env)

    def test_single_trans_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            ContinuousActionVecMockEnv(dtype=torch.float64),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self, dtype_fixture):  # noqa: F811
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(dtype=torch.float64),
                DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(
        self,
        dtype_fixture,  # noqa: F811
        maybe_fork_ParallelEnv,
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(dtype=torch.float64),
                DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
            )

        try:
            env = maybe_fork_ParallelEnv(1, make_env)
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass
            del env

    def test_trans_serial_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv(dtype=torch.float64)),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(
        self,
        dtype_fixture,  # noqa: F811
        maybe_fork_ParallelEnv,
    ):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(
                2, lambda: ContinuousActionVecMockEnv(dtype=torch.float64)
            ),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self, dtype_fixture):  # noqa: F811
        t = DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"])
        td = TensorDict(
            {"observation": torch.randn(10, 4, 5)},
            [10, 4],
        )
        assert td["observation"].dtype is torch.double
        out = t._call(td)
        assert out["observation"].dtype is torch.float

    def test_transform_inverse(
        self,
    ):
        t = DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"])
        td = TensorDict(
            {"action": torch.randn(10, 4, 5)},
            [10, 4],
        )
        assert td["action"].dtype is torch.float
        out = t.inv(td)
        assert out["action"].dtype is torch.double

    def test_transform_compose(self, dtype_fixture):  # noqa: F811
        t = Compose(DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]))
        td = TensorDict(
            {"observation": torch.randn(10, 4, 5)},
            [10, 4],
        )
        assert td["observation"].dtype is torch.double
        out = t._call(td)
        assert out["observation"].dtype is torch.float

    def test_transform_compose_invserse(
        self,
    ):
        t = Compose(DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]))
        td = TensorDict(
            {"action": torch.randn(10, 4, 5)},
            [10, 4],
        )
        assert td["action"].dtype is torch.float
        out = t.inv(td)
        assert out["action"].dtype is torch.double

    def test_transform_env(self, dtype_fixture):  # noqa: F811
        raise pytest.skip("Tested in test_transform_inverse")

    def test_transform_model(self, dtype_fixture):  # noqa: F811
        t = DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"])
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {"observation": torch.randn(10, 4, 5)},
            [10, 4],
        )
        assert td["observation"].dtype is torch.double
        td = model(td)
        assert td["observation"].dtype is torch.float

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        rb = rbclass(storage=LazyTensorStorage(10))
        t = DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"])
        rb.append_transform(t)
        td = TensorDict(
            {
                "observation": torch.randn(10, 4, 5, dtype=torch.double),
                "action": torch.randn(10, 4, 5),
            },
            [10, 4],
        )
        assert td["observation"].dtype is torch.double
        assert td["action"].dtype is torch.float
        rb.extend(td)
        storage = rb._storage[:]
        # observation is not part of in_keys_inv
        assert storage["observation"].dtype is torch.double
        # action is part of in_keys_inv
        assert storage["action"].dtype is torch.double
        td = rb.sample(10)
        assert td["observation"].dtype is torch.float
        assert td["action"].dtype is torch.double
