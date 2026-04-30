# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from functools import partial

import numpy as np

import pytest

import torch

from _transforms_common import _has_transformers, TransformBase
from tensordict import NonTensorData, NonTensorStack, TensorDict, TensorDictBase
from torch import nn

from torchrl.data import (
    Categorical,
    Composite,
    LazyTensorStorage,
    NonTensor,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorSpec,
    Unbounded,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    EnvBase,
    ExcludeTransform,
    Hash,
    RemoveEmptySpecs,
    RenameTransform,
    SelectTransform,
    SerialEnv,
    Tokenizer,
    TransformedEnv,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms.transforms import Transform
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
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingEnv,
    CountingEnvWithString,
    NestedCountingEnv,
)


class TestExcludeTransform(TransformBase):
    class EnvWithManyKeys(EnvBase):
        def __init__(self):
            super().__init__()
            self.observation_spec = Composite(
                a=Unbounded(3),
                b=Unbounded(3),
                c=Unbounded(3),
            )
            self.reward_spec = Unbounded(1)
            self.action_spec = Unbounded(2)

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            return self.observation_spec.rand().update(
                {
                    "reward": self.reward_spec.rand(),
                    "done": torch.zeros(1, dtype=torch.bool),
                }
            )

        def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
            return self.observation_spec.rand().update(
                {"done": torch.zeros(1, dtype=torch.bool)}
            )

        def _set_seed(self, seed: int | None) -> None:
            ...

    def test_single_trans_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            ExcludeTransform("observation_copy"),
        )
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                ExcludeTransform("observation_copy"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                ExcludeTransform("observation_copy"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            ExcludeTransform("observation_copy"),
        )
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), t)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            ExcludeTransform("observation_copy"),
        )
        env = TransformedEnv(maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), t)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_env(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, ExcludeTransform("a"))
        assert "a" not in env.reset().keys()
        assert "b" in env.reset().keys()
        assert "c" in env.reset().keys()

    def test_exclude_done(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, ExcludeTransform("a", "done"))
        assert "done" not in env.done_keys
        check_env_specs(env)
        env = TransformedEnv(base_env, ExcludeTransform("a"))
        assert "done" in env.done_keys
        check_env_specs(env)

    def test_exclude_reward(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, ExcludeTransform("a", "reward"))
        assert "reward" not in env.reward_keys
        check_env_specs(env)
        env = TransformedEnv(base_env, ExcludeTransform("a"))
        assert "reward" in env.reward_keys
        check_env_specs(env)

    @pytest.mark.parametrize("nest_done", [True, False])
    @pytest.mark.parametrize("nest_reward", [True, False])
    def test_nested(self, nest_reward, nest_done):
        env = NestedCountingEnv(
            nest_reward=nest_reward,
            nest_done=nest_done,
        )
        transformed_env = TransformedEnv(env, ExcludeTransform())
        td = transformed_env.rollout(1)
        td_keys = td.keys(True, True)
        assert ("next", env.reward_key) in td_keys
        for done_key in env.done_keys:
            assert ("next", done_key) in td_keys
            assert done_key in td_keys
        assert env.action_key in td_keys
        assert ("data", "states") in td_keys
        assert ("next", "data", "states") in td_keys

        transformed_env = TransformedEnv(env, ExcludeTransform(("data", "states")))
        td = transformed_env.rollout(1)
        td_keys = td.keys(True, True)
        assert ("next", env.reward_key) in td_keys
        for done_key in env.done_keys:
            assert ("next", done_key) in td_keys
            assert done_key in td_keys
        assert env.action_key in td_keys
        assert ("data", "states") not in td_keys
        assert ("next", "data", "states") not in td_keys

    def test_transform_no_env(self):
        t = ExcludeTransform("a")
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": {
                    "d": torch.randn(1),
                },
            },
            [],
        )
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()
        t = ExcludeTransform("a", ("c", "d"))
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()
        assert ("c", "d") not in td.keys(True, True)
        t = ExcludeTransform("a", "c")
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" not in td.keys()
        assert ("c", "d") not in td.keys(True, True)

    def test_transform_compose(self):
        t = Compose(ExcludeTransform("a"))
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        )
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    def test_transform_model(self):
        t = ExcludeTransform("a")
        t = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        )
        td = t(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = ExcludeTransform("a")
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        ).expand(3)
        rb.extend(td)
        td = rb.sample(4)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    def test_transform_inverse(self):
        raise pytest.skip("no inverse for ExcludeTransform")


class TestSelectTransform(TransformBase):
    class EnvWithManyKeys(EnvBase):
        def __init__(self):
            super().__init__()
            self.observation_spec = Composite(
                a=Unbounded(3),
                b=Unbounded(3),
                c=Unbounded(3),
            )
            self.reward_spec = Unbounded(1)
            self.action_spec = Unbounded(2)

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            return self.observation_spec.rand().update(
                {
                    "reward": self.reward_spec.rand(),
                    "done": torch.zeros(1, dtype=torch.bool),
                }
            )

        def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
            return self.observation_spec.rand().update(
                {"done": torch.zeros(1, dtype=torch.bool)}
            )

        def _set_seed(self, seed: int | None) -> None:
            ...

    def test_single_trans_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            SelectTransform("observation", "observation_orig"),
        )
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                SelectTransform("observation", "observation_orig"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                SelectTransform("observation", "observation_orig"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            SelectTransform("observation", "observation_orig"),
        )
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), t)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            SelectTransform("observation", "observation_orig"),
        )
        env = TransformedEnv(maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), t)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_env(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, SelectTransform("b", "c"))
        assert "a" not in env.reset().keys()
        assert "b" in env.reset().keys()
        assert "c" in env.reset().keys()

    @pytest.mark.parametrize("keep_done", [True, False])
    def test_select_done(self, keep_done):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(
            base_env, SelectTransform("b", "c", "done", keep_dones=keep_done)
        )
        assert "done" in env.done_keys
        check_env_specs(env)
        env = TransformedEnv(base_env, SelectTransform("b", "c", keep_dones=keep_done))
        if keep_done:
            assert "done" in env.done_keys
        else:
            assert "done" not in env.done_keys
        check_env_specs(env)

    @pytest.mark.parametrize("keep_reward", [True, False])
    def test_select_reward(self, keep_reward):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(
            base_env, SelectTransform("b", "c", "reward", keep_rewards=keep_reward)
        )
        assert "reward" in env.reward_keys
        check_env_specs(env)
        env = TransformedEnv(
            base_env, SelectTransform("b", "c", keep_rewards=keep_reward)
        )
        if keep_reward:
            assert "reward" in env.reward_keys
        else:
            assert "reward" not in env.reward_keys
        check_env_specs(env)

    @pytest.mark.parametrize("nest_done", [True, False])
    @pytest.mark.parametrize("nest_reward", [True, False])
    def test_nested(self, nest_reward, nest_done):
        env = NestedCountingEnv(
            nest_reward=nest_reward,
            nest_done=nest_done,
        )
        transformed_env = TransformedEnv(env, SelectTransform())
        td = transformed_env.rollout(1)
        td_keys = td.keys(True, True)
        assert ("next", env.reward_key) in td_keys
        for done_key in env.done_keys:
            assert ("next", done_key) in td_keys
            assert done_key in td_keys
        assert env.action_key in td_keys
        assert ("data", "states") not in td_keys
        assert ("next", "data", "states") not in td_keys

        transformed_env = TransformedEnv(env, SelectTransform(("data", "states")))
        td = transformed_env.rollout(1)
        td_keys = td.keys(True, True)
        assert ("next", env.reward_key) in td_keys
        for done_key in env.done_keys:
            assert ("next", done_key) in td_keys
            assert done_key in td_keys
        assert env.action_key in td_keys
        assert ("data", "states") in td_keys
        assert ("next", "data", "states") in td_keys

    def test_transform_no_env(self):
        t = SelectTransform("b", "c")
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        )
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    def test_transform_compose(self):
        t = Compose(SelectTransform("b", "c"))
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        )
        td = t._call(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    def test_transform_model(self):
        t = SelectTransform("b", "c")
        t = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        )
        td = t(td)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = SelectTransform("b", "c")
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {
                "a": torch.randn(1),
                "b": torch.randn(1),
                "c": torch.randn(1),
            },
            [],
        ).expand(3)
        rb.extend(td)
        td = rb.sample(4)
        assert "a" not in td.keys()
        assert "b" in td.keys()
        assert "c" in td.keys()

    def test_transform_inverse(self):
        raise pytest.skip("no inverse for SelectTransform")


@pytest.mark.parametrize("create_copy", [True, False])
class TestRenameTransform(TransformBase):
    @pytest.mark.parametrize("compose", [True, False])
    def test_single_trans_env_check(self, create_copy, compose):
        t = RenameTransform(
            ["observation"],
            ["stuff"],
            create_copy=create_copy,
        )
        if compose:
            t = Compose(t)
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        check_env_specs(env)
        t = RenameTransform(
            ["observation_orig"],
            ["stuff"],
            ["observation_orig"],
            ["stuff"],
            create_copy=create_copy,
        )
        if compose:
            t = Compose(t)
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        check_env_specs(env)

    def test_serial_trans_env_check(self, create_copy):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation"],
                    ["stuff"],
                    create_copy=create_copy,
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation_orig"],
                    ["stuff"],
                    ["observation_orig"],
                    ["stuff"],
                    create_copy=create_copy,
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, create_copy, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation"],
                    ["stuff"],
                    create_copy=create_copy,
                ),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation_orig"],
                    ["stuff"],
                    ["observation_orig"],
                    ["stuff"],
                    create_copy=create_copy,
                ),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self, create_copy):
        def make_env():
            return ContinuousActionVecMockEnv()

        env = TransformedEnv(
            SerialEnv(2, make_env),
            RenameTransform(
                ["observation"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        check_env_specs(env)
        env = TransformedEnv(
            SerialEnv(2, make_env),
            RenameTransform(
                ["observation_orig"],
                ["stuff"],
                ["observation_orig"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, create_copy, maybe_fork_ParallelEnv):
        def make_env():
            return ContinuousActionVecMockEnv()

        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, make_env),
            RenameTransform(
                ["observation"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, make_env),
            RenameTransform(
                ["observation_orig"],
                ["stuff"],
                ["observation_orig"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("mode", ["forward", "_call"])
    @pytest.mark.parametrize(
        "in_out_key",
        [
            ("a", "b"),
            (("nested", "stuff"), "b"),
            (("nested", "stuff"), "b"),
            (("nested", "stuff"), ("nested", "other")),
        ],
    )
    def test_transform_no_env(self, create_copy, mode, in_out_key):
        in_key, out_key = in_out_key
        t = RenameTransform([in_key], [out_key], create_copy=create_copy)
        tensordict = TensorDict({in_key: torch.randn(())}, [])
        if mode == "forward":
            t(tensordict)
        elif mode == "_call":
            t._call(tensordict)
        else:
            raise NotImplementedError
        assert out_key in tensordict.keys(True, True)
        if create_copy:
            assert in_key in tensordict.keys(True, True)
        else:
            assert in_key not in tensordict.keys(True, True)

    @pytest.mark.parametrize("mode", ["forward", "_call"])
    def test_transform_compose(self, create_copy, mode):
        t = Compose(RenameTransform(["a"], ["b"], create_copy=create_copy))
        tensordict = TensorDict({"a": torch.randn(())}, [])
        if mode == "forward":
            t(tensordict)
        elif mode == "_call":
            t._call(tensordict)
        else:
            raise NotImplementedError
        assert "b" in tensordict.keys()
        if create_copy:
            assert "a" in tensordict.keys()
        else:
            assert "a" not in tensordict.keys()

    def test_transform_env(self, create_copy):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            RenameTransform(
                ["observation"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        r = env.rollout(3)
        if create_copy:
            assert "observation" in r.keys()
            assert ("next", "observation") in r.keys(True)
        else:
            assert "observation" not in r.keys()
            assert ("next", "observation") not in r.keys(True)
        assert "stuff" in r.keys()
        assert ("next", "stuff") in r.keys(True)

        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            RenameTransform(
                ["observation_orig"],
                ["stuff"],
                ["observation_orig"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        r = env.rollout(3)
        if create_copy:
            assert "observation_orig" in r.keys()
            assert ("next", "observation_orig") in r.keys(True)
        else:
            assert "observation_orig" not in r.keys()
            assert ("next", "observation_orig") not in r.keys(True)
        assert "stuff" in r.keys()
        assert ("next", "stuff") in r.keys(True)

    def test_rename_done_reward(self, create_copy):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            RenameTransform(
                ["done"],
                [("nested", "other_done")],
                create_copy=create_copy,
            ),
        )
        assert ("nested", "other_done") in env.done_keys
        check_env_specs(env)
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            RenameTransform(
                ["reward"],
                [("nested", "reward")],
                create_copy=create_copy,
            ),
        )
        assert ("nested", "reward") in env.reward_keys
        check_env_specs(env)

    def test_transform_model(self, create_copy):
        t = RenameTransform(["a"], ["b"], create_copy=create_copy)
        tensordict = TensorDict({"a": torch.randn(())}, [])
        model = nn.Sequential(t)
        model(tensordict)
        assert "b" in tensordict.keys()
        if create_copy:
            assert "a" in tensordict.keys()
        else:
            assert "a" not in tensordict.keys()

    @pytest.mark.parametrize(
        "inverse",
        [
            False,
            True,
        ],
    )
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, create_copy, inverse, rbclass):
        if not inverse:
            t = RenameTransform(["a"], ["b"], create_copy=create_copy)
            tensordict = TensorDict({"a": torch.randn(())}, []).expand(10)
        else:
            t = RenameTransform(["a"], ["b"], ["a"], ["b"], create_copy=create_copy)
            tensordict = TensorDict({"b": torch.randn(())}, []).expand(10)
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        rb.extend(tensordict)

        assert "a" in rb._storage._storage.keys()
        sample = rb.sample(2)
        if create_copy:
            assert "a" in sample.keys()
        else:
            assert "a" not in sample.keys()
        assert "b" in sample.keys()

    def test_transform_inverse(self, create_copy):
        t = RenameTransform(["a"], ["b"], ["a"], ["b"], create_copy=create_copy)
        tensordict = TensorDict({"b": torch.randn(())}, []).expand(10)
        tensordict = t.inv(tensordict)
        assert "a" in tensordict.keys()
        if create_copy:
            assert "b" in tensordict.keys()
        else:
            assert "b" not in tensordict.keys()

    def test_rename_action(self, create_copy):
        base_env = ContinuousActionVecMockEnv()
        env = base_env.append_transform(
            RenameTransform(
                in_keys=[],
                out_keys=[],
                in_keys_inv=["action"],
                out_keys_inv=[("renamed", "action")],
                create_copy=create_copy,
            )
        )
        r = env.rollout(3)
        assert ("renamed", "action") in env.action_keys, env.action_keys
        assert ("renamed", "action") in r
        assert env.full_action_spec[("renamed", "action")] is not None
        if create_copy:
            assert "action" in env.action_keys
            assert "action" in r
        else:
            assert "action" not in env.action_keys
            assert "action" not in r


class TestRemoveEmptySpecs(TransformBase):
    class DummyEnv(EnvBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observation_spec = Composite(
                observation=Unbounded((*self.batch_size, 3)),
                other=Composite(
                    another_other=Composite(shape=self.batch_size),
                    shape=self.batch_size,
                ),
                shape=self.batch_size,
            )
            self.action_spec = Unbounded((*self.batch_size, 3))
            self.done_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
            self.full_done_spec["truncated"] = self.full_done_spec["terminated"].clone()
            self.reward_spec = Composite(
                reward=Unbounded(*self.batch_size, 1),
                other_reward=Composite(shape=self.batch_size),
                shape=self.batch_size,
            )
            self.state_spec = Composite(
                state=Composite(
                    sub=Composite(shape=self.batch_size), shape=self.batch_size
                ),
                shape=self.batch_size,
            )

        def _reset(self, tensordict):
            return self.observation_spec.rand().update(self.full_done_spec.zero())

        def _step(self, tensordict):
            return (
                TensorDict()
                .update(self.observation_spec.rand())
                .update(self.full_done_spec.zero())
                .update(self.full_reward_spec.rand())
            )

        def _set_seed(self, seed: int | None) -> None:
            ...

    def test_single_trans_env_check(self):
        env = TransformedEnv(self.DummyEnv(), RemoveEmptySpecs())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(2, lambda: TransformedEnv(self.DummyEnv(), RemoveEmptySpecs()))
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2, lambda: TransformedEnv(self.DummyEnv(), RemoveEmptySpecs())
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        with pytest.raises(
            RuntimeError, match="The environment passed to SerialEnv has empty specs"
        ):
            TransformedEnv(SerialEnv(2, self.DummyEnv), RemoveEmptySpecs())

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        with pytest.raises(
            RuntimeError, match="The environment passed to ParallelEnv has empty specs"
        ):
            env = TransformedEnv(
                maybe_fork_ParallelEnv(2, self.DummyEnv), RemoveEmptySpecs()
            )

    def test_transform_no_env(self):
        td = TensorDict({"a": {"b": {"c": {}}}}, [])
        t = RemoveEmptySpecs()
        t._call(td)
        assert len(td.keys()) == 0

    def test_transform_compose(self):
        td = TensorDict({"a": {"b": {"c": {}}}}, [])
        t = Compose(RemoveEmptySpecs())
        t._call(td)
        assert len(td.keys()) == 0

    def test_transform_env(self):
        base_env = self.DummyEnv()
        r = base_env.rollout(2)
        assert ("next", "other", "another_other") in r.keys(True)
        env = TransformedEnv(base_env, RemoveEmptySpecs())
        r = env.rollout(2)
        assert ("other", "another_other") not in r.keys(True)
        assert "other" not in r.keys(True)

    def test_transform_model(self):
        td = TensorDict({"a": {"b": {"c": {}}}}, [])
        t = nn.Sequential(Compose(RemoveEmptySpecs()))
        td = t(td)
        assert len(td.keys()) == 0

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Compose(RemoveEmptySpecs())

        batch = (20,)
        td = TensorDict({"a": {"b": {"c": {}}}}, batch)

        torch.manual_seed(0)
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        rb.extend(td)
        td = rb.sample(1)
        if "index" in td.keys():
            del td["index"]
        assert len(td.keys()) == 0

    def test_transform_inverse(self):
        td = TensorDict({"a": {"b": {"c": {}}}}, [])
        t = RemoveEmptySpecs()
        t.inv(td)
        assert len(td.keys()) != 0
        env = TransformedEnv(self.DummyEnv(), RemoveEmptySpecs())
        td2 = env.transform.inv(TensorDict())
        assert ("state", "sub") in td2.keys(True)


class TestHash(TransformBase):
    @pytest.mark.parametrize("datatype", ["tensor", "str", "NonTensorStack"])
    def test_transform_no_env(self, datatype):
        if datatype == "tensor":
            obs = torch.tensor(10)
            hash_fn = lambda x: torch.tensor(hash(x))
        elif datatype == "str":
            obs = "abcdefg"
            hash_fn = Hash.reproducible_hash
        elif datatype == "NonTensorStack":
            obs = torch.stack(
                [
                    NonTensorData(data="abcde"),
                    NonTensorData(data="fghij"),
                    NonTensorData(data="klmno"),
                ]
            )

            def fn0(x):
                # return tuple([tuple(Hash.reproducible_hash(x_).tolist()) for x_ in x])
                return torch.stack([Hash.reproducible_hash(x_) for x_ in x])

            hash_fn = fn0
        else:
            raise RuntimeError(f"please add a test case for datatype {datatype}")

        td = TensorDict(
            {
                "observation": obs,
            }
        )

        t = Hash(in_keys=["observation"], out_keys=["hashing"], hash_fn=hash_fn)
        td_hashed = t(td)

        assert td_hashed.get("observation") is td.get("observation")

        if datatype == "NonTensorStack":
            assert (
                td_hashed["hashing"] == hash_fn(td.get("observation").tolist())
            ).all()
        elif datatype == "str":
            assert all(td_hashed["hashing"] == hash_fn(td["observation"]))
        else:
            assert td_hashed["hashing"] == hash_fn(td["observation"])

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_single_trans_env_check(self, datatype):
        if datatype == "tensor":
            t = Hash(
                in_keys=["observation"],
                out_keys=["hashing"],
                hash_fn=lambda x: torch.tensor(hash(x)),
            )
            base_env = CountingEnv()
        elif datatype == "str":
            t = Hash(
                in_keys=["string"],
                out_keys=["hashing"],
            )
            base_env = CountingEnvWithString()
        env = TransformedEnv(base_env, t)
        check_env_specs(env)

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_serial_trans_env_check(self, datatype):
        def make_env():
            if datatype == "tensor":
                t = Hash(
                    in_keys=["observation"],
                    out_keys=["hashing"],
                    hash_fn=lambda x: torch.tensor(hash(x)),
                )
                base_env = CountingEnv()

            elif datatype == "str":
                t = Hash(
                    in_keys=["string"],
                    out_keys=["hashing"],
                )
                base_env = CountingEnvWithString()

            return TransformedEnv(base_env, t)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv, datatype):
        def make_env():
            if datatype == "tensor":
                t = Hash(
                    in_keys=["observation"],
                    out_keys=["hashing"],
                    hash_fn=lambda x: torch.tensor(hash(x)),
                )
                base_env = CountingEnv()
            elif datatype == "str":
                t = Hash(
                    in_keys=["string"],
                    out_keys=["hashing"],
                )
                base_env = CountingEnvWithString()
            return TransformedEnv(base_env, t)

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_trans_serial_env_check(self, datatype):
        if datatype == "tensor":
            t = Hash(
                in_keys=["observation"],
                out_keys=["hashing"],
                hash_fn=lambda x: torch.tensor([hash(x[0]), hash(x[1])]),
            )
            base_env = CountingEnv
        elif datatype == "str":
            t = Hash(
                in_keys=["string"],
                out_keys=["hashing"],
                hash_fn=lambda x: torch.stack([Hash.reproducible_hash(x_) for x_ in x]),
            )
            base_env = CountingEnvWithString

        env = TransformedEnv(SerialEnv(2, base_env), t)
        check_env_specs(env)

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv, datatype):
        if datatype == "tensor":
            t = Hash(
                in_keys=["observation"],
                out_keys=["hashing"],
                hash_fn=lambda x: torch.tensor([hash(x[0]), hash(x[1])]),
            )
            base_env = CountingEnv
        elif datatype == "str":
            t = Hash(
                in_keys=["string"],
                out_keys=["hashing"],
                hash_fn=lambda x: torch.stack([Hash.reproducible_hash(x_) for x_ in x]),
            )
            base_env = CountingEnvWithString

        env = TransformedEnv(maybe_fork_ParallelEnv(2, base_env), t)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("datatype", ["tensor", "str"])
    def test_transform_compose(self, datatype):
        if datatype == "tensor":
            obs = torch.tensor(10)
        elif datatype == "str":
            obs = "abcdefg"

        td = TensorDict(
            {
                "observation": obs,
            }
        )
        t = Hash(
            in_keys=["observation"],
            out_keys=["hashing"],
            hash_fn=lambda x: torch.tensor(hash(x)),
        )
        t = Compose(t)
        td_hashed = t(td)

        assert td_hashed["observation"] is td["observation"]
        assert td_hashed["hashing"] == hash(td["observation"])

    def test_transform_model(self):
        t = Hash(
            in_keys=[("next", "observation"), ("observation",)],
            out_keys=[("next", "hashing"), ("hashing",)],
            hash_fn=lambda x: torch.tensor(hash(x)),
        )
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {("next", "observation"): torch.randn(3), "observation": torch.randn(3)}, []
        )
        td_out = model(td)
        assert ("next", "hashing") in td_out.keys(True)
        assert ("hashing",) in td_out.keys(True)
        assert td_out["next", "hashing"] == hash(td["next", "observation"])
        assert td_out["hashing"] == hash(td["observation"])

    @pytest.mark.skipif(not _has_gym, reason="Gym not found")
    def test_transform_env(self):
        t = Hash(
            in_keys=["observation"],
            out_keys=["hashing"],
            hash_fn=lambda x: torch.tensor(hash(x)),
        )
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), t)
        assert env.observation_spec["hashing"]
        assert "observation" in env.observation_spec
        assert "observation" in env.base_env.observation_spec
        check_env_specs(env)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Hash(
            in_keys=[("next", "observation"), ("observation",)],
            out_keys=[("next", "hashing"), ("hashing",)],
            hash_fn=lambda x: torch.tensor([hash(x[0]), hash(x[1])]),
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {
                "observation": torch.randn(3, 4),
                "next": TensorDict(
                    {"observation": torch.randn(3, 4)},
                    [],
                ),
            },
            [],
        ).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        assert "hashing" in td.keys()
        assert "observation" in td.keys()
        assert ("next", "observation") in td.keys(True)

    @pytest.mark.parametrize("repertoire_gen", [lambda: None, lambda: {}])
    def test_transform_inverse(self, repertoire_gen):
        repertoire = repertoire_gen()
        t = Hash(
            in_keys=["observation"],
            out_keys=["hashing"],
            in_keys_inv=["observation"],
            out_keys_inv=["hashing"],
            repertoire=repertoire,
        )
        inputs = [
            TensorDict({"observation": "test string"}),
            TensorDict({"observation": torch.randn(10)}),
            TensorDict({"observation": "another string"}),
            TensorDict({"observation": torch.randn(3, 2, 1, 8)}),
        ]
        outputs = [t(input.clone()).exclude("observation") for input in inputs]

        # Run the inputs through again, just to make sure that using the same
        # inputs doesn't overwrite the repertoire.
        for input in inputs:
            t(input.clone())

        assert len(t._repertoire) == 4

        inv_inputs = [t.inv(output.clone()) for output in outputs]

        for input, inv_input in zip(inputs, inv_inputs):
            if torch.is_tensor(input["observation"]):
                assert (input["observation"] == inv_input["observation"]).all()
            else:
                assert input["observation"] == inv_input["observation"]

    @pytest.mark.parametrize("repertoire_gen", [lambda: None, lambda: {}])
    def test_repertoire(self, repertoire_gen):
        repertoire = repertoire_gen()
        t = Hash(in_keys=["observation"], out_keys=["hashing"], repertoire=repertoire)
        inputs = [
            "string",
            ["a", "b"],
            torch.randn(3, 4, 1),
            torch.randn(()),
            torch.randn(0),
            1234,
            [1, 2, 3, 4],
        ]
        outputs = []

        for input in inputs:
            td = TensorDict({"observation": input})
            outputs.append(t(td.clone()).clone()["hashing"])

        for output, input in zip(outputs, inputs):
            if repertoire is not None:
                stored_input = repertoire[t.hash_to_repertoire_key(output)]
                assert stored_input is t.get_input_from_hash(output)

                if torch.is_tensor(stored_input):
                    assert (stored_input == torch.as_tensor(input)).all()
                elif isinstance(stored_input, np.ndarray):
                    assert (stored_input == np.asarray(input)).all()

                else:
                    assert stored_input == input
            else:
                with pytest.raises(RuntimeError):
                    stored_input = t.get_input_from_hash(output)


@pytest.mark.skipif(
    not _has_transformers, reason="transformers needed to test tokenizers"
)
class TestTokenizer(TransformBase):
    @pytest.mark.parametrize("datatype", ["str", "NonTensorStack"])
    def test_transform_no_env(self, datatype):
        if datatype == "str":
            obs = "abcdefg"
        elif datatype == "NonTensorStack":
            obs = torch.stack(
                [
                    NonTensorData(data="abcde"),
                    NonTensorData(data="fghij"),
                    NonTensorData(data="klmno"),
                ]
            )
        else:
            raise RuntimeError(f"please add a test case for datatype {datatype}")

        td = TensorDict(
            {
                "observation": obs,
            }
        )

        t = Tokenizer(in_keys=["observation"], out_keys=["tokens"])
        td_tokenized = t(td)
        t_inv = Tokenizer([], [], in_keys_inv=["observation"], out_keys_inv=["tokens"])
        td_recon = t_inv.inv(td_tokenized.clone().exclude("observation"))
        assert td_tokenized.get("observation") is td.get("observation")
        assert td_recon["observation"] == td["observation"]

    @pytest.mark.parametrize("datatype", ["str"])
    def test_single_trans_env_check(self, datatype):
        if datatype == "str":
            t = Tokenizer(
                in_keys=["string"],
                out_keys=["tokens"],
                max_length=5,
            )
            base_env = CountingEnvWithString(max_size=4, min_size=4)
        env = TransformedEnv(base_env, t)
        check_env_specs(env, return_contiguous=False)

    @pytest.mark.parametrize("datatype", ["str"])
    def test_serial_trans_env_check(self, datatype):
        def make_env():
            if datatype == "str":
                t = Tokenizer(
                    in_keys=["string"],
                    out_keys=["tokens"],
                    max_length=5,
                )
                base_env = CountingEnvWithString(max_size=4, min_size=4)

            return TransformedEnv(base_env, t)

        env = SerialEnv(2, make_env)
        check_env_specs(env, return_contiguous=False)

    @pytest.mark.parametrize("datatype", ["str"])
    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv, datatype):
        def make_env():
            if datatype == "str":
                t = Tokenizer(
                    in_keys=["string"],
                    out_keys=["tokens"],
                    max_length=5,
                )
                base_env = CountingEnvWithString(max_size=4, min_size=4)
            return TransformedEnv(base_env, t)

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env, return_contiguous=False)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("datatype", ["str"])
    def test_trans_serial_env_check(self, datatype):
        if datatype == "str":
            t = Tokenizer(
                in_keys=["string"],
                out_keys=["tokens"],
                max_length=5,
            )
            base_env = partial(CountingEnvWithString, max_size=4, min_size=4)

        env = TransformedEnv(SerialEnv(2, base_env), t)
        check_env_specs(env, return_contiguous=False)

    @pytest.mark.parametrize("datatype", ["str"])
    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv, datatype):
        if datatype == "str":
            t = Tokenizer(
                in_keys=["string"],
                out_keys=["tokens"],
                max_length=5,
            )
            base_env = partial(CountingEnvWithString, max_size=4, min_size=4)

        env = TransformedEnv(maybe_fork_ParallelEnv(2, base_env), t)
        try:
            check_env_specs(env, return_contiguous=False)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("datatype", ["str"])
    def test_transform_compose(self, datatype):
        if datatype == "str":
            obs = "abcdefg"

        td = TensorDict(
            {
                "observation": obs,
            }
        )
        t = Tokenizer(
            in_keys=["observation"],
            out_keys=["tokens"],
            max_length=5,
        )
        t = Compose(t)
        td_tokenized = t(td)

        assert td_tokenized["observation"] is td["observation"]
        assert (
            td_tokenized["tokens"]
            == t[0].tokenizer.encode(
                obs,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                max_length=5,
            )
        ).all()

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_transform_model(self, n):
        t = Tokenizer(
            in_keys=["observation"],
            out_keys=["tokens"],
            max_length=n,
        )
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({"observation": "a string!"})
        td_out = model(td)
        assert (
            td_out["tokens"] == torch.tensor([1037, 5164, 999] + [0] * (n - 3))
        ).all()

    def test_transform_env(self):
        import random

        random.seed(0)
        t = Tokenizer(
            in_keys=["string"],
            out_keys=["tokens"],
            max_length=10,
        )
        base_env = CountingEnvWithString(max_steps=10, max_size=4, min_size=4)
        env = TransformedEnv(base_env, t)
        policy = lambda td: env.full_action_spec.one()
        r = env.rollout(100, policy)
        assert r["string"] == [
            "mzjp",
            "sgqe",
            "eydt",
            "rwzt",
            "jdxc",
            "prdl",
            "ktug",
            "oqib",
            "cxmw",
            "tpkh",
            "wcgs",
        ]
        assert (
            env.transform.tokenizer.batch_decode(r["tokens"], skip_special_tokens=True)
            == r["string"]
        )

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Tokenizer(
            in_keys=["observation"],
            out_keys=["tokens"],
            max_length=5,
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(
            {
                "observation": NonTensorStack(
                    "mzjp",
                    "sgqe",
                    "eydt",
                    "rwzt",
                    "jdxc",
                    "prdl",
                    "ktug",
                    "oqib",
                    "cxmw",
                    "tpkh",
                ),
            },
            [10],
        )
        rb.extend(td)
        td = rb.sample(2)
        assert (
            t.tokenizer.batch_decode(td["tokens"], skip_special_tokens=True)
            == td["observation"]
        )

    def test_transform_inverse(self):
        torch.manual_seed(0)
        t = Tokenizer(
            in_keys=[],
            out_keys=[],
            # The policy produces tokens
            out_keys_inv=["tokens"],
            # The env must see strings
            in_keys_inv=["strings"],
            max_length=5,
        )
        base_env = CountingEnv()

        class CheckString(Transform):
            def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
                assert "strings" in tensordict
                tensordict.pop("strings")
                return tensordict

            def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
                action_spec["strings"] = NonTensor(
                    shape=action_spec.shape, example_data="a string!"
                )
                return action_spec

        env = TransformedEnv(base_env, Compose(CheckString(), t))

        def policy(td):
            td.set("tokens", torch.randint(0, 10000, (10,)))
            td.update(env.full_action_spec.one())
            return td

        env.check_env_specs()
