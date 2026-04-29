# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import re

import pytest

import tensordict.tensordict
import torch

from _transforms_common import TransformBase
from tensordict import TensorDict, unravel_key
from tensordict.utils import _unravel_key_to_tuple
from torch import nn
from torchrl._utils import _replace_last

from torchrl.data import (
    BoundedContinuous,
    Categorical,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorSpec,
    Unbounded,
    UnboundedContinuous,
)
from torchrl.envs import (
    BinarizeReward,
    Compose,
    EnvBase,
    LineariseRewards,
    ParallelEnv,
    Reward2GoTransform,
    RewardClipping,
    RewardScaling,
    RewardSum,
    SerialEnv,
    SignTransform,
    TransformedEnv,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
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
    CountingBatchedEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)


class TestBinarizeReward(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), BinarizeReward())
        check_env_specs(env)
        env.close()

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2, lambda: TransformedEnv(ContinuousActionVecMockEnv(), BinarizeReward())
        )
        check_env_specs(env)
        env.close()

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2, lambda: TransformedEnv(ContinuousActionVecMockEnv(), BinarizeReward())
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()), BinarizeReward()
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
            BinarizeReward(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    @pytest.mark.parametrize("in_key", ["reward", ("agents", "reward")])
    def test_transform_no_env(self, device, batch, in_key):
        torch.manual_seed(0)
        br = BinarizeReward(in_keys=[in_key])
        reward = torch.randn(*batch, 1, device=device)
        reward_copy = reward.clone()
        misc = torch.randn(*batch, 1, device=device)
        misc_copy = misc.clone()

        td = TensorDict(
            {"misc": misc, in_key: reward},
            batch,
            device=device,
        )
        br(td)
        assert (td[in_key] != reward_copy).all()
        assert (td["misc"] == misc_copy).all()
        assert (torch.count_nonzero(td[in_key]) == torch.sum(reward_copy > 0)).all()

    def test_nested(self):
        orig_env = NestedCountingEnv()
        env = TransformedEnv(orig_env, BinarizeReward(in_keys=[orig_env.reward_key]))
        env.rollout(3)
        assert "data" in env._output_spec["full_reward_spec"]

    def test_transform_compose(self):
        torch.manual_seed(0)
        br = Compose(BinarizeReward())
        batch = (2,)
        device = "cpu"
        reward = torch.randn(*batch, 1, device=device)
        reward_copy = reward.clone()
        misc = torch.randn(*batch, 1, device=device)
        misc_copy = misc.clone()

        td = TensorDict(
            {"misc": misc, "reward": reward},
            batch,
            device=device,
        )
        br(td)
        assert (td["reward"] != reward_copy).all()
        assert (td["misc"] == misc_copy).all()
        assert (torch.count_nonzero(td["reward"]) == torch.sum(reward_copy > 0)).all()

    def test_transform_env(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), BinarizeReward())
        rollout = env.rollout(3)
        assert env.reward_spec.is_in(rollout["next", "reward"])

    def test_transform_model(self):
        device = "cpu"
        batch = [4]
        torch.manual_seed(0)
        br = BinarizeReward()

        class RewardPlus(nn.Module):
            def forward(self, td):
                return td["reward"] + 1

        reward = torch.randn(*batch, 1, device=device)
        misc = torch.randn(*batch, 1, device=device)

        td = TensorDict(
            {"misc": misc, "reward": reward},
            batch,
            device=device,
        )
        chain = nn.Sequential(br, RewardPlus())
        reward = chain(td)
        assert ((reward - 1) == td["reward"]).all()
        assert ((reward - 1 == 0) | (reward - 1 == 1)).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        device = "cpu"
        batch = [20]
        torch.manual_seed(0)
        br = BinarizeReward()
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(br)
        reward = torch.randn(*batch, 1, device=device)
        misc = torch.randn(*batch, 1, device=device)
        td = TensorDict(
            {"misc": misc, "reward": reward},
            batch,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample(20)
        assert ((sample["reward"] == 0) | (sample["reward"] == 1)).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for BinarizedReward")


class TestRewardClipping(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), RewardClipping(-0.1, 0.1))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), RewardClipping(-0.1, 0.1)
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), RewardClipping(-0.1, 0.1)
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
            SerialEnv(2, ContinuousActionVecMockEnv), RewardClipping(-0.1, 0.1)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            RewardClipping(-0.1, 0.1),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("reward_key", ["reward", ("agents", "reward")])
    def test_transform_no_env(self, reward_key):
        t = RewardClipping(-0.1, 0.1, in_keys=[reward_key])
        td = TensorDict({reward_key: torch.randn(10)}, [])
        t._call(td)
        assert (td[reward_key] <= 0.1).all()
        assert (td[reward_key] >= -0.1).all()

    def test_transform_compose(self):
        t = Compose(RewardClipping(-0.1, 0.1))
        td = TensorDict({"reward": torch.randn(10)}, [])
        t._call(td)
        assert (td["reward"] <= 0.1).all()
        assert (td["reward"] >= -0.1).all()

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    def test_transform_env(self):
        t = Compose(RewardClipping(-0.1, 0.1))
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), t)
        td = env.rollout(3)
        assert (td["next", "reward"] <= 0.1).all()
        assert (td["next", "reward"] >= -0.1).all()

    def test_transform_model(self):
        t = RewardClipping(-0.1, 0.1)
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({"reward": torch.randn(10)}, [])
        model(td)
        assert (td["reward"] <= 0.1).all()
        assert (td["reward"] >= -0.1).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = RewardClipping(-0.1, 0.1)
        rb = rbclass(storage=LazyTensorStorage(10))
        td = TensorDict({"reward": torch.randn(10)}, []).expand(10)
        rb.append_transform(t)
        rb.extend(td)
        td = rb.sample(2)
        assert (td["reward"] <= 0.1).all()
        assert (td["reward"] >= -0.1).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for RewardClipping")


class TestRewardScaling(TransformBase):
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("scale", [0.1, 10])
    @pytest.mark.parametrize("loc", [1, 5])
    @pytest.mark.parametrize("keys", [None, ["reward_1"], [("nested", "key")]])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_reward_scaling(self, batch, scale, loc, keys, device, standard_normal):
        torch.manual_seed(0)
        if keys is None:
            keys_total = set()
        else:
            keys_total = set(keys)
        reward_scaling = RewardScaling(
            in_keys=keys, scale=scale, loc=loc, standard_normal=standard_normal
        )
        td = TensorDict(
            {
                **{key: torch.randn(*batch, 1, device=device) for key in keys_total},
                "reward": torch.randn(*batch, 1, device=device),
            },
            batch,
            device=device,
        )
        td.set("dont touch", torch.randn(*batch, 1, device=device))
        td_copy = td.clone()
        reward_scaling(td)
        for key in keys_total:
            if standard_normal:
                original_key = td.get(key)
                scaled_key = (td_copy.get(key) - loc) / scale
                torch.testing.assert_close(original_key, scaled_key)
            else:
                original_key = td.get(key)
                scaled_key = td_copy.get(key) * scale + loc
                torch.testing.assert_close(original_key, scaled_key)
        if keys is None:
            if standard_normal:
                original_key = td.get("reward")
                scaled_key = (td_copy.get("reward") - loc) / scale
                torch.testing.assert_close(original_key, scaled_key)
            else:
                original_key = td.get("reward")
                scaled_key = td_copy.get("reward") * scale + loc
                torch.testing.assert_close(original_key, scaled_key)

        assert (td.get("dont touch") == td_copy.get("dont touch")).all()

        if len(keys_total) == 1:
            reward_spec = Unbounded(device=device)
            reward_spec = reward_scaling.transform_reward_spec(reward_spec)
            assert reward_spec.shape == torch.Size([1])

    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), RewardScaling(0.5, 1.5))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), RewardScaling(0.5, 1.5))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), RewardScaling(0.5, 1.5))

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
            SerialEnv(2, ContinuousActionVecMockEnv), RewardScaling(0.5, 1.5)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            RewardScaling(0.5, 1.5),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_no_env(self, standard_normal):
        loc = 0.5
        scale = 1.5
        t = RewardScaling(0.5, 1.5, standard_normal=standard_normal)
        reward = torch.randn(10)
        td = TensorDict({"reward": reward}, [])
        t._call(td)
        if standard_normal:
            assert torch.allclose((reward - loc) / scale, td["reward"])
        else:
            assert torch.allclose((td["reward"] - loc) / scale, reward)

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_compose(self, standard_normal):
        loc = 0.5
        scale = 1.5
        t = RewardScaling(0.5, 1.5, standard_normal=standard_normal)
        t = Compose(t)
        reward = torch.randn(10)
        td = TensorDict({"reward": reward}, [])
        t._call(td)
        if standard_normal:
            assert torch.allclose((reward - loc) / scale, td["reward"])
        else:
            assert torch.allclose((td["reward"] - loc) / scale, reward)

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_env(self, standard_normal):
        loc = 0.5
        scale = 1.5
        t = Compose(RewardScaling(0.5, 1.5, standard_normal=standard_normal))
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), t)
        torch.manual_seed(0)
        env.set_seed(0)
        td = env.rollout(3)
        torch.manual_seed(0)
        env.set_seed(0)
        td_base = env.base_env.rollout(3)
        reward = td_base["next", "reward"]
        if standard_normal:
            assert torch.allclose((reward - loc) / scale, td["next", "reward"])
        else:
            assert torch.allclose((td["next", "reward"] - loc) / scale, reward)

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_model(self, standard_normal):
        loc = 0.5
        scale = 1.5
        t = RewardScaling(0.5, 1.5, standard_normal=standard_normal)
        model = nn.Sequential(t, nn.Identity())
        reward = torch.randn(10)
        td = TensorDict({"reward": reward}, [])
        model(td)
        if standard_normal:
            assert torch.allclose((reward - loc) / scale, td["reward"])
        else:
            assert torch.allclose((td["reward"] - loc) / scale, reward)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_rb(self, rbclass, standard_normal):
        loc = 0.5
        scale = 1.5
        t = RewardScaling(0.5, 1.5, standard_normal=standard_normal)
        rb = rbclass(storage=LazyTensorStorage(10))
        reward = torch.randn(10)
        td = TensorDict({"reward": reward}, []).expand(10)
        rb.append_transform(t)
        rb.extend(td)
        td = rb.sample(2)
        if standard_normal:
            assert torch.allclose((reward - loc) / scale, td["reward"])
        else:
            assert torch.allclose((td["reward"] - loc) / scale, reward)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for RewardScaling")


class TestRewardSum(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
        )
        check_env_specs(env)
        r = env.rollout(4)
        assert r["next", "episode_reward"].unique().numel() > 1

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)
        r = env.rollout(4)
        assert r["next", "episode_reward"].unique().numel() > 1

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
            r = env.rollout(4)
            assert r["next", "episode_reward"].unique().numel() > 1
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
        )
        check_env_specs(env)
        r = env.rollout(4)
        assert r["next", "episode_reward"].unique().numel() > 1

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
        )
        try:
            check_env_specs(env)
            r = env.rollout(4)
            assert r["next", "episode_reward"].unique().numel() > 1
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("has_in_keys,", [True, False])
    @pytest.mark.parametrize(
        "reset_keys,", [[("some", "nested", "reset")], ["_reset"] * 3, None]
    )
    def test_trans_multi_key(
        self, has_in_keys, reset_keys, n_workers=2, batch_size=(3, 2), max_steps=5
    ):
        torch.manual_seed(0)
        env_fun = lambda: MultiKeyCountingEnv(batch_size=batch_size)
        base_env = SerialEnv(n_workers, env_fun)
        kwargs = (
            {}
            if not has_in_keys
            else {"in_keys": ["reward", ("nested_1", "gift"), ("nested_2", "reward")]}
        )
        t = RewardSum(reset_keys=reset_keys, **kwargs)
        env = TransformedEnv(
            base_env,
            Compose(t),
        )
        policy = MultiKeyCountingEnvPolicy(
            full_action_spec=env.action_spec, deterministic=True
        )
        with (
            pytest.raises(ValueError, match="Could not match the env reset_keys")
            if reset_keys == [("some", "nested", "reset")]
            else contextlib.nullcontext()
        ):
            check_env_specs(env)
        if reset_keys != [("some", "nested", "reset")]:
            td = env.rollout(max_steps, policy=policy)
            for reward_key in env.reward_keys:
                reward_key = _unravel_key_to_tuple(reward_key)
                assert (
                    td.get(
                        ("next", _replace_last(reward_key, f"episode_{reward_key[-1]}"))
                    )[(0,) * (len(batch_size) + 1)][-1]
                    == max_steps
                ).all()

    @pytest.mark.parametrize("in_key", ["reward", ("some", "nested")])
    def test_transform_no_env(self, in_key):
        t = RewardSum(in_keys=[in_key], out_keys=[("some", "nested_sum")])
        reward = torch.randn(10)
        td = TensorDict({("next", in_key): reward}, [])
        with pytest.raises(
            ValueError, match="At least one dimension of the tensordict"
        ):
            t(td)
        td.batch_size = [10]
        td.names = ["time"]
        with pytest.raises(KeyError):
            t(td)
        t = RewardSum(
            in_keys=[unravel_key(("next", in_key))],
            out_keys=[("some", "nested_sum")],
        )
        res = t(td)
        assert ("some", "nested_sum") in res.keys(True, True)

    def test_transform_compose(
        self,
    ):
        # reset keys should not be needed for offline run
        t = Compose(RewardSum(in_keys=["reward"], out_keys=["episode_reward"]))
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        with pytest.raises(
            ValueError, match="At least one dimension of the tensordict"
        ):
            t(td)
        td.batch_size = [10]
        td.names = ["time"]
        with pytest.raises(KeyError):
            t(td)
        t = RewardSum(
            in_keys=[("next", "reward")], out_keys=[("next", "episode_reward")]
        )
        t(td)

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    @pytest.mark.parametrize("out_key", ["reward_sum", ("some", "nested")])
    @pytest.mark.parametrize("reward_spec", [False, True])
    def test_transform_env(self, out_key, reward_spec):
        t = Compose(
            RewardSum(in_keys=["reward"], out_keys=[out_key], reward_spec=reward_spec)
        )
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), t)
        if reward_spec:
            assert out_key in env.reward_keys
            assert out_key not in env.observation_spec.keys(True)
        else:
            assert out_key not in env.reward_keys
            assert out_key in env.observation_spec.keys(True)

        env.set_seed(0)
        torch.manual_seed(0)
        td = env.rollout(3)
        env.set_seed(0)
        torch.manual_seed(0)
        td_base = env.base_env.rollout(3)
        reward = td_base["next", "reward"]
        final_reward = td_base["next", "reward"].sum(-2)
        assert torch.allclose(td["next", "reward"], reward)
        assert torch.allclose(td["next", out_key][..., -1, :], final_reward)

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_rewardsum_batching(self, batched_class, break_when_any_done):
        from torchrl.testing import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())), RewardSum()
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2, lambda: TransformedEnv(GymEnv(CARTPOLE_VERSIONED()), RewardSum())
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    def test_transform_model(
        self,
    ):
        t = RewardSum(
            in_keys=[("next", "reward")], out_keys=[("next", "episode_reward")]
        )
        model = nn.Sequential(t, nn.Identity())
        env = TransformedEnv(ContinuousActionVecMockEnv(), RewardSum())
        data = env.rollout(10)
        data_exclude = data.exclude(("next", "episode_reward"))
        model(data_exclude)
        assert (
            data_exclude["next", "episode_reward"] == data["next", "episode_reward"]
        ).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
    ):
        t = RewardSum(
            in_keys=[("next", "reward")], out_keys=[("next", "episode_reward")]
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        env = TransformedEnv(ContinuousActionVecMockEnv(), RewardSum())
        data = env.rollout(10)
        data_exclude = data.exclude(("next", "episode_reward"))
        rb.append_transform(t)
        rb.add(data_exclude)
        sample = rb.sample(1).squeeze(0)
        assert (
            sample["next", "episode_reward"] == data["next", "episode_reward"]
        ).all()

    @pytest.mark.parametrize(
        "keys",
        [["done", "reward"]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_sum_reward(self, keys, device):
        torch.manual_seed(0)
        batch = 4
        rs = RewardSum()
        td = TensorDict(
            {
                "next": {
                    "done": torch.zeros((batch, 1), dtype=torch.bool),
                    "reward": torch.rand((batch, 1)),
                },
                "episode_reward": torch.zeros((batch, 1), dtype=torch.bool),
            },
            device=device,
            batch_size=[batch],
        )

        # apply one time, episode_reward should be equal to reward again
        td_next = rs._step(td, td.get("next"))
        assert "episode_reward" in td.keys()
        assert (td_next.get("episode_reward") == td_next.get("reward")).all()

        # apply a second time, episode_reward should twice the reward
        td["episode_reward"] = td["next", "episode_reward"]
        td_next = rs._step(td, td.get("next"))
        assert (td_next.get("episode_reward") == 2 * td_next.get("reward")).all()

        # reset environments
        td.set("_reset", torch.ones(batch, dtype=torch.bool, device=device))
        with pytest.raises(TypeError, match="reset_keys not provided but parent"):
            rs._reset(td, td)
        rs._reset_keys = ["_reset"]
        td_reset = rs._reset(td, td.empty())
        td = td_reset.set("next", td.get("next"))

        # apply a third time, episode_reward should be equal to reward again
        td_next = rs._step(td, td.get("next"))
        assert (td_next.get("episode_reward") == td_next.get("reward")).all()

        # test transform_observation_spec
        base_env = ContinuousActionVecMockEnv(
            reward_spec=Unbounded(shape=(3, 16, 16)),
        )
        transfomed_env = TransformedEnv(base_env, RewardSum())
        transformed_observation_spec1 = transfomed_env.observation_spec
        assert isinstance(transformed_observation_spec1, Composite)
        assert "episode_reward" in transformed_observation_spec1.keys()
        assert "observation" in transformed_observation_spec1.keys()

        base_env = ContinuousActionVecMockEnv(
            reward_spec=Unbounded(),
            observation_spec=Composite(
                observation=Unbounded(),
                some_extra_observation=Unbounded(),
            ),
        )
        transfomed_env = TransformedEnv(base_env, RewardSum())
        transformed_observation_spec2 = transfomed_env.observation_spec
        assert isinstance(transformed_observation_spec2, Composite)
        assert "some_extra_observation" in transformed_observation_spec2.keys()
        assert "episode_reward" in transformed_observation_spec2.keys()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for RewardSum")

    @pytest.mark.parametrize("in_keys", [["reward"], ["reward_1", "reward_2"]])
    @pytest.mark.parametrize(
        "out_keys", [["episode_reward"], ["episode_reward_1", "episode_reward_2"]]
    )
    @pytest.mark.parametrize("reset_keys", [["_reset"], ["_reset1", "_reset2"]])
    def test_keys_length_errors(self, in_keys, reset_keys, out_keys, batch=10):
        reset_dict = {
            reset_key: torch.zeros(batch, dtype=torch.bool) for reset_key in reset_keys
        }
        reward_sum_dict = {out_key: torch.randn(batch) for out_key in out_keys}
        reset_dict.update(reward_sum_dict)
        td = TensorDict(reset_dict, [])

        if len(in_keys) != len(out_keys):
            with pytest.raises(
                ValueError,
                match="RewardSum expects the same number of input and output keys",
            ):
                RewardSum(in_keys=in_keys, reset_keys=reset_keys, out_keys=out_keys)
        else:
            t = RewardSum(in_keys=in_keys, reset_keys=reset_keys, out_keys=out_keys)

            if len(in_keys) != len(reset_keys):
                with pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Could not match the env reset_keys {reset_keys} with the in_keys {in_keys}"
                    ),
                ):
                    t._reset(td, td.empty())
            else:
                t._reset(td, td.empty())


class TestReward2Go(TransformBase):
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    @pytest.mark.parametrize("t", [3, 20])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, done_flags, gamma, t, device, rbclass):
        batch = 10
        batch_size = [batch, t]
        torch.manual_seed(0)
        out_key = "reward2go"
        r2g = Reward2GoTransform(gamma=gamma, out_keys=[out_key])
        rb = rbclass(storage=LazyTensorStorage(batch), transform=r2g)
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)

        td = TensorDict(
            {"misc": misc, "next": {"done": done, "reward": reward}},
            batch_size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample(13)
        assert sample[out_key].shape == (13, t, 1)
        assert (sample[out_key] != 0).all()

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    @pytest.mark.parametrize("t", [3, 20])
    def test_transform_offline_rb(self, done_flags, gamma, t, device):
        batch = 10
        batch_size = [batch, t]
        torch.manual_seed(0)
        out_key = "reward2go"
        r2g = Reward2GoTransform(gamma=gamma, out_keys=[out_key])
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(batch), transform=r2g)
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)

        td = TensorDict(
            {"misc": misc, "next": {"done": done, "reward": reward}},
            batch_size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample(13)
        assert sample[out_key].shape == (13, t, 1)
        assert (sample[out_key] != 0).all()

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    def test_transform_err(self, gamma, done_flags):
        device = "cpu"
        batch = [20]
        torch.manual_seed(0)
        done = torch.zeros(*batch, 1, dtype=torch.bool, device=device)
        done_flags = torch.randint(0, *batch, size=(done_flags,))
        done[done_flags] = True
        reward = torch.randn(*batch, 1, device=device)
        misc = torch.randn(*batch, 1, device=device)
        r2g = Reward2GoTransform(gamma=gamma)
        td = TensorDict(
            {"misc": misc, "reward": reward, "next": {"done": done}},
            batch,
            device=device,
        )
        with pytest.raises(KeyError, match="Could not find"):
            _ = r2g.inv(td)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    def test_transform_inverse(self, gamma, done_flags, device):
        batch = 10
        t = 20
        batch_size = [batch, t]
        torch.manual_seed(0)
        out_key = "reward2go"
        r2g = Reward2GoTransform(gamma=gamma, out_keys=[out_key])
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)

        td = TensorDict(
            {"misc": misc, "next": {"done": done, "reward": reward}},
            batch_size,
            device=device,
        )
        td = r2g.inv(td)
        assert td[out_key].shape == (batch, t, 1)
        assert (td[out_key] != 0).all()

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    def test_transform(self, gamma, done_flags):
        device = "cpu"
        batch = 10
        t = 20
        batch_size = [batch, t]
        torch.manual_seed(0)
        r2g = Reward2GoTransform(gamma=gamma)
        done = torch.zeros(*batch_size, 1, dtype=torch.bool, device=device)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)

        td = TensorDict(
            {"misc": misc, "next": {"done": done, "reward": reward}},
            batch_size,
            device=device,
        )
        td_out = r2g(td.clone())
        # assert that no transforms are done in the forward pass
        assert (td_out == td).all()

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_transform_env(self, gamma):
        t = Reward2GoTransform(gamma=gamma)
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = TransformedEnv(CountingBatchedEnv(), t)
        t = Reward2GoTransform(gamma=gamma)
        t = Compose(t)
        env = TransformedEnv(CountingBatchedEnv())
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            env.append_transform(t)

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_parallel_trans_env_check(self, gamma):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                Reward2GoTransform(gamma=gamma),
            )

        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = ParallelEnv(2, make_env)

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_single_trans_env_check(self, gamma):
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = TransformedEnv(
                ContinuousActionVecMockEnv(),
                Reward2GoTransform(gamma=gamma),
            )

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_serial_trans_env_check(self, gamma):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                Reward2GoTransform(gamma=gamma),
            )

        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = SerialEnv(2, make_env)

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_trans_serial_env_check(self, gamma):
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = TransformedEnv(
                SerialEnv(2, ContinuousActionVecMockEnv),
                Reward2GoTransform(gamma=gamma),
            )

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    def test_trans_parallel_env_check(self, gamma):
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            _ = TransformedEnv(
                ParallelEnv(2, ContinuousActionVecMockEnv),
                Reward2GoTransform(gamma=gamma),
            )

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    @pytest.mark.parametrize(
        "in_key", [("next", "reward"), ("next", "other", "nested")]
    )
    def test_transform_no_env(self, gamma, done_flags, in_key):
        device = "cpu"
        torch.manual_seed(0)
        batch = 10
        t = 20
        out_key = "reward2go"
        batch_size = [batch, t]
        torch.manual_seed(0)
        r2g = Reward2GoTransform(gamma=gamma, in_keys=[in_key], out_keys=[out_key])
        done = torch.zeros(*batch_size, 1, dtype=torch.bool)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)
        td = TensorDict(
            {
                "misc": misc,
                "next": {
                    "reward": reward,
                    "done": done,
                    "other": {"nested": reward.clone()},
                },
            },
            batch,
            device=device,
        )
        td = r2g.inv(td)
        assert td[out_key].shape == (batch, t, 1)
        assert td[out_key].all() != 0

    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    def test_transform_compose(self, gamma, done_flags):
        device = "cpu"
        torch.manual_seed(0)
        batch = 10
        t = 20
        out_key = "reward2go"
        compose = Compose(Reward2GoTransform(gamma=gamma, out_keys=[out_key]))
        batch_size = [batch, t]
        torch.manual_seed(0)
        done = torch.zeros(*batch_size, 1, dtype=torch.bool)
        for i in range(batch):
            while not done[i].any():
                done[i] = done[i].bernoulli_(0.1)
        reward = torch.randn(*batch_size, 1, device=device)
        misc = torch.randn(*batch_size, 1, device=device)
        td = TensorDict(
            {"misc": misc, "next": {"reward": reward, "done": done}},
            batch,
            device=device,
        )
        td_out = compose(td.clone())
        assert (td_out == td).all()
        td_out = compose.inv(td.clone())
        assert out_key in td_out.keys()

    def test_transform_model(self):
        raise pytest.skip("No model transform for Reward2Go")


class TestLineariseRewards(TransformBase):
    def test_weight_shape_error(self):
        with pytest.raises(
            ValueError, match="Expected weights to be a unidimensional tensor"
        ):
            LineariseRewards(in_keys=("reward",), weights=torch.ones(size=(2, 4)))

    def test_weight_no_sign_error(self):
        LineariseRewards(in_keys=("reward",), weights=-torch.ones(size=(2,)))

    def test_discrete_spec_error(self):
        with pytest.raises(
            NotImplementedError,
            match="Aggregation of rewards that take discrete values is not supported.",
        ):
            transform = LineariseRewards(in_keys=("reward",))
            reward_spec = Categorical(n=2)
            transform.transform_reward_spec(reward_spec)

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=2),
        ],
    )
    def test_single_trans_env_check(self, reward_spec: TensorSpec):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(reward_spec=reward_spec),
            LineariseRewards(in_keys=["reward"]),  # will use default weights
        )
        check_env_specs(env)

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=2),
        ],
    )
    def test_serial_trans_env_check(self, reward_spec: TensorSpec):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(reward_spec=reward_spec),
                LineariseRewards(in_keys=["reward"]),  # will use default weights
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=2),
        ],
    )
    def test_parallel_trans_env_check(
        self, maybe_fork_ParallelEnv, reward_spec: TensorSpec
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(reward_spec=reward_spec),
                LineariseRewards(in_keys=["reward"]),  # will use default weights
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=2),
        ],
    )
    def test_trans_serial_env_check(self, reward_spec: TensorSpec):
        def make_env():
            return ContinuousActionVecMockEnv(reward_spec=reward_spec)

        env = TransformedEnv(
            SerialEnv(2, make_env), LineariseRewards(in_keys=["reward"])
        )
        check_env_specs(env)

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=2),
        ],
    )
    def test_trans_parallel_env_check(
        self, maybe_fork_ParallelEnv, reward_spec: TensorSpec
    ):
        def make_env():
            return ContinuousActionVecMockEnv(reward_spec=reward_spec)

        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, make_env),
            LineariseRewards(in_keys=["reward"]),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("reward_key", [("reward",), ("agents", "reward")])
    @pytest.mark.parametrize(
        "num_rewards, weights",
        [
            (1, None),
            (3, None),
            (2, [1.0, 2.0]),
        ],
    )
    def test_transform_no_env(self, reward_key, num_rewards, weights):
        out_keys = reward_key[:-1] + ("scalar_reward",)
        t = LineariseRewards(in_keys=[reward_key], out_keys=[out_keys], weights=weights)
        td = TensorDict({reward_key: torch.randn(num_rewards)}, [])
        t._call(td)

        weights = torch.ones(num_rewards) if weights is None else torch.tensor(weights)
        expected = (weights * td[reward_key]).sum(-1, keepdim=True)
        torch.testing.assert_close(td[out_keys], expected)

    @pytest.mark.parametrize("reward_key", [("reward",), ("agents", "reward")])
    @pytest.mark.parametrize(
        "num_rewards, weights",
        [
            (1, None),
            (3, None),
            (2, [1.0, 2.0]),
        ],
    )
    def test_transform_compose(self, reward_key, num_rewards, weights):
        out_keys = reward_key[:-1] + ("scalar_reward",)
        t = Compose(
            LineariseRewards(in_keys=[reward_key], out_keys=[out_keys], weights=weights)
        )
        td = TensorDict({reward_key: torch.randn(num_rewards)}, [])
        t._call(td)

        weights = torch.ones(num_rewards) if weights is None else torch.tensor(weights)
        expected = (weights * td[reward_key]).sum(-1, keepdim=True)
        torch.testing.assert_close(td[out_keys], expected)

    def test_compose_with_reward_scaling(self):
        """Test that LineariseRewards properly registers output keys for use in Compose.

        This test reproduces the issue from GitHub #3237 where LineariseRewards
        does not register its output keys in the spec, causing subsequent transforms
        to fail during initialization.
        """
        # Create a simple env with multi-objective rewards
        env = self._DummyMultiObjectiveEnv(num_rewards=3)

        # Create a composed transform with LineariseRewards and RewardScaling
        # This should work without KeyError since transform_output_spec properly validates
        transform = Compose(
            LineariseRewards(
                in_keys=[("reward",)],
                out_keys=[
                    (
                        "nested",
                        "scalar_reward",
                    )
                ],
                weights=[1.0, 2.0, 3.0],
            ),
            RewardScaling(
                in_keys=[
                    (
                        "nested",
                        "scalar_reward",
                    )
                ],
                loc=0.0,
                scale=10.0,
            ),
        )

        # Apply transform to environment
        transformed_env = TransformedEnv(env, transform)

        # Check that specs are valid
        check_env_specs(transformed_env)

        # Verify the transform works correctly
        rollout = transformed_env.rollout(5)
        assert ("next", "nested", "scalar_reward") in rollout.keys(True)
        assert rollout[("next", "nested", "scalar_reward")].shape[-1] == 1

    def test_compose_with_nested_keys(self):
        """Test LineariseRewards with nested keys as described in GitHub #3237."""

        # Create a dummy env that produces nested rewards
        class _NestedRewardEnv(EnvBase):
            def __init__(self):
                super().__init__()
                self.observation_spec = Composite(
                    observation=UnboundedContinuous((*self.batch_size, 3))
                )
                self.action_spec = Categorical(
                    2, (*self.batch_size, 1), dtype=torch.bool
                )
                self.done_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
                self.full_done_spec["truncated"] = self.full_done_spec[
                    "terminated"
                ].clone()
                # Nested reward spec
                self.reward_spec = Composite(
                    agent1=Composite(
                        reward_vec=UnboundedContinuous(*self.batch_size, 2)
                    )
                )

            def _reset(self, tensordict: TensorDict) -> TensorDict:
                return self.observation_spec.sample()

            def _step(self, tensordict: TensorDict) -> TensorDict:
                return TensorDict(
                    {
                        ("observation"): self.observation_spec["observation"].sample(),
                        ("done"): False,
                        ("terminated"): False,
                        ("agent1", "reward_vec"): torch.randn(2),
                    }
                )

            def _set_seed(self, seed: int | None = None) -> None:
                pass

        env = _NestedRewardEnv()

        # This is the exact scenario from the GitHub issue
        transform = Compose(
            transforms=[
                LineariseRewards(
                    in_keys=[("agent1", "reward_vec")],
                    out_keys=[("agent1", "weighted_reward")],
                ),
                RewardScaling(
                    in_keys=[("agent1", "weighted_reward")], loc=0.0, scale=2.0
                ),
            ],
        )

        # This should work without KeyError
        transformed_env = TransformedEnv(env, transform)

        # Check that specs are valid
        check_env_specs(transformed_env)

        # Verify the transform works correctly
        rollout = transformed_env.rollout(5)
        assert ("next", "agent1", "weighted_reward") in rollout.keys(True)
        assert rollout[("next", "agent1", "weighted_reward")].shape[-1] == 1

    class _DummyMultiObjectiveEnv(EnvBase):
        """A dummy multi-objective environment."""

        def __init__(self, num_rewards: int) -> None:
            super().__init__()
            self._num_rewards = num_rewards

            self.observation_spec = Composite(
                observation=UnboundedContinuous((*self.batch_size, 3))
            )
            self.action_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
            self.done_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
            self.full_done_spec["truncated"] = self.full_done_spec["terminated"].clone()
            self.reward_spec = UnboundedContinuous(*self.batch_size, num_rewards)

        def _reset(self, tensordict: TensorDict) -> TensorDict:
            return self.observation_spec.sample()

        def _step(self, tensordict: TensorDict) -> TensorDict:
            done, terminated = False, False
            reward = torch.randn((self._num_rewards,))

            return TensorDict(
                {
                    ("observation"): self.observation_spec["observation"].sample(),
                    ("done"): done,
                    ("terminated"): terminated,
                    ("reward"): reward,
                }
            )

        def _set_seed(self, seed: int | None = None) -> None:
            pass

    @pytest.mark.parametrize(
        "num_rewards, weights",
        [
            (1, None),
            (3, None),
            (2, [1.0, 2.0]),
            (2, [1.0, -1.0]),
        ],
    )
    def test_transform_env(self, num_rewards, weights):
        weights = weights if weights is not None else [1.0 for _ in range(num_rewards)]

        transform = LineariseRewards(
            in_keys=("reward",), out_keys=("scalar_reward",), weights=weights
        )
        env = TransformedEnv(self._DummyMultiObjectiveEnv(num_rewards), transform)
        rollout = env.rollout(10)
        scalar_reward = rollout.get(("next", "scalar_reward"))
        assert scalar_reward.shape[-1] == 1

        expected = sum(
            w * r
            for w, r in zip(weights, rollout.get(("next", "reward")).split(1, dim=-1))
        )
        torch.testing.assert_close(scalar_reward, expected)

    @pytest.mark.parametrize(
        "num_rewards, weights",
        [
            (1, None),
            (3, None),
            (2, [1.0, 2.0]),
        ],
    )
    def test_transform_model(self, num_rewards, weights):
        weights_list = (
            weights if weights is not None else [1.0 for _ in range(num_rewards)]
        )
        transform = LineariseRewards(
            in_keys=("reward",), out_keys=("scalar_reward",), weights=weights_list
        )

        model = nn.Sequential(transform, nn.Identity())
        td = TensorDict({"reward": torch.randn(num_rewards)}, [])
        model(td)

        w = torch.tensor(weights_list)
        expected = (w * td["reward"]).sum(-1, keepdim=True)
        torch.testing.assert_close(td["scalar_reward"], expected)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        num_rewards = 3
        weights = None
        transform = LineariseRewards(
            in_keys=("reward",), out_keys=("scalar_reward",), weights=weights
        )

        rb = rbclass(storage=LazyTensorStorage(10))
        td = TensorDict({"reward": torch.randn(num_rewards)}, []).expand(10)
        rb.append_transform(transform)
        rb.extend(td)

        td = rb.sample(2)
        torch.testing.assert_close(
            td["scalar_reward"], td["reward"].sum(-1, keepdim=True)
        )

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for LineariseReward")

    @pytest.mark.parametrize(
        "weights, reward_spec, expected_spec",
        [
            (None, UnboundedContinuous(shape=3), UnboundedContinuous(shape=1)),
            (
                None,
                BoundedContinuous(0, 1, shape=3),
                BoundedContinuous(0, 3, shape=1),
            ),
            (
                None,
                BoundedContinuous(low=[-1.0, -2.0], high=[1.0, 2.0]),
                BoundedContinuous(low=-3.0, high=3.0, shape=1),
            ),
            (
                [1.0, 0.0],
                BoundedContinuous(
                    low=[-1.0, -2.0],
                    high=[1.0, 2.0],
                    shape=2,
                ),
                BoundedContinuous(low=-1.0, high=1.0, shape=1),
            ),
            (
                [1.0, -1.0],
                BoundedContinuous(
                    low=[-1.0, -2.0],
                    high=[1.0, 2.0],
                    shape=2,
                ),
                BoundedContinuous(low=-3.0, high=3.0, shape=1),
            ),
        ],
    )
    def test_reward_spec(
        self,
        weights,
        reward_spec: TensorSpec,
        expected_spec: TensorSpec,
    ) -> None:
        transform = LineariseRewards(in_keys=("reward",), weights=weights)
        assert transform.transform_reward_spec(reward_spec) == expected_spec

    def test_composite_reward_spec(self) -> None:
        weights = None
        reward_spec = Composite(
            agent_0=Composite(
                reward=BoundedContinuous(low=[0, 0, 0], high=[1, 1, 1], shape=3)
            ),
            agent_1=Composite(
                reward=BoundedContinuous(
                    low=[-1, -1, -1],
                    high=[1, 1, 1],
                    shape=3,
                )
            ),
        )
        expected_reward_spec = Composite(
            agent_0=Composite(reward=BoundedContinuous(low=0, high=3, shape=1)),
            agent_1=Composite(reward=BoundedContinuous(low=-3, high=3, shape=1)),
        )
        transform = LineariseRewards(
            in_keys=[("agent_0", "reward"), ("agent_1", "reward")], weights=weights
        )
        assert transform.transform_reward_spec(reward_spec) == expected_reward_spec

    @pytest.mark.parametrize("weights", [None, torch.ones(3)])
    def test_apply_transform_keepdim(self, weights):
        """Test that _apply_transform preserves the last dimension (keepdim).

        LineariseRewards spec sets shape [..., 1], so _apply_transform must
        output a tensor with the same trailing 1-dim, not squeeze it away.
        """
        transform = LineariseRewards(in_keys=("reward",), weights=weights)
        reward = torch.randn(2, 3)
        result = transform._apply_transform(reward)
        # The result should keep the last dimension as size 1
        assert result.shape == torch.Size(
            [2, 1]
        ), f"Expected shape [2, 1], got {result.shape}"

    @pytest.mark.parametrize(
        "reward_spec",
        [
            UnboundedContinuous(shape=3),
            BoundedContinuous(0, 1, shape=3),
        ],
    )
    def test_output_matches_spec(self, reward_spec):
        """Test that the reward tensor produced by the transform matches the spec."""
        env = TransformedEnv(
            ContinuousActionVecMockEnv(reward_spec=reward_spec),
            LineariseRewards(in_keys=["reward"]),
        )
        td = env.reset()
        td = env.rand_action(td)
        td = env.step(td)
        reward = td["next", "reward"]
        spec = env.reward_spec
        assert spec.is_in(
            reward
        ), f"Reward shape {reward.shape} doesn't match spec shape {spec.shape}"


class TestSignTransform(TransformBase):
    @staticmethod
    def check_sign_applied(tensor):
        return torch.logical_or(
            torch.logical_or(tensor == -1, tensor == 1), tensor == 0.0
        ).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        torch.manual_seed(0)
        rb = rbclass(storage=LazyTensorStorage(20))

        t = Compose(
            SignTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_sign", "reward_sign"],
                # What is stored within
                in_keys_inv=["input_signed"],
                # What the outside world sees
                out_keys_inv=["input_unsigned"],
            )
        )
        rb.append_transform(t)
        data = TensorDict({"observation": 1, "reward": 2, "input_unsigned": 3}, [])
        rb.add(data)
        sample = rb.sample(20)

        assert (sample["observation"] == 1).all()
        assert self.check_sign_applied(sample["obs_sign"])

        assert (sample["reward"] == 2).all()
        assert self.check_sign_applied(sample["reward_sign"])

        assert (sample["input_unsigned"] == 3).all()
        assert self.check_sign_applied(sample["input_signed"])

    def test_single_trans_env_check(self):
        env = ContinuousActionVecMockEnv()
        env = TransformedEnv(
            env,
            SignTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
            ),
        )
        check_env_specs(env)

    def test_transform_compose(self):
        t = Compose(
            SignTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_sign", "reward_sign"],
            )
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert self.check_sign_applied(data["obs_sign"])
        assert data["reward"] == 2
        assert self.check_sign_applied(data["reward_sign"])

    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, device):
        base_env = ContinuousActionVecMockEnv(device=device)
        env = TransformedEnv(
            base_env,
            SignTransform(
                in_keys=["observation", "reward"],
            ),
        )
        r = env.rollout(3)
        assert r.device == device
        assert self.check_sign_applied(r["observation"])
        assert self.check_sign_applied(r["next", "observation"])
        assert self.check_sign_applied(r["next", "reward"])
        check_env_specs(env)

    def test_transform_inverse(self):
        t = SignTransform(
            # What is seen inside
            in_keys_inv=["obs_signed", "reward_signed"],
            # What the outside world sees
            out_keys_inv=["obs", "reward"],
        )
        data = TensorDict({"obs": 1, "reward": 2}, [])
        data = t.inv(data)
        assert data["obs"] == 1
        assert self.check_sign_applied(data["obs_signed"])
        assert data["reward"] == 2
        assert self.check_sign_applied(data["reward_signed"])

    def test_transform_model(self):
        t = nn.Sequential(
            SignTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_sign", "reward_sign"],
            )
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert self.check_sign_applied(data["obs_sign"])
        assert data["reward"] == 2
        assert self.check_sign_applied(data["reward_sign"])

    def test_transform_no_env(self):
        t = SignTransform(
            in_keys=["observation", "reward"],
            out_keys=["obs_sign", "reward_sign"],
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert self.check_sign_applied(data["obs_sign"])
        assert data["reward"] == 2
        assert self.check_sign_applied(data["reward_sign"])

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = ContinuousActionVecMockEnv()
            return TransformedEnv(
                env,
                SignTransform(
                    in_keys=["observation", "reward"],
                    in_keys_inv=["observation_orig"],
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

    def test_serial_trans_env_check(self):
        def make_env():
            env = ContinuousActionVecMockEnv()
            return TransformedEnv(
                env,
                SignTransform(
                    in_keys=["observation", "reward"],
                    in_keys_inv=["observation_orig"],
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            SignTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
            ),
        )
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
            SignTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
            ),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass
