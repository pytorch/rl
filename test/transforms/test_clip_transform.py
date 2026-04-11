# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import torch

from _transforms_common import TransformBase
from tensordict import TensorDict
from torch import nn

from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import ClipTransform, Compose, SerialEnv, TransformedEnv
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


class TestClipTransform(TransformBase):
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        torch.manual_seed(0)
        rb = rbclass(storage=LazyTensorStorage(20))

        t = Compose(
            ClipTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_clip", "reward_clip"],
                in_keys_inv=["input_clip"],
                out_keys_inv=["input"],
                low=-0.1,
                high=0.1,
            )
        )
        rb.append_transform(t)
        data = TensorDict({"observation": 1, "reward": 2, "input": 3}, [])
        rb.add(data)
        sample = rb.sample(20)

        assert (sample["observation"] == 1).all()
        assert (sample["obs_clip"] == 0.1).all()
        assert (sample["reward"] == 2).all()
        assert (sample["reward_clip"] == 0.1).all()
        assert (sample["input"] == 3).all()
        assert (sample["input_clip"] == 0.1).all()

    def test_single_trans_env_check(self):
        env = ContinuousActionVecMockEnv()
        env = TransformedEnv(
            env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-0.1,
                high=0.1,
            ),
        )
        check_env_specs(env)

    def test_transform_compose(self):
        t = Compose(
            ClipTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_clip", "reward_clip"],
                low=-0.1,
                high=0.1,
            )
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert data["obs_clip"] == 0.1
        assert data["reward"] == 2
        assert data["reward_clip"] == 0.1

    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, device):
        base_env = ContinuousActionVecMockEnv(device=device)
        env = TransformedEnv(
            base_env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-0.1,
                high=0.1,
            ),
        )
        r = env.rollout(3)
        assert r.device == device
        assert (r["observation"] <= 0.1).all()
        assert (r["next", "observation"] <= 0.1).all()
        assert (r["next", "reward"] <= 0.1).all()
        assert (r["observation"] >= -0.1).all()
        assert (r["next", "observation"] >= -0.1).all()
        assert (r["next", "reward"] >= -0.1).all()
        check_env_specs(env)
        with pytest.raises(
            TypeError, match="Either one or both of `high` and `low` must be provided"
        ):
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=None,
                high=None,
            )
        with pytest.raises(TypeError, match="low and high must be scalars or None"):
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=torch.randn(2),
                high=None,
            )
        with pytest.raises(
            ValueError, match="`low` must be strictly lower than `high`"
        ):
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=1.0,
                high=-1.0,
            )
        env = TransformedEnv(
            base_env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=1.0,
                high=None,
            ),
        )
        check_env_specs(env)
        env = TransformedEnv(
            base_env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=None,
                high=1.0,
            ),
        )
        check_env_specs(env)
        env = TransformedEnv(
            base_env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-1,
                high=1,
            ),
        )
        check_env_specs(env)
        env = TransformedEnv(
            base_env,
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-torch.ones(()),
                high=1,
            ),
        )
        check_env_specs(env)

    def test_transform_inverse(self):
        t = ClipTransform(
            # What the outside world sees
            out_keys_inv=["observation", "reward"],
            # What the env expects
            in_keys_inv=["obs_clip", "reward_clip"],
            low=-0.1,
            high=0.1,
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t.inv(data)
        assert data["observation"] == 1
        assert data["obs_clip"] == 0.1
        assert data["reward"] == 2
        assert data["reward_clip"] == 0.1

    def test_transform_model(self):
        t = nn.Sequential(
            ClipTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_clip", "reward_clip"],
                low=-0.1,
                high=0.1,
            )
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert data["obs_clip"] == 0.1
        assert data["reward"] == 2
        assert data["reward_clip"] == 0.1

    def test_transform_no_env(self):
        t = ClipTransform(
            in_keys=["observation", "reward"],
            out_keys=["obs_clip", "reward_clip"],
            low=-0.1,
            high=0.1,
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t(data)
        assert data["observation"] == 1
        assert data["obs_clip"] == 0.1
        assert data["reward"] == 2
        assert data["reward_clip"] == 0.1

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = ContinuousActionVecMockEnv()
            return TransformedEnv(
                env,
                ClipTransform(
                    in_keys=["observation", "reward"],
                    in_keys_inv=["observation_orig"],
                    low=-0.1,
                    high=0.1,
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
                ClipTransform(
                    in_keys=["observation", "reward"],
                    in_keys_inv=["observation_orig"],
                    low=-0.1,
                    high=0.1,
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = ContinuousActionVecMockEnv()
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-0.1,
                high=0.1,
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
        env = ContinuousActionVecMockEnv()
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            ClipTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
                low=-0.1,
                high=0.1,
            ),
        )
        check_env_specs(env)
