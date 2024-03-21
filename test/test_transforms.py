# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import argparse
import contextlib
import importlib.util

import itertools
import pickle
import re
import sys
from copy import copy
from functools import partial

import numpy as np
import pytest

import tensordict.tensordict
import torch

from _utils_internal import (  # noqa
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionConvMockEnvNumpy,
    IncrementingEnv,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)
from tensordict import TensorDict, TensorDictBase, unravel_key
from tensordict.nn import TensorDictSequential
from tensordict.utils import _unravel_key_to_tuple, assert_allclose_td
from torch import multiprocessing as mp, nn, Tensor
from torchrl._utils import _replace_last, prod
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorStorage,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    ActionMask,
    BinarizeReward,
    BurnInTransform,
    CatFrames,
    CatTensors,
    CenterCrop,
    ClipTransform,
    Compose,
    DeviceCastTransform,
    DiscreteActionProjection,
    DMControlEnv,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvBase,
    EnvCreator,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    InitTracker,
    MultiStepTransform,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    PermuteTransform,
    PinMemoryTransform,
    R3MTransform,
    RandomCropTensorDict,
    RemoveEmptySpecs,
    RenameTransform,
    Resize,
    Reward2GoTransform,
    RewardClipping,
    RewardScaling,
    RewardSum,
    SelectTransform,
    SerialEnv,
    SignTransform,
    SqueezeTransform,
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TimeMaxPool,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
    VC1Transform,
    VIPTransform,
)
from torchrl.envs.libs.dm_control import _has_dm_control
from torchrl.envs.libs.gym import _has_gym, GymEnv, set_gym_backend
from torchrl.envs.transforms import VecNorm
from torchrl.envs.transforms.r3m import _R3MNet
from torchrl.envs.transforms.rlhf import KLRewardTransform
from torchrl.envs.transforms.transforms import _has_tv, FORWARD_NOT_IMPLEMENTED
from torchrl.envs.transforms.vc1 import _has_vc
from torchrl.envs.transforms.vip import _VIPNet, VIPRewardTransform
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.modules import GRUModule, LSTMModule, MLP, ProbabilisticActor, TanhNormal

TIMEOUT = 100.0

_has_gymnasium = importlib.util.find_spec("gymnasium") is not None


class TransformBase:
    """A base class for transform test.

    We ask for every new transform tests to be coded following this minimum requirement class.

    Of course, specific behaviours can also be tested separately.

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
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()), BinarizeReward()
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, lambda: ContinuousActionVecMockEnv()),
            BinarizeReward(),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

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
        raise pytest.skip("No inverse for BinerizedReward")


class TestClipTransform(TransformBase):
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        device = "cpu"
        batch = [20]
        torch.manual_seed(0)
        rb = rbclass(storage=LazyTensorStorage(20))

        t = Compose(
            ClipTransform(
                in_keys=["observation", "reward"],
                out_keys=["obs_clip", "reward_clip"],
                in_keys_inv=["input"],
                out_keys_inv=["input_clip"],
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
        with pytest.raises(ValueError, match="`low` must be stricly lower than `high`"):
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
            in_keys_inv=["observation", "reward"],
            out_keys_inv=["obs_clip", "reward_clip"],
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
            env.close()

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
            env.close()

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


class TestCatFrames(TransformBase):
    @pytest.mark.parametrize("out_keys", [None, ["obs2"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            CatFrames(dim=-1, N=3, in_keys=["observation"], out_keys=out_keys),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        check_env_specs(env)
        env2 = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, lambda: ContinuousActionVecMockEnv()),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_catframes_batching(
        self, batched_class, break_when_any_done, maybe_fork_ParallelEnv
    ):
        from _utils_internal import CARTPOLE_VERSIONED

        if batched_class is ParallelEnv:
            batched_class = maybe_fork_ParallelEnv

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            CatFrames(
                dim=-1, N=3, in_keys=["observation"], out_keys=["observation_cat"]
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                CatFrames(
                    dim=-1, N=3, in_keys=["observation"], out_keys=["observation_cat"]
                ),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    def test_nested(self, nested_dim=3, batch_size=(32, 1), rollout_length=6, cat_N=5):
        env = NestedCountingEnv(
            max_steps=20, nested_dim=nested_dim, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.action_spec, action_key=env.action_key
        )
        td = env.rollout(rollout_length, policy=policy)
        assert td[("data", "states")].shape == (
            *batch_size,
            rollout_length,
            nested_dim,
            1,
        )
        transformed_env = TransformedEnv(
            env, CatFrames(dim=-1, N=cat_N, in_keys=[("data", "states")])
        )
        td = transformed_env.rollout(rollout_length, policy=policy)
        assert td[("data", "states")].shape == (
            *batch_size,
            rollout_length,
            nested_dim,
            cat_N,
        )
        assert (
            (td[("data", "states")][0, 0, -1, 0]).eq(torch.arange(1, 1 + cat_N)).all()
        )
        assert (
            (td[("next", "data", "states")][0, 0, -1, 0])
            .eq(torch.arange(2, 2 + cat_N))
            .all()
        )

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        td = env.reset()
        assert td["observation"].shape[-1] == 9
        assert (td["observation"][..., :3] == td["observation"][..., 3:6]).all()
        assert (td["observation"][..., 3:6] == td["observation"][..., 6:9]).all()
        old = td["observation"][..., 3:6].clone()
        td = env.rand_step(td)
        assert (td["next", "observation"][..., :3] == old).all()
        assert (
            td["next", "observation"][..., :3] == td["next", "observation"][..., 3:6]
        ).all()
        assert (
            td["next", "observation"][..., 3:6] != td["next", "observation"][..., 6:9]
        ).any()

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env_clone(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        td = env.reset()
        td = env.rand_step(td)
        cloned = env.transform.clone()
        value_at_clone = td["next", "observation"].clone()
        for _ in range(10):
            td = env.rand_step(td)
            td = step_mdp(td)
        assert (td["observation"] != value_at_clone).any()
        assert (td["observation"] == env.transform._cat_buffers_observation).all()
        assert (
            cloned._cat_buffers_observation == env.transform._cat_buffers_observation
        ).all()
        assert cloned is not env.transform

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["zeros", "constant", "same"])
    def test_transform_model(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )
        envbase = ContinuousActionVecMockEnv()
        env = TransformedEnv(envbase, cat_frames)

        torch.manual_seed(10)
        env.set_seed(10)
        td = env.rollout(10)

        torch.manual_seed(10)
        envbase.set_seed(10)
        tdbase = envbase.rollout(10)

        tdbase0 = tdbase.clone()

        model = nn.Sequential(cat_frames2, nn.Identity())
        model(tdbase)
        assert assert_allclose_td(td, tdbase)

        with pytest.warns(UserWarning):
            tdbase0.names = None
            model(tdbase0)
        tdbase0.batch_size = []
        with pytest.raises(
            ValueError, match="CatFrames cannot process unbatched tensordict"
        ):
            model(tdbase0)
        tdbase0.batch_size = [10]
        tdbase0 = tdbase0.expand(5, 10)
        tdbase0_copy = tdbase0.transpose(0, 1)
        tdbase0.refine_names("time", None)
        tdbase0_copy.names = [None, "time"]
        v1 = model(tdbase0)
        v2 = model(tdbase0_copy)
        # check that swapping dims and names leads to same result
        assert_allclose_td(v1, v2.transpose(0, 1))

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "zeros", "constant"])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, dim, N, padding, rbclass):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )

        env = TransformedEnv(ContinuousActionVecMockEnv(), cat_frames)
        td = env.rollout(10)

        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(cat_frames2)
        rb.add(td.exclude(*out_keys, ("next", out_keys[0])))
        tdsample = rb.sample(1).squeeze(0).exclude("index")
        for key in td.keys(True, True):
            assert (tdsample[key] == td[key]).all(), key
        assert (tdsample["out_" + key1] == td["out_" + key1]).all()
        assert (tdsample["next", "out_" + key1] == td["next", "out_" + key1]).all()

    @pytest.mark.parametrize("dim", [-1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "zeros", "constant"])
    def test_transform_as_inverse(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        in_keys = ["observation", ("next", "observation")]
        rollout_length = 10
        cat_frames = CatFrames(
            N=N, in_keys=in_keys, dim=dim, padding=padding, as_inverse=True
        )

        env1 = TransformedEnv(
            ContinuousActionVecMockEnv(),
        )
        env2 = TransformedEnv(
            ContinuousActionVecMockEnv(),
            CatFrames(N=N, in_keys=in_keys, dim=dim, padding=padding, as_inverse=True),
        )
        obs_dim = env1.observation_spec["observation_orig"].shape[0]
        td = env1.rollout(rollout_length)

        transformed_td = cat_frames._inv_call(td)
        assert transformed_td.get(in_keys[0]).shape == (rollout_length, obs_dim * N)
        assert transformed_td.get(in_keys[1]).shape == (rollout_length, obs_dim * N)
        with pytest.raises(
            Exception,
            match="CatFrames as inverse is not supported as a transform for environments, only for replay buffers.",
        ):
            env2.rollout(rollout_length)

    def test_catframes_transform_observation_spec(self):
        N = 4
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        cat_frames = CatFrames(
            N=N,
            in_keys=keys,
            dim=-3,
        )
        mins = [0, 0.5]
        maxes = [0.5, 1]
        observation_spec = CompositeSpec(
            {
                key: BoundedTensorSpec(
                    space_min, space_max, (1, 3, 3), dtype=torch.double
                )
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        result = cat_frames.transform_observation_spec(observation_spec)
        observation_spec = CompositeSpec(
            {
                key: BoundedTensorSpec(
                    space_min, space_max, (1, 3, 3), dtype=torch.double
                )
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        final_spec = result[key2]
        assert final_spec.shape[0] == N
        for key in keys:
            for i in range(N):
                assert torch.equal(
                    result[key].space.high[i], observation_spec[key].space.high[0]
                )
                assert torch.equal(
                    result[key].space.low[i], observation_spec[key].space.low[0]
                )

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(1, 4))
    @pytest.mark.parametrize("dim", [-3, -2, 1])
    @pytest.mark.parametrize("N", [2, 4])
    def test_transform_no_env(self, device, d, batch_size, dim, N):
        key1 = "first key"
        key2 = ("second", "key")
        keys = [key1, key2]
        extra_d = (3,) * (-dim - 1)
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        if dim > 0:
            with pytest.raises(
                ValueError, match="dim must be < 0 to accomodate for tensordict"
            ):
                cat_frames = CatFrames(N=N, in_keys=keys, dim=dim)
            return
        cat_frames = CatFrames(N=N, in_keys=keys, dim=dim)

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        slices = (slice(None),) * (-dim - 1)
        index1 = (Ellipsis, slice(None, -d), *slices)
        index2 = (Ellipsis, slice(-d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v1 = latest_frame[index1]

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        index1 = (Ellipsis, slice(None, -2 * d), *slices)
        index2 = (Ellipsis, slice(-2 * d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v2 = latest_frame[index1]

        # we don't want the same tensor to be returned twice, but they're all copies of the same buffer
        assert v1 is not v2

    @pytest.mark.skipif(not _has_gym, reason="gym required for this test")
    @pytest.mark.parametrize("padding", ["zeros", "constant", "same"])
    @pytest.mark.parametrize("envtype", ["gym", "conv"])
    def test_tranform_offline_against_online(self, padding, envtype):
        torch.manual_seed(0)
        key = "observation" if envtype == "gym" else "pixels"
        env = SerialEnv(
            3,
            lambda: TransformedEnv(
                GymEnv("CartPole-v1")
                if envtype == "gym"
                else DiscreteActionConvMockEnv(),
                CatFrames(
                    dim=-3 if envtype == "conv" else -1,
                    N=5,
                    in_keys=[key],
                    out_keys=[f"{key}_cat"],
                    padding=padding,
                ),
            ),
        )
        env.set_seed(0)

        r = env.rollout(100, break_when_any_done=False)

        c = CatFrames(
            dim=-3 if envtype == "conv" else -1,
            N=5,
            in_keys=[key, ("next", key)],
            out_keys=[f"{key}_cat2", ("next", f"{key}_cat2")],
            padding=padding,
        )

        r2 = c(r)

        torch.testing.assert_close(r2[f"{key}_cat2"], r2[f"{key}_cat"])
        torch.testing.assert_close(r2["next", f"{key}_cat2"], r2["next", f"{key}_cat"])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(2, 3))
    @pytest.mark.parametrize(
        "dim",
        [-3],
    )
    @pytest.mark.parametrize("N", [2, 4])
    def test_transform_compose(self, device, d, batch_size, dim, N):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        extra_d = (3,) * (-dim - 1)
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        cat_frames = Compose(CatFrames(N=N, in_keys=keys, dim=dim))

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        slices = (slice(None),) * (-dim - 1)
        index1 = (Ellipsis, slice(None, -d), *slices)
        index2 = (Ellipsis, slice(-d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v1 = latest_frame[index1]

        tdclone = cat_frames._call(td.clone())
        latest_frame = tdclone.get(key2)

        assert latest_frame.shape[dim] == N * d
        index1 = (Ellipsis, slice(None, -2 * d), *slices)
        index2 = (Ellipsis, slice(-2 * d, None), *slices)
        assert (latest_frame[index1] == 0).all()
        assert (latest_frame[index2] == 1).all()
        v2 = latest_frame[index1]

        # we don't want the same tensor to be returned twice, but they're all copies of the same buffer
        assert v1 is not v2

    @pytest.mark.parametrize("device", get_default_devices())
    def test_catframes_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        cat_frames = CatFrames(N=N, in_keys=keys, dim=-3, reset_key="_reset")

        cat_frames._call(td.clone())
        buffer = getattr(cat_frames, f"_cat_buffers_{key1}")

        tdc = td.clone()
        passed_back_td = cat_frames._reset(tdc, tdc)

        # assert tdc is passed_back_td
        # assert (buffer == 0).all()
        #
        # _ = cat_frames._call(tdc)
        assert (buffer != 0).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for CatFrames")

    @pytest.mark.parametrize("padding_value", [2, 0.5, -1])
    def test_constant_padding(self, padding_value):
        key1 = "first_key"
        N = 4
        key1_tensor = torch.zeros((1, 1))
        td = TensorDict({key1: key1_tensor}, [1])
        cat_frames = CatFrames(
            N=N,
            in_keys=key1,
            out_keys="cat_" + key1,
            dim=-1,
            padding="constant",
            padding_value=padding_value,
        )

        cat_td = cat_frames._call(td.clone())
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 1
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 2
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 3
        cat_td = cat_frames._call(cat_td)
        assert (cat_td.get("cat_first_key") == padding_value).sum() == N - 4


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="Testing R3M on cuda only")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@pytest.mark.parametrize(
    "model",
    [
        "resnet18",
    ],
)  # 1226: "resnet34", "resnet50"])
class TestR3M(TransformBase):
    def test_transform_inverse(self, model, device):
        raise pytest.skip("no inverse for R3MTransform")

    def test_transform_compose(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = Compose(
            R3MTransform(
                model,
                in_keys=in_keys,
                out_keys=out_keys,
                tensor_pixels_keys=tensor_pixels_key,
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        r3m(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 512

    def test_transform_no_env(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        r3m(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 512

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, model, device, rbclass):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(r3m)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 512

    def test_transform_model(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(r3m, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 512

    def test_parallel_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                R3MTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = ParallelEnv(2, make_env)
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                R3MTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), r3m
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_trans_serial_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), r3m
        )
        check_env_specs(transformed_env)

    def test_single_trans_env_check(self, model, device):
        if model != "resnet18":
            # we don't test other resnets for the sake of speed and we don't use skip
            # to avoid polluting the log with it
            return
        tensor_pixels_key = None
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), r3m
        )
        check_env_specs(transformed_env)

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_transform_env(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, r3m)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        if tensor_pixels_key:
            expected_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("stack_images", [True, False])
    @pytest.mark.parametrize(
        "parallel",
        [
            True,
            False,
        ],
    )
    def test_r3m_mult_images(self, model, device, stack_images, parallel):
        in_keys = ["pixels", "pixels2"]
        out_keys = ["vec"] if stack_images else ["vec", "vec2"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            stack_images=stack_images,
        )

        def base_env_constructor():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                CatTensors(["pixels"], "pixels2", del_keys=False),
            )

        assert base_env_constructor().device == device
        if parallel:
            base_env = ParallelEnv(2, base_env_constructor)
        else:
            base_env = base_env_constructor()
        assert base_env.device == device

        transformed_env = TransformedEnv(base_env, r3m)
        assert transformed_env.device == device
        assert r3m.device == device

        td = transformed_env.reset()
        assert td.device == device
        if stack_images:
            expected_keys = {"pixels_orig", "done", "vec", "terminated"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys
        else:
            expected_keys = {"pixels_orig", "done", "vec", "vec2", "terminated"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        if not stack_images:
            expected_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == expected_keys, set(td.keys()) - expected_keys
        transformed_env.close()

    def test_r3m_parallel(self, model, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(4, lambda: DiscreteActionConvMockEnvNumpy().to(device))
        transformed_env = TransformedEnv(base_env, r3m)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys(True)) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "next",
            }
        )
        assert set(td.keys(True)) == expected_keys, set(td.keys()) - expected_keys
        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    @pytest.mark.parametrize(
        "in_keys",
        [["pixels"], ["pixels_1", "pixels_2", "pixels_3"]],
    )
    @pytest.mark.parametrize(
        "out_keys",
        [["r3m_vec"], ["r3m_vec_1", "r3m_vec_2", "r3m_vec_3"]],
    )
    def test_r3mnet_transform_observation_spec(
        self, in_keys, out_keys, del_keys, device, model
    ):
        r3m_net = _R3MNet(in_keys, out_keys, model, del_keys)

        observation_spec = CompositeSpec(
            {key: BoundedTensorSpec(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = CompositeSpec(
                {
                    key: UnboundedContinuousTensorSpec(r3m_net.outdim, device)
                    for key in out_keys
                }
            )

            observation_spec_out = r3m_net.transform_observation_spec(observation_spec)

            for key in in_keys:
                assert key not in observation_spec_out
            for key in out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].device == exp_ts[key].device
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
        else:
            ts_dict = {}
            for key in in_keys:
                ts_dict[key] = observation_spec[key]
            for key in out_keys:
                ts_dict[key] = UnboundedContinuousTensorSpec(r3m_net.outdim, device)
            exp_ts = CompositeSpec(ts_dict)

            observation_spec_out = r3m_net.transform_observation_spec(observation_spec)

            for key in in_keys + out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
                assert observation_spec_out[key].device == exp_ts[key].device

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_r3m_spec_against_real(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        r3m = R3MTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, r3m)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + list(transformed_env.observation_spec.keys())
            + ["action"]
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
                "terminated",
                "done",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


class TestStepCounter(TransformBase):
    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_step_count_gym(self):
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), StepCounter(max_steps=30))
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_step_count_gym_doublecount(self):
        # tests that 2 truncations can be used together
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(
                StepCounter(max_steps=2),
                StepCounter(max_steps=3),  # this one will be ignored
            ),
        )
        r = env.rollout(10, break_when_any_done=False)
        assert (
            r.get(("next", "truncated")).squeeze().nonzero().squeeze(-1)
            == torch.arange(1, 10, 2)
        ).all()

    @pytest.mark.skipif(not _has_dm_control, reason="no dm_control detected")
    def test_step_count_dmc(self):
        env = TransformedEnv(DMControlEnv("cheetah", "run"), StepCounter(max_steps=30))
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_stepcount_batching(self, batched_class, break_when_any_done):
        from _utils_internal import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            StepCounter(max_steps=15),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()), StepCounter(max_steps=15)
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    @pytest.mark.parametrize("update_done", [False, True])
    @pytest.mark.parametrize("max_steps", [10, None])
    def test_single_trans_env_check(self, update_done, max_steps):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            StepCounter(max_steps=max_steps, update_done=update_done),
        )
        check_env_specs(env)
        r = env.rollout(100, break_when_any_done=False)
        if update_done and max_steps:
            assert r["next", "done"][r["next", "truncated"]].all()
        elif max_steps:
            assert not r["next", "done"][r["next", "truncated"]].all()
        else:
            assert "truncated" not in r.keys()
            assert ("next", "truncated") not in r.keys(True)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), StepCounter(10)
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), StepCounter(10))
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Gym not found")
    def test_transform_env(self):
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), StepCounter(10))
        td = env.rollout(100, break_when_any_done=False)
        assert td["step_count"].max() == 9
        assert td.shape[-1] == 100

    @pytest.mark.parametrize("step_key", ["step_count", "other-key"])
    @pytest.mark.parametrize("max_steps", [None, 10])
    @pytest.mark.parametrize("nested_done", [True, False])
    def test_nested(
        self, step_key, nested_done, max_steps, batch_size=(32, 2), rollout_length=15
    ):
        env = NestedCountingEnv(
            max_steps=20, nest_done=nested_done, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.action_spec, action_key=env.action_key
        )
        transformed_env = TransformedEnv(
            env,
            StepCounter(
                max_steps=max_steps,
                step_count_key=step_key,
            ),
        )
        step_key = transformed_env.transform.step_count_keys[0]
        td = transformed_env.rollout(
            rollout_length, policy=policy, break_when_any_done=False
        )
        if nested_done:
            step = td[step_key][0, 0, :, 0, 0].clone()
            last_step = td[step_key][:, :, -1, :, :].clone()
        else:
            step = td[step_key][0, 0, :, 0].clone()
            last_step = td[step_key][:, :, -1, :].clone()
        if max_steps is None:
            assert step.eq(torch.arange(rollout_length)).all()
        else:
            assert step[:max_steps].eq(torch.arange(max_steps)).all()
            assert step[max_steps:].eq(torch.arange(rollout_length - max_steps)).all()

        if nested_done:
            for done_key in env.done_keys:
                reset_key = (*done_key[:-1], "_reset")
                _reset = env.full_done_spec[done_key].rand()
                break
        else:
            reset_key = "_reset"
            _reset = env.full_done_spec["done"].rand()
        td_reset = transformed_env.reset(
            TensorDict(
                {reset_key: _reset, step_key: last_step},
                batch_size=env.batch_size,
                device=env.device,
            )
        )
        assert (td_reset[step_key][_reset] == 0).all()
        assert (td_reset[step_key][~_reset] == last_step[~_reset]).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        transform = StepCounter(10)
        rb = rbclass(storage=LazyTensorStorage(20))
        td = TensorDict({"a": torch.randn(10)}, [10])
        rb.extend(td)
        rb.append_transform(transform)
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            rb.sample(5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    @pytest.mark.parametrize("max_steps", [None, 1, 5, 50])
    @pytest.mark.parametrize("reset_workers", [True, False])
    def test_transform_compose(self, max_steps, device, batch, reset_workers):
        torch.manual_seed(0)
        step_counter = Compose(StepCounter(max_steps))
        done = torch.zeros(*batch, 1, dtype=torch.bool)
        td = TensorDict({"done": done, ("next", "done"): done}, batch, device=device)
        _reset = torch.zeros((), dtype=torch.bool, device=device)
        while not _reset.any() and reset_workers:
            _reset = torch.randn(done.shape, device=device) < 0
            td.set("_reset", _reset)
            td.set("done", _reset)
            td.set("terminated", _reset)
            td.set(("next", "terminated"), done)
            td.set(("next", "done"), done)
        td.set("step_count", torch.zeros(*batch, 1, dtype=torch.int))
        step_counter[0]._step_count_keys = ["step_count"]
        step_counter[0]._terminated_keys = ["terminated"]
        step_counter[0]._truncated_keys = ["truncated"]
        step_counter[0]._reset_keys = ["_reset"]
        step_counter[0]._done_keys = ["done"]
        td_reset = td.empty()
        td_reset = step_counter._reset(td, td_reset)
        assert not torch.all(td_reset.get("step_count"))
        i = 0
        td_next = td.get("next")
        td = td_reset
        while max_steps is None or i < max_steps:
            td_next = step_counter._step(td, td_next)
            td.set("next", td_next)

            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            td["next", "terminated"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td_reset = td.empty()
        if reset_workers:
            td.set("_reset", _reset)
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), _reset) == 0
            )
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), ~_reset) == i
            )
        else:
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(td_reset.get("step_count") == 0)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for StepCounter")

    def test_transform_model(self):
        transform = StepCounter(10)
        _ = TensorDict({"a": torch.randn(10)}, [10])
        model = nn.Sequential(transform, nn.Identity())
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            model(5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    @pytest.mark.parametrize("max_steps", [None, 1, 5, 50])
    @pytest.mark.parametrize("reset_workers", [True, False])
    def test_transform_no_env(self, max_steps, device, batch, reset_workers):
        torch.manual_seed(0)
        step_counter = StepCounter(max_steps)
        done = torch.zeros(*batch, 1, dtype=torch.bool)
        td = TensorDict({"done": done, ("next", "done"): done}, batch, device=device)
        _reset = torch.zeros((), dtype=torch.bool, device=device)
        while not _reset.any() and reset_workers:
            _reset = torch.randn(done.shape, device=device) < 0
            td.set("_reset", _reset)
            td.set("terminated", _reset)
            td.set(("next", "terminated"), done)
            td.set("done", _reset)
            td.set(("next", "done"), done)
        td.set("step_count", torch.zeros(*batch, 1, dtype=torch.int))
        step_counter._step_count_keys = ["step_count"]
        step_counter._done_keys = ["done"]
        step_counter._terminated_keys = ["terminated"]
        step_counter._truncated_keys = ["truncated"]
        step_counter._reset_keys = ["_reset"]
        step_counter._completed_keys = ["completed"]

        td_reset = td.empty()
        td_reset = step_counter._reset(td, td_reset)
        assert not torch.all(td_reset.get("step_count"))
        i = 0
        td_next = td.get("next")
        td = td_reset
        while max_steps is None or i < max_steps:
            td_next = step_counter._step(td, td_next)
            td.set("next", td_next)

            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            td["next", "terminated"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td_reset = td.empty()
        if reset_workers:
            td.set("_reset", _reset)
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), _reset) == 0
            )
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), ~_reset) == i
            )
        else:
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(td_reset.get("step_count") == 0)

    def test_step_counter_observation_spec(self):
        transformed_env = TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(50))
        check_env_specs(transformed_env)
        transformed_env.close()

    def test_stepcounter_ignore(self):
        # checks that step_count_keys respect the convention that nested dones should
        # be ignored if there is a done in a root td
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=True, nest_done=True), StepCounter()
        )
        assert len(env.transform.step_count_keys) == 1
        assert env.transform.step_count_keys[0] == "step_count"
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=False, nest_done=True), StepCounter()
        )
        assert len(env.transform.step_count_keys) == 1
        assert env.transform.step_count_keys[0] == ("data", "step_count")


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

    def test_parallel_trans_env_check(self):
        def make_env():
            ct = CatTensors(
                in_keys=["observation", "observation_orig"],
                out_key="observation_out",
                dim=-1,
                del_keys=False,
            )
            return TransformedEnv(ContinuousActionVecMockEnv(), ct)

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        ct = CatTensors(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )

        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        ct = CatTensors(
            in_keys=["observation", "observation_orig"],
            out_key="observation_out",
            dim=-1,
            del_keys=False,
        )

        env = TransformedEnv(ParallelEnv(2, ContinuousActionVecMockEnv), ct)
        try:
            check_env_specs(env)
        finally:
            env.close()

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
            observation_spec = BoundedTensorSpec(0, 1, (1, 4, 32))
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, len(keys) * 4, 32])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(0, 1, (1, 4, 32)) for key in keys}
            )
            observation_spec = cattensors.transform_observation_spec(observation_spec)
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
class TestCenterCrop(TransformBase):
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("h", [None, 21])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        cc(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = cc.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, h])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = cc.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, h])

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize(
        "batch",
        [
            [2],
        ],
    )
    @pytest.mark.parametrize(
        "h",
        [
            None,
        ],
    )
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_model(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = nn.Sequential(cc, nn.Identity())
        model(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])
        assert (td.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize(
        "batch",
        [
            [2],
        ],
    )
    @pytest.mark.parametrize(
        "h",
        [
            None,
        ],
    )
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        model = Compose(cc)
        tdc = model(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()
        tdc = model._call(td.clone())
        for key in keys:
            assert tdc.get(key).shape[-2:] == torch.Size([20, h])
        assert (tdc.get("dont touch") == dont_touch).all()

    @pytest.mark.parametrize("nchannels", [3])
    @pytest.mark.parametrize(
        "batch",
        [
            [2],
        ],
    )
    @pytest.mark.parametrize(
        "h",
        [
            None,
        ],
    )
    @pytest.mark.parametrize("keys", [["observation_pixels"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
        keys,
        h,
        nchannels,
        batch,
    ):
        torch.manual_seed(0)
        dont_touch = torch.randn(
            *batch,
            nchannels,
            16,
            16,
        )
        cc = CenterCrop(w=20, h=h, in_keys=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(
                    *batch,
                    nchannels,
                    16,
                    16,
                )
                for key in keys
            },
            batch,
        )
        td.set("dont touch", dont_touch.clone())
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(cc)
        rb.extend(td)
        td = rb.sample(10)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, h])

    def test_single_trans_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        keys = ["pixels"]

        def make_env():
            ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
            return TransformedEnv(DiscreteActionConvMockEnvNumpy(), ct)

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(SerialEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(ParallelEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_gym, reason="No Gym detected")
    @pytest.mark.parametrize("out_key", [None, ["outkey"], [("out", "key")]])
    def test_transform_env(self, out_key):
        keys = ["pixels"]
        ct = Compose(
            ToTensorImage(), CenterCrop(out_keys=out_key, w=20, h=20, in_keys=keys)
        )
        env = TransformedEnv(GymEnv(PONG_VERSIONED()), ct)
        td = env.reset()
        if out_key is None:
            assert td["pixels"].shape == torch.Size([3, 20, 20])
        else:
            assert td[out_key[0]].shape == torch.Size([3, 20, 20])
        check_env_specs(env)

    def test_transform_inverse(self):
        raise pytest.skip("CenterCrop does not have an inverse method.")


class TestDiscreteActionProjection(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            DiscreteActionProjection(7, 10),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            DiscreteActionProjection(7, 10),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("action_key", ["action", ("nested", "stuff")])
    def test_transform_no_env(self, action_key):
        t = DiscreteActionProjection(7, 10, action_key=action_key)
        td = TensorDict(
            {action_key: nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        assert td[action_key].shape[-1] == 10
        assert (td[action_key].sum(-1) == 1).all()
        out = t.inv(td)
        assert out[action_key].shape[-1] == 7
        assert (out[action_key].sum(-1) == 1).all()

    def test_transform_compose(self):
        t = Compose(DiscreteActionProjection(7, 10))
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        assert td["action"].shape[-1] == 10
        assert (td["action"].sum(-1) == 1).all()
        out = t.inv(td)
        assert out["action"].shape[-1] == 7
        assert (out["action"].sum(-1) == 1).all()

    def test_transform_env(self):
        raise pytest.skip("Tested in test_transform_inverse")

    @pytest.mark.parametrize("include_forward", [True, False])
    def test_transform_model(self, include_forward):
        t = DiscreteActionProjection(7, 10, include_forward=include_forward)
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(7, (10, 4, 1)), 7)},
            [10, 4],
        )
        td = model(td)
        assert td["action"].shape[-1] == 10 if include_forward else 7

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("include_forward", [True, False])
    def test_transform_rb(self, include_forward, rbclass):
        rb = rbclass(storage=LazyTensorStorage(10))
        t = DiscreteActionProjection(7, 10, include_forward=include_forward)
        rb.append_transform(t)
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        rb.extend(td)

        storage = rb._storage._storage[:]

        assert storage["action"].shape[-1] == 7
        td = rb.sample(10)
        assert td["action"].shape[-1] == 10 if include_forward else 7

    def test_transform_inverse(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
        )
        assert env.action_spec.space.n == 10
        assert env.action_spec.rand().shape == torch.Size([10])
        # check that transforming the action does not affect the outer td
        td = env.reset()
        td_out = env.rand_step(td)
        assert td_out["action"].shape == torch.Size([10])
        assert td is td_out


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
            action_spec = BoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
            input_spec = CompositeSpec(
                full_action_spec=CompositeSpec(action=action_spec), full_state_spec=None
            )
            action_spec = double2float.transform_input_spec(input_spec)
            assert action_spec.dtype == torch.float
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
                    for key in keys
                }
            )
            observation_spec = double2float.transform_observation_spec(observation_spec)
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
        base_env = ContinuousActionVecMockEnv()
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

    def test_parallel_trans_env_check(self, dtype_fixture):  # noqa: F811
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(dtype=torch.float64),
                DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
            )

        try:
            env = ParallelEnv(1, make_env)
            check_env_specs(env)
        finally:
            env.close()
            del env

    def test_trans_serial_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv(dtype=torch.float64)),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            ParallelEnv(2, lambda: ContinuousActionVecMockEnv(dtype=torch.float64)),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

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


class TestExcludeTransform(TransformBase):
    class EnvWithManyKeys(EnvBase):
        def __init__(self):
            super().__init__()
            self.observation_spec = CompositeSpec(
                a=UnboundedContinuousTensorSpec(3),
                b=UnboundedContinuousTensorSpec(3),
                c=UnboundedContinuousTensorSpec(3),
            )
            self.reward_spec = UnboundedContinuousTensorSpec(1)
            self.action_spec = UnboundedContinuousTensorSpec(2)

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

        def _set_seed(self, seed):
            return seed + 1

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

    def test_parallel_trans_env_check(self):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                ExcludeTransform("observation_copy"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            ExcludeTransform("observation_copy"),
        )
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), t)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            ExcludeTransform("observation_copy"),
        )
        env = TransformedEnv(ParallelEnv(2, ContinuousActionVecMockEnv), t)
        try:
            check_env_specs(env)
        finally:
            env.close()

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
            self.observation_spec = CompositeSpec(
                a=UnboundedContinuousTensorSpec(3),
                b=UnboundedContinuousTensorSpec(3),
                c=UnboundedContinuousTensorSpec(3),
            )
            self.reward_spec = UnboundedContinuousTensorSpec(1)
            self.action_spec = UnboundedContinuousTensorSpec(2)

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

        def _set_seed(self, seed):
            return seed + 1

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

    def test_parallel_trans_env_check(self):
        def make_env():
            t = Compose(
                CatTensors(
                    in_keys=["observation"], out_key="observation_copy", del_keys=False
                ),
                SelectTransform("observation", "observation_orig"),
            )
            env = TransformedEnv(ContinuousActionVecMockEnv(), t)
            return env

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            SelectTransform("observation", "observation_orig"),
        )
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), t)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        t = Compose(
            CatTensors(
                in_keys=["observation"], out_key="observation_copy", del_keys=False
            ),
            SelectTransform("observation", "observation_orig"),
        )
        env = TransformedEnv(ParallelEnv(2, ContinuousActionVecMockEnv), t)
        try:
            check_env_specs(env)
        finally:
            env.close()

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


class TestFlattenObservation(TransformBase):
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            FlattenObservation(-3, -1, out_keys=out_keys),
        )
        check_env_specs(env)
        if out_keys:
            assert out_keys[0] in env.reset().keys()

    def test_serial_trans_env_check(self):
        def make_env():
            env = TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), FlattenObservation(-3, -1)
            )
            return env

        env = SerialEnv(2, make_env)

    def test_parallel_trans_env_check(self):
        def make_env():
            env = TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), FlattenObservation(-3, -1)
            )
            return env

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            FlattenObservation(
                -3,
                -1,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            FlattenObservation(
                -3,
                -1,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, size, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        start_dim = -3 - len(size)
        flatten = FlattenObservation(start_dim, -3, in_keys=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        flatten(td)
        expected_size = prod(size + [nchannels])
        for key in keys:
            assert td.get(key).shape[-3] == expected_size
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = flatten.transform_observation_spec(observation_spec)
            assert observation_spec.shape[-3] == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = flatten.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, size, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        start_dim = -3 - len(size)
        flatten = Compose(FlattenObservation(start_dim, -3, in_keys=keys))
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        flatten(td)
        expected_size = prod(size + [nchannels])
        for key in keys:
            assert td.get(key).shape[-3] == expected_size
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = flatten.transform_observation_spec(observation_spec)
            assert observation_spec.shape[-3] == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = flatten.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED()), FlattenObservation(-3, -1, out_keys=out_keys)
        )
        check_env_specs(env)
        if out_keys:
            assert out_keys[0] in env.reset().keys(True, True)
            assert env.rollout(3)[out_keys[0]].ndimension() == 2
        else:
            assert env.rollout(3)["pixels"].ndimension() == 2

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        t = FlattenObservation(-3, -1, out_keys=out_keys)
        td = TensorDict({"pixels": torch.randint(255, (10, 10, 3))}, [])
        module = nn.Sequential(t, nn.Identity())
        if out_keys:
            assert module(td)[out_keys[0]].ndimension() == 1
        else:
            assert module(td)["pixels"].ndimension() == 1

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, out_keys, rbclass):
        t = FlattenObservation(-3, -1, out_keys=out_keys)
        td = TensorDict({"pixels": torch.randint(255, (10, 10, 3))}, []).expand(10)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        rb.extend(td)
        td = rb.sample(2)
        if out_keys:
            assert td[out_keys[0]].ndimension() == 2
        else:
            assert td["pixels"].ndimension() == 2

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for FlattenObservation (yet).")


class TestFrameSkipTransform(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
            return env

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        t = FrameSkipTransform(2)
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict, tensordict.get("next"))

    def test_transform_compose(self):
        t = Compose(FrameSkipTransform(2))
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict, tensordict.get("next"))

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    @pytest.mark.parametrize("skip", [-1, 1, 2, 3])
    def test_transform_env(self, skip):
        """Tests that the built-in frame_skip and the transform lead to the same results."""
        torch.manual_seed(0)
        if skip < 0:
            with pytest.raises(
                ValueError,
                match="frame_skip should have a value greater or equal to one",
            ):
                FrameSkipTransform(skip)
            return
        else:
            fs = FrameSkipTransform(skip)
        base_env = GymEnv(PENDULUM_VERSIONED(), frame_skip=skip)
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), fs)
        base_env.set_seed(0)
        env.base_env.set_seed(0)
        td1 = base_env.reset()
        td2 = env.reset()
        for key in td1.keys():
            torch.testing.assert_close(td1[key], td2[key])
        for i in range(10):
            td1 = base_env.step(tensordicts[i].clone()).flatten_keys()
            td2 = env.step(tensordicts[i].clone()).flatten_keys()
            for key in td1.keys():
                torch.testing.assert_close(td1[key], td2[key])

    def test_nested(self, skip=4):
        env = NestedCountingEnv(max_steps=20)
        policy = CountingEnvCountPolicy(
            action_spec=env.action_spec, action_key=env.action_key
        )
        trnasformed_env = TransformedEnv(env, FrameSkipTransform(frame_skip=skip))
        td = trnasformed_env.rollout(2, policy=policy)
        (td[0] == 0).all()
        (td[1] == skip).all()

    def test_transform_model(self):
        t = FrameSkipTransform(2)
        t = nn.Sequential(t, nn.Identity())
        tensordict = TensorDict({}, [])
        with pytest.raises(
            RuntimeError,
            match="FrameSkipTransform can only be used when appended to a transformed env",
        ):
            t(tensordict)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = FrameSkipTransform(2)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        tensordict = TensorDict({"a": torch.zeros(10)}, [10])
        rb.extend(tensordict)
        with pytest.raises(
            RuntimeError,
            match="FrameSkipTransform can only be used when appended to a transformed env",
        ):
            rb.sample(10)

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    @pytest.mark.parametrize("skip", [-1, 1, 2, 3])
    def test_frame_skip_transform_unroll(self, skip):
        torch.manual_seed(0)
        if skip < 0:
            with pytest.raises(
                ValueError,
                match="frame_skip should have a value greater or equal to one",
            ):
                FrameSkipTransform(skip)
            return
        else:
            fs = FrameSkipTransform(skip)
        base_env = GymEnv(PENDULUM_VERSIONED())
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), fs)
        base_env.set_seed(0)
        env.base_env.set_seed(0)
        td1 = base_env.reset()
        td2 = env.reset()
        for key in td1.keys():
            torch.testing.assert_close(td1[key], td2[key])
        for i in range(10):
            r = 0.0
            for _ in range(skip):
                td1 = base_env.step(tensordicts[i].clone()).flatten_keys(".")
                r = td1.get("next.reward") + r
            td1.set("next.reward", r)
            td2 = env.step(tensordicts[i].clone()).flatten_keys(".")
            for key in td1.keys():
                torch.testing.assert_close(td1[key], td2[key])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for FrameSkipTransform")


class TestGrayScale(TransformBase):
    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = GrayScale(in_keys=keys)
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 16, 16, device=device) for key in keys},
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        gs(td)
        for key in keys:
            assert td.get(key).shape[-3] == 1
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = gs.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, 16, 16])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys",
        [
            [("next", "observation"), ("some_other", "nested_key")],
            [("next", "observation_pixels")],
        ],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = Compose(GrayScale(in_keys=keys))
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 16, 16, device=device) for key in keys},
            [1],
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        gs(td)
        for key in keys:
            assert td.get(key).shape[-3] == 1
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = gs.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, 16, 16])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        out_keys = None

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        out_keys = None

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        out_keys = None
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        out_keys = None
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), GrayScale(out_keys=out_keys)),
        )
        r = env.rollout(3)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        td = TensorDict({"pixels": torch.rand(3, 12, 12)}, []).expand(3)
        model = nn.Sequential(GrayScale(out_keys=out_keys), nn.Identity())
        r = model(td)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys, rbclass):
        td = TensorDict({"pixels": torch.rand(3, 12, 12)}, []).expand(3)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(GrayScale(out_keys=out_keys))
        rb.extend(td)
        r = rb.sample(3)
        if out_keys:
            assert "pixels" in r.keys()
            assert "stuff" in r.keys()
            assert r["pixels"].shape[-3] == 3
            assert r["stuff"].shape[-3] == 1
        else:
            assert "pixels" in r.keys()
            assert "stuff" not in r.keys()
            assert r["pixels"].shape[-3] == 1

    def test_transform_inverse(self):
        raise pytest.skip("No inversee for grayscale")


class TestNoop(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), NoopResetEnv())
        with pytest.raises(
            ValueError,
            match="The parent environment batch-size is non-null",
        ):
            check_env_specs(env)

    def test_trans_parallel_env_check(self):
        raise pytest.skip("Skipped as error tested by test_trans_serial_env_check.")

    def test_transform_no_env(self):
        t = NoopResetEnv()
        with pytest.raises(
            RuntimeError,
            match="NoopResetEnv.parent not found. Make sure that the parent is set.",
        ):
            td = TensorDict({"next": {}}, [])
            t._reset(td, td.empty())
        td = TensorDict({"next": {}}, [])
        t._step(td, td.get("next"))

    def test_transform_compose(self):
        t = Compose(NoopResetEnv())
        with pytest.raises(
            RuntimeError,
            match="NoopResetEnv.parent not found. Make sure that the parent is set.",
        ):
            td = TensorDict({"next": {}}, [])
            td = t._reset(td, td.empty())
        td = TensorDict({"next": {}}, [])
        t._step(td, td.get("next"))

    def test_transform_model(self):
        t = nn.Sequential(NoopResetEnv(), nn.Identity())
        td = TensorDict({}, [])
        t(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = NoopResetEnv()
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({}, [10])
        rb.extend(td)
        rb.sample(1)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for NoopResetEnv")

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, random, device, compose):
        torch.manual_seed(0)
        env = ContinuousActionVecMockEnv()
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        if compose:
            transformed_env = TransformedEnv(env)
            transformed_env.append_transform(noop_reset_env)
        else:
            transformed_env = TransformedEnv(env, noop_reset_env)
        transformed_env = transformed_env.to(device)
        transformed_env.reset()
        if random:
            assert transformed_env.step_count > 0
        else:
            assert transformed_env.step_count == 30

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    def test_nested(self, random, compose):
        torch.manual_seed(0)
        env = NestedCountingEnv(nest_done=False, max_steps=50, nested_dim=6)
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        if compose:
            transformed_env = TransformedEnv(env)
            transformed_env.append_transform(noop_reset_env)
        else:
            transformed_env = TransformedEnv(env, noop_reset_env)
        transformed_env.reset()
        if random:
            assert transformed_env.count > 0
        else:
            assert transformed_env.count == 30

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_noop_reset_env_error(self, random, device, compose):
        torch.manual_seed(0)
        env = SerialEnv(2, lambda: ContinuousActionVecMockEnv())
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        transformed_env = TransformedEnv(env)
        transformed_env.append_transform(noop_reset_env)
        with pytest.raises(
            ValueError,
            match="The parent environment batch-size is non-null",
        ):
            transformed_env.reset()

    @pytest.mark.parametrize("noops", [0, 2, 8])
    @pytest.mark.parametrize("max_steps", [0, 5, 9])
    def test_noop_reset_limit_exceeded(self, noops, max_steps):
        env = IncrementingEnv(max_steps=max_steps)
        check_env_specs(env)
        noop_reset_env = NoopResetEnv(noops=noops, random=False)
        transformed_env = TransformedEnv(env, noop_reset_env)
        if noops <= max_steps:  # Normal behavior.
            result = transformed_env.reset()
            assert result["observation"] == noops
        elif noops > max_steps:  # Raise error as reset limit exceeded.
            with pytest.raises(RuntimeError):
                transformed_env.reset()


class TestObservationNorm(TransformBase):
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_single_trans_env_check(
        self,
        out_keys,
    ):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=torch.zeros(7),
                scale=1.0,
                in_keys=["observation"],
                out_keys=out_keys,
            ),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(
        self,
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                ObservationNorm(
                    loc=torch.zeros(7),
                    in_keys=["observation"],
                    scale=1.0,
                ),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(
        self,
    ):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                ObservationNorm(
                    loc=torch.zeros(7),
                    in_keys=["observation"],
                    scale=1.0,
                ),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(
        self,
    ):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            ObservationNorm(
                loc=torch.zeros(7),
                in_keys=["observation"],
                scale=1.0,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(
        self,
    ):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            ObservationNorm(
                loc=torch.zeros(7),
                in_keys=["observation"],
                scale=1.0,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("in_key", ["observation", ("some_other", "observation")])
    @pytest.mark.parametrize(
        "out_keys", [None, ["stuff"], [("some_other", "nested_key")]]
    )
    def test_transform_no_env(self, out_keys, standard_normal, in_key):
        t = ObservationNorm(in_keys=[in_key], out_keys=out_keys)
        # test that init fails
        with pytest.raises(
            RuntimeError,
            match="Cannot initialize the transform if parent env is not defined",
        ):
            t.init_stats(num_iter=5)
        t = ObservationNorm(
            loc=torch.ones(7),
            scale=0.5,
            in_keys=[in_key],
            out_keys=out_keys,
            standard_normal=standard_normal,
        )
        obs = torch.randn(7)
        td = TensorDict({in_key: obs}, [])
        t(td)
        if out_keys:
            assert out_keys[0] in td.keys(True, True)
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td[in_key]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_compose(self, out_keys, standard_normal):
        t = Compose(ObservationNorm(in_keys=["observation"], out_keys=out_keys))
        # test that init fails
        with pytest.raises(
            RuntimeError,
            match="Cannot initialize the transform if parent env is not defined",
        ):
            t[0].init_stats(num_iter=5)
        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, [])
        t(td)
        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys, standard_normal):
        if standard_normal:
            scale = 1_000_000
        else:
            scale = 0.0
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=0.0,
                scale=scale,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            ),
        )
        if out_keys:
            assert out_keys[0] in env.reset().keys()
            obs = env.rollout(3)[out_keys[0]]
        else:
            obs = env.rollout(3)["observation"]

        assert (abs(obs) < 1e-2).all()

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_env_clone(self, standard_normal):
        out_keys = ["stuff"]
        if standard_normal:
            scale = 1_000_000
        else:
            scale = 0.0
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            ObservationNorm(
                loc=0.0,
                scale=scale,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            ),
        )
        cloned = env.transform.clone()
        env.transform.loc += 1
        env.transform.scale += 1
        torch.testing.assert_close(
            env.transform.loc, torch.ones_like(env.transform.loc)
        )
        torch.testing.assert_close(
            env.transform.scale, scale + torch.ones_like(env.transform.scale)
        )
        assert env.transform.loc == cloned.loc
        assert env.transform.scale == cloned.scale

    def test_transform_model(self):
        standard_normal = True
        out_keys = ["stuff"]

        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        model = nn.Sequential(t, nn.Identity())
        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, [])
        model(td)

        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        standard_normal = True
        out_keys = ["stuff"]

        t = Compose(
            ObservationNorm(
                loc=torch.ones(7),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                standard_normal=standard_normal,
            )
        )
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)

        obs = torch.randn(7)
        td = TensorDict({"observation": obs}, []).expand(3)
        rb.extend(td)
        td = rb.sample(5)

        if out_keys:
            assert out_keys[0] in td.keys()
            obs_tr = td[out_keys[0]]
        else:
            obs_tr = td["observation"]
        if standard_normal:
            assert torch.allclose((obs - 1) / 0.5, obs_tr)
        else:
            assert torch.allclose(0.5 * obs + 1, obs_tr)

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_key_inv", ["action_inv", ("nested", "action_inv")])
    @pytest.mark.parametrize(
        "out_key", ["observation_out", ("nested", "observation_out")]
    )
    def test_transform_inverse(self, out_key, out_key_inv):
        standard_normal = True
        out_keys = [out_key]
        in_keys_inv = ["action"]
        out_keys_inv = [out_key_inv]
        t = Compose(
            ObservationNorm(
                loc=torch.ones(()),
                scale=0.5,
                in_keys=["observation"],
                out_keys=out_keys,
                in_keys_inv=in_keys_inv,
                out_keys_inv=out_keys_inv,
                standard_normal=standard_normal,
            )
        )
        base_env = GymEnv(PENDULUM_VERSIONED())
        env = TransformedEnv(base_env, t)
        td = env.rollout(3)
        check_env_specs(env)
        env.set_seed(0)
        assert torch.allclose(td["action"] * 0.5 + 1, t.inv(td)[out_key_inv])
        assert torch.allclose((td["observation"] - 1) / 0.5, td[out_key])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize(
        ["loc", "scale"],
        [
            (0, 1),
            (1, 2),
            (torch.ones(16, 16), torch.ones(1)),
            (torch.ones(1), torch.ones(16, 16)),
        ],
    )
    def test_observationnorm(
        self, batch, keys, device, nchannels, loc, scale, standard_normal
    ):
        torch.manual_seed(0)
        nchannels = 3
        if isinstance(loc, Tensor):
            loc = loc.to(device)
        if isinstance(scale, Tensor):
            scale = scale.to(device)
        on = ObservationNorm(loc, scale, in_keys=keys, standard_normal=standard_normal)
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.zeros(1, nchannels, 16, 16, device=device) for key in keys}, [1]
        )
        td.set("dont touch", dont_touch.clone())
        on(td)
        for key in keys:
            if standard_normal:
                assert (td.get(key) == -loc / scale).all()
            else:
                assert (td.get(key) == loc).all()
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(
                0, 1, (nchannels, 16, 16), device=device
            )
            observation_spec = on.transform_observation_spec(observation_spec)
            if standard_normal:
                assert (observation_spec.space.minimum == -loc / scale).all()
                assert (observation_spec.space.maximum == (1 - loc) / scale).all()
            else:
                assert (observation_spec.space.minimum == loc).all()
                assert (observation_spec.space.maximum == scale + loc).all()

        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(0, 1, (nchannels, 16, 16), device=device)
                    for key in keys
                }
            )
            observation_spec = on.transform_observation_spec(observation_spec)
            for key in keys:
                if standard_normal:
                    assert (observation_spec[key].space.low == -loc / scale).all()
                    assert (observation_spec[key].space.high == (1 - loc) / scale).all()
                else:
                    assert (observation_spec[key].space.low == loc).all()
                    assert (observation_spec[key].space.high == scale + loc).all()

    @pytest.mark.parametrize("keys", [["observation"], ["observation", "next_pixel"]])
    @pytest.mark.parametrize("size", [1, 3])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_observationnorm_init_stats(
        self, keys, size, device, standard_normal, parallel
    ):
        def make_env():
            base_env = ContinuousActionVecMockEnv(
                observation_spec=CompositeSpec(
                    observation=BoundedTensorSpec(
                        low=1, high=1, shape=torch.Size([size])
                    ),
                    observation_orig=BoundedTensorSpec(
                        low=1, high=1, shape=torch.Size([size])
                    ),
                ),
                action_spec=BoundedTensorSpec(low=1, high=1, shape=torch.Size((size,))),
                seed=0,
            )
            base_env.out_key = "observation"
            return base_env

        if parallel:
            base_env = SerialEnv(2, make_env)
            reduce_dim = (0, 1)
            cat_dim = 1
        else:
            base_env = make_env()
            reduce_dim = 0
            cat_dim = 0

        t_env = TransformedEnv(
            base_env,
            transform=ObservationNorm(in_keys=keys, standard_normal=standard_normal),
        )
        if len(keys) > 1:
            t_env.transform.init_stats(
                num_iter=11, key="observation", cat_dim=cat_dim, reduce_dim=reduce_dim
            )
        else:
            t_env.transform.init_stats(
                num_iter=11, reduce_dim=reduce_dim, cat_dim=cat_dim
            )
        batch_dims = len(t_env.batch_size)
        assert (
            t_env.transform.loc.shape
            == t_env.observation_spec["observation"].shape[batch_dims:]
        )
        assert (
            t_env.transform.scale.shape
            == t_env.observation_spec["observation"].shape[batch_dims:]
        )
        assert t_env.transform.loc.dtype == t_env.observation_spec["observation"].dtype
        assert (
            t_env.transform.loc.device == t_env.observation_spec["observation"].device
        )

    @pytest.mark.parametrize("keys", [["pixels"], ["pixels", "stuff"]])
    @pytest.mark.parametrize("size", [1, 3])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_observationnorm_init_stats_pixels(
        self, keys, size, device, standard_normal, parallel
    ):
        def make_env():
            base_env = DiscreteActionConvMockEnvNumpy(
                seed=0,
            )
            base_env.out_key = "pixels"
            return base_env

        if parallel:
            base_env = SerialEnv(2, make_env)
            reduce_dim = (0, 1, 3, 4)
            keep_dim = (3, 4)
            cat_dim = 1
        else:
            base_env = make_env()
            reduce_dim = (0, 2, 3)
            keep_dim = (2, 3)
            cat_dim = 0

        t_env = TransformedEnv(
            base_env,
            transform=ObservationNorm(in_keys=keys, standard_normal=standard_normal),
        )
        if len(keys) > 1:
            t_env.transform.init_stats(
                num_iter=11,
                key="pixels",
                cat_dim=cat_dim,
                reduce_dim=reduce_dim,
                keep_dims=keep_dim,
            )
        else:
            t_env.transform.init_stats(
                num_iter=11,
                reduce_dim=reduce_dim,
                cat_dim=cat_dim,
                keep_dims=keep_dim,
            )

        assert t_env.transform.loc.shape == torch.Size(
            [t_env.observation_spec["pixels"].shape[-3], 1, 1]
        )
        assert t_env.transform.scale.shape == torch.Size(
            [t_env.observation_spec["pixels"].shape[-3], 1, 1]
        )

    def test_observationnorm_stats_already_initialized_error(self):
        transform = ObservationNorm(in_keys=["next_observation"], loc=0, scale=1)

        with pytest.raises(RuntimeError, match="Loc/Scale are already initialized"):
            transform.init_stats(num_iter=11)

    def test_observationnorm_wrong_catdim(self):
        transform = ObservationNorm(in_keys=["next_observation"], loc=0, scale=1)

        with pytest.raises(
            ValueError, match="cat_dim must be part of or equal to reduce_dim"
        ):
            transform.init_stats(num_iter=11, cat_dim=1)

        with pytest.raises(
            ValueError, match="cat_dim must be part of or equal to reduce_dim"
        ):
            transform.init_stats(num_iter=11, cat_dim=2, reduce_dim=(0, 1))

        with pytest.raises(
            ValueError,
            match="cat_dim must be specified if reduce_dim is not an integer",
        ):
            transform.init_stats(num_iter=11, reduce_dim=(0, 1))

    def test_observationnorm_init_stats_multiple_keys_error(self):
        transform = ObservationNorm(in_keys=["next_observation", "next_pixels"])

        err_msg = "Transform has multiple in_keys but no specific key was passed as an argument"
        with pytest.raises(RuntimeError, match=err_msg):
            transform.init_stats(num_iter=11)

    def test_observationnorm_initialization_order_error(self):
        base_env = ContinuousActionVecMockEnv()
        t_env = TransformedEnv(base_env)

        transform1 = ObservationNorm(in_keys=["next_observation"])
        transform2 = ObservationNorm(in_keys=["next_observation"])
        t_env.append_transform(transform1)
        t_env.append_transform(transform2)

        err_msg = (
            "ObservationNorms need to be initialized in the right order."
            "Trying to initialize an ObservationNorm while a parent ObservationNorm transform is still uninitialized"
        )
        with pytest.raises(RuntimeError, match=err_msg):
            transform2.init_stats(num_iter=10, key="observation")

    def test_observationnorm_uninitialized_stats_error(self):
        transform = ObservationNorm(in_keys=["next_observation", "next_pixels"])

        err_msg = (
            "Loc/Scale have not been initialized. Either pass in values in the constructor "
            "or call the init_stats method"
        )
        with pytest.raises(RuntimeError, match=err_msg):
            transform._apply_transform(torch.Tensor([1]))


@pytest.mark.skipif(not _has_tv, reason="no torchvision")
class TestResize(TransformBase):
    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, interpolation, keys, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        resize = Resize(w=20, h=21, interpolation=interpolation, in_keys=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        resize(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, 21])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = resize.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, interpolation, keys, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        resize = Compose(Resize(w=20, h=21, interpolation=interpolation, in_keys=keys))
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        resize(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, 21])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = resize.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    def test_single_trans_env_check(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        check_env_specs(env)
        assert "pixels" in env.observation_spec.keys()

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    @pytest.mark.parametrize("out_key", ["pixels", ("agents", "pixels")])
    def test_transform_env(self, out_key):
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED()),
            Compose(
                ToTensorImage(), Resize(20, 21, in_keys=["pixels"], out_keys=[out_key])
            ),
        )
        check_env_specs(env)
        td = env.rollout(3)
        assert td[out_key].shape[-3:] == torch.Size([3, 20, 21])

    def test_transform_model(self):
        module = nn.Sequential(Resize(20, 21, in_keys=["pixels"]), nn.Identity())
        td = TensorDict({"pixels": torch.randn(3, 32, 32)}, [])
        module(td)
        assert td["pixels"].shape == torch.Size([3, 20, 21])

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = Resize(20, 21, in_keys=["pixels"])
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"pixels": torch.randn(3, 32, 32)}, []).expand(10)
        rb.extend(td)
        td = rb.sample(2)
        assert td["pixels"].shape[-3:] == torch.Size([3, 20, 21])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for Resize")


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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), RewardClipping(-0.1, 0.1)
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), RewardClipping(-0.1, 0.1)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), RewardClipping(-0.1, 0.1)
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

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
            reward_spec = UnboundedContinuousTensorSpec(device=device)
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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), RewardScaling(0.5, 1.5))

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), RewardScaling(0.5, 1.5)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), RewardScaling(0.5, 1.5)
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
            r = env.rollout(4)
            assert r["next", "episode_reward"].unique().numel() > 1
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
        )
        check_env_specs(env)
        r = env.rollout(4)
        assert r["next", "episode_reward"].unique().numel() > 1

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            Compose(RewardScaling(loc=-1, scale=1), RewardSum()),
        )
        try:
            check_env_specs(env)
            r = env.rollout(4)
            assert r["next", "episode_reward"].unique().numel() > 1
        finally:
            env.close()

    @pytest.mark.parametrize("has_in_keys,", [True, False])
    @pytest.mark.parametrize("reset_keys,", [None, ["_reset"] * 3])
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
        with pytest.raises(
            ValueError, match="Could not match the env reset_keys"
        ) if reset_keys is None else contextlib.nullcontext():
            check_env_specs(env)
        if reset_keys is not None:
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
    def test_transform_env(self, out_key):
        t = Compose(RewardSum(in_keys=["reward"], out_keys=[out_key]))
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), t)
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
        from _utils_internal import CARTPOLE_VERSIONED

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
            reward_spec=UnboundedContinuousTensorSpec(shape=(3, 16, 16)),
        )
        transfomed_env = TransformedEnv(base_env, RewardSum())
        transformed_observation_spec1 = transfomed_env.observation_spec
        assert isinstance(transformed_observation_spec1, CompositeSpec)
        assert "episode_reward" in transformed_observation_spec1.keys()
        assert "observation" in transformed_observation_spec1.keys()

        base_env = ContinuousActionVecMockEnv(
            reward_spec=UnboundedContinuousTensorSpec(),
            observation_spec=CompositeSpec(
                observation=UnboundedContinuousTensorSpec(),
                some_extra_observation=UnboundedContinuousTensorSpec(),
            ),
        )
        transfomed_env = TransformedEnv(base_env, RewardSum())
        transformed_observation_spec2 = transfomed_env.observation_spec
        assert isinstance(transformed_observation_spec2, CompositeSpec)
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
                    t.reset(td)
            else:
                t.reset(td)


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


class TestUnsqueezeTransform(TransformBase):
    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", ("some_other", "nested_key")], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(
        self, keys, size, nchannels, batch, device, unsqueeze_dim
    ):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = UnsqueezeTransform(
            unsqueeze_dim, in_keys=keys, allow_positive_dim=True
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
        if unsqueeze_dim >= 0 and unsqueeze_dim < len(batch):
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                unsqueeze(td)
            return
        unsqueeze(td)
        expected_size = [*batch, *size, nchannels, 16, 16]
        if unsqueeze_dim < 0:
            expected_size.insert(len(expected_size) + unsqueeze_dim + 1, 1)
        else:
            expected_size.insert(unsqueeze_dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys:
            assert td.get(key).shape == expected_size, (
                batch,
                size,
                nchannels,
                unsqueeze_dim,
            )
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(
                -1, 1, (*batch, *size, nchannels, 16, 16)
            )
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
            assert observation_spec.shape == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(-1, 1, (*batch, *size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == expected_size

    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
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
    def test_unsqueeze_inv(
        self, keys, keys_inv, size, nchannels, batch, device, unsqueeze_dim
    ):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        unsqueeze = UnsqueezeTransform(
            unsqueeze_dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
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

        if expected_size[unsqueeze_dim] == 1:
            del expected_size[unsqueeze_dim]
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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                UnsqueezeTransform(-1, in_keys=["observation"]),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            UnsqueezeTransform(-1, in_keys=["observation"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            UnsqueezeTransform(-1, in_keys=["observation"]),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(
        self, keys, size, nchannels, batch, device, unsqueeze_dim
    ):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = Compose(
            UnsqueezeTransform(unsqueeze_dim, in_keys=keys, allow_positive_dim=True)
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
        if unsqueeze_dim >= 0 and unsqueeze_dim < len(batch):
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                unsqueeze(td)
            return
        unsqueeze(td)
        expected_size = [*batch, *size, nchannels, 16, 16]
        if unsqueeze_dim < 0:
            expected_size.insert(len(expected_size) + unsqueeze_dim + 1, 1)
        else:
            expected_size.insert(unsqueeze_dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys:
            assert td.get(key).shape == expected_size, (
                batch,
                size,
                nchannels,
                unsqueeze_dim,
            )
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(
                -1, 1, (*batch, *size, nchannels, 16, 16)
            )
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
            assert observation_spec.shape == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(-1, 1, (*batch, *size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
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
    @pytest.mark.parametrize("unsqueeze_dim", [-1, 1])
    def test_transform_model(self, out_keys, unsqueeze_dim):
        t = UnsqueezeTransform(
            unsqueeze_dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        td = TensorDict(
            {"observation": TensorDict({"stuff": torch.randn(3, 4)}, [3, 4])}, []
        )
        t(td)
        expected_shape = [3, 4]
        if unsqueeze_dim >= 0:
            expected_shape.insert(unsqueeze_dim, 1)
        else:
            expected_shape.insert(len(expected_shape) + unsqueeze_dim + 1, 1)
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    @pytest.mark.parametrize("unsqueeze_dim", [-1, 1])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass, out_keys, unsqueeze_dim):
        t = UnsqueezeTransform(
            unsqueeze_dim,
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
        if unsqueeze_dim >= 0:
            expected_shape.insert(unsqueeze_dim, 1)
        else:
            expected_shape.insert(len(expected_shape) + unsqueeze_dim + 1, 1)
        if out_keys is None:
            assert td["observation"].shape == torch.Size(expected_shape)
        else:
            assert td[out_keys[0]].shape == torch.Size(expected_shape)

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    def test_transform_inverse(self):
        env = TransformedEnv(
            GymEnv(HALFCHEETAH_VERSIONED()),
            # the order is inverted
            Compose(
                UnsqueezeTransform(
                    -1, in_keys_inv=["action_t"], out_keys_inv=["action"]
                ),
                SqueezeTransform(-1, in_keys_inv=["action"], out_keys_inv=["action_t"]),
            ),
        )
        td = env.rollout(3)
        assert env.action_spec.shape[-1] == 6
        assert td["action"].shape[-1] == 6


class TestSqueezeTransform(TransformBase):
    @pytest.mark.parametrize("squeeze_dim", [1, -2])
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
        self, keys, keys_inv, size, nchannels, batch, device, squeeze_dim
    ):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(
            squeeze_dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
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

        if expected_size[squeeze_dim] == 1:
            del expected_size[squeeze_dim]
        for key in keys:
            assert td.get(key).shape == torch.Size(expected_size)

    @pytest.mark.parametrize("squeeze_dim", [1, -2])
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
    def test_squeeze_inv(
        self, keys, keys_inv, size, nchannels, batch, device, squeeze_dim
    ):
        torch.manual_seed(0)
        if squeeze_dim >= 0:
            squeeze_dim = squeeze_dim + len(batch)
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(
            squeeze_dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
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

        if squeeze_dim < 0:
            expected_size.insert(len(expected_size) + squeeze_dim + 1, 1)
        else:
            expected_size.insert(squeeze_dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys_inv:
            assert td.get(key).shape == torch.Size(expected_size), squeeze_dim

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
            UnsqueezeTransform(-1, in_keys_inv=["action_un"], out_keys_inv=["action"]),
            SqueezeTransform(-1, in_keys_inv=["action"], out_keys_inv=["action_un"]),
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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), self._circular_transform
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), self._circular_transform
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), self._circular_transform
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("squeeze_dim", [1, -2])
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
        self, keys, keys_inv, size, nchannels, batch, device, squeeze_dim
    ):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        squeeze = Compose(
            SqueezeTransform(
                squeeze_dim, in_keys=keys, in_keys_inv=keys_inv, allow_positive_dim=True
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

        if expected_size[squeeze_dim] == 1:
            del expected_size[squeeze_dim]
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
        squeeze_dim = 1
        t = SqueezeTransform(
            squeeze_dim,
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
        squeeze_dim = -2
        t = SqueezeTransform(
            squeeze_dim,
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

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    def test_transform_inverse(self):
        env = TransformedEnv(
            GymEnv(HALFCHEETAH_VERSIONED()), self._inv_circular_transform
        )
        check_env_specs(env)
        r = env.rollout(3)
        r2 = GymEnv(HALFCHEETAH_VERSIONED()).rollout(3)
        assert (r.zero_() == r2.zero_()).all()


class TestTargetReturn(TransformBase):
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, batch, mode, device):
        torch.manual_seed(0)
        t = TargetReturn(target_return=10.0, mode=mode)
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        td = env.rollout(2)
        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()
        else:
            assert (td["next", "target_return"] == 10.0).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, batch, mode, device):
        torch.manual_seed(0)
        t = Compose(
            TargetReturn(
                in_keys=["reward"],
                out_keys=["target_return"],
                target_return=10.0,
                mode=mode,
                reset_key="_reset",
            )
        )
        next_reward = torch.rand((*batch, 1))
        td = TensorDict(
            {
                "next": {
                    "reward": next_reward,
                },
            },
            device=device,
            batch_size=batch,
        )
        td_reset = t._reset(td, td.empty())
        next_td = td.get("next")
        next_td = t._step(td_reset, next_td)
        td.set("next", next_td)

        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()

        else:
            assert (td["next", "target_return"] == 10.0).all()

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_single_trans_env_check(self, mode, device):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_serial_trans_env_check(self, mode, device):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TargetReturn(target_return=10.0, mode=mode).to(device),
                device=device,
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_parallel_trans_env_check(self, mode, device):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TargetReturn(target_return=10.0, mode=mode).to(device),
                device=device,
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_trans_serial_env_check(self, mode, device):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_trans_parallel_env_check(self, mode, device):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode),
            device=device,
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [SerialEnv, ParallelEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_targetreturn_batching(self, batched_class, break_when_any_done):
        from _utils_internal import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TargetReturn(target_return=10.0, mode="reduce"),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TargetReturn(target_return=10.0, mode="reduce"),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TargetReturn")

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("in_key", ["reward", ("agents", "reward")])
    @pytest.mark.parametrize("out_key", ["target_return", ("agents", "target_return")])
    def test_transform_no_env(self, mode, in_key, out_key):
        t = TargetReturn(
            target_return=10.0,
            mode=mode,
            in_keys=[in_key],
            out_keys=[out_key],
            reset_key="_reset",
        )
        reward = torch.randn(10, 1)
        td = TensorDict({("next", in_key): reward}, [10])
        td_reset = t._reset(td, td.empty())
        td_next = t._step(td_reset, td.get("next"))
        td.set("next", td_next)
        if mode == "reduce":
            assert (td["next", out_key] + td["next", in_key] == 10.0).all()
        else:
            assert (td["next", out_key] == 10.0).all()

    def test_transform_model(
        self,
    ):
        t = TargetReturn(target_return=10.0)
        model = nn.Sequential(t, nn.Identity())
        reward = torch.randn(10, 1)
        td = TensorDict({("next", "reward"): reward}, [10])
        with pytest.raises(
            NotImplementedError, match="cannot be executed without a parent"
        ):
            model(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
    ):
        t = TargetReturn(target_return=10.0)
        rb = rbclass(storage=LazyTensorStorage(10))
        reward = torch.randn(10, 1)
        td = TensorDict({("next", "reward"): reward}, [10])
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="cannot be executed without a parent"
        ):
            _ = rb.sample(2)


class TestToTensorImage(TransformBase):
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = ToTensorImage(in_keys=keys)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 16, 16, 3), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        totensorimage(td)
        for key in keys:
            assert td.get(key).shape[-3:] == torch.Size([3, 16, 16])
            assert td.get(key).device == device
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(0, 255, (16, 16, 3), dtype=torch.uint8)
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(0, 255, (16, 16, 3), dtype=torch.uint8)
                    for key in keys
                }
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.low == 0).all()
                assert (observation_spec[key].space.high == 1).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = Compose(ToTensorImage(in_keys=keys))
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 16, 16, 3), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        totensorimage(td)
        for key in keys:
            assert td.get(key).shape[-3:] == torch.Size([3, 16, 16])
            assert td.get(key).device == device
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(0, 255, (16, 16, 3), dtype=torch.uint8)
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(0, 255, (16, 16, 3), dtype=torch.uint8)
                    for key in keys
                }
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.low == 0).all()
                assert (observation_spec[key].space.high == 1).all()

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            ToTensorImage(in_keys=["pixels"], out_keys=out_keys),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                ToTensorImage(in_keys=["pixels"], out_keys=None),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(),
                ToTensorImage(in_keys=["pixels"], out_keys=None),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            ToTensorImage(in_keys=["pixels"], out_keys=None),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            ToTensorImage(in_keys=["pixels"], out_keys=None),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("out_keys", [None, ["stuff"], [("nested", "stuff")]])
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_transform_env(self, out_keys, default_dtype):
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(),
            ToTensorImage(in_keys=["pixels"], out_keys=out_keys),
        )
        r = env.rollout(3)
        if out_keys is not None:
            assert out_keys[0] in r.keys(True, True)
            obs = r[out_keys[0]]
        else:
            obs = r["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is default_dtype
        torch.set_default_dtype(prev_dtype)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_model(self, out_keys):
        t = ToTensorImage(in_keys=["pixels"], out_keys=out_keys)
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({"pixels": torch.randint(255, (21, 22, 3))}, [])
        model(td)
        if out_keys is not None:
            assert out_keys[0] in td.keys()
            obs = td[out_keys[0]]
        else:
            obs = td["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is torch.float32

    @pytest.mark.parametrize("from_int", [None, True, False])
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.uint8])
    def test_transform_scale(self, from_int, default_dtype):
        totensorimage = ToTensorImage(in_keys=["pixels"], from_int=from_int)
        fill_value = 150 if default_dtype == torch.uint8 else 0.5
        td = TensorDict(
            {"pixels": torch.full((21, 22, 3), fill_value, dtype=default_dtype)}, []
        )
        # Save whether or not the tensor is floating point before the transform changes it
        # to floating point type.
        is_floating_point = torch.is_floating_point(td["pixels"])
        totensorimage(td)

        if from_int is None:
            expected_pixel_value = (
                fill_value / 255 if not is_floating_point else fill_value
            )
        else:
            expected_pixel_value = fill_value / 255 if from_int else fill_value
        assert (td["pixels"] == expected_pixel_value).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys, rbclass):
        t = ToTensorImage(in_keys=["pixels"], out_keys=out_keys)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"pixels": torch.randint(255, (21, 22, 3))}, [])
        rb.extend(td.expand(10))
        td = rb.sample(2)
        if out_keys is not None:
            assert out_keys[0] in td.keys()
            obs = td[out_keys[0]]
        else:
            obs = td["pixels"]
        assert obs.shape[-3] == 3
        assert obs.dtype is torch.float32

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for ToTensorImage")


class TestTensorDictPrimer(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3])),
        )
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        assert ("next", "mykey") in env.rollout(3).keys(True)

    def test_transform_no_env(self):
        t = TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3]))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "mykey" in td.keys()

    def test_transform_model(self):
        t = TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3]))
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({}, [])
        model(td)
        assert "mykey" in td.keys()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        batch_size = (2,)
        t = TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([*batch_size, 3]))
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"a": torch.zeros(())}, [])
        rb.extend(td.expand(10))
        td = rb.sample(*batch_size)
        assert "mykey" in td.keys()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TensorDictPrimer")

    def test_transform_compose(self):
        t = Compose(TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3])))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "mykey" in td.keys()

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3])),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            assert ("next", "mykey") in env.rollout(3).keys(True)
        finally:
            env.close()

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3])),
            )

        env = SerialEnv(2, make_env)
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            assert ("next", "mykey") in env.rollout(3).keys(True)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([2, 4])),
        )
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            r = env.rollout(3)
            assert ("next", "mykey") in r.keys(True)
            assert r["next", "mykey"].shape == torch.Size([2, 3, 4])
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        with pytest.raises(RuntimeError, match="The leading shape of the primer specs"):
            env = TransformedEnv(
                SerialEnv(2, ContinuousActionVecMockEnv),
                TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([4])),
            )
            _ = env.observation_spec

        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([2, 4])),
        )
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        r = env.rollout(3)
        assert ("next", "mykey") in r.keys(True)
        assert r["next", "mykey"].shape == torch.Size([2, 3, 4])

    @pytest.mark.parametrize(
        "default_keys", [["action"], ["action", "monkeys jumping on the bed"]]
    )
    @pytest.mark.parametrize(
        "spec",
        [
            CompositeSpec(b=BoundedTensorSpec(-3, 3, [4])),
            BoundedTensorSpec(-3, 3, [4]),
        ],
    )
    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    @pytest.mark.parametrize("serial", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(
        self,
        default_keys,
        spec,
        random,
        value,
        serial,
        device,
    ):
        if random and value != 0.0:
            return pytest.skip("no need to check random=True with more than one value")
        torch.manual_seed(0)
        num_defaults = len(default_keys)

        def make_env():
            env = ContinuousActionVecMockEnv()
            env.set_seed(100)
            kwargs = {
                key: spec.clone() if key != "action" else env.action_spec.clone()
                # copy to avoid having the same spec for all keys
                for key in default_keys
            }
            reset_transform = TensorDictPrimer(
                random=random, default_value=value, **kwargs
            )
            transformed_env = TransformedEnv(env, reset_transform).to(device)
            return transformed_env

        if serial:
            env = SerialEnv(2, make_env)
        else:
            env = make_env()

        tensordict = env.reset()
        tensordict_select = tensordict.select(
            *[key for key in tensordict.keys() if key in default_keys]
        )
        assert len(list(tensordict_select.keys())) == num_defaults
        if random:
            assert (tensordict_select != value).any()
        else:
            assert (tensordict_select == value).all()

        if isinstance(spec, CompositeSpec) and any(
            key != "action" for key in default_keys
        ):
            for key in default_keys:
                if key in ("action",):
                    continue
                assert key in tensordict.keys()
                assert tensordict[key, "b"] is not None

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_tensordictprimer_batching(self, batched_class, break_when_any_done):
        from _utils_internal import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([2, 4])),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([4])),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)


class TestTimeMaxPool(TransformBase):
    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_no_env(self, T, seq_len, device):
        batch = 1
        nodes = 4
        keys = ["observation", ("nested", "key")]
        time_max_pool = TimeMaxPool(keys, T=T)

        tensor_list = []
        for _ in range(seq_len):
            tensor_list.append(torch.rand(batch, nodes).to(device))
        max_vals, _ = torch.max(torch.stack(tensor_list[-T:]), dim=0)

        for i in range(seq_len):
            env_td = TensorDict(
                {
                    "observation": tensor_list[i],
                    ("nested", "key"): tensor_list[i].clone(),
                },
                device=device,
                batch_size=[batch],
            )
            transformed_td = time_max_pool._call(env_td)

        assert (max_vals == transformed_td["observation"]).all()
        assert (max_vals == transformed_td["nested", "key"]).all()

    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, T, seq_len, device):
        batch = 1
        nodes = 4
        keys = ["observation"]
        time_max_pool = Compose(TimeMaxPool(keys, T=T))

        tensor_list = []
        for _ in range(seq_len):
            tensor_list.append(torch.rand(batch, nodes).to(device))
        max_vals, _ = torch.max(torch.stack(tensor_list[-T:]), dim=0)

        for i in range(seq_len):
            env_td = TensorDict(
                {
                    "observation": tensor_list[i],
                },
                device=device,
                batch_size=[batch],
            )
            transformed_td = time_max_pool._call(env_td)

        assert (max_vals == transformed_td["observation"]).all()

    @pytest.mark.parametrize("out_keys", [None, ["obs2"]])
    def test_single_trans_env_check(self, out_keys):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TimeMaxPool(in_keys=["observation"], T=3, out_keys=out_keys),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                TimeMaxPool(
                    in_keys=["observation"],
                    T=3,
                ),
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                TimeMaxPool(
                    in_keys=["observation"],
                    T=3,
                ),
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()),
            TimeMaxPool(
                in_keys=["observation"],
                T=3,
            ),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, lambda: ContinuousActionVecMockEnv()),
            TimeMaxPool(
                in_keys=["observation"],
                T=3,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_timemax_batching(self, batched_class, break_when_any_done):
        from _utils_internal import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TimeMaxPool(
                in_keys=["observation"],
                out_keys=["observation_max"],
                T=3,
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TimeMaxPool(
                    in_keys=["observation"],
                    out_keys=["observation_max"],
                    T=3,
                ),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    @pytest.mark.parametrize("out_keys", [None, ["obs2"], [("some", "other")]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED(), frame_skip=4),
            TimeMaxPool(
                in_keys=["observation"],
                out_keys=out_keys,
                T=3,
            ),
        )
        td = env.reset()
        if out_keys:
            assert td[out_keys[0]].shape[-1] == 3
        else:
            assert td["observation"].shape[-1] == 3

    def test_transform_model(self):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        dim = -2
        d = 4
        N = 3
        batch_size = (5,)
        extra_d = (3,) * (-dim - 1)
        device = "cpu"
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        t = TimeMaxPool(
            in_keys=["observation"],
            T=3,
        )

        model = nn.Sequential(t, nn.Identity())
        with pytest.raises(
            NotImplementedError, match="TimeMaxPool cannot be called independently"
        ):
            model(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        dim = -2
        d = 4
        N = 3
        batch_size = (5,)
        extra_d = (3,) * (-dim - 1)
        device = "cpu"
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        t = TimeMaxPool(
            in_keys=["observation"],
            T=3,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="TimeMaxPool cannot be called independently"
        ):
            _ = rb.sample(10)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_tmp_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        t = TimeMaxPool(in_keys=key1, T=3, reset_key="_reset")

        t._call(td.clone())
        buffer = getattr(t, f"_maxpool_buffer_{key1}")

        tdc = td.clone()
        passed_back_td = t._reset(tdc, tdc.empty())

        # assert tdc is passed_back_td
        assert (buffer != 0).any()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for TimeMaxPool")


class TestgSDE(TransformBase):
    @pytest.mark.parametrize("action_dim,state_dim", [(None, None), (7, 7)])
    def test_single_trans_env_check(self, action_dim, state_dim):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            gSDENoise(state_dim=state_dim, action_dim=action_dim),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            state_dim = 7
            action_dim = 7
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                gSDENoise(state_dim=state_dim, action_dim=action_dim),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            state_dim = 7
            action_dim = 7
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                gSDENoise(state_dim=state_dim, action_dim=action_dim),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        state_dim = 7
        action_dim = 7
        with pytest.raises(RuntimeError, match="The leading shape of the primer"):
            env = TransformedEnv(
                SerialEnv(2, ContinuousActionVecMockEnv),
                gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=()),
            )
            check_env_specs(env)
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        state_dim = 7
        action_dim = 7
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        state_dim = 7
        action_dim = 5
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size(
            [
                2,
                action_dim,
                state_dim,
            ]
        )

    def test_transform_model(self):
        state_dim = 7
        action_dim = 5
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,))
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict({}, [])
        model(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        state_dim = 7
        action_dim = 5
        batch_size = (2,)
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=batch_size)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"a": torch.zeros(())}, [])
        rb.extend(td.expand(10))
        td = rb.sample(*batch_size)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TensorDictPrimer")

    def test_transform_compose(self):
        state_dim = 7
        action_dim = 5
        t = Compose(gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "_eps_gSDE" in td.keys()
        assert (td["_eps_gSDE"] != 0.0).all()
        assert td["_eps_gSDE"].shape == torch.Size([2, action_dim, state_dim])

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()), gSDENoise(state_dim=3, action_dim=1)
        )
        check_env_specs(env)
        assert (env.reset()["_eps_gSDE"] != 0.0).all()


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="Testing VIP on cuda only")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
@pytest.mark.parametrize("model", ["resnet50"])
class TestVIP(TransformBase):
    def test_transform_inverse(self, model, device):
        raise pytest.skip("no inverse for VIPTransform")

    def test_single_trans_env_check(self, model, device):
        tensor_pixels_key = None
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), vip
        )
        check_env_specs(transformed_env)

    def test_trans_serial_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vip
        )
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vip
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                VIPTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_parallel_trans_env_check(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]

        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                VIPTransform(
                    model,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    tensor_pixels_keys=tensor_pixels_key,
                ),
            )

        transformed_env = ParallelEnv(2, make_env)
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_transform_model(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(vip, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 1024

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, model, device, rbclass):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(vip)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 1024

    def test_transform_no_env(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 1024

    def test_transform_compose(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = Compose(
            VIPTransform(
                model,
                in_keys=in_keys,
                out_keys=out_keys,
                tensor_pixels_keys=tensor_pixels_key,
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 1024

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_vip_instantiation(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vip)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if tensor_pixels_key:
            expected_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("stack_images", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_vip_mult_images(self, model, device, stack_images, parallel):
        in_keys = ["pixels", "pixels2"]
        out_keys = ["vec"] if stack_images else ["vec", "vec2"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            stack_images=stack_images,
        )

        def base_env_constructor():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                CatTensors(["pixels"], "pixels2", del_keys=False),
            )

        assert base_env_constructor().device == device
        if parallel:
            base_env = ParallelEnv(2, base_env_constructor)
        else:
            base_env = base_env_constructor()
        assert base_env.device == device

        transformed_env = TransformedEnv(base_env, vip)
        assert transformed_env.device == device
        assert vip.device == device

        td = transformed_env.reset()
        assert td.device == device
        if stack_images:
            expected_keys = {"pixels_orig", "done", "vec", "terminated"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys
        else:
            expected_keys = {"pixels_orig", "done", "vec", "vec2", "terminated"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not stack_images:
            expected_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    def test_transform_env(self, model, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(4, lambda: DiscreteActionConvMockEnvNumpy().to(device))
        transformed_env = TransformedEnv(base_env, vip)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"vec", "done", "pixels_orig", "terminated"}
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()
        del transformed_env

    def test_vip_parallel_reward(self, model, device, dtype_fixture):  # noqa
        torch.manual_seed(1)
        in_keys = ["pixels"]
        out_keys = ["vec"]
        tensor_pixels_key = None
        vip = VIPRewardTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = ParallelEnv(4, lambda: DiscreteActionConvMockEnvNumpy().to(device))
        transformed_env = TransformedEnv(base_env, vip)
        tensordict_reset = TensorDict(
            {"goal_image": torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)},
            [4],
            device=device,
        )
        with pytest.raises(
            KeyError,
            match=r"VIPRewardTransform.* requires .* key to be present in the input tensordict",
        ):
            _ = transformed_env.reset()
        with pytest.raises(
            KeyError,
            match=r"VIPRewardTransform.* requires .* key to be present in the input tensordict",
        ):
            _ = transformed_env.reset(tensordict_reset.empty())

        td = transformed_env.reset(tensordict_reset)
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {
            "vec",
            "done",
            "pixels_orig",
            "goal_embedding",
            "goal_image",
            "terminated",
        }
        if tensor_pixels_key:
            expected_keys.add(tensor_pixels_key)
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        assert set(td.keys(True)) == expected_keys, td

        torch.manual_seed(1)
        tensordict_reset = TensorDict(
            {"goal_image": torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)},
            [4],
            device=device,
        )
        td = transformed_env.rollout(
            5, auto_reset=False, tensordict=transformed_env.reset(tensordict_reset)
        )
        assert set(td.keys(True)) == expected_keys, td
        # test that we do compute the reward we want
        cur_embedding = td["next", "vec"]
        goal_embedding = td["goal_embedding"]
        last_embedding = td["vec"]

        # test that there is only one goal embedding
        goal = td["goal_embedding"]
        goal_expand = td["goal_embedding"][:, :1].expand_as(td["goal_embedding"])
        torch.testing.assert_close(goal, goal_expand)

        torch.testing.assert_close(cur_embedding[:, :-1], last_embedding[:, 1:])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(cur_embedding[:, 1:], last_embedding[:, :-1])

        explicit_reward = -torch.linalg.norm(cur_embedding - goal_embedding, dim=-1) - (
            -torch.linalg.norm(last_embedding - goal_embedding, dim=-1)
        )
        torch.testing.assert_close(explicit_reward, td["next", "reward"].squeeze())

        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    @pytest.mark.parametrize(
        "in_keys",
        [["pixels"], ["pixels_1", "pixels_2", "pixels_3"]],
    )
    @pytest.mark.parametrize(
        "out_keys",
        [["vip_vec"], ["vip_vec_1", "vip_vec_2", "vip_vec_3"]],
    )
    def test_vipnet_transform_observation_spec(
        self, in_keys, out_keys, del_keys, device, model
    ):
        vip_net = _VIPNet(in_keys, out_keys, model, del_keys)

        observation_spec = CompositeSpec(
            {key: BoundedTensorSpec(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = CompositeSpec(
                {key: UnboundedContinuousTensorSpec(1024, device) for key in out_keys}
            )

            observation_spec_out = vip_net.transform_observation_spec(observation_spec)

            for key in in_keys:
                assert key not in observation_spec_out
            for key in out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].device == exp_ts[key].device
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
        else:
            ts_dict = {}
            for key in in_keys:
                ts_dict[key] = observation_spec[key]
            for key in out_keys:
                ts_dict[key] = UnboundedContinuousTensorSpec(1024, device)
            exp_ts = CompositeSpec(ts_dict)

            observation_spec_out = vip_net.transform_observation_spec(observation_spec)

            for key in in_keys + out_keys:
                assert observation_spec_out[key].shape == exp_ts[key].shape
                assert observation_spec_out[key].dtype == exp_ts[key].dtype
                assert observation_spec_out[key].device == exp_ts[key].device

    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_vip_spec_against_real(self, model, tensor_pixels_key, device):
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vip)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + ["action"]
            + list(transformed_env.observation_spec.keys())
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [
                ("next", "reward"),
                ("next", "done"),
                "done",
                ("next", "terminated"),
                "terminated",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


@pytest.mark.skipif(not _has_vc, reason="vc_models not installed")
@pytest.mark.skipif(not torch.cuda.device_count(), reason="VC1 should run on cuda")
@pytest.mark.parametrize("device", [torch.device("cuda:0")])
class TestVC1(TransformBase):
    def test_transform_inverse(self, device):
        raise pytest.skip("no inverse for VC1Transform")

    def test_single_trans_env_check(self, device):
        del_keys = False
        in_keys = ["pixels"]
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device), vc1
        )
        check_env_specs(transformed_env)

    def test_trans_serial_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            SerialEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vc1
        )
        check_env_specs(transformed_env)

    def test_trans_parallel_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), vc1
        )
        try:
            check_env_specs(transformed_env)
        finally:
            transformed_env.close()

    def test_serial_trans_env_check(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]

        def make_env():
            t = VC1Transform(
                in_keys=in_keys,
                out_keys=out_keys,
                del_keys=del_keys,
                model_name="default",
            )

            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy().to(device),
                t,
            )

        transformed_env = SerialEnv(2, make_env)
        check_env_specs(transformed_env)

    def test_parallel_trans_env_check(self, device):
        # let's spare this one
        return

    def test_transform_model(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        module = nn.Sequential(vc1, nn.Identity())
        sample = module(td)
        assert "vec" in sample.keys()
        if del_keys:
            assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 16

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, device, rbclass):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(vc1)
        td = TensorDict({"pixels": torch.randint(255, (10, 244, 244, 3))}, [10])
        rb.extend(td)
        sample = rb.sample(10)
        assert "vec" in sample.keys()
        if del_keys:
            assert "pixels" not in sample.keys()
        assert sample["vec"].shape[-1] == 16

    def test_transform_no_env(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vc1(td)
        assert "vec" in td.keys()
        if del_keys:
            assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 16

    def test_transform_compose(self, device):
        in_keys = ["pixels"]
        del_keys = False
        out_keys = ["vec"]
        vip = Compose(
            VC1Transform(
                in_keys=in_keys,
                out_keys=out_keys,
                del_keys=del_keys,
                model_name="default",
            )
        )
        td = TensorDict({"pixels": torch.randint(255, (244, 244, 3))}, [])
        vip(td)
        assert "vec" in td.keys()
        if del_keys:
            assert "pixels" not in td.keys()
        assert td["vec"].shape[-1] == 16

    @pytest.mark.parametrize("del_keys", [False, True])
    def test_vc1_instantiation(self, del_keys, device):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vc1)
        td = transformed_env.reset()
        assert td.device == device
        expected_keys = {"nested", "done", "pixels_orig", "terminated"}
        if not del_keys:
            expected_keys.add("pixels")
        assert set(td.keys()) == expected_keys, set(td.keys()) - expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "nested"),
                ("next", "nested", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("nested", "vec"),
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not del_keys:
            expected_keys.add(("next", "pixels"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()

    @pytest.mark.parametrize("del_keys", [True, False])
    def test_transform_env(self, device, del_keys):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]

        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = ParallelEnv(4, lambda: DiscreteActionConvMockEnvNumpy().to(device))
        transformed_env = TransformedEnv(base_env, vc1)
        td = transformed_env.reset()
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        expected_keys = {"nested", "done", "pixels_orig", "terminated"}
        if not del_keys:
            expected_keys.add("pixels")
        assert set(td.keys()) == expected_keys

        td = transformed_env.rand_step(td)
        expected_keys = expected_keys.union(
            {
                ("next", "nested"),
                ("next", "nested", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("nested", "vec"),
                ("next", "reward"),
                ("next", "done"),
                ("next", "terminated"),
            }
        )
        if not del_keys:
            expected_keys.add(("next", "pixels"))
        assert set(td.keys(True)) == expected_keys, set(td.keys(True)) - expected_keys
        transformed_env.close()
        del transformed_env

    @pytest.mark.parametrize("del_keys", [True, False])
    def test_vc1_spec_against_real(self, del_keys, device):
        in_keys = ["pixels"]
        out_keys = [("nested", "vec")]
        vc1 = VC1Transform(
            in_keys=in_keys,
            out_keys=out_keys,
            del_keys=del_keys,
            model_name="default",
        )
        base_env = DiscreteActionConvMockEnvNumpy().to(device)
        transformed_env = TransformedEnv(base_env, vc1)
        expected_keys = (
            list(transformed_env.state_spec.keys())
            + ["action"]
            + list(transformed_env.observation_spec.keys(True))
            + [
                unravel_key(("next", key))
                for key in transformed_env.observation_spec.keys(True)
            ]
            + [
                ("next", "reward"),
                ("next", "done"),
                "done",
                ("next", "terminated"),
                "terminated",
                "next",
            ]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


class TestVecNorm:
    SEED = -1

    @staticmethod
    def _test_vecnorm_subproc_auto(
        idx, make_env, queue_out: mp.Queue, queue_in: mp.Queue
    ):
        env = make_env()
        env.set_seed(1000 + idx)
        tensordict = env.reset()
        for _ in range(10):
            tensordict = env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put(True)
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        t = env.transform
        obs_sum = t._td.get("observation_sum").clone()
        obs_ssq = t._td.get("observation_ssq").clone()
        obs_count = t._td.get("observation_count").clone()
        reward_sum = t._td.get("reward_sum").clone()
        reward_ssq = t._td.get("reward_ssq").clone()
        reward_count = t._td.get("reward_count").clone()

        queue_out.put(
            (obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count)
        )
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        env.close()
        queue_out.close()
        queue_in.close()
        del queue_in, queue_out

    @pytest.mark.parametrize("nprc", [2, 5])
    def test_vecnorm_parallel_auto(self, nprc):
        queues = []
        prcs = []
        if _has_gym:
            make_env = EnvCreator(
                lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED()), VecNorm(decay=1.0))
            )
        else:
            make_env = EnvCreator(
                lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm(decay=1.0))
            )

        for idx in range(nprc):
            prc_queue_in = mp.Queue(1)
            prc_queue_out = mp.Queue(1)
            p = mp.Process(
                target=self._test_vecnorm_subproc_auto,
                args=(
                    idx,
                    make_env,
                    prc_queue_in,
                    prc_queue_out,
                ),
            )
            p.start()
            prcs.append(p)
            queues.append((prc_queue_in, prc_queue_out))

        dones = [queue[0].get() for queue in queues]
        assert all(dones)
        msg = "all_done"
        for idx in range(nprc):
            queues[idx][1].put(msg)

        td = make_env.state_dict()["_extra_state"]["td"]

        obs_sum = td.get("observation_sum").clone()
        obs_ssq = td.get("observation_ssq").clone()
        obs_count = td.get("observation_count").clone()
        reward_sum = td.get("reward_sum").clone()
        reward_ssq = td.get("reward_ssq").clone()
        reward_count = td.get("reward_count").clone()

        assert obs_count == nprc * 11 + 2  # 10 steps + reset + init

        for idx in range(nprc):
            tup = queues[idx][0].get(timeout=TIMEOUT)
            (
                _obs_sum,
                _obs_ssq,
                _obs_count,
                _reward_sum,
                _reward_ssq,
                _reward_count,
            ) = tup
            assert (obs_sum == _obs_sum).all(), "sum"
            assert (obs_ssq == _obs_ssq).all(), "ssq"
            assert (obs_count == _obs_count).all(), "count"
            assert (reward_sum == _reward_sum).all(), "sum"
            assert (reward_ssq == _reward_ssq).all(), "ssq"
            assert (reward_count == _reward_count).all(), "count"

            obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count = (
                _obs_sum,
                _obs_ssq,
                _obs_count,
                _reward_sum,
                _reward_ssq,
                _reward_count,
            )
        msg = "all_done"
        for idx in range(nprc):
            queues[idx][1].put(msg)

        del queues
        for p in prcs:
            p.join()

    @staticmethod
    def _run_parallelenv(parallel_env, queue_in, queue_out):
        tensordict = parallel_env.reset()
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("first round")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("second round")
        parallel_env.close()
        queue_out.close()
        queue_in.close()
        del parallel_env, queue_out, queue_in

    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason="Nested spawned multiprocessed is currently failing in python 3.11. "
        "See https://github.com/python/cpython/pull/108568 for info and fix.",
    )
    def test_parallelenv_vecnorm(self):
        if _has_gym:
            make_env = EnvCreator(
                lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED()), VecNorm())
            )
        else:
            make_env = EnvCreator(
                lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm())
            )
        parallel_env = ParallelEnv(
            2,
            make_env,
        )
        queue_out = mp.Queue(1)
        queue_in = mp.Queue(1)
        proc = mp.Process(
            target=self._run_parallelenv, args=(parallel_env, queue_out, queue_in)
        )
        proc.start()
        parallel_sd = parallel_env.state_dict()
        assert "worker0" in parallel_sd
        worker_sd = parallel_sd["worker0"]
        td = worker_sd["_extra_state"]["td"]
        queue_out.put("start")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "first round"
        values = td.clone()
        queue_out.put("start")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "second round"
        new_values = td.clone()
        for k, item in values.items():
            if k in ["reward_sum", "reward_ssq"] and not _has_gym:
                # mocking env rewards are sparse
                continue
            assert (item != new_values.get(k)).any(), k
        proc.join()
        if not parallel_env.is_closed:
            parallel_env.close()

    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.skipif(not _has_gym, reason="no gym library found")
    @pytest.mark.parametrize(
        "parallel",
        [
            None,
            False,
            True,
        ],
    )
    def test_vecnorm_rollout(self, parallel, thr=0.2, N=200):
        self.SEED += 1
        torch.manual_seed(self.SEED)

        if parallel is None:
            env = GymEnv(PENDULUM_VERSIONED())
        elif parallel:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )
        else:
            env = SerialEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )

        env.set_seed(self.SEED)
        t = VecNorm(decay=1.0)
        env_t = TransformedEnv(env, t)
        td = env_t.reset()
        tds = []
        for _ in range(N):
            td = env_t.rand_step(td)
            tds.append(td.clone())
            td = step_mdp(td)
            if td.get("done").any():
                td = env_t.reset()
        tds = torch.stack(tds, 0)
        obs = tds.get(("next", "observation"))
        obs = obs.view(-1, obs.shape[-1])
        mean = obs.mean(0)
        assert (abs(mean) < thr).all()
        std = obs.std(0)
        assert (abs(std - 1) < thr).all()
        if not env_t.is_closed:
            env_t.close()
        self.SEED = 0

    def test_pickable(self):

        transform = VecNorm()
        serialized = pickle.dumps(transform)
        transform2 = pickle.loads(serialized)
        assert transform.__dict__.keys() == transform2.__dict__.keys()
        for key in sorted(transform.__dict__.keys()):
            assert isinstance(transform.__dict__[key], type(transform2.__dict__[key]))


def test_added_transforms_are_in_eval_mode_trivial():
    base_env = ContinuousActionVecMockEnv()
    t = TransformedEnv(base_env)
    assert not t.transform.training

    t.train()
    assert t.transform.training


def test_added_transforms_are_in_eval_mode():
    base_env = ContinuousActionVecMockEnv()
    r = RewardScaling(0, 1)
    t = TransformedEnv(base_env, r)
    assert not t.transform.training
    t.append_transform(RewardScaling(0, 1))
    assert not t.transform[1].training

    t.train()
    assert t.transform.training
    assert t.transform[0].training
    assert t.transform[1].training


class TestTransformedEnv:
    def test_independent_obs_specs_from_shared_env(self):
        obs_spec = CompositeSpec(
            observation=BoundedTensorSpec(low=0, high=10, shape=torch.Size((1,)))
        )
        base_env = ContinuousActionVecMockEnv(observation_spec=obs_spec)
        t1 = TransformedEnv(
            base_env, transform=ObservationNorm(in_keys=["observation"], loc=3, scale=2)
        )
        t2 = TransformedEnv(
            base_env, transform=ObservationNorm(in_keys=["observation"], loc=1, scale=6)
        )

        t1_obs_spec = t1.observation_spec
        t2_obs_spec = t2.observation_spec

        assert t1_obs_spec["observation"].space.low == 3
        assert t1_obs_spec["observation"].space.high == 23

        assert t2_obs_spec["observation"].space.low == 1
        assert t2_obs_spec["observation"].space.high == 61

        assert base_env.observation_spec["observation"].space.low == 0
        assert base_env.observation_spec["observation"].space.high == 10

    def test_independent_reward_specs_from_shared_env(self):
        reward_spec = UnboundedContinuousTensorSpec()
        base_env = ContinuousActionVecMockEnv(reward_spec=reward_spec)
        t1 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=0, clamp_max=4)
        )
        t2 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=-2, clamp_max=2)
        )

        t1_reward_spec = t1.reward_spec
        t2_reward_spec = t2.reward_spec

        assert t1_reward_spec.space.minimum == 0
        assert t1_reward_spec.space.maximum == 4

        assert t2_reward_spec.space.minimum == -2
        assert t2_reward_spec.space.maximum == 2

        assert base_env.reward_spec.space.minimum == -np.inf
        assert base_env.reward_spec.space.maximum == np.inf

    def test_allow_done_after_reset(self):
        base_env = ContinuousActionVecMockEnv(allow_done_after_reset=True)
        assert base_env._allow_done_after_reset
        t1 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=0, clamp_max=4)
        )
        assert t1._allow_done_after_reset
        with pytest.raises(
            RuntimeError,
            match="_allow_done_after_reset is a read-only property for TransformedEnvs",
        ):
            t1._allow_done_after_reset = False
        base_env._allow_done_after_reset = False
        assert not t1._allow_done_after_reset


def test_nested_transformed_env():
    base_env = ContinuousActionVecMockEnv()
    t1 = RewardScaling(0, 1)
    t2 = RewardScaling(0, 2)
    env = TransformedEnv(TransformedEnv(base_env, t1), t2)

    assert env.base_env is base_env
    assert isinstance(env.transform, Compose)
    children = list(env.transform.transforms.children())
    assert len(children) == 2
    assert children[0].scale == 1
    assert children[1].scale == 2


def test_transform_parent():
    base_env = ContinuousActionVecMockEnv()
    t1 = RewardScaling(0, 1)
    t2 = RewardScaling(0, 2)
    env = TransformedEnv(TransformedEnv(base_env, t1), t2)
    t3 = RewardClipping(0.1, 0.5)
    env.append_transform(t3)

    t1_parent_gt = t1._container
    t2_parent_gt = t2._container
    t3_parent_gt = t3._container

    _ = t1.parent
    _ = t2.parent
    _ = t3.parent

    assert t1_parent_gt == t1._container
    assert t2_parent_gt == t2._container
    assert t3_parent_gt == t3._container


def test_transform_parent_cache():
    """Tests the caching and uncaching of the transformed envs."""
    env = TransformedEnv(
        ContinuousActionVecMockEnv(),
        FrameSkipTransform(3),
    )

    # print the parent
    assert (
        type(env.transform.parent.transform) is Compose
        and len(env.transform.parent.transform) == 0
    )
    transform = env.transform
    parent1 = env.transform.parent
    parent2 = env.transform.parent
    assert parent1 is parent2

    # change the env, re-print the parent
    env.insert_transform(0, NoopResetEnv(3))
    parent3 = env.transform[-1].parent
    assert parent1 is not parent3
    assert type(parent3.transform[0]) is NoopResetEnv

    # change the env, re-print the parent
    env.insert_transform(0, CatTensors(["observation"]))
    parent4 = env.transform[-1].parent
    assert parent1 is not parent4
    assert parent3 is not parent4
    assert type(parent4.transform[0]) is CatTensors
    assert type(parent4.transform[1]) is NoopResetEnv

    # check that we don't keep track of the wrong parent
    env.transform = NoopResetEnv(3)
    assert transform.parent is None


class TestTransforms:
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(
            in_keys=keys,
            N=4,
            dim=-3,
        )
        t2 = FiniteTensorDictCheck()
        t3 = ExcludeTransform()
        compose = Compose(t1, t2, t3)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, nchannels, 16, 16), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        if not batch:
            with pytest.raises(
                ValueError,
                match="CatFrames cannot process unbatched tensordict instances",
            ):
                compose(td.clone(False))
        compose._call(td)
        for key in keys:
            assert td.get(key).shape[-3] == nchannels * N
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(0, 255, (nchannels, 16, 16))
            # StepCounter does not want non composite specs
            observation_spec = compose[:2].transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels * N, 16, 16])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(0, 255, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = compose.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size(
                    [nchannels * N, 16, 16]
                )

    def test_lambda_functions(self):
        def trsf(data):
            if "y" in data.keys():
                data["y"] += 1
                return data
            return data.set("y", torch.zeros(data.shape))

        env = TransformedEnv(CountingEnv(5), trsf)
        env.append_transform(trsf)
        env.insert_transform(0, trsf)
        # With Compose
        env.transform.append(trsf)
        assert env.reset().get("y") == 3
        env.transform = trsf
        assert env.reset().get("y") == 0

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("keys_inv_1", [["action_1"], []])
    @pytest.mark.parametrize("keys_inv_2", [["action_2"], []])
    def test_compose_inv(self, keys_inv_1, keys_inv_2, device):
        torch.manual_seed(0)
        keys_to_transform = set(keys_inv_1 + keys_inv_2)
        keys_total = {"action_1", "action_2", "dont_touch"}
        double2float_1 = DoubleToFloat(in_keys_inv=keys_inv_1)
        double2float_2 = DoubleToFloat(in_keys_inv=keys_inv_2)
        compose = Compose(double2float_1, double2float_2)
        td = TensorDict(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.float32, device=device)
                for key in keys_total
            },
            [1],
            device=device,
        )

        td = compose.inv(td)
        for key in keys_to_transform:
            assert td.get(key).dtype == torch.double
        for key in keys_total - keys_to_transform:
            assert td.get(key).dtype == torch.float32

    def test_compose_indexing(self):
        c = Compose(
            ObservationNorm(loc=1.0, scale=1.0, in_keys=["observation"]),
            RewardScaling(loc=0, scale=1),
            ObservationNorm(loc=2.0, scale=2.0, in_keys=["observation"]),
        )
        base_env = ContinuousActionVecMockEnv()
        env = TransformedEnv(base_env, c)
        last_t = env.transform[-1]
        assert last_t.scale == 2
        env.transform[-1].scale += 1
        assert last_t.scale == 3
        # indexing a sequence of transforms involves re-creating a Compose, which requires a clone
        # because we need to deparent the transforms
        sub_compose = env.transform[1:]
        assert isinstance(sub_compose, Compose)
        last_t2 = sub_compose[-1]
        assert last_t2.scale == 3
        # this involves clone, but the value of the registered buffer should still match
        env.transform[1:][-1].scale += 1
        assert last_t.scale == 4
        assert last_t2.scale == 4

    @pytest.mark.parametrize("device", get_default_devices())
    def test_finitetensordictcheck(self, device):
        ftd = FiniteTensorDictCheck()
        td = TensorDict(
            {key: torch.randn(1, 3, 3, device=device) for key in ["a", "b", "c"]}, [1]
        )
        ftd._call(td)
        td.set("inf", torch.zeros(1, 3).fill_(float("inf")))
        with pytest.raises(ValueError, match="Encountered a non-finite tensor"):
            ftd._call(td)
        with pytest.raises(ValueError, match="Encountered a non-finite tensor"):
            ftd(td)

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device found")
    @pytest.mark.parametrize("device", get_default_devices())
    def test_pin_mem(self, device):
        pin_mem = PinMemoryTransform()
        td = TensorDict(
            {key: torch.randn(3) for key in ["a", "b", "c"]}, [], device=device
        )
        if device.type == "cuda":
            with pytest.raises(RuntimeError, match="cannot pin"):
                pin_mem(td)
            with pytest.raises(RuntimeError, match="cannot pin"):
                pin_mem._call(td)
            return
        pin_mem(td)
        for item in td.values():
            assert item.is_pinned

    def test_append(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        (key,) = itertools.islice(obs_spec.keys(), 1)

        env = TransformedEnv(env)
        env.append_transform(CatFrames(N=4, dim=-1, in_keys=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 1
        obs_spec = env.observation_spec
        obs_spec = obs_spec[key]
        assert obs_spec.shape[-1] == 4 * env.base_env.observation_spec[key].shape[-1]

    def test_insert(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        (key,) = itertools.islice(obs_spec.keys(), 1)
        env = TransformedEnv(env)

        # we start by asking the spec. That will create the private attributes
        _ = env.action_spec
        _ = env.observation_spec
        _ = env.reward_spec

        assert env._input_spec is not None
        assert "full_action_spec" in env._input_spec
        assert env._input_spec["full_action_spec"] is not None
        assert env._output_spec["full_observation_spec"] is not None
        assert env._output_spec["full_reward_spec"] is not None
        assert env._output_spec["full_done_spec"] is not None

        env.insert_transform(0, CatFrames(N=4, dim=-1, in_keys=[key]))

        # transformed envs do not have spec after insert -- they need to be computed
        assert env._input_spec is None
        assert env._output_spec is None

        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 1
        obs_spec = env.observation_spec
        obs_spec = obs_spec[key]
        assert obs_spec.shape[-1] == 4 * env.base_env.observation_spec[key].shape[-1]

        env.insert_transform(1, FiniteTensorDictCheck())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 2
        assert isinstance(env.transform[-1], FiniteTensorDictCheck)
        assert isinstance(env.transform[0], CatFrames)

        env.insert_transform(0, NoopResetEnv())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 3
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], CatFrames)
        assert isinstance(env.transform[2], FiniteTensorDictCheck)

        env.insert_transform(2, NoopResetEnv())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 4
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], CatFrames)
        assert isinstance(env.transform[2], NoopResetEnv)
        assert isinstance(env.transform[3], FiniteTensorDictCheck)

        env.insert_transform(-3, PinMemoryTransform())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 5
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], PinMemoryTransform)
        assert isinstance(env.transform[2], CatFrames)
        assert isinstance(env.transform[3], NoopResetEnv)
        assert isinstance(env.transform[4], FiniteTensorDictCheck)

        assert env._input_spec is None
        assert env._output_spec is None

        env.insert_transform(-5, CatFrames(N=4, dim=-1, in_keys=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 6

        assert isinstance(env.transform[0], CatFrames)
        assert isinstance(env.transform[1], NoopResetEnv)
        assert isinstance(env.transform[2], PinMemoryTransform)
        assert isinstance(env.transform[3], CatFrames)
        assert isinstance(env.transform[4], NoopResetEnv)
        assert isinstance(env.transform[5], FiniteTensorDictCheck)

        assert env._input_spec is None
        assert env._output_spec is None

        _ = copy(env.action_spec)
        _ = copy(env.observation_spec)
        _ = copy(env.reward_spec)

        with pytest.raises(ValueError):
            env.insert_transform(-7, FiniteTensorDictCheck())

        with pytest.raises(ValueError):
            env.insert_transform(7, FiniteTensorDictCheck())

        with pytest.raises(ValueError):
            env.insert_transform(4, "ffff")


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_locked_transformed(device):
    env = TransformedEnv(
        MockBatchedLockedEnv(device),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    env.step(td)

    with pytest.raises(
        RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
    ):
        env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked_transformed(device):
    env = TransformedEnv(
        MockBatchedUnLockedEnv(device),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    env.step(td)
    env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked_with_batch_size_transformed(device):
    env = TransformedEnv(
        MockBatchedUnLockedEnv(device, batch_size=torch.Size([2])),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    env.step(td)
    td_expanded = td.expand(2, 2).reshape(-1).to_tensordict()

    with pytest.raises(
        RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
    ):
        env.step(td_expanded)


transforms = [
    ToTensorImage,
    pytest.param(
        partial(RewardClipping, clamp_min=0.1, clamp_max=0.9), id="RewardClipping"
    ),
    BinarizeReward,
    pytest.param(
        partial(Resize, w=2, h=2),
        id="Resize",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(CenterCrop, w=1),
        id="CenterCrop",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(FlattenObservation, first_dim=-3, last_dim=-3), id="FlattenObservation"
    ),
    pytest.param(
        partial(UnsqueezeTransform, unsqueeze_dim=-1), id="UnsqueezeTransform"
    ),
    pytest.param(partial(SqueezeTransform, squeeze_dim=-1), id="SqueezeTransform"),
    GrayScale,
    pytest.param(
        partial(ObservationNorm, in_keys=["observation"]), id="ObservationNorm"
    ),
    pytest.param(partial(CatFrames, dim=-3, N=4), id="CatFrames"),
    pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
    FiniteTensorDictCheck,
    DoubleToFloat,
    CatTensors,
    pytest.param(
        partial(DiscreteActionProjection, max_actions=1, num_actions_effective=1),
        id="DiscreteActionProjection",
    ),
    NoopResetEnv,
    TensorDictPrimer,
    PinMemoryTransform,
    gSDENoise,
    VecNorm,
]


@pytest.mark.parametrize("transform", transforms)
def test_smoke_compose_transform(transform):
    Compose(transform())


@pytest.mark.parametrize("transform", transforms)
def test_clone_parent(transform):
    base_env1 = ContinuousActionVecMockEnv()
    base_env2 = ContinuousActionVecMockEnv()
    env = TransformedEnv(base_env1, transform())
    env_clone = TransformedEnv(base_env2, env.transform.clone())

    assert env_clone.transform.parent.base_env is not base_env1
    assert env_clone.transform.parent.base_env is base_env2
    assert env.transform.parent.base_env is not base_env2
    assert env.transform.parent.base_env is base_env1


@pytest.mark.parametrize("transform", transforms)
def test_clone_parent_compose(transform):
    base_env1 = ContinuousActionVecMockEnv()
    base_env2 = ContinuousActionVecMockEnv()
    env = TransformedEnv(base_env1, Compose(ToTensorImage(), transform()))
    t = env.transform.clone()

    assert t.parent is None
    assert t[0].parent is None
    assert t[1].parent is None

    env_clone = TransformedEnv(base_env2, Compose(ToTensorImage(), *t))

    assert env_clone.transform[0].parent.base_env is not base_env1
    assert env_clone.transform[0].parent.base_env is base_env2
    assert env.transform[0].parent.base_env is not base_env2
    assert env.transform[0].parent.base_env is base_env1
    assert env_clone.transform[1].parent.base_env is not base_env1
    assert env_clone.transform[1].parent.base_env is base_env2
    assert env.transform[1].parent.base_env is not base_env2
    assert env.transform[1].parent.base_env is base_env1


class TestCroSeq:
    def test_crop_dim1(self):
        tensordict = TensorDict(
            {
                "a": torch.arange(20).view(1, 1, 1, 20).expand(3, 4, 2, 20),
                "b": TensorDict(
                    {"c": torch.arange(20).view(1, 1, 1, 20, 1).expand(3, 4, 2, 20, 1)},
                    [3, 4, 2, 20, 1],
                ),
            },
            [3, 4, 2, 20],
        )
        t = RandomCropTensorDict(11, -1)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 2, 11])
        assert tensordict_crop["b"].shape == torch.Size([3, 4, 2, 11, 1])
        assert (
            tensordict_crop["a"][:, :, :, :-1] + 1 == tensordict_crop["a"][:, :, :, 1:]
        ).all()

    def test_crop_dim2(self):
        tensordict = TensorDict(
            {"a": torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2)},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(11, -2)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 11, 2])
        assert (
            tensordict_crop["a"][:, :, :-1] + 1 == tensordict_crop["a"][:, :, 1:]
        ).all()

    def test_crop_error(self):
        tensordict = TensorDict(
            {"a": torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2)},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(21, -2)
        with pytest.raises(RuntimeError, match="Cannot sample trajectories of length"):
            _ = t(tensordict)

    @pytest.mark.parametrize("mask_key", ("mask", ("collector", "mask")))
    def test_crop_mask(self, mask_key):
        a = torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2).clone()
        mask = a < 21
        mask[0] = a[0] < 15
        mask[1] = a[1] < 16
        mask[1] = a[2] < 14
        tensordict = TensorDict(
            {"a": a, mask_key: mask},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(15, -2, mask_key=mask_key)
        with pytest.raises(RuntimeError, match="Cannot sample trajectories of length"):
            _ = t(tensordict)
        t = RandomCropTensorDict(13, -2, mask_key=mask_key)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 13, 2])
        assert tensordict_crop[mask_key].all()


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

    def test_parallel_trans_env_check(self, create_copy):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation"],
                    ["stuff"],
                    create_copy=create_copy,
                ),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

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

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

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

    def test_trans_parallel_env_check(self, create_copy):
        def make_env():
            return ContinuousActionVecMockEnv()

        env = TransformedEnv(
            ParallelEnv(2, make_env),
            RenameTransform(
                ["observation"],
                ["stuff"],
                create_copy=create_copy,
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()
        env = TransformedEnv(
            ParallelEnv(2, make_env),
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
            env.close()

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


class TestInitTracker(TransformBase):
    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_init_gym(
        self,
    ):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(StepCounter(max_steps=30), InitTracker()),
        )
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_dm_control, reason="no dm_control detected")
    def test_init_dmc(self):
        env = TransformedEnv(
            DMControlEnv("cheetah", "run"),
            Compose(StepCounter(max_steps=30), InitTracker()),
        )
        env.rollout(1000)
        check_env_specs(env)

    def test_single_trans_env_check(self):
        env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
        env = TransformedEnv(env, InitTracker())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            env = TransformedEnv(env, InitTracker())
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            env = TransformedEnv(env, InitTracker())
            return env

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = SerialEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = ParallelEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        with pytest.raises(ValueError, match="init_key can only be of type str"):
            InitTracker(init_key=("some", "nested"))
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            InitTracker()(None)

    def test_transform_compose(self):
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            Compose(InitTracker())(None)

    def test_transform_env(self):
        policy = lambda tensordict: tensordict.set(
            "action", torch.ones(tensordict.shape, dtype=torch.int32)
        )
        env = CountingBatchedEnv(max_steps=torch.tensor([3, 4]), batch_size=[2])
        env = TransformedEnv(env, InitTracker())
        r = env.rollout(100, policy, break_when_any_done=False)
        assert (r["is_init"].view(r.batch_size).sum(-1) == torch.tensor([25, 20])).all()

    def test_transform_model(self):
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            td = TensorDict({}, [])
            chain = nn.Sequential(InitTracker())
            chain(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        batch = [1]
        device = "cpu"
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(InitTracker())
        reward = torch.randn(*batch, 1, device=device)
        misc = torch.randn(*batch, 1, device=device)
        td = TensorDict(
            {"misc": misc, "reward": reward},
            batch,
            device=device,
        )
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            _ = rb.sample(20)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for InitTracker")

    @pytest.mark.parametrize("init_key", ["is_init", "loool"])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("max_steps", [5])
    def test_nested(
        self,
        nested_done,
        max_steps,
        init_key,
        batch_size=(32, 2),
        rollout_length=9,
    ):
        env = NestedCountingEnv(
            max_steps=max_steps, nest_done=nested_done, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.action_spec, action_key=env.action_key
        )
        transformed_env = TransformedEnv(
            env,
            InitTracker(init_key=init_key),
        )
        init_key = transformed_env.transform.init_keys[0]
        td = transformed_env.rollout(
            rollout_length, policy=policy, break_when_any_done=False
        )
        if nested_done:
            is_init = td[init_key][0, 0, :, 0, 0].clone()
        else:
            is_init = td[init_key][0, 0, :, 0].clone()
        if max_steps == 20:
            assert torch.all(is_init[0] == 1)
            assert torch.all(is_init[1:] == 0)
        else:
            assert torch.all(is_init[0] == 1)
            assert torch.all(is_init[1 : max_steps + 1] == 0)
            assert torch.all(is_init[max_steps + 1] == 1)
            assert torch.all(is_init[max_steps + 2 :] == 0)

        td_reset = TensorDict(
            rand_reset(transformed_env),
            batch_size=env.batch_size,
            device=env.device,
        )
        if nested_done:
            reset = td_reset["data", "_reset"]
        else:
            reset = td_reset["_reset"]

        td_reset = transformed_env.reset(td_reset)
        assert (td_reset[init_key] == reset).all()

    def test_inittracker_ignore(self):
        # checks that init keys respect the convention that nested dones should
        # be ignored if there is a done in a root td
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=True, nest_done=True), InitTracker()
        )
        assert len(env.transform.init_keys) == 1
        assert env.transform.init_keys[0] == "is_init"
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=False, nest_done=True), InitTracker()
        )
        assert len(env.transform.init_keys) == 1
        assert env.transform.init_keys[0] == ("data", "is_init")


class TestKLRewardTransform(TransformBase):
    envclass = ContinuousActionVecMockEnv

    def _make_actor(self):
        from tensordict.nn import NormalParamExtractor, TensorDictModule as Mod

        env = self.envclass()
        n_obs = env.observation_spec["observation"].shape[-1]
        n_act = env.action_spec.shape[-1]

        module = Mod(
            nn.Sequential(nn.Linear(n_obs, n_act * 2), NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        actor = ProbabilisticActor(
            module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )
        return actor

    def _make_transform_env(self, out_key, base_env):
        actor = self._make_actor()
        transform = KLRewardTransform(actor, out_keys=out_key)
        return Compose(
            TensorDictPrimer(
                primers={
                    "sample_log_prob": UnboundedContinuousTensorSpec(
                        shape=base_env.action_spec.shape[:-1]
                    )
                }
            ),
            transform,
        )

    @pytest.mark.parametrize("in_key", [None, "some_stuff", ["some_stuff"], ["a", "b"]])
    @pytest.mark.parametrize(
        "out_key", [None, "some_stuff", ["some_stuff"], ["a", "b"]]
    )
    def test_transform_no_env(self, in_key, out_key):
        actor = self._make_actor()
        if any(isinstance(key, list) and len(key) > 1 for key in (in_key, out_key)):
            with pytest.raises(ValueError):
                KLRewardTransform(actor, in_keys=in_key, out_keys=out_key)
            return
        t = KLRewardTransform(actor, in_keys=in_key, out_keys=out_key)
        batch = [2, 3]
        tensordict = TensorDict(
            {
                "action": torch.randn(*batch, 7),
                "observation": torch.randn(*batch, 7),
                "sample_log_prob": torch.randn(*batch),
            },
            batch,
        )
        next_td = TensorDict({t.in_keys[0]: torch.zeros(*batch, 1)}, batch)
        next_td = t._step(tensordict, next_td)
        tensordict.set("next", next_td)
        assert (tensordict.get("next").get(t.out_keys[0]) != 0).all()

    def test_transform_compose(self):
        actor = self._make_actor()
        t = Compose(KLRewardTransform(actor))
        batch = [2, 3]
        tensordict = TensorDict(
            {
                "action": torch.randn(*batch, 7),
                "observation": torch.randn(*batch, 7),
                "next": {t[0].in_keys[0]: torch.zeros(*batch, 1)},
                "sample_log_prob": torch.randn(*batch),
            },
            batch,
        )
        t(tensordict)
        assert (tensordict.get("next").get("reward") != 0).all()

    @torch.no_grad()
    @pytest.mark.parametrize(
        "out_key",
        [
            None,
            "some_stuff",
            ["some_stuff"],
        ],
    )
    def test_transform_env(self, out_key):
        base_env = self.envclass()
        actor = self._make_actor()
        # we need to patch the env and create a sample_log_prob spec to make check_env_specs happy
        env = TransformedEnv(
            base_env,
            Compose(
                RewardScaling(0.0, 0.0),  # make reward 0 to check the effect of kl
                KLRewardTransform(actor, out_keys=out_key),
            ),
        )
        actor = self._make_actor()
        td1 = env.rollout(3, actor)
        tdparams = TensorDict(dict(actor.named_parameters()), []).unflatten_keys(".")
        assert (
            tdparams
            == env.transform[-1].frozen_params.select(*tdparams.keys(True, True))
        ).all()

        def update(x):
            x.data += 1
            return x

        tdparams.apply_(update)
        td2 = env.rollout(3, actor)
        assert not (
            tdparams
            == env.transform[-1].frozen_params.select(*tdparams.keys(True, True))
        ).any()
        out_key = env.transform[-1].out_keys[0]
        assert (td1.get("next").get(out_key) != td2.get("next").get(out_key)).all()

    @pytest.mark.parametrize("out_key", [None, "some_stuff", ["some_stuff"]])
    def test_single_trans_env_check(self, out_key):
        base_env = self.envclass()
        # we need to patch the env and create a sample_log_prob spec to make check_env_specs happy
        env = TransformedEnv(base_env, self._make_transform_env(out_key, base_env))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        out_key = "reward"

        def make_env():
            base_env = self.envclass()
            return TransformedEnv(base_env, self._make_transform_env(out_key, base_env))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        out_key = "reward"

        def make_env():
            base_env = self.envclass()
            return TransformedEnv(base_env, self._make_transform_env(out_key, base_env))

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        out_key = "reward"
        base_env = SerialEnv(2, self.envclass)
        env = TransformedEnv(base_env, self._make_transform_env(out_key, base_env))
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        out_key = "reward"
        base_env = ParallelEnv(2, self.envclass)
        env = TransformedEnv(base_env, self._make_transform_env(out_key, base_env))
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_model(self):
        actor = self._make_actor()
        t = KLRewardTransform(actor, in_keys="reward", out_keys="reward")
        batch = [2, 3]
        tensordict = TensorDict(
            {
                "action": torch.randn(*batch, 7),
                "observation": torch.randn(*batch, 7),
                "next": {t.in_keys[0]: torch.zeros(*batch, 1)},
                "sample_log_prob": torch.randn(*batch),
            },
            batch,
        )
        t = TensorDictSequential(t)
        t(tensordict)
        assert (tensordict.get("next").get(t.out_keys[0]) != 0).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        actor = self._make_actor()
        t = KLRewardTransform(actor, in_keys="reward", out_keys="reward")
        batch = [2, 3]
        tensordict = TensorDict(
            {
                "action": torch.randn(*batch, 7),
                "observation": torch.randn(*batch, 7),
                "next": {t.in_keys[0]: torch.zeros(*batch, 1)},
                "sample_log_prob": torch.randn(*batch),
            },
            batch,
        )
        rb = rbclass(storage=LazyTensorStorage(100), transform=t)
        rb.extend(tensordict)
        sample = rb.sample(3)
        assert (sample.get(("next", "reward")) != 0).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for KLRewardTransform")

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_kl_diff(self, requires_grad):
        actor = self._make_actor()
        t = KLRewardTransform(
            actor, in_keys="reward", out_keys="reward", requires_grad=requires_grad
        )
        assert t.frozen_params.requires_grad is requires_grad

    def test_kl_lstm(self):
        from tensordict.nn import (
            NormalParamExtractor,
            ProbabilisticTensorDictModule,
            ProbabilisticTensorDictSequential,
            TensorDictModule,
        )

        env = TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())
        lstm_module = LSTMModule(
            input_size=env.observation_spec["observation"].shape[-1],
            hidden_size=2,
            in_keys=["observation", "rs_h", "rs_c"],
            out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
        )
        mlp = MLP(num_cells=[2], out_features=env.action_spec.shape[-1] * 2)
        policy = ProbabilisticTensorDictSequential(
            lstm_module,
            TensorDictModule(mlp, in_keys=["intermediate"], out_keys=["intermediate"]),
            TensorDictModule(
                NormalParamExtractor(),
                in_keys=["intermediate"],
                out_keys=["loc", "scale"],
            ),
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
            ),
        )
        policy(env.reset())
        klt = KLRewardTransform(policy)
        # check that this runs: it can only run if the params are nn.Parameter instances
        klt(env.rollout(3, policy))


class TestActionMask(TransformBase):
    @property
    def _env_class(self):
        from torchrl.data import BinaryDiscreteTensorSpec, DiscreteTensorSpec

        class MaskedEnv(EnvBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.action_spec = DiscreteTensorSpec(4)
                self.state_spec = CompositeSpec(
                    action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool)
                )
                self.observation_spec = CompositeSpec(
                    obs=UnboundedContinuousTensorSpec(3),
                    action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool),
                )
                self.reward_spec = UnboundedContinuousTensorSpec(1)

            def _reset(self, tensordict):
                td = self.observation_spec.rand()
                td.update(torch.ones_like(self.state_spec.rand()))
                return td

            def _step(self, data):
                td = self.observation_spec.rand()
                mask = data.get("action_mask")
                action = data.get("action")
                mask = mask.scatter(-1, action.unsqueeze(-1), 0)

                td.set("action_mask", mask)
                td.set("reward", self.reward_spec.rand())
                td.set("done", ~(mask.any().view(1)))
                return td

            def _set_seed(self, seed):
                return seed

        return MaskedEnv

    def test_single_trans_env_check(self):
        env = self._env_class()
        env = TransformedEnv(env, ActionMask())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(2, lambda: TransformedEnv(self._env_class(), ActionMask()))
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(2, lambda: TransformedEnv(self._env_class(), ActionMask()))
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, self._env_class), ActionMask())
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(ParallelEnv(2, self._env_class), ActionMask())
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        t = ActionMask()
        with pytest.raises(RuntimeError, match="parent cannot be None"):
            t._call(TensorDict({}, []))

    def test_transform_compose(self):
        env = self._env_class()
        env = TransformedEnv(env, Compose(ActionMask()))
        check_env_specs(env)

    def test_transform_env(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), ActionMask())
        with pytest.raises(ValueError, match="The action spec must be one of"):
            env.rollout(2)
        env = self._env_class()
        env = TransformedEnv(env, ActionMask())
        td = env.reset()
        for _ in range(1000):
            td = env.rand_action(td)
            assert env.action_spec.is_in(td.get("action"))
            td = env.step(td)
            td = step_mdp(td)
            if td.get("done"):
                break
        else:
            raise RuntimeError
        assert not td.get("action_mask").any()

    def test_transform_model(self):
        t = ActionMask()
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            t(TensorDict({}, []))

    def test_transform_rb(self):
        t = ActionMask()
        rb = ReplayBuffer(storage=LazyTensorStorage(100))
        rb.append_transform(t)
        rb.extend(TensorDict({"a": [1]}, [1]).expand(10))
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            rb.sample(3)

    def test_transform_inverse(self):
        # no inverse transform
        return


class TestDeviceCastTransform(TransformBase):
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

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(device="cpu:0"), DeviceCastTransform("cpu:1")
            )

        env = ParallelEnv(2, make_env)
        assert env.device == torch.device("cpu:1")
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(SerialEnv(2, make_env), DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        def make_env():
            return ContinuousActionVecMockEnv(device="cpu:0")

        env = TransformedEnv(ParallelEnv(2, make_env), DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        t = DeviceCastTransform("cpu:1", "cpu:0")
        assert t._call(TensorDict({}, [], device="cpu:0")).device == torch.device(
            "cpu:1"
        )

    def test_transform_compose(self):
        t = Compose(DeviceCastTransform("cpu:1", "cpu:0"))
        assert t._call(TensorDict({}, [], device="cpu:0")).device == torch.device(
            "cpu:1"
        )
        assert t._inv_call(TensorDict({}, [], device="cpu:1")).device == torch.device(
            "cpu:0"
        )

    def test_transform_env(self):
        env = ContinuousActionVecMockEnv(device="cpu:0")
        assert env.device == torch.device("cpu:0")
        env = TransformedEnv(env, DeviceCastTransform("cpu:1"))
        assert env.device == torch.device("cpu:1")
        assert env.transform.device == torch.device("cpu:1")
        assert env.transform.orig_device == torch.device("cpu:0")

    def test_transform_model(self):
        t = Compose(DeviceCastTransform("cpu:1", "cpu:0"))
        m = nn.Sequential(t)
        assert t(TensorDict({}, [], device="cpu:0")).device == torch.device("cpu:1")

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
        assert t._inv_call(TensorDict({}, [], device="cpu:1")).device == torch.device(
            "cpu:0"
        )


class TestPermuteTransform(TransformBase):
    envclass = DiscreteActionConvMockEnv

    @classmethod
    def _get_permute(cls):
        return PermuteTransform(
            (-1, -2, -3), in_keys=["pixels_orig", "pixels"], in_keys_inv=["pixels_orig"]
        )

    def test_single_trans_env_check(self):
        base_env = TestPermuteTransform.envclass()
        env = TransformedEnv(base_env, TestPermuteTransform._get_permute())
        check_env_specs(env)
        assert env.observation_spec["pixels"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] == env.observation_spec["pixels_orig"]

    def test_serial_trans_env_check(self):
        env = SerialEnv(
            2,
            lambda: TransformedEnv(
                TestPermuteTransform.envclass(), TestPermuteTransform._get_permute()
            ),
        )
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(
            2,
            lambda: TransformedEnv(
                TestPermuteTransform.envclass(), TestPermuteTransform._get_permute()
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, TestPermuteTransform.envclass),
            TestPermuteTransform._get_permute(),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, TestPermuteTransform.envclass),
            TestPermuteTransform._get_permute(),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_compose(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = Compose(
            PermuteTransform(
                dims=(-1, -4, -2, -3),
                in_keys=["pixels"],
            )
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        td = trans(td)
        assert td["pixels"].shape == torch.Size((*batch, C, D, H, W))

    def test_transform_env(self):
        base_env = TestPermuteTransform.envclass()
        env = TransformedEnv(base_env, TestPermuteTransform._get_permute())
        check_env_specs(env)
        assert env.observation_spec["pixels"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] == env.observation_spec["pixels_orig"]
        assert env.state_spec["pixels_orig"] != base_env.state_spec["pixels_orig"]
        assert env.observation_spec["pixels"] != base_env.observation_spec["pixels"]

        td = env.rollout(3)
        assert td["pixels"].shape == torch.Size([3, 7, 7, 1])

        # check error
        with pytest.raises(ValueError, match="Only tailing dims with negative"):
            t = PermuteTransform((-1, -10))

    def test_transform_model(self):
        batch = [2]
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        out_channels = 4
        from tensordict.nn import TensorDictModule

        model = nn.Sequential(
            trans,
            TensorDictModule(
                nn.Conv3d(C, out_channels, 3, padding=1),
                in_keys=["pixels"],
                out_keys=["pixels"],
            ),
        )
        td = model(td)
        assert td["pixels"].shape == torch.Size((*batch, out_channels, D, H, W))

    def test_transform_rb(self):
        batch = [6]
        D, W, H, C = 4, 5, 6, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(5), transform=trans)
        rb.extend(td)
        sample = rb.sample(2)
        assert sample["pixels"].shape == torch.Size([2, C, D, H, W])

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_inverse(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys_inv=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, C, D, H, W))}, batch_size=batch)
        td = trans.inv(td)
        assert td["pixels"].shape == torch.Size((*batch, D, W, H, C))

    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    def test_transform_no_env(self, batch):
        D, W, H, C = 8, 32, 64, 3
        trans = PermuteTransform(
            dims=(-1, -4, -2, -3),
            in_keys=["pixels"],
        )  # DxWxHxC => CxDxHxW
        td = TensorDict({"pixels": torch.randn((*batch, D, W, H, C))}, batch_size=batch)
        td = trans(td)
        assert td["pixels"].shape == torch.Size((*batch, C, D, H, W))


@pytest.mark.skipif(
    not _has_gymnasium,
    reason="EndOfLifeTransform can only be tested when Gym is present.",
)
class TestEndOfLife(TransformBase):
    def test_trans_parallel_env_check(self):
        def make():
            with set_gym_backend("gymnasium"):
                return GymEnv("ALE/Breakout-v5")

        with pytest.warns(UserWarning, match="The base_env is not a gym env"):
            with pytest.raises(AttributeError):
                env = TransformedEnv(
                    ParallelEnv(2, make), transform=EndOfLifeTransform()
                )
                check_env_specs(env)

    def test_trans_serial_env_check(self):
        def make():
            with set_gym_backend("gymnasium"):
                return GymEnv("ALE/Breakout-v5")

        with pytest.warns(UserWarning, match="The base_env is not a gym env"):
            env = TransformedEnv(SerialEnv(2, make), transform=EndOfLifeTransform())
            check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_single_trans_env_check(self, eol_key, lives_key):
        with set_gym_backend("gymnasium"):
            env = TransformedEnv(
                GymEnv("ALE/Breakout-v5"),
                transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
            )
        check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_serial_trans_env_check(self, eol_key, lives_key):
        def make():
            with set_gym_backend("gymnasium"):
                return TransformedEnv(
                    GymEnv("ALE/Breakout-v5"),
                    transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
                )

        env = SerialEnv(2, make)
        check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_parallel_trans_env_check(self, eol_key, lives_key):
        def make():
            with set_gym_backend("gymnasium"):
                return TransformedEnv(
                    GymEnv("ALE/Breakout-v5"),
                    transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
                )

        env = ParallelEnv(2, make)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_transform_no_env(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=t.NO_PARENT_ERR.format(type(t))):
            t._step(TensorDict({}, []), TensorDict({}, []))

    def test_transform_compose(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=t.NO_PARENT_ERR.format(type(t))):
            Compose(t)._step(TensorDict({}, []), TensorDict({}, []))

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_transform_env(self, eol_key, lives_key):
        from tensordict.nn import TensorDictModule
        from torchrl.objectives import DQNLoss
        from torchrl.objectives.value import GAE

        with set_gym_backend("gymnasium"):
            env = TransformedEnv(
                GymEnv("ALE/Breakout-v5"),
                transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
            )
        check_env_specs(env)
        loss = DQNLoss(nn.Identity(), action_space="categorical")
        env.transform.register_keys(loss)
        assert ("next", eol_key) in loss.in_keys
        gae = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=TensorDictModule(nn.Identity(), ["x"], ["y"]),
        )
        env.transform.register_keys(gae)
        assert ("next", eol_key) in gae.in_keys

    def test_transform_model(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            nn.Sequential(t)(TensorDict({}, []))

    def test_transform_rb(self):
        pass

    def test_transform_inverse(self):
        pass


class TestBurnInTransform(TransformBase):
    def _make_gru_module(self, input_size=4, hidden_size=4, device="cpu"):
        return GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            in_keys=["observation", "rhs", "is_init"],
            out_keys=["output", ("next", "rhs")],
            device=device,
        ).set_recurrent_mode(True)

    def _make_lstm_module(self, input_size=4, hidden_size=4, device="cpu"):
        return LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            in_keys=["observation", "rhs_h", "rhs_c", "is_init"],
            out_keys=["output", ("next", "rhs_h"), ("next", "rhs_c")],
            device=device,
        ).set_recurrent_mode(True)

    def _make_batch(self, batch_size: int = 2, sequence_length: int = 5):
        observation = torch.randn(batch_size, sequence_length + 1, 4)
        is_init = torch.zeros(batch_size, sequence_length, 1, dtype=torch.bool)
        batch = TensorDict(
            {
                "observation": observation[:, :-1],
                "is_init": is_init,
                "next": TensorDict(
                    {
                        "observation": observation[:, 1:],
                    },
                    batch_size=[batch_size, sequence_length],
                ),
            },
            batch_size=[batch_size, sequence_length],
        )
        return batch

    def test_single_trans_env_check(self):
        module = self._make_gru_module()
        burn_in_transform = BurnInTransform(module, burn_in=2)
        with pytest.raises(
            RuntimeError,
            match="BurnInTransform can only be appended to a ReplayBuffer.",
        ):
            env = TransformedEnv(ContinuousActionVecMockEnv(), burn_in_transform)
            check_env_specs(env)
            env.close()

    def test_serial_trans_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_parallel_trans_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_trans_serial_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_trans_parallel_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_no_env(self, module, batch_size, sequence_length, burn_in):
        """tests the transform on dummy data, without an env."""
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        data = burn_in_transform(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_compose(self, module, batch_size, sequence_length, burn_in):
        """tests the transform on dummy data, without an env but inside a Compose."""
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_compose = Compose(BurnInTransform(module, burn_in=burn_in))
        data = burn_in_compose(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    def test_transform_env(self):
        module = self._make_gru_module()
        burn_in_transform = BurnInTransform(module, burn_in=2)
        env = TransformedEnv(ContinuousActionVecMockEnv(), burn_in_transform)
        with pytest.raises(
            RuntimeError,
            match="BurnInTransform can only be appended to a ReplayBuffer.",
        ):
            rollout = env.rollout(3)

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_model(self, module, batch_size, sequence_length, burn_in):
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        module = nn.Sequential(burn_in_transform, nn.Identity())
        data = module(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, module, batch_size, sequence_length, burn_in, rbclass):
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(burn_in_transform)
        rb.extend(data)
        batch = rb.sample(2)
        assert batch.shape[-1] == sequence_length - burn_in

        for key in batch.keys():
            if key.startswith("rhs"):
                assert batch[:, 0].get(key).abs().sum() > 0.0
                assert batch[:, 1:].get(key).sum() == 0.0

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for BurnInTransform")


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
                in_keys_inv=["input"],
                out_keys_inv=["input_sign"],
            )
        )
        rb.append_transform(t)
        data = TensorDict({"observation": 1, "reward": 2, "input": 3}, [])
        rb.add(data)
        sample = rb.sample(20)

        assert (sample["observation"] == 1).all()
        assert self.check_sign_applied(sample["obs_sign"])

        assert (sample["reward"] == 2).all()
        assert self.check_sign_applied(sample["reward_sign"])

        assert (sample["input"] == 3).all()
        assert self.check_sign_applied(sample["input_sign"])

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
            in_keys_inv=["observation", "reward"],
            out_keys_inv=["obs_sign", "reward_sign"],
        )
        data = TensorDict({"observation": 1, "reward": 2}, [])
        data = t.inv(data)
        assert data["observation"] == 1
        assert self.check_sign_applied(data["obs_sign"])
        assert data["reward"] == 2
        assert self.check_sign_applied(data["reward_sign"])

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

    def test_parallel_trans_env_check(self):
        def make_env():
            env = ContinuousActionVecMockEnv()
            return TransformedEnv(
                env,
                SignTransform(
                    in_keys=["observation", "reward"],
                    in_keys_inv=["observation_orig"],
                ),
            )

        env = ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            env.close()

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

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            SignTransform(
                in_keys=["observation", "reward"],
                in_keys_inv=["observation_orig"],
            ),
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

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
            env.close()


class TestRemoveEmptySpecs(TransformBase):
    class DummyEnv(EnvBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec((*self.batch_size, 3)),
                other=CompositeSpec(
                    another_other=CompositeSpec(shape=self.batch_size),
                    shape=self.batch_size,
                ),
                shape=self.batch_size,
            )
            self.action_spec = UnboundedContinuousTensorSpec((*self.batch_size, 3))
            self.done_spec = DiscreteTensorSpec(
                2, (*self.batch_size, 1), dtype=torch.bool
            )
            self.full_done_spec["truncated"] = self.full_done_spec["terminated"].clone()
            self.reward_spec = CompositeSpec(
                reward=UnboundedContinuousTensorSpec(*self.batch_size, 1),
                other_reward=CompositeSpec(shape=self.batch_size),
                shape=self.batch_size,
            )
            self.state_spec = CompositeSpec(
                state=CompositeSpec(
                    sub=CompositeSpec(shape=self.batch_size), shape=self.batch_size
                ),
                shape=self.batch_size,
            )

        def _reset(self, tensordict):
            return self.observation_spec.rand().update(self.full_done_spec.zero())

        def _step(self, tensordict):
            return (
                TensorDict({}, batch_size=[])
                .update(self.observation_spec.rand())
                .update(self.full_done_spec.zero())
                .update(self.full_reward_spec.rand())
            )

        def _set_seed(self, seed):
            return seed + 1

    def test_single_trans_env_check(self):
        env = TransformedEnv(self.DummyEnv(), RemoveEmptySpecs())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(2, lambda: TransformedEnv(self.DummyEnv(), RemoveEmptySpecs()))
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(
            2, lambda: TransformedEnv(self.DummyEnv(), RemoveEmptySpecs())
        )
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_trans_serial_env_check(self):
        with pytest.raises(
            RuntimeError, match="The environment passed to SerialEnv has empty specs"
        ):
            env = TransformedEnv(SerialEnv(2, self.DummyEnv), RemoveEmptySpecs())

    def test_trans_parallel_env_check(self):
        with pytest.raises(
            RuntimeError, match="The environment passed to ParallelEnv has empty specs"
        ):
            env = TransformedEnv(ParallelEnv(2, self.DummyEnv), RemoveEmptySpecs())

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
        td2 = env.transform.inv(TensorDict({}, []))
        assert ("state", "sub") in td2.keys(True)


class TestMultiStepTransform:
    def test_multistep_transform(self):
        env = TransformedEnv(
            SerialEnv(
                2, [lambda: CountingEnv(max_steps=4), lambda: CountingEnv(max_steps=10)]
            ),
            StepCounter(),
        )

        env.set_seed(0)
        torch.manual_seed(0)

        t = MultiStepTransform(3, 0.98)

        outs_2 = []
        td = env.reset().contiguous()
        for _ in range(1):
            rollout = env.rollout(
                250, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            out = t._inv_call(rollout)
            td = rollout[..., -1]
            outs_2.append(out)
        # This will break if we don't have the appropriate number of frames
        outs_2 = torch.cat(outs_2, -1).split([47, 50, 50, 50, 50], -1)

        t = MultiStepTransform(3, 0.98)

        env.set_seed(0)
        torch.manual_seed(0)

        outs = []
        td = env.reset().contiguous()
        for i in range(5):
            rollout = env.rollout(
                50, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            out = t._inv_call(rollout)
            # tests that the data is insensitive to the collection schedule
            assert_allclose_td(out, outs_2[i])
            td = rollout[..., -1]["next"].exclude("reward")
            outs.append(out)

        outs = torch.cat(outs, -1)

        # Test with a very tiny window and across the whole collection
        t = MultiStepTransform(3, 0.98)

        env.set_seed(0)
        torch.manual_seed(0)

        outs_3 = []
        td = env.reset().contiguous()
        for _ in range(125):
            rollout = env.rollout(
                2, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            out = t._inv_call(rollout)
            td = rollout[..., -1]["next"]
            if out is not None:
                outs_3.append(out)

        outs_3 = torch.cat(outs_3, -1)

        assert_allclose_td(outs, outs_3)

    def test_multistep_transform_changes(self):
        data = TensorDict(
            {
                "steps": torch.arange(100),
                "next": {
                    "steps": torch.arange(1, 101),
                    "reward": torch.ones(100, 1),
                    "done": torch.zeros(100, 1, dtype=torch.bool),
                    "terminated": torch.zeros(100, 1, dtype=torch.bool),
                    "truncated": torch.zeros(100, 1, dtype=torch.bool),
                },
            },
            batch_size=[100],
        )
        data_splits = data.split(10)
        t = MultiStepTransform(3, 0.98)
        rb = ReplayBuffer(storage=LazyTensorStorage(100), transform=t)
        for data in data_splits:
            rb.extend(data)
            t.n_steps = t.n_steps + 1
            assert (rb[:]["steps"] == torch.arange(len(rb))).all()
            assert rb[:]["next", "steps"][-1] == data["steps"][-1]
            assert t._buffer["steps"][-1] == data["steps"][-1]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
