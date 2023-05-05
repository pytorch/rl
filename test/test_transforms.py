# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import argparse
import itertools
from copy import copy, deepcopy
from functools import partial

import numpy as np
import pytest
import torch

from _utils_internal import (  # noqa
    dtype_fixture,
    get_available_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    retry,
)
from mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    DiscreteActionConvMockEnvNumpy,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
)
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import multiprocessing as mp, nn, Tensor
from torchrl._utils import prod
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    LazyTensorStorage,
    ReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DiscreteActionProjection,
    DoubleToFloat,
    EnvBase,
    EnvCreator,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    InitTracker,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    PinMemoryTransform,
    R3MTransform,
    RandomCropTensorDict,
    RenameTransform,
    Resize,
    Reward2GoTransform,
    RewardClipping,
    RewardScaling,
    RewardSum,
    SelectTransform,
    SerialEnv,
    SqueezeTransform,
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TimeMaxPool,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
    VIPTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import VecNorm
from torchrl.envs.transforms.r3m import _R3MNet
from torchrl.envs.transforms.transforms import _has_tv
from torchrl.envs.transforms.vip import _VIPNet, VIPRewardTransform
from torchrl.envs.utils import check_env_specs, step_mdp

TIMEOUT = 100.0


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

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(
            2, lambda: TransformedEnv(ContinuousActionVecMockEnv(), BinarizeReward())
        )
        check_env_specs(env)
        env.close()

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()), BinarizeReward()
        )
        check_env_specs(env)
        env.close()

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, lambda: ContinuousActionVecMockEnv()), BinarizeReward()
        )
        check_env_specs(env)
        env.close()

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    def test_transform_no_env(self, device, batch):
        torch.manual_seed(0)
        br = BinarizeReward()
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

    def test_transform_rb(self):
        device = "cpu"
        batch = [20]
        torch.manual_seed(0)
        br = BinarizeReward()
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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

    def test_parallel_trans_env_check(self):
        env = ParallelEnv(
            2,
            lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                CatFrames(dim=-1, N=3, in_keys=["observation"]),
            ),
        )
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: ContinuousActionVecMockEnv()),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, lambda: ContinuousActionVecMockEnv()),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED, frame_skip=4),
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
            GymEnv(PENDULUM_VERSIONED, frame_skip=4),
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        td = env.reset()
        td = env.rand_step(td)
        cloned = env.transform.clone()
        value_at_clone = td["next", "observation"].clone()
        for _ in range(10):
            td = env.rand_step(td)
        assert (td["next", "observation"] != value_at_clone).any()
        assert (
            td["next", "observation"] == env.transform._cat_buffers_observation
        ).all()
        assert (
            cloned._cat_buffers_observation == env.transform._cat_buffers_observation
        ).all()
        assert cloned is not env.transform

    @pytest.mark.parametrize("dim", [-2, -1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "zeros"])
    def test_transform_model(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=out_keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )
        envbase = ContinuousActionVecMockEnv()
        env = TransformedEnv(
            envbase,
            Compose(
                UnsqueezeTransform(dim, in_keys=keys, out_keys=out_keys), cat_frames
            ),
        )
        torch.manual_seed(10)
        env.set_seed(10)
        td = env.rollout(10)
        torch.manual_seed(10)
        envbase.set_seed(10)
        tdbase = envbase.rollout(10)

        model = nn.Sequential(cat_frames2, nn.Identity())
        model(tdbase)
        assert (td == tdbase).all()

    @pytest.mark.parametrize("dim", [-2, -1])
    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("padding", ["same", "zeros"])
    def test_transform_rb(self, dim, N, padding):
        # test equivalence between transforms within an env and within a rb
        key1 = "observation"
        keys = [key1]
        out_keys = ["out_" + key1]
        cat_frames = CatFrames(
            N=N, in_keys=out_keys, out_keys=out_keys, dim=dim, padding=padding
        )
        cat_frames2 = CatFrames(
            N=N,
            in_keys=keys + [("next", keys[0])],
            out_keys=out_keys + [("next", out_keys[0])],
            dim=dim,
            padding=padding,
        )

        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            Compose(
                UnsqueezeTransform(dim, in_keys=keys, out_keys=out_keys), cat_frames
            ),
        )
        td = env.rollout(10)

        rb = ReplayBuffer(storage=LazyTensorStorage(20))
        rb.append_transform(cat_frames2)
        rb.add(td.exclude(*out_keys, ("next", out_keys[0])))
        tdsample = rb.sample(1).squeeze(0).exclude("index")
        for key in td.keys(True, True):
            assert (tdsample[key] == td[key]).all(), key
        assert (tdsample["out_" + key1] == td["out_" + key1]).all()
        assert (tdsample["next", "out_" + key1] == td["next", "out_" + key1]).all()

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
                    result[key].space.maximum[i], observation_spec[key].space.maximum[0]
                )
                assert torch.equal(
                    result[key].space.minimum[i], observation_spec[key].space.minimum[0]
                )

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(1, 4))
    @pytest.mark.parametrize("dim", [-3, -2, 1])
    @pytest.mark.parametrize("N", [2, 4])
    def test_transform_no_env(self, device, d, batch_size, dim, N):
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        extra_d = (3,) * (-dim - 1)
        key1_tensor = torch.ones(*batch_size, d, *extra_d, device=device) * 2
        key2_tensor = torch.ones(*batch_size, d, *extra_d, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), batch_size, device=device)
        if dim > 0:
            with pytest.raises(
                ValueError, match="dim must be > 0 to accomodate for tensordict"
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

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("batch_size", [(), (1,), (1, 2)])
    @pytest.mark.parametrize("d", range(2, 3))
    @pytest.mark.parametrize(
        "dim",
        [
            -3,
        ],
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

    @pytest.mark.parametrize("device", get_available_devices())
    def test_catframes_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        cat_frames = CatFrames(N=N, in_keys=keys, dim=-3)

        cat_frames._call(td.clone())
        buffer = getattr(cat_frames, f"_cat_buffers_{key1}")

        tdc = td.clone()
        passed_back_td = cat_frames.reset(tdc)

        assert tdc is passed_back_td
        assert (buffer == 0).all()

        _ = cat_frames._call(tdc)
        assert (buffer != 0).all()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for CatFrames")


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("model", ["resnet18", "resnet34", "resnet50"])
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

    def test_transform_rb(self, model, device):
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
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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
        check_env_specs(transformed_env)

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
        check_env_specs(transformed_env)

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
        exp_keys = {"vec", "done", "pixels_orig"}
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == exp_keys, set(td.keys()) - exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                "next",
            }
        )
        if tensor_pixels_key:
            exp_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == exp_keys, set(td.keys(True)) - exp_keys
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
            exp_keys = {"pixels_orig", "done", "vec"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == exp_keys
        else:
            exp_keys = {"pixels_orig", "done", "vec", "vec2"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                "next",
            }
        )
        if not stack_images:
            exp_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == exp_keys, set(td.keys()) - exp_keys
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
        exp_keys = {"vec", "done", "pixels_orig"}
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key)
        assert set(td.keys(True)) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "action",
                ("next", "reward"),
                ("next", "done"),
                "next",
            }
        )
        assert set(td.keys(True)) == exp_keys, set(td.keys()) - exp_keys
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
            list(transformed_env.input_spec.keys())
            + list(transformed_env.observation_spec.keys())
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [("next", "reward"), ("next", "done"), "done", "next"]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


class TestStepCounter(TransformBase):
    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), StepCounter(10)
        )
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), StepCounter(10))
        check_env_specs(env)

    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Gym not found")
    def test_transform_env(self):
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), StepCounter(10))
        td = env.rollout(100, break_when_any_done=False)
        assert td["step_count"].max() == 9
        assert td.shape[-1] == 100

    def test_transform_rb(self):
        transform = StepCounter(10)
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
        td = TensorDict({"a": torch.randn(10)}, [10])
        rb.extend(td)
        rb.append_transform(transform)
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            rb.sample(5)

    @pytest.mark.parametrize("device", get_available_devices())
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
            td.set(("next", "done"), done)

        td = step_counter.reset(td)
        assert not torch.all(td.get("step_count"))
        i = 0
        while max_steps is None or i < max_steps:
            td = step_counter._step(td)
            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td = step_counter.reset(td)
        if reset_workers:
            assert torch.all(torch.masked_select(td.get("step_count"), _reset) == 0)
            assert torch.all(torch.masked_select(td.get("step_count"), ~_reset) == i)
        else:
            assert torch.all(td.get("step_count") == 0)

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

    @pytest.mark.parametrize("device", get_available_devices())
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
            _reset = torch.randn(batch, device=device) < 0
            td.set("_reset", _reset)
            td.set("done", _reset)
            td.set(("next", "done"), done)

        td = step_counter.reset(td)
        assert not torch.all(td.get("step_count"))
        i = 0
        while max_steps is None or i < max_steps:
            td = step_counter._step(td)
            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td = step_counter.reset(td)
        if reset_workers:
            assert torch.all(torch.masked_select(td.get("step_count"), _reset) == 0)
            assert torch.all(torch.masked_select(td.get("step_count"), ~_reset) == i)
        else:
            assert torch.all(td.get("step_count") == 0)

    def test_step_counter_observation_spec(self):
        transformed_env = TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(50))
        check_env_specs(transformed_env)
        transformed_env.close()


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
        check_env_specs(env)

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
        check_env_specs(env)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", "observation_other"],
            ["observation_pixels"],
        ],
    )
    def test_transform_no_env(self, keys, device):
        cattensors = CatTensors(in_keys=keys, out_key="observation_out", dim=-2)

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

        if len(keys) == 1:
            observation_spec = BoundedTensorSpec(0, 1, (1, 4, 32))
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, len(keys) * 4, 32])
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(0, 1, (1, 4, 32)) for key in keys}
            )
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec["observation_out"].shape == torch.Size(
                [1, len(keys) * 4, 32]
            )

    @pytest.mark.parametrize("device", get_available_devices())
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
    def test_transform_env(self, del_keys):
        ct = CatTensors(
            in_keys=[
                "observation",
            ],
            out_key="observation_out",
            dim=-1,
            del_keys=del_keys,
        )
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), ct)
        assert env.observation_spec["observation_out"]
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

    def test_transform_rb(self):
        ct = CatTensors(
            in_keys=[("next", "observation"), "action"],
            out_key="observation_out",
            dim=-1,
            del_keys=True,
        )
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
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
    def test_transform_rb(
        self,
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
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(SerialEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        keys = ["pixels"]
        ct = Compose(ToTensorImage(), CenterCrop(w=20, h=20, in_keys=keys))
        env = TransformedEnv(ParallelEnv(2, DiscreteActionConvMockEnvNumpy), ct)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="No Gym detected")
    @pytest.mark.parametrize("out_key", [None, ["outkey"]])
    def test_transform_env(self, out_key):
        keys = ["pixels"]
        ct = Compose(
            ToTensorImage(), CenterCrop(out_keys=out_key, w=20, h=20, in_keys=keys)
        )
        env = TransformedEnv(GymEnv(PONG_VERSIONED), ct)
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
        check_env_specs(env)

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
        check_env_specs(env)

    def test_transform_no_env(self):
        t = DiscreteActionProjection(7, 10)
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        assert td["action"].shape[-1] == 10
        assert (td["action"].sum(-1) == 1).all()
        out = t.inv(td)
        assert out["action"].shape[-1] == 7
        assert (out["action"].sum(-1) == 1).all()

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

    @pytest.mark.parametrize("include_forward", [True, False])
    def test_transform_rb(self, include_forward):
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        t = DiscreteActionProjection(7, 10, include_forward=include_forward)
        rb.append_transform(t)
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        rb.extend(td)
        assert rb._storage._storage["action"].shape[-1] == 7
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
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", "some_other_key"],
            ["observation_pixels"],
            ["action"],
        ],
    )
    @pytest.mark.parametrize(
        "keys_inv",
        [
            ["action", "some_other_key"],
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
            input_spec = CompositeSpec(action=action_spec)
            action_spec = double2float.transform_input_spec(input_spec)
            assert action_spec.dtype == torch.float

        elif len(keys) == 1:
            observation_spec = BoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
            observation_spec = double2float.transform_observation_spec(observation_spec)
            assert observation_spec.dtype == torch.float

        else:
            observation_spec = CompositeSpec(
                {
                    key: BoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
                    for key in keys
                }
            )
            observation_spec = double2float.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].dtype == torch.float

    def test_single_trans_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self, dtype_fixture):  # noqa: F811
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, dtype_fixture):  # noqa: F811
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
            )

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

    def test_trans_serial_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, dtype_fixture):  # noqa: F811
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            DoubleToFloat(in_keys=["observation"], in_keys_inv=["action"]),
        )
        check_env_specs(env)

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

    def test_transform_rb(
        self,
    ):
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        # observation is not part of in_keys_inv
        assert rb._storage._storage["observation"].dtype is torch.double
        # action is part of in_keys_inv
        assert rb._storage._storage["action"].dtype is torch.double
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
            self.input_spec = CompositeSpec(action=UnboundedContinuousTensorSpec(2))

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
        check_env_specs(env)

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
        check_env_specs(env)

    def test_transform_env(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, ExcludeTransform("a"))
        assert "a" not in env.reset().keys()
        assert "b" in env.reset().keys()
        assert "c" in env.reset().keys()

    def test_transform_no_env(self):
        t = ExcludeTransform("a")
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

    def test_transform_rb(self):
        t = ExcludeTransform("a")
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
            self.input_spec = CompositeSpec(action=UnboundedContinuousTensorSpec(2))

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
        check_env_specs(env)

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
        check_env_specs(env)

    def test_transform_env(self):
        base_env = TestExcludeTransform.EnvWithManyKeys()
        env = TransformedEnv(base_env, SelectTransform("b", "c"))
        assert "a" not in env.reset().keys()
        assert "b" in env.reset().keys()
        assert "c" in env.reset().keys()

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

    def test_transform_rb(self):
        t = SelectTransform("b", "c")
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_env(self, out_keys):
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED), FlattenObservation(-3, -1, out_keys=out_keys)
        )
        check_env_specs(env)
        if out_keys:
            assert out_keys[0] in env.reset().keys()
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
    def test_transform_rb(self, out_keys):
        t = FlattenObservation(-3, -1, out_keys=out_keys)
        td = TensorDict({"pixels": torch.randint(255, (10, 10, 3))}, []).expand(10)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        check_env_specs(env)

    def test_transform_no_env(self):
        t = FrameSkipTransform(2)
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict)

    def test_transform_compose(self):
        t = Compose(FrameSkipTransform(2))
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict)

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
        base_env = GymEnv(PENDULUM_VERSIONED, frame_skip=skip)
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), fs)
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

    def test_transform_model(self):
        t = FrameSkipTransform(2)
        t = nn.Sequential(t, nn.Identity())
        tensordict = TensorDict({}, [])
        with pytest.raises(
            RuntimeError,
            match="FrameSkipTransform can only be used when appended to a transformed env",
        ):
            t(tensordict)

    def test_transform_rb(self):
        t = FrameSkipTransform(2)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        base_env = GymEnv(PENDULUM_VERSIONED)
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), fs)
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
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
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
        check_env_specs(env)

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
        check_env_specs(env)

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

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys):
        td = TensorDict({"pixels": torch.rand(3, 12, 12)}, []).expand(3)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), NoopResetEnv())
        with pytest.raises(
            ValueError,
            match="there is more than one done state in the parent environment",
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
            t.reset(TensorDict({"next": {}}, []))
        t._step(TensorDict({"next": {}}, []))

    def test_transform_compose(self):
        t = Compose(NoopResetEnv())
        with pytest.raises(
            RuntimeError,
            match="NoopResetEnv.parent not found. Make sure that the parent is set.",
        ):
            t.reset(TensorDict({"next": {}}, []))
        t._step(TensorDict({"next": {}}, []))

    def test_transform_model(self):
        t = nn.Sequential(NoopResetEnv(), nn.Identity())
        td = TensorDict({}, [])
        t(td)

    def test_transform_rb(self):
        t = NoopResetEnv()
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({}, [10])
        rb.extend(td)
        rb.sample(1)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for NoopResetEnv")

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
    def test_noop_reset_env_error(self, random, device, compose):
        torch.manual_seed(0)
        env = SerialEnv(2, lambda: ContinuousActionVecMockEnv())
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        transformed_env = TransformedEnv(env)
        transformed_env.append_transform(noop_reset_env)
        with pytest.raises(
            ValueError,
            match="there is more than one done state in the parent environment",
        ):
            transformed_env.reset()


class TestObservationNorm(TransformBase):
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
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
        check_env_specs(env)

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
        check_env_specs(env)

    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_no_env(self, out_keys, standard_normal):
        t = ObservationNorm(in_keys=["observation"], out_keys=out_keys)
        # test that init fails
        with pytest.raises(
            RuntimeError,
            match="Cannot initialize the transform if parent env is not defined",
        ):
            t.init_stats(num_iter=5)
        t = ObservationNorm(
            loc=torch.ones(7),
            scale=0.5,
            in_keys=["observation"],
            out_keys=out_keys,
            standard_normal=standard_normal,
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

    def test_transform_rb(self):
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
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
    def test_transform_inverse(self):
        standard_normal = True
        out_keys = ["observation_out"]
        in_keys_inv = ["action"]
        out_keys_inv = ["action_inv"]
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
        base_env = GymEnv(PENDULUM_VERSIONED)
        env = TransformedEnv(base_env, t)
        td = env.rollout(3)
        check_env_specs(env)
        env.set_seed(0)
        # assert "observation_inv" in env.input_spec.keys()
        # "observation_inv" should not appear in the tensordict
        assert torch.allclose(td["action"] * 0.5 + 1, t.inv(td)["action_inv"])
        assert torch.allclose((td["observation"] - 1) / 0.5, td["observation_out"])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
                    assert (observation_spec[key].space.minimum == -loc / scale).all()
                    assert (
                        observation_spec[key].space.maximum == (1 - loc) / scale
                    ).all()
                else:
                    assert (observation_spec[key].space.minimum == loc).all()
                    assert (observation_spec[key].space.maximum == scale + loc).all()

    @pytest.mark.parametrize("keys", [["observation"], ["observation", "next_pixel"]])
    @pytest.mark.parametrize("size", [1, 3])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("standard_normal", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_observationnorm_init_stats(
        self, keys, size, device, standard_normal, parallel
    ):
        def make_env():
            base_env = ContinuousActionVecMockEnv(
                observation_spec=CompositeSpec(
                    observation=BoundedTensorSpec(
                        minimum=1, maximum=1, shape=torch.Size([size])
                    ),
                    observation_orig=BoundedTensorSpec(
                        minimum=1, maximum=1, shape=torch.Size([size])
                    ),
                ),
                action_spec=BoundedTensorSpec(
                    minimum=1, maximum=1, shape=torch.Size((size,))
                ),
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
    @pytest.mark.parametrize("device", get_available_devices())
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
        transform = ObservationNorm(in_keys="next_observation", loc=0, scale=1)

        with pytest.raises(RuntimeError, match="Loc/Scale are already initialized"):
            transform.init_stats(num_iter=11)

    def test_observationnorm_wrong_catdim(self):
        transform = ObservationNorm(in_keys="next_observation", loc=0, scale=1)

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
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
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
        check_env_specs(env)

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
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="No gym")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PONG_VERSIONED),
            Compose(ToTensorImage(), Resize(20, 21, in_keys=["pixels"])),
        )
        check_env_specs(env)
        td = env.rollout(3)
        assert td["pixels"].shape[-3:] == torch.Size([3, 20, 21])

    def test_transform_model(self):
        module = nn.Sequential(Resize(20, 21, in_keys=["pixels"]), nn.Identity())
        td = TensorDict({"pixels": torch.randn(3, 32, 32)}, [])
        module(td)
        assert td["pixels"].shape == torch.Size([3, 20, 21])

    def test_transform_rb(self):
        t = Resize(20, 21, in_keys=["pixels"])
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), RewardClipping(-0.1, 0.1)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), RewardClipping(-0.1, 0.1)
        )
        check_env_specs(env)

    def test_transform_no_env(self):
        t = RewardClipping(-0.1, 0.1)
        td = TensorDict({"reward": torch.randn(10)}, [])
        t._call(td)
        assert (td["reward"] <= 0.1).all()
        assert (td["reward"] >= -0.1).all()

    def test_transform_compose(self):
        t = Compose(RewardClipping(-0.1, 0.1))
        td = TensorDict({"reward": torch.randn(10)}, [])
        t._call(td)
        assert (td["reward"] <= 0.1).all()
        assert (td["reward"] >= -0.1).all()

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    def test_transform_env(self):
        t = Compose(RewardClipping(-0.1, 0.1))
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), t)
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

    def test_transform_rb(self):
        t = RewardClipping(-0.1, 0.1)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
    @pytest.mark.parametrize("keys", [None, ["reward_1"]])
    @pytest.mark.parametrize("device", get_available_devices())
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), RewardScaling(0.5, 1.5)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), RewardScaling(0.5, 1.5)
        )
        check_env_specs(env)

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
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), t)
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

    @pytest.mark.parametrize("standard_normal", [True, False])
    def test_transform_rb(self, standard_normal):
        loc = 0.5
        scale = 1.5
        t = RewardScaling(0.5, 1.5, standard_normal=standard_normal)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        env = TransformedEnv(ContinuousActionVecMockEnv(), RewardSum())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), RewardSum())

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), RewardSum())

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), RewardSum())
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(ParallelEnv(2, ContinuousActionVecMockEnv), RewardSum())
        check_env_specs(env)

    def test_transform_no_env(
        self,
    ):
        t = RewardSum()
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        with pytest.raises(NotImplementedError):
            t(td)

    def test_transform_compose(
        self,
    ):
        t = Compose(RewardSum())
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        with pytest.raises(NotImplementedError):
            t(td)

    @pytest.mark.skipif(not _has_gym, reason="No Gym")
    def test_transform_env(
        self,
    ):
        t = Compose(RewardSum())
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED), t)
        env.set_seed(0)
        torch.manual_seed(0)
        td = env.rollout(3)
        env.set_seed(0)
        torch.manual_seed(0)
        td_base = env.base_env.rollout(3)
        reward = td_base["next", "reward"]
        final_reward = td_base["next", "reward"].sum(-2)
        assert torch.allclose(td["next", "reward"], reward)
        assert torch.allclose(td["next", "episode_reward"][..., -1, :], final_reward)

    def test_transform_model(
        self,
    ):
        t = RewardSum()
        model = nn.Sequential(t, nn.Identity())
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        with pytest.raises(NotImplementedError):
            model(td)

    def test_transform_rb(
        self,
    ):
        t = RewardSum()
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, []).expand(10)
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(NotImplementedError):
            _ = rb.sample(2)

    @pytest.mark.parametrize(
        "keys",
        [["done", "reward"]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
        td = rs._step(td)
        td_next = td["next"]
        assert "episode_reward" in td.keys()
        assert (td_next.get("episode_reward") == td_next.get("reward")).all()

        # apply a second time, episode_reward should twice the reward
        td["episode_reward"] = td["next", "episode_reward"]
        td = rs._step(td)
        td_next = td["next"]
        assert (td_next.get("episode_reward") == 2 * td_next.get("reward")).all()

        # reset environments
        td.set("_reset", torch.ones(batch, dtype=torch.bool, device=device))
        rs.reset(td)

        # apply a third time, episode_reward should be equal to reward again
        td = rs._step(td)
        td_next = td["next"]
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


class TestReward2Go(TransformBase):
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.99, 1.0])
    @pytest.mark.parametrize("done_flags", [1, 5])
    @pytest.mark.parametrize("t", [3, 20])
    def test_transform_rb(self, done_flags, gamma, t, device):
        batch = 10
        batch_size = [batch, t]
        torch.manual_seed(0)
        out_key = "reward2go"
        r2g = Reward2GoTransform(gamma=gamma, out_keys=[out_key])
        rb = ReplayBuffer(storage=LazyTensorStorage(batch), transform=r2g)
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

    @pytest.mark.parametrize("device", get_available_devices())
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
        t = Compose(t)
        env = TransformedEnv(CountingBatchedEnv())
        env.append_transform(t)

        env.set_seed(0)
        torch.manual_seed(0)
        with pytest.raises(ValueError, match=Reward2GoTransform.ENV_ERR):
            env.rollout(3)

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
    def test_transform_no_env(self, gamma, done_flags):
        device = "cpu"
        torch.manual_seed(0)
        batch = 10
        t = 20
        out_key = "reward2go"
        batch_size = [batch, t]
        torch.manual_seed(0)
        r2g = Reward2GoTransform(gamma=gamma, out_keys=[out_key])
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
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys_inv", [[], ["action", "some_other_key"], ["observation_pixels"]]
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
        check_env_specs(env)

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
        check_env_specs(env)

    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
    def test_transform_rb(self, out_keys, unsqueeze_dim):
        t = UnsqueezeTransform(
            unsqueeze_dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
            GymEnv(HALFCHEETAH_VERSIONED),
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
        # inverse transforms are now hidden from outer scope
        # assert env.input_spec["action_t"].shape[-1] == 1
        # assert td["action_t"].shape[-1] == 1


class TestSqueezeTransform(TransformBase):
    @pytest.mark.parametrize("squeeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys_inv", [[], ["action", "some_other_key"], [("next", "observation_pixels")]]
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
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys_inv", [[], ["action", "some_other_key"], ["observation_pixels"]]
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), self._circular_transform
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv), self._circular_transform
        )
        check_env_specs(env)

    @pytest.mark.parametrize("squeeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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

    def test_transform_env(self):
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
    def test_transform_rb(self, out_keys):
        squeeze_dim = -2
        t = SqueezeTransform(
            squeeze_dim,
            in_keys=["observation"],
            out_keys=out_keys,
            allow_positive_dim=True,
        )
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
            GymEnv(HALFCHEETAH_VERSIONED), self._inv_circular_transform
        )
        check_env_specs(env)
        r = env.rollout(3)
        r2 = GymEnv(HALFCHEETAH_VERSIONED).rollout(3)
        assert (r.zero_() == r2.zero_()).all()


class TestTargetReturn(TransformBase):
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
    def test_transform_compose(self, batch, mode, device):
        torch.manual_seed(0)
        t = Compose(TargetReturn(target_return=10.0, mode=mode))
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
        td = t.reset(td)
        td = t._step(td)

        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()

        else:
            assert (td["next", "target_return"] == 10.0).all()

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_single_trans_env_check(self, mode, device):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
    def test_parallel_trans_env_check(self, mode, device):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TargetReturn(target_return=10.0, mode=mode).to(device),
                device=device,
            )

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_trans_serial_env_check(self, mode, device):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_trans_parallel_env_check(self, mode, device):
        env = TransformedEnv(
            ParallelEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode),
            device=device,
        )
        check_env_specs(env)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TargetReturn")

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    def test_transform_no_env(self, mode):
        t = TargetReturn(target_return=10.0, mode=mode)
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        td = t.reset(td)
        td = t._step(td)
        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()
        else:
            assert (td["next", "target_return"] == 10.0).all()

    def test_transform_model(
        self,
    ):
        t = TargetReturn(target_return=10.0)
        model = nn.Sequential(t, nn.Identity())
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, [])
        with pytest.raises(
            NotImplementedError, match="cannot be executed without a parent"
        ):
            model(td)

    def test_transform_rb(
        self,
    ):
        t = TargetReturn(target_return=10.0)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        reward = torch.randn(10)
        td = TensorDict({("next", "reward"): reward}, []).expand(10)
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
    @pytest.mark.parametrize("device", get_available_devices())
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
            observation_spec = BoundedTensorSpec(0, 255, (16, 16, 3))
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(0, 255, (16, 16, 3)) for key in keys}
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.minimum == 0).all()
                assert (observation_spec[key].space.maximum == 1).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
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
            observation_spec = BoundedTensorSpec(0, 255, (16, 16, 3))
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                {key: BoundedTensorSpec(0, 255, (16, 16, 3)) for key in keys}
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 16, 16])
                assert (observation_spec[key].space.minimum == 0).all()
                assert (observation_spec[key].space.maximum == 1).all()

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
        check_env_specs(env)

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
        check_env_specs(env)

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
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
            assert out_keys[0] in r.keys()
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

    @pytest.mark.parametrize("out_keys", [None, ["stuff"]])
    def test_transform_rb(self, out_keys):
        t = ToTensorImage(in_keys=["pixels"], out_keys=out_keys)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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

    def test_transform_rb(self):
        batch_size = (2,)
        t = TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([*batch_size, 3]))
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        assert ("next", "mykey") in env.rollout(3).keys(True)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([3])),
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        assert ("next", "mykey") in env.rollout(3).keys(True)

    def test_trans_parallel_env_check(self):
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            TensorDictPrimer(mykey=UnboundedContinuousTensorSpec([2, 4])),
        )
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        r = env.rollout(3)
        assert ("next", "mykey") in r.keys(True)
        assert r["next", "mykey"].shape == torch.Size([2, 3, 4])

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
    @pytest.mark.parametrize("device", get_available_devices())
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
                key: deepcopy(spec) if key != "action" else deepcopy(env.action_spec)
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


class TestTimeMaxPool(TransformBase):
    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_transform_no_env(self, T, seq_len, device):
        batch = 1
        nodes = 4
        keys = ["observation"]
        time_max_pool = TimeMaxPool(keys, T=T)

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

    @pytest.mark.parametrize("T", [2, 4])
    @pytest.mark.parametrize("seq_len", [8])
    @pytest.mark.parametrize("device", get_available_devices())
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
        check_env_specs(env)

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
            CatFrames(dim=-1, N=3, in_keys=["observation"]),
        )
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    def test_transform_env(self):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED, frame_skip=4),
            TimeMaxPool(
                in_keys=["observation"],
                T=3,
            ),
        )
        td = env.reset()
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

    def test_transform_rb(self):
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
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="TimeMaxPool cannot be called independently"
        ):
            _ = rb.sample(10)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tmp_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.randn(1, 1, 3, 3, device=device)
        key2_tensor = torch.randn(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        t = TimeMaxPool(
            in_keys=key1,
            T=3,
        )

        t._call(td.clone())
        buffer = getattr(t, f"_maxpool_buffer_{key1}")

        tdc = td.clone()
        passed_back_td = t.reset(tdc)

        assert tdc is passed_back_td
        assert (buffer == 0).all()

        _ = t._call(tdc)
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
        check_env_specs(env)

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
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        state_dim = 7
        action_dim = 7
        env = TransformedEnv(
            ParallelEnv(2, ContinuousActionVecMockEnv),
            gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=(2,)),
        )
        check_env_specs(env)

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

    def test_transform_rb(self):
        state_dim = 7
        action_dim = 5
        batch_size = (2,)
        t = gSDENoise(state_dim=state_dim, action_dim=action_dim, shape=batch_size)
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
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
            GymEnv(PENDULUM_VERSIONED), gSDENoise(state_dim=3, action_dim=1)
        )
        check_env_specs(env)
        assert (env.reset()["_eps_gSDE"] != 0.0).all()


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.parametrize("device", get_available_devices())
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
        r3m = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        transformed_env = TransformedEnv(
            ParallelEnv(2, lambda: DiscreteActionConvMockEnvNumpy().to(device)), r3m
        )
        check_env_specs(transformed_env)

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
        check_env_specs(transformed_env)

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

    def test_transform_rb(self, model, device):
        in_keys = ["pixels"]
        tensor_pixels_key = None
        out_keys = ["vec"]
        vip = VIPTransform(
            model,
            in_keys=in_keys,
            out_keys=out_keys,
            tensor_pixels_keys=tensor_pixels_key,
        )
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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
        exp_keys = {"vec", "done", "pixels_orig"}
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key[0])
        assert set(td.keys()) == exp_keys, set(td.keys()) - exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
            }
        )
        if tensor_pixels_key:
            exp_keys.add(("next", tensor_pixels_key[0]))
        assert set(td.keys(True)) == exp_keys, set(td.keys(True)) - exp_keys
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
            exp_keys = {"pixels_orig", "done", "vec"}
            # assert td["vec"].shape[0] == 2
            assert td["vec"].ndimension() == 1 + parallel
            assert set(td.keys()) == exp_keys
        else:
            exp_keys = {"pixels_orig", "done", "vec", "vec2"}
            assert td["vec"].shape[0 + parallel] != 2
            assert td["vec"].ndimension() == 1 + parallel
            assert td["vec2"].shape[0 + parallel] != 2
            assert td["vec2"].ndimension() == 1 + parallel
            assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
            }
        )
        if not stack_images:
            exp_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == exp_keys, set(td.keys(True)) - exp_keys
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
        exp_keys = {"vec", "done", "pixels_orig"}
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key)
        assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
            }
        )
        assert set(td.keys(True)) == exp_keys, set(td.keys(True)) - exp_keys
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
            _ = transformed_env.reset(tensordict_reset.select())

        td = transformed_env.reset(tensordict_reset)
        assert td.device == device
        assert td.batch_size == torch.Size([4])
        exp_keys = {
            "vec",
            "done",
            "pixels_orig",
            "goal_embedding",
            "goal_image",
        }
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key)
        assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {
                ("next", "vec"),
                ("next", "pixels_orig"),
                "next",
                "action",
                ("next", "reward"),
                ("next", "done"),
            }
        )
        assert set(td.keys(True)) == exp_keys, td

        torch.manual_seed(1)
        tensordict_reset = TensorDict(
            {"goal_image": torch.randint(0, 255, (4, 7, 7, 3), dtype=torch.uint8)},
            [4],
            device=device,
        )
        td = transformed_env.rollout(
            5, auto_reset=False, tensordict=transformed_env.reset(tensordict_reset)
        )
        assert set(td.keys(True)) == exp_keys, td
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

        explicit_reward = -torch.norm(cur_embedding - goal_embedding, dim=-1) - (
            -torch.norm(last_embedding - goal_embedding, dim=-1)
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
            list(transformed_env.input_spec.keys())
            + list(transformed_env.observation_spec.keys())
            + [("next", key) for key in transformed_env.observation_spec.keys()]
            + [("next", "reward"), ("next", "done"), "done", "next"]
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
                lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED), VecNorm(decay=1.0))
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
        queue_out.put("first round")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
        queue_out.put("second round")
        parallel_env.close()
        queue_out.close()
        queue_in.close()
        del parallel_env, queue_out, queue_in

    def test_parallelenv_vecnorm(self):
        if _has_gym:
            make_env = EnvCreator(
                lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED), VecNorm())
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
            env = GymEnv(PENDULUM_VERSIONED)
        elif parallel:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED)
            )
        else:
            env = SerialEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED)
            )

        env.set_seed(self.SEED)
        t = VecNorm(decay=1.0)
        env_t = TransformedEnv(env, t)
        td = env_t.reset()
        tds = []
        for _ in range(N):
            td = env_t.rand_step(td)
            tds.append(td.clone())
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
            observation=BoundedTensorSpec(minimum=0, maximum=10, shape=torch.Size((1,)))
        )
        base_env = ContinuousActionVecMockEnv(observation_spec=obs_spec)
        t1 = TransformedEnv(base_env, transform=ObservationNorm(loc=3, scale=2))
        t2 = TransformedEnv(base_env, transform=ObservationNorm(loc=1, scale=6))

        t1_obs_spec = t1.observation_spec
        t2_obs_spec = t2.observation_spec

        assert t1_obs_spec["observation"].space.minimum == 3
        assert t1_obs_spec["observation"].space.maximum == 23

        assert t2_obs_spec["observation"].space.minimum == 1
        assert t2_obs_spec["observation"].space.maximum == 61

        assert base_env.observation_spec["observation"].space.minimum == 0
        assert base_env.observation_spec["observation"].space.maximum == 10

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
    @pytest.mark.parametrize("device", get_available_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(
            in_keys=keys,
            N=4,
            dim=-3,
        )
        t2 = FiniteTensorDictCheck()
        t3 = StepCounter()
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
                ValueError, match="The last dimension of the tensordict"
            ):
                compose(td.clone(False))
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            compose[1:](td.clone(False))
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

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys_inv_1",
        [
            ["action_1"],
            [],
        ],
    )
    @pytest.mark.parametrize(
        "keys_inv_2",
        [
            ["action_2"],
            [],
        ],
    )
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

    @pytest.mark.parametrize("device", get_available_devices())
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
    @pytest.mark.parametrize("device", get_available_devices())
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
        assert "action" in env._input_spec
        assert env._input_spec["action"] is not None
        assert env._output_spec["observation"] is not None
        assert env._output_spec["reward"] is not None

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

        try:
            env.insert_transform(-7, FiniteTensorDictCheck())
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._input_spec is not None
            assert "action" in env._input_spec
            assert env._input_spec["action"] is not None
            assert env._output_spec["observation"] is not None
            assert env._output_spec["reward"] is not None

        try:
            env.insert_transform(7, FiniteTensorDictCheck())
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._input_spec is not None
            assert "action" in env._input_spec
            assert env._input_spec["action"] is not None
            assert env._output_spec["observation"] is not None
            assert env._output_spec["reward"] is not None

        try:
            env.insert_transform(4, "ffff")
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._input_spec is not None
            assert "action" in env._input_spec
            assert env._input_spec["action"] is not None
            assert env._output_spec["observation"] is not None
            assert env._output_spec["reward"] is not None


@pytest.mark.parametrize("device", get_available_devices())
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
        RuntimeError, match="Expected a tensordict with shape==env.shape, "
    ):
        env.step(td_expanded)


@pytest.mark.parametrize("device", get_available_devices())
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


@pytest.mark.parametrize("device", get_available_devices())
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
        RuntimeError, match="Expected a tensordict with shape==env.shape, "
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
    ObservationNorm,
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
            [
                "stuff",
            ],
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
                    [
                        "observation",
                    ],
                    [
                        "stuff",
                    ],
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
                    [
                        "stuff",
                    ],
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
                    [
                        "observation",
                    ],
                    [
                        "stuff",
                    ],
                    create_copy=create_copy,
                ),
            )

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                RenameTransform(
                    ["observation_orig"],
                    ["stuff"],
                    ["observation_orig"],
                    [
                        "stuff",
                    ],
                    create_copy=create_copy,
                ),
            )

        env = ParallelEnv(2, make_env)
        check_env_specs(env)

    def test_trans_serial_env_check(self, create_copy):
        def make_env():
            return ContinuousActionVecMockEnv()

        env = TransformedEnv(
            SerialEnv(2, make_env),
            RenameTransform(
                [
                    "observation",
                ],
                [
                    "stuff",
                ],
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
                [
                    "stuff",
                ],
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
                [
                    "observation",
                ],
                [
                    "stuff",
                ],
                create_copy=create_copy,
            ),
        )
        check_env_specs(env)
        env = TransformedEnv(
            ParallelEnv(2, make_env),
            RenameTransform(
                ["observation_orig"],
                ["stuff"],
                ["observation_orig"],
                [
                    "stuff",
                ],
                create_copy=create_copy,
            ),
        )
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["forward", "_call"])
    def test_transform_no_env(self, create_copy, mode):
        t = RenameTransform(["a"], ["b"], create_copy=create_copy)
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
                [
                    "observation",
                ],
                [
                    "stuff",
                ],
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
                [
                    "stuff",
                ],
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
    def test_transform_rb(self, create_copy, inverse):
        if not inverse:
            t = RenameTransform(["a"], ["b"], create_copy=create_copy)
            tensordict = TensorDict({"a": torch.randn(())}, []).expand(10)
        else:
            t = RenameTransform(["a"], ["b"], ["a"], ["b"], create_copy=create_copy)
            tensordict = TensorDict({"b": torch.randn(())}, []).expand(10)
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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
        check_env_specs(env)

    def test_trans_serial_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = SerialEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        check_env_specs(env)

    def test_trans_parallel_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = ParallelEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        check_env_specs(env)

    def test_transform_no_env(self):
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

    def test_transform_rb(self):
        batch = [1]
        device = "cpu"
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
