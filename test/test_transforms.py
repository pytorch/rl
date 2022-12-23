# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
    PENDULUM_VERSIONED,
    retry,
)
from mocking_classes import (
    ContinuousActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
)
from tensordict import TensorDict
from torch import multiprocessing as mp, Tensor
from torchrl._utils import prod
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    Compose,
    DoubleToFloat,
    EnvCreator,
    FiniteTensorDictCheck,
    FlattenObservation,
    GrayScale,
    ObservationNorm,
    ParallelEnv,
    R3MTransform,
    Resize,
    RewardClipping,
    RewardScaling,
    SerialEnv,
    ToTensorImage,
    VIPTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.envs.transforms.r3m import _R3MNet
from torchrl.envs.transforms.transforms import (
    _has_tv,
    CenterCrop,
    DiscreteActionProjection,
    FrameSkipTransform,
    gSDENoise,
    NoopResetEnv,
    PinMemoryTransform,
    SqueezeTransform,
    TensorDictPrimer,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.vip import _VIPNet, VIPRewardTransform

TIMEOUT = 10.0


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
            env_input_keys = None
        else:
            make_env = EnvCreator(
                lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm())
            )
            env_input_keys = ["action", ContinuousActionVecMockEnv._out_key]
        parallel_env = ParallelEnv(3, make_env, env_input_keys=env_input_keys)
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
        obs_spec = CompositeSpec(observation=BoundedTensorSpec(minimum=0, maximum=10))
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
    assert children[0] == t1
    assert children[1] == t2


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
    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_resize(self, interpolation, keys, nchannels, batch, device):
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
            observation_spec = NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = resize.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("h", [None, 21])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_centercrop(self, keys, h, nchannels, batch, device):
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
            observation_spec = NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = cc.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, h])
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = cc.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, h])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_flatten(self, keys, size, nchannels, batch, device):
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
            observation_spec = NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = flatten.transform_observation_spec(observation_spec)
            assert observation_spec.shape[-3] == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = flatten.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    @pytest.mark.parametrize("skip", [-1, 1, 2, 3])
    def test_frame_skip_transform_builtin(self, skip):
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
                td1 = base_env.step(tensordicts[i].clone()).flatten_keys()
                r = td1.get("reward") + r
            td1.set("reward", r)
            td2 = env.step(tensordicts[i].clone()).flatten_keys()
            for key in td1.keys():
                torch.testing.assert_close(td1[key], td2[key])

    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["observation", "some_other_key"], ["observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_unsqueeze(self, keys, size, nchannels, batch, device, unsqueeze_dim):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = UnsqueezeTransform(unsqueeze_dim, in_keys=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        unsqueeze(td)
        expected_size = [*size, nchannels, 16, 16]
        if unsqueeze_dim < 0:
            expected_size.insert(len(expected_size) + unsqueeze_dim + 1, 1)
        else:
            expected_size.insert(unsqueeze_dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys:
            assert td.get(key).shape[len(batch) :] == expected_size, (
                batch,
                size,
                nchannels,
                unsqueeze_dim,
            )
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
            assert observation_spec.shape == expected_size
        else:
            observation_spec = CompositeSpec(
                {
                    key: NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
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
            unsqueeze_dim, in_keys=keys, in_keys_inv=keys_inv
        )
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )

        unsqueeze.inv(td)

        expected_size = [*size, nchannels, 16, 16]
        for key in keys_total.difference(keys_inv):
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

        if expected_size[unsqueeze_dim] == 1:
            del expected_size[unsqueeze_dim]
        for key in keys_inv:
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

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
    def test_squeeze(self, keys, keys_inv, size, nchannels, batch, device, squeeze_dim):
        torch.manual_seed(0)
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(squeeze_dim, in_keys=keys, in_keys_inv=keys_inv)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )
        squeeze(td)

        expected_size = [*size, nchannels, 16, 16]
        for key in keys_total.difference(keys):
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

        if expected_size[squeeze_dim] == 1:
            del expected_size[squeeze_dim]
        for key in keys:
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

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
        keys_total = set(keys + keys_inv)
        squeeze = SqueezeTransform(squeeze_dim, in_keys=keys, in_keys_inv=keys_inv)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys_total
            },
            batch,
        )
        squeeze.inv(td)

        expected_size = [*size, nchannels, 16, 16]
        for key in keys_total.difference(keys_inv):
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

        if squeeze_dim < 0:
            expected_size.insert(len(expected_size) + squeeze_dim + 1, 1)
        else:
            expected_size.insert(squeeze_dim, 1)
        expected_size = torch.Size(expected_size)

        for key in keys_inv:
            assert td.get(key).shape[len(batch) :] == torch.Size(expected_size)

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_grayscale(self, keys, device):
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
            observation_spec = NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16))
            observation_spec = gs.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, 16, 16])
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [[("next", "observation"), "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_totensorimage(self, keys, batch, device):
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
            observation_spec = NdBoundedTensorSpec(0, 255, (16, 16, 3))
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 16, 16])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(0, 255, (16, 16, 3)) for key in keys}
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
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(in_keys=keys, N=4)
        t2 = FiniteTensorDictCheck()
        compose = Compose(t1, t2)
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
        compose(td)
        for key in keys:
            assert td.get(key).shape[-3] == nchannels * N
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(0, 255, (nchannels, 16, 16))
            observation_spec = compose.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels * N, 16, 16])
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(0, 255, (nchannels, 16, 16)) for key in keys}
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

        compose.inv(td)
        for key in keys_to_transform:
            assert td.get(key).dtype == torch.double
        for key in keys_total - keys_to_transform:
            assert td.get(key).dtype == torch.float32

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
            observation_spec = NdBoundedTensorSpec(
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
                    key: NdBoundedTensorSpec(0, 1, (nchannels, 16, 16), device=device)
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
                    observation=NdBoundedTensorSpec(
                        minimum=1, maximum=1, shape=torch.Size([size])
                    ),
                    observation_orig=NdBoundedTensorSpec(
                        minimum=1, maximum=1, shape=torch.Size([size])
                    ),
                ),
                action_spec=NdBoundedTensorSpec(
                    minimum=1, maximum=1, shape=torch.Size((size,))
                ),
                seed=0,
            )
            base_env.out_key = "observation"
            return base_env

        if parallel:
            base_env = SerialEnv(3, make_env)
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

        assert t_env.transform.loc.shape == t_env.observation_spec["observation"].shape
        assert (
            t_env.transform.scale.shape == t_env.observation_spec["observation"].shape
        )
        assert t_env.transform.loc.dtype == t_env.observation_spec["observation"].dtype
        assert (
            t_env.transform.loc.device == t_env.observation_spec["observation"].device
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

    @pytest.mark.parametrize("device", get_available_devices())
    def test_observationnorm_infinite_stats_error(self, device):
        base_env = ContinuousActionVecMockEnv(
            observation_spec=CompositeSpec(
                observation=NdBoundedTensorSpec(
                    minimum=1, maximum=1, shape=torch.Size([1])
                ),
                observation_orig=NdBoundedTensorSpec(
                    minimum=1, maximum=1, shape=torch.Size([1])
                ),
            ),
            action_spec=NdBoundedTensorSpec(
                minimum=1, maximum=1, shape=torch.Size((1,))
            ),
            seed=0,
        )
        base_env.out_key = "observation"
        t_env = TransformedEnv(
            base_env,
            transform=ObservationNorm(in_keys="observation"),
        )
        t_env.append_transform(ObservationNorm(in_keys="observation"))
        err_msg = "Non-finite values found in"
        with pytest.raises(RuntimeError, match=err_msg):
            for transform in t_env.transform:
                transform.init_stats(num_iter=100)

    def test_catframes_transform_observation_spec(self):
        N = 4
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        cat_frames = CatFrames(N=N, in_keys=keys)
        mins = [0, 0.5]
        maxes = [0.5, 1]
        observation_spec = CompositeSpec(
            {
                key: NdBoundedTensorSpec(
                    space_min, space_max, (1, 3, 3), dtype=torch.double
                )
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        result = cat_frames.transform_observation_spec(observation_spec)
        observation_spec = CompositeSpec(
            {
                key: NdBoundedTensorSpec(
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
    def test_catframes_buffer_check_latest_frame(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.zeros(1, 1, 3, 3, device=device)
        key2_tensor = torch.ones(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        cat_frames = CatFrames(N=N, in_keys=keys)

        cat_frames(td)
        latest_frame = td.get(key2)

        assert latest_frame.shape[1] == N
        for i in range(0, N - 1):
            assert torch.equal(latest_frame[0][i], key2_tensor[0][0])
        assert torch.equal(latest_frame[0][N - 1], key1_tensor[0][0])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_catframes_reset(self, device):
        key1 = "first key"
        key2 = "second key"
        N = 4
        keys = [key1, key2]
        key1_tensor = torch.zeros(1, 1, 3, 3, device=device)
        key2_tensor = torch.ones(1, 1, 3, 3, device=device)
        key_tensors = [key1_tensor, key2_tensor]
        td = TensorDict(dict(zip(keys, key_tensors)), [1], device=device)
        cat_frames = CatFrames(N=N, in_keys=keys)

        cat_frames(td)
        buffer_length1 = len(cat_frames.buffer)
        passed_back_td = cat_frames.reset(td)

        assert buffer_length1 == 2
        assert td is passed_back_td
        assert 0 == len(cat_frames.buffer)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_finitetensordictcheck(self, device):
        ftd = FiniteTensorDictCheck()
        td = TensorDict(
            {key: torch.randn(1, 3, 3, device=device) for key in ["a", "b", "c"]}, [1]
        )
        ftd(td)
        td.set("inf", torch.zeros(1, 3).fill_(float("inf")))
        with pytest.raises(ValueError, match="Encountered a non-finite tensor"):
            ftd(td)

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
        double2float(td)
        for key in keys:
            assert td.get(key).dtype == torch.float
        assert td.get("dont touch").dtype == torch.double

        double2float.inv(td)
        for key in keys_inv:
            assert td.get(key).dtype == torch.double
        assert td.get("dont touch").dtype == torch.double

        if len(keys_total) == 1 and len(keys_inv) and keys[0] == "action":
            action_spec = NdBoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
            input_spec = CompositeSpec(action=action_spec)
            action_spec = double2float.transform_input_spec(input_spec)
            assert action_spec.dtype == torch.float

        elif len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
            observation_spec = double2float.transform_observation_spec(observation_spec)
            assert observation_spec.dtype == torch.float

        else:
            observation_spec = CompositeSpec(
                {
                    key: NdBoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
                    for key in keys
                }
            )
            observation_spec = double2float.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].dtype == torch.float

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["observation", "observation_other"],
            ["observation_pixels"],
        ],
    )
    def test_cattensors(self, keys, device):
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

        cattensors(td)
        assert td.get("observation_out").shape[-2] == len(keys) * 4
        assert td.get("dont touch").shape == dont_touch.shape

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(0, 1, (1, 4, 32))
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, len(keys) * 4, 32])
        else:
            observation_spec = CompositeSpec(
                {key: NdBoundedTensorSpec(0, 1, (1, 4, 32)) for key in keys}
            )
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec["observation_out"].shape == torch.Size(
                [1, len(keys) * 4, 32]
            )

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

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_noop_reset_env(self, random, device, compose):
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
        env = SerialEnv(3, lambda: ContinuousActionVecMockEnv())
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        transformed_env = TransformedEnv(env)
        transformed_env.append_transform(noop_reset_env)
        with pytest.raises(
            ValueError,
            match="there is more than one done state in the parent environment",
        ):
            transformed_env.reset()

    @pytest.mark.parametrize(
        "default_keys", [["action"], ["action", "monkeys jumping on the bed"]]
    )
    @pytest.mark.parametrize(
        "spec",
        [
            CompositeSpec(b=NdBoundedTensorSpec(-3, 3, [4])),
            NdBoundedTensorSpec(-3, 3, [4]),
        ],
    )
    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    @pytest.mark.parametrize("serial", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_tensordict_primer(
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
            env = SerialEnv(3, make_env)
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

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    def test_binarized_reward(self, device, batch):
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
        assert td["reward"] is reward
        assert (td["reward"] != reward_copy).all()
        assert (td["misc"] == misc_copy).all()
        assert (torch.count_nonzero(td["reward"]) == torch.sum(reward_copy > 0)).all()

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

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device found")
    @pytest.mark.parametrize("device", get_available_devices())
    def test_pin_mem(self, device):
        pin_mem = PinMemoryTransform()
        td = TensorDict(
            {key: torch.randn(3) for key in ["a", "b", "c"]}, [], device=device
        )
        pin_mem(td)
        for item in td.values():
            assert item.is_pinned

    def test_append(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        (key,) = itertools.islice(obs_spec.keys(), 1)

        env = TransformedEnv(env)
        env.append_transform(CatFrames(N=4, cat_dim=-1, in_keys=[key]))
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
        assert env._observation_spec is not None
        assert env._reward_spec is not None

        env.insert_transform(0, CatFrames(N=4, cat_dim=-1, in_keys=[key]))

        # transformed envs do not have spec after insert -- they need to be computed
        assert env._input_spec is None
        assert env._observation_spec is None
        assert env._reward_spec is None

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
        assert env._observation_spec is None
        assert env._reward_spec is None

        env.insert_transform(-5, CatFrames(N=4, cat_dim=-1, in_keys=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 6

        assert isinstance(env.transform[0], CatFrames)
        assert isinstance(env.transform[1], NoopResetEnv)
        assert isinstance(env.transform[2], PinMemoryTransform)
        assert isinstance(env.transform[3], CatFrames)
        assert isinstance(env.transform[4], NoopResetEnv)
        assert isinstance(env.transform[5], FiniteTensorDictCheck)

        assert env._input_spec is None
        assert env._observation_spec is None
        assert env._reward_spec is None

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
            assert env._observation_spec is not None
            assert env._reward_spec is not None

        try:
            env.insert_transform(7, FiniteTensorDictCheck())
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._input_spec is not None
            assert "action" in env._input_spec
            assert env._input_spec["action"] is not None
            assert env._observation_spec is not None
            assert env._reward_spec is not None

        try:
            env.insert_transform(4, "ffff")
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._input_spec is not None
            assert "action" in env._input_spec
            assert env._input_spec["action"] is not None
            assert env._observation_spec is not None
            assert env._reward_spec is not None


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("model", ["resnet18", "resnet34", "resnet50"])
class TestR3M:
    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_r3m_instantiation(self, model, tensor_pixels_key, device):
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
            {("next", "vec"), ("next", "pixels_orig"), "action", "reward", "next"}
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
            base_env = ParallelEnv(3, base_env_constructor)
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
            {("next", "vec"), ("next", "pixels_orig"), "action", "reward", "next"}
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
            {("next", "vec"), ("next", "pixels_orig"), "action", "reward", "next"}
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
            {key: NdBoundedTensorSpec(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = CompositeSpec(
                {
                    key: NdUnboundedContinuousTensorSpec(r3m_net.outdim, device)
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
                ts_dict[key] = NdUnboundedContinuousTensorSpec(r3m_net.outdim, device)
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
            + ["reward", "done", "next"]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


@pytest.mark.skipif(not _has_tv, reason="torchvision not installed")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("model", ["resnet50"])
class TestVIP:
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
            {("next", "vec"), ("next", "pixels_orig"), "next", "action", "reward"}
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
            base_env = ParallelEnv(3, base_env_constructor)
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
            {("next", "vec"), ("next", "pixels_orig"), "next", "action", "reward"}
        )
        if not stack_images:
            exp_keys.add(("next", "vec2"))
        assert set(td.keys(True)) == exp_keys, set(td.keys(True)) - exp_keys
        transformed_env.close()

    def test_vip_parallel(self, model, device):
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
            {("next", "vec"), ("next", "pixels_orig"), "next", "action", "reward"}
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
        exp_keys = {"vec", "done", "pixels_orig", "goal_embedding", "goal_image"}
        if tensor_pixels_key:
            exp_keys.add(tensor_pixels_key)
        assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union(
            {("next", "vec"), ("next", "pixels_orig"), "next", "action", "reward"}
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
        torch.testing.assert_close(explicit_reward, td["reward"].squeeze())

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
            {key: NdBoundedTensorSpec(-1, 1, (3, 16, 16), device) for key in in_keys}
        )
        if del_keys:
            exp_ts = CompositeSpec(
                {
                    key: NdUnboundedContinuousTensorSpec(vip_net.outdim, device)
                    for key in out_keys
                }
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
                ts_dict[key] = NdUnboundedContinuousTensorSpec(vip_net.outdim, device)
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
            + ["reward", "done", "next"]
        )
        assert set(expected_keys) == set(transformed_env.rollout(3).keys(True))


@pytest.mark.parametrize("device", get_available_devices())
def test_batch_locked_transformed(device):
    env = TransformedEnv(
        MockBatchedLockedEnv(device),
        Compose(
            ObservationNorm(in_keys=[("next", "observation")], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand(env.batch_size)
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
            ObservationNorm(in_keys=[("next", "observation")], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand(env.batch_size)
    td_expanded = td.expand(2).clone()
    env.step(td)
    env.step(td_expanded)


@pytest.mark.parametrize("device", get_available_devices())
def test_batch_unlocked_with_batch_size_transformed(device):
    env = TransformedEnv(
        MockBatchedUnLockedEnv(device, batch_size=torch.Size([2])),
        Compose(
            ObservationNorm(in_keys=[("next", "observation")], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False

    td = env.reset()
    td["action"] = env.action_spec.rand(env.batch_size)
    td_expanded = td.expand(2, 2).reshape(-1).to_tensordict()
    env.step(td)

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
    pytest.param(partial(FlattenObservation, first_dim=-3), id="FlattenObservation"),
    pytest.param(
        partial(UnsqueezeTransform, unsqueeze_dim=-1), id="UnsqueezeTransform"
    ),
    pytest.param(partial(SqueezeTransform, squeeze_dim=-1), id="SqueezeTransform"),
    GrayScale,
    ObservationNorm,
    CatFrames,
    pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
    FiniteTensorDictCheck,
    DoubleToFloat,
    CatTensors,
    pytest.param(
        partial(DiscreteActionProjection, max_n=1, m=1), id="DiscreteActionProjection"
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
