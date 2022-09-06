# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from copy import copy

import pytest
import torch
from _utils_internal import get_available_devices
from mocking_classes import ContinuousActionVecMockEnv, DiscreteActionConvMockEnvNumpy
from torch import Tensor
from torch import multiprocessing as mp
from torchrl import prod
from torchrl.data import (
    NdBoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data import TensorDict
from torchrl.envs import EnvCreator, SerialEnv
from torchrl.envs import ParallelEnv
from torchrl.envs import (
    Resize,
    GrayScale,
    ToTensorImage,
    Compose,
    ObservationNorm,
    CatFrames,
    FiniteTensorDictCheck,
    DoubleToFloat,
    CatTensors,
    FlattenObservation,
    RewardScaling,
    BinarizeReward,
    R3MTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import VecNorm, TransformedEnv
from torchrl.envs.transforms.transforms import (
    _has_tv,
    NoopResetEnv,
    PinMemoryTransform,
    CenterCrop,
    UnsqueezeTransform,
)

TIMEOUT = 10.0


def _test_vecnorm_subproc_auto(idx, make_env, queue_out: mp.Queue, queue_in: mp.Queue):
    env = make_env()
    env.set_seed(1000 + idx)
    tensordict = env.reset()
    for _ in range(10):
        tensordict = env.rand_step(tensordict)
    queue_out.put(True)
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"
    t = env.transform
    obs_sum = t._td.get("next_observation_sum").clone()
    obs_ssq = t._td.get("next_observation_ssq").clone()
    obs_count = t._td.get("next_observation_count").clone()
    reward_sum = t._td.get("reward_sum").clone()
    reward_ssq = t._td.get("reward_ssq").clone()
    reward_count = t._td.get("reward_count").clone()

    queue_out.put((obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count))
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"
    env.close()
    queue_out.close()
    queue_in.close()
    del queue_in, queue_out


@pytest.mark.parametrize("nprc", [2, 5])
def test_vecnorm_parallel_auto(nprc):

    queues = []
    prcs = []
    if _has_gym:
        make_env = EnvCreator(
            lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm(decay=1.0))
        )
    else:
        make_env = EnvCreator(
            lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm(decay=1.0))
        )

    for idx in range(nprc):
        prc_queue_in = mp.Queue(1)
        prc_queue_out = mp.Queue(1)
        p = mp.Process(
            target=_test_vecnorm_subproc_auto,
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

    obs_sum = td.get("next_observation_sum").clone()
    obs_ssq = td.get("next_observation_ssq").clone()
    obs_count = td.get("next_observation_count").clone()
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


def test_parallelenv_vecnorm():
    if _has_gym:
        make_env = EnvCreator(lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm()))
        env_input_keys = None
    else:
        make_env = EnvCreator(
            lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm())
        )
        env_input_keys = ["action", ContinuousActionVecMockEnv._out_key]
    parallel_env = ParallelEnv(3, make_env, env_input_keys=env_input_keys)
    queue_out = mp.Queue(1)
    queue_in = mp.Queue(1)
    proc = mp.Process(target=_run_parallelenv, args=(parallel_env, queue_out, queue_in))
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


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
@pytest.mark.parametrize(
    "parallel",
    [
        None,
        False,
        True,
    ],
)
def test_vecnorm(parallel, thr=0.2, N=200):  # 10000):
    torch.manual_seed(0)

    if parallel is None:
        env = GymEnv("Pendulum-v1")
    elif parallel:
        env = ParallelEnv(num_workers=5, create_env_fn=lambda: GymEnv("Pendulum-v1"))
    else:
        env = SerialEnv(num_workers=5, create_env_fn=lambda: GymEnv("Pendulum-v1"))

    env.set_seed(0)
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
    obs = tds.get("next_observation")
    obs = obs.view(-1, obs.shape[-1])
    mean = obs.mean(0)
    assert (abs(mean) < thr).all()
    std = obs.std(0)
    assert (abs(std - 1) < thr).all()
    if not env_t.is_closed:
        env_t.close()


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


class TestTransforms:
    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_resize(self, interpolation, keys, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        resize = Resize(w=20, h=21, interpolation=interpolation, keys_in=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
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
                **{key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("h", [None, 21])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_centercrop(self, keys, h, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        cc = CenterCrop(w=20, h=h, keys_in=keys)
        if h is None:
            h = 20
        td = TensorDict(
            {
                key: torch.randn(*batch, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
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
                **{key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = cc.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, h])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_flatten(self, keys, size, nchannels, batch, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        start_dim = -3 - len(size)
        flatten = FlattenObservation(start_dim, -3, keys_in=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
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
                **{
                    key: NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = flatten.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape[-3] == expected_size

    @pytest.mark.parametrize("unsqueeze_dim", [1, -2])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize("batch", [[], [2], [2, 4]])
    @pytest.mark.parametrize("size", [[], [4]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_unsqueeze(self, keys, size, nchannels, batch, device, unsqueeze_dim):
        torch.manual_seed(0)
        dont_touch = torch.randn(*batch, *size, nchannels, 16, 16, device=device)
        unsqueeze = UnsqueezeTransform(unsqueeze_dim, keys_in=keys)
        td = TensorDict(
            {
                key: torch.randn(*batch, *size, nchannels, 16, 16, device=device)
                for key in keys
            },
            batch,
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
                **{
                    key: NdBoundedTensorSpec(-1, 1, (*size, nchannels, 16, 16))
                    for key in keys
                }
            )
            observation_spec = unsqueeze.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == expected_size

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_grayscale(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = GrayScale(keys_in=keys)
        dont_touch = torch.randn(1, nchannels, 16, 16, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 16, 16, device=device) for key in keys}, [1]
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
                **{key: NdBoundedTensorSpec(-1, 1, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 16, 16])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_totensorimage(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = ToTensorImage(keys_in=keys)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 16, 16, 3), device=device)
                for key in keys
            },
            batch,
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
                **{key: NdBoundedTensorSpec(0, 255, (16, 16, 3)) for key in keys}
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
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(keys_in=keys, N=4)
        t2 = FiniteTensorDictCheck()
        compose = Compose(t1, t2)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, nchannels, 16, 16), device=device)
                for key in keys
            },
            batch,
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
                **{
                    key: NdBoundedTensorSpec(0, 255, (nchannels, 16, 16))
                    for key in keys
                }
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
        keys_total = set(["action_1", "action_2", "dont_touch"])
        double2float_1 = DoubleToFloat(keys_inv_in=keys_inv_1)
        double2float_2 = DoubleToFloat(keys_inv_in=keys_inv_2)
        compose = Compose(double2float_1, double2float_2)
        td = TensorDict(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.float32, device=device)
                for key in keys_total
            },
            [1],
        )

        compose.inv(td)
        for key in keys_to_transform:
            assert td.get(key).dtype == torch.double
        for key in keys_total - keys_to_transform:
            assert td.get(key).dtype == torch.float32

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
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
        on = ObservationNorm(loc, scale, keys_in=keys, standard_normal=standard_normal)
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
                **{
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

    def test_catframes_transform_observation_spec(self):
        N = 4
        key1 = "first key"
        key2 = "second key"
        keys = [key1, key2]
        cat_frames = CatFrames(N=N, keys_in=keys)
        mins = [0, 0.5]
        maxes = [0.5, 1]
        observation_spec = CompositeSpec(
            **{
                key: NdBoundedTensorSpec(
                    space_min, space_max, (1, 3, 3), dtype=torch.double
                )
                for key, space_min, space_max in zip(keys, mins, maxes)
            }
        )

        result = cat_frames.transform_observation_spec(observation_spec)
        observation_spec = CompositeSpec(
            **{
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
        td = TensorDict(dict(zip(keys, key_tensors)), [1])
        cat_frames = CatFrames(N=N, keys_in=keys)

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
        td = TensorDict(dict(zip(keys, key_tensors)), [1])
        cat_frames = CatFrames(N=N, keys_in=keys)

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
        with pytest.raises(ValueError, match="Found non-finite elements"):
            ftd(td)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "keys",
        [
            ["next_observation", "some_other_key"],
            ["next_observation_pixels"],
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
        double2float = DoubleToFloat(keys_in=keys, keys_inv_in=keys_inv)
        dont_touch = torch.randn(1, 3, 3, dtype=torch.double, device=device)
        td = TensorDict(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.double, device=device)
                for key in keys_total
            },
            [1],
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
            action_spec = double2float.transform_action_spec(action_spec)
            assert action_spec.dtype == torch.float

        elif len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(0, 1, (1, 3, 3), dtype=torch.double)
            observation_spec = double2float.transform_observation_spec(observation_spec)
            assert observation_spec.dtype == torch.float

        else:
            observation_spec = CompositeSpec(
                **{
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
            ["next_observation", "next_observation_other"],
            ["next_observation_pixels"],
        ],
    )
    def test_cattensors(self, keys, device):
        cattensors = CatTensors(keys_in=keys, out_key="observation_out", dim=-2)

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
                **{key: NdBoundedTensorSpec(0, 1, (1, 4, 32)) for key in keys}
            )
            observation_spec = cattensors.transform_observation_spec(observation_spec)
            assert observation_spec["observation_out"].shape == torch.Size(
                [1, len(keys) * 4, 32]
            )

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
            {
                "misc": misc,
                "reward": reward,
            },
            batch,
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
    def test_reward_scaling(self, batch, scale, loc, keys, device):
        torch.manual_seed(0)
        if keys is None:
            keys_total = set([])
        else:
            keys_total = set(keys)
        reward_scaling = RewardScaling(keys_in=keys, scale=scale, loc=loc)
        td = TensorDict(
            {
                **{key: torch.randn(*batch, 1, device=device) for key in keys_total},
                "reward": torch.randn(*batch, 1, device=device),
            },
            batch,
        )
        td.set("dont touch", torch.randn(*batch, 1, device=device))
        td_copy = td.clone()
        reward_scaling(td)
        for key in keys_total:
            assert (td.get(key) == td_copy.get(key).mul_(scale).add_(loc)).all()
        assert (td.get("dont touch") == td_copy.get("dont touch")).all()
        if len(keys_total) == 0:
            assert (
                td.get("reward") == td_copy.get("reward").mul_(scale).add_(loc)
            ).all()
        elif len(keys_total) == 1:
            reward_spec = UnboundedContinuousTensorSpec(device=device)
            reward_spec = reward_scaling.transform_reward_spec(reward_spec)
            assert reward_spec.shape == torch.Size([1])

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device found")
    @pytest.mark.parametrize("device", get_available_devices())
    def test_pin_mem(self, device):
        pin_mem = PinMemoryTransform()
        td = TensorDict({key: torch.randn(3) for key in ["a", "b", "c"]}, [])
        pin_mem(td)
        for item in td.values():
            assert item.is_pinned

    def test_append(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        key = list(obs_spec.keys())[0]

        env = TransformedEnv(env)
        env.append_transform(CatFrames(N=4, cat_dim=-1, keys_in=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 1
        obs_spec = env.observation_spec
        obs_spec = obs_spec[key]
        assert obs_spec.shape[-1] == 4 * env.base_env.observation_spec[key].shape[-1]

    def test_insert(self):

        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        key = list(obs_spec.keys())[0]
        env = TransformedEnv(env)

        _ = env.action_spec
        _ = env.observation_spec
        _ = env.reward_spec

        assert env._action_spec is not None
        assert env._observation_spec is not None
        assert env._reward_spec is not None

        env.insert_transform(0, CatFrames(N=4, cat_dim=-1, keys_in=[key]))

        assert env._action_spec is None
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
        assert env._action_spec is None
        assert env._observation_spec is None
        assert env._reward_spec is None

        env.insert_transform(-5, CatFrames(N=4, cat_dim=-1, keys_in=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 6

        assert isinstance(env.transform[0], CatFrames)
        assert isinstance(env.transform[1], NoopResetEnv)
        assert isinstance(env.transform[2], PinMemoryTransform)
        assert isinstance(env.transform[3], CatFrames)
        assert isinstance(env.transform[4], NoopResetEnv)
        assert isinstance(env.transform[5], FiniteTensorDictCheck)
        assert env._action_spec is None
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
            assert env._action_spec is not None
            assert env._observation_spec is not None
            assert env._reward_spec is not None

        try:
            env.insert_transform(7, FiniteTensorDictCheck())
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._action_spec is not None
            assert env._observation_spec is not None
            assert env._reward_spec is not None

        try:
            env.insert_transform(4, "ffff")
            assert 1 == 6
        except ValueError:
            assert len(env.transform) == 6
            assert env._action_spec is not None
            assert env._observation_spec is not None
            assert env._reward_spec is not None


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("model", ["resnet18", "resnet34", "resnet50"])
class TestR3M:
    @pytest.mark.parametrize("tensor_pixels_key", [None, ["funny_key"]])
    def test_r3m_instantiation(self, model, tensor_pixels_key, device):
        keys_in = ["next_pixels"]
        keys_out = ["next_vec"]
        r3m = R3MTransform(
            model,
            keys_in=keys_in,
            keys_out=keys_out,
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
        exp_keys = exp_keys.union({"next_vec", "next_pixels_orig", "action", "reward"})
        assert set(td.keys()) == exp_keys, set(td.keys()) - exp_keys
        transformed_env.close()

    @pytest.mark.parametrize("stack_images", [True, False])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_r3m_mult_images(self, model, device, stack_images, parallel):
        keys_in = ["next_pixels", "next_pixels2"]
        keys_out = ["next_vec"] if stack_images else ["next_vec", "next_vec2"]
        r3m = R3MTransform(
            model,
            keys_in=keys_in,
            keys_out=keys_out,
            stack_images=stack_images,
        )

        base_env_constructor = lambda: TransformedEnv(
            DiscreteActionConvMockEnvNumpy().to(device),
            CatTensors(["next_pixels"], "next_pixels2", del_keys=False),
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
        exp_keys = exp_keys.union({"next_vec", "next_pixels_orig", "action", "reward"})
        if not stack_images:
            exp_keys = exp_keys.union({"next_vec2"})
        assert set(td.keys()) == exp_keys, set(td.keys()) - exp_keys
        transformed_env.close()

    def test_r3m_parallel(self, model, device):
        keys_in = ["next_pixels"]
        keys_out = ["next_vec"]
        tensor_pixels_key = None
        r3m = R3MTransform(
            model,
            keys_in=keys_in,
            keys_out=keys_out,
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
        assert set(td.keys()) == exp_keys

        td = transformed_env.rand_step(td)
        exp_keys = exp_keys.union({"next_vec", "next_pixels_orig", "action", "reward"})
        assert set(td.keys()) == exp_keys, set(td.keys()) - exp_keys
        transformed_env.close()
        del transformed_env


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
