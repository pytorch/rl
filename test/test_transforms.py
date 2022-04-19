# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from _utils_internal import get_available_devices
from mocking_classes import ContinuousActionVecMockEnv
from torch import Tensor
from torch import multiprocessing as mp
from torchrl.agents.env_creator import EnvCreator
from torchrl.data import NdBoundedTensorSpec, CompositeSpec
from torchrl.data import TensorDict
from torchrl.envs import GymEnv, ParallelEnv
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
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import VecNorm, TransformedEnv
from torchrl.envs.transforms.transforms import (
    _has_tv,
    NoopResetEnv,
    PinMemoryTransform,
)

TIMEOUT = 10.0


def _test_vecnorm_subproc(idx, queue_out: mp.Queue, queue_in: mp.Queue):
    td = queue_in.get(timeout=TIMEOUT)
    if _has_gym:
        env = GymEnv("Pendulum-v1")
    else:
        env = ContinuousActionVecMockEnv()
    t = VecNorm(shared_td=td)
    env = TransformedEnv(env, t)
    env.set_seed(1000 + idx)
    env.reset()
    assert env.current_tensordict is not None
    for _ in range(10):
        env.rand_step()
    queue_out.put(True)
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"
    obs_sum = t._td.get("next_observation_sum").clone()
    obs_ssq = t._td.get("next_observation_ssq").clone()
    obs_count = t._td.get("next_observation_ssq").clone()
    reward_sum = t._td.get("reward_sum").clone()
    reward_ssq = t._td.get("reward_ssq").clone()
    reward_count = t._td.get("reward_ssq").clone()

    td_out = TensorDict(
        {
            "obs_sum": obs_sum,
            "obs_ssq": obs_ssq,
            "obs_count": obs_count,
            "reward_sum": reward_sum,
            "reward_ssq": reward_ssq,
            "reward_count": reward_count,
        },
        [],
    ).share_memory_()
    queue_out.put(td_out)
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"


@pytest.mark.parametrize("nprc", [2, 5])
def test_vecnorm_parallel(nprc):
    queues = []
    prcs = []
    if _has_gym:
        td = VecNorm.build_td_for_shared_vecnorm(GymEnv("Pendulum-v1"))
    else:
        td = VecNorm.build_td_for_shared_vecnorm(ContinuousActionVecMockEnv())
    for idx in range(nprc):
        prc_queue_in = mp.Queue(1)
        prc_queue_out = mp.Queue(1)
        p = mp.Process(
            target=_test_vecnorm_subproc,
            args=(
                idx,
                prc_queue_in,
                prc_queue_out,
            ),
        )
        p.start()
        prc_queue_out.put(td)
        prcs.append(p)
        queues.append((prc_queue_in, prc_queue_out))

    dones = [queue[0].get(timeout=TIMEOUT) for queue in queues]
    assert all(dones)
    msg = "all_done"
    for idx in range(nprc):
        queues[idx][1].put(msg)

    obs_sum = td.get("next_observation_sum").clone()
    obs_ssq = td.get("next_observation_ssq").clone()
    obs_count = td.get("next_observation_ssq").clone()
    reward_sum = td.get("reward_sum").clone()
    reward_ssq = td.get("reward_ssq").clone()
    reward_count = td.get("reward_ssq").clone()

    for idx in range(nprc):
        td_out = queues[idx][0].get(timeout=TIMEOUT)
        _obs_sum = td_out.get("obs_sum")
        _obs_ssq = td_out.get("obs_ssq")
        _obs_count = td_out.get("obs_count")
        _reward_sum = td_out.get("reward_sum")
        _reward_ssq = td_out.get("reward_ssq")
        _reward_count = td_out.get("reward_count")
        assert (obs_sum == _obs_sum).all()
        assert (obs_ssq == _obs_ssq).all()
        assert (obs_count == _obs_count).all()
        assert (reward_sum == _reward_sum).all()
        assert (reward_ssq == _reward_ssq).all()
        assert (reward_count == _reward_count).all()

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


def _test_vecnorm_subproc_auto(idx, make_env, queue_out: mp.Queue, queue_in: mp.Queue):
    env = make_env()
    env.set_seed(1000 + idx)
    env.reset()
    for _ in range(10):
        env.rand_step()
    queue_out.put(True)
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"
    t = env.transform
    obs_sum = t._td.get("next_observation_sum").clone()
    obs_ssq = t._td.get("next_observation_ssq").clone()
    obs_count = t._td.get("next_observation_ssq").clone()
    reward_sum = t._td.get("reward_sum").clone()
    reward_ssq = t._td.get("reward_ssq").clone()
    reward_count = t._td.get("reward_ssq").clone()

    queue_out.put((obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count))
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "all_done"
    env.close()


@pytest.mark.parametrize("nprc", [2, 5])
def test_vecnorm_parallel_auto(nprc):
    queues = []
    prcs = []
    if _has_gym:
        make_env = EnvCreator(lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm()))
    else:
        make_env = EnvCreator(
            lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm())
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

    td = list(make_env.state_dict().values())[0]
    dones = [queue[0].get() for queue in queues]
    assert all(dones)
    msg = "all_done"
    for idx in range(nprc):
        queues[idx][1].put(msg)

    obs_sum = td.get("next_observation_sum").clone()
    obs_ssq = td.get("next_observation_ssq").clone()
    obs_count = td.get("next_observation_ssq").clone()
    reward_sum = td.get("reward_sum").clone()
    reward_ssq = td.get("reward_ssq").clone()
    reward_count = td.get("reward_ssq").clone()

    for idx in range(nprc):
        tup = queues[idx][0].get(timeout=TIMEOUT)
        _obs_sum, _obs_ssq, _obs_count, _reward_sum, _reward_ssq, _reward_count = tup
        assert (obs_sum == _obs_sum).all(), (_obs_sum, obs_sum)
        assert (obs_ssq == _obs_ssq).all()
        assert (obs_count == _obs_count).all()
        assert (reward_sum == _reward_sum).all()
        assert (reward_ssq == _reward_ssq).all()
        assert (reward_count == _reward_count).all()

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


def _run_parallelenv(parallel_env, queue_in, queue_out):
    parallel_env.reset()

    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "start"
    for _ in range(10):
        parallel_env.rand_step()
    queue_out.put("first round")
    msg = queue_in.get(timeout=TIMEOUT)
    assert msg == "start"
    for _ in range(10):
        parallel_env.rand_step()
    queue_out.put("second round")
    parallel_env.close()
    del parallel_env


def test_parallelenv_vecnorm():
    if _has_gym:
        make_env = EnvCreator(lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm()))
    else:
        make_env = EnvCreator(
            lambda: TransformedEnv(ContinuousActionVecMockEnv(), VecNorm())
        )
    parallel_env = ParallelEnv(3, make_env)
    queue_out = mp.Queue(1)
    queue_in = mp.Queue(1)
    proc = mp.Process(target=_run_parallelenv, args=(parallel_env, queue_out, queue_in))
    proc.start()
    parallel_sd = parallel_env.state_dict()
    assert "worker0" in parallel_sd
    worker_sd = parallel_sd["worker0"]
    td = list(worker_sd.values())[0]
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
    parallel_env.close()


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
@pytest.mark.parametrize("parallel", [False, True])
def test_vecnorm(parallel, thr=0.2, N=200):  # 10000):
    torch.manual_seed(0)

    if parallel:
        env = ParallelEnv(num_workers=5, create_env_fn=lambda: GymEnv("Pendulum-v1"))
    else:
        env = GymEnv("Pendulum-v1")

    env.set_seed(0)
    t = VecNorm()
    env = TransformedEnv(env, t)
    env.reset()
    tds = []
    for _ in range(N):
        td = env.rand_step()
        if td.get("done").any():
            env.reset()
        tds.append(td)
    tds = torch.stack(tds, 0)
    obs = tds.get("next_observation")
    obs = obs.view(-1, obs.shape[-1])
    mean = obs.mean(0)
    assert (abs(mean) < thr).all()
    std = obs.std(0)
    assert (abs(std - 1) < thr).all()
    env.close()


class TestTransforms:
    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("nchannels", [1, 3])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_resize(self, interpolation, keys, nchannels, device):
        torch.manual_seed(0)
        dont_touch = torch.randn(1, nchannels, 32, 32, device=device)
        resize = Resize(w=20, h=21, interpolation=interpolation, keys=keys)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 32, 32, device=device) for key in keys}, [1]
        )
        td.set("dont touch", dont_touch.clone())
        resize(td)
        for key in keys:
            assert td.get(key).shape[-2:] == torch.Size([20, 21])
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(-1, 1, (nchannels, 32, 32))
            observation_spec = resize.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels, 20, 21])
        else:
            observation_spec = CompositeSpec(
                **{key: NdBoundedTensorSpec(-1, 1, (nchannels, 32, 32)) for key in keys}
            )
            observation_spec = resize.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([nchannels, 20, 21])

    @pytest.mark.skipif(not _has_tv, reason="no torchvision")
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_grayscale(self, keys, device):
        torch.manual_seed(0)
        nchannels = 3
        gs = GrayScale(keys=keys)
        dont_touch = torch.randn(1, nchannels, 32, 32, device=device)
        td = TensorDict(
            {key: torch.randn(1, nchannels, 32, 32, device=device) for key in keys}, [1]
        )
        td.set("dont touch", dont_touch.clone())
        gs(td)
        for key in keys:
            assert td.get(key).shape[-3] == 1
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(-1, 1, (nchannels, 32, 32))
            observation_spec = gs.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([1, 32, 32])
        else:
            observation_spec = CompositeSpec(
                **{key: NdBoundedTensorSpec(-1, 1, (nchannels, 32, 32)) for key in keys}
            )
            observation_spec = gs.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size([1, 32, 32])

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_totensorimage(self, keys, batch, device):
        torch.manual_seed(0)
        nchannels = 3
        totensorimage = ToTensorImage(keys=keys)
        dont_touch = torch.randn(*batch, nchannels, 32, 32, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, 32, 32, 3), device=device)
                for key in keys
            },
            batch,
        )
        td.set("dont touch", dont_touch.clone())
        totensorimage(td)
        for key in keys:
            assert td.get(key).shape[-3:] == torch.Size([3, 32, 32])
            assert td.get(key).device == device
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(0, 255, (32, 32, 3))
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            assert observation_spec.shape == torch.Size([3, 32, 32])
            assert (observation_spec.space.minimum == 0).all()
            assert (observation_spec.space.maximum == 1).all()
        else:
            observation_spec = CompositeSpec(
                **{key: NdBoundedTensorSpec(0, 255, (32, 32, 3)) for key in keys}
            )
            observation_spec = totensorimage.transform_observation_spec(
                observation_spec
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size([3, 32, 32])
                assert (observation_spec[key].space.minimum == 0).all()
                assert (observation_spec[key].space.maximum == 1).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(keys=keys, N=4)
        t2 = FiniteTensorDictCheck()
        compose = Compose(t1, t2)
        dont_touch = torch.randn(*batch, nchannels, 32, 32, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, nchannels, 32, 32), device=device)
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
            observation_spec = NdBoundedTensorSpec(0, 255, (nchannels, 32, 32))
            observation_spec = compose.transform_observation_spec(observation_spec)
            assert observation_spec.shape == torch.Size([nchannels * N, 32, 32])
        else:
            observation_spec = CompositeSpec(
                **{
                    key: NdBoundedTensorSpec(0, 255, (nchannels, 32, 32))
                    for key in keys
                }
            )
            observation_spec = compose.transform_observation_spec(observation_spec)
            for key in keys:
                assert observation_spec[key].shape == torch.Size(
                    [nchannels * N, 32, 32]
                )

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
            (torch.ones(32, 32), torch.ones(1)),
            (torch.ones(1), torch.ones(32, 32)),
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
        on = ObservationNorm(loc, scale, keys=keys, standard_normal=standard_normal)
        dont_touch = torch.randn(1, nchannels, 32, 32, device=device)
        td = TensorDict(
            {key: torch.zeros(1, nchannels, 32, 32, device=device) for key in keys}, [1]
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
            observation_spec = NdBoundedTensorSpec(0, 1, (nchannels, 32, 32))
            observation_spec = on.transform_observation_spec(observation_spec)
            if standard_normal:
                assert (observation_spec.space.minimum == -loc / scale).all()
                assert (observation_spec.space.maximum == (1 - loc) / scale).all()
            else:
                assert (observation_spec.space.minimum == loc).all()
                assert (observation_spec.space.maximum == scale + loc).all()

        else:
            observation_spec = CompositeSpec(
                **{key: NdBoundedTensorSpec(0, 1, (nchannels, 32, 32)) for key in keys}
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

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys", [["next_observation", "some_other_key"], ["next_observation_pixels"]]
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [1, 4, 5])
    def test_catframes(self, batch, keys, device, N):
        pass

    @pytest.mark.parametrize("device", get_available_devices())
    def test_finitetensordictcheck(self, device):
        ftd = FiniteTensorDictCheck()
        td = TensorDict(
            {key: torch.randn(1, 32, 32, device=device) for key in ["a", "b", "c"]}, [1]
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
    def test_double2float(self, keys, device):
        torch.manual_seed(0)
        double2float = DoubleToFloat(keys=keys)
        dont_touch = torch.randn(1, 32, 32, dtype=torch.double, device=device)
        td = TensorDict(
            {
                key: torch.zeros(1, 32, 32, dtype=torch.double, device=device)
                for key in keys
            },
            [1],
        )
        td.set("dont touch", dont_touch.clone())
        double2float(td)
        for key in keys:
            assert td.get(key).dtype == torch.float
        assert td.get("dont touch").dtype == torch.double

        double2float.inv(td)
        for key in keys:
            assert td.get(key).dtype == torch.double
        assert td.get("dont touch").dtype == torch.double

        if len(keys) == 1 and keys[0] == "action":
            action_spec = NdBoundedTensorSpec(0, 1, (1, 32, 32), dtype=torch.double)
            action_spec = double2float.transform_action_spec(action_spec)
            assert action_spec.dtype == torch.float

        elif len(keys) == 1:
            observation_spec = NdBoundedTensorSpec(
                0, 1, (1, 32, 32), dtype=torch.double
            )
            observation_spec = double2float.transform_observation_spec(observation_spec)
            assert observation_spec.dtype == torch.float

        else:
            observation_spec = CompositeSpec(
                **{
                    key: NdBoundedTensorSpec(0, 1, (1, 32, 32), dtype=torch.double)
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
        cattensors = CatTensors(keys=keys, out_key="observation_out", dim=-2)

        dont_touch = torch.randn(1, 32, 32, dtype=torch.double, device=device)
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
    def test_noop_reset_env(self, random):
        torch.manual_seed(0)
        env = ContinuousActionVecMockEnv()
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(env=env, random=random)
        transformed_env = TransformedEnv(env, noop_reset_env)
        transformed_env.reset()
        if random:
            assert transformed_env.step_count > 0
        else:
            assert transformed_env.step_count == 30

    @pytest.mark.parametrize("device", get_available_devices())
    def test_binerized_reward(self, device):
        pass

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device found")
    @pytest.mark.parametrize("device", get_available_devices())
    def test_pin_mem(self, device):
        pin_mem = PinMemoryTransform()
        td = TensorDict({key: torch.randn(3) for key in ["a", "b", "c"]}, [])
        pin_mem(td)
        for key, item in td.items():
            assert item.is_pinned


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
