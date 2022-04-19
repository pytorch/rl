# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from mocking_classes import ContinuousActionVecMockEnv
from torch import multiprocessing as mp
from torchrl.agents.env_creator import EnvCreator
from torchrl.data import TensorDict
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import VecNorm, TransformedEnv

TIMEOUT = 10.0


def _test_vecnorm_subproc(idx, queue_out: mp.Queue, queue_in: mp.Queue):
    td = queue_in.get(timeout=TIMEOUT)
    if _has_gym:
        env = GymEnv("Pendulum-v1")
    else:
        env = ContinuousActionVecMockEnv()
    t = VecNorm(shared_td=td)
    env = TransformedEnv(env, t)
    env.set_seed(1000+idx)
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
    env.set_seed(1000+idx)
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
        assert (item != new_values.get(k)).any(), k
    proc.join()
    parallel_env.close()


@pytest.mark.parametrize("parallel", [False, True])
def test_vecnorm(parallel, thr=0.2, N=200):  # 10000):
    torch.manual_seed(0)

    if parallel:
        if _has_gym:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: GymEnv("Pendulum-v1")
            )
        else:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: ContinuousActionVecMockEnv()
            )
    elif _has_gym:
        env = GymEnv("Pendulum-v1")
    else:
        env = ContinuousActionVecMockEnv()

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
