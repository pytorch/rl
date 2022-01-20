import time

import torch
from torch import multiprocessing as mp
import pytest

from torchrl.data.transforms import VecNorm, TransformedEnv
from torchrl.envs import GymEnv, ParallelEnv


def _test_vecnorm_subproc(idx, queue_out: mp.Queue, queue_in: mp.Queue):
    td = queue_in.get()
    env = GymEnv("Pendulum-v0")
    env.set_seed(idx)
    t = VecNorm(shared_td = td)
    env = TransformedEnv(env, t)
    for _ in range(10):
        env.rand_step()
    queue_out.put(True)
    msg = queue_in.get()
    assert msg == "all_done"
    obs_sum = t._td.get("next_observation_sum").clone()
    obs_ssq = t._td.get("next_observation_ssq").clone()
    obs_count = t._td.get("next_observation_ssq").clone()
    reward_sum = t._td.get("reward_sum").clone()
    reward_ssq = t._td.get("reward_ssq").clone()
    reward_count = t._td.get("reward_ssq").clone()

    queue_out.put((
        obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count
    ))

@pytest.mark.parametrize("nprc", [2,5])
def test_vecnorm_parallel(nprc):
    queues = []
    prcs = []
    td = VecNorm.build_td_for_shared_vecnorm(GymEnv("Pendulum-v0"))
    for idx in range(nprc):
        prc_queue_in = mp.Queue(1)
        prc_queue_out = mp.Queue(1)
        p = mp.Process(target=_test_vecnorm_subproc, args=(idx, prc_queue_in, prc_queue_out, ))
        p.start()
        prc_queue_out.put(td)
        prcs.append(p)
        queues.append((prc_queue_in, prc_queue_out))

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
        tup = queues[idx][0].get()
        _obs_sum, _obs_ssq, _obs_count, _reward_sum, _reward_ssq, _reward_count = tup
        assert (obs_sum == _obs_sum).all()
        assert (obs_ssq == _obs_ssq).all()
        assert (obs_count == _obs_count).all()
        assert (reward_sum == _reward_sum).all()
        assert (reward_ssq == _reward_ssq).all()
        assert (reward_count == _reward_count).all()

        obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count = _obs_sum, _obs_ssq, _obs_count, _reward_sum, _reward_ssq, _reward_count

@pytest.mark.parametrize("parallel", [False, True])
def test_vecnorm(parallel, thr=0.2):
    torch.manual_seed(0)

    if parallel:
        env = ParallelEnv(num_workers=5, create_env_fn=lambda: GymEnv("Pendulum-v0"))
    else:
        env = GymEnv("Pendulum-v0")
    env.set_seed(0)
    t = VecNorm()
    env = TransformedEnv(env, t)
    env.reset()
    tds = []
    for _ in range(10000):
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


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, '--capture', 'no'])
