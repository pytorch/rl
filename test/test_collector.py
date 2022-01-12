import numpy as np
import pytest
import torch
from torch import nn

from mocking_classes import DiscreteActionConvMockEnv, DiscreteActionVecMockEnv, DiscreteActionVecPolicy, \
    DiscreteActionConvPolicy
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.envs import ParallelEnv


def make_make_env(env_name="conv"):
    def make_transformed_env():
        if env_name == "conv":
            return DiscreteActionConvMockEnv()
        elif env_name == "vec":
            return DiscreteActionVecMockEnv()

    return make_transformed_env


def dummypolicy_vec():
    policy = DiscreteActionVecPolicy()
    return policy


def dummypolicy_conv():
    policy = DiscreteActionConvPolicy()
    return policy


def make_policy(env):
    if env == "conv":
        return dummypolicy_conv()
    elif env == "vec":
        return dummypolicy_vec()
    else:
        raise NotImplementedError


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_concurrent_collector_consistency(num_env, env_name, seed=100):
    print("concurrent")
    if num_env == 1:
        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env
    else:
        def env_fn(seed):
            env = ParallelEnv(num_workers=num_env, create_env_fn=make_make_env(env_name))
            env.set_seed(seed)
            return env
    policy = make_policy(env_name)

    print("\tserial")
    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=20,
        max_steps_per_traj=20,
        total_frames=200,
        device='cpu',
        pin_memory=False,
    )
    for i, d in enumerate(collector):
        if i == 0:
            b1 = d
        elif i == 1:
            b2 = d
        else:
            break
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)

    print("\tparallel")
    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=20,
        max_steps_per_traj=20,
        total_frames=200,
        pin_memory=False,
    )
    for i, d in enumerate(ccollector):
        if i == 0:
            b1c = d
        elif i == 1:
            b2c = d
        else:
            break
    print(b1c)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1c, b2c)
    assert_allclose_td(b1c, b1)
    assert_allclose_td(b2c, b2)
    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec", "conv"])
def test_concurrent_collector_seed(num_env, env_name, seed=100):
    print("concurrent")
    if num_env == 1:
        def env_fn():
            env = make_make_env(env_name)()
            return env
    else:
        def env_fn():
            env = ParallelEnv(num_workers=num_env, create_env_fn=make_make_env(env_name))
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)
    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={},
        policy=policy,
        frames_per_batch=20,
        max_steps_per_traj=20,
        total_frames=300,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, data in enumerate(ccollector):
        if i == 0:
            b1 = data
            ccollector.set_seed(seed)
        elif i == 1:
            b2 = data
        elif i == 2:
            b3 = data
        else:
            break
    assert_allclose_td(b1, b2)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b3)
    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_collector_consistency(num_env, env_name, seed=100):
    print("collector")
    if num_env == 1:
        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env
    else:
        def env_fn(seed):
            env = ParallelEnv(num_workers=num_env, create_env_fn=make_make_env(env_name))
            env.set_seed(seed)
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)

    # Get a single rollout with dummypolicy
    env = env_fn(seed)
    rollout1a = env.rollout(policy=policy, n_steps=20, auto_reset=True)
    env.set_seed(seed)
    rollout1b = env.rollout(policy=policy, n_steps=20, auto_reset=True)
    rollout2 = env.rollout(policy=policy, n_steps=20, auto_reset=True)
    assert assert_allclose_td(rollout1a, rollout1b)
    with pytest.raises(AssertionError):
        assert_allclose_td(rollout1a, rollout2)
    env.close()

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=20,
        max_steps_per_traj=20,
        total_frames=200,
        device='cpu',
        pin_memory=False,
    )
    collector = iter(collector)
    b1 = next(collector)
    b2 = next(collector)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)

    if num_env == 1:
        # rollouts collected through DataCollector are padded using pad_sequence, which introduces a first dimension
        rollout1a = rollout1a.unsqueeze(0)
    assert (
            rollout1a.batch_size == b1.batch_size
    ), f"got batch_size {rollout1a.batch_size} and {b1.batch_size}"

    assert_allclose_td(rollout1a, b1.select(*rollout1a.keys()))


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("collector_class", [SyncDataCollector, aSyncDataCollector])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_traj_len_consistency(num_env, env_name, collector_class, seed=100):
    """
    Tests that various frames_per_batch lead to the same results
    Args:
        num_env:
        env_name:
        seed:

    Returns:

    """
    if num_env == 1:
        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env
    else:
        def env_fn(seed):
            env = ParallelEnv(num_workers=num_env, create_env_fn=make_make_env(env_name))
            env.set_seed(seed)
            return env

    max_steps_per_traj = 20

    policy = make_policy(env_name)

    collector1 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=1,
        max_steps_per_traj=max_steps_per_traj,
        total_frames=2 * num_env * max_steps_per_traj,
        device='cpu',
        seed=seed,
        pin_memory=False
    )
    data1 = []
    for d in collector1:
        data1.append(d)
    data1 = torch.cat(data1, 1)

    collector10 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=10,
        max_steps_per_traj=max_steps_per_traj,
        total_frames=2 * num_env * max_steps_per_traj,
        device='cpu',
        seed=seed,
        pin_memory=False
    )
    data10 = []
    for d in collector10:
        data10.append(d)
    data10 = torch.cat(data10, 1)

    collector20 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={'seed': seed},
        policy=policy,
        frames_per_batch=20,
        max_steps_per_traj=max_steps_per_traj,
        total_frames=2 * num_env * max_steps_per_traj,
        device='cpu',
        seed=seed,
        pin_memory=False
    )
    data20 = []
    for d in collector20:
        data20.append(d)
    data20 = torch.cat(data20, 1)
    data20 = data20[:, :data10.shape[1]]

    assert_allclose_td(data1, data20)
    assert_allclose_td(data10, data20)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == "__main__":
    pytest.main([__file__, '--capture', 'no'])
