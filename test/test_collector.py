import numpy as np
import pytest
import torch
from torch import nn

from mocking_classes import DiscreteActionConvMockEnv, DiscreteActionVecMockEnv, DiscreteActionVecPolicy, \
    DiscreteActionConvPolicy
from torchrl.agents.env_creator import EnvCreator
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy, MultiSyncDataCollector, MultiaSyncDataCollector
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.data.transforms import TransformedEnv, VecNorm
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import _has_gym


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


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
def test_collector_vecnorm_envcreator():
    """
    High level test of the following pipeline:
     (1) Design a function that creates an environment with VecNorm
     (2) Wrap that function in an EnvCreator to instantiate the shared tensordict
     (3) Create a ParallelEnv that dispatches this env across workers
     (4) Run several ParallelEnv synchronously
    The function tests that the tensordict gathered from the workers match at certain moments in time, and that they
    are modified after the collector is run for more steps.

    """
    from torchrl.envs import GymEnv
    env_make = EnvCreator(lambda: TransformedEnv(GymEnv("Pendulum-v0"), VecNorm()))
    env_make = ParallelEnv(4, env_make)

    policy = RandomPolicy(env_make.action_spec)
    c = MultiSyncDataCollector([env_make, env_make], policy=policy, total_frames=int(1e6))
    final_seed = c.set_seed(0)
    assert final_seed == 7

    c_iter = iter(c)
    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td1 = s["worker0"]["worker3"]['_extra_state'].clone()
    td2 = s["worker1"]["worker0"]['_extra_state'].clone()
    assert (td1 == td2).all()

    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td3 = s["worker0"]["worker3"]['_extra_state'].clone()
    td4 = s["worker1"]["worker0"]['_extra_state'].clone()
    assert (td3 == td4).all()
    assert (td1 != td4).any()

    del c


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.skipif(torch.cuda.device_count() <= 1, reason="no cuda device found")
def test_update_weights(use_async):
    policy = torch.nn.Linear(3, 4).cuda(1)
    policy.share_memory()
    collector_class = MultiSyncDataCollector if not use_async else MultiaSyncDataCollector
    collector = collector_class(
        [lambda: DiscreteActionVecMockEnv()] * 3,
        policy=policy,
        devices=[torch.device("cuda:0")] * 3,
        passing_devices=[torch.device("cuda:0")] * 3,
    )
    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f'worker{worker}']['policy_state_dict']:
            torch.testing.assert_allclose(state_dict[f'worker{worker}']['policy_state_dict'][k],
                                          policy_state_dict[k].cpu())

    # change policy weights
    for p in policy.parameters():
        p.data += torch.randn_like(p)

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    # check they don't match
    for worker in range(3):
        for k in state_dict[f'worker{worker}']['policy_state_dict']:
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(state_dict[f'worker{worker}']['policy_state_dict'][k],
                                              policy_state_dict[k].cpu())

    # update weights
    collector.update_policy_weights_()

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f'worker{worker}']['policy_state_dict']:
            torch.testing.assert_allclose(state_dict[f'worker{worker}']['policy_state_dict'][k],
                                          policy_state_dict[k].cpu())

    collector.shutdown()
    del collector

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == "__main__":
    pytest.main([__file__, '--capture', 'no'])
