# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch
from _utils_internal import generate_seeds
from mocking_classes import (
    DiscreteActionConvMockEnv,
    DiscreteActionVecMockEnv,
    DiscreteActionVecPolicy,
    DiscreteActionConvPolicy,
    ContinuousActionVecMockEnv,
    MockSerialEnv,
)
from torch import nn
from torchrl._utils import seed_generator
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from torchrl.collectors.collectors import (
    RandomPolicy,
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
)
from torchrl.data import (
    CompositeSpec,
    NdUnboundedContinuousTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.envs import EnvCreator
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.modules import LSTMNet, TensorDictModule
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper, Actor

# torch.set_default_dtype(torch.double)


class ParametricPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, obs):
        max_obs = (obs == obs.max(dim=-1, keepdim=True)[0]).cumsum(-1).argmax(-1)
        k = obs.shape[-1]
        max_obs = (max_obs + 1) % k
        action = torch.nn.functional.one_hot(max_obs, k)
        return action


class ParametricPolicy(Actor):
    def __init__(self):
        super().__init__(
            ParametricPolicyNet(),
            in_keys=["observation"],
        )


def make_make_env(env_name="conv"):
    def make_transformed_env(seed=None):
        if env_name == "conv":
            env = DiscreteActionConvMockEnv()
        elif env_name == "vec":
            env = DiscreteActionVecMockEnv()
        if seed is not None:
            env.set_seed(seed)
        return env

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


def _is_consistent_device_type(
    device_type, policy_device_type, passing_device_type, tensordict_device_type
):
    if passing_device_type is None:
        if device_type is None:
            if policy_device_type is None:
                return tensordict_device_type == "cpu"

            return tensordict_device_type == policy_device_type

        return tensordict_device_type == device_type

    return tensordict_device_type == passing_device_type


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("device", ["cuda", "cpu", None])
@pytest.mark.parametrize("policy_device", ["cuda", "cpu", None])
@pytest.mark.parametrize("passing_device", ["cuda", "cpu", None])
def test_output_device_consistency(
    num_env, device, policy_device, passing_device, seed=40
):
    if (
        device == "cuda" or policy_device == "cuda" or passing_device == "cuda"
    ) and not torch.cuda.is_available():
        pytest.skip("cuda is not available")

    _device = "cuda:0" if device == "cuda" else device
    _policy_device = "cuda:0" if policy_device == "cuda" else policy_device
    _passing_device = "cuda:0" if passing_device == "cuda" else passing_device

    if num_env == 1:

        def env_fn(seed):
            env = make_make_env("vec")()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env("vec"),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

    if _policy_device is None:
        policy = make_policy("vec")
    else:
        policy = ParametricPolicy().to(torch.device(_policy_device))

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=_device,
        passing_device=_passing_device,
        pin_memory=False,
    )
    for _, d in enumerate(collector):
        assert _is_consistent_device_type(
            device, policy_device, passing_device, d.device.type
        )
        break

    collector.shutdown()

    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=_device,
        passing_device=_passing_device,
        pin_memory=False,
    )

    for _, d in enumerate(ccollector):
        assert _is_consistent_device_type(
            device, policy_device, passing_device, d.device.type
        )
        break

    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_concurrent_collector_consistency(num_env, env_name, seed=40):
    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env(env_name),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
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
    collector.shutdown()

    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        pin_memory=False,
    )
    for i, d in enumerate(ccollector):
        if i == 0:
            b1c = d
        elif i == 1:
            b2c = d
        else:
            break
    with pytest.raises(AssertionError):
        assert_allclose_td(b1c, b2c)

    assert_allclose_td(b1c, b1)
    assert_allclose_td(b2c, b2)

    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec"])
def test_collector_done_persist(num_env, env_name, seed=5):
    if num_env == 1:

        def env_fn(seed):
            env = MockSerialEnv(device="cpu")
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            def make_env(seed):
                env = MockSerialEnv(device="cpu")
                env.set_seed(seed)
                return env

            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_env,
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
                allow_step_when_done=True,
            )
            env.set_seed(seed)
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=200 * num_env,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
        pin_memory=False,
        reset_when_done=False,
    )
    for _, d in enumerate(collector):  # noqa
        break

    assert (d["done"].sum(-2) >= 1).all()
    assert torch.unique(d["traj_ids"], dim=-1).shape[-1] == 1

    del collector


# TODO: design a test that ensures that collectors are interrupted even if __del__ is not called
# @pytest.mark.parametrize("should_shutdown", [True, False])
# def test_shutdown_collector(should_shutdown, num_env=3, env_name="vec", seed=40):
#     def env_fn(seed):
#         env = ParallelEnv(
#             num_workers=num_env,
#             create_env_fn=make_make_env(env_name),
#             create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
#         )
#         return env
#
#     policy = make_policy(env_name)
#
#     ccollector = aSyncDataCollector(
#         create_env_fn=env_fn,
#         create_env_kwargs={"seed": seed},
#         policy=policy,
#         frames_per_batch=20,
#         max_frames_per_traj=2000,
#         total_frames=20000,
#         pin_memory=False,
#     )
#     for i, d in enumerate(ccollector):
#         if i == 0:
#             b1c = d
#         elif i == 1:
#             b2c = d
#         else:
#             break
#     with pytest.raises(AssertionError):
#         assert_allclose_td(b1c, b2c)
#
#     if should_shutdown:
#         ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec", "conv"])
def test_collector_batch_size(num_env, env_name, seed=100):
    if num_env == 1:

        def env_fn():
            env = make_make_env(env_name)()
            return env

    else:

        def env_fn():
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)
    num_workers = 4
    frames_per_batch = 20
    ccollector = MultiaSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, b in enumerate(ccollector):
        assert b.numel() == -(-frames_per_batch // num_env) * num_env
        if i == 5:
            break
    ccollector.shutdown()

    ccollector = MultiSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
        pin_memory=False,
    )
    ccollector.set_seed(seed)
    for i, b in enumerate(ccollector):
        assert (
            b.numel()
            == -(-frames_per_batch // num_env // num_workers) * num_env * num_workers
        )
        if i == 5:
            break
    ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("env_name", ["vec", "conv"])
def test_concurrent_collector_seed(num_env, env_name, seed=100):
    if num_env == 1:

        def env_fn():
            env = make_make_env(env_name)()
            return env

    else:

        def env_fn():
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)
    ccollector = aSyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=20,
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
    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env,
                create_env_fn=make_make_env(env_name),
                create_env_kwargs=[{"seed": s} for s in generate_seeds(seed, num_env)],
            )
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)

    # Get a single rollout with dummypolicy
    env = env_fn(seed)
    rollout1a = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    env.set_seed(seed)
    rollout1b = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    rollout2 = env.rollout(policy=policy, max_steps=20, auto_reset=True)
    assert_allclose_td(rollout1a, rollout1b)
    with pytest.raises(AssertionError):
        assert_allclose_td(rollout1a, rollout2)
    env.close()

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20 * num_env,
        max_frames_per_traj=20,
        total_frames=200,
        device="cpu",
        pin_memory=False,
    )
    collector_iter = iter(collector)
    b1 = next(collector_iter)
    b2 = next(collector_iter)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)

    if num_env == 1:
        # rollouts collected through DataCollector are padded using pad_sequence, which introduces a first dimension
        rollout1a = rollout1a.unsqueeze(0)
    assert (
        rollout1a.batch_size == b1.batch_size
    ), f"got batch_size {rollout1a.batch_size} and {b1.batch_size}"

    assert_allclose_td(rollout1a, b1.select(*rollout1a.keys()))
    collector.shutdown()


@pytest.mark.parametrize("num_env", [1, 3])
@pytest.mark.parametrize("collector_class", [SyncDataCollector, aSyncDataCollector])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_traj_len_consistency(num_env, env_name, collector_class, seed=100):
    """
    Tests that various frames_per_batch lead to the same results
    """
    if num_env == 1:

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

    else:

        def env_fn(seed):
            env = ParallelEnv(
                num_workers=num_env, create_env_fn=make_make_env(env_name)
            )
            env.set_seed(seed)
            return env

    max_frames_per_traj = 20

    policy = make_policy(env_name)

    def make_frames_per_batch(frames_per_batch):
        return -(-frames_per_batch // num_env) * num_env

    collector1 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=1 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data1 = []
    for d in collector1:
        data1.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    data1 = torch.cat(data1, 1)
    data1 = data1[:, :max_frames_per_traj]

    collector1.shutdown()
    del collector1

    collector10 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=10 * num_env,
        max_frames_per_traj=20,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data10 = []
    for d in collector10:
        data10.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    data10 = torch.cat(data10, 1)
    data10 = data10[:, :max_frames_per_traj]

    collector10.shutdown()
    del collector10

    collector20 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=20 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
        seed=seed,
        pin_memory=False,
    )
    count = 0
    data20 = []
    for d in collector20:
        data20.append(d)
        count += d.shape[1]
        if count > max_frames_per_traj:
            break

    collector20.shutdown()
    del collector20
    data20 = torch.cat(data20, 1)
    data20 = data20[:, :max_frames_per_traj]

    assert_allclose_td(data1, data20)
    assert_allclose_td(data10, data20)


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize("static_seed", [True, False])
def test_collector_vecnorm_envcreator(static_seed):
    """
    High level test of the following pipeline:
     (1) Design a function that creates an environment with VecNorm
     (2) Wrap that function in an EnvCreator to instantiate the shared tensordict
     (3) Create a ParallelEnv that dispatches this env across workers
     (4) Run several ParallelEnv synchronously
    The function tests that the tensordict gathered from the workers match at certain moments in time, and that they
    are modified after the collector is run for more steps.

    """
    from torchrl.envs.libs.gym import GymEnv

    num_envs = 4
    env_make = EnvCreator(lambda: TransformedEnv(GymEnv("Pendulum-v1"), VecNorm()))
    env_make = ParallelEnv(num_envs, env_make)

    policy = RandomPolicy(env_make.action_spec)
    num_data_collectors = 2
    c = MultiSyncDataCollector(
        [env_make] * num_data_collectors, policy=policy, total_frames=int(1e6)
    )

    init_seed = 0
    new_seed = c.set_seed(init_seed, static_seed=static_seed)
    if static_seed:
        assert new_seed == init_seed
    else:
        assert new_seed != init_seed

    seed = init_seed
    for _ in range(num_envs * num_data_collectors):
        seed = seed_generator(seed)
    if not static_seed:
        assert new_seed == seed
    else:
        assert new_seed != seed

    c_iter = iter(c)
    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td1 = s["worker0"]["env_state_dict"]["worker3"]["_extra_state"]["td"].clone()
    td2 = s["worker1"]["env_state_dict"]["worker0"]["_extra_state"]["td"].clone()
    assert (td1 == td2).all()

    next(c_iter)
    next(c_iter)

    s = c.state_dict()

    td3 = s["worker0"]["env_state_dict"]["worker3"]["_extra_state"]["td"].clone()
    td4 = s["worker1"]["env_state_dict"]["worker0"]["_extra_state"]["td"].clone()
    assert (td3 == td4).all()
    assert (td1 != td4).any()

    del c


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
def test_update_weights(use_async):
    def create_env():
        return ContinuousActionVecMockEnv()

    n_actions = ContinuousActionVecMockEnv().action_spec.shape[-1]
    policy = TensorDictModule(
        torch.nn.LazyLinear(n_actions), in_keys=["observation"], out_keys=["action"]
    )
    policy(create_env().reset())

    collector_class = (
        MultiSyncDataCollector if not use_async else MultiaSyncDataCollector
    )
    collector = collector_class(
        [create_env] * 3,
        policy=policy,
        devices=[torch.device("cuda:0")] * 3,
        passing_devices=[torch.device("cuda:0")] * 3,
    )
    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            torch.testing.assert_close(
                state_dict[f"worker{worker}"]["policy_state_dict"][k],
                policy_state_dict[k].cpu(),
            )

    # change policy weights
    for p in policy.parameters():
        p.data += torch.randn_like(p)

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    # check they don't match
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    state_dict[f"worker{worker}"]["policy_state_dict"][k],
                    policy_state_dict[k].cpu(),
                )

    # update weights
    collector.update_policy_weights_()

    # collect state_dict
    state_dict = collector.state_dict()
    policy_state_dict = policy.state_dict()
    for worker in range(3):
        for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
            torch.testing.assert_close(
                state_dict[f"worker{worker}"]["policy_state_dict"][k],
                policy_state_dict[k].cpu(),
            )

    collector.shutdown()
    del collector


@pytest.mark.parametrize(
    "collector_class",
    [MultiSyncDataCollector, MultiaSyncDataCollector, SyncDataCollector],
)
@pytest.mark.parametrize("exclude", [True, False])
def test_excluded_keys(collector_class, exclude):
    if not exclude and collector_class is not SyncDataCollector:
        pytest.skip("defining _exclude_private_keys is not possible")

    def make_env():
        return ContinuousActionVecMockEnv()

    dummy_env = make_env()
    obs_spec = dummy_env.observation_spec["next_observation"]
    policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
    policy = Actor(policy_module, spec=dummy_env.action_spec)
    policy_explore = OrnsteinUhlenbeckProcessWrapper(policy)

    collector_kwargs = {
        "create_env_fn": make_env,
        "policy": policy_explore,
        "frames_per_batch": 30,
    }
    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(3)
        ]

    collector = collector_class(**collector_kwargs)
    collector._exclude_private_keys = exclude
    for b in collector:
        keys = b.keys()
        if exclude:
            assert not any(key.startswith("_") for key in keys)
        else:
            assert any(key.startswith("_") for key in keys)
        break
    collector.shutdown()
    dummy_env.close()


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class",
    [
        SyncDataCollector,
        MultiaSyncDataCollector,
        MultiSyncDataCollector,
    ],
)
@pytest.mark.parametrize("init_random_frames", [0, 50])
@pytest.mark.parametrize("explicit_spec", [True, False])
def test_collector_output_keys(collector_class, init_random_frames, explicit_spec):
    from torchrl.envs.libs.gym import GymEnv

    out_features = 1
    hidden_size = 12
    total_frames = 200
    frames_per_batch = 20
    num_envs = 3

    net = LSTMNet(
        out_features,
        {"input_size": hidden_size, "hidden_size": hidden_size},
        {"out_features": hidden_size},
    )

    policy_kwargs = {
        "module": net,
        "in_keys": ["observation", "hidden1", "hidden2"],
        "out_keys": ["action", "hidden1", "hidden2", "next_hidden1", "next_hidden2"],
    }
    if explicit_spec:
        hidden_spec = NdUnboundedContinuousTensorSpec((1, hidden_size))
        policy_kwargs["spec"] = CompositeSpec(
            action=UnboundedContinuousTensorSpec(),
            hidden1=hidden_spec,
            hidden2=hidden_spec,
            next_hidden1=hidden_spec,
            next_hidden2=hidden_spec,
        )

    policy = TensorDictModule(**policy_kwargs)

    env_maker = lambda: GymEnv("Pendulum-v1")

    policy(env_maker().reset())

    collector_kwargs = {
        "create_env_fn": env_maker,
        "policy": policy,
        "total_frames": total_frames,
        "frames_per_batch": frames_per_batch,
        "init_random_frames": init_random_frames,
    }

    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(num_envs)
        ]

    collector = collector_class(**collector_kwargs)

    keys = [
        "action",
        "done",
        "hidden1",
        "hidden2",
        "mask",
        "next_hidden1",
        "next_hidden2",
        "next_observation",
        "observation",
        "reward",
        "step_count",
        "traj_ids",
    ]
    b = next(iter(collector))

    assert set(b.keys()) == set(keys)
    collector.shutdown()
    del collector


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
