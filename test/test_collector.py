# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import gc

import sys

import numpy as np
import pytest
import torch

from _utils_internal import (
    check_rollout_consistency_multikey_env,
    decorate_thread_sub_func,
    generate_seeds,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    retry,
)
from mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionConvPolicy,
    DiscreteActionVecMockEnv,
    DiscreteActionVecPolicy,
    HeterogeneousCountingEnv,
    HeterogeneousCountingEnvPolicy,
    MockSerialEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)
from tensordict import assert_allclose_td, LazyStackedTensorDict, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential

from torch import nn
from torchrl._utils import _replace_last, logger as torchrl_logger, prod, seed_generator
from torchrl.collectors import aSyncDataCollector, SyncDataCollector
from torchrl.collectors.collectors import (
    _Interruptor,
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data import (
    CompositeSpec,
    LazyTensorStorage,
    ReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import (
    EnvBase,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    SerialEnv,
    StepCounter,
)
from torchrl.envs.libs.gym import _has_gym, gym_backend, GymEnv, set_gym_backend
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    check_env_specs,
    PARTIAL_MISSING_ERR,
    RandomPolicy,
)
from torchrl.modules import Actor, LSTMNet, OrnsteinUhlenbeckProcessWrapper, SafeModule

# torch.set_default_dtype(torch.double)
IS_WINDOWS = sys.platform == "win32"
IS_OSX = sys.platform == "darwin"
PYTHON_3_10 = sys.version_info.major == 3 and sys.version_info.minor == 10
PYTHON_3_7 = sys.version_info.major == 3 and sys.version_info.minor == 7


class WrappablePolicy(nn.Module):
    def __init__(self, out_features: int, multiple_outputs: bool = False):
        super().__init__()
        self.multiple_outputs = multiple_outputs
        self.linear = nn.LazyLinear(out_features)

    def forward(self, observation):
        output = self.linear(observation)
        if self.multiple_outputs:
            return output, output.sum(), output.min(), output.max()
        return self.linear(observation)


class TensorDictCompatiblePolicy(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.in_keys = ["observation"]
        self.out_keys = ["action"]
        self.linear = nn.LazyLinear(out_features)

    def forward(self, tensordict):
        return tensordict.set(
            self.out_keys[0], self.linear(tensordict.get(self.in_keys[0]))
        )


class UnwrappablePolicy(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.linear = nn.Linear(2, out_features)

    def forward(self, observation, other_stuff):
        return self.linear(observation), other_stuff.sum()


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


# def _is_consistent_device_type(
#     device_type, policy_device_type, storing_device_type, tensordict_device_type
# ):
#     if storing_device_type is None:
#         if device_type is None:
#             if policy_device_type is None:
#                 return tensordict_device_type == "cpu"
#
#             return tensordict_device_type == policy_device_type
#
#         return tensordict_device_type == device_type
#
#     return tensordict_device_type == storing_device_type


class TestCollectorDevices:
    class DeviceLessEnv(EnvBase):
        # receives data on cpu, outputs on gpu -- tensordict has no device
        def __init__(self, default_device):
            self.default_device = default_device
            super().__init__(device=None)
            self.observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec((), device=default_device)
            )
            self.reward_spec = UnboundedContinuousTensorSpec(1, device=default_device)
            self.full_done_spec = CompositeSpec(
                done=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
                truncated=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
                terminated=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
            )
            self.action_spec = UnboundedContinuousTensorSpec((), device=None)
            assert self.device is None
            assert self.full_observation_spec is not None
            assert self.full_done_spec is not None
            assert self.full_state_spec is not None
            assert self.full_action_spec is not None
            assert self.full_reward_spec is not None

        def _step(self, tensordict):
            assert tensordict.device is None
            with torch.device(self.default_device):
                return TensorDict(
                    {
                        "observation": torch.zeros(()),
                        "reward": torch.zeros((1,)),
                        "done": torch.zeros((1,), dtype=torch.bool),
                        "terminated": torch.zeros((1,), dtype=torch.bool),
                        "truncated": torch.zeros((1,), dtype=torch.bool),
                    },
                    batch_size=[],
                    device=None,
                )

        def _reset(self, tensordict=None):
            with torch.device(self.default_device):
                return TensorDict(
                    {
                        "observation": torch.zeros(()),
                        "done": torch.zeros((1,), dtype=torch.bool),
                        "terminated": torch.zeros((1,), dtype=torch.bool),
                        "truncated": torch.zeros((1,), dtype=torch.bool),
                    },
                    batch_size=[],
                    device=None,
                )

        def _set_seed(self, seed: int | None = None):
            return seed

    class EnvWithDevice(EnvBase):
        def __init__(self, default_device):
            self.default_device = default_device
            super().__init__(device=self.default_device)
            self.observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    (), device=self.default_device
                )
            )
            self.reward_spec = UnboundedContinuousTensorSpec(
                1, device=self.default_device
            )
            self.full_done_spec = CompositeSpec(
                done=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
                truncated=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
                terminated=UnboundedContinuousTensorSpec(
                    1, dtype=torch.bool, device=self.default_device
                ),
                device=self.default_device,
            )
            self.action_spec = UnboundedContinuousTensorSpec(
                (), device=self.default_device
            )
            assert self.device == torch.device(self.default_device)
            assert self.full_observation_spec is not None
            assert self.full_done_spec is not None
            assert self.full_state_spec is not None
            assert self.full_action_spec is not None
            assert self.full_reward_spec is not None

        def _step(self, tensordict):
            assert tensordict.device == torch.device(self.default_device)
            with torch.device(self.default_device):
                return TensorDict(
                    {
                        "observation": torch.zeros(()),
                        "reward": torch.zeros((1,)),
                        "done": torch.zeros((1,), dtype=torch.bool),
                        "terminated": torch.zeros((1,), dtype=torch.bool),
                        "truncated": torch.zeros((1,), dtype=torch.bool),
                    },
                    batch_size=[],
                    device=self.default_device,
                )

        def _reset(self, tensordict=None):
            with torch.device(self.default_device):
                return TensorDict(
                    {
                        "observation": torch.zeros(()),
                        "done": torch.zeros((1,), dtype=torch.bool),
                        "terminated": torch.zeros((1,), dtype=torch.bool),
                        "truncated": torch.zeros((1,), dtype=torch.bool),
                    },
                    batch_size=[],
                    device=self.default_device,
                )

        def _set_seed(self, seed: int | None = None):
            return seed

    class DeviceLessPolicy(TensorDictModuleBase):
        in_keys = ["observation"]
        out_keys = ["action"]

        # receives data on gpu and outputs on cpu
        def forward(self, tensordict):
            assert tensordict.device is None
            return tensordict.set("action", torch.zeros((), device="cpu"))

    class PolicyWithDevice(TensorDictModuleBase):
        in_keys = ["observation"]
        out_keys = ["action"]
        # receives and sends data on gpu
        default_device = "cuda:0" if torch.cuda.device_count() else "cpu"

        def forward(self, tensordict):
            assert tensordict.device == torch.device(self.default_device)
            return tensordict.set("action", torch.zeros((), device=self.default_device))

    @pytest.mark.parametrize("main_device", get_default_devices())
    @pytest.mark.parametrize("storing_device", [None, *get_default_devices()])
    def test_output_device(self, main_device, storing_device):

        # env has no device, policy is strictly on GPU
        device = None
        env_device = None
        policy_device = main_device
        env = self.DeviceLessEnv(main_device)
        policy = self.PolicyWithDevice()
        collector = SyncDataCollector(
            env,
            policy,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            frames_per_batch=1,
            total_frames=10,
        )
        for data in collector:  # noqa: B007
            break

        assert data.device == storing_device

        # env is on cuda, policy has no device
        device = None
        env_device = main_device
        policy_device = None
        env = self.EnvWithDevice(main_device)
        policy = self.DeviceLessPolicy()
        collector = SyncDataCollector(
            env,
            policy,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            frames_per_batch=1,
            total_frames=10,
        )
        for data in collector:  # noqa: B007
            break
        assert data.device == storing_device

        # env and policy are on device
        device = main_device
        env_device = None
        policy_device = None
        env = self.EnvWithDevice(main_device)
        policy = self.PolicyWithDevice()
        collector = SyncDataCollector(
            env,
            policy,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            frames_per_batch=1,
            total_frames=10,
        )
        for data in collector:  # noqa: B007
            break
        assert data.device == main_device

        # same but more specific
        device = None
        env_device = main_device
        policy_device = main_device
        env = self.EnvWithDevice(main_device)
        policy = self.PolicyWithDevice()
        collector = SyncDataCollector(
            env,
            policy,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            frames_per_batch=1,
            total_frames=10,
        )
        for data in collector:  # noqa: B007
            break
        assert data.device == main_device

        # none has a device
        device = None
        env_device = None
        policy_device = None
        env = self.DeviceLessEnv(main_device)
        policy = self.DeviceLessPolicy()
        collector = SyncDataCollector(
            env,
            policy,
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            frames_per_batch=1,
            total_frames=10,
        )
        for data in collector:  # noqa: B007
            break
        assert data.device == storing_device


# @pytest.mark.skipif(
#     IS_WINDOWS and PYTHON_3_10,
#     reason="Windows Access Violation in torch.multiprocessing / BrokenPipeError in multiprocessing.connection",
# )
# @pytest.mark.parametrize("num_env", [2])
# @pytest.mark.parametrize("device", ["cuda", "cpu", None])
# @pytest.mark.parametrize("policy_device", ["cuda", "cpu", None])
# @pytest.mark.parametrize("storing_device", ["cuda", "cpu", None])
# def test_output_device_consistency(
#     num_env, device, policy_device, storing_device, seed=40
# ):
#     if (
#         device == "cuda" or policy_device == "cuda" or storing_device == "cuda"
#     ) and not torch.cuda.is_available():
#         pytest.skip("cuda is not available")
#
#     if IS_WINDOWS and PYTHON_3_7:
#         if device == "cuda" and policy_device == "cuda" and device is None:
#             pytest.skip(
#                 "BrokenPipeError in multiprocessing.connection with Python 3.7 on Windows"
#             )
#
#     _device = "cuda:0" if device == "cuda" else device
#     _policy_device = "cuda:0" if policy_device == "cuda" else policy_device
#     _storing_device = "cuda:0" if storing_device == "cuda" else storing_device
#
#     if num_env == 1:
#
#         def env_fn(seed):
#             env = make_make_env("vec")()
#             env.set_seed(seed)
#             return env
#
#     else:
#
#         def env_fn(seed):
#             # 1226: faster execution
#             # env = ParallelEnv(
#             env = SerialEnv(
#                 num_workers=num_env,
#                 create_env_fn=make_make_env("vec"),
#                 create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
#             )
#             return env
#
#     if _policy_device is None:
#         policy = make_policy("vec")
#     else:
#         policy = ParametricPolicy().to(torch.device(_policy_device))
#
#     collector = SyncDataCollector(
#         create_env_fn=env_fn,
#         create_env_kwargs={"seed": seed},
#         policy=policy,
#         frames_per_batch=20,
#         max_frames_per_traj=2000,
#         total_frames=20000,
#         device=_device,
#         storing_device=_storing_device,
#     )
#     for _, d in enumerate(collector):
#         assert _is_consistent_device_type(
#             device, policy_device, storing_device, d.device.type
#         )
#         break
#     assert d.names[-1] == "time"
#
#     collector.shutdown()
#
#     ccollector = aSyncDataCollector(
#         create_env_fn=env_fn,
#         create_env_kwargs={"seed": seed},
#         policy=policy,
#         frames_per_batch=20,
#         max_frames_per_traj=2000,
#         total_frames=20000,
#         device=_device,
#         storing_device=_storing_device,
#     )
#
#     for _, d in enumerate(ccollector):
#         assert _is_consistent_device_type(
#             device, policy_device, storing_device, d.device.type
#         )
#         break
#     assert d.names[-1] == "time"
#
#     ccollector.shutdown()
#     del ccollector


@pytest.mark.parametrize("num_env", [1, 2])
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
    )
    for i, d in enumerate(collector):
        if i == 0:
            b1 = d
        elif i == 1:
            b2 = d
        else:
            break
    assert d.names[-1] == "time"
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
    )
    for i, d in enumerate(ccollector):
        if i == 0:
            b1c = d
        elif i == 1:
            b2c = d
        else:
            break
    assert d.names[-1] == "time"
    with pytest.raises(AssertionError):
        assert_allclose_td(b1c, b2c)

    assert_allclose_td(b1c, b1)
    assert_allclose_td(b2c, b2)

    ccollector.shutdown()


@pytest.mark.skipif(not _has_gym, reason="gym library is not installed")
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "constr",
    [
        functools.partial(split_trajectories, prefix="collector"),
        functools.partial(split_trajectories),
        functools.partial(split_trajectories, trajectory_key=("collector", "traj_ids")),
    ],
)
def test_collector_env_reset(constr, parallel):
    torch.manual_seed(0)

    def make_env():
        # This is currently necessary as the methods in GymWrapper may have mismatching backend
        # versions.
        with set_gym_backend(gym_backend()):
            return TransformedEnv(GymEnv(PONG_VERSIONED(), frame_skip=4), StepCounter())

    if parallel:
        env = ParallelEnv(2, make_env)
    else:
        env = SerialEnv(2, make_env)
    try:
        # env = SerialEnv(2, lambda: GymEnv("CartPole-v1", frame_skip=4))
        env.set_seed(0)
        collector = SyncDataCollector(
            env,
            policy=None,
            total_frames=10001,
            frames_per_batch=10000,
            split_trajs=False,
        )
        for _data in collector:
            break
        steps = _data["next", "step_count"][..., 1:, :]
        done = _data["next", "done"][..., :-1, :]
        # we don't want just one done
        assert done.sum() > 3
        # check that after a done, the next step count is always 1
        assert (steps[done] == 1).all()
        # check that if the env is not done, the next step count is > 1
        assert (steps[~done] > 1).all()
        # check that if step is 1, then the env was done before
        assert (steps == 1)[done].all()
        # check that split traj has a minimum total reward of -21 (for pong only)
        _data = constr(_data)
        assert _data["next", "reward"].sum(-2).min() == -21
    finally:
        env.close()
        del env


# Deprecated reset_when_done
# @pytest.mark.parametrize("num_env", [1, 2])
# @pytest.mark.parametrize("env_name", ["vec"])
# def test_collector_done_persist(num_env, env_name, seed=5):
#     if num_env == 1:
#
#         def env_fn(seed):
#             env = MockSerialEnv(device="cpu")
#             env.set_seed(seed)
#             return env
#
#     else:
#
#         def env_fn(seed):
#             def make_env(seed):
#                 env = MockSerialEnv(device="cpu")
#                 env.set_seed(seed)
#                 return env
#
#             env = ParallelEnv(
#                 num_workers=num_env,
#                 create_env_fn=make_env,
#                 create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
#             )
#             env.set_seed(seed)
#             return env
#
#     policy = make_policy(env_name)
#
#     collector = SyncDataCollector(
#         create_env_fn=env_fn,
#         create_env_kwargs={"seed": seed},
#         policy=policy,
#         frames_per_batch=200 * num_env,
#         max_frames_per_traj=2000,
#         total_frames=20000,
#         device="cpu",
#         reset_when_done=False,
#     )
#     for _, d in enumerate(collector):  # noqa
#         break
#
#     assert (d["done"].sum(-2) >= 1).all()
#     assert torch.unique(d["collector", "traj_ids"], dim=-1).shape[-1] == 1
#
#     del collector


@pytest.mark.parametrize("frames_per_batch", [200, 10])
@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize("env_name", ["vec"])
def test_split_trajs(num_env, env_name, frames_per_batch, seed=5):
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

            env = SerialEnv(
                num_workers=num_env,
                create_env_fn=make_env,
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            env.set_seed(seed)
            return env

    policy = make_policy(env_name)

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=frames_per_batch * num_env,
        max_frames_per_traj=2000,
        total_frames=20000,
        device="cpu",
        reset_when_done=True,
        split_trajs=True,
    )
    for _, d in enumerate(collector):  # noqa
        break

    assert d.ndimension() == 2
    assert d["collector", "mask"].shape == d.shape
    assert d["next", "step_count"].shape == d["next", "done"].shape
    assert d["collector", "traj_ids"].shape == d.shape
    for traj in d.unbind(0):
        assert traj["collector", "traj_ids"].unique().numel() == 1
        assert (
            traj["next", "step_count"][1:] - traj["next", "step_count"][:-1] == 1
        ).all()

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


@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize(
    "env_name",
    [
        "vec",
    ],
)  # 1226: for efficiency, we just test vec, not "conv"
def test_collector_batch_size(
    num_env, env_name, seed=100, num_workers=2, frames_per_batch=20
):
    """Tests that there are 'frames_per_batch' frames in each batch of a collection."""
    if num_env == 3 and IS_WINDOWS:
        pytest.skip("Test timeout (> 10 min) on CI pipeline Windows machine with GPU")
    if num_env == 1:

        def env_fn():
            env = make_make_env(env_name)()
            return env

    else:

        def env_fn():
            # 1226: For efficiency, we don't use Parallel but Serial
            # env = ParallelEnv(
            env = SerialEnv(num_workers=num_env, create_env_fn=make_make_env(env_name))
            return env

    policy = make_policy(env_name)

    torch.manual_seed(0)
    np.random.seed(0)

    ccollector = MultiaSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
    )
    try:
        ccollector.set_seed(seed)
        for i, b in enumerate(ccollector):
            assert b.numel() == -(-frames_per_batch // num_env) * num_env
            if i == 5:
                break
        assert b.names[-1] == "time"
    finally:
        ccollector.shutdown()

    ccollector = MultiSyncDataCollector(
        create_env_fn=[env_fn for _ in range(num_workers)],
        policy=policy,
        frames_per_batch=frames_per_batch,
        max_frames_per_traj=1000,
        total_frames=frames_per_batch * 100,
        cat_results="stack",
    )
    try:
        ccollector.set_seed(seed)
        for i, b in enumerate(ccollector):
            assert (
                b.numel()
                == -(-frames_per_batch // num_env // num_workers)
                * num_env
                * num_workers
            )
            if i == 5:
                break
        assert b.names[-1] == "time"
    finally:
        ccollector.shutdown()
        del ccollector


@pytest.mark.parametrize("num_env", [1, 2])
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
    )
    try:
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
    finally:
        ccollector.shutdown()


@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize("env_name", ["conv", "vec"])
def test_collector_consistency(num_env, env_name, seed=100):
    """Tests that a rollout gathered with env.rollout matches one gathered with the collector."""
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
    env = TransformedEnv(env, StepCounter(20))
    rollout1a = env.rollout(policy=policy, max_steps=50, auto_reset=True)
    env.set_seed(seed)
    rollout1b = env.rollout(policy=policy, max_steps=50, auto_reset=True)
    rollout2 = env.rollout(policy=policy, max_steps=50, auto_reset=True)
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
    )
    collector_iter = iter(collector)
    b1 = next(collector_iter)
    b2 = next(collector_iter)
    with pytest.raises(AssertionError):
        assert_allclose_td(b1, b2)

    # if num_env == 1:
    #     # rollouts collected through DataCollector are padded using pad_sequence, which introduces a first dimension
    #     rollout1a = rollout1a.unsqueeze(0)
    assert (
        rollout1a.batch_size == b1.batch_size
    ), f"got batch_size {rollout1a.batch_size} and {b1.batch_size}"
    assert_allclose_td(rollout1a, b1.select(*rollout1a.keys(True, True)))
    collector.shutdown()


@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize(
    "collector_class",
    [
        SyncDataCollector,
    ],
)  # aSyncDataCollector])
@pytest.mark.parametrize("env_name", ["vec"])  # 1226: removing "conv" for efficiency
def test_traj_len_consistency(num_env, env_name, collector_class, seed=100):
    """Tests that various frames_per_batch lead to the same results."""

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

    collector1 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=1 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
    )
    collector1.set_seed(seed)
    count = 0
    data1 = []
    for d in collector1:
        data1.append(d)
        count += d.shape[-1]
        if count > max_frames_per_traj:
            break

    data1 = torch.cat(data1, d.ndim - 1)
    data1 = data1[..., :max_frames_per_traj]

    collector1.shutdown()
    del collector1

    collector10 = collector_class(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": seed},
        policy=policy,
        frames_per_batch=10 * num_env,
        max_frames_per_traj=2000,
        total_frames=2 * num_env * max_frames_per_traj,
        device="cpu",
    )
    collector10.set_seed(seed)
    count = 0
    data10 = []
    for d in collector10:
        data10.append(d)
        count += d.shape[-1]
        if count > max_frames_per_traj:
            break

    data10 = torch.cat(data10, data1.ndim - 1)
    data10 = data10[..., :max_frames_per_traj]

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
    )
    collector20.set_seed(seed)
    count = 0
    data20 = []
    for d in collector20:
        data20.append(d)
        count += d.shape[-1]
        if count > max_frames_per_traj:
            break

    collector20.shutdown()
    del collector20
    data20 = torch.cat(data20, data1.ndim - 1)
    data20 = data20[..., :max_frames_per_traj]

    assert_allclose_td(data1, data20)
    assert_allclose_td(data10, data20)


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Nested spawned multiprocessed is currently failing in python 3.11. "
    "See https://github.com/python/cpython/pull/108568 for info and fix.",
)
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
    env_make = EnvCreator(
        lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED()), VecNorm())
    )
    env_make = ParallelEnv(num_envs, env_make)

    policy = RandomPolicy(env_make.action_spec)
    num_data_collectors = 2
    c = MultiSyncDataCollector(
        [env_make] * num_data_collectors,
        policy=policy,
        total_frames=int(1e6),
        frames_per_batch=200,
        cat_results="stack",
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
    c.shutdown()
    del c


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
def test_update_weights(use_async):
    def create_env():
        return ContinuousActionVecMockEnv()

    n_actions = ContinuousActionVecMockEnv().action_spec.shape[-1]
    policy = SafeModule(
        torch.nn.LazyLinear(n_actions), in_keys=["observation"], out_keys=["action"]
    )
    policy(create_env().reset())

    collector_class = (
        MultiSyncDataCollector if not use_async else MultiaSyncDataCollector
    )
    collector = collector_class(
        [create_env] * 3,
        policy=policy,
        device=[torch.device("cuda:0")] * 3,
        storing_device=[torch.device("cuda:0")] * 3,
        frames_per_batch=20,
        cat_results="stack",
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
    [
        functools.partial(MultiSyncDataCollector, cat_results="stack"),
        MultiaSyncDataCollector,
        SyncDataCollector,
    ],
)
@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize("out_key", ["_dummy", ("out", "_dummy"), ("_out", "dummy")])
def test_excluded_keys(collector_class, exclude, out_key):
    if not exclude and collector_class is not SyncDataCollector:
        pytest.skip("defining _exclude_private_keys is not possible")

    def make_env():
        return TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())

    dummy_env = make_env()
    obs_spec = dummy_env.observation_spec["observation"]
    policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
    policy = TensorDictModule(
        policy_module, in_keys=["observation"], out_keys=["action"]
    )
    copier = TensorDictModule(lambda x: x, in_keys=["observation"], out_keys=[out_key])
    policy = TensorDictSequential(policy, copier)
    policy_explore = OrnsteinUhlenbeckProcessWrapper(policy)

    collector_kwargs = {
        "create_env_fn": make_env,
        "policy": policy_explore,
        "frames_per_batch": 30,
        "total_frames": -1,
    }
    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(3)
        ]

    collector = collector_class(**collector_kwargs)
    collector._exclude_private_keys = exclude
    for b in collector:
        keys = set(b.keys())
        if exclude:
            assert not any(key.startswith("_") for key in keys)
            assert out_key not in b.keys(True, True)
        else:
            assert any(key.startswith("_") for key in keys)
            assert out_key in b.keys(True, True)
        break
    collector.shutdown()
    dummy_env.close()
    del collector


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class",
    [
        SyncDataCollector,
        MultiaSyncDataCollector,
        functools.partial(MultiSyncDataCollector, cat_results="stack"),
    ],
)
@pytest.mark.parametrize("init_random_frames", [50])  # 1226: faster execution
@pytest.mark.parametrize(
    "explicit_spec,split_trajs", [[True, True], [False, False]]
)  # 1226: faster execution
def test_collector_output_keys(
    collector_class, init_random_frames, explicit_spec, split_trajs
):
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
        "out_keys": [
            "action",
            "hidden1",
            "hidden2",
            ("next", "hidden1"),
            ("next", "hidden2"),
        ],
    }
    if explicit_spec:
        hidden_spec = UnboundedContinuousTensorSpec((1, hidden_size))
        policy_kwargs["spec"] = CompositeSpec(
            action=UnboundedContinuousTensorSpec(),
            hidden1=hidden_spec,
            hidden2=hidden_spec,
            next=CompositeSpec(hidden1=hidden_spec, hidden2=hidden_spec),
        )

    policy = SafeModule(**policy_kwargs)

    env_maker = lambda: GymEnv(PENDULUM_VERSIONED())

    policy(env_maker().reset())

    collector_kwargs = {
        "create_env_fn": env_maker,
        "policy": policy,
        "total_frames": total_frames,
        "frames_per_batch": frames_per_batch,
        "init_random_frames": init_random_frames,
        "split_trajs": split_trajs,
    }

    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(num_envs)
        ]

    collector = collector_class(**collector_kwargs)

    keys = {
        "action",
        "done",
        "collector",
        "hidden1",
        "hidden2",
        ("next", "hidden1"),
        ("next", "hidden2"),
        ("next", "observation"),
        ("next", "done"),
        ("next", "reward"),
        "next",
        "observation",
        ("collector", "traj_ids"),
    }
    if split_trajs:
        keys.add(("collector", "mask"))

    keys.add(("next", "terminated"))
    keys.add("terminated")
    keys.add(("next", "truncated"))
    keys.add("truncated")
    b = next(iter(collector))

    assert set(b.keys(True)) == keys
    collector.shutdown()
    del collector


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("storing_device", ["cuda", "cpu"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
def test_collector_device_combinations(device, storing_device):
    if IS_WINDOWS and PYTHON_3_10 and storing_device == "cuda" and device == "cuda":
        pytest.skip("Windows fatal exception: access violation in torch.storage")

    def env_fn(seed):
        env = make_make_env("conv")()
        env.set_seed(seed)
        return env

    policy = dummypolicy_conv()

    collector = SyncDataCollector(
        create_env_fn=env_fn,
        create_env_kwargs={"seed": 0},
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=device,
        storing_device=storing_device,
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(storing_device)
    collector.shutdown()

    collector = MultiSyncDataCollector(
        create_env_fn=[
            env_fn,
        ],
        create_env_kwargs=[
            {"seed": 0},
        ],
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=[
            device,
        ],
        storing_device=[
            storing_device,
        ],
        cat_results="stack",
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(storing_device)
    collector.shutdown()

    collector = MultiaSyncDataCollector(
        create_env_fn=[
            env_fn,
        ],
        create_env_kwargs=[
            {"seed": 0},
        ],
        policy=policy,
        frames_per_batch=20,
        max_frames_per_traj=2000,
        total_frames=20000,
        device=[
            device,
        ],
        storing_device=[
            storing_device,
        ],
    )
    batch = next(collector.iterator())
    assert batch.device == torch.device(storing_device)
    collector.shutdown()
    del collector


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class",
    [
        SyncDataCollector,
        MultiaSyncDataCollector,
        functools.partial(MultiSyncDataCollector, cat_results="stack"),
    ],
)
class TestAutoWrap:
    num_envs = 1

    @pytest.fixture
    def env_maker(self):
        from torchrl.envs.libs.gym import GymEnv

        return lambda: GymEnv(PENDULUM_VERSIONED())

    def _create_collector_kwargs(self, env_maker, collector_class, policy):
        collector_kwargs = {
            "create_env_fn": env_maker,
            "policy": policy,
            "frames_per_batch": 200,
            "total_frames": -1,
        }

        if collector_class is not SyncDataCollector:
            collector_kwargs["create_env_fn"] = [
                collector_kwargs["create_env_fn"] for _ in range(self.num_envs)
            ]

        return collector_kwargs

    @pytest.mark.parametrize("multiple_outputs", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_auto_wrap_modules(
        self, collector_class, multiple_outputs, env_maker, device
    ):
        policy = WrappablePolicy(
            out_features=env_maker().action_spec.shape[-1],
            multiple_outputs=multiple_outputs,
        )
        # init lazy params
        policy(env_maker().reset().get("observation"))

        collector = collector_class(
            **self._create_collector_kwargs(env_maker, collector_class, policy),
            device=device,
        )

        out_keys = ["action"]
        if multiple_outputs:
            out_keys.extend(f"output{i}" for i in range(1, 4))

        if collector_class is SyncDataCollector:
            assert isinstance(collector.policy, TensorDictModule)
            assert collector.policy.out_keys == out_keys
            # this does not work now that we force the device of the policy
            # assert collector.policy.module is policy

        for i, data in enumerate(collector):
            if i == 0:
                assert (data["action"] != 0).any()
                for p in policy.parameters():
                    p.data.zero_()
                    assert p.device == torch.device("cpu")
                collector.update_policy_weights_()
            elif i == 4:
                assert (data["action"] == 0).all()
                break

        collector.shutdown()
        del collector

    # Deprecated as from v0.3
    # def test_no_wrap_compatible_module(self, collector_class, env_maker):
    #     policy = TensorDictCompatiblePolicy(
    #         out_features=env_maker().action_spec.shape[-1]
    #     )
    #     policy(env_maker().reset())
    #
    #     collector = collector_class(
    #         **self._create_collector_kwargs(env_maker, collector_class, policy)
    #     )
    #
    #     if collector_class is not SyncDataCollector:
    #         # We now do the casting only on the remote workers
    #         pass
    #     else:
    #         assert isinstance(collector.policy, TensorDictCompatiblePolicy)
    #         assert collector.policy.out_keys == ["action"]
    #         assert collector.policy is policy
    #
    #     for i, data in enumerate(collector):
    #         if i == 0:
    #             assert (data["action"] != 0).any()
    #             for p in policy.parameters():
    #                 p.data.zero_()
    #                 assert p.device == torch.device("cpu")
    #             collector.update_policy_weights_()
    #         elif i == 4:
    #             assert (data["action"] == 0).all()
    #             break
    #
    #     collector.shutdown()
    #     del collector

    def test_auto_wrap_error(self, collector_class, env_maker):
        policy = UnwrappablePolicy(out_features=env_maker().action_spec.shape[-1])
        with pytest.raises(
            TypeError,
            match=(r"Arguments to policy.forward are incompatible with entries in"),
        ) if collector_class is SyncDataCollector else pytest.raises(EOFError):
            collector_class(
                **self._create_collector_kwargs(env_maker, collector_class, policy)
            )


@pytest.mark.parametrize("env_class", [CountingEnv, CountingBatchedEnv])
def test_initial_obs_consistency(env_class, seed=1):
    # non regression test on #938
    torch.manual_seed(seed)
    start_val = 4
    if env_class == CountingEnv:
        num_envs = 1
        env = CountingEnv(device="cpu", max_steps=8, start_val=start_val)
        max_steps = 8
    elif env_class == CountingBatchedEnv:
        num_envs = 2
        env = CountingBatchedEnv(
            device="cpu",
            batch_size=[num_envs],
            max_steps=torch.arange(num_envs) + 17,
            start_val=torch.ones([num_envs]) * start_val,
        )
        max_steps = env.max_steps.max().item()
    env.set_seed(seed)
    policy = lambda tensordict: tensordict.set(
        "action", torch.ones(tensordict.shape, dtype=torch.int)
    )
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        frames_per_batch=((max_steps - 3) * 2 + 2) * num_envs,  # at least two episodes
        split_trajs=False,
        total_frames=-1,
    )
    for _d in collector:
        break
    obs = _d["observation"].squeeze()
    if env_class == CountingEnv:
        arange_0 = start_val + torch.arange(max_steps - 3)
        arange = start_val + torch.arange(2)
        expected = torch.cat([arange_0, arange_0, arange])
    else:
        # the first env has a shorter horizon than the second
        arange_0 = start_val + torch.arange(max_steps - 3 - 1)
        arange = start_val + torch.arange(start_val)
        expected_0 = torch.cat([arange_0, arange_0, arange])
        arange_0 = start_val + torch.arange(max_steps - 3)
        arange = start_val + torch.arange(2)
        expected_1 = torch.cat([arange_0, arange_0, arange])
        expected = torch.stack([expected_0, expected_1])
    assert torch.allclose(obs, expected.to(obs.dtype))
    collector.shutdown()
    del collector


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


@pytest.mark.skipif(IS_OSX, reason="Queue.qsize does not work on osx.")
class TestPreemptiveThreshold:
    @pytest.mark.parametrize("env_name", ["conv", "vec"])
    def test_sync_collector_interruptor_mechanism(self, env_name, seed=100):
        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

        policy = make_policy(env_name)
        interruptor = _Interruptor()
        interruptor.start_collection()

        collector = SyncDataCollector(
            create_env_fn=env_fn,
            create_env_kwargs={"seed": seed},
            policy=policy,
            frames_per_batch=50,
            total_frames=200,
            device="cpu",
            interruptor=interruptor,
            split_trajs=False,
        )

        interruptor.stop_collection()
        for batch in collector:
            assert batch["collector", "traj_ids"][0] != -1
            assert batch["collector", "traj_ids"][1] == -1
        collector.shutdown()
        del collector

    @pytest.mark.parametrize(
        "env_name", ["vec"]
    )  # 1226: removing "conv" for efficiency
    def test_multisync_collector_interruptor_mechanism(self, env_name, seed=100):

        frames_per_batch = 800

        def env_fn(seed):
            env = make_make_env(env_name)()
            env.set_seed(seed)
            return env

        policy = make_policy(env_name)

        collector = MultiSyncDataCollector(
            create_env_fn=[env_fn] * 4,
            create_env_kwargs=[{"seed": seed}] * 4,
            policy=policy,
            total_frames=800,
            max_frames_per_traj=50,
            frames_per_batch=frames_per_batch,
            init_random_frames=-1,
            reset_at_each_iter=False,
            device=get_default_devices()[0],
            split_trajs=False,
            preemptive_threshold=0.0,  # stop after one iteration
            cat_results="stack",
        )

        for batch in collector:
            trajectory_ids = batch["collector", "traj_ids"]
            trajectory_ids_mask = trajectory_ids != -1  # valid frames mask
            assert trajectory_ids[trajectory_ids_mask].numel() < frames_per_batch
        collector.shutdown()
        del collector


def test_maxframes_error():
    env = TransformedEnv(CountingEnv(), StepCounter(2))
    _ = SyncDataCollector(
        env, RandomPolicy(env.action_spec), total_frames=10_000, frames_per_batch=1000
    )
    with pytest.raises(ValueError):
        _ = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            total_frames=10_000,
            frames_per_batch=1000,
            max_frames_per_traj=2,
        )


@retry(AssertionError, tries=10, delay=0)
@pytest.mark.parametrize("policy_device", [None, *get_available_devices()])
@pytest.mark.parametrize("env_device", [None, *get_available_devices()])
@pytest.mark.parametrize("storing_device", [None, *get_available_devices()])
@pytest.mark.parametrize("parallel", [False, True])
def test_reset_heterogeneous_envs(
    policy_device: torch.device,
    env_device: torch.device,
    storing_device: torch.device,
    parallel,
):
    if (
        policy_device is not None
        and policy_device.type == "cuda"
        and env_device is None
    ):
        env_device = torch.device("cpu")  # explicit mapping
    elif env_device is not None and env_device.type == "cuda" and policy_device is None:
        policy_device = torch.device("cpu")
    env1 = lambda: TransformedEnv(CountingEnv(device="cpu"), StepCounter(2))
    env2 = lambda: TransformedEnv(CountingEnv(device="cpu"), StepCounter(3))
    if parallel:
        cls = ParallelEnv
    else:
        cls = SerialEnv
    env = cls(2, [env1, env2], device=env_device, share_individual_td=True)
    collector = SyncDataCollector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=10_000,
        frames_per_batch=100,
        policy_device=policy_device,
        env_device=env_device,
        storing_device=storing_device,
    )
    try:
        for data in collector:  # noqa: B007
            break
        data_device = storing_device if storing_device is not None else env_device
        assert (
            data[0]["next", "truncated"].squeeze()
            == torch.tensor([False, True], device=data_device).repeat(25)[:50]
        ).all(), data[0]["next", "truncated"]
        assert (
            data[1]["next", "truncated"].squeeze()
            == torch.tensor([False, False, True], device=data_device).repeat(17)[:50]
        ).all(), data[1]["next", "truncated"][:10]
    finally:
        collector.shutdown()
        del collector


def test_policy_with_mask():
    env = CountingBatchedEnv(start_val=torch.tensor(10), max_steps=torch.tensor(1e5))

    def policy(td):
        obs = td.get("observation")
        # This policy cannot work with obs all 0s
        if not obs.any():
            raise AssertionError
        action = obs.clone()
        td.set("action", action)
        return td

    collector = SyncDataCollector(
        env, policy=policy, frames_per_batch=10, total_frames=20
    )
    for _ in collector:
        break
    collector.shutdown()


class TestNestedEnvsCollector:
    def test_multi_collector_nested_env_consistency(self, seed=1):
        torch.manual_seed(seed)
        env_fn = lambda: TransformedEnv(NestedCountingEnv(), InitTracker())
        env = NestedCountingEnv()
        policy = CountingEnvCountPolicy(env.action_spec, env.action_key)

        ccollector = MultiaSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=20,
            total_frames=100,
            device=get_default_devices()[0],
        )
        try:
            for i, d in enumerate(ccollector):
                if i == 0:
                    c1 = d
                elif i == 1:
                    c2 = d
                else:
                    break
            assert d.names[-1] == "time"
            with pytest.raises(AssertionError):
                assert_allclose_td(c1, c2)
        finally:
            ccollector.shutdown()
            del ccollector

        ccollector = MultiSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=20,
            total_frames=100,
            device=get_default_devices()[0],
            cat_results="stack",
        )
        try:
            for i, d in enumerate(ccollector):
                if i == 0:
                    d1 = d
                elif i == 1:
                    d2 = d
                else:
                    break
            assert d.names[-1] == "time"
            with pytest.raises(AssertionError):
                assert_allclose_td(d1, d2)
        finally:
            ccollector.shutdown()
            del ccollector
        assert_allclose_td(c1, d1.reshape(c1.shape))
        assert_allclose_td(c2, d2.reshape(c2.shape))

    @pytest.mark.parametrize("nested_obs_action", [True, False])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("nested_reward", [True, False])
    def test_collector_nested_env_combinations(
        self,
        nested_obs_action,
        nested_done,
        nested_reward,
        seed=1,
        frames_per_batch=20,
    ):
        env = NestedCountingEnv(
            nest_reward=nested_reward,
            nest_done=nested_done,
            nest_obs_action=nested_obs_action,
        )
        torch.manual_seed(seed)
        policy = CountingEnvCountPolicy(env.action_spec, env.action_key)
        ccollector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=get_default_devices()[0],
        )

        for _td in ccollector:
            break
        ccollector.shutdown()
        del ccollector

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 2)])
    def test_nested_env_dims(self, batch_size, nested_dim=5, frames_per_batch=20):
        from mocking_classes import CountingEnvCountPolicy, NestedCountingEnv

        env = NestedCountingEnv(batch_size=batch_size, nested_dim=nested_dim)
        env_fn = lambda: NestedCountingEnv(batch_size=batch_size, nested_dim=nested_dim)
        torch.manual_seed(0)
        policy = CountingEnvCountPolicy(env.action_spec, env.action_key)
        policy(env.reset())
        ccollector = SyncDataCollector(
            create_env_fn=env_fn,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=get_default_devices()[0],
        )

        for _td in ccollector:
            break
        ccollector.shutdown()
        del ccollector
        assert ("data", "reward") not in _td.keys(True)
        assert _td.batch_size == (*batch_size, frames_per_batch // prod(batch_size))
        assert _td["data"].batch_size == (
            *batch_size,
            frames_per_batch // prod(batch_size),
            nested_dim,
        )
        assert _td["next", "data"].batch_size == (
            *batch_size,
            frames_per_batch // prod(batch_size),
            nested_dim,
        )


class TestHeterogeneousEnvsCollector:
    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("frames_per_batch", [4, 8, 16])
    def test_collector_heterogeneous_env(
        self, batch_size, frames_per_batch, seed=1, max_steps=4
    ):
        batch_size = torch.Size(batch_size)
        env = HeterogeneousCountingEnv(max_steps=max_steps - 1, batch_size=batch_size)
        torch.manual_seed(seed)
        device = get_default_devices()[0]
        policy = HeterogeneousCountingEnvPolicy(env.input_spec["full_action_spec"])
        ccollector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=device,
        )

        for _td in ccollector:
            break
        ccollector.shutdown()
        collected_frames = frames_per_batch // batch_size.numel()

        for i in range(env.n_nested_dim):
            if collected_frames >= max_steps:
                agent_obs = _td["lazy"][(0,) * len(batch_size)][..., i][f"tensor_{i}"]
                for _ in range(i + 1):
                    agent_obs = agent_obs.mean(-1)
                assert (
                    agent_obs
                    == torch.arange(max_steps, device=device).repeat(
                        collected_frames // max_steps
                    )
                ).all()  # Check reset worked
            assert (_td["lazy"][..., i]["action"] == 1).all()
        del ccollector

    def test_multi_collector_heterogeneous_env_consistency(
        self, seed=1, frames_per_batch=20, batch_dim=10
    ):
        env = HeterogeneousCountingEnv(max_steps=3, batch_size=(batch_dim,))
        torch.manual_seed(seed)
        env_fn = lambda: TransformedEnv(env, InitTracker())
        check_env_specs(env_fn(), return_contiguous=False)
        policy = HeterogeneousCountingEnvPolicy(env.input_spec["full_action_spec"])

        ccollector = MultiaSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=get_default_devices()[0],
        )
        try:
            for i, d in enumerate(ccollector):
                if i == 0:
                    c1 = d
                elif i == 1:
                    c2 = d
                else:
                    break
            assert d.names[-1] == "time"
            with pytest.raises(AssertionError):
                assert_allclose_td(c1, c2)
        finally:
            ccollector.shutdown()

        ccollector = MultiSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=get_default_devices()[0],
            cat_results="stack",
        )
        try:
            for i, d in enumerate(ccollector):
                if i == 0:
                    d1 = d
                elif i == 1:
                    d2 = d
                else:
                    break
            assert d.names[-1] == "time"
            with pytest.raises(AssertionError):
                assert_allclose_td(d1, d2)
        finally:
            ccollector.shutdown()
            del ccollector

        assert_allclose_td(c1.unsqueeze(0), d1)
        assert_allclose_td(c2.unsqueeze(0), d2)


class TestMultiKeyEnvsCollector:
    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 1)])
    @pytest.mark.parametrize("frames_per_batch", [4, 8, 16])
    @pytest.mark.parametrize("max_steps", [2, 3])
    def test_collector(self, batch_size, frames_per_batch, max_steps, seed=1):
        env = MultiKeyCountingEnv(batch_size=batch_size, max_steps=max_steps)
        torch.manual_seed(seed)
        device = get_default_devices()[0]
        policy = MultiKeyCountingEnvPolicy(
            env.input_spec["full_action_spec"].to(device)
        )
        ccollector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=device,
        )

        for _td in ccollector:
            break
        ccollector.shutdown()
        del ccollector
        for done_key in env.done_keys:
            assert _replace_last(done_key, "_reset") not in _td.keys(True, True)
        check_rollout_consistency_multikey_env(_td, max_steps=max_steps)

    def test_multi_collector_consistency(
        self, seed=1, frames_per_batch=20, batch_dim=10
    ):
        env = MultiKeyCountingEnv(batch_size=(batch_dim,))
        env_fn = lambda: env
        torch.manual_seed(seed)
        device = get_default_devices()[0]
        policy = MultiKeyCountingEnvPolicy(
            env.input_spec["full_action_spec"].to(device), deterministic=True
        )

        ccollector = MultiaSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=device,
        )
        for i, d in enumerate(ccollector):
            if i == 0:
                c1 = d
            elif i == 1:
                c2 = d
            else:
                break
        assert d.names[-1] == "time"
        with pytest.raises(AssertionError):
            assert_allclose_td(c1, c2)
        ccollector.shutdown()

        ccollector = MultiSyncDataCollector(
            create_env_fn=[env_fn],
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=100,
            device=get_default_devices()[0],
            cat_results="stack",
        )
        for i, d in enumerate(ccollector):
            if i == 0:
                d1 = d
            elif i == 1:
                d2 = d
            else:
                break
        assert d.names[-1] == "time"
        with pytest.raises(AssertionError):
            assert_allclose_td(d1, d2)
        ccollector.shutdown()
        del ccollector

        assert_allclose_td(c1.unsqueeze(0), d1)
        assert_allclose_td(c2.unsqueeze(0), d2)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No casting if no cuda")
class TestUpdateParams:
    class DummyEnv(EnvBase):
        def __init__(self, device, batch_size=[]):  # noqa: B006
            super().__init__(batch_size=batch_size, device=device)
            self.state = torch.zeros(self.batch_size, device=device)
            self.observation_spec = CompositeSpec(
                state=UnboundedContinuousTensorSpec(shape=(), device=device)
            )
            self.action_spec = UnboundedContinuousTensorSpec(
                shape=batch_size, device=device
            )
            self.reward_spec = UnboundedContinuousTensorSpec(
                shape=(*batch_size, 1), device=device
            )

        def _step(
            self,
            tensordict,
        ):
            action = tensordict.get("action")
            self.state += action
            return TensorDict(
                {
                    "state": self.state.clone(),
                    "reward": self.reward_spec.zero(),
                    **self.full_done_spec.zero(),
                },
                self.batch_size,
                device=self.device,
            )

        def _reset(self, tensordict=None):
            self.state.zero_()
            return TensorDict(
                {"state": self.state.clone()}, self.batch_size, device=self.device
            )

        def _set_seed(self, seed):
            return seed

    class Policy(TensorDictModuleBase):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(()))
            self.register_buffer("buf", torch.zeros(()))
            self.in_keys = []
            self.out_keys = ["action"]

        def forward(self, td):
            td["action"] = (self.param + self.buf).expand(td.shape)
            return td

    @pytest.mark.parametrize(
        "collector",
        [
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
            MultiaSyncDataCollector,
        ],
    )
    @pytest.mark.parametrize("give_weights", [True, False])
    @pytest.mark.parametrize(
        "policy_device,env_device",
        [
            ["cpu", "cuda"],
            ["cuda", "cpu"],
            # ["cpu", "cuda:0"],  # 1226: faster execution
            # ["cuda:0", "cpu"],
            # ["cuda", "cuda:0"],
            # ["cuda:0", "cuda"],
        ],
    )
    def test_param_sync(self, give_weights, collector, policy_device, env_device):
        policy = TestUpdateParams.Policy().to(policy_device)

        env = EnvCreator(lambda: TestUpdateParams.DummyEnv(device=env_device))
        device = env().device
        env = [env]
        col = collector(
            env, policy, device=device, total_frames=200, frames_per_batch=10
        )
        try:
            for i, data in enumerate(col):
                if i == 0:
                    assert (data["action"] == 0).all()
                    # update policy
                    policy.param.data += 1
                    policy.buf.data += 2
                    if give_weights:
                        d = dict(policy.named_parameters())
                        d.update(policy.named_buffers())
                        p_w = TensorDict(d, [])
                    else:
                        p_w = None
                    col.update_policy_weights_(p_w)
                elif i == 20:
                    if (data["action"] == 1).all():
                        raise RuntimeError("Failed to update buffer")
                    elif (data["action"] == 2).all():
                        raise RuntimeError("Failed to update params")
                    elif (data["action"] == 0).all():
                        raise RuntimeError("Failed to update params and buffers")
                    assert (data["action"] == 3).all()
        finally:
            col.shutdown()
            del col


class TestAggregateReset:
    def test_aggregate_reset_to_root(self):
        # simple
        td = TensorDict({"_reset": torch.zeros((1,), dtype=torch.bool)}, [])
        assert _aggregate_end_of_traj(td).shape == ()
        # td with batch size
        td = TensorDict({"_reset": torch.zeros((1,), dtype=torch.bool)}, [1])
        assert _aggregate_end_of_traj(td).shape == (1,)
        td = TensorDict({"_reset": torch.zeros((1, 2), dtype=torch.bool)}, [1])
        assert _aggregate_end_of_traj(td).shape == (1,)
        # nested td
        td = TensorDict(
            {
                "_reset": torch.zeros((1,), dtype=torch.bool),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1],
        )
        assert _aggregate_end_of_traj(td).shape == (1,)
        # nested td with greater number of dims
        td = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        # test reduction
        assert _aggregate_end_of_traj(td).shape == (1, 2)
        td = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        # test reduction, partial
        assert _aggregate_end_of_traj(td).shape == (1, 2)
        td = TensorDict(
            {
                "_reset": torch.tensor([True, False]).view(1, 2),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        assert (
            _aggregate_end_of_traj(td) == torch.tensor([True, False]).view(1, 2)
        ).all()
        # with a stack
        td0 = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
                "b": {"c": torch.randn(1, 2)},
            },
            [1, 2],
        )
        td1 = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
                "b": {"c": torch.randn(1, 2, 5)},
            },
            [1, 2],
        )
        td = LazyStackedTensorDict.lazy_stack([td0, td1], 0)
        assert _aggregate_end_of_traj(td).all()

    def test_aggregate_reset_to_root_keys(self):
        # simple
        td = TensorDict({"_reset": torch.zeros((1,), dtype=torch.bool)}, [])
        assert _aggregate_end_of_traj(td, reset_keys=["_reset"]).shape == ()
        # td with batch size
        td = TensorDict({"_reset": torch.zeros((1,), dtype=torch.bool)}, [1])
        assert _aggregate_end_of_traj(td, reset_keys=["_reset"]).shape == (1,)
        td = TensorDict({"_reset": torch.zeros((1, 2), dtype=torch.bool)}, [1])
        assert _aggregate_end_of_traj(td, reset_keys=["_reset"]).shape == (1,)
        # nested td
        td = TensorDict(
            {
                "_reset": torch.zeros((1,), dtype=torch.bool),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1],
        )
        assert _aggregate_end_of_traj(
            td, reset_keys=["_reset", ("a", "_reset")]
        ).shape == (1,)
        # nested td with greater number of dims
        td = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        # test reduction
        assert _aggregate_end_of_traj(
            td, reset_keys=["_reset", ("a", "_reset")]
        ).shape == (
            1,
            2,
        )
        td = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        assert _aggregate_end_of_traj(td, reset_keys=["_reset", ("a", "_reset")]).all()
        # test reduction, partial
        assert _aggregate_end_of_traj(
            td, reset_keys=["_reset", ("a", "_reset")]
        ).shape == (
            1,
            2,
        )
        td = TensorDict(
            {
                "_reset": torch.tensor(
                    [True, False],
                ).view(1, 2),
                "a": {"_reset": torch.zeros((1, 2), dtype=torch.bool)},
            },
            [1, 2],
        )
        assert (
            _aggregate_end_of_traj(td, reset_keys=["_reset", ("a", "_reset")])
            == torch.tensor([True, False]).view(1, 2)
        ).all()
        # with a stack
        td0 = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
                "b": {"c": torch.randn(1, 2)},
            },
            [1, 2],
        )
        td1 = TensorDict(
            {
                "_reset": torch.zeros(
                    (1, 2),
                    dtype=torch.bool,
                ),
                "a": {"_reset": torch.ones((1, 2), dtype=torch.bool)},
                "b": {"c": torch.randn(1, 2, 5)},
            },
            [1, 2],
        )
        td = LazyStackedTensorDict.lazy_stack([td0, td1], 0)
        assert _aggregate_end_of_traj(td, reset_keys=["_reset", ("a", "_reset")]).all()

    def test_aggregate_reset_to_root_errors(self):
        # the order matters: if the first or another key is missing, the ValueError is raised at a different line
        with pytest.raises(ValueError, match=PARTIAL_MISSING_ERR):
            _aggregate_end_of_traj(
                TensorDict({"_reset": False}, []),
                reset_keys=["_reset", ("another", "_reset")],
            )
        with pytest.raises(ValueError, match=PARTIAL_MISSING_ERR):
            _aggregate_end_of_traj(
                TensorDict({"_reset": False}, []),
                reset_keys=[("another", "_reset"), "_reset"],
            )


@pytest.mark.parametrize(
    "collector_class",
    [
        functools.partial(MultiSyncDataCollector, cat_results="stack"),
        MultiaSyncDataCollector,
        SyncDataCollector,
    ],
)
def test_collector_reloading(collector_class):
    def make_env():
        return ContinuousActionVecMockEnv()

    dummy_env = make_env()
    obs_spec = dummy_env.observation_spec["observation"]
    policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
    policy = Actor(policy_module, spec=dummy_env.action_spec)
    policy_explore = OrnsteinUhlenbeckProcessWrapper(policy)

    collector_kwargs = {
        "create_env_fn": make_env,
        "policy": policy_explore,
        "frames_per_batch": 30,
        "total_frames": 90,
    }
    if collector_class is not SyncDataCollector:
        collector_kwargs["create_env_fn"] = [
            collector_kwargs["create_env_fn"] for _ in range(3)
        ]

    collector = collector_class(**collector_kwargs)
    for i, _ in enumerate(collector):
        if i == 3:
            break
    collector_frames = collector._frames
    collector_iter = collector._iter
    collector_state_dict = collector.state_dict()
    collector.shutdown()

    collector = collector_class(**collector_kwargs)
    collector.load_state_dict(collector_state_dict)
    assert collector._frames == collector_frames
    assert collector._iter == collector_iter
    for _ in enumerate(collector):
        raise AssertionError
    collector.shutdown()
    del collector


class TestLibThreading:
    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_num_threads(self):
        from torchrl.collectors import collectors

        _main_async_collector_saved = collectors._main_async_collector
        collectors._main_async_collector = decorate_thread_sub_func(
            collectors._main_async_collector, num_threads=3
        )
        num_threads = torch.get_num_threads()
        try:
            env = ContinuousActionVecMockEnv()
            c = MultiSyncDataCollector(
                [env],
                policy=RandomPolicy(env.action_spec),
                num_threads=7,
                num_sub_threads=3,
                total_frames=200,
                frames_per_batch=200,
                cat_results="stack",
            )
            assert torch.get_num_threads() == 7
            for _ in c:
                pass
        finally:
            try:
                c.shutdown()
                del c
            except Exception:
                torchrl_logger.info("Failed to shut down collector")
            # reset vals
            collectors._main_async_collector = _main_async_collector_saved
            torch.set_num_threads(num_threads)

    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_auto_num_threads(self):
        init_threads = torch.get_num_threads()
        try:
            collector = MultiSyncDataCollector(
                [ContinuousActionVecMockEnv],
                RandomPolicy(ContinuousActionVecMockEnv().full_action_spec),
                frames_per_batch=3,
                cat_results="stack",
            )
            for _ in collector:
                assert torch.get_num_threads() == init_threads - 1
                break
            collector.shutdown()
            assert torch.get_num_threads() == init_threads
            del collector
            gc.collect()
        finally:
            torch.set_num_threads(init_threads)

        try:
            collector = MultiSyncDataCollector(
                [ParallelEnv(2, ContinuousActionVecMockEnv)],
                RandomPolicy(ContinuousActionVecMockEnv().full_action_spec.expand(2)),
                frames_per_batch=3,
                cat_results="stack",
            )
            for _ in collector:
                assert torch.get_num_threads() == init_threads - 2
                break
            collector.shutdown()
            assert torch.get_num_threads() == init_threads
            del collector
            gc.collect()
        finally:
            torch.set_num_threads(init_threads)


class TestUniqueTraj:
    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    @pytest.mark.parametrize("cat_results", ["stack", 0])
    def test_unique_traj_sync(self, cat_results):
        stack_results = cat_results == "stack"
        buffer = ReplayBuffer(
            storage=LazyTensorStorage(900, ndim=2 + stack_results), batch_size=16
        )
        c = MultiSyncDataCollector(
            [SerialEnv(2, EnvCreator(lambda: GymEnv("CartPole-v1")))] * 3,
            policy=RandomPolicy(GymEnv("CartPole-v1").action_spec),
            total_frames=900,
            frames_per_batch=300,
            cat_results=cat_results,
        )
        try:
            for d in c:
                buffer.extend(d)
            traj_ids = buffer[:].get(("collector", "traj_ids"))
            # check that we have as many trajs as expected (no skip)
            assert traj_ids.unique().numel() == traj_ids.max() + 1
            # check that trajs are not overlapping
            if stack_results:
                sets = [
                    set(batch)
                    for collectors in traj_ids.tolist()
                    for batch in collectors
                ]
            else:
                sets = [set(batch) for batch in traj_ids.tolist()]

            for i in range(len(sets) - 1):
                for j in range(i + 1, len(sets)):
                    assert sets[i].intersection(sets[j]) == set()
        finally:
            c.shutdown()
            del c


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
