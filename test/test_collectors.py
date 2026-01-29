# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import torch

import torchrl.collectors._multi_base
import torchrl.collectors._runner
from packaging import version
from pyvers import implement_for
from tensordict import (
    assert_allclose_td,
    LazyStackedTensorDict,
    NonTensorData,
    TensorDict,
    TensorDictBase,
)
from tensordict.nn import (
    CudaGraphModule,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import nn
from torchrl._utils import (
    _make_ordinal_device,
    _replace_last,
    logger as torchrl_logger,
    prod,
    seed_generator,
)
from torchrl.collectors import (
    AsyncCollector,
    Collector,
    MultiAsyncCollector,
    MultiSyncCollector,
    ProfileConfig,
    WeightUpdaterBase,
)
from torchrl.collectors._constants import _Interruptor
from torchrl.collectors._multi_base import MultiCollector

from torchrl.collectors.utils import split_trajectories
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    LazyTensorStorage,
    NonTensor,
    ReplayBuffer,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import (
    EnvBase,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    SerialEnv,
    StepCounter,
    Transform,
)
from torchrl.envs.libs.gym import _has_gym, gym_backend, GymEnv, set_gym_backend
from torchrl.envs.transforms import TransformedEnv, VecNorm
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    check_env_specs,
    PARTIAL_MISSING_ERR,
)
from torchrl.modules import (
    Actor,
    OrnsteinUhlenbeckProcessModule,
    RandomPolicy,
    SafeModule,
)

from torchrl.testing import (
    CARTPOLE_VERSIONED,
    check_rollout_consistency_multikey_env,
    generate_seeds,
    get_available_devices,
    get_default_devices,
    LSTMNet,
    PENDULUM_VERSIONED,
    retry,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionConvPolicy,
    DiscreteActionVecMockEnv,
    DiscreteActionVecPolicy,
    EnvThatErrorsAfter10Iters,
    EnvWithDynamicSpec,
    HeterogeneousCountingEnv,
    HeterogeneousCountingEnvPolicy,
    MockSerialEnv,
    MultiKeyCountingEnv,
    MultiKeyCountingEnvPolicy,
    NestedCountingEnv,
)
from torchrl.testing.modules import BiasModule, NonSerializableBiasModule
from torchrl.testing.mp_helpers import decorate_thread_sub_func
from torchrl.weight_update import (
    MultiProcessWeightSyncScheme,
    SharedMemWeightSyncScheme,
)

# torch.set_default_dtype(torch.double)
IS_WINDOWS = sys.platform == "win32"
IS_OSX = sys.platform == "darwin"
PYTHON_3_10 = sys.version_info.major == 3 and sys.version_info.minor == 10
PYTHON_3_7 = sys.version_info.major == 3 and sys.version_info.minor == 7
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
_has_cuda = torch.cuda.is_available()


@implement_for("torch", "2.5")
def has_mps():
    return torch.mps.is_available()


@implement_for("torch", None, "2.5")
def has_mps():  # noqa: F811
    return torch.backends.mps.is_available()


class WrappablePolicy(nn.Module):
    def __init__(self, out_features: int, multiple_outputs: bool = False):
        super().__init__()
        self.multiple_outputs = multiple_outputs
        self.linear = nn.LazyLinear(out_features)

    def forward(self, observation):
        output = self.linear(observation)
        if self.multiple_outputs:
            return output, output.sum(), output.min(), output.max()
        return output


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


class DeterministicZeroPolicyNet(nn.Module):
    """A simple policy that always outputs action 0 (for discrete action spaces)."""

    def forward(self, observation):
        return torch.zeros(
            observation.shape[:-1], dtype=torch.long, device=observation.device
        )


class DeterministicZeroPolicy(Actor):
    """A deterministic policy that always outputs action 0.

    Useful for testing init_random_frames to distinguish from random actions.
    """

    def __init__(self):
        super().__init__(
            DeterministicZeroPolicyNet(),
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


def _pendulum_env_maker():
    return GymEnv(PENDULUM_VERSIONED())


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


class TestCollectorGeneric:
    @pytest.mark.parametrize("num_env", [1, 2])
    # 1226: for efficiency, we just test vec, not "conv"
    @pytest.mark.parametrize("env_name", ["vec"])
    def test_collector_batch_size(
        self, num_env, env_name, seed=100, num_workers=2, frames_per_batch=20
    ):
        """Tests that there are 'frames_per_batch' frames in each batch of a collection."""
        if num_env == 3 and IS_WINDOWS:
            pytest.skip(
                "Test timeout (> 10 min) on CI pipeline Windows machine with GPU"
            )
        if num_env == 1:

            def env_fn():
                env = make_make_env(env_name)()
                return env

        else:

            def env_fn():
                # 1226: For efficiency, we don't use Parallel but Serial
                # env = ParallelEnv(
                env = SerialEnv(
                    num_workers=num_env, create_env_fn=make_make_env(env_name)
                )
                return env

        policy = make_policy(env_name)

        torch.manual_seed(0)
        np.random.seed(0)

        ccollector = MultiAsyncCollector(
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

        ccollector = MultiSyncCollector(
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
    @pytest.mark.parametrize("env_name", ["conv", "vec"])
    def test_collector_consistency(self, num_env, env_name, seed=100):
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
                    create_env_kwargs=[
                        {"seed": s} for s in generate_seeds(seed, num_env)
                    ],
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
        try:
            assert_allclose_td(rollout1a, rollout1b)
            with pytest.raises(AssertionError):
                assert_allclose_td(rollout1a, rollout2)
        finally:
            env.close()

        collector = Collector(
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

        # if num_env == 1:
        #     # rollouts collected through DataCollector are padded using pad_sequence, which introduces a first dimension
        #     rollout1a = rollout1a.unsqueeze(0)
        try:
            with pytest.raises(AssertionError):
                assert_allclose_td(b1, b2)
            assert (
                rollout1a.batch_size == b1.batch_size
            ), f"got batch_size {rollout1a.batch_size} and {b1.batch_size}"
            assert_allclose_td(rollout1a, b1.select(*rollout1a.keys(True, True)))
        finally:
            collector.shutdown()

    @pytest.mark.skipif(not _has_gym, reason="gym library is not installed")
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize(
        "constr",
        [
            functools.partial(split_trajectories, prefix="collector"),
            functools.partial(split_trajectories),
            functools.partial(
                split_trajectories, trajectory_key=("collector", "traj_ids")
            ),
        ],
    )
    def test_collector_env_reset(self, constr, parallel):
        torch.manual_seed(0)

        def make_env():
            # This is currently necessary as the methods in GymWrapper may have mismatching backend
            # versions.
            with set_gym_backend(gym_backend()):
                return TransformedEnv(
                    GymEnv(CARTPOLE_VERSIONED(), frame_skip=4), StepCounter()
                )

        if parallel:
            env = ParallelEnv(2, make_env)
        else:
            env = SerialEnv(2, make_env)
        try:
            # env = SerialEnv(2, lambda: GymEnv("CartPole-v1", frame_skip=4))
            env.set_seed(0)
            collector = Collector(
                env,
                policy=None,
                total_frames=2001,
                frames_per_batch=2000,
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
            # check that split traj has reasonable reward structure
            _data = constr(_data)
            assert _data["next", "reward"].sum(-2).min() >= 0
        finally:
            env.close()
            del env

    @pytest.mark.parametrize(
        "break_when_any_done,break_when_all_done",
        [[True, False], [False, True], [False, False]],
    )
    @pytest.mark.parametrize("n_envs", [1, 4])
    def test_collector_outplace_policy(
        self, n_envs, break_when_any_done, break_when_all_done
    ):
        def policy_inplace(td):
            td.set("action", torch.ones(td.shape + (1,)))
            return td

        def policy_outplace(td):
            return td.empty().set("action", torch.ones(td.shape + (1,)))

        if n_envs == 1:
            env = CountingEnv(10)
        else:
            env = SerialEnv(
                n_envs,
                [functools.partial(CountingEnv, 10 + i) for i in range(n_envs)],
            )
        env.reset()
        c_inplace = Collector(
            env, policy_inplace, frames_per_batch=10, total_frames=100
        )
        d_inplace = torch.cat(list(c_inplace), dim=0)
        env.reset()
        c_outplace = Collector(
            env, policy_outplace, frames_per_batch=10, total_frames=100
        )
        d_outplace = torch.cat(list(c_outplace), dim=0)
        assert_allclose_td(d_inplace, d_outplace)

    @pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
            MultiAsyncCollector,
            functools.partial(MultiSyncCollector, cat_results="stack"),
        ],
    )
    @pytest.mark.parametrize("init_random_frames", [0, 50])  # 1226: faster execution
    @pytest.mark.parametrize(
        "explicit_spec,split_trajs", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_collector_output_keys(
        self, collector_class, init_random_frames, explicit_spec, split_trajs
    ):
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
            hidden_spec = Unbounded((1, hidden_size))
            policy_kwargs["spec"] = Composite(
                action=Unbounded(),
                hidden1=hidden_spec,
                hidden2=hidden_spec,
                next=Composite(hidden1=hidden_spec, hidden2=hidden_spec),
            )

        policy = SafeModule(**policy_kwargs)

        env_maker = _pendulum_env_maker

        policy(env_maker().reset())

        collector_kwargs = {
            "create_env_fn": env_maker,
            "policy": policy,
            "total_frames": total_frames,
            "frames_per_batch": frames_per_batch,
            "init_random_frames": init_random_frames,
            "split_trajs": split_trajs,
        }

        if collector_class is not Collector:
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

    @pytest.mark.parametrize(
        "collector_class",
        [
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
            Collector,
        ],
    )
    def test_collector_reloading(self, collector_class):
        def make_env():
            return ContinuousActionVecMockEnv()

        dummy_env = make_env()
        obs_spec = dummy_env.observation_spec["observation"]
        policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
        policy = Actor(policy_module, spec=dummy_env.action_spec)
        policy_explore = TensorDictSequential(
            policy, OrnsteinUhlenbeckProcessModule(spec=policy.spec)
        )

        collector_kwargs = {
            "create_env_fn": make_env,
            "policy": policy_explore,
            "frames_per_batch": 30,
            "total_frames": 90,
        }
        if collector_class is not Collector:
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

    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason="Nested spawned multiprocessed is currently failing in python 3.11. "
        "See https://github.com/python/cpython/pull/108568 for info and fix.",
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.8.0"),
        reason="VecNorm shared memory synchronization requires PyTorch >= 2.8 "
        "when using spawn multiprocessing start method with file_system sharing strategy.",
    )
    @pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
    @pytest.mark.parametrize("static_seed", [True, False])
    def test_collector_vecnorm_envcreator(self, static_seed):
        """
        High level test of the following pipeline:
         (1) Design a function that creates an environment with VecNorm
         (2) Wrap that function in an EnvCreator to instantiate the shared tensordict
         (3) Create a ParallelEnv that dispatches this env across workers
         (4) Run several ParallelEnv synchronously
        The function tests that the tensordict gathered from the workers match at certain moments in time, and that they
        are modified after the collector is run for more steps.

        """
        num_envs = 4
        env_make = EnvCreator(
            lambda: TransformedEnv(GymEnv(PENDULUM_VERSIONED()), VecNorm())
        )
        env_make = ParallelEnv(num_envs, env_make)

        policy = RandomPolicy(env_make.action_spec)
        num_data_collectors = 2
        c = MultiSyncCollector(
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

        td1 = (
            TensorDict(s["worker0"]["env_state_dict"]["worker3"]["_extra_state"])
            .unflatten_keys(VecNorm.SEP)
            .clone()
        )
        td2 = (
            TensorDict(s["worker1"]["env_state_dict"]["worker0"]["_extra_state"])
            .unflatten_keys(VecNorm.SEP)
            .clone()
        )
        assert (td1 == td2).all()

        next(c_iter)
        next(c_iter)

        s = c.state_dict()

        td3 = (
            TensorDict(s["worker0"]["env_state_dict"]["worker3"]["_extra_state"])
            .unflatten_keys(VecNorm.SEP)
            .clone()
        )
        td4 = (
            TensorDict(s["worker1"]["env_state_dict"]["worker0"]["_extra_state"])
            .unflatten_keys(VecNorm.SEP)
            .clone()
        )
        assert (td3 == td4).all()
        assert (td1 != td4).any()
        c.shutdown()
        del c

    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize("env_name", ["conv", "vec"])
    def test_concurrent_collector_consistency(self, num_env, env_name, seed=40):
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
                    create_env_kwargs=[
                        {"seed": i} for i in range(seed, seed + num_env)
                    ],
                )
                return env

        policy = make_policy(env_name)
        torchrl_logger.info("Sync")
        collector = Collector(
            create_env_fn=env_fn,
            create_env_kwargs={"seed": seed},
            policy=policy,
            frames_per_batch=20,
            max_frames_per_traj=200,
            total_frames=200,
            device="cpu",
        )
        torchrl_logger.info("Loop")
        try:
            assert collector._use_buffers
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
        finally:
            torchrl_logger.info("Shutting down sync")
            collector.shutdown()

        torchrl_logger.info("Concurrent")
        ccollector = AsyncCollector(
            create_env_fn=env_fn,
            create_env_kwargs={"seed": seed},
            policy=policy,
            frames_per_batch=20,
            max_frames_per_traj=2000,
            total_frames=20000,
        )
        torchrl_logger.info("Loop")
        for i, d in enumerate(ccollector):
            if i == 0:
                b1c = d
            elif i == 1:
                b2c = d
            else:
                break

        try:
            assert ccollector._use_buffers
            assert d.names[-1] == "time"

            with pytest.raises(AssertionError):
                assert_allclose_td(b1c, b2c)

            assert_allclose_td(b1c, b1)
            assert_allclose_td(b2c, b2)
        finally:
            torchrl_logger.info("Shutting down concurrent")
            ccollector.shutdown()
            del ccollector

    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize("env_name", ["vec", "conv"])
    def test_concurrent_collector_seed(self, num_env, env_name, seed=100):
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
        ccollector = AsyncCollector(
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

    @pytest.mark.parametrize(
        "ctype", [Collector, MultiAsyncCollector, MultiSyncCollector]
    )
    def test_env_that_errors(self, ctype):
        make_env = EnvThatErrorsAfter10Iters
        policy = RandomPolicy(make_env().action_spec)
        if ctype is Collector:
            collector = Collector(
                make_env, policy=policy, frames_per_batch=30, total_frames=60
            )
        else:
            collector = ctype(
                [make_env, make_env],
                policy=policy,
                frames_per_batch=30,
                total_frames=60,
            )
        with pytest.raises(RuntimeError):
            for _ in collector:
                break

    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.parametrize("to", [3, 10])
    @pytest.mark.parametrize(
        "collector_cls", ["MultiSyncCollector", "MultiAsyncCollector"]
    )
    def test_env_that_waits(self, to, collector_cls):
        # Tests that the collector fails if the MAX_IDLE_COUNT<waiting time, but succeeds otherwise
        # We run this in a subprocess to control the env variable.
        script = f"""import os

os.environ['MAX_IDLE_COUNT'] = '{to}'

from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded
from typing import Optional
from tensordict import TensorDictBase, TensorDict
import time
from torchrl.collectors import {collector_cls}, RandomPolicy


class EnvThatWaitsFor1Sec(EnvBase):
    def __init__(self):
        self.action_spec = Composite(action=Unbounded((1,)))
        self.reward_spec = Composite(reward=Unbounded((1,)))
        self.done_spec = Composite(done=Unbounded((1,)))
        self.observation_spec = Composite(observation=Unbounded((1,)))
        super().__init__()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDict:
        return self.full_observation_spec.zero().update(self.full_done_spec.zero())

    def _step(self, tensordict: TensorDictBase, **kwargs) -> TensorDict:
        time.sleep(1)
        return (
            self.full_observation_spec.zero()
            .update(self.full_done_spec.zero())
            .update(self.full_reward_spec.zero())
        )

    def _set_seed(self, seed: Optional[int]) -> None:
        ...

if __name__ == "__main__":
    policy = RandomPolicy(EnvThatWaitsFor1Sec().action_spec)
    c = {collector_cls}([EnvThatWaitsFor1Sec], policy=policy, total_frames=15, frames_per_batch=5)
    for d in c:
        break
    c.shutdown()
"""
        result = subprocess.run(
            ["python", "-c", script], capture_output=True, text=True
        )
        # This errors if the timeout is too short (3), succeeds if long enough (10)
        assert result.returncode == int(
            to == 3
        ), f"Test failed with output: {result.stdout}"

    @pytest.mark.parametrize(
        "collector_class",
        [
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
            Collector,
        ],
    )
    @pytest.mark.parametrize("exclude", [True, False])
    @pytest.mark.parametrize(
        "out_key", ["_dummy", ("out", "_dummy"), ("_out", "dummy")]
    )
    def test_excluded_keys(self, collector_class, exclude, out_key):
        if not exclude and collector_class is not Collector:
            pytest.skip("defining _exclude_private_keys is not possible")

        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), InitTracker())

        dummy_env = make_env()
        obs_spec = dummy_env.observation_spec["observation"]
        policy_module = nn.Linear(obs_spec.shape[-1], dummy_env.action_spec.shape[-1])
        policy = TensorDictModule(
            policy_module, in_keys=["observation"], out_keys=["action"]
        )
        copier = TensorDictModule(
            lambda x: x, in_keys=["observation"], out_keys=[out_key]
        )
        policy_explore = TensorDictSequential(
            policy,
            copier,
            OrnsteinUhlenbeckProcessModule(
                spec=Composite({key: None for key in policy.out_keys})
            ),
        )

        collector_kwargs = {
            "create_env_fn": make_env,
            "policy": policy_explore,
            "frames_per_batch": 30,
            "total_frames": -1,
        }
        if collector_class is not Collector:
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

    @pytest.mark.parametrize("env_class", [CountingEnv, CountingBatchedEnv])
    def test_initial_obs_consistency(self, env_class, seed=1):
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
        collector = Collector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=((max_steps - 3) * 2 + 2)
            * num_envs,  # at least two episodes
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

    def test_maxframes_error(self):
        env = TransformedEnv(CountingEnv(), StepCounter(2))
        _ = Collector(
            env,
            RandomPolicy(env.action_spec),
            total_frames=10_000,
            frames_per_batch=1000,
        )
        with pytest.raises(ValueError):
            _ = Collector(
                env,
                RandomPolicy(env.action_spec),
                total_frames=10_000,
                frames_per_batch=1000,
                max_frames_per_traj=2,
            )

    @pytest.mark.filterwarnings(
        "error::UserWarning", "ignore:Tensordict is registered in PyTree:UserWarning"
    )
    @pytest.mark.parametrize(
        "collector_type",
        [
            Collector,
            MultiAsyncCollector,
            functools.partial(MultiSyncCollector, cat_results="stack"),
        ],
    )
    def test_no_deepcopy_policy(self, collector_type):
        # Tests that the collector instantiation does not make a deepcopy of the policy if not necessary.
        #
        # The only situation where we want to deepcopy the policy is when the policy_device differs from the actual device
        # of the policy. This can only be checked if the policy is an nn.Module and any of the params is not on the desired
        # device.
        #
        # If the policy is not a nn.Module or has no parameter, policy_device should warn (we don't know what to do but we
        # can trust that the user knows what to do).

        # warnings.warn("Tensordict is registered in PyTree", category=UserWarning)

        # Skip multi-collectors on macOS with older PyTorch when MPS is available.
        # On macOS: "fork" causes segfaults after MPS initialization (even with CPU tensors),
        # and "spawn" on older PyTorch (<2.5) can't handle some multiprocessing scenarios.
        is_multi_collector = collector_type is not Collector
        is_macos = sys.platform == "darwin"
        is_old_pytorch = version.parse(torch.__version__).base_version < "2.5.0"
        mps_available = torch.backends.mps.is_available()
        if is_multi_collector and is_macos and is_old_pytorch and mps_available:
            pytest.skip(
                "Multi-collectors are not supported on macOS with MPS available and PyTorch < 2.5.0 "
                "due to multiprocessing compatibility issues with MPS initialization."
            )

        shared_device = torch.device("cpu")
        if torch.cuda.is_available():
            original_device = torch.device("cuda:0")
        elif has_mps():
            original_device = torch.device("mps")
        else:
            pytest.skip("No GPU or MPS device")

        def make_policy(device=None, nn_module=True):
            if nn_module:
                return TensorDictModule(
                    nn.Linear(7, 7, device=device),
                    in_keys=["observation"],
                    out_keys=["action"],
                )
            policy = make_policy(device=device)
            return CloudpickleWrapper(policy)

        def make_and_test_policy(
            policy,
            policy_device=None,
            env_device=None,
            device=None,
            trust_policy=None,
        ):
            # make sure policy errors when copied

            policy.__deepcopy__ = __deepcopy_error__
            envs = ContinuousActionVecMockEnv(device=env_device)
            if collector_type is not Collector:
                envs = [envs, envs]
            c = collector_type(
                envs,
                policy=policy,
                total_frames=100,
                frames_per_batch=10,
                policy_device=policy_device,
                env_device=env_device,
                device=device,
                trust_policy=trust_policy,
            )
            for _ in c:
                return

        # Simplest use cases
        policy = make_policy()
        make_and_test_policy(policy)

        if collector_type is Collector or original_device.type != "mps":
            # mps cannot be shared
            policy = make_policy(device=original_device)
            make_and_test_policy(policy, env_device=original_device)

        if collector_type is Collector or original_device.type != "mps":
            policy = make_policy(device=original_device)
            make_and_test_policy(
                policy, policy_device=original_device, env_device=original_device
            )

        # Test that we DON'T raise deepcopy errors anymore even when policy_device differs
        # These scenarios previously would have triggered deepcopy, but now use meta device context manager
        if collector_type is not Collector:
            # policy_device differs from the actual device - previously required deepcopy, now works!
            policy = make_policy(device=original_device)
            make_and_test_policy(
                policy, policy_device=shared_device, env_device=shared_device
            )

        if collector_type is not Collector:
            # device differs from the actual device - previously required deepcopy, now works!
            policy = make_policy(device=original_device)
            make_and_test_policy(policy, device=shared_device)

        # If there is no policy_device, we assume that the user is doing things right too but don't warn
        if collector_type is Collector or original_device.type != "mps":
            policy = make_policy(original_device, nn_module=False)
            make_and_test_policy(policy, env_device=original_device)

        # If the policy is a CudaGraphModule, we know it's on cuda - no need to warn
        if torch.cuda.is_available() and collector_type is Collector:
            policy = make_policy(original_device)
            cudagraph_policy = CudaGraphModule(policy)
            make_and_test_policy(
                cudagraph_policy,
                policy_device=original_device,
                env_device=shared_device,
            )

    @pytest.mark.parametrize(
        "ctype", [Collector, MultiAsyncCollector, MultiSyncCollector]
    )
    def test_no_stopiteration(self, ctype):
        # Tests that there is no StopIteration raised and that the length of the collector is properly set
        if ctype is Collector:
            envs = SerialEnv(16, CountingEnv)
        else:
            envs = [SerialEnv(8, CountingEnv), SerialEnv(8, CountingEnv)]

        collector = ctype(create_env_fn=envs, frames_per_batch=173, total_frames=300)
        try:
            c_iter = iter(collector)
            assert len(collector) == 2
            for i in range(len(collector)):  # noqa: B007
                c = next(c_iter)
                assert c is not None
            assert i == 1
        finally:
            collector.shutdown()
            del collector

    def test_policy_with_mask(self):
        env = CountingBatchedEnv(
            start_val=torch.tensor(10), max_steps=torch.tensor(1e5)
        )

        def policy(td):
            obs = td.get("observation")
            # This policy cannot work with obs all 0s
            if not obs.any():
                raise AssertionError
            action = obs.clone()
            td.set("action", action)
            return td

        collector = Collector(env, policy=policy, frames_per_batch=10, total_frames=20)
        for _ in collector:
            break
        collector.shutdown()

    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.parametrize("policy_device", [None, *get_available_devices()])
    @pytest.mark.parametrize("env_device", [None, *get_available_devices()])
    @pytest.mark.parametrize("storing_device", [None, *get_available_devices()])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("share_individual_td", [False, True])
    def test_reset_heterogeneous_envs(
        self,
        policy_device: torch.device,
        env_device: torch.device,
        storing_device: torch.device,
        parallel,
        share_individual_td,
    ):
        if (
            policy_device is not None
            and policy_device.type == "cuda"
            and env_device is None
        ):
            env_device = torch.device("cpu")  # explicit mapping
        elif (
            env_device is not None
            and env_device.type == "cuda"
            and policy_device is None
        ):
            policy_device = torch.device("cpu")
        env1 = lambda: TransformedEnv(CountingEnv(device="cpu"), StepCounter(2))
        env2 = lambda: TransformedEnv(CountingEnv(device="cpu"), StepCounter(3))
        if parallel:
            cls = ParallelEnv
        else:
            cls = SerialEnv
        env = cls(
            2, [env1, env2], device=env_device, share_individual_td=share_individual_td
        )
        collector = Collector(
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
                == torch.tensor([False, False, True], device=data_device).repeat(17)[
                    :50
                ]
            ).all(), data[1]["next", "truncated"][:10]
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize(
        "collector_cls",
        [Collector, MultiSyncCollector, MultiAsyncCollector],
    )
    def test_set_truncated(self, collector_cls):
        env_fn = lambda: TransformedEnv(
            NestedCountingEnv(), InitTracker()
        ).add_truncated_keys()
        env = env_fn()
        policy = CloudpickleWrapper(env.rand_action)
        if collector_cls == Collector:
            collector = collector_cls(
                env,
                policy=policy,
                frames_per_batch=20,
                total_frames=-1,
                set_truncated=True,
                trust_policy=True,
            )
        else:
            collector = collector_cls(
                [env_fn, env_fn],
                policy=policy,
                frames_per_batch=20,
                total_frames=-1,
                cat_results="stack",
                set_truncated=True,
                trust_policy=True,
            )
        try:
            for data in collector:
                assert data[..., -1]["next", "data", "truncated"].all()
                break
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize("frames_per_batch", [200, 10])
    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize("env_name", ["vec"])
    def test_split_trajs(self, num_env, env_name, frames_per_batch, seed=5):
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
                    create_env_kwargs=[
                        {"seed": i} for i in range(seed, seed + num_env)
                    ],
                )
                env.set_seed(seed)
                return env

        policy = make_policy(env_name)

        collector = Collector(
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

    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
        ],
    )  # AsyncCollector])
    @pytest.mark.parametrize(
        "env_name", ["vec"]
    )  # 1226: removing "conv" for efficiency
    def test_traj_len_consistency(self, num_env, env_name, collector_class, seed=100):
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

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize(
        "cudagraph", [False, True] if torch.cuda.is_available() else [False]
    )
    @pytest.mark.parametrize(
        "weight_sync_scheme",
        [None, MultiProcessWeightSyncScheme, SharedMemWeightSyncScheme],
    )
    # @pytest.mark.skipif(not torch.cuda.is_available() and not torch.mps.is_available(), reason="no cuda/mps device found")
    def test_update_weights(self, use_async, cudagraph, weight_sync_scheme):
        def create_env():
            return ContinuousActionVecMockEnv()

        n_actions = ContinuousActionVecMockEnv().action_spec.shape[-1]
        policy = SafeModule(
            torch.nn.LazyLinear(n_actions), in_keys=["observation"], out_keys=["action"]
        )
        policy(create_env().reset())

        collector_class = MultiSyncCollector if not use_async else MultiAsyncCollector
        kwargs = {}
        if weight_sync_scheme is not None:
            kwargs["weight_sync_schemes"] = {"policy": weight_sync_scheme()}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        collector = collector_class(
            [create_env] * 3,
            policy=policy,
            device=[torch.device(device)] * 3,
            storing_device=[torch.device(device)] * 3,
            frames_per_batch=20,
            cat_results="stack",
            cudagraph_policy=cudagraph,
            **kwargs,
        )
        try:
            # collect state_dict
            state_dict = collector.state_dict()
            policy_state_dict = policy.state_dict()
            for worker in range(3):
                assert "policy_state_dict" in state_dict[f"worker{worker}"], state_dict[
                    f"worker{worker}"
                ].keys()
                for k in state_dict[f"worker{worker}"]["policy_state_dict"]:
                    torch.testing.assert_close(
                        state_dict[f"worker{worker}"]["policy_state_dict"][k].cpu(),
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
                    with pytest.raises(
                        AssertionError
                    ) if torch.cuda.is_available() else nullcontext():
                        torch.testing.assert_close(
                            state_dict[f"worker{worker}"]["policy_state_dict"][k].cpu(),
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
                        state_dict[f"worker{worker}"]["policy_state_dict"][k].cpu(),
                        policy_state_dict[k].cpu(),
                    )
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize(
        "use_async", [True]
    )  # MultiSync has known indexing issues with SharedMem
    def test_update_weights_shared_mem(self, use_async):
        """Test shared memory weight synchronization scheme."""

        def create_env():
            return ContinuousActionVecMockEnv()

        n_actions = ContinuousActionVecMockEnv().action_spec.shape[-1]
        policy = SafeModule(
            torch.nn.LazyLinear(n_actions), in_keys=["observation"], out_keys=["action"]
        )
        policy(create_env().reset())

        # Get policy weights and put them in shared memory
        policy_weights = TensorDict.from_module(policy)
        policy_weights.share_memory_()

        # Create shared memory weight sync scheme
        weight_sync_scheme = SharedMemWeightSyncScheme()
        # Use the new init_on_sender API with params_map
        # All 3 workers share the same CPU weights in shared memory
        weight_sync_scheme.init_on_sender(
            params_map={0: policy_weights, 1: policy_weights, 2: policy_weights},
        )

        collector_class = MultiSyncCollector if not use_async else MultiAsyncCollector
        collector = collector_class(
            [create_env] * 3,
            policy=policy,
            frames_per_batch=20,
            cat_results="stack",
            weight_sync_schemes={"policy": weight_sync_scheme},
        )
        try:
            # Collect first batch
            for _ in collector:
                break

            # Change policy weights
            old_weight = policy.module.weight.data.clone()
            for p in policy.parameters():
                p.data += torch.randn_like(p)
            new_weight = policy.module.weight.data.clone()

            # Verify weights changed
            assert not torch.allclose(old_weight, new_weight)

            # Update weights using shared memory
            collector.update_policy_weights_()

            # Collect another batch - should use new weights
            for _ in collector:
                break

            # Verify shared memory was updated
            assert torch.allclose(policy_weights["module", "weight"], new_weight)

        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize("env_name", ["vec"])
    @pytest.mark.parametrize("frames_per_batch_worker", [[10, 10], [15, 5]])
    def test_collector_frames_per_batch_worker(
        self,
        num_env,
        env_name,
        frames_per_batch_worker,
        seed=100,
        num_workers=2,
    ):
        """Tests that there are 'sum(frames_per_batch_worker)' frames in each batch of a collection."""
        if num_env == 1:

            def env_fn():
                env = make_make_env(env_name)()
                return env

        else:

            def env_fn():
                # 1226: For efficiency, we don't use Parallel but Serial
                # env = ParallelEnv(
                env = SerialEnv(
                    num_workers=num_env, create_env_fn=make_make_env(env_name)
                )
                return env

        policy = make_policy(env_name)

        torch.manual_seed(0)
        np.random.seed(0)

        frames_per_batch = sum(frames_per_batch_worker)

        collector = MultiAsyncCollector(
            create_env_fn=[env_fn for _ in range(num_workers)],
            policy=policy,
            frames_per_batch=frames_per_batch_worker,
            max_frames_per_traj=1000,
            total_frames=frames_per_batch * 100,
        )
        try:
            collector.set_seed(seed)
            for i, b in enumerate(collector):
                assert b.numel() == -(-frames_per_batch // num_env) * num_env
                if i == 5:
                    break
            assert b.names[-1] == "time"
        finally:
            collector.shutdown()

        collector = MultiSyncCollector(
            create_env_fn=[env_fn for _ in range(num_workers)],
            policy=policy,
            frames_per_batch=frames_per_batch,
            max_frames_per_traj=1000,
            total_frames=frames_per_batch * 100,
            cat_results="stack",
        )
        try:
            collector.set_seed(seed)
            for i, b in enumerate(collector):
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
            collector.shutdown()
            del collector

        with pytest.raises(
            ValueError,
            match="If `frames_per_batch` is provided as a sequence, it should contain exactly one value per worker.",
        ):
            collector = MultiSyncCollector(
                create_env_fn=[env_fn for _ in range(num_workers)],
                policy=policy,
                frames_per_batch=frames_per_batch_worker[:-1],
                max_frames_per_traj=1000,
                total_frames=frames_per_batch * 100,
            )

    class FixedIDEnv(EnvBase):
        """
        A simple mock environment that returns a fixed ID as its sole observation.

        This environment is designed to test MultiSyncDataCollector ordering.
        Each environment instance is initialized with a unique env_id, which it
        returns as the observation at every step.
        """

        def __init__(
            self,
            env_id: int,
            max_steps: int = 10,
            sleep_odd_only: bool = False,
            **kwargs,
        ):
            """
            Args:
                env_id: The ID to return as observation. This will be returned as a tensor.
                max_steps: Maximum number of steps before the environment terminates.
            """
            super().__init__(device="cpu", batch_size=torch.Size([]))
            self.env_id = env_id
            self.max_steps = max_steps
            self.sleep_odd_only = sleep_odd_only
            self._step_count = 0

            # Define specs
            self.observation_spec = Composite(
                observation=Unbounded(shape=(1,), dtype=torch.float32)
            )
            self.action_spec = Composite(
                action=Unbounded(shape=(1,), dtype=torch.float32)
            )
            self.reward_spec = Composite(
                reward=Unbounded(shape=(1,), dtype=torch.float32)
            )
            self.done_spec = Composite(
                done=Unbounded(shape=(1,), dtype=torch.bool),
                terminated=Unbounded(shape=(1,), dtype=torch.bool),
                truncated=Unbounded(shape=(1,), dtype=torch.bool),
            )

        def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
            """Reset the environment and return initial observation."""
            # Add sleep to simulate real-world timing variations
            # This helps test that the collector properly handles different reset times
            if not self.sleep_odd_only:
                # Random sleep up to 10ms
                time.sleep(torch.rand(1).item() * 0.01)
            elif self.env_id % 2 == 1:
                time.sleep(0.1)

            self._step_count = 0
            return TensorDict(
                {
                    "observation": torch.tensor(
                        [float(self.env_id)], dtype=torch.float32
                    ),
                    "done": torch.tensor([False], dtype=torch.bool),
                    "terminated": torch.tensor([False], dtype=torch.bool),
                    "truncated": torch.tensor([False], dtype=torch.bool),
                },
                batch_size=self.batch_size,
            )

        def _step(self, tensordict: TensorDict) -> TensorDict:
            """Execute one step and return the env_id as observation."""
            self._step_count += 1
            done = self._step_count >= self.max_steps

            if self.sleep_odd_only and self.env_id % 2 == 1:
                time.sleep(0.1)

            return TensorDict(
                {
                    "observation": torch.tensor(
                        [float(self.env_id)], dtype=torch.float32
                    ),
                    "reward": torch.tensor([1.0], dtype=torch.float32),
                    "done": torch.tensor([done], dtype=torch.bool),
                    "terminated": torch.tensor([done], dtype=torch.bool),
                    "truncated": torch.tensor([False], dtype=torch.bool),
                },
                batch_size=self.batch_size,
            )

        def _set_seed(self, seed: int | None) -> int | None:
            """Set the seed for reproducibility."""
            if seed is not None:
                torch.manual_seed(seed)
            return seed

    @pytest.mark.parametrize("num_envs,n_steps", [(8, 5)])
    @pytest.mark.parametrize("with_preempt", [False, True])
    @pytest.mark.parametrize("cat_results", ["stack", -1])
    def test_multi_sync_data_collector_ordering(
        self, num_envs: int, n_steps: int, with_preempt: bool, cat_results: str | int
    ):
        """
        Test that MultiSyncDataCollector returns data in the correct order.

        We create num_envs environments, each returning its env_id as the observation.
        After collection, we verify that the observations correspond to the correct env_ids in order
        """
        if with_preempt and IS_OSX:
            pytest.skip(
                "Cannot use preemption on OSX due to Queue.qsize() not being implemented on this platform."
            )

        # Create environment factories using partial - one for each env_id
        # This pattern mirrors CrossPlayEvaluator._rollout usage
        env_factories = [
            functools.partial(
                self.FixedIDEnv, env_id=i, max_steps=10, sleep_odd_only=with_preempt
            )
            for i in range(num_envs)
        ]

        collector = MultiSyncCollector(
            create_env_fn=env_factories,
            frames_per_batch=num_envs * n_steps,
            total_frames=num_envs * n_steps,
            device="cpu",
            preemptive_threshold=0.5 if with_preempt else None,
            cat_results=cat_results,
            init_random_frames=n_steps,  # no need of a policy
            use_buffers=True,
        )

        try:
            # Collect one batch
            for batch in collector:
                # Verify that each environment's observations match its env_id
                # batch has shape [num_envs, frames_per_env]
                # In the pre-emption case, we have that envs with odd ids are order of magnitude slower.
                # These should be skipped by pre-emption (since they are the 50% slowest)

                # Recover rectangular shape of batch to uniform checks
                if cat_results != "stack":
                    if not with_preempt:
                        batch = batch.reshape(num_envs, n_steps)
                    else:
                        traj_ids = batch["collector", "traj_ids"]
                        traj_ids[traj_ids == 0] = 99  # avoid using traj_ids = 0
                        # Split trajectories to recover correct shape
                        # thanks to having a single trajectory per env
                        # Pads with zeros!
                        batch = split_trajectories(
                            batch, trajectory_key=("collector", "traj_ids")
                        )
                        # Use -1 for padding to uniform with other preemption
                        is_padded = batch["collector", "traj_ids"] == 0
                        batch[is_padded] = -1

                #
                for env_idx in range(num_envs):
                    if with_preempt and env_idx % 2 == 1:
                        # This is a slow env, should have been preempted after first step
                        assert (batch["collector", "traj_ids"][env_idx, 1:] == -1).all()
                        continue
                    # This is a fast env, no preemption happened
                    assert (batch["collector", "traj_ids"][env_idx] != -1).all()

                    env_data = batch[env_idx]
                    observations = env_data["observation"]
                    # All observations from this environment should equal its env_id
                    expected_id = float(env_idx)
                    actual_ids = observations.flatten().unique()

                    assert len(actual_ids) == 1, (
                        f"Env {env_idx} should only produce observations with value {expected_id}, "
                        f"but got {actual_ids.tolist()}"
                    )
                    assert (
                        actual_ids[0].item() == expected_id
                    ), f"Environment {env_idx} should produce observation {expected_id}, but got {actual_ids[0].item()}"
        finally:
            collector.shutdown()

    def test_collector_next_method(self):
        """Non-regression test: next() should work correctly after __iter__.

        Previously, `__iter__` set `_iterator = True` as a flag, but `next()` expected
        `_iterator` to be either `None` or an actual iterator object. This test ensures
        that calling `next()` works correctly.
        """
        env = ContinuousActionVecMockEnv()
        policy = RandomPolicy(env.action_spec)

        collector = Collector(
            env,
            policy,
            total_frames=500,
            frames_per_batch=50,
        )
        try:
            # Test calling next() multiple times
            data1 = collector.next()
            assert data1 is not None, "next() should return data"
            assert data1.numel() == 50, f"Expected 50 frames, got {data1.numel()}"

            data2 = collector.next()
            assert data2 is not None, "second next() should return data"
            assert data2.numel() == 50, f"Expected 50 frames, got {data2.numel()}"

            # Test that we can still iterate after calling next()
            count = 0
            for data in collector:
                assert data.numel() == 50
                count += 1
                if count >= 2:
                    break
        finally:
            collector.shutdown()

    @pytest.mark.parametrize("use_buffers", [True, False])
    @pytest.mark.parametrize("storing_device", [None, "cpu"])
    def test_unbatched_env_traj_ids_shape_consistency(
        self, use_buffers, storing_device
    ):
        """Regression test for issue #3137: traj_ids shape inconsistency with unbatched envs.

        When using SyncDataCollector with an unbatched environment (batch_size=()),
        the traj_ids should maintain consistent shapes across all steps, even when
        done=True triggers trajectory updates.

        See: https://github.com/pytorch/rl/issues/3137
        """

        class UnbatchedDoneEnv(EnvBase):
            """Unbatched environment that returns done=True after N steps."""

            def __init__(self, done_after_n_steps=6):
                super().__init__(batch_size=torch.Size([]))
                self.done_after_n_steps = done_after_n_steps
                self._step_count = 0

                self.observation_spec = Composite(
                    observation=Unbounded(shape=(3,)),
                )
                self.action_spec = Composite(
                    action=Unbounded(shape=(1,)),
                )
                self.reward_spec = Composite(
                    reward=Unbounded(shape=(1,)),
                )
                self.full_done_spec = Composite(
                    done=Unbounded(shape=(1,), dtype=torch.bool),
                    terminated=Unbounded(shape=(1,), dtype=torch.bool),
                    truncated=Unbounded(shape=(1,), dtype=torch.bool),
                )

            def _reset(self, tensordict=None):
                self._step_count = 0
                return TensorDict(
                    {
                        "observation": torch.rand(3),
                        "done": torch.tensor([False]),
                        "terminated": torch.tensor([False]),
                        "truncated": torch.tensor([False]),
                    },
                    batch_size=self.batch_size,
                )

            def _step(self, tensordict):
                self._step_count += 1
                done = self._step_count >= self.done_after_n_steps

                return TensorDict(
                    {
                        "observation": torch.rand(3),
                        "reward": torch.tensor([1.0]),
                        "done": torch.tensor([done]),
                        "terminated": torch.tensor([done]),
                        "truncated": torch.tensor([False]),
                    },
                    batch_size=self.batch_size,
                )

            def _set_seed(self, seed):
                torch.manual_seed(seed)

        env = UnbatchedDoneEnv(done_after_n_steps=6)
        policy = RandomPolicy(env.action_spec)

        collector = Collector(
            create_env_fn=lambda: UnbatchedDoneEnv(done_after_n_steps=6),
            policy=policy,
            total_frames=100,
            frames_per_batch=30,
            use_buffers=use_buffers,
            storing_device=storing_device,
        )

        try:
            for i, data in enumerate(collector):
                # Verify the data has the expected shape
                assert data.shape == torch.Size(
                    [30]
                ), f"Batch {i}: expected shape [30], got {data.shape}"

                # Verify traj_ids exists and has consistent shape
                traj_ids = data.get(("collector", "traj_ids"))
                assert traj_ids is not None, "traj_ids should be present"
                assert traj_ids.shape == torch.Size(
                    [30]
                ), f"Batch {i}: traj_ids expected shape [30], got {traj_ids.shape}"

                # Verify traj_ids values are valid (non-negative integers)
                assert (traj_ids >= 0).all(), "traj_ids should be non-negative"

                if i >= 2:
                    break
        finally:
            collector.shutdown()


class TestCollectorDevices:
    class DeviceLessEnv(EnvBase):
        # receives data on cpu, outputs on gpu -- tensordict has no device
        def __init__(self, default_device):
            self.default_device = default_device
            super().__init__(device=None)
            self.observation_spec = Composite(
                observation=Unbounded((), device=default_device)
            )
            self.reward_spec = Unbounded(1, device=default_device)
            self.full_done_spec = Composite(
                done=Unbounded(1, dtype=torch.bool, device=self.default_device),
                truncated=Unbounded(1, dtype=torch.bool, device=self.default_device),
                terminated=Unbounded(1, dtype=torch.bool, device=self.default_device),
            )
            self.action_spec = Unbounded((), device=None)
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

        def _set_seed(self, seed: int | None = None) -> None:
            ...

    class EnvWithDevice(EnvBase):
        def __init__(self, default_device):
            self.default_device = default_device
            super().__init__(device=self.default_device)
            self.observation_spec = Composite(
                observation=Unbounded((), device=self.default_device)
            )
            self.reward_spec = Unbounded(1, device=self.default_device)
            self.full_done_spec = Composite(
                done=Unbounded(1, dtype=torch.bool, device=self.default_device),
                truncated=Unbounded(1, dtype=torch.bool, device=self.default_device),
                terminated=Unbounded(1, dtype=torch.bool, device=self.default_device),
                device=self.default_device,
            )
            self.action_spec = Unbounded((), device=self.default_device)
            assert self.device == _make_ordinal_device(
                torch.device(self.default_device)
            )
            assert self.full_observation_spec is not None
            assert self.full_done_spec is not None
            assert self.full_state_spec is not None
            assert self.full_action_spec is not None
            assert self.full_reward_spec is not None

        def _step(self, tensordict):
            assert tensordict.device == _make_ordinal_device(
                torch.device(self.default_device)
            )
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

        def _set_seed(self, seed: int | None = None) -> None:
            ...

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

        def __init__(self, default_device=None):
            super().__init__()
            self.default_device = (
                default_device
                if default_device is not None
                else ("cuda:0" if torch.cuda.device_count() else "cpu")
            )

        def forward(self, tensordict):
            assert tensordict.device == _make_ordinal_device(
                torch.device(self.default_device)
            )
            return tensordict.set("action", torch.zeros((), device=self.default_device))

    @pytest.mark.parametrize("main_device", get_default_devices())
    @pytest.mark.parametrize("storing_device", [None, *get_default_devices()])
    def test_output_device(self, main_device, storing_device):

        # env has no device, policy is strictly on GPU
        device = None
        env_device = None
        policy_device = main_device
        env = self.DeviceLessEnv(main_device)
        policy = self.PolicyWithDevice(main_device)
        collector = Collector(
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
        collector = Collector(
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
        policy = self.PolicyWithDevice(main_device)
        collector = Collector(
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
        # When storing_device is None, it falls back to device
        expected_device = storing_device if storing_device is not None else main_device
        assert data.device == expected_device

        # same but more specific
        device = None
        env_device = main_device
        policy_device = main_device
        env = self.EnvWithDevice(main_device)
        policy = self.PolicyWithDevice(main_device)
        collector = Collector(
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
        # When storing_device is None, and env_device == policy_device, it falls back to env_device
        expected_device = storing_device if storing_device is not None else main_device
        assert data.device == expected_device

        # none has a device
        device = None
        env_device = None
        policy_device = None
        env = self.DeviceLessEnv(main_device)
        policy = self.DeviceLessPolicy()
        collector = Collector(
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

    class CudaPolicy(TensorDictSequential):
        def __init__(self, n_obs):
            module = torch.nn.Linear(n_obs, n_obs, device="cuda")
            module.weight.data.copy_(torch.eye(n_obs))
            module.bias.data.fill_(0)
            m0 = TensorDictModule(module, in_keys=["observation"], out_keys=["hidden"])
            m1 = TensorDictModule(
                lambda a: a + 1, in_keys=["hidden"], out_keys=["action"]
            )
            super().__init__(m0, m1)

    class GoesThroughEnv(EnvBase):
        def __init__(self, n_obs, device):
            self.observation_spec = Composite(observation=Unbounded(n_obs))
            self.action_spec = Unbounded(n_obs)
            self.reward_spec = Unbounded(1)
            self.full_done_specs = Composite(done=Unbounded(1, dtype=torch.bool))
            super().__init__(device=device)

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            a = tensordict["action"]
            if self.device is not None:
                assert a.device == self.device
            out = tensordict.empty()
            out["observation"] = tensordict["observation"] + (
                a - tensordict["observation"]
            )
            out["reward"] = torch.zeros((1,), device=self.device)
            out["done"] = torch.zeros((1,), device=self.device, dtype=torch.bool)
            return out

        def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
            return self.full_done_specs.zeros().update(self.observation_spec.zeros())

        def _set_seed(self, seed: int | None) -> None:
            ...

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    @pytest.mark.parametrize("env_device", ["cuda:0", "cpu"])
    @pytest.mark.parametrize("storing_device", [None, "cuda:0", "cpu"])
    @pytest.mark.parametrize("no_cuda_sync", [True, False])
    def test_no_synchronize(self, env_device, storing_device, no_cuda_sync):
        """Tests that no_cuda_sync avoids any call to torch.cuda.synchronize() and that the data is not corrupted."""
        should_raise = not no_cuda_sync
        should_raise = should_raise & (
            (env_device == "cpu") or (storing_device == "cpu")
        )
        with patch("torch.cuda.synchronize") as mock_synchronize, pytest.raises(
            AssertionError, match="Expected 'synchronize' to not have been called."
        ) if should_raise else contextlib.nullcontext():
            collector = Collector(
                create_env_fn=functools.partial(
                    self.GoesThroughEnv, n_obs=1000, device=None
                ),
                policy=self.CudaPolicy(n_obs=1000),
                frames_per_batch=100,
                total_frames=1000,
                env_device=env_device,
                storing_device=storing_device,
                policy_device="cuda:0",
                no_cuda_sync=no_cuda_sync,
            )
            assert collector.env.device == torch.device(env_device)
            i = 0
            for d in collector:
                for _d in d.unbind(0):
                    u = _d["observation"].unique()
                    assert u.numel() == 1, i
                    assert u == i, i
                    i += 1
                    u = _d["next", "observation"].unique()
                    assert u.numel() == 1, i
                    assert u == i, i
                mock_synchronize.assert_not_called()

    @pytest.mark.gpu
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("storing_device", ["cuda", "cpu"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
    def test_collector_device_combinations(self, device, storing_device):
        if IS_WINDOWS and PYTHON_3_10 and storing_device == "cuda" and device == "cuda":
            pytest.skip("Windows fatal exception: access violation in torch.storage")

        def env_fn(seed):
            env = make_make_env("conv")()
            env.set_seed(seed)
            return env

        policy = dummypolicy_conv()

        collector = Collector(
            create_env_fn=env_fn,
            create_env_kwargs={"seed": 0},
            policy=policy,
            frames_per_batch=20,
            max_frames_per_traj=2000,
            total_frames=20000,
            device=device,
            storing_device=storing_device,
        )
        assert collector._use_buffers
        batch = next(collector.iterator())
        assert batch.device == _make_ordinal_device(torch.device(storing_device))
        collector.shutdown()

        collector = MultiSyncCollector(
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
        assert batch.device == _make_ordinal_device(torch.device(storing_device))
        collector.shutdown()

        collector = MultiAsyncCollector(
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
        assert batch.device == _make_ordinal_device(torch.device(storing_device))
        collector.shutdown()
        del collector


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
#     collector = Collector(
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
#     ccollector = AsyncCollector(
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
#     collector = Collector(
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
#     ccollector = AsyncCollector(
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


@pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
@pytest.mark.parametrize(
    "collector_class,num_envs",
    [
        (Collector, 1),
        (MultiAsyncCollector, 1),
        (functools.partial(MultiSyncCollector, cat_results="stack"), 1),
        (MultiAsyncCollector, 2),
        (functools.partial(MultiSyncCollector, cat_results="stack"), 2),
    ],
)
class TestAutoWrap:
    @pytest.fixture
    def env_maker(self):
        return lambda: GymEnv(PENDULUM_VERSIONED())

    def _create_collector_kwargs(self, env_maker, collector_class, policy, num_envs):
        collector_kwargs = {
            "create_env_fn": env_maker,
            "policy": policy,
            "frames_per_batch": 200,
            "total_frames": -1,
        }

        if collector_class is not Collector:
            collector_kwargs["create_env_fn"] = [
                collector_kwargs["create_env_fn"] for _ in range(num_envs)
            ]

        return collector_kwargs

    @pytest.mark.parametrize("multiple_outputs", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_auto_wrap_modules(
        self, collector_class, multiple_outputs, env_maker, device, num_envs
    ):
        policy = WrappablePolicy(
            out_features=env_maker().action_spec.shape[-1],
            multiple_outputs=multiple_outputs,
        )
        # init lazy params
        policy(env_maker().reset().get("observation"))

        collector = collector_class(
            **self._create_collector_kwargs(
                env_maker, collector_class, policy, num_envs
            ),
            device=device,
        )
        if isinstance(collector, MultiCollector):
            assert collector._weight_sync_schemes is not None
            assert "policy" in collector._weight_sync_schemes

        try:
            out_keys = ["action"]
            if multiple_outputs:
                out_keys.extend(f"output{i}" for i in range(1, 4))

            if collector_class is Collector:
                assert isinstance(collector._wrapped_policy, TensorDictModule)
                assert collector._wrapped_policy.out_keys == out_keys
                # this does not work now that we force the device of the policy
                # assert collector.policy.module is policy

            for i, data in enumerate(collector):  # noqa: B007
                # Debug: iteration {i}
                if i == 0:
                    assert (data["action"] != 0).any()
                    for p in policy.parameters():
                        p.data.zero_()
                        assert p.device == torch.device("cpu")
                    # Debug: updating policy weights
                    torchrl_logger.debug("Calling update_policy_weights_")
                    collector.update_policy_weights_()
                    # Debug: updated policy weights
                elif i == 4:
                    assert (data["action"] == 0).all()
                    break
        finally:
            # Debug: shutting down collector
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
    #     if collector_class is not Collector:
    #         # We now do the casting only on the remote workers
    #         pass
    #     else:
    #         assert isinstance(collector.policy, TensorDictCompatiblePolicy)
    #         assert collector.policy.out_keys == ["action"]
    #         assert collector.policy is policy
    #
    #     for i, data in enumerate(collector):  # noqa: B007
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

    def test_auto_wrap_error(self, collector_class, env_maker, num_envs):
        policy = UnwrappablePolicy(out_features=env_maker().action_spec.shape[-1])
        with pytest.raises(
            TypeError,
            match=(
                "Arguments to policy.forward are incompatible with entries in|Failed to wrap the policy. If the policy needs to be trusted, set trust_policy=True."
            ),
        ):
            collector_class(
                **self._create_collector_kwargs(
                    env_maker, collector_class, policy, num_envs
                )
            )


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

        collector = Collector(
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

        collector = MultiSyncCollector(
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

    def test_multisync_split_trajs_set_seed(self):
        """Test that MultiSyncCollector with split_trajs=True and set_seed works without errors."""
        from torchrl.testing.mocking_classes import CountingEnv

        env_maker = lambda: CountingEnv(max_steps=100)
        policy = RandomPolicy(env_maker().action_spec)
        collector = MultiSyncCollector(
            create_env_fn=[env_maker, env_maker],
            policy=policy,
            total_frames=2000,
            max_frames_per_traj=50,
            frames_per_batch=200,
            init_random_frames=-1,
            reset_at_each_iter=False,
            device="cpu",
            storing_device="cpu",
            cat_results="stack",
            split_trajs=True,
        )
        collector.set_seed(42)
        try:
            for i, data in enumerate(collector):  # noqa: B007
                if i == 2:
                    break
            # Check that traj_ids are unique across the batch
            traj_ids = data.get(("collector", "traj_ids"))
            # Each row is one trajectory; all elements in a row share the same traj_id
            # Check that each trajectory has a unique id
            traj_ids_per_traj = traj_ids.select(-1, 0)
            assert (
                traj_ids_per_traj.unique().numel() == traj_ids_per_traj.numel()
            ), "traj_ids should be unique across trajectories"
        finally:
            collector.shutdown()
            del collector


class TestNestedEnvsCollector:
    def test_multi_collector_nested_env_consistency(self, seed=1):
        torch.manual_seed(seed)
        env_fn = lambda: TransformedEnv(NestedCountingEnv(), InitTracker())
        env = NestedCountingEnv()
        policy = CountingEnvCountPolicy(
            env.full_action_spec[env.action_key], env.action_key
        )

        ccollector = MultiAsyncCollector(
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

        ccollector = MultiSyncCollector(
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
        policy = CountingEnvCountPolicy(
            env.full_action_spec[env.action_key], env.action_key
        )
        ccollector = Collector(
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
        env = NestedCountingEnv(batch_size=batch_size, nested_dim=nested_dim)
        env_fn = lambda: NestedCountingEnv(batch_size=batch_size, nested_dim=nested_dim)
        torch.manual_seed(0)
        policy = CountingEnvCountPolicy(
            env.full_action_spec[env.action_key], env.action_key
        )
        policy(env.reset())
        ccollector = Collector(
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
        ccollector = Collector(
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

        ccollector = MultiAsyncCollector(
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

        ccollector = MultiSyncCollector(
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
        ccollector = Collector(
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

        ccollector = MultiAsyncCollector(
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

        ccollector = MultiSyncCollector(
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() and (not has_mps()),
    reason="No casting if no cuda",
)
class TestUpdateParams:
    class DummyEnv(EnvBase):
        def __init__(self, device, batch_size=[]):  # noqa: B006
            super().__init__(batch_size=batch_size, device=device)
            self.state = torch.zeros(self.batch_size, device=device)
            self.observation_spec = Composite(state=Unbounded(shape=(), device=device))
            self.action_spec = Unbounded(shape=batch_size, device=device)
            self.reward_spec = Unbounded(shape=(*batch_size, 1), device=device)

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

        def _set_seed(self, seed: int | None) -> None:
            ...

    class Policy(TensorDictModuleBase):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(()))
            self.register_buffer("buf", torch.zeros(()))
            self.in_keys = []
            self.out_keys = ["action"]

        def forward(self, td):
            td["action"] = (self.param + self.buf.to(self.param.device)).expand(
                td.shape
            )
            return td

    @pytest.mark.parametrize(
        "collector",
        [
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("give_weights", [True, False])
    @pytest.mark.parametrize(
        "policy_device,env_device",
        [
            ["cpu", get_default_devices()[0]],
            [get_default_devices()[0], "cpu"],
            # ["cpu", "cuda:0"],  # 1226: faster execution
            # ["cuda:0", "cpu"],
            # ["cuda", "cuda:0"],
            # ["cuda:0", "cuda"],
        ],
    )
    @pytest.mark.parametrize(
        "weight_sync_scheme",
        [None, MultiProcessWeightSyncScheme, SharedMemWeightSyncScheme],
    )
    def test_param_sync(
        self, give_weights, collector, policy_device, env_device, weight_sync_scheme
    ):
        policy = TestUpdateParams.Policy().to(policy_device)

        env = EnvCreator(lambda: TestUpdateParams.DummyEnv(device=env_device))
        device = env().device
        env = [env]
        kwargs = {}
        if weight_sync_scheme is not None:
            kwargs["weight_sync_schemes"] = {"policy": weight_sync_scheme()}
        col = collector(
            env,
            policy,
            device=device,
            total_frames=200,
            frames_per_batch=10,
            **kwargs,
        )
        try:
            for i, data in enumerate(col):
                if i == 0:
                    assert (data["action"] == 0).all()
                    # update policy
                    policy.param.data += 1
                    policy.buf.data += 2
                    if give_weights:
                        p_w = TensorDict.from_module(policy)
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

    @pytest.mark.parametrize(
        "collector",
        [
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("give_weights", [True, False])
    @pytest.mark.parametrize(
        "policy_device,env_device",
        [
            ["cpu", get_default_devices()[0]],
            [get_default_devices()[0], "cpu"],
            # ["cpu", "cuda:0"],  # 1226: faster execution
            # ["cuda:0", "cpu"],
            # ["cuda", "cuda:0"],
            # ["cuda:0", "cuda"],
        ],
    )
    @pytest.mark.parametrize(
        "weight_sync_scheme",
        [None, MultiProcessWeightSyncScheme, SharedMemWeightSyncScheme],
    )
    def test_param_sync_mixed_device(
        self, give_weights, collector, policy_device, env_device, weight_sync_scheme
    ):
        # Skip multi-collectors on macOS with older PyTorch when MPS is available.
        # On macOS: "fork" causes segfaults after MPS initialization (even with CPU tensors),
        # and "spawn" on older PyTorch (<2.5) can't handle some multiprocessing scenarios.
        is_multi_collector = collector is not Collector
        is_macos = sys.platform == "darwin"
        is_old_pytorch = version.parse(torch.__version__).base_version < "2.5.0"
        mps_available = torch.backends.mps.is_available()
        if is_multi_collector and is_macos and is_old_pytorch and mps_available:
            pytest.skip(
                "Multi-collectors are not supported on macOS with MPS available and PyTorch < 2.5.0 "
                "due to multiprocessing compatibility issues with MPS initialization."
            )

        with torch.device("cpu"):
            policy = TestUpdateParams.Policy()
        policy.param = nn.Parameter(policy.param.data.to(policy_device))
        assert policy.buf.device == torch.device("cpu")

        env = EnvCreator(lambda: TestUpdateParams.DummyEnv(device=env_device))
        device = env().device
        env = [env]
        kwargs = {}
        if weight_sync_scheme is not None:
            kwargs["weight_sync_schemes"] = {"policy": weight_sync_scheme()}
        col = collector(
            env,
            policy,
            device=device,
            total_frames=200,
            frames_per_batch=10,
            **kwargs,
        )
        try:
            for i, data in enumerate(col):
                if i == 0:
                    assert (data["action"] == 0).all()
                    # update policy
                    policy.param.data += 1
                    policy.buf.data += 2
                    assert policy.buf.device == torch.device("cpu")
                    if give_weights:
                        p_w = TensorDict.from_module(policy)
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

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 3,
        reason="requires at least 3 CUDA devices",
    )
    def test_shared_device_weight_update(self):
        """Test that weight updates work correctly when multiple workers share the same device.

        This test specifically validates the per-worker queue implementation in SharedMemWeightSyncScheme.
        When workers 0 and 2 share cuda:2, each should receive its own copy of the weights through
        dedicated queues, preventing race conditions that could occur with a single shared queue.

        Note: This test only uses SharedMemWeightSyncScheme (not MultiProcessWeightSyncScheme) because
        the latter sends tensors through pipes, which we want to avoid.
        """
        # Create policy on cuda:0
        policy = TensorDictModule(
            nn.Linear(7, 7, device="cuda:0"),
            in_keys=["observation"],
            out_keys=["action"],
        )

        def make_env():
            return ContinuousActionVecMockEnv()

        # Create collector with workers on cuda:2, cuda:1, cuda:2
        # Workers 0 and 2 share cuda:2 - this is the key test case
        collector = MultiAsyncCollector(
            [make_env, make_env, make_env],
            policy=policy,
            frames_per_batch=30,
            total_frames=300,
            device=["cuda:2", "cuda:1", "cuda:2"],
            storing_device=["cuda:2", "cuda:1", "cuda:2"],
            weight_sync_schemes={"policy": SharedMemWeightSyncScheme()},
        )

        try:
            # Collect first batch to initialize workers
            for _ in collector:
                break

            # Get initial weights
            old_weight = policy.module.weight.data.clone()

            # Modify policy weights on cuda:0
            for p in policy.parameters():
                p.data += torch.randn_like(p)

            new_weight = policy.module.weight.data.clone()
            assert not torch.allclose(
                old_weight, new_weight
            ), "Weights should have changed"

            # Update weights - this should propagate to all workers via their dedicated queues
            collector.update_policy_weights_()

            # Collect more batches to ensure weights are propagated
            for i, _ in enumerate(collector):
                if i >= 2:
                    break

            # Get state dict from all workers
            state_dict = collector.state_dict()

            # Verify all workers have the new weights, including both workers on cuda:2
            for worker_idx in range(3):
                worker_key = f"worker{worker_idx}"
                assert (
                    "policy_state_dict" in state_dict[worker_key]
                ), f"Worker {worker_idx} should have policy_state_dict"
                worker_weight = state_dict[worker_key]["policy_state_dict"][
                    "module.weight"
                ]
                torch.testing.assert_close(
                    worker_weight.cpu(),
                    new_weight.cpu(),
                    msg=(
                        f"Worker {worker_idx} weights don't match expected weights. "
                        f"Workers 0 and 2 share device cuda:2, worker 1 is on cuda:1. "
                        f"This test validates that the per-worker queue system correctly "
                        f"distributes weights even when multiple workers share a device."
                    ),
                )
        finally:
            collector.shutdown()
            del collector


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


def _subprocess_test_worker(func, error_queue):
    """Worker function that runs a test function and reports errors via queue."""
    try:
        func()
    except Exception as e:
        error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
    else:
        error_queue.put(None)


def _run_test_in_subprocess(func, timeout=120):
    """Run a test function in a fresh subprocess to avoid thread pool initialization issues.

    This is necessary because torch.set_num_threads() may not work correctly
    if the thread pool has already been initialized in the parent process.
    Running in a fresh subprocess ensures a clean PyTorch state.

    Args:
        func: The test function to run. Must be picklable (module-level function).
        timeout: Timeout in seconds for the subprocess.

    Raises:
        AssertionError: If the test function raises an exception in the subprocess.
    """
    ctx = torch.multiprocessing.get_context("spawn")
    error_queue = ctx.Queue()

    proc = ctx.Process(target=_subprocess_test_worker, args=(func, error_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise AssertionError(f"Test timed out after {timeout} seconds")

    if proc.exitcode != 0:
        try:
            result = error_queue.get_nowait()
        except Exception:
            result = None

        if result is not None:
            exc_type, exc_msg, tb = result
            raise AssertionError(f"Test failed with {exc_type}: {exc_msg}\n{tb}")
        else:
            raise AssertionError(f"Test subprocess exited with code {proc.exitcode}")

    # Check if there was an exception even with exitcode 0
    try:
        result = error_queue.get_nowait()
        if result is not None:
            exc_type, exc_msg, tb = result
            raise AssertionError(f"Test failed with {exc_type}: {exc_msg}\n{tb}")
    except Exception:
        pass


def _test_num_threads_impl():
    """Implementation of test_num_threads that runs in a subprocess."""
    env = ContinuousActionVecMockEnv()
    _main_async_collector_saved = torchrl.collectors._multi_base._main_async_collector
    torchrl.collectors._multi_base._main_async_collector = decorate_thread_sub_func(
        torchrl.collectors._multi_base._main_async_collector, num_threads=3
    )
    num_threads = torch.get_num_threads()
    try:
        c = MultiSyncCollector(
            [env],
            policy=RandomPolicy(env.action_spec),
            num_threads=7,
            num_sub_threads=3,
            total_frames=200,
            frames_per_batch=200,
            cat_results="stack",
        )
        assert (
            torch.get_num_threads() == 7
        ), f"Expected 7 threads, got {torch.get_num_threads()}"
        for _ in c:
            pass
    finally:
        try:
            c.shutdown()
            del c
        except Exception:
            pass
        torchrl.collectors._multi_base._main_async_collector = (
            _main_async_collector_saved
        )
        torch.set_num_threads(num_threads)


def _test_auto_num_threads_impl():
    """Implementation of test_auto_num_threads that runs in a subprocess."""
    init_threads = torch.get_num_threads()

    # Test 1: Single env
    try:
        collector = MultiSyncCollector(
            [ContinuousActionVecMockEnv],
            RandomPolicy(ContinuousActionVecMockEnv().full_action_spec),
            frames_per_batch=3,
            cat_results="stack",
        )
        for _ in collector:
            current = torch.get_num_threads()
            expected = init_threads - 1
            assert current == expected, f"Expected {expected} threads, got {current}"
            break
        collector.shutdown()
        current = torch.get_num_threads()
        assert (
            current == init_threads
        ), f"After shutdown: expected {init_threads} threads, got {current}"
        del collector
        gc.collect()
    finally:
        torch.set_num_threads(init_threads)

    # Test 2: ParallelEnv with 2 workers
    try:
        collector = MultiSyncCollector(
            [ParallelEnv(2, ContinuousActionVecMockEnv)],
            RandomPolicy(ContinuousActionVecMockEnv().full_action_spec.expand(2)),
            frames_per_batch=3,
            cat_results="stack",
        )
        for _ in collector:
            current = torch.get_num_threads()
            expected = init_threads - 2
            assert current == expected, f"Expected {expected} threads, got {current}"
            break
        collector.shutdown()
        current = torch.get_num_threads()
        assert (
            current == init_threads
        ), f"After shutdown: expected {init_threads} threads, got {current}"
        del collector
        gc.collect()
    finally:
        torch.set_num_threads(init_threads)


class TestLibThreading:
    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_num_threads(self):
        _run_test_in_subprocess(_test_num_threads_impl)

    @pytest.mark.skipif(
        IS_OSX or IS_WINDOWS,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_auto_num_threads(self):
        _run_test_in_subprocess(_test_auto_num_threads_impl)


class TestUniqueTraj:
    @pytest.mark.skipif(not _has_gym, reason="Gym not available")
    @pytest.mark.parametrize("cat_results", ["stack", 0])
    def test_unique_traj_sync(self, cat_results):
        stack_results = cat_results == "stack"
        buffer = ReplayBuffer(
            storage=LazyTensorStorage(900, ndim=2 + stack_results), batch_size=16
        )
        c = MultiSyncCollector(
            [SerialEnv(2, EnvCreator(lambda: GymEnv("CartPole-v1")))] * 3,
            policy=RandomPolicy(GymEnv("CartPole-v1").action_spec),
            total_frames=900,
            frames_per_batch=300,
            cat_results=cat_results,
        )
        try:
            for d in c:
                buffer.extend(d)
            assert c._use_buffers
            traj_ids = buffer[:].get(("collector", "traj_ids"))
            # Ideally, we'd like that (sorted_traj.values == sorted_traj.indices).all()
            #  but in practice, one env can reach the end of the rollout and do a reset
            #  (which we don't want to prevent) and increment the global traj count,
            #  when the others have not finished yet. In that case, this traj number will never
            #  appear.
            # sorted_traj = traj_ids.unique().sort()
            # assert (sorted_traj.values == sorted_traj.indices).all()
            # assert traj_ids.unique().numel() == traj_ids.max() + 1

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


class TestDynamicEnvs:
    def test_dynamic_sync_collector(self):
        env = EnvWithDynamicSpec()
        policy = RandomPolicy(env.action_spec)
        collector = Collector(env, policy, frames_per_batch=20, total_frames=100)
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert data.names[-1] == "time"

    @pytest.mark.parametrize("policy_device", [None, *get_default_devices()])
    def test_dynamic_multisync_collector(self, policy_device):
        env = EnvWithDynamicSpec
        spec = env().action_spec
        if policy_device is not None:
            spec = spec.to(policy_device)
        policy = RandomPolicy(spec)
        collector = MultiSyncCollector(
            [env],
            policy,
            frames_per_batch=20,
            total_frames=100,
            use_buffers=False,
            cat_results="stack",
            policy_device=policy_device,
            env_device="cpu",
            storing_device="cpu",
        )
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert data.names[-1] == "time"

    def test_dynamic_multiasync_collector(self):
        env = EnvWithDynamicSpec
        policy = RandomPolicy(env().action_spec)
        collector = MultiAsyncCollector(
            [env],
            policy,
            frames_per_batch=20,
            total_frames=100,
            # use_buffers=False,
        )
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert data.names[-1] == "time"


@pytest.mark.skipif(not _has_gym, reason="gym required for this test")
class TestCollectorsNonTensor:
    class AddNontTensorData(Transform):
        def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
            next_tensordict[
                "nt"
            ] = f"a string! - {next_tensordict.get('step_count').item()}"
            return next_tensordict

        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            return tensordict_reset.set("nt", NonTensorData("reset!"))

        def transform_observation_spec(
            self, observation_spec: TensorSpec
        ) -> TensorSpec:
            observation_spec["nt"] = NonTensor(shape=())
            return observation_spec

    @classmethod
    def make_env(cls):
        return (
            GymEnv(CARTPOLE_VERSIONED())
            .append_transform(StepCounter())
            .append_transform(cls.AddNontTensorData())
        )

    def test_simple(self):
        torch.manual_seed(0)
        env = self.make_env()
        env.set_seed(0)
        collector = Collector(env, frames_per_batch=10, total_frames=200)
        result = []
        for data in collector:
            result.append(data)
        result = torch.cat(result)
        for i, val in enumerate(result["nt"][1:]):
            if val == "a string! - 1":
                assert result["nt"][i] == "reset!"
            elif val.startswith("a string!"):
                assert result["next", "nt"][i] == val
                int1 = int(val.split(" - ")[-1])
                int0 = int(result["nt"][i].split(" - ")[-1])
                assert int0 + 1 == int1
            elif val == "reset!":
                assert result["next", "nt"][i + 1] == "a string! - 1", i

    @pytest.mark.parametrize("use_buffers", [True, False])
    def test_sync(self, use_buffers):
        torch.manual_seed(0)
        collector = MultiSyncCollector(
            [self.make_env, self.make_env],
            frames_per_batch=10,
            total_frames=200,
            cat_results="stack",
            use_buffers=use_buffers,
        )
        try:
            result = []
            for data in collector:
                result.append(data)
            results = torch.cat(result)
            for result in results.unbind(0):
                for i, val in enumerate(result["nt"][1:]):
                    if val == "a string! - 1":
                        assert result["nt"][i] == "reset!"
                    elif val.startswith("a string!"):
                        assert result["next", "nt"][i] == val
                        int1 = int(val.split(" - ")[-1])
                        int0 = int(result["nt"][i].split(" - ")[-1])
                        assert int0 + 1 == int1
                    elif val == "reset!":
                        assert result["next", "nt"][i + 1] == "a string! - 1", i
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize("use_buffers", [True, False])
    def test_async(self, use_buffers):
        torch.manual_seed(0)
        collector = MultiAsyncCollector(
            [self.make_env, self.make_env],
            frames_per_batch=10,
            total_frames=200,
            use_buffers=use_buffers,
        )
        try:
            results = []
            for data in collector:
                results.append(data)
            for result in results:
                for i, val in enumerate(result["nt"][1:]):
                    if val == "a string! - 1":
                        assert result["nt"][i] == "reset!"
                    elif val.startswith("a string!"):
                        assert result["next", "nt"][i] == val
                        int1 = int(val.split(" - ")[-1])
                        int0 = int(result["nt"][i].split(" - ")[-1])
                        assert int0 + 1 == int1
                    elif val == "reset!":
                        assert result["next", "nt"][i + 1] == "a string! - 1", i
        finally:
            collector.shutdown()
            del collector


class TestCollectorRB:
    @pytest.mark.skipif(not _has_gym, reason="requires gym.")
    def test_collector_rb_sync(self):
        env = SerialEnv(8, lambda cp=CARTPOLE_VERSIONED(): GymEnv(cp))
        env.set_seed(0)
        rb = ReplayBuffer(storage=LazyTensorStorage(256, ndim=2), batch_size=5)
        collector = Collector(
            env,
            RandomPolicy(env.action_spec),
            replay_buffer=rb,
            total_frames=256,
            frames_per_batch=16,
        )
        torch.manual_seed(0)

        for c in collector:
            assert c is None
            rb.sample()
        rbdata0 = rb[:].clone()
        collector.shutdown()
        if not env.is_closed:
            env.close()
        del collector, env

        env = SerialEnv(8, lambda cp=CARTPOLE_VERSIONED(): GymEnv(cp))
        env.set_seed(0)
        rb = ReplayBuffer(storage=LazyTensorStorage(256, ndim=2), batch_size=5)
        collector = Collector(
            env, RandomPolicy(env.action_spec), total_frames=256, frames_per_batch=16
        )
        torch.manual_seed(0)

        for i, c in enumerate(collector):
            rb.extend(c)
            torch.testing.assert_close(
                rbdata0[:, : (i + 1) * 2]["observation"], rb[:]["observation"]
            )
            assert c is not None
            rb.sample()

        rbdata1 = rb[:].clone()
        collector.shutdown()
        if not env.is_closed:
            env.close()
        del collector, env
        assert assert_allclose_td(rbdata0, rbdata1)

    @pytest.mark.skipif(not _has_gym, reason="requires gym.")
    @pytest.mark.parametrize("extend_buffer", [False, True])
    @pytest.mark.parametrize("env_creator", [False, True])
    @pytest.mark.parametrize("storagetype", [LazyTensorStorage, LazyMemmapStorage])
    def test_collector_rb_multisync(
        self, extend_buffer, env_creator, storagetype, tmpdir
    ):
        if not env_creator:
            env = GymEnv(CARTPOLE_VERSIONED()).append_transform(StepCounter())
            env.set_seed(0)
            action_spec = env.action_spec
            env = lambda env=env: env
        else:
            env = EnvCreator(
                lambda cp=CARTPOLE_VERSIONED(): GymEnv(cp).append_transform(
                    StepCounter()
                )
            )
            action_spec = env.meta_data.specs["input_spec", "full_action_spec"]

        if storagetype == LazyMemmapStorage:
            storagetype = functools.partial(LazyMemmapStorage, scratch_dir=tmpdir)
        rb = ReplayBuffer(storage=storagetype(256), batch_size=5)

        collector = MultiSyncCollector(
            [env, env],
            RandomPolicy(action_spec),
            replay_buffer=rb,
            total_frames=256,
            frames_per_batch=32,
            extend_buffer=extend_buffer,
        )
        torch.manual_seed(0)
        pred_len = 0
        for c in collector:
            pred_len += 32
            assert c is None
            assert len(rb) == pred_len
        collector.shutdown()
        assert len(rb) == 256
        if extend_buffer:
            steps_counts = rb["step_count"].squeeze().split(16)
            collector_ids = rb["collector", "traj_ids"].squeeze().split(16)
            for step_count, ids in zip(steps_counts, collector_ids):
                step_countdiff = step_count.diff()
                idsdiff = ids.diff()
                assert (
                    (step_countdiff == 1) | (step_countdiff < 0)
                ).all(), steps_counts
                assert (idsdiff >= 0).all()

    @pytest.mark.skipif(not _has_gym, reason="requires gym.")
    @pytest.mark.parametrize("extend_buffer", [False, True])
    @pytest.mark.parametrize("env_creator", [False, True])
    @pytest.mark.parametrize("storagetype", [LazyTensorStorage, LazyMemmapStorage])
    def test_collector_rb_multiasync(
        self, extend_buffer, env_creator, storagetype, tmpdir
    ):
        if not env_creator:
            env = GymEnv(CARTPOLE_VERSIONED()).append_transform(StepCounter())
            env.set_seed(0)
            action_spec = env.action_spec
            env = lambda env=env: env
        else:
            env = EnvCreator(
                lambda cp=CARTPOLE_VERSIONED(): GymEnv(cp).append_transform(
                    StepCounter()
                )
            )
            action_spec = env.meta_data.specs["input_spec", "full_action_spec"]

        if storagetype == LazyMemmapStorage:
            storagetype = functools.partial(LazyMemmapStorage, scratch_dir=tmpdir)
        rb = ReplayBuffer(storage=storagetype(256), batch_size=5)

        collector = MultiAsyncCollector(
            [env, env],
            RandomPolicy(action_spec),
            replay_buffer=rb,
            total_frames=256,
            frames_per_batch=16,
            extend_buffer=extend_buffer,
        )
        torch.manual_seed(0)
        pred_len = 0
        for c in collector:
            pred_len += 16
            assert c is None
            assert len(rb) >= pred_len
        collector.shutdown()
        assert len(rb) == 256
        if extend_buffer:
            steps_counts = rb["step_count"].squeeze().split(16)
            collector_ids = rb["collector", "traj_ids"].squeeze().split(16)
            for step_count, ids in zip(steps_counts, collector_ids):
                step_countdiff = step_count.diff()
                idsdiff = ids.diff()
                assert (
                    (step_countdiff == 1) | (step_countdiff < 0)
                ).all(), steps_counts
                assert (idsdiff >= 0).all()

    @pytest.mark.skipif(not _has_gym, reason="requires gym.")
    @pytest.mark.parametrize(
        "collector_class", [MultiSyncCollector, MultiAsyncCollector]
    )
    @pytest.mark.parametrize("extend_buffer", [True, False])
    def test_parallel_env_with_multi_collector_and_replay_buffer(
        self, collector_class, extend_buffer
    ):
        """Test that ParallelEnv works with multi-collectors when replay_buffer is given.

        Regression test for issue #3240 / PR #3341.
        The bug was that `_main_async_collector` hardcoded `extend_buffer=False`
        instead of forwarding the user's setting, causing dimension mismatches
        when using ParallelEnv with multi-collectors and replay buffers.
        """

        # Create a ParallelEnv factory - this is the key component that was failing
        def make_parallel_env():
            return ParallelEnv(
                num_workers=2,
                create_env_fn=lambda cp=CARTPOLE_VERSIONED(): GymEnv(
                    cp
                ).append_transform(StepCounter()),
            )

        # Get action spec from a temporary env
        temp_env = make_parallel_env()
        action_spec = temp_env.action_spec
        temp_env.close(raise_if_closed=False)
        del temp_env

        # Create replay buffer with ndim=2 to handle the batch dimension from ParallelEnv
        rb = ReplayBuffer(storage=LazyTensorStorage(512, ndim=2), batch_size=5)

        # Create the multi-collector with ParallelEnv and replay_buffer
        # This combination was failing before the fix
        collector = collector_class(
            [
                make_parallel_env,
                make_parallel_env,
            ],  # 2 workers, each with ParallelEnv(2)
            RandomPolicy(action_spec),
            replay_buffer=rb,
            total_frames=256,
            frames_per_batch=32,
            extend_buffer=extend_buffer,
        )

        try:
            # Collect data - this should not raise dimension mismatch errors
            for c in collector:
                # When replay_buffer is used, iterator yields None
                assert c is None

            # Verify buffer was populated correctly
            assert len(rb) >= 256, f"Expected at least 256 frames, got {len(rb)}"

            # If extend_buffer=True, verify trajectory structure is preserved
            if extend_buffer:
                # Each batch should have consecutive step counts (with resets)
                steps_counts = rb["step_count"].squeeze()
                # Just verify we have valid step counts (StepCounter starts at 0)
                assert steps_counts.min() >= 0
                assert steps_counts.numel() >= 256

        finally:
            collector.shutdown()
            del collector

    @staticmethod
    def _zero_postproc(td):
        # Apply zero to all tensor values in the tensordict
        return torch.zeros_like(td)

    @pytest.mark.parametrize(
        "collector_class",
        [
            Collector,
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize("use_replay_buffer", [True, False])
    @pytest.mark.parametrize("extend_buffer", [True, False])
    def test_collector_postproc_zeros(
        self, collector_class, use_replay_buffer, extend_buffer
    ):
        """Test that postproc functionality works correctly across all collector types.

        This test verifies that:
        1. Postproc is applied correctly when no replay buffer is used
        2. Postproc is applied correctly when replay buffer is used with extend_buffer=True
        3. Postproc is not applied when replay buffer is used with extend_buffer=False
        4. The behavior is consistent across Sync, MultiaSync, and MultiSync collectors
        """
        # Skip multi-collectors with replay buffer on older Python.
        # There's a known shared memory visibility race condition with Python < 3.10 and the
        # "spawn" multiprocessing start method. The child process writes to shared memory,
        # but the main process may sample before the writes are fully visible.
        is_multi_collector = collector_class != Collector
        if is_multi_collector and use_replay_buffer and sys.version_info < (3, 10):
            pytest.skip(
                "Multi-collectors with replay buffer are not supported on Python < 3.10 "
                "due to shared memory visibility issues with the 'spawn' start method."
            )

        # Create a simple dummy environment
        def make_env():
            env = DiscreteActionVecMockEnv()
            env.set_seed(0)
            return env

        # Create a simple dummy policy
        def make_policy(env):
            return RandomPolicy(env.action_spec)

        # Test parameters
        total_frames = 64
        frames_per_batch = 16

        if use_replay_buffer:
            # Create replay buffer
            rb = ReplayBuffer(
                storage=LazyTensorStorage(256), batch_size=5, compilable=False
            )

            # Test with replay buffer
            if collector_class == Collector:
                collector = collector_class(
                    make_env(),
                    make_policy(make_env()),
                    replay_buffer=rb,
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    extend_buffer=extend_buffer,
                    postproc=self._zero_postproc if extend_buffer else None,
                )
            else:
                # MultiSync and MultiaSync collectors
                collector = collector_class(
                    [make_env, make_env],
                    make_policy(make_env()),
                    replay_buffer=rb,
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    extend_buffer=extend_buffer,
                    postproc=self._zero_postproc if extend_buffer else None,
                )
            try:
                # Collect data
                collected_frames = 0
                for _ in collector:
                    collected_frames += frames_per_batch
                    if extend_buffer:
                        # With extend_buffer=True, postproc should be applied
                        # Check that the replay buffer contains zeros
                        sample = rb.sample(5)
                        torch.testing.assert_close(
                            sample["observation"],
                            torch.zeros_like(sample["observation"]),
                        )
                        torch.testing.assert_close(
                            sample["action"], torch.zeros_like(sample["action"])
                        )
                        # Check next.reward instead of reward
                        torch.testing.assert_close(
                            sample["next", "reward"],
                            torch.zeros_like(sample["next", "reward"]),
                        )
                    else:
                        # With extend_buffer=False, postproc should not be applied
                        # Check that the replay buffer contains non-zero values
                        sample = rb.sample(5)
                        assert torch.any(sample["observation"] != 0.0)
                        assert torch.any(sample["action"] != 0.0)

                    if collected_frames >= total_frames:
                        break
            finally:
                collector.shutdown()

        else:
            # Test without replay buffer
            if collector_class == Collector:
                collector = collector_class(
                    make_env(),
                    make_policy(make_env()),
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    postproc=self._zero_postproc,
                )
            else:
                # MultiSync and MultiaSync collectors
                collector = collector_class(
                    [make_env, make_env],
                    make_policy(make_env()),
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    postproc=self._zero_postproc,
                )
            try:
                # Collect data and verify postproc is applied
                for batch in collector:
                    # All values should be zero due to postproc
                    assert torch.all(batch["observation"] == 0.0)
                    assert torch.all(batch["action"] == 0.0)
                    # Check next.reward instead of reward
                    assert torch.all(batch["next", "reward"] == 0.0)
                    break  # Just check first batch
            finally:
                collector.shutdown()


def __deepcopy_error__(*args, **kwargs):
    raise RuntimeError("deepcopy not allowed")


class TestPolicyFactory:
    class MPSWeightUpdaterBase(WeightUpdaterBase):
        def __init__(self, policy_weights, num_workers):
            # Weights are on mps device, which cannot be shared
            self.policy_weights = policy_weights.data
            self.num_workers = num_workers

        def _sync_weights_with_worker(
            self, worker_id: int | torch.device, server_weights: TensorDictBase
        ) -> TensorDictBase:
            # Send weights on cpu - the local workers will do the cpu->mps copy
            self.collector.pipes[worker_id].send((server_weights, "update"))
            val, msg = self.collector.pipes[worker_id].recv()
            assert msg == "updated"
            return server_weights

        def _get_server_weights(self) -> TensorDictBase:
            return self.policy_weights.cpu()

        def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
            return server_weights

        def all_worker_ids(self) -> list[int] | list[torch.device]:
            return list(range(self.num_workers))

    @pytest.mark.skipif(not _has_gym, reason="requires gym")
    @pytest.mark.parametrize(
        "weight_updater", ["scheme_shared", "scheme_mp", "weight_updater"]
    )
    def test_update_weights(self, weight_updater):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env_maker = lambda: GymEnv(PENDULUM_VERSIONED(), device="cpu")
        policy_factory = lambda: TensorDictModule(
            nn.Linear(3, 1, device=device), in_keys=["observation"], out_keys=["action"]
        )
        policy = policy_factory()
        policy_weights = TensorDict.from_module(policy)
        kwargs = {}
        if weight_updater == "scheme_shared":
            scheme = SharedMemWeightSyncScheme()
            kwargs = {"weight_sync_schemes": {"policy": scheme}}
        elif weight_updater == "scheme_mp":
            scheme = MultiProcessWeightSyncScheme()
            kwargs = {"weight_sync_schemes": {"policy": scheme}}
        elif weight_updater == "weight_updater":
            scheme = None
            kwargs = {"weight_updater": self.MPSWeightUpdaterBase(policy_weights, 2)}
        else:
            raise NotImplementedError

        if scheme is not None:
            scheme.init_on_sender(
                model=policy_factory(), devices=[device] * 2, model_id="policy"
            )

        collector = MultiSyncCollector(
            create_env_fn=[env_maker, env_maker],
            policy_factory=policy_factory,
            total_frames=2000,
            max_frames_per_traj=50,
            frames_per_batch=200,
            init_random_frames=-1,
            reset_at_each_iter=False,
            device=device,
            storing_device="cpu",
            **kwargs,
        )
        try:
            if weight_updater == "weight_updater":
                assert collector._legacy_weight_updater

            # When using policy_factory, must pass weights explicitly
            collector.update_policy_weights_(policy_weights)

            for i, data in enumerate(collector):  # noqa: B007
                if i == 2:
                    assert (data["action"] != 0).any()
                    # zero the policy
                    policy_weights.data.zero_()
                    # When using policy_factory, must pass weights explicitly
                    collector.update_policy_weights_(policy_weights)
                elif i == 3:
                    assert (data["action"] == 0).all(), data["action"]
                    break
        finally:
            collector.shutdown()

    @pytest.mark.parametrize(
        "collector_cls",
        [
            functools.partial(MultiSyncCollector, cat_results="stack"),
            MultiAsyncCollector,
        ],
    )
    @pytest.mark.parametrize(
        "weight_sync_scheme_cls",
        [MultiProcessWeightSyncScheme, SharedMemWeightSyncScheme],
    )
    def test_nonserializable_policy_with_factory_and_weight_sync(
        self, collector_cls, weight_sync_scheme_cls
    ):
        """Test that a non-serializable policy can be used on the main node alongside a policy_factory.

        The policy instance is used only for weight extraction on the main node, while
        the policy_factory is what gets sent to and instantiated on workers.
        """

        # Simple continuous-control env
        def create_env():
            return ContinuousActionVecMockEnv()

        # Non-serializable policy instance on main node
        base_module = NonSerializableBiasModule(0.0)
        policy = TensorDictModule(
            base_module, in_keys=["observation"], out_keys=["action"]
        )

        # Serializable factory used to build worker policies
        def policy_factory():
            return TensorDictModule(
                BiasModule(0.0), in_keys=["observation"], out_keys=["action"]
            )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Weight sync scheme will be initialized on the sender side by the collector,
        # using the policy instance passed above as the source of weights.
        weight_sync_scheme = weight_sync_scheme_cls()

        collector = collector_cls(
            [create_env, create_env],
            policy=policy,
            policy_factory=policy_factory,
            frames_per_batch=16,
            total_frames=64,
            device=device,
            storing_device="cpu",
            weight_sync_schemes={"policy": weight_sync_scheme},
        )

        try:
            # Ensure we can collect at least one batch without serialization issues
            iterator = iter(collector)
            _ = next(iterator)

            # Change the main-node policy weights and update workers without passing weights explicitly
            with torch.no_grad():
                base_module.bias.add_(1.0)

            # This call should:
            # - Use the (non-serializable) policy to extract weights via TensorDict.from_module()
            # - Send those weights through the weight sync scheme
            # - NOT attempt to serialize the policy itself
            collector.update_policy_weights_()

            # Collect again to exercise the updated weights path and ensure workers didn't crash
            _ = next(iterator)
        finally:
            collector.shutdown()


class TestAsyncCollection:
    @pytest.mark.parametrize("total_frames", [-1, 1_000_000_000])
    def test_start_single(self, total_frames):
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))
        env = CountingEnv()
        policy = RandomPolicy(action_spec=env.action_spec)
        collector = Collector(
            env,
            policy,
            replay_buffer=rb,
            total_frames=total_frames,
            frames_per_batch=16,
        )
        try:
            collector.start()
            for _ in range(10):
                time.sleep(0.1)
                if len(rb) >= 16:
                    break
            else:
                raise RuntimeError("RB is empty")
            assert len(rb) >= 16
        finally:
            collector.async_shutdown(timeout=10)
            del collector

    def test_pause(self):
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))
        env = CountingEnv()
        policy = RandomPolicy(action_spec=env.action_spec)
        collector = AsyncCollector(
            CountingEnv,
            policy,
            replay_buffer=rb,
            total_frames=-1,
            frames_per_batch=16,
        )
        try:
            num_pauses = 0
            collector.start()
            for _ in range(10):
                time.sleep(0.1)
                if len(rb) >= 16:
                    with collector.pause():
                        num_pauses += 1
                        n = rb.write_count
                        for _ in range(10):
                            assert rb.write_count == n
                            time.sleep(0.1)
                    time.sleep(1)
                    assert rb.write_count > n
                    if num_pauses == 2:
                        break
            else:
                raise RuntimeError("RB is empty")
            assert len(rb) >= 16
        finally:
            collector.async_shutdown(timeout=10)
            del collector

    @pytest.mark.parametrize("total_frames", [-1, 1_000_000_000])
    @pytest.mark.parametrize("cls", [MultiAsyncCollector, MultiSyncCollector])
    def test_start_multi(self, total_frames, cls):
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))
        policy = RandomPolicy(action_spec=CountingEnv().action_spec)
        collector = cls(
            [CountingEnv, CountingEnv],
            policy,
            replay_buffer=rb,
            total_frames=total_frames,
            frames_per_batch=16,
        )
        try:
            collector.start()
            for _ in range(10):
                time.sleep(0.1)  # Use asyncio.sleep instead of time.sleep
                if len(rb) >= 16:
                    break
            else:
                raise RuntimeError("RB is empty")
        finally:
            collector.async_shutdown()
            del collector

    @pytest.mark.parametrize("total_frames", [-1, 1_000_000_000])
    @pytest.mark.parametrize(
        "cls", [Collector, MultiAsyncCollector, MultiSyncCollector]
    )
    @pytest.mark.parametrize(
        "weight_sync_scheme",
        [None, MultiProcessWeightSyncScheme, SharedMemWeightSyncScheme],
    )
    @pytest.mark.flaky(reruns=3, reruns_delay=0.5)
    def test_start_update_policy(self, total_frames, cls, weight_sync_scheme):
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))
        env = CountingEnv()
        m = nn.Linear(env.observation_spec["observation"].shape[-1], 1)
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)
        policy = TensorDictSequential(
            TensorDictModule(
                lambda x: x.float(), in_keys=["observation"], out_keys=["action"]
            ),
            TensorDictModule(m, in_keys=["action"], out_keys=["action"]),
            TensorDictModule(
                lambda x: x.to(torch.int8), in_keys=["action"], out_keys=["action"]
            ),
        )
        td = TensorDict.from_module(policy).data.clone()
        if cls != Collector:
            env = [CountingEnv] * 2

        # Add weight sync schemes for multi-process collectors
        kwargs = {}
        if cls != Collector and weight_sync_scheme is not None:
            kwargs["weight_sync_schemes"] = {"policy": weight_sync_scheme()}

        collector = cls(
            env,
            policy,
            replay_buffer=rb,
            total_frames=total_frames,
            frames_per_batch=16,
            **kwargs,
        )
        try:
            if not isinstance(collector, Collector):
                if weight_sync_scheme is not None:
                    assert isinstance(
                        collector._weight_sync_schemes["policy"], weight_sync_scheme
                    )
                else:
                    assert isinstance(
                        collector._weight_sync_schemes["policy"],
                        SharedMemWeightSyncScheme,
                    )
            collector.start()
            for _ in range(10):
                time.sleep(0.1)
                if len(rb) >= 16:
                    break
            else:
                raise RuntimeError("RB is empty")
            assert (rb[-16:]["action"] == 0).all()
            td["module", "1", "module", "bias"] += 1
            collector.update_policy_weights_(td)
            for _ in range(10):
                time.sleep(0.1)
                if (rb[-16:]["action"] == 1).all():
                    break
            else:
                raise RuntimeError("Failed to update policy weights")
        finally:
            collector.async_shutdown(timeout=10)
            del collector


class TestInitRandomFramesWithStart:
    """Tests for init_random_frames with .start() method for collectors."""

    @pytest.mark.skipif(not _has_gym, reason="requires gym.")
    @pytest.mark.parametrize("cls", [MultiSyncCollector, MultiAsyncCollector])
    @pytest.mark.flaky(reruns=3, reruns_delay=0.5)
    def test_init_random_frames_with_start(self, cls):
        """Test that init_random_frames works with .start() for multi-process collectors.

        This test verifies that:
        1. Collection starts without error when init_random_frames is provided
        2. Data collection proceeds beyond init_random_frames
        3. The replay buffer is properly populated
        """
        init_random_frames = 64
        frames_per_batch = 16
        total_to_collect = 256

        # Create env to get action spec for policy
        env = GymEnv(CARTPOLE_VERSIONED())
        policy = RandomPolicy(env.action_spec)
        env.close()

        rb = ReplayBuffer(storage=LazyTensorStorage(total_to_collect), batch_size=5)

        env_fns = [
            lambda: GymEnv(CARTPOLE_VERSIONED()),
            lambda: GymEnv(CARTPOLE_VERSIONED()),
        ]
        collector = cls(
            env_fns,
            policy,
            replay_buffer=rb,
            total_frames=-1,
            frames_per_batch=frames_per_batch,
            init_random_frames=init_random_frames,
        )

        try:
            # Start the collector - this should NOT raise an error even with init_random_frames
            collector.start()

            # Wait for enough data to be collected - should go beyond init_random_frames
            for _ in range(100):
                time.sleep(0.1)
                if rb.write_count >= total_to_collect:
                    break
            else:
                raise RuntimeError(
                    f"Not enough data collected: {rb.write_count} < {total_to_collect}. "
                    f"init_random_frames was {init_random_frames}."
                )

            # Verify that collection proceeded beyond init_random_frames
            assert (
                rb.write_count >= total_to_collect
            ), f"Expected at least {total_to_collect} frames, got {rb.write_count}"

            # Verify that data has expected structure
            sample = rb[:16]
            assert "observation" in sample.keys()
            assert "action" in sample.keys()
            assert "next" in sample.keys()

        finally:
            collector.async_shutdown(timeout=10)
            del collector


class TestCollectorProfiling:
    """Tests for the collector profiling feature."""

    def test_profile_config_validation(self):
        """Test ProfileConfig validation."""
        # Valid config
        config = ProfileConfig(
            workers=[0],
            num_rollouts=5,
            warmup_rollouts=2,
        )
        assert config.workers == [0]
        assert config.num_rollouts == 5
        assert config.warmup_rollouts == 2

        # Invalid: num_rollouts <= warmup_rollouts
        with pytest.raises(ValueError, match="num_rollouts.*must be greater"):
            ProfileConfig(num_rollouts=2, warmup_rollouts=2)

        with pytest.raises(ValueError, match="num_rollouts.*must be greater"):
            ProfileConfig(num_rollouts=2, warmup_rollouts=3)

        # Invalid: negative warmup
        with pytest.raises(ValueError, match="warmup_rollouts must be >= 0"):
            ProfileConfig(num_rollouts=5, warmup_rollouts=-1)

    def test_profile_config_get_save_path(self):
        """Test ProfileConfig.get_save_path method."""
        from pathlib import Path

        # Default path
        config = ProfileConfig(save_path=None)
        path = config.get_save_path(worker_idx=0)
        assert path == Path("./collector_profile_0.json")

        # Custom path with placeholder
        config = ProfileConfig(save_path="./traces/worker_{worker_idx}/trace.json")
        path = config.get_save_path(worker_idx=2)
        assert path == Path("./traces/worker_2/trace.json")

    def test_profile_config_should_profile_worker(self):
        """Test ProfileConfig.should_profile_worker method."""
        config = ProfileConfig(workers=[0, 2])
        assert config.should_profile_worker(0) is True
        assert config.should_profile_worker(1) is False
        assert config.should_profile_worker(2) is True
        assert config.should_profile_worker(3) is False

    @pytest.mark.parametrize(
        "use_gpu",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_profile_config_get_activities(self, use_gpu):
        """Test ProfileConfig.get_activities method."""
        if use_gpu:
            config = ProfileConfig(activities=["cpu", "cuda"])
            activities = config.get_activities()
            assert torch.profiler.ProfilerActivity.CPU in activities
            assert torch.profiler.ProfilerActivity.CUDA in activities
        else:
            config = ProfileConfig(activities=["cpu"])
            activities = config.get_activities()
            assert torch.profiler.ProfilerActivity.CPU in activities

    def test_enable_profile_single_collector(self, tmp_path):
        """Test enable_profile on a single-process Collector."""
        if not _has_gym:
            pytest.skip("Gym not available")

        trace_path = tmp_path / "trace_{worker_idx}.json"

        env = GymEnv(PENDULUM_VERSIONED())
        policy = RandomPolicy(env.action_spec)

        collector = Collector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=50,
            total_frames=300,
        )

        # Enable profiling
        collector.enable_profile(
            workers=[0],  # Ignored for single-process
            num_rollouts=3,
            warmup_rollouts=1,
            save_path=str(trace_path),
        )

        assert collector.profile_config is not None
        assert collector.profile_config.num_rollouts == 3
        assert collector.profile_config.warmup_rollouts == 1

        # Run collection
        data_count = 0
        for _data in collector:
            data_count += 1
            if data_count >= 5:
                break

        collector.shutdown()

        # Check that the trace file was created
        expected_trace = tmp_path / "trace_0.json"
        assert expected_trace.exists(), f"Trace file not found at {expected_trace}"

    def test_enable_profile_cannot_call_after_iteration(self):
        """Test that enable_profile raises error after iteration starts."""
        if not _has_gym:
            pytest.skip("Gym not available")

        env = GymEnv(PENDULUM_VERSIONED())
        policy = RandomPolicy(env.action_spec)

        collector = Collector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=50,
            total_frames=200,
        )

        # Start iteration
        it = iter(collector)
        next(it)

        # Now enable_profile should fail
        with pytest.raises(
            RuntimeError, match="Cannot enable profiling after iteration"
        ):
            collector.enable_profile(num_rollouts=3, warmup_rollouts=1)

        collector.shutdown()

    @pytest.mark.slow
    def test_enable_profile_multi_sync_collector(self, tmp_path):
        """Test enable_profile on MultiSyncCollector."""
        if not _has_gym:
            pytest.skip("Gym not available")

        trace_path = tmp_path / "trace_{worker_idx}.json"

        def env_fn():
            return GymEnv(PENDULUM_VERSIONED())

        policy = RandomPolicy(GymEnv(PENDULUM_VERSIONED()).action_spec)

        collector = MultiSyncCollector(
            create_env_fn=[env_fn, env_fn],
            policy=policy,
            frames_per_batch=50,
            total_frames=300,
        )

        # Enable profiling - only profile worker 0
        collector.enable_profile(
            workers=[0],
            num_rollouts=3,
            warmup_rollouts=1,
            save_path=str(trace_path),
        )

        assert collector.profile_config is not None

        # Run collection
        data_count = 0
        for _data in collector:
            data_count += 1
            if data_count >= 5:
                break

        collector.shutdown()

        # Check that the trace file was created for worker 0
        expected_trace = tmp_path / "trace_0.json"
        assert expected_trace.exists(), f"Trace file not found at {expected_trace}"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main(
        [__file__, "--capture", "no", "--exitfirst", "--timeout", "180"] + unknown
    )
