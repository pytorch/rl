# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import gc
import importlib
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging import version
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
    aSyncDataCollector,
    SyncDataCollector,
    WeightUpdateSenderBase,
)
from torchrl.collectors.collectors import (
    _Interruptor,
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
)

from torchrl.collectors.llm import LLMCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    LazyStackStorage,
    LazyTensorStorage,
    NonTensor,
    ReplayBuffer,
    TensorSpec,
    Unbounded,
)
from torchrl.data.llm.dataset import _has_transformers
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import (
    AsyncEnvPool,
    EnvBase,
    EnvCreator,
    InitTracker,
    LLMEnv,
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
    RandomPolicy,
)
from torchrl.modules import (
    Actor,
    OrnsteinUhlenbeckProcessModule,
    SafeModule,
    TransformersWrapper,
    vLLMWrapper,
)

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.rl.test._utils_internal import (
        CARTPOLE_VERSIONED,
        check_rollout_consistency_multikey_env,
        decorate_thread_sub_func,
        generate_seeds,
        get_available_devices,
        get_default_devices,
        LSTMNet,
        PENDULUM_VERSIONED,
        PONG_VERSIONED,
        retry,
    )
    from pytorch.rl.test.mocking_classes import (
        ContinuousActionVecMockEnv,
        CountingBatchedEnv,
        CountingEnv,
        CountingEnvCountPolicy,
        DiscreteActionConvMockEnv,
        DiscreteActionConvPolicy,
        DiscreteActionVecMockEnv,
        DiscreteActionVecPolicy,
        DummyStrDataLoader,
        EnvThatErrorsAfter10Iters,
        EnvWithDynamicSpec,
        HeterogeneousCountingEnv,
        HeterogeneousCountingEnvPolicy,
        MockSerialEnv,
        MultiKeyCountingEnv,
        MultiKeyCountingEnvPolicy,
        NestedCountingEnv,
    )
else:
    IS_FB = False
    from _utils_internal import (
        CARTPOLE_VERSIONED,
        check_rollout_consistency_multikey_env,
        decorate_thread_sub_func,
        generate_seeds,
        get_available_devices,
        get_default_devices,
        LSTMNet,
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
        DummyStrDataLoader,
        EnvThatErrorsAfter10Iters,
        EnvWithDynamicSpec,
        HeterogeneousCountingEnv,
        HeterogeneousCountingEnvPolicy,
        MockSerialEnv,
        MultiKeyCountingEnv,
        MultiKeyCountingEnvPolicy,
        NestedCountingEnv,
    )

# torch.set_default_dtype(torch.double)
IS_WINDOWS = sys.platform == "win32"
IS_OSX = sys.platform == "darwin"
PYTHON_3_10 = sys.version_info.major == 3 and sys.version_info.minor == 10
PYTHON_3_7 = sys.version_info.major == 3 and sys.version_info.minor == 7
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
_has_cuda = torch.cuda.is_available()
_has_vllm = importlib.util.find_spec("vllm") is not None


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
                    GymEnv(PONG_VERSIONED(), frame_skip=4), StepCounter()
                )

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
        c_inplace = SyncDataCollector(
            env, policy_inplace, frames_per_batch=10, total_frames=100
        )
        d_inplace = torch.cat(list(c_inplace), dim=0)
        env.reset()
        c_outplace = SyncDataCollector(
            env, policy_outplace, frames_per_batch=10, total_frames=100
        )
        d_outplace = torch.cat(list(c_outplace), dim=0)
        assert_allclose_td(d_inplace, d_outplace)

    @pytest.mark.skipif(not _has_gym, reason="test designed with GymEnv")
    @pytest.mark.parametrize(
        "collector_class",
        [
            SyncDataCollector,
            MultiaSyncDataCollector,
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
        ],
    )
    @pytest.mark.parametrize("init_random_frames", [0, 50])  # 1226: faster execution
    @pytest.mark.parametrize(
        "explicit_spec,split_trajs", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_collector_output_keys(
        self, collector_class, init_random_frames, explicit_spec, split_trajs
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
            hidden_spec = Unbounded((1, hidden_size))
            policy_kwargs["spec"] = Composite(
                action=Unbounded(),
                hidden1=hidden_spec,
                hidden2=hidden_spec,
                next=Composite(hidden1=hidden_spec, hidden2=hidden_spec),
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

    @pytest.mark.parametrize(
        "collector_class",
        [
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
            MultiaSyncDataCollector,
            SyncDataCollector,
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

    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason="Nested spawned multiprocessed is currently failing in python 3.11. "
        "See https://github.com/python/cpython/pull/108568 for info and fix.",
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

        collector = SyncDataCollector(
            create_env_fn=env_fn,
            create_env_kwargs={"seed": seed},
            policy=policy,
            frames_per_batch=20,
            max_frames_per_traj=2000,
            total_frames=20000,
            device="cpu",
        )
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

        try:
            assert ccollector._use_buffers
            assert d.names[-1] == "time"

            with pytest.raises(AssertionError):
                assert_allclose_td(b1c, b2c)

            assert_allclose_td(b1c, b1)
            assert_allclose_td(b2c, b2)
        finally:
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

    @pytest.mark.parametrize(
        "ctype", [SyncDataCollector, MultiaSyncDataCollector, MultiSyncDataCollector]
    )
    def test_env_that_errors(self, ctype):
        make_env = EnvThatErrorsAfter10Iters
        policy = RandomPolicy(make_env().action_spec)
        if ctype is SyncDataCollector:
            collector = SyncDataCollector(
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
    @pytest.mark.skipif(IS_FB, reason="Not compatible with fbcode")
    @pytest.mark.parametrize("to", [3, 10])
    @pytest.mark.parametrize(
        "collector_cls", ["MultiSyncDataCollector", "MultiaSyncDataCollector"]
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

    def _set_seed(self, seed: Optional[int]):
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
        # This errors if the timeout is 5 secs, not 15
        assert result.returncode == int(
            to == 3
        ), f"Test failed with output: {result.stdout}"

    @pytest.mark.parametrize(
        "collector_class",
        [
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
            MultiaSyncDataCollector,
            SyncDataCollector,
        ],
    )
    @pytest.mark.parametrize("exclude", [True, False])
    @pytest.mark.parametrize(
        "out_key", ["_dummy", ("out", "_dummy"), ("_out", "dummy")]
    )
    def test_excluded_keys(self, collector_class, exclude, out_key):
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
        collector = SyncDataCollector(
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
        _ = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            total_frames=10_000,
            frames_per_batch=1000,
        )
        with pytest.raises(ValueError):
            _ = SyncDataCollector(
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
            SyncDataCollector,
            MultiaSyncDataCollector,
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
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

        shared_device = torch.device("cpu")
        if torch.cuda.is_available():
            original_device = torch.device("cuda:0")
        elif torch.mps.is_available():
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
            if collector_type is not SyncDataCollector:
                envs = [envs, envs]
            c = collector_type(
                envs,
                policy=policy,
                total_frames=1000,
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

        if collector_type is SyncDataCollector or original_device.type != "mps":
            # mps cannot be shared
            policy = make_policy(device=original_device)
            make_and_test_policy(policy, env_device=original_device)

        if collector_type is SyncDataCollector or original_device.type != "mps":
            policy = make_policy(device=original_device)
            make_and_test_policy(
                policy, policy_device=original_device, env_device=original_device
            )

        # a deepcopy must occur when the policy_device differs from the actual device
        with pytest.raises(RuntimeError, match="deepcopy not allowed"):
            policy = make_policy(device=original_device)
            make_and_test_policy(
                policy, policy_device=shared_device, env_device=shared_device
            )

        # a deepcopy must occur when device differs from the actual device
        with pytest.raises(RuntimeError, match="deepcopy not allowed"):
            policy = make_policy(device=original_device)
            make_and_test_policy(policy, device=shared_device)

        # If the policy is not an nn.Module, we can't cast it to device, so we assume that the policy device
        # is there to inform us
        substitute_device = (
            original_device if torch.cuda.is_available() else torch.device("cpu")
        )
        policy = make_policy(substitute_device, nn_module=False)
        with pytest.warns(UserWarning):
            make_and_test_policy(
                policy, policy_device=substitute_device, env_device=substitute_device
            )
        # For instance, if the env is on CPU, knowing the policy device helps with casting stuff on the right device
        with pytest.warns(UserWarning):
            make_and_test_policy(
                policy, policy_device=substitute_device, env_device=shared_device
            )
        make_and_test_policy(
            policy,
            policy_device=substitute_device,
            env_device=shared_device,
            trust_policy=True,
        )

        # If there is no policy_device, we assume that the user is doing things right too but don't warn
        if collector_type is SyncDataCollector or original_device.type != "mps":
            policy = make_policy(original_device, nn_module=False)
            make_and_test_policy(policy, env_device=original_device)

        # If the policy is a CudaGraphModule, we know it's on cuda - no need to warn
        if torch.cuda.is_available() and collector_type is SyncDataCollector:
            policy = make_policy(original_device)
            cudagraph_policy = CudaGraphModule(policy)
            make_and_test_policy(
                cudagraph_policy,
                policy_device=original_device,
                env_device=shared_device,
            )

    @pytest.mark.parametrize(
        "ctype", [SyncDataCollector, MultiaSyncDataCollector, MultiSyncDataCollector]
    )
    def test_no_stopiteration(self, ctype):
        # Tests that there is no StopIteration raised and that the length of the collector is properly set
        if ctype is SyncDataCollector:
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

        collector = SyncDataCollector(
            env, policy=policy, frames_per_batch=10, total_frames=20
        )
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
                == torch.tensor([False, False, True], device=data_device).repeat(17)[
                    :50
                ]
            ).all(), data[1]["next", "truncated"][:10]
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.parametrize(
        "collector_cls",
        [SyncDataCollector, MultiSyncDataCollector, MultiaSyncDataCollector],
    )
    def test_set_truncated(self, collector_cls):
        env_fn = lambda: TransformedEnv(
            NestedCountingEnv(), InitTracker()
        ).add_truncated_keys()
        env = env_fn()
        policy = CloudpickleWrapper(env.rand_action)
        if collector_cls == SyncDataCollector:
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

    @pytest.mark.parametrize("num_env", [1, 2])
    @pytest.mark.parametrize(
        "collector_class",
        [
            SyncDataCollector,
        ],
    )  # aSyncDataCollector])
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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device found")
    def test_update_weights(self, use_async):
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

        def _set_seed(self, seed: int | None = None):
            return seed

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

        def _set_seed(self, seed: int | None):
            return seed

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
            collector = SyncDataCollector(
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
        assert collector._use_buffers
        batch = next(collector.iterator())
        assert batch.device == _make_ordinal_device(torch.device(storing_device))
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
        assert batch.device == _make_ordinal_device(torch.device(storing_device))
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
            match=("Arguments to policy.forward are incompatible with entries in"),
        ):
            collector_class(
                **self._create_collector_kwargs(env_maker, collector_class, policy)
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
        if os.getenv("PYTORCH_TEST_FBCODE"):
            from pytorch.rl.test.mocking_classes import (
                CountingEnvCountPolicy,
                NestedCountingEnv,
            )
        else:
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


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.mps.is_available(),
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
            td["action"] = (self.param + self.buf.to(self.param.device)).expand(
                td.shape
            )
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
            ["cpu", get_default_devices()[0]],
            [get_default_devices()[0], "cpu"],
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
            functools.partial(MultiSyncDataCollector, cat_results="stack"),
            MultiaSyncDataCollector,
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
    def test_param_sync_mixed_device(
        self, give_weights, collector, policy_device, env_device
    ):
        with torch.device("cpu"):
            policy = TestUpdateParams.Policy()
        policy.param = nn.Parameter(policy.param.data.to(policy_device))
        assert policy.buf.device == torch.device("cpu")

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
        IS_OSX or IS_WINDOWS,
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
        collector = SyncDataCollector(
            env, policy, frames_per_batch=20, total_frames=100
        )
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
        collector = MultiSyncDataCollector(
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
        collector = MultiaSyncDataCollector(
            [env],
            policy,
            frames_per_batch=20,
            total_frames=100,
            # use_buffers=False,
        )
        for data in collector:
            assert isinstance(data, LazyStackedTensorDict)
            assert data.names[-1] == "time"


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
)
@pytest.mark.skipif(IS_WINDOWS, reason="windows is not supported for compile tests.")
class TestCompile:
    @pytest.mark.parametrize(
        "collector_cls",
        # Clearing compiled policies causes segfault on machines with cuda
        [SyncDataCollector, MultiaSyncDataCollector, MultiSyncDataCollector]
        if not torch.cuda.is_available()
        else [SyncDataCollector],
    )
    @pytest.mark.parametrize("compile_policy", [True, {}, {"mode": "default"}])
    @pytest.mark.parametrize(
        "device", [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    )
    def test_compiled_policy(self, collector_cls, compile_policy, device):
        policy = TensorDictModule(
            nn.Linear(7, 7, device=device), in_keys=["observation"], out_keys=["action"]
        )
        make_env = functools.partial(ContinuousActionVecMockEnv, device=device)
        if collector_cls is SyncDataCollector:
            torch._dynamo.reset_code_caches()
            collector = SyncDataCollector(
                make_env(),
                policy,
                frames_per_batch=30,
                total_frames=120,
                compile_policy=compile_policy,
            )
            assert collector.compiled_policy
        else:
            collector = collector_cls(
                [make_env] * 2,
                policy,
                frames_per_batch=30,
                total_frames=120,
                compile_policy=compile_policy,
            )
            assert collector.compiled_policy
        try:
            for data in collector:
                assert data is not None
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.parametrize(
        "collector_cls",
        [SyncDataCollector],
    )
    @pytest.mark.parametrize("cudagraph_policy", [True, {}, {"warmup": 10}])
    def test_cudagraph_policy(self, collector_cls, cudagraph_policy):
        device = torch.device("cuda:0")
        policy = TensorDictModule(
            nn.Linear(7, 7, device=device), in_keys=["observation"], out_keys=["action"]
        )
        make_env = functools.partial(ContinuousActionVecMockEnv, device=device)
        if collector_cls is SyncDataCollector:
            collector = SyncDataCollector(
                make_env(),
                policy,
                frames_per_batch=30,
                total_frames=120,
                cudagraph_policy=cudagraph_policy,
                device=device,
            )
            assert collector.cudagraphed_policy
        else:
            collector = collector_cls(
                [make_env] * 2,
                policy,
                frames_per_batch=30,
                total_frames=120,
                cudagraph_policy=cudagraph_policy,
                device=device,
            )
            assert collector.cudagraphed_policy
        try:
            for data in collector:
                assert data is not None
        finally:
            collector.shutdown()
            del collector


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
        collector = SyncDataCollector(env, frames_per_batch=10, total_frames=200)
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
        collector = MultiSyncDataCollector(
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
        collector = MultiaSyncDataCollector(
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
        collector = SyncDataCollector(
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
        collector = SyncDataCollector(
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

        collector = MultiSyncDataCollector(
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
        if not extend_buffer:
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

        collector = MultiaSyncDataCollector(
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
        if not extend_buffer:
            steps_counts = rb["step_count"].squeeze().split(16)
            collector_ids = rb["collector", "traj_ids"].squeeze().split(16)
            for step_count, ids in zip(steps_counts, collector_ids):
                step_countdiff = step_count.diff()
                idsdiff = ids.diff()
                assert (
                    (step_countdiff == 1) | (step_countdiff < 0)
                ).all(), steps_counts
                assert (idsdiff >= 0).all()


def __deepcopy_error__(*args, **kwargs):
    raise RuntimeError("deepcopy not allowed")


class TestPolicyFactory:
    class MPSWeightUpdaterBase(WeightUpdateSenderBase):
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

    @pytest.mark.skipif(not _has_cuda, reason="requires cuda another device than CPU.")
    def test_weight_update(self):
        device = "cuda:0"
        env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        policy_factory = lambda: TensorDictModule(
            nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
        ).to(device)
        policy = policy_factory()
        policy_weights = TensorDict.from_module(policy)

        collector = MultiSyncDataCollector(
            create_env_fn=[env_maker, env_maker],
            policy_factory=policy_factory,
            total_frames=2000,
            max_frames_per_traj=50,
            frames_per_batch=200,
            init_random_frames=-1,
            reset_at_each_iter=False,
            device=device,
            storing_device="cpu",
            weight_update_sender=self.MPSWeightUpdaterBase(policy_weights, 2),
        )

        collector.update_policy_weights_()
        try:
            for i, data in enumerate(collector):
                if i == 2:
                    assert (data["action"] != 0).any()
                    # zero the policy
                    policy_weights.data.zero_()
                    collector.update_policy_weights_()
                elif i == 3:
                    assert (data["action"] == 0).all(), data["action"]
                    break
        finally:
            collector.shutdown()


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
class TestLLMCollector:
    @pytest.fixture(scope="module")
    def vllm_instance(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("gpt2")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def vllm_instance_opt(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("facebook/opt-125m")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def transformers_instance(self):
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel(GPT2Config()).eval()
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTForCausalLM(OPTConfig())

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [True, False])
    @pytest.mark.parametrize("total_steps", [1, 10, 20])
    def test_llm_collector_with_vllm(self, rb, total_steps, vllm_instance):
        # NOTE: if VLLM fails with CUDA multiprocessing, try setting
        # `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
        policy = vLLMWrapper(vllm_instance)
        tokenizer = vllm_instance.get_tokenizer()
        self._run_collector_test(total_steps, rb, policy, tokenizer)

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [True, False])
    @pytest.mark.parametrize("total_steps", [1, 10, 20])
    def test_llm_collector_with_transformers(
        self, rb, total_steps, transformers_instance
    ):
        model, tokenizer = transformers_instance
        policy = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=True,
            generate=True,
            return_log_probs=True,
        )
        self._run_collector_test(total_steps, rb, policy, tokenizer)

    def _run_collector_test(self, total_steps, rb, policy, tokenizer):
        bsz = 4
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            steps_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_steps=total_steps,
        )

        stack = []
        for data in collector:
            # Should be moved to replay buffer
            if rb is not None:
                assert data is None
            else:
                stack.append(data)

        if rb is not None:
            # Now check the buffer
            assert len(rb) >= total_steps
            sample = rb.sample(4)
            assert sample.shape == (4,)
            assert not sample._has_exclusive_keys
            # Should match length
            assert len(sample["text"]) == 4
            # assert len(sample["text"][0]) == 10, sample["text"][0]
            # Should be non-empty
            assert sample["text_response"] is not None
            for i in range(4):
                # Check that there are more chars in the next step
                assert len(sample["text"][i]) < len(sample["next", "text"][i])
        else:
            stack = torch.cat(stack)
            assert not stack._has_exclusive_keys
            assert stack.numel() == max(-(total_steps // -4) * 4, 4)
            stack = stack.view(-1)
            for i in range(stack.numel()):
                # Check that there are more chars in the next step
                assert len(stack["text"][i]) < len(stack["next", "text"][i])
        assert collector._frames >= total_steps

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_llm_collector_start(self, vllm_instance):
        total_steps = 20
        policy = vLLMWrapper(vllm_instance)
        vllm_instance.get_tokenizer()
        bsz = 4
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
        )

        rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            steps_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_steps=total_steps,
        )
        torchrl_logger.info("starting")
        collector.start()

        j = 0
        while True:
            if not len(rb):
                await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep
            sample = rb.sample(10)
            assert sample.ndim == 1
            for i in range(10):
                # Check that there are more chars in the next step
                assert len(sample["text"][i]) < len(sample["next", "text"][i])
            assert not sample._has_exclusive_keys, sample
            j += 1
            if j == 5:
                break
        assert collector._frames >= total_steps

        try:
            # Assuming collector._task is the task created in start()
            await asyncio.wait_for(collector.async_shutdown(), timeout=30)
        except asyncio.TimeoutError:
            torchrl_logger.info("Collector shutdown timed out")

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [False, True])
    @pytest.mark.parametrize("yield_only_last_steps", [False, True])
    def test_llm_collector_completed(
        self, vllm_instance_opt, rb, yield_only_last_steps
    ):
        torch.manual_seed(0)
        policy = vLLMWrapper(vllm_instance_opt)
        tokenizer = vllm_instance_opt.get_tokenizer()
        bsz = 4
        total_steps = 20
        max_steps = 20
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        # To make sure the env breaks at some point
        env = env.append_transform(StepCounter(max_steps=max_steps))

        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            steps_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_steps=total_steps,
            yield_completed_trajectories=True,
            yield_only_last_steps=yield_only_last_steps,
        )
        assert collector.yield_completed_trajectories
        assert collector.yield_only_last_steps is yield_only_last_steps

        cur_total_steps = 0
        has_found_one_with_more_steps = False
        for data in collector:
            if rb is None:
                assert data.ndim == 1
                # assert (data["next", "step_count"] < max_steps-1).all()
                cur_total_steps += data.numel()
                for i in range(data.numel()):
                    if data[i]["next", "step_count"] == max_steps:
                        continue
                    if data[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(data["text"][i]) < len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )
                    else:
                        assert len(data["text"][i]) == len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )

                if yield_only_last_steps:
                    assert data.shape == (1,)
                else:
                    has_found_one_with_more_steps |= data.numel() > 1
            else:
                assert data is None
                sample = rb.sample(5)
                for i in range(sample.numel()):
                    if sample[i]["next", "step_count"] == max_steps:
                        continue
                    if sample[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(sample["text"][i]) < len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )
                    else:
                        assert len(sample["text"][i]) == len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )

                assert sample.ndim == 1
                assert sample.shape == (5,)
                assert (sample["next", "step_count"] < 99).all()
                cur_total_steps += 1
            assert collector._frames >= cur_total_steps
        if rb is None and not yield_only_last_steps:
            assert has_found_one_with_more_steps
        assert collector._frames >= total_steps

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [False, True])
    @pytest.mark.parametrize("yield_only_last_steps", [False, True])
    def test_llm_collector_completed_async(
        self, vllm_instance_opt, rb, yield_only_last_steps
    ):
        torch.manual_seed(0)
        policy = vLLMWrapper(vllm_instance_opt)
        tokenizer = vllm_instance_opt.get_tokenizer()
        bsz = 4
        total_steps = 20
        max_steps = 20
        dataloader = DummyStrDataLoader(bsz)

        def env_maker():
            env = LLMEnv.from_dataloader(
                dataloader=dataloader,
                from_text=True,
                batch_size=(),
                group_repeats=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            # To make sure the env breaks at some point
            env = env.append_transform(StepCounter(max_steps=max_steps))
            return env

        env = AsyncEnvPool([env_maker] * bsz, backend="threading", stack="lazy")

        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            steps_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_steps=total_steps,
            yield_completed_trajectories=True,
            yield_only_last_steps=yield_only_last_steps,
        )
        assert collector.yield_completed_trajectories
        assert collector.yield_only_last_steps is yield_only_last_steps

        cur_total_steps = 0
        has_found_one_with_more_steps = False
        for data in collector:
            if rb is None:
                assert data.ndim == 1
                # assert (data["next", "step_count"] < max_steps-1).all()
                cur_total_steps += data.numel()
                for i in range(data.numel()):
                    if data[i]["next", "step_count"] == max_steps:
                        continue
                    if data[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(data["text"][i]) < len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )
                    else:
                        assert len(data["text"][i]) == len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )

                if yield_only_last_steps:
                    assert data.shape == (1,)
                else:
                    has_found_one_with_more_steps |= data.numel() > 1
            else:
                assert data is None
                sample = rb.sample(5)
                for i in range(sample.numel()):
                    if sample[i]["next", "step_count"] == max_steps:
                        continue
                    if sample[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(sample["text"][i]) < len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )
                    else:
                        assert len(sample["text"][i]) == len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )

                assert sample.ndim == 1
                assert sample.shape == (5,)
                assert (sample["next", "step_count"] < 99).all()
                cur_total_steps += 1
            assert collector._frames >= cur_total_steps
        if rb is None and not yield_only_last_steps:
            assert has_found_one_with_more_steps
        assert collector._frames >= total_steps


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
