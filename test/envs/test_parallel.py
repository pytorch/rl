# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc
import os

import pytest
import torch

from _envs_common import _has_gym, IS_OSX, IS_WIN, mp_ctx
from tensordict import (
    assert_allclose_td,
    LazyStackedTensorDict,
    set_list_to_stack,
    TensorDict,
)
from tensordict.nn import TensorDictModuleBase
from torch import multiprocessing as mp, nn

from torchrl import set_auto_unwrap_transformed_env
from torchrl.collectors import Collector, MultiSyncCollector
from torchrl.data.tensor_specs import Composite
from torchrl.envs import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    SerialEnv,
    TransformedEnv,
)
from torchrl.envs.batched_envs import _stackable
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import Compose, StepCounter
from torchrl.modules import ActorCriticOperator, MLP, SafeModule, ValueOperator
from torchrl.testing import (
    CARTPOLE_VERSIONED,
    get_default_devices,
    make_envs as _make_envs,
    PENDULUM_VERSIONED,
    rand_reset,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnv,
    DiscreteActionVecMockEnv,
    MockBatchedLockedEnv,
    NestedCountingEnv,
)


class TestParallel:
    @pytest.fixture(autouse=True, scope="class")
    def disable_autowrap(self):
        with set_auto_unwrap_transformed_env(False):
            yield

    # Helper classes for test_parallel_env_chained_attr
    class _NestedObject:
        value = 42

        def get_value(self):
            return self.value

    class _EnvWithNestedAttr(DiscreteActionVecMockEnv):
        def __init__(self):
            super().__init__()
            self.nested = TestParallel._NestedObject()

    def test_create_env_fn(self, maybe_fork_ParallelEnv):
        def make_env():
            return GymEnv(PENDULUM_VERSIONED())

        with pytest.raises(
            RuntimeError, match="len\\(create_env_fn\\) and num_workers mismatch"
        ):
            maybe_fork_ParallelEnv(4, [make_env, make_env])

    def test_create_env_kwargs(self, maybe_fork_ParallelEnv):
        def make_env():
            return GymEnv(PENDULUM_VERSIONED())

        with pytest.raises(
            RuntimeError, match="len\\(create_env_kwargs\\) and num_workers mismatch"
        ):
            maybe_fork_ParallelEnv(
                4, make_env, create_env_kwargs=[{"seed": 0}, {"seed": 1}]
            )

    @pytest.mark.skipif(
        not torch.cuda.device_count(), reason="No cuda device detected."
    )
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("hetero", [True, False])
    @pytest.mark.parametrize("pdevice", [None, "cpu", "cuda"])
    @pytest.mark.parametrize("edevice", ["cpu", "cuda"])
    @pytest.mark.parametrize("bwad", [True, False])
    def test_parallel_devices(
        self, parallel, hetero, pdevice, edevice, bwad, maybe_fork_ParallelEnv
    ):
        if parallel:
            cls = maybe_fork_ParallelEnv
        else:
            cls = SerialEnv
        if not hetero:
            env = cls(
                2, lambda: ContinuousActionVecMockEnv(device=edevice), device=pdevice
            )
        else:
            env1 = lambda: ContinuousActionVecMockEnv(device=edevice)
            env2 = lambda: TransformedEnv(ContinuousActionVecMockEnv(device=edevice))
            env = cls(2, [env1, env2], device=pdevice)
        try:
            r = env.rollout(2, break_when_any_done=bwad)
            if pdevice is not None:
                assert env.device.type == torch.device(pdevice).type
                assert r.device.type == torch.device(pdevice).type
                assert all(
                    item.device.type == torch.device(pdevice).type
                    for item in r.values(True, True)
                )
            else:
                assert env.device.type == torch.device(edevice).type
                assert r.device.type == torch.device(edevice).type
                assert all(
                    item.device.type == torch.device(edevice).type
                    for item in r.values(True, True)
                )
            if parallel:
                assert (
                    env.shared_tensordict_parent.device.type
                    == torch.device(edevice).type
                )
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("start_method", [None, mp_ctx])
    def test_serial_for_single(self, maybe_fork_ParallelEnv, start_method):
        gc.collect()
        try:
            env = ParallelEnv(
                1,
                ContinuousActionVecMockEnv,
                serial_for_single=True,
                mp_start_method=start_method,
            )
            assert isinstance(env, SerialEnv)
            env = ParallelEnv(
                1, ContinuousActionVecMockEnv, mp_start_method=start_method
            )
            assert isinstance(env, ParallelEnv)
            env = ParallelEnv(
                2,
                ContinuousActionVecMockEnv,
                serial_for_single=True,
                mp_start_method=start_method,
            )
            assert isinstance(env, ParallelEnv)
        finally:
            env.close(raise_if_closed=False)

    def test_lambda_wrapping(self, maybe_fork_ParallelEnv):
        """Test that ParallelEnv automatically wraps lambda functions with EnvCreator.

        Lambda functions cannot be pickled with standard pickle (required for spawn
        start method), but EnvCreator uses cloudpickle which can handle them.
        This test verifies that lambda functions work correctly with ParallelEnv.
        """
        # Test single lambda function
        env = maybe_fork_ParallelEnv(2, lambda: ContinuousActionVecMockEnv())
        try:
            rollout = env.rollout(3)
            assert rollout.shape[0] == 2
            assert rollout.shape[1] == 3
        finally:
            env.close(raise_if_closed=False)

        # Test list of lambda functions (heterogeneous envs)
        env1 = lambda: ContinuousActionVecMockEnv()
        env2 = lambda: ContinuousActionVecMockEnv()
        env = maybe_fork_ParallelEnv(2, [env1, env2])
        try:
            rollout = env.rollout(3)
            assert rollout.shape[0] == 2
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("num_parallel_env", [1, 10])
    @pytest.mark.parametrize("env_batch_size", [[], (32,), (32, 1)])
    def test_env_with_batch_size(
        self, num_parallel_env, env_batch_size, maybe_fork_ParallelEnv
    ):
        try:
            env = MockBatchedLockedEnv(
                device="cpu", batch_size=torch.Size(env_batch_size)
            )
            env.set_seed(1)
            parallel_env = maybe_fork_ParallelEnv(num_parallel_env, lambda: env)
            assert parallel_env.batch_size == (num_parallel_env, *env_batch_size)
        finally:
            env.close(raise_if_closed=False)
            parallel_env.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_dmc, reason="no dm_control")
    @pytest.mark.parametrize("env_task", ["stand,stand,stand", "stand,walk,stand"])
    @pytest.mark.parametrize("share_individual_td", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_multi_task_serial_parallel(
        self, env_task, share_individual_td, maybe_fork_ParallelEnv, device
    ):
        tasks = env_task.split(",")
        if len(tasks) == 1:
            single_task = True

            def env_make():
                return DMControlEnv("humanoid", tasks[0], device=device)

        elif len(set(tasks)) == 1 and len(tasks) == 3:
            single_task = True
            env_make = [lambda: DMControlEnv("humanoid", tasks[0], device=device)] * 3
        else:
            single_task = False
            env_make = [
                lambda task=task: DMControlEnv("humanoid", task, device=device)
                for task in tasks
            ]

        env_serial = SerialEnv(3, env_make, share_individual_td=share_individual_td)
        try:
            env_serial.start()
            assert env_serial._single_task is single_task

            env_serial.set_seed(0)
            torch.manual_seed(0)
            td_serial = env_serial.rollout(max_steps=50)
        finally:
            env_serial.close(raise_if_closed=False)
            gc.collect()

        try:
            env_parallel = maybe_fork_ParallelEnv(
                3, env_make, share_individual_td=share_individual_td
            )
            env_parallel.start()
            assert env_parallel._single_task is single_task

            env_parallel.set_seed(0)
            torch.manual_seed(0)
            td_parallel = env_parallel.rollout(max_steps=50)

            assert_allclose_td(td_serial, td_parallel)
        finally:
            env_parallel.close(raise_if_closed=False)
            gc.collect()

    @pytest.mark.skipif(not _has_dmc, reason="no dm_control")
    def test_multitask(self, maybe_fork_ParallelEnv):
        env1 = DMControlEnv("humanoid", "stand")
        env1_obs_keys = list(env1.observation_spec.keys())
        env2 = DMControlEnv("humanoid", "walk")
        env2_obs_keys = list(env2.observation_spec.keys())

        assert len(env1_obs_keys)
        assert len(env2_obs_keys)

        def env1_maker():
            return TransformedEnv(
                DMControlEnv("humanoid", "stand"),
                Compose(
                    CatTensors(env1_obs_keys, "observation_stand", del_keys=False),
                    CatTensors(env1_obs_keys, "observation"),
                    DoubleToFloat(
                        in_keys=["observation_stand", "observation"],
                        in_keys_inv=["action"],
                    ),
                ),
            )

        def env2_maker():
            return TransformedEnv(
                DMControlEnv("humanoid", "walk"),
                Compose(
                    CatTensors(env2_obs_keys, "observation_walk", del_keys=False),
                    CatTensors(env2_obs_keys, "observation"),
                    DoubleToFloat(
                        in_keys=["observation_walk", "observation"],
                        in_keys_inv=["action"],
                    ),
                ),
            )

        try:
            env = maybe_fork_ParallelEnv(2, [env1_maker, env2_maker])
            # env = SerialEnv(2, [env1_maker, env2_maker])
            assert not env._single_task

            td = env.rollout(10, return_contiguous=False)
            assert "observation_walk" not in td.keys()
            assert "observation_walk" in td[1].keys()
            assert "observation_walk" not in td[0].keys()
            assert "observation_stand" in td[0].keys()
            assert "observation_stand" not in td[1].keys()
            assert "observation_walk" in td[:, 0][1].keys()
            assert "observation_walk" not in td[:, 0][0].keys()
            assert "observation_stand" in td[:, 0][0].keys()
            assert "observation_stand" not in td[:, 0][1].keys()
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED, CARTPOLE_VERSIONED]
    )  # 1226: faster execution
    @pytest.mark.parametrize("frame_skip", [4])  # 1226: faster execution
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_parallel_env(
        self, env_name, frame_skip, transformed_in, transformed_out, T=10, N=3
    ):
        env_name = env_name()
        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        try:
            td = TensorDict(
                source={"action": env0.action_spec.rand((N,))}, batch_size=[N]
            )
            env_parallel.reset()
            td1 = env_parallel.step(td)
            assert not td1.is_shared()
            assert ("next", "done") in td1.keys(True)
            assert ("next", "reward") in td1.keys(True)

            with pytest.raises(RuntimeError):
                # number of actions does not match number of workers
                td = TensorDict(
                    source={"action": env0.action_spec.rand((N - 1,))},
                    batch_size=[N - 1],
                )
                _ = env_parallel.step(td)

            td_reset = TensorDict(source=rand_reset(env_parallel), batch_size=[N])
            env_parallel.reset(tensordict=td_reset)

            # check that interruption occurred because of max_steps or done
            td = env_parallel.rollout(policy=None, max_steps=T)
            assert (
                td.shape == torch.Size([N, T]) or td.get(("next", "done")).sum(1).any()
            )
        finally:
            env_parallel.close(raise_if_closed=False)
            # env_serial.close()  # never opened
            env0.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [4])  # 1226: faster execution
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: faster execution
    def test_parallel_env_with_policy(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        T=10,
        N=3,
    ):
        env_name = env_name()
        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        try:
            policy = ActorCriticOperator(
                SafeModule(
                    spec=None,
                    module=nn.LazyLinear(12),
                    in_keys=["observation"],
                    out_keys=["hidden"],
                ),
                SafeModule(
                    spec=None,
                    module=nn.LazyLinear(env0.action_spec.shape[-1]),
                    in_keys=["hidden"],
                    out_keys=["action"],
                ),
                ValueOperator(
                    module=MLP(out_features=1, num_cells=[]),
                    in_keys=["hidden", "action"],
                ),
            )

            td = TensorDict(
                source={"action": env0.action_spec.rand((N,))}, batch_size=[N]
            )
            env_parallel.reset()
            td1 = env_parallel.step(td)
            assert not td1.is_shared()
            assert ("next", "done") in td1.keys(True)
            assert ("next", "reward") in td1.keys(True)

            with pytest.raises(RuntimeError):
                # number of actions does not match number of workers
                td = TensorDict(
                    source={"action": env0.action_spec.rand((N - 1,))},
                    batch_size=[N - 1],
                )
                _ = env_parallel.step(td)

            td_reset = TensorDict(source=rand_reset(env_parallel), batch_size=[N])
            env_parallel.reset(tensordict=td_reset)

            td = env_parallel.rollout(policy=policy, max_steps=T)
            assert (
                td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
            ), f"{td.shape}, {td.get('done').sum(1)}"
        finally:
            env_parallel.close(raise_if_closed=False)
            # env_serial.close()
            env0.close(raise_if_closed=False)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_transform_env_transform_no_device(
        self, heterogeneous, maybe_fork_ParallelEnv
    ):
        # Tests non-regression on 1865
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(), StepCounter(max_steps=3)
            )

        if heterogeneous:
            make_envs = [EnvCreator(make_env), EnvCreator(make_env)]
        else:
            make_envs = make_env
        penv = maybe_fork_ParallelEnv(2, make_envs)
        r = penv.rollout(6, break_when_any_done=False)
        assert r.shape == (2, 6)
        try:
            env = TransformedEnv(penv)
            r = env.rollout(6, break_when_any_done=False)
            assert r.shape == (2, 6)
        finally:
            penv.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name",
        [PENDULUM_VERSIONED],
    )  # PONG_VERSIONED])  # 1226: efficiency
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize(
        "transformed_in,transformed_out", [[True, True], [False, False]]
    )  # 1226: effociency
    @pytest.mark.parametrize("static_seed", [False, True])
    def test_parallel_env_seed(
        self, env_name, frame_skip, transformed_in, transformed_out, static_seed
    ):
        env_name = env_name()
        env_parallel, env_serial, _, _ = _make_envs(
            env_name, frame_skip, transformed_in, transformed_out, 5
        )
        try:
            out_seed_serial = env_serial.set_seed(0, static_seed=static_seed)
            if static_seed:
                assert out_seed_serial == 0
            td0_serial = env_serial.reset()
            torch.manual_seed(0)

            td_serial = env_serial.rollout(
                max_steps=10, auto_reset=False, tensordict=td0_serial
            ).contiguous()
            key = "pixels" if "pixels" in td_serial.keys() else "observation"
            torch.testing.assert_close(
                td_serial[:, 0].get(("next", key)), td_serial[:, 1].get(key)
            )

            out_seed_parallel = env_parallel.set_seed(0, static_seed=static_seed)
            if static_seed:
                assert out_seed_serial == 0
            td0_parallel = env_parallel.reset()

            torch.manual_seed(0)
            assert out_seed_parallel == out_seed_serial
            td_parallel = env_parallel.rollout(
                max_steps=10, auto_reset=False, tensordict=td0_parallel
            ).contiguous()
            torch.testing.assert_close(
                td_parallel[:, :-1].get(("next", key)), td_parallel[:, 1:].get(key)
            )
            assert_allclose_td(td0_serial, td0_parallel)
            assert_allclose_td(td_serial[:, 0], td_parallel[:, 0])  # first step
            assert_allclose_td(td_serial[:, 1], td_parallel[:, 1])  # second step
            assert_allclose_td(td_serial, td_parallel)
        finally:
            env_parallel.close(raise_if_closed=False)
            env_serial.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_parallel_env_shutdown(self, maybe_fork_ParallelEnv):
        env_make = EnvCreator(lambda: GymEnv(PENDULUM_VERSIONED()))
        env = maybe_fork_ParallelEnv(4, env_make)
        try:
            env.reset()
            assert not env.is_closed
            env.rand_step()
            assert not env.is_closed
            env.close()
            assert env.is_closed
            env.reset()
            assert not env.is_closed
            env.close()
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_env_custom_method(self, parallel, maybe_fork_ParallelEnv):
        # define env

        if parallel:
            env = maybe_fork_ParallelEnv(2, lambda: DiscreteActionVecMockEnv())
        else:
            env = SerialEnv(2, lambda: DiscreteActionVecMockEnv())
        try:
            # we must start the environment first
            env.reset()
            assert all(result == 0 for result in env.custom_fun())
            assert all(result == 1 for result in env.custom_attr)
            assert all(result == 2 for result in env.custom_prop)  # to be fixed
        finally:
            env.close(raise_if_closed=False)

    def test_parallel_env_chained_attr(self, maybe_fork_ParallelEnv):
        """Test chained attribute access like env.nested.value works in ParallelEnv."""
        env = maybe_fork_ParallelEnv(2, TestParallel._EnvWithNestedAttr)
        try:
            env.reset()
            # Test chained attribute access
            results = list(env.nested.value)
            assert all(result == 42 for result in results)
            # Test chained method access
            results = list(env.nested.get_value())
            assert all(result == 42 for result in results)
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda to test on")
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED]
    )  # 1226: Skip PONG for efficiency
    @pytest.mark.parametrize(
        "transformed_in,transformed_out,open_before",
        [  # 1226: efficiency
            [True, True, True],
            [True, True, False],
            [False, False, True],
        ],
    )
    def test_parallel_env_cast(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        device,
        open_before,
        N=3,
    ):
        env_name = env_name()
        # tests casting to device
        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        try:
            if open_before:
                td_cpu = env0.rollout(max_steps=10)
                assert td_cpu.device == torch.device("cpu")
            env0 = env0.to(device)
            assert env0.observation_spec.device == torch.device(device)
            assert env0.action_spec.device == torch.device(device)
            assert env0.reward_spec.device == torch.device(device)
            assert env0.device == torch.device(device)
            td_device = env0.reset()
            assert td_device.device == torch.device(device), env0
            td_device = env0.rand_step()
            assert td_device.device == torch.device(device), env0
            td_device = env0.rollout(max_steps=10)
            assert td_device.device == torch.device(device), env0

            if open_before:
                td_cpu = env_serial.rollout(max_steps=10)
                assert td_cpu.device == torch.device("cpu")
            observation_spec = env_serial.observation_spec.clone()
            done_spec = env_serial.done_spec.clone()
            reward_spec = env_serial.reward_spec.clone()
            action_spec = env_serial.action_spec.clone()
            state_spec = env_serial.state_spec.clone()
            env_serial = env_serial.to(device)
            assert env_serial.observation_spec.device == torch.device(device)
            assert env_serial.action_spec.device == torch.device(device)
            assert env_serial.reward_spec.device == torch.device(device)
            assert env_serial.device == torch.device(device)
            assert env_serial.observation_spec == observation_spec.to(device)
            assert env_serial.action_spec == action_spec.to(device)
            assert env_serial.reward_spec == reward_spec.to(device)
            assert env_serial.done_spec == done_spec.to(device)
            assert env_serial.state_spec == state_spec.to(device)
            td_device = env_serial.reset()
            assert td_device.device == torch.device(device), env_serial
            td_device = env_serial.rand_step()
            assert td_device.device == torch.device(device), env_serial
            td_device = env_serial.rollout(max_steps=10)
            assert td_device.device == torch.device(device), env_serial

            if open_before:
                td_cpu = env_parallel.rollout(max_steps=10)
                assert td_cpu.device == torch.device("cpu")
            observation_spec = env_parallel.observation_spec.clone()
            done_spec = env_parallel.done_spec.clone()
            reward_spec = env_parallel.reward_spec.clone()
            action_spec = env_parallel.action_spec.clone()
            state_spec = env_parallel.state_spec.clone()
            env_parallel = env_parallel.to(device)
            assert env_parallel.observation_spec.device == torch.device(device)
            assert env_parallel.action_spec.device == torch.device(device)
            assert env_parallel.reward_spec.device == torch.device(device)
            assert env_parallel.device == torch.device(device)
            assert env_parallel.observation_spec == observation_spec.to(device)
            assert env_parallel.action_spec == action_spec.to(device)
            assert env_parallel.reward_spec == reward_spec.to(device)
            assert env_parallel.done_spec == done_spec.to(device)
            assert env_parallel.state_spec == state_spec.to(device)
            td_device = env_parallel.reset()
            assert td_device.device == torch.device(device), env_parallel
            td_device = env_parallel.rand_step()
            assert td_device.device == torch.device(device), env_parallel
            td_device = env_parallel.rollout(max_steps=10)
            assert td_device.device == torch.device(device), env_parallel
        finally:
            env_parallel.close(raise_if_closed=False)
            env_serial.close(raise_if_closed=False)
            env0.close(raise_if_closed=False)

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device detected")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])  # 1226: efficiency
    @pytest.mark.parametrize(
        "transformed_in,transformed_out",
        [  # 1226
            [True, True],
            [False, False],
        ],
    )
    def test_parallel_env_device(
        self, env_name, frame_skip, transformed_in, transformed_out, device
    ):
        env_name = env_name()
        # tests creation on device
        torch.manual_seed(0)
        N = 3

        env_parallel, env_serial, _, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            device=device,
            N=N,
            local_mp_ctx="spawn",
        )

        try:
            assert env0.device == torch.device(device)
            out = env0.rollout(max_steps=20)
            assert out.device == torch.device(device)

            assert env_serial.device == torch.device(device)
            out = env_serial.rollout(max_steps=20)
            assert out.device == torch.device(device)

            assert env_parallel.device == torch.device(device)
            out = env_parallel.rollout(max_steps=20)
            assert out.device == torch.device(device)
        finally:
            env_parallel.close(raise_if_closed=False)
            env_serial.close(raise_if_closed=False)
            env0.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_device", [None, "cpu"])
    def test_parallel_env_device_vs_no_device(self, maybe_fork_ParallelEnv, env_device):
        def make_env() -> GymEnv:
            env = GymEnv(PENDULUM_VERSIONED(), device=env_device)
            return env.append_transform(DoubleToFloat())

        # Rollouts work with a regular env
        parallel_env = maybe_fork_ParallelEnv(
            num_workers=1, create_env_fn=make_env, device=None
        )
        parallel_env.reset()
        parallel_env.set_seed(0)
        torch.manual_seed(0)

        parallel_rollout = parallel_env.rollout(max_steps=10)

        # Rollout doesn't work with Parallelnv
        parallel_env = maybe_fork_ParallelEnv(
            num_workers=1, create_env_fn=make_env, device="cpu"
        )
        parallel_env.reset()
        parallel_env.set_seed(0)
        torch.manual_seed(0)

        parallel_rollout_cpu = parallel_env.rollout(max_steps=10)
        assert_allclose_td(parallel_rollout, parallel_rollout_cpu)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    @pytest.mark.parametrize(
        "env_name", [PENDULUM_VERSIONED]
    )  # 1226: No pong for efficiency
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize(
        "device",
        [torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")],
    )
    def test_parallel_env_transform_consistency(self, env_name, frame_skip, device):
        env_name = env_name()
        env_parallel_in, env_serial_in, _, env0_in = _make_envs(
            env_name,
            frame_skip,
            transformed_in=True,
            transformed_out=False,
            device=device,
            N=3,
            local_mp_ctx="spawn" if torch.cuda.device_count() else mp_ctx,
        )
        env_parallel_out, env_serial_out, _, env0_out = _make_envs(
            env_name,
            frame_skip,
            transformed_in=False,
            transformed_out=True,
            device=device,
            N=3,
            local_mp_ctx="spawn" if torch.cuda.device_count() else mp_ctx,
        )
        try:
            torch.manual_seed(0)
            env_parallel_in.set_seed(0)
            r_in = env_parallel_in.rollout(max_steps=20)
            torch.manual_seed(0)
            env_parallel_out.set_seed(0)
            r_out = env_parallel_out.rollout(max_steps=20)
            assert_allclose_td(r_in, r_out)
            env_parallel_in.close()
            env_parallel_out.close()

            torch.manual_seed(0)
            env_serial_in.set_seed(0)
            r_in = env_serial_in.rollout(max_steps=20)
            torch.manual_seed(0)
            env_serial_out.set_seed(0)
            r_out = env_serial_out.rollout(max_steps=20)
            assert_allclose_td(r_in, r_out)
            env_serial_in.close()
            env_serial_out.close()

            torch.manual_seed(0)
            env0_in.set_seed(0)
            r_in = env0_in.rollout(max_steps=20)
            torch.manual_seed(0)
            env0_out.set_seed(0)
            r_out = env0_out.rollout(max_steps=20)
            assert_allclose_td(r_in, r_out)
        finally:
            env0_in.close(raise_if_closed=False)
            env0_in.close(raise_if_closed=False)

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_env_kwargs_set(self, parallel, maybe_fork_ParallelEnv):
        num_env = 2

        def make_make_env():
            def make_transformed_env(seed=None):
                env = DiscreteActionConvMockEnv()
                if seed is not None:
                    env.set_seed(seed)
                return env

            return make_transformed_env

        _class = maybe_fork_ParallelEnv if parallel else SerialEnv

        def env_fn1(seed):
            env = _class(
                num_workers=num_env,
                create_env_fn=make_make_env(),
                create_env_kwargs=[{"seed": i} for i in range(seed, seed + num_env)],
            )
            return env

        def env_fn2(seed):
            env = _class(
                num_workers=num_env,
                create_env_fn=make_make_env(),
            )
            env.update_kwargs([{"seed": i} for i in range(seed, seed + num_env)])
            return env

        env1 = env_fn1(100)
        env2 = env_fn2(100)
        try:
            env1.start()
            env2.start()
            for c1, c2 in zip(env1.counter, env2.counter):
                assert c1 == c2
        finally:
            env1.close(raise_if_closed=False)
            env2.close(raise_if_closed=False)

    @pytest.mark.parametrize("parallel", [True, False])
    def test_parallel_env_update_kwargs(self, parallel, maybe_fork_ParallelEnv):
        def make_env(seed=None):
            env = DiscreteActionConvMockEnv()
            if seed is not None:
                env.set_seed(seed)
            return env

        _class = maybe_fork_ParallelEnv if parallel else SerialEnv
        env = _class(
            num_workers=2,
            create_env_fn=make_env,
            create_env_kwargs=[{"seed": 0}, {"seed": 1}],
        )
        with pytest.raises(
            RuntimeError, match="len\\(kwargs\\) and num_workers mismatch"
        ):
            env.update_kwargs([{"seed": 42}])

    @pytest.mark.parametrize("batch_size", [(32, 5), (4,), (1,), ()])
    @pytest.mark.parametrize("n_workers", [2, 1])
    def test_parallel_env_reset_flag(
        self, batch_size, n_workers, maybe_fork_ParallelEnv, max_steps=3
    ):
        torch.manual_seed(1)
        env = maybe_fork_ParallelEnv(
            n_workers, lambda: CountingEnv(max_steps=max_steps, batch_size=batch_size)
        )
        try:
            env.set_seed(1)
            action = env.full_action_spec[env.action_key].rand()
            action[:] = 1
            for i in range(max_steps):
                td = env.step(
                    TensorDict(
                        {"action": action}, batch_size=env.batch_size, device=env.device
                    )
                )
                assert (td["next", "done"] == 0).all()
                assert (td["next"]["observation"] == i + 1).all()

            td = env.step(
                TensorDict(
                    {"action": action}, batch_size=env.batch_size, device=env.device
                )
            )
            assert (td["next", "done"] == 1).all()
            assert (td["next"]["observation"] == max_steps + 1).all()

            td_reset = TensorDict(
                rand_reset(env), batch_size=env.batch_size, device=env.device
            )
            td_reset.update(td.get("next").exclude("reward"))
            reset = td_reset["_reset"]
            td_reset = env.reset(td_reset)

            assert (td_reset["done"][reset] == 0).all()
            assert (td_reset["observation"][reset] == 0).all()
            assert (td_reset["done"][~reset] == 1).all()
            assert (td_reset["observation"][~reset] == max_steps + 1).all()
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("nested_obs_action", [True, False])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("nested_reward", [True, False])
    @pytest.mark.parametrize("env_type", ["serial", "parallel"])
    def test_parallel_env_nested(
        self,
        nested_obs_action,
        nested_done,
        nested_reward,
        env_type,
        maybe_fork_ParallelEnv,
        n_envs=2,
        batch_size=(32,),
        nested_dim=5,
        rollout_length=3,
        seed=1,
    ):
        env_fn = lambda: NestedCountingEnv(
            nest_done=nested_done,
            nest_reward=nested_reward,
            nest_obs_action=nested_obs_action,
            batch_size=batch_size,
            nested_dim=nested_dim,
        )

        if env_type == "serial":
            env = SerialEnv(n_envs, env_fn)
        else:
            env = maybe_fork_ParallelEnv(n_envs, env_fn)

        try:
            env.set_seed(seed)

            batch_size = (n_envs, *batch_size)

            td = env.reset()
            assert td.batch_size == batch_size
            if nested_done or nested_obs_action:
                assert td["data"].batch_size == (*batch_size, nested_dim)
            if not nested_done and not nested_reward and not nested_obs_action:
                assert "data" not in td.keys()

            policy = CountingEnvCountPolicy(
                env.full_action_spec[env.action_key], env.action_key
            )
            td = env.rollout(rollout_length, policy)
            assert td.batch_size == (*batch_size, rollout_length)
            if nested_done or nested_obs_action:
                assert td["data"].batch_size == (
                    *batch_size,
                    rollout_length,
                    nested_dim,
                )
            if nested_reward or nested_done or nested_obs_action:
                assert td["next", "data"].batch_size == (
                    *batch_size,
                    rollout_length,
                    nested_dim,
                )
            if not nested_done and not nested_reward and not nested_obs_action:
                assert "data" not in td.keys()
                assert "data" not in td["next"].keys()

            if nested_obs_action:
                assert "observation" not in td.keys()
                assert (td[..., -1]["data", "states"] == 2).all()
            else:
                assert ("data", "states") not in td.keys(True, True)
                assert (td[..., -1]["observation"] == 2).all()
        finally:
            env.close(raise_if_closed=False)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
class TestConcurrentEnvs:
    """Concurrent parallel envs on multiple procs can interfere."""

    class Policy(TensorDictModuleBase):
        in_keys = []
        out_keys = ["action"]

        def __init__(self, spec):
            super().__init__()
            self.spec = spec

        def forward(self, tensordict):
            tensordict.set("action", self.spec["action"].zero() + 1)
            return tensordict

    @staticmethod
    def main_penv(j, q=None):
        gc.collect()
        device = "cpu" if not torch.cuda.device_count() else "cuda:0"
        n_workers = 1
        env_p = ParallelEnv(
            n_workers,
            [
                lambda i=i: CountingEnv(i, device=device)
                for i in range(j, j + n_workers)
            ],
        )
        env_s = SerialEnv(
            n_workers,
            [
                lambda i=i: CountingEnv(i, device=device)
                for i in range(j, j + n_workers)
            ],
        )
        spec = env_p.action_spec
        policy = TestConcurrentEnvs.Policy(Composite(action=spec.to(device)))
        N = 10
        r_p = []
        r_s = []
        for _ in range(N):
            with torch.no_grad():
                r_p.append(env_s.rollout(100, break_when_any_done=False, policy=policy))
                r_s.append(env_p.rollout(100, break_when_any_done=False, policy=policy))

        td_equals = torch.stack(r_p) == torch.stack(r_s)
        if td_equals.all():
            if q is not None:
                q.put(("passed", j))
            else:
                pass
        else:
            if q is not None:
                s = ""
                for key, item in td_equals.items(True, True):
                    if not item.all():
                        s = s + f"\t{key}"
                q.put((f"failed: {s}", j))
            else:
                raise RuntimeError()

    @staticmethod
    def main_collector(j, q=None):
        device = "cpu" if not torch.cuda.device_count() else "cuda:0"
        N = 10
        n_workers = 1
        make_envs = [
            lambda i=i: CountingEnv(i, device=device) for i in range(j, j + n_workers)
        ]
        spec = make_envs[0]().action_spec
        policy = TestConcurrentEnvs.Policy(Composite(action=spec))
        collector = MultiSyncCollector(
            make_envs,
            policy,
            frames_per_batch=n_workers * 100,
            total_frames=N * n_workers * 100,
            storing_device=device,
            device=device,
            trust_policy=True,
            cat_results=-1,
        )
        single_collectors = [
            Collector(
                make_envs[i](),
                policy,
                frames_per_batch=n_workers * 100,
                total_frames=N * n_workers * 100,
                storing_device=device,
                trust_policy=True,
                device=device,
            )
            for i in range(n_workers)
        ]
        iter_collector = iter(collector)
        iter_single_collectors = [iter(sc) for sc in single_collectors]

        r_p = []
        r_s = []
        for _ in range(N):
            with torch.no_grad():
                r_p.append(next(iter_collector).clone())
                r_s.append(torch.cat([next(sc) for sc in iter_single_collectors]))

        collector.shutdown()
        for sc in single_collectors:
            sc.shutdown()
        del collector
        del single_collectors
        r_p = torch.stack(r_p).contiguous()
        r_s = torch.stack(r_s).contiguous()
        td_equals = r_p == r_s

        if td_equals.all():
            if q is not None:
                q.put(("passed", j))
            else:
                pass
        else:
            if q is not None:
                s = ""
                for key, item in td_equals.items(True, True):
                    if not item.all():
                        s = s + f"\t{key}"
                q.put((f"failed: {s}", j))
            else:
                raise RuntimeError()

    @pytest.mark.parametrize("nproc", [3, 1])
    def test_mp_concurrent(self, nproc):
        if nproc == 1:
            self.main_penv(3)
            self.main_penv(6)
            self.main_penv(9)
        else:
            q = mp.Queue(3)
            ps = []
            try:
                for k in range(3, 10, 3):
                    p = mp.Process(target=type(self).main_penv, args=(k, q))
                    ps.append(p)
                    p.start()
                for _ in range(3):
                    msg, j = q.get(timeout=100)
                    assert msg == "passed", j
            finally:
                for p in ps:
                    p.join()

    @pytest.mark.parametrize("nproc", [3, 1])
    def test_mp_collector(self, nproc):
        if nproc == 1:
            self.main_collector(3)
            self.main_collector(6)
            self.main_collector(9)
        else:
            q = mp.Queue(3)
            ps = []
            try:
                for j in range(3, 10, 3):
                    p = mp.Process(target=type(self).main_collector, args=(j, q))
                    ps.append(p)
                    p.start()
                for _ in range(3):
                    msg, j = q.get(timeout=100)
                    assert msg == "passed", j
            finally:
                for p in ps:
                    p.join(timeout=2)


class TestLibThreading:
    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_num_threads(self):
        gc.collect()
        num_threads = torch.get_num_threads()
        main_pid = os.getpid()
        try:
            # Wrap the env factory to check thread count inside the subprocess.
            # The env is created AFTER torch.set_num_threads() is called in the worker.
            # Note: the factory is also called in the main process to get metadata,
            # so we only check thread count when running in a subprocess.
            def make_env():
                if os.getpid() != main_pid:
                    # Only check thread count in subprocess, not during metadata extraction
                    assert (
                        torch.get_num_threads() == 3
                    ), f"Expected 3 threads, got {torch.get_num_threads()}"
                return ContinuousActionVecMockEnv()

            env = ParallelEnv(2, make_env, num_sub_threads=3, num_threads=7)
            # We could test that the number of threads isn't changed until we start the procs.
            # Even though it's unlikely that we have 7 threads, we still disable this for safety
            # assert torch.get_num_threads() != 7
            env.rollout(3)
            assert torch.get_num_threads() == 7
        finally:
            torch.set_num_threads(num_threads)

    @pytest.mark.skipif(
        IS_OSX,
        reason="setting different threads across workers can randomly fail on OSX.",
    )
    def test_auto_num_threads(self, maybe_fork_ParallelEnv):
        gc.collect()
        init_threads = torch.get_num_threads()

        try:
            env3 = maybe_fork_ParallelEnv(3, ContinuousActionVecMockEnv)
            env3.rollout(2)

            assert torch.get_num_threads() == max(1, init_threads - 3)

            env2 = maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv)
            env2.rollout(2)

            assert torch.get_num_threads() == max(1, init_threads - 5)

            env2.close()
            del env2
            gc.collect()

            assert torch.get_num_threads() == max(1, init_threads - 3)

            env3.close()
            del env3
            gc.collect()

            assert torch.get_num_threads() == init_threads
        finally:
            torch.set_num_threads(init_threads)


@pytest.mark.skipif(IS_WIN, reason="fork not available on windows 10")
def test_parallel_another_ctx():
    gc.collect()

    try:
        sm = mp.get_start_method()
        if sm == "spawn":
            other_sm = "fork"
        else:
            other_sm = "spawn"
        env = ParallelEnv(2, ContinuousActionVecMockEnv, mp_start_method=other_sm)
        assert env.rollout(3) is not None
        assert env._workers[0]._start_method == other_sm
    finally:
        try:
            env.close()
            del env
        except Exception:
            pass


@pytest.mark.skipif(not _has_gym, reason="gym not found")
def test_single_task_share_individual_td():
    cartpole = CARTPOLE_VERSIONED()
    env = SerialEnv(2, lambda: GymEnv(cartpole))
    assert not env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(2, lambda: GymEnv(cartpole), share_individual_td=True)
    assert env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    env = SerialEnv(2, [lambda: GymEnv(cartpole)] * 2)
    assert not env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(2, [lambda: GymEnv(cartpole)] * 2, share_individual_td=True)
    assert env.share_individual_td
    assert env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    env = SerialEnv(2, [EnvCreator(lambda: GymEnv(cartpole)) for _ in range(2)])
    assert not env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, TensorDict)

    env = SerialEnv(
        2,
        [EnvCreator(lambda: GymEnv(cartpole)) for _ in range(2)],
        share_individual_td=True,
    )
    assert env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    # Change shape: makes results non-stackable
    env = SerialEnv(
        2,
        [
            EnvCreator(lambda: GymEnv(cartpole)),
            EnvCreator(
                lambda: TransformedEnv(
                    GymEnv(cartpole), CatFrames(N=4, dim=-1, in_keys=["observation"])
                )
            ),
        ],
    )
    assert env.share_individual_td
    assert not env._single_task
    env.rollout(2)
    assert isinstance(env.shared_tensordict_parent, LazyStackedTensorDict)

    with pytest.raises(ValueError, match="share_individual_td=False"):
        SerialEnv(
            2,
            [
                EnvCreator(lambda: GymEnv(cartpole)),
                EnvCreator(
                    lambda: TransformedEnv(
                        GymEnv(cartpole),
                        CatFrames(N=4, dim=-1, in_keys=["observation"]),
                    )
                ),
            ],
            share_individual_td=False,
        )


@set_list_to_stack(True)
def test_stackable():
    # Tests the _stackable util
    stack = [TensorDict({"a": 0}, []), TensorDict({"b": 1}, [])]
    assert not _stackable(*stack), torch.stack(stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": 1}, [])]
    assert not _stackable(*stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": [1]}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": [0]}, []), TensorDict({"a": [1], "b": {}}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": {"b": [0]}}, []), TensorDict({"a": {"b": [1]}}, [])]
    assert _stackable(*stack)
    stack = [TensorDict({"a": {"b": [0]}}, []), TensorDict({"a": {"b": 1}}, [])]
    assert not _stackable(*stack)
    stack = [TensorDict({"a": "a string"}, []), TensorDict({"a": "another string"}, [])]
    assert _stackable(*stack)
