# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import gc
import pickle
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch

from _envs_common import _has_ale, _has_gym, mp_ctx
from packaging import version
from tensordict import assert_allclose_td, TensorDict, TensorDictBase
from tensordict.tensorclass import TensorClass
from torch import nn

from torchrl.data.tensor_specs import Composite, NonTensor, Unbounded
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from torchrl.envs.libs.gym import gym_backend, GymEnv
from torchrl.envs.transforms import StepCounter, TransformedEnv
from torchrl.envs.utils import check_env_specs, make_composite_from_td, step_mdp
from torchrl.modules import Actor
from torchrl.testing import (
    CARTPOLE_VERSIONED,
    get_default_devices,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    EnvThatDoesNothing,
    EnvThatErrorsBecauseOfStack,
    EnvWithMetadata,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
    MockSerialEnv,
    NestedCountingEnv,
)


class TestEnvBase:
    def test_run_type_checks(self):
        env = ContinuousActionVecMockEnv()
        env.adapt_dtype = False
        env._run_type_checks = False
        check_env_specs(env)
        env._run_type_checks = True
        check_env_specs(env)
        env.output_spec.unlock_(recurse=True)
        # check type check on done
        env.output_spec["full_done_spec", "done"].dtype = torch.int
        with pytest.raises(TypeError, match="expected done.dtype to"):
            check_env_specs(env)
        env.output_spec["full_done_spec", "done"].dtype = torch.bool
        # check type check on reward
        env.output_spec["full_reward_spec", "reward"].dtype = torch.int
        with pytest.raises(TypeError, match="expected"):
            check_env_specs(env)
        env.output_spec["full_reward_spec", "reward"].dtype = torch.float
        # check type check on obs
        env.output_spec["full_observation_spec", "observation"].dtype = torch.float16
        with pytest.raises(TypeError):
            check_env_specs(env)

    def test_check_env_specs_state_spec_keys(self):
        """Regression test for https://github.com/pytorch/rl/issues/3260.

        check_env_specs() should succeed when state_spec contains keys that are
        not present in observation_spec (e.g. RNG seeds, auxiliary state).
        Such keys may be returned by _step and appear in the real rollout's
        "next", but are intentionally excluded from the key comparison in
        check_env_specs() since their presence is optional.
        """

        class EnvWithExtraState(EnvBase):
            def __init__(self):
                super().__init__()
                self.observation_spec = Composite(
                    observation=Unbounded(shape=(3,), dtype=torch.float32)
                )
                self.action_spec = Unbounded(shape=(1,), dtype=torch.float32)
                self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
                # state_spec has a key ("hidden_state") absent from observation_spec
                self.state_spec = Composite(
                    hidden_state=Unbounded(shape=(4,), dtype=torch.float32)
                )

            def _reset(self, tensordict):
                return TensorDict(
                    {
                        "observation": torch.zeros(3),
                        "hidden_state": torch.zeros(4),
                    }
                )

            def _step(self, tensordict):
                return TensorDict(
                    {
                        "observation": torch.zeros(3),
                        "hidden_state": torch.zeros(4),
                        "reward": torch.zeros(1),
                        "done": torch.zeros(1, dtype=torch.bool),
                        "terminated": torch.zeros(1, dtype=torch.bool),
                    }
                )

            def _set_seed(self, seed):
                pass

        # check_env_specs() must not raise even though _step returns
        # "hidden_state" (a state_spec key) in "next"
        check_env_specs(EnvWithExtraState())

    class MyEnv(EnvBase):
        def __init__(self):
            super().__init__()
            self.observation_spec = Unbounded(())
            self.action_spec = Unbounded(())

        def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
            ...

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            ...

        def _set_seed(self, seed: int | None) -> None:
            ...

    def test_env_lock(self):

        env = self.MyEnv()
        for _ in range(2):
            assert env.is_spec_locked
            assert env.output_spec.is_locked
            assert env.input_spec.is_locked
            with pytest.raises(RuntimeError, match="lock"):
                env.input_spec["full_action_spec", "action"] = Unbounded(())
            env = pickle.loads(pickle.dumps(env))

        env = self.MyEnv(spec_locked=False)
        assert not env.is_spec_locked
        assert not env.output_spec.is_locked
        assert not env.input_spec.is_locked
        env.input_spec["full_action_spec", "action"] = Unbounded(())

    def test_single_env_spec(self):
        env = NestedCountingEnv(batch_size=[3, 1, 7])
        assert not env.full_action_spec_unbatched.shape
        assert not env.full_done_spec_unbatched.shape
        assert not env.input_spec_unbatched.shape
        assert not env.full_observation_spec_unbatched.shape
        assert not env.output_spec_unbatched.shape
        assert not env.full_reward_spec_unbatched.shape

        assert env.full_action_spec_unbatched[env.action_key].shape
        assert env.full_reward_spec_unbatched[env.reward_key].shape

        assert env.output_spec.is_in(env.output_spec_unbatched.zeros(env.shape))
        assert env.input_spec.is_in(env.input_spec_unbatched.zeros(env.shape))

    @pytest.mark.skipif(not _has_gym, reason="Gym required for this test")
    def test_non_td_policy(self):
        env = GymEnv("CartPole-v1", categorical_action_encoding=True)

        class ArgMaxModule(nn.Module):
            def forward(self, values):
                return values.argmax(-1)

        policy = nn.Sequential(
            nn.Linear(
                env.observation_spec["observation"].shape[-1],
                env.full_action_spec[env.action_key].n,
            ),
            ArgMaxModule(),
        )
        env.rollout(10, policy)
        env = SerialEnv(
            2, lambda: GymEnv("CartPole-v1", categorical_action_encoding=True)
        )
        env.rollout(10, policy)

    def test_stack_error(self):
        env = EnvThatErrorsBecauseOfStack()
        assert not env._has_dynamic_specs
        cm = pytest.raises(
            RuntimeError,
            match="The reward key was present in the root tensordict of at least one of the tensordicts to stack",
        )
        with cm:
            env.check_env_specs()
        with cm:
            env.rollout(10, break_when_any_done=True, return_contiguous=True)
        with cm:
            env.rollout(10, break_when_any_done=False, return_contiguous=True)

    @pytest.mark.parametrize("dynamic_shape", [True, False])
    def test_make_spec_from_td(self, dynamic_shape):
        data = TensorDict(
            {
                "obs": torch.randn(3),
                "action": torch.zeros(2, dtype=torch.int),
                "next": {
                    "obs": torch.randn(3),
                    "reward": torch.randn(1),
                    "done": torch.zeros(1, dtype=torch.bool),
                },
            },
            [],
        )
        spec = make_composite_from_td(data, dynamic_shape=dynamic_shape)
        assert (spec.zero() == data.zero_()).all()
        for key, val in data.items(True, True):
            assert val.dtype is spec[key].dtype
        if dynamic_shape:
            assert all(s.shape[-1] == -1 for s in spec.values(True, True))

    def test_make_spec_from_tc(self):
        class Scratch(TensorClass):
            obs: torch.Tensor
            string: str
            some_object: Any

        class Whatever:
            ...

        td = TensorDict(
            a=Scratch(
                obs=torch.ones(5, 3),
                string="another string!",
                some_object=Whatever(),
                batch_size=(5,),
            ),
            b="a string!",
            batch_size=(5,),
        )
        spec = make_composite_from_td(td)
        assert isinstance(spec, Composite)
        assert isinstance(spec["a"], Composite)
        assert isinstance(spec["b"], NonTensor)
        assert spec["b"].example_data == "a string!", spec["b"].example_data
        assert spec["a", "string"].example_data == "another string!"
        one = spec.one()
        assert isinstance(one["a"], Scratch)
        assert isinstance(one["b"], str)
        assert isinstance(one["a"].string, str)
        assert isinstance(one["a"].some_object, Whatever)
        assert (one == td).all()

    def test_env_that_does_nothing(self):
        env = EnvThatDoesNothing()
        env.check_env_specs()
        r = env.rollout(3)
        r.exclude(
            "done", "terminated", ("next", "done"), ("next", "terminated"), inplace=True
        )
        assert r.is_empty()
        p_env = SerialEnv(2, EnvThatDoesNothing)
        p_env.check_env_specs()
        r = p_env.rollout(3)
        r.exclude(
            "done", "terminated", ("next", "done"), ("next", "terminated"), inplace=True
        )
        assert r.is_empty()
        p_env = ParallelEnv(2, EnvThatDoesNothing)
        try:
            p_env.check_env_specs()
            r = p_env.rollout(3)
            r.exclude(
                "done",
                "terminated",
                ("next", "done"),
                ("next", "terminated"),
                inplace=True,
            )
            assert r.is_empty()
        finally:
            p_env.close()
            del p_env

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("share_individual_td", [True, False])
    def test_backprop(self, device, maybe_fork_ParallelEnv, share_individual_td):
        gc.collect()

        # Tests that backprop through a series of single envs and through a serial env are identical
        # Also tests that no backprop can be achieved with parallel env.
        class DifferentiableEnv(EnvBase):
            def __init__(self, device):
                super().__init__(device=device)
                self.observation_spec = Composite(
                    observation=Unbounded(3, device=device),
                    device=device,
                )
                self.action_spec = Composite(
                    action=Unbounded(3, device=device), device=device
                )
                self.reward_spec = Composite(
                    reward=Unbounded(1, device=device), device=device
                )
                self.seed = 0

            def _set_seed(self, seed: int | None) -> None:
                self.seed = seed

            def _reset(self, tensordict):
                td = self.observation_spec.zero().update(self.done_spec.zero())
                td["observation"] = (
                    td["observation"].clone() + self.seed % 10
                ).requires_grad_()
                return td

            def _step(self, tensordict):
                action = tensordict.get("action")
                obs = (tensordict.get("observation") + action) / action.norm()
                return TensorDict(
                    {
                        "reward": action.sum().unsqueeze(0),
                        **self.full_done_spec.zero(),
                        "observation": obs,
                    },
                    batch_size=[],
                )

        torch.manual_seed(0)
        policy = Actor(torch.nn.Linear(3, 3, device=device))
        env0 = DifferentiableEnv(device=device)
        seed = env0.set_seed(0)
        env1 = DifferentiableEnv(device=device)
        env1.set_seed(seed)
        r0 = env0.rollout(10, policy)
        r1 = env1.rollout(10, policy)
        r = torch.stack([r0, r1])
        g = torch.autograd.grad(r["next", "reward"].sum(), policy.parameters())

        def make_env(seed, device=device):
            env = DifferentiableEnv(device=device)
            env.set_seed(seed)
            return env

        serial_env = SerialEnv(
            2,
            [
                functools.partial(make_env, seed=0),
                functools.partial(make_env, seed=seed),
            ],
            device=device,
            share_individual_td=share_individual_td,
        )
        if share_individual_td:
            r_serial = serial_env.rollout(10, policy)
        else:
            with pytest.raises(
                RuntimeError, match="Cannot update a view of a tensordict"
            ):
                r_serial = serial_env.rollout(10, policy)
            return

        g_serial = torch.autograd.grad(
            r_serial["next", "reward"].sum(), policy.parameters()
        )
        torch.testing.assert_close(g, g_serial)

        p_env = maybe_fork_ParallelEnv(
            2,
            [
                functools.partial(make_env, seed=0),
                functools.partial(make_env, seed=seed),
            ],
            device=device,
        )
        try:
            r_parallel = p_env.rollout(10, policy)
            assert not r_parallel.exclude("action").requires_grad
        finally:
            p_env.close()
            del p_env

    @pytest.mark.parametrize("env_type", [CountingEnv, EnvWithMetadata])
    def test_auto_spec(self, env_type):
        if env_type is EnvWithMetadata:
            obs_vals = ["tensor", "non_tensor"]
        else:
            obs_vals = "observation"
        env = env_type()
        td = env.reset()

        policy = lambda td, action_spec=env.full_action_spec.clone(): td.update(
            action_spec.rand()
        )

        env.full_observation_spec = Composite(
            shape=env.full_observation_spec.shape,
            device=env.full_observation_spec.device,
        )
        env.full_action_spec = Composite(
            action=Unbounded((0,)),
            shape=env.full_action_spec.shape,
            device=env.full_action_spec.device,
        )
        env.full_reward_spec = Composite(
            shape=env.full_reward_spec.shape, device=env.full_reward_spec.device
        )
        env.full_done_spec = Composite(
            shape=env.full_done_spec.shape, device=env.full_done_spec.device
        )
        env.full_state_spec = Composite(
            shape=env.full_state_spec.shape, device=env.full_state_spec.device
        )
        env.auto_specs_(policy, tensordict=td.copy(), observation_key=obs_vals)
        env.check_env_specs(tensordict=td.copy())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device found.")
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_auto_cast_to_device(self, break_when_any_done):
        gc.collect()

        env = ContinuousActionVecMockEnv(device="cpu")
        policy = Actor(
            nn.Linear(
                env.observation_spec["observation"].shape[-1],
                env.full_action_spec[env.action_key].shape[-1],
                device="cuda:0",
            ),
            in_keys=["observation"],
        )
        with pytest.raises(RuntimeError):
            env.rollout(10, policy)
        torch.manual_seed(0)
        env.set_seed(0)
        rollout0 = env.rollout(
            100,
            policy,
            auto_cast_to_device=True,
            break_when_any_done=break_when_any_done,
        )
        torch.manual_seed(0)
        env.set_seed(0)
        rollout1 = env.rollout(
            100,
            policy.cpu(),
            auto_cast_to_device=False,
            break_when_any_done=break_when_any_done,
        )
        assert_allclose_td(rollout0, rollout1)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED, CARTPOLE_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [1, 4])
    def test_env_seed(self, env_name, frame_skip, seed=0):
        env_name = env_name()
        env = GymEnv(env_name, frame_skip=frame_skip)
        action = env.full_action_spec[env.action_key].rand()

        env.set_seed(seed)
        td0a = env.reset()
        td1a = env.step(td0a.clone().set("action", action))

        env.set_seed(seed)
        td0b = env.fake_tensordict()
        td0b = env.reset(tensordict=td0b)
        td1b = env.step(td0b.exclude("next").clone().set("action", action))

        assert_allclose_td(td0a, td0b.select(*td0a.keys()))
        assert_allclose_td(td1a, td1b)

        env.set_seed(
            seed=seed + 10,
        )
        td0c = env.reset()
        td1c = env.step(td0c.clone().set("action", action))

        with pytest.raises(AssertionError):
            assert_allclose_td(td0a, td0c.select(*td0a.keys()))
        with pytest.raises(AssertionError):
            assert_allclose_td(td1a, td1c)
        env.close()

    # Check that the "terminated" key is filled in automatically if only the "done"
    # key is provided in `_step`.
    def test_done_key_completion_done(self):
        class DoneEnv(CountingEnv):
            def _step(
                self,
                tensordict: TensorDictBase,
            ) -> TensorDictBase:
                self.count += 1
                tensordict = TensorDict(
                    source={
                        "observation": self.count.clone(),
                        "done": self.count > self.max_steps,
                        "reward": torch.zeros_like(self.count, dtype=torch.float),
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                return tensordict

        env = DoneEnv(max_steps=torch.tensor([[0], [1]]), batch_size=(2,))
        td = env.reset()
        env.rand_action(td)
        td = env.step(td)
        assert torch.equal(td[("next", "done")], torch.tensor([[True], [False]]))
        assert torch.equal(td[("next", "terminated")], torch.tensor([[True], [False]]))

    # Check that the "done" key is filled in automatically if only the "terminated"
    # key is provided in `_step`.
    def test_done_key_completion_terminated(self):
        class TerminatedEnv(CountingEnv):
            def _step(
                self,
                tensordict: TensorDictBase,
            ) -> TensorDictBase:
                self.count += 1
                tensordict = TensorDict(
                    source={
                        "observation": self.count.clone(),
                        "terminated": self.count > self.max_steps,
                        "reward": torch.zeros_like(self.count, dtype=torch.float),
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                return tensordict

        env = TerminatedEnv(max_steps=torch.tensor([[0], [1]]), batch_size=(2,))
        td = env.reset()
        env.rand_action(td)
        td = env.step(td)
        assert torch.equal(td[("next", "done")], torch.tensor([[True], [False]]))
        assert torch.equal(td[("next", "terminated")], torch.tensor([[True], [False]]))

    @pytest.mark.parametrize("batch_size", [(), (2,), (32, 5)])
    def test_env_base_reset_flag(self, batch_size, max_steps=3):
        torch.manual_seed(0)
        env = CountingEnv(max_steps=max_steps, batch_size=batch_size)
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
            assert (td["next", "observation"] == i + 1).all()

        td = env.step(
            TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        )
        assert (td["next", "done"] == 1).all()
        assert (td["next", "observation"] == max_steps + 1).all()

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

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_seed(self):
        torch.manual_seed(0)
        env1 = GymEnv(PENDULUM_VERSIONED())
        env1.set_seed(0)
        state0_1 = env1.reset()
        state1_1 = env1.step(state0_1.set("action", env1.action_spec.rand()))

        torch.manual_seed(0)
        env2 = GymEnv(PENDULUM_VERSIONED())
        env2.set_seed(0)
        state0_2 = env2.reset()
        state1_2 = env2.step(state0_2.set("action", env2.action_spec.rand()))

        assert_allclose_td(state0_1, state0_2)
        assert_allclose_td(state1_1, state1_2)

        env1.set_seed(0)
        torch.manual_seed(0)
        rollout1 = env1.rollout(max_steps=30)

        env2.set_seed(0)
        torch.manual_seed(0)
        rollout2 = env2.rollout(max_steps=30)

        torch.testing.assert_close(
            rollout1["observation"][1:], rollout1[("next", "observation")][:-1]
        )
        torch.testing.assert_close(
            rollout2["observation"][1:], rollout2[("next", "observation")][:-1]
        )
        torch.testing.assert_close(rollout1["observation"], rollout2["observation"])

    @pytest.mark.parametrize("device", get_default_devices())
    def test_batch_locked(self, device):
        env = MockBatchedLockedEnv(device)
        assert env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            env.batch_locked = False
        td = env.reset()
        td["action"] = env.full_action_spec[env.action_key].rand()
        td_expanded = td.expand(2).clone()
        _ = env.step(td)

        with pytest.raises(
            RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
        ):
            env.step(td_expanded)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_batch_unlocked(self, device):
        env = MockBatchedUnLockedEnv(device)
        assert not env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            env.batch_locked = False
        td = env.reset()
        td["action"] = env.full_action_spec[env.action_key].rand()
        td_expanded = td.expand(2).clone()
        td = env.step(td)

        env.step(td_expanded)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_batch_unlocked_with_batch_size(self, device):
        env = MockBatchedUnLockedEnv(device, batch_size=torch.Size([2]))
        assert not env.batch_locked

        with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
            env.batch_locked = False

        td = env.reset()
        td["action"] = env.full_action_spec[env.action_key].rand()
        td_expanded = td.expand(2, 2).reshape(-1).to_tensordict()
        td = env.step(td)

        with pytest.raises(RuntimeError, match="Expected a tensordict with shape"):
            env.step(td_expanded)

    # We should have a way to cache the values of the env
    # but as noted in torchrl.envs.common._cache_value, this is unsafe unless the specs
    # are carefully locked (which would be bc-breaking).
    # def test_env_cache(self):
    #     env = CountingEnv()
    #     for _ in range(2):
    #         env.reward_keys
    #         assert "reward_keys" in env._cache
    #         env.action_keys
    #         assert "action_keys" in env._cache
    #         env.rollout(3)
    #         assert "_step_mdp" in env._cache
    #         env.observation_spec = env.observation_spec.clone()
    #         assert not env._cache

    @pytest.mark.parametrize("storing_device", get_default_devices())
    def test_storing_device(self, storing_device):
        """Ensure rollout data tensors are moved to the requested storing_device."""
        env = ContinuousActionVecMockEnv(device="cpu")

        td = env.rollout(
            10,
            storing_device=torch.device(storing_device)
            if storing_device is not None
            else None,
        )

        expected_device = (
            torch.device(storing_device) if storing_device is not None else env.device
        )

        assert td.device == expected_device

        for _, item in td.items(True, True):
            if isinstance(item, torch.Tensor):
                assert item.device == expected_device


class TestRollout:
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED, PONG_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [1, 4])
    def test_rollout(self, env_name, frame_skip, seed=0):
        if env_name is PONG_VERSIONED and not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        if env_name is PONG_VERSIONED and version.parse(
            gym_backend().__version__
        ) < version.parse("0.19"):
            # Then 100 steps in pong are not sufficient to detect a difference
            pytest.skip("can't detect difference in gym rollout with this gym version.")

        env_name = env_name()
        env = GymEnv(env_name, frame_skip=frame_skip)

        torch.manual_seed(seed)
        np.random.seed(seed)
        env.set_seed(seed)
        env.reset()
        rollout1 = env.rollout(max_steps=100)
        assert rollout1.names[-1] == "time"

        torch.manual_seed(seed)
        np.random.seed(seed)
        env.set_seed(seed)
        env.reset()
        rollout2 = env.rollout(max_steps=100)
        assert rollout2.names[-1] == "time"

        assert_allclose_td(rollout1, rollout2)

        torch.manual_seed(seed)
        env.set_seed(seed + 10)
        env.reset()
        rollout3 = env.rollout(max_steps=100)
        with pytest.raises(AssertionError):
            assert_allclose_td(rollout1, rollout3)
        env.close()

    def test_rollout_set_truncated(self):
        env = ContinuousActionVecMockEnv()
        with pytest.raises(RuntimeError, match="set_truncated was set to True"):
            env.rollout(max_steps=10, set_truncated=True, break_when_any_done=False)
        env.add_truncated_keys()
        r = env.rollout(max_steps=10, set_truncated=True, break_when_any_done=False)
        assert r.shape == torch.Size([10])
        assert r[..., -1]["next", "truncated"].all()
        assert r[..., -1]["next", "done"].all()

    @pytest.mark.parametrize("max_steps", [1, 5])
    def test_rollouts_chaining(self, max_steps, batch_size=(4,), epochs=4):
        # CountingEnv is done at max_steps + 1, so to emulate it being done at max_steps, we feed max_steps=max_steps - 1
        env = CountingEnv(max_steps=max_steps - 1, batch_size=batch_size)
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key], action_key=env.action_key
        )

        input_td = env.reset()
        for _ in range(epochs):
            rollout_td = env.rollout(
                max_steps=max_steps,
                policy=policy,
                auto_reset=False,
                break_when_any_done=False,
                tensordict=input_td,
            )
            assert (env.count == max_steps).all()
            input_td = step_mdp(
                rollout_td[..., -1],
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=env.reward_keys,
                action_keys=env.action_keys,
                done_keys=env.done_keys,
            )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_rollout_predictability(self, device):
        env = MockSerialEnv(device=device)
        env.set_seed(100)
        first = 100 % 17
        policy = Actor(torch.nn.Linear(1, 1, bias=False)).to(device)
        for p in policy.parameters():
            p.data.fill_(1.0)
        td_out = env.rollout(policy=policy, max_steps=200)
        assert (
            torch.arange(first, first + 100, device=device)
            == td_out.get("observation").squeeze()
        ).all()
        assert (
            torch.arange(first + 1, first + 101, device=device)
            == td_out.get(("next", "observation")).squeeze()
        ).all()
        assert (
            torch.arange(first + 1, first + 101, device=device)
            == td_out.get(("next", "reward")).squeeze()
        ).all()
        assert (
            torch.arange(first, first + 100, device=device)
            == td_out.get("action").squeeze()
        ).all()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [1])
    @pytest.mark.parametrize("truncated_key", ["truncated", "done"])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_rollout_reset(
        self,
        env_name,
        frame_skip,
        parallel,
        truncated_key,
        maybe_fork_ParallelEnv,
        seed=0,
    ):
        env_name = env_name()
        envs = []
        for horizon in [20, 30, 40]:
            envs.append(
                lambda horizon=horizon: TransformedEnv(
                    GymEnv(env_name, frame_skip=frame_skip),
                    StepCounter(horizon, truncated_key=truncated_key),
                )
            )
        if parallel:
            env = maybe_fork_ParallelEnv(3, envs)
        else:
            env = SerialEnv(3, envs)
        env.set_seed(100)
        out = env.rollout(100, break_when_any_done=False)
        assert out.names[-1] == "time"
        assert out.shape == torch.Size([3, 100])
        assert (
            out[..., -1]["step_count"].squeeze().cpu() == torch.tensor([19, 9, 19])
        ).all()
        assert (
            out[..., -1]["next", "step_count"].squeeze().cpu()
            == torch.tensor([20, 10, 20])
        ).all()
        assert (
            out["next", truncated_key].squeeze().sum(-1) == torch.tensor([5, 3, 2])
        ).all()

    @pytest.mark.parametrize(
        "break_when_any_done,break_when_all_done",
        [[True, False], [False, True], [False, False]],
    )
    @pytest.mark.parametrize("n_envs,serial", [[1, None], [4, True], [4, False]])
    def test_rollout_outplace_policy(
        self, n_envs, serial, break_when_any_done, break_when_all_done
    ):
        def policy_inplace(td):
            td.set("action", torch.ones(td.shape + (1,)))
            return td

        def policy_outplace(td):
            return td.empty().set("action", torch.ones(td.shape + (1,)))

        if n_envs == 1:
            env = CountingEnv(10)
        elif serial:
            env = SerialEnv(
                n_envs,
                [partial(CountingEnv, 10 + i) for i in range(n_envs)],
            )
        else:
            env = ParallelEnv(
                n_envs,
                [partial(CountingEnv, 10 + i) for i in range(n_envs)],
                mp_start_method=mp_ctx,
            )
        r_inplace = env.rollout(
            40,
            policy_inplace,
            break_when_all_done=break_when_all_done,
            break_when_any_done=break_when_any_done,
        )
        r_outplace = env.rollout(
            40,
            policy_outplace,
            break_when_all_done=break_when_all_done,
            break_when_any_done=break_when_any_done,
        )
        if break_when_any_done:
            assert r_outplace.shape[-1:] == (11,)
        elif break_when_all_done:
            if n_envs > 1:
                assert r_outplace.shape[-1:] == (14,)
            else:
                assert r_outplace.shape[-1:] == (11,)
        else:
            assert r_outplace.shape[-1:] == (40,)
        assert_allclose_td(r_inplace, r_outplace)
