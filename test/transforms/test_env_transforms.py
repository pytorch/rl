# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

import tensordict.tensordict
import torch

from _transforms_common import _has_ale, _has_gymnasium, TransformBase
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.collectors import MultiSyncCollector
from torchrl.data import (
    Binary,
    Categorical,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    Unbounded,
)
from torchrl.envs import (
    BurnInTransform,
    Compose,
    DMControlEnv,
    EndOfLifeTransform,
    EnvBase,
    EnvCreator,
    ExpandAs,
    FrameSkipTransform,
    InitTracker,
    NoopResetEnv,
    ParallelEnv,
    RandomTruncationTransform,
    RenameTransform,
    SerialEnv,
    StepCounter,
    TargetReturn,
    TrajCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import _has_dm_control
from torchrl.envs.libs.gym import _has_gym, GymEnv, set_gym_backend
from torchrl.envs.transforms.transforms import FORWARD_NOT_IMPLEMENTED
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.modules import GRUModule, LSTMModule

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingBatchedEnv,
    CountingEnv,
    CountingEnvCountPolicy,
    DiscreteActionConvMockEnvNumpy,
    IncrementingEnv,
    MultiAgentCountingEnv,
    NestedCountingEnv,
)


class TestStepCounter(TransformBase):
    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_step_count_gym(self):
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), StepCounter(max_steps=30))
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_step_count_gym_doublecount(self):
        # tests that 2 truncations can be used together
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(
                StepCounter(max_steps=2),
                StepCounter(max_steps=3),  # this one will be ignored
            ),
        )
        r = env.rollout(10, break_when_any_done=False)
        assert (
            r.get(("next", "truncated")).squeeze().nonzero().squeeze(-1)
            == torch.arange(1, 10, 2)
        ).all()

    @pytest.mark.skipif(not _has_dm_control, reason="no dm_control detected")
    def test_step_count_dmc(self):
        env = TransformedEnv(DMControlEnv("cheetah", "run"), StepCounter(max_steps=30))
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_stepcount_batching(self, batched_class, break_when_any_done):
        from torchrl.testing import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            StepCounter(max_steps=10),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(30, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()), StepCounter(max_steps=10)
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(30, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    @pytest.mark.parametrize("update_done", [False, True])
    @pytest.mark.parametrize("max_steps", [10, None])
    def test_single_trans_env_check(self, update_done, max_steps):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            StepCounter(max_steps=max_steps, update_done=update_done),
        )
        check_env_specs(env)
        r = env.rollout(100, break_when_any_done=False)
        if update_done and max_steps:
            assert r["next", "done"][r["next", "truncated"]].all()
        elif max_steps:
            assert not r["next", "done"][r["next", "truncated"]].all()
        else:
            assert "truncated" not in r.keys()
            assert ("next", "truncated") not in r.keys(True)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(10))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), StepCounter(10)
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), StepCounter(10))
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="Gym not found")
    def test_transform_env(self):
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), StepCounter(10))
        td = env.rollout(100, break_when_any_done=False)
        assert td["step_count"].max() == 9
        assert td.shape[-1] == 100

    @pytest.mark.parametrize("step_key", ["step_count", "other-key"])
    @pytest.mark.parametrize("max_steps", [None, 10])
    @pytest.mark.parametrize("nested_done", [True, False])
    def test_nested(
        self, step_key, nested_done, max_steps, batch_size=(32, 2), rollout_length=15
    ):
        env = NestedCountingEnv(
            max_steps=20, nest_done=nested_done, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key], action_key=env.action_key
        )
        transformed_env = TransformedEnv(
            env,
            StepCounter(
                max_steps=max_steps,
                step_count_key=step_key,
            ),
        )
        step_key = transformed_env.transform.step_count_keys[0]
        td = transformed_env.rollout(
            rollout_length, policy=policy, break_when_any_done=False
        )
        if nested_done:
            step = td[step_key][0, 0, :, 0, 0].clone()
            last_step = td[step_key][:, :, -1, :, :].clone()
        else:
            step = td[step_key][0, 0, :, 0].clone()
            last_step = td[step_key][:, :, -1, :].clone()
        if max_steps is None:
            assert step.eq(torch.arange(rollout_length)).all()
        else:
            assert step[:max_steps].eq(torch.arange(max_steps)).all()
            assert step[max_steps:].eq(torch.arange(rollout_length - max_steps)).all()

        if nested_done:
            for done_key in env.done_keys:
                reset_key = (*done_key[:-1], "_reset")
                _reset = env.full_done_spec[done_key].rand()
                break
        else:
            reset_key = "_reset"
            _reset = env.full_done_spec["done"].rand()
        td_reset = transformed_env.reset(
            TensorDict(
                {reset_key: _reset, step_key: last_step},
                batch_size=env.batch_size,
                device=env.device,
            )
        )
        assert (td_reset[step_key][_reset] == 0).all()
        assert (td_reset[step_key][~_reset] == last_step[~_reset]).all()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        transform = StepCounter(10)
        rb = rbclass(storage=LazyTensorStorage(20))
        td = TensorDict({"a": torch.randn(10)}, [10])
        rb.extend(td)
        rb.append_transform(transform)
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            rb.sample(5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    @pytest.mark.parametrize("max_steps", [None, 1, 5, 50])
    @pytest.mark.parametrize("reset_workers", [True, False])
    def test_transform_compose(self, max_steps, device, batch, reset_workers):
        torch.manual_seed(0)
        step_counter = Compose(StepCounter(max_steps))
        done = torch.zeros(*batch, 1, dtype=torch.bool)
        td = TensorDict({"done": done, ("next", "done"): done}, batch, device=device)
        _reset = torch.zeros((), dtype=torch.bool, device=device)
        while not _reset.any() and reset_workers:
            _reset = torch.randn(done.shape, device=device) < 0
            td.set("_reset", _reset)
            td.set("done", _reset)
            td.set("terminated", _reset)
            td.set(("next", "terminated"), done)
            td.set(("next", "done"), done)
        td.set("step_count", torch.zeros(*batch, 1, dtype=torch.int))
        step_counter[0]._step_count_keys = ["step_count"]
        step_counter[0]._terminated_keys = ["terminated"]
        step_counter[0]._truncated_keys = ["truncated"]
        step_counter[0]._reset_keys = ["_reset"]
        step_counter[0]._done_keys = ["done"]
        td_reset = td.empty()
        td_reset = step_counter._reset(td, td_reset)
        assert not torch.all(td_reset.get("step_count"))
        i = 0
        td_next = td.get("next")
        td = td_reset
        while max_steps is None or i < max_steps:
            td_next = step_counter._step(td, td_next)
            td.set("next", td_next)

            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            td["next", "terminated"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td_reset = td.empty()
        if reset_workers:
            td.set("_reset", _reset)
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), _reset) == 0
            )
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), ~_reset) == i
            )
        else:
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(td_reset.get("step_count") == 0)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for StepCounter")

    def test_transform_model(self):
        transform = StepCounter(10)
        _ = TensorDict({"a": torch.randn(10)}, [10])
        model = nn.Sequential(transform, nn.Identity())
        with pytest.raises(
            NotImplementedError, match="StepCounter cannot be called independently"
        ):
            model(5)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    @pytest.mark.parametrize("max_steps", [None, 1, 5, 50])
    @pytest.mark.parametrize("reset_workers", [True, False])
    def test_transform_no_env(self, max_steps, device, batch, reset_workers):
        torch.manual_seed(0)
        step_counter = StepCounter(max_steps)
        done = torch.zeros(*batch, 1, dtype=torch.bool)
        td = TensorDict({"done": done, ("next", "done"): done}, batch, device=device)
        _reset = torch.zeros((), dtype=torch.bool, device=device)
        while not _reset.any() and reset_workers:
            _reset = torch.randn(done.shape, device=device) < 0
            td.set("_reset", _reset)
            td.set("terminated", _reset)
            td.set(("next", "terminated"), done)
            td.set("done", _reset)
            td.set(("next", "done"), done)
        td.set("step_count", torch.zeros(*batch, 1, dtype=torch.int))
        step_counter._step_count_keys = ["step_count"]
        step_counter._done_keys = ["done"]
        step_counter._terminated_keys = ["terminated"]
        step_counter._truncated_keys = ["truncated"]
        step_counter._reset_keys = ["_reset"]
        step_counter._completed_keys = ["completed"]

        td_reset = td.empty()
        td_reset = step_counter._reset(td, td_reset)
        assert not torch.all(td_reset.get("step_count"))
        i = 0
        td_next = td.get("next")
        td = td_reset
        while max_steps is None or i < max_steps:
            td_next = step_counter._step(td, td_next)
            td.set("next", td_next)

            i += 1
            assert torch.all(td.get(("next", "step_count")) == i), (
                td.get(("next", "step_count")),
                i,
            )
            td = step_mdp(td)
            td["next", "done"] = done
            td["next", "terminated"] = done
            if max_steps is None:
                break

        if max_steps is not None:
            assert torch.all(td.get("step_count") == max_steps)
            assert torch.all(td.get("truncated"))
        td_reset = td.empty()
        if reset_workers:
            td.set("_reset", _reset)
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), _reset) == 0
            )
            assert torch.all(
                torch.masked_select(td_reset.get("step_count"), ~_reset) == i
            )
        else:
            td_reset = step_counter._reset(td, td_reset)
            assert torch.all(td_reset.get("step_count") == 0)

    def test_step_counter_observation_spec(self):
        transformed_env = TransformedEnv(ContinuousActionVecMockEnv(), StepCounter(50))
        check_env_specs(transformed_env)
        transformed_env.close()

    def test_stepcounter_reset_heterogeneous_observation_spec(self):
        class HeterogeneousEnv(EnvBase):
            def __init__(self) -> None:
                super().__init__(batch_size=torch.Size([2]))
                self.observation_spec = Composite(
                    {
                        "shaped_obs": Unbounded(shape=torch.Size([2, 5, 5])),
                        "binary_obs": Binary(n=2, shape=torch.Size([2])),
                    },
                    shape=torch.Size([2]),
                )
                self.action_spec = Categorical(2, shape=torch.Size([2, 1]))
                self.reward_spec = Unbounded(shape=torch.Size([2, 1]))
                self.done_spec = Composite(
                    {
                        "done": Categorical(
                            2, shape=torch.Size([2, 1]), dtype=torch.bool
                        ),
                        "terminated": Categorical(
                            2, shape=torch.Size([2, 1]), dtype=torch.bool
                        ),
                        "truncated": Categorical(
                            2, shape=torch.Size([2, 1]), dtype=torch.bool
                        ),
                    },
                    shape=torch.Size([2]),
                )

            def _reset(
                self, tensordict: TensorDictBase | None = None, **kwargs
            ) -> TensorDictBase:
                return TensorDict(
                    {
                        "shaped_obs": torch.zeros(2, 5, 5),
                        "binary_obs": torch.zeros(2, dtype=torch.int8),
                    },
                    batch_size=[2],
                )

            def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
                raise NotImplementedError

            def _set_seed(self, seed: int | None) -> None:
                return None

        env = TransformedEnv(HeterogeneousEnv(), StepCounter(max_steps=10))
        td = env.reset()

        assert td["step_count"].shape == torch.Size([2, 1])
        assert td["step_count"].eq(0).all()

    def test_stepcounter_nested(self):
        # checks that step_count_keys are only created at root level when both exist
        # (nested keys are filtered out to avoid shape mismatches)
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=True, nest_done=True), StepCounter()
        )
        assert len(env.transform.step_count_keys) == 1
        assert env.transform.step_count_keys[0] == "step_count"
        # all_truncated_keys should include both root and nested
        assert len(env.transform.all_truncated_keys) == 2
        assert "truncated" in env.transform.all_truncated_keys
        assert ("data", "truncated") in env.transform.all_truncated_keys
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=False, nest_done=True), StepCounter()
        )
        assert len(env.transform.step_count_keys) == 1
        assert env.transform.step_count_keys[0] == ("data", "step_count")

    def test_stepcounter_marl_truncation(self):
        # Regression test for https://github.com/pytorch/rl/issues/3400
        # In MARL envs with both root and nested done keys, StepCounter should
        # propagate done to nested levels when max_steps is reached.
        max_steps = 3
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=True, nest_done=True, max_steps=10),
            StepCounter(max_steps=max_steps),
        )

        # step_count is only tracked at root level (to avoid shape mismatches)
        assert "step_count" in env.transform.step_count_keys
        assert ("data", "step_count") not in env.transform.step_count_keys

        # done should be propagated to nested keys via all_done_keys
        assert "done" in env.transform.all_done_keys
        assert ("data", "done") in env.transform.all_done_keys

        # Roll out past max_steps
        td = env.rollout(max_steps + 1)

        # Check that done is set at BOTH root and nested levels when max_steps is reached
        root_done = td["next", "done"]
        nested_done = td["next", "data", "done"]

        # At step max_steps, both should be True (due to truncation)
        assert root_done[max_steps - 1].all(), "Root done should be True at max_steps"
        assert nested_done[
            max_steps - 1
        ].all(), "Nested (agent-level) done should also be True at max_steps"

        # Before max_steps, root done should be False (nested may be True due to agent deaths)
        assert not root_done[
            : max_steps - 1
        ].any(), "Root done should be False before max_steps"


class TestRandomTruncationTransform(TransformBase):
    def _make_transform(self):
        return Compose(
            StepCounter(),
            RandomTruncationTransform(prob=1.0, min_horizon=3, max_horizon=5),
        )

    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), self._make_transform())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), self._make_transform())

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), self._make_transform())

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), self._make_transform()
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            self._make_transform(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        transform = RandomTruncationTransform(prob=1.0, min_horizon=3, max_horizon=5)
        td = TensorDict(
            {
                "step_count": torch.tensor([[3]]),
                ("next", "step_count"): torch.tensor([[4]]),
                ("next", "done"): torch.tensor([[False]]),
                ("next", "truncated"): torch.tensor([[False]]),
            },
            [],
        )
        # Without initialization, _step is a no-op
        result = transform._step(td, td["next"])
        assert not result["done"].any()

    def test_transform_compose(self):
        transform = RandomTruncationTransform(prob=1.0, min_horizon=3, max_horizon=5)
        td = TensorDict(
            {
                "step_count": torch.zeros(4, 1, dtype=torch.int64),
                "done": torch.zeros(4, 1, dtype=torch.bool),
                ("next", "step_count"): torch.ones(4, 1, dtype=torch.int64),
                ("next", "done"): torch.zeros(4, 1, dtype=torch.bool),
                ("next", "truncated"): torch.zeros(4, 1, dtype=torch.bool),
            },
            [4],
        )
        result = transform._step(td, td["next"])
        assert "truncated" in result.keys()

    def test_transform_env(self):
        torch.manual_seed(0)
        max_horizon = 5
        env = TransformedEnv(
            CountingEnv(max_steps=100),
            Compose(
                StepCounter(),
                RandomTruncationTransform(
                    prob=1.0, min_horizon=1, max_horizon=max_horizon
                ),
            ),
        )
        rollout = env.rollout(200)
        step_counts = rollout["next", "step_count"]
        assert step_counts.max() <= max_horizon

    def test_transform_model(self):
        # RandomTruncationTransform doesn't modify observations, so a model
        # reading obs is unaffected.
        env = TransformedEnv(ContinuousActionVecMockEnv(), self._make_transform())
        td = env.reset()
        obs_dim = td["observation"].shape[-1]
        model = nn.Linear(obs_dim, 1)
        model(td["observation"])

    def test_transform_rb(self):
        transform = RandomTruncationTransform(prob=1.0, min_horizon=3, max_horizon=5)
        rb = ReplayBuffer(storage=LazyTensorStorage(20))
        td = TensorDict({"a": torch.randn(10)}, [10])
        rb.extend(td)
        rb.append_transform(transform)
        with pytest.raises(NotImplementedError):
            rb.sample(5)

    def test_transform_inverse(self):
        # RandomTruncationTransform is not invertible — nothing to test
        pass

    def test_basic_truncation(self):
        """Episodes are truncated before max_steps of the underlying env."""
        torch.manual_seed(0)
        max_horizon = 5
        env = TransformedEnv(
            CountingEnv(max_steps=100),
            Compose(
                StepCounter(),
                RandomTruncationTransform(
                    prob=1.0, min_horizon=1, max_horizon=max_horizon
                ),
            ),
        )
        rollout = env.rollout(200)
        step_counts = rollout["next", "step_count"]
        assert step_counts.max() <= max_horizon

    def test_first_reset_spreads_horizons(self):
        """First reset assigns diverse horizons to decorrelate envs."""
        torch.manual_seed(42)
        max_horizon = 100
        transform = RandomTruncationTransform(
            prob=0.0, min_horizon=50, max_horizon=max_horizon
        )
        n_envs = 64
        env = TransformedEnv(
            CountingBatchedEnv(
                max_steps=torch.full((n_envs,), 200), batch_size=[n_envs]
            ),
            Compose(StepCounter(), transform),
        )
        env.reset()
        assert transform._initialized
        assert transform._horizons is not None
        assert transform._horizons.shape == (n_envs, 1)
        assert transform._horizons.unique().numel() > 1
        assert (transform._horizons >= 1).all()
        assert (transform._horizons <= max_horizon).all()

    def test_first_episode_prob(self):
        """first_episode_prob=1.0 ensures first_episode flag is set after initial reset."""
        torch.manual_seed(0)
        transform = RandomTruncationTransform(
            prob=0.0,
            min_horizon=5,
            max_horizon=10,
            first_episode_prob=1.0,
        )
        n_envs = 32
        env = TransformedEnv(
            CountingBatchedEnv(
                max_steps=torch.full((n_envs,), 200), batch_size=[n_envs]
            ),
            Compose(StepCounter(), transform),
        )
        env.reset()
        assert transform._first_episode.all()

    def test_prob_zero_uses_max_horizon(self):
        """With prob=0 and first_episode_prob=0, subsequent resets use max_horizon."""
        torch.manual_seed(0)
        max_horizon = 10
        transform = RandomTruncationTransform(
            min_horizon=1,
            max_horizon=max_horizon,
            prob=0.0,
        )
        n_envs = 8
        env = TransformedEnv(
            CountingBatchedEnv(
                max_steps=torch.full((n_envs,), 200), batch_size=[n_envs]
            ),
            Compose(StepCounter(), transform),
        )
        # Roll out past the max_horizon so some envs reset
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key],
            action_key=env.action_key,
        )
        rollout = env.rollout(max_horizon + 5, policy=policy, break_when_any_done=False)
        # After first episode resets, horizons should be max_horizon
        # (prob=0 and first_episode_prob=0 both mean "keep full")
        reset_envs = ~transform._first_episode.squeeze(-1)
        if reset_envs.any():
            assert (transform._horizons[reset_envs] == max_horizon).all()

    def test_requires_step_counter(self):
        with pytest.raises(RuntimeError, match="requires a StepCounter"):
            TransformedEnv(
                CountingEnv(max_steps=100),
                RandomTruncationTransform(min_horizon=1, max_horizon=5),
            )

    def test_validation(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="prob must be in"):
            RandomTruncationTransform(min_horizon=1, max_horizon=5, prob=-0.1)
        with pytest.raises(ValueError, match="prob must be in"):
            RandomTruncationTransform(min_horizon=1, max_horizon=5, prob=1.5)
        with pytest.raises(ValueError, match="first_episode_prob must be in"):
            RandomTruncationTransform(
                min_horizon=1, max_horizon=5, first_episode_prob=-0.1
            )
        with pytest.raises(ValueError, match="min_horizon must be >= 1"):
            RandomTruncationTransform(min_horizon=0, max_horizon=5)
        with pytest.raises(ValueError, match="min_horizon.*must be <= max_horizon"):
            RandomTruncationTransform(min_horizon=100, max_horizon=50)
        with pytest.raises(ValueError, match="max_horizon must be >= 1"):
            RandomTruncationTransform(min_horizon=1, max_horizon=0)


class TestTrajCounter(TransformBase):
    def test_single_trans_env_check(self):
        torch.manual_seed(0)
        env = TransformedEnv(CountingEnv(max_steps=4), TrajCounter())
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())
        check_env_specs(env)

    @pytest.mark.parametrize("predefined", [True, False])
    def test_parallel_trans_env_check(self, predefined):
        if predefined:
            t = TrajCounter()
        else:
            t = None

        def make_env(max_steps=4, t=t):
            if t is None:
                t = TrajCounter()
            env = TransformedEnv(CountingEnv(max_steps=max_steps), t.clone())
            env.transform.transform_observation_spec(
                env.base_env.observation_spec.clone()
            )
            return env

        if predefined:
            penv = ParallelEnv(
                2,
                [EnvCreator(make_env, max_steps=4), EnvCreator(make_env, max_steps=5)],
                mp_start_method="spawn",
            )
        else:
            make_env_c0 = EnvCreator(make_env)
            make_env_c1 = make_env_c0.make_variant(max_steps=5)
            penv = ParallelEnv(
                2,
                [make_env_c0, make_env_c1],
                mp_start_method="spawn",
            )

        r = penv.rollout(100, break_when_any_done=False)
        s0 = set(r[0]["traj_count"].squeeze().tolist())
        s1 = set(r[1]["traj_count"].squeeze().tolist())
        assert len(s1.intersection(s0)) == 0

    @pytest.mark.parametrize("predefined", [True, False])
    def test_serial_trans_env_check(self, predefined):
        if predefined:
            t = TrajCounter()
        else:
            t = None

        def make_env(max_steps=4, t=t):
            if t is None:
                t = TrajCounter()
            else:
                t = t.clone()
            env = TransformedEnv(CountingEnv(max_steps=max_steps), t)
            env.transform.transform_observation_spec(
                env.base_env.observation_spec.clone()
            )
            return env

        if predefined:
            penv = SerialEnv(
                2,
                [EnvCreator(make_env, max_steps=4), EnvCreator(make_env, max_steps=5)],
            )
        else:
            make_env_c0 = EnvCreator(make_env)
            make_env_c1 = make_env_c0.make_variant(max_steps=5)
            penv = SerialEnv(
                2,
                [make_env_c0, make_env_c1],
            )

        r = penv.rollout(100, break_when_any_done=False)
        s0 = set(r[0]["traj_count"].squeeze().tolist())
        s1 = set(r[1]["traj_count"].squeeze().tolist())
        assert len(s1.intersection(s0)) == 0

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(
                2, [lambda: CountingEnv(max_steps=4), lambda: CountingEnv(max_steps=5)]
            ),
            TrajCounter(),
        )
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())
        r = env.rollout(
            100,
            lambda td: td.set("action", torch.ones(env.shape + (1,))),
            break_when_any_done=False,
        )
        check_env_specs(env)
        assert r["traj_count"].max() == 36

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(
                2, [lambda: CountingEnv(max_steps=4), lambda: CountingEnv(max_steps=5)]
            ),
            TrajCounter(),
        )
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())
        r = env.rollout(
            100,
            lambda td: td.set("action", torch.ones(env.shape + (1,))),
            break_when_any_done=False,
        )
        check_env_specs(env)
        assert r["traj_count"].max() == 36

    def test_transform_env(self):
        torch.manual_seed(0)
        env = TransformedEnv(CountingEnv(max_steps=4), TrajCounter())
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())
        r = env.rollout(100, lambda td: td.set("action", 1), break_when_any_done=False)
        assert r["traj_count"].max() == 19

    def test_nested(self):
        torch.manual_seed(0)
        env = TransformedEnv(
            CountingEnv(max_steps=4),
            Compose(
                RenameTransform("done", ("nested", "done"), create_copy=True),
                TrajCounter(out_key=(("nested"), (("traj_count",),))),
            ),
        )
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())
        r = env.rollout(100, lambda td: td.set("action", 1), break_when_any_done=False)
        assert r["nested", "traj_count"].max() == 19

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = TrajCounter()
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(t)
        td = (
            TensorDict(
                {("next", "observation"): torch.randn(3), "action": torch.randn(2)}, []
            )
            .expand(10)
            .contiguous()
        )
        rb.extend(td)
        with pytest.raises(
            RuntimeError,
            match="TrajCounter can only be called within an environment step or reset",
        ):
            td = rb.sample(10)

    @retry(AssertionError, tries=10, delay=0)
    def test_collector_match(self):
        torch.manual_seed(0)

        # The counter in the collector should match the one from the transform
        t = TrajCounter()

        def make_env(max_steps=4):
            env = TransformedEnv(CountingEnv(max_steps=max_steps), t.clone())
            env.transform.transform_observation_spec(
                env.base_env.observation_spec.clone()
            )
            return env

        collector = MultiSyncCollector(
            [EnvCreator(make_env, max_steps=5), EnvCreator(make_env, max_steps=4)],
            total_frames=32,
            frames_per_batch=8,
        )

        try:
            traj_ids_collector = []
            traj_ids_env = []
            for d in collector:
                traj_ids_collector.extend(d["collector", "traj_ids"].view(-1).tolist())
                traj_ids_env.extend(d["next", "traj_count"].view(-1).tolist())
            assert len(set(traj_ids_env)) == len(set(traj_ids_collector))
        finally:
            collector.shutdown()
            del collector

    def test_transform_compose(self):
        t = TrajCounter()
        t = nn.Sequential(t)
        td = (
            TensorDict(
                {("next", "observation"): torch.randn(3), "action": torch.randn(2)}, []
            )
            .expand(10)
            .contiguous()
        )

        with pytest.raises(
            RuntimeError,
            match="TrajCounter can only be called within an environment step or reset",
        ):
            td = t(td)

    def test_transform_inverse(self):
        pytest.skip("No inverse transform for TrajCounter")

    def test_transform_model(self):
        t = TrajCounter()
        td = (
            TensorDict(
                {("next", "observation"): torch.randn(3), "action": torch.randn(2)}, []
            )
            .expand(10)
            .contiguous()
        )

        with pytest.raises(
            RuntimeError,
            match="TrajCounter can only be called within an environment step or reset",
        ):
            td = t(td)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [[], [4], [6, 4]])
    def test_transform_no_env(self, device, batch):
        pytest.skip("TrajCounter cannot be called without env")


class TestInitTracker(TransformBase):
    @pytest.mark.skipif(not _has_gym, reason="no gym detected")
    def test_init_gym(
        self,
    ):
        env = TransformedEnv(
            GymEnv(PENDULUM_VERSIONED()),
            Compose(StepCounter(max_steps=30), InitTracker()),
        )
        env.rollout(1000)
        check_env_specs(env)

    @pytest.mark.skipif(not _has_dm_control, reason="no dm_control detected")
    def test_init_dmc(self):
        env = TransformedEnv(
            DMControlEnv("cheetah", "run"),
            Compose(StepCounter(max_steps=30), InitTracker()),
        )
        env.rollout(1000)
        check_env_specs(env)

    def test_single_trans_env_check(self):
        env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
        env = TransformedEnv(env, InitTracker())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            env = TransformedEnv(env, InitTracker())
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            env = TransformedEnv(env, InitTracker())
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = SerialEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = CountingBatchedEnv(max_steps=torch.tensor([4, 5]), batch_size=[2])
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        env = TransformedEnv(env, InitTracker())
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        with pytest.raises(ValueError, match="init_key can only be of type str"):
            InitTracker(init_key=("some", "nested"))
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            InitTracker()(None)

    def test_transform_compose(self):
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            Compose(InitTracker())(None)

    def test_transform_env(self):
        policy = lambda tensordict: tensordict.set(
            "action", torch.ones(tensordict.shape, dtype=torch.int32)
        )
        env = CountingBatchedEnv(max_steps=torch.tensor([3, 4]), batch_size=[2])
        env = TransformedEnv(env, InitTracker())
        r = env.rollout(100, policy, break_when_any_done=False)
        assert (r["is_init"].view(r.batch_size).sum(-1) == torch.tensor([25, 20])).all()

    def test_transform_model(self):
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            td = TensorDict()
            chain = nn.Sequential(InitTracker())
            chain(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        batch = [1]
        device = "cpu"
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(InitTracker())
        reward = torch.randn(*batch, 1, device=device)
        misc = torch.randn(*batch, 1, device=device)
        td = TensorDict(
            {"misc": misc, "reward": reward},
            batch,
            device=device,
        )
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="InitTracker cannot be executed without a parent"
        ):
            _ = rb.sample(20)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for InitTracker")

    @pytest.mark.parametrize("init_key", ["is_init", "loool"])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("max_steps", [5])
    def test_nested(
        self,
        nested_done,
        max_steps,
        init_key,
        batch_size=(32, 2),
        rollout_length=9,
    ):
        env = NestedCountingEnv(
            max_steps=max_steps, nest_done=nested_done, batch_size=batch_size
        )
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key], action_key=env.action_key
        )
        transformed_env = TransformedEnv(
            env,
            InitTracker(init_key=init_key),
        )
        init_key = transformed_env.transform.init_keys[0]
        td = transformed_env.rollout(
            rollout_length, policy=policy, break_when_any_done=False
        )
        if nested_done:
            is_init = td[init_key][0, 0, :, 0, 0].clone()
        else:
            is_init = td[init_key][0, 0, :, 0].clone()
        if max_steps == 20:
            assert torch.all(is_init[0] == 1)
            assert torch.all(is_init[1:] == 0)
        else:
            assert torch.all(is_init[0] == 1)
            assert torch.all(is_init[1 : max_steps + 1] == 0)
            assert torch.all(is_init[max_steps + 1] == 1)
            assert torch.all(is_init[max_steps + 2 :] == 0)

        td_reset = TensorDict(
            rand_reset(transformed_env),
            batch_size=env.batch_size,
            device=env.device,
        )
        if nested_done:
            reset = td_reset["data", "_reset"]
        else:
            reset = td_reset["_reset"]

        td_reset = transformed_env.reset(td_reset)
        assert (td_reset[init_key] == reset).all()

    def test_inittracker_ignore(self):
        # checks that init keys respect the convention that nested dones should
        # be ignored if there is a done in a root td
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=True, nest_done=True), InitTracker()
        )
        assert len(env.transform.init_keys) == 1
        assert env.transform.init_keys[0] == "is_init"
        env = TransformedEnv(
            NestedCountingEnv(has_root_done=False, nest_done=True), InitTracker()
        )
        assert len(env.transform.init_keys) == 1
        assert env.transform.init_keys[0] == ("data", "is_init")


class TestFrameSkipTransform(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
            return env

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            env = TransformedEnv(ContinuousActionVecMockEnv(), FrameSkipTransform(2))
            return env

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv), FrameSkipTransform(2)
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        t = FrameSkipTransform(2)
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict, tensordict.get("next"))

    def test_transform_compose(self):
        t = Compose(FrameSkipTransform(2))
        tensordict = TensorDict({"next": {}}, [])
        with pytest.raises(
            RuntimeError, match="parent not found for FrameSkipTransform"
        ):
            t._step(tensordict, tensordict.get("next"))

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    @pytest.mark.parametrize("skip", [-1, 1, 2, 3])
    def test_transform_env(self, skip):
        """Tests that the built-in frame_skip and the transform lead to the same results."""
        torch.manual_seed(0)
        if skip < 0:
            with pytest.raises(
                ValueError,
                match="frame_skip should have a value greater or equal to one",
            ):
                FrameSkipTransform(skip)
            return
        else:
            fs = FrameSkipTransform(skip)
        base_env = GymEnv(PENDULUM_VERSIONED(), frame_skip=skip)
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), fs)
        base_env.set_seed(0)
        env.base_env.set_seed(0)
        td1 = base_env.reset()
        td2 = env.reset()
        for key in td1.keys():
            torch.testing.assert_close(td1[key], td2[key])
        for i in range(10):
            td1 = base_env.step(tensordicts[i].clone()).flatten_keys()
            td2 = env.step(tensordicts[i].clone()).flatten_keys()
            for key in td1.keys():
                torch.testing.assert_close(td1[key], td2[key])

    def test_nested(self, skip=4):
        env = NestedCountingEnv(max_steps=20)
        policy = CountingEnvCountPolicy(
            action_spec=env.full_action_spec[env.action_key], action_key=env.action_key
        )
        trnasformed_env = TransformedEnv(env, FrameSkipTransform(frame_skip=skip))
        td = trnasformed_env.rollout(2, policy=policy)
        (td[0] == 0).all()
        (td[1] == skip).all()

    def test_transform_model(self):
        t = FrameSkipTransform(2)
        t = nn.Sequential(t, nn.Identity())
        tensordict = TensorDict()
        with pytest.raises(
            RuntimeError,
            match="FrameSkipTransform can only be used when appended to a transformed env",
        ):
            t(tensordict)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = FrameSkipTransform(2)
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        tensordict = TensorDict({"a": torch.zeros(10)}, [10])
        rb.extend(tensordict)
        with pytest.raises(
            RuntimeError,
            match="FrameSkipTransform can only be used when appended to a transformed env",
        ):
            rb.sample(10)

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    @pytest.mark.parametrize("skip", [-1, 1, 2, 3])
    def test_frame_skip_transform_unroll(self, skip):
        torch.manual_seed(0)
        if skip < 0:
            with pytest.raises(
                ValueError,
                match="frame_skip should have a value greater or equal to one",
            ):
                FrameSkipTransform(skip)
            return
        else:
            fs = FrameSkipTransform(skip)
        base_env = GymEnv(PENDULUM_VERSIONED())
        tensordicts = TensorDict({"action": base_env.action_spec.rand((10,))}, [10])
        env = TransformedEnv(GymEnv(PENDULUM_VERSIONED()), fs)
        base_env.set_seed(0)
        env.base_env.set_seed(0)
        td1 = base_env.reset()
        td2 = env.reset()
        for key in td1.keys():
            torch.testing.assert_close(td1[key], td2[key])
        for i in range(10):
            r = 0.0
            for _ in range(skip):
                td1 = base_env.step(tensordicts[i].clone()).flatten_keys(".")
                r = td1.get("next.reward") + r
            td1.set("next.reward", r)
            td2 = env.step(tensordicts[i].clone()).flatten_keys(".")
            for key in td1.keys():
                torch.testing.assert_close(td1[key], td2[key])

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for FrameSkipTransform")


class TestNoop(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(ContinuousActionVecMockEnv(), NoopResetEnv())

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, ContinuousActionVecMockEnv), NoopResetEnv())
        with pytest.raises(
            ValueError,
            match="The parent environment batch-size is non-null",
        ):
            check_env_specs(env)

    def test_trans_parallel_env_check(self):
        raise pytest.skip("Skipped as error tested by test_trans_serial_env_check.")

    def test_transform_no_env(self):
        t = NoopResetEnv()
        with pytest.raises(
            RuntimeError,
            match="NoopResetEnv.parent not found. Make sure that the parent is set.",
        ):
            td = TensorDict({"next": {}}, [])
            t._reset(td, td.empty())
        td = TensorDict({"next": {}}, [])
        t._step(td, td.get("next"))

    def test_transform_compose(self):
        t = Compose(NoopResetEnv())
        with pytest.raises(
            RuntimeError,
            match="NoopResetEnv.parent not found. Make sure that the parent is set.",
        ):
            td = TensorDict({"next": {}}, [])
            td = t._reset(td, td.empty())
        td = TensorDict({"next": {}}, [])
        t._step(td, td.get("next"))

    def test_transform_model(self):
        t = nn.Sequential(NoopResetEnv(), nn.Identity())
        td = TensorDict()
        t(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        t = NoopResetEnv()
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict(batch_size=[10])
        rb.extend(td)
        rb.sample(1)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for NoopResetEnv")

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, random, device, compose):
        torch.manual_seed(0)
        env = ContinuousActionVecMockEnv()
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        if compose:
            transformed_env = TransformedEnv(env)
            transformed_env.append_transform(noop_reset_env)
        else:
            transformed_env = TransformedEnv(env, noop_reset_env)
        transformed_env = transformed_env.to(device)
        transformed_env.reset()
        if random:
            assert transformed_env.step_count > 0
        else:
            assert transformed_env.step_count == 30

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    def test_nested(self, random, compose):
        torch.manual_seed(0)
        env = NestedCountingEnv(nest_done=False, max_steps=50, nested_dim=6)
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        if compose:
            transformed_env = TransformedEnv(env)
            transformed_env.append_transform(noop_reset_env)
        else:
            transformed_env = TransformedEnv(env, noop_reset_env)
        transformed_env.reset()
        if random:
            assert transformed_env.count > 0
        else:
            assert transformed_env.count == 30

    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("compose", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_noop_reset_env_error(self, random, device, compose):
        torch.manual_seed(0)
        env = SerialEnv(2, lambda: ContinuousActionVecMockEnv())
        env.set_seed(100)
        noop_reset_env = NoopResetEnv(random=random)
        transformed_env = TransformedEnv(env)
        transformed_env.append_transform(noop_reset_env)
        with pytest.raises(
            ValueError,
            match="The parent environment batch-size is non-null",
        ):
            transformed_env.reset()

    @pytest.mark.parametrize("noops", [0, 2, 8])
    @pytest.mark.parametrize("max_steps", [0, 5, 9])
    def test_noop_reset_limit_exceeded(self, noops, max_steps):
        env = IncrementingEnv(max_steps=max_steps)
        check_env_specs(env)
        noop_reset_env = NoopResetEnv(noops=noops, random=False)
        transformed_env = TransformedEnv(env, noop_reset_env)
        if noops <= max_steps:  # Normal behavior.
            result = transformed_env.reset()
            assert result["observation"] == noops
        elif noops > max_steps:  # Raise error as reset limit exceeded.
            with pytest.raises(RuntimeError):
                transformed_env.reset()


@pytest.mark.skipif(
    not _has_gymnasium,
    reason="EndOfLifeTransform can only be tested when Gym is present.",
)
@pytest.mark.skipif(
    not _has_ale,
    reason="ALE not available (missing ale_py); skipping Atari gym tests.",
)
class TestEndOfLife(TransformBase):
    pytest.mark.filterwarnings("ignore:The base_env is not a gym env")

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        def make():
            with set_gym_backend("gymnasium"):
                return GymEnv(BREAKOUT_VERSIONED())

        # Chained attribute access on ParallelEnv now works with __getattr__
        with pytest.warns(UserWarning, match="The base_env is not a gym env"):
            env = TransformedEnv(
                maybe_fork_ParallelEnv(2, make), transform=EndOfLifeTransform()
            )
            try:
                check_env_specs(env)
            finally:
                try:
                    env.close()
                except RuntimeError:
                    pass

    def test_trans_serial_env_check(self):
        def make():
            with set_gym_backend("gymnasium"):
                return GymEnv(BREAKOUT_VERSIONED())

        with pytest.warns(UserWarning, match="The base_env is not a gym env"):
            env = TransformedEnv(SerialEnv(2, make), transform=EndOfLifeTransform())
            check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_single_trans_env_check(self, eol_key, lives_key):
        with set_gym_backend("gymnasium"):
            env = TransformedEnv(
                GymEnv(BREAKOUT_VERSIONED()),
                transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
            )
        check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_serial_trans_env_check(self, eol_key, lives_key):
        def make():
            with set_gym_backend("gymnasium"):
                return TransformedEnv(
                    GymEnv(BREAKOUT_VERSIONED()),
                    transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
                )

        env = SerialEnv(2, make)
        check_env_specs(env)

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_parallel_trans_env_check(self, eol_key, lives_key, maybe_fork_ParallelEnv):
        def make():
            with set_gym_backend("gymnasium"):
                return TransformedEnv(
                    GymEnv(BREAKOUT_VERSIONED()),
                    transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
                )

        env = maybe_fork_ParallelEnv(2, make)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=t.NO_PARENT_ERR.format(type(t))):
            t._step(TensorDict(), TensorDict())

    def test_transform_compose(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=t.NO_PARENT_ERR.format(type(t))):
            Compose(t)._step(TensorDict(), TensorDict())

    @pytest.mark.parametrize("eol_key", ["eol_key", ("nested", "eol")])
    @pytest.mark.parametrize("lives_key", ["lives_key", ("nested", "lives")])
    def test_transform_env(self, eol_key, lives_key):
        from tensordict.nn import TensorDictModule
        from torchrl.objectives import DQNLoss
        from torchrl.objectives.value import GAE

        with set_gym_backend("gymnasium"):
            env = TransformedEnv(
                GymEnv(BREAKOUT_VERSIONED()),
                transform=EndOfLifeTransform(eol_key=eol_key, lives_key=lives_key),
            )
        check_env_specs(env)
        loss = DQNLoss(nn.Identity(), action_space="categorical")
        env.transform.register_keys(loss)
        assert ("next", eol_key) in loss.in_keys
        gae = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=TensorDictModule(nn.Identity(), ["x"], ["y"]),
        )
        env.transform.register_keys(gae)
        assert ("next", eol_key) in gae.in_keys

    def test_transform_model(self):
        t = EndOfLifeTransform()
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            nn.Sequential(t)(TensorDict())

    def test_transform_rb(self):
        pass

    def test_transform_inverse(self):
        pass


class TestBurnInTransform(TransformBase):
    def _make_gru_module(self, input_size=4, hidden_size=4, device="cpu"):
        return GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            in_keys=["observation", "rhs", "is_init"],
            out_keys=["output", ("next", "rhs")],
            device=device,
            default_recurrent_mode=True,
        )

    def _make_lstm_module(self, input_size=4, hidden_size=4, device="cpu"):
        return LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            in_keys=["observation", "rhs_h", "rhs_c", "is_init"],
            out_keys=["output", ("next", "rhs_h"), ("next", "rhs_c")],
            device=device,
            default_recurrent_mode=True,
        )

    def _make_batch(self, batch_size: int = 2, sequence_length: int = 5):
        observation = torch.randn(batch_size, sequence_length + 1, 4)
        is_init = torch.zeros(batch_size, sequence_length, 1, dtype=torch.bool)
        batch = TensorDict(
            {
                "observation": observation[:, :-1],
                "is_init": is_init,
                "next": TensorDict(
                    {
                        "observation": observation[:, 1:],
                    },
                    batch_size=[batch_size, sequence_length],
                ),
            },
            batch_size=[batch_size, sequence_length],
        )
        return batch

    def test_single_trans_env_check(self):
        module = self._make_gru_module()
        burn_in_transform = BurnInTransform(module, burn_in=2)
        with pytest.raises(
            RuntimeError,
            match="BurnInTransform can only be appended to a ReplayBuffer.",
        ):
            env = TransformedEnv(ContinuousActionVecMockEnv(), burn_in_transform)
            check_env_specs(env)
            env.close()

    def test_serial_trans_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_parallel_trans_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_trans_serial_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    def test_trans_parallel_env_check(self):
        raise pytest.skip(
            "BurnInTransform can only be appended to a ReplayBuffer, not to a TransformedEnv."
        )

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_no_env(self, module, batch_size, sequence_length, burn_in):
        """tests the transform on dummy data, without an env."""
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        data = burn_in_transform(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_compose(self, module, batch_size, sequence_length, burn_in):
        """tests the transform on dummy data, without an env but inside a Compose."""
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_compose = Compose(BurnInTransform(module, burn_in=burn_in))
        data = burn_in_compose(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    def test_transform_env(self):
        module = self._make_gru_module()
        burn_in_transform = BurnInTransform(module, burn_in=2)
        env = TransformedEnv(ContinuousActionVecMockEnv(), burn_in_transform)
        with pytest.raises(
            RuntimeError,
            match="BurnInTransform can only be appended to a ReplayBuffer.",
        ):
            env.rollout(3)

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    def test_transform_model(self, module, batch_size, sequence_length, burn_in):
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        module = nn.Sequential(burn_in_transform, nn.Identity())
        data = module(data)
        assert data.shape[-1] == sequence_length - burn_in

        for key in data.keys():
            if key.startswith("rhs"):
                assert data[:, 0].get(key).abs().sum() > 0.0
                assert data[:, 1:].get(key).sum() == 0.0

    @pytest.mark.parametrize("module", ["gru", "lstm"])
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("sequence_length", [4, 8])
    @pytest.mark.parametrize("burn_in", [2])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, module, batch_size, sequence_length, burn_in, rbclass):
        torch.manual_seed(0)
        data = self._make_batch(batch_size, sequence_length)

        if module == "gru":
            module = self._make_gru_module()
            hidden = torch.zeros(
                data.batch_size + (module.gru.num_layers, module.gru.hidden_size)
            )
            data.set("rhs", hidden)
        else:
            module = self._make_lstm_module()
            hidden_h = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            hidden_c = torch.zeros(
                data.batch_size + (module.lstm.num_layers, module.lstm.hidden_size)
            )
            data.set("rhs_h", hidden_h)
            data.set("rhs_c", hidden_c)

        burn_in_transform = BurnInTransform(module, burn_in=burn_in)
        rb = rbclass(storage=LazyTensorStorage(20))
        rb.append_transform(burn_in_transform)
        rb.extend(data)
        batch = rb.sample(2)
        assert batch.shape[-1] == sequence_length - burn_in

        for key in batch.keys():
            if key.startswith("rhs"):
                assert batch[:, 0].get(key).abs().sum() > 0.0
                assert batch[:, 1:].get(key).sum() == 0.0

    def test_transform_inverse(self):
        raise pytest.skip("No inverse for BurnInTransform")


class TestTargetReturn(TransformBase):
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(self, batch, mode, device):
        torch.manual_seed(0)
        t = TargetReturn(target_return=10.0, mode=mode)
        env = TransformedEnv(ContinuousActionVecMockEnv(), t)
        td = env.rollout(2)
        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()
        else:
            assert (td["next", "target_return"] == 10.0).all()

    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_compose(self, batch, mode, device):
        torch.manual_seed(0)
        t = Compose(
            TargetReturn(
                in_keys=["reward"],
                out_keys=["target_return"],
                target_return=10.0,
                mode=mode,
                reset_key="_reset",
            )
        )
        next_reward = torch.rand((*batch, 1))
        td = TensorDict(
            {
                "next": {
                    "reward": next_reward,
                },
            },
            device=device,
            batch_size=batch,
        )
        td_reset = t._reset(td, td.empty())
        next_td = td.get("next")
        next_td = t._step(td_reset, next_td)
        td.set("next", next_td)

        if mode == "reduce":
            assert (td["next", "target_return"] + td["next", "reward"] == 10.0).all()

        else:
            assert (td["next", "target_return"] == 10.0).all()

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_single_trans_env_check(self, mode, device):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_serial_trans_env_check(self, mode, device):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TargetReturn(target_return=10.0, mode=mode).to(device),
                device=device,
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_parallel_trans_env_check(self, mode, device, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TargetReturn(target_return=10.0, mode=mode).to(device),
                device=device,
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_trans_serial_env_check(self, mode, device):
        env = TransformedEnv(
            SerialEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode).to(device),
            device=device,
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_trans_parallel_env_check(self, mode, device, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy).to(device),
            TargetReturn(target_return=10.0, mode=mode),
            device=device,
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [SerialEnv, ParallelEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_targetreturn_batching(self, batched_class, break_when_any_done):
        from torchrl.testing import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TargetReturn(target_return=10.0, mode="reduce"),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TargetReturn(target_return=10.0, mode="reduce"),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.tensordict.assert_allclose_td(r0, r1)

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TargetReturn")

    @pytest.mark.parametrize("mode", ["reduce", "constant"])
    @pytest.mark.parametrize("in_key", ["reward", ("agents", "reward")])
    @pytest.mark.parametrize("out_key", ["target_return", ("agents", "target_return")])
    def test_transform_no_env(self, mode, in_key, out_key):
        t = TargetReturn(
            target_return=10.0,
            mode=mode,
            in_keys=[in_key],
            out_keys=[out_key],
            reset_key="_reset",
        )
        reward = torch.randn(10, 1)
        td = TensorDict({("next", in_key): reward}, [10])
        td_reset = t._reset(td, td.empty())
        td_next = t._step(td_reset, td.get("next"))
        td.set("next", td_next)
        if mode == "reduce":
            assert (td["next", out_key] + td["next", in_key] == 10.0).all()
        else:
            assert (td["next", out_key] == 10.0).all()

    def test_transform_model(
        self,
    ):
        t = TargetReturn(target_return=10.0)
        model = nn.Sequential(t, nn.Identity())
        reward = torch.randn(10, 1)
        td = TensorDict({("next", "reward"): reward}, [10])
        with pytest.raises(
            NotImplementedError, match="cannot be executed without a parent"
        ):
            model(td)

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(
        self,
        rbclass,
    ):
        t = TargetReturn(target_return=10.0)
        rb = rbclass(storage=LazyTensorStorage(10))
        reward = torch.randn(10, 1)
        td = TensorDict({("next", "reward"): reward}, [10])
        rb.append_transform(t)
        rb.extend(td)
        with pytest.raises(
            NotImplementedError, match="cannot be executed without a parent"
        ):
            _ = rb.sample(2)


class TestExpandAs(TransformBase):
    @staticmethod
    def _make_transform(out_key: str = "env_done_expanded") -> ExpandAs:
        return ExpandAs(
            in_key="env_done",
            ref_key=("agents", "agent_0", "observation"),
            out_key=out_key,
        )

    def test_single_trans_env_check(self):
        env = TransformedEnv(MultiAgentCountingEnv(n_agents=4), self._make_transform())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                MultiAgentCountingEnv(n_agents=4), self._make_transform()
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                MultiAgentCountingEnv(n_agents=4), self._make_transform()
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(
            SerialEnv(2, lambda: MultiAgentCountingEnv(n_agents=4)),
            self._make_transform(),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, lambda: MultiAgentCountingEnv(n_agents=4)),
            self._make_transform(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        t = self._make_transform()
        td = TensorDict(
            {
                ("agents", "agent_0", "observation"): torch.randn(5, 3, 4),
                "env_done": torch.tensor([[False], [True], [False], [True], [False]]),
            },
            batch_size=[5],
        )
        td = t(td)
        assert (
            td["env_done_expanded"].shape
            == td["agents", "agent_0", "observation"].shape
        )
        assert (
            td["env_done_expanded"]
            == td["env_done"]
            .unsqueeze(-1)
            .expand_as(td["agents", "agent_0", "observation"])
        ).all()

    def test_transform_env(self):
        env = TransformedEnv(MultiAgentCountingEnv(n_agents=4), self._make_transform())
        td = env.reset()
        assert "env_done_expanded" in td.keys(True, True)
        assert (
            td["env_done_expanded"].shape
            == td["agents", "agent_0", "observation"].shape
        )
        td = env.rand_step(td)
        assert (
            td["next", "env_done_expanded"].shape
            == td["next", "agents", "agent_0", "observation"].shape
        )

    def test_transform_model(self):
        model = nn.Sequential(self._make_transform(), nn.Identity())
        td = TensorDict(
            {
                ("agents", "agent_0", "observation"): torch.randn(10, 7),
                "env_done": torch.tensor(
                    [
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                    ]
                ),
            },
            [10],
        )
        td = model(td)
        assert td["env_done_expanded"].shape == torch.Size([10, 7])

    def test_transform_compose(self):
        t = Compose(self._make_transform())
        td = TensorDict(
            {
                ("agents", "agent_0", "observation"): torch.randn(10, 7),
                "env_done": torch.tensor(
                    [
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                    ]
                ),
            },
            [10],
        )
        td = t(td)
        assert td["env_done_expanded"].shape == torch.Size([10, 7])
        assert (td["env_done_expanded"] == td["env_done"]).all()

    def test_transform_rb(self):
        t = self._make_transform()
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        td = TensorDict(
            {
                ("agents", "agent_0", "observation"): torch.randn(10, 7),
                "env_done": torch.tensor(
                    [
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                        [False],
                        [True],
                    ]
                ),
            },
            [10],
        )
        rb.append_transform(t)
        rb.extend(td)
        sample = rb.sample(2)
        assert sample["env_done_expanded"].shape == torch.Size([2, 7])

    def test_transform_inplace(self):
        t = self._make_transform(out_key="env_done")
        td = TensorDict(
            {
                ("agents", "agent_0", "observation"): torch.randn(5, 3, 4),
                "env_done": torch.tensor([[False], [True], [False], [True], [False]]),
            },
            batch_size=[5],
        )
        td = t(td)
        assert td["env_done"].shape == td["agents", "agent_0", "observation"].shape

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for ExpandAs")
