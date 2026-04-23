# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections

from functools import partial

import numpy as np

import pytest

import torch

from _transforms_common import (
    _has_gymnasium,
    _has_mujoco,
    IS_WIN,
    mp_ctx,
    TORCH_VERSION,
    TransformBase,
)
from packaging import version
from tensordict import NonTensorData, TensorDict, TensorDictBase
from tensordict.nn import WrapModule
from torch import nn

from torchrl.data import (
    Composite,
    LazyTensorStorage,
    NonTensor,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorSpec,
    Unbounded,
)
from torchrl.envs import (
    ActionMask,
    Compose,
    ConditionalPolicySwitch,
    ConditionalSkip,
    DiscreteActionProjection,
    EnvBase,
    MultiAction,
    ParallelEnv,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms.transforms import (
    ActionDiscretizer,
    FORWARD_NOT_IMPLEMENTED,
    Transform,
)
from torchrl.envs.utils import check_env_specs, step_mdp

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
    CountingEnv,
    DiscreteActionConvMockEnvNumpy,
    EnvWithScalarAction,
    StateLessCountingEnv,
)


class TestActionMask(TransformBase):
    @property
    def _env_class(self):
        from torchrl.data import Binary, Categorical

        class MaskedEnv(EnvBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.action_spec = Categorical(4)
                self.state_spec = Composite(action_mask=Binary(4, dtype=torch.bool))
                self.observation_spec = Composite(
                    obs=Unbounded(3),
                    action_mask=Binary(4, dtype=torch.bool),
                )
                self.reward_spec = Unbounded(1)

            def _reset(self, tensordict):
                td = self.observation_spec.rand()
                td.update(torch.ones_like(self.state_spec.rand()))
                return td

            def _step(self, data):
                td = self.observation_spec.rand()
                mask = data.get("action_mask")
                action = data.get("action")
                mask = mask.scatter(-1, action.unsqueeze(-1), 0)

                td.set("action_mask", mask)
                td.set("reward", self.reward_spec.rand())
                td.set("done", ~(mask.any().view(1)))
                return td

            def _set_seed(self, seed: int | None) -> None:
                ...

        return MaskedEnv

    def test_single_trans_env_check(self):
        env = self._env_class()
        env = TransformedEnv(env, ActionMask())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        env = SerialEnv(2, lambda: TransformedEnv(self._env_class(), ActionMask()))
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        env = maybe_fork_ParallelEnv(
            2, lambda: TransformedEnv(self._env_class(), ActionMask())
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_serial_env_check(self):
        env = TransformedEnv(SerialEnv(2, self._env_class), ActionMask())
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(maybe_fork_ParallelEnv(2, self._env_class), ActionMask())
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        t = ActionMask()
        with pytest.raises(RuntimeError, match="parent cannot be None"):
            t._call(TensorDict())

    def test_transform_compose(self):
        env = self._env_class()
        env = TransformedEnv(env, Compose(ActionMask()))
        check_env_specs(env)

    def test_transform_env(self):
        env = TransformedEnv(ContinuousActionVecMockEnv(), ActionMask())
        with pytest.raises(ValueError, match="The action spec must be one of"):
            env.rollout(2)
        env = self._env_class()
        env = TransformedEnv(env, ActionMask())
        td = env.reset()
        for _ in range(1000):
            td = env.rand_action(td)
            assert env.action_spec.is_in(td.get("action"))
            td = env.step(td)
            td = step_mdp(td)
            if td.get("done"):
                break
        else:
            raise RuntimeError
        assert not td.get("action_mask").any()

    def test_transform_model(self):
        t = ActionMask()
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            t(TensorDict())

    def test_transform_rb(self):
        t = ActionMask()
        rb = ReplayBuffer(storage=LazyTensorStorage(100))
        rb.append_transform(t)
        rb.extend(TensorDict({"a": [1]}, [1]).expand(10))
        with pytest.raises(RuntimeError, match=FORWARD_NOT_IMPLEMENTED.format(type(t))):
            rb.sample(3)

    def test_transform_inverse(self):
        # no inverse transform
        return

    @pytest.mark.skipif(not _has_gymnasium, reason="gymnasium required for this test")
    @pytest.mark.parametrize("categorical_action_encoding", [True, False])
    def test_multidiscrete_action_mask_gym(self, categorical_action_encoding):
        """Test that ActionMask works with MultiDiscrete action space when mask shape matches nvec.

        This tests the fix for issue #3242: when an environment has a MultiDiscrete action space
        (e.g., [5, 5]) and provides an action_mask with matching shape (5, 5), the action spec
        is converted to a flattened Categorical/OneHot so the mask can represent all 25 possible
        action combinations.
        """
        import gymnasium as gym
        from gymnasium import spaces

        from torchrl.envs import GymWrapper, TransformedEnv
        from torchrl.envs.transforms import ActionMask
        from torchrl.envs.utils import check_env_specs

        class MultiDiscreteActionMaskEnv(gym.Env):
            """Minimal environment with MultiDiscrete action space and 2D action mask."""

            def __init__(self):
                super().__init__()
                self.action_space = spaces.MultiDiscrete([5, 5])
                self.observation_space = spaces.Dict(
                    {
                        "observation": spaces.Box(
                            low=0, high=1, shape=(5, 5), dtype=float
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=(5, 5), dtype=bool
                        ),
                    }
                )

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                obs = {
                    "observation": self.observation_space["observation"].sample(),
                    "action_mask": np.ones((5, 5), dtype=bool),
                }
                return obs, {}

            def step(self, action):
                obs = {
                    "observation": self.observation_space["observation"].sample(),
                    "action_mask": np.ones((5, 5), dtype=bool),
                }
                return obs, 0.0, False, False, {}

        # Wrap the environment
        env = GymWrapper(
            MultiDiscreteActionMaskEnv(),
            categorical_action_encoding=categorical_action_encoding,
        )

        # Apply ActionMask transform
        env = TransformedEnv(env, ActionMask())

        # This would fail before the fix with:
        # RuntimeError: Cannot expand mask to the desired shape.
        check_env_specs(env)

        # Verify we can do a rollout
        td = env.rollout(3)
        assert td is not None


class TestActionDiscretizer(TransformBase):
    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "env_cls",
        [
            ContinuousActionVecMockEnv,
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    def test_single_trans_env_check(self, categorical, env_cls):
        base_env = env_cls()
        env = base_env.append_transform(
            ActionDiscretizer(num_intervals=5, categorical=categorical)
        )
        check_env_specs(env)

    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "env_cls",
        [
            ContinuousActionVecMockEnv,
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    def test_serial_trans_env_check(self, categorical, env_cls):
        def make_env():
            base_env = env_cls()
            return base_env.append_transform(
                ActionDiscretizer(num_intervals=5, categorical=categorical)
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "env_cls",
        [
            ContinuousActionVecMockEnv,
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    def test_parallel_trans_env_check(self, categorical, env_cls):
        def make_env():
            base_env = env_cls()
            env = base_env.append_transform(
                ActionDiscretizer(num_intervals=5, categorical=categorical)
            )
            return env

        env = ParallelEnv(2, make_env, mp_start_method=mp_ctx)
        check_env_specs(env)

    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "env_cls",
        [
            ContinuousActionVecMockEnv,
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    def test_trans_serial_env_check(self, categorical, env_cls):
        env = SerialEnv(2, env_cls).append_transform(
            ActionDiscretizer(num_intervals=5, categorical=categorical)
        )
        check_env_specs(env)

    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "env_cls",
        [
            ContinuousActionVecMockEnv,
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    def test_trans_parallel_env_check(self, categorical, env_cls):
        env = ParallelEnv(2, env_cls, mp_start_method=mp_ctx).append_transform(
            ActionDiscretizer(num_intervals=5, categorical=categorical)
        )
        check_env_specs(env)

    def test_transform_no_env(self):
        categorical = True
        with pytest.raises(RuntimeError, match="Cannot execute transform"):
            ActionDiscretizer(num_intervals=5, categorical=categorical)._init()

    def test_transform_compose(self):
        categorical = True
        env = SerialEnv(2, ContinuousActionVecMockEnv).append_transform(
            Compose(ActionDiscretizer(num_intervals=5, categorical=categorical))
        )
        check_env_specs(env)

    @pytest.mark.skipif(not _has_gym, reason="gym required for this test")
    @pytest.mark.parametrize("interval_as_tensor", [False, True])
    @pytest.mark.parametrize("categorical", [True, False])
    @pytest.mark.parametrize(
        "sampling",
        [
            None,
            ActionDiscretizer.SamplingStrategy.MEDIAN,
            ActionDiscretizer.SamplingStrategy.LOW,
            ActionDiscretizer.SamplingStrategy.HIGH,
            ActionDiscretizer.SamplingStrategy.RANDOM,
        ],
    )
    @pytest.mark.parametrize(
        "env_cls",
        [
            "cheetah",
            "pendulum",
            partial(EnvWithScalarAction, singleton=True),
            partial(EnvWithScalarAction, singleton=False),
        ],
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    def test_transform_env(self, env_cls, interval_as_tensor, categorical, sampling):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if env_cls == "cheetah":
            if not _has_mujoco:
                pytest.skip(
                    "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
                )
            base_env = GymEnv(
                HALFCHEETAH_VERSIONED(),
                device=device,
            )
            num_intervals = torch.arange(5, 11)
        elif env_cls == "pendulum":
            base_env = GymEnv(
                PENDULUM_VERSIONED(),
                device=device,
            )
            num_intervals = torch.arange(5, 6)
        else:
            base_env = env_cls(
                device=device,
            )
            num_intervals = torch.arange(5, 6)

        if not interval_as_tensor:
            # override
            num_intervals = 5
        t = ActionDiscretizer(
            num_intervals=num_intervals,
            categorical=categorical,
            sampling=sampling,
            out_action_key="action_disc",
        )
        env = base_env.append_transform(t)
        check_env_specs(env)
        r = env.rollout(4)
        assert r["action"].dtype == torch.float
        if categorical:
            assert r["action_disc"].dtype == torch.int64
        else:
            assert r["action_disc"].dtype == torch.bool
        if t.sampling in (
            t.SamplingStrategy.LOW,
            t.SamplingStrategy.MEDIAN,
            t.SamplingStrategy.RANDOM,
        ):
            assert (r["action"] < base_env.action_spec.high).all()
        if t.sampling in (
            t.SamplingStrategy.HIGH,
            t.SamplingStrategy.MEDIAN,
            t.SamplingStrategy.RANDOM,
        ):
            assert (r["action"] > base_env.action_spec.low).all()

    def test_transform_model(self):
        pytest.skip("Tested elsewhere")

    def test_transform_rb(self):
        pytest.skip("Tested elsewhere")

    def test_transform_inverse(self):
        pytest.skip("Tested elsewhere")


class TestDiscreteActionProjection(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
        )
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
            )

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
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
            SerialEnv(2, DiscreteActionConvMockEnvNumpy),
            DiscreteActionProjection(7, 10),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, DiscreteActionConvMockEnvNumpy),
            DiscreteActionProjection(7, 10),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("action_key", ["action", ("nested", "stuff")])
    def test_transform_no_env(self, action_key):
        t = DiscreteActionProjection(7, 10, action_key=action_key)
        td = TensorDict(
            {action_key: nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        assert td[action_key].shape[-1] == 10
        assert (td[action_key].sum(-1) == 1).all()
        out = t.inv(td)
        assert out[action_key].shape[-1] == 7
        assert (out[action_key].sum(-1) == 1).all()

    def test_transform_compose(self):
        t = Compose(DiscreteActionProjection(7, 10))
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        assert td["action"].shape[-1] == 10
        assert (td["action"].sum(-1) == 1).all()
        out = t.inv(td)
        assert out["action"].shape[-1] == 7
        assert (out["action"].sum(-1) == 1).all()

    def test_transform_env(self):
        raise pytest.skip("Tested in test_transform_inverse")

    @pytest.mark.parametrize("include_forward", [True, False])
    def test_transform_model(self, include_forward):
        t = DiscreteActionProjection(7, 10, include_forward=include_forward)
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(7, (10, 4, 1)), 7)},
            [10, 4],
        )
        td = model(td)
        assert td["action"].shape[-1] == 10 if include_forward else 7

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    @pytest.mark.parametrize("include_forward", [True, False])
    def test_transform_rb(self, include_forward, rbclass):
        rb = rbclass(storage=LazyTensorStorage(10))
        t = DiscreteActionProjection(7, 10, include_forward=include_forward)
        rb.append_transform(t)
        td = TensorDict(
            {"action": nn.functional.one_hot(torch.randint(10, (10, 4, 1)), 10)},
            [10, 4],
        )
        rb.extend(td)

        storage = rb._storage._storage[:]

        assert storage["action"].shape[-1] == 7
        td = rb.sample(10)
        assert td["action"].shape[-1] == 10 if include_forward else 7

    def test_transform_inverse(self):
        env = TransformedEnv(
            DiscreteActionConvMockEnvNumpy(), DiscreteActionProjection(7, 10)
        )
        assert env.action_spec.space.n == 10
        assert env.action_spec.rand().shape == torch.Size([10])
        # check that transforming the action does not affect the outer td
        td = env.reset()
        td_out = env.rand_step(td)
        assert td_out["action"].shape == torch.Size([10])
        assert td is td_out


class TestMultiAction(TransformBase):
    @pytest.mark.parametrize("bwad", [False, True])
    def test_single_trans_env_check(self, bwad):
        base_env = CountingEnv(max_steps=10)
        env = TransformedEnv(
            base_env,
            Compose(
                StepCounter(step_count_key="before_count"),
                MultiAction(),
                StepCounter(step_count_key="after_count"),
            ),
        )
        env.check_env_specs()

        def policy(td):
            # 3 action per step
            td["action"] = torch.ones(3, 1)
            return td

        r = env.rollout(10, policy)
        assert r["action"].shape == (4, 3, 1)
        assert r["next", "done"].any()
        assert r["next", "done"][-1].all()
        assert (r["observation"][0] == 0).all()
        assert (r["next", "observation"][0] == 3).all()
        assert (r["next", "observation"][-1] == 11).all()
        # Check that before_count is incremented but not after_count
        assert r["before_count"].max() == 9
        assert r["after_count"].max() == 3

    def _batched_trans_env_check(self, cls, bwad, within):
        if within:

            def make_env(i):
                base_env = CountingEnv(max_steps=i)
                env = TransformedEnv(
                    base_env,
                    Compose(
                        StepCounter(step_count_key="before_count"),
                        MultiAction(),
                        StepCounter(step_count_key="after_count"),
                    ),
                )
                return env

            env = cls(2, [partial(make_env, i=10), partial(make_env, i=20)])
        else:
            base_env = cls(
                2,
                [
                    partial(CountingEnv, max_steps=10),
                    partial(CountingEnv, max_steps=20),
                ],
            )
            env = TransformedEnv(
                base_env,
                Compose(
                    StepCounter(step_count_key="before_count"),
                    MultiAction(),
                    StepCounter(step_count_key="after_count"),
                ),
            )

        try:
            env.check_env_specs()

            def policy(td):
                # 3 action per step
                td["action"] = torch.ones(2, 3, 1)
                return td

            r = env.rollout(10, policy, break_when_any_done=bwad)
            # r0
            r0 = r[0]
            if bwad:
                assert r["action"].shape == (2, 4, 3, 1)
            else:
                assert r["action"].shape == (2, 10, 3, 1)
            assert r0["next", "done"].any()
            if bwad:
                assert r0["next", "done"][-1].all()
            else:
                assert r0["next", "done"].sum() == 2

            assert (r0["observation"][0] == 0).all()
            assert (r0["next", "observation"][0] == 3).all()
            if bwad:
                assert (r0["next", "observation"][-1] == 11).all()
            else:
                assert (r0["next", "observation"][-1] == 6).all(), r0[
                    "next", "observation"
                ]
            # Check that before_count is incremented but not after_count
            assert r0["before_count"].max() == 9
            assert r0["after_count"].max() == 3
            # r1
            r1 = r[1]
            if bwad:
                assert not r1["next", "done"].any()
            else:
                assert r1["next", "done"].any()
                assert r1["next", "done"].sum() == 1
            assert (r1["observation"][0] == 0).all()
            assert (r1["next", "observation"][0] == 3).all()
            if bwad:
                # r0 cannot go above 11 but r1 can - so we see a 12 because one more step was done
                assert (r1["next", "observation"][-1] == 12).all()
            else:
                assert (r1["next", "observation"][-1] == 9).all()
            # Check that before_count is incremented but not after_count
            if bwad:
                assert r1["before_count"].max() == 9
                assert r1["after_count"].max() == 3
            else:
                assert r1["before_count"].max() == 18
                assert r1["after_count"].max() == 6
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("bwad", [False, True])
    def test_serial_trans_env_check(self, bwad):
        self._batched_trans_env_check(SerialEnv, bwad, within=True)

    @pytest.mark.parametrize("bwad", [False, True])
    def test_parallel_trans_env_check(self, bwad):
        self._batched_trans_env_check(
            partial(ParallelEnv, mp_start_method=mp_ctx), bwad, within=True
        )

    @pytest.mark.parametrize("bwad", [False, True])
    def test_trans_serial_env_check(self, bwad):
        self._batched_trans_env_check(SerialEnv, bwad, within=False)

    @pytest.mark.parametrize("bwad", [True, False])
    @pytest.mark.parametrize("buffers", [True, False])
    def test_trans_parallel_env_check(self, bwad, buffers):
        self._batched_trans_env_check(
            partial(ParallelEnv, use_buffers=buffers, mp_start_method=mp_ctx),
            bwad,
            within=False,
        )

    def test_transform_no_env(self):
        ...

    def test_transform_compose(self):
        ...

    @pytest.mark.parametrize("bwad", [True, False])
    def test_transform_env(self, bwad):
        # tests stateless (batch-unlocked) envs
        torch.manual_seed(0)
        env = StateLessCountingEnv()

        def policy(td):
            td["action"] = torch.ones(td.shape + (1,))
            return td

        r = env.rollout(
            10,
            tensordict=env.reset().expand(4),
            auto_reset=False,
            break_when_any_done=False,
            policy=policy,
        )
        assert (r["count"] == torch.arange(10).expand(4, 10).view(4, 10, 1)).all()
        td_reset = env.reset().expand(4).clone()
        td_reset["max_count"] = torch.arange(4, 8).view(4, 1)
        env = TransformedEnv(env, MultiAction())

        def policy(td):
            td["action"] = torch.ones(td.shape + (3,) + (1,))
            return td

        r = env.rollout(
            20,
            policy=policy,
            auto_reset=False,
            tensordict=td_reset,
            break_when_any_done=bwad,
        )

    def test_transform_model(self):
        ...

    def test_transform_rb(self):
        return

    def test_transform_inverse(self):
        return


@pytest.mark.skipif(IS_WIN, reason="Test is flaky on Windows")
class TestConditionalSkip(TransformBase):
    def check_non_tensor_match(self, td):
        q = collections.deque()
        obs_str = td["obs_str"]
        obs = td["observation"]
        q.extend(list(zip(obs_str, obs.unbind(0))))
        next_obs_str = td["next", "obs_str"]
        next_obs = td["next", "observation"]
        q.extend(zip(next_obs_str, next_obs.unbind(0)))
        while len(q):
            o_str, o = q.popleft()
            if isinstance(o_str, list):
                q.extend(zip(o_str, o.unbind(0)))
            else:
                assert o_str == str(o), (obs, obs_str, next_obs, next_obs_str)

    class ToString(Transform):
        def _apply_transform(self, obs: torch.Tensor) -> None:
            return NonTensorData(str(obs), device=self.parent.device)

        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            reset_data = self._call(tensordict_reset)
            return reset_data

        def transform_observation_spec(
            self, observation_spec: TensorSpec
        ) -> TensorSpec:
            observation_spec["obs_str"] = NonTensor(
                example_data="a string!",
                shape=observation_spec.shape,
                device=self.parent.device,
            )
            return observation_spec

    class CountinEnvWithString(TransformedEnv):
        def __init__(self, *args, **kwargs):
            base_env = CountingEnv()
            super().__init__(
                base_env,
                TestConditionalSkip.ToString(
                    in_keys=["observation"], out_keys=["obs_str"]
                ),
            )

    @pytest.mark.parametrize("bwad", [False, True])
    def test_single_trans_env_check(self, bwad):
        env = TestConditionalSkip.CountinEnvWithString()
        base_env = TransformedEnv(
            env,
            Compose(
                StepCounter(step_count_key="other_count"),
                ConditionalSkip(cond=lambda td: td["step_count"] % 2 == 1),
            ),
        )
        env = TransformedEnv(base_env, StepCounter(), auto_unwrap=False)
        env.set_seed(0)
        env.check_env_specs()
        policy = lambda td: td.set("action", torch.ones((1,)))
        r = env.rollout(10, policy, break_when_any_done=bwad)
        assert (r["step_count"] == torch.arange(10).view(10, 1)).all()
        assert (r["other_count"] == torch.arange(1, 11).view(10, 1) // 2).all()
        self.check_non_tensor_match(r)

    @pytest.mark.parametrize("bwad", [False, True])
    @pytest.mark.parametrize("device", [None])
    def test_serial_trans_env_check(self, bwad, device):
        def make_env(i):
            env = TestConditionalSkip.CountinEnvWithString()
            base_env = TransformedEnv(
                env,
                Compose(
                    StepCounter(step_count_key="other_count"),
                    ConditionalSkip(cond=lambda td, i=i: (td["step_count"] % 2 == i)),
                ),
            )
            return TransformedEnv(
                base_env,
                StepCounter(),
                auto_unwrap=False,
            )

        env = SerialEnv(
            2, [partial(make_env, i=0), partial(make_env, i=1)], device=device
        )
        env.check_env_specs()
        policy = lambda td: td.set("action", torch.ones((2, 1)))
        r = env.rollout(10, policy, break_when_any_done=bwad)
        assert (r["step_count"] == torch.arange(10).view(10, 1).expand(2, 10, 1)).all()
        assert (r["other_count"][0] == torch.arange(0, 10).view(10, 1) // 2).all()
        assert (r["other_count"][1] == torch.arange(1, 11).view(10, 1) // 2).all()
        self.check_non_tensor_match(r)

    @pytest.mark.parametrize("bwad", [False, True])
    @pytest.mark.parametrize("device", [None])
    def test_parallel_trans_env_check(self, bwad, device):
        def make_env(i):
            env = TestConditionalSkip.CountinEnvWithString()
            base_env = TransformedEnv(
                env,
                Compose(
                    StepCounter(step_count_key="other_count"),
                    ConditionalSkip(cond=lambda td, i=i: (td["step_count"] % 2 == i)),
                ),
            )
            return TransformedEnv(
                base_env,
                StepCounter(),
                auto_unwrap=False,
            )

        env = ParallelEnv(
            2,
            [partial(make_env, i=0), partial(make_env, i=1)],
            mp_start_method=mp_ctx,
            device=device,
        )
        try:
            env.check_env_specs()
            policy = lambda td: td.set("action", torch.ones((2, 1)))
            r = env.rollout(10, policy, break_when_any_done=bwad)
            assert (
                r["step_count"] == torch.arange(10).view(10, 1).expand(2, 10, 1)
            ).all()
            assert (r["other_count"][0] == torch.arange(0, 10).view(10, 1) // 2).all()
            assert (r["other_count"][1] == torch.arange(1, 11).view(10, 1) // 2).all()
            self.check_non_tensor_match(r)
        finally:
            env.close()
            del env

    @pytest.mark.parametrize("bwad", [False, True])
    def test_trans_serial_env_check(self, bwad):
        def make_env():
            env = TestConditionalSkip.CountinEnvWithString(max_steps=100)
            base_env = TransformedEnv(env, StepCounter(step_count_key="other_count"))
            return base_env

        base_env = SerialEnv(2, [make_env, make_env])

        def cond(td):
            sc = td["step_count"] + torch.tensor([[0], [1]])
            return sc.squeeze() % 2 == 0

        env = TransformedEnv(base_env, ConditionalSkip(cond))
        env = TransformedEnv(env, StepCounter(), auto_unwrap=False)
        env.check_env_specs()
        policy = lambda td: td.set("action", torch.ones((2, 1)))
        r = env.rollout(10, policy, break_when_any_done=bwad)
        assert (r["step_count"] == torch.arange(10).view(10, 1).expand(2, 10, 1)).all()
        assert (r["other_count"][0] == torch.arange(0, 10).view(10, 1) // 2).all()
        assert (r["other_count"][1] == torch.arange(1, 11).view(10, 1) // 2).all()
        self.check_non_tensor_match(r)

    @pytest.mark.parametrize("bwad", [True, False])
    @pytest.mark.parametrize("buffers", [True, False])
    def test_trans_parallel_env_check(self, bwad, buffers):
        def make_env():
            env = TestConditionalSkip.CountinEnvWithString(max_steps=100)
            base_env = TransformedEnv(env, StepCounter(step_count_key="other_count"))
            return base_env

        base_env = ParallelEnv(
            2, [make_env, make_env], mp_start_method=mp_ctx, use_buffers=buffers
        )
        try:

            def cond(td):
                sc = td["step_count"] + torch.tensor([[0], [1]])
                return sc.squeeze() % 2 == 0

            env = TransformedEnv(base_env, ConditionalSkip(cond))
            env = TransformedEnv(env, StepCounter(), auto_unwrap=False)
            env.check_env_specs()
            policy = lambda td: td.set("action", torch.ones((2, 1)))
            r = env.rollout(10, policy, break_when_any_done=bwad)
            assert (
                r["step_count"] == torch.arange(10).view(10, 1).expand(2, 10, 1)
            ).all()
            assert (r["other_count"][0] == torch.arange(0, 10).view(10, 1) // 2).all()
            assert (r["other_count"][1] == torch.arange(1, 11).view(10, 1) // 2).all()
            self.check_non_tensor_match(r)
        finally:
            base_env.close()
            del base_env

    def test_transform_no_env(self):
        t = ConditionalSkip(lambda td: torch.arange(td.numel()).view(td.shape) % 2 == 0)
        assert not t._inv_call(TensorDict())["_step"]
        assert t._inv_call(TensorDict())["_step"].shape == ()
        assert t._inv_call(TensorDict(batch_size=(2, 3)))["_step"].shape == (2, 3)

    def test_transform_compose(self):
        t = Compose(
            ConditionalSkip(lambda td: torch.arange(td.numel()).view(td.shape) % 2 == 0)
        )
        assert not t._inv_call(TensorDict())["_step"]
        assert t._inv_call(TensorDict())["_step"].shape == ()
        assert t._inv_call(TensorDict(batch_size=(2, 3)))["_step"].shape == (2, 3)

    def test_transform_env(self):
        # tested above
        return

    def test_transform_model(self):
        t = Compose(
            ConditionalSkip(lambda td: torch.arange(td.numel()).view(td.shape) % 2 == 0)
        )
        with pytest.raises(NotImplementedError):
            t(TensorDict())["_step"]

    def test_transform_rb(self):
        return

    def test_transform_inverse(self):
        return


class TestConditionalPolicySwitch(TransformBase):
    def test_single_trans_env_check(self):
        base_env = CountingEnv(max_steps=15)
        condition = lambda td: ((td.get("step_count") % 2) == 0).all()
        # Player 0
        policy_odd = lambda td: td.set("action", env.action_spec.zero())
        policy_even = lambda td: td.set("action", env.action_spec.one())
        transforms = Compose(
            StepCounter(),
            ConditionalPolicySwitch(condition=condition, policy=policy_even),
        )
        env = base_env.append_transform(transforms)
        env.check_env_specs()

    def _create_policy_odd(self, base_env):
        return WrapModule(
            lambda td, base_env=base_env: td.set(
                "action", base_env.action_spec_unbatched.zero(td.shape)
            ),
            out_keys=["action"],
        )

    def _create_policy_even(self, base_env):
        return WrapModule(
            lambda td, base_env=base_env: td.set(
                "action", base_env.action_spec_unbatched.one(td.shape)
            ),
            out_keys=["action"],
        )

    def _create_transforms(self, condition, policy_even):
        return Compose(
            StepCounter(),
            ConditionalPolicySwitch(condition=condition, policy=policy_even),
        )

    def _make_env(self, max_count, env_cls):
        torch.manual_seed(0)
        condition = lambda td: ((td.get("step_count") % 2) == 0).squeeze(-1)
        base_env = env_cls(max_steps=max_count)
        policy_even = self._create_policy_even(base_env)
        transforms = self._create_transforms(condition, policy_even)
        return base_env.append_transform(transforms)

    def _test_env(self, env, policy_odd):
        env.check_env_specs()
        env.set_seed(0)
        r = env.rollout(100, policy_odd, break_when_any_done=False)
        # Check results are independent: one reset / step in one env should not impact results in another
        r0, r1, r2 = r.unbind(0)
        r0_split = r0.split(6)
        assert all((r == r0_split[0][: r.numel()]).all() for r in r0_split[1:])
        r1_split = r1.split(7)
        assert all((r == r1_split[0][: r.numel()]).all() for r in r1_split[1:])
        r2_split = r2.split(8)
        assert all((r == r2_split[0][: r.numel()]).all() for r in r2_split[1:])

    def test_trans_serial_env_check(self):
        torch.manual_seed(0)
        base_env = SerialEnv(
            3,
            [partial(CountingEnv, 6), partial(CountingEnv, 7), partial(CountingEnv, 8)],
        )
        condition = lambda td: ((td.get("step_count") % 2) == 0).squeeze(-1)
        policy_odd = self._create_policy_odd(base_env)
        policy_even = self._create_policy_even(base_env)
        transforms = self._create_transforms(condition, policy_even)
        env = base_env.append_transform(transforms)
        self._test_env(env, policy_odd)

    def test_trans_parallel_env_check(self):
        torch.manual_seed(0)
        base_env = ParallelEnv(
            3,
            [partial(CountingEnv, 6), partial(CountingEnv, 7), partial(CountingEnv, 8)],
            mp_start_method=mp_ctx,
        )
        condition = lambda td: ((td.get("step_count") % 2) == 0).squeeze(-1)
        policy_odd = self._create_policy_odd(base_env)
        policy_even = self._create_policy_even(base_env)
        transforms = self._create_transforms(condition, policy_even)
        env = base_env.append_transform(transforms)
        self._test_env(env, policy_odd)

    def test_serial_trans_env_check(self):
        condition = lambda td: ((td.get("step_count") % 2) == 0).squeeze(-1)
        policy_odd = self._create_policy_odd(CountingEnv())

        def make_env(max_count):
            return partial(self._make_env, max_count, CountingEnv)

        env = SerialEnv(3, [make_env(6), make_env(7), make_env(8)])
        self._test_env(env, policy_odd)

    def test_parallel_trans_env_check(self):
        condition = lambda td: ((td.get("step_count") % 2) == 0).squeeze(-1)
        policy_odd = self._create_policy_odd(CountingEnv())

        def make_env(max_count):
            return partial(self._make_env, max_count, CountingEnv)

        env = ParallelEnv(
            3, [make_env(6), make_env(7), make_env(8)], mp_start_method=mp_ctx
        )
        self._test_env(env, policy_odd)

    def test_transform_no_env(self):
        policy_odd = lambda td: td
        policy_even = lambda td: td
        condition = lambda td: True
        transforms = ConditionalPolicySwitch(condition=condition, policy=policy_even)
        with pytest.raises(
            RuntimeError,
            match="ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional.",
        ):
            transforms(TensorDict())

    def test_transform_compose(self):
        policy_odd = lambda td: td
        policy_even = lambda td: td
        condition = lambda td: True
        transforms = Compose(
            ConditionalPolicySwitch(condition=condition, policy=policy_even),
        )
        with pytest.raises(
            RuntimeError,
            match="ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional.",
        ):
            transforms(TensorDict())

    def test_transform_env(self):
        base_env = CountingEnv(max_steps=15)
        condition = lambda td: ((td.get("step_count") % 2) == 0).all()
        # Player 0
        policy_odd = lambda td: td.set("action", env.action_spec.zero())
        policy_even = lambda td: td.set("action", env.action_spec.one())
        transforms = Compose(
            StepCounter(),
            ConditionalPolicySwitch(condition=condition, policy=policy_even),
        )
        env = base_env.append_transform(transforms)
        env.check_env_specs()
        r = env.rollout(1000, policy_odd, break_when_all_done=True)
        assert r.shape[0] == 15
        assert (r["action"] == 0).all()
        assert (
            r["step_count"] == torch.arange(1, r.numel() * 2, 2).unsqueeze(-1)
        ).all()
        assert r["next", "done"].any()

        # Player 1
        condition = lambda td: ((td.get("step_count") % 2) == 1).all()
        transforms = Compose(
            StepCounter(),
            ConditionalPolicySwitch(condition=condition, policy=policy_odd),
        )
        env = base_env.append_transform(transforms)
        r = env.rollout(1000, policy_even, break_when_all_done=True)
        assert r.shape[0] == 16
        assert (r["action"] == 1).all()
        assert (
            r["step_count"] == torch.arange(0, r.numel() * 2, 2).unsqueeze(-1)
        ).all()
        assert r["next", "done"].any()

    def test_transform_model(self):
        policy_odd = lambda td: td
        policy_even = lambda td: td
        condition = lambda td: True
        transforms = nn.Sequential(
            ConditionalPolicySwitch(condition=condition, policy=policy_even),
        )
        with pytest.raises(
            RuntimeError,
            match="ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional.",
        ):
            transforms(TensorDict())

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        policy_odd = lambda td: td
        policy_even = lambda td: td
        condition = lambda td: True
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(
            ConditionalPolicySwitch(condition=condition, policy=policy_even)
        )
        rb.extend(TensorDict(batch_size=[2]))
        with pytest.raises(
            RuntimeError,
            match="ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional.",
        ):
            rb.sample(2)

    def test_transform_inverse(self):
        return
