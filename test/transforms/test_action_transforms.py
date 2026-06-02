# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
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
    Binary,
    Bounded,
    Categorical,
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
    ActionScaling,
    Compose,
    ConditionalPolicySwitch,
    ConditionalSkip,
    DiscreteActionProjection,
    EnvBase,
    FlattenAction,
    GymWrapper,
    HumanoidMacroAction,
    MacroPrimitiveTransform,
    MultiAction,
    ParallelEnv,
    RobotMacroAction,
    SatelliteAttitudeTransform,
    SatelliteMacroAction,
    SerialEnv,
    StepCounter,
    TransformedEnv,
    URScriptPrimitive,
    URScriptPrimitiveTransform,
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


class TestActionScaling(TransformBase):
    @staticmethod
    def _bounded_env(low=-2.0, high=4.0, shape=(7,)):
        return ContinuousActionVecMockEnv(
            action_spec=Bounded(low=low, high=high, shape=shape)
        )

    def test_single_trans_env_check(self):
        env = TransformedEnv(self._bounded_env(), ActionScaling())
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(self._bounded_env(), ActionScaling())

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(self._bounded_env(), ActionScaling())

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
            SerialEnv(2, self._bounded_env),
            ActionScaling(),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, self._bounded_env),
            ActionScaling(),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        # Without an env, ``loc`` and ``scale`` can be passed explicitly.
        t = ActionScaling(loc=2.0, scale=3.0)
        # inv: normalized -> env action; 1 -> 2 + 3 = 5, -1 -> 2 - 3 = -1
        out = t._inv_apply_transform(torch.tensor([1.0, -1.0, 0.0]))
        assert torch.allclose(out, torch.tensor([5.0, -1.0, 2.0]))
        # forward: env -> normalized; 5 -> 1, -1 -> -1, 2 -> 0
        back = t._apply_transform(out)
        assert torch.allclose(back, torch.tensor([1.0, -1.0, 0.0]))

    def test_transform_compose(self):
        t = ActionScaling()
        env = TransformedEnv(self._bounded_env(), Compose(t))
        spec = env.action_spec
        assert torch.allclose(spec.space.low, -torch.ones(7))
        assert torch.allclose(spec.space.high, torch.ones(7))
        # ``inv`` returns a new tensordict with the rescaled action.
        td = TensorDict({"action": torch.ones(7)}, [])
        out = env.transform.inv(td)
        assert torch.allclose(out["action"], torch.full((7,), 4.0))

    def test_transform_env(self):
        captured = {}

        class CaptureEnv(ContinuousActionVecMockEnv):
            def _step(self, td):
                captured["action"] = td["action"].clone()
                return super()._step(td)

        env = TransformedEnv(
            CaptureEnv(action_spec=Bounded(low=-2.0, high=4.0, shape=(7,))),
            ActionScaling(),
        )
        spec = env.action_spec
        assert torch.allclose(spec.space.low, -torch.ones(7))
        assert torch.allclose(spec.space.high, torch.ones(7))

        # Drive the env with a normalized action of +1: env should receive +4.
        td = env.reset()
        td["action"] = torch.ones(7)
        env.step(td)
        assert torch.allclose(captured["action"], torch.full((7,), 4.0))

        # Normalized action of -1 -> env should receive -2.
        td = env.reset()
        td["action"] = -torch.ones(7)
        env.step(td)
        assert torch.allclose(captured["action"], torch.full((7,), -2.0))

        # Normalized action of 0 -> env should receive midpoint (1.0).
        td = env.reset()
        td["action"] = torch.zeros(7)
        env.step(td)
        assert torch.allclose(captured["action"], torch.full((7,), 1.0))

    def test_transform_model(self):
        # When used in a model / replay buffer context, ``forward`` maps
        # env-scale actions back to the normalized range.
        t = ActionScaling(loc=1.0, scale=3.0)
        td = TensorDict({"action": torch.tensor([4.0, -2.0, 1.0])}, [])
        model = nn.Sequential(t, nn.Identity())
        td = model(td)
        assert torch.allclose(td["action"], torch.tensor([1.0, -1.0, 0.0]))

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        # Storage path: extend applies ``inv`` (normalized -> env-scale),
        # sample applies ``forward`` (env-scale -> normalized), so each
        # sampled action recovers one of the original normalized values.
        t = ActionScaling(loc=1.0, scale=3.0)
        normalized = torch.tensor([-1.0, -0.5, 0.0, 0.5])
        td = TensorDict({"action": normalized}, [4])
        rb = rbclass(storage=LazyTensorStorage(4))
        rb.append_transform(t)
        rb.extend(td)
        # Stored data should be in env-scale (-2, -0.5, 1, 2.5).
        stored = rb._storage._storage[:]
        torch.testing.assert_close(
            stored["action"].sort().values,
            torch.tensor([-2.0, -0.5, 1.0, 2.5]),
        )
        sampled = rb.sample(16)
        # Sampling is stochastic — every sampled action must be one of the
        # original normalized values.
        for value in sampled["action"].tolist():
            assert any(
                abs(value - x) < 1e-6 for x in normalized.tolist()
            ), f"sampled value {value} is not among the stored normalized values"

    def test_transform_inverse(self):
        # Round-trip a batch of normalized actions through inv then forward and
        # confirm we recover the input.
        t = ActionScaling(loc=1.0, scale=3.0)
        norm = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        env_action = t._inv_apply_transform(norm)
        recovered = t._apply_transform(env_action)
        assert torch.allclose(recovered, norm)

    # ActionScaling-specific tests
    def test_scaling_math_default(self):
        env = TransformedEnv(self._bounded_env(low=-2.0, high=4.0), ActionScaling())
        _ = env.action_spec  # trigger initialization
        t = env.transform
        # loc = (high + low) / 2 = 1, scale = (high - low) / 2 = 3
        assert torch.allclose(t.loc, torch.full((7,), 1.0))
        assert torch.allclose(t.scale, torch.full((7,), 3.0))

    def test_scaling_per_dim_bounds(self):
        # Different bounds per dimension.
        low = torch.tensor([-1.0, 0.0, -5.0])
        high = torch.tensor([1.0, 2.0, 5.0])
        env = TransformedEnv(
            ContinuousActionVecMockEnv(
                action_spec=Bounded(low=low, high=high, shape=(3,))
            ),
            ActionScaling(),
        )
        _ = env.action_spec
        t = env.transform
        assert torch.allclose(t.loc, torch.tensor([0.0, 1.0, 0.0]))
        assert torch.allclose(t.scale, torch.tensor([1.0, 1.0, 5.0]))

    def test_standard_normal_false(self):
        captured = {}

        class CaptureEnv(ContinuousActionVecMockEnv):
            def _step(self, td):
                captured["action"] = td["action"].clone()
                return super()._step(td)

        env = TransformedEnv(
            CaptureEnv(action_spec=Bounded(low=-2.0, high=4.0, shape=(7,))),
            ActionScaling(standard_normal=False),
        )
        spec = env.action_spec
        # standard_normal=False -> exposed spec is [0, 1]
        assert torch.allclose(spec.space.low, torch.zeros(7))
        assert torch.allclose(spec.space.high, torch.ones(7))

        td = env.reset()
        td["action"] = torch.zeros(7)
        env.step(td)
        # 0 in [0,1] -> low (-2) in env range
        assert torch.allclose(captured["action"], torch.full((7,), -2.0))

        td = env.reset()
        td["action"] = torch.ones(7)
        env.step(td)
        # 1 in [0,1] -> high (4) in env range
        assert torch.allclose(captured["action"], torch.full((7,), 4.0))

    def test_unbounded_action_spec_raises(self):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(action_spec=Unbounded(shape=(7,))),
            ActionScaling(),
        )
        with pytest.raises(RuntimeError, match="ActionScaling"):
            _ = env.action_spec

    def test_partially_unbounded_action_spec_raises(self):
        spec = Bounded(low=float("-inf"), high=1.0, shape=(7,))
        env = TransformedEnv(
            ContinuousActionVecMockEnv(action_spec=spec), ActionScaling()
        )
        with pytest.raises(RuntimeError, match="ActionScaling"):
            _ = env.action_spec

    def test_finfo_extreme_bound_raises(self):
        # Bound equal to ``finfo.max`` is treated as unbounded.
        hi = torch.tensor([torch.finfo(torch.float32).max] * 7)
        lo = torch.tensor([-2.0] * 7)
        spec = Bounded(low=lo, high=hi, shape=(7,))
        env = TransformedEnv(
            ContinuousActionVecMockEnv(action_spec=spec), ActionScaling()
        )
        with pytest.raises(RuntimeError, match="ActionScaling"):
            _ = env.action_spec

    def test_explicit_loc_scale_inconsistent_raises(self):
        with pytest.raises(ValueError, match="loc and scale"):
            ActionScaling(loc=1.0)
        with pytest.raises(ValueError, match="loc and scale"):
            ActionScaling(scale=1.0)

    def test_explicit_zero_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            ActionScaling(loc=0.0, scale=0.0)

    def test_multiple_in_keys_raises(self):
        with pytest.raises(ValueError, match="single action key"):
            ActionScaling(in_keys_inv=["action_a", "action_b"])

    def test_rollout_action_in_bounds(self):
        env = TransformedEnv(self._bounded_env(low=-2.0, high=4.0), ActionScaling())
        torch.manual_seed(0)
        r = env.rollout(20, auto_cast_to_device=False)
        # The recorded ``action`` lives in the normalized [-1, 1] space.
        assert (r["action"] >= -1.0).all()
        assert (r["action"] <= 1.0).all()

    def test_nested_action_key(self):
        # Verify that ActionScaling rewrites the spec and applies the inv
        # transform for dict-structured action spaces (issue #1209).
        class _NestedActionEnv(EnvBase):
            def __init__(self):
                super().__init__()
                self.action_spec = Composite(
                    {("agent", "action"): Bounded(low=-2.0, high=4.0, shape=(7,))}
                )
                self.observation_spec = Composite(observation=Unbounded(shape=(4,)))
                self.reward_spec = Unbounded(shape=(1,))
                self.full_done_spec = Composite(
                    done=Categorical(2, shape=(1,), dtype=torch.bool),
                    terminated=Categorical(2, shape=(1,), dtype=torch.bool),
                )
                self.last_action = None

            def _reset(self, tensordict=None):
                out = TensorDict({}, batch_size=[])
                out.update(self.observation_spec.zero())
                out.update(self.full_done_spec.zero())
                return out

            def _step(self, tensordict):
                self.last_action = tensordict["agent", "action"].clone()
                out = TensorDict({}, batch_size=[])
                out.update(self.observation_spec.zero())
                out.update(self.full_done_spec.zero())
                out["reward"] = self.reward_spec.zero()
                return out

            def _set_seed(self, seed):
                pass

        base_env = _NestedActionEnv()
        env = TransformedEnv(base_env, ActionScaling(in_keys_inv=[("agent", "action")]))
        leaf = env.full_action_spec[("agent", "action")]
        assert torch.allclose(leaf.space.low, -torch.ones(7))
        assert torch.allclose(leaf.space.high, torch.ones(7))

        td = env.reset()
        td["agent", "action"] = torch.ones(7)
        env.step(td)
        # normalized +1 -> env-scale +4 (high)
        assert torch.allclose(base_env.last_action, torch.full((7,), 4.0))

        td = env.reset()
        td["agent", "action"] = -torch.ones(7)
        env.step(td)
        # normalized -1 -> env-scale -2 (low)
        assert torch.allclose(base_env.last_action, torch.full((7,), -2.0))


class TestURScriptPrimitiveTransform:
    def test_movej_expands_to_fixed_length_action_sequence(self):
        transform = URScriptPrimitiveTransform(macro_steps=4)
        target = torch.arange(7, dtype=torch.float32).view(1, 7)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[1]]),
                "target_qpos": target,
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        out = transform.inv(td)
        action = out["action"]
        assert action.shape == torch.Size([1, 4, 7])
        torch.testing.assert_close(action[:, -1], target)
        torch.testing.assert_close(action[:, 0], target / 4)

    def test_urscript_primitive_enum(self):
        assert int(URScriptPrimitive.MOVEL) == 2
        assert str(URScriptPrimitive.OPEN_GRIPPER) == "open_gripper"

        transform = URScriptPrimitiveTransform(macro_steps=1)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[int(URScriptPrimitive.MOVEJ)]]),
                "target_qpos": torch.ones(1, 7),
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1], torch.ones(1, 7))

    def test_nested_action_key(self):
        transform = URScriptPrimitiveTransform(
            action_key=("agent", "action"), macro_steps=3
        )
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[3], [4]]),
                "robot_qpos": torch.zeros(2, 6),
                "gripper_qpos": torch.zeros(2, 2),
            },
            batch_size=[2],
        )

        out = transform.inv(td)
        action = out[("agent", "action")]
        assert action.shape == torch.Size([2, 3, 7])
        assert (action[0, :, -1] == 0).all()
        assert action[1, -1, -1] == 255
        assert (action[1, :, -1].diff() > 0).all()

    def test_make_primitive_and_action_sequence_with_nested_keys(self):
        transform = URScriptPrimitiveTransform(
            action_key=("agent", "action"),
            primitive_id_key=("agent", "primitive_id"),
            target_qpos_key=("agent", "target_qpos"),
            target_pose_key=("agent", "target_pose"),
            gripper_key=("agent", "gripper"),
            robot_qpos_key=("obs", "robot_qpos"),
            gripper_qpos_key=("obs", "gripper_qpos"),
            macro_steps=3,
            close_gripper_ctrl=0.5,
        )
        td = TensorDict(
            {
                "obs": TensorDict(
                    {
                        "robot_qpos": torch.zeros(2, 6),
                        "gripper_qpos": torch.zeros(2, 2),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[2],
        )
        target_qpos = transform.low_level_action(td["obs", "robot_qpos"], gripper=0.5)
        primitive = transform.make_primitive(
            td,
            URScriptPrimitive.MOVEJ,
            target_qpos=target_qpos,
            gripper=0.5,
        )

        assert primitive[("agent", "primitive_id")].shape == torch.Size([2, 1])
        assert (primitive[("agent", "primitive_id")] == URScriptPrimitive.MOVEJ).all()
        torch.testing.assert_close(primitive[("agent", "target_qpos")], target_qpos)

        action = transform.action_sequence(
            td,
            URScriptPrimitive.MOVEJ,
            target_qpos=target_qpos,
            gripper=0.5,
        )
        assert action.shape == torch.Size([2, 3, 7])
        torch.testing.assert_close(action[:, -1], target_qpos)

    def test_settle_steps_repeat_final_action_with_nested_key(self):
        transform = URScriptPrimitiveTransform(
            action_key=("agent", "action"), macro_steps=3, settle_steps=2
        )
        target = torch.arange(7, dtype=torch.float32).view(1, 7)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[int(URScriptPrimitive.MOVEJ)]]),
                "target_qpos": target,
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)[("agent", "action")]
        assert action.shape == torch.Size([1, 5, 7])
        torch.testing.assert_close(action[:, 2], target)
        torch.testing.assert_close(action[:, 3], target)
        torch.testing.assert_close(action[:, 4], target)

    def test_current_action_and_gripper_override(self):
        transform = URScriptPrimitiveTransform(macro_steps=2)
        current = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 255.0]])
        target = torch.zeros(1, 7)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[1]]),
                "target_qpos": target,
                "gripper": torch.tensor([[255.0]]),
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
                "action": current,
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        expected_target = target.clone()
        expected_target[..., -1] = 255.0
        torch.testing.assert_close(action[:, -1], expected_target)
        torch.testing.assert_close(
            action[:, 0], current + 0.5 * (expected_target - current)
        )

    def test_movel_uses_cartesian_solver(self):
        def solver(target_pose, start_action):
            out = start_action.clone()
            out[..., :3] = target_pose[..., :3]
            return out

        transform = URScriptPrimitiveTransform(macro_steps=1, cartesian_solver=solver)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[2]]),
                "target_pose": torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]),
                "action": torch.zeros(1, 7),
                "target_qpos": torch.zeros(1, 7),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1, :3], torch.tensor([[1.0, 2.0, 3.0]]))

    def test_structured_robot_macro_action_reach_pose(self):
        def solver(target_pose, start_action):
            out = start_action.clone()
            out[..., :3] = target_pose[..., :3]
            return out

        transform = URScriptPrimitiveTransform(macro_steps=1, cartesian_solver=solver)
        robot_macro_action = RobotMacroAction.reach_pose(
            position=torch.tensor([[1.0, 2.0, 3.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            gripper="closed",
            steps=3,
            settle_steps=2,
        )
        td = TensorDict(
            {
                "action": robot_macro_action,
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 5, 7])
        torch.testing.assert_close(action[:, 2, :3], torch.tensor([[1.0, 2.0, 3.0]]))
        torch.testing.assert_close(action[:, -1, -1:], torch.tensor([[255.0]]))

    def test_robot_macro_action_reach_pose_closed_holds_gripper_command(self):
        def solver(target_pose, start_action):
            out = start_action.clone()
            out[..., :3] = target_pose[..., :3]
            return out

        transform = URScriptPrimitiveTransform(
            macro_steps=4,
            cartesian_solver=solver,
            close_gripper_ctrl=200.0,
        )
        td = TensorDict(
            {
                "action": RobotMacroAction.reach_pose(
                    position=torch.zeros(1, 3),
                    gripper="closed",
                    gripper_command=154.0,
                    steps=4,
                ),
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.full((1, 8), 0.62),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[..., -1], torch.full((1, 4), 154.0))

    def test_robot_macro_action_close_gripper_keeps_arm(self):
        transform = URScriptPrimitiveTransform(macro_steps=4)
        robot_qpos = torch.arange(6, dtype=torch.float32).view(1, 6)
        td = TensorDict(
            {
                "action": RobotMacroAction.close_gripper(steps=4),
                "robot_qpos": robot_qpos,
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1, :6], robot_qpos)
        torch.testing.assert_close(action[:, -1, -1:], torch.full((1, 1), 255.0))

    def test_robot_macro_action_close_gripper_command_override(self):
        transform = URScriptPrimitiveTransform(macro_steps=4)
        robot_qpos = torch.arange(6, dtype=torch.float32).view(1, 6)
        td = TensorDict(
            {
                "action": RobotMacroAction.close_gripper(command=154.0, steps=4),
                "robot_qpos": robot_qpos,
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1, :6], robot_qpos)
        torch.testing.assert_close(action[:, -1, -1:], torch.full((1, 1), 154.0))

    def test_structured_robot_macro_action_reach_joints_keep_gripper(self):
        transform = URScriptPrimitiveTransform(macro_steps=4)
        robot_macro_action = RobotMacroAction.reach_joints(joints=torch.ones(1, 6))
        td = TensorDict(
            {
                "action": robot_macro_action,
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.full((1, 2), 7.0),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1, :6], torch.ones(1, 6))
        torch.testing.assert_close(action[:, -1, -1:], torch.full((1, 1), 7.0))

    def test_robot_macro_action_reset_uses_parent_home_qpos(self):
        class FakeParent:
            robot_home_qpos = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        class FakeTransform(URScriptPrimitiveTransform):
            @property
            def parent(self):
                return FakeParent()

        transform = FakeTransform(macro_steps=2)
        td = TensorDict(
            {
                "action": RobotMacroAction.RESET,
                "robot_qpos": torch.zeros(1, 6),
                "gripper_qpos": torch.zeros(1, 2),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(
            action[:, -1, :6], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        )
        torch.testing.assert_close(action[:, -1, -1:], torch.zeros(1, 1))

    def test_urscript_execute_returns_multi_action_compose(self):
        transform = URScriptPrimitiveTransform(macro_steps=2, execute=True)

        assert isinstance(transform, Compose)
        assert isinstance(transform[0], MultiAction)
        assert isinstance(transform[1], URScriptPrimitiveTransform)

    def test_macro_transform_string_proxies(self):
        transform = MacroPrimitiveTransform(
            primitive_library="basic",
            adapter="tensordict",
            solver="joint_interpolation",
            macro_steps=2,
            action_dim=7,
        )
        target = torch.arange(7, dtype=torch.float32).view(1, 7)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[2]]),
                "target_qpos": target,
                "target_pose": torch.zeros(1, 7),
                "action": torch.zeros(1, 7),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1], target)

    def test_macro_transform_non_gripper_action_dim(self):
        transform = MacroPrimitiveTransform(action_dim=4, macro_steps=3)
        target = torch.arange(8, dtype=torch.float32).view(2, 4)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[1], [1]]),
                "target_qpos": target,
            },
            batch_size=[2],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([2, 3, 4])
        torch.testing.assert_close(action[:, -1], target)

    def test_humanoid_macro_action_uses_generic_transform(self):
        transform = MacroPrimitiveTransform(action_dim=4, macro_steps=2)
        target = torch.ones(1, 4)
        td = TensorDict(
            {
                "action": HumanoidMacroAction.reach_control(
                    target, steps=3, settle_steps=1
                )
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 4, 4])
        torch.testing.assert_close(action[:, 2], target)
        torch.testing.assert_close(action[:, 3], target)

    def test_structured_primitive_tensordict_under_action_key(self):
        transform = MacroPrimitiveTransform(action_dim=3, macro_steps=2)
        td = TensorDict(
            {
                "action": TensorDict(
                    {
                        "primitive_id": torch.tensor([[1]]),
                        "target_qpos": torch.tensor([[1.0, 2.0, 3.0]]),
                        "steps": torch.tensor([[3]]),
                    },
                    batch_size=[1],
                )
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 3, 3])
        torch.testing.assert_close(action[:, -1], torch.tensor([[1.0, 2.0, 3.0]]))

    def test_satellite_macro_action_transform_identity_target_is_zero(self):
        transform = SatelliteAttitudeTransform(num_cmgs=4, macro_steps=2)
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        td = TensorDict(
            {
                "action": SatelliteMacroAction.slew_attitude(
                    target,
                    steps=2,
                    settle_steps=0,
                ),
                "bus_quat": target.clone(),
                "bus_omega": torch.zeros(1, 3),
                "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 2, 4])
        torch.testing.assert_close(action, torch.zeros_like(action))

    def test_satellite_attitude_transform_accepts_target_tensor_action(self):
        transform = SatelliteAttitudeTransform(
            num_cmgs=4,
            macro_steps=2,
            settle_steps=0,
        )
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        td = TensorDict(
            {
                "action": target.clone(),
                "bus_quat": target.clone(),
                "bus_omega": torch.zeros(1, 3),
                "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 2, 4])
        torch.testing.assert_close(action, torch.zeros_like(action))

    def test_satellite_attitude_transform_accepts_nested_attitude_action(self):
        transform = SatelliteAttitudeTransform(
            num_cmgs=4,
            macro_steps=2,
            settle_steps=0,
        )
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        td = TensorDict(
            {
                "action": TensorDict({"attitude": target.clone()}, batch_size=[1]),
                "bus_quat": target.clone(),
                "bus_omega": torch.zeros(1, 3),
                "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        assert action.shape == torch.Size([1, 2, 4])
        torch.testing.assert_close(action, torch.zeros_like(action))

    def test_satellite_attitude_transform_exposes_attitude_action_spec(self):
        transform = SatelliteAttitudeTransform(num_cmgs=4)
        input_spec = Composite(
            full_action_spec=Composite(
                action=Bounded(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 4),
                    dtype=torch.float32,
                ),
                shape=(1,),
            ),
            full_observation_spec=Composite(
                bus_quat=Unbounded(shape=(1, 4), dtype=torch.float64),
                shape=(1,),
            ),
            shape=(1,),
        )

        output_spec = transform.transform_input_spec(input_spec)
        action_spec = output_spec["full_action_spec"]

        assert ("action", "attitude") in action_spec.keys(True, True)
        assert action_spec["action", "attitude"].shape == torch.Size([1, 4])
        assert action_spec["action", "attitude"].dtype is torch.float64
        assert ("action", "attitude") in action_spec.rand().keys(True, True)

    def test_macro_transform_custom_solver_object(self):
        class Solver:
            def movel(self, target_pose, start, fallback, *, transform, tensordict):
                del fallback, transform, tensordict
                out = start.clone()
                out[..., :3] = target_pose[..., :3]
                return out

        transform = MacroPrimitiveTransform(solver=Solver(), macro_steps=1)
        td = TensorDict(
            {
                "primitive_id": torch.tensor([[2]]),
                "target_pose": torch.tensor([[4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]]),
                "action": torch.zeros(1, 7),
            },
            batch_size=[1],
        )

        action = transform.inv(td)["action"]
        torch.testing.assert_close(action[:, -1, :3], torch.tensor([[4.0, 5.0, 6.0]]))


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


class _MultiDimActionEnv(EnvBase):
    """Helper env exposing a multi-dimensional bounded continuous action.

    Stores the action received on each step as an instance attribute so that
    tests do not share state across env instances (which would otherwise race
    under SerialEnv / ParallelEnv).
    """

    def __init__(self, action_shape=(3, 5), batch_size=(), action_key="action"):
        super().__init__(batch_size=torch.Size(batch_size))
        self._action_key = action_key
        if action_key == "action":
            self.action_spec = Bounded(
                low=-1.0, high=1.0, shape=(*batch_size, *action_shape)
            )
        else:
            # Build nested action via the input_spec.
            self.action_spec = Composite(
                {
                    action_key[0]: Composite(
                        {
                            action_key[1]: Bounded(
                                low=-1.0,
                                high=1.0,
                                shape=(*batch_size, *action_shape),
                            )
                        },
                        shape=batch_size,
                    )
                },
                shape=batch_size,
            )
        self.observation_spec = Composite(
            observation=Unbounded(shape=(*batch_size, 4)), shape=batch_size
        )
        self.reward_spec = Unbounded(shape=(*batch_size, 1))
        self.full_done_spec = Composite(
            done=Categorical(2, shape=(*batch_size, 1), dtype=torch.bool),
            terminated=Categorical(2, shape=(*batch_size, 1), dtype=torch.bool),
            shape=batch_size,
        )
        self.last_action = None

    def _reset(self, tensordict=None):
        out = TensorDict({}, batch_size=self.batch_size)
        out.update(self.observation_spec.rand())
        out.update(self.full_done_spec.zero())
        return out

    def _step(self, tensordict):
        self.last_action = tensordict.get(self._action_key).clone()
        out = TensorDict({}, batch_size=self.batch_size)
        out.update(self.observation_spec.rand())
        out.update(self.full_done_spec.zero())
        out["reward"] = self.reward_spec.zero()
        return out

    def _set_seed(self, seed):
        pass


class TestFlattenAction(TransformBase):
    @staticmethod
    def _env(action_shape=(3, 5), action_key="action"):
        return _MultiDimActionEnv(action_shape=action_shape, action_key=action_key)

    def test_single_trans_env_check(self):
        env = TransformedEnv(self._env(), FlattenAction(first_dim=-2, last_dim=-1))
        check_env_specs(env)

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(self._env(), FlattenAction(first_dim=-2, last_dim=-1))

        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(self._env(), FlattenAction(first_dim=-2, last_dim=-1))

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
            SerialEnv(2, self._env),
            FlattenAction(first_dim=-2, last_dim=-1),
        )
        check_env_specs(env)

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, self._env),
            FlattenAction(first_dim=-2, last_dim=-1),
        )
        try:
            check_env_specs(env)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_transform_no_env(self):
        # ``_apply_transform`` flattens regardless of parent state.
        t = FlattenAction(first_dim=-2, last_dim=-1)
        action = torch.arange(15.0).reshape(3, 5)
        out = t._apply_transform(action)
        assert out.shape == (15,)
        assert torch.allclose(out, torch.arange(15.0))

    def test_transform_compose(self):
        t = FlattenAction(first_dim=-2, last_dim=-1)
        env = TransformedEnv(self._env(), Compose(t))
        spec = env.action_spec
        assert tuple(spec.shape) == (15,)
        # Inv direction: flat in -> 2-D out.
        td = TensorDict({"action": torch.arange(15.0)}, [])
        out = env.transform.inv(td)
        assert out["action"].shape == (3, 5)

    def test_transform_env(self):
        base = self._env()
        env = TransformedEnv(base, FlattenAction(first_dim=-2, last_dim=-1))
        spec = env.action_spec
        assert tuple(spec.shape) == (15,)
        td = env.reset()
        td["action"] = torch.arange(15.0)
        env.step(td)
        assert base.last_action.shape == (3, 5)
        assert torch.allclose(base.last_action, torch.arange(15.0).reshape(3, 5))

    def test_transform_model(self):
        t = FlattenAction(first_dim=-2, last_dim=-1)
        td = TensorDict({"action": torch.arange(15.0).reshape(3, 5)}, [])
        module = nn.Sequential(t, nn.Identity())
        out = module(td)
        assert out["action"].shape == (15,)
        assert torch.allclose(out["action"], torch.arange(15.0))

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        # ``extend`` runs the inv direction (flat -> shaped); ``sample`` runs
        # the forward direction (shaped -> flat). The round-trip recovers the
        # flat representation. ``action_shape`` lets the transform work without
        # a parent env.
        t = FlattenAction(first_dim=-2, last_dim=-1, action_shape=(3, 5))
        flat = torch.arange(60.0).reshape(4, 15)
        td = TensorDict({"action": flat}, [4])
        rb = rbclass(storage=LazyTensorStorage(4))
        rb.append_transform(t)
        rb.extend(td)
        stored = rb._storage._storage[:]
        assert stored["action"].shape == torch.Size([4, 3, 5])
        sampled = rb.sample(8)
        assert sampled["action"].shape == torch.Size([8, 15])

    def test_transform_inverse(self):
        # Round-trip: flatten then unflatten recovers the original.
        t = FlattenAction(first_dim=-2, last_dim=-1, action_shape=(3, 5))
        action = torch.arange(15.0)
        unflat = t._inv_apply_transform(action)
        assert unflat.shape == (3, 5)
        flat = t._apply_transform(unflat)
        assert torch.allclose(flat, action)

    # FlattenAction-specific tests
    def test_flatten_spec_shape(self):
        env = TransformedEnv(
            self._env(action_shape=(3, 5)),
            FlattenAction(first_dim=-2, last_dim=-1),
        )
        spec = env.action_spec
        assert tuple(spec.shape) == (15,)
        assert spec.space.low.shape == torch.Size([15])
        assert spec.space.high.shape == torch.Size([15])

    def test_flatten_3d_action_partial(self):
        env = TransformedEnv(
            self._env(action_shape=(2, 3, 5)),
            FlattenAction(first_dim=-2, last_dim=-1),
        )
        assert tuple(env.action_spec.shape) == (2, 15)

    def test_flatten_3d_action_full(self):
        env = TransformedEnv(
            self._env(action_shape=(2, 3, 5)),
            FlattenAction(first_dim=-3, last_dim=-1),
        )
        assert tuple(env.action_spec.shape) == (30,)

    def test_flatten_1d_noop(self):
        base = self._env(action_shape=(7,))
        env = TransformedEnv(base, FlattenAction(first_dim=-1, last_dim=-1))
        assert tuple(env.action_spec.shape) == (7,)
        td = env.reset()
        td["action"] = torch.ones(7)
        env.step(td)
        assert base.last_action.shape == (7,)

    def test_nested_action_key(self):
        base = self._env(action_shape=(3, 5), action_key=("nested", "action"))
        env = TransformedEnv(
            base,
            FlattenAction(
                first_dim=-2, last_dim=-1, in_keys_inv=[("nested", "action")]
            ),
        )
        leaf = env.full_action_spec[("nested", "action")]
        assert tuple(leaf.shape) == (15,)
        td = env.reset()
        td["nested", "action"] = torch.arange(15.0)
        env.step(td)
        assert base.last_action.shape == (3, 5)
        assert torch.allclose(base.last_action, torch.arange(15.0).reshape(3, 5))
        check_env_specs(env)

    def test_positive_dim_raises(self):
        with pytest.raises(ValueError, match="first_dim"):
            FlattenAction(first_dim=0, last_dim=-1)
        with pytest.raises(ValueError, match="last_dim"):
            FlattenAction(first_dim=-2, last_dim=0)

    def test_first_dim_after_last_dim_raises(self):
        with pytest.raises(ValueError, match="first_dim"):
            FlattenAction(first_dim=-1, last_dim=-2)

    def test_allow_positive_dim(self):
        # Construction with positive dims succeeds when ``allow_positive_dim``
        # is set, and the runtime path picks up the same effective span as the
        # negative-dim equivalent when attached to an env with batch_size=().
        base = self._env(action_shape=(3, 5))
        t = FlattenAction(first_dim=0, last_dim=1, allow_positive_dim=True)
        env = TransformedEnv(base, t)
        assert tuple(env.action_spec.shape) == (15,)
        td = env.reset()
        td["action"] = torch.arange(15.0)
        env.step(td)
        assert base.last_action.shape == (3, 5)
        assert torch.allclose(base.last_action, torch.arange(15.0).reshape(3, 5))

    def test_inv_apply_transform_multi_key_raises(self):
        # ``_inv_apply_transform`` cannot disambiguate between multiple keys
        # (no key information is passed to it), so it must raise rather than
        # silently picking one. Multi-key inversion still works via
        # ``inv()`` / ``_inv_call`` which iterate keys explicitly.
        t = FlattenAction(
            first_dim=-2,
            last_dim=-1,
            in_keys_inv=["a", "b"],
            action_shape=(3, 5),
        )
        with pytest.raises(RuntimeError, match="disambiguate"):
            t._inv_apply_transform(torch.arange(15.0))

    def test_inv_recovers_env_action(self):
        # Driving the env with several normalized actions and confirming the
        # captured env action always has the expected (3, 5) shape.
        base = self._env(action_shape=(3, 5))
        env = TransformedEnv(base, FlattenAction(first_dim=-2, last_dim=-1))
        for i in range(5):
            td = env.reset()
            td["action"] = torch.full((15,), float(i))
            env.step(td)
            assert base.last_action.shape == (3, 5)
            assert torch.allclose(base.last_action, torch.full((3, 5), float(i)))

    def test_rollout_action_shape(self):
        env = TransformedEnv(
            self._env(action_shape=(3, 5)),
            FlattenAction(first_dim=-2, last_dim=-1),
        )
        r = env.rollout(4)
        assert r["action"].shape == (4, 15)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
