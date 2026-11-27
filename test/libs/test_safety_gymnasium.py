# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import importlib.util
import warnings

import numpy as np
import pytest
import torch

from torchrl.envs.libs.safety_gymnasium import (
    _has_safety_gymnasium,
    SafetyGymnasiumEnv,
    SafetyGymnasiumWrapper,
)
from torchrl.envs.utils import check_env_specs

_has_gymnasium = importlib.util.find_spec("gymnasium") is not None


def _make_counting_cost_env(episode_frames, cost_scale=1.0):
    """Build a fake safety env whose n-th frame costs ``n * cost_scale`` and whose episode lasts ``episode_frames`` frames."""
    import gymnasium

    class CountingCostEnv(gymnasium.Env):
        """Episode of ``episode_frames`` frames whose n-th frame has cost ``n * cost_scale``."""

        observation_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            """Start a new episode at frame 0."""
            self.frame = 0
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            """Advance one frame and return the safety-gymnasium 6-tuple."""
            self.frame += 1
            terminated = self.frame == episode_frames
            observation = np.zeros(3, dtype=np.float32)
            cost = float(self.frame) * cost_scale
            return observation, 1.0, cost, terminated, False, {}

    return CountingCostEnv()


@pytest.mark.skipif(not _has_safety_gymnasium, reason="safety-gymnasium not installed")
class TestSafetyGymnasium:
    def test_wrapper_specs(self):
        import safety_gymnasium

        base = safety_gymnasium.make("SafetyPointGoal1-v0")
        env = SafetyGymnasiumWrapper(base)
        check_env_specs(env)
        assert "cost" in env.observation_spec.keys()

    def test_env_from_name_specs(self):
        env = SafetyGymnasiumEnv(env_name="SafetyPointGoal1-v0")
        check_env_specs(env)
        assert "cost" in env.observation_spec.keys()

    def test_rollout_exposes_cost(self):
        env = SafetyGymnasiumEnv(env_name="SafetyPointGoal1-v0")
        env.set_seed(0)
        td = env.rollout(5)
        assert ("next", "cost") in td.keys(True)
        assert td["next", "cost"].dtype == torch.float64
        assert td["next", "cost"].shape == td["next", "reward"].shape[:-1]

    def test_cost_fires_on_hazard_contact(self):
        # SafetyCarPush2-v0 has dense hazards; under random actions we expect
        # at least one positive cost in a long rollout. Without this signal
        # being plumbed through, every cost would be zero.
        env = SafetyGymnasiumEnv(env_name="SafetyCarPush2-v0")
        env.set_seed(0)
        td = env.rollout(2000, break_when_any_done=False)
        assert (td["next", "cost"] > 0).any(), (
            "Expected at least one nonzero cost over 2000 random steps; "
            "cost signal may not be plumbed correctly."
        )

    def test_cost_summed_with_frame_skip(self):
        # Each frame costs 0 or 1, so a step can only cost more than 1 if the
        # cost of its 4 frames is summed.
        env = SafetyGymnasiumEnv(env_name="SafetyCarPush2-v0", frame_skip=4)
        check_env_specs(env)
        # Seed the random actions too, not only the env.
        torch.manual_seed(0)
        env.set_seed(0)
        td = env.rollout(1000, break_when_any_done=False)
        assert td["next", "cost"].max() > 1
        assert td["next", "cost"].max() <= 4

    def test_vector_env(self):
        """Wrapping a safety-gymnasium vector env builds valid specs (#4749)."""
        import safety_gymnasium

        base = safety_gymnasium.vector.make(
            "SafetyCarPush2-v0", num_envs=3, asynchronous=False
        )
        env = SafetyGymnasiumWrapper(base, cost_trailing_dim=True)
        check_env_specs(env)
        td = env.rollout(10)
        assert td["next", "cost"].shape == td["next", "reward"].shape == (3, 10, 1)

    def test_env_num_envs(self):
        """SafetyGymnasiumEnv with num_envs runs, instead of failing on the 6-tuple step."""
        env = SafetyGymnasiumEnv(
            env_name="SafetyCarPush2-v0", num_envs=2, cost_trailing_dim=False
        )
        try:
            check_env_specs(env)
            td = env.rollout(5)
            assert td["next", "cost"].shape == (2, 5)
        finally:
            env.close()

    def test_vector_env_cost_values(self):
        """Each sub-env keeps its own cost, including on the step where its episode ends and auto-resets."""
        import safety_gymnasium

        # Sub-env 0: 4-frame episodes, n-th frame costs n.
        # Sub-env 1: 6-frame episodes, n-th frame costs 2 * n.
        env_makers = [
            functools.partial(_make_counting_cost_env, 4, 1.0),
            functools.partial(_make_counting_cost_env, 6, 2.0),
        ]
        base = safety_gymnasium.vector.SafetySyncVectorEnv(env_makers)
        env = SafetyGymnasiumWrapper(base, frame_skip=2, cost_trailing_dim=True)
        rollout = env.rollout(4, break_when_any_done=False)
        # Sub-env 0 ends after step 2 and restarts; sub-env 1 ends after step 3.
        expected_next_costs = torch.tensor(
            [
                [1.0 + 2.0, 3.0 + 4.0, 1.0 + 2.0, 3.0 + 4.0],
                [2.0 + 4.0, 6.0 + 8.0, 10.0 + 12.0, 2.0 + 4.0],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            rollout["next", "cost"], expected_next_costs.unsqueeze(-1)
        )
        # The observation right after an auto-reset carries no cost.
        expected_costs = torch.tensor(
            [[0.0, 3.0, 0.0, 3.0], [0.0, 6.0, 14.0, 0.0]], dtype=torch.float64
        )
        torch.testing.assert_close(rollout["cost"], expected_costs.unsqueeze(-1))


@pytest.mark.skipif(not _has_gymnasium, reason="gymnasium not installed")
class TestSafetyGymnasiumFrameSkip:
    @pytest.mark.parametrize(
        "frame_skip,episode_frames,expected_costs",
        [
            (2, 4, [1.0 + 2.0, 3.0 + 4.0]),
            # The last step ends after 2 of its 3 frames.
            (3, 5, [1.0 + 2.0 + 3.0, 4.0 + 5.0]),
        ],
    )
    def test_cost_summed_over_skipped_frames(
        self, frame_skip, episode_frames, expected_costs
    ):
        """The cost of a step is the sum over the frames it played, like the reward (#4738)."""
        env = SafetyGymnasiumWrapper(
            _make_counting_cost_env(episode_frames), frame_skip=frame_skip
        )
        rollout = env.rollout(10)
        torch.testing.assert_close(
            rollout["next", "cost"],
            torch.tensor(expected_costs, dtype=torch.float64),
        )


@pytest.mark.skipif(not _has_gymnasium, reason="gymnasium not installed")
class TestSafetyGymnasiumCostShape:
    @pytest.mark.parametrize(
        "cost_trailing_dim,expected_shape,warns",
        [
            (None, (2,), True),
            (False, (2,), False),
            (True, (2, 1), False),
        ],
    )
    def test_cost_trailing_dim(self, cost_trailing_dim, expected_shape, warns):
        """The default keeps the released cost shape and warns; an explicit choice is silent."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            env = SafetyGymnasiumWrapper(
                _make_counting_cost_env(episode_frames=2),
                cost_trailing_dim=cost_trailing_dim,
            )
        cost_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, FutureWarning)
            and "cost_trailing_dim" in str(warning.message)
        ]
        assert bool(cost_warnings) == warns
        check_env_specs(env)
        rollout = env.rollout(10)
        assert rollout["next", "cost"].shape == expected_shape


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
