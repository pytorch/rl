# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch

from torchrl.envs.libs.safety_gymnasium import (
    _has_safety_gymnasium,
    SafetyGymnasiumEnv,
    SafetyGymnasiumWrapper,
)
from torchrl.envs.utils import check_env_specs


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
