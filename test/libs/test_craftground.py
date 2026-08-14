# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CraftGround (Minecraft) wrapper tests.

These tests boot a real Minecraft client through CraftGround's Gradle
project, which requires OpenJDK 21, an X display (e.g. Xvfb on headless
machines) and network access to Mojang's servers on the first run. They run
in the dedicated ``unittests-craftground`` CI job; see
``knowledge_base/MINECRAFT.md`` for the setup and the Minecraft ownership
and licensing requirements.
"""
from __future__ import annotations

import pytest
import torch
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.libs.craftground import (
    _has_craftground,
    CraftGroundEnv,
    CraftGroundWrapper,
)
from torchrl.envs.utils import check_env_specs


def _tiny_config():
    """A small, fast-to-render superflat world configuration."""
    from craftground.initial_environment_config import (
        InitialEnvironmentConfig,
        WorldType,
    )

    return InitialEnvironmentConfig(
        image_width=114,
        image_height=64,
        world_type=WorldType.SUPERFLAT,
        render_distance=2,
        simulation_distance=5,
        hud_hidden=True,
    )


@pytest.mark.skipif(not _has_craftground, reason="craftground not installed")
class TestCraftGround:
    @pytest.fixture(scope="class")
    def env_v1(self):
        # Booting Minecraft is expensive: share one environment (with fast
        # resets) across the tests that use the default v1 action space.
        env = CraftGroundEnv(initial_env_config=_tiny_config())
        yield env
        env.close(raise_if_closed=False)

    def test_env_specs(self, env_v1):
        check_env_specs(env_v1)

    def test_rollout_pixels(self, env_v1):
        td = env_v1.rollout(3)
        pixels = td["next", "pixels"]
        assert pixels.dtype == torch.uint8
        assert pixels.shape == torch.Size([3, 64, 114, 3])
        # the base env is a sandbox: no reward, no termination
        assert (td["next", "reward"] == 0).all()
        assert not td["next", "done"].any()

    def test_transform_composition(self, env_v1):
        # rewards/termination are composed on top of the sandbox with
        # transforms: a StepCounter must truncate the rollout
        env = TransformedEnv(env_v1, StepCounter(max_steps=2))
        td = env.rollout(5)
        assert td.shape == torch.Size([2])
        assert td["next", "truncated"][-1]

    def test_wrapper_specs(self):
        import craftground

        base = craftground.make(initial_env_config=_tiny_config())
        env = CraftGroundWrapper(base)
        try:
            check_env_specs(env)
        finally:
            env.close(raise_if_closed=False)

    def test_v2_action_space(self):
        import craftground

        env = CraftGroundEnv(
            initial_env_config=_tiny_config(),
            action_space_version=craftground.ActionSpaceVersion.V2_MINERL_HUMAN,
        )
        try:
            action_keys = set(env.full_action_spec.keys(True, True))
            # dotted upstream keys ("hotbar.1") are exposed with underscores
            assert "hotbar_1" in action_keys
            assert "camera" in action_keys
            td = env.rollout(2)
            assert td["next", "pixels"].dtype == torch.uint8
        finally:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    pytest.main([__file__])
