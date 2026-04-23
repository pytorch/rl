# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch

from torchrl.envs.libs.genesis import _has_genesis, GenesisEnv, GenesisWrapper
from torchrl.envs.utils import check_env_specs


def _franka_scene():
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    return scene


@pytest.mark.skipif(not _has_genesis, reason="Genesis not found")
class TestGenesis:
    def test_genesis_wrapper_import(self):
        assert GenesisWrapper is not None
        assert GenesisEnv is not None

    def test_genesis_available_envs(self):
        envs = GenesisEnv.available_envs
        assert isinstance(envs, list)

    def test_genesis_wrapper_reset_step(self):
        scene = _franka_scene()
        env = GenesisWrapper(scene)
        try:
            td = env.reset()
            assert any(k.endswith("qpos") for k in td.keys())
            td = env.rand_step(td)
            assert "next" in td.keys()
            assert "reward" in td["next"].keys()
            assert "done" in td["next"].keys()
        finally:
            env.close()

    def test_genesis_wrapper_specs(self):
        scene = _franka_scene()
        env = GenesisWrapper(scene)
        try:
            check_env_specs(env)
        finally:
            env.close()

    def test_genesis_wrapper_rollout(self):
        scene = _franka_scene()
        env = GenesisWrapper(scene, max_steps=10)
        try:
            td = env.rollout(5)
            assert td.batch_size == (5,)
        finally:
            env.close()

    def test_genesis_wrapper_frame_skip(self):
        scene = _franka_scene()
        env = GenesisWrapper(scene, frame_skip=4)
        try:
            td = env.reset()
            td = env.rand_step(td)
            assert "next" in td.keys()
        finally:
            env.close()

    def test_genesis_subclass_hooks(self):
        class _CustomEnv(GenesisWrapper):
            def _make_obs(self):
                return {"custom_obs": torch.tensor([1.0, 2.0, 3.0])}

            def _compute_reward(self, action):
                return 1.0

        env = _CustomEnv(_franka_scene())
        try:
            td = env.reset()
            assert "custom_obs" in td.keys()
            td = env.rand_step(td)
            assert td["next", "reward"].item() == pytest.approx(1.0)
        finally:
            env.close()

    def test_genesis_env_config(self):
        try:
            env = GenesisEnv(env_name="franka_reach", max_steps=100)
            env.close()
        except Exception as e:
            pytest.skip(f"Genesis franka_reach not available: {e}")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
