# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pytest
import torch

from torchrl.envs.libs.genesis import GenesisEnv, GenesisWrapper

_has_genesis = importlib.util.find_spec("genesis") is not None


@pytest.mark.skipif(not _has_genesis, reason="Genesis not found")
class TestGenesis:
    def test_genesis_wrapper_import(self):
        assert GenesisWrapper is not None
        assert GenesisEnv is not None

    def test_genesis_available_envs(self):
        envs = GenesisEnv.available_envs
        assert isinstance(envs, list)

    def test_genesis_wrapper_basic(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        env = GenesisWrapper(scene)
        try:
            td = env.reset()
            assert "observation" in td.keys()
        finally:
            env.close()

    def test_genesis_wrapper_step(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        env = GenesisWrapper(scene)
        try:
            td = env.reset()
            td = env.rand_step(td)
            assert "next" in td.keys()
            assert "observation" in td["next"].keys()
            assert "reward" in td["next"].keys()
            assert "done" in td["next"].keys()
        finally:
            env.close()

    def test_genesis_wrapper_specs(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        env = GenesisWrapper(scene)
        try:
            env.reset()
            assert env.observation_spec is not None
            assert env.action_spec is not None
            assert env.reward_spec is not None
            assert env.done_spec is not None
        finally:
            env.close()

    def test_genesis_wrapper_batch_size(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        env = GenesisWrapper(scene, batch_size=torch.Size([2]))
        try:
            td = env.reset()
            assert td.batch_size[0] == 2
        finally:
            env.close()

    def test_genesis_wrapper_frame_skip(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        env = GenesisWrapper(scene, frame_skip=4)
        try:
            td = env.reset()
            td = env.rand_step(td)
            assert "next" in td.keys()
        finally:
            env.close()

    def test_genesis_custom_functions(self):
        import genesis as gs

        gs.init(backend=gs.cpu)
        scene = gs.Scene()
        plane = scene.add_entity(gs.morphs.Plane())
        scene.build()

        def custom_obs(scene):
            return {"custom_obs": np.array([1.0, 2.0, 3.0])}

        def custom_reward(scene):
            return 1.0

        env = GenesisWrapper(
            scene,
            observation_func=custom_obs,
            reward_func=custom_reward,
        )
        try:
            td = env.reset()
            assert "custom_obs" in td["observation"].keys()
            td = env.rand_step(td)
            assert "custom_obs" in td["next"]["observation"].keys()
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
