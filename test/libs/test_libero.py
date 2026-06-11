# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the LIBERO environment adapter.

LIBERO is not installed in any CI environment yet (it ships from source,
not PyPI): this module currently only runs where LIBERO is installed
locally (see the LiberoWrapper docstring for install notes; on headless
Linux set ``MUJOCO_GL=egl``). The demo-replay acceptance test additionally
needs the official demo datasets (set ``LIBERO_DATASET_PATH``).
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import pytest
import torch

from torchrl.envs.libs.libero import _has_libero, LiberoEnv, LiberoWrapper
from torchrl.envs.utils import check_env_specs

_has_h5py = importlib.util.find_spec("h5py") is not None

# Keep simulator instantiations cheap: small renders, few settle steps.
_FAST = {"camera_height": 64, "camera_width": 64, "settle_steps": 2}


@pytest.mark.skipif(not _has_libero, reason="libero not found")
class TestLibero:
    def test_available_envs(self):
        suites = LiberoEnv.available_envs
        assert "libero_spatial" in suites
        assert "libero_object" in suites
        assert "libero_goal" in suites
        assert "libero_10" in suites

    def test_specs_and_rollout(self):
        env = LiberoEnv("libero_spatial", task_id=0, max_episode_steps=10, **_FAST)
        try:
            env.set_seed(0)
            td = env.reset()
            assert td["observation", "image"].dtype == torch.uint8
            assert td["observation", "image"].shape == (3, 64, 64)
            assert td["observation", "state"].dtype == torch.float32
            assert isinstance(td["language_instruction"], str)
            assert "bowl" in td["language_instruction"]
            assert not td["success"].any()
            check_env_specs(env)
            # truncation at the horizon, in env steps
            rollout = env.rollout(15)
            assert rollout.batch_size == (10,)
            assert rollout["next", "done"][-1].all()
            assert rollout["next", "truncated"][-1].all()
            assert not rollout["next", "terminated"][-1].any()
        finally:
            env.close(raise_if_closed=False)

    def test_wrapper(self):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        base = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            ),
            camera_heights=64,
            camera_widths=64,
        )
        env = LiberoWrapper(base, settle_steps=2, max_episode_steps=8)
        try:
            # the instruction defaults to the one parsed from the BDDL file
            # (the suite's curated task.language can differ; LiberoEnv uses
            # the latter)
            assert isinstance(env.instruction, str)
            assert env.instruction
            check_env_specs(env)
        finally:
            env.close(raise_if_closed=False)

    def test_fixed_init_state_deterministic(self):
        env = LiberoEnv(
            "libero_spatial",
            task_id=0,
            init_state_mode="fixed",
            init_state_id=3,
            **_FAST,
        )
        try:
            first = env.reset()["observation", "state"]
            second = env.reset()["observation", "state"]
            torch.testing.assert_close(first, second, atol=1e-4, rtol=1e-4)
        finally:
            env.close(raise_if_closed=False)

    def test_grouped_inits(self):
        # the same init state is replayed group_repeats times and the group
        # id (with the worker offset) identifies the group
        env = LiberoEnv(
            "libero_spatial",
            task_id=0,
            group_repeats=2,
            group_id_offset=100,
            **_FAST,
        )
        try:
            env.set_seed(0)
            states, group_ids = [], []
            for _ in range(4):
                td = env.reset()
                states.append(td["observation", "state"])
                group_ids.append(td["group_id"].item())
            assert group_ids == [100, 100, 101, 101]
            torch.testing.assert_close(states[0], states[1], atol=1e-4, rtol=1e-4)
            # the group id rides every step
            td["action"] = env.full_action_spec["action"].zero()
            assert env.step(td)["next", "group_id"].item() == 101
        finally:
            env.close(raise_if_closed=False)

    def test_cycle_init_states(self):
        env = LiberoEnv("libero_spatial", task_id=0, init_state_mode="cycle", **_FAST)
        try:
            first = env.reset()["observation", "state"]
            second = env.reset()["observation", "state"]
            # consecutive resets walk through distinct init states
            assert not torch.allclose(first, second, atol=1e-4)
        finally:
            env.close(raise_if_closed=False)

    def test_wrist_camera_and_proprio(self):
        env = LiberoEnv(
            "libero_spatial",
            task_id=0,
            wrist_camera="robot0_eye_in_hand",
            proprio_keys=("robot0_eef_pos", "robot0_gripper_qpos"),
            **_FAST,
        )
        try:
            td = env.reset()
            assert td["observation", "wrist_image"].shape == (3, 64, 64)
            assert td["observation", "state"].shape == (5,)
            check_env_specs(env)
        finally:
            env.close(raise_if_closed=False)

    def test_validation(self):
        with pytest.raises(ValueError, match="init_state_mode"):
            LiberoEnv("libero_spatial", task_id=0, init_state_mode="bad", **_FAST)
        with pytest.raises(ValueError, match="Unknown task suite"):
            LiberoEnv("not_a_suite", task_id=0, **_FAST)
        with pytest.raises(ValueError, match="out of range"):
            LiberoEnv("libero_spatial", task_id=1000, **_FAST)

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found")
    @pytest.mark.skipif(
        "LIBERO_DATASET_PATH" not in os.environ,
        reason="set LIBERO_DATASET_PATH to a directory holding the LIBERO demo "
        "hdf5 datasets to run the demo-replay acceptance test",
    )
    def test_demo_replay_success(self):
        # acceptance gate: replaying a successful demonstration through the
        # adapter must fire the success flag
        import h5py
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        demo_path = os.path.join(
            os.environ["LIBERO_DATASET_PATH"],
            task.problem_folder,
            f"{task.name}_demo.hdf5",
        )
        if not os.path.exists(demo_path):
            pytest.skip(f"demo file not found: {demo_path}")
        with h5py.File(demo_path, "r") as h5:
            demo = h5["data/demo_0"]
            actions = demo["actions"][:]
            initial_state = demo["states"][0]
        base = OffScreenRenderEnv(
            bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            ),
            camera_heights=64,
            camera_widths=64,
        )
        env = LiberoWrapper(
            base,
            init_states=initial_state[None],
            init_state_mode="fixed",
            settle_steps=0,
            max_episode_steps=len(actions) + 1,
        )
        try:
            td = env.reset()
            success = False
            for action in actions:
                td["action"] = torch.as_tensor(action, dtype=torch.float32)
                td = env.step(td)["next"]
                success = success or bool(td["success"].any())
                if td["done"].any():
                    break
            assert success, "replaying a successful demo must trigger success"
        finally:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
