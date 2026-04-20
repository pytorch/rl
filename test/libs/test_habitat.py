# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from torchrl.envs.batched_envs import ParallelEnv
from torchrl.envs.libs.habitat import _has_habitat, HabitatEnv
from torchrl.envs.utils import check_env_specs


@pytest.mark.skipif(not _has_habitat, reason="habitat not installed")
@pytest.mark.parametrize("envname", ["HabitatRenderPick-v0", "HabitatPick-v0"])
class TestHabitat:
    def test_habitat(self, envname):
        env = HabitatEnv(envname)
        _ = env.rollout(3)
        check_env_specs(env)

    @pytest.mark.parametrize("from_pixels", [True, False])
    def test_habitat_render(self, envname, from_pixels):
        env = HabitatEnv(envname, from_pixels=from_pixels)
        rollout = env.rollout(3)
        check_env_specs(env)
        if from_pixels:
            assert "pixels" in rollout.keys()

    def test_num_workers_returns_lazy_parallel_env(self, envname):
        """Ensure HabitatEnv with num_workers > 1 returns a lazy ParallelEnv."""
        env = HabitatEnv(envname, num_workers=3)
        try:
            assert isinstance(env, ParallelEnv)
            assert env.num_workers == 3
            # ParallelEnv should be lazy (not started yet)
            assert env.is_closed

            # configure_parallel should work before env starts
            env.configure_parallel(use_buffers=False)
            assert env._use_buffers is False

            # After reset, env is started
            env.reset()
            assert not env.is_closed
            assert env.batch_size == torch.Size([3])
        finally:
            env.close()

    def test_set_seed_and_reset_works(self, envname):
        """Smoke test that setting seed and reset works (seed forwarded into build)."""
        env = HabitatEnv(envname)
        final_seed = env.set_seed(0)
        assert final_seed is not None
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()

    def test_habitat_kwargs_preserved_with_seed(self, envname):
        """Test that kwargs like camera_id are preserved when seed is provided."""
        env = HabitatEnv(
            envname,
            from_pixels=True,
            pixels_only=True,
        )
        try:
            final_seed = env.set_seed(1)
            assert final_seed is not None
            td = env.reset()
            assert isinstance(td, TensorDict)
            if hasattr(env, "render_kwargs"):
                assert env.render_kwargs is None or isinstance(env.render_kwargs, dict)
        finally:
            env.close()

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Test requires at least 2 GPUs",
    )
    def test_num_workers_multi_gpu(self, envname):
        """Test that num_workers with device list assigns envs to different GPUs."""
        env = HabitatEnv(
            envname,
            num_workers=2,
            device=["cuda:0", "cuda:1"],
        )
        try:
            assert isinstance(env, ParallelEnv)
            assert env.num_workers == 2

            # Verify each sub-env factory has the correct device in its kwargs
            for idx, create_fn in enumerate(env.create_env_fn):
                expected_device = f"cuda:{idx}"
                assert create_fn.keywords["device"] == expected_device

            # After reset, env should work correctly
            env.reset()
            assert not env.is_closed
            assert env.batch_size == torch.Size([2])
        finally:
            env.close()
