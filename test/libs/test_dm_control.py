# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import importlib.util

import numpy as np
import pytest
import torch
from packaging import version
from tensordict import assert_allclose_td, TensorDict

from torchrl.collectors import Collector
from torchrl.envs import EnvCreator
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv, DMControlWrapper
from torchrl.envs.libs.gym import _has_gym, gym_backend, GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType
from torchrl.modules import RandomPolicy
from torchrl.testing import HALFCHEETAH_VERSIONED, PONG_VERSIONED

_has_dm_control = importlib.util.find_spec("dm_control") is not None

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_OSX = __import__("sys").platform == "darwin"

_has_ale = importlib.util.find_spec("ale_py") is not None
_has_mujoco = (
    importlib.util.find_spec("mujoco") is not None
    or importlib.util.find_spec("mujoco_py") is not None
)


@pytest.mark.skipif(not _has_dmc, reason="no dm_control library found")
class TestDMControl:
    @pytest.mark.parametrize("env_name,task", [["cheetah", "run"]])
    @pytest.mark.parametrize("frame_skip", [1, 3])
    @pytest.mark.parametrize(
        "from_pixels,pixels_only", [[True, True], [True, False], [False, False]]
    )
    def test_dmcontrol(self, env_name, task, frame_skip, from_pixels, pixels_only):
        if from_pixels and (not torch.has_cuda or not torch.cuda.device_count()):
            raise pytest.skip("no cuda device")

        tds = []
        tds_reset = []
        final_seed = []
        for _ in range(2):
            env0 = DMControlEnv(
                env_name,
                task,
                frame_skip=frame_skip,
                from_pixels=from_pixels,
                pixels_only=pixels_only,
            )
            torch.manual_seed(0)
            np.random.seed(0)
            final_seed0 = env0.set_seed(0)
            tdreset0 = env0.reset()
            rollout0 = env0.rollout(max_steps=50)
            env0.close()
            del env0
            tds_reset.append(tdreset0)
            tds.append(rollout0)
            final_seed.append(final_seed0)

        tdreset1, tdreset0 = tds_reset
        rollout0, rollout1 = tds
        final_seed0, final_seed1 = final_seed

        assert_allclose_td(tdreset1, tdreset0)
        assert final_seed0 == final_seed1
        assert_allclose_td(rollout0, rollout1)

        env1 = DMControlEnv(
            env_name,
            task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        torch.manual_seed(1)
        np.random.seed(1)
        final_seed1 = env1.set_seed(1)
        tdreset1 = env1.reset()
        rollout1 = env1.rollout(max_steps=50)
        env1.close()
        del env1

        with pytest.raises(AssertionError):
            assert_allclose_td(tdreset1, tdreset0)
            assert final_seed0 == final_seed1
            assert_allclose_td(rollout0, rollout1)

        from dm_control import suite

        base_env = suite.load(env_name, task)
        if from_pixels:
            from dm_control.suite.wrappers import pixels

            render_kwargs = {"camera_id": 0}
            base_env = pixels.Wrapper(
                base_env, pixels_only=pixels_only, render_kwargs=render_kwargs
            )
        env2 = DMControlWrapper(base_env, frame_skip=frame_skip)
        torch.manual_seed(0)
        np.random.seed(0)
        final_seed2 = env2.set_seed(0)
        tdreset2 = env2.reset()
        rollout2 = env2.rollout(max_steps=50)

        assert_allclose_td(tdreset0, tdreset2)
        assert final_seed0 == final_seed2
        assert_allclose_td(rollout0, rollout2)

    def test_num_workers_returns_lazy_parallel_env(self):
        """Ensure DMControlEnv with num_workers > 1 returns a lazy ParallelEnv."""
        # When num_workers > 1, should return ParallelEnv directly (lazy)
        env = DMControlEnv("cheetah", "run", num_workers=3)
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

    def test_set_seed_and_reset_works(self):
        """Smoke test that setting seed and reset works (seed forwarded into build)."""
        env = DMControlEnv("cheetah", "run")
        final_seed = env.set_seed(0)
        assert final_seed is not None
        td = env.reset()

        assert isinstance(td, TensorDict)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_dmcontrol_kwargs_preserved_with_seed(self):
        """Test that kwargs like camera_id are preserved when seed is provided.

        Regression test for a bug where `kwargs = {"random": ...}` replaced
        all kwargs instead of updating them when _seed was not None.
        """
        # Create env with custom camera_id and from_pixels=True
        # The camera_id should be preserved even when seed is set internally
        env = DMControlEnv(
            "cheetah",
            "run",
            from_pixels=True,
            pixels_only=True,
            camera_id=1,  # Non-default camera_id
        )
        try:
            # Verify the render_kwargs were set correctly
            assert env.render_kwargs["camera_id"] == 1
            # Verify env works
            td = env.reset()
            assert "pixels" in td.keys()
        finally:
            env.close()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    @pytest.mark.parametrize("env_name,task", [["cheetah", "run"]])
    @pytest.mark.parametrize("frame_skip", [1, 3])
    @pytest.mark.parametrize(
        "from_pixels,pixels_only", [[True, True], [True, False], [False, False]]
    )
    def test_dmcontrol_device_consistency(
        self, env_name, task, frame_skip, from_pixels, pixels_only
    ):
        env0 = DMControlEnv(
            env_name,
            task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            device="cpu",
        )

        env1 = DMControlEnv(
            env_name,
            task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            device="cuda",
        )

        env0.set_seed(0)
        r0 = env0.rollout(100, break_when_any_done=False)
        assert r0.device == torch.device("cpu")
        actions = collections.deque(r0["action"].unbind(0))

        def policy(td):
            return td.set("action", actions.popleft())

        env1.set_seed(0)
        r1 = env1.rollout(100, policy, break_when_any_done=False)
        assert r1.device == torch.device("cuda:0")
        assert_allclose_td(r0, r1.cpu())

    @pytest.mark.parametrize("env_name,task", [["cheetah", "run"]])
    @pytest.mark.parametrize("frame_skip", [1, 3])
    @pytest.mark.parametrize(
        "from_pixels,pixels_only", [[True, True], [True, False], [False, False]]
    )
    def test_faketd(self, env_name, task, frame_skip, from_pixels, pixels_only):
        if from_pixels and not torch.cuda.device_count():
            raise pytest.skip("no cuda device")

        env = DMControlEnv(
            env_name,
            task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        check_env_specs(env)

    def test_truncated(self):
        env = DMControlEnv("walker", "walk")
        r = env.rollout(1001)
        assert r.shape == (1000,)
        assert r[-1]["next", "truncated"]
        assert r[-1]["next", "done"]
        assert not r[-1]["next", "terminated"]


params = []
if _has_dmc:
    params = [
        # [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
    ]
if _has_gym:
    if _has_mujoco:
        params += [
            # [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
            [GymEnv, (HALFCHEETAH_VERSIONED(),), {"from_pixels": False}],
        ]
    if _has_ale:
        params += [
            [GymEnv, (PONG_VERSIONED(),), {}],
        ]


@pytest.mark.skipif(
    IS_OSX,
    reason="rendering unstable on osx, skipping (mujoco.FatalError: gladLoadGL error)",
)
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
)
@pytest.mark.parametrize("env_lib,env_args,env_kwargs", params)
def test_td_creation_from_spec(env_lib, env_args, env_kwargs):
    if (
        _has_gym
        and version.parse(gym_backend().__version__) < version.parse("0.26.0")
        and env_kwargs.get("from_pixels", False)
        and torch.cuda.device_count() == 0
    ):
        raise pytest.skip(
            "Skipping test as rendering is not supported in tests before gym 0.26."
        )
    env = env_lib(*env_args, **env_kwargs)
    td = env.rollout(max_steps=5)
    td0 = td[0]
    fake_td = env.fake_tensordict()

    assert set(fake_td.keys(include_nested=True, leaves_only=True)) == set(
        td.keys(include_nested=True, leaves_only=True)
    )
    for key in fake_td.keys(include_nested=True, leaves_only=True):
        assert fake_td.get(key).shape == td.get(key)[0].shape
    for key in fake_td.keys(include_nested=True, leaves_only=True):
        assert fake_td.get(key).shape == td0.get(key).shape
        assert fake_td.get(key).dtype == td0.get(key).dtype
        assert fake_td.get(key).device == td0.get(key).device


params = []
if _has_dmc:
    params += [
        # [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
    ]
if _has_gym:
    if _has_mujoco:
        params += [
            # [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
            [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": False}],
            # [GymEnv, (PONG_VERSIONED,), {}],  # 1226: skipping
        ]


# @pytest.mark.skipif(IS_OSX, reason="rendering unstable on osx, skipping")
@pytest.mark.parametrize("env_lib,env_args,env_kwargs", params)
@pytest.mark.parametrize(
    "device",
    [torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")],
)
class TestCollectorLib:
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    def test_collector_run(self, env_lib, env_args, env_kwargs, device):
        env_args = tuple(arg() if callable(arg) else arg for arg in env_args)
        if not _has_dmc and env_lib is DMControlEnv:
            raise pytest.skip("no dmc")
        if not _has_gym and env_lib is GymEnv:
            raise pytest.skip("no gym")

        from_pixels = env_kwargs.get("from_pixels", False)
        if from_pixels and (not torch.has_cuda or not torch.cuda.device_count()):
            raise pytest.skip("no cuda device")

        env_fn = EnvCreator(lambda: env_lib(*env_args, **env_kwargs, device=device))
        env = SerialEnv(3, env_fn)
        # env = ParallelEnv(3, env_fn)  # 1226: Serial for efficiency reasons
        # check_env_specs(env)

        # env = ParallelEnv(3, env_fn)
        frames_per_batch = 21
        collector = Collector(  # 1226: not using MultiaSync for perf reasons
            create_env_fn=env,
            policy=RandomPolicy(action_spec=env.action_spec),
            total_frames=-1,
            max_frames_per_traj=100,
            frames_per_batch=frames_per_batch,
            init_random_frames=-1,
            reset_at_each_iter=False,
            split_trajs=True,
            device=device,
            storing_device=device,
            exploration_type=ExplorationType.RANDOM,
        )
        for i, _data in enumerate(collector):
            if i == 3:
                break
        collector.shutdown()
        assert _data.shape[1] == -(frames_per_batch // -env.num_workers)
        assert _data.shape[0] == frames_per_batch // _data.shape[1]
        del env
