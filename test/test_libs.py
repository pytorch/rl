# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs.libs.dm_control import _has_dmc
from torchrl.envs.libs.gym import _has_gym, _is_from_pixels

if _has_gym:
    import gym
    from gym.wrappers.pixel_observation import PixelObservationWrapper
if _has_dmc:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

from sys import platform

from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.dm_control import DMControlEnv, DMControlWrapper
from torchrl.envs.libs.gym import GymEnv, GymWrapper

IS_OSX = platform == "darwin"


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
@pytest.mark.parametrize(
    "env_name",
    [
        "ALE/Pong-v5",
        "Pendulum-v1",
    ],
)
@pytest.mark.parametrize("frame_skip", [1, 3])
@pytest.mark.parametrize(
    "from_pixels,pixels_only",
    [
        [False, False],
        [True, True],
        [True, False],
    ],
)
def test_gym(env_name, frame_skip, from_pixels, pixels_only):
    if env_name == "ALE/Pong-v5" and not from_pixels:
        raise pytest.skip("already pixel")
    elif (
        env_name == "Pendulum-v1"
        and from_pixels
        and (not torch.has_cuda or not torch.cuda.device_count())
    ):
        raise pytest.skip("no cuda device")

    tdreset = []
    tdrollout = []
    final_seed = []
    for _ in range(2):
        env0 = GymEnv(
            env_name,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        torch.manual_seed(0)
        np.random.seed(0)
        final_seed.append(env0.set_seed(0))
        tdreset.append(env0.reset())
        tdrollout.append(env0.rollout(max_steps=50))
        assert env0.from_pixels is from_pixels
        env0.close()
        env_type = type(env0._env)
        del env0

    assert_allclose_td(*tdreset)
    assert_allclose_td(*tdrollout)
    final_seed0, final_seed1 = final_seed
    assert final_seed0 == final_seed1

    if env_name == "ALE/Pong-v5":
        base_env = gym.make(env_name, frameskip=frame_skip)
        frame_skip = 1
    else:
        base_env = gym.make(env_name)

    if from_pixels and not _is_from_pixels(base_env):
        base_env = PixelObservationWrapper(base_env, pixels_only=pixels_only)
    assert type(base_env) is env_type
    env1 = GymWrapper(base_env, frame_skip=frame_skip)
    torch.manual_seed(0)
    np.random.seed(0)
    final_seed2 = env1.set_seed(0)
    tdreset2 = env1.reset()
    rollout2 = env1.rollout(max_steps=50)
    assert env1.from_pixels is from_pixels
    env1.close()
    del env1, base_env

    assert_allclose_td(tdreset[0], tdreset2)
    assert final_seed0 == final_seed2
    assert_allclose_td(tdrollout[0], rollout2)


@pytest.mark.skipif(not _has_dmc, reason="no dm_control library found")
@pytest.mark.parametrize("env_name,task", [["cheetah", "run"]])
@pytest.mark.parametrize("frame_skip", [1, 3])
@pytest.mark.parametrize(
    "from_pixels,pixels_only",
    [
        [True, True],
        [True, False],
        [False, False],
    ],
)
def test_dmcontrol(env_name, task, frame_skip, from_pixels, pixels_only):
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

    base_env = suite.load(env_name, task)
    if from_pixels:
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


@pytest.mark.skipif(IS_OSX, reason="rendeing unstable on osx, skipping")
@pytest.mark.skipif(not (_has_dmc and _has_gym), reason="gym or dm_control not present")
@pytest.mark.parametrize(
    "env_lib,env_args,env_kwargs",
    [
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [GymEnv, ("HalfCheetah-v4",), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
        [GymEnv, ("HalfCheetah-v4",), {"from_pixels": False}],
        [GymEnv, ("ALE/Pong-v5",), {}],
    ],
)
def test_td_creation_from_spec(env_lib, env_args, env_kwargs):
    env = env_lib(*env_args, **env_kwargs)
    td = env.rollout(max_steps=5)[0]
    fake_td = env.fake_tensordict()
    assert set(fake_td.keys()) == set(td.keys())
    for key in fake_td.keys():
        assert fake_td.get(key).shape == td.get(key).shape
        assert fake_td.get(key).dtype == td.get(key).dtype
        assert fake_td.get(key).device == td.get(key).device


@pytest.mark.skipif(IS_OSX, reason="rendeing unstable on osx, skipping")
@pytest.mark.skipif(not (_has_dmc and _has_gym), reason="gym or dm_control not present")
@pytest.mark.parametrize(
    "env_lib,env_args,env_kwargs",
    [
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [GymEnv, ("HalfCheetah-v4",), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
        [GymEnv, ("HalfCheetah-v4",), {"from_pixels": False}],
        [GymEnv, ("ALE/Pong-v5",), {}],
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
class TestCollectorLib:
    def test_collector_run(self, env_lib, env_args, env_kwargs, device):
        env_fn = EnvCreator(lambda: env_lib(*env_args, **env_kwargs, device=device))
        env = ParallelEnv(3, env_fn)
        collector = MultiaSyncDataCollector(
            create_env_fn=[env, env],
            policy=RandomPolicy(env.action_spec),
            total_frames=-1,
            max_frames_per_traj=100,
            frames_per_batch=21,
            init_random_frames=-1,
            reset_at_each_iter=False,
            split_trajs=True,
            devices=[device, device],
            passing_devices=[device, device],
            update_at_each_batch=False,
            init_with_lag=False,
            exploration_mode="random",
        )
        for i, data in enumerate(collector):
            if i == 3:
                assert data.shape[0] == 3
                assert data.shape[1] == 7
                break
        collector.shutdown()
        del env


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
