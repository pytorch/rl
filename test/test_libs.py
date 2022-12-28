# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from sys import platform

import numpy as np
import pytest
import torch
from _utils_internal import (
    get_available_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
)
from packaging import version
from tensordict.tensordict import assert_allclose_td
from torchrl._utils import implement_for
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.brax import _has_brax, BraxEnv
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv, DMControlWrapper
from torchrl.envs.libs.gym import _has_gym, _is_from_pixels, GymEnv, GymWrapper
from torchrl.envs.libs.habitat import _has_habitat, HabitatEnv
from torchrl.envs.libs.jumanji import _has_jumanji, JumanjiEnv
from torchrl.envs.utils import check_env_specs

if _has_gym:
    import gym

    gym_version = version.parse(gym.__version__)
    if gym_version > version.parse("0.19"):
        from gym.wrappers.pixel_observation import PixelObservationWrapper
    else:
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )

if _has_dmc:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

IS_OSX = platform == "darwin"


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
@pytest.mark.parametrize(
    "env_name",
    [
        PONG_VERSIONED,
        PENDULUM_VERSIONED,
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
class TestGym:
    def test_gym(self, env_name, frame_skip, from_pixels, pixels_only):
        if env_name == PONG_VERSIONED and not from_pixels:
            raise pytest.skip("already pixel")
        elif (
            env_name != PONG_VERSIONED
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

        if env_name == PONG_VERSIONED:
            base_env = gym.make(env_name, frameskip=frame_skip)
            frame_skip = 1
        else:
            base_env = _make_gym_environment(env_name)

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

        assert_allclose_td(tdreset[0], tdreset2, rtol=1e-4, atol=1e-4)
        assert final_seed0 == final_seed2
        assert_allclose_td(tdrollout[0], rollout2, rtol=1e-4, atol=1e-4)

    def test_gym_fake_td(self, env_name, frame_skip, from_pixels, pixels_only):
        if env_name == PONG_VERSIONED and not from_pixels:
            raise pytest.skip("already pixel")
        elif (
            env_name != PONG_VERSIONED
            and from_pixels
            and (not torch.has_cuda or not torch.cuda.device_count())
        ):
            raise pytest.skip("no cuda device")

        env = GymEnv(
            env_name,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        check_env_specs(env)


@implement_for("gym", None, "0.26")
def _make_gym_environment(env_name):  # noqa: F811
    return gym.make(env_name)


@implement_for("gym", "0.26", None)
def _make_gym_environment(env_name):  # noqa: F811
    return gym.make(env_name, render_mode="rgb_array")


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
class TestDMControl:
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

    def test_faketd(self, env_name, task, frame_skip, from_pixels, pixels_only):
        if from_pixels and (not torch.has_cuda or not torch.cuda.device_count()):
            raise pytest.skip("no cuda device")

        env = DMControlEnv(
            env_name,
            task,
            frame_skip=frame_skip,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        check_env_specs(env)


@pytest.mark.skipif(
    IS_OSX,
    reason="rendering unstable on osx, skipping (mujoco.FatalError: gladLoadGL error)",
)
@pytest.mark.skipif(not (_has_dmc and _has_gym), reason="gym or dm_control not present")
@pytest.mark.parametrize(
    "env_lib,env_args,env_kwargs",
    [
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": False}],
        [GymEnv, (PONG_VERSIONED,), {}],
    ],
)
def test_td_creation_from_spec(env_lib, env_args, env_kwargs):
    if (
        gym_version < version.parse("0.26.0")
        and env_kwargs.get("from_pixels", False)
        and torch.cuda.device_count() == 0
    ):
        pytest.skip(
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


@pytest.mark.skipif(IS_OSX, reason="rendering unstable on osx, skipping")
@pytest.mark.skipif(not (_has_dmc and _has_gym), reason="gym or dm_control not present")
@pytest.mark.parametrize(
    "env_lib,env_args,env_kwargs",
    [
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": False}],
        [GymEnv, (PONG_VERSIONED,), {}],
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
class TestCollectorLib:
    def test_collector_run(self, env_lib, env_args, env_kwargs, device):
        from_pixels = env_kwargs.get("from_pixels", False)
        if from_pixels and (not torch.has_cuda or not torch.cuda.device_count()):
            raise pytest.skip("no cuda device")

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


@pytest.mark.skipif(not _has_habitat, reason="habitat not installed")
@pytest.mark.parametrize("envname", ["HabitatRenderPick-v0", "HabitatRenderPick-v0"])
class TestHabitat:
    def test_habitat(self, envname):
        env = HabitatEnv(envname)
        rollout = env.rollout(3)
        check_env_specs(env)


@pytest.mark.skipif(not _has_jumanji, reason="jumanji not installed")
@pytest.mark.parametrize(
    "envname",
    [
        "TSP50-v0",
        "Snake-6x6-v0",
    ],
)
class TestJumanji:
    def test_jumanji_seeding(self, envname):
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = JumanjiEnv(envname)
            torch.manual_seed(0)
            np.random.seed(0)
            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=50))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_jumanji_batch_size(self, envname, batch_size):
        env = JumanjiEnv(envname, batch_size=batch_size)
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=50)
        env.close()
        del env
        assert tdreset.batch_size == batch_size
        assert tdrollout.batch_size[:-1] == batch_size

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_jumanji_spec_rollout(self, envname, batch_size):
        env = JumanjiEnv(envname, batch_size=batch_size)
        env.set_seed(0)
        check_env_specs(env)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_jumanji_consistency(self, envname, batch_size):
        import jax
        import jax.numpy as jnp
        import numpy as onp
        from torchrl.envs.libs.jax_utils import _tree_flatten

        env = JumanjiEnv(envname, batch_size=batch_size)
        obs_keys = list(env.observation_spec.keys(True))
        env.set_seed(1)
        rollout = env.rollout(10)

        env.set_seed(1)
        key = env.key
        base_env = env._env
        key, *keys = jax.random.split(key, np.prod(batch_size) + 1)
        state, timestep = jax.vmap(base_env.reset)(jnp.stack(keys))
        # state = env._reshape(state)
        # timesteps.append(timestep)
        for i in range(rollout.shape[-1]):
            action = rollout[..., i]["action"]
            # state = env._flatten(state)
            action = _tree_flatten(env.read_action(action), env.batch_size)
            state, timestep = jax.vmap(base_env.step)(state, action)
            # state = env._reshape(state)
            # timesteps.append(timestep)
            checked = False
            for _key in obs_keys:
                if isinstance(_key, str):
                    _key = (_key,)
                try:
                    t2 = getattr(timestep, _key[0])
                except AttributeError:
                    try:
                        t2 = getattr(timestep.observation, _key[0])
                    except AttributeError:
                        continue
                t1 = rollout[..., i][("next", *_key)]
                for __key in _key[1:]:
                    t2 = getattr(t2, _key)
                t2 = torch.tensor(onp.asarray(t2)).view_as(t1)
                torch.testing.assert_close(t1, t2)
                checked = True
            if not checked:
                raise AttributeError(
                    f"None of the keys matched: {rollout}, {list(timestep.__dict__.keys())}"
                )


@pytest.mark.skipif(not _has_brax, reason="brax not installed")
@pytest.mark.parametrize("envname", ["fast"])
class TestBrax:
    def test_brax_seeding(self, envname):
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = BraxEnv(envname)
            torch.manual_seed(0)
            np.random.seed(0)
            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=50))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_brax_batch_size(self, envname, batch_size):
        env = BraxEnv(envname, batch_size=batch_size)
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=50)
        env.close()
        del env
        assert tdreset.batch_size == batch_size
        assert tdrollout.batch_size[:-1] == batch_size

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_brax_spec_rollout(self, envname, batch_size):
        env = BraxEnv(envname, batch_size=batch_size)
        env.set_seed(0)
        check_env_specs(env)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_brax_consistency(self, envname, batch_size, requires_grad):
        import jax
        import jax.numpy as jnp
        from torchrl.envs.libs.jax_utils import (
            _ndarray_to_tensor,
            _tensor_to_ndarray,
            _tree_flatten,
        )

        env = BraxEnv(envname, batch_size=batch_size, requires_grad=requires_grad)
        env.set_seed(1)
        rollout = env.rollout(10)

        env.set_seed(1)
        key = env._key
        base_env = env._env
        key, *keys = jax.random.split(key, np.prod(batch_size) + 1)
        state = jax.vmap(base_env.reset)(jnp.stack(keys))
        for i in range(rollout.shape[-1]):
            action = rollout[..., i]["action"]
            action = _tensor_to_ndarray(action.clone())
            action = _tree_flatten(action, env.batch_size)
            state = jax.vmap(base_env.step)(state, action)
            t1 = rollout[..., i][("next", "observation")]
            t2 = _ndarray_to_tensor(state.obs).view_as(t1)
            torch.testing.assert_close(t1, t2)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_brax_grad(self, envname, batch_size):
        batch_size = (1,)
        env = BraxEnv(envname, batch_size=batch_size, requires_grad=True)
        env.set_seed(0)
        td1 = env.reset()
        action = torch.randn(batch_size + env.action_spec.shape)
        action.requires_grad_(True)
        td1["action"] = action
        td2 = env.step(td1)
        td2["reward"].mean().backward()
        env.close()
        del env


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
