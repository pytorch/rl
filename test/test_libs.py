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
    _make_envs,
    get_available_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
)
from packaging import version
from tensordict.tensordict import assert_allclose_td, TensorDict
from torch import nn
from torchrl._utils import implement_for
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv, DMControlWrapper
from torchrl.envs.libs.gym import _has_gym, _is_from_pixels, GymEnv, GymWrapper
from torchrl.envs.libs.habitat import _has_habitat, HabitatEnv
from torchrl.envs.libs.jumanji import _has_jumanji, JumanjiEnv
from torchrl.envs.utils import check_env_specs

from torchrl.envs.vec_env import _has_envpool
from torchrl.modules import ActorCriticOperator, MLP, SafeModule, ValueOperator

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
@pytest.mark.parametrize("envname", ["Snake-6x6-v0", "TSP50-v0"])
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
            action = env._flatten(env.read_action(action))
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


@pytest.mark.skipif(not _has_envpool, reason="no envpool library found")
class TestEnvPool:
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PONG_VERSIONED, PENDULUM_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [False, True])
    @pytest.mark.parametrize("transformed_out", [False, True])
    def test_parallel_env(
        self, env_name, frame_skip, transformed_in, transformed_out, T=10, N=3
    ):
        env_parallel, env_serial, env_multithreaded, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        envs_to_test = [env_parallel]
        if _has_envpool:
            envs_to_test.append(env_multithreaded)
        for env_test in envs_to_test:
            td = TensorDict(
                source={"action": env0.action_spec.rand((N,))},
                batch_size=[
                    N,
                ],
            )
            print(f"env0.action_spec={env0.action_spec} td={td}")
            td1 = env_test.step(td)
            assert not td1.is_shared()
            assert "done" in td1.keys()
            assert "reward" in td1.keys()

            with pytest.raises(RuntimeError):
                # number of actions does not match number of workers
                td = TensorDict(
                    source={"action": env0.action_spec.rand((N - 1,))},
                    batch_size=[N - 1],
                )
                td1 = env_test.step(td)

            td_reset = TensorDict(
                source={
                    "reset_workers": torch.zeros(N, 1, dtype=torch.bool).bernoulli_()
                },
                batch_size=[
                    N,
                ],
            )
            env_test.reset(tensordict=td_reset)

            td = env_test.rollout(policy=None, max_steps=T)
            assert (
                td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
            ), f"{td.shape}, {td.get('done').sum(1)}"

            env_test.close()

        # env_serial.close()  # never opened
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", [PENDULUM_VERSIONED])
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [True, False])
    @pytest.mark.parametrize(
        "selected_keys",
        [
            ["action", "observation", "next_observation", "done", "reward"],
            ["hidden", "action", "observation", "next_observation", "done", "reward"],
            None,
        ],
    )
    def test_parallel_env_with_policy(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        selected_keys,
        T=10,
        N=3,
    ):
        env_parallel, env_serial, env_multithreaded, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
            selected_keys=selected_keys,
        )

        policy = ActorCriticOperator(
            SafeModule(
                spec=None,
                module=nn.LazyLinear(12),
                in_keys=["observation"],
                out_keys=["hidden"],
            ),
            SafeModule(
                spec=None,
                module=nn.LazyLinear(env0.action_spec.shape[-1]),
                in_keys=["hidden"],
                out_keys=["action"],
            ),
            ValueOperator(
                module=MLP(out_features=1, num_cells=[]), in_keys=["hidden", "action"]
            ),
        )

        envs_to_test = [env_parallel]
        if _has_envpool:
            envs_to_test.append(env_multithreaded)
        for env_test in envs_to_test:
            td = TensorDict(
                source={"action": env0.action_spec.rand((N,))},
                batch_size=[
                    N,
                ],
            )
            td1 = env_test.step(td)
            assert not td1.is_shared()
            assert "done" in td1.keys()
            assert "reward" in td1.keys()

            with pytest.raises(RuntimeError):
                # number of actions does not match number of workers
                td = TensorDict(
                    source={"action": env0.action_spec.rand((N - 1,))},
                    batch_size=[N - 1],
                )
                td1 = env_test.step(td)

            td_reset = TensorDict(
                source={
                    "reset_workers": torch.zeros(N, 1, dtype=torch.bool).bernoulli_()
                },
                batch_size=[
                    N,
                ],
            )
            env_test.reset(tensordict=td_reset)

            td = env_test.rollout(policy=policy, max_steps=T)
            assert (
                td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
            ), f"{td.shape}, {td.get('done').sum(1)}"
            env_test.close()

        # env_serial.close()
        env0.close()

    @pytest.mark.skipif(not _has_envpool, reason="no envpool")
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name",
        [
            PENDULUM_VERSIONED,
            PONG_VERSIONED,
        ],
    )
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_in", [False, True])
    @pytest.mark.parametrize("transformed_out", [True, False])
    @pytest.mark.parametrize("static_seed", [True, False])
    def test_multithreaded_env_seed(
        self, env_name, frame_skip, transformed_in, transformed_out, static_seed
    ):
        _, env_serial, env_multithread, env0 = _make_envs(
            env_name, frame_skip, transformed_in, transformed_out, 1
        )
        env0.set_seed(0, static_seed=static_seed)

        out_seed_serial = env_serial.set_seed(0, static_seed=static_seed)
        if static_seed:
            assert out_seed_serial == 0
        td0_serial = env_serial.reset()
        torch.manual_seed(0)

        max_steps = 2

        td_serial = env_serial.rollout(
            max_steps=max_steps, auto_reset=False, tensordict=td0_serial
        ).contiguous()
        key = "pixels" if "pixels" in td_serial.keys() else "observation"
        torch.testing.assert_close(
            td_serial[:, 0].get(("next", key)), td_serial[:, 1].get(key)
        )

        out_seed_multithread = env_multithread.set_seed(0, static_seed=static_seed)
        if static_seed:
            assert out_seed_serial == 0
        td0_multithread = env_multithread.reset()

        torch.manual_seed(0)
        assert out_seed_multithread == out_seed_serial
        td_multithread = env_multithread.rollout(
            max_steps=max_steps, auto_reset=False, tensordict=td0_multithread
        ).contiguous()
        torch.testing.assert_close(
            td_multithread[:, :-1].get(("next", key)), td_multithread[:, 1:].get(key)
        )
        print(td0_serial["observation"], td0_multithread["observation"])
        assert_allclose_td(td0_serial, td0_multithread)
        assert_allclose_td(td_serial[:, 0], td_multithread[:, 0])  # first step
        assert_allclose_td(td_serial[:, 1], td_multithread[:, 1])  # second step
        assert_allclose_td(td_serial, td_multithread)

        env_multithread.close()
        env_serial.close()

    @pytest.mark.skipif(not _has_envpool, reason="no envpool")
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_multithread_env_shutdown(self):
        _, _, env, _ = _make_envs(
            PENDULUM_VERSIONED,
            1,
            transformed_in=True,
            transformed_out=False,
            N=3,
        )
        env.reset()
        assert not env.is_closed
        env.rand_step()
        assert not env.is_closed
        env.close()
        assert env.is_closed
        env.reset()
        assert not env.is_closed
        env.close()

    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda to test on")
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", [PONG_VERSIONED, PENDULUM_VERSIONED])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [False, True])
    @pytest.mark.parametrize("open_before", [False, True])
    def test_parallel_env_cast(
        self,
        env_name,
        frame_skip,
        transformed_in,
        transformed_out,
        device,
        open_before,
        T=10,
        N=3,
    ):
        # tests casting to device
        env_parallel, env_serial, env_multithread, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            N=N,
        )
        if open_before:
            td_cpu = env0.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env0 = env0.to(device)
        assert env0.observation_spec.device == torch.device(device)
        assert env0.action_spec.device == torch.device(device)
        assert env0.reward_spec.device == torch.device(device)
        assert env0.device == torch.device(device)
        td_device = env0.reset()
        assert td_device.device == torch.device(device), env0
        td_device = env0.rand_step()
        assert td_device.device == torch.device(device), env0
        td_device = env0.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env0

        if open_before:
            td_cpu = env_serial.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env_serial = env_serial.to(device)
        assert env_serial.observation_spec.device == torch.device(device)
        assert env_serial.action_spec.device == torch.device(device)
        assert env_serial.reward_spec.device == torch.device(device)
        assert env_serial.device == torch.device(device)
        td_device = env_serial.reset()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rand_step()
        assert td_device.device == torch.device(device), env_serial
        td_device = env_serial.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env_serial

        envs_to_test = [env_parallel]
        if _has_envpool:
            envs_to_test.append(env_multithread)

        for env_test in envs_to_test:
            if open_before:
                td_cpu = env_test.rollout(max_steps=10)
                assert td_cpu.device == torch.device("cpu")
            env_test = env_test.to(device)
            assert env_test.observation_spec.device == torch.device(device)
            assert env_test.action_spec.device == torch.device(device)
            assert env_test.reward_spec.device == torch.device(device)
            assert env_test.device == torch.device(device)
            td_device = env_test.reset()
            assert td_device.device == torch.device(device), env_test
            td_device = env_test.rand_step()
            assert td_device.device == torch.device(device), env_test
            td_device = env_test.rollout(max_steps=10)
            assert td_device.device == torch.device(device), env_test
            env_test.close()

        env_serial.close()
        env0.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device detected")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", [PONG_VERSIONED, PENDULUM_VERSIONED])
    @pytest.mark.parametrize("transformed_in", [True, False])
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_parallel_env_device(
        self, env_name, frame_skip, transformed_in, transformed_out, device
    ):
        # tests creation on device
        torch.manual_seed(0)
        N = 3

        env_parallel, env_serial, env0 = _make_envs(
            env_name,
            frame_skip,
            transformed_in=transformed_in,
            transformed_out=transformed_out,
            device=device,
            N=N,
        )

        assert env0.device == torch.device(device)
        out = env0.rollout(max_steps=20)
        assert out.device == torch.device(device)

        assert env_serial.device == torch.device(device)
        out = env_serial.rollout(max_steps=20)
        assert out.device == torch.device(device)

        assert env_parallel.device == torch.device(device)
        out = env_parallel.rollout(max_steps=20)
        assert out.device == torch.device(device)

        env_parallel.close()
        env_serial.close()
        env0.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
