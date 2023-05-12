# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import importlib

import time
from sys import platform
from typing import Optional, Union

import numpy as np
import pytest
import torch

import torchrl
from _utils_internal import (
    _make_multithreaded_env,
    CARTPOLE_VERSIONED,
    get_available_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
)
from packaging import version
from tensordict.tensordict import assert_allclose_td, TensorDict
from torchrl._utils import implement_for
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.openml import OpenMLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    RenameTransform,
)
from torchrl.envs.libs.brax import _has_brax, BraxEnv
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv, DMControlWrapper
from torchrl.envs.libs.gym import (
    _has_gym,
    _is_from_pixels,
    GymEnv,
    GymWrapper,
    MOGymEnv,
    MOGymWrapper,
)
from torchrl.envs.libs.habitat import _has_habitat, HabitatEnv
from torchrl.envs.libs.jumanji import _has_jumanji, JumanjiEnv
from torchrl.envs.libs.openml import OpenMLEnv
from torchrl.envs.libs.vmas import _has_vmas, VmasEnv, VmasWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType
from torchrl.envs.vec_env import _has_envpool, MultiThreadedEnvWrapper, SerialEnv
from torchrl.modules import ActorCriticOperator, MLP, SafeModule, ValueOperator

_has_d4rl = importlib.util.find_spec("d4rl") is not None

_has_mo = importlib.util.find_spec("mo_gymnasium") is not None

_has_sklearn = importlib.util.find_spec("sklearn") is not None


if _has_gym:
    try:
        import gymnasium as gym
        from gymnasium import __version__ as gym_version

        gym_version = version.parse(gym_version)
        from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
    except ModuleNotFoundError:
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

if _has_vmas:
    import vmas

if _has_envpool:
    import envpool

IS_OSX = platform == "darwin"
RTOL = 1e-1
ATOL = 1e-1


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
class TestGym:
    @pytest.mark.parametrize(
        "env_name",
        [
            PONG_VERSIONED,
            # PENDULUM_VERSIONED,
            HALFCHEETAH_VERSIONED,
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
    def test_gym(self, env_name, frame_skip, from_pixels, pixels_only):
        if env_name == PONG_VERSIONED and not from_pixels:
            # raise pytest.skip("already pixel")
            # we don't skip because that would raise an exception
            return
        elif (
            env_name != PONG_VERSIONED and from_pixels and torch.cuda.device_count() < 1
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

        assert_allclose_td(*tdreset, rtol=RTOL, atol=ATOL)
        assert_allclose_td(*tdrollout, rtol=RTOL, atol=ATOL)
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

        assert_allclose_td(tdreset[0], tdreset2, rtol=RTOL, atol=ATOL)
        assert final_seed0 == final_seed2
        assert_allclose_td(tdrollout[0], rollout2, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize(
        "env_name",
        [
            PONG_VERSIONED,
            # PENDULUM_VERSIONED,
            HALFCHEETAH_VERSIONED,
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
    def test_gym_fake_td(self, env_name, frame_skip, from_pixels, pixels_only):
        if env_name == PONG_VERSIONED and not from_pixels:
            # raise pytest.skip("already pixel")
            return
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

    @pytest.mark.parametrize("frame_skip", [1, 3])
    @pytest.mark.parametrize(
        "from_pixels,pixels_only",
        [
            [False, False],
            [True, True],
            [True, False],
        ],
    )
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_mo(self, frame_skip, from_pixels, pixels_only, wrapper):
        if importlib.util.find_spec("gymnasium") is not None and not _has_mo:
            raise pytest.skip("mo-gym not found")
        else:
            # avoid skipping, which we consider as errors in the gym CI
            return

        def make_env():
            import mo_gymnasium

            if wrapper:
                return MOGymWrapper(
                    mo_gymnasium.make("minecart-v0"),
                    frame_skip=frame_skip,
                    from_pixels=from_pixels,
                    pixels_only=pixels_only,
                )
            else:
                return MOGymEnv(
                    "minecart-v0",
                    frame_skip=frame_skip,
                    from_pixels=from_pixels,
                    pixels_only=pixels_only,
                )

        env = make_env()
        check_env_specs(env)
        env = SerialEnv(2, make_env)
        check_env_specs(env)

    def test_info_reader(self):
        try:
            import gym_super_mario_bros as mario_gym
        except ImportError as err:
            try:
                import gym

                # with 0.26 we must have installed gym_super_mario_bros
                # Since we capture the skips as errors, we raise a skip in this case
                # Otherwise, we just return
                if (
                    version.parse("0.26.0")
                    <= version.parse(gym.__version__)
                    < version.parse("0.27.0")
                ):
                    raise pytest.skip(f"no super mario bros: error=\n{err}")
            except ImportError:
                pass
            return

        env = mario_gym.make("SuperMarioBros-v0", apply_api_compatibility=True)
        env = GymWrapper(env)

        def info_reader(info, tensordict):
            assert isinstance(info, dict)  # failed before bugfix

        env.info_dict_reader = info_reader
        env.reset()
        env.rand_step()
        env.rollout(3)


@implement_for("gym", None, "0.26")
def _make_gym_environment(env_name):  # noqa: F811
    return gym.make(env_name)


@implement_for("gym", "0.26", None)
def _make_gym_environment(env_name):  # noqa: F811
    return gym.make(env_name, render_mode="rgb_array")


@implement_for("gymnasium", "0.27", None)
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


params = []
if _has_dmc:
    params = [
        # [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
    ]
if _has_gym:
    params += [
        # [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": False}],
        [GymEnv, (PONG_VERSIONED,), {}],
    ]


@pytest.mark.skipif(
    IS_OSX,
    reason="rendering unstable on osx, skipping (mujoco.FatalError: gladLoadGL error)",
)
@pytest.mark.parametrize("env_lib,env_args,env_kwargs", params)
def test_td_creation_from_spec(env_lib, env_args, env_kwargs):
    if (
        gym_version < version.parse("0.26.0")
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
    params += [
        # [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
        [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": False}],
        [GymEnv, (PONG_VERSIONED,), {}],
    ]


# @pytest.mark.skipif(IS_OSX, reason="rendering unstable on osx, skipping")
@pytest.mark.parametrize("env_lib,env_args,env_kwargs", params)
@pytest.mark.parametrize("device", get_available_devices())
class TestCollectorLib:
    def test_collector_run(self, env_lib, env_args, env_kwargs, device):
        if not _has_dmc and env_lib is DMControlEnv:
            raise pytest.skip("no dmc")
        if not _has_gym and env_lib is GymEnv:
            raise pytest.skip("no gym")

        from_pixels = env_kwargs.get("from_pixels", False)
        if from_pixels and (not torch.has_cuda or not torch.cuda.device_count()):
            raise pytest.skip("no cuda device")

        env_fn = EnvCreator(lambda: env_lib(*env_args, **env_kwargs, device=device))
        # env = SerialEnv(3, env_fn)
        env = ParallelEnv(3, env_fn)
        frames_per_batch = 21
        collector = MultiaSyncDataCollector(
            create_env_fn=[env, env],
            policy=RandomPolicy(action_spec=env.action_spec),
            total_frames=-1,
            max_frames_per_traj=100,
            frames_per_batch=frames_per_batch,
            init_random_frames=-1,
            reset_at_each_iter=False,
            split_trajs=True,
            devices=[device, device],
            storing_devices=[device, device],
            update_at_each_batch=False,
            exploration_type=ExplorationType.RANDOM,
        )
        for i, _data in enumerate(collector):
            if i == 3:
                break
        collector.shutdown()
        assert _data.shape[1] == -(frames_per_batch // -env.num_workers)
        assert _data.shape[0] == frames_per_batch // _data.shape[1]
        del env


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


@pytest.mark.skipif(not _has_jumanji, reason="jumanji not installed")
@pytest.mark.parametrize(
    "envname",
    [
        "TSP-v1",
        "Snake-v1",
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
            rollout = env.rollout(max_steps=50)
            tdrollout.append(rollout)
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


ENVPOOL_CLASSIC_CONTROL_ENVS = [
    PENDULUM_VERSIONED,
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    CARTPOLE_VERSIONED,
]
ENVPOOL_ATARI_ENVS = []  # PONG_VERSIONED]
ENVPOOL_GYM_ENVS = ENVPOOL_CLASSIC_CONTROL_ENVS + ENVPOOL_ATARI_ENVS
ENVPOOL_DM_ENVS = ["CheetahRun-v1"]
ENVPOOL_ALL_ENVS = ENVPOOL_GYM_ENVS + ENVPOOL_DM_ENVS


@pytest.mark.skipif(not _has_envpool, reason="No envpool library found")
class TestEnvPool:
    @pytest.mark.parametrize("env_name", ENVPOOL_ALL_ENVS)
    def test_env_wrapper_creation(self, env_name):
        env_name = env_name.replace("ALE/", "")  # EnvPool naming convention
        envpool_env = envpool.make(
            task_id=env_name, env_type="gym", num_envs=4, gym_reset_return_info=True
        )
        env = MultiThreadedEnvWrapper(envpool_env)
        env.reset()
        env.rand_step()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize(
        "env_name", ENVPOOL_GYM_ENVS
    )  # Not working for CheetahRun-v1 yet
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_out", [False, True])
    def test_specs(self, env_name, frame_skip, transformed_out, T=10, N=3):
        env_multithreaded = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=transformed_out,
            N=N,
        )
        check_env_specs(env_multithreaded)

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ENVPOOL_ALL_ENVS)
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_out", [False, True])
    def test_env_basic_operation(
        self, env_name, frame_skip, transformed_out, T=10, N=3
    ):
        torch.manual_seed(0)
        env_multithreaded = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=transformed_out,
            N=N,
        )
        td = TensorDict(
            source={"action": env_multithreaded.action_spec.rand()},
            batch_size=[
                N,
            ],
        )
        td1 = env_multithreaded.step(td)
        assert not td1.is_shared()
        assert ("next", "done") in td1.keys(True)
        assert ("next", "reward") in td1.keys(True)

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env_multithreaded.action_spec.rand()},
                batch_size=[N - 1],
            )
            _ = env_multithreaded.step(td)

        _reset = torch.zeros(N, dtype=torch.bool).bernoulli_()
        td_reset = TensorDict(
            source={"_reset": _reset},
            batch_size=[N],
        )
        env_multithreaded.reset(tensordict=td_reset)

        td = env_multithreaded.rollout(
            policy=None, max_steps=T, break_when_any_done=False
        )
        assert (
            td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
        ), f"{td.shape}, {td.get('done').sum(1)}"

        env_multithreaded.close()

    # Don't run on Atari envs because output is uint8
    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ENVPOOL_CLASSIC_CONTROL_ENVS + ENVPOOL_DM_ENVS)
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_env_with_policy(
        self,
        env_name,
        frame_skip,
        transformed_out,
        T=10,
        N=3,
    ):
        class DiscreteChoice(torch.nn.Module):
            """Dummy module producing discrete output. Necessary when the action space is discrete."""

            def __init__(self, out_dim: int, dtype: Optional[Union[torch.dtype, str]]):
                super().__init__()
                self.lin = torch.nn.LazyLinear(out_dim, dtype=dtype)

            def forward(self, x):
                res = torch.argmax(self.lin(x), axis=-1, keepdim=True)
                return res

        env_multithreaded = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=transformed_out,
            N=N,
        )
        if env_name == "CheetahRun-v1":
            in_keys = [("velocity")]
            dtype = torch.float64
        else:
            in_keys = ["observation"]
            dtype = torch.float32

        if env_multithreaded.action_spec.shape:
            module = torch.nn.LazyLinear(
                env_multithreaded.action_spec.shape[-1], dtype=dtype
            )
        else:
            # Action space is discrete
            module = DiscreteChoice(env_multithreaded.action_spec.space.n, dtype=dtype)

        policy = ActorCriticOperator(
            SafeModule(
                spec=None,
                module=torch.nn.LazyLinear(12, dtype=dtype),
                in_keys=in_keys,
                out_keys=["hidden"],
            ),
            SafeModule(
                spec=None,
                module=module,
                in_keys=["hidden"],
                out_keys=["action"],
            ),
            ValueOperator(
                module=MLP(out_features=1, num_cells=[], layer_kwargs={"dtype": dtype}),
                in_keys=["hidden", "action"],
            ),
        )

        td = TensorDict(
            source={"action": env_multithreaded.action_spec.rand()},
            batch_size=[
                N,
            ],
        )

        td1 = env_multithreaded.step(td)
        assert not td1.is_shared()
        assert ("next", "done") in td1.keys(True)
        assert ("next", "reward") in td1.keys(True)

        with pytest.raises(RuntimeError):
            # number of actions does not match number of workers
            td = TensorDict(
                source={"action": env_multithreaded.action_spec.rand()},
                batch_size=[N - 1],
            )
            _ = env_multithreaded.step(td)

        reset = torch.zeros(N, dtype=torch.bool).bernoulli_()
        td_reset = TensorDict(
            source={"_reset": reset},
            batch_size=[N],
        )
        env_multithreaded.reset(tensordict=td_reset)
        td = env_multithreaded.rollout(
            policy=policy, max_steps=T, break_when_any_done=False
        )
        assert (
            td.shape == torch.Size([N, T]) or td.get("done").sum(1).all()
        ), f"{td.shape}, {td.get('done').sum(1)}"

        env_multithreaded.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.parametrize("env_name", ENVPOOL_ALL_ENVS)
    @pytest.mark.parametrize("frame_skip", [4, 1])
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_multithreaded_env_seed(
        self, env_name, frame_skip, transformed_out, seed=100, N=4
    ):
        # Create the first env, set the seed, and perform a sequence of operations
        env = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=True,
            N=N,
        )
        action = env.action_spec.rand()
        env.set_seed(seed)
        td0a = env.reset()
        td1a = env.step(td0a.clone().set("action", action))
        td2a = env.rollout(max_steps=10)

        # Create a new env, set the seed, and repeat same operations
        env = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=True,
            N=N,
        )
        env.set_seed(seed)
        td0b = env.reset()
        td1b = env.step(td0b.clone().set("action", action))
        td2b = env.rollout(max_steps=10)

        # Check that results on two envs are identical
        assert_allclose_td(td0a, td0b.select(*td0a.keys()))
        assert_allclose_td(td1a, td1b)
        assert_allclose_td(td2a, td2b)

        # Check that results are different if seed is different
        # Skip Pong, since there different actions can lead to the same result
        if env_name != PONG_VERSIONED:
            env.set_seed(
                seed=seed + 10,
            )
            td0c = env.reset()
            td1c = env.step(td0c.clone().set("action", action))
            with pytest.raises(AssertionError):
                assert_allclose_td(td0a, td0c.select(*td0a.keys()))
            with pytest.raises(AssertionError):
                assert_allclose_td(td1a, td1c)
        env.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    def test_multithread_env_shutdown(self):
        env = _make_multithreaded_env(
            PENDULUM_VERSIONED,
            1,
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
    @pytest.mark.parametrize("env_name", ENVPOOL_ALL_ENVS)
    @pytest.mark.parametrize("transformed_out", [False, True])
    @pytest.mark.parametrize("open_before", [False, True])
    def test_multithreaded_env_cast(
        self,
        env_name,
        frame_skip,
        transformed_out,
        device,
        open_before,
        T=10,
        N=3,
    ):
        # tests casting to device
        env_multithread = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=transformed_out,
            N=N,
        )
        if open_before:
            td_cpu = env_multithread.rollout(max_steps=10)
            assert td_cpu.device == torch.device("cpu")
        env_multithread = env_multithread.to(device)
        assert env_multithread.observation_spec.device == torch.device(device)
        assert env_multithread.action_spec.device == torch.device(device)
        assert env_multithread.reward_spec.device == torch.device(device)
        assert env_multithread.device == torch.device(device)
        td_device = env_multithread.reset()
        assert td_device.device == torch.device(device), env_multithread
        td_device = env_multithread.rand_step()
        assert td_device.device == torch.device(device), env_multithread
        td_device = env_multithread.rollout(max_steps=10)
        assert td_device.device == torch.device(device), env_multithread
        env_multithread.close()

    @pytest.mark.skipif(not _has_gym, reason="no gym")
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device detected")
    @pytest.mark.parametrize("frame_skip", [4])
    @pytest.mark.parametrize("device", [0])
    @pytest.mark.parametrize("env_name", ENVPOOL_ALL_ENVS)
    @pytest.mark.parametrize("transformed_out", [True, False])
    def test_env_device(self, env_name, frame_skip, transformed_out, device):
        # tests creation on device
        torch.manual_seed(0)
        N = 3

        env_multithreaded = _make_multithreaded_env(
            env_name,
            frame_skip,
            transformed_out=transformed_out,
            device=device,
            N=N,
        )

        assert env_multithreaded.device == torch.device(device)
        out = env_multithreaded.rollout(max_steps=20)
        assert out.device == torch.device(device)

        env_multithreaded.close()


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
    @pytest.mark.parametrize(
        "requires_grad",
        [
            True,
            False,
        ],
    )
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
        action = torch.randn(env.action_spec.shape)
        action.requires_grad_(True)
        td1["action"] = action
        td2 = env.step(td1)
        td2[("next", "reward")].mean().backward()
        env.close()
        del env

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_brax_parallel(self, envname, batch_size, parallel, n=1):
        def make_brax():
            env = BraxEnv(envname, batch_size=batch_size, requires_grad=False)
            env.set_seed(1)
            return env

        if parallel:
            env = ParallelEnv(n, make_brax)
        else:
            env = SerialEnv(n, make_brax)
        check_env_specs(env)
        tensordict = env.rollout(3)
        assert tensordict.shape == torch.Size([n, *batch_size, 3])


@pytest.mark.skipif(not _has_vmas, reason="vmas not installed")
class TestVmas:
    @pytest.mark.parametrize("scenario_name", torchrl.envs.libs.vmas._get_envs())
    def test_all_vmas_scenarios(self, scenario_name):
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=4,
        )
        env.set_seed(0)
        env.reset()
        env.rollout(10)

    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_seeding(self, scenario_name):
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=4,
            )
            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=10))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize(
        "batch_size", [(), (12,), (12, 2), (12, 3), (12, 3, 1), (12, 3, 4)]
    )
    @pytest.mark.parametrize("scenario_name", torchrl.envs.libs.vmas._get_envs())
    def test_vmas_batch_size_error(self, scenario_name, batch_size):
        num_envs = 12
        n_agents = 2
        if len(batch_size) > 1:
            with pytest.raises(
                TypeError,
                match="Batch size used in constructor is not compatible with vmas.",
            ):
                _ = VmasEnv(
                    scenario=scenario_name,
                    num_envs=num_envs,
                    n_agents=n_agents,
                    batch_size=batch_size,
                )
        elif len(batch_size) == 1 and batch_size != (num_envs,):
            with pytest.raises(
                TypeError,
                match="Batch size used in constructor does not match vmas batch size.",
            ):
                _ = VmasEnv(
                    scenario=scenario_name,
                    num_envs=num_envs,
                    n_agents=n_agents,
                    batch_size=batch_size,
                )
        else:
            _ = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                batch_size=batch_size,
            )

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_batch_size(self, scenario_name, num_envs, n_agents):
        torch.manual_seed(0)
        n_rollout_samples = 5
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
        )
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=n_rollout_samples)
        env.close()

        assert tdreset.batch_size == (num_envs,)
        assert tdreset["observation"].shape[1] == env.n_agents
        assert tdreset["done"].shape[1] == env.n_agents

        assert tdrollout.batch_size == (num_envs, n_rollout_samples)
        assert tdrollout["observation"].shape[2] == env.n_agents
        assert tdrollout["next", "reward"].shape[2] == env.n_agents
        assert tdrollout["action"].shape[2] == env.n_agents
        assert tdrollout["done"].shape[2] == env.n_agents
        del env

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_spec_rollout(
        self, scenario_name, num_envs, n_agents, continuous_actions
    ):
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
            continuous_actions=continuous_actions,
        )
        wrapped = VmasWrapper(
            vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                continuous_actions=continuous_actions,
            )
        )
        for e in [env, wrapped]:
            e.set_seed(0)
            check_env_specs(e)
            del e

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize("scenario_name", torchrl.envs.libs.vmas._get_envs())
    def test_vmas_repr(self, scenario_name, num_envs, n_agents):
        if n_agents == 1 and scenario_name == "balance":
            return
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
        )
        assert str(env) == (
            f"{VmasEnv.__name__}(num_envs={num_envs}, n_agents={env.n_agents},"
            f" batch_size={torch.Size((num_envs,))}, device={env.device}) (scenario={scenario_name})"
        )

    @pytest.mark.parametrize("num_envs", [1, 10])
    @pytest.mark.parametrize("n_workers", [1, 3])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_parallel(
        self,
        scenario_name,
        num_envs,
        n_workers,
        continuous_actions,
        n_agents=5,
        n_rollout_samples=3,
    ):
        torch.manual_seed(0)

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                continuous_actions=continuous_actions,
            )
            env.set_seed(0)
            return env

        env = ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert tensordict.shape == torch.Size(
            [n_workers, list(env.num_envs)[0], n_rollout_samples]
        )

    @pytest.mark.parametrize("num_envs", [1, 10])
    @pytest.mark.parametrize("n_workers", [1, 3])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_reset(
        self,
        scenario_name,
        num_envs,
        n_workers,
        n_agents=5,
        n_rollout_samples=3,
        max_steps=3,
    ):
        torch.manual_seed(0)

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                max_steps=max_steps,
            )
            env.set_seed(0)
            return env

        env = ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert (
            tensordict["next", "done"]
            .sum(
                tuple(range(tensordict.batch_dims, tensordict["next", "done"].ndim)),
                dtype=torch.bool,
            )[..., -1]
            .all()
        )

        _reset = env.done_spec.rand()
        while not _reset.any():
            _reset = env.done_spec.rand()

        tensordict = env.reset(
            TensorDict({"_reset": _reset}, batch_size=env.batch_size, device=env.device)
        )
        assert not tensordict["done"][_reset].all().item()
        # vmas resets all the agent dimension if only one of the agents needs resetting
        # thus, here we check that where we did not reset any agent, all agents are still done
        assert tensordict["done"].all(dim=2)[~_reset.any(dim=2)].all().item()

    @pytest.mark.skipif(len(get_available_devices()) < 2, reason="not enough devices")
    @pytest.mark.parametrize("first", [0, 1])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_to_device(self, scenario_name: str, first: int):
        torch.manual_seed(0)
        devices = get_available_devices()

        def make_vmas():
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=7,
                n_agents=3,
                seed=0,
                device=devices[first],
            )
            return env

        env = ParallelEnv(2, make_vmas)

        assert env.rollout(max_steps=3).device == devices[first]

        env.to(devices[1 - first])

        assert env.rollout(max_steps=3).device == devices[1 - first]


@pytest.mark.skipif(not _has_d4rl, reason="D4RL not found")
class TestD4RL:
    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    def test_terminate_on_end(self, task):
        t0 = time.time()
        data_true = D4RLExperienceReplay(
            task,
            split_trajs=True,
            from_env=False,
            terminate_on_end=True,
            batch_size=2,
            use_timeout_as_done=False,
        )
        _ = D4RLExperienceReplay(
            task,
            split_trajs=True,
            from_env=False,
            terminate_on_end=False,
            batch_size=2,
            use_timeout_as_done=False,
        )
        data_from_env = D4RLExperienceReplay(
            task,
            split_trajs=True,
            from_env=True,
            batch_size=2,
            use_timeout_as_done=False,
        )
        keys = set(data_from_env._storage._storage.keys(True, True))
        keys = keys.intersection(data_true._storage._storage.keys(True, True))
        assert_allclose_td(
            data_true._storage._storage.select(*keys),
            data_from_env._storage._storage.select(*keys),
        )

    @pytest.mark.parametrize(
        "task",
        [
            # "antmaze-medium-play-v0",
            # "hammer-cloned-v1",
            # "maze2d-open-v0",
            # "maze2d-open-dense-v0",
            # "relocate-human-v1",
            "walker2d-medium-replay-v2",
            # "ant-medium-v2",
            # # "flow-merge-random-v0",
            # "kitchen-partial-v0",
            # # "carla-town-v0",
        ],
    )
    def test_d4rl_dummy(self, task):
        t0 = time.time()
        _ = D4RLExperienceReplay(task, split_trajs=True, from_env=True, batch_size=2)
        print(f"completed test after {time.time()-t0}s")

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("split_trajs", [True, False])
    @pytest.mark.parametrize("from_env", [True, False])
    def test_dataset_build(self, task, split_trajs, from_env):
        t0 = time.time()
        data = D4RLExperienceReplay(
            task, split_trajs=split_trajs, from_env=from_env, batch_size=2
        )
        sample = data.sample()
        env = GymWrapper(gym.make(task))
        rollout = env.rollout(2)
        for key in rollout.keys(True, True):
            sim = rollout[key]
            offline = sample[key]
            assert sim.dtype == offline.dtype, key
            assert sim.shape[-1] == offline.shape[-1], key
        print(f"completed test after {time.time()-t0}s")

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("split_trajs", [True, False])
    def test_d4rl_iteration(self, task, split_trajs):
        t0 = time.time()
        batch_size = 3
        data = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=False,
            terminate_on_end=True,
            batch_size=batch_size,
            sampler=SamplerWithoutReplacement(drop_last=True),
        )
        i = 0
        for sample in data:  # noqa: B007
            i += 1
        assert len(data) // i == batch_size
        print(f"completed test after {time.time()-t0}s")


@pytest.mark.skipif(not _has_sklearn, reason="Scikit-learn not found")
@pytest.mark.parametrize(
    "dataset",
    [
        "adult_num",
        "adult_onehot",
        "mushroom_num",
        "mushroom_onehot",
        "covertype",
        "shuttle",
        "magic",
    ],
)
class TestOpenML:
    @pytest.mark.parametrize("batch_size", [(), (2,), (2, 3)])
    def test_env(self, dataset, batch_size):
        env = OpenMLEnv(dataset, batch_size=batch_size)
        td = env.reset()
        assert td.shape == torch.Size(batch_size)
        td = env.rand_step(td)
        assert td.shape == torch.Size(batch_size)
        assert "index" not in td.keys()
        check_env_specs(env)

    def test_data(self, dataset):
        data = OpenMLExperienceReplay(
            dataset,
            batch_size=2048,
            transform=Compose(
                RenameTransform(["X"], ["observation"]),
                DoubleToFloat(["observation"]),
            ),
        )
        # check that dataset eventually runs out
        for i, _ in enumerate(data):  # noqa: B007
            continue
        assert len(data) // 2048 in (i, i - 1)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
