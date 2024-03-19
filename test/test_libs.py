# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
from contextlib import nullcontext
from pathlib import Path

from torchrl._utils import logger as torchrl_logger

from torchrl.data.datasets.gen_dgrl import GenDGRLExperienceReplay

from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.modules import MaskedCategorical

_has_isaac = importlib.util.find_spec("isaacgym") is not None

if _has_isaac:
    # isaac gym asks to be imported before torch...
    import isaacgym  # noqa
    import isaacgymenvs  # noqa
    from torchrl.envs.libs.isaacgym import IsaacGymEnv

import argparse
import importlib

import time
from sys import platform
from typing import Optional, Union

import numpy as np
import pytest
import torch

from _utils_internal import (
    _make_multithreaded_env,
    CARTPOLE_VERSIONED,
    get_available_devices,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    rollout_consistency_assertion,
)
from packaging import version
from tensordict import assert_allclose_td, LazyStackedTensorDict, TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn
from torchrl._utils import implement_for
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    ReplayBuffer,
    ReplayBufferEnsemble,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.data.datasets.atari_dqn import AtariDQNExperienceReplay
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.datasets.openml import OpenMLExperienceReplay
from torchrl.data.datasets.openx import OpenXExperienceReplay
from torchrl.data.datasets.roboset import RobosetExperienceReplay
from torchrl.data.datasets.vd4rl import VD4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    EnvBase,
    EnvCreator,
    RemoveEmptySpecs,
    RenameTransform,
)
from torchrl.envs.batched_envs import SerialEnv
from torchrl.envs.libs.brax import _has_brax, BraxEnv
from torchrl.envs.libs.dm_control import _has_dmc, DMControlEnv, DMControlWrapper
from torchrl.envs.libs.envpool import _has_envpool, MultiThreadedEnvWrapper
from torchrl.envs.libs.gym import (
    _gym_to_torchrl_spec_transform,
    _has_gym,
    _is_from_pixels,
    _torchrl_to_gym_spec_transform,
    gym_backend,
    GymEnv,
    GymWrapper,
    MOGymEnv,
    MOGymWrapper,
    set_gym_backend,
)
from torchrl.envs.libs.habitat import _has_habitat, HabitatEnv
from torchrl.envs.libs.jumanji import _has_jumanji, JumanjiEnv
from torchrl.envs.libs.openml import OpenMLEnv
from torchrl.envs.libs.pettingzoo import _has_pettingzoo, PettingZooEnv
from torchrl.envs.libs.robohive import _has_robohive, RoboHiveEnv
from torchrl.envs.libs.smacv2 import _has_smacv2, SMACv2Env
from torchrl.envs.libs.vmas import _has_vmas, VmasEnv, VmasWrapper
from torchrl.envs.utils import (
    check_env_specs,
    ExplorationType,
    MarlGroupMapType,
    RandomPolicy,
)
from torchrl.modules import ActorCriticOperator, MLP, SafeModule, ValueOperator

_has_d4rl = importlib.util.find_spec("d4rl") is not None

_has_mo = importlib.util.find_spec("mo_gymnasium") is not None

_has_sklearn = importlib.util.find_spec("sklearn") is not None

_has_gym_robotics = importlib.util.find_spec("gymnasium_robotics") is not None

_has_minari = importlib.util.find_spec("minari") is not None

_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
_has_gym_regular = importlib.util.find_spec("gym") is not None
if _has_gymnasium:
    set_gym_backend("gymnasium").set()
    import gymnasium

    assert gym_backend() is gymnasium
elif _has_gym:
    set_gym_backend("gym").set()
    import gym

    assert gym_backend() is gym


def get_gym_pixel_wrapper():
    try:
        # works whenever gym_version > version.parse("0.19")
        PixelObservationWrapper = gym_backend(
            "wrappers.pixel_observation"
        ).PixelObservationWrapper
    except Exception as err:
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )
    return PixelObservationWrapper


if _has_gym:
    try:
        from gymnasium import __version__ as gym_version

        gym_version = version.parse(gym_version)
    except ModuleNotFoundError:
        from gym import __version__ as gym_version

        gym_version = version.parse(gym_version)

if _has_dmc:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels

if _has_vmas:
    import vmas


if _has_envpool:
    import envpool

_has_pytree = True
try:
    from torch.utils._pytree import tree_flatten
except ImportError:
    _has_pytree = False
IS_OSX = platform == "darwin"
RTOL = 1e-1
ATOL = 1e-1


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
class TestGym:
    class DummyEnv(EnvBase):
        def __init__(self, arg1, *, arg2, **kwargs):
            super().__init__(**kwargs)

            assert arg1 == 1
            assert arg2 == 2

            self.observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec((*self.batch_size, 3)),
                other=CompositeSpec(
                    another_other=UnboundedContinuousTensorSpec((*self.batch_size, 3)),
                    shape=self.batch_size,
                ),
                shape=self.batch_size,
            )
            self.action_spec = UnboundedContinuousTensorSpec((*self.batch_size, 3))
            self.done_spec = DiscreteTensorSpec(
                2, (*self.batch_size, 1), dtype=torch.bool
            )
            self.full_done_spec["truncated"] = self.full_done_spec["terminated"].clone()

        def _reset(self, tensordict):
            return self.observation_spec.rand()

        def _step(self, tensordict):
            action = tensordict.get("action")
            return TensorDict(
                {
                    "observation": action.clone(),
                    "other": {"another_other": torch.zeros_like(action)},
                    "reward": action.sum(-1, True),
                    "done": ~action.any(-1, True),
                    "terminated": ~action.any(-1, True),
                    "truncated": torch.zeros((*self.batch_size, 1), dtype=torch.bool),
                },
                batch_size=[],
            )

        def _set_seed(self, seed):
            return seed + 1

    @implement_for("gym", None, "0.18")
    def _make_spec(self, batch_size, cat, cat_shape, multicat, multicat_shape):
        return CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(*batch_size, 1)),
            b=CompositeSpec(
                c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size
            ),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=BoundedTensorSpec(-3, 4, shape=(*batch_size, 1)),
            # g=UnboundedDiscreteTensorSpec(shape=(*batch_size, 1), dtype=torch.long),
            h=BinaryDiscreteTensorSpec(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @implement_for("gym", "0.18", None)
    def _make_spec(  # noqa: F811
        self, batch_size, cat, cat_shape, multicat, multicat_shape
    ):
        return CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(*batch_size, 1)),
            b=CompositeSpec(
                c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size
            ),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=BoundedTensorSpec(-3, 4, shape=(*batch_size, 1)),
            g=UnboundedDiscreteTensorSpec(shape=(*batch_size, 1), dtype=torch.long),
            h=BinaryDiscreteTensorSpec(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @implement_for("gymnasium")
    def _make_spec(  # noqa: F811
        self, batch_size, cat, cat_shape, multicat, multicat_shape
    ):
        return CompositeSpec(
            a=UnboundedContinuousTensorSpec(shape=(*batch_size, 1)),
            b=CompositeSpec(
                c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size
            ),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=BoundedTensorSpec(-3, 4, shape=(*batch_size, 1)),
            g=UnboundedDiscreteTensorSpec(shape=(*batch_size, 1), dtype=torch.long),
            h=BinaryDiscreteTensorSpec(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @pytest.mark.parametrize("categorical", [True, False])
    def test_gym_spec_cast(self, categorical):

        batch_size = [3, 4]
        cat = DiscreteTensorSpec if categorical else OneHotDiscreteTensorSpec
        cat_shape = batch_size if categorical else (*batch_size, 5)
        multicat = (
            MultiDiscreteTensorSpec if categorical else MultiOneHotDiscreteTensorSpec
        )
        multicat_shape = 2 if categorical else 5
        spec = self._make_spec(batch_size, cat, cat_shape, multicat, multicat_shape)
        recon = _gym_to_torchrl_spec_transform(
            _torchrl_to_gym_spec_transform(
                spec, categorical_action_encoding=categorical
            ),
            categorical_action_encoding=categorical,
            batch_size=batch_size,
        )
        for (key0, spec0), (key1, spec1) in zip(
            spec.items(True, True), recon.items(True, True)
        ):
            assert spec0 == spec1, (key0, key1, spec0, spec1)
        assert spec == recon
        assert recon.shape == spec.shape

    _BACKENDS = [None]
    if _has_gymnasium:
        _BACKENDS += ["gymnasium"]
    if _has_gym_regular:
        _BACKENDS += ["gym"]

    @pytest.mark.skipif(not _has_pytree, reason="pytree needed for torchrl_to_gym test")
    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("numpy", [True, False])
    def test_torchrl_to_gym(self, backend, numpy):
        from torchrl.envs.libs.gym import gym_backend, set_gym_backend

        gb = gym_backend()
        try:
            EnvBase.register_gym(
                f"Dummy-{numpy}-{backend}-v0",
                entry_point=self.DummyEnv,
                to_numpy=numpy,
                backend=backend,
                arg1=1,
                arg2=2,
            )

            with set_gym_backend(backend) if backend is not None else nullcontext():
                envgym = gym_backend().make(f"Dummy-{numpy}-{backend}-v0")
                envgym.reset()
                obs, *_ = envgym.step(envgym.action_space.sample())
                assert "observation" in obs
                assert "other" in obs
                if numpy:
                    assert all(
                        isinstance(val, np.ndarray) for val in tree_flatten(obs)[0]
                    )
                else:
                    assert all(
                        isinstance(val, torch.Tensor) for val in tree_flatten(obs)[0]
                    )

                # with a transform
                transform = Compose(
                    CatTensors(["observation", ("other", "another_other")]),
                    RemoveEmptySpecs(),
                )
                envgym = gym_backend().make(
                    f"Dummy-{numpy}-{backend}-v0",
                    transform=transform,
                )
                envgym.reset()
                obs, *_ = envgym.step(envgym.action_space.sample())
                assert "observation_other" not in obs
                assert "observation" not in obs
                assert "other" not in obs
                if numpy:
                    assert all(
                        isinstance(val, np.ndarray) for val in tree_flatten(obs)[0]
                    )
                else:
                    assert all(
                        isinstance(val, torch.Tensor) for val in tree_flatten(obs)[0]
                    )

            # register with transform
            transform = Compose(
                CatTensors(["observation", ("other", "another_other")]),
                RemoveEmptySpecs(),
            )
            EnvBase.register_gym(
                f"Dummy-{numpy}-{backend}-transform-v0",
                entry_point=self.DummyEnv,
                backend=backend,
                to_numpy=numpy,
                arg1=1,
                arg2=2,
                transform=transform,
            )

            with set_gym_backend(backend) if backend is not None else nullcontext():
                envgym = gym_backend().make(f"Dummy-{numpy}-{backend}-transform-v0")
                envgym.reset()
                obs, *_ = envgym.step(envgym.action_space.sample())
                assert "observation_other" not in obs
                assert "observation" not in obs
                assert "other" not in obs
                if numpy:
                    assert all(
                        isinstance(val, np.ndarray) for val in tree_flatten(obs)[0]
                    )
                else:
                    assert all(
                        isinstance(val, torch.Tensor) for val in tree_flatten(obs)[0]
                    )

            # register with transform
            EnvBase.register_gym(
                f"Dummy-{numpy}-{backend}-noarg-v0",
                entry_point=self.DummyEnv,
                backend=backend,
                to_numpy=numpy,
            )
            with set_gym_backend(backend) if backend is not None else nullcontext():
                with pytest.raises(AssertionError):
                    envgym = gym_backend().make(
                        f"Dummy-{numpy}-{backend}-noarg-v0", arg1=None, arg2=None
                    )
                envgym = gym_backend().make(
                    f"Dummy-{numpy}-{backend}-noarg-v0", arg1=1, arg2=2
                )

            # Get info dict
            gym_info_at_reset = version.parse(
                gym_backend().__version__
            ) >= version.parse("0.26.0")
            with set_gym_backend(backend) if backend is not None else nullcontext():
                envgym = gym_backend().make(
                    f"Dummy-{numpy}-{backend}-noarg-v0",
                    arg1=1,
                    arg2=2,
                    info_keys=("other",),
                )
                if gym_info_at_reset:
                    out, info = envgym.reset()
                    if numpy:
                        assert all(
                            isinstance(val, np.ndarray)
                            for val in tree_flatten((obs, info))[0]
                        )
                    else:
                        assert all(
                            isinstance(val, torch.Tensor)
                            for val in tree_flatten((obs, info))[0]
                        )
                else:
                    out = envgym.reset()
                    info = {}
                    if numpy:
                        assert all(
                            isinstance(val, np.ndarray)
                            for val in tree_flatten((obs, info))[0]
                        )
                    else:
                        assert all(
                            isinstance(val, torch.Tensor)
                            for val in tree_flatten((obs, info))[0]
                        )
                assert "observation" in out
                assert "other" not in out

                if gym_info_at_reset:
                    assert "other" in info

                out, *_, info = envgym.step(envgym.action_space.sample())
                assert "observation" in out
                assert "other" not in out
                assert "other" in info
                if numpy:
                    assert all(
                        isinstance(val, np.ndarray)
                        for val in tree_flatten((obs, info))[0]
                    )
                else:
                    assert all(
                        isinstance(val, torch.Tensor)
                        for val in tree_flatten((obs, info))[0]
                    )

            EnvBase.register_gym(
                f"Dummy-{numpy}-{backend}-info-v0",
                entry_point=self.DummyEnv,
                backend=backend,
                to_numpy=numpy,
                info_keys=("other",),
            )
            with set_gym_backend(backend) if backend is not None else nullcontext():
                envgym = gym_backend().make(
                    f"Dummy-{numpy}-{backend}-info-v0", arg1=1, arg2=2
                )
                if gym_info_at_reset:
                    out, info = envgym.reset()
                    if numpy:
                        assert all(
                            isinstance(val, np.ndarray)
                            for val in tree_flatten((obs, info))[0]
                        )
                    else:
                        assert all(
                            isinstance(val, torch.Tensor)
                            for val in tree_flatten((obs, info))[0]
                        )
                else:
                    out = envgym.reset()
                    info = {}
                    if numpy:
                        assert all(
                            isinstance(val, np.ndarray)
                            for val in tree_flatten((obs, info))[0]
                        )
                    else:
                        assert all(
                            isinstance(val, torch.Tensor)
                            for val in tree_flatten((obs, info))[0]
                        )
                assert "observation" in out
                assert "other" not in out

                if gym_info_at_reset:
                    assert "other" in info

                out, *_, info = envgym.step(envgym.action_space.sample())
                assert "observation" in out
                assert "other" not in out
                assert "other" in info
                if numpy:
                    assert all(
                        isinstance(val, np.ndarray)
                        for val in tree_flatten((obs, info))[0]
                    )
                else:
                    assert all(
                        isinstance(val, torch.Tensor)
                        for val in tree_flatten((obs, info))[0]
                    )
        finally:
            set_gym_backend(gb).set()

    @pytest.mark.parametrize(
        "env_name",
        [
            HALFCHEETAH_VERSIONED(),
            PONG_VERSIONED(),
            # PENDULUM_VERSIONED,
        ],
    )
    @pytest.mark.parametrize("frame_skip", [1, 3])
    @pytest.mark.parametrize(
        "from_pixels,pixels_only",
        [
            [True, True],
            [True, False],
            [False, False],
        ],
    )
    def test_gym(self, env_name, frame_skip, from_pixels, pixels_only):

        if env_name == PONG_VERSIONED() and not from_pixels:
            # raise pytest.skip("already pixel")
            # we don't skip because that would raise an exception
            return
        elif (
            env_name != PONG_VERSIONED()
            and from_pixels
            and torch.cuda.device_count() < 1
        ):
            raise pytest.skip("no cuda device")

        def non_null_obs(batched_td):
            if from_pixels:
                pix_norm = batched_td.get("pixels").flatten(-3, -1).float().norm(dim=-1)
                pix_norm_next = (
                    batched_td.get(("next", "pixels"))
                    .flatten(-3, -1)
                    .float()
                    .norm(dim=-1)
                )
                idx = (pix_norm > 1) & (pix_norm_next > 1)
                # eliminate batch size: all idx must be True (otherwise one could be filled with 0s)
                while idx.ndim > 1:
                    idx = idx.all(0)
                idx = idx.nonzero().squeeze(-1)
                assert idx.numel(), "Did not find pixels with norm > 1"
                return idx
            return slice(None)

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
            rollout = env0.rollout(max_steps=50)
            tdrollout.append(rollout)
            assert env0.from_pixels is from_pixels
            env0.close()
            env_type = type(env0._env)

        assert_allclose_td(*tdreset, rtol=RTOL, atol=ATOL)
        tdrollout = torch.stack(tdrollout, 0)

        # custom filtering of non-null obs: mujoco rendering sometimes fails
        # and renders black images. To counter this in the tests, we select
        # tensordicts with all non-null observations
        idx = non_null_obs(tdrollout)
        assert_allclose_td(
            tdrollout[0][..., idx], tdrollout[1][..., idx], rtol=RTOL, atol=ATOL
        )
        final_seed0, final_seed1 = final_seed
        assert final_seed0 == final_seed1

        if env_name == PONG_VERSIONED():
            base_env = gym_backend().make(env_name, frameskip=frame_skip)
            frame_skip = 1
        else:
            base_env = _make_gym_environment(env_name)

        if from_pixels and not _is_from_pixels(base_env):
            PixelObservationWrapper = get_gym_pixel_wrapper()
            base_env = PixelObservationWrapper(base_env, pixels_only=pixels_only)
        assert type(base_env) is env_type

        # Compare GymEnv output with GymWrapper output
        env1 = GymWrapper(base_env, frame_skip=frame_skip)
        assert env0.get_library_name(env0._env) == env1.get_library_name(env1._env)
        # check that we didn't do more wrapping
        assert type(env0._env) == type(env1._env)  # noqa: E721
        assert env0.output_spec == env1.output_spec
        assert env0.input_spec == env1.input_spec
        del env0
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
        # same magic trick for mujoco as above
        tdrollout = torch.stack([tdrollout[0], rollout2], 0)
        idx = non_null_obs(tdrollout)
        assert_allclose_td(
            tdrollout[0][..., idx], tdrollout[1][..., idx], rtol=RTOL, atol=ATOL
        )

    @pytest.mark.parametrize(
        "env_name",
        [
            PONG_VERSIONED(),
            # PENDULUM_VERSIONED,
            HALFCHEETAH_VERSIONED(),
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
        if env_name == PONG_VERSIONED() and not from_pixels:
            # raise pytest.skip("already pixel")
            return
        elif (
            env_name != PONG_VERSIONED()
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

    def test_info_reader_mario(self):
        try:
            import gym_super_mario_bros as mario_gym
        except ImportError as err:
            try:
                gym = gym_backend()

                # with 0.26 we must have installed gym_super_mario_bros
                # Since we capture the skips as errors, we raise a skip in this case
                # Otherwise, we just return
                gym_version = version.parse(gym.__version__)
                if version.parse(
                    "0.26.0"
                ) <= gym_version and gym_version < version.parse("0.27"):
                    raise pytest.skip(f"no super mario bros: error=\n{err}")
            except ImportError:
                pass
            return

        gb = gym_backend()
        try:
            with set_gym_backend("gym"):
                env = mario_gym.make("SuperMarioBros-v0")
                env = GymWrapper(env)
                check_env_specs(env)

                def info_reader(info, tensordict):
                    assert isinstance(info, dict)  # failed before bugfix

                env.info_dict_reader = info_reader
                check_env_specs(env)
        finally:
            set_gym_backend(gb).set()

    @implement_for("gymnasium")
    def test_one_hot_and_categorical(self):
        # tests that one-hot and categorical work ok when an integer is expected as action
        cliff_walking = GymEnv("CliffWalking-v0", categorical_action_encoding=True)
        cliff_walking.rollout(10)
        check_env_specs(cliff_walking)

        cliff_walking = GymEnv("CliffWalking-v0", categorical_action_encoding=False)
        cliff_walking.rollout(10)
        check_env_specs(cliff_walking)

    @implement_for("gym")
    def test_one_hot_and_categorical(self):  # noqa: F811
        # we do not skip (bc we may want to make sure nothing is skipped)
        # but CliffWalking-v0 in earlier Gym versions uses np.bool, which
        # was deprecated after np 1.20, and we don't want to install multiple np
        # versions.
        return

    @implement_for("gymnasium")
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_wrapper(self, envname):
        import gymnasium

        # we can't use parametrize with implement_for
        env = GymWrapper(
            gymnasium.vector.SyncVectorEnv(
                2 * [lambda envname=envname: gymnasium.make(envname)]
            )
        )
        assert env.batch_size == torch.Size([2])
        check_env_specs(env)
        env = GymWrapper(
            gymnasium.vector.AsyncVectorEnv(
                2 * [lambda envname=envname: gymnasium.make(envname)]
            )
        )
        assert env.batch_size == torch.Size([2])
        check_env_specs(env)

    @implement_for("gymnasium")
    # this env has Dict-based observation which is a nice thing to test
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_env(self, envname):
        gb = gym_backend()
        try:
            with set_gym_backend("gymnasium"):
                env = GymEnv(envname, num_envs=2, from_pixels=False)
                env.set_seed(0)
                assert env.get_library_name(env._env) == "gymnasium"
            # rollouts can be executed without decorator
            check_env_specs(env)
            rollout = env.rollout(100, break_when_any_done=False)
            for obs_key in env.observation_spec.keys(True, True):
                rollout_consistency_assertion(
                    rollout,
                    done_key="done",
                    observation_key=obs_key,
                    done_strict="CartPole" in envname,
                )
            env.close()
            del env
        finally:
            set_gym_backend(gb).set()

    @implement_for("gym", "0.18")
    @pytest.mark.parametrize(
        "envname",
        ["CartPole-v1", "HalfCheetah-v4"],
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_wrapper(self, envname):  # noqa: F811
        gym = gym_backend()
        # we can't use parametrize with implement_for
        for envname in ["CartPole-v1", "HalfCheetah-v4"]:
            env = GymWrapper(
                gym.vector.SyncVectorEnv(
                    2 * [lambda envname=envname: gym.make(envname)]
                )
            )
            assert env.batch_size == torch.Size([2])
            check_env_specs(env)
            env = GymWrapper(
                gym.vector.AsyncVectorEnv(
                    2 * [lambda envname=envname: gym.make(envname)]
                )
            )
            assert env.batch_size == torch.Size([2])
            check_env_specs(env)
            env.close()
            del env

    @implement_for("gym", "0.18")
    @pytest.mark.parametrize(
        "envname",
        ["cp", "hc"],
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_env(self, envname):  # noqa: F811
        gb = gym_backend()
        try:
            with set_gym_backend("gym"):
                if envname == "hc":
                    envname = HALFCHEETAH_VERSIONED()
                else:
                    envname = CARTPOLE_VERSIONED()
                env = GymEnv(envname, num_envs=2, from_pixels=False)
                env.set_seed(0)
                assert env.get_library_name(env._env) == "gym"
            # rollouts can be executed without decorator
            check_env_specs(env)
            rollout = env.rollout(100, break_when_any_done=False)
            for obs_key in env.observation_spec.keys(True, True):
                rollout_consistency_assertion(
                    rollout,
                    done_key="done",
                    observation_key=obs_key,
                    done_strict="CartPole" in envname,
                )
            env.close()
            del env
            if envname != "CartPole-v1":
                with set_gym_backend("gym"):
                    env = GymEnv(envname, num_envs=2, from_pixels=True)
                    env.set_seed(0)
                # rollouts can be executed without decorator
                check_env_specs(env)
                env.close()
                del env
        finally:
            set_gym_backend(gb).set()

    @implement_for("gym", None, "0.18")
    @pytest.mark.parametrize(
        "envname",
        ["CartPole-v1", "HalfCheetah-v4"],
    )
    def test_vecenvs_wrapper(self, envname):  # noqa: F811
        # skipping tests for older versions of gym
        ...

    @implement_for("gym", None, "0.18")
    @pytest.mark.parametrize(
        "envname",
        ["CartPole-v1", "HalfCheetah-v4"],
    )
    def test_vecenvs_env(self, envname):  # noqa: F811
        # skipping tests for older versions of gym
        ...

    @implement_for("gym", None, "0.26")
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_gym_output_num(self, wrapper):
        # gym has 4 outputs, no truncation
        gym = gym_backend()
        try:
            if wrapper:
                env = GymWrapper(gym.make(PENDULUM_VERSIONED()))
            else:
                with set_gym_backend("gym"):
                    env = GymEnv(PENDULUM_VERSIONED())
            # truncated is read from the info
            assert "truncated" in env.done_keys
            assert "terminated" in env.done_keys
            assert "done" in env.done_keys
            check_env_specs(env)
        finally:
            set_gym_backend(gym).set()

    @implement_for("gym", "0.26")
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_gym_output_num(self, wrapper):  # noqa: F811
        # gym has 5 outputs, with truncation
        gym = gym_backend()
        try:
            if wrapper:
                env = GymWrapper(gym.make(PENDULUM_VERSIONED()))
            else:
                with set_gym_backend("gym"):
                    env = GymEnv(PENDULUM_VERSIONED())
            assert "truncated" in env.done_keys
            assert "terminated" in env.done_keys
            assert "done" in env.done_keys
            check_env_specs(env)

            if wrapper:
                # let's further test with a wrapper that exposes the env with old API
                from gym.wrappers.compatibility import EnvCompatibility

                with pytest.raises(
                    ValueError,
                    match="GymWrapper does not support the gym.wrapper.compatibility.EnvCompatibility",
                ):
                    GymWrapper(EnvCompatibility(gym.make("CartPole-v1")))
        finally:
            set_gym_backend(gym).set()

    @implement_for("gymnasium")
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_gym_output_num(self, wrapper):  # noqa: F811
        # gym has 5 outputs, with truncation
        gym = gym_backend()
        try:
            if wrapper:
                env = GymWrapper(gym.make(PENDULUM_VERSIONED()))
            else:
                with set_gym_backend("gymnasium"):
                    env = GymEnv(PENDULUM_VERSIONED())
            assert "truncated" in env.done_keys
            assert "terminated" in env.done_keys
            assert "done" in env.done_keys
            check_env_specs(env)
        finally:
            set_gym_backend(gym).set()

    def test_gym_gymnasium_parallel(self, maybe_fork_ParallelEnv):
        # tests that both gym and gymnasium work with wrappers without
        # decorating with set_gym_backend during execution
        gym = gym_backend()
        try:
            if importlib.util.find_spec("gym") is not None:
                with set_gym_backend("gym"):
                    gym = gym_backend()

                old_api = version.parse(gym.__version__) < version.parse("0.26")
                make_fun = EnvCreator(
                    lambda: GymWrapper(gym.make(PENDULUM_VERSIONED()))
                )
            elif importlib.util.find_spec("gymnasium") is not None:
                import gymnasium

                old_api = False
                make_fun = EnvCreator(
                    lambda: GymWrapper(gymnasium.make(PENDULUM_VERSIONED()))
                )
            else:
                raise ImportError  # unreachable under pytest.skipif
            penv = maybe_fork_ParallelEnv(2, make_fun)
            rollout = penv.rollout(2)
            if old_api:
                assert "terminated" in rollout.keys()
                # truncated is read from info
                assert "truncated" in rollout.keys()
            else:
                assert "terminated" in rollout.keys()
                assert "truncated" in rollout.keys()
            check_env_specs(penv)
        finally:
            set_gym_backend(gym).set()

    @implement_for("gym", None, "0.22.0")
    def test_vecenvs_nan(self):  # noqa: F811
        # old versions of gym must return nan for next values when there is a done state
        torch.manual_seed(0)
        env = GymEnv("CartPole-v0", num_envs=2)
        env.set_seed(0)
        rollout = env.rollout(200)
        assert torch.isfinite(rollout.get("observation")).all()
        assert not torch.isfinite(rollout.get(("next", "observation"))).all()
        env.close()
        del env

        # same with collector
        env = GymEnv("CartPole-v0", num_envs=2)
        env.set_seed(0)
        c = SyncDataCollector(
            env, RandomPolicy(env.action_spec), total_frames=2000, frames_per_batch=200
        )
        for rollout in c:
            assert torch.isfinite(rollout.get("observation")).all()
            assert not torch.isfinite(rollout.get(("next", "observation"))).all()
            break
        del c
        return

    @implement_for("gym", "0.22.0", None)
    def test_vecenvs_nan(self):  # noqa: F811
        # new versions of gym must never return nan for next values when there is a done state
        torch.manual_seed(0)
        env = GymEnv("CartPole-v0", num_envs=2)
        env.set_seed(0)
        rollout = env.rollout(200)
        assert torch.isfinite(rollout.get("observation")).all()
        assert torch.isfinite(rollout.get(("next", "observation"))).all()
        env.close()
        del env

        # same with collector
        env = GymEnv("CartPole-v0", num_envs=2)
        env.set_seed(0)
        c = SyncDataCollector(
            env, RandomPolicy(env.action_spec), total_frames=2000, frames_per_batch=200
        )
        for rollout in c:
            assert torch.isfinite(rollout.get("observation")).all()
            assert torch.isfinite(rollout.get(("next", "observation"))).all()
            break
        del c
        return

    @implement_for("gymnasium")
    def test_vecenvs_nan(self):  # noqa: F811
        # new versions of gym must never return nan for next values when there is a done state
        torch.manual_seed(0)
        env = GymEnv("CartPole-v1", num_envs=2)
        env.set_seed(0)
        rollout = env.rollout(200)
        assert torch.isfinite(rollout.get("observation")).all()
        assert torch.isfinite(rollout.get(("next", "observation"))).all()
        env.close()
        del env

        # same with collector
        env = GymEnv("CartPole-v1", num_envs=2)
        env.set_seed(0)
        c = SyncDataCollector(
            env, RandomPolicy(env.action_spec), total_frames=2000, frames_per_batch=200
        )
        for rollout in c:
            assert torch.isfinite(rollout.get("observation")).all()
            assert torch.isfinite(rollout.get(("next", "observation"))).all()
            break
        del c
        return


@implement_for("gym", None, "0.26")
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name)


@implement_for("gym", "0.26", None)
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name, render_mode="rgb_array")


@implement_for("gymnasium")
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name, render_mode="rgb_array")


@pytest.mark.skipif(not _has_dmc, reason="no dm_control library found")
@pytest.mark.parametrize("env_name,task", [["cheetah", "run"]])
@pytest.mark.parametrize("frame_skip", [1, 3])
@pytest.mark.parametrize(
    "from_pixels,pixels_only", [[True, True], [True, False], [False, False]]
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


params = []
if _has_dmc:
    params = [
        # [DMControlEnv, ("cheetah", "run"), {"from_pixels": True}],
        [DMControlEnv, ("cheetah", "run"), {"from_pixels": False}],
    ]
if _has_gym:
    params += [
        # [GymEnv, (HALFCHEETAH_VERSIONED,), {"from_pixels": True}],
        [GymEnv, (HALFCHEETAH_VERSIONED(),), {"from_pixels": False}],
        [GymEnv, (PONG_VERSIONED(),), {}],
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
        # [GymEnv, (PONG_VERSIONED,), {}],  # 1226: skipping
    ]


# @pytest.mark.skipif(IS_OSX, reason="rendering unstable on osx, skipping")
@pytest.mark.parametrize("env_lib,env_args,env_kwargs", params)
@pytest.mark.parametrize(
    "device",
    [torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")],
)
class TestCollectorLib:
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
        collector = SyncDataCollector(  # 1226: not using MultiaSync for perf reasons
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
        key, *keys = jax.random.split(key, int(np.prod(batch_size) + 1))
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
    PENDULUM_VERSIONED(),
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    CARTPOLE_VERSIONED(),
]
ENVPOOL_ATARI_ENVS = []  # PONG_VERSIONED]
ENVPOOL_GYM_ENVS = ENVPOOL_CLASSIC_CONTROL_ENVS + ENVPOOL_ATARI_ENVS
ENVPOOL_DM_ENVS = ["CheetahRun-v1"]
ENVPOOL_ALL_ENVS = ENVPOOL_GYM_ENVS + ENVPOOL_DM_ENVS


@pytest.mark.skipif(not _has_envpool, reason="No envpool library found")
class TestEnvPool:
    def test_lib(self):
        import envpool

        assert MultiThreadedEnvWrapper.lib is envpool

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
        if env_name != PONG_VERSIONED():
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
            PENDULUM_VERSIONED(),
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
        key, *keys = jax.random.split(key, int(np.prod(batch_size) + 1))
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
    def test_brax_parallel(
        self, envname, batch_size, parallel, maybe_fork_ParallelEnv, n=1
    ):
        def make_brax():
            env = BraxEnv(envname, batch_size=batch_size, requires_grad=False)
            env.set_seed(1)
            return env

        if parallel:
            env = maybe_fork_ParallelEnv(n, make_brax)
        else:
            env = SerialEnv(n, make_brax)
        check_env_specs(env)
        tensordict = env.rollout(3)
        assert tensordict.shape == torch.Size([n, *batch_size, 3])


@pytest.mark.skipif(not _has_vmas, reason="vmas not installed")
class TestVmas:
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
    @pytest.mark.parametrize("continuous_actions", [True, False])
    def test_all_vmas_scenarios(self, scenario_name, continuous_actions):
        env = VmasEnv(
            scenario=scenario_name,
            continuous_actions=continuous_actions,
            num_envs=4,
        )
        env.set_seed(0)
        env.reset()
        env.rollout(10)
        env.close()

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
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
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
            env = VmasEnv(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                batch_size=batch_size,
            )
            env.close()

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize(
        "scenario_name",
        ["simple_reference", "simple_tag", "waterfall", "flocking", "discovery"],
    )
    def test_vmas_batch_size(self, scenario_name, num_envs, n_agents):
        torch.manual_seed(0)
        n_rollout_samples = 5
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        )
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(
            max_steps=n_rollout_samples,
            return_contiguous=False if env.het_specs else True,
        )
        env.close()

        if env.het_specs:
            assert isinstance(tdreset["agents"], LazyStackedTensorDict)
        else:
            assert isinstance(tdreset["agents"], TensorDict)

        assert tdreset.batch_size == (num_envs,)
        assert tdreset["agents"].batch_size == (num_envs, env.n_agents)
        if not env.het_specs:
            assert tdreset["agents", "observation"].shape[1] == env.n_agents
        assert tdreset["done"].shape[1] == 1

        assert tdrollout.batch_size == (num_envs, n_rollout_samples)
        assert tdrollout["agents"].batch_size == (
            num_envs,
            n_rollout_samples,
            env.n_agents,
        )
        if not env.het_specs:
            assert tdrollout["agents", "observation"].shape[2] == env.n_agents
        assert tdrollout["next", "agents", "reward"].shape[2] == env.n_agents
        assert tdrollout["agents", "action"].shape[2] == env.n_agents
        assert tdrollout["done"].shape[2] == 1
        del env

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize(
        "scenario_name",
        ["simple_reference", "simple_tag", "waterfall", "flocking", "discovery"],
    )
    def test_vmas_spec_rollout(
        self, scenario_name, num_envs, n_agents, continuous_actions
    ):
        vmas_env = VmasEnv(
            scenario=scenario_name,
            num_envs=num_envs,
            n_agents=n_agents,
            continuous_actions=continuous_actions,
        )
        vmas_wrapped_env = VmasWrapper(
            vmas.make_env(
                scenario=scenario_name,
                num_envs=num_envs,
                n_agents=n_agents,
                continuous_actions=continuous_actions,
            )
        )
        for env in [vmas_env, vmas_wrapped_env]:
            env.set_seed(0)
            check_env_specs(env, return_contiguous=False if env.het_specs else True)
            env.close()

    @pytest.mark.parametrize("num_envs", [1, 20])
    @pytest.mark.parametrize("n_agents", [1, 5])
    @pytest.mark.parametrize("scenario_name", VmasWrapper.available_envs)
    def test_vmas_repr(self, scenario_name, num_envs, n_agents):
        if (
            n_agents == 1
            and scenario_name == "balance"
            or scenario_name == "simple_adversary"
        ):
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
        env.close()

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
        maybe_fork_ParallelEnv,
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

        env = maybe_fork_ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert tensordict.shape == torch.Size(
            [n_workers, list(env.num_envs)[0], n_rollout_samples]
        )
        env.close()

    @pytest.mark.parametrize("num_envs", [1, 2])
    @pytest.mark.parametrize("n_workers", [1, 3])
    @pytest.mark.parametrize(
        "scenario_name", ["simple_reference", "waterfall", "flocking", "discovery"]
    )
    def test_vmas_reset(
        self,
        scenario_name,
        num_envs,
        n_workers,
        maybe_fork_ParallelEnv,
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

        env = maybe_fork_ParallelEnv(n_workers, make_vmas)
        tensordict = env.rollout(max_steps=n_rollout_samples)

        assert (
            tensordict["next", "done"]
            .sum(
                tuple(range(tensordict.batch_dims, tensordict["next", "done"].ndim)),
                dtype=torch.bool,
            )[..., -1]
            .all()
        )

        td_reset = TensorDict(
            rand_reset(env), batch_size=env.batch_size, device=env.device
        )
        # it is good practice to have a "complete" input tensordict for reset
        for done_key in env.done_keys:
            td_reset.set(done_key, tensordict[..., -1].get(("next", done_key)))
        reset = td_reset["_reset"]
        tensordict = env.reset(td_reset)

        assert not tensordict.get("done")[reset].any()
        assert tensordict.get("done")[~reset].all()
        env.close()

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

        env = make_vmas()

        assert env.rollout(max_steps=3).device == devices[first]

        env.to(devices[1 - first])

        assert env.rollout(max_steps=3).device == devices[1 - first]
        env.close()

    @pytest.mark.parametrize("n_envs", [1, 4])
    @pytest.mark.parametrize("n_workers", [1, 2])
    @pytest.mark.parametrize("n_agents", [1, 3])
    def test_collector(
        self, n_envs, n_workers, n_agents, maybe_fork_ParallelEnv, frames_per_batch=80
    ):
        torch.manual_seed(1)
        env_fun = lambda: VmasEnv(
            scenario="flocking", num_envs=n_envs, n_agents=n_agents, max_steps=7
        )

        env = maybe_fork_ParallelEnv(n_workers, env_fun)

        n_actions_per_agent = env.action_spec.shape[-1]
        n_observations_per_agent = env.observation_spec["agents", "observation"].shape[
            -1
        ]

        policy = SafeModule(
            nn.Linear(
                n_observations_per_agent,
                n_actions_per_agent,
            ),
            in_keys=[("agents", "observation")],
            out_keys=[env.action_key],
            spec=env.action_spec,
            safe=True,
        )
        ccollector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=1000,
            device="cpu",
        )

        for i, _td in enumerate(ccollector):
            if i == 1:
                break
        ccollector.shutdown()

        td_batch = (n_workers, n_envs, frames_per_batch // (n_workers * n_envs))
        agents_td_batch = td_batch + (n_agents,)

        assert _td.shape == td_batch
        assert _td["next"].shape == td_batch
        assert _td["agents"].shape == agents_td_batch
        assert _td["agents", "info"].shape == agents_td_batch
        assert _td["next", "agents"].shape == agents_td_batch
        assert _td["next", "agents", "info"].shape == agents_td_batch
        assert _td["collector"].shape == td_batch

        assert _td[env.action_key].shape == agents_td_batch + (n_actions_per_agent,)
        assert _td["agents", "observation"].shape == agents_td_batch + (
            n_observations_per_agent,
        )
        assert _td["next", "agents", "observation"].shape == agents_td_batch + (
            n_observations_per_agent,
        )
        assert _td["next", env.reward_key].shape == agents_td_batch + (1,)
        for done_key in env.done_keys:
            assert _td[done_key].shape == td_batch + (1,)
            assert _td["next", done_key].shape == td_batch + (1,)

        assert env.reward_key not in _td.keys(True, True)
        assert env.action_key not in _td["next"].keys(True, True)

    def test_collector_heterogeneous(self, n_envs=10, frames_per_batch=20):
        env = VmasEnv(
            scenario="simple_tag",
            num_envs=n_envs,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        )
        torch.manual_seed(1)

        ccollector = SyncDataCollector(
            create_env_fn=env,
            policy=None,
            frames_per_batch=frames_per_batch,
            total_frames=1000,
            device="cpu",
        )

        for i, _td in enumerate(ccollector):
            if i == 1:
                break
        ccollector.shutdown()

        td_batch = (n_envs, frames_per_batch // n_envs)
        agents_td_batch = td_batch + (env.n_agents,)

        assert _td.shape == td_batch
        assert _td["next"].shape == td_batch
        assert _td["agents"].shape == agents_td_batch
        assert _td["next", "agents"].shape == agents_td_batch
        assert _td["collector"].shape == td_batch
        assert _td["next", env.reward_key].shape == agents_td_batch + (1,)
        for done_key in env.done_keys:
            assert _td[done_key].shape == td_batch + (1,)
            assert _td["next", done_key].shape == td_batch + (1,)

        assert env.reward_key not in _td.keys(True, True)
        assert env.action_key not in _td["next"].keys(True, True)

    @pytest.mark.parametrize("n_agents", [1, 5])
    def test_grouping(self, n_agents, scenario_name="dispersion", n_envs=2):
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=n_envs,
            n_agents=n_agents,
        )
        env = VmasEnv(
            scenario=scenario_name,
            num_envs=n_envs,
            n_agents=n_agents,
            # Put each agent in a group with its name
            group_map={
                agent_name: [agent_name] for agent_name in reversed(env.agent_names)
            },
        )

        # Check that when setting the action for a specific group, it is reflected to the right agent in the backend
        for group in env.group_map.keys():
            env.reset()
            action = env.full_action_spec.zero()
            action.set((group, "action"), action.get((group, "action")) + 1.0)
            prev_pos = {agent.name: agent.state.pos.clone() for agent in env.agents}
            env.step(action)
            pos = {agent.name: agent.state.pos.clone() for agent in env.agents}
            for agent_name in env.agent_names:
                if agent_name == group:
                    assert (pos[agent_name] > prev_pos[agent_name]).all()
                else:
                    assert (pos[agent_name] == prev_pos[agent_name]).all()


@pytest.mark.slow
class TestGenDGRL:
    @staticmethod
    @pytest.fixture
    def _patch_traj_len():
        # avoids processing the entire dataset
        _get_category_len = GenDGRLExperienceReplay._get_category_len

        def new_get_category_len(cls, category_name):
            return 100

        GenDGRLExperienceReplay._get_category_len = classmethod(new_get_category_len)

        yield
        GenDGRLExperienceReplay._get_category_len = _get_category_len

    @pytest.mark.parametrize("dataset_num", [4])
    def test_gen_dgrl_preproc(self, dataset_num, tmpdir, _patch_traj_len):
        dataset_id = GenDGRLExperienceReplay.available_datasets[dataset_num]
        tmpdir = Path(tmpdir)
        dataset = GenDGRLExperienceReplay(
            dataset_id, batch_size=32, root=tmpdir / "1", download="force"
        )
        from torchrl.envs import Compose, GrayScale, Resize

        t = Compose(
            Resize(32, in_keys=["observation", ("next", "observation")]),
            GrayScale(in_keys=["observation", ("next", "observation")]),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            num_frames=100,
            dest=tmpdir / "2",
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["observation"].shape == torch.Size([32, 1, 32, 32])
        assert sample["next", "observation"].shape == torch.Size([32, 1, 32, 32])

    @pytest.mark.parametrize("dataset_num", [0, 4, 8])
    def test_gen_dgrl(self, dataset_num, tmpdir, _patch_traj_len):
        dataset_id = GenDGRLExperienceReplay.available_datasets[dataset_num]
        dataset = GenDGRLExperienceReplay(dataset_id, batch_size=32, root=tmpdir)
        for batch in dataset:  # noqa: B007
            break
        assert batch.get(("next", "observation")).shape[-3] == 3
        for key in (
            ("next", "done"),
            ("next", "truncated"),
            ("next", "terminated"),
            "observation",
            "action",
            ("next", "reward"),
        ):
            assert key in batch.keys(True, True)
        for key in (
            ("next", "done"),
            ("next", "truncated"),
            ("next", "terminated"),
            "terminated",
            "truncated",
            "done",
            ("next", "reward"),
        ):
            val = batch.get(key)
            assert val.shape[:-1] == batch.shape


@pytest.mark.skipif(not _has_d4rl, reason="D4RL not found")
@pytest.mark.slow
class TestD4RL:
    def test_d4rl_preproc(self, tmpdir):
        dataset_id = "walker2d-medium-replay-v2"
        tmpdir = Path(tmpdir)
        dataset = D4RLExperienceReplay(
            dataset_id,
            batch_size=32,
            root=tmpdir / "1",
            download="force",
            direct_download=True,
        )
        from torchrl.envs import CatTensors, Compose

        t = Compose(
            CatTensors(
                in_keys=["observation", ("info", "qpos"), ("info", "qvel")],
                out_key="data",
            ),
            CatTensors(
                in_keys=[
                    ("next", "observation"),
                    ("next", "info", "qpos"),
                    ("next", "info", "qvel"),
                ],
                out_key=("next", "data"),
            ),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir / "2",
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["data"].shape == torch.Size([32, 35])
        assert sample["next", "data"].shape == torch.Size([32, 35])

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    @pytest.mark.parametrize("use_truncated_as_done", [True, False])
    @pytest.mark.parametrize("split_trajs", [True, False])
    def test_terminate_on_end(self, task, use_truncated_as_done, split_trajs, tmpdir):
        root1 = tmpdir / "1"
        root2 = tmpdir / "2"
        root3 = tmpdir / "3"

        with pytest.warns(
            UserWarning, match="Using use_truncated_as_done=True"
        ) if use_truncated_as_done else nullcontext():
            data_true = D4RLExperienceReplay(
                task,
                split_trajs=split_trajs,
                from_env=False,
                terminate_on_end=True,
                batch_size=2,
                use_truncated_as_done=use_truncated_as_done,
                download="force",
                root=root1,
            )
        _ = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=False,
            terminate_on_end=False,
            batch_size=2,
            use_truncated_as_done=use_truncated_as_done,
            download="force",
            root=root2,
        )
        data_from_env = D4RLExperienceReplay(
            task,
            split_trajs=split_trajs,
            from_env=True,
            batch_size=2,
            use_truncated_as_done=use_truncated_as_done,
            download="force",
            root=root3,
        )
        if not use_truncated_as_done:
            keys = set(data_from_env[:].keys(True, True))
            keys = keys.intersection(data_true[:].keys(True, True))
            assert data_true[:].shape == data_from_env[:].shape
            # for some reason, qlearning_dataset overwrites the next obs that is contained in the buffer,
            # resulting in tiny changes in the value contained for that key. Over 99.99% of the values
            # match, but the test still fails because of this.
            # We exclude that entry from the comparison.
            keys.discard(("next", "observation"))
            assert_allclose_td(
                data_true[:].select(*keys),
                data_from_env[:].select(*keys),
            )
        else:
            leaf_names = data_from_env[:].keys(True)
            leaf_names = [
                name[-1] if isinstance(name, tuple) else name for name in leaf_names
            ]
            assert "truncated" in leaf_names
            leaf_names = data_true[:].keys(True)
            leaf_names = [
                name[-1] if isinstance(name, tuple) else name for name in leaf_names
            ]
            assert "truncated" not in leaf_names

    @pytest.mark.parametrize("task", ["walker2d-medium-replay-v2"])
    def test_direct_download(self, task, tmpdir):
        root1 = tmpdir / "1"
        root2 = tmpdir / "2"
        data_direct = D4RLExperienceReplay(
            task,
            split_trajs=False,
            from_env=False,
            batch_size=2,
            use_truncated_as_done=True,
            direct_download=True,
            download="force",
            root=root1,
        )
        data_d4rl = D4RLExperienceReplay(
            task,
            split_trajs=False,
            from_env=True,
            batch_size=2,
            use_truncated_as_done=True,
            direct_download=False,
            terminate_on_end=True,  # keep the last time step
            download="force",
            root=root2,
        )
        keys = set(data_direct[:].keys(True, True))
        keys = keys.intersection(data_d4rl[:].keys(True, True))
        assert len(keys)
        assert_allclose_td(
            data_direct[:].select(*keys).apply(lambda t: t.float()),
            data_d4rl[:].select(*keys).apply(lambda t: t.float()),
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
        torchrl_logger.info(f"terminated test after {time.time()-t0}s")

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
            if "truncated" in key:
                # truncated is missing from static datasets
                continue
            sim = rollout.get(key)
            offline = sample.get(key)
            # assert sim.dtype == offline.dtype, key
            assert sim.shape[-1] == offline.shape[-1], key
        torchrl_logger.info(f"terminated test after {time.time()-t0}s")

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
        torchrl_logger.info(f"terminated test after {time.time()-t0}s")


_MINARI_DATASETS = []


def _minari_selected_datasets():
    if not _has_minari or not _has_gymnasium:
        return
    global _MINARI_DATASETS
    import minari

    torch.manual_seed(0)

    keys = list(minari.list_remote_datasets())
    indices = torch.randperm(len(keys))[:10]
    keys = [keys[idx] for idx in indices]
    keys = [
        key
        for key in keys
        if "=0.4" in minari.list_remote_datasets()[key]["minari_version"]
    ]
    assert len(keys) > 5
    _MINARI_DATASETS += keys


_minari_selected_datasets()


@pytest.mark.skipif(not _has_minari or not _has_gymnasium, reason="Minari not found")
@pytest.mark.slow
class TestMinari:
    @pytest.mark.parametrize("split", [False, True])
    @pytest.mark.parametrize("selected_dataset", _MINARI_DATASETS)
    def test_load(self, selected_dataset, split):
        torchrl_logger.info(f"dataset {selected_dataset}")
        data = MinariExperienceReplay(
            selected_dataset, batch_size=32, split_trajs=split
        )
        t0 = time.time()
        for i, sample in enumerate(data):
            t1 = time.time()
            torchrl_logger.info(f"sampling time {1000 * (t1-t0): 4.4f}ms")
            assert data.metadata["action_space"].is_in(sample["action"])
            assert data.metadata["observation_space"].is_in(sample["observation"])
            t0 = time.time()
            if i == 10:
                break

    def test_minari_preproc(self, tmpdir):
        selected_dataset = _MINARI_DATASETS[0]
        dataset = MinariExperienceReplay(
            selected_dataset,
            batch_size=32,
            split_trajs=False,
            download="force",
        )

        from torchrl.envs import CatTensors, Compose

        t = Compose(
            CatTensors(
                in_keys=[
                    ("observation", "observation"),
                    ("info", "qpos"),
                    ("info", "qvel"),
                ],
                out_key="data",
            ),
            CatTensors(
                in_keys=[
                    ("next", "observation", "observation"),
                    ("next", "info", "qpos"),
                    ("next", "info", "qvel"),
                ],
                out_key=("next", "data"),
            ),
        )

        def fn(data):
            return t(data)

        new_storage = dataset.preprocess(
            fn,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            num_frames=100,
            dest=tmpdir,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        sample = dataset.sample()
        assert len(dataset) == 100
        assert sample["data"].shape == torch.Size([32, 8])
        assert sample["next", "data"].shape == torch.Size([32, 8])


@pytest.mark.slow
class TestRoboset:
    def test_load(self):
        selected_dataset = RobosetExperienceReplay.available_datasets[0]
        data = RobosetExperienceReplay(
            selected_dataset,
            batch_size=32,
        )
        t0 = time.time()
        for i, _ in enumerate(data):
            t1 = time.time()
            torchrl_logger.info(f"sampling time {1000 * (t1-t0): 4.4f}ms")
            t0 = time.time()
            if i == 10:
                break

    def test_roboset_preproc(self, tmpdir):
        dataset = RobosetExperienceReplay(
            "FK1-v4(expert)/FK1_MicroOpenRandom_v2d-v4", batch_size=32, download="force"
        )

        def func(data):
            return data.set("obs_norm", data.get("observation").norm(dim=-1))

        new_storage = dataset.preprocess(
            func,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir,
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert "obs_norm" in sample.keys()


@pytest.mark.slow
class TestVD4RL:
    @pytest.mark.parametrize("image_size", [None, (37, 33)])
    def test_load(self, image_size):
        torch.manual_seed(0)
        datasets = VD4RLExperienceReplay.available_datasets
        for idx in torch.randperm(len(datasets)).tolist()[:4]:
            selected_dataset = datasets[idx]
            data = VD4RLExperienceReplay(
                selected_dataset,
                batch_size=32,
                image_size=image_size,
            )
            t0 = time.time()
            for i, batch in enumerate(data):
                if image_size:
                    assert batch.get("pixels").shape == (32, 3, *image_size)
                    assert batch.get(("next", "pixels")).shape == (32, 3, *image_size)
                else:
                    assert batch.get("pixels").shape[:2] == (32, 3)
                    assert batch.get(("next", "pixels")).shape[:2] == (32, 3)

                assert batch.get("pixels").dtype is torch.float32
                assert batch.get(("next", "pixels")).dtype is torch.float32
                assert (batch.get("pixels") != 0).any()
                assert (batch.get(("next", "pixels")) != 0).any()
                t1 = time.time()
                torchrl_logger.info(f"sampling time {1000 * (t1-t0): 4.4f}ms")
                t0 = time.time()
                if i == 10:
                    break

    def test_vd4rl_preproc(self, tmpdir):
        torch.manual_seed(0)
        datasets = VD4RLExperienceReplay.available_datasets
        dataset_id = list(datasets)[4]
        dataset = VD4RLExperienceReplay(dataset_id, batch_size=32, download="force")
        from torchrl.envs import Compose, GrayScale, ToTensorImage

        func = Compose(
            ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
            GrayScale(in_keys=["pixels", ("next", "pixels")]),
        )
        new_storage = dataset.preprocess(
            func,
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=1000,
            mp_start_method="fork",
            dest=tmpdir,
            num_frames=100,
        )
        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100
        sample = dataset.sample()
        assert sample["next", "pixels"].shape == torch.Size([32, 1, 64, 64])


@pytest.mark.slow
class TestAtariDQN:
    @pytest.fixture(scope="class")
    def limit_max_runs(self):
        prev_val = AtariDQNExperienceReplay._max_runs
        AtariDQNExperienceReplay._max_runs = 3
        yield
        AtariDQNExperienceReplay._max_runs = prev_val

    @pytest.mark.parametrize("dataset_id", ["Asterix/1", "Pong/4"])
    @pytest.mark.parametrize(
        "num_slices,slice_len", [[None, None], [None, 8], [2, None]]
    )
    def test_single_dataset(self, dataset_id, slice_len, num_slices, limit_max_runs):
        dataset = AtariDQNExperienceReplay(
            dataset_id, slice_len=slice_len, num_slices=num_slices
        )
        sample = dataset.sample(64)
        for key in (
            ("next", "observation"),
            ("next", "truncated"),
            ("next", "terminated"),
            ("next", "done"),
            ("next", "reward"),
            "observation",
            "action",
            "done",
            "truncated",
            "terminated",
        ):
            assert key in sample.keys(True)
        assert sample.shape == (64,)
        assert sample.get_non_tensor("metadata")["dataset_id"] == dataset_id

    @pytest.mark.parametrize(
        "num_slices,slice_len", [[None, None], [None, 8], [2, None]]
    )
    def test_double_dataset(self, slice_len, num_slices, limit_max_runs):
        dataset_pong = AtariDQNExperienceReplay(
            "Pong/4", slice_len=slice_len, num_slices=num_slices
        )
        dataset_asterix = AtariDQNExperienceReplay(
            "Asterix/1", slice_len=slice_len, num_slices=num_slices
        )
        dataset = ReplayBufferEnsemble(
            dataset_pong, dataset_asterix, sample_from_all=True, batch_size=128
        )
        sample = dataset.sample()
        assert sample.shape == (2, 64)
        assert sample[0].get_non_tensor("metadata")["dataset_id"] == "Pong/4"
        assert sample[1].get_non_tensor("metadata")["dataset_id"] == "Asterix/1"

    @pytest.mark.parametrize("dataset_id", ["Pong/4"])
    def test_atari_preproc(self, dataset_id, tmpdir):
        from torchrl.envs import Compose, RenameTransform, Resize, UnsqueezeTransform

        dataset = AtariDQNExperienceReplay(
            dataset_id,
            slice_len=None,
            num_slices=8,
            batch_size=64,
            # num_procs=max(0, os.cpu_count() - 4),
            num_procs=0,
        )

        t = Compose(
            UnsqueezeTransform(
                unsqueeze_dim=-3, in_keys=["observation", ("next", "observation")]
            ),
            Resize(32, in_keys=["observation", ("next", "observation")]),
            RenameTransform(in_keys=["action"], out_keys=["other_action"]),
        )

        def preproc(data):
            return t(data)

        new_storage = dataset.preprocess(
            preproc,
            num_workers=max(1, os.cpu_count() - 4),
            num_chunks=1000,
            mp_start_method="fork",
            pbar=True,
            dest=tmpdir,
            num_frames=100,
        )

        dataset = ReplayBuffer(storage=new_storage, batch_size=32)
        assert len(dataset) == 100


@pytest.mark.slow
class TestOpenX:
    @pytest.mark.parametrize(
        "download,padding",
        [[True, None], [False, None], [False, 0], [False, True], [False, False]],
    )
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("replacement", [True, False])
    @pytest.mark.parametrize(
        "batch_size,num_slices,slice_len",
        [
            [3000, 2, None],
            [32, 32, None],
            [32, None, 1],
            [3000, None, 1500],
            [None, None, 32],
            [None, None, 1500],
        ],
    )
    def test_openx(
        self, download, shuffle, replacement, padding, batch_size, num_slices, slice_len
    ):
        torch.manual_seed(0)
        np.random.seed(0)

        streaming = not download
        cm = (
            pytest.raises(RuntimeError, match="shuffle=False")
            if not streaming and not shuffle and replacement
            else pytest.raises(
                RuntimeError,
                match="replacement=True is not available with streamed datasets",
            )
            if streaming and replacement
            else nullcontext()
        )
        dataset = None
        with cm:
            dataset = OpenXExperienceReplay(
                "cmu_stretch",
                download=download,
                streaming=streaming,
                batch_size=batch_size,
                shuffle=shuffle,
                num_slices=num_slices,
                slice_len=slice_len,
                pad=padding,
                replacement=replacement,
            )
        if dataset is None:
            return
        # iterating
        if padding is None and (
            (batch_size is not None and batch_size > 1000)
            or (slice_len is not None and slice_len > 1000)
        ):
            raises_cm = pytest.raises(
                RuntimeError,
                match="The trajectory length (.*) is shorter than the slice length|"
                #       "Some stored trajectories have a length shorter than the slice that was asked for|"
                "Did not find a single trajectory with sufficient length",
            )
            with raises_cm:
                for data in dataset:  # noqa: B007
                    break
            if batch_size is None and slice_len is not None:
                with raises_cm:
                    dataset.sample(2 * slice_len)
                return

        else:
            for data in dataset:  # noqa: B007
                break
            # check data shape
            if batch_size is not None:
                assert data.shape[0] == batch_size
            elif slice_len is not None:
                assert data.shape[0] == slice_len
            if batch_size is not None:
                if num_slices is not None:
                    assert data.get(("next", "done")).sum(-2) == num_slices
                elif streaming:
                    assert (
                        data.get(("next", "done")).sum(-2)
                        == data.get("episode").unique().numel()
                    )

        # sampling
        if batch_size is None:
            if slice_len is not None:
                batch_size = 2 * slice_len
            elif num_slices is not None:
                batch_size = num_slices * 32
            sample = dataset.sample(batch_size)
        else:
            if padding is None and (batch_size > 1000):
                with pytest.raises(
                    RuntimeError,
                    match="Did not find a single trajectory with sufficient length"
                    if not streaming
                    else "The trajectory length (.*) is shorter than the slice length",
                ):
                    sample = dataset.sample()
                return
            else:
                sample = dataset.sample()
                assert sample.shape == (batch_size,)
        if slice_len is not None:
            assert sample.get(("next", "done")).sum() == int(
                batch_size // slice_len
            ), sample.get(("next", "done"))
        elif num_slices is not None:
            assert sample.get(("next", "done")).sum() == num_slices

    def test_openx_preproc(self, tmpdir):
        dataset = OpenXExperienceReplay(
            "cmu_stretch",
            download=True,
            streaming=False,
            batch_size=64,
            shuffle=True,
            num_slices=8,
            slice_len=None,
        )
        from torchrl.envs import Compose, RenameTransform, Resize

        t = Compose(
            Resize(
                64,
                64,
                in_keys=[("observation", "image"), ("next", "observation", "image")],
            ),
            RenameTransform(
                in_keys=[
                    ("observation", "image"),
                    ("next", "observation", "image"),
                    ("observation", "state"),
                    ("next", "observation", "state"),
                ],
                out_keys=["pixels", ("next", "pixels"), "state", ("next", "state")],
            ),
        )

        def fn(data: TensorDict):
            data.unlock_()
            data = data.select(
                "action",
                "done",
                "episode",
                ("next", "done"),
                ("next", "observation"),
                ("next", "reward"),
                ("next", "terminated"),
                ("next", "truncated"),
                "observation",
                "terminated",
                "truncated",
            )
            data = t(data)
            data = data.select(*data.keys(True, True))
            return data

        new_storage = dataset.preprocess(
            CloudpickleWrapper(fn),
            num_workers=max(1, os.cpu_count() - 2),
            num_chunks=500,
            mp_start_method="fork",
            dest=tmpdir,
        )
        dataset = ReplayBuffer(storage=new_storage)
        sample = dataset.sample(32)
        assert "observation" not in sample.keys()
        assert "pixels" in sample.keys()
        assert ("next", "pixels") in sample.keys(True)
        assert "state" in sample.keys()
        assert ("next", "state") in sample.keys(True)
        assert sample["pixels"].shape == torch.Size([32, 3, 64, 64])


@pytest.mark.skipif(not _has_sklearn, reason="Scikit-learn not found")
@pytest.mark.parametrize(
    "dataset",
    [
        # "adult_num", # 1226: Expensive to test
        # "adult_onehot", # 1226: Expensive to test
        "mushroom_num",
        "mushroom_onehot",
        # "covertype",  # 1226: Expensive to test
        "shuttle",
        "magic",
    ],
)
@pytest.mark.slow
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


@pytest.mark.skipif(not _has_isaac, reason="IsaacGym not found")
@pytest.mark.parametrize(
    "task",
    [
        "AllegroHand",
        # "AllegroKuka",
        # "AllegroKukaTwoArms",
        # "AllegroHandManualDR",
        # "AllegroHandADR",
        "Ant",
        # "Anymal",
        # "AnymalTerrain",
        # "BallBalance",
        # "Cartpole",
        # "FactoryTaskGears",
        # "FactoryTaskInsertion",
        # "FactoryTaskNutBoltPick",
        # "FactoryTaskNutBoltPlace",
        # "FactoryTaskNutBoltScrew",
        # "FrankaCabinet",
        # "FrankaCubeStack",
        "Humanoid",
        # "HumanoidAMP",
        # "Ingenuity",
        # "Quadcopter",
        # "ShadowHand",
        "Trifinger",
    ],
)
@pytest.mark.parametrize("num_envs", [10, 20])
@pytest.mark.parametrize("device", get_default_devices())
class TestIsaacGym:
    @classmethod
    def _run_on_proc(cls, q, task, num_envs, device):
        try:
            env = IsaacGymEnv(task=task, num_envs=num_envs, device=device)
            check_env_specs(env)
            q.put(("succeeded!", None))
        except Exception as err:
            q.put(("failed!", err))
            raise err

    def test_env(self, task, num_envs, device):
        from torch import multiprocessing as mp

        q = mp.Queue(1)
        proc = mp.Process(target=self._run_on_proc, args=(q, task, num_envs, device))
        try:
            proc.start()
            msg, error = q.get()
            if msg != "succeeded!":
                raise error
        finally:
            q.close()
            proc.join()

    #
    # def test_collector(self, task, num_envs, device):
    #     env = IsaacGymEnv(task=task, num_envs=num_envs, device=device)
    #     collector = SyncDataCollector(
    #         env,
    #         policy=SafeModule(nn.LazyLinear(out_features=env.observation_spec['obs'].shape[-1]), in_keys=["obs"], out_keys=["action"]),
    #         frames_per_batch=20,
    #         total_frames=-1
    #     )
    #     for c in collector:
    #         assert c.shape == torch.Size([num_envs, 20])
    #         break


@pytest.mark.skipif(not _has_pettingzoo, reason="PettingZoo not found")
class TestPettingZoo:
    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("continuous_actions", [True, False])
    @pytest.mark.parametrize("use_mask", [True])
    @pytest.mark.parametrize("return_state", [True, False])
    @pytest.mark.parametrize(
        "group_map",
        [None, MarlGroupMapType.ALL_IN_ONE_GROUP, MarlGroupMapType.ONE_GROUP_PER_AGENT],
    )
    def test_pistonball(
        self, parallel, continuous_actions, use_mask, return_state, group_map
    ):

        kwargs = {"n_pistons": 21, "continuous": continuous_actions}

        env = PettingZooEnv(
            task="pistonball_v6",
            parallel=parallel,
            seed=0,
            return_state=return_state,
            use_mask=use_mask,
            group_map=group_map,
            **kwargs,
        )

        check_env_specs(env)

    @pytest.mark.parametrize(
        "wins_player_0",
        [True, False],
    )
    def test_tic_tac_toe(self, wins_player_0):
        env = PettingZooEnv(
            task="tictactoe_v3",
            parallel=False,
            group_map={"player": ["player_1", "player_2"]},
            categorical_actions=False,
            seed=0,
            use_mask=True,
        )

        class Policy:

            action = 0
            t = 0

            def __call__(self, td):
                new_td = env.input_spec["full_action_spec"].zero()

                player_acting = 0 if self.t % 2 == 0 else 1
                other_player = 1 if self.t % 2 == 0 else 0
                # The acting player has "mask" True and "action_mask" set to the available actions
                assert td["player", "mask"][player_acting].all()
                assert td["player", "action_mask"][player_acting].any()
                # The non-acting player has "mask" False and "action_mask" set to all Trues
                assert not td["player", "mask"][other_player].any()
                assert td["player", "action_mask"][other_player].all()

                if self.t % 2 == 0:
                    if not wins_player_0 and self.t == 4:
                        new_td["player", "action"][0][self.action + 1] = 1
                    else:
                        new_td["player", "action"][0][self.action] = 1
                else:
                    new_td["player", "action"][1][self.action + 6] = 1
                if td["player", "mask"][1].all():
                    self.action += 1
                self.t += 1
                return td.update(new_td)

        td = env.rollout(100, policy=Policy())

        assert td.batch_size[0] == (5 if wins_player_0 else 6)
        assert (td[:-1]["next", "player", "reward"] == 0).all()
        if wins_player_0:
            assert (
                td[-1]["next", "player", "reward"] == torch.tensor([[1], [-1]])
            ).all()
        else:
            assert (
                td[-1]["next", "player", "reward"] == torch.tensor([[-1], [1]])
            ).all()

    @pytest.mark.parametrize(
        "task",
        [
            "multiwalker_v9",
            "waterworld_v4",
            "pursuit_v4",
            "simple_spread_v3",
            "simple_v3",
            "rps_v2",
            "cooperative_pong_v5",
            "pistonball_v6",
        ],
    )
    def test_envs_one_group_parallel(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=True,
            seed=0,
            use_mask=False,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "multiwalker_v9",
            "waterworld_v4",
            "pursuit_v4",
            "simple_spread_v3",
            "simple_v3",
            "rps_v2",
            "cooperative_pong_v5",
            "pistonball_v6",
            "connect_four_v3",
            "tictactoe_v3",
            "chess_v6",
            "gin_rummy_v4",
            "tictactoe_v3",
        ],
    )
    def test_envs_one_group_aec(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=False,
            seed=0,
            use_mask=True,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "simple_adversary_v3",
            "simple_crypto_v3",
            "simple_push_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
            "simple_tag_v3",
            "simple_world_comm_v3",
            "knights_archers_zombies_v10",
            "basketball_pong_v3",
            "boxing_v2",
            "foozpong_v3",
        ],
    )
    def test_envs_more_groups_parallel(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=True,
            seed=0,
            use_mask=False,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize(
        "task",
        [
            "simple_adversary_v3",
            "simple_crypto_v3",
            "simple_push_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
            "simple_tag_v3",
            "simple_world_comm_v3",
            "knights_archers_zombies_v10",
            "basketball_pong_v3",
            "boxing_v2",
            "foozpong_v3",
            "go_v5",
        ],
    )
    def test_envs_more_groups_aec(self, task):
        env = PettingZooEnv(
            task=task,
            parallel=False,
            seed=0,
            use_mask=True,
        )
        check_env_specs(env)
        env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize("task", ["knights_archers_zombies_v10", "pistonball_v6"])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_vec_env(self, task, parallel, maybe_fork_ParallelEnv):
        env_fun = lambda: PettingZooEnv(
            task=task,
            parallel=parallel,
            seed=0,
            use_mask=not parallel,
        )
        vec_env = maybe_fork_ParallelEnv(2, create_env_fn=env_fun)
        vec_env.rollout(100, break_when_any_done=False)

    @pytest.mark.parametrize("task", ["knights_archers_zombies_v10", "pistonball_v6"])
    @pytest.mark.parametrize("parallel", [True, False])
    def test_collector(self, task, parallel):
        env_fun = lambda: PettingZooEnv(
            task=task,
            parallel=parallel,
            seed=0,
            use_mask=not parallel,
        )
        coll = SyncDataCollector(
            create_env_fn=env_fun, frames_per_batch=30, total_frames=60, policy=None
        )
        for _ in coll:
            break


@pytest.mark.skipif(not _has_robohive, reason="SMACv2 not found")
class TestRoboHive:
    # unfortunately we must import robohive to get the available envs
    # and this import will occur whenever pytest is run on this file.
    # The other option would be not to use parametrize but that also
    # means less informative error trace stacks.
    # In the CI, robohive should not coexist with other libs so that's fine.
    # Locally these imports can be annoying, especially given the amount of
    # stuff printed by robohive.
    @pytest.mark.parametrize("from_pixels", [True, False])
    @set_gym_backend("gym")
    def test_robohive(self, from_pixels):
        for envname in RoboHiveEnv.available_envs:
            try:
                if any(
                    substr in envname
                    for substr in ("_vr3m", "_vrrl", "_vflat", "_vvc1s")
                ):
                    torchrl_logger.info("not testing envs with prebuilt rendering")
                    return
                if "Adroit" in envname:
                    torchrl_logger.info("tcdm are broken")
                    return
                try:
                    env = RoboHiveEnv(envname)
                except AttributeError as err:
                    if "'MjData' object has no attribute 'get_body_xipos'" in str(err):
                        torchrl_logger.info("tcdm are broken")
                        return
                    else:
                        raise err
                if (
                    from_pixels
                    and len(RoboHiveEnv.get_available_cams(env_name=envname)) == 0
                ):
                    torchrl_logger.info("no camera")
                    return
                check_env_specs(env)
            except Exception as err:
                raise RuntimeError(f"Test with robohive end {envname} failed.") from err


@pytest.mark.skipif(not _has_smacv2, reason="SMACv2 not found")
class TestSmacv2:
    def test_env_procedural(self):
        distribution_config = {
            "n_units": 5,
            "n_enemies": 6,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine", "marauder", "medivac"],
                "exception_unit_types": ["medivac"],
                "weights": [0.5, 0.2, 0.3],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 5,
                "map_x": 32,
                "map_y": 32,
            },
        }
        env = SMACv2Env(
            map_name="10gen_terran",
            capability_config=distribution_config,
            seed=0,
        )
        check_env_specs(env, seed=None)
        env.close()

    @pytest.mark.parametrize("categorical_actions", [True, False])
    @pytest.mark.parametrize("map", ["MMM2", "3s_vs_5z"])
    def test_env(self, map: str, categorical_actions):
        env = SMACv2Env(
            map_name=map,
            categorical_actions=categorical_actions,
            seed=0,
        )
        check_env_specs(env, seed=None)
        env.close()

    def test_parallel_env(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(
                num_workers=2,
                create_env_fn=lambda: SMACv2Env(
                    map_name="3s_vs_5z",
                    seed=0,
                ),
            ),
            ActionMask(
                action_key=("agents", "action"), mask_key=("agents", "action_mask")
            ),
        )
        check_env_specs(env, seed=None)
        env.close()

    def test_collector(self):
        env = SMACv2Env(map_name="MMM2", seed=0, categorical_actions=True)
        in_feats = env.observation_spec["agents", "observation"].shape[-1]
        out_feats = env.action_spec.space.n

        module = TensorDictModule(
            nn.Linear(in_feats, out_feats),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        prob = ProbabilisticTensorDictModule(
            in_keys={"logits": ("agents", "logits"), "mask": ("agents", "action_mask")},
            out_keys=[("agents", "action")],
            distribution_class=MaskedCategorical,
        )
        actor = TensorDictSequential(module, prob)

        collector = SyncDataCollector(
            env, policy=actor, frames_per_batch=20, total_frames=40
        )
        for _ in collector:
            break
        collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
