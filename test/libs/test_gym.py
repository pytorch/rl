# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import gc
import importlib.util
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from packaging import version
from tensordict import assert_allclose_td, TensorDict

from torchrl._utils import implement_for, logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    MultiCategorical,
    MultiOneHot,
    NonTensor,
    OneHot,
    Unbounded,
    UnboundedDiscrete,
)
from torchrl.envs import CatTensors, Compose, EnvBase, EnvCreator, RemoveEmptySpecs
from torchrl.envs.batched_envs import ParallelEnv, SerialEnv
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
    register_gym_spec_conversion,
    set_gym_backend,
)
from torchrl.envs.utils import check_env_specs
from torchrl.modules import RandomPolicy
from torchrl.testing import (
    CARTPOLE_VERSIONED,
    CLIFFWALKING_VERSIONED,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rollout_consistency_assertion,
)

_has_ale = importlib.util.find_spec("ale_py") is not None
_has_atari_py = False
if importlib.util.find_spec("atari_py") is not None:
    try:
        import atari_py

        _has_atari_py = hasattr(atari_py, "get_game_path")
    except Exception:
        _has_atari_py = False
_has_mujoco = (
    importlib.util.find_spec("mujoco") is not None
    or importlib.util.find_spec("mujoco_py") is not None
)
_has_mo = importlib.util.find_spec("mo_gymnasium") is not None
_has_gym_robotics = importlib.util.find_spec("gymnasium_robotics") is not None
_has_gym_regular = importlib.util.find_spec("gym") is not None
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
_has_minigrid = importlib.util.find_spec("minigrid") is not None

if _has_gymnasium:
    import gymnasium

try:
    from torch.utils._pytree import tree_flatten

    _has_pytree = True
except ImportError:
    _has_pytree = False

    def tree_flatten(x):
        raise RuntimeError("pytree required")


RTOL = 1e-1
ATOL = 1e-1


def _has_atari_for_gym():
    """Check if Atari support is available for the current gym backend."""
    return False


@implement_for("gym", None, "0.25.0")
def _has_atari_for_gym():  # noqa: F811
    """For gym < 0.25: requires functional atari_py."""
    return _has_atari_py


@implement_for("gym", "0.25.0", None)
def _has_atari_for_gym():  # noqa: F811
    """For gym >= 0.25: requires ale_py."""
    return _has_ale


@implement_for("gymnasium")
def _has_atari_for_gym():  # noqa: F811
    """For gymnasium: requires ale_py."""
    return _has_ale


@implement_for("gym")
def get_gym_pixel_wrapper():
    try:
        # works whenever gym_version > version.parse("0.19")
        PixelObservationWrapper = gym_backend(
            "wrappers.pixel_observation"
        ).PixelObservationWrapper
    except Exception:
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )
    return PixelObservationWrapper


@implement_for("gymnasium", None, "1.1.0")
def get_gym_pixel_wrapper():  # noqa: F811
    try:
        # works whenever gym_version > version.parse("0.19")
        PixelObservationWrapper = gym_backend(
            "wrappers.pixel_observation"
        ).PixelObservationWrapper
    except Exception:
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )
    return PixelObservationWrapper


@implement_for("gymnasium", "1.1.0")
def get_gym_pixel_wrapper():  # noqa: F811
    # works whenever gym_version > version.parse("0.19")
    def PixelObservationWrapper(*args, pixels_only=False, **kwargs):
        return gym_backend("wrappers").AddRenderObservation(
            *args, render_only=pixels_only, **kwargs
        )

    return PixelObservationWrapper


@pytest.mark.skipif(not _has_gym, reason="no gym library found")
class TestGym:
    class DummyEnv(EnvBase):
        def __init__(self, arg1, *, arg2, **kwargs):
            super().__init__(**kwargs)

            assert arg1 == 1
            assert arg2 == 2

            self.observation_spec = Composite(
                observation=Unbounded((*self.batch_size, 3)),
                other=Composite(
                    another_other=Unbounded((*self.batch_size, 3)),
                    shape=self.batch_size,
                ),
                shape=self.batch_size,
            )
            self.action_spec = Unbounded((*self.batch_size, 3))
            self.done_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
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

        def _set_seed(self, seed: int | None) -> None:
            ...

    @implement_for("gym", None, "0.18")
    def _make_spec(self, batch_size, cat, cat_shape, multicat, multicat_shape):
        return Composite(
            a=Unbounded(shape=(*batch_size, 1)),
            b=Composite(c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=Bounded(-3, 4, shape=(*batch_size, 1)),
            # g=UnboundedDiscrete(shape=(*batch_size, 1), dtype=torch.long),
            h=Binary(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @implement_for("gym", "0.18", None)
    def _make_spec(  # noqa: F811
        self, batch_size, cat, cat_shape, multicat, multicat_shape
    ):
        return Composite(
            a=Unbounded(shape=(*batch_size, 1)),
            b=Composite(c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=Bounded(-3, 4, shape=(*batch_size, 1)),
            g=UnboundedDiscrete(shape=(*batch_size, 1), dtype=torch.long),
            h=Binary(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @implement_for("gymnasium", None, "1.0.0")
    def _make_spec(  # noqa: F811
        self, batch_size, cat, cat_shape, multicat, multicat_shape
    ):
        return Composite(
            a=Unbounded(shape=(*batch_size, 1)),
            b=Composite(c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=Bounded(-3, 4, shape=(*batch_size, 1)),
            g=UnboundedDiscrete(shape=(*batch_size, 1), dtype=torch.long),
            h=Binary(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @implement_for("gymnasium", "1.1.0")
    def _make_spec(  # noqa: F811
        self, batch_size, cat, cat_shape, multicat, multicat_shape
    ):
        return Composite(
            a=Unbounded(shape=(*batch_size, 1)),
            b=Composite(c=cat(5, shape=cat_shape, dtype=torch.int64), shape=batch_size),
            d=cat(5, shape=cat_shape, dtype=torch.int64),
            e=multicat([2, 3], shape=(*batch_size, multicat_shape), dtype=torch.int64),
            f=Bounded(-3, 4, shape=(*batch_size, 1)),
            g=UnboundedDiscrete(shape=(*batch_size, 1), dtype=torch.long),
            h=Binary(n=5, shape=(*batch_size, 5)),
            shape=batch_size,
        )

    @pytest.mark.parametrize("categorical", [True, False])
    def test_gym_spec_cast(self, categorical):
        batch_size = [3, 4]
        cat = Categorical if categorical else OneHot
        cat_shape = batch_size if categorical else (*batch_size, 5)
        multicat = MultiCategorical if categorical else MultiOneHot
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

    def test_gym_new_spec_reg(self):
        Space = gym_backend("spaces").Space

        class MySpaceParent(Space):
            ...

        s_parent = MySpaceParent()

        class MySpaceChild(MySpaceParent):
            ...

        # We intentionally register first the child then the parent
        @register_gym_spec_conversion(MySpaceChild)
        def convert_myspace_child(spec, **kwargs):
            return NonTensor((), example_data="child")

        @register_gym_spec_conversion(MySpaceParent)
        def convert_myspace_parent(spec, **kwargs):
            return NonTensor((), example_data="parent")

        s_child = MySpaceChild()
        assert _gym_to_torchrl_spec_transform(s_parent).example_data == "parent"
        assert _gym_to_torchrl_spec_transform(s_child).example_data == "child"

        class NoConversionSpace(Space):
            ...

        s_no_conv = NoConversionSpace()
        with pytest.raises(
            KeyError, match="No conversion tool could be found with the gym space"
        ):
            _gym_to_torchrl_spec_transform(s_no_conv)

    @pytest.mark.parametrize("order", ["tuple_seq"])
    @implement_for("gym")
    def test_gym_spec_cast_tuple_sequential(self, order):
        torchrl_logger.info("Sequence not available in gym")
        return

    @pytest.mark.parametrize("order", ["tuple_seq"])
    @implement_for("gymnasium", "1.1.0")
    def test_gym_spec_cast_tuple_sequential(self, order):  # noqa: F811
        self._test_gym_spec_cast_tuple_sequential(order)

    @pytest.mark.parametrize("order", ["tuple_seq"])
    @implement_for("gymnasium", None, "1.0.0")
    def test_gym_spec_cast_tuple_sequential(self, order):  # noqa: F811
        # Sequence.stack parameter was added in gymnasium 1.0.0, skip for older versions
        torchrl_logger.info("Sequence.stack not available in gymnasium < 1.0.0")
        return

    def _test_gym_spec_cast_tuple_sequential(self, order):  # noqa: F811
        with set_gym_backend("gymnasium"):
            if order == "seq_tuple":
                # Requires nested tensors to be created along dim=1, disabling
                space = gym_backend("spaces").Dict(
                    feature=gym_backend("spaces").Sequence(
                        gym_backend("spaces").Tuple(
                            (
                                gym_backend("spaces").Box(-1, 1, shape=(2, 2)),
                                gym_backend("spaces").Box(-1, 1, shape=(1, 2)),
                            )
                        ),
                        stack=True,
                    )
                )
            elif order == "tuple_seq":
                space = gym_backend("spaces").Dict(
                    feature=gym_backend("spaces").Tuple(
                        (
                            gym_backend("spaces").Sequence(
                                gym_backend("spaces").Box(-1, 1, shape=(2, 2)),
                                stack=True,
                            ),
                            gym_backend("spaces").Sequence(
                                gym_backend("spaces").Box(-1, 1, shape=(1, 2)),
                                stack=True,
                            ),
                        ),
                    )
                )
            else:
                raise NotImplementedError
            sample = space.sample()

            # Custom tree_map that treats tuples and tensors as leaves
            def custom_tree_map(fn, obj):
                if isinstance(obj, (tuple, torch.Tensor)):
                    return fn(obj)
                elif isinstance(obj, dict):
                    return {k: custom_tree_map(fn, v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [custom_tree_map(fn, item) for item in obj]
                else:
                    return fn(obj)

            def stack_tuples(item):
                if isinstance(item, tuple):
                    try:
                        return torch.stack(
                            [custom_tree_map(stack_tuples, x) for x in item]
                        )
                    except RuntimeError:
                        item = [custom_tree_map(stack_tuples, x) for x in item]
                        try:
                            return torch.nested.nested_tensor(item)
                        except RuntimeError:
                            return tuple(item)
                return torch.as_tensor(item)

            sample_pt = custom_tree_map(stack_tuples, sample)
            spec = _gym_to_torchrl_spec_transform(space)
            rand = spec.rand()

            assert spec.contains(rand), (rand, spec)
            assert spec.contains(sample_pt), (rand, sample_pt)

            space_recon = _torchrl_to_gym_spec_transform(spec)
            assert space_recon == space, (space_recon, space)
            rand_numpy = rand.numpy()
            assert space.contains(rand_numpy)

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

    @implement_for("gym", None, "0.26")
    def test_gym_dict_action_space(self):
        torchrl_logger.info("tested for gym > 0.26 - no backward issue")
        return

    @implement_for("gym", "0.26", None)
    def test_gym_dict_action_space(self):  # noqa: F811
        import gym
        from gym import Env

        class CompositeActionEnv(Env):
            def __init__(self):
                self.action_space = gym.spaces.Dict(
                    a0=gym.spaces.Discrete(2), a1=gym.spaces.Box(-1, 1)
                )
                self.observation_space = gym.spaces.Box(-1, 1)

            def step(self, action):
                assert isinstance(action, dict)
                assert isinstance(action["a0"], np.ndarray)
                assert isinstance(action["a1"], np.ndarray)
                return (0.5, 0.0, False, False, {})

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict | None = None,
            ):
                return (0.0, {})

        env = CompositeActionEnv()
        torchrl_env = GymWrapper(env)
        assert isinstance(torchrl_env.action_spec, Composite)
        assert len(torchrl_env.action_keys) == 2
        r = torchrl_env.rollout(10)
        assert isinstance(r[0]["a0"], torch.Tensor)
        assert isinstance(r[0]["a1"], torch.Tensor)
        assert r[0]["observation"] == 0
        assert r[1]["observation"] == 0.5

    @implement_for("gymnasium")
    def test_gym_dict_action_space(self):  # noqa: F811
        import gymnasium as gym
        from gymnasium import Env

        class CompositeActionEnv(Env):
            def __init__(self):
                self.action_space = gym.spaces.Dict(
                    a0=gym.spaces.Discrete(2), a1=gym.spaces.Box(-1, 1)
                )
                self.observation_space = gym.spaces.Box(-1, 1)

            def step(self, action):
                assert isinstance(action, dict)
                assert isinstance(action["a0"], np.ndarray)
                assert isinstance(action["a1"], np.ndarray)
                return (0.5, 0.0, False, False, {})

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict | None = None,
            ):
                return (0.0, {})

        env = CompositeActionEnv()
        torchrl_env = GymWrapper(env)
        assert isinstance(torchrl_env.action_spec, Composite)
        assert len(torchrl_env.action_keys) == 2
        r = torchrl_env.rollout(10)
        assert isinstance(r[0]["a0"], torch.Tensor)
        assert isinstance(r[0]["a1"], torch.Tensor)
        assert r[0]["observation"] == 0
        assert r[1]["observation"] == 0.5

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
        if env_name == PONG_VERSIONED() and not _has_atari_for_gym():
            pytest.skip(
                "Atari not available for current gym version; skipping Atari gym test."
            )
        if env_name == HALFCHEETAH_VERSIONED() and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )

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
        if env_name == PONG_VERSIONED() and not _has_atari_for_gym():
            pytest.skip(
                "Atari not available for current gym version; skipping Atari gym test."
            )
        if env_name == HALFCHEETAH_VERSIONED() and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )

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

    @pytest.mark.parametrize("num_envs", [0, 1, 2])
    def test_mo_num_envs_vector_reward_spec(self, num_envs):
        if not _has_mo:
            pytest.skip("mo-gym not found")

        env = MOGymEnv("mo-mountaincarcontinuous-v0", num_envs=num_envs, device="cpu")
        try:
            expected_shape = torch.Size([2])
            if num_envs:
                expected_shape = torch.Size([num_envs, *expected_shape])
            assert env.reward_spec.shape == expected_shape
            td = env.rand_step(env.reset())
            assert td["next", "reward"].shape == expected_shape
        finally:
            env.close()

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
            # Check gym version - gym_super_mario_bros is not compatible with gym 0.26+
            # because it uses the old reset() API that returns only obs, not (obs, info)
            gym = gym_backend()
            gym_version = version.parse(gym.__version__)
            if gym_version >= version.parse("0.26.0"):
                pytest.skip(
                    "gym_super_mario_bros is not compatible with gym >= 0.26 "
                    "(uses old reset() API that returns only obs, not (obs, info))"
                )

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

    @implement_for("gymnasium", "1.1.0")
    def test_one_hot_and_categorical(self):
        self._test_one_hot_and_categorical()

    @implement_for("gymnasium", None, "1.0.0")
    def test_one_hot_and_categorical(self):  # noqa
        self._test_one_hot_and_categorical()

    def _test_one_hot_and_categorical(self):
        # tests that one-hot and categorical work ok when an integer is expected as action
        cliff_walking = GymEnv(
            CLIFFWALKING_VERSIONED(), categorical_action_encoding=True
        )
        cliff_walking.rollout(10)
        check_env_specs(cliff_walking)

        cliff_walking = GymEnv(
            CLIFFWALKING_VERSIONED(), categorical_action_encoding=False
        )
        cliff_walking.rollout(10)
        check_env_specs(cliff_walking)

    @implement_for("gym")
    def test_one_hot_and_categorical(self):  # noqa: F811
        # we do not skip (bc we may want to make sure nothing is skipped)
        # but CliffWalking-v0 in earlier Gym versions uses np.bool, which
        # was deprecated after np 1.20, and we don't want to install multiple np
        # versions.
        return

    @implement_for("gymnasium", "1.1.0")
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_wrapper(self, envname):
        import gymnasium

        if envname.startswith("ALE/") and not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        if "HalfCheetah" in envname and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )

        with set_gym_backend("gymnasium"):
            self._test_vecenvs_wrapper(
                envname,
                kwargs={"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP},
            )

    @implement_for("gymnasium", None, "1.0.0")
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_wrapper(self, envname):  # noqa
        if envname.startswith("ALE/") and not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        if "HalfCheetah" in envname and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )

        with set_gym_backend("gymnasium"):
            self._test_vecenvs_wrapper(envname)

    def _test_vecenvs_wrapper(self, envname, kwargs=None):
        import gymnasium

        # Skip if short env name is passed (from gym-decorated test parameterization)
        # This can happen due to implement_for/pytest parametrization interaction
        if envname in ("cp", "hc"):
            pytest.skip(
                f"Short env name '{envname}' not valid for gymnasium; "
                "this may be due to implement_for decorator issues."
            )

        if kwargs is None:
            kwargs = {}
        # we can't use parametrize with implement_for
        env = GymWrapper(
            gymnasium.vector.SyncVectorEnv(
                2 * [lambda envname=envname: gymnasium.make(envname)], **kwargs
            )
        )
        assert env.batch_size == torch.Size([2])
        check_env_specs(env)
        env = GymWrapper(
            gymnasium.vector.AsyncVectorEnv(
                2 * [lambda envname=envname: gymnasium.make(envname)], **kwargs
            )
        )
        assert env.batch_size == torch.Size([2])
        check_env_specs(env)

    @implement_for("gymnasium", "1.1.0")
    # this env has Dict-based observation which is a nice thing to test
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_env(self, envname):
        if envname.startswith("ALE/") and not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        if "HalfCheetah" in envname and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        self._test_vecenvs_env(envname)

    @implement_for("gymnasium", None, "1.0.0")
    # this env has Dict-based observation which is a nice thing to test
    @pytest.mark.parametrize(
        "envname",
        ["HalfCheetah-v4", "CartPole-v1", "ALE/Pong-v5"]
        + (["FetchReach-v2"] if _has_gym_robotics else []),
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_env(self, envname):  # noqa
        if envname.startswith("ALE/") and not _has_ale:
            pytest.skip("ALE not available (missing ale_py); skipping Atari gym test.")
        if "HalfCheetah" in envname and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        self._test_vecenvs_env(envname)

    def _test_vecenvs_env(self, envname):
        # Skip if short env name is passed (from gym-decorated test parameterization)
        # This can happen due to implement_for/pytest parametrization interaction
        if envname in ("cp", "hc"):
            pytest.skip(
                f"Short env name '{envname}' not valid for gymnasium; "
                "this may be due to implement_for decorator issues."
            )

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
        ["cp", "hc"],
    )
    @pytest.mark.flaky(reruns=5, reruns_delay=1)
    def test_vecenvs_wrapper(self, envname):  # noqa: F811
        if envname == "hc" and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        with set_gym_backend("gym"):
            gym = gym_backend()
            if envname == "hc":
                envname = HALFCHEETAH_VERSIONED()
            else:
                envname = CARTPOLE_VERSIONED()
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
        if envname == "hc" and not _has_mujoco:
            pytest.skip(
                "MuJoCo not available (missing mujoco); skipping MuJoCo gym test."
            )
        # Skip HalfCheetah with gym 0.25.x due to AsyncVectorEnv subprocess issues
        if envname == "hc":
            gym = gym_backend()
            gym_version = version.parse(gym.__version__)
            if version.parse("0.25.0") <= gym_version < version.parse("0.26.0"):
                pytest.skip(
                    "Skipping HalfCheetah vecenvs test for gym 0.25.x due to AsyncVectorEnv subprocess issues"
                )
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
            if "CartPole" not in envname:
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
        ["cp", "hc"],
    )
    def test_vecenvs_wrapper(self, envname):  # noqa: F811
        # skipping tests for older versions of gym
        ...

    @implement_for("gym", None, "0.18")
    @pytest.mark.parametrize(
        "envname",
        ["cp", "hc"],
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

    @implement_for("gymnasium", "1.1.0")
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_gym_output_num(self, wrapper):  # noqa: F811
        self._test_gym_output_num(wrapper)

    @implement_for("gymnasium", None, "1.0.0")
    @pytest.mark.parametrize("wrapper", [True, False])
    def test_gym_output_num(self, wrapper):  # noqa: F811
        self._test_gym_output_num(wrapper)

    def _test_gym_output_num(self, wrapper):  # noqa: F811
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
        c = Collector(
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
        c = Collector(
            env, RandomPolicy(env.action_spec), total_frames=2000, frames_per_batch=200
        )
        for rollout in c:
            assert torch.isfinite(rollout.get("observation")).all()
            assert torch.isfinite(rollout.get(("next", "observation"))).all()
            break
        del c
        return

    @implement_for("gymnasium", "1.1.0")
    def test_vecenvs_nan(self):  # noqa: F811
        self._test_vecenvs_nan()

    @implement_for("gymnasium", None, "1.0.0")
    def test_vecenvs_nan(self):  # noqa: F811
        self._test_vecenvs_nan()

    def _test_vecenvs_nan(self):  # noqa: F811
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
        c = Collector(
            env, RandomPolicy(env.action_spec), total_frames=2000, frames_per_batch=200
        )
        for rollout in c:
            assert torch.isfinite(rollout.get("observation")).all()
            assert torch.isfinite(rollout.get(("next", "observation"))).all()
            break
        del c
        return

    def _get_dummy_gym_env(self, backend, **kwargs):
        with set_gym_backend(backend):

            class CustomEnv(gym_backend().Env):
                def __init__(self, dim=3, use_termination=True, max_steps=4):
                    self.dim = dim
                    self.use_termination = use_termination
                    self.observation_space = gym_backend("spaces").Box(
                        low=-np.inf, high=np.inf, shape=(self.dim,), dtype=np.float32
                    )
                    self.action_space = gym_backend("spaces").Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                    )
                    self.max_steps = max_steps

                def _get_info(self):
                    return {"field1": self.state**2}

                def _get_obs(self):
                    return self.state.copy()

                def reset(self, seed=0, options=None):
                    self.state = np.zeros(
                        self.observation_space.shape, dtype=np.float32
                    )
                    observation = self._get_obs()
                    info = self._get_info()
                    assert (observation < self.max_steps).all()
                    return observation, info

                def step(self, action):
                    # self.state += action.item()
                    self.state += 1
                    truncated, terminated = False, False
                    if self.use_termination:
                        terminated = self.state[0] == 4
                    reward = 1 if terminated else 0  # Binary sparse rewards
                    observation = self._get_obs()
                    info = self._get_info()
                    return observation, reward, terminated, truncated, info

            return CustomEnv(**kwargs)

    def counting_env(self):
        import gymnasium as gym
        from gymnasium import Env

        class CountingEnvRandomReset(Env):
            def __init__(self, i=0):
                self.counter = 1
                self.i = i
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
                self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
                self.rng = np.random.RandomState(0)

            def step(self, action):
                self.counter += 1
                done = bool(self.rng.random() < 0.05)
                return (
                    np.asarray(
                        [
                            self.counter,
                        ]
                    ),
                    0,
                    done,
                    done,
                    {},
                )

            def reset(
                self,
                *,
                seed: int | None = None,
                options=None,
            ):
                self.counter = 1
                if seed is not None:
                    self.rng = np.random.RandomState(seed)
                return (
                    np.asarray(
                        [
                            self.counter,
                        ]
                    ),
                    {},
                )

        return CountingEnvRandomReset

    @implement_for("gym")
    def test_gymnasium_autoreset(self, venv):
        return

    @implement_for("gymnasium", None, "1.1.0")
    def test_gymnasium_autoreset(self, venv):  # noqa
        return

    @implement_for("gymnasium", "1.1.0")
    @pytest.mark.parametrize("venv", ["sync", "async"])
    def test_gymnasium_autoreset(self, venv):  # noqa
        import gymnasium as gym

        set_gym_backend("gymnasium").set()

        counting_env = self.counting_env()
        if venv == "sync":
            venv = gym.vector.SyncVectorEnv
        else:
            venv = gym.vector.AsyncVectorEnv
        envs0 = venv(
            [lambda i=i: counting_env(i) for i in range(2)],
            autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        )
        env = GymWrapper(envs0)
        envs0.reset(seed=0)
        torch.manual_seed(0)
        r0 = env.rollout(20, break_when_any_done=False)
        envs1 = venv(
            [lambda i=i: counting_env(i) for i in range(2)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        env = GymWrapper(envs1)
        envs1.reset(seed=0)
        # env.set_seed(0)
        torch.manual_seed(0)
        r1 = []
        t_ = env.reset()
        for s in r0.unbind(-1):
            t_.set("action", s["action"])
            t, t_ = env.step_and_maybe_reset(t_)
            r1.append(t)
        r1 = torch.stack(r1, -1)
        torch.testing.assert_close(r0["observation"], r1["observation"])
        torch.testing.assert_close(r0["next", "observation"], r1["next", "observation"])
        torch.testing.assert_close(r0["next", "done"], r1["next", "done"])

    @implement_for("gym")
    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_resetting_strategies(self, heterogeneous):
        return

    @implement_for("gymnasium", None, "1.0.0")
    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_resetting_strategies(self, heterogeneous):  # noqa
        # Skip for gymnasium < 1.0.0 because the autoreset behavior is different
        # and doesn't match the test's expectations for observation tracking
        torchrl_logger.info(
            "Skipping test_resetting_strategies for gymnasium < 1.0.0 due to different autoreset behavior"
        )
        return

    @implement_for("gymnasium", "1.1.0")
    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_resetting_strategies(self, heterogeneous):  # noqa
        import gymnasium as gym

        self._test_resetting_strategies(
            heterogeneous, {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
        )

    def _test_resetting_strategies(self, heterogeneous, kwargs):
        if _has_gymnasium:
            backend = "gymnasium"
        else:
            backend = "gym"
        with set_gym_backend(backend):
            if version.parse(gym_backend().__version__) < version.parse("0.26"):
                torchrl_logger.info(
                    "Running into unrelated errors with older versions of gym."
                )
                return
            steps = 5
            if not heterogeneous:
                env = GymWrapper(
                    gym_backend().vector.AsyncVectorEnv(
                        [functools.partial(self._get_dummy_gym_env, backend=backend)]
                        * 4,
                        **kwargs,
                    )
                )
            else:
                env = GymWrapper(
                    gym_backend().vector.AsyncVectorEnv(
                        [
                            functools.partial(
                                self._get_dummy_gym_env,
                                max_steps=i + 4,
                                backend=backend,
                            )
                            for i in range(4)
                        ],
                        **kwargs,
                    )
                )
            try:
                check_env_specs(env)
                td = env.rollout(steps, break_when_any_done=False)
                if not heterogeneous:
                    assert not (td["observation"] == 4).any()
                    assert (td["next", "observation"] == 4).sum() == 3 * 4

                # check with manual reset
                torch.manual_seed(0)
                env.set_seed(0)
                reset = env.reset(
                    TensorDict({"_reset": torch.ones(4, 1, dtype=torch.bool)}, [4])
                )
                r0 = env.rollout(
                    10, break_when_any_done=False, auto_reset=False, tensordict=reset
                )
                torch.manual_seed(0)
                env.set_seed(0)
                reset = env.reset()
                r1 = env.rollout(
                    10, break_when_any_done=False, auto_reset=False, tensordict=reset
                )
                torch.manual_seed(0)
                env.set_seed(0)
                r2 = env.rollout(10, break_when_any_done=False)
                assert_allclose_td(r0, r1)
                assert_allclose_td(r1, r2)
                for r in (r0, r1, r2):
                    torch.testing.assert_close(r["field1"], r["observation"].pow(2))
                    torch.testing.assert_close(
                        r["next", "field1"], r["next", "observation"].pow(2)
                    )

            finally:
                if not env.is_closed:
                    env.close()
                del env
                gc.collect()

    def test_is_from_pixels_simple_env(self):
        """Test that _is_from_pixels correctly identifies non-pixel environments."""
        from torchrl.envs.libs.gym import _is_from_pixels

        # Test with a simple environment that doesn't have pixels
        class SimpleEnv:
            def __init__(self):
                try:
                    import gymnasium as gym
                except ImportError:
                    import gym
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

        env = SimpleEnv()

        # This should return False since it's not a pixel environment
        result = _is_from_pixels(env)
        assert result is False, f"Expected False for simple environment, got {result}"

    def test_is_from_pixels_box_env(self):
        """Test that _is_from_pixels correctly identifies pixel Box environments."""
        from torchrl.envs.libs.gym import _is_from_pixels

        # Test with a pixel-like environment
        class PixelEnv:
            def __init__(self):
                try:
                    import gymnasium as gym
                except ImportError:
                    import gym
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3)
                )

        pixel_env = PixelEnv()

        # This should return True since it's a pixel environment
        result = _is_from_pixels(pixel_env)
        assert result is True, f"Expected True for pixel environment, got {result}"

    def test_is_from_pixels_dict_env(self):
        """Test that _is_from_pixels correctly identifies Dict environments with pixels."""
        from torchrl.envs.libs.gym import _is_from_pixels

        # Test with a Dict environment that has pixels
        class DictPixelEnv:
            def __init__(self):
                try:
                    import gymnasium as gym
                except ImportError:
                    import gym
                self.observation_space = gym.spaces.Dict(
                    {
                        "pixels": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3)),
                        "state": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                    }
                )

        dict_pixel_env = DictPixelEnv()

        # This should return True since it has a "pixels" key
        result = _is_from_pixels(dict_pixel_env)
        assert (
            result is True
        ), f"Expected True for Dict environment with pixels, got {result}"

    def test_is_from_pixels_dict_env_no_pixels(self):
        """Test that _is_from_pixels correctly identifies Dict environments without pixels."""
        from torchrl.envs.libs.gym import _is_from_pixels

        # Test with a Dict environment that doesn't have pixels
        class DictNoPixelEnv:
            def __init__(self):
                try:
                    import gymnasium as gym
                except ImportError:
                    import gym
                self.observation_space = gym.spaces.Dict(
                    {
                        "state": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                        "features": gym.spaces.Box(low=0, high=1, shape=(5,)),
                    }
                )

        dict_no_pixel_env = DictNoPixelEnv()

        # This should return False since it doesn't have a "pixels" key
        result = _is_from_pixels(dict_no_pixel_env)
        assert (
            result is False
        ), f"Expected False for Dict environment without pixels, got {result}"

    def test_num_workers_returns_parallel_env(self):
        """Ensure explicit TorchRL `num_workers` returns a lazy ParallelEnv, while gym's
        native `num_envs` remains a gym-native vectorization."""

        # TorchRL-managed parallelism: should return ParallelEnv
        env = GymEnv("CartPole-v1", num_workers=3)
        try:
            assert isinstance(env, ParallelEnv)
            # accept either attribute name used by ParallelEnv implementations
            nworkers = getattr(env, "num_workers", None)
            if nworkers is None:
                nworkers = getattr(env, "num_envs", None)
            assert nworkers == 3
            # start workers on first use
            env.reset()
            assert env.batch_size == torch.Size([3])
        finally:
            env.close()

        # Gym-native vectorization should NOT be converted implicitly by TorchRL
        env_gymvec = GymEnv("CartPole-v1", num_envs=3)
        try:
            assert not isinstance(env_gymvec, ParallelEnv)
        finally:
            env_gymvec.close()

    def test_num_workers_kwargs_modifiable(self):
        """Ensure the kwargs preserved by the GymEnv factory can be modified via
        `configure_parallel` before workers start."""

        env = GymEnv("CartPole-v1", num_workers=3)
        try:
            # should return a lazy ParallelEnv
            assert isinstance(env, ParallelEnv)

            # configure_parallel should accept kwargs and be callable before start
            env.configure_parallel(use_buffers=True, num_threads=1)

            # starting the environment should work after configuring
            td = env.reset()
            assert isinstance(td, TensorDict)
        finally:
            env.close()

    def test_set_seed_and_reset_works(self):
        """Smoke test that setting seed and reset works (seed forwarded into build)."""
        env = GymEnv("CartPole-v1")
        final_seed = env.set_seed(0)
        assert final_seed is not None
        td = env.reset()

        assert isinstance(td, TensorDict)
        env.close()

        # Also verify behavior for TorchRL-managed parallel envs
        penv = GymEnv("CartPole-v1", num_workers=2)
        try:
            final_seed = penv.set_seed(0)
            assert final_seed is not None
            td = penv.reset()
            assert isinstance(td, TensorDict)
        finally:
            penv.close()

    def test_gym_kwargs_preserved_with_seed(self):
        """Test that kwargs like frame_skip are preserved when seed is provided.
        Regression test for a bug where `kwargs` were overwritten when `_seed` was not None.
        """
        # Use Pendulum instead of CartPole because CartPole can terminate
        # early due to pole falling, especially with frame_skip=4
        env = GymEnv(PENDULUM_VERSIONED(), frame_skip=4, from_pixels=False)
        try:
            td = env.reset()
            rollout = env.rollout(max_steps=5)
            assert rollout.shape[0] == 5
            assert "observation" in td.keys()
        finally:
            env.close()

    def test_is_from_pixels_wrapper_env(self):
        """Test that _is_from_pixels correctly identifies wrapped environments."""
        from torchrl.envs.libs.gym import _is_from_pixels

        # Test with a mock environment that simulates being wrapped with a pixel wrapper
        class MockWrappedEnv:
            def __init__(self):
                try:
                    import gymnasium as gym
                except ImportError:
                    import gym
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3)
                )

        # Mock the isinstance check to simulate the wrapper detection
        import torchrl.envs.libs.utils

        original_isinstance = isinstance

        def mock_isinstance(obj, cls):
            if cls == torchrl.envs.libs.utils.GymPixelObservationWrapper:
                return True
            return original_isinstance(obj, cls)

        # Temporarily patch isinstance
        import builtins

        builtins.isinstance = mock_isinstance

        try:
            wrapped_env = MockWrappedEnv()

            # This should return True since it's detected as a pixel wrapper
            result = _is_from_pixels(wrapped_env)
            assert (
                result is True
            ), f"Expected True for wrapped environment, got {result}"
        finally:
            # Restore original isinstance
            builtins.isinstance = original_isinstance

    @pytest.mark.parametrize("num_envs", [0, 1, 2])
    def test_gymnasium_num_envs(self, num_envs, request):
        if not _has_gymnasium:
            pytest.skip("gymnasium not found")

        gym_version = version.parse(gymnasium.__version__)
        if version.parse("1.0.0") <= gym_version < version.parse("1.1.0"):
            pytest.skip("gymnasium 1.0 is not supported")

        with set_gym_backend("gymnasium"):
            env = GymEnv("CartPole-v1", num_envs=num_envs)
        request.addfinalizer(env.close)

        if num_envs > 0:
            expected_batch_size = torch.Size([num_envs])
            expected_reward_shape = torch.Size([num_envs, 1])
        else:
            expected_batch_size = torch.Size([])
            expected_reward_shape = torch.Size([1])

        assert env.batch_size == expected_batch_size
        check_env_specs(env)
        td = env.rand_step(env.reset())
        assert td["next", "reward"].shape == expected_reward_shape


@pytest.mark.skipif(
    not _has_minigrid or not _has_gymnasium, reason="MiniGrid not found"
)
class TestMiniGrid:
    @pytest.mark.parametrize(
        "id",
        [
            "BabyAI-KeyCorridorS6R3-v0",
            "MiniGrid-Empty-16x16-v0",
            "MiniGrid-BlockedUnlockPickup-v0",
        ],
    )
    def test_minigrid(self, id):
        env_base = gymnasium.make(id)
        env = GymWrapper(env_base)
        check_env_specs(env)


@implement_for("gym", None, "0.25")
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name)


@implement_for("gym", "0.25", None)
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name, render_mode="rgb_array")


@implement_for("gymnasium", None, "1.0.0")
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name, render_mode="rgb_array")


@implement_for("gymnasium", "1.1.0")
def _make_gym_environment(env_name):  # noqa: F811
    gym = gym_backend()
    return gym.make(env_name, render_mode="rgb_array")
