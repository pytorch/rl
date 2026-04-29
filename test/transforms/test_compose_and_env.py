# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import itertools

from copy import copy
from functools import partial

import pytest

import tensordict.tensordict
import torch

from _transforms_common import mp_ctx, TransformBase
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import assert_allclose_td
from torch import nn
from torchrl._utils import set_auto_unwrap_transformed_env

from torchrl.data import (
    Bounded,
    Categorical,
    Composite,
    LazyTensorStorage,
    RandomSampler,
    ReplayBuffer,
    TensorDictReplayBuffer,
    TensorSpec,
    Unbounded,
    UnboundedContinuous,
)
from torchrl.envs import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DiscreteActionProjection,
    DoubleToFloat,
    EnvBase,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    MultiStepTransform,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    PinMemoryTransform,
    RandomCropTensorDict,
    RenameTransform,
    Resize,
    RewardClipping,
    RewardScaling,
    SerialEnv,
    SqueezeTransform,
    StepCounter,
    TensorDictPrimer,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import VecNorm
from torchrl.envs.transforms.transforms import _has_tv, BatchSizeTransform, Transform
from torchrl.envs.utils import check_env_specs
from torchrl.modules import GRUModule
from torchrl.modules.utils import get_primers_from_module

from torchrl.testing import (  # noqa
    BREAKOUT_VERSIONED,
    dtype_fixture,
    get_default_devices,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
    rand_reset,
    retry,
)
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingEnv,
    MockBatchedLockedEnv,
    MockBatchedUnLockedEnv,
    MultiKeyCountingEnv,
)


def test_added_transforms_are_in_eval_mode_trivial():
    base_env = ContinuousActionVecMockEnv()
    t = TransformedEnv(base_env)
    assert not t.transform.training

    t.train()
    assert t.transform.training


def test_added_transforms_are_in_eval_mode():
    base_env = ContinuousActionVecMockEnv()
    r = RewardScaling(0, 1)
    t = TransformedEnv(base_env, r)
    assert not t.transform.training
    t.append_transform(RewardScaling(0, 1))
    assert not t.transform[1].training

    t.train()
    assert t.transform.training
    assert t.transform[0].training
    assert t.transform[1].training


class TestTransformedEnv:
    class DummyCompositeEnv(EnvBase):  # type: ignore[misc]
        """A dummy environment with a composite action set."""

        def __init__(self) -> None:
            super().__init__()

            self.observation_spec = Composite(
                observation=UnboundedContinuous((*self.batch_size, 3))
            )

            self.action_spec = Composite(
                action=Composite(
                    head_0=Composite(
                        action=Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
                    ),
                    head_1=Composite(
                        action=Categorical(2, (*self.batch_size, 1), dtype=torch.bool)
                    ),
                )
            )

            self.done_spec = Categorical(2, (*self.batch_size, 1), dtype=torch.bool)

            self.full_done_spec["truncated"] = self.full_done_spec["terminated"].clone()

            self.reward_spec = UnboundedContinuous(*self.batch_size, 1)

        def _reset(self, tensordict: TensorDict) -> TensorDict:
            return TensorDict(
                {"observation": torch.randn((*self.batch_size, 3)), "done": False}
            )

        def _step(self, tensordict: TensorDict) -> TensorDict:
            return TensorDict(
                {
                    "observation": torch.randn((*self.batch_size, 3)),
                    "done": False,
                    "reward": torch.randn((*self.batch_size, 1)),
                }
            )

        def _set_seed(self, seed: int) -> None:
            pass

    def test_no_modif_specs(self) -> None:
        base_env = self.DummyCompositeEnv()
        specs = base_env.specs.clone()
        transformed_env = TransformedEnv(
            base_env,
            RenameTransform(
                in_keys=[],
                out_keys=[],
                in_keys_inv=[("action", "head_0", "action")],
                out_keys_inv=[("action", "head_99", "action")],
            ),
        )
        td = transformed_env.reset()
        # A second reset with a TD passed fails due to override of the `input_spec`
        td = transformed_env.reset(td)
        specs_after = base_env.specs.clone()
        assert specs == specs_after

    @pytest.mark.filterwarnings("error")
    def test_nested_transformed_env(self):
        base_env = ContinuousActionVecMockEnv()
        t1 = RewardScaling(0, 1)
        t2 = RewardScaling(0, 2)

        def test_unwrap():
            env = TransformedEnv(TransformedEnv(base_env, t1), t2)
            assert env.base_env is base_env
            assert isinstance(env.transform, Compose)
            children = list(env.transform.transforms.children())
            assert len(children) == 2
            assert children[0].scale == 1
            assert children[1].scale == 2

        def test_wrap(auto_unwrap=None):
            env = TransformedEnv(
                TransformedEnv(base_env, t1), t2, auto_unwrap=auto_unwrap
            )
            assert env.base_env is not base_env
            assert isinstance(env.base_env.transform, RewardScaling)
            assert isinstance(env.transform, RewardScaling)

        with pytest.warns(FutureWarning):
            test_unwrap()

        test_wrap(False)

        with set_auto_unwrap_transformed_env(True):
            test_unwrap()

        with set_auto_unwrap_transformed_env(False):
            test_wrap()

    def test_attr_error(self):
        class BuggyTransform(Transform):
            def transform_observation_spec(
                self, observation_spec: TensorSpec
            ) -> TensorSpec:
                raise AttributeError

            def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
                raise RuntimeError("reward!")

        env = TransformedEnv(CountingEnv(), BuggyTransform())
        with pytest.raises(
            AttributeError, match="because an internal error was raised"
        ):
            env.observation_spec
        with pytest.raises(
            AttributeError, match="'CountingEnv' object has no attribute 'tralala'"
        ):
            env.tralala
        with pytest.raises(RuntimeError, match="reward!"):
            env.transform.transform_reward_spec(env.base_env.full_reward_spec)

    def test_independent_obs_specs_from_shared_env(self):
        obs_spec = Composite(
            observation=Bounded(low=0, high=10, shape=torch.Size((1,)))
        )
        base_env = ContinuousActionVecMockEnv(observation_spec=obs_spec)
        t1 = TransformedEnv(
            base_env, transform=ObservationNorm(in_keys=["observation"], loc=3, scale=2)
        )
        t2 = TransformedEnv(
            base_env, transform=ObservationNorm(in_keys=["observation"], loc=1, scale=6)
        )

        t1_obs_spec = t1.observation_spec
        t2_obs_spec = t2.observation_spec

        assert t1_obs_spec["observation"].space.low == 3
        assert t1_obs_spec["observation"].space.high == 23

        assert t2_obs_spec["observation"].space.low == 1
        assert t2_obs_spec["observation"].space.high == 61

        assert base_env.observation_spec["observation"].space.low == 0
        assert base_env.observation_spec["observation"].space.high == 10

    def test_independent_reward_specs_from_shared_env(self):
        reward_spec = Unbounded()
        base_env = ContinuousActionVecMockEnv(reward_spec=reward_spec)
        t1 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=0, clamp_max=4)
        )
        t2 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=-2, clamp_max=2)
        )

        t1_reward_spec = t1.reward_spec
        t2_reward_spec = t2.reward_spec

        assert t1_reward_spec.space.low == 0
        assert t1_reward_spec.space.high == 4

        assert t2_reward_spec.space.low == -2
        assert t2_reward_spec.space.high == 2

        assert (
            base_env.reward_spec.space.low
            == torch.finfo(base_env.reward_spec.dtype).min
        )
        assert (
            base_env.reward_spec.space.high
            == torch.finfo(base_env.reward_spec.dtype).max
        )

    def test_allow_done_after_reset(self):
        base_env = ContinuousActionVecMockEnv(allow_done_after_reset=True)
        assert base_env._allow_done_after_reset
        t1 = TransformedEnv(
            base_env, transform=RewardClipping(clamp_min=0, clamp_max=4)
        )
        assert t1._allow_done_after_reset
        with pytest.raises(
            RuntimeError,
            match="_allow_done_after_reset is a read-only property for TransformedEnvs",
        ):
            t1._allow_done_after_reset = False
        base_env._allow_done_after_reset = False
        assert not t1._allow_done_after_reset


def test_transform_parent():
    base_env = ContinuousActionVecMockEnv()
    t1 = RewardScaling(0, 1)
    t2 = RewardScaling(0, 2)
    env = TransformedEnv(TransformedEnv(base_env, t1), t2)
    t3 = RewardClipping(0.1, 0.5)
    env.append_transform(t3)

    t1_parent_gt = t1._container
    t2_parent_gt = t2._container
    t3_parent_gt = t3._container

    _ = t1.parent
    _ = t2.parent
    _ = t3.parent

    assert t1_parent_gt == t1._container
    assert t2_parent_gt == t2._container
    assert t3_parent_gt == t3._container


def test_transform_parent_cache():
    """Tests the caching and uncaching of the transformed envs."""
    env = TransformedEnv(
        ContinuousActionVecMockEnv(),
        FrameSkipTransform(3),
    )

    # print the parent
    assert (
        type(env.transform.parent.transform) is Compose
        and len(env.transform.parent.transform) == 0
    )
    transform = env.transform
    parent1 = env.transform.parent
    parent2 = env.transform.parent
    assert parent1 is parent2

    # change the env, re-print the parent
    env.insert_transform(0, NoopResetEnv(3))
    parent3 = env.transform[-1].parent
    assert parent1 is not parent3
    assert type(parent3.transform[0]) is NoopResetEnv

    # change the env, re-print the parent
    env.insert_transform(0, CatTensors(["observation"]))
    parent4 = env.transform[-1].parent
    assert parent1 is not parent4
    assert parent3 is not parent4
    assert type(parent4.transform[0]) is CatTensors
    assert type(parent4.transform[1]) is NoopResetEnv

    # check that we don't keep track of the wrong parent
    env.transform = NoopResetEnv(3)
    assert transform.parent is None


class TestTransforms:
    @pytest.mark.parametrize("batch", [[], [1], [3, 2]])
    @pytest.mark.parametrize(
        "keys",
        [["next_observation", "some_other_key"], [("next", "observation_pixels")]],
    )
    @pytest.mark.parametrize("device", get_default_devices())
    def test_compose(self, keys, batch, device, nchannels=1, N=4):
        torch.manual_seed(0)
        t1 = CatFrames(
            in_keys=keys,
            N=4,
            dim=-3,
        )
        t2 = FiniteTensorDictCheck()
        t3 = ExcludeTransform()
        compose = Compose(t1, t2, t3)
        dont_touch = torch.randn(*batch, nchannels, 16, 16, device=device)
        td = TensorDict(
            {
                key: torch.randint(255, (*batch, nchannels, 16, 16), device=device)
                for key in keys
            },
            batch,
            device=device,
        )
        td.set("dont touch", dont_touch.clone())
        if not batch:
            with pytest.raises(
                ValueError,
                match="CatFrames cannot process unbatched tensordict instances",
            ):
                compose(td.clone(False))
        compose._call(td)
        for key in keys:
            assert td.get(key).shape[-3] == nchannels * N
        assert (td.get("dont touch") == dont_touch).all()

        if len(keys) == 1:
            observation_spec = Bounded(0, 255, (nchannels, 16, 16))
            # StepCounter does not want non composite specs
            observation_spec = compose[:2].transform_observation_spec(
                observation_spec.clone()
            )
            assert observation_spec.shape == torch.Size([nchannels * N, 16, 16])
        else:
            observation_spec = Composite(
                {key: Bounded(0, 255, (nchannels, 16, 16)) for key in keys}
            )
            observation_spec = compose.transform_observation_spec(
                observation_spec.clone()
            )
            for key in keys:
                assert observation_spec[key].shape == torch.Size(
                    [nchannels * N, 16, 16]
                )

    def test_compose_pop(self):
        t1 = CatFrames(in_keys=["a", "b"], N=2, dim=-1)
        t2 = FiniteTensorDictCheck()
        t3 = ExcludeTransform()
        compose = Compose(t1, t2, t3)
        assert len(compose.transforms) == 3
        p = compose.pop()
        assert p is t3
        assert len(compose.transforms) == 2
        p = compose.pop(0)
        assert p is t1
        assert len(compose.transforms) == 1
        p = compose.pop()
        assert p is t2
        assert len(compose.transforms) == 0
        with pytest.raises(IndexError, match="index -1 is out of range"):
            compose.pop()

    def test_compose_pop_parent_modification(self):
        t1 = CatFrames(in_keys=["a", "b"], N=2, dim=-1)
        t2 = FiniteTensorDictCheck()
        t3 = ExcludeTransform()
        compose = Compose(t1, t2, t3)
        env = TransformedEnv(ContinuousActionVecMockEnv(), compose)
        p = t2.parent
        assert isinstance(p.transform[0], CatFrames)
        env.transform.pop(0)
        assert env.transform[0] is t2
        new_p = t2.parent
        assert new_p is not p
        assert len(new_p.transform) == 0

    def test_lambda_functions(self):
        def trsf(data):
            if "y" in data.keys():
                data["y"] += 1
                return data
            return data.set("y", torch.zeros(data.shape))

        env = TransformedEnv(CountingEnv(5), trsf)
        env.append_transform(trsf)
        env.insert_transform(0, trsf)
        # With Compose
        env.transform.append(trsf)
        assert env.reset().get("y") == 3
        env.transform = trsf
        assert env.reset().get("y") == 0

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("keys_inv_1", [["action_1"], []])
    @pytest.mark.parametrize("keys_inv_2", [["action_2"], []])
    def test_compose_inv(self, keys_inv_1, keys_inv_2, device):
        torch.manual_seed(0)
        keys_to_transform = set(keys_inv_1 + keys_inv_2)
        keys_total = {"action_1", "action_2", "dont_touch"}
        double2float_1 = DoubleToFloat(in_keys_inv=keys_inv_1)
        double2float_2 = DoubleToFloat(in_keys_inv=keys_inv_2)
        compose = Compose(double2float_1, double2float_2)
        td = TensorDict(
            {
                key: torch.zeros(1, 3, 3, dtype=torch.float32, device=device)
                for key in keys_total
            },
            [1],
            device=device,
        )

        td = compose.inv(td)
        for key in keys_to_transform:
            assert td.get(key).dtype == torch.double
        for key in keys_total - keys_to_transform:
            assert td.get(key).dtype == torch.float32

    def test_compose_indexing(self):
        c = Compose(
            ObservationNorm(loc=1.0, scale=1.0, in_keys=["observation"]),
            RewardScaling(loc=0, scale=1),
            ObservationNorm(loc=2.0, scale=2.0, in_keys=["observation"]),
        )
        base_env = ContinuousActionVecMockEnv()
        env = TransformedEnv(base_env, c)
        last_t = env.transform[-1]
        assert last_t.scale == 2
        env.transform[-1].scale += 1
        assert last_t.scale == 3
        # indexing a sequence of transforms involves re-creating a Compose, which requires a clone
        # because we need to deparent the transforms
        sub_compose = env.transform[1:]
        assert isinstance(sub_compose, Compose)
        last_t2 = sub_compose[-1]
        assert last_t2.scale == 3
        # this involves clone, but the value of the registered buffer should still match
        env.transform[1:][-1].scale += 1
        assert last_t.scale == 4
        assert last_t2.scale == 4

    def test_compose_action_spec(self):
        # Create a Compose transform that renames "action" to "action_1" and then to "action_2"
        c = Compose(
            RenameTransform(
                in_keys=(),
                out_keys=(),
                in_keys_inv=("action",),
                out_keys_inv=("action_1",),
            ),
            RenameTransform(
                in_keys=(),
                out_keys=(),
                in_keys_inv=("action_1",),
                out_keys_inv=("action_2",),
            ),
        )
        base_env = ContinuousActionVecMockEnv()
        env = TransformedEnv(base_env, c)

        # Check the `full_action_spec`s
        assert "action_2" in env.full_action_spec
        # Ensure intermediate keys are no longer in the action spec
        assert "action_1" not in env.full_action_spec
        assert "action" not in env.full_action_spec

        # Final check to ensure clean sampling from the action_spec
        action = env.rand_action()
        assert "action_2" in action

    @pytest.mark.parametrize("device", get_default_devices())
    def test_finitetensordictcheck(self, device):
        ftd = FiniteTensorDictCheck()
        td = TensorDict(
            {key: torch.randn(1, 3, 3, device=device) for key in ["a", "b", "c"]}, [1]
        )
        ftd._call(td)
        td.set("inf", torch.zeros(1, 3).fill_(float("inf")))
        with pytest.raises(ValueError, match="Encountered a non-finite tensor"):
            ftd._call(td)
        with pytest.raises(ValueError, match="Encountered a non-finite tensor"):
            ftd(td)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda device found")
    @pytest.mark.parametrize("device", get_default_devices())
    def test_pin_mem(self, device):
        pin_mem = PinMemoryTransform()
        td = TensorDict(
            {key: torch.randn(3) for key in ["a", "b", "c"]}, [], device=device
        )
        if device.type == "cuda":
            with pytest.raises(RuntimeError, match="cannot pin"):
                pin_mem(td)
            with pytest.raises(RuntimeError, match="cannot pin"):
                pin_mem._call(td)
            return
        pin_mem(td)
        for item in td.values():
            assert item.is_pinned

    def test_append(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        (key,) = itertools.islice(obs_spec.keys(), 1)

        env = TransformedEnv(env)
        env.append_transform(CatFrames(N=4, dim=-1, in_keys=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 1
        obs_spec = env.observation_spec
        obs_spec = obs_spec[key]
        assert obs_spec.shape[-1] == 4 * env.base_env.observation_spec[key].shape[-1]

    def test_insert(self):
        env = ContinuousActionVecMockEnv()
        obs_spec = env.observation_spec
        (key,) = itertools.islice(obs_spec.keys(), 1)
        env = TransformedEnv(env)

        # we start by asking the spec. That will create the private attributes
        _ = env.action_spec
        _ = env.observation_spec
        _ = env.reward_spec

        assert env._input_spec is not None
        assert "full_action_spec" in env._input_spec
        assert env._input_spec["full_action_spec"] is not None
        assert env._output_spec["full_observation_spec"] is not None
        assert env._output_spec["full_reward_spec"] is not None
        assert env._output_spec["full_done_spec"] is not None

        env.insert_transform(0, CatFrames(N=4, dim=-1, in_keys=[key]))

        # transformed envs do not have spec after insert -- they need to be computed
        assert env._input_spec is None
        assert env._output_spec is None

        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 1
        obs_spec = env.observation_spec
        obs_spec = obs_spec[key]
        assert obs_spec.shape[-1] == 4 * env.base_env.observation_spec[key].shape[-1]

        env.insert_transform(1, FiniteTensorDictCheck())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 2
        assert isinstance(env.transform[-1], FiniteTensorDictCheck)
        assert isinstance(env.transform[0], CatFrames)

        env.insert_transform(0, NoopResetEnv())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 3
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], CatFrames)
        assert isinstance(env.transform[2], FiniteTensorDictCheck)

        env.insert_transform(2, NoopResetEnv())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 4
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], CatFrames)
        assert isinstance(env.transform[2], NoopResetEnv)
        assert isinstance(env.transform[3], FiniteTensorDictCheck)

        env.insert_transform(-3, PinMemoryTransform())
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 5
        assert isinstance(env.transform[0], NoopResetEnv)
        assert isinstance(env.transform[1], PinMemoryTransform)
        assert isinstance(env.transform[2], CatFrames)
        assert isinstance(env.transform[3], NoopResetEnv)
        assert isinstance(env.transform[4], FiniteTensorDictCheck)

        assert env._input_spec is None
        assert env._output_spec is None

        env.insert_transform(-5, CatFrames(N=4, dim=-1, in_keys=[key]))
        assert isinstance(env.transform, Compose)
        assert len(env.transform) == 6

        assert isinstance(env.transform[0], CatFrames)
        assert isinstance(env.transform[1], NoopResetEnv)
        assert isinstance(env.transform[2], PinMemoryTransform)
        assert isinstance(env.transform[3], CatFrames)
        assert isinstance(env.transform[4], NoopResetEnv)
        assert isinstance(env.transform[5], FiniteTensorDictCheck)

        assert env._input_spec is None
        assert env._output_spec is None

        _ = copy(env.action_spec)
        _ = copy(env.observation_spec)
        _ = copy(env.reward_spec)

        with pytest.raises(ValueError):
            env.insert_transform(-7, FiniteTensorDictCheck())

        with pytest.raises(ValueError):
            env.insert_transform(7, FiniteTensorDictCheck())

        with pytest.raises(ValueError):
            env.insert_transform(4, "ffff")


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_locked_transformed(device):
    env = TransformedEnv(
        MockBatchedLockedEnv(device),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    env.step(td)

    with pytest.raises(
        RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
    ):
        env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked_transformed(device):
    env = TransformedEnv(
        MockBatchedUnLockedEnv(device),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    td_expanded = td.expand(2).clone()
    env.step(td)
    env.step(td_expanded)


@pytest.mark.parametrize("device", get_default_devices())
def test_batch_unlocked_with_batch_size_transformed(device):
    env = TransformedEnv(
        MockBatchedUnLockedEnv(device, batch_size=torch.Size([2])),
        Compose(
            ObservationNorm(in_keys=["observation"], loc=0.5, scale=1.1),
            RewardClipping(0, 0.1),
        ),
    )
    assert not env.batch_locked

    with pytest.raises(RuntimeError, match="batch_locked is a read-only property"):
        env.batch_locked = False
    td = env.reset()
    td["action"] = env.action_spec.rand()
    env.step(td)
    td_expanded = td.expand(2, 2).reshape(-1).to_tensordict()

    with pytest.raises(
        RuntimeError, match="Expected a tensordict with shape==env.batch_size, "
    ):
        env.step(td_expanded)


transforms = [
    ToTensorImage,
    pytest.param(
        partial(RewardClipping, clamp_min=0.1, clamp_max=0.9), id="RewardClipping"
    ),
    BinarizeReward,
    pytest.param(
        partial(Resize, w=2, h=2),
        id="Resize",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(CenterCrop, w=1),
        id="CenterCrop",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(FlattenObservation, first_dim=-3, last_dim=-3), id="FlattenObservation"
    ),
    pytest.param(partial(UnsqueezeTransform, dim=-1), id="UnsqueezeTransform"),
    pytest.param(partial(SqueezeTransform, dim=-1), id="SqueezeTransform"),
    GrayScale,
    pytest.param(
        partial(ObservationNorm, in_keys=["observation"]), id="ObservationNorm"
    ),
    pytest.param(partial(CatFrames, dim=-3, N=4), id="CatFrames"),
    pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
    FiniteTensorDictCheck,
    DoubleToFloat,
    CatTensors,
    pytest.param(
        partial(DiscreteActionProjection, max_actions=1, num_actions_effective=1),
        id="DiscreteActionProjection",
    ),
    NoopResetEnv,
    TensorDictPrimer,
    PinMemoryTransform,
    gSDENoise,
    VecNorm,
]


@pytest.mark.parametrize("transform", transforms)
def test_smoke_compose_transform(transform):
    Compose(transform())


@pytest.mark.parametrize("transform", transforms)
def test_clone_parent(transform):
    base_env1 = ContinuousActionVecMockEnv()
    base_env2 = ContinuousActionVecMockEnv()
    env = TransformedEnv(base_env1, transform())
    env_clone = TransformedEnv(base_env2, env.transform.clone())

    assert env_clone.transform.parent.base_env is not base_env1
    assert env_clone.transform.parent.base_env is base_env2
    assert env.transform.parent.base_env is not base_env2
    assert env.transform.parent.base_env is base_env1


@pytest.mark.parametrize("transform", transforms)
def test_clone_parent_compose(transform):
    base_env1 = ContinuousActionVecMockEnv()
    base_env2 = ContinuousActionVecMockEnv()
    env = TransformedEnv(base_env1, Compose(ToTensorImage(), transform()))
    t = env.transform.clone()

    assert t.parent is None
    assert t[0].parent is None
    assert t[1].parent is None

    env_clone = TransformedEnv(base_env2, Compose(ToTensorImage(), *t))

    assert env_clone.transform[0].parent.base_env is not base_env1
    assert env_clone.transform[0].parent.base_env is base_env2
    assert env.transform[0].parent.base_env is not base_env2
    assert env.transform[0].parent.base_env is base_env1
    assert env_clone.transform[1].parent.base_env is not base_env1
    assert env_clone.transform[1].parent.base_env is base_env2
    assert env.transform[1].parent.base_env is not base_env2
    assert env.transform[1].parent.base_env is base_env1


class TestCroSeq:
    def test_crop_dim1(self):
        tensordict = TensorDict(
            {
                "a": torch.arange(20).view(1, 1, 1, 20).expand(3, 4, 2, 20),
                "b": TensorDict(
                    {"c": torch.arange(20).view(1, 1, 1, 20, 1).expand(3, 4, 2, 20, 1)},
                    [3, 4, 2, 20, 1],
                ),
            },
            [3, 4, 2, 20],
        )
        t = RandomCropTensorDict(11, -1)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 2, 11])
        assert tensordict_crop["b"].shape == torch.Size([3, 4, 2, 11, 1])
        assert (
            tensordict_crop["a"][:, :, :, :-1] + 1 == tensordict_crop["a"][:, :, :, 1:]
        ).all()

    def test_crop_dim2(self):
        tensordict = TensorDict(
            {"a": torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2)},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(11, -2)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 11, 2])
        assert (
            tensordict_crop["a"][:, :, :-1] + 1 == tensordict_crop["a"][:, :, 1:]
        ).all()

    def test_crop_error(self):
        tensordict = TensorDict(
            {"a": torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2)},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(21, -2)
        with pytest.raises(RuntimeError, match="Cannot sample trajectories of length"):
            _ = t(tensordict)

    @pytest.mark.parametrize("mask_key", ("mask", ("collector", "mask")))
    def test_crop_mask(self, mask_key):
        a = torch.arange(20).view(1, 1, 20, 1).expand(3, 4, 20, 2).clone()
        mask = a < 21
        mask[0] = a[0] < 15
        mask[1] = a[1] < 16
        mask[1] = a[2] < 14
        tensordict = TensorDict(
            {"a": a, mask_key: mask},
            [3, 4, 20, 2],
        )
        t = RandomCropTensorDict(15, -2, mask_key=mask_key)
        with pytest.raises(RuntimeError, match="Cannot sample trajectories of length"):
            _ = t(tensordict)
        t = RandomCropTensorDict(13, -2, mask_key=mask_key)
        tensordict_crop = t(tensordict)
        assert tensordict_crop.shape == torch.Size([3, 4, 13, 2])
        assert tensordict_crop[mask_key].all()


class TestMultiStepTransform:
    def test_multistep_transform(self):
        env = TransformedEnv(
            SerialEnv(
                2, [lambda: CountingEnv(max_steps=4), lambda: CountingEnv(max_steps=10)]
            ),
            StepCounter(),
        )

        env.set_seed(0)
        torch.manual_seed(0)

        t = MultiStepTransform(3, 0.98)

        outs_2 = []
        td = env.reset().contiguous()
        assert "reward" not in td
        for _ in range(1):
            rollout = env.rollout(
                250, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            out = t._inv_call(rollout)
            td = rollout[..., -1]
            outs_2.append(out)
        # This will break if we don't have the appropriate number of frames
        outs_2 = torch.cat(outs_2, -1).split([47, 50, 50, 50, 50], -1)

        t = MultiStepTransform(3, 0.98)

        env.set_seed(0)
        torch.manual_seed(0)

        outs = []
        td = env.reset().contiguous()
        for i in range(5):
            rollout = env.rollout(
                50, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            out = t._inv_call(rollout)
            # tests that the data is insensitive to the collection schedule
            assert_allclose_td(out, outs_2[i])
            td = rollout[..., -1]["next"].exclude("reward")
            outs.append(out)

        outs = torch.cat(outs, -1)

        # Test with a very tiny window and across the whole collection
        t = MultiStepTransform(3, 0.98)

        env.set_seed(0)
        torch.manual_seed(0)

        outs_3 = []
        td = env.reset().contiguous()
        for _ in range(125):
            rollout = env.rollout(
                2, auto_reset=False, tensordict=td, break_when_any_done=False
            ).contiguous()
            assert rollout.shape[:-1] == env.batch_size
            assert "reward" not in rollout.keys()
            out = t._inv_call(rollout)
            td = rollout[..., -1]["next"].exclude("reward")
            if out is not None:
                outs_3.append(out)

        outs_3 = torch.cat(outs_3, -1)

        assert_allclose_td(outs, outs_3)

    def test_multistep_transform_changes(self):
        data = TensorDict(
            {
                "steps": torch.arange(100),
                "next": {
                    "steps": torch.arange(1, 101),
                    "reward": torch.ones(100, 1),
                    "done": torch.zeros(100, 1, dtype=torch.bool),
                    "terminated": torch.zeros(100, 1, dtype=torch.bool),
                    "truncated": torch.zeros(100, 1, dtype=torch.bool),
                },
            },
            batch_size=[100],
        )
        data_splits = data.split(10)
        t = MultiStepTransform(3, 0.98)
        rb = ReplayBuffer(storage=LazyTensorStorage(100), transform=t)
        for data in data_splits:
            rb.extend(data)
            t.n_steps = t.n_steps + 1
            assert (rb[:]["steps"] == torch.arange(len(rb))).all()
            assert rb[:]["next", "steps"][-1] == data["steps"][-1]
            assert t._buffer["steps"][-1] == data["steps"][-1]

    @pytest.mark.parametrize("add_or_extend", ["add", "extend"])
    def test_multisteptransform_single_item(self, add_or_extend):
        # Configuration
        buffer_size = 1000
        n_step = 3
        gamma = 0.99
        device = "cpu"

        rb = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=device, ndim=1),
            sampler=RandomSampler(),
            transform=MultiStepTransform(n_steps=n_step, gamma=gamma),
        )
        obs_dict = lambda i: {"observation": torch.full((4,), i)}  # 4-dim observation
        next_obs_dict = lambda i: {"observation": torch.full((4,), i)}

        for i in range(10):
            # Create transition with batch_size=[] (no batch dimension)
            transition = TensorDict(
                {
                    "obs": TensorDict(obs_dict(i), batch_size=[]),
                    "action": torch.full((2,), i),  # 2-dim action
                    "next": TensorDict(
                        {
                            "obs": TensorDict(next_obs_dict(i), batch_size=[]),
                            "done": torch.tensor(False, dtype=torch.bool),
                            "reward": torch.tensor(float(i), dtype=torch.float32),
                        },
                        batch_size=[],
                    ),
                },
                batch_size=[],
            )

            if add_or_extend == "add":
                rb.add(transition)
            else:
                rb.extend(transition.unsqueeze(0))
        rbcontent = rb[:]
        assert (rbcontent["steps_to_next_obs"] == 3).all()
        assert rbcontent.shape == (7,)
        assert (rbcontent["next", "original_reward"] == torch.arange(7)).all()
        assert (
            rbcontent["next", "reward"] > rbcontent["next", "original_reward"]
        ).all()


class TestBatchSizeTransform(TransformBase):
    class MyEnv(EnvBase):
        batch_locked = False

        def __init__(self):
            super().__init__()
            self.observation_spec = Composite(observation=Unbounded(3))
            self.reward_spec = Unbounded(1)
            self.action_spec = Unbounded(1)

        def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
            tensordict_batch_size = (
                tensordict.batch_size if tensordict is not None else torch.Size([])
            )
            result = self.observation_spec.rand(tensordict_batch_size)
            result.update(self.full_done_spec.zero(tensordict_batch_size))
            return result

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            result = self.observation_spec.rand(tensordict.batch_size)
            result.update(self.full_done_spec.zero(tensordict.batch_size))
            result.update(self.full_reward_spec.zero(tensordict.batch_size))
            return result

        def _set_seed(self, seed: int | None) -> None:
            ...

    @classmethod
    def reset_func(tensordict, tensordict_reset, env):
        result = env.observation_spec.rand()
        result.update(env.full_done_spec.zero())
        assert result.batch_size != torch.Size([])
        return result

    @pytest.mark.parametrize(
        "stateless,reshape_fn",
        [
            [False, "reshape"],
            [False, "unsqueeze"],
            [False, "unflatten"],
            [False, "squeeze"],
            [False, "flatten"],
            [True, None],
        ],
    )
    def test_single_trans_env_check(self, stateless, reshape_fn):
        if stateless:
            base_env = self.MyEnv()
            transform = BatchSizeTransform(batch_size=[10])
            expected_batch_size = torch.Size([10])
            assert transform.reshape_fn is None
        else:
            if reshape_fn == "reshape":
                base_env = CountingEnv(max_steps=3)
                reshape_fn = lambda x: x.reshape(1, 1)
                expected_batch_size = torch.Size([1, 1])
            elif reshape_fn == "unsqueeze":
                base_env = CountingEnv(max_steps=3)
                reshape_fn = lambda x: x.unsqueeze(0)
                expected_batch_size = torch.Size([1])
            elif reshape_fn == "unflatten":
                base_env = SerialEnv(1, lambda: CountingEnv(max_steps=3))
                reshape_fn = lambda x: x.unflatten(0, (1, 1))
                expected_batch_size = torch.Size([1, 1])
            elif reshape_fn == "squeeze":
                base_env = SerialEnv(1, lambda: CountingEnv(max_steps=3))
                reshape_fn = lambda x: x.squeeze(0)
                expected_batch_size = torch.Size([])
            elif reshape_fn == "flatten":
                base_env = SerialEnv(1, lambda: CountingEnv(max_steps=3))
                reshape_fn = lambda x: x.unflatten(0, (1, 1)).flatten(0, 1)
                expected_batch_size = torch.Size([1])
            else:
                raise NotImplementedError(reshape_fn)

            transform = BatchSizeTransform(reshape_fn=reshape_fn)
            assert transform.batch_size is None

        env = TransformedEnv(base_env, transform)
        assert env.batch_size == expected_batch_size
        check_env_specs(env)

    @pytest.mark.parametrize(
        "stateless,reshape_fn",
        [
            [False, "reshape"],
            [True, None],
        ],
    )
    def test_serial_trans_env_check(self, stateless, reshape_fn):
        def make_env(stateless=stateless, reshape_fn=reshape_fn):
            if stateless:
                base_env = self.MyEnv()
                transform = BatchSizeTransform(batch_size=[10])
                expected_batch_size = torch.Size([10])
                assert transform.reshape_fn is None
            else:
                if reshape_fn == "reshape":
                    base_env = CountingEnv(max_steps=3)
                    reshape_fn = lambda x: x.reshape(1, 1)
                    expected_batch_size = torch.Size([1, 1])
                else:
                    raise NotImplementedError(reshape_fn)

                transform = BatchSizeTransform(reshape_fn=reshape_fn)
                assert transform.batch_size is None

            env = TransformedEnv(base_env, transform)
            assert env.batch_size == expected_batch_size
            return env

        env = SerialEnv(2, make_env)
        assert env.batch_size == (2, *make_env().batch_size)
        check_env_specs(env)

    @pytest.mark.parametrize(
        "stateless,reshape_fn",
        [
            [False, "reshape"],
            [True, None],
        ],
    )
    def test_parallel_trans_env_check(self, stateless, reshape_fn):
        def make_env(stateless=stateless, reshape_fn=reshape_fn):
            if stateless:
                base_env = self.MyEnv()
                transform = BatchSizeTransform(batch_size=[10])
                expected_batch_size = torch.Size([10])
                assert transform.reshape_fn is None
            else:
                if reshape_fn == "reshape":
                    base_env = CountingEnv(max_steps=3)
                    reshape_fn = lambda x: x.reshape(1, 1)
                    expected_batch_size = torch.Size([1, 1])
                else:
                    raise NotImplementedError(reshape_fn)

                transform = BatchSizeTransform(reshape_fn=reshape_fn)
                assert transform.batch_size is None

            env = TransformedEnv(base_env, transform)
            assert env.batch_size == expected_batch_size
            return env

        env = ParallelEnv(2, make_env, mp_start_method=mp_ctx)
        assert env.batch_size == (2, *make_env().batch_size)
        check_env_specs(env)

    @pytest.mark.parametrize(
        "stateless,reshape_fn",
        [
            [False, "reshape"],
        ],
    )
    def test_trans_serial_env_check(self, stateless, reshape_fn):
        def make_env(stateless=stateless, reshape_fn=reshape_fn):
            if reshape_fn == "reshape":
                base_env = CountingEnv(max_steps=3)
            else:
                raise NotImplementedError(reshape_fn)
            return base_env

        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1, 2)
            expected_batch_size = torch.Size([1, 2])
        else:
            raise NotImplementedError(reshape_fn)

        transform = BatchSizeTransform(reshape_fn=reshape_fn)
        assert transform.batch_size is None

        env = TransformedEnv(SerialEnv(2, make_env), transform)
        assert env.batch_size == expected_batch_size
        check_env_specs(env)

    @pytest.mark.parametrize(
        "stateless,reshape_fn",
        [
            [False, "reshape"],
        ],
    )
    def test_trans_parallel_env_check(self, stateless, reshape_fn):
        def make_env(stateless=stateless, reshape_fn=reshape_fn):
            if reshape_fn == "reshape":
                base_env = CountingEnv(max_steps=3)
            else:
                raise NotImplementedError(reshape_fn)
            return base_env

        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1, 2)
            expected_batch_size = torch.Size([1, 2])
        else:
            raise NotImplementedError(reshape_fn)

        transform = BatchSizeTransform(reshape_fn=reshape_fn)
        assert transform.batch_size is None

        env = TransformedEnv(
            ParallelEnv(2, make_env, mp_start_method=mp_ctx), transform
        )
        assert env.batch_size == expected_batch_size
        check_env_specs(env)

    @pytest.mark.parametrize("stateless,reshape_fn", [[False, "reshape"]])
    def test_transform_no_env(self, stateless, reshape_fn):
        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1)
            expected_batch_size = torch.Size([1])
        else:
            raise NotImplementedError(reshape_fn)
        transform = BatchSizeTransform(reshape_fn=reshape_fn)
        base_env = CountingEnv(max_steps=3)
        assert transform._call(base_env.reset()).batch_size == expected_batch_size

    @pytest.mark.parametrize("stateless,reshape_fn", [[False, "reshape"]])
    def test_transform_compose(self, stateless, reshape_fn):
        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1)
            expected_batch_size = torch.Size([1])
        else:
            raise NotImplementedError(reshape_fn)
        transform = Compose(BatchSizeTransform(reshape_fn=reshape_fn))
        base_env = CountingEnv(max_steps=3)
        assert transform(base_env.reset()).batch_size == expected_batch_size

    @pytest.mark.parametrize("stateless,reshape_fn", [[False, "reshape"]])
    def test_transform_env(self, stateless, reshape_fn):
        # tested in single_env
        return

    @pytest.mark.parametrize("stateless,reshape_fn", [[False, "reshape"]])
    def test_transform_model(self, stateless, reshape_fn):
        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1)
            expected_batch_size = torch.Size([1])
        else:
            raise NotImplementedError(reshape_fn)
        transform = nn.Sequential(Compose(BatchSizeTransform(reshape_fn=reshape_fn)))
        base_env = CountingEnv(max_steps=3)
        assert transform(base_env.reset()).batch_size == expected_batch_size

    @pytest.mark.parametrize("stateless,reshape_fn", [[False, "reshape"]])
    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass, stateless, reshape_fn):
        if reshape_fn == "reshape":
            reshape_fn = lambda x: x.reshape(1, -1)
            expected_batch_size = torch.Size([1, 12])
        else:
            raise NotImplementedError(reshape_fn)
        rb = rbclass(storage=LazyTensorStorage(20))
        transform = Compose(BatchSizeTransform(reshape_fn=reshape_fn))
        rb.append_transform(transform)

        batch = (20, 3)
        td = TensorDict({"a": {"b": {"c": {}}}}, batch)

        rb.extend(td)
        if rbclass is TensorDictReplayBuffer:
            with pytest.raises(RuntimeError, match="Failed to set the metadata"):
                assert rb.sample(4).shape == expected_batch_size
        else:
            assert rb.sample(4).shape == expected_batch_size

    def test_transform_inverse(self):
        # Tested in single_env
        return


class TestTensorDictPrimer(TransformBase):
    def test_single_trans_env_check(self):
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TensorDictPrimer(mykey=Unbounded([3])),
        )
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        assert ("next", "mykey") in env.rollout(3).keys(True)

    def test_nested_key_env(self):
        env = MultiKeyCountingEnv()
        env_obs_spec_prior_primer = env.observation_spec.clone()
        env = TransformedEnv(
            env,
            TensorDictPrimer(
                Composite(
                    {
                        "nested_1": Composite(
                            {"mykey": Unbounded((env.nested_dim_1, 4))},
                            shape=(env.nested_dim_1,),
                        )
                    }
                ),
                reset_key="_reset",
            ),
        )
        check_env_specs(env)
        env_obs_spec_post_primer = env.observation_spec.clone()
        assert ("nested_1", "mykey") in env_obs_spec_post_primer.keys(True, True)
        del env_obs_spec_post_primer[("nested_1", "mykey")]
        assert env_obs_spec_post_primer == env_obs_spec_prior_primer

        assert ("nested_1", "mykey") in env.reset().keys(True, True)
        assert ("next", "nested_1", "mykey") in env.rollout(3).keys(True, True)

    def test_transform_no_env(self):
        t = TensorDictPrimer(mykey=Unbounded([3]))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "mykey" in td.keys()

    def test_transform_model(self):
        t = TensorDictPrimer(mykey=Unbounded([3]))
        model = nn.Sequential(t, nn.Identity())
        td = TensorDict()
        model(td)
        assert "mykey" in td.keys()

    @pytest.mark.parametrize("rbclass", [ReplayBuffer, TensorDictReplayBuffer])
    def test_transform_rb(self, rbclass):
        batch_size = (2,)
        t = TensorDictPrimer(mykey=Unbounded([*batch_size, 3]))
        rb = rbclass(storage=LazyTensorStorage(10))
        rb.append_transform(t)
        td = TensorDict({"a": torch.zeros(())}, [])
        rb.extend(td.expand(10))
        td = rb.sample(*batch_size)
        assert "mykey" in td.keys()

    def test_transform_inverse(self):
        raise pytest.skip("No inverse method for TensorDictPrimer")

    def test_transform_compose(self):
        t = Compose(TensorDictPrimer(mykey=Unbounded([3])))
        td = TensorDict({"a": torch.zeros(())}, [])
        t(td)
        assert "mykey" in td.keys()

    def test_parallel_trans_env_check(self, maybe_fork_ParallelEnv):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TensorDictPrimer(mykey=Unbounded([3])),
            )

        env = maybe_fork_ParallelEnv(2, make_env)
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            assert ("next", "mykey") in env.rollout(3).keys(True)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_serial_trans_env_check(self):
        def make_env():
            return TransformedEnv(
                ContinuousActionVecMockEnv(),
                TensorDictPrimer(mykey=Unbounded([3])),
            )

        env = SerialEnv(2, make_env)
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            assert ("next", "mykey") in env.rollout(3).keys(True)
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    def test_trans_parallel_env_check(self, maybe_fork_ParallelEnv):
        env = TransformedEnv(
            maybe_fork_ParallelEnv(2, ContinuousActionVecMockEnv),
            TensorDictPrimer(mykey=Unbounded([4]), expand_specs=True),
        )
        try:
            check_env_specs(env)
            assert "mykey" in env.reset().keys()
            r = env.rollout(3)
            assert ("next", "mykey") in r.keys(True)
            assert r["next", "mykey"].shape == torch.Size([2, 3, 4])
        finally:
            try:
                env.close()
            except RuntimeError:
                pass

    @pytest.mark.parametrize("spec_shape", [[4], [2, 4]])
    @pytest.mark.parametrize("expand_specs", [True, False, None])
    def test_trans_serial_env_check(self, spec_shape, expand_specs):
        if expand_specs is None:
            with pytest.raises(RuntimeError):
                env = TransformedEnv(
                    SerialEnv(2, ContinuousActionVecMockEnv),
                    TensorDictPrimer(
                        mykey=Unbounded(spec_shape), expand_specs=expand_specs
                    ),
                )
                env.observation_spec
            return
        elif expand_specs is True:
            shape = spec_shape[:-1]
            env = TransformedEnv(
                SerialEnv(2, ContinuousActionVecMockEnv),
                TensorDictPrimer(
                    Composite(mykey=Unbounded(spec_shape), shape=shape),
                    expand_specs=expand_specs,
                ),
            )
        else:
            # If we don't expand, we can't use [4]
            env = TransformedEnv(
                SerialEnv(2, ContinuousActionVecMockEnv),
                TensorDictPrimer(
                    mykey=Unbounded(spec_shape), expand_specs=expand_specs
                ),
            )
            if spec_shape == [4]:
                with pytest.raises(ValueError):
                    env.observation_spec
                return

        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        r = env.rollout(3)
        assert ("next", "mykey") in r.keys(True)
        assert r["next", "mykey"].shape == torch.Size([2, 3, 4])

    @pytest.mark.parametrize(
        "default_keys", [["action"], ["action", "monkeys jumping on the bed"]]
    )
    @pytest.mark.parametrize(
        "spec",
        [
            Composite(b=Bounded(-3, 3, [4])),
            Bounded(-3, 3, [4]),
        ],
    )
    @pytest.mark.parametrize("random", [True, False])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    @pytest.mark.parametrize("serial", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_transform_env(
        self,
        default_keys,
        spec,
        random,
        value,
        serial,
        device,
    ):
        if random and value != 0.0:
            return pytest.skip("no need to check random=True with more than one value")
        torch.manual_seed(0)
        num_defaults = len(default_keys)

        def make_env():
            env = ContinuousActionVecMockEnv()
            env.set_seed(100)
            kwargs = {
                key: spec.clone() if key != "action" else env.action_spec.clone()
                # copy to avoid having the same spec for all keys
                for key in default_keys
            }
            reset_transform = TensorDictPrimer(
                random=random, default_value=value, **kwargs
            )
            transformed_env = TransformedEnv(env, reset_transform).to(device)
            return transformed_env

        if serial:
            env = SerialEnv(2, make_env)
        else:
            env = make_env()

        tensordict = env.reset()
        tensordict_select = tensordict.select(
            *[key for key in tensordict.keys() if key in default_keys]
        )
        assert len(list(tensordict_select.keys())) == num_defaults
        if random:
            assert (tensordict_select != value).any()
        else:
            assert (tensordict_select == value).all()

        if isinstance(spec, Composite) and any(key != "action" for key in default_keys):
            for key in default_keys:
                if key in ("action",):
                    continue
                assert key in tensordict.keys()
                assert tensordict[key, "b"] is not None

    @pytest.mark.skipif(not _has_gym, reason="Test executed on gym")
    @pytest.mark.parametrize("batched_class", [ParallelEnv, SerialEnv])
    @pytest.mark.parametrize("break_when_any_done", [True, False])
    def test_tensordictprimer_batching(self, batched_class, break_when_any_done):
        from torchrl.testing import CARTPOLE_VERSIONED

        env = TransformedEnv(
            batched_class(2, lambda: GymEnv(CARTPOLE_VERSIONED())),
            TensorDictPrimer(Composite({"mykey": Unbounded([2, 4])}, shape=[2])),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r0 = env.rollout(100, break_when_any_done=break_when_any_done)

        env = batched_class(
            2,
            lambda: TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                TensorDictPrimer(mykey=Unbounded([4])),
            ),
        )
        torch.manual_seed(0)
        env.set_seed(0)
        r1 = env.rollout(100, break_when_any_done=break_when_any_done)
        tensordict.assert_close(r0, r1)

    def test_callable_default_value(self):
        def create_tensor():
            return torch.ones(3)

        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TensorDictPrimer(mykey=Unbounded([3]), default_value=create_tensor),
        )
        check_env_specs(env)
        assert "mykey" in env.reset().keys()
        assert ("next", "mykey") in env.rollout(3).keys(True)

    def test_dict_default_value(self):
        # Test with a dict of float default values
        key1_spec = Unbounded([3])
        key2_spec = Unbounded([3])
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TensorDictPrimer(
                mykey1=key1_spec,
                mykey2=key2_spec,
                default_value={
                    "mykey1": 1.0,
                    "mykey2": 2.0,
                },
            ),
        )
        check_env_specs(env)
        reset_td = env.reset()
        assert "mykey1" in reset_td.keys()
        assert "mykey2" in reset_td.keys()
        rollout_td = env.rollout(3)
        assert ("next", "mykey1") in rollout_td.keys(True)
        assert ("next", "mykey2") in rollout_td.keys(True)
        assert (rollout_td.get(("next", "mykey1")) == 1.0).all()
        assert (rollout_td.get(("next", "mykey2")) == 2.0).all()

        # Test with a dict of callable default values
        key1_spec = Unbounded([3])
        key2_spec = Categorical(3, dtype=torch.int64)
        env = TransformedEnv(
            ContinuousActionVecMockEnv(),
            TensorDictPrimer(
                mykey1=key1_spec,
                mykey2=key2_spec,
                default_value={
                    "mykey1": lambda: torch.ones(3),
                    "mykey2": lambda: torch.tensor(1, dtype=torch.int64),
                },
            ),
        )
        check_env_specs(env)
        reset_td = env.reset()
        assert "mykey1" in reset_td.keys()
        assert "mykey2" in reset_td.keys()
        rollout_td = env.rollout(3)
        assert ("next", "mykey1") in rollout_td.keys(True)
        assert ("next", "mykey2") in rollout_td.keys(True)
        assert (rollout_td.get(("next", "mykey1")) == torch.ones(3)).all
        assert (
            rollout_td.get(("next", "mykey2")) == torch.tensor(1, dtype=torch.int64)
        ).all

    @pytest.mark.skipif(not _has_gym, reason="GYM not found")
    def test_spec_shape_inplace_correction(self):
        hidden_size = input_size = num_layers = 2
        model = GRUModule(
            input_size, hidden_size, num_layers, in_key="observation", out_key="action"
        )
        env = TransformedEnv(
            SerialEnv(2, lambda: GymEnv(PENDULUM_VERSIONED())),
        )
        # These primers do not have the leading batch dimension
        # since model is agnostic to batch dimension that will be used.
        primers = get_primers_from_module(model)
        for primer in primers.primers:
            assert primers.primers.get(primer).shape == torch.Size(
                [num_layers, hidden_size]
            )
        env.append_transform(primers)
        # Reset should add the batch dimension to the primers
        # since the parent exists and is batch_locked.
        td = env.reset()
        for primer in primers.primers:
            assert primers.primers.get(primer).shape == torch.Size(
                [2, num_layers, hidden_size]
            )
            assert td.get(primer).shape == torch.Size([2, num_layers, hidden_size])
