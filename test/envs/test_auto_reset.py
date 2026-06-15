# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import gc
import signal
import time
from sys import platform

import pytest
import torch

from _envs_common import _has_gym, _has_transformers, mp_ctx
from packaging import version
from tensordict import (
    assert_allclose_td,
    LazyStackedTensorDict,
    set_capture_non_tensor_stack,
    TensorDict,
    TensorDictBase,
)

from torchrl import set_auto_unwrap_transformed_env
from torchrl.data import LazyStackStorage, ReplayBuffer, SliceSampler
from torchrl.data.tensor_specs import (
    Binary,
    Categorical,
    Composite,
    NonTensor,
    Unbounded,
)
from torchrl.envs import CatFrames, EnvBase, ParallelEnv, SerialEnv, TrajCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    Compose,
    InitTracker,
    RandomTruncationTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)
from torchrl.envs.transforms.transforms import (
    AutoResetEnv,
    AutoResetTransform,
    Tokenizer,
    Transform,
    UnsqueezeTransform,
)
from torchrl.envs.utils import check_env_specs
from torchrl.testing import CARTPOLE_VERSIONED
from torchrl.testing.mocking_classes import (
    AutoResetHeteroCountingEnv,
    AutoResettingCountingEnv,
    CountingEnv,
    EnvWithDynamicSpec,
    EnvWithMetadata,
    EnvWithTensorClass,
    Str2StrEnv,
)

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)


class TestAutoReset:
    @staticmethod
    def _step_native_auto_reset_until_done(env):
        tensordict = env.reset()
        for _ in range(10):
            tensordict.update(env.full_action_spec.one())
            tensordict, tensordict_ = env.step_and_maybe_reset(
                tensordict,
            )
            if tensordict["next", "done"].all():
                return tensordict, tensordict_
            tensordict = tensordict_
        raise RuntimeError("Auto-resetting counting env did not terminate.")

    def test_native_auto_reset_resets_transform_state(self):
        class RewardingAutoResettingCountingEnv(AutoResettingCountingEnv):
            def _step(self, tensordict):
                tensordict = super()._step(tensordict)
                tensordict["reward"] = torch.ones_like(tensordict["reward"])
                return tensordict

        env = TransformedEnv(
            RewardingAutoResettingCountingEnv(3),
            Compose(StepCounter(), RewardSum(), TrajCounter()),
        )
        env._torchrl_native_autoreset = True
        env.full_observation_spec

        tensordict, tensordict_ = self._step_native_auto_reset_until_done(env)

        assert tensordict["next", "done"].all()
        assert not tensordict_["done"].any()
        assert (tensordict_["observation"] == 0).all()
        assert (tensordict_["step_count"] == 0).all()
        assert (tensordict_["episode_reward"] == 0).all()
        assert (tensordict_["traj_count"] == 1).all()

    def test_native_auto_reset_reset_mask_reaches_all_transforms(self):
        class ResetMaskRecorder(Transform):
            def __init__(self, key):
                super().__init__()
                self.key = key

            def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
                tensordict_reset.set(self.key, tensordict.get("_reset"))
                return tensordict_reset

        env = TransformedEnv(
            AutoResettingCountingEnv(3),
            Compose(
                ResetMaskRecorder("reset_before"),
                StepCounter(),
                ResetMaskRecorder("reset_after"),
            ),
        )
        env._torchrl_native_autoreset = True
        env.full_observation_spec

        tensordict, tensordict_ = self._step_native_auto_reset_until_done(env)

        assert tensordict["next", "done"].all()
        assert not tensordict_["done"].any()
        assert tensordict_["reset_before"].all()
        assert tensordict_["reset_after"].all()
        assert (tensordict_["step_count"] == 0).all()

    def test_native_auto_reset_resets_catframes_state(self):
        class FloatAutoResettingCountingEnv(AutoResettingCountingEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.observation_spec = Composite(
                    observation=Unbounded(
                        (*self.batch_size, 1),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=self.batch_size,
                    device=self.device,
                )

            def _reset(self, tensordict=None):
                tensordict = super()._reset(tensordict)
                tensordict["observation"] = tensordict["observation"].to(torch.float32)
                return tensordict

            def _step(self, tensordict):
                tensordict = super()._step(tensordict)
                tensordict["observation"] = tensordict["observation"].to(torch.float32)
                return tensordict

        env = TransformedEnv(
            FloatAutoResettingCountingEnv(3),
            CatFrames(N=2, dim=-1, in_keys=["observation"]),
        )
        env._torchrl_native_autoreset = True
        env.full_observation_spec

        tensordict, tensordict_ = self._step_native_auto_reset_until_done(env)

        assert tensordict["next", "done"].all()
        assert not tensordict_["done"].any()
        assert (tensordict_["observation"] == 0).all()

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"),
        reason="Dynamo cannot trace `EnvBase.device` (a class-level property "
        "using `self.__dict__.get`) on PyTorch < 2.5.",
    )
    def test_native_auto_reset_compile_step_and_maybe_reset(self):
        class BranchlessAutoResetCountingEnv(EnvBase):
            def __init__(self, max_steps=3, **kwargs):
                super().__init__(**kwargs)
                self.max_steps = max_steps
                self.observation_spec = Composite(
                    observation=Unbounded((1,), dtype=torch.float32)
                )
                self.reward_spec = Unbounded((1,))
                self.done_spec = Categorical(2, dtype=torch.bool, shape=(1,))
                self.action_spec = Binary(n=1, shape=(1,))
                self.register_buffer("count", torch.zeros(1, dtype=torch.float32))

            def _reset(self, tensordict=None, **kwargs):
                self.count.zero_()
                done = self.count.bool()
                return TensorDict(
                    {
                        "observation": self.count.clone(),
                        "done": done,
                        "terminated": done,
                    },
                    [],
                )

            def _step(self, tensordict):
                next_count = self.count + tensordict["action"].to(torch.float32)
                done = next_count > self.max_steps
                count = torch.where(done, torch.zeros_like(next_count), next_count)
                self.count.copy_(count)
                return TensorDict(
                    {
                        "observation": count.clone(),
                        "done": done,
                        "terminated": done,
                        "reward": torch.ones_like(count),
                    },
                    [],
                )

            def _set_seed(self, seed):
                return None

        env = TransformedEnv(
            BranchlessAutoResetCountingEnv(3),
            Compose(
                StepCounter(),
                RandomTruncationTransform(prob=1.0, min_horizon=1, max_horizon=3),
                RewardSum(),
                VecNormV2(in_keys=["observation"]),
            ),
        )
        env._torchrl_native_autoreset = True
        env.full_observation_spec
        tensordict = env.reset()

        env.compile(backend="eager", fullgraph=True)
        for _ in range(6):
            tensordict.update(env.full_action_spec.one())
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict)
            tensordict = tensordict_
        env.eager()

        assert tensordict["step_count"].le(3).all()
        assert tensordict["episode_reward"].le(3).all()
        assert "_compiled_step_and_maybe_reset" not in env.__dict__

    def test_env_constructor_compile_kwarg(self):
        env = CountingEnv(3, compile=False)
        assert "_compiled_step_and_maybe_reset" not in env.__dict__

        env = CountingEnv(3, compile=True)
        assert "_compiled_step_and_maybe_reset" in env.__dict__

        env = CountingEnv(
            3,
            compile={"warmup": 2, "backend": "eager", "fullgraph": True},
        )
        assert "_compiled_step_and_maybe_reset" in env.__dict__

        env_t = TransformedEnv(
            CountingEnv(3),
            StepCounter(),
            compile={"warmup": 1, "backend": "eager"},
        )
        assert "_compiled_step_and_maybe_reset" in env_t.__dict__

        with pytest.raises(TypeError, match="compile must be"):
            CountingEnv(3, compile="please")

    def test_transformed_env_getattr_no_recursion_on_module_internals(self):
        """TransformedEnv.__getattr__ short-circuits nn.Module internal slots.

        ``nn.Module.__dir__`` reads ``_parameters``, ``_buffers``, ``_modules``
        via attribute access; routing those through the base-env fallback
        used to re-enter ``__getattr__`` and infinite-recurse under
        ``torch.compile(step_and_maybe_reset)``. The fallback now refuses
        the explicit set of nn.Module instance slots (and all dunders),
        while still letting other single-underscore attributes (e.g. test
        envs' ``_counter`` / wrapper handles' ``_env``) flow through to
        ``base_env`` as before.
        """
        env = TransformedEnv(CountingEnv(3), StepCounter())
        for name in (
            "_parameters",
            "_buffers",
            "_modules",
            "_backward_hooks",
            "_forward_hooks",
            "_non_persistent_buffers_set",
            "__weakref__",
        ):
            with pytest.raises(AttributeError):
                env.__getattr__(name)
        assert "base_env" in dir(env)

    def test_transformed_env_getattr_delegates_user_private_attrs(self):
        """Non-internal single-underscore attrs still delegate to base_env.

        Wrappers like ``GymEnv`` store the underlying gym handle as
        ``_env``; ``CountingEnv``-style test fixtures store ``_counter``.
        Both are public-by-convention private fields of the wrapped env
        and the long-standing convention is that the
        :class:`TransformedEnv` wrapper transparently forwards them.
        """

        class _UnderscoreAttrEnv(CountingEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._sentinel = "marker"

        base = _UnderscoreAttrEnv(3)
        env = TransformedEnv(base, StepCounter())
        # Delegation goes through __getattr__'s base_env fallback.
        assert env._sentinel == "marker"
        # And ``base_env`` is still accessible by name (regression for the
        # ``"base_env" in self.__dir__()`` removal).
        assert env.base_env is base

    def test_native_auto_reset_wrapped_vecnorm_step_and_maybe_reset(self):
        class CountingVecNormV2(VecNormV2):
            def __init__(self, *args, **kwargs):
                self.step_calls = 0
                self.reset_calls = 0
                super().__init__(*args, **kwargs)

            def _step(self, tensordict, next_tensordict):
                self.step_calls += 1
                return super()._step(tensordict, next_tensordict)

            def _reset(self, tensordict, tensordict_reset):
                self.reset_calls += 1
                return super()._reset(tensordict, tensordict_reset)

        env = TransformedEnv(AutoResettingCountingEnv(3), InitTracker())
        env._torchrl_native_autoreset = True
        vecnorm = CountingVecNormV2(in_keys=["observation"])
        env = TransformedEnv(env, vecnorm)
        env.full_observation_spec
        tensordict = env.reset()

        initial_step_calls = vecnorm.step_calls
        initial_reset_calls = vecnorm.reset_calls
        assert vecnorm.reset_calls == 1

        for i in range(4):
            tensordict.update(env.full_action_spec.zero().apply(lambda x: x + 1))
            tensordict, tensordict_ = env.step_and_maybe_reset(
                tensordict,
            )
            assert vecnorm.step_calls == initial_step_calls + i + 1
            assert vecnorm.reset_calls == initial_reset_calls

        assert tensordict["next", "done"].all()
        assert not tensordict_["done"].any()
        assert tensordict_["is_init"].all()

    def test_native_auto_reset_step_and_maybe_reset(self):
        env = TransformedEnv(AutoResettingCountingEnv(3), InitTracker())
        env._torchrl_native_autoreset = True
        tensordict = env.reset()

        for _ in range(4):
            tensordict.update(env.full_action_spec.zero().apply(lambda x: x + 1))
            tensordict, tensordict_ = env.step_and_maybe_reset(
                tensordict,
            )

        assert tensordict["next", "done"].all()
        assert (tensordict["next", "observation"] == 0).all()
        assert not tensordict_["done"].any()
        assert tensordict_["is_init"].all()

    def test_auto_reset(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )

        env = AutoResettingCountingEnv(4, auto_reset=True)
        assert isinstance(env, TransformedEnv) and isinstance(
            env.transform, AutoResetTransform
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([20])
        assert r["next", "done"].sum() == 4
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all(), r[
            "next", "observation"
        ][r["next", "done"].squeeze()]
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 1
        assert not r["done"].any()

    def test_auto_reset_transform(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = TransformedEnv(
            AutoResettingCountingEnv(4, auto_reset=True), StepCounter()
        )
        assert isinstance(env, TransformedEnv) and isinstance(
            env.base_env.transform, AutoResetTransform
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([20])
        assert r["next", "done"].sum() == 4
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 1
        assert not r["done"].any()

    def test_auto_reset_serial(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = SerialEnv(
            2, functools.partial(AutoResettingCountingEnv, 4, auto_reset=True)
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert r["next", "done"].sum() == 8
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 2
        assert not r["done"].any()

    def test_auto_reset_serial_hetero(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = SerialEnv(
            2,
            [
                functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
                functools.partial(AutoResettingCountingEnv, 5, auto_reset=True),
            ],
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        assert not r["done"].any()

    def test_auto_reset_parallel(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = ParallelEnv(
            2,
            functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
            mp_start_method=mp_ctx,
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert r["next", "done"].sum() == 8
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        r = env.rollout(20, policy, break_when_any_done=True)
        assert r["next", "done"].sum() == 2
        assert not r["done"].any()

    def test_auto_reset_parallel_hetero(self):
        policy = lambda td: td.set(
            "action", torch.ones((*td.shape, 1), dtype=torch.int64)
        )
        env = ParallelEnv(
            2,
            [
                functools.partial(AutoResettingCountingEnv, 4, auto_reset=True),
                functools.partial(AutoResettingCountingEnv, 5, auto_reset=True),
            ],
            mp_start_method=mp_ctx,
        )
        r = env.rollout(20, policy, break_when_any_done=False)
        assert r.shape == torch.Size([2, 20])
        assert (r["next", "observation"][r["next", "done"].squeeze()] == -1).all()
        assert (
            r[..., 1:]["observation"][r[..., :-1]["next", "done"].squeeze()] == 0
        ).all()
        assert not r["done"].any()

    def test_auto_reset_heterogeneous_env(self):
        torch.manual_seed(0)
        env = TransformedEnv(
            AutoResetHeteroCountingEnv(4, auto_reset=True), StepCounter()
        )

        def policy(td):
            return td.update(
                env.full_action_spec.zero().apply(lambda x: x.bernoulli_(0.5))
            )

        assert isinstance(env.base_env, AutoResetEnv) and isinstance(
            env.base_env.transform, AutoResetTransform
        )
        check_env_specs(env)
        r = env.rollout(40, policy, break_when_any_done=False)
        assert (r["next", "lazy", "step_count"] - 1 == r["lazy", "step_count"]).all()
        done = r["next", "lazy", "done"].squeeze(-1)[:-1]
        assert (
            r["next", "lazy", "step_count"][1:][~done]
            == r["next", "lazy", "step_count"][:-1][~done] + 1
        ).all()
        assert (
            r["next", "lazy", "step_count"][1:][done]
            != r["next", "lazy", "step_count"][:-1][done] + 1
        ).all()
        done_split = r["next", "lazy", "done"].unbind(1)
        lazy_slit = r["next", "lazy"].unbind(1)
        lazy_roots = r["lazy"].unbind(1)
        for lazy, lazy_root, done in zip(lazy_slit, lazy_roots, done_split):
            assert lazy["lidar"][done.squeeze()].isnan().all()
            assert not lazy["lidar"][~done.squeeze()].isnan().any()
            assert (lazy_root["lidar"][1:][done[:-1].squeeze()] == 0).all()


class TestEnvWithDynamicSpec:
    def test_dynamic_rollout(self):
        env = EnvWithDynamicSpec()
        rollout = env.rollout(4)
        assert isinstance(rollout, LazyStackedTensorDict)
        rollout = env.rollout(4, return_contiguous=False)
        assert isinstance(rollout, LazyStackedTensorDict)
        with pytest.raises(
            RuntimeError,
            match="The environment specs are dynamic. Call rollout with return_contiguous=False",
        ):
            env.rollout(4, return_contiguous=True)
        env.rollout(4)
        env.rollout(4, return_contiguous=False)
        check_env_specs(env, return_contiguous=False)

    @pytest.mark.skipif(not _has_gym, reason="requires gym to be installed")
    @pytest.mark.parametrize("penv", [SerialEnv, ParallelEnv])
    def test_batched_nondynamic(self, penv):
        # Tests not using buffers in batched envs
        env_buffers = penv(
            3,
            lambda: GymEnv(CARTPOLE_VERSIONED(), device=None),
            use_buffers=True,
            mp_start_method=mp_ctx if penv is ParallelEnv else None,
        )
        try:
            env_buffers.set_seed(0)
            torch.manual_seed(0)
            rollout_buffers = env_buffers.rollout(
                20, return_contiguous=True, break_when_any_done=False
            )
        finally:
            env_buffers.close(raise_if_closed=False)
            del env_buffers
        gc.collect()
        # Add a small delay to allow multiprocessing resource_sharer threads
        # to fully clean up before creating the next environment. This prevents
        # a race condition where the old resource_sharer service thread is still
        # active when the new environment starts, causing a deadlock.
        # See: https://bugs.python.org/issue30289
        if penv is ParallelEnv:
            time.sleep(0.1)

        env_no_buffers = penv(
            3,
            lambda: GymEnv(CARTPOLE_VERSIONED(), device=None),
            use_buffers=False,
            mp_start_method=mp_ctx if penv is ParallelEnv else None,
        )
        try:
            env_no_buffers.set_seed(0)
            torch.manual_seed(0)
            rollout_no_buffers = env_no_buffers.rollout(
                20, return_contiguous=True, break_when_any_done=False
            )
        finally:
            env_no_buffers.close(raise_if_closed=False)
            del env_no_buffers
        gc.collect()
        assert_allclose_td(rollout_buffers, rollout_no_buffers)

    @pytest.mark.parametrize("break_when_any_done", [False, True])
    def test_batched_dynamic(self, break_when_any_done):
        list_of_envs = [EnvWithDynamicSpec(i + 4) for i in range(3)]
        dummy_rollouts = [
            env.rollout(
                20, return_contiguous=False, break_when_any_done=break_when_any_done
            )
            for env in list_of_envs
        ]
        t = min(dr.shape[0] for dr in dummy_rollouts)
        dummy_rollouts = TensorDict.maybe_dense_stack([dr[:t] for dr in dummy_rollouts])
        del list_of_envs

        # Tests not using buffers in batched envs
        env_no_buffers = SerialEnv(
            3,
            [lambda i=i + 4: EnvWithDynamicSpec(i) for i in range(3)],
            use_buffers=False,
        )
        env_no_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_no_buffers_serial = env_no_buffers.rollout(
            20, return_contiguous=False, break_when_any_done=break_when_any_done
        )
        del env_no_buffers
        gc.collect()
        assert_allclose_td(
            dummy_rollouts.exclude("action"),
            rollout_no_buffers_serial.exclude("action"),
        )

        env_no_buffers = ParallelEnv(
            3,
            [lambda i=i + 4: EnvWithDynamicSpec(i) for i in range(3)],
            use_buffers=False,
            mp_start_method=mp_ctx,
        )
        env_no_buffers.set_seed(0)
        torch.manual_seed(0)
        rollout_no_buffers_parallel = env_no_buffers.rollout(
            20, return_contiguous=False, break_when_any_done=break_when_any_done
        )
        del env_no_buffers
        gc.collect()

        assert_allclose_td(
            dummy_rollouts.exclude("action"),
            rollout_no_buffers_parallel.exclude("action"),
        )
        assert_allclose_td(rollout_no_buffers_serial, rollout_no_buffers_parallel)


class TestNonTensorEnv:
    @pytest.fixture(scope="class", autouse=True)
    def set_capture(self):
        with set_capture_non_tensor_stack(False):
            yield None
        return

    @pytest.mark.parametrize("bwad", [True, False])
    def test_single(self, bwad):
        env = EnvWithMetadata()
        r = env.rollout(10, break_when_any_done=bwad)
        assert r.get("non_tensor").tolist() == list(range(10))

    @pytest.mark.parametrize("bwad", [True, False])
    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_serial(self, bwad, use_buffers):
        N = 50
        env = SerialEnv(2, EnvWithMetadata, use_buffers=use_buffers)
        r = env.rollout(N, break_when_any_done=bwad)
        assert r.get("non_tensor").tolist() == [list(range(N))] * 2

    # @pytest.mark.forked  # Run in isolated subprocess to avoid resource_sharer pollution from other tests
    @pytest.mark.parametrize("bwad", [True, False])
    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_parallel(self, bwad, use_buffers, maybe_fork_ParallelEnv):
        N = 50
        env = maybe_fork_ParallelEnv(2, EnvWithMetadata, use_buffers=use_buffers)
        try:
            r = env.rollout(N, break_when_any_done=bwad)
            assert r.get("non_tensor").tolist() == [list(range(N))] * 2
        finally:
            env.close(raise_if_closed=False)
            del env
            time.sleep(0.1)
            gc.collect()

    @pytest.mark.parametrize("use_buffers", [False, True])
    def test_parallel_partial_reset(self, use_buffers, maybe_fork_ParallelEnv):
        """Regression: a *partial* reset (subset of workers) with NonTensor data.

        ``out`` carries no NonTensor leaf (NonTensor is not shared-memory
        backed), so a partial fancy-index assignment used to raise
        ``IndexError: list index out of range``. The reset workers must get the
        fresh value while the untouched worker keeps its current value.
        """
        env = maybe_fork_ParallelEnv(3, EnvWithMetadata, use_buffers=use_buffers)
        try:
            env.set_seed(0)
            td = env.reset()
            for _ in range(4):
                td.set("action", torch.zeros(3, 1))
                td = env.step_mdp(env.step(td))
            assert td.get("non_tensor").tolist() == [4, 4, 4]
            # partial reset: workers 0 and 2 only; worker 1 keeps its state
            reset = td.select("non_tensor")
            reset.set("_reset", torch.tensor([True, False, True]).reshape(3, 1))
            out = env.reset(reset)
            assert out.get("non_tensor").tolist() == [0, 4, 0]
        finally:
            env.close(raise_if_closed=False)
            del env
            time.sleep(0.1)
            gc.collect()

    @pytest.mark.skipif(
        platform == "win32", reason="signal-based timeout not supported."
    )
    def test_parallel_large_non_tensor_does_not_deadlock(self, maybe_fork_ParallelEnv):
        """Regression test: large non-tensor payloads must not deadlock ParallelEnv in buffer mode.

        In shared-buffer mode, non-tensor leaves are sent over the Pipe. If the worker
        blocks on `send()` (pipe buffer full) before setting its completion event,
        the parent can hang forever waiting for that event. We guard against this by
        using a signal alarm and a very large non-tensor payload.
        """

        class _LargeNonTensorEnv(EnvWithMetadata):
            def __init__(self, payload_size: int = 5_000_000):
                super().__init__()
                self._payload = b"x" * payload_size

            def _reset(self, tensordict):
                data = self._saved_obs_spec.zero()
                data.set_non_tensor("non_tensor", self._payload)
                data.update(self.full_done_spec.zero())
                return data

            def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
                data = self._saved_obs_spec.zero()
                data.set_non_tensor("non_tensor", self._payload)
                data.update(self.full_done_spec.zero())
                data.update(self._saved_full_reward_spec.zero())
                return data

        def _alarm_handler(signum, frame):
            raise TimeoutError(
                "ParallelEnv deadlocked while waiting for workers with large non-tensor payloads."
            )

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(15)
        env = maybe_fork_ParallelEnv(2, _LargeNonTensorEnv, use_buffers=True)
        try:
            td = env.reset()
            td = td.set("action", torch.zeros(2, 1))
            _ = env.step(td)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            env.close(raise_if_closed=False)
            del env
            time.sleep(0.1)
            gc.collect()

    class AddString(Transform):
        def __init__(self):
            super().__init__()
            self._str = "0"

        def _call(self, td):
            td["string"] = str(int(self._str) + 1)
            self._str = td["string"]
            return td

        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            self._str = "0"
            tensordict_reset["string"] = self._str
            return tensordict_reset

        def transform_observation_spec(self, observation_spec):
            observation_spec["string"] = NonTensor(())
            return observation_spec

    @pytest.mark.parametrize("batched", ["serial", "parallel"])
    def test_partial_reset(self, batched):
        with set_capture_non_tensor_stack(False):
            env0 = lambda: CountingEnv(5).append_transform(self.AddString())
            env1 = lambda: CountingEnv(6).append_transform(self.AddString())
            if batched == "parallel":
                env = ParallelEnv(2, [env0, env1], mp_start_method=mp_ctx)
            else:
                env = SerialEnv(2, [env0, env1])
            try:
                s = env.reset()
                i = 0
                for i in range(10):  # noqa: B007
                    s, s_ = env.step_and_maybe_reset(
                        s.set("action", torch.ones(2, 1, dtype=torch.int))
                    )
                    if s.get(("next", "done")).any():
                        break
                    s = s_
                assert i == 5
                assert (s["next", "done"] == torch.tensor([[True], [False]])).all()
                assert s_["string"] == ["0", "6"]
                assert s["next", "string"] == ["6", "6"]
            finally:
                env.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_transformers, reason="transformers required")
    def test_from_text_env_tokenizer(self):
        env = Str2StrEnv()
        env.set_seed(0)
        env = env.append_transform(
            Tokenizer(
                in_keys=["observation"],
                out_keys=["obs_tokens"],
                in_keys_inv=["action"],
                out_keys_inv=["action_tokens"],
            )
        )
        env.check_env_specs()
        assert env._has_dynamic_specs
        r = env.rollout(3, return_contiguous=False)
        assert len(r) == 3
        assert isinstance(r["observation"], list)
        r = r.densify(layout=torch.jagged)
        assert isinstance(r["observation"], list)
        assert isinstance(r["obs_tokens"], torch.Tensor)
        assert isinstance(r["action_tokens"], torch.Tensor)

    @pytest.mark.skipif(not _has_transformers, reason="transformers required")
    @set_auto_unwrap_transformed_env(False)
    def test_from_text_env_tokenizer_catframes(self):
        """Tests that we can use Unsqueeze + CatFrames with tokenized strings of variable lengths."""
        env = Str2StrEnv()
        env.set_seed(0)
        env = env.append_transform(
            Tokenizer(
                in_keys=["observation"],
                out_keys=["obs_tokens"],
                in_keys_inv=["action"],
                out_keys_inv=["action_tokens"],
                # We must use max_length otherwise we can't call cat
                # Perhaps we could use NJT here?
                max_length=10,
            )
        )
        env = env.append_transform(
            UnsqueezeTransform(
                dim=-2, in_keys=["obs_tokens"], out_keys=["obs_tokens_cat"]
            ),
        )
        env = env.append_transform(CatFrames(N=4, dim=-2, in_keys=["obs_tokens_cat"]))
        r = env.rollout(3)
        assert r["obs_tokens_cat"].shape == (3, 4, 10)

    @pytest.mark.skipif(not _has_transformers, reason="transformers required")
    def test_from_text_rb_slicesampler(self):
        """Dedicated test for replay buffer sampling of trajectories with variable token length"""
        env = Str2StrEnv()
        env.set_seed(0)
        env = env.append_transform(
            Tokenizer(
                in_keys=["observation"],
                out_keys=["obs_tokens"],
                in_keys_inv=["action"],
                out_keys_inv=["action_tokens"],
            )
        )
        env = env.append_transform(StepCounter(max_steps=10))
        env = env.append_transform(TrajCounter())
        rb = ReplayBuffer(
            storage=LazyStackStorage(100),
            sampler=SliceSampler(slice_len=10, end_key=("next", "done")),
        )
        r0 = env.rollout(20, break_when_any_done=False)
        rb.extend(r0)
        has_0 = False
        has_1 = False
        for _ in range(100):
            v0 = rb.sample(10)
            assert (v0["step_count"].squeeze() == torch.arange(10)).all()
            assert (v0["next", "step_count"].squeeze() == torch.arange(1, 11)).all()
            try:
                traj = v0["traj_count"].unique().item()
            except Exception:
                raise RuntimeError(
                    f"More than one traj found in single slice: {v0['traj_count']}"
                )
            has_0 |= traj == 0
            has_1 |= traj == 1
            if has_0 and has_1:
                break
        else:
            raise RuntimeError("Failed to sample both trajs")

    def test_env_with_tensorclass(self):
        env = EnvWithTensorClass()
        env.check_env_specs()
        r = env.reset()
        for _ in range(3):
            assert isinstance(r["tc"], env.tc_cls)
            a = env.rand_action(r)
            s = env.step(a)
            assert isinstance(s["tc"], env.tc_cls)
            r = env.step_mdp(s)

    @pytest.mark.parametrize("cls", [SerialEnv, ParallelEnv])
    def test_env_with_tensorclass_batched(self, cls):
        env = cls(2, EnvWithTensorClass)
        env.check_env_specs()
        r = env.reset()
        for _ in range(3):
            assert isinstance(r["tc"], EnvWithTensorClass.tc_cls)
            a = env.rand_action(r)
            s = env.step(a)
            assert isinstance(s["tc"], EnvWithTensorClass.tc_cls)
            r = env.step_mdp(s)
