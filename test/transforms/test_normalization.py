# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pickle

import sys

import pytest

import torch

from _transforms_common import TIMEOUT, TORCH_VERSION
from packaging import version
from tensordict import assert_close, TensorDict, TensorDictBase
from tensordict.utils import assert_allclose_td
from torch import multiprocessing as mp

from torchrl.data import Composite, Unbounded
from torchrl.envs import (
    Compose,
    EnvBase,
    EnvCreator,
    ParallelEnv,
    RenameTransform,
    SerialEnv,
    TransformedEnv,
    VecNormV2,
)
from torchrl.envs.libs.gym import _has_gym, GymEnv
from torchrl.envs.transforms import VecNorm
from torchrl.envs.utils import check_env_specs, step_mdp

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
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv, CountingEnv


class TestVecNormV2:
    SEED = -1

    class SimpleEnv(EnvBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.full_reward_spec = Composite(reward=Unbounded((1,)))
            self.full_observation_spec = Composite(observation=Unbounded(()))
            self.full_action_spec = Composite(action=Unbounded(()))

        def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
            tensordict = (
                TensorDict()
                .update(self.full_observation_spec.rand())
                .update(self.full_done_spec.zero())
            )
            return tensordict

        def _step(
            self,
            tensordict: TensorDictBase,
        ) -> TensorDictBase:
            tensordict = (
                TensorDict()
                .update(self.full_observation_spec.rand())
                .update(self.full_done_spec.zero())
            )
            tensordict["reward"] = self.reward_spec.rand()
            return tensordict

        def _set_seed(self, seed: int | None) -> None:
            ...

    @pytest.mark.parametrize("batched", [False, True])
    def test_vecnorm2_decay1(self, batched):
        env = self.SimpleEnv()
        if batched:
            env = SerialEnv(2, [lambda env=env: env] * 2)
        env = env.append_transform(
            VecNormV2(
                in_keys=["reward", "observation"],
                out_keys=["reward_norm", "obs_norm"],
                decay=1,
                reduce_batch_dims=True,
            )
        )
        s_ = env.reset()
        ss = []
        N = 20
        for i in range(N):
            s, s_ = env.step_and_maybe_reset(env.rand_action(s_))
            ss.append(s)
            sstack = torch.stack(ss)
            if i >= 2:
                for k in ("reward",):
                    loc = sstack[: i + 1]["next", k].mean().unsqueeze(-1)
                    scale = (
                        sstack[: i + 1]["next", k]
                        .std(unbiased=False)
                        .clamp_min(1e-6)
                        .unsqueeze(-1)
                    )
                    # Assert that loc and scale match the expected values
                    torch.testing.assert_close(
                        loc,
                        env.transform.loc[k],
                    )
                    torch.testing.assert_close(
                        scale,
                        env.transform.scale[k],
                    )
        if batched:
            assert env.transform._loc.ndim == 0
            assert env.transform._var.ndim == 0

    @pytest.mark.skipif(not _has_gym, reason="gym not available")
    @pytest.mark.parametrize("stateful", [True, False])
    def test_stateful_and_stateless_specs(self, stateful):
        torch.manual_seed(0)
        env = GymEnv(PENDULUM_VERSIONED())
        env.set_seed(0)
        env = env.append_transform(
            VecNorm(
                in_keys=["observation"],
                out_keys=["obs_norm"],
                stateful=stateful,
                new_api=True,
            )
        )
        # check that transform output spec runs
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.check_env_specs()

    @pytest.mark.skipif(not _has_gym, reason="gym not available")
    def test_stateful_vs_stateless(self):
        vals = []
        locs = []
        vars = []
        counts = []
        for stateful in [True, False]:
            torch.manual_seed(0)
            env = GymEnv(PENDULUM_VERSIONED())
            env.set_seed(0)
            env = env.append_transform(
                VecNorm(
                    in_keys=["observation"],
                    out_keys=["obs_norm"],
                    stateful=stateful,
                    new_api=True,
                )
            )
            # check that transform output spec runs
            env.transform.transform_output_spec(env.base_env.output_spec)
            r = env.rollout(10)
            if stateful:
                locs.append(env.transform._loc["observation"])
                vars.append(env.transform._var["observation"])
                counts.append(env.transform._count)
            else:
                locs.append(r[-1]["next", "_vecnorm_loc", "observation"])
                vars.append(r[-1]["next", "_vecnorm_var", "observation"])
                counts.append(r[-1]["next", "_vecnorm_count"])
            env.close()
            vals.append(r)
            del env
        torch.testing.assert_close(
            counts[0].apply(lambda c1, c2: c1.expand_as(c2), counts[1]), counts[1]
        )
        torch.testing.assert_close(locs[0], locs[1])
        torch.testing.assert_close(vars[0], vars[1])
        assert_close(vals[0], vals[1], intersection=True)

    @pytest.mark.parametrize("stateful", [True, False])
    def test_vecnorm_stack(self, stateful):
        env = CountingEnv()
        env = env.append_transform(
            VecNorm(in_keys=["observation"], stateful=stateful, new_api=True)
        )
        env = env.append_transform(
            VecNorm(in_keys=["reward"], stateful=stateful, new_api=True)
        )
        env.check_env_specs(break_when_any_done="both")

    def test_init_stateful(self):
        env = CountingEnv()
        vecnorm = VecNorm(
            in_keys=["observation"], out_keys=["obs_norm"], stateful=True, new_api=True
        )
        assert vecnorm._loc is None
        env = env.append_transform(vecnorm)
        assert vecnorm._loc is not None

    @staticmethod
    def _test_vecnorm_subproc_auto(
        idx, make_env, queue_out: mp.Queue, queue_in: mp.Queue
    ):
        env = make_env()
        env.set_seed(1000 + idx)
        tensordict = env.reset()
        for _ in range(10):
            tensordict = env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put(True)
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        t = env.transform[1]
        loc = t._loc
        var = t._var
        count = t._count

        queue_out.put((loc, var, count))
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        env.close()
        queue_out.close()
        queue_in.close()
        del queue_in, queue_out

    @property
    def rename_t(self):
        return RenameTransform(in_keys=["observation"], out_keys=[("some", "obs")])

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.8.0"),
        reason="VecNorm shared memory synchronization requires PyTorch >= 2.8 "
        "when using spawn multiprocessing start method.",
    )
    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.parametrize("nprc", [2, 5])
    def test_vecnorm_parallel_auto(self, nprc):
        queues = []
        prcs = []
        if _has_gym:
            maker = lambda: TransformedEnv(
                GymEnv(PENDULUM_VERSIONED()),
                Compose(
                    self.rename_t,
                    VecNorm(
                        decay=0.9, in_keys=[("some", "obs"), "reward"], new_api=True
                    ),
                ),
            )
            check_env_specs(maker())
            make_env = EnvCreator(maker)
        else:
            maker = lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                Compose(
                    self.rename_t,
                    VecNorm(
                        decay=0.9, in_keys=[("some", "obs"), "reward"], new_api=True
                    ),
                ),
            )
            check_env_specs(maker())
            make_env = EnvCreator(maker)

        for idx in range(nprc):
            prc_queue_in = mp.Queue(1)
            prc_queue_out = mp.Queue(1)
            p = mp.Process(
                target=self._test_vecnorm_subproc_auto,
                args=(
                    idx,
                    make_env,
                    prc_queue_in,
                    prc_queue_out,
                ),
            )
            p.start()
            prcs.append(p)
            queues.append((prc_queue_in, prc_queue_out))

        try:
            dones = [queue[0].get() for queue in queues]
            assert all(dones)
            msg = "all_done"
            for idx in range(nprc):
                queues[idx][1].put(msg)

            td = TensorDict(
                make_env.state_dict()["transforms.1._extra_state"]
            ).unflatten_keys(VecNormV2.SEP)

            _loc = td["loc"]
            _var = td["var"]
            _count = td["count"]

            assert (_count == nprc * 11 + 2)[
                "some", "obs"
            ].all()  # 10 steps + reset + init
            assert (_count == nprc * 10 + 1)["reward"].all(), _count[
                "reward"
            ]  # 10 steps + init

            for idx in range(nprc):
                tup = queues[idx][0].get(timeout=TIMEOUT)
                (loc, var, count) = tup
                assert (loc == _loc).all(), "loc"
                assert (var == _var).all(), "var"
                assert (count == _count).all(), "count"

                loc, var, count = (_loc, _var, _count)
            msg = "all_done"
            for idx in range(nprc):
                queues[idx][1].put(msg)
        finally:
            del queues
            for p in prcs:
                try:
                    p.join(timeout=5)
                except TimeoutError:
                    p.terminate()

    @staticmethod
    def _run_parallelenv(parallel_env, queue_in, queue_out):
        tensordict = parallel_env.reset()
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("first round")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("second round")
        parallel_env.close()
        queue_out.close()
        queue_in.close()
        del parallel_env, queue_out, queue_in

    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason="Nested spawned multiprocessed is currently failing in python 3.11. "
        "See https://github.com/python/cpython/pull/108568 for info and fix.",
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.8.0"),
        reason="VecNorm shared memory synchronization requires PyTorch >= 2.8 "
        "when using spawn multiprocessing start method.",
    )
    def test_parallelenv_vecnorm(self):
        if _has_gym:
            make_env = EnvCreator(
                lambda: TransformedEnv(
                    GymEnv(PENDULUM_VERSIONED()),
                    Compose(
                        self.rename_t,
                        VecNorm(in_keys=[("some", "obs"), "reward"], new_api=True),
                    ),
                )
            )
        else:
            make_env = EnvCreator(
                lambda: TransformedEnv(
                    ContinuousActionVecMockEnv(),
                    Compose(
                        self.rename_t,
                        VecNorm(in_keys=[("some", "obs"), "reward"], new_api=True),
                    ),
                )
            )
        parallel_env = ParallelEnv(
            2,
            make_env,
        )
        try:
            queue_out = mp.Queue(1)
            queue_in = mp.Queue(1)
            proc = mp.Process(
                target=self._run_parallelenv, args=(parallel_env, queue_out, queue_in)
            )
            proc.start()
            parallel_sd = parallel_env.state_dict()
            assert "worker0" in parallel_sd
            worker_sd = parallel_sd["worker0"]
            td = TensorDict(worker_sd["transforms.1._extra_state"]).unflatten_keys(
                VecNormV2.SEP
            )
            queue_out.put("start")
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "first round"
            values = td.clone()
            queue_out.put("start")
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "second round"
            new_values = td.clone()
            for k, item in values.items():
                assert (item != new_values.get(k)).any(), k
        finally:
            try:
                proc.join(timeout=5)
            except TimeoutError:
                proc.terminate()
            if not parallel_env.is_closed:
                parallel_env.close(raise_if_closed=False)

    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.skipif(not _has_gym, reason="no gym library found")
    @pytest.mark.parametrize(
        "parallel",
        [
            None,
            False,
            True,
        ],
    )
    def test_vecnorm_rollout(self, parallel, thr=0.2, N=200, warmup=100):
        self.SEED += 1
        torch.manual_seed(self.SEED)

        if parallel is None:
            env = GymEnv(PENDULUM_VERSIONED())
        elif parallel:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )
        else:
            env = SerialEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )
        try:
            env.set_seed(self.SEED)
            t = VecNorm(decay=0.9, in_keys=["observation", "reward"], new_api=True)
            env_t = TransformedEnv(env, t)
            td = env_t.reset()
            tds = []
            for _ in range(N + warmup):
                td, td_ = env_t.step_and_maybe_reset(env.rand_action(td))
                tds.append(td)
                td = td_
            tds = torch.stack(tds[-N:], 0)
            obs = tds.get(("next", "observation"))
            obs = obs.view(-1, obs.shape[-1])
            mean = obs.mean(0)
            assert (abs(mean) < thr).all()
            std = obs.std(0)
            assert (abs(std - 1) < thr).all()
            self.SEED = 0
        finally:
            env.close(raise_if_closed=False)

    @pytest.mark.parametrize("stateful", [True, False])
    def test_denorm(self, stateful):
        """Test that denorm recovers original data for stateful VecNormV2."""
        torch.manual_seed(0)
        # Use in_keys == out_keys, the realistic use case where original is overwritten
        vecnorm = VecNormV2(
            in_keys=["observation"],
            stateful=stateful,
        )

        if not stateful:
            # Stateless mode should raise NotImplementedError for denorm
            td = TensorDict({"observation": torch.randn(10, 1)}, [10])
            with pytest.raises(NotImplementedError):
                vecnorm.denorm(td.clone())
        else:
            # Stateful mode: normalize then denormalize
            td_original = TensorDict({"observation": torch.randn(10, 1)}, [10])
            td_norm = td_original.clone()
            vecnorm._step(td_norm, td_norm)  # Normalizes 'observation' in-place

            # Verify normalization happened (observation was modified)
            assert not torch.allclose(
                td_norm["observation"], td_original["observation"]
            )

            # Denormalize - should recover original
            td_denorm = vecnorm.denorm(td_norm)

            # Check recovery of original data
            torch.testing.assert_close(
                td_denorm["observation"],
                td_original["observation"],
                rtol=1e-5,
                atol=1e-5,
            )

    def test_pickable(self):
        transform = VecNorm(in_keys=["observation"], new_api=True)
        env = CountingEnv()
        env = env.append_transform(transform)
        serialized = pickle.dumps(transform)
        transform2 = pickle.loads(serialized)
        assert transform.__dict__.keys() == transform2.__dict__.keys()
        for key in sorted(transform.__dict__.keys()):
            assert isinstance(transform.__dict__[key], type(transform2.__dict__[key]))

    def test_state_dict_vecnorm(self):
        transform0 = Compose(
            VecNorm(
                in_keys=["a", ("b", "c")],
                out_keys=["a_avg", ("b", "c_avg")],
                new_api=True,
            )
        )
        td = TensorDict({"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4])
        with pytest.warns(UserWarning, match="Querying state_dict on an uninitialized"):
            sd_empty = transform0.state_dict()
        assert not transform0[0].initialized

        transform1 = transform0.clone()
        # works fine
        transform1.load_state_dict(sd_empty)
        transform1._step(td, td)
        assert not transform0[0].initialized
        with pytest.raises(
            RuntimeError,
            match=r"called with a void state-dict while the instance is initialized.",
        ):
            transform1.load_state_dict(sd_empty)

        transform0._step(td, td)
        sd = transform0.state_dict()

        transform1 = transform0.clone()
        assert transform0[0]._loc.is_shared() is transform1[0]._loc.is_shared()

        # A clone does not have the same data ptr
        def assert_differs(a, b):
            assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()

        transform1[0]._loc.apply(assert_differs, transform0[0]._loc, filter_empty=True)

        transform1.load_state_dict(transform0.state_dict())

        def assert_same(a, b):
            assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

        transform1[0]._loc.apply(assert_same, transform0[0]._loc, filter_empty=True)

        transform1 = Compose(
            VecNorm(
                in_keys=["a", ("b", "c")],
                out_keys=["a_avg", ("b", "c_avg")],
                new_api=True,
            )
        )
        assert transform1[0]._loc is None
        with pytest.warns(
            UserWarning,
            match="VecNorm wasn't initialized and the tensordict is not shared",
        ):
            transform1.load_state_dict(sd)
        transform1._step(td, td)

        transform1 = Compose(
            VecNorm(
                in_keys=["a", ("b", "c")],
                out_keys=["a_avg", ("b", "c_avg")],
                new_api=True,
            )
        )
        transform1._step(td, td)
        transform1.load_state_dict(sd)

    def test_to_obsnorm_multikeys(self):
        transform0 = Compose(
            VecNorm(
                in_keys=["a", ("b", "c")],
                out_keys=["a_avg", ("b", "c_avg")],
                new_api=True,
            )
        )
        for _ in range(10):
            td = TensorDict(
                {"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4]
            )
            td0 = transform0._step(td, td.clone())
        # td0.update(transform0[0]._stateful_norm(td.select(*transform0[0].in_keys)))
        td1 = transform0[0].to_observation_norm()._step(td, td.clone())
        assert_allclose_td(td0, td1)

        loc = transform0[0].loc
        scale = transform0[0].scale
        keys = list(transform0[0].in_keys)
        td2 = (td.select(*keys) - loc) / (scale.clamp_min(torch.finfo(scale.dtype).eps))
        td2.rename_key_("a", "a_avg")
        td2.rename_key_(("b", "c"), ("b", "c_avg"))
        assert_allclose_td(td0.select(*td2.keys(True, True)), td2)

    def test_frozen(self):
        transform0 = VecNorm(
            in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")], new_api=True
        )
        with pytest.raises(
            RuntimeError, match="Make sure the VecNorm has been initialized"
        ):
            transform0.frozen_copy()
        td = TensorDict({"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4])
        td0 = transform0._step(td, td.clone())
        # td0.update(transform0._stateful_norm(td0.select(*transform0.in_keys)))

        transform1 = transform0.frozen_copy()
        td1 = transform1._step(td, td.clone())
        assert_allclose_td(td0, td1)

        td += 1
        td2 = transform0._step(td, td.clone())
        transform1._step(td, td.clone())
        # assert_allclose_td(td2, td3)
        with pytest.raises(AssertionError):
            assert_allclose_td(td0, td2)

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
            pytest.param(
                "mps",
                marks=pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS not available"
                ),
            ),
        ],
    )
    def test_shared_stats_stay_shared_on_device_move(self, device):
        shared = TensorDict(
            loc=TensorDict({"observation": torch.zeros(3)}, []),
            var=TensorDict({"observation": torch.zeros(3)}, []),
            count=TensorDict({"observation": torch.zeros((), dtype=torch.int64)}, []),
            batch_size=[],
        ).share_memory_()

        transform = VecNormV2(
            in_keys=["observation"],
            decay=0.9,
            shared_data=shared,
            reduce_batch_dims=True,
        ).freeze()
        ptr_before = transform._loc["observation"].data_ptr()

        transform = transform.to(device)

        assert transform._loc["observation"].device.type == "cpu"
        assert transform._loc["observation"].data_ptr() == ptr_before
        assert transform._stats_are_shared()

        updater = VecNormV2(
            in_keys=["observation"],
            decay=0.9,
            shared_data=shared,
            reduce_batch_dims=True,
        )
        updater._stateful_update(TensorDict({"observation": torch.ones(4, 3)}, [4]))
        batch = TensorDict(
            {"observation": torch.ones(4, 3, device=device)}, [4], device=device
        )
        normalized = transform._stateful_norm(batch)

        torch.testing.assert_close(
            transform._loc["observation"], updater._loc["observation"]
        )
        assert torch.isfinite(normalized["observation"]).all()

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
            pytest.param(
                "mps",
                marks=pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS not available"
                ),
            ),
        ],
    )
    @pytest.mark.skipif(not _has_gym, reason="gym not available")
    def test_vecnorm_gpu_device_handling(self, device):
        """Test that VecNorm(new_api=True) properly handles device movement to GPU/MPS.

        This test ensures that when an environment with VecNormV2 is moved to a
        non-CPU device, the internal statistics (_loc, _var, _count) are also moved
        to avoid device mismatch errors during normalization.
        """

        def assert_stats_on_device(transform, device_type, stage=""):
            """Helper to verify VecNorm statistics are on the expected device."""
            prefix = f"{stage} - " if stage else ""
            for key, val in transform._loc.items():
                assert (
                    val.device.type == device_type
                ), f"{prefix}_loc[{key}] not on {device_type}"
            for key, val in transform._var.items():
                assert (
                    val.device.type == device_type
                ), f"{prefix}_var[{key}] not on {device_type}"
            # _count can be a TensorDict or a plain tensor
            if isinstance(transform._count, TensorDictBase):
                for key, val in transform._count.items():
                    assert (
                        val.device.type == device_type
                    ), f"{prefix}_count[{key}] not on {device_type}"
            else:
                assert (
                    transform._count.device.type == device_type
                ), f"{prefix}_count not on {device_type}"

        env = GymEnv("CartPole-v1")
        env = env.append_transform(
            VecNorm(
                in_keys=["observation"],
                out_keys=["observation_norm"],
                new_api=True,
            )
        )
        env = env.to(device)

        td_reset = env.reset()
        assert td_reset.device.type == device
        assert td_reset["observation_norm"].device.type == device

        vecnorm_transform = env.transform
        assert isinstance(vecnorm_transform, VecNormV2)
        assert vecnorm_transform.initialized
        assert_stats_on_device(vecnorm_transform, device, "After initialization")

        for _ in range(5):
            action = env.rand_action(td_reset)
            td_step = env.step(td_reset.update(action))
            assert td_step["next", "observation_norm"].device.type == device
            td_reset = td_step["next"]

        assert_stats_on_device(vecnorm_transform, device, "After updates")

        env.close()


class TestVecNorm:
    SEED = -1

    @staticmethod
    def _test_vecnorm_subproc_auto(
        idx, make_env, queue_out: mp.Queue, queue_in: mp.Queue
    ):
        env = make_env()
        env.set_seed(1000 + idx)
        tensordict = env.reset()
        for _ in range(10):
            tensordict = env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put(True)
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        t = env.transform[1]
        obs_sum = t._td.get(("some", "obs_sum")).clone()
        obs_ssq = t._td.get(("some", "obs_ssq")).clone()
        obs_count = t._td.get(("some", "obs_count")).clone()
        reward_sum = t._td.get("reward_sum").clone()
        reward_ssq = t._td.get("reward_ssq").clone()
        reward_count = t._td.get("reward_count").clone()

        queue_out.put(
            (obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count)
        )
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "all_done"
        env.close()
        queue_out.close()
        queue_in.close()
        del queue_in, queue_out

    @property
    def rename_t(self):
        return RenameTransform(in_keys=["observation"], out_keys=[("some", "obs")])

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.8.0"),
        reason="VecNorm shared memory synchronization requires PyTorch >= 2.8 "
        "when using spawn multiprocessing start method.",
    )
    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.parametrize("nprc", [2, 5])
    def test_vecnorm_parallel_auto(self, nprc):
        queues = []
        prcs = []
        if _has_gym:
            maker = lambda: TransformedEnv(
                GymEnv(PENDULUM_VERSIONED()),
                Compose(
                    self.rename_t,
                    VecNorm(decay=1.0, in_keys=[("some", "obs"), "reward"]),
                ),
            )
            check_env_specs(maker())
            make_env = EnvCreator(maker)
        else:
            maker = lambda: TransformedEnv(
                ContinuousActionVecMockEnv(),
                Compose(
                    self.rename_t,
                    VecNorm(decay=1.0, in_keys=[("some", "obs"), "reward"]),
                ),
            )
            check_env_specs(maker())
            make_env = EnvCreator(maker)

        for idx in range(nprc):
            prc_queue_in = mp.Queue(1)
            prc_queue_out = mp.Queue(1)
            p = mp.Process(
                target=self._test_vecnorm_subproc_auto,
                args=(
                    idx,
                    make_env,
                    prc_queue_in,
                    prc_queue_out,
                ),
            )
            p.start()
            prcs.append(p)
            queues.append((prc_queue_in, prc_queue_out))

        dones = [queue[0].get() for queue in queues]
        assert all(dones)
        msg = "all_done"
        for idx in range(nprc):
            queues[idx][1].put(msg)

        td = TensorDict(
            make_env.state_dict()["transforms.1._extra_state"]
        ).unflatten_keys(VecNorm.SEP)

        obs_sum = td.get(("some", "obs_sum")).clone()
        obs_ssq = td.get(("some", "obs_ssq")).clone()
        obs_count = td.get(("some", "obs_count")).clone()
        reward_sum = td.get("reward_sum").clone()
        reward_ssq = td.get("reward_ssq").clone()
        reward_count = td.get("reward_count").clone()

        assert obs_count == nprc * 11 + 2  # 10 steps + reset + init

        for idx in range(nprc):
            tup = queues[idx][0].get(timeout=TIMEOUT)
            (
                _obs_sum,
                _obs_ssq,
                _obs_count,
                _reward_sum,
                _reward_ssq,
                _reward_count,
            ) = tup
            assert (obs_sum == _obs_sum).all(), "sum"
            assert (obs_ssq == _obs_ssq).all(), "ssq"
            assert (obs_count == _obs_count).all(), "count"
            assert (reward_sum == _reward_sum).all(), "sum"
            assert (reward_ssq == _reward_ssq).all(), "ssq"
            assert (reward_count == _reward_count).all(), "count"

            obs_sum, obs_ssq, obs_count, reward_sum, reward_ssq, reward_count = (
                _obs_sum,
                _obs_ssq,
                _obs_count,
                _reward_sum,
                _reward_ssq,
                _reward_count,
            )
        msg = "all_done"
        for idx in range(nprc):
            queues[idx][1].put(msg)

        del queues
        for p in prcs:
            p.join()

    @staticmethod
    def _run_parallelenv(parallel_env, queue_in, queue_out):
        tensordict = parallel_env.reset()
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("first round")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "start"
        for _ in range(10):
            tensordict = parallel_env.rand_step(tensordict)
            tensordict = step_mdp(tensordict)
        queue_out.put("second round")
        parallel_env.close()
        queue_out.close()
        queue_in.close()
        del parallel_env, queue_out, queue_in

    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason="Nested spawned multiprocessed is currently failing in python 3.11. "
        "See https://github.com/python/cpython/pull/108568 for info and fix.",
    )
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.8.0"),
        reason="VecNorm shared memory synchronization requires PyTorch >= 2.8 "
        "when using spawn multiprocessing start method.",
    )
    def test_parallelenv_vecnorm(self):
        if _has_gym:
            make_env = EnvCreator(
                lambda: TransformedEnv(
                    GymEnv(PENDULUM_VERSIONED()),
                    Compose(
                        self.rename_t,
                        VecNorm(in_keys=[("some", "obs"), "reward"]),
                    ),
                )
            )
        else:
            make_env = EnvCreator(
                lambda: TransformedEnv(
                    ContinuousActionVecMockEnv(),
                    Compose(
                        self.rename_t,
                        VecNorm(in_keys=[("some", "obs"), "reward"]),
                    ),
                )
            )
        parallel_env = ParallelEnv(
            2,
            make_env,
        )
        queue_out = mp.Queue(1)
        queue_in = mp.Queue(1)
        proc = mp.Process(
            target=self._run_parallelenv, args=(parallel_env, queue_out, queue_in)
        )
        proc.start()
        parallel_sd = parallel_env.state_dict()
        assert "worker0" in parallel_sd
        worker_sd = parallel_sd["worker0"]
        td = TensorDict(worker_sd["transforms.1._extra_state"]).unflatten_keys(
            VecNorm.SEP
        )
        queue_out.put("start")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "first round"
        values = td.clone()
        queue_out.put("start")
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "second round"
        new_values = td.clone()
        for k, item in values.items():
            if k in ["reward_sum", "reward_ssq"] and not _has_gym:
                # mocking env rewards are sparse
                continue
            assert (item != new_values.get(k)).any(), k
        proc.join()
        if not parallel_env.is_closed:
            parallel_env.close()

    @retry(AssertionError, tries=10, delay=0)
    @pytest.mark.skipif(not _has_gym, reason="no gym library found")
    @pytest.mark.parametrize(
        "parallel",
        [
            None,
            False,
            True,
        ],
    )
    def test_vecnorm_rollout(self, parallel, thr=0.2, N=200):
        self.SEED += 1
        torch.manual_seed(self.SEED)

        if parallel is None:
            env = GymEnv(PENDULUM_VERSIONED())
        elif parallel:
            env = ParallelEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )
        else:
            env = SerialEnv(
                num_workers=5, create_env_fn=lambda: GymEnv(PENDULUM_VERSIONED())
            )

        env.set_seed(self.SEED)
        t = VecNorm(decay=1.0)
        env_t = TransformedEnv(env, t)
        td = env_t.reset()
        tds = []
        for _ in range(N):
            td = env_t.rand_step(td)
            tds.append(td.clone())
            td = step_mdp(td)
            if td.get("done").any():
                td = env_t.reset()
        tds = torch.stack(tds, 0)
        obs = tds.get(("next", "observation"))
        obs = obs.view(-1, obs.shape[-1])
        mean = obs.mean(0)
        assert (abs(mean) < thr).all()
        std = obs.std(0)
        assert (abs(std - 1) < thr).all()
        if not env_t.is_closed:
            env_t.close()
        self.SEED = 0

    def test_pickable(self):
        transform = VecNorm()
        serialized = pickle.dumps(transform)
        transform2 = pickle.loads(serialized)
        assert transform.__dict__.keys() == transform2.__dict__.keys()
        for key in sorted(transform.__dict__.keys()):
            assert isinstance(transform.__dict__[key], type(transform2.__dict__[key]))

    def test_denorm(self):
        """Test that denorm recovers original data for VecNorm."""
        torch.manual_seed(0)
        # Use in_keys == out_keys, the realistic use case where original is overwritten
        vecnorm = VecNorm(in_keys=["observation"])
        td_original = TensorDict({"observation": torch.randn(10, 1)}, [10])
        td_norm = td_original.clone()
        vecnorm._step(td_norm, td_norm)  # Normalizes 'observation' in-place

        # Verify normalization happened (observation was modified)
        assert not torch.allclose(td_norm["observation"], td_original["observation"])

        # Denormalize - should recover original
        td_denorm = vecnorm.denorm(td_norm)

        # Check recovery of original data
        torch.testing.assert_close(
            td_denorm["observation"], td_original["observation"], rtol=1e-5, atol=1e-5
        )

    def test_state_dict_vecnorm(self):
        transform0 = Compose(
            VecNorm(in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")])
        )
        td = TensorDict({"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4])
        with pytest.warns(UserWarning, match="Querying state_dict on an uninitialized"):
            sd_empty = transform0.state_dict()

        transform1 = transform0.clone()
        # works fine
        transform1.load_state_dict(sd_empty)
        transform1._step(td, td)
        with pytest.raises(KeyError, match="Could not find a tensordict"):
            transform1.load_state_dict(sd_empty)

        transform0._step(td, td)
        sd = transform0.state_dict()

        transform1 = transform0.clone()
        assert transform0[0]._td.is_shared() is transform1[0]._td.is_shared()

        def assert_differs(a, b):
            assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

        transform1[0]._td.apply(assert_differs, transform0[0]._td, filter_empty=True)

        transform1 = Compose(
            VecNorm(in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")])
        )
        with pytest.warns(UserWarning, match="VecNorm wasn't initialized"):
            transform1.load_state_dict(sd)
        transform1._step(td, td)

        transform1 = Compose(
            VecNorm(in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")])
        )
        transform1._step(td, td)
        transform1.load_state_dict(sd)

    def test_to_obsnorm_multikeys(self):
        transform0 = Compose(
            VecNorm(in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")])
        )
        td = TensorDict({"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4])
        td0 = transform0._step(td, td.clone())
        td1 = transform0[0].to_observation_norm()._step(td, td.clone())
        assert_allclose_td(td0, td1)

        loc = transform0[0].loc
        scale = transform0[0].scale
        keys = list(transform0[0].in_keys)
        td2 = (td.select(*keys) - loc) / (scale + torch.finfo(scale.dtype).eps)
        td2.rename_key_("a", "a_avg")
        td2.rename_key_(("b", "c"), ("b", "c_avg"))
        assert_allclose_td(td0.select(*td2.keys(True, True)), td2)

    def test_frozen(self):
        transform0 = VecNorm(
            in_keys=["a", ("b", "c")], out_keys=["a_avg", ("b", "c_avg")]
        )
        with pytest.raises(
            RuntimeError, match="Make sure the VecNorm has been initialized"
        ):
            transform0.frozen_copy()
        td = TensorDict({"a": torch.randn(3, 4), ("b", "c"): torch.randn(3, 4)}, [3, 4])
        td0 = transform0._step(td, td.clone())
        transform1 = transform0.frozen_copy()
        td1 = transform1._step(td, td.clone())
        assert_allclose_td(td0, td1)

        td += 1
        td2 = transform0._step(td, td.clone())
        td3 = transform1._step(td, td.clone())
        assert_allclose_td(td2, td3)
        with pytest.raises(AssertionError):
            assert_allclose_td(td0, td2)
