# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc
import importlib.util
import os

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import SerialEnv
from torchrl.envs.batched_envs import ParallelEnv
from torchrl.envs.libs.brax import _has_brax, BraxEnv, BraxWrapper
from torchrl.envs.libs.jax_utils import (
    _ndarray_to_tensor,
    _tensor_to_ndarray,
    _tree_flatten,
)
from torchrl.envs.utils import check_env_specs
from torchrl.testing import get_available_devices

_has_jax = importlib.util.find_spec("jax") is not None
_has_psutil = importlib.util.find_spec("psutil") is not None


@pytest.mark.skipif(not _has_brax, reason="brax not installed")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("envname", ["fast"])
class TestBrax:
    @pytest.fixture(autouse=True)
    def _setup_jax(self):
        """Configure JAX for proper GPU initialization."""
        import jax

        # Set JAX environment variables for better GPU handling
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

        # Try to initialize JAX with GPU, fallback to CPU if it fails
        try:
            jax.devices()
        except Exception:
            # Fallback to CPU
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            jax.config.update("jax_platform_name", "cpu")

        yield

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_brax_constructor(self, envname, requires_grad, device):
        env0 = BraxEnv(envname, requires_grad=requires_grad, device=device)
        env1 = BraxWrapper(env0._env, requires_grad=requires_grad, device=device)

        env0.set_seed(0)
        torch.manual_seed(0)
        init = env0.reset()
        if requires_grad:
            init = init.apply(
                lambda x: x.requires_grad_(True) if x.is_floating_point() else x
            )
        r0 = env0.rollout(10, tensordict=init, auto_reset=False)
        assert r0.requires_grad == requires_grad

        env1.set_seed(0)
        torch.manual_seed(0)
        init = env1.reset()
        if requires_grad:
            init = init.apply(
                lambda x: x.requires_grad_(True) if x.is_floating_point() else x
            )
        r1 = env1.rollout(10, tensordict=init, auto_reset=False)
        assert r1.requires_grad == requires_grad
        assert_allclose_td(r0.data, r1.data)

    def test_brax_seeding(self, envname, device):
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = BraxEnv(envname, device=device)
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
    def test_brax_batch_size(self, envname, batch_size, device):
        env = BraxEnv(envname, batch_size=batch_size, device=device)
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=50)
        env.close()
        del env
        assert tdreset.batch_size == batch_size
        assert tdrollout.batch_size[:-1] == batch_size

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_brax_spec_rollout(self, envname, batch_size, device):
        env = BraxEnv(envname, batch_size=batch_size, device=device)
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
    def test_brax_consistency(self, envname, batch_size, requires_grad, device):
        import jax
        import jax.numpy as jnp

        env = BraxEnv(
            envname, batch_size=batch_size, requires_grad=requires_grad, device=device
        )
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
    def test_brax_grad(self, envname, batch_size, device):
        batch_size = (1,)
        env = BraxEnv(envname, batch_size=batch_size, requires_grad=True, device=device)
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
        self, envname, batch_size, parallel, maybe_fork_ParallelEnv, device, n=1
    ):
        def make_brax():
            env = BraxEnv(
                envname, batch_size=batch_size, requires_grad=False, device=device
            )
            env.set_seed(1)
            return env

        if parallel:
            env = maybe_fork_ParallelEnv(n, make_brax)
        else:
            env = SerialEnv(n, make_brax)
        check_env_specs(env)
        tensordict = env.rollout(3)
        assert tensordict.shape == torch.Size([n, *batch_size, 3])

    def test_brax_memory_leak(self, envname, device):
        """Test memory usage with different cache clearing strategies."""
        import psutil

        process = psutil.Process(os.getpid())
        env = BraxEnv(
            envname,
            batch_size=[10],
            requires_grad=True,
            device=device,
        )
        env.clear_cache()
        gc.collect()
        env.set_seed(0)
        next_td = env.reset()
        num_steps = 200
        policy = TensorDictModule(
            torch.nn.Linear(
                env.observation_spec[env.observation_keys[0]].shape[-1],
                env.action_spec.shape[-1],
                device=device,
            ),
            in_keys=env.observation_keys[:1],
            out_keys=["action"],
        )
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        for i in range(num_steps):
            policy(next_td)
            out_td, next_td = env.step_and_maybe_reset(next_td)
            if i % 50 == 0:
                loss = out_td["next", "observation"].sum()
                loss.backward()
                next_td = next_td.detach().clone()
            # gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        assert (
            memory_increase < 100
        ), f"Memory leak with automatic clearing: {memory_increase:.2f} MB"

    def test_brax_cache_clearing(self, envname, device):
        env = BraxEnv(envname, batch_size=[1], requires_grad=True, device=device)
        env.clear_cache()
        for _ in range(5):
            env.clear_cache()

    def test_num_workers_returns_lazy_parallel_env(self, envname, device):
        """Ensure BraxEnv with num_workers > 1 returns a lazy ParallelEnv."""
        env = BraxEnv(envname, num_workers=3, device=device)
        try:
            assert isinstance(env, ParallelEnv)
            assert env.num_workers == 3
            # ParallelEnv should be lazy (not started yet)
            assert env.is_closed
            # configure_parallel should work before env starts
            env.configure_parallel(use_buffers=False)
            env.reset()
            assert not env.is_closed
            assert env.batch_size == torch.Size([3])
        finally:
            env.close()

    def test_set_seed_and_reset_works(self, envname, device):
        """Smoke test that setting seed and reset works for BraxEnv."""
        env = BraxEnv(envname, device=device)
        try:
            final_seed = env.set_seed(0)
            assert final_seed is not None
            td = env.reset()
            assert isinstance(td, TensorDict)
        finally:
            env.close()

    @pytest.mark.parametrize("freq", [10, None, False])
    def test_brax_automatic_cache_clearing_parameter(self, envname, device, freq):
        env = BraxEnv(
            envname,
            batch_size=[1],
            requires_grad=True,
            device=device,
            cache_clear_frequency=freq,
        )
        if freq is False:
            assert env._cache_clear_frequency is False
        elif freq is None:
            assert env._cache_clear_frequency == 20  # Default value
        else:
            assert env._cache_clear_frequency == freq
        env.set_seed(0)
        next_td = env.reset()
        for i in range(10):
            action = env.action_spec.rand()
            next_td["action"] = action
            out_td, next_td = env.step_and_maybe_reset(next_td)
            assert env._step_count == i + 1

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_brax_kwargs_preserved_with_seed(self, envname, device):
        """Test that constructor kwargs are preserved when seed is provided.

        Regression test for a bug where `kwargs` were overwritten when `_seed`
        was not None.
        """
        env = BraxEnv(
            envname,
            device=device,
        )
        try:
            final_seed = env.set_seed(1)
            assert final_seed is not None
            td = env.reset()
            assert isinstance(td, TensorDict)
            preserved = False
            if hasattr(env, "_constructor_kwargs") and isinstance(
                env._constructor_kwargs, dict
            ):
                preserved = env._constructor_kwargs.get("env_name") == envname
            assert preserved, "constructor kwargs were not preserved after set_seed"
        finally:
            env.close()
