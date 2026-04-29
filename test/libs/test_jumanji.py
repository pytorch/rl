# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, TensorDict

from torchrl.envs.libs.jumanji import _has_jumanji, JumanjiEnv
from torchrl.envs.utils import check_env_specs


def _jumanji_envs():
    if not _has_jumanji:
        return ()
    return JumanjiEnv.available_envs[-10:-5]


@pytest.mark.skipif(not _has_jumanji, reason="jumanji not installed")
@pytest.mark.slow
@pytest.mark.parametrize("envname", _jumanji_envs())
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
        env = JumanjiEnv(envname, batch_size=batch_size, jit=True)
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=50)
        env.close()
        del env
        assert tdreset.batch_size == batch_size
        assert tdrollout.batch_size[:-1] == batch_size

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_jumanji_spec_rollout(self, envname, batch_size):
        env = JumanjiEnv(envname, batch_size=batch_size, jit=True)
        env.set_seed(0)
        check_env_specs(env)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_jumanji_consistency(self, envname, batch_size):
        import jax
        import jax.numpy as jnp
        import numpy as onp
        from torchrl.envs.libs.jax_utils import _tree_flatten

        env = JumanjiEnv(envname, batch_size=batch_size, jit=True)
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

    @pytest.mark.parametrize("batch_size", [[3], []])
    def test_jumanji_rendering(self, envname, batch_size):
        # check that this works with a batch-size
        env = JumanjiEnv(envname, from_pixels=True, batch_size=batch_size, jit=True)
        env.set_seed(0)
        env.transform.transform_observation_spec(env.base_env.observation_spec.clone())

        r = env.rollout(10)
        pixels = r["pixels"]
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(np.asarray(pixels))
            assert batch_size
        else:
            assert not batch_size
        assert pixels.unique().numel() > 1
        assert pixels.dtype == torch.uint8

        check_env_specs(env)

    @pytest.mark.parametrize("jit", [True, False])
    def test_jumanji_batch_unlocked(self, envname, jit):
        torch.manual_seed(0)
        env = JumanjiEnv(envname, jit=jit)
        env.set_seed(0)
        assert not env.batch_locked
        reset = env.reset(TensorDict(batch_size=[16]))
        assert reset.batch_size == (16,)
        env.rand_step(reset)
        r = env.rollout(
            2000, auto_reset=False, tensordict=reset, break_when_all_done=True
        )
        assert r.batch_size[0] == 16
        done = r["next", "done"]
        assert done.any(-2).all() or (r.shape[-1] == 2000)
