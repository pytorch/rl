# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch
from tensordict import assert_allclose_td, TensorDict

from torchrl.envs import SerialEnv
from torchrl.envs.batched_envs import ParallelEnv
from torchrl.envs.libs.mujoco_playground import (
    _has_mujoco_playground,
    KNOWN_MARL_MAPPINGS,
    MujocoPlaygroundAgentMapping,
    MujocoPlaygroundAgentSpec,
    MujocoPlaygroundEnv,
    MujocoPlaygroundWrapper,
)
from torchrl.envs.utils import check_env_specs
from torchrl.testing import get_available_devices

_has_jax = importlib.util.find_spec("jax") is not None

# Default config: force JAX backend on all envs (CPU-only machines lack other backends)
_JAX_CONFIG = {"impl": "jax"}

# Fast flat-obs environment used for most tests
_FLAT_OBS_ENV = "CartpoleBalance"

# Multi-agent tests use CheetahRun (action_size=6, flat obs)
_MARL_ENV = "CheetahRun"

# dm_control_suite environments known to exist (subset)
_KNOWN_DM_ENVS = ["CartpoleBalance", "CheetahRun", "WalkerWalk"]

# Locomotion environments known to exist (subset)
_KNOWN_LOCOMOTION_ENVS = ["Go1JoystickFlatTerrain"]

# Manipulation environments known to exist (subset)
_KNOWN_MANIPULATION_ENVS = ["PandaPickCube"]


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("envname", [_FLAT_OBS_ENV])
class TestMujocoPlayground:
    @pytest.fixture(autouse=True)
    def _setup_jax(self):
        """Configure JAX for proper GPU initialization."""
        import jax

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

        try:
            jax.devices()
        except Exception:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            jax.config.update("jax_platform_name", "cpu")

        yield

    def test_constructor_wrapper(self, envname, device):
        """MujocoPlaygroundWrapper works from a pre-built env."""
        from mujoco_playground import dm_control_suite

        base_env = dm_control_suite.load(envname, config_overrides=_JAX_CONFIG)
        env = MujocoPlaygroundWrapper(base_env, device=device)
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()

    def test_constructor_env(self, envname, device):
        """MujocoPlaygroundEnv wraps env by name."""
        env0 = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        from mujoco_playground import dm_control_suite

        base_env = dm_control_suite.load(envname, config_overrides=_JAX_CONFIG)
        env1 = MujocoPlaygroundWrapper(base_env, device=device)

        env0.set_seed(0)
        torch.manual_seed(0)
        r0 = env0.rollout(5)
        env1.set_seed(0)
        torch.manual_seed(0)
        r1 = env1.rollout(5)
        assert_allclose_td(r0.data, r1.data)
        env0.close()
        env1.close()

    def test_seeding(self, envname, device):
        """Same seed produces identical rollouts."""
        final_seed = []
        tdreset = []
        tdrollout = []
        for _ in range(2):
            env = MujocoPlaygroundEnv(
                envname, device=device, config_overrides=_JAX_CONFIG
            )
            torch.manual_seed(0)
            np.random.seed(0)
            final_seed.append(env.set_seed(0))
            tdreset.append(env.reset())
            tdrollout.append(env.rollout(max_steps=20))
            env.close()
            del env
        assert final_seed[0] == final_seed[1]
        assert_allclose_td(*tdreset)
        assert_allclose_td(*tdrollout)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_batch_size(self, envname, batch_size, device):
        """batch_size propagates to reset/rollout output shapes."""
        env = MujocoPlaygroundEnv(
            envname, batch_size=batch_size, device=device, config_overrides=_JAX_CONFIG
        )
        env.set_seed(0)
        tdreset = env.reset()
        tdrollout = env.rollout(max_steps=10)
        env.close()
        del env
        assert tdreset.batch_size == torch.Size(batch_size)
        assert tdrollout.batch_size[:-1] == torch.Size(batch_size)

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    def test_spec_rollout(self, envname, batch_size, device):
        """check_env_specs passes for all batch sizes."""
        env = MujocoPlaygroundEnv(
            envname, batch_size=batch_size, device=device, config_overrides=_JAX_CONFIG
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    @pytest.mark.parametrize("batch_size", [(), (5,), (5, 4)])
    @pytest.mark.parametrize("parallel", [False, True])
    def test_parallel(
        self, envname, batch_size, parallel, maybe_fork_ParallelEnv, device, n=1
    ):
        """Env works inside SerialEnv and ParallelEnv."""

        def make_env():
            env = MujocoPlaygroundEnv(
                envname,
                batch_size=batch_size,
                device=device,
                config_overrides=_JAX_CONFIG,
            )
            env.set_seed(1)
            return env

        if parallel:
            env = maybe_fork_ParallelEnv(n, make_env)
        else:
            env = SerialEnv(n, make_env)
        check_env_specs(env)
        tensordict = env.rollout(3)
        assert tensordict.shape == torch.Size([n, *batch_size, 3])

    def test_num_workers_returns_lazy_parallel_env(self, envname, device):
        """num_workers > 1 returns a lazy ParallelEnv."""
        env = MujocoPlaygroundEnv(
            envname, num_workers=3, device=device, config_overrides=_JAX_CONFIG
        )
        try:
            assert isinstance(env, ParallelEnv)
            assert env.num_workers == 3
            assert env.is_closed
            env.configure_parallel(use_buffers=False)
            env.reset()
            assert not env.is_closed
        finally:
            env.close()

    def test_set_seed_and_reset_works(self, envname, device):
        """Setting seed then reset produces a valid TensorDict."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        try:
            final_seed = env.set_seed(0)
            assert final_seed is not None
            td = env.reset()
            assert isinstance(td, TensorDict)
        finally:
            env.close()

    def test_flat_obs_has_observation_key(self, envname, device):
        """Flat-obs environments expose an 'observation' key."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        env.set_seed(0)
        assert "observation" in env.observation_spec.keys()
        td = env.reset()
        assert "observation" in td.keys()
        env.close()

    def test_config_override(self, envname, device):
        """config_overrides are forwarded to the environment constructor."""
        env = MujocoPlaygroundEnv(
            envname,
            config_overrides={"impl": "jax", "episode_length": 200},
            device=device,
        )
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundAvailableEnvs:
    @pytest.fixture(autouse=True)
    def _setup_jax(self):
        import jax

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        try:
            jax.devices()
        except Exception:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            jax.config.update("jax_platform_name", "cpu")
        yield

    def test_available_envs_contains_all_suites(self, device):
        """available_envs spans dm_control_suite, locomotion, and manipulation."""
        envs = MujocoPlaygroundEnv.available_envs
        assert len(envs) > 0
        for name in _KNOWN_DM_ENVS:
            assert name in envs, f"{name} missing from available_envs"
        for name in _KNOWN_LOCOMOTION_ENVS:
            assert name in envs, f"{name} missing from available_envs"
        for name in _KNOWN_MANIPULATION_ENVS:
            assert name in envs, f"{name} missing from available_envs"

    @pytest.mark.parametrize("envname", _KNOWN_DM_ENVS[:1])
    def test_dm_control_suite_env(self, envname, device):
        """dm_control_suite env constructs and resets."""
        env = MujocoPlaygroundEnv(envname, device=device, config_overrides=_JAX_CONFIG)
        env.set_seed(0)
        td = env.reset()
        assert isinstance(td, TensorDict)
        env.close()

    def test_env_name_property(self, device):
        """env_name property returns correct name."""
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        assert env.env_name == _FLAT_OBS_ENV
        env.close()

    def test_repr(self, device):
        """__repr__ includes env name, batch_size, device."""
        env = MujocoPlaygroundEnv(
            _FLAT_OBS_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        r = repr(env)
        assert _FLAT_OBS_ENV in r
        assert "batch_size" in r
        env.close()


def _make_two_agent_mapping(env):
    """Build a 2-agent mapping for a flat-obs env by splitting obs/action in half."""
    action_size = env._env.action_size
    obs_size = env._env.observation_size
    half_act = action_size // 2
    half_obs = obs_size // 2
    return MujocoPlaygroundAgentMapping(
        agents=[
            MujocoPlaygroundAgentSpec(
                name="agent_0",
                action_indices=list(range(half_act)),
                observation_indices=list(range(half_obs)),
            ),
            MujocoPlaygroundAgentSpec(
                name="agent_1",
                action_indices=list(range(half_act, action_size)),
                observation_indices=list(range(half_obs, obs_size)),
            ),
        ],
        homogenization_mode="none",
    )


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
@pytest.mark.parametrize("device", get_available_devices())
class TestMujocoPlaygroundMultiAgent:
    @pytest.fixture(autouse=True)
    def _setup_jax(self):
        import jax

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        try:
            jax.devices()
        except Exception:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            jax.config.update("jax_platform_name", "cpu")
        yield

    def _make_env(self, device, mapping=None, batch_size=()):
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            batch_size=batch_size,
            device=device,
            config_overrides=_JAX_CONFIG,
        )
        if mapping is None:
            mapping = _make_two_agent_mapping(env)
        env.close()
        return MujocoPlaygroundEnv(
            _MARL_ENV,
            batch_size=batch_size,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )

    def test_construction_two_agents(self, device):
        env = self._make_env(device)
        env.set_seed(0)
        env.close()

    def test_action_spec_is_nested_composite(self, device):
        env = self._make_env(device)
        from torchrl.data.tensor_specs import Composite

        assert isinstance(env.action_spec, Composite)
        assert "agent_0" in env.action_spec.keys()
        assert "agent_1" in env.action_spec.keys()
        env.close()

    def test_reset_output_keys(self, device):
        env = self._make_env(device)
        env.set_seed(0)
        td = env.reset()
        assert "agent_0" in td.keys()
        assert "agent_1" in td.keys()
        assert "done" in td.keys()
        assert "terminated" in td.keys()
        assert "observation" in td["agent_0"].keys()
        assert "observation" in td["agent_1"].keys()
        env.close()

    def test_reset_obs_shapes_none_mode(self, device):
        # Build mapping explicitly so we know obs sizes
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        act_size = base_env._env.action_size
        base_env.close()

        half_obs = obs_size // 2
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)),
                    observation_indices=list(range(half_obs)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(act_size // 2, act_size)),
                    observation_indices=list(range(half_obs, obs_size)),
                ),
            ],
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()
        assert td["agent_0", "observation"].shape[-1] == half_obs
        assert td["agent_1", "observation"].shape[-1] == obs_size - half_obs
        env.close()

    def test_state_not_in_tensordict(self, device):
        # JAX state is kept as instance var; it must not appear in TensorDict output
        env = self._make_env(device)
        env.set_seed(0)
        td = env.reset()
        assert "state" not in td.keys()
        assert "state" not in td["agent_0"].keys()
        assert "state" not in td["agent_1"].keys()
        env.close()

    def test_rollout_works(self, device):
        env = self._make_env(device)
        env.set_seed(0)
        td = env.rollout(3)
        assert td.shape[-1] == 3
        env.close()

    def test_reward_duplicated_to_all_agents(self, device):
        env = self._make_env(device)
        env.set_seed(0)
        td = env.rollout(1)
        r0 = td[..., 0]["next", "agent_0", "reward"]
        r1 = td[..., 0]["next", "agent_1", "reward"]
        torch.testing.assert_close(r0, r1)
        env.close()

    def test_action_shapes_in_rollout(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        act_size = base_env._env.action_size
        base_env.close()

        env = self._make_env(device)
        env.set_seed(0)
        td = env.rollout(2)
        half = act_size // 2
        assert td["agent_0", "action"].shape[-1] == half
        assert td["agent_1", "action"].shape[-1] == act_size - half
        env.close()

    def test_validation_overlapping_actions_raises(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1, 2],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3, 4, 5],  # index 2 overlaps
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="overlap"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_validation_missing_action_indices_raises(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],  # only 2 of 6 — missing 2,3,4,5
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3],  # still missing 4,5
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="Missing indices"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_validation_out_of_range_action_index_raises(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        act_size = base_env._env.action_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)) + [999],
                    observation_indices=list(range(obs_size)),
                ),
            ],
        )
        with pytest.raises(ValueError, match="out of range"):
            MujocoPlaygroundEnv(
                _MARL_ENV,
                device=device,
                agent_mapping=mapping,
                config_overrides=_JAX_CONFIG,
            )

    def test_homogenization_max_shapes(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        base_env.close()

        # Unequal action splits: 2 vs 4
        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=[2, 3, 4, 5],
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="max",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()

        max_act = 4  # max(2, 4)
        max_obs = max(obs_size // 2, obs_size - obs_size // 2) + 2  # +2 agents one-hot
        assert td["agent_0", "observation"].shape[-1] == max_obs
        assert td["agent_1", "observation"].shape[-1] == max_obs
        assert env.action_spec["agent_0", "action"].shape[-1] == max_act
        assert env.action_spec["agent_1", "action"].shape[-1] == max_act
        env.close()

    def test_homogenization_concat_shapes(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        act_size = base_env._env.action_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)),
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(act_size // 2, act_size)),
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="concat",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()

        assert td["agent_0", "observation"].shape[-1] == obs_size
        assert td["agent_1", "observation"].shape[-1] == obs_size
        assert env.action_spec["agent_0", "action"].shape[-1] == act_size
        assert env.action_spec["agent_1", "action"].shape[-1] == act_size
        env.close()

    def test_check_env_specs_none_mode(self, device):
        env = self._make_env(device)
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_check_env_specs_max_mode(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        act_size = base_env._env.action_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=[0, 1],
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(2, act_size)),
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="max",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_check_env_specs_concat_mode(self, device):
        base_env = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        obs_size = base_env._env.observation_size
        act_size = base_env._env.action_size
        base_env.close()

        mapping = MujocoPlaygroundAgentMapping(
            agents=[
                MujocoPlaygroundAgentSpec(
                    name="agent_0",
                    action_indices=list(range(act_size // 2)),
                    observation_indices=list(range(obs_size // 2)),
                ),
                MujocoPlaygroundAgentSpec(
                    name="agent_1",
                    action_indices=list(range(act_size // 2, act_size)),
                    observation_indices=list(range(obs_size // 2, obs_size)),
                ),
            ],
            homogenization_mode="concat",
        )
        env = MujocoPlaygroundEnv(
            _MARL_ENV,
            device=device,
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    @pytest.mark.parametrize("batch_size", [(4,), (4, 3)])
    def test_batch_size_with_agent_mapping(self, device, batch_size):
        env = self._make_env(device, batch_size=batch_size)
        env.set_seed(0)
        td = env.reset()
        assert td.batch_size == torch.Size(batch_size)
        assert td["agent_0", "observation"].shape[: len(batch_size)] == torch.Size(
            batch_size
        )
        assert td["agent_1", "observation"].shape[: len(batch_size)] == torch.Size(
            batch_size
        )
        env.close()

    def test_no_mapping_regression(self, device):
        env_marl = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        env_plain = MujocoPlaygroundEnv(
            _MARL_ENV, device=device, config_overrides=_JAX_CONFIG
        )
        env_marl.set_seed(42)
        env_plain.set_seed(42)
        td_plain = env_plain.rollout(5)
        # single-agent mode: observation key present at top level
        assert "observation" in td_plain.keys()
        assert "agent_0" not in td_plain.keys()
        env_marl.close()
        env_plain.close()


@pytest.mark.skipif(
    not _has_mujoco_playground, reason="mujoco_playground not installed"
)
class TestKnownMarlMappings:
    """Tests for KNOWN_MARL_MAPPINGS and string-based agent_mapping lookup."""

    def test_known_mappings_present(self):
        expected = {
            "ant_4x2",
            "halfcheetah_6x1",
            "hopper_3x1",
            "humanoid_9|8",
            "walker2d_2x3",
        }
        assert set(KNOWN_MARL_MAPPINGS.keys()) == expected

    def test_known_mappings_are_agent_mapping_instances(self):
        for name, mapping in KNOWN_MARL_MAPPINGS.items():
            assert isinstance(mapping, MujocoPlaygroundAgentMapping), name
            assert len(mapping.agents) > 0, name
            for agent in mapping.agents:
                assert isinstance(agent, MujocoPlaygroundAgentSpec), name
                assert len(agent.action_indices) > 0, name
                assert len(agent.observation_indices) > 0, name

    def test_string_mapping_resolves(self):
        # CheetahRun has action_size=6 — matches halfcheetah_6x1
        env = MujocoPlaygroundEnv(
            "CheetahRun",
            device="cpu",
            agent_mapping="halfcheetah_6x1",
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()
        for i in range(6):
            assert f"agent_{i}" in td.keys(), f"agent_{i} missing from reset td"
        env.close()

    def test_string_mapping_check_env_specs(self):
        env = MujocoPlaygroundEnv(
            "CheetahRun",
            device="cpu",
            agent_mapping="halfcheetah_6x1",
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        check_env_specs(env)
        env.close()

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown agent_mapping"):
            MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="bogus_mapping",
                config_overrides=_JAX_CONFIG,
            )

    def test_mismatch_env_raises(self):
        # ant_4x2 expects action_size=8; CheetahRun has action_size=6 — mismatch
        with pytest.raises(ValueError):
            MujocoPlaygroundEnv(
                "CheetahRun",
                device="cpu",
                agent_mapping="ant_4x2",
                config_overrides=_JAX_CONFIG,
            )

    def test_object_mapping_still_works(self):
        mapping = KNOWN_MARL_MAPPINGS["halfcheetah_6x1"]
        env = MujocoPlaygroundEnv(
            "CheetahRun",
            device="cpu",
            agent_mapping=mapping,
            config_overrides=_JAX_CONFIG,
        )
        env.set_seed(0)
        td = env.reset()
        assert "agent_0" in td.keys()
        env.close()
