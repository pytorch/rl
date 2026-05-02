# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the MuJoCo custom envs (humanoid / ant / walker / hopper /
satellite) across the three physics backends."""

from __future__ import annotations

import pytest
import torch
from torchrl.envs import (
    AntEnv,
    HopperEnv,
    HumanoidEnv,
    MujocoEnv,
    ParallelEnv,
    SatelliteEnv,
    SerialEnv,
    Walker2dEnv,
)
from torchrl.envs.custom.mujoco._backends import (
    _has_jax,
    _has_mjx,
    _has_mujoco,
    _has_mujoco_torch,
)
from torchrl.envs.utils import check_env_specs

_AVAILABLE_BACKENDS: list[str] = []
if _has_mujoco_torch:
    _AVAILABLE_BACKENDS.append("mujoco-torch")
if _has_mjx and _has_jax:
    _AVAILABLE_BACKENDS.append("mjx")
if _has_mujoco:
    _AVAILABLE_BACKENDS.append("mujoco")

_VMAP_BACKENDS = [b for b in _AVAILABLE_BACKENDS if b in ("mujoco-torch", "mjx")]
_LOCOMOTION_ENVS = [HumanoidEnv, AntEnv, Walker2dEnv, HopperEnv]


@pytest.mark.skipif(
    not _AVAILABLE_BACKENDS,
    reason="No MuJoCo backend installed (mujoco-torch / mjx / mujoco).",
)
class TestMujoco:
    # ------------------------------------------------------------------
    # Spec / rollout coverage across all available backends.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("cls", _LOCOMOTION_ENVS)
    def test_locomotion_env_specs(self, cls, backend):
        if backend == "mujoco":
            # Single-env semantics for C-bindings backend.
            env = cls(num_envs=1, seed=0, backend=backend)
        else:
            env = cls(num_envs=2, seed=0, backend=backend)
        check_env_specs(env)
        assert env.observation_spec["observation"].shape[0] == env.batch_size[0]
        assert env.action_spec.shape[0] == env.batch_size[0]

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("cls", _LOCOMOTION_ENVS)
    def test_locomotion_rollout(self, cls, backend):
        n = 1 if backend == "mujoco" else 2
        env = cls(num_envs=n, seed=0, backend=backend)
        td = env.rollout(5)
        reward = td.get(("next", "reward"))
        assert reward.shape[-1] == 1
        assert torch.isfinite(reward).all()

    # ------------------------------------------------------------------
    # Satellite: spec, dim sanity, finite singularity reward.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("num_cmgs", [4, 6])
    def test_satellite_specs(self, num_cmgs, backend):
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=num_cmgs, num_envs=n, seed=0, backend=backend)
        check_env_specs(env)
        # action_spec dim = N_GIMBALS, not nu (rotors are held constant).
        assert env.action_spec.shape == torch.Size([n, num_cmgs])
        # obs = quat_err(3) + bus_omega(3) + gimbal_angles(N) + gimbal_rates(N)
        assert env.observation_spec["observation"].shape == torch.Size(
            [n, 6 + 2 * num_cmgs]
        )

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("num_cmgs", [4, 6])
    def test_satellite_reward_finite(self, num_cmgs, backend):
        """Singularity term must never explode: ``+eps`` in ``manipulability``
        guards ``1/sqrt(det(JJ^T))`` against rank-deficient configurations.
        """
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=num_cmgs, num_envs=n, seed=0, backend=backend)
        td = env.rollout(50)
        assert torch.isfinite(td.get(("next", "reward"))).all()

    # ------------------------------------------------------------------
    # Backend dispatch: vmap backends reject num_workers / parallel;
    # the C-bindings backend composes via ParallelEnv / SerialEnv.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_vmap_backend_rejects_num_workers(self, backend):
        with pytest.raises(ValueError, match="num_envs"):
            HopperEnv(backend=backend, num_workers=2, seed=0)

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_vmap_backend_rejects_parallel(self, backend):
        with pytest.raises(ValueError, match="parallel"):
            HopperEnv(backend=backend, parallel=True, seed=0)

    @pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")
    def test_mujoco_backend_num_envs_aliases_num_workers(self):
        """For the C-bindings backend, ``num_envs`` and ``num_workers``
        are aliases; both produce a :class:`ParallelEnv` of N copies."""
        env_a = HopperEnv(backend="mujoco", num_envs=2, seed=0)
        env_b = HopperEnv(backend="mujoco", num_workers=2, seed=0)
        # Lazy ParallelEnvs -- don't start workers, just shape-check.
        assert isinstance(env_a, ParallelEnv)
        assert isinstance(env_b, ParallelEnv)
        assert env_a.batch_size == env_b.batch_size

    @pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")
    def test_mujoco_backend_rejects_both_envs_and_workers(self):
        with pytest.raises(ValueError, match="aliases"):
            HopperEnv(backend="mujoco", num_envs=2, num_workers=2, seed=0)

    @pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")
    def test_mujoco_backend_serial_dispatch(self):
        env = HopperEnv(backend="mujoco", num_envs=2, parallel=False, seed=0)
        assert isinstance(env, SerialEnv)
        td = env.rollout(3)
        assert torch.isfinite(td.get(("next", "reward"))).all()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")
    def test_mujoco_backend_parallel_rollout(self):
        env = HopperEnv(backend="mujoco", num_envs=2, seed=0)
        assert isinstance(env, ParallelEnv)
        td = env.rollout(3)
        assert torch.isfinite(td.get(("next", "reward"))).all()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="mujoco not installed")
    def test_mujoco_backend_single_env_passthrough(self):
        """``backend='mujoco'`` with N=1 returns a bare ``HopperEnv``,
        not a ``ParallelEnv`` wrapper."""
        env = HopperEnv(backend="mujoco", num_envs=1, seed=0)
        assert isinstance(env, HopperEnv)
        assert env.batch_size == torch.Size([1])

    # ------------------------------------------------------------------
    # Compile / unknown-backend / custom XML.
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not _has_mujoco_torch, reason="mujoco-torch not installed")
    def test_torch_backend_compile_smoke(self):
        """``compile_step=True`` must not raise on the default backend."""
        env = HopperEnv(num_envs=2, seed=0, compile_step=True)
        td = env.rollout(3)
        assert torch.isfinite(td.get(("next", "reward"))).all()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            HopperEnv(num_envs=1, seed=0, backend="not-a-backend")

    # ------------------------------------------------------------------
    # Rendering / from_pixels.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_from_pixels_spec_and_rollout(self, backend):
        """``from_pixels=True`` adds a ``pixels`` key with a ``uint8`` spec
        of shape ``(num_envs, H, W, 3)`` and values in ``[0, 255]``.

        Uses :class:`SatelliteEnv` because locomotion envs terminate
        early under random actions, masking rollout shape assertions.
        """
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(
            num_cmgs=4,
            num_envs=n,
            seed=0,
            backend=backend,
            from_pixels=True,
            render_width=32,
            render_height=32,
        )
        check_env_specs(env)
        assert env.observation_spec["pixels"].shape == torch.Size([n, 32, 32, 3])
        assert env.observation_spec["pixels"].dtype == torch.uint8

        td = env.rollout(2)
        pixels = td.get(("next", "pixels"))
        assert pixels.shape == torch.Size([n, 2, 32, 32, 3])
        assert pixels.dtype == torch.uint8
        assert (pixels >= 0).all() and (pixels <= 255).all()
        # Real RGB content -- not all-zero, not saturated.
        assert pixels.float().std().item() > 0

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_pixel_only_drops_observation_key(self, backend):
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(
            num_cmgs=4,
            num_envs=n,
            seed=0,
            backend=backend,
            from_pixels=True,
            pixel_only=True,
            render_width=32,
            render_height=32,
        )
        check_env_specs(env)
        keys = set(env.observation_spec.keys())
        assert keys == {"pixels"}, f"pixel_only must drop 'observation', got {keys}"

    def test_pixel_only_without_from_pixels_raises(self):
        with pytest.raises(ValueError, match="pixel_only"):
            HopperEnv(num_envs=1, seed=0, pixel_only=True)

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_render_method(self, backend):
        """``env.render()`` returns the same shape as the pixel obs."""
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        env.reset()
        rgb = env.render(width=24, height=24)
        assert rgb.shape == torch.Size([n, 24, 24, 3])
        assert rgb.dtype == torch.uint8

    def test_xml_path_kwarg_overrides_class_attr(self, tmp_path):
        """Custom ``xml_path=`` overrides the class-level :attr:`XML_PATH`."""
        backend = _AVAILABLE_BACKENDS[0]
        # A trivial single-hinge model -- pure-XML, no external mesh deps.
        xml = (
            "<mujoco><worldbody>"
            "<body name='b' pos='0 0 1'>"
            "<joint name='j' type='hinge'/>"
            "<geom size='0.1' mass='1'/>"
            "</body></worldbody>"
            "<actuator><motor name='a' joint='j' gear='1' ctrlrange='-1 1'/></actuator>"
            "</mujoco>"
        )
        path = tmp_path / "tiny.xml"
        path.write_text(xml)

        class TinyEnv(MujocoEnv):
            FRAME_SKIP = 2

            def _compute_reward(self, state, action, next_state):
                return torch.zeros(self.num_envs, 1, device=self.device)

            def _compute_done(self, state, next_state):
                return torch.zeros(
                    self.num_envs, 1, dtype=torch.bool, device=self.device
                )

        env = TinyEnv(xml_path=str(path), backend=backend, num_envs=1, seed=0)
        check_env_specs(env)
        td = env.rollout(3)
        assert td.shape == torch.Size([1, 3])
