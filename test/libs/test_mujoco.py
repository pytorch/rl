# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the MuJoCo custom envs (humanoid / ant / walker / hopper /
satellite) across the three physics backends."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tensordict import TensorDict
from torchrl.envs import (
    AntEnv,
    CubeBowlEnv,
    Compose,
    HopperEnv,
    HumanoidEnv,
    MujocoEnv,
    MultiAction,
    ParallelEnv,
    RobotAction,
    SatelliteEnv,
    SerialEnv,
    TransformedEnv,
    URScriptPrimitiveTransform,
    Walker2dEnv,
)
from torchrl.envs.custom.mujoco._backends import (
    _has_jax,
    _has_mjx,
    _has_mujoco,
    _has_mujoco_torch,
)
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
    random_unit_quat,
)
from torchrl.envs.utils import check_env_specs, step_mdp

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
        # The observation is exposed as named sub-keys so a
        # CatTensors transform can pack the dynamics-relevant ones into
        # a single policy input while keeping ``manipulability``
        # available for logging.
        obs_spec = env.observation_spec
        assert obs_spec["quat_err"].shape == torch.Size([n, 3])
        assert obs_spec["bus_omega"].shape == torch.Size([n, 3])
        assert obs_spec["gimbal_angles"].shape == torch.Size([n, 2 * num_cmgs])
        assert obs_spec["gimbal_rates"].shape == torch.Size([n, num_cmgs])
        assert obs_spec["manipulability"].shape == torch.Size([n, 1])

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

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_satellite_reward_guard_reports_nonfinite_component(self, backend):
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        # The reward guard is off by default on the hot path (it forces
        # a CUDA->CPU sync). Enable it explicitly for this assertion.
        env._debug_finite = True
        env.reset()
        state = env._state_td()
        action = torch.zeros(env.action_spec.shape, dtype=env.dtype, device=env.device)
        action[0, 0] = torch.finfo(env.dtype).max
        with pytest.raises(RuntimeError, match="reward/control_cost"):
            env._compute_reward(state, action, state)

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_satellite_action_changes_gimbal_state(self, backend):
        n = 1 if backend == "mujoco" else 2
        env_zero = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        env_one = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        td_zero = env_zero.reset()
        td_one = env_one.reset()
        zero_action = torch.zeros(
            env_zero.action_spec.shape, dtype=env_zero.dtype, device=env_zero.device
        )
        one_action = torch.ones(
            env_one.action_spec.shape, dtype=env_one.dtype, device=env_one.device
        )

        td_zero = env_zero.step(td_zero.set("action", zero_action))["next"]
        td_one = env_one.step(td_one.set("action", one_action))["next"]

        assert not torch.allclose(td_zero["gimbal_angles"], td_one["gimbal_angles"])

    def test_quat_log_uses_short_arc(self):
        q = random_unit_quat((1024,), generator=torch.Generator().manual_seed(0))
        log_q = quat_log(q)
        log_neg_q = quat_log(-q)
        assert torch.allclose(log_q, log_neg_q, atol=1e-5, rtol=1e-5)
        assert log_q.norm(dim=-1).max() <= torch.pi + 1e-5

    def test_cmg_geometry_helpers_match_xml_assets(self):
        """The ``(gimbal_axes, rotor_axes_ref)`` helpers must match the
        joint ``axis="..."`` attributes in the satellite XML assets,
        column by column. A mismatch (labels swapped, signs flipped,
        order shuffled) produces a Jacobian that's correct at
        ``theta=0`` -- where the manipulability is sign-invariant --
        but quietly diverges from physics once the gimbals leave the
        nominal posture. Pinned here so the reward signal can't drift.
        """
        import re
        from pathlib import Path

        assets_dir = (
            Path(__import__("torchrl").envs.custom.mujoco.__file__).parent / "assets"
        )

        def parse_axes(
            xml_path: Path, n_cmgs: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            text = xml_path.read_text()
            g_cols, r_cols = [], []
            for i in range(1, n_cmgs + 1):
                m_g = re.search(rf'<joint\s+name="gimbal{i}"[^>]*axis="([^"]+)"', text)
                m_r = re.search(rf'<joint\s+name="rotor{i}"[^>]*axis="([^"]+)"', text)
                assert m_g and m_r, f"missing CMG{i} joints in {xml_path.name}"
                g_cols.append([float(s) for s in m_g.group(1).split()])
                r_cols.append([float(s) for s in m_r.group(1).split()])
            return (
                torch.tensor(g_cols, dtype=torch.float64).T,
                torch.tensor(r_cols, dtype=torch.float64).T,
            )

        g_xml4, r_xml4 = parse_axes(assets_dir / "satellite_large.xml", 4)
        g_h4, r_h4 = pyramid_4cmg_geometry(dtype=torch.float64)
        torch.testing.assert_close(g_h4, g_xml4, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r_h4, r_xml4, rtol=1e-3, atol=1e-3)

        g_xml6, r_xml6 = parse_axes(assets_dir / "satellite_small.xml", 6)
        g_h6, r_h6 = orthogonal_6cmg_geometry(dtype=torch.float64)
        torch.testing.assert_close(g_h6, g_xml6, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r_h6, r_xml6, rtol=1e-3, atol=1e-3)

    def test_cmg_jacobian_at_pi_over_4_matches_reference(self):
        """Independent reference computation of the CMG Jacobian at
        non-trivial gimbal angles. Walks each gimbal to ``pi/4`` (well
        past where the existing torque-direction test exercises) and
        rebuilds the Jacobian from first principles to confirm the
        helper geometry, the Rodrigues rotation, and the cross-product
        sign all agree.
        """
        import math as _math

        g, r0 = pyramid_4cmg_geometry(dtype=torch.float64)
        theta = torch.full((1, 4), _math.pi / 4, dtype=torch.float64)
        h = 100.0
        jac = cmg_jacobian(theta, g, r0, h).squeeze(0)

        ref = torch.empty(3, 4, dtype=torch.float64)
        c, s = _math.cos(_math.pi / 4), _math.sin(_math.pi / 4)
        for i in range(4):
            gi, ri0 = g[:, i], r0[:, i]
            # Rodrigues: r(theta) = c*r0 + s*(g x r0) + (1-c)*(g.r0)*g.
            g_x_r = torch.linalg.cross(gi, ri0)
            g_dot_r = (gi * ri0).sum()
            ri = c * ri0 + s * g_x_r + (1.0 - c) * g_dot_r * gi
            ref[:, i] = h * torch.linalg.cross(gi, ri)
        torch.testing.assert_close(jac, ref, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_satellite_gimbal_observation_is_periodic(self, backend):
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        td = env.rollout(1000)
        gimbal_obs = td["next", "gimbal_angles"]
        assert torch.isfinite(gimbal_obs).all()
        assert (gimbal_obs.abs() <= 1.0 + 1e-6).all()

    # ------------------------------------------------------------------
    # Satellite physics-correctness: specific (state, action) -> (next
    # state, reward) transitions verified against analytical
    # predictions. These tests catch bugs that pure spec / finite-value
    # tests miss (e.g. a wrong gimbal index in the obs builder, an
    # inverted torque sign, an off-by-one in the reset override).
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sat(backend: str, n: int = 2, **kwargs) -> SatelliteEnv:
        if backend == "mujoco":
            n = 1
        return SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend, **kwargs)

    @staticmethod
    def _bus_quat(env: SatelliteEnv) -> torch.Tensor:
        return env._backend.qpos[..., 3:7].to(env.dtype).clone()

    @staticmethod
    def _bus_omega(env: SatelliteEnv) -> torch.Tensor:
        return env._backend.qvel[..., 3:6].to(env.dtype).clone()

    @staticmethod
    def _step_with_action(
        env: SatelliteEnv, action: torch.Tensor, n_steps: int
    ) -> None:
        """Drive the env for ``n_steps`` substeps with a fixed ``action``."""
        td = env.reset() if not getattr(env, "_was_reset", False) else None
        if td is None:
            # Re-read current state into a fresh td (without resetting).
            td = TensorDict(
                {"action": action}, batch_size=env.batch_size, device=env.device
            )
        else:
            td.set("action", action)
        for _ in range(n_steps):
            td = env.step(td)
            td = td["next"].select(*env.observation_spec.keys())
            td.set("action", action)

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_zero_action_preserves_orientation(self, backend):
        """Zero gimbal command + symmetric pyramid CMG cluster (sum of
        rotor moments == 0 at theta=0) means the bus has zero net
        torque applied to it. Roll out 200 steps with zero action and
        confirm the bus quaternion drifts < 1 deg from its initial
        attitude.
        """
        env = self._make_sat(backend, n=2)
        # Use a non-trivial init_bus_quat so we'd notice if the env
        # were silently re-initialising every step.
        init_q = torch.tensor(
            [[0.7071, 0.0, 0.7071, 0.0], [0.7071, 0.7071, 0.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        env.reset(TensorDict({"init_bus_quat": init_q}, batch_size=env.batch_size))
        bus0 = self._bus_quat(env)

        zero_action = torch.zeros(
            env.action_spec.shape, dtype=env.dtype, device=env.device
        )
        td = TensorDict({"action": zero_action}, batch_size=env.batch_size)
        for _ in range(200):
            td = env.step(td)
            td = td["next"].select(*env.observation_spec.keys())
            td.set("action", zero_action)

        bus1 = self._bus_quat(env)
        # Compare via shortest-arc angle: cos(angle/2) = |<q0, q1>|.
        cos_half = (bus0 * bus1).sum(dim=-1).abs().clamp(-1.0, 1.0)
        angle_deg = (2.0 * torch.acos(cos_half)).rad2deg()
        assert angle_deg.max().item() < 1.0, (
            f"Bus drifted {angle_deg.tolist()} deg under zero action; "
            "expected < 1 deg from rotor-induced numerical noise alone."
        )
        # Bus angular velocity should also stay near zero.
        omega = self._bus_omega(env).norm(dim=-1)
        assert omega.max().item() < 0.05, (
            f"Bus omega = {omega.tolist()} rad/s under zero action; "
            "the satellite should be inertially still."
        )

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_init_bus_quat_is_honored(self, backend):
        """``reset({"init_bus_quat": q})`` must place ``qpos[..., 3:7]``
        at ``q`` (post-normalization) and propagate to the
        ``quat_err`` observation according to ``q_err = q^-1 * target``.
        """
        env = self._make_sat(backend, n=2)
        init_q = torch.tensor(
            [[0.7071, 0.0, 0.0, 0.7071], [0.6, 0.0, 0.8, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        target_q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        td = env.reset(
            TensorDict(
                {"init_bus_quat": init_q, "target_quat": target_q},
                batch_size=env.batch_size,
            )
        )
        # Backend stored the (normalized) init quat verbatim.
        init_q_norm = init_q / init_q.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(
            self._bus_quat(env), init_q_norm, rtol=1e-4, atol=1e-4
        )
        # quat_err observation = quat_log(init^-1 * target).
        target_q_norm = target_q / target_q.norm(dim=-1, keepdim=True)
        expected_qerr = quat_log(quat_mul(quat_conj(init_q_norm), target_q_norm))
        torch.testing.assert_close(td["quat_err"], expected_qerr, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_quat_err_is_zero_at_target(self, backend):
        """Setting ``init_bus_quat == target_quat`` makes the
        observation ``quat_err`` start at zero (within reset noise)."""
        env = self._make_sat(backend, n=2)
        q = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        td = env.reset(
            TensorDict(
                {"init_bus_quat": q, "target_quat": q},
                batch_size=env.batch_size,
            )
        )
        # Reset noise is RESET_NOISE_SCALE = 1e-3 on qpos, so quat_err
        # should be small (a few mrad) but not exactly zero.
        assert (
            td["quat_err"].abs().max().item() < 5e-2
        ), f"quat_err = {td['quat_err']} when init == target; expected near zero."

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_180deg_target_gives_pi_attitude_error(self, backend):
        """A 180-deg rotation about an axis is the maximum SO(3)
        distance. ``||quat_log(q_err)||`` should equal ``pi`` (within
        reset noise)."""
        env = self._make_sat(backend, n=2)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        # 180 deg about +x and 180 deg about +y respectively.
        target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        td = env.reset(
            TensorDict(
                {"init_bus_quat": identity, "target_quat": target},
                batch_size=env.batch_size,
            )
        )
        att_err = td["quat_err"].norm(dim=-1)
        assert torch.allclose(
            att_err, torch.full_like(att_err, torch.pi), atol=1e-2
        ), f"||quat_err|| = {att_err.tolist()}, expected ~pi"

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_gimbal_action_torques_bus(self, backend):
        """Driving CMG #1 alone with action=+1 produces a bus torque
        whose direction is **opposite** to the column of
        :func:`cmg_jacobian` for that CMG.

        Why opposite: ``cmg_jacobian`` returns ``h * (g_i x r_i)`` per
        unit gimbal rate. By Newton's third law that's the torque on
        the *rotor* (whose angular momentum is rotating with the
        gimbal), and the *bus* sees the reaction torque
        ``-h * (g_i x r_i)``. The manipulability metric used in the
        reward (``sqrt(det(J J^T))``) is sign-invariant so the env
        reward is unaffected, but the body-frame slewing direction
        flips. This test pins the sign convention so a future
        refactor can't silently invert it.
        """
        env = self._make_sat(backend, n=2, action_scale=3.0)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * env.num_envs,
            dtype=env.dtype,
            device=env.device,
        )
        env.reset(TensorDict({"init_bus_quat": identity}, batch_size=env.batch_size))

        action = torch.zeros(env.action_spec.shape, dtype=env.dtype, device=env.device)
        action[..., 0] = 1.0  # only CMG 1
        td = TensorDict({"action": action}, batch_size=env.batch_size)
        for _ in range(20):
            td = env.step(td)
            td = td["next"].select(*env.observation_spec.keys())
            td.set("action", action)

        omega = self._bus_omega(env)
        # Predicted torque on the BUS = -h * (g_1 x r_1(0)).
        g, r0 = pyramid_4cmg_geometry(device=env.device, dtype=env.dtype)
        jac = cmg_jacobian(
            torch.zeros(1, 4, device=env.device, dtype=env.dtype),
            g,
            r0,
            float(env.ROTOR_SPEED),
        ).squeeze(0)
        bus_torque_dir = -jac[:, 0]  # reaction on the bus
        # Bus omega should align with the (sign of) bus_torque_dir on
        # the axes where the predicted torque is large; the y-axis
        # contribution is structurally zero for CMG 1 in the pyramid.
        omega_signs = torch.sign(omega)
        torque_signs = torch.sign(bus_torque_dir).unsqueeze(0).expand_as(omega)
        big_axes = bus_torque_dir.abs() > 0.1
        match = omega_signs[..., big_axes] == torque_signs[..., big_axes]
        assert match.all(), (
            f"Bus omega = {omega.tolist()} does not match expected reaction "
            f"sign pattern -cmg_jacobian[:, 0] = {bus_torque_dir.tolist()}."
        )
        # Magnitude must actually grow (not just numerical noise).
        assert omega.norm(dim=-1).min().item() > 0.05, (
            f"|bus_omega| = {omega.norm(dim=-1).tolist()} rad/s after 20 "
            "steps of saturated CMG-1 command; expected the bus to slew."
        )

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_reward_at_zero_error_is_baseline(self, backend):
        """With ``init_bus_quat == target_quat`` and zero action, the
        reward should equal the singularity baseline:

            r ~= -singularity_weight / (manip / rotor_h^3)

        which for the nominal pyramid with ``rotor_h = 100`` is
        approximately ``-0.5 / 1.0 = -0.5`` per step (control cost is
        zero, attitude error is at the reset-noise floor).
        """
        env = self._make_sat(backend, n=2, action_scale=3.0, singularity_weight=0.5)
        q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
            dtype=env.dtype,
            device=env.device,
        )
        td = env.reset(
            TensorDict(
                {"init_bus_quat": q, "target_quat": q}, batch_size=env.batch_size
            )
        )
        zero_action = torch.zeros(
            env.action_spec.shape, dtype=env.dtype, device=env.device
        )
        td.set("action", zero_action)
        td = env.step(td)
        reward = td["next", "reward"].squeeze(-1)
        # Allow generous tolerance: reset noise + 1 step of dynamics.
        assert (reward > -0.7).all() and (reward < -0.3).all(), (
            f"Reward at zero attitude error = {reward.tolist()}; "
            "expected ~-0.5 (singularity baseline only)."
        )

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_reward_at_180deg_is_pi_plus_baseline(self, backend):
        """At a 180-deg attitude error with zero action, the reward
        should equal ``-pi - singularity_weight/manip_norm`` per step,
        i.e. about ``-3.64`` for the default weights.
        """
        env = self._make_sat(backend, n=2, action_scale=3.0, singularity_weight=0.5)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * env.num_envs,
            dtype=env.dtype,
            device=env.device,
        )
        target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=env.dtype,
            device=env.device,
        )
        td = env.reset(
            TensorDict(
                {"init_bus_quat": identity, "target_quat": target},
                batch_size=env.batch_size,
            )
        )
        zero_action = torch.zeros(
            env.action_spec.shape, dtype=env.dtype, device=env.device
        )
        td.set("action", zero_action)
        td = env.step(td)
        reward = td["next", "reward"].squeeze(-1)
        expected = -torch.pi - 0.5
        # Bus has barely moved in 1 step, so attitude error stays near pi.
        assert (reward > expected - 0.2).all() and (
            reward < expected + 0.2
        ).all(), (
            f"Reward at 180-deg error = {reward.tolist()}; expected ~{expected:.2f}."
        )

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_observation_matches_state(self, backend):
        """Observation channels are read directly off ``qpos`` /
        ``qvel`` -- not synthesised. After one non-trivial step:

        * ``bus_omega == qvel[..., 3:6]``
        * ``gimbal_rates == qvel[..., gimbal_rate_idx]``
        * ``gimbal_angles == [sin(qpos_gimbals), cos(qpos_gimbals)]``
        """
        env = self._make_sat(backend, n=2, action_scale=3.0)
        env.reset()
        action = torch.full(
            env.action_spec.shape, 0.5, dtype=env.dtype, device=env.device
        )
        td = TensorDict({"action": action}, batch_size=env.batch_size)
        for _ in range(10):
            td = env.step(td)
            td = td["next"].select(*env.observation_spec.keys())
            td.set("action", action)

        qpos = env._backend.qpos.to(env.dtype)
        qvel = env._backend.qvel.to(env.dtype)
        gimbal_idx = [7 + 2 * i for i in range(env.N_GIMBALS)]
        gimbal_rate_idx = [6 + 2 * i for i in range(env.N_GIMBALS)]

        torch.testing.assert_close(
            td["bus_omega"], qvel[..., 3:6], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            td["gimbal_rates"],
            qvel[..., gimbal_rate_idx],
            rtol=1e-5,
            atol=1e-5,
        )
        # gimbal_angles is concat([sin, cos]) over the gimbal qpos.
        gimbals = qpos[..., gimbal_idx]
        expected = torch.cat([gimbals.sin(), gimbals.cos()], dim=-1)
        torch.testing.assert_close(td["gimbal_angles"], expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("backend", _VMAP_BACKENDS)
    def test_satellite_reset_is_reproducible(self, backend):
        """Same ``(init_bus_quat, target_quat)`` and same action sequence
        must produce byte-identical bus quaternion trajectories. This is
        the determinism guarantee that the eval pipeline relies on
        (the :class:`TestSetPrimer` replays the same starts every
        iteration -- if the env is non-deterministic, eval comparisons
        between iterations are meaningless).
        """
        n = 2
        init_q = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [0.7071, 0.7071, 0.0, 0.0]],
        )
        target_q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        )

        def _trajectory(seed: int) -> torch.Tensor:
            env = SatelliteEnv(
                num_cmgs=4,
                num_envs=n,
                seed=seed,
                backend=backend,
                action_scale=3.0,
            )
            env.reset(
                TensorDict(
                    {
                        "init_bus_quat": init_q.to(env.dtype).to(env.device),
                        "target_quat": target_q.to(env.dtype).to(env.device),
                    },
                    batch_size=env.batch_size,
                )
            )
            action = torch.full(
                env.action_spec.shape, 0.3, dtype=env.dtype, device=env.device
            )
            traj = []
            td = TensorDict({"action": action}, batch_size=env.batch_size)
            for _ in range(20):
                td = env.step(td)
                traj.append(self._bus_quat(env))
                td = td["next"].select(*env.observation_spec.keys())
                td.set("action", action)
            return torch.stack(traj, dim=0)

        # Same (init, target, seed, action) -> identical trajectory.
        torch.testing.assert_close(_trajectory(0), _trajectory(0))

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
    def test_pixels_only_drops_observation_key(self, backend):
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(
            num_cmgs=4,
            num_envs=n,
            seed=0,
            backend=backend,
            from_pixels=True,
            pixels_only=True,
            render_width=32,
            render_height=32,
        )
        check_env_specs(env)
        keys = set(env.observation_spec.keys())
        assert keys == {"pixels"}, f"pixels_only must drop 'observation', got {keys}"

    def test_pixels_only_without_from_pixels_raises(self):
        with pytest.raises(ValueError, match="pixels_only"):
            HopperEnv(num_envs=1, seed=0, pixels_only=True)

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_render_method(self, backend):
        """``env.render()`` returns the same shape as the pixel obs."""
        n = 1 if backend == "mujoco" else 2
        env = SatelliteEnv(num_cmgs=4, num_envs=n, seed=0, backend=backend)
        env.reset()
        rgb = env.render(width=24, height=24)
        assert rgb.shape == torch.Size([n, 24, 24, 3])
        assert rgb.dtype == torch.uint8

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_specs_and_rollout(self):
        env = CubeBowlEnv(seed=0, max_episode_steps=3)
        check_env_specs(env)
        assert env.action_spec.shape == torch.Size([1, 7])
        assert env.observation_spec["robot_qpos"].shape == torch.Size([1, 6])
        assert env.observation_spec["gripper_qpos"].shape == torch.Size([1, 2])
        assert env.observation_spec["pinch_quat"].shape == torch.Size([1, 4])
        assert env.observation_spec["gripper_left_pad_pos"].shape == torch.Size([1, 3])
        assert env.observation_spec["cube_pos"].shape == torch.Size([1, 3])
        assert env.observation_spec["bowl_pos"].shape == torch.Size([1, 3])

        td = env.rollout(2)
        assert torch.isfinite(td.get(("next", "reward"))).all()

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_sparse_coordinate_reward(self):
        env = CubeBowlEnv(seed=0, max_episode_steps=3)
        observation = env.reset()
        hold_action = env.low_level_action(observation["robot_qpos"])

        default_transition = env.step(observation.clone().set("action", hold_action))
        torch.testing.assert_close(
            default_transition["next", "reward"],
            torch.zeros(1, 1, dtype=env.dtype),
        )

        observation = env.reset(
            TensorDict({"cube_pos": env._target_pos().clone()}, batch_size=[1])
        )
        transform = env.make_urscript_transform(macro_steps=1)
        expanded_action = transform.action_sequence(
            observation,
            URScriptPrimitiveTransform.WAIT,
            gripper=env.gripper_open_ctrl,
        )
        target_transition = env.step(
            observation.clone().set("action", expanded_action[:, 0])
        )
        torch.testing.assert_close(
            target_transition["next", "reward"],
            torch.ones(1, 1, dtype=env.dtype),
        )
        assert target_transition["next", "success"].all()

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_construction_time_positions(self):
        cube_position = (0.31, -0.11, 0.035)
        bowl_position = (0.24, 0.17, 0.01)
        env = CubeBowlEnv(
            cube_position=cube_position,
            bowl_position=bowl_position,
            seed=0,
        )
        td = env.reset()
        torch.testing.assert_close(
            td["cube_pos"],
            torch.tensor([cube_position], dtype=td["cube_pos"].dtype),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            td["bowl_pos"],
            torch.tensor([[0.24, 0.17, 0.05]], dtype=td["bowl_pos"].dtype),
            atol=1e-6,
            rtol=0.0,
        )

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_macro_helper_api(self):
        env = CubeBowlEnv(seed=0, max_episode_steps=3)
        observation = env.reset()
        transform = env.make_urscript_transform(macro_steps=2)
        target_qpos = env.low_level_action(
            observation["robot_qpos"], gripper=env.gripper_close_ctrl
        )
        primitive = transform.make_primitive(
            observation,
            URScriptPrimitiveTransform.MOVEJ,
            target_qpos=target_qpos,
            gripper=env.gripper_close_ctrl,
        )
        action = transform.action_sequence(primitive)

        assert action.shape == torch.Size([1, 2, 7])
        torch.testing.assert_close(action[:, -1], target_qpos)
        assert env.gripper_cube_distance(observation).shape == torch.Size([1, 1])
        assert env.pose_at(observation["cube_pos"]).shape == torch.Size([1, 7])
        grasp_width = 2 * env.OBJECT_HALF_SIZE - 0.001
        grasp_ctrl = env.gripper_ctrl_for_width(grasp_width)
        assert env.gripper_open_ctrl <= grasp_ctrl <= env.gripper_close_ctrl

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_menagerie_ur5e_when_available(self):
        menagerie_path = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
        if menagerie_path is None or not Path(menagerie_path).exists():
            return

        env = CubeBowlEnv(
            robot_model="menagerie_ur5e",
            menagerie_path=menagerie_path,
            seed=0,
            max_episode_steps=3,
        )
        check_env_specs(env)
        assert env.action_spec.shape == torch.Size([1, 7])
        assert env.observation_spec["robot_qpos"].shape == torch.Size([1, 6])
        assert env.observation_spec["gripper_qpos"].shape == torch.Size([1, 8])
        assert env.observation_spec["pinch_quat"].shape == torch.Size([1, 4])
        assert env.observation_spec["gripper_left_pad_pos"].shape == torch.Size([1, 3])
        assert env.observation_spec["gripper_right_pad_pos"].shape == torch.Size([1, 3])
        assert env._backend.nq == 21
        assert env._backend.nv == 20
        assert env._backend.nu == 7

        observation = env.reset(
            TensorDict({"cube_pos": env._target_pos().clone()}, batch_size=[1])
        )
        hold_action = torch.zeros_like(env.action_spec.rand())
        hold_action[..., :6] = observation["robot_qpos"]
        transition = env.step(observation.clone().set("action", hold_action))
        torch.testing.assert_close(
            transition["next", "reward"],
            torch.ones(1, 1, dtype=env.dtype),
        )
        env.close()

        env = CubeBowlEnv(
            robot_model="menagerie_ur5e",
            menagerie_path=menagerie_path,
            seed=0,
            max_episode_steps=8000,
        )
        observation = env.reset()

        def pose_with_quat(xyz, quat):
            return torch.cat([xyz, quat.expand(*xyz.shape[:-1], 4)], dim=-1)

        def action_from_robot_qpos(robot_qpos, gripper):
            action = torch.zeros(*robot_qpos.shape[:-1], 7, dtype=robot_qpos.dtype)
            action[..., :6] = robot_qpos
            action[..., -1] = float(gripper)
            return action

        def primitive_td(observation, primitive_id, target_pose=None, gripper=0.0):
            if target_pose is None:
                target_pose = torch.zeros(*observation.batch_size, 7)
            td = observation.clone()
            td["primitive_id"] = torch.full((1, 1), primitive_id, dtype=torch.long)
            td["target_pose"] = target_pose
            td["target_qpos"] = action_from_robot_qpos(
                observation["robot_qpos"], gripper
            )
            td["gripper"] = torch.full((1, 1), float(gripper))
            return td

        def gripper_cube_distance(observation):
            cube_pos = observation["cube_pos"]
            half_size = torch.full_like(cube_pos, CubeBowlEnv.OBJECT_HALF_SIZE)

            def pad_to_cube(pad_pos):
                q = (pad_pos - cube_pos).abs() - half_size
                outside = q.clamp_min(0.0).norm(dim=-1, keepdim=True)
                inside = q.max(dim=-1, keepdim=True).values.clamp_max(0.0)
                return outside + inside

            return torch.minimum(
                pad_to_cube(observation["gripper_left_pad_pos"]),
                pad_to_cube(observation["gripper_right_pad_pos"]),
            ).clamp_min(0.0)

        def run_primitive(observation, transform, primitive):
            primitive.update(observation.select(*env.observation_keys))
            expanded = transform.inv(primitive)
            start_cube = observation["cube_pos"].clone()
            min_gripper_distance = torch.full_like(start_cube[..., :1], float("inf"))
            max_reward = torch.zeros_like(start_cube[..., :1])
            last_reward = torch.zeros_like(start_cube[..., :1])
            for action in expanded["action"][0]:
                transition = env.step(
                    observation.clone().set("action", action.view(1, 7))
                )
                observation = step_mdp(transition)
                min_gripper_distance = torch.minimum(
                    min_gripper_distance, gripper_cube_distance(observation)
                )
                last_reward = transition["next", "reward"]
                max_reward = torch.maximum(max_reward, last_reward)
            return observation, min_gripper_distance, max_reward, last_reward

        def solver(target_pose, start_action):
            return env._cartesian_pose_to_joint_target(
                target_pose,
                start_action,
                iterations=220,
                orientation_weight=1.0,
                step_size=0.7,
                damping=1e-4,
            )

        def make_transform(macro_steps, settle_steps=0):
            return URScriptPrimitiveTransform(
                macro_steps=macro_steps,
                settle_steps=settle_steps,
                cartesian_solver=solver,
                open_gripper_ctrl=0.0,
                close_gripper_ctrl=255.0,
            )

        approach_transform = make_transform(180, settle_steps=60)
        close_transform = make_transform(160, settle_steps=80)
        lift_transform = make_transform(120, settle_steps=60)
        carry_transform = make_transform(80, settle_steps=20)
        drop_transform = make_transform(100, settle_steps=40)
        open_transform = make_transform(100, settle_steps=20)
        home_transform = make_transform(250, settle_steps=800)
        gripper_open = 0.0
        gripper_close = 255.0
        initial_robot_qpos = observation["robot_qpos"].clone()
        gripper_quat = observation["pinch_quat"].clone()
        grasp_distance = torch.full_like(observation["cube_pos"][..., :1], float("inf"))
        cube_motion_while_closed = torch.zeros_like(grasp_distance)
        cube_lift_while_closed = torch.zeros_like(grasp_distance)
        max_reward = torch.zeros_like(grasp_distance)
        last_reward = torch.zeros_like(grasp_distance)

        def update_closed_motion(reference_cube):
            nonlocal cube_motion_while_closed, cube_lift_while_closed
            cube_motion_while_closed = torch.maximum(
                cube_motion_while_closed,
                (observation["cube_pos"] - reference_cube).norm(
                    dim=-1, keepdim=True
                ),
            )
            cube_lift_while_closed = torch.maximum(
                cube_lift_while_closed,
                observation["cube_pos"][..., 2:3] - reference_cube[..., 2:3],
            )

        for _ in range(20):
            transition = env.step(
                observation.clone().set(
                    "action", action_from_robot_qpos(observation["robot_qpos"], 0.0)
                )
            )
            observation = step_mdp(transition)
            last_reward = transition["next", "reward"]
            max_reward = torch.maximum(max_reward, last_reward)

        observation, _, reward, last_reward = run_primitive(
            observation,
            open_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.OPEN_GRIPPER,
                gripper=gripper_open,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)

        cube = observation["cube_pos"].clone()
        bowl = observation["bowl_pos"].clone()
        above_cube = pose_with_quat(
            cube + torch.tensor([[0.0, 0.0, 0.18]], dtype=cube.dtype),
            gripper_quat,
        )
        observation, _, reward, last_reward = run_primitive(
            observation,
            approach_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.MOVEL,
                target_pose=above_cube,
                gripper=gripper_open,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)

        grasp_cube = pose_with_quat(
            cube + torch.tensor([[0.0, 0.0, -0.005]], dtype=cube.dtype),
            gripper_quat,
        )
        observation, _, reward, last_reward = run_primitive(
            observation,
            approach_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.MOVEL,
                target_pose=grasp_cube,
                gripper=gripper_open,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)

        observation, distance, reward, last_reward = run_primitive(
            observation,
            close_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.CLOSE_GRIPPER,
                gripper=gripper_close,
            ),
        )
        grasp_distance = torch.minimum(grasp_distance, distance)
        max_reward = torch.maximum(max_reward, reward)
        closed_reference_cube = observation["cube_pos"].clone()
        update_closed_motion(closed_reference_cube)

        cube = observation["cube_pos"].clone()
        pinch_to_cube = observation["pinch_pos"].clone() - cube
        lift_cube = pose_with_quat(
            cube
            + pinch_to_cube
            + torch.tensor([[0.0, 0.0, 0.20]], dtype=cube.dtype),
            gripper_quat,
        )
        observation, _, reward, last_reward = run_primitive(
            observation,
            lift_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.MOVEL,
                target_pose=lift_cube,
                gripper=gripper_close,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)
        update_closed_motion(closed_reference_cube)

        start_y = observation["cube_pos"][..., 1:2].clone()
        target_y = bowl[..., 1:2]
        for waypoint in range(1, 5):
            alpha = float(waypoint) / 4.0
            desired_cube = torch.cat(
                [
                    bowl[..., :1],
                    start_y + alpha * (target_y - start_y),
                    torch.full_like(bowl[..., 2:3], 0.24),
                ],
                dim=-1,
            )
            cube = observation["cube_pos"].clone()
            pinch_to_cube = observation["pinch_pos"].clone() - cube
            observation, _, reward, last_reward = run_primitive(
                observation,
                carry_transform,
                primitive_td(
                    observation,
                    URScriptPrimitiveTransform.MOVEL,
                    target_pose=pose_with_quat(
                        desired_cube + pinch_to_cube, gripper_quat
                    ),
                    gripper=gripper_close,
                ),
            )
            max_reward = torch.maximum(max_reward, reward)
            update_closed_motion(closed_reference_cube)

        cube = observation["cube_pos"].clone()
        pinch_to_cube = observation["pinch_pos"].clone() - cube
        drop_cube = torch.cat(
            [
                bowl[..., :1],
                bowl[..., 1:2],
                torch.full_like(bowl[..., 2:3], 0.13),
            ],
            dim=-1,
        )
        observation, _, reward, last_reward = run_primitive(
            observation,
            drop_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.MOVEL,
                target_pose=pose_with_quat(drop_cube + pinch_to_cube, gripper_quat),
                gripper=gripper_close,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)
        update_closed_motion(closed_reference_cube)

        observation, _, reward, last_reward = run_primitive(
            observation,
            open_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.OPEN_GRIPPER,
                gripper=gripper_open,
            ),
        )
        max_reward = torch.maximum(max_reward, reward)

        for _ in range(240):
            transition = env.step(
                observation.clone().set(
                    "action", action_from_robot_qpos(observation["robot_qpos"], 0.0)
                )
            )
            observation = step_mdp(transition)
            last_reward = transition["next", "reward"]
            max_reward = torch.maximum(max_reward, last_reward)

        home_target = action_from_robot_qpos(initial_robot_qpos, gripper_open)
        observation, _, reward, last_reward = run_primitive(
            observation,
            home_transform,
            primitive_td(
                observation,
                URScriptPrimitiveTransform.MOVEJ,
                gripper=gripper_open,
            ).set("target_qpos", home_target),
        )
        max_reward = torch.maximum(max_reward, reward)
        for _ in range(800):
            transition = env.step(observation.clone().set("action", home_target))
            observation = step_mdp(transition)
            last_reward = transition["next", "reward"]
            max_reward = torch.maximum(max_reward, last_reward)

        robot_home_error = (observation["robot_qpos"] - initial_robot_qpos).norm(
            dim=-1, keepdim=True
        )
        assert grasp_distance.item() <= 0.025
        assert cube_motion_while_closed.item() >= 0.05
        assert cube_lift_while_closed.item() >= 0.08
        assert robot_home_error.item() <= 0.03
        torch.testing.assert_close(max_reward, torch.ones_like(max_reward))
        torch.testing.assert_close(last_reward, torch.ones_like(last_reward))
        assert observation["success"].all()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="CubeBowlEnv uses MuJoCo C bindings.")
    def test_cube_bowl_urscript_macro_smoke(self):
        base = CubeBowlEnv(seed=0, max_episode_steps=20)
        env = TransformedEnv(
            base,
            URScriptPrimitiveTransform(
                macro_steps=2,
                execute=True,
                open_gripper_ctrl=0.0,
                close_gripper_ctrl=0.038,
            ),
        )
        td = env.reset()
        td["action"] = RobotAction.reach_pose(
            position=td["cube_pos"] + torch.tensor([[0.0, 0.0, 0.08]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            gripper="open",
            steps=2,
        )

        out = env.step(td)
        assert torch.isfinite(out.get(("next", "reward"))).all()

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


if __name__ == "__main__":
    pytest.main([__file__])
