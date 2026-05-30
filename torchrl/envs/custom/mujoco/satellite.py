# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Satellite attitude-control task with Control Moment Gyros (CMGs).

The agent commands gimbal rates of either 4 (pyramid) or 6 (orthogonal)
CMGs to slew the satellite to a target attitude sampled uniformly on
``SO(3)`` at reset, while *avoiding internal singularities* of the CMG
cluster. The singularity penalty uses the manipulability metric of the
gimbal Jacobian: ``m(J) = sqrt(det(J J^T) + eps)``.

Reward
    ``r = - ||q_err|| - lambda_s / m(J) - lambda_u * ||a||^2``

Termination
    Never (the satellite cannot crash). Use a
    :class:`~torchrl.envs.transforms.TimeLimit` transform for truncation.
"""

from __future__ import annotations

import os
import re
from typing import Literal

import torch
from tensordict import TensorDictBase
from torchrl._utils import is_compiling
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    manipulability,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
    random_unit_quat,
)
from torchrl.envs.custom.mujoco.base import MujocoEnv


class SatelliteEnv(MujocoEnv):
    r"""Attitude-control task with 4 or 6 CMGs.

    Args:
        num_cmgs: ``4`` (pyramid) or ``6`` (orthogonal cluster).
        attitude_weight: weight on ``-||q_err||``. Default ``1.0``.
        singularity_weight: weight on ``-1 / m_norm(J)`` where
            ``m_norm(J) = m(J) / rotor_h^3``. The normalization makes
            the metric rotor-speed invariant; default ``0.5`` so the
            penalty is comparable in magnitude to the attitude term
            for near-singular postures and effectively zero for the
            nominal cluster.
        ctrl_cost_weight: weight on ``||a||^2``. Default ``0.01``.
        omega_weight: weight on ``-||bus_omega||^2``. Penalizes body
            angular velocity so the policy is incentivised to *stop*
            at the target attitude rather than swing through it.
            Default ``0.1``: with omega magnitudes around 1 rad/s
            during a slew this contributes ``~-0.1`` per step,
            comparable to the singularity term.
        action_scale: rescaling factor applied inside
            :meth:`_prepare_ctrl`: the agent emits actions in
            ``[-1, 1]`` and the simulator sees a commanded gimbal
            rate of ``action_scale * action`` rad/s. Default ``3.0``,
            chosen to keep saturated 180-deg slews physically
            reachable inside ~1500-step episodes.
        rotor_h: scalar rotor angular momentum used in the Jacobian.
            Defaults to :attr:`ROTOR_SPEED` (proxy unit, since the
            absolute scale only changes the singularity threshold).
        \*\*kwargs: forwarded to :class:`MujocoEnv`.

    Example:
        >>> from torchrl.envs import SatelliteEnv  # doctest: +SKIP
        >>> env = SatelliteEnv(num_cmgs=4, num_envs=4)  # doctest: +SKIP
        >>> td = env.rollout(50)  # doctest: +SKIP

    Reference:
        Wie, Bong (2008). "Space Vehicle Dynamics and Control" --
        chapter on CMG steering laws and singularity escape.
    """

    FRAME_SKIP = 10
    RESET_NOISE_SCALE = 0.001
    ROTOR_SPEED_4 = 100.0
    ROTOR_SPEED_6 = 200.0
    RENDER_BACKGROUND = (0.0, 0.0, 0.05)

    def __init__(
        self,
        *,
        num_cmgs: Literal[4, 6] = 4,
        attitude_weight: float = 1.0,
        singularity_weight: float = 0.5,
        singularity_clamp_min: float = 1e-6,
        singularity_mode: Literal["inverse", "exp"] = "inverse",
        singularity_exp_k: float = 5.0,
        ctrl_cost_weight: float = 0.01,
        omega_weight: float = 0.1,
        action_scale: float = 3.0,
        rotor_h: float | None = None,
        **kwargs,
    ) -> None:
        if num_cmgs == 4:
            self.N_GIMBALS = 4
            self.ROTOR_SPEED = self.ROTOR_SPEED_4
            self.XML_PATH = "satellite_large.xml"
        elif num_cmgs == 6:
            self.N_GIMBALS = 6
            self.ROTOR_SPEED = self.ROTOR_SPEED_6
            self.XML_PATH = "satellite_small.xml"
        else:
            raise ValueError(f"num_cmgs must be 4 or 6, got {num_cmgs}")
        self.num_cmgs = num_cmgs
        self.attitude_weight = float(attitude_weight)
        self.singularity_weight = float(singularity_weight)
        self.singularity_clamp_min = float(singularity_clamp_min)
        self.singularity_mode = str(singularity_mode)
        self.singularity_exp_k = float(singularity_exp_k)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.omega_weight = float(omega_weight)
        self.action_scale = float(action_scale)
        self._rotor_h = float(rotor_h) if rotor_h is not None else self.ROTOR_SPEED

        super().__init__(**kwargs)

        # CMG geometry, cached on the env's device / dtype.
        if num_cmgs == 4:
            g, r0 = pyramid_4cmg_geometry(device=self.device, dtype=self.dtype)
        else:
            g, r0 = orthogonal_6cmg_geometry(device=self.device, dtype=self.dtype)
        self._gimbal_axes = g
        self._rotor_axes_ref = r0

        # Per-env target attitude, populated on reset.
        self._target_quat = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=self.dtype
        )
        self._target_quat[:, 0] = 1.0

        # Cached per-step manipulability so the obs builder can surface
        # it without recomputing the Jacobian. Filled by
        # :meth:`_compute_reward`; on the first reset, defaults to a
        # large value (a non-singular nominal posture).
        self._last_manip = torch.full(
            (self.num_envs, 1), 1.0, device=self.device, dtype=self.dtype
        )

        # Per-step ``isfinite`` checks are off by default. They force a
        # CUDA -> CPU sync on each call (via ``bool(all_finite)``), which
        # under hot-path collection erodes throughput. Set
        # ``TORCHRL_SATELLITE_DEBUG=1`` (or pass ``debug_finite=True`` to
        # the constructor) to re-enable them; the test suite uses this
        # to validate reward guards without changing default semantics.
        env_flag = os.environ.get("TORCHRL_SATELLITE_DEBUG", "0")
        self._debug_finite = env_flag not in ("0", "", "false", "False")

    def _assert_finite(self, name: str, value: torch.Tensor) -> None:
        if is_compiling():
            # ``torch._assert`` is a compile-time hint; cheap inside a
            # compiled graph and skipped at runtime entirely.
            torch._assert(
                torch.isfinite(value).all(),
                f"SatelliteEnv produced non-finite values in {name}.",
            )
            return
        if not self._debug_finite:
            # Default eager path: no host-device sync, no overhead.
            return
        finite = torch.isfinite(value)
        if bool(finite.all()):
            return

        bad = ~finite
        nan_count = torch.isnan(value).sum().item()
        inf_count = torch.isinf(value).sum().item()
        if value.ndim > 0 and value.shape[0] == self.num_envs:
            bad_rows = bad.reshape(self.num_envs, -1).any(dim=-1).nonzero().flatten()
            row_list = bad_rows[:8].detach().cpu().tolist()
            sample = value.index_select(0, bad_rows[:3]).detach().cpu()
        else:
            row_list = []
            sample = value[bad][:8].detach().cpu()
        raise RuntimeError(
            f"SatelliteEnv produced non-finite values in {name}: "
            f"nan_count={nan_count}, inf_count={inf_count}, "
            f"bad_rows={row_list}, sample={sample}."
        )

    # ------------------------------------------------------------------
    # XML patching: no ground floor in space.
    # ------------------------------------------------------------------

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        xml = re.sub(r"<camera\b[^/]*/>\s*", "", xml)
        xml = re.sub(r"<light\b[^/]*/>\s*", "", xml)
        camera = (
            '<camera name="side" pos="3 -3 2" '
            'xyaxes="0.707 0.707 0 -0.302 0.302 0.905" fovy="60"/>'
        )
        light = (
            '<light name="top" pos="0 0 4" dir="0 0 -1" '
            'diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" directional="true"/>'
        )
        return xml.replace("<worldbody>", f"<worldbody>\n  {camera}\n  {light}")

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def _make_obs_spec(self) -> Composite:
        # Semantically-split observation: the dynamics-relevant pieces
        # are exposed as separate keys so downstream transforms (e.g.
        # :class:`CatTensors`) can recombine them while keeping
        # ``manipulability`` available for logging without feeding it
        # into the policy.
        n = self.N_GIMBALS
        return Composite(
            quat_err=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            bus_omega=Unbounded(
                shape=(self.num_envs, 3), dtype=self.dtype, device=self.device
            ),
            gimbal_angles=Unbounded(
                shape=(self.num_envs, 2 * n), dtype=self.dtype, device=self.device
            ),
            gimbal_rates=Unbounded(
                shape=(self.num_envs, n), dtype=self.dtype, device=self.device
            ),
            manipulability=Unbounded(
                shape=(self.num_envs, 1), dtype=self.dtype, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _make_specs(self) -> None:
        super()._make_specs()
        # Override the action spec: agent controls only gimbals, not rotors.
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.N_GIMBALS),
            dtype=self.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State plumbing
    # ------------------------------------------------------------------

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos, qvel = super()._sample_initial_state(n, tensordict)
        # Pin rotor velocities (qvel layout: [bus_lin(3), bus_ang(3),
        # then per-CMG (gimbal_rate, rotor_rate) pairs]).
        rotor_vel_idx = [6 + 2 * i + 1 for i in range(self.N_GIMBALS)]
        qvel[..., rotor_vel_idx] = self.ROTOR_SPEED
        # Optional starting attitude override -- when present in the
        # input tensordict, write into ``qpos[..., 3:7]`` (the bus
        # quaternion) so the simulator boots at the requested attitude.
        # The caller is expected to pass a full-batch ``(num_envs, 4)``
        # quaternion; the backend's ``reset_mask`` picks the rows it
        # actually resets, so we can write through without masking.
        if tensordict is not None and "init_bus_quat" in tensordict.keys():
            init_q = tensordict.get("init_bus_quat").to(qpos.dtype).to(qpos.device)
            init_q = init_q / init_q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            qpos[..., 3:7] = init_q
        return qpos, qvel

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        # Scale the agent's [-1, 1] command up to ``[-action_scale,
        # action_scale]`` rad/s of gimbal rate. With the velocity
        # actuators in the XML, the saturation cap on commanded rate
        # is the dominant control-authority knob, so leaving it at 1
        # rad/s makes 180-deg slews infeasible in any reasonable
        # horizon. Keep the policy's action range at [-1, 1] (clean
        # for ``TanhNormal``) and absorb the scaling here.
        scaled = action * self.action_scale
        rotor_ctrl = torch.full(
            (scaled.shape[0], self.N_GIMBALS),
            self.ROTOR_SPEED,
            dtype=scaled.dtype,
            device=scaled.device,
        )
        return torch.cat([scaled, rotor_ctrl], dim=-1)

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        if tensordict is not None and "target_quat" in tensordict.keys():
            t = tensordict.get("target_quat").to(self.dtype).to(self.device)
            self._target_quat = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            self._target_quat = random_unit_quat(
                (self.num_envs,),
                generator=self.rng,
                device=self.device,
                dtype=self.dtype,
            )
        # Manipulability is only computed inside :meth:`_compute_reward`.
        # Reset the cache to a non-singular default so the first obs
        # after reset is well-defined.
        self._last_manip = self._compute_manip_from_qpos(
            self._backend.qpos.to(self.dtype)
        )

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        # Build a full-batch target_quat candidate (either the user-
        # supplied override or a freshly-sampled one) and mux only the
        # masked rows in via ``torch.where``. Keeps the path free of
        # data-dependent shape ops (no ``mask.sum().item()`` sync, no
        # boolean indexing).
        if tensordict is not None and "target_quat" in tensordict.keys():
            t = tensordict.get("target_quat").to(self.dtype).to(self.device)
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            t = random_unit_quat(
                (self.num_envs,),
                generator=self.rng,
                device=self.device,
                dtype=self.dtype,
            )
        m = mask.unsqueeze(-1)
        self._target_quat = torch.where(m, t, self._target_quat)
        new_manip = self._compute_manip_from_qpos(self._backend.qpos.to(self.dtype))
        self._last_manip = torch.where(m, new_manip, self._last_manip)

    # ------------------------------------------------------------------
    # Observation, reward, done
    # ------------------------------------------------------------------

    def _attitude_error(self, qpos: torch.Tensor) -> torch.Tensor:
        bus_quat = qpos[..., 3:7].to(self.dtype)
        # q_err such that target = current * q_err  =>  q_err = current^-1 * target
        q_err = quat_mul(quat_conj(bus_quat), self._target_quat)
        return quat_log(q_err)  # (B, 3)

    def _compute_manip_from_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        """Compute the per-env manipulability from a ``qpos`` snapshot.

        Returns shape ``(num_envs, 1)``. Used to refresh
        :attr:`_last_manip` on reset and inside :meth:`_compute_reward`.
        """
        gimbal_idx = [7 + 2 * i for i in range(self.N_GIMBALS)]
        gimbal_angles = qpos[..., gimbal_idx]
        jac = cmg_jacobian(
            gimbal_angles, self._gimbal_axes, self._rotor_axes_ref, self._rotor_h
        )
        self._assert_finite("reset/qpos", qpos)
        self._assert_finite("reset/gimbal_angles", gimbal_angles)
        self._assert_finite("reset/cmg_jacobian", jac)
        manip = manipulability(jac)
        self._assert_finite("reset/manipulability", manip)
        return manip.unsqueeze(-1)

    def _make_obs_split(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        """Return the observation as a dict of named sub-keys.

        Replaces the legacy single-tensor ``_make_obs``. Downstream
        :class:`CatTensors` transforms can recombine the dynamics-relevant
        keys into a single observation while keeping ``manipulability``
        separate for logging.
        """
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        n = self.N_GIMBALS
        gimbal_idx = [7 + 2 * i for i in range(n)]
        gimbal_rate_idx = [6 + 2 * i for i in range(n)]
        return {
            "quat_err": self._attitude_error(qpos),
            "bus_omega": qvel[..., 3:6].clone(),
            "gimbal_angles": torch.cat(
                [qpos[..., gimbal_idx].sin(), qpos[..., gimbal_idx].cos()],
                dim=-1,
            ),
            "gimbal_rates": qvel[..., gimbal_rate_idx].clone(),
            "manipulability": self._last_manip,
        }

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if not self.pixel_only:
            out.update(self._make_obs_split(state))
        if self.from_pixels:
            out["pixels"] = self._backend.render(
                camera_id=self.camera_id,
                width=self.render_width,
                height=self.render_height,
                background=self.RENDER_BACKGROUND,
            )
        return out

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        qpos = next_state["qpos"].to(self.dtype)
        gimbal_idx = [7 + 2 * i for i in range(self.N_GIMBALS)]
        gimbal_angles = qpos[..., gimbal_idx]

        self._assert_finite("reward/qpos", qpos)
        self._assert_finite("reward/action", action)
        self._assert_finite("reward/target_quat", self._target_quat)
        quat_err = self._attitude_error(qpos)
        self._assert_finite("reward/quat_err", quat_err)
        att_err = quat_err.norm(dim=-1)
        self._assert_finite("reward/attitude_error", att_err)
        jac = cmg_jacobian(
            gimbal_angles, self._gimbal_axes, self._rotor_axes_ref, self._rotor_h
        )
        self._assert_finite("reward/gimbal_angles", gimbal_angles)
        self._assert_finite("reward/cmg_jacobian", jac)
        jjt = jac @ jac.transpose(-1, -2)
        self._assert_finite("reward/cmg_jacobian_jjt", jjt)
        det = torch.linalg.det(jjt)
        self._assert_finite("reward/cmg_jacobian_det", det)
        manip = torch.sqrt(det.clamp_min(0.0) + 1e-8)
        self._assert_finite("reward/manipulability", manip)
        # Normalize manipulability by ``rotor_h ** 3`` so the metric
        # is rotor-speed invariant. ``det(J J^T) ~ h^6`` scales with
        # the rotor angular momentum, so without this rescaling the
        # 1/manip penalty is ~5e-8 per step at ``rotor_h=100`` and
        # never trains the avoid-singularity half of the task. After
        # rescaling, ``manip_norm`` lives in ``[0, ~1]`` and the
        # default ``singularity_weight=0.5`` yields ~0.5 / O(1) ≈ 0.5
        # contribution on the nominal cluster, growing as the cluster
        # approaches a singular configuration.
        manip_norm = manip / (self._rotor_h**3)
        # Cache the *unnormalized* metric for the obs (so logging
        # remains physically interpretable across rotor speeds), but
        # use the normalized one for the penalty.
        self._last_manip = manip.unsqueeze(-1)
        ctrl_cost = (action.to(self.dtype) ** 2).sum(dim=-1)
        self._assert_finite("reward/control_cost", ctrl_cost)
        # Body-rate penalty -- the slew-and-stop incentive. Without
        # this, the optimal "swing through the target" policy is
        # locally rewarded as much as the proper "settle at the
        # target" one, because attitude error alone is zero at the
        # crossing instant.
        qvel = next_state["qvel"].to(self.dtype)
        bus_omega = qvel[..., 3:6]
        omega_cost = (bus_omega**2).sum(dim=-1)
        self._assert_finite("reward/omega_cost", omega_cost)
        if self.singularity_mode == "exp":
            # Bounded smooth penalty: ``w * exp(-k * manip_norm)`` saturates
            # at ``-w`` near the singularity instead of diverging via 1/0.
            sing_penalty = self.singularity_weight * torch.exp(
                -self.singularity_exp_k * manip_norm
            )
        else:  # "inverse"
            sing_penalty = self.singularity_weight / manip_norm.clamp_min(
                self.singularity_clamp_min
            )
        reward = (
            -self.attitude_weight * att_err
            - sing_penalty
            - self.ctrl_cost_weight * ctrl_cost
            - self.omega_weight * omega_cost
        )
        self._assert_finite("reward/total", reward)
        reward = reward.unsqueeze(-1)
        self._assert_finite("reward/total_unsqueezed", reward)
        return reward

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
