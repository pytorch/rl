# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pure-torch math helpers for the MuJoCo custom envs.

Quaternions follow the MuJoCo convention ``(w, x, y, z)``. CMG geometry
helpers support the satellite env's manipulability-based singularity penalty.
"""

from __future__ import annotations

import math

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two unit quaternions, ``(..., 4)`` -> ``(..., 4)``."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithm map of a unit quaternion to a 3-vector axis-angle.

    For ``q = (cos(a/2), sin(a/2) n)`` with axis ``n`` and angle ``a``,
    ``quat_log(q) = a * n``. Range: ``[0, pi]`` in magnitude.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    # q and -q encode the same SO(3) rotation. Use the representative
    # with non-negative scalar part so the log map is the shortest arc.
    sign = torch.where(q[..., 0:1] < 0, -1.0, 1.0)
    q = q * sign
    w = q[..., 0:1].clamp(-1.0, 1.0)
    v = q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    angle = 2.0 * torch.atan2(v_norm, w)
    return v / v_norm * angle


def random_unit_quat(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform random unit quaternion on ``SO(3)``."""
    q = torch.randn(*shape, 4, generator=generator, device=device, dtype=dtype)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _rodrigues_rotate(
    g: torch.Tensor, r0: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Rotate ``r0`` around unit axis ``g`` by angle ``theta``.

    ``g``, ``r0``: ``(3, N)``. ``theta``: ``(..., N)``. Output: ``(..., 3, N)``.
    """
    cos_t = torch.cos(theta).unsqueeze(-2)
    sin_t = torch.sin(theta).unsqueeze(-2)
    g_dot_r = (g * r0).sum(dim=0)
    g_cross_r = torch.linalg.cross(g, r0, dim=0)
    return cos_t * r0 + sin_t * g_cross_r + (1.0 - cos_t) * g_dot_r * g


def cmg_jacobian(
    gimbal_angles: torch.Tensor,
    gimbal_axes: torch.Tensor,
    rotor_axes_ref: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """CMG output-torque Jacobian over gimbal rates.

    For each CMG with gimbal axis ``g_i`` and rotor axis ``r_i(theta_i)``,
    the rate-of-change of the rotor's angular momentum per unit gimbal
    rate is ``h * (g_i x r_i(theta_i))``. Stacked into a ``(..., 3, N)``
    matrix. The torque applied to the *bus* is the Newton's-third-law
    reaction, ``-h * (g_i x r_i)``; this function returns the
    rotor-frame quantity because the manipulability metric
    ``sqrt(det(J J^T))`` -- the only consumer in this module -- is
    sign-invariant. Callers that care about the body-frame slewing
    direction must negate the result.

    Args:
        gimbal_angles: ``(..., N)`` current gimbal angles in radians.
        gimbal_axes: ``(3, N)`` fixed gimbal axes in body frame, unit norm.
        rotor_axes_ref: ``(3, N)`` rotor axes at ``theta=0``, unit norm,
            perpendicular to the corresponding gimbal axis.
        h: scalar rotor angular momentum magnitude.

    Returns:
        ``(..., 3, N)`` Jacobian whose ``i``-th column is the
        rotor-momentum-rate per unit ``i``-th gimbal rate. Negate to
        get the bus torque per unit gimbal rate.
    """
    r = _rodrigues_rotate(gimbal_axes, rotor_axes_ref, gimbal_angles)
    g_expanded = gimbal_axes.expand_as(r)
    return h * torch.linalg.cross(g_expanded, r, dim=-2)


def manipulability(jac: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``sqrt(det(J J^T) + eps)`` -- proxy for distance from singularity.

    ``jac`` shape: ``(..., 3, N)`` with ``N >= 3``. Output: ``(...,)``.
    """
    jjt = jac @ jac.transpose(-1, -2)
    det = torch.linalg.det(jjt).clamp_min(0.0)
    return torch.sqrt(det + eps)


def pyramid_4cmg_geometry(
    skew_deg: float = 54.7356,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard 4-CMG pyramid matching ``satellite_large.xml``.

    Each gimbal axis lies in the body ``xy`` plane (one per face of the
    bus). The four rotor reference axes are tilted by ``skew_deg`` from
    ``+z`` toward the corresponding face. The default skew angle
    ``arctan(sqrt(2)) ~ 54.74 deg`` gives a spherical momentum envelope
    (the textbook configuration).

    Order matches the four CMGs declared in
    ``satellite_large.xml`` (``cmg1``..``cmg4``):

    ============  ================  ================
     CMG          gimbal axis        rotor axis @ 0
    ============  ================  ================
     cmg1 (+x)    (0, 1, 0)          (sb, 0, cb)
     cmg2 (+y)    (-1, 0, 0)         (0, sb, cb)
     cmg3 (-x)    (0, -1, 0)         (-sb, 0, cb)
     cmg4 (-y)    (1, 0, 0)          (0, -sb, cb)
    ============  ================  ================

    Returns ``(gimbal_axes, rotor_axes_ref)``, both ``(3, 4)`` and
    column-aligned with the XML's gimbal joint order.
    """
    beta = math.radians(skew_deg)
    cb, sb = math.cos(beta), math.sin(beta)
    g = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    r0 = torch.tensor(
        [
            [sb, 0.0, -sb, 0.0],
            [0.0, sb, 0.0, -sb],
            [cb, cb, cb, cb],
        ],
        device=device,
        dtype=dtype,
    )
    return g, r0


def orthogonal_6cmg_geometry(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """6-CMG cluster matching ``satellite_small.xml``.

    One CMG per face of the bus; the six gimbal axes are pair-wise drawn
    from ``{x, y, z}`` so that each principal axis carries two rotor
    spin vectors when the gimbals are at zero. The cluster is full-rank
    at ``theta = 0``.

    Order matches the six CMGs declared in
    ``satellite_small.xml`` (``cmg1``..``cmg6``):

    ============  ================  ================
     CMG          gimbal axis        rotor axis @ 0
    ============  ================  ================
     cmg1 (+x)    (0, 1, 0)          (0, 0, 1)
     cmg2 (-x)    (0, 0, 1)          (0, 1, 0)
     cmg3 (+y)    (0, 0, 1)          (1, 0, 0)
     cmg4 (-y)    (1, 0, 0)          (0, 0, 1)
     cmg5 (+z)    (1, 0, 0)          (0, 1, 0)
     cmg6 (-z)    (0, 1, 0)          (1, 0, 0)
    ============  ================  ================

    Returns ``(gimbal_axes, rotor_axes_ref)``, both ``(3, 6)`` and
    column-aligned with the XML's gimbal joint order.
    """
    g = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    r0 = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    return g, r0
