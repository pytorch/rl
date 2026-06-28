# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Physics-engine adapters for the MuJoCo custom envs.

Three backends share a common contract (:class:`_PhysicsBackend`):

* ``mujoco-torch`` -- native torch, batched via :func:`torch.vmap`.
* ``mjx`` -- JAX-vectorized via :func:`jax.vmap` + :func:`jax.jit`,
  bridged to torch through DLPack.
* ``mujoco`` -- official C-bindings, batched by Python loop.

Each backend owns the model and the per-env simulation state. The env
calls :meth:`reset`, :meth:`step`, and reads ``qpos``/``qvel``/``time``
to compute observations and rewards.
"""

from __future__ import annotations

import abc
import importlib.util
import urllib.request
from pathlib import Path
from typing import Literal

import torch

_has_mujoco_torch = importlib.util.find_spec("mujoco_torch") is not None
_has_mujoco = importlib.util.find_spec("mujoco") is not None
_has_jax = importlib.util.find_spec("jax") is not None
_has_mjx = _has_mujoco and importlib.util.find_spec("mujoco.mjx") is not None


BackendName = Literal["mujoco-torch", "mjx", "mujoco"]


def resolve_xml_string(path_or_url: str | Path) -> str:
    """Read XML from a local path or http(s) URL.

    URLs let users point at remote XML assets without vendoring them.
    """
    p = str(path_or_url)
    if p.startswith(("http://", "https://")):
        with urllib.request.urlopen(p) as resp:
            return resp.read().decode("utf-8")
    return Path(p).read_text()


class _PhysicsBackend(abc.ABC):
    """Common contract across the three engines.

    After construction, attributes ``nq, nv, nu, timestep, qpos0,
    actuator_lo, actuator_hi`` are populated from the model. ``qpos0`` is
    the initial ``qpos`` produced by ``mj_forward`` on a fresh ``MjData``.
    """

    nq: int
    nv: int
    nu: int
    timestep: float
    qpos0: torch.Tensor
    qvel0: torch.Tensor
    actuator_lo: torch.Tensor
    actuator_hi: torch.Tensor
    actuator_ctrllimited: torch.Tensor

    def __init__(
        self, xml_string: str, *, num_envs: int, device: torch.device | None
    ) -> None:
        self.num_envs = num_envs
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._init_model(xml_string)

    @abc.abstractmethod
    def _init_model(self, xml_string: str) -> None:
        """Parse XML and prepare the batched data state.

        Populates ``nq, nv, nu, timestep, qpos0, qvel0, actuator_lo,
        actuator_hi`` from the parsed model.
        """

    @abc.abstractmethod
    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Set the full batched state to the given ``qpos, qvel``.

        Shapes: ``(num_envs, nq)`` and ``(num_envs, nv)``.
        """

    @abc.abstractmethod
    def reset_mask(
        self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> None:
        """Reset the subset of envs where ``mask`` is True.

        ``mask`` is shape ``(num_envs,)`` bool. ``qpos`` / ``qvel`` are
        full-batch ``(num_envs, nq)`` / ``(num_envs, nv)`` -- the backend
        is responsible for masking internally. Full-batch tensors keep
        shapes data-independent so the reset path stays sync-free under
        ``torch.compile`` / cudagraphs.
        """

    @abc.abstractmethod
    def step(self, ctrl: torch.Tensor, frame_skip: int) -> None:
        """Advance state by ``frame_skip`` physics substeps."""

    @property
    @abc.abstractmethod
    def qpos(self) -> torch.Tensor:
        """Current ``qpos``, shape ``(num_envs, nq)`` on ``self.device``."""

    @property
    @abc.abstractmethod
    def qvel(self) -> torch.Tensor:
        """Current ``qvel``, shape ``(num_envs, nv)`` on ``self.device``."""

    @property
    @abc.abstractmethod
    def time(self) -> torch.Tensor:
        """Current simulation time per env, shape ``(num_envs,)``."""

    def render(
        self,
        *,
        camera_id: int = 0,
        width: int = 64,
        height: int = 64,
        background: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Render every env to RGB pixels.

        Returns a ``(num_envs, height, width, 3)`` ``uint8`` tensor on
        ``self.device``. Default raises ``NotImplementedError`` --
        backends override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rendering."
        )

    def set_static_body_pose(
        self,
        body_name: str,
        position: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> None:
        """Update a non-jointed visual body when the backend supports it.

        Backends that keep immutable compiled models can safely ignore this.
        The hook is for renderer-only markers and must not be used for
        simulated bodies.
        """
        return


# ----------------------------------------------------------------------
# mujoco-torch backend
# ----------------------------------------------------------------------


class _TorchBackend(_PhysicsBackend):
    """Native-torch backend powered by ``mujoco-torch``.

    Uses :func:`torch.vmap` to batch and (optionally) :func:`torch.compile`
    for the per-step physics. State lives in a ``mujoco_torch.Data`` object
    with mutable ``qpos`` / ``qvel`` / ``ctrl`` torch tensors.
    """

    def __init__(
        self,
        xml_string: str,
        *,
        num_envs: int,
        device: torch.device | None,
        compile_step: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        if not _has_mujoco_torch:
            raise ImportError(
                "backend='mujoco-torch' requires the `mujoco-torch` package. "
                "Install with `pip install mujoco-torch`."
            )
        self._compile_step = compile_step
        self._compile_kwargs = compile_kwargs or {}
        super().__init__(xml_string, num_envs=num_envs, device=device)

    def _init_model(self, xml_string: str) -> None:
        import mujoco
        import mujoco_torch

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)
        self._m_mj = m_mj

        mx = mujoco_torch.device_put(m_mj)
        dx0 = mujoco_torch.device_put(d_mj)
        if self.device != torch.device("cpu"):
            mx = mx.to(self.device)
            dx0 = dx0.to(self.device)
        # One step so all derived dtypes match what vmap(step) produces.
        dx0 = mujoco_torch.step(mx, dx0)

        self._mx = mx
        self._dx0 = dx0
        self._sim_dtype = dx0.qpos.dtype
        self._ctrl_dtype = dx0.ctrl.dtype

        self.nq = int(m_mj.nq)
        self.nv = int(m_mj.nv)
        self.nu = int(m_mj.nu)
        self.timestep = float(m_mj.opt.timestep)
        self.qpos0 = dx0.qpos.detach().clone()
        self.qvel0 = dx0.qvel.detach().clone()
        ar = torch.as_tensor(m_mj.actuator_ctrlrange, device=self.device)
        self.actuator_lo = ar[:, 0].to(self._ctrl_dtype)
        self.actuator_hi = ar[:, 1].to(self._ctrl_dtype)
        self.actuator_ctrllimited = torch.as_tensor(
            m_mj.actuator_ctrllimited, device=self.device, dtype=torch.bool
        )

        # Build the batched state and the (compiled) physics step.
        self._dx = self._dx0.expand(self.num_envs).clone()
        self._build_step_fn()

    def _build_step_fn(self) -> None:
        import mujoco_torch

        mx = self._mx
        single = self.num_envs == 1

        if single:

            def _one_step(d):
                return mujoco_torch.step(mx, d)

            base = _one_step
        else:
            _vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

            def _vmap_one(d):
                return _vmap_step(d)

            base = _vmap_one

        # When ``compile_step`` is set, compile only the per-step fn and
        # keep the frame_skip loop in Python. Compiling the unrolled loop
        # blows up the graph (50x more nodes), which sends inductor's
        # fusion analysis into multi-hour territory on CUDA backends.
        if self._compile_step:
            base_compiled = torch.compile(base, **self._compile_kwargs)

            def _multi_step(d, frame_skip: int):
                for _ in range(frame_skip):
                    d = base_compiled(d)
                return d

        else:

            def _multi_step(d, frame_skip: int):
                for _ in range(frame_skip):
                    d = base(d)
                return d

        self._physics_step = _multi_step
        self._single_env = single

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        # Refresh from the warm reference so all derived data is sane,
        # then overwrite qpos / qvel.
        self._dx = self._dx0.expand(self.num_envs).clone()
        self._dx.qpos.copy_(qpos.to(self._sim_dtype))
        self._dx.qvel.copy_(qvel.to(self._sim_dtype))

    def reset_mask(
        self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> None:
        # qpos / qvel are full-batch (num_envs, *). We mux in the
        # masked rows via ``torch.where`` so the path stays sync-free
        # (no data-dependent shapes, no ``.item()`` calls).
        fresh = self._dx0.expand(self.num_envs).clone()
        fresh.qpos.copy_(qpos.to(self._sim_dtype))
        fresh.qvel.copy_(qvel.to(self._sim_dtype))
        # The simulator state contains both per-env tensors and 0-dim /
        # non-batched leaves shared across envs (``nefc``, ``ncon``,
        # ...). Mask only the batched ones; the shared scalars don't
        # need touching. We enumerate the field names via the public
        # ``to_tensordict`` view, then fetch the *live* tensor off the
        # tensorclass with ``getattr`` so the in-place ``copy_`` mirrors
        # back onto the simulator state.
        cur_td = self._dx.to_tensordict()
        new_td = fresh.to_tensordict()
        for k in cur_td.keys():
            dst = getattr(self._dx, k, None)
            src = new_td.get(k, default=None)
            if dst is None or src is None:
                continue
            if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
                continue
            if dst.ndim == 0 or dst.shape[0] != self.num_envs:
                continue
            # A stride-0 batch dimension means every env row aliases the same
            # storage, so there is no row-local value to update during a
            # masked reset.
            if dst.stride(0) == 0:
                continue
            m = mask.view((self.num_envs,) + (1,) * (dst.ndim - 1))
            dst.copy_(torch.where(m, src.to(dst.dtype), dst))

    def step(self, ctrl: torch.Tensor, frame_skip: int) -> None:
        ctrl = ctrl.to(self._ctrl_dtype)
        # Clamp only actuators that MuJoCo marks as ctrl-limited. Unlimited
        # actuators report a default ctrlrange of [0, 0], which is not an
        # active range and must not zero out controls.
        clamped = torch.minimum(torch.maximum(ctrl, self.actuator_lo), self.actuator_hi)
        ctrl = torch.where(self.actuator_ctrllimited, clamped, ctrl)
        self._dx.update_(ctrl=ctrl)
        if self._single_env:
            stepped = self._physics_step(self._dx[0], frame_skip)
            self._dx = stepped.unsqueeze(0)
        else:
            self._dx = self._physics_step(self._dx, frame_skip)

    @property
    def qpos(self) -> torch.Tensor:
        return self._dx.qpos

    @property
    def qvel(self) -> torch.Tensor:
        return self._dx.qvel

    @property
    def time(self) -> torch.Tensor:
        # mujoco-torch's Data exposes a per-env time scalar.
        t = self._dx.time
        if t.ndim == 0:
            t = t.expand(self.num_envs)
        return t

    def render(
        self,
        *,
        camera_id: int = 0,
        width: int = 64,
        height: int = 64,
        background: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        import mujoco_torch

        # Cache precomputed render data lazily (depends on the model only).
        if not hasattr(self, "_render_precomp"):
            self._render_precomp = mujoco_torch.precompute_render_data(self._mx)
        frames = []
        for i in range(self.num_envs):
            rgb, _, _ = mujoco_torch.render(
                self._mx,
                self._dx[i],
                camera_id=camera_id,
                width=width,
                height=height,
                precomp=self._render_precomp,
                background=background,
            )
            frames.append((rgb * 255).clamp(0, 255).to(torch.uint8))
        return torch.stack(frames)


# ----------------------------------------------------------------------
# mujoco C-bindings backend
# ----------------------------------------------------------------------


class _MujocoBackend(_PhysicsBackend):
    """Reference backend: official ``mujoco`` C-bindings, **single env**.

    The C-bindings can't vmap, so we don't fake batching here. To run
    multiple environments in parallel with this backend, compose with
    :class:`~torchrl.envs.ParallelEnv` (multiprocess) or
    :class:`~torchrl.envs.SerialEnv` (in-process loop). The
    :class:`~torchrl.envs.custom.mujoco.MujocoEnv` metaclass does this
    dispatch automatically when ``num_workers > 1`` or ``num_envs > 1``
    is requested with ``backend='mujoco'``.
    """

    def __init__(
        self,
        xml_string: str,
        *,
        num_envs: int,
        device: torch.device | None,
    ) -> None:
        if not _has_mujoco:
            raise ImportError(
                "backend='mujoco' requires the `mujoco` package. "
                "Install with `pip install mujoco`."
            )
        if num_envs != 1:
            raise ValueError(
                "backend='mujoco' is single-env. To batch, wrap with "
                "torchrl.envs.ParallelEnv or torchrl.envs.SerialEnv "
                "(MujocoEnv does this automatically when num_workers>1 "
                "or num_envs>1 is passed)."
            )
        super().__init__(xml_string, num_envs=num_envs, device=device)

    def _init_model(self, xml_string: str) -> None:
        import mujoco

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        self._m = m_mj
        self._d = d_mj
        self._mujoco = mujoco

        self.nq = int(m_mj.nq)
        self.nv = int(m_mj.nv)
        self.nu = int(m_mj.nu)
        self.timestep = float(m_mj.opt.timestep)
        self.qpos0 = torch.as_tensor(d_mj.qpos.copy(), device=self.device).to(
            torch.float32
        )
        self.qvel0 = torch.as_tensor(d_mj.qvel.copy(), device=self.device).to(
            torch.float32
        )
        ar = torch.as_tensor(
            m_mj.actuator_ctrlrange, device=self.device, dtype=torch.float32
        )
        self.actuator_lo = ar[:, 0]
        self.actuator_hi = ar[:, 1]
        self.actuator_ctrllimited = torch.as_tensor(
            m_mj.actuator_ctrllimited, device=self.device, dtype=torch.bool
        )

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        import mujoco

        self._d.qpos[:] = qpos.detach().cpu().double().numpy()[0]
        self._d.qvel[:] = qvel.detach().cpu().double().numpy()[0]
        self._d.time = 0.0
        mujoco.mj_forward(self._m, self._d)

    def reset_mask(
        self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> None:
        # Single-env: mask is shape (1,). Either reset or no-op.
        if bool(mask.any()):
            self.reset(qpos, qvel)

    def step(self, ctrl: torch.Tensor, frame_skip: int) -> None:
        import mujoco

        clamped = torch.minimum(torch.maximum(ctrl, self.actuator_lo), self.actuator_hi)
        clamped = torch.where(self.actuator_ctrllimited, clamped, ctrl)
        self._d.ctrl[:] = clamped.detach().cpu().double().numpy()[0]
        for _ in range(frame_skip):
            mujoco.mj_step(self._m, self._d)

    @property
    def qpos(self) -> torch.Tensor:
        return torch.as_tensor(
            self._d.qpos.copy(), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    @property
    def qvel(self) -> torch.Tensor:
        return torch.as_tensor(
            self._d.qvel.copy(), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    @property
    def time(self) -> torch.Tensor:
        return torch.tensor([self._d.time], device=self.device, dtype=torch.float32)

    def render(
        self,
        *,
        camera_id: int = 0,
        width: int = 64,
        height: int = 64,
        background: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        import mujoco
        import numpy as np

        if (
            not hasattr(self, "_renderer")
            or self._renderer.height != height
            or self._renderer.width != width
        ):
            self._renderer = mujoco.Renderer(self._m, height=height, width=width)
        self._renderer.update_scene(self._d, camera=camera_id)
        rgb = self._renderer.render()  # (H, W, 3) uint8 numpy
        if background is not None:
            # Tint background pixels (those at the far plane). Approximation:
            # uses the geom-id buffer is overkill -- skip when not requested.
            pass
        return torch.as_tensor(np.ascontiguousarray(rgb), device=self.device).unsqueeze(
            0
        )

    def set_static_body_pose(
        self,
        body_name: str,
        position: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> None:
        body_id = self._mujoco.mj_name2id(
            self._m,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            return
        pos = position.reshape(-1, 3)[0].detach().cpu().to(torch.float64)
        quat = quaternion.reshape(-1, 4)[0].detach().cpu().to(torch.float64)
        quat = quat / quat.norm().clamp_min(1e-12)
        self._m.body_pos[body_id] = pos.numpy()
        self._m.body_quat[body_id] = quat.numpy()
        self._mujoco.mj_forward(self._m, self._d)


# ----------------------------------------------------------------------
# MJX (JAX) backend
# ----------------------------------------------------------------------


class _MJXBackend(_PhysicsBackend):
    """JAX-vectorized backend via ``mujoco.mjx``.

    Mirrors the pattern in :mod:`torchrl.envs.libs.brax`: build an
    ``mjx.Model``, ``vmap+jit`` the step function, bridge JAX arrays to
    torch tensors via DLPack on each call.
    """

    def __init__(
        self,
        xml_string: str,
        *,
        num_envs: int,
        device: torch.device | None,
    ) -> None:
        if not (_has_mjx and _has_jax):
            raise ImportError(
                "backend='mjx' requires `mujoco>=3.0` (with mjx) and `jax`. "
                "Install with `pip install mujoco-mjx jax`."
            )
        super().__init__(xml_string, num_envs=num_envs, device=device)

    def _init_model(self, xml_string: str) -> None:
        import jax
        import mujoco
        from mujoco import mjx

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        mx = mjx.put_model(m_mj)
        dx0_single = mjx.make_data(mx)
        dx0_single = mjx.forward(mx, dx0_single)

        self._mjx = mjx
        self._jax = jax
        self._m_mj = m_mj  # kept for rendering via mujoco.Renderer
        self._mx = mx
        self._dx0_single = dx0_single
        # All JAX arrays must live on this device or jit will fail with
        # "Received incompatible devices for jitted computation". The
        # model's data state lives wherever `mjx.put_model` placed it
        # (JAX default device), and any tensors we splice in (qpos /
        # qvel / time) must be put_d to match.
        self._jax_device = dx0_single.qpos.device

        self.nq = int(m_mj.nq)
        self.nv = int(m_mj.nv)
        self.nu = int(m_mj.nu)
        self.timestep = float(m_mj.opt.timestep)
        self.qpos0 = self._jax_to_torch(dx0_single.qpos)
        self.qvel0 = self._jax_to_torch(dx0_single.qvel)
        ar = torch.as_tensor(
            m_mj.actuator_ctrlrange, device=self.device, dtype=torch.float32
        )
        self.actuator_lo = ar[:, 0]
        self.actuator_hi = ar[:, 1]
        self.actuator_ctrllimited = torch.as_tensor(
            m_mj.actuator_ctrllimited, device=self.device, dtype=torch.bool
        )

        self._vmap_step = jax.jit(jax.vmap(lambda d: mjx.step(mx, d)))
        self._vmap_forward = jax.jit(jax.vmap(lambda d: mjx.forward(mx, d)))

        # Initial batched state.
        self._dx = jax.vmap(lambda _: dx0_single)(jax.numpy.arange(self.num_envs))

    def _jax_to_torch(self, x) -> torch.Tensor:
        from torchrl.envs.libs.jax_utils import _ndarray_to_tensor

        return _ndarray_to_tensor(x).to(self.device).to(torch.float32)

    def _torch_to_jax(self, x: torch.Tensor):
        from torchrl.envs.libs.jax_utils import _tensor_to_ndarray

        arr = _tensor_to_ndarray(x.contiguous())
        # Force the array onto the model's JAX device. Torch tensors on
        # CPU would otherwise yield a CPU JAX array even when the model
        # lives on GPU, causing a mixed-device pytree at jit time.
        return self._jax.device_put(arr, self._jax_device)

    def _broadcast_dx0(self):
        """Build a batched copy of ``_dx0_single`` on ``self._jax_device``."""
        jax = self._jax
        idx = jax.device_put(jax.numpy.arange(self.num_envs), self._jax_device)
        return jax.vmap(lambda _: self._dx0_single)(idx)

    def reset(self, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        qpos_j = self._torch_to_jax(qpos)
        qvel_j = self._torch_to_jax(qvel)
        # Broadcast _dx0_single (with time=0) over the batch, splice in
        # the new qpos/qvel, then re-run mjx.forward so derived quantities
        # are consistent. Time stays at zero from the reference state.
        dx = self._broadcast_dx0()
        dx = dx.replace(qpos=qpos_j, qvel=qvel_j)
        self._dx = self._vmap_forward(dx)

    def reset_mask(
        self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> None:
        # ``qpos`` and ``qvel`` are full-batch (num_envs, *). Mux the
        # reset rows in via ``torch.where`` on the torch side, then push
        # back to JAX. Partial reset isn't on the hot path; the JAX
        # bridge itself dominates the cost here.
        full_qpos = self.qpos.clone()
        full_qvel = self.qvel.clone()
        m_q = mask.view(self.num_envs, *([1] * (full_qpos.ndim - 1)))
        m_v = mask.view(self.num_envs, *([1] * (full_qvel.ndim - 1)))
        full_qpos = torch.where(m_q, qpos.to(full_qpos.dtype), full_qpos)
        full_qvel = torch.where(m_v, qvel.to(full_qvel.dtype), full_qvel)
        time = self.time.clone()
        time = torch.where(mask, torch.zeros_like(time), time)

        qpos_j = self._torch_to_jax(full_qpos)
        qvel_j = self._torch_to_jax(full_qvel)
        time_j = self._torch_to_jax(time)
        dx = self._broadcast_dx0()
        dx = dx.replace(qpos=qpos_j, qvel=qvel_j, time=time_j)
        self._dx = self._vmap_forward(dx)

    def step(self, ctrl: torch.Tensor, frame_skip: int) -> None:
        clamped = torch.minimum(torch.maximum(ctrl, self.actuator_lo), self.actuator_hi)
        ctrl = torch.where(self.actuator_ctrllimited, clamped, ctrl)
        ctrl_j = self._torch_to_jax(ctrl)
        self._dx = self._dx.replace(ctrl=ctrl_j)
        for _ in range(frame_skip):
            self._dx = self._vmap_step(self._dx)

    @property
    def qpos(self) -> torch.Tensor:
        return self._jax_to_torch(self._dx.qpos)

    @property
    def qvel(self) -> torch.Tensor:
        return self._jax_to_torch(self._dx.qvel)

    @property
    def time(self) -> torch.Tensor:
        return self._jax_to_torch(self._dx.time)

    def render(
        self,
        *,
        camera_id: int = 0,
        width: int = 64,
        height: int = 64,
        background: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Render via mujoco's CPU renderer after copying mjx state to MjData.

        Slow path (one MjData per env, sequential render). Acceptable for
        eval / video; not for high-throughput pixel-based training.
        """
        import mujoco
        import numpy as np

        if (
            not hasattr(self, "_renderer")
            or self._renderer.height != height
            or self._renderer.width != width
        ):
            self._renderer = mujoco.Renderer(self._m_mj, height=height, width=width)
        if not hasattr(self, "_render_d"):
            self._render_d = mujoco.MjData(self._m_mj)

        qpos = self.qpos.detach().cpu().double().numpy()
        qvel = self.qvel.detach().cpu().double().numpy()
        frames = []
        for i in range(self.num_envs):
            self._render_d.qpos[:] = qpos[i]
            self._render_d.qvel[:] = qvel[i]
            mujoco.mj_forward(self._m_mj, self._render_d)
            self._renderer.update_scene(self._render_d, camera=camera_id)
            frames.append(self._renderer.render().copy())
        rgb = np.stack(frames, axis=0)
        return torch.as_tensor(np.ascontiguousarray(rgb), device=self.device)


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def make_backend(
    name: BackendName,
    xml_string: str,
    *,
    num_envs: int,
    device: torch.device | None,
    compile_step: bool = False,
    compile_kwargs: dict | None = None,
) -> _PhysicsBackend:
    """Instantiate the requested backend.

    Raises ``ImportError`` with an actionable message when the underlying
    package is missing. ``compile_step`` is honored only by the
    ``mujoco-torch`` backend (mjx already JITs; the C-bindings backend is
    a Python loop).
    """
    if name == "mujoco-torch":
        return _TorchBackend(
            xml_string,
            num_envs=num_envs,
            device=device,
            compile_step=compile_step,
            compile_kwargs=compile_kwargs,
        )
    if name == "mjx":
        return _MJXBackend(xml_string, num_envs=num_envs, device=device)
    if name == "mujoco":
        return _MujocoBackend(xml_string, num_envs=num_envs, device=device)
    raise ValueError(
        f"unknown backend {name!r}; expected one of 'mujoco-torch', 'mjx', 'mujoco'"
    )
