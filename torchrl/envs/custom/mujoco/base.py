# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Base class for MuJoCo-backed custom envs with selectable physics backend.

The core idea: subclasses describe the *task* (reward, termination,
observation map) while the base class handles spec construction,
step/reset plumbing, and dispatches the simulation to one of three
backends (``mujoco-torch``, ``mjx``, ``mujoco``) -- see
:mod:`torchrl.envs.custom.mujoco._backends`.
"""

from __future__ import annotations

import abc
import re
from copy import copy
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Binary, Bounded, Composite, Unbounded
from torchrl.envs.common import _EnvPostInit, EnvBase
from torchrl.envs.custom.mujoco._backends import (
    _PhysicsBackend,
    BackendName,
    make_backend,
    resolve_xml_string,
)

_ASSETS_DIR = Path(__file__).parent / "assets"


def _normalize_mujoco_index(
    item: Any,
    num_envs: int,
    device: torch.device | None,
) -> tuple[slice | torch.Tensor, int]:
    if isinstance(item, bool):
        raise NotImplementedError("Boolean masks are not supported for MuJoCo envs.")
    if isinstance(item, int):
        item = item + num_envs if item < 0 else item
        if item < 0 or item >= num_envs:
            raise IndexError(f"index {item} is out of bounds for {num_envs} envs.")
        return torch.tensor([item], device=device, dtype=torch.long), 1
    if isinstance(item, slice):
        indexed_num_envs = len(range(num_envs)[item])
        if indexed_num_envs == 0:
            raise IndexError("Empty MuJoCo env indexing is not supported.")
        return item, indexed_num_envs
    if isinstance(item, np.ndarray):
        if item.dtype == np.bool_:
            raise NotImplementedError(
                "Boolean masks are not supported for MuJoCo envs."
            )
        if item.ndim == 0:
            return _normalize_mujoco_index(int(item), num_envs, device)
        item = torch.as_tensor(item, dtype=torch.long, device=device)
    elif isinstance(item, torch.Tensor):
        if item.dtype == torch.bool:
            raise NotImplementedError(
                "Boolean masks are not supported for MuJoCo envs."
            )
        if item.ndim == 0:
            return _normalize_mujoco_index(int(item.item()), num_envs, device)
        item = item.to(device=device, dtype=torch.long)
    else:
        try:
            item = torch.as_tensor(item, device=device)
        except (TypeError, ValueError) as err:
            raise TypeError(
                "MuJoCo env indices must be integers, slices, integer NumPy "
                f"arrays, or integer torch tensors, got {type(item).__name__}."
            ) from err
        if item.dtype == torch.bool:
            raise NotImplementedError(
                "Boolean masks are not supported for MuJoCo envs."
            )
        item = item.to(dtype=torch.long)

    if item.ndim != 1:
        raise IndexError(
            f"MuJoCo env indices must be scalar or 1D, got shape {item.shape}."
        )
    if item.numel() == 0:
        raise IndexError("Empty MuJoCo env indexing is not supported.")
    item = torch.where(item < 0, item + num_envs, item)
    out_of_bounds = (item < 0) | (item >= num_envs)
    if bool(out_of_bounds.any()):
        bad = int(item[out_of_bounds][0].item())
        raise IndexError(f"index {bad} is out of bounds for {num_envs} envs.")
    return item, int(item.numel())


def _clone_generator(rng: torch.Generator | None, device: torch.device | None):
    if rng is None:
        return None
    cloned = torch.Generator() if device is None else torch.Generator(device=device)
    cloned.set_state(rng.get_state())
    return cloned


class _MujocoMeta(_EnvPostInit):
    """Metaclass for :class:`MujocoEnv` that dispatches batching.

    Backends ``"mujoco-torch"`` and ``"mjx"`` vmap natively over
    ``num_envs``; passing ``num_workers > 1`` or ``parallel`` to those
    backends raises ``ValueError``. Backend ``"mujoco"`` (single-env
    C-bindings) is composed via :class:`~torchrl.envs.ParallelEnv` (when
    ``parallel=True``, the default) or :class:`~torchrl.envs.SerialEnv`
    (when ``parallel=False``) when ``num_workers > 1`` or ``num_envs >
    1``. ``num_workers`` and ``num_envs`` are mutually exclusive aliases
    for that backend.
    """

    def __call__(
        cls,
        *args,
        num_workers: int = 1,
        parallel: bool | None = None,
        **kwargs,
    ):
        backend = kwargs.get("backend", getattr(cls, "DEFAULT_BACKEND", "mujoco-torch"))
        num_envs = int(kwargs.get("num_envs", 1))
        num_workers = int(num_workers)

        if backend in ("mujoco-torch", "mjx"):
            if num_workers > 1:
                raise ValueError(
                    f"backend={backend!r} batches via vmap; pass num_envs=N, "
                    "not num_workers=N. (num_workers / parallel are only "
                    "valid for backend='mujoco'.)"
                )
            if parallel is not None:
                raise ValueError(
                    f"backend={backend!r} only supports vmap-based batching; "
                    "the `parallel` kwarg is only valid for backend='mujoco'."
                )
            return super().__call__(*args, **kwargs)

        if backend == "mujoco":
            if num_workers > 1 and num_envs > 1:
                raise ValueError(
                    "For backend='mujoco', set either num_envs or num_workers, "
                    "not both -- they are aliases for the same N copies."
                )
            n = max(num_workers, num_envs)
            if n > 1:
                from torchrl.envs.batched_envs import ParallelEnv, SerialEnv

                wrap_cls = SerialEnv if parallel is False else ParallelEnv
                inner_kwargs = dict(kwargs)
                inner_kwargs["num_envs"] = 1

                def _factory(_args=args, _kwargs=inner_kwargs):
                    # Re-enters this metaclass with N=1 -> falls through.
                    return cls(*_args, **_kwargs)

                return wrap_cls(n, _factory)
            # Single env: pass through.
            return super().__call__(*args, **kwargs)

        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            "'mujoco-torch', 'mjx', 'mujoco'"
        )


class MujocoEnv(EnvBase, abc.ABC, metaclass=_MujocoMeta):
    """Base TorchRL environment backed by a swappable MuJoCo physics engine.

    Subclasses implement the task by overriding :meth:`_compute_reward` and
    :meth:`_compute_done` (and optionally :meth:`_make_obs`,
    :meth:`_make_obs_spec`, :meth:`_sample_initial_state`,
    :meth:`_prepare_ctrl`, :meth:`_patch_xml`).

    The XML asset is resolved in this order: explicit ``xml_path`` kwarg >
    class attribute :attr:`XML_PATH` > class attribute :attr:`XML_URL`.
    Either may be a local path or an ``http(s)`` URL -- URLs let users
    point at remote assets we don't (or can't) vendor.

    Args:
        xml_path: optional override for the XML asset (path or URL).
        backend: ``"mujoco-torch"`` (default), ``"mjx"``, or ``"mujoco"``.
        num_envs: batch size; the env's ``batch_size`` is ``(num_envs,)``.
        device: torch device for observations / rewards / actions.
        seed: RNG seed for the reset noise distribution and any
            subclass-defined randomization (e.g. random target attitudes).
        frame_skip: physics substeps per agent action; defaults to
            :attr:`FRAME_SKIP`.
        reset_noise_scale: uniform noise added to ``qpos0`` and zero
            ``qvel`` at reset; defaults to :attr:`RESET_NOISE_SCALE`.
        max_episode_steps: per-env truncation horizon. The env emits
            ``terminated`` (from :meth:`_compute_done`), ``truncated``
            (``step_count >= max_episode_steps``), and ``done`` (the
            OR of the two) as separate keys, matching the Gymnasium
            convention.
        dtype: floating-point dtype for action / observation / reward
            tensors. Backend-internal dtype may differ.
        compile_step: when ``backend="mujoco-torch"``, wrap the per-env
            physics step in :func:`torch.compile`. Ignored otherwise.
        compile_kwargs: forwarded to :func:`torch.compile` when applicable.
        from_pixels: if ``True``, include a ``"pixels"`` observation rendered
            from MuJoCo at reset and after every step.
        pixels_only: if ``True``, return only the ``"pixels"`` observation.
            Requires ``from_pixels=True``.
        render_width: pixel observation width used when ``from_pixels=True``.
        render_height: pixel observation height used when ``from_pixels=True``.
        render_every: render a fresh frame every ``render_every`` steps and
            reuse the previous frame in between; resets always render fresh.
            Rendering often dominates step cost, so consumers that subsample
            frames anyway (e.g. :class:`~torchrl.record.VideoRecorder` with
            ``skip``) should set this to the same stride instead of paying
            for frames that are dropped. Defaults to ``1`` (render every
            step).
        camera_id: MuJoCo camera id used for pixel observations and by
            :meth:`render` when no camera override is provided.

    Example:
        >>> from torchrl.envs import HumanoidEnv  # doctest: +SKIP
        >>> env = HumanoidEnv(num_envs=4)         # doctest: +SKIP
        >>> td = env.rollout(10)                  # doctest: +SKIP

    See Also:
        :class:`~torchrl.envs.custom.mujoco._backends._PhysicsBackend`.
    """

    DEFAULT_BACKEND: ClassVar[BackendName] = "mujoco-torch"
    XML_PATH: ClassVar[str | Path | None] = None
    XML_URL: ClassVar[str | None] = None
    FRAME_SKIP: ClassVar[int] = 5
    RESET_NOISE_SCALE: ClassVar[float] = 0.01
    SKIP_QPOS: ClassVar[int] = 0
    """How many leading entries of ``qpos`` to drop in the default obs."""
    RENDER_BACKGROUND: ClassVar[tuple[float, float, float] | None] = None
    """Background color for the ``mujoco-torch`` ray-cast renderer.
    Subclasses (e.g. satellite) override to a deep-space tone."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    batch_locked = True
    _has_frame_skip = True
    # ``_reset`` can honor ``qpos``/``qvel`` passed through the reset tensordict
    # (deterministic snapshot/branch), so this env supports
    # ``reset(td, set_state=True)``. This is purely additive: by default the env
    # samples a fresh state, so no historical behavior changes.
    _supports_set_state = True

    def __init__(
        self,
        *,
        xml_path: str | Path | None = None,
        backend: Literal["mujoco-torch", "mjx", "mujoco"] = "mujoco-torch",
        num_envs: int = 1,
        device: torch.device | None = None,
        seed: int | None = None,
        frame_skip: int | None = None,
        reset_noise_scale: float | None = None,
        max_episode_steps: int = 1000,
        dtype: torch.dtype = torch.float32,
        compile_step: bool = False,
        compile_kwargs: dict | None = None,
        from_pixels: bool = False,
        pixels_only: bool = False,
        render_width: int = 64,
        render_height: int = 64,
        render_every: int = 1,
        camera_id: int = 0,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.num_envs = num_envs
        self.dtype = dtype
        self.frame_skip = int(frame_skip if frame_skip is not None else self.FRAME_SKIP)
        self.reset_noise_scale = float(
            reset_noise_scale
            if reset_noise_scale is not None
            else self.RESET_NOISE_SCALE
        )
        self.max_episode_steps = int(max_episode_steps)
        self.backend_name: BackendName = backend
        self.from_pixels = bool(from_pixels)
        self.pixels_only = bool(pixels_only)
        if pixels_only and not from_pixels:
            raise ValueError("pixels_only=True requires from_pixels=True.")
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.render_every = int(render_every)
        if self.render_every < 1:
            raise ValueError(f"render_every must be >= 1, got {render_every}.")
        self._last_pixels: torch.Tensor | None = None
        self._render_counter = 0
        self.camera_id = int(camera_id)

        xml_string = self._load_xml(xml_path)
        self._backend: _PhysicsBackend = make_backend(
            backend,
            xml_string,
            num_envs=num_envs,
            device=self.device,
            compile_step=compile_step,
            compile_kwargs=compile_kwargs,
        )

        self.rng: torch.Generator | None = None
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.set_seed(seed)
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._make_specs()

    # ------------------------------------------------------------------
    # XML resolution + patching
    # ------------------------------------------------------------------

    def _load_xml(self, xml_path: str | Path | None) -> str:
        # An explicit ``xml_path=...`` is treated as a hard request: if
        # it can't be resolved, surface the error rather than silently
        # falling back to the class-level defaults. Subclass-level
        # ``XML_PATH`` / ``XML_URL`` defaults *are* tried in order with
        # the last failure preserved as ``__cause__``.
        if xml_path is not None:
            try:
                return self._patch_xml(resolve_xml_string(xml_path))
            except OSError as e:
                raise FileNotFoundError(
                    f"{type(self).__name__}: explicit xml_path={xml_path!r} "
                    "could not be resolved."
                ) from e

        candidates: list[str | Path] = []
        if self.XML_PATH is not None:
            p = Path(self.XML_PATH)
            if not p.is_absolute() and not str(p).startswith(("http://", "https://")):
                p = _ASSETS_DIR / p
            candidates.append(p)
        if self.XML_URL is not None:
            candidates.append(self.XML_URL)
        if not candidates:
            raise ValueError(
                f"{type(self).__name__} requires an XML asset: pass `xml_path=...` "
                "or set the `XML_PATH` / `XML_URL` class attribute."
            )
        last_exc: Exception | None = None
        for cand in candidates:
            try:
                xml_string = resolve_xml_string(cand)
                return self._patch_xml(xml_string)
            except OSError as e:
                last_exc = e
        raise FileNotFoundError(
            f"{type(self).__name__}: none of the XML candidates resolved: "
            f"{candidates}"
        ) from last_exc

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        """Hook to mutate the XML string before parsing.

        Default replaces all ``<camera>`` and ``<light>`` tags with a
        single fixed-viewpoint setup and injects a ground plane if none
        exists. Subclasses can override (e.g. the satellite env skips
        the ground plane).
        """
        xml = re.sub(r"<camera\b[^/]*/>\s*", "", xml)
        xml = re.sub(r"<light\b[^/]*/>\s*", "", xml)
        camera = (
            '<camera name="side" pos="0 -4 3" ' 'xyaxes="1 0 0 0 0.45 1" fovy="60"/>'
        )
        light = (
            '<light name="top" pos="0 0 4" dir="0 0 -1" '
            'diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" directional="true"/>'
        )
        floor = ""
        if not re.search(r'<geom\b[^>]*type="plane"', xml):
            floor = (
                '\n  <geom name="floor" type="plane" size="10 10 0.1" '
                'rgba="0.8 0.85 0.8 1" conaffinity="1" condim="3"/>'
            )
        return xml.replace("<worldbody>", f"<worldbody>\n  {camera}\n  {light}{floor}")

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        """Per-env reward, shape ``(num_envs, 1)``, dtype :attr:`dtype`."""

    @abc.abstractmethod
    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        """Per-env termination flag, shape ``(num_envs, 1)``, dtype bool."""

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        """Default observation: ``cat(qpos[SKIP_QPOS:], qvel)``.

        Override for richer observations. The shape produced here must
        match :meth:`_make_obs_spec`.
        """
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        return torch.cat([qpos[..., self.SKIP_QPOS :], qvel], dim=-1)

    def _make_obs_spec(self) -> Composite:
        """Default obs spec matching the default :meth:`_make_obs`."""
        nq = self._backend.nq
        nv = self._backend.nv
        obs_dim = (nq - self.SKIP_QPOS) + nv
        return Composite(
            observation=Unbounded(
                shape=(self.num_envs, obs_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _pixels_spec(self) -> Bounded:
        return Bounded(
            low=0,
            high=255,
            shape=(self.num_envs, self.render_height, self.render_width, 3),
            dtype=torch.uint8,
            device=self.device,
        )

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(qpos, qvel)`` for ``n`` envs at reset.

        Default: ``qpos0`` plus uniform noise on both ``qpos`` and
        ``qvel``. Override to e.g. set fixed rotor speeds (satellite),
        or to read user-supplied keys from ``tensordict`` (e.g. a fixed
        starting attitude provided by a :class:`TensorDictPrimer`).

        ``tensordict`` is the same object passed to :meth:`_reset` and
        may carry extra keys like ``init_bus_quat`` etc. for environments
        that support starting-state overrides.
        """
        backend = self._backend
        qpos = backend.qpos0.unsqueeze(0).expand(n, -1).to(self.device).clone()
        qvel = backend.qvel0.unsqueeze(0).expand(n, -1).to(self.device).clone()
        if self.reset_noise_scale > 0:
            noise_q = torch.empty_like(qpos).uniform_(
                -self.reset_noise_scale, self.reset_noise_scale, generator=self.rng
            )
            noise_v = torch.empty_like(qvel).uniform_(
                -self.reset_noise_scale, self.reset_noise_scale, generator=self.rng
            )
            qpos = qpos + noise_q
            qvel = qvel + noise_v
        return qpos, qvel

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        """Map the agent action onto the simulator ``ctrl`` vector.

        Default: identity. Override for partial actuation (e.g. when the
        agent controls a subset of actuators while others are held
        constant -- satellite CMG rotors).
        """
        return action

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _render_pixels(self) -> torch.Tensor:
        """Render the pixel observation, honoring ``render_every``.

        Off-cadence steps return the previous frame unchanged (the cached
        tensor is replaced, never mutated in place, so handing it out
        repeatedly is safe). ``_reset`` clears the cache so the first frame
        of an episode is always fresh.
        """
        last = self._last_pixels
        if (
            self.render_every <= 1
            or last is None
            or last.shape[0] != self.num_envs
            or self._render_counter % self.render_every == 0
        ):
            last = self._backend.render(
                camera_id=self.camera_id,
                width=self.render_width,
                height=self.render_height,
                background=self.RENDER_BACKGROUND,
            )
            self._last_pixels = last
        return last

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        """Assemble the obs dict, including ``pixels`` when ``from_pixels``."""
        out: dict[str, torch.Tensor] = {}
        if not self.pixels_only:
            out["observation"] = self._make_obs(state)
        if self.from_pixels:
            out["pixels"] = self._render_pixels()
        return out

    def _make_specs(self) -> None:
        backend = self._backend
        obs_spec = self._make_obs_spec()
        if self.from_pixels:
            if self.pixels_only:
                obs_spec = Composite(
                    pixels=self._pixels_spec(),
                    shape=(self.num_envs,),
                    device=self.device,
                )
            else:
                obs_spec["pixels"] = self._pixels_spec()
        self.observation_spec = obs_spec
        # Action spec from actuator_ctrlrange (clamp infinities to a sane
        # default for unbounded actuators).
        lo = backend.actuator_lo.to(self.dtype).to(self.device)
        hi = backend.actuator_hi.to(self.dtype).to(self.device)
        # Mujoco emits 0/0 for limit-less actuators -- treat as +/-inf.
        unlimited = (lo == 0) & (hi == 0)
        lo = torch.where(unlimited, torch.full_like(lo, -1.0), lo)
        hi = torch.where(unlimited, torch.full_like(hi, 1.0), hi)
        # Broadcast over batch.
        lo_b = lo.unsqueeze(0).expand(self.num_envs, -1).clone()
        hi_b = hi.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.action_spec = Bounded(
            low=lo_b,
            high=hi_b,
            shape=(self.num_envs, backend.nu),
            dtype=self.dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(self.num_envs, 1), dtype=self.dtype, device=self.device
        )
        # Emit ``terminated`` (subclass-driven) and ``truncated``
        # (max_episode_steps-driven) as first-class keys so downstream
        # consumers (collectors, GAE) can tell the two apart instead of
        # only seeing the OR'd ``done`` flag.
        done_shape = (self.num_envs, 1)
        self.full_done_spec = Composite(
            done=Binary(n=1, shape=done_shape, dtype=torch.bool, device=self.device),
            terminated=Binary(
                n=1, shape=done_shape, dtype=torch.bool, device=self.device
            ),
            truncated=Binary(
                n=1, shape=done_shape, dtype=torch.bool, device=self.device
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _state_td(self) -> TensorDict:
        """Snapshot of the backend state as a TensorDict.

        Keys: ``qpos`` ``(num_envs, nq)``, ``qvel`` ``(num_envs, nv)``,
        ``time`` ``(num_envs,)``. Backend-internal dtype is preserved;
        cast inside :meth:`_make_obs` / :meth:`_compute_reward` if needed.
        """
        return TensorDict(
            {
                "qpos": self._backend.qpos,
                "qvel": self._backend.qvel,
                "time": self._backend.time,
            },
            batch_size=(self.num_envs,),
            device=self.device,
        )

    def get_state(self) -> TensorDict:
        """Return a detached snapshot of the MuJoCo simulator state.

        Returns:
            A TensorDict containing ``qpos``, ``qvel``, and ``time`` with the
            environment batch size.
        """
        return self._state_td().clone()

    def _index_extra_state(self, index: slice | torch.Tensor) -> dict[str, Any]:
        """Return subclass-owned batched state for an indexed env snapshot."""
        del index
        return {}

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        """Load subclass-owned state into an indexed env snapshot."""
        del state

    def _set_indexed_extra_state(
        self,
        index: slice | torch.Tensor,
        source: MujocoEnv,
    ) -> None:
        """Write subclass-owned state from an indexed snapshot into ``self``."""
        del index, source

    def __getitem__(self, item: Any) -> MujocoEnv:
        """Return a detached snapshot of one or more environments in the batch.

        The returned environment starts from the selected simulator state but is
        independent from the parent. Assign it back with ``env[item] = sub_env``
        to explicitly write its state back into the parent batch.
        """
        index, indexed_num_envs = _normalize_mujoco_index(
            item, self.num_envs, self.device
        )
        env = copy(self)
        was_locked = env.is_spec_locked
        if was_locked:
            env.set_spec_lock_(False)
        env.num_envs = indexed_num_envs
        env.__dict__["_input_spec"] = self.input_spec[index].clone()
        env.__dict__["_output_spec"] = self.output_spec[index].clone()
        env.batch_size = torch.Size([indexed_num_envs])
        env._backend = self._backend.clone_batch(index, indexed_num_envs)
        env._step_count = self._step_count[index].clone()
        env.rng = _clone_generator(self.rng, self.device)
        env._load_indexed_extra_state(self._index_extra_state(index))
        env.__dict__["_cache"] = {}
        if was_locked:
            env.set_spec_lock_(True)
        return env

    def __setitem__(self, item: Any, source: MujocoEnv) -> None:
        """Write an indexed MuJoCo env snapshot back into this env batch."""
        if not isinstance(source, MujocoEnv):
            raise TypeError(
                f"Expected a MujocoEnv source, got {type(source).__name__}."
            )
        index, indexed_num_envs = _normalize_mujoco_index(
            item, self.num_envs, self.device
        )
        if source.num_envs != indexed_num_envs:
            raise ValueError(
                "The source env batch does not match the indexed destination: "
                f"got source.num_envs={source.num_envs} and index selects "
                f"{indexed_num_envs} envs."
            )
        if self.backend_name != source.backend_name:
            raise TypeError(
                "Cannot write back a MuJoCo env snapshot from backend "
                f"{source.backend_name!r} into backend {self.backend_name!r}."
            )
        if (
            self._backend.nq != source._backend.nq
            or self._backend.nv != source._backend.nv
        ):
            raise ValueError(
                "Cannot write back MuJoCo envs with different state sizes."
            )
        self._backend.set_batch(index, source._backend)
        self._step_count[index] = source._step_count.to(
            device=self._step_count.device, dtype=self._step_count.dtype
        )
        self._set_indexed_extra_state(index, source)

    # ------------------------------------------------------------------
    # EnvBase interface
    # ------------------------------------------------------------------

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        # ``set_state`` is resolved by ``EnvBase.reset``: when truthy and the
        # input tensordict carries ``qpos``/``qvel``, reset deterministically to
        # that snapshot instead of sampling. Composes with a partial ``_reset``
        # mask (the backend muxes the masked rows).
        set_state = bool(kwargs.get("set_state"))
        reset_mask = None
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"]
            if reset_mask.ndim > 1:
                reset_mask = reset_mask.squeeze(-1)

        if (
            set_state
            and tensordict is not None
            and "qpos" in tensordict.keys()
            and "qvel" in tensordict.keys()
        ):
            qpos = tensordict.get("qpos").to(
                device=self.device, dtype=self._backend.qpos0.dtype
            )
            qvel = tensordict.get("qvel").to(
                device=self.device, dtype=self._backend.qvel0.dtype
            )
        else:
            # Always sample full-batch (num_envs, *) initial states; the
            # backend's ``reset_mask`` muxes the masked rows internally with
            # ``torch.where`` so this path is free of data-dependent shapes
            # (no ``.item()`` syncs, friendly to ``torch.compile``).
            qpos, qvel = self._sample_initial_state(self.num_envs, tensordict)
        if reset_mask is None:
            self._backend.reset(qpos, qvel)
            self._step_count.zero_()
            self._on_reset_all(tensordict)
        else:
            self._backend.reset_mask(reset_mask, qpos, qvel)
            self._step_count = torch.where(
                reset_mask, torch.zeros_like(self._step_count), self._step_count
            )
            self._on_reset_mask(reset_mask, tensordict)

        # Force a fresh render on the first frame of the episode.
        self._render_counter = 0
        self._last_pixels = None

        state = self._state_td()
        obs = self._build_obs_dict(state)
        zero_flag = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        out = TensorDict(
            {
                **obs,
                "done": zero_flag.clone(),
                "terminated": zero_flag.clone(),
                "truncated": zero_flag,
            },
            batch_size=(self.num_envs,),
            device=self.device,
        )
        return out

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        """Hook called after a full backend reset.

        Override for per-episode randomization that lives outside the
        simulator (e.g. sample a new target attitude). ``tensordict`` is
        the same object passed to :meth:`_reset`.
        """

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        """Hook for partial reset.

        Override alongside :meth:`_on_reset_all`. ``tensordict`` is the
        same object passed to :meth:`_reset`.
        """

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict["action"].to(self.dtype)
        ctrl = self._prepare_ctrl(action)

        state = self._state_td()
        self._backend.step(ctrl, self.frame_skip)
        next_state = self._state_td()
        self._step_count += 1
        self._render_counter += 1

        reward = self._compute_reward(state, action, next_state).to(self.dtype)
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        terminated = self._compute_done(state, next_state).to(torch.bool)
        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        done = terminated | truncated

        obs = self._build_obs_dict(next_state)
        return TensorDict(
            {
                **obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=(self.num_envs,),
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(seed))
        self.rng = rng

    def render(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        camera_id: int | None = None,
    ) -> torch.Tensor:
        """Render every env to an ``(num_envs, H, W, 3)`` ``uint8`` tensor.

        Pulls dimensions from the constructor kwargs by default; pass
        explicit values to render at a different resolution.
        """
        return self._backend.render(
            camera_id=self.camera_id if camera_id is None else int(camera_id),
            width=self.render_width if width is None else int(width),
            height=self.render_height if height is None else int(height),
            background=self.RENDER_BACKGROUND,
        )
