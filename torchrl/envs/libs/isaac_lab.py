# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Isaac Lab environment wrapper.

This module exposes :class:`IsaacLabWrapper`, a thin specialisation of
:class:`~torchrl.envs.libs.gym.GymWrapper` for Isaac Lab's vectorised
environments. In addition to the auto-reset / spec handling inherited from
the base wrapper, this module surfaces Isaac Lab's per-index reset and
``reset_to`` APIs through the standard torchrl ``_reset`` mask, so that a
caller can reset an arbitrary subset of sub-environments from the
tensordict (and have the transform stack -- ``RewardSum``, ``InitTracker``,
recurrent primers, ``VecNormV2``, ... -- fire correctly on the reset
indices only).
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any, Literal

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Unbounded
from torchrl.envs.libs.gym import GymWrapper

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None
_has_isaaclab_newton = importlib.util.find_spec("isaaclab_newton") is not None
_has_isaaclab_ov = importlib.util.find_spec("isaaclab_ov") is not None


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment
            instance to wrap. ``ManagerBasedEnv``, ``ManagerBasedRLEnv``,
            ``DirectRLEnv`` and ``DirectMARLEnv`` are all supported.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``True``.
        native_autoreset (bool, optional): if ``True``, keeps Isaac Lab's native
            auto-reset observations in the collector hot path and avoids the
            synthetic reset call in :meth:`~torchrl.envs.EnvBase.step_and_maybe_reset`.
            The terminal ``"next"`` observation remains unavailable and is
            marked with ``NaN``; the native reset observation is cloned into
            the next root observation.
            Defaults to ``False``.
        from_tiled_camera (bool, optional): if ``True``, reads pixels from an
            Isaac Lab tiled camera sensor and writes them under ``pixels_key``.
            This is the recommended headless rendering path. Defaults to
            ``False``.
        tiled_camera_name (str, optional): Name of the sensor in
            ``env.scene.sensors``. Defaults to ``"tiled_camera"``.
        tiled_camera_data_type (str, optional): Camera data type to read.
            Defaults to ``"rgb"``.
        pixels_key (NestedKey, optional): TensorDict key where pixels are
            written. Defaults to ``"pixels"``.
        pixels_dtype (torch.dtype, optional): dtype used for the output pixels.
            If ``torch.uint8`` is requested and the camera returns floating
            point data, values are scaled from ``[0, 1]`` to ``[0, 255]``.
            Defaults to ``torch.uint8``.
        pixels_channels (int, optional): Number of channels to keep from the
            camera output. Defaults to ``3``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab doc for installation instructions <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_.

    Per-index reset
    ---------------

    Isaac Lab's underlying envs let the caller reset an arbitrary subset of
    sub-environments without disturbing the others. This wrapper plumbs that
    capability through the standard torchrl ``"_reset"`` mask: when the
    tensordict passed to :meth:`reset` carries a partial ``"_reset"`` boolean
    mask (i.e. neither all ``True`` nor all ``False``), only the masked
    sub-envs are reset and the others keep their state and ``episode_length_buf``.
    The transform stack (``RewardSum``, ``InitTracker``, recurrent primers,
    ``VecNormV2``, ...) fires on the reset rows only, exactly like a normal
    reset.

    The per-index reset path is gated on ``native_autoreset=True``: with the
    default ``native_autoreset=False`` it would conflict with the
    :class:`~torchrl.envs.transforms.VecGymEnvTransform`-based obs-swap path
    that :meth:`~torchrl.envs.EnvBase.step_and_maybe_reset` triggers on every
    "done" row (this would double-reset the affected envs). Set
    ``native_autoreset=True`` if you want partial-reset semantics.

    State-based reset
    -----------------

    For deterministic branching from a snapshot, snapshot the scene with
    :meth:`get_state` and restore it with ``env.reset(td, set_state=True,
    scene_state=snapshot)`` (or the :meth:`reset_to_state` convenience, which
    routes through the same path). Manager-based envs only; both work in
    conjunction with the partial ``"_reset"`` mask.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    # Supports deterministic resets via ``reset(td, set_state=True,
    # scene_state=...)`` (manager-based envs). The snapshot is honored through
    # Isaac's ``reset_to`` rather than from tensordict state entries.
    _supports_set_state = True

    def __init__(
        self,
        env: isaaclab.envs.ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        native_autoreset: bool = False,
        from_tiled_camera: bool = False,
        tiled_camera_name: str = "tiled_camera",
        tiled_camera_data_type: str = "rgb",
        pixels_key: NestedKey = "pixels",
        pixels_dtype: torch.dtype | None = torch.uint8,
        pixels_channels: int | None = 3,
        **kwargs,
    ):
        self.from_tiled_camera = from_tiled_camera
        self.tiled_camera_name = tiled_camera_name
        self.tiled_camera_data_type = tiled_camera_data_type
        self.pixels_key = pixels_key
        self.pixels_dtype = pixels_dtype
        self.pixels_channels = pixels_channels
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )

    @staticmethod
    def add_tiled_camera_config(
        env_cfg,
        *,
        sensor_name: str = "tiled_camera",
        data_type: str = "rgb",
        renderer_backend: Literal["isaac_rtx", "newton_warp", "ovrtx"] | None = None,
        renderer_cfg: object | None = None,
        width: int = 320,
        height: int = 240,
        pos: tuple[float, float, float] = (-7.0, 0.0, 3.0),
        rot: tuple[float, float, float, float] = (0.9945, 0.0, 0.1045, 0.0),
        convention: Literal["opengl", "ros", "world"] = "world",
        focal_length: float = 24.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.955,
        clipping_range: tuple[float, float] = (0.1, 100.0),
        render_interval: int | None = None,
    ):
        """Attach an Isaac Lab :class:`~isaaclab.sensors.TiledCameraCfg`.

        This helper mutates an Isaac Lab environment config before the env is
        instantiated, making headless RGB capture available through
        :class:`IsaacLabWrapper` with ``from_tiled_camera=True``.

        Args:
            env_cfg: Isaac Lab environment config to mutate.
            sensor_name (str, optional): Name used in ``env.scene.sensors``.
                Defaults to ``"tiled_camera"``.
            data_type (str, optional): Camera output data type. Defaults to
                ``"rgb"``.
            renderer_backend (str, optional): Renderer backend to use for the
                tiled camera. ``"isaac_rtx"`` keeps Isaac Lab's default RTX
                renderer, ``"newton_warp"`` uses the Isaac Lab Newton Warp
                renderer, and ``"ovrtx"`` uses the OVRTX renderer. Defaults to
                ``None``, which keeps the Isaac Lab default.
            renderer_cfg (object, optional): Explicit renderer configuration
                object to pass to :class:`~isaaclab.sensors.TiledCameraCfg`.
                This cannot be combined with ``renderer_backend``.
            width (int, optional): Camera image width. Defaults to ``320``.
            height (int, optional): Camera image height. Defaults to ``240``.
            pos (tuple of float, optional): Camera position offset.
            rot (tuple of float, optional): Camera orientation offset.
            convention (str, optional): Orientation convention. Defaults to
                ``"world"``.
            focal_length (float, optional): Pinhole camera focal length.
            focus_distance (float, optional): Pinhole camera focus distance.
            horizontal_aperture (float, optional): Pinhole camera aperture.
            clipping_range (tuple of float, optional): Near and far clipping
                planes.
            render_interval (int, optional): Simulation render interval. If
                omitted, the existing config value is kept.

        Returns:
            The mutated ``env_cfg``.

        Examples:
            >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
            >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
            >>> cfg = IsaacLabWrapper.add_tiled_camera_config(AntEnvCfg())
        """
        if not _has_isaaclab:
            raise ImportError("Isaac Lab is required to add a tiled camera config.")
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        if renderer_backend is not None and renderer_cfg is not None:
            raise ValueError(
                "Only one of renderer_backend or renderer_cfg can be provided."
            )
        if renderer_backend == "newton_warp":
            if not _has_isaaclab_newton:
                raise ImportError(
                    "Isaac Lab Newton is required to use "
                    "renderer_backend='newton_warp'."
                )
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            renderer_cfg = NewtonWarpRendererCfg()
        elif renderer_backend == "ovrtx":
            if not _has_isaaclab_ov:
                raise ImportError(
                    "Isaac Lab OVRTX is required to use renderer_backend='ovrtx'."
                )
            from isaaclab_ov.renderers import OVRTXRendererCfg

            renderer_cfg = OVRTXRendererCfg()

        camera_kwargs = {}
        if renderer_cfg is not None:
            camera_kwargs["renderer_cfg"] = renderer_cfg
        camera_cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos,
                rot=rot,
                convention=convention,
            ),
            data_types=[data_type],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            ),
            width=width,
            height=height,
            **camera_kwargs,
        )
        setattr(env_cfg.scene, sensor_name, camera_cfg)
        if render_interval is not None:
            env_cfg.sim.render_interval = render_interval
        return env_cfg

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _make_specs(self, env, batch_size=None) -> None:
        super()._make_specs(env, batch_size=batch_size)
        if not self.from_tiled_camera:
            return
        camera = self._get_tiled_camera()
        cfg = camera.cfg
        channels = self.pixels_channels
        if channels is None:
            if self.tiled_camera_data_type == "rgb":
                channels = 3
            else:
                channels = camera.data.output[self.tiled_camera_data_type].shape[-1]
        dtype = self.pixels_dtype
        if dtype is None:
            dtype = camera.data.output[self.tiled_camera_data_type].dtype
        shape = (*self.batch_size, cfg.height, cfg.width, channels)
        if dtype == torch.uint8:
            pixels_spec = Bounded(0, 255, shape=shape, dtype=dtype, device=self.device)
        else:
            pixels_spec = Unbounded(shape=shape, dtype=dtype, device=self.device)
        self.observation_spec[self.pixels_key] = pixels_spec

    def _get_tiled_camera(self):
        env = self._env.unwrapped
        scene = env.scene
        try:
            return scene.sensors[self.tiled_camera_name]
        except KeyError as err:
            raise KeyError(
                f"Could not find tiled camera sensor {self.tiled_camera_name!r} "
                "in env.scene.sensors. Add one to the Isaac Lab config before "
                "calling gym.make, for example with "
                "IsaacLabWrapper.add_tiled_camera_config(...)."
            ) from err

    def _read_tiled_camera_pixels(self) -> torch.Tensor:
        pixels = self._get_tiled_camera().data.output[self.tiled_camera_data_type]
        pixels = torch.as_tensor(pixels, device=self.device)
        if self.pixels_channels is not None:
            pixels = pixels[..., : self.pixels_channels]
        if self.pixels_dtype is not None and pixels.dtype != self.pixels_dtype:
            if self.pixels_dtype == torch.uint8 and pixels.dtype.is_floating_point:
                pixels = pixels.mul(255).clamp(0, 255)
            pixels = pixels.to(self.pixels_dtype)
        return pixels

    def _add_tiled_camera_pixels(self, observations):
        if not self.from_tiled_camera:
            return self._normalize_observation_keys(observations)
        if isinstance(observations, Mapping):
            observations = dict(observations)
        else:
            observations = {"observation": observations}
        observations[self.pixels_key] = self._read_tiled_camera_pixels()
        return self._normalize_observation_keys(observations)

    def _normalize_observation_keys(self, observations):
        if not isinstance(observations, Mapping):
            return observations
        spec_keys = set(self.observation_spec.keys(True, True))
        if "policy" in observations and "policy" not in spec_keys:
            if "observation" in spec_keys:
                observations = dict(observations)
                observations["observation"] = observations.pop("policy")
        return observations

    def _reset_output_transform(self, reset_data):  # noqa: F811
        observations, info = super()._reset_output_transform(reset_data)
        return self._add_tiled_camera_pixels(observations), info

    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        observations = self._add_tiled_camera_pixels(observations)
        done = terminated | truncated
        reward = reward.unsqueeze(-1)  # to get to (num_envs, 1)
        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )

    # ------------------------------------------------------------------
    # Isaac Lab env detection
    # ------------------------------------------------------------------

    @staticmethod
    def _supported_isaac_env_classes() -> tuple[type, ...]:
        """Returns the tuple of Isaac Lab env classes this wrapper can bridge.

        ``ManagerBasedEnv`` and ``ManagerBasedRLEnv`` (subclass) expose
        ``reset(env_ids=..., seed=..., options=...)`` and ``reset_to``.
        ``DirectRLEnv`` and ``DirectMARLEnv`` only expose ``_reset_idx``;
        for those we rebuild the post-reset observation manually.
        """
        from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv

        return (ManagerBasedEnv, DirectRLEnv, DirectMARLEnv)

    @classmethod
    def _supports_native_autoreset(cls, env: Any) -> bool:
        """Return ``True`` iff ``env`` (assumed already unwrapped) is a batched Isaac Lab env.

        This is the single source of truth used by
        :class:`~torchrl.envs.libs.gym._GymAsyncMeta` to decide whether to
        install the :class:`~torchrl.envs.transforms.VecGymEnvTransform`
        adapter and register the ``_torchrl_native_autoreset`` flag for an
        Isaac Lab env (regardless of whether the env was wrapped via
        :class:`IsaacLabWrapper` or the generic :class:`GymWrapper`).
        """
        if not _has_isaaclab:
            return False
        return isinstance(env, cls._supported_isaac_env_classes())

    @property
    def _isaac_unwrapped(self):
        return self._env.unwrapped

    # ------------------------------------------------------------------
    # Per-index reset bridge
    # ------------------------------------------------------------------

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        # Deterministic state-based reset is the single ``set_state=True`` path
        # (``set_state`` is resolved by ``EnvBase.reset``). The snapshot is passed
        # as the ``scene_state`` kwarg (from :meth:`get_state`) rather than via
        # the tensordict, since it carries opaque simulator state. When set, it
        # overrides the regular reset path entirely (Isaac Lab's ``reset_to``).
        set_state = kwargs.pop("set_state", None)
        scene_state = kwargs.pop("scene_state", None)
        is_relative = kwargs.pop("is_relative", False)

        if scene_state is not None and not set_state:
            raise ValueError(
                "A `scene_state` snapshot was passed to reset() without "
                "`set_state=True`. Pass `set_state=True` to deterministically "
                "reset to the snapshot."
            )

        reset = None
        if tensordict is not None:
            reset = tensordict.get("_reset", None)

        if set_state:
            if scene_state is None:
                raise ValueError(
                    "reset(set_state=True) on an Isaac Lab env requires a "
                    "`scene_state` snapshot (obtained from `env.get_state()`)."
                )
            return self._reset_to_state_at(
                scene_state, reset=reset, is_relative=is_relative, **kwargs
            )

        if not self._is_batched:
            return super()._reset(tensordict, **kwargs)

        if reset is None or reset.all():
            # Full reset: defer to GymWrapper / GymLikeEnv.
            if tensordict is not None and "_reset" in tensordict.keys():
                tensordict = tensordict.exclude("_reset")
            return super()._reset(tensordict, **kwargs)

        if not reset.any():
            # Nothing to reset: return a sentinel reset tensordict so the
            # surrounding _reset_proc_data / _update_during_reset machinery
            # preserves the incoming state on every sub-env.
            return tensordict.exclude("_reset")

        # Per-index reset path. Gated on native_autoreset=True: with
        # native_autoreset=False, partial-mask reset calls are issued by
        # EnvBase.maybe_reset on every step that has any "done" row, and
        # the surrounding VecGymEnvTransform already handles the obs swap
        # without re-entering Isaac. Firing unwrapped.reset(env_ids=...)
        # here would double-reset those envs. We keep the historical
        # no-op semantics in that case so explicit partial resets
        # remain a feature you opt into via native_autoreset=True.
        if not self._native_autoreset_enabled:
            return tensordict.exclude("_reset")

        env_ids = self._reset_mask_to_env_ids(reset)
        obs, info = self._partial_reset(env_ids=env_ids, **kwargs)
        return self._build_reset_tensordict(obs, info)

    def _input_td_has_state(self, tensordict: TensorDictBase | None) -> bool:
        # Isaac Lab honors a ``scene_state`` snapshot passed as a reset kwarg,
        # never tensordict state entries, so a reset tensordict carrying
        # observations must not trigger the implicit-state transition warning.
        return False

    @property
    def _native_autoreset_enabled(self) -> bool:
        """Whether the wrapping pipeline has ``native_autoreset=True``.

        Set on the instance by :class:`~torchrl.envs.libs.gym._GymAsyncMeta`
        when the wrapper is constructed via :class:`IsaacLabWrapper(env,
        native_autoreset=True)`. Mirrored on the surrounding
        :class:`~torchrl.envs.TransformedEnv` via ``_torchrl_native_autoreset``.
        """
        return self.__dict__.get("_torchrl_native_autoreset", False)

    def _reset_mask_to_env_ids(self, reset: torch.Tensor) -> torch.Tensor:
        """Convert a ``_reset`` boolean mask to a 1-D ``env_ids`` tensor."""
        # The mask is shaped (num_envs, 1) or (num_envs,) and may live on a
        # different device than the underlying Isaac env (e.g. CPU in tests).
        return (
            reset.reshape(-1).nonzero(as_tuple=True)[0].to(self._isaac_unwrapped.device)
        )

    def _partial_reset(
        self,
        *,
        env_ids: torch.Tensor,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> tuple[Any, dict | None]:
        """Reset the listed sub-envs without touching the rest."""
        from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv

        unwrapped = self._isaac_unwrapped
        if isinstance(unwrapped, ManagerBasedEnv):
            reset_kwargs: dict[str, Any] = {"env_ids": env_ids}
            if seed is not None:
                reset_kwargs["seed"] = seed
            if options is not None:
                reset_kwargs["options"] = options
            return unwrapped.reset(**reset_kwargs)

        if isinstance(unwrapped, (DirectRLEnv, DirectMARLEnv)):
            # DirectRLEnv.reset() does not accept env_ids -- fall back to the
            # internal _reset_idx primitive and replay the post-reset bookkeeping
            # that DirectRLEnv / DirectMARLEnv reset() would normally do.
            if seed is not None:
                unwrapped.seed(seed)
            unwrapped._reset_idx(env_ids)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            return unwrapped._get_observations(), unwrapped.extras

        raise TypeError(
            f"Per-index reset is not supported for Isaac Lab env of type "
            f"{type(unwrapped).__name__}. Supported bases are ManagerBasedEnv "
            "(and subclasses), DirectRLEnv and DirectMARLEnv."
        )

    def _build_reset_tensordict(self, obs: Any, info: dict | None) -> TensorDictBase:
        """Rebuild a torchrl-style reset tensordict from Isaac's (obs, info)."""
        obs = self._add_tiled_camera_pixels(obs)
        source = self.read_obs(obs)
        tensordict_out = TensorDict(source=source, batch_size=self.batch_size)
        if self.info_dict_reader and info is not None:
            for info_dict_reader in self.info_dict_reader:
                out = info_dict_reader(info, tensordict_out)
                if out is not None:
                    tensordict_out = out
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)
        return tensordict_out

    # ------------------------------------------------------------------
    # State-based reset bridge
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Return the current Isaac Lab scene state.

        This is exactly what ``InteractiveScene.get_state()`` returns on the
        underlying env. It can later be passed back via
        ``env.reset(td, set_state=True, scene_state=...)`` (or the
        :meth:`reset_to_state` convenience) to deterministically branch back to
        this checkpoint.

        Returns:
            The scene-state dict (as defined by Isaac Lab; not converted).
        """
        return self._isaac_unwrapped.scene.get_state()

    def reset_to_state(
        self,
        state: Any,
        tensordict: TensorDictBase | None = None,
        *,
        is_relative: bool = False,
        seed: int | None = None,
    ) -> TensorDictBase:
        """Deterministically reset to ``state`` (per-index when a ``_reset`` mask is set).

        ``state`` is the dict format returned by :meth:`get_state` (i.e.
        ``InteractiveScene.get_state()``). Only manager-based envs are
        supported (they are the only ones that expose ``reset_to``).

        Args:
            state: scene state to restore.
            tensordict: optional input tensordict. If it contains a ``"_reset"``
                mask, only the masked sub-envs are restored; otherwise every
                sub-env is restored.

        Keyword Args:
            is_relative: if ``True``, ``state`` is interpreted relative to the
                env origin (matches Isaac Lab's ``reset_to(is_relative=True)``).
                Defaults to ``False``.
            seed: optional seed forwarded to Isaac's ``reset_to``.

        Returns:
            A reset tensordict in torchrl's standard shape.

        .. note::
            This is a thin convenience around the unified deterministic-reset
            path; it is exactly equivalent to::

                env.reset(td, set_state=True, scene_state=state, is_relative=...)

            If the env is wrapped in a :class:`~torchrl.envs.TransformedEnv` and
            the transform stack (``RewardSum``, ``InitTracker``, primers,
            ``VecNormV2``, ...) must fire on the restored rows, call that
            ``reset(...)`` form on the top-level env so it routes through
            :meth:`TransformedEnv._reset` and triggers every transform on the
            way down.
        """
        kwargs: dict[str, Any] = {"scene_state": state, "is_relative": is_relative}
        if seed is not None:
            kwargs["seed"] = seed
        return self.reset(tensordict, set_state=True, **kwargs)

    def _reset_to_state_at(
        self,
        state: Any,
        *,
        reset: torch.Tensor | None,
        is_relative: bool,
        seed: int | None = None,
        **kwargs,
    ) -> TensorDictBase:
        unwrapped = self._isaac_unwrapped
        if not hasattr(unwrapped, "reset_to"):
            raise RuntimeError(
                f"Isaac Lab env of type {type(unwrapped).__name__} does not "
                "expose reset_to (only manager-based envs do). For state-based "
                "reset on Direct envs, branch from the underlying scene state "
                "by hand."
            )
        if reset is None:
            env_ids = torch.arange(
                self.batch_size.numel(),
                device=unwrapped.device,
                dtype=torch.long,
            )
        else:
            env_ids = self._reset_mask_to_env_ids(reset)
            state = self._index_state_for_env_ids(state, env_ids)
        reset_to_kwargs: dict[str, Any] = {
            "env_ids": env_ids,
            "is_relative": is_relative,
        }
        if seed is not None:
            reset_to_kwargs["seed"] = seed
        obs, info = unwrapped.reset_to(state, **reset_to_kwargs)
        return self._build_reset_tensordict(obs, info)

    def _index_state_for_env_ids(self, state: Any, env_ids: torch.Tensor) -> Any:
        num_envs = self.batch_size.numel()
        if isinstance(state, torch.Tensor):
            if state.shape[:1] == (num_envs,):
                return state.index_select(0, env_ids.to(state.device))
            return state
        if isinstance(state, Mapping):
            return {
                key: self._index_state_for_env_ids(value, env_ids)
                for key, value in state.items()
            }
        if isinstance(state, tuple):
            return tuple(self._index_state_for_env_ids(item, env_ids) for item in state)
        if isinstance(state, list):
            return [self._index_state_for_env_ids(item, env_ids) for item in state]
        return state
