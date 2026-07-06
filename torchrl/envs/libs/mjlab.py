# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab environment wrapper.

This module exposes :class:`MJLabWrapper`, a TorchRL adapter for
``mjlab.envs.ManagerBasedRlEnv``, and :class:`MJLabEnv`, a convenience
constructor that builds an mjlab task from mjlab's task registry.
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import _EnvPostInit, _EnvWrapper
from torchrl.envs.utils import _classproperty

_has_mjlab = importlib.util.find_spec("mjlab") is not None

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _canonical_device(device: DEVICE_TYPING | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return device


def _device_from_mjlab(device: DEVICE_TYPING | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _canonical_device(device)


def _space_dtype(space: Any) -> torch.dtype:
    dtype = getattr(space, "dtype", torch.float32)
    if isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPE_MAP.get(str(dtype), torch.float32)


def _space_shape(space: Any) -> tuple[int, ...]:
    shape = getattr(space, "shape", ())
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _is_dict_space(space: Any) -> bool:
    spaces = getattr(space, "spaces", None)
    return isinstance(spaces, Mapping)


def _finite_bound(bound: Any) -> bool:
    try:
        tensor = torch.as_tensor(bound)
    except (TypeError, ValueError):
        return False
    if tensor.numel() == 0:
        return True
    return bool(torch.isfinite(tensor).all().item())


def _make_spec_from_space(
    space: Any,
    *,
    batch_size: torch.Size,
    device: torch.device,
) -> TensorSpec:
    if _is_dict_space(space):
        entries = {
            key: _make_spec_from_space(
                subspace,
                batch_size=batch_size,
                device=device,
            )
            for key, subspace in space.spaces.items()
        }
        return Composite(**entries, shape=batch_size, device=device)

    shape = (*batch_size, *_space_shape(space))
    dtype = _space_dtype(space)
    low = getattr(space, "low", -math.inf)
    high = getattr(space, "high", math.inf)
    if _finite_bound(low) and _finite_bound(high):
        return Bounded(low=low, high=high, shape=shape, dtype=dtype, device=device)
    return Unbounded(shape=shape, dtype=dtype, device=device)


def _single_space(env: Any, name: str) -> Any:
    single_name = f"single_{name}_space"
    if hasattr(env, single_name):
        return getattr(env, single_name)
    return getattr(env, f"{name}_space")


def _to_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def _format_done(value: Any, *, batch_size: torch.Size, device: torch.device):
    tensor = _to_tensor(value, device=device, dtype=torch.bool)
    return tensor.reshape(*batch_size, 1)


def _format_reward(value: Any, *, batch_size: torch.Size, device: torch.device):
    tensor = _to_tensor(value, device=device, dtype=torch.float32)
    return tensor.reshape(*batch_size, 1)


def _validate_batch_size(batch_size: torch.Size, num_envs: int) -> torch.Size:
    if len(batch_size) != 1:
        raise ValueError(
            "mjlab exposes a flat vectorized environment. "
            f"Expected a one-dimensional batch_size of ({num_envs},), "
            f"got {tuple(batch_size)}."
        )
    if batch_size[0] != num_envs:
        raise ValueError(
            f"batch_size={tuple(batch_size)} does not match the mjlab env's "
            f"num_envs={num_envs}. Pass batch_size=[{num_envs}] or omit it."
        )
    return batch_size


def _sensor_name(sensor: Any) -> str | None:
    cfg = getattr(sensor, "cfg", sensor)
    return getattr(cfg, "prefixed_name", getattr(cfg, "name", None))


def _supports_rgb_camera_sensor(sensor: Any) -> bool:
    cfg = getattr(sensor, "cfg", sensor)
    data_types = getattr(cfg, "data_types", ())
    return "rgb" in data_types


def _rgb_camera_sensor_names_from_cfg(cfg: Any) -> list[str]:
    scene = getattr(cfg, "scene", None)
    sensors = getattr(scene, "sensors", ())
    names = []
    for sensor in sensors:
        if _supports_rgb_camera_sensor(sensor):
            name = _sensor_name(sensor)
            if name is not None:
                names.append(name)
    return names


def _cfg_has_rgb_camera_sensor(cfg: Any, pixels_sensor: str | None) -> bool:
    names = _rgb_camera_sensor_names_from_cfg(cfg)
    if pixels_sensor is None:
        return bool(names)
    return pixels_sensor in names


def _env_sensor_mapping(env: Any) -> Mapping[str, Any]:
    scene = getattr(env, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if isinstance(sensors, Mapping):
        return sensors
    return {}


def _select_rgb_camera_sensor(env: Any, pixels_sensor: str | None) -> str | None:
    sensors = _env_sensor_mapping(env)
    if pixels_sensor is not None:
        if pixels_sensor not in sensors:
            raise ValueError(
                f"pixels_sensor={pixels_sensor!r} was not found in the mjlab "
                f"scene sensors. Available sensors: {sorted(sensors)}."
            )
        if not _supports_rgb_camera_sensor(sensors[pixels_sensor]):
            raise ValueError(
                f"pixels_sensor={pixels_sensor!r} does not expose RGB data. "
                "Use an mjlab CameraSensorCfg with data_types containing 'rgb'."
            )
        return pixels_sensor

    candidates = [
        name for name, sensor in sensors.items() if _supports_rgb_camera_sensor(sensor)
    ]
    if len(candidates) > 1:
        raise ValueError(
            "from_pixels=True found multiple mjlab RGB camera sensors "
            f"({sorted(candidates)}). Pass pixels_sensor=... to select one."
        )
    if candidates:
        return candidates[0]
    return None


class MJLabWrapper(_EnvWrapper):
    """TorchRL wrapper for a pre-built ``mjlab.envs.ManagerBasedRlEnv``.

    Args:
        env: The mjlab manager-based RL environment to wrap.
        device: Torch device for actions and returned tensors. When ``None``,
            it is inferred from ``env.device``. The TorchRL device must match
            the mjlab simulation device; a mismatch raises an error instead of
            silently copying tensors across devices in the hot path.
        batch_size: Batch size of the wrapper. When ``None``, it is inferred as
            ``[env.num_envs]``. mjlab exposes a flat vectorized batch, so only
            one-dimensional batch sizes are accepted.
        native_autoreset: If ``False`` (default), TorchRL drives resets through
            the standard ``"_reset"`` mask by setting ``env.cfg.auto_reset`` to
            ``False`` and calling ``env.reset(env_ids=...)`` on done rows. If
            ``True``, mjlab's native auto-reset is kept and TorchRL marks the
            terminal ``"next"`` observation as invalid while carrying mjlab's
            post-reset observation into the next root tensordict, matching the
            native-auto-reset contract used by Isaac Lab.
        from_pixels: If ``True``, append RGB pixels under ``pixels_key`` at
            reset and step time. If the mjlab scene has an RGB
            ``CameraSensor``, TorchRL uses the sensor's batched output with
            shape ``[num_envs, H, W, 3]``. Otherwise it falls back to
            :meth:`render`, which requires ``num_envs == 1`` and
            ``render_mode="rgb_array"`` because mjlab viewer rendering returns
            one frame for the whole scene rather than one image per
            environment row.
        pixels_only: If ``True``, return only the pixel observation. Requires
            ``from_pixels=True``.
        pixels_key: TensorDict key for pixel observations. Defaults to
            ``"pixels"``.
        pixels_sensor: Name of the mjlab RGB ``CameraSensor`` to use for
            ``from_pixels=True``. If ``None`` and exactly one RGB camera sensor
            is present, it is selected automatically.
        allow_done_after_reset: Passed to :class:`~torchrl.envs.EnvBase`.
            Defaults to ``False``.

    mjlab reference: Zakka et al., "mjlab: A Lightweight Framework for
    GPU-Accelerated Robot Learning", arXiv:2601.22074.

    Examples:
        >>> from mjlab.envs import ManagerBasedRlEnv  # doctest: +SKIP
        >>> from mjlab.tasks.registry import load_env_cfg  # doctest: +SKIP
        >>> from torchrl.envs import MJLabWrapper  # doctest: +SKIP
        >>> cfg = load_env_cfg("Mjlab-Cartpole-Balance")  # doctest: +SKIP
        >>> cfg.scene.num_envs = 4  # doctest: +SKIP
        >>> base_env = ManagerBasedRlEnv(cfg, device="cuda:0")  # doctest: +SKIP
        >>> env = MJLabWrapper(base_env)  # doctest: +SKIP
        >>> td = env.rollout(10)  # doctest: +SKIP
    """

    git_url = "https://github.com/mujocolab/mjlab"
    libname = "mjlab"
    _has_frame_skip = True

    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib
        import mjlab

        cls._lib = mjlab
        return mjlab

    @_classproperty
    def available_envs(cls) -> list[str]:
        if not _has_mjlab:
            return []
        import mjlab.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks

        return list_tasks()

    def __init__(
        self,
        env: Any = None,
        *,
        device: DEVICE_TYPING | None = None,
        batch_size: torch.Size | list[int] | tuple[int, ...] | None = None,
        native_autoreset: bool = False,
        from_pixels: bool = False,
        pixels_only: bool = False,
        pixels_key: NestedKey = "pixels",
        pixels_sensor: str | None = None,
        allow_done_after_reset: bool = False,
        **kwargs,
    ) -> None:
        if env is None:
            raise TypeError("MJLabWrapper requires a pre-built mjlab env.")

        env_device = _canonical_device(env.device)
        if device is None:
            device = env_device
        else:
            device = _canonical_device(device)
            if device != env_device:
                raise ValueError(
                    f"device={device} does not match the wrapped mjlab env device "
                    f"{env_device}. Build the mjlab env on the requested device "
                    "or omit the device argument."
                )

        num_envs = int(env.num_envs)
        if batch_size is None:
            batch_size = torch.Size([num_envs])
        else:
            batch_size = _validate_batch_size(torch.Size(batch_size), num_envs)

        self._from_pixels = bool(from_pixels)
        self._pixels_only = bool(pixels_only)
        self._pixels_key = pixels_key
        self._pixels_sensor_name = pixels_sensor
        self._pixels_source: str | None = None
        self._native_autoreset = bool(native_autoreset)
        kwargs["env"] = env
        kwargs["from_pixels"] = from_pixels
        kwargs["pixels_only"] = pixels_only
        kwargs["pixels_sensor"] = pixels_sensor
        kwargs["native_autoreset"] = native_autoreset
        super().__init__(
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
            **kwargs,
        )
        self._torchrl_native_autoreset = bool(native_autoreset)

    def _check_kwargs(self, kwargs: dict) -> None:
        if "env" not in kwargs:
            raise TypeError("Expected an 'env' keyword argument.")
        env = kwargs["env"]
        for attr in ("reset", "step", "render", "num_envs", "device"):
            if not hasattr(env, attr):
                raise TypeError(
                    f"env is missing required attribute {attr!r}. Expected a "
                    "mjlab.envs.ManagerBasedRlEnv-compatible object."
                )

    def _build_env(
        self,
        env: Any,
        *,
        from_pixels: bool = False,
        pixels_only: bool = False,
        pixels_sensor: str | None = None,
        native_autoreset: bool = False,
        **kwargs,
    ) -> Any:
        if kwargs:
            raise ValueError(f"Unsupported kwargs: {sorted(kwargs)}")
        if pixels_only and not from_pixels:
            raise ValueError("pixels_only=True requires from_pixels=True.")
        if pixels_sensor is not None and not from_pixels:
            raise ValueError("pixels_sensor requires from_pixels=True.")
        if from_pixels:
            selected_sensor = _select_rgb_camera_sensor(env, pixels_sensor)
            if selected_sensor is not None:
                self._pixels_source = "sensor"
                self._pixels_sensor_name = selected_sensor
            else:
                self._pixels_source = "render"
                if self.batch_size != torch.Size([1]):
                    raise ValueError(
                        "from_pixels=True with num_envs > 1 requires an mjlab "
                        "RGB CameraSensor because mjlab.render() returns one "
                        "viewer frame for the whole scene, not one image per "
                        "environment row. Add a CameraSensorCfg with "
                        "data_types containing 'rgb', or pass pixels_sensor=... "
                        "if one is already configured."
                    )
                if getattr(env, "render_mode", None) != "rgb_array":
                    raise ValueError(
                        "from_pixels=True without an RGB CameraSensor requires "
                        "the mjlab env to be built with render_mode='rgb_array'. "
                        "Use MJLabEnv(..., from_pixels=True) or construct "
                        "ManagerBasedRlEnv(..., render_mode='rgb_array') before "
                        "wrapping."
                    )

        cfg = getattr(env, "cfg", None)
        if cfg is not None and hasattr(cfg, "auto_reset"):
            cfg.auto_reset = bool(native_autoreset)
        return env

    def _init_env(self) -> int | None:
        return None

    def _make_specs(self, env: Any) -> None:
        self.observation_spec = self._make_observation_spec(env)
        self.action_spec = _make_spec_from_space(
            _single_space(env, "action"), batch_size=self.batch_size, device=self.device
        )
        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1), dtype=torch.float32, device=self.device
        )
        done_spec = Categorical(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_spec.clone(),
            terminated=done_spec.clone(),
            truncated=done_spec.clone(),
            shape=self.batch_size,
            device=self.device,
        )

    def _make_observation_spec(self, env: Any) -> Composite:
        obs_spec = _make_spec_from_space(
            _single_space(env, "observation"),
            batch_size=self.batch_size,
            device=self.device,
        )
        if not isinstance(obs_spec, Composite):
            obs_spec = Composite(observation=obs_spec, shape=self.batch_size)
        if self._from_pixels:
            pixels_spec = self._make_pixels_spec()
            if self._pixels_only:
                obs_spec = Composite(shape=self.batch_size, device=self.device)
            obs_spec[self._pixels_key] = pixels_spec
        return obs_spec

    def _obs_to_tensordict(self, obs: Any) -> TensorDictBase:
        if not isinstance(obs, Mapping):
            obs = {"observation": obs}
        if self._pixels_only:
            source: dict[str, Any] = {}
        else:
            source = self._map_obs(obs)
        td = TensorDict(source, batch_size=self.batch_size, device=self.device)
        if self._from_pixels:
            td.set(self._pixels_key, self._get_pixels())
        return td

    def _map_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        source = {}
        for key, value in obs.items():
            if isinstance(value, Mapping):
                source[key] = self._map_obs(value)
            else:
                source[key] = _to_tensor(value, device=self.device)
        return source

    def _make_pixels_spec(self) -> TensorSpec:
        if self._pixels_source == "sensor":
            sensor = self._pixels_sensor()
            cfg = getattr(sensor, "cfg", None)
            height = getattr(cfg, "height", None)
            width = getattr(cfg, "width", None)
            if height is not None and width is not None:
                return Bounded(
                    low=0,
                    high=255,
                    shape=(*self.batch_size, int(height), int(width), 3),
                    dtype=torch.uint8,
                    device=self.device,
                )
        pixels = self._get_pixels()
        if pixels.dtype == torch.uint8:
            return Bounded(
                low=0,
                high=255,
                shape=pixels.shape,
                dtype=torch.uint8,
                device=self.device,
            )
        return Unbounded(shape=pixels.shape, dtype=pixels.dtype, device=self.device)

    def _pixels_sensor(self) -> Any:
        if self._pixels_sensor_name is None:
            raise RuntimeError("No mjlab pixels sensor has been selected.")
        sensors = _env_sensor_mapping(self._env)
        try:
            return sensors[self._pixels_sensor_name]
        except KeyError as err:
            raise RuntimeError(
                f"mjlab pixels sensor {self._pixels_sensor_name!r} is no longer "
                "available in the scene."
            ) from err

    def _get_pixels(self) -> torch.Tensor:
        if self._pixels_source == "sensor":
            return self._sensor_pixels()
        return self._render_pixels()

    def _sensor_pixels(self) -> torch.Tensor:
        sensor = self._pixels_sensor()
        pixels = getattr(sensor.data, "rgb", None)
        if pixels is None:
            raise RuntimeError(
                f"mjlab pixels sensor {self._pixels_sensor_name!r} did not return "
                "RGB data. Use a CameraSensorCfg with data_types containing 'rgb'."
            )
        pixels = _to_tensor(pixels, device=self.device)
        if pixels.ndim != 4 or pixels.shape[:1] != self.batch_size:
            raise RuntimeError(
                "Expected mjlab camera sensor pixels to have leading shape "
                f"{tuple(self.batch_size)}, got {tuple(pixels.shape)}."
            )
        return pixels

    def _render_pixels(self) -> torch.Tensor:
        pixels = self.render(mode="rgb_array")
        if pixels is None:
            raise RuntimeError("mjlab render() returned None while from_pixels=True.")
        pixels = _to_tensor(pixels, device=self.device)
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.shape[:1] != self.batch_size:
            raise RuntimeError(
                "Expected rendered pixels to have leading shape "
                f"{tuple(self.batch_size)} after batching, got "
                f"{tuple(pixels.shape)}."
            )
        return pixels

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if kwargs:
            raise ValueError(f"Unsupported reset kwargs: {sorted(kwargs)}")
        reset = tensordict.get("_reset", None) if tensordict is not None else None
        if reset is not None:
            reset = _to_tensor(reset, device=self.device, dtype=torch.bool)
            if not reset.any():
                return tensordict.exclude("_reset")
            env_ids = None if reset.all() else reset.reshape(-1).nonzero().squeeze(-1)
        else:
            env_ids = None
        obs, _info = self._env.reset(seed=seed, env_ids=env_ids, options=options)
        return self._obs_to_tensordict(obs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        action = _to_tensor(action, device=self.device)
        obs, reward, terminated, truncated, _extras = self._env.step(action)
        terminated = _format_done(
            terminated, batch_size=self.batch_size, device=self.device
        )
        truncated = _format_done(
            truncated, batch_size=self.batch_size, device=self.device
        )
        done = terminated | truncated
        td = self._obs_to_tensordict(obs)
        td.set(
            "reward",
            _format_reward(reward, batch_size=self.batch_size, device=self.device),
        )
        td.set("terminated", terminated)
        td.set("truncated", truncated)
        td.set("done", done)
        return td

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        if hasattr(self._env, "seed"):
            self._env.seed(int(seed))
        cfg = getattr(self._env, "cfg", None)
        if cfg is not None and hasattr(cfg, "seed"):
            cfg.seed = int(seed)

    def render(self, mode: str = "rgb_array") -> Any:
        """Render the wrapped mjlab environment.

        Args:
            mode: Only ``"rgb_array"`` is supported.

        Returns:
            An RGB array returned by ``mjlab``'s renderer.
        """
        if mode != "rgb_array":
            raise ValueError("MJLabWrapper.render only supports mode='rgb_array'.")
        return self._env.render()

    def close(self, *, raise_if_closed: bool = True) -> None:
        super().close(raise_if_closed=raise_if_closed)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_envs={self.batch_size[0]}, "
            f"device={self.device}, native_autoreset={self._native_autoreset})"
        )


class _MJLabEnvMeta(_EnvPostInit):
    """Return a lazy ParallelEnv when ``num_workers > 1``."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        if num_workers is None:
            num_workers = kwargs.pop("num_workers", 1)
        else:
            kwargs.pop("num_workers", None)
        num_workers = int(num_workers) if num_workers is not None else 1
        if cls.__name__ == "MJLabEnv" and num_workers > 1:
            from torchrl.envs import EnvCreator, ParallelEnv

            task_id = args[0] if len(args) >= 1 else kwargs.get("task_id")
            env_kwargs = {
                key: value for key, value in kwargs.items() if key != "task_id"
            }

            def make_env(_task_id=task_id, _kwargs=env_kwargs):
                return cls(_task_id, num_workers=1, **_kwargs)

            return ParallelEnv(num_workers, EnvCreator(make_env))
        return super().__call__(*args, **kwargs)


class MJLabEnv(MJLabWrapper, metaclass=_MJLabEnvMeta):
    """Build and wrap an mjlab task from mjlab's task registry.

    Args:
        task_id: Registered mjlab task id, for example
            ``"Mjlab-Velocity-Flat-Unitree-G1"``.
        cfg: Optional mjlab ``ManagerBasedRlEnvCfg``. When omitted, ``task_id``
            is loaded from ``mjlab.tasks.registry``. The config is deep-copied
            before TorchRL mutates ``scene.num_envs`` or ``auto_reset``.
        play: If ``True`` and ``cfg`` is omitted, load mjlab's play/evaluation
            config for ``task_id``.
        num_envs: Number of parallel mjlab worlds. Overrides
            ``cfg.scene.num_envs``.
        device: Simulation device. Defaults to ``"cuda:0"`` when CUDA is
            available, otherwise ``"cpu"``.
        batch_size: TorchRL batch size. Must be ``[num_envs]``. If provided and
            ``num_envs`` is omitted, it sets ``num_envs``.
        render_mode: mjlab render mode. Set to ``"rgb_array"`` to enable
            :meth:`render`. Automatically set when ``from_pixels=True`` uses
            the single-env render fallback. It is not required when pixels come
            from an mjlab RGB ``CameraSensor``.
        native_autoreset: See :class:`MJLabWrapper`.
        from_pixels: See :class:`MJLabWrapper`.
        pixels_only: See :class:`MJLabWrapper`.
        pixels_key: See :class:`MJLabWrapper`.
        pixels_sensor: See :class:`MJLabWrapper`.
        num_workers: If greater than one, return a lazy
            :class:`~torchrl.envs.ParallelEnv` with one mjlab env per worker.

    See also :class:`~torchrl.trainers.algorithms.configs.MJLabEnvConfig`.

    mjlab reference: Zakka et al., "mjlab: A Lightweight Framework for
    GPU-Accelerated Robot Learning", arXiv:2601.22074.

    Examples:
        >>> from torchrl.envs import MJLabEnv  # doctest: +SKIP
        >>> env = MJLabEnv(  # doctest: +SKIP
        ...     "Mjlab-Velocity-Flat-Unitree-G1", num_envs=1024, device="cuda:0"
        ... )
        >>> td = env.reset()  # doctest: +SKIP
    """

    def __init__(
        self,
        task_id: str,
        *,
        cfg: Any | None = None,
        play: bool = False,
        num_envs: int | None = None,
        device: DEVICE_TYPING | None = None,
        batch_size: torch.Size | list[int] | tuple[int, ...] | None = None,
        render_mode: str | None = None,
        native_autoreset: bool = False,
        from_pixels: bool = False,
        pixels_only: bool = False,
        pixels_key: NestedKey = "pixels",
        pixels_sensor: str | None = None,
        allow_done_after_reset: bool = False,
        num_workers: int | None = None,
        **kwargs,
    ) -> None:
        del num_workers
        if not _has_mjlab:
            raise ImportError(
                "mjlab python package was not found. Install it with: pip install mjlab"
            )
        import mjlab.tasks  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        self._task_id = task_id
        cfg = deepcopy(load_env_cfg(task_id, play=play) if cfg is None else cfg)

        if batch_size is not None:
            batch_size = torch.Size(batch_size)
            if len(batch_size) != 1:
                raise ValueError(
                    "MJLabEnv only supports a flat one-dimensional batch_size, "
                    f"got {tuple(batch_size)}."
                )
            if num_envs is None:
                num_envs = int(batch_size[0])
        if num_envs is not None:
            cfg.scene.num_envs = int(num_envs)
        if batch_size is not None:
            _validate_batch_size(batch_size, int(cfg.scene.num_envs))

        if hasattr(cfg, "auto_reset"):
            cfg.auto_reset = bool(native_autoreset)
        device = _device_from_mjlab(device)
        has_rgb_camera_sensor = _cfg_has_rgb_camera_sensor(cfg, pixels_sensor)
        if from_pixels and render_mode is None and not has_rgb_camera_sensor:
            render_mode = "rgb_array"
        if render_mode not in (None, "rgb_array"):
            raise ValueError("MJLabEnv only supports render_mode=None or 'rgb_array'.")
        if from_pixels and int(cfg.scene.num_envs) != 1 and not has_rgb_camera_sensor:
            raise ValueError(
                "from_pixels=True with num_envs > 1 requires an mjlab RGB "
                "CameraSensor because mjlab.render() returns one viewer frame "
                "for the whole scene, not one image per environment row. Add "
                "a CameraSensorCfg with data_types containing 'rgb', or pass "
                "pixels_sensor=... if one is already configured."
            )

        env = ManagerBasedRlEnv(
            cfg=cfg,
            device=str(device),
            render_mode=render_mode,
            **kwargs,
        )
        super().__init__(
            env=env,
            device=device,
            batch_size=batch_size,
            native_autoreset=native_autoreset,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            pixels_key=pixels_key,
            pixels_sensor=pixels_sensor,
            allow_done_after_reset=allow_done_after_reset,
        )

    @property
    def task_id(self) -> str:
        return self._task_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(task_id={self.task_id!r}, "
            f"num_envs={self.batch_size[0]}, device={self.device})"
        )
