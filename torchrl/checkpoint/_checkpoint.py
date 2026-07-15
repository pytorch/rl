# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import base64
import json
import os
import random
import shutil
import tempfile
import uuid
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import tensordict
import torch

from torchrl import __version__ as torchrl_version
from torchrl._utils import logger as torchrl_logger

CheckpointFormat = Literal["directory", "archive"]
CheckpointStrictness = Literal["error", "warn", "ignore"]
ArchiveCompression = Literal["stored", "deflate"]
StateDictFormat = Literal["directory", "archive", "consolidated", "torch"]

_MANIFEST_NAME = "manifest.json"
_FORMAT_NAME = "torchrl.checkpoint"
_FORMAT_VERSION = 1


def _to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-compatible.")


class CheckpointError(RuntimeError):
    """Error raised when a checkpoint cannot be saved or restored.

    Args:
        message: Description of the checkpoint failure.
        result: Optional partial load result associated with the failure.

    Examples:
        >>> from torchrl.checkpoint import CheckpointError
        >>> error = CheckpointError("invalid checkpoint")
        >>> str(error)
        'invalid checkpoint'
    """

    def __init__(
        self, message: str, result: CheckpointLoadResult | None = None
    ) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CheckpointOptions:
    """Arguments forwarded to a component's serialization methods.

    Operation-level options are merged over registration-time options. Keyword
    arguments are shallow-merged and explicitly supplied positional arguments
    replace the registration-time tuple.

    Args:
        save_args: Positional arguments forwarded after the checkpoint path for
            ``dump`` adapters, or to ``state_dict`` for state-dict adapters.
        save_kwargs: Keyword arguments forwarded during saving.
        load_args: Positional arguments forwarded after the checkpoint path for
            ``load`` adapters, or after the state mapping for state-dict adapters.
        load_kwargs: Keyword arguments forwarded during restoration.

    Examples:
        >>> from torchrl.checkpoint import CheckpointOptions
        >>> options = CheckpointOptions(save_kwargs={"compression": "zstd"})
        >>> options.save_kwargs["compression"]
        'zstd'
    """

    save_args: tuple[Any, ...] | None = None
    save_kwargs: Mapping[str, Any] | None = None
    load_args: tuple[Any, ...] | None = None
    load_kwargs: Mapping[str, Any] | None = None

    def merged(self, override: CheckpointOptions | None) -> CheckpointOptions:
        """Return these options with an operation-level override applied."""
        if override is None:
            return self
        save_kwargs = dict(self.save_kwargs or {})
        save_kwargs.update(override.save_kwargs or {})
        load_kwargs = dict(self.load_kwargs or {})
        load_kwargs.update(override.load_kwargs or {})
        return CheckpointOptions(
            save_args=self.save_args
            if override.save_args is None
            else override.save_args,
            save_kwargs=save_kwargs,
            load_args=self.load_args
            if override.load_args is None
            else override.load_args,
            load_kwargs=load_kwargs,
        )


@dataclass
class CheckpointLoadResult:
    """Structured result returned by :meth:`Checkpoint.load`.

    Args:
        loaded: Components restored successfully.
        missing: Requested components absent from the checkpoint.
        incompatible: Components that could not be restored, mapped to reasons.
        unrequested: Manifest components intentionally not requested.
        values: Values returned by adapters, including immutable JSON components.
        manifest: Parsed checkpoint manifest.

    Examples:
        >>> from torchrl.checkpoint import CheckpointLoadResult
        >>> result = CheckpointLoadResult()
        >>> result.loaded
        set()
    """

    loaded: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    incompatible: dict[str, str] = field(default_factory=dict)
    unrequested: set[str] = field(default_factory=set)
    values: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


class CheckpointAdapter(abc.ABC):
    """Interface used to save and restore one checkpoint component.

    Adapters receive a real local directory even when the outer checkpoint is
    an archive. Custom adapters can therefore write any number of files without
    depending on the container implementation.

    Examples:
        >>> from torchrl.checkpoint import CheckpointAdapter
        >>> issubclass(CheckpointAdapter, object)
        True
    """

    adapter_id: str
    format_version: int = 1

    @abc.abstractmethod
    def save(
        self,
        component: Any,
        path: Path,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Save ``component`` below ``path``."""

    @abc.abstractmethod
    def load(
        self,
        component: Any,
        path: Path,
        *,
        map_location: Any,
        tensor_load_kwargs: Mapping[str, Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Restore ``component`` from ``path`` and optionally return a value."""

    def _detach_from_load_path(self, component: Any) -> None:
        """Detach a loaded component from temporary archive payload files."""
        del component


class DumpLoadCheckpointAdapter(CheckpointAdapter):
    """Adapter for objects exposing ``dump(path)`` and ``load(path)``.

    Examples:
        >>> from torchrl.checkpoint import DumpLoadCheckpointAdapter
        >>> DumpLoadCheckpointAdapter().adapter_id
        'torchrl.dump_load'
    """

    adapter_id = "torchrl.dump_load"

    def save(
        self,
        component: Any,
        path: Path,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        path.mkdir(parents=True, exist_ok=True)
        return component.dump(path, *args, **kwargs)

    def load(
        self,
        component: Any,
        path: Path,
        *,
        map_location: Any,
        tensor_load_kwargs: Mapping[str, Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        del map_location, tensor_load_kwargs
        component.load(path, *args, **kwargs)
        return None

    def _detach_from_load_path(self, component: Any) -> None:
        detach = getattr(component, "_torchrl_checkpoint_detach_from_load_path", None)
        if detach is not None:
            detach()


def _encode_state_dict_value(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value.detach()
        return {"kind": "tensor", "key": key, "device": str(value.device)}
    if isinstance(value, np.ndarray):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = torch.from_numpy(value.copy())
        return {"kind": "numpy", "key": key, "dtype": value.dtype.str}
    if isinstance(value, Mapping):
        metadata = getattr(value, "_metadata", None)
        mapping_type = (
            "ordered_dict"
            if isinstance(value, OrderedDict) or metadata is not None
            else "dict"
        )
        result = {
            "kind": mapping_type,
            "items": [
                [
                    _encode_state_dict_value(key, tensors),
                    _encode_state_dict_value(item, tensors),
                ]
                for key, item in value.items()
            ],
        }
        if metadata is not None:
            result["metadata"] = _encode_state_dict_value(metadata, tensors)
        return result
    if isinstance(value, torch.Size):
        return {"kind": "torch_size", "items": list(value)}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_state_dict_value(item, tensors) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_state_dict_value(item, tensors) for item in value],
        }
    if isinstance(value, set):
        return {
            "kind": "set",
            "items": [_encode_state_dict_value(item, tensors) for item in value],
        }
    if isinstance(value, frozenset):
        return {
            "kind": "frozenset",
            "items": [_encode_state_dict_value(item, tensors) for item in value],
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "items": [
                _encode_state_dict_value(item, tensors)
                for item in (value.start, value.stop, value.step)
            ],
        }
    if isinstance(value, range):
        return {
            "kind": "range",
            "items": [value.start, value.stop, value.step],
        }
    if isinstance(value, torch.device):
        return {"kind": "torch_device", "value": str(value)}
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, bytearray):
        return {
            "kind": "bytearray",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, complex):
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return {
            "kind": "numpy_scalar",
            "dtype": value.dtype.str,
            "value": _to_json_value(value),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "value", "value": value}
    raise TypeError(
        f"State-dict value of type {type(value).__name__} is not supported by "
        "the TensorDict checkpoint format. Register "
        "StateDictCheckpointAdapter(payload_format='torch') for state dicts "
        "that require pickle serialization."
    )


def _mapped_device(saved_device: str, map_location: Any) -> torch.device:
    if map_location is None:
        return torch.device(saved_device)
    if isinstance(map_location, Mapping):
        target = map_location.get(saved_device)
        if target is None:
            target = map_location.get(torch.device(saved_device), saved_device)
        return torch.device(target)
    if callable(map_location):
        raise TypeError(
            "Callable map_location is only supported by the torch state-dict "
            "payload format."
        )
    return torch.device(map_location)


def _decode_state_dict_value(
    value: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], map_location: Any
) -> Any:
    kind = value["kind"]
    if kind == "tensor":
        tensor = tensors[value["key"]]
        return tensor.to(_mapped_device(value["device"], map_location), copy=True)
    if kind == "numpy":
        return tensors[value["key"]].cpu().numpy().astype(value["dtype"], copy=True)
    if kind in ("dict", "ordered_dict"):
        mapping = OrderedDict() if kind == "ordered_dict" else {}
        for key, item in value["items"]:
            mapping[
                _decode_state_dict_value(key, tensors, map_location)
            ] = _decode_state_dict_value(item, tensors, map_location)
        metadata = value.get("metadata")
        if metadata is not None:
            mapping._metadata = _decode_state_dict_value(  # type: ignore[attr-defined]
                metadata, tensors, map_location
            )
        return mapping
    if kind == "torch_size":
        return torch.Size(value["items"])
    if kind in ("tuple", "list", "set", "frozenset"):
        items = [
            _decode_state_dict_value(item, tensors, map_location)
            for item in value["items"]
        ]
        return {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset,}[
            kind
        ](items)
    if kind == "slice":
        return slice(
            *(
                _decode_state_dict_value(item, tensors, map_location)
                for item in value["items"]
            )
        )
    if kind == "range":
        return range(*value["items"])
    if kind == "torch_device":
        return torch.device(value["value"])
    if kind == "torch_dtype":
        dtype = getattr(torch, value["value"], None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Unknown torch dtype {value['value']!r} in checkpoint.")
        return dtype
    if kind == "path":
        return Path(value["value"])
    if kind in ("bytes", "bytearray"):
        decoded = base64.b64decode(value["value"])
        return decoded if kind == "bytes" else bytearray(decoded)
    if kind == "complex":
        return complex(value["real"], value["imag"])
    if kind == "numpy_scalar":
        return np.dtype(value["dtype"]).type(value["value"])
    if kind == "ellipsis":
        return Ellipsis
    if kind == "value":
        return value["value"]
    raise TypeError(f"Unknown state-dict schema kind {kind!r}.")


class StateDictCheckpointAdapter(CheckpointAdapter):
    """Adapter for ``state_dict`` / ``load_state_dict`` objects.

    TensorDict directory payloads are used by default. ``payload_format`` can
    select a TensorDict archive, a consolidated TensorDict, or the pickle-based
    :func:`torch.save` format. Loading auto-detects all four payload formats.

    Args:
        payload_format: Format used for new payloads. One of ``"directory"``,
            ``"archive"``, ``"consolidated"``, or ``"torch"``.
        archive_compression: Compression passed to TensorDict archive saves.

    Examples:
        >>> from torchrl.checkpoint import StateDictCheckpointAdapter
        >>> StateDictCheckpointAdapter().payload_format
        'directory'
        >>> StateDictCheckpointAdapter(payload_format="torch").payload_format
        'torch'
    """

    adapter_id = "torchrl.state_dict"
    state_directory = "state"
    state_archive = "state.tdz"
    state_consolidated = "state.tdc"
    state_torch = "state.pt"
    schema_filename = "state.json"

    def __init__(
        self,
        payload_format: StateDictFormat = "directory",
        *,
        archive_compression: str | int | None = None,
    ) -> None:
        if payload_format not in ("directory", "archive", "consolidated", "torch"):
            raise ValueError(
                "payload_format must be 'directory', 'archive', 'consolidated', "
                f"or 'torch', got {payload_format!r}."
            )
        self.payload_format = payload_format
        self.archive_compression = archive_compression

    def save(
        self,
        component: Any,
        path: Path,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        path.mkdir(parents=True, exist_ok=True)
        state = component.state_dict(*args, **kwargs)
        if self.payload_format == "torch":
            torch.save(state, path / self.state_torch)
            return state
        tensors: dict[str, torch.Tensor] = {}
        schema = _encode_state_dict_value(state, tensors)
        tensor_state = tensordict.TensorDict(tensors, batch_size=[])
        if self.payload_format == "directory":
            tensordict.save(tensor_state, path / self.state_directory)
        elif self.payload_format == "archive":
            tensor_state.save(
                path / self.state_archive,
                archive=True,
                compression=self.archive_compression,
            )
        else:
            tensor_state.consolidate(path / self.state_consolidated)
        with (path / self.schema_filename).open("w", encoding="utf-8") as file:
            json.dump(schema, file, separators=(",", ":"))
            file.write("\n")
        return state

    def load(
        self,
        component: Any,
        path: Path,
        *,
        map_location: Any,
        tensor_load_kwargs: Mapping[str, Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        torch_path = path / self.state_torch
        if torch_path.exists():
            load_kwargs: dict[str, Any] = {"weights_only": True}
            load_kwargs.update(tensor_load_kwargs)
            if map_location is not None:
                load_kwargs["map_location"] = map_location
            state = torch.load(torch_path, **load_kwargs)
        else:
            unsupported_load_kwargs = set(tensor_load_kwargs).difference(
                {"mmap", "weights_only"}
            )
            if unsupported_load_kwargs:
                raise TypeError(
                    "The following tensor_load_kwargs are only accepted for torch "
                    f"state-dict payloads: {sorted(unsupported_load_kwargs)}."
                )
            directory = path / self.state_directory
            archive = path / self.state_archive
            consolidated = path / self.state_consolidated
            if directory.is_dir():
                tensor_state = tensordict.load(directory)
            elif archive.is_file():
                tensor_state = tensordict.load(archive)
            elif consolidated.is_file():
                tensor_state = tensordict.from_consolidated(consolidated)
            else:
                raise FileNotFoundError(
                    f"No state-dict payload was found below {path!s}."
                )
            with (path / self.schema_filename).open(encoding="utf-8") as file:
                schema = json.load(file)
            state = _decode_state_dict_value(schema, tensor_state, map_location)
        component.load_state_dict(state, *args, **kwargs)
        return None


class JSONCheckpointAdapter(CheckpointAdapter):
    """Adapter for JSON-compatible configuration, metrics, and metadata.

    Mutable mappings and lists are updated in place during restoration. Other
    values are returned through :attr:`CheckpointLoadResult.values`.

    Examples:
        >>> from torchrl.checkpoint import JSONCheckpointAdapter
        >>> JSONCheckpointAdapter().adapter_id
        'torchrl.json'
    """

    adapter_id = "torchrl.json"
    state_filename = "value.json"

    def save(
        self,
        component: Any,
        path: Path,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if args or kwargs:
            raise TypeError("JSON checkpoint components do not accept save options.")
        path.mkdir(parents=True, exist_ok=True)
        with (path / self.state_filename).open("w", encoding="utf-8") as file:
            json.dump(_to_json_value(component), file, indent=2, sort_keys=True)
        return component

    def load(
        self,
        component: Any,
        path: Path,
        *,
        map_location: Any,
        tensor_load_kwargs: Mapping[str, Any],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        del map_location, tensor_load_kwargs
        if args or kwargs:
            raise TypeError("JSON checkpoint components do not accept load options.")
        with (path / self.state_filename).open(encoding="utf-8") as file:
            value = json.load(file)
        if isinstance(component, MutableMapping) and isinstance(value, Mapping):
            component.clear()
            component.update(value)
            return component
        if isinstance(component, list) and isinstance(value, list):
            component[:] = value
            return component
        return value


class GlobalRNGState:
    """Checkpointable process-global random-number-generator state.

    The object captures Python, NumPy, Torch CPU, and initialized accelerator
    RNGs when :meth:`state_dict` is called.

    Examples:
        >>> import torch
        >>> from torchrl.checkpoint import GlobalRNGState
        >>> state = GlobalRNGState().state_dict()
        >>> "torch_cpu" in state
        True
    """

    def state_dict(self) -> dict[str, Any]:
        """Capture all supported process-global RNG state."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        numpy_state = np.random.get_state()
        state["numpy"] = (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        if torch.cuda.is_initialized() and torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        xpu = getattr(torch, "xpu", None)
        if (
            xpu is not None
            and getattr(xpu, "is_initialized", lambda: False)()
            and xpu.is_available()
        ):
            state["torch_xpu"] = xpu.get_rng_state_all()
        mps = getattr(torch, "mps", None)
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps is not None
            and mps_backend is not None
            and getattr(mps, "_default_mps_generator", None) is not None
            and mps_backend.is_available()
        ):
            state["torch_mps"] = mps.get_rng_state()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore process-global RNG state."""
        random.setstate(state_dict["python"])
        numpy_state = state_dict["numpy"]
        if isinstance(numpy_state[1], torch.Tensor):
            numpy_state = (
                numpy_state[0],
                numpy_state[1].cpu().numpy().astype(np.uint32, copy=False),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(state_dict["torch_cpu"].cpu())
        if "torch_cuda" in state_dict:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state_dict["torch_cuda"]]
            )
        if "torch_xpu" in state_dict:
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                raise RuntimeError(
                    "Checkpoint contains XPU RNG state but XPU is unavailable."
                )
            xpu.set_rng_state_all(
                [rng_state.cpu() for rng_state in state_dict["torch_xpu"]]
            )
        if "torch_mps" in state_dict:
            mps = getattr(torch, "mps", None)
            mps_backend = getattr(torch.backends, "mps", None)
            if mps is None or mps_backend is None or not mps_backend.is_available():
                raise RuntimeError(
                    "Checkpoint contains MPS RNG state but MPS is unavailable."
                )
            mps.set_rng_state(state_dict["torch_mps"].cpu())


@dataclass
class _Component:
    value: Any
    adapter: CheckpointAdapter | None
    options: CheckpointOptions


class Checkpoint:
    """Standard TorchRL checkpoint coordinator.

    Components are bound to the checkpoint and may be independently selected
    for each save or restore. The default directory format supports direct,
    lazy access to component payloads; ``format="archive"`` stores the same
    manifest and component tree in one ZIP file.

    Args:
        format: Default output format, ``"directory"`` or ``"archive"``.
        strict: Default restoration behavior for missing or incompatible
            requested components.
        archive_compression: ``"stored"`` avoids recompressing payloads;
            ``"deflate"`` enables ZIP deflate compression.
        save_components: Optional default component selection for saves. This
            is useful for excluding large components from scheduled saves.
        **components: Named objects or JSON-compatible values to register.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from torchrl.checkpoint import Checkpoint
        >>> source = torch.nn.Linear(2, 1)
        >>> target = torch.nn.Linear(2, 1)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     Checkpoint(policy=source).save(f"{tmpdir}/checkpoint")
        ...     result = Checkpoint(policy=target).load(f"{tmpdir}/checkpoint")
        >>> result.loaded
        {'policy'}
    """

    _format_migrations: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def __init__(
        self,
        *,
        format: CheckpointFormat = "directory",
        strict: CheckpointStrictness = "error",
        archive_compression: ArchiveCompression = "stored",
        save_components: Collection[str] | None = None,
        **components: Any,
    ) -> None:
        self._validate_format(format)
        self._validate_strict(strict)
        if archive_compression not in ("stored", "deflate"):
            raise ValueError(
                "archive_compression must be 'stored' or 'deflate', got "
                f"{archive_compression!r}."
            )
        self.format = format
        self.strict = strict
        self.archive_compression = archive_compression
        self.save_components = (
            None if save_components is None else frozenset(save_components)
        )
        self._components: dict[str, _Component] = {}
        self._adapter_registry: list[tuple[type, CheckpointAdapter]] = []
        for name, component in components.items():
            self.register(name, component)

    @property
    def components(self) -> Mapping[str, Any]:
        """Read-only mapping view of registered component values."""
        return {name: component.value for name, component in self._components.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def register(
        self,
        name: str,
        component: Any,
        *,
        adapter: CheckpointAdapter | None = None,
        options: CheckpointOptions | None = None,
    ) -> Checkpoint:
        """Register a named component and return ``self``.

        Args:
            name: Stable manifest component name.
            component: Live object or JSON-compatible value.
            adapter: Optional explicit serialization adapter.
            options: Persistent component method arguments.

        Returns:
            This checkpoint.
        """
        self._validate_component_name(name)
        if name in self._components:
            raise KeyError(f"Checkpoint component {name!r} is already registered.")
        if adapter is not None and not isinstance(adapter, CheckpointAdapter):
            raise TypeError("adapter must be a CheckpointAdapter instance.")
        self._components[name] = _Component(
            component, adapter, options or CheckpointOptions()
        )
        return self

    def register_adapter(
        self, component_type: type, adapter: CheckpointAdapter
    ) -> Checkpoint:
        """Register an adapter for a type on this checkpoint instance."""
        if not isinstance(component_type, type):
            raise TypeError("component_type must be a type.")
        if not isinstance(adapter, CheckpointAdapter):
            raise TypeError("adapter must be a CheckpointAdapter instance.")
        self._adapter_registry.insert(0, (component_type, adapter))
        return self

    def save(
        self,
        path: str | Path,
        *,
        components: Collection[str] | None = None,
        component_options: Mapping[str, CheckpointOptions] | None = None,
        format: CheckpointFormat | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save selected registered components atomically.

        Args:
            path: Local checkpoint destination.
            components: Component names to save. ``None`` uses
                ``save_components`` from construction, or saves all when no
                default selection was configured.
            component_options: Per-operation option overrides.
            format: Per-operation container override.
            metadata: JSON-compatible manifest metadata.

        Returns:
            Expanded destination path.
        """
        destination = self._local_path(path)
        output_format = self.format if format is None else format
        self._validate_format(output_format)
        selected = self._selection(
            self.save_components if components is None else components
        )
        options = component_options or {}
        unknown_options = set(options).difference(selected)
        if unknown_options:
            raise KeyError(
                "Options were provided for unselected checkpoint components: "
                f"{sorted(unknown_options)}."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.parent / (f".{destination.name}.stage-{uuid.uuid4().hex}")
        stage.mkdir(mode=0o777)
        try:
            component_records: dict[str, Any] = {}
            for index, name in enumerate(sorted(selected)):
                registration = self._components[name]
                adapter = self._resolve_adapter(registration)
                relative_path = Path("components") / self._component_dir(index, name)
                component_path = stage / relative_path
                resolved_options = registration.options.merged(options.get(name))
                adapter.save(
                    registration.value,
                    component_path,
                    args=resolved_options.save_args or (),
                    kwargs=resolved_options.save_kwargs or {},
                )
                files = sorted(
                    file.relative_to(component_path).as_posix()
                    for file in component_path.rglob("*")
                    if file.is_file()
                )
                component_records[name] = {
                    "adapter": adapter.adapter_id,
                    "adapter_version": adapter.format_version,
                    "path": relative_path.as_posix(),
                    "files": files,
                }
            manifest = self._new_manifest(
                output_format, component_records, metadata or {}
            )
            with (stage / _MANIFEST_NAME).open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, sort_keys=True)
                file.write("\n")
            if output_format == "directory":
                self._publish_path(stage, destination)
                stage = None
            else:
                archive = destination.parent / (
                    f".{destination.name}.archive-{uuid.uuid4().hex}"
                )
                try:
                    compression = (
                        zipfile.ZIP_STORED
                        if self.archive_compression == "stored"
                        else zipfile.ZIP_DEFLATED
                    )
                    with zipfile.ZipFile(archive, "w", compression=compression) as file:
                        for source in sorted(stage.rglob("*")):
                            if source.is_file():
                                file.write(source, source.relative_to(stage).as_posix())
                    self._publish_path(archive, destination)
                finally:
                    self._remove_path(archive)
            return destination
        finally:
            if stage is not None:
                self._remove_path(stage)

    def load(
        self,
        path: str | Path,
        *,
        components: Collection[str] | None = None,
        component_options: Mapping[str, CheckpointOptions] | None = None,
        map_location: Any = None,
        tensor_load_kwargs: Mapping[str, Any] | None = None,
        strict: CheckpointStrictness | None = None,
    ) -> CheckpointLoadResult:
        """Restore selected components from a local checkpoint.

        Args:
            path: Directory or archive checkpoint.
            components: Registered names to restore. ``None`` selects all
                registered components.
            component_options: Per-operation option overrides.
            map_location: Device mapping used while reading tensor payloads.
            tensor_load_kwargs: Additional keyword arguments passed to
                :func:`torch.load` for state-dict components explicitly saved
                with the torch payload format. ``weights_only`` defaults to
                ``True``.
            strict: Per-operation strictness override.

        Returns:
            A structured component load report.
        """
        source = self._local_path(path)
        load_strict = self.strict if strict is None else strict
        self._validate_strict(load_strict)
        manifest = self.manifest(source)
        requested = self._selection(components)
        manifest_components: Mapping[str, Any] = manifest["components"]
        result = CheckpointLoadResult(
            missing=requested.difference(manifest_components),
            unrequested=set(manifest_components).difference(requested),
            manifest=manifest,
        )
        options = component_options or {}
        unknown_options = set(options).difference(requested)
        if unknown_options:
            raise KeyError(
                "Options were provided for unrequested checkpoint components: "
                f"{sorted(unknown_options)}."
            )
        adapters: dict[str, CheckpointAdapter] = {}
        for name in requested.intersection(manifest_components):
            registration = self._components[name]
            try:
                adapter = self._resolve_adapter(registration)
            except Exception as error:
                result.incompatible[name] = str(error)
                continue
            record = manifest_components[name]
            if record["adapter"] != adapter.adapter_id:
                result.incompatible[name] = (
                    f"manifest adapter is {record['adapter']!r}, target adapter is "
                    f"{adapter.adapter_id!r}"
                )
                continue
            if record["adapter_version"] != adapter.format_version:
                result.incompatible[name] = (
                    f"manifest adapter version is {record['adapter_version']}, target "
                    f"adapter version is {adapter.format_version}"
                )
                continue
            adapters[name] = adapter
        self._handle_load_issues(result, load_strict)
        selected_records = {name: manifest_components[name] for name in adapters}
        with self._materialize(source, manifest, selected_records) as root:
            for name in sorted(adapters):
                registration = self._components[name]
                resolved_options = registration.options.merged(options.get(name))
                record = manifest_components[name]
                try:
                    value = adapters[name].load(
                        registration.value,
                        root / record["path"].replace("\\", "/"),
                        map_location=map_location,
                        tensor_load_kwargs=tensor_load_kwargs or {},
                        args=resolved_options.load_args or (),
                        kwargs=resolved_options.load_kwargs or {},
                    )
                    if not source.is_dir():
                        adapters[name]._detach_from_load_path(registration.value)
                except Exception as error:
                    result.incompatible[name] = str(error)
                    if load_strict == "error":
                        raise CheckpointError(
                            f"Failed to load checkpoint component {name!r}: {error}",
                            result,
                        ) from error
                    if load_strict == "warn":
                        torchrl_logger.warning(
                            "Failed to load checkpoint component %r: %s", name, error
                        )
                    continue
                result.loaded.add(name)
                if value is not None:
                    result.values[name] = value
        return result

    @classmethod
    def is_checkpoint(cls, path: str | Path) -> bool:
        """Return whether ``path`` contains a TorchRL checkpoint manifest."""
        try:
            cls.manifest(path)
        except (CheckpointError, OSError, TypeError, ValueError):
            return False
        return True

    @classmethod
    def register_migration(
        cls,
        from_version: int,
        migration: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register one manifest migration from ``from_version`` to the next.

        Migrations must return a new manifest whose ``format_version`` is
        exactly ``from_version + 1``. Payload migrations remain the
        responsibility of the component adapter identified by the migrated
        manifest.
        """
        if not isinstance(from_version, int) or from_version < 0:
            raise ValueError("from_version must be a non-negative integer.")
        if not callable(migration):
            raise TypeError("migration must be callable.")
        cls._format_migrations[from_version] = migration

    @classmethod
    def manifest(cls, path: str | Path) -> dict[str, Any]:
        """Read and validate a checkpoint manifest without loading payloads."""
        source = cls._local_path(path)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {source}.")
        try:
            if source.is_dir():
                with (source / _MANIFEST_NAME).open(encoding="utf-8") as file:
                    manifest = json.load(file)
            elif zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as archive:
                    with archive.open(_MANIFEST_NAME) as file:
                        manifest = json.load(file)
            else:
                raise CheckpointError(
                    f"Path is not a TorchRL checkpoint directory or archive: {source}."
                )
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
            raise CheckpointError(
                f"Invalid checkpoint manifest at {source}."
            ) from error
        if manifest.get("format") != _FORMAT_NAME:
            raise CheckpointError(
                f"Path does not contain a TorchRL checkpoint: {source}."
            )
        version = manifest.get("format_version")
        if (
            isinstance(version, int)
            and not isinstance(version, bool)
            and version < _FORMAT_VERSION
        ):
            while version < _FORMAT_VERSION:
                migration = cls._format_migrations.get(version)
                if migration is None:
                    raise CheckpointError(
                        f"Unsupported older checkpoint format version {version!r}; "
                        f"no migration to version {version + 1} is registered."
                    )
                manifest = migration(dict(manifest))
                next_version = manifest.get("format_version")
                if next_version != version + 1 or isinstance(next_version, bool):
                    raise CheckpointError(
                        f"Checkpoint migration from version {version} returned "
                        f"version {next_version!r}; expected {version + 1}."
                    )
                version = next_version
        if version != _FORMAT_VERSION or isinstance(version, bool):
            direction = (
                "newer"
                if isinstance(version, int)
                and not isinstance(version, bool)
                and version > 1
                else "invalid"
            )
            raise CheckpointError(
                f"Unsupported {direction} checkpoint format version {version!r}; "
                f"this build supports version {_FORMAT_VERSION}."
            )
        if not isinstance(manifest.get("components"), dict):
            raise CheckpointError("Checkpoint manifest has no component mapping.")
        for name, record in manifest["components"].items():
            cls._validate_component_record(name, record)
        return manifest

    def _resolve_adapter(self, component: _Component) -> CheckpointAdapter:
        if component.adapter is not None:
            return component.adapter
        for component_type, adapter in self._adapter_registry:
            if isinstance(component.value, component_type):
                return adapter
        dump = getattr(component.value, "dump", None)
        load = getattr(component.value, "load", None)
        if callable(dump) and callable(load):
            return DumpLoadCheckpointAdapter()
        state_dict = getattr(component.value, "state_dict", None)
        load_state_dict = getattr(component.value, "load_state_dict", None)
        if callable(state_dict) and callable(load_state_dict):
            return StateDictCheckpointAdapter()
        if self._is_json_value(component.value):
            return JSONCheckpointAdapter()
        raise TypeError(
            f"No checkpoint adapter could be inferred for {type(component.value).__name__}. "
            "Provide an adapter, implement dump/load, or implement "
            "state_dict/load_state_dict."
        )

    def _selection(self, components: Collection[str] | None) -> set[str]:
        selected = set(self._components) if components is None else set(components)
        unknown = selected.difference(self._components)
        if unknown:
            raise KeyError(f"Unregistered checkpoint components: {sorted(unknown)}.")
        return selected

    @staticmethod
    def _new_manifest(
        format: CheckpointFormat,
        components: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not Checkpoint._is_json_value(metadata):
            raise TypeError("Checkpoint manifest metadata must be JSON-compatible.")
        return {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "container": format,
            "checkpoint_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "torchrl": torchrl_version,
                "tensordict": tensordict.__version__,
                "torch": torch.__version__,
            },
            "metadata": _to_json_value(metadata),
            "components": dict(components),
        }

    @classmethod
    @contextmanager
    def _materialize(
        cls,
        source: Path,
        manifest: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
    ):
        if source.is_dir():
            yield source
            return
        temporary = Path(tempfile.mkdtemp(prefix="torchrl-checkpoint-load-"))
        try:
            with zipfile.ZipFile(source) as archive:
                for record in records.values():
                    root = Path(record["path"].replace("\\", "/"))
                    (temporary / root).mkdir(parents=True, exist_ok=True)
                    for relative_file in record["files"]:
                        member = root / relative_file.replace("\\", "/")
                        cls._validate_archive_member(member)
                        destination = temporary / member
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member.as_posix()) as source_file:
                            with destination.open("wb") as destination_file:
                                shutil.copyfileobj(source_file, destination_file)
            yield temporary
        finally:
            cls._remove_path(temporary)

    @staticmethod
    def _handle_load_issues(
        result: CheckpointLoadResult, strict: CheckpointStrictness
    ) -> None:
        if not result.missing and not result.incompatible:
            return
        parts = []
        if result.missing:
            parts.append(f"missing={sorted(result.missing)}")
        if result.incompatible:
            parts.append(f"incompatible={result.incompatible}")
        message = "Checkpoint restore issues: " + ", ".join(parts)
        if strict == "error":
            raise CheckpointError(message, result)
        if strict == "warn":
            torchrl_logger.warning(message)

    @staticmethod
    def _publish_path(source: Path, destination: Path) -> None:
        backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
        moved_destination = False
        try:
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup)
                moved_destination = True
            os.replace(source, destination)
        except Exception:
            if moved_destination and not destination.exists() and backup.exists():
                os.replace(backup, destination)
            raise
        finally:
            Checkpoint._remove_path(backup)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _local_path(path: str | Path) -> Path:
        if "://" in str(path):
            raise ValueError(
                f"Only local checkpoint paths are supported, got {path!s}."
            )
        return Path(path).expanduser().absolute()

    @staticmethod
    def _component_dir(index: int, name: str) -> str:
        return f"{index:04d}-{quote(name, safe='._-')}"

    @staticmethod
    def _validate_component_name(name: str) -> None:
        if not isinstance(name, str) or not name or name in (".", ".."):
            raise ValueError("Checkpoint component names must be non-empty strings.")
        if "/" in name or "\\" in name:
            raise ValueError(
                "Checkpoint component names cannot contain path separators."
            )

    @staticmethod
    def _validate_archive_member(path: Path) -> None:
        if path.is_absolute() or ".." in path.parts:
            raise CheckpointError(f"Unsafe archive member path: {path}.")

    @classmethod
    def _validate_component_record(cls, name: Any, record: Any) -> None:
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise CheckpointError("Checkpoint manifest has an invalid component entry.")
        if (
            not isinstance(record.get("adapter"), str)
            or not isinstance(record.get("adapter_version"), int)
            or isinstance(record.get("adapter_version"), bool)
        ):
            raise CheckpointError(
                f"Checkpoint component {name!r} has invalid adapter metadata."
            )
        relative_path = record.get("path")
        files = record.get("files")
        if not isinstance(relative_path, str) or not isinstance(files, list):
            raise CheckpointError(
                f"Checkpoint component {name!r} has an invalid file inventory."
            )
        cls._validate_archive_member(Path(relative_path.replace("\\", "/")))
        for relative_file in files:
            if not isinstance(relative_file, str):
                raise CheckpointError(
                    f"Checkpoint component {name!r} has an invalid file inventory."
                )
            cls._validate_archive_member(Path(relative_file.replace("\\", "/")))

    @staticmethod
    def _validate_format(format: str) -> None:
        if format not in ("directory", "archive"):
            raise ValueError(
                f"Checkpoint format must be 'directory' or 'archive', got {format!r}."
            )

    @staticmethod
    def _validate_strict(strict: str) -> None:
        if strict not in ("error", "warn", "ignore"):
            raise ValueError(
                "Checkpoint strictness must be 'error', 'warn', or 'ignore', got "
                f"{strict!r}."
            )

    @staticmethod
    def _is_json_value(value: Any) -> bool:
        try:
            json.dumps(_to_json_value(value))
        except (TypeError, ValueError, OverflowError):
            return False
        return True
