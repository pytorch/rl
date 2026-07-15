# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._distributed import _DistributedInferenceTransport
from torchrl.modules.inference_server._monarch import MonarchTransport
from torchrl.modules.inference_server._mp import MPTransport
from torchrl.modules.inference_server._ray import RayTransport
from torchrl.modules.inference_server._shared_memory import SharedMemoryTransport
from torchrl.modules.inference_server._slot import SlotTransport
from torchrl.modules.inference_server._threading import ThreadingTransport
from torchrl.modules.inference_server._transport import InferenceTransport


def _validate_inference_transport_selection(
    transport: InferenceTransport | str | None,
    *,
    service_backend: str,
    transport_options: dict[str, Any] | None,
    request_spec: TensorDictBase | None,
    response_spec: TensorDictBase | None,
) -> None:
    """Reject unsupported explicit selectors before creating an owner."""
    if isinstance(transport, InferenceTransport):
        if transport_options:
            raise ValueError(
                "transport_options cannot be used with an explicit transport instance."
            )
        return
    kind = "auto" if transport is None else transport
    allowed = {
        "thread": {"auto", "thread", "queue", "direct"},
        "process": {"auto", "process", "shared_memory", "distributed"},
        "ray": {"auto", "ray", "distributed"},
    }[service_backend]
    if kind not in allowed:
        raise ValueError(
            f"transport={kind!r} is incompatible with "
            f"service_backend={service_backend!r}."
        )
    if (request_spec is None) != (response_spec is None):
        raise ValueError("request_spec and response_spec must be provided together.")
    if kind == "shared_memory" and request_spec is None:
        raise ValueError(
            "transport='shared_memory' requires request_spec and response_spec."
        )
    if kind == "distributed" and service_backend == "process" and request_spec is None:
        raise ValueError(
            "Process-owned distributed inference requires request_spec and response_spec."
        )


def _make_inference_transport(
    transport: InferenceTransport | str | None,
    *,
    service_backend: str,
    transport_options: dict[str, Any] | None = None,
    request_spec: TensorDictBase | None = None,
    response_spec: TensorDictBase | None = None,
    num_clients: int | None = None,
) -> InferenceTransport:
    """Resolve an inference transport without exposing deployment machinery."""
    _validate_inference_transport_selection(
        transport,
        service_backend=service_backend,
        transport_options=transport_options,
        request_spec=request_spec,
        response_spec=response_spec,
    )
    if isinstance(transport, InferenceTransport):
        if transport_options:
            raise ValueError(
                "transport_options cannot be used with an explicit transport instance."
            )
        return transport

    options = dict(transport_options or {})
    kind = "auto" if transport is None else transport
    if kind == "auto":
        kind = {
            "thread": "thread",
            "process": "process",
            "ray": "ray",
            "monarch": "monarch",
        }[service_backend]

    if kind in ("thread", "queue"):
        if service_backend != "thread":
            raise ValueError(
                f"transport={transport!r} is incompatible with "
                f"service_backend={service_backend!r}."
            )
        return ThreadingTransport(**options)
    if kind == "process":
        if service_backend != "process":
            raise ValueError(
                f"transport='process' is incompatible with "
                f"service_backend={service_backend!r}."
            )
        options.setdefault("use_manager", True)
        return MPTransport(**options)
    if kind == "ray":
        if service_backend != "ray":
            raise ValueError(
                f"transport='ray' is incompatible with "
                f"service_backend={service_backend!r}."
            )
        return RayTransport(**options)
    if kind == "monarch":
        if service_backend != "monarch":
            raise ValueError(
                f"transport='monarch' is incompatible with "
                f"service_backend={service_backend!r}."
            )
        return MonarchTransport(**options)
    if kind == "shared_memory":
        if service_backend != "process":
            raise ValueError(
                "transport='shared_memory' requires service_backend='process'."
            )
        if request_spec is None or response_spec is None:
            raise ValueError(
                "transport='shared_memory' requires request_spec and response_spec."
            )
        options.setdefault("num_slots", num_clients or 1)
        return SharedMemoryTransport(request_spec, response_spec, **options)
    if kind == "direct":
        if service_backend != "thread":
            raise ValueError("transport='direct' requires service_backend='thread'.")
        if num_clients is None:
            raise ValueError("transport='direct' requires num_clients.")
        return SlotTransport(num_clients, **options)
    if kind == "distributed":
        if service_backend not in ("ray", "process"):
            raise ValueError(
                "transport='distributed' requires service_backend='ray' or 'process'."
            )
        if request_spec is None or response_spec is None:
            raise ValueError(
                "transport='distributed' requires request_spec and response_spec."
            )
        return _DistributedInferenceTransport(request_spec, response_spec, **options)
    raise ValueError(
        f"Unknown inference transport {transport!r}. Expected one of 'auto', "
        "'thread', 'process', 'ray', 'shared_memory', 'direct', or 'distributed'."
    )


def _inference_transport_kind(transport: InferenceTransport) -> str:
    """Return the compact selector for a resolved inference transport."""
    if isinstance(transport, SharedMemoryTransport):
        return "shared_memory"
    if isinstance(transport, SlotTransport):
        return "direct"
    if isinstance(transport, RayTransport):
        return "ray"
    if isinstance(transport, MonarchTransport):
        return "monarch"
    if isinstance(transport, MPTransport):
        return "process"
    if isinstance(transport, ThreadingTransport):
        return "thread"
    if isinstance(transport, _DistributedInferenceTransport):
        return "distributed"
    return "custom"


__all__ = [
    "_inference_transport_kind",
    "_make_inference_transport",
    "_validate_inference_transport_selection",
]
