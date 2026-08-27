# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator
from typing import cast, Literal, TypeAlias

ServiceBackend: TypeAlias = Literal[
    "direct",
    "thread",
    "process",
    "ray",
    "rpc",
    "submitit",
    "monarch",
    "distributed",
]
ServiceBackendAlias: TypeAlias = Literal["threading", "multiprocessing"]
TransportBackend: TypeAlias = Literal[
    "auto",
    "direct",
    "thread",
    "queue",
    "process",
    "shared_memory",
    "ray",
    "monarch",
    "distributed",
]

_SERVICE_BACKEND_ALIASES = {
    "threading": "thread",
    "multiprocessing": "process",
}
_SERVICE_BACKENDS = frozenset(
    {
        "direct",
        "thread",
        "process",
        "ray",
        "rpc",
        "submitit",
        "monarch",
        "distributed",
    }
)
_TRANSPORT_BACKENDS = frozenset(
    {
        "auto",
        "direct",
        "thread",
        "queue",
        "process",
        "shared_memory",
        "ray",
        "monarch",
        "distributed",
    }
)

_SERVICE_BACKEND_CONTEXT: contextvars.ContextVar[tuple[int, ServiceBackend] | None]
_SERVICE_BACKEND_CONTEXT = contextvars.ContextVar(
    "torchrl_service_backend", default=None
)
_TRANSPORT_BACKEND_CONTEXT: contextvars.ContextVar[tuple[int, TransportBackend] | None]
_TRANSPORT_BACKEND_CONTEXT = contextvars.ContextVar(
    "torchrl_transport_backend", default=None
)


def normalize_service_backend(
    backend: ServiceBackend | ServiceBackendAlias | str,
) -> ServiceBackend:
    """Return the canonical spelling of a TorchRL service backend.

    The long ``threading`` and ``multiprocessing`` spellings remain permanent
    aliases because they are part of the stable ``AsyncBatchedCollector`` API.
    New APIs expose the shorter canonical atoms.

    Args:
        backend: Backend name to normalize.

    Returns:
        The canonical backend name.

    Raises:
        ValueError: If the backend is unknown.
    """
    canonical = _SERVICE_BACKEND_ALIASES.get(backend, backend)
    if canonical not in _SERVICE_BACKENDS:
        choices = sorted(_SERVICE_BACKENDS | set(_SERVICE_BACKEND_ALIASES))
        raise ValueError(
            f"Unsupported service backend {backend!r}. Expected one of {choices}."
        )
    return cast(ServiceBackend, canonical)


def _normalize_transport_backend(backend: TransportBackend | str) -> TransportBackend:
    """Return the canonical spelling of a TorchRL payload transport.

    Args:
        backend: Transport selector to validate.

    Returns:
        The validated transport selector.

    Raises:
        ValueError: If the transport is unknown.
    """
    if backend not in _TRANSPORT_BACKENDS:
        raise ValueError(
            f"Unsupported transport backend {backend!r}. "
            f"Expected one of {sorted(_TRANSPORT_BACKENDS)}."
        )
    return cast(TransportBackend, backend)


def _context_value(
    context: contextvars.ContextVar[tuple[int, str] | None]
) -> str | None:
    value = context.get()
    if value is None:
        return None
    pid, backend = value
    if pid != os.getpid():
        # ``ContextVar`` state is copied by ``fork``. Backend construction
        # defaults belong to the process that entered the context and must not
        # recursively select another owner inside a worker process.
        return None
    return backend


def _get_service_backend() -> ServiceBackend | None:
    return cast(ServiceBackend | None, _context_value(_SERVICE_BACKEND_CONTEXT))


def _get_transport_backend() -> TransportBackend | None:
    return cast(TransportBackend | None, _context_value(_TRANSPORT_BACKEND_CONTEXT))


def _contextual_backend_error(
    message: str,
    *,
    service: bool = False,
    transport: bool = False,
) -> str:
    """Annotate a validation error with the scoped default that selected it."""
    contexts = []
    if service:
        contexts.append("torchrl.service_backend")
    if transport:
        contexts.append("torchrl.transport_backend")
    if not contexts:
        return message
    context_names = " and ".join(contexts)
    return f"{message} (selected by an enclosing {context_names} context.)"


def _resolve_service_backend(
    backend: ServiceBackend | ServiceBackendAlias | str | None,
    *,
    default: ServiceBackend | ServiceBackendAlias | str,
) -> ServiceBackend:
    if backend is not None:
        return normalize_service_backend(backend)
    contextual_backend = _get_service_backend()
    if contextual_backend is not None:
        return contextual_backend
    return normalize_service_backend(default)


def _resolve_transport_backend(
    backend: TransportBackend | str | None,
    *,
    default: TransportBackend | str,
) -> TransportBackend:
    if backend is not None:
        return _normalize_transport_backend(backend)
    contextual_backend = _get_transport_backend()
    if contextual_backend is not None:
        return contextual_backend
    return _normalize_transport_backend(default)


@contextlib.contextmanager
def service_backend(
    backend: ServiceBackend | ServiceBackendAlias | str,
) -> Iterator[None]:
    """Set the default service backend within the current context.

    Explicit constructor arguments take precedence over this scoped default.
    The context is local to the current execution context and process, restores
    the previous value on exit, and is not inherited by forked workers.

    Args:
        backend: Service backend used by compatible objects constructed in the
            context.

    Examples:
        >>> from torchrl import service_backend
        >>> from torchrl.data import ReplayBuffer
        >>> with service_backend("direct"):
        ...     replay = ReplayBuffer()
        >>> replay.service_backend
        'direct'
    """
    backend = normalize_service_backend(backend)
    token = _SERVICE_BACKEND_CONTEXT.set((os.getpid(), backend))
    try:
        yield
    finally:
        _SERVICE_BACKEND_CONTEXT.reset(token)


@contextlib.contextmanager
def transport_backend(backend: TransportBackend | str) -> Iterator[None]:
    """Set the default payload transport within the current context.

    Explicit ``transport`` constructor arguments take precedence. Components
    still validate whether the selected transport is compatible with their
    service backend.

    Args:
        backend: Transport used by compatible objects constructed in the
            context.

    Examples:
        >>> from torchrl import service_backend, transport_backend
        >>> from torchrl.data import ReplayBuffer
        >>> with service_backend("direct"), transport_backend("direct"):
        ...     replay = ReplayBuffer()
        >>> replay.service_backend
        'direct'
    """
    backend = _normalize_transport_backend(backend)
    token = _TRANSPORT_BACKEND_CONTEXT.set((os.getpid(), backend))
    try:
        yield
    finally:
        _TRANSPORT_BACKEND_CONTEXT.reset(token)
