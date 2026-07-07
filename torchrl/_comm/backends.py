# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import cast, Literal, TypeAlias

ServiceBackend: TypeAlias = Literal[
    "direct", "thread", "process", "ray", "monarch", "distributed"
]
ServiceBackendAlias: TypeAlias = Literal["threading", "multiprocessing"]

_SERVICE_BACKEND_ALIASES = {
    "threading": "thread",
    "multiprocessing": "process",
}
_SERVICE_BACKENDS = frozenset(
    {"direct", "thread", "process", "ray", "monarch", "distributed"}
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
