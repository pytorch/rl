# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import threading
from typing import Any

_has_ray = importlib.util.find_spec("ray") is not None


class _RayRuntimeLease:
    """Process-local lease for a Ray runtime used by TorchRL owners."""

    _lock = threading.RLock()
    _leases = 0
    _owns_runtime = False

    def __init__(self) -> None:
        self._released = False

    @classmethod
    def acquire(cls, ray_init_config: dict[str, Any] | None = None) -> _RayRuntimeLease:
        if not _has_ray:
            raise ImportError("Ray is required for service_backend='ray'.")
        import ray

        with cls._lock:
            if not ray.is_initialized():
                ray.init(**dict(ray_init_config or {}))
                cls._owns_runtime = True
            cls._leases += 1
        return cls()

    def release(self) -> None:
        if self._released:
            return
        import ray

        type(self)._lock.acquire()
        try:
            if self._released:
                return
            self._released = True
            type(self)._leases -= 1
            if type(self)._leases == 0 and type(self)._owns_runtime:
                if ray.is_initialized():
                    ray.shutdown()
                type(self)._owns_runtime = False
        finally:
            type(self)._lock.release()


class _RayActorLiveness:
    """Picklable liveness flag backed by a Ray actor system probe."""

    def __init__(self, actor) -> None:
        self._actor = actor

    def is_set(self) -> bool:
        import ray

        try:
            ray.get(self._actor.__ray_ready__.remote(), timeout=0.5)
            return True
        except Exception:
            return False


def _set_ray_client_liveness(client, actor) -> None:
    """Attach actor liveness to a private distributed client."""
    if hasattr(client, "_peer_alive"):
        client._peer_alive = _RayActorLiveness(actor)


__all__ = [
    "_RayActorLiveness",
    "_RayRuntimeLease",
    "_set_ray_client_liveness",
]
