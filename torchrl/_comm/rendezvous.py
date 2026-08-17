# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import pickle
import time
from collections.abc import Callable, MutableMapping
from datetime import timedelta
from typing import Any, Protocol

_has_ray = importlib.util.find_spec("ray") is not None


class Rendezvous(Protocol):
    """Key/value rendezvous used by distributed connection handshakes."""

    def publish(self, key: str, value: Any) -> None:
        ...

    def read(self, key: str) -> Any:
        ...

    def wait(self, key: str, timeout: float | None = None) -> Any:
        ...


class MappingRendezvous:
    """Rendezvous adapter for a shared mapping."""

    def __init__(self, mapping: MutableMapping[str, Any]) -> None:
        self._mapping = mapping

    def publish(self, key: str, value: Any) -> None:
        self._mapping[key] = value

    def read(self, key: str) -> Any:
        return self._mapping[key]

    def wait(self, key: str, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        while key not in self._mapping:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for rendezvous key {key!r}.")
            time.sleep(0.01)
        return self._mapping[key]


class TCPStoreRendezvous:
    """Rendezvous adapter for :class:`torch.distributed.TCPStore`."""

    def __init__(
        self,
        store,
        *,
        encode: Callable[[Any], bytes] = pickle.dumps,
        decode: Callable[[bytes], Any] = pickle.loads,
    ) -> None:
        self._store = store
        self._encode = encode
        self._decode = decode

    def publish(self, key: str, value: Any) -> None:
        self._store.set(key, self._encode(value))

    def read(self, key: str) -> Any:
        return self._decode(self._store.get(key))

    def wait(self, key: str, timeout: float | None = None) -> Any:
        if timeout is None:
            self._store.wait([key])
        else:
            self._store.wait([key], timedelta(seconds=timeout))
        return self.read(key)


class RayRendezvous:
    """Rendezvous adapter for a Ray actor exposing ``set`` and ``get``."""

    def __init__(self, actor) -> None:
        if not _has_ray:
            raise ImportError("Ray is required for RayRendezvous.")
        self._actor = actor
        self._ray = None

    @property
    def ray(self):
        if self._ray is None:
            import ray

            self._ray = ray
        return self._ray

    def publish(self, key: str, value: Any) -> None:
        self.ray.get(self._actor.set.remote(key, value))

    def read(self, key: str) -> Any:
        return self.ray.get(self._actor.get.remote(key))

    def wait(self, key: str, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self.read(key)
            except KeyError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for rendezvous key {key!r}."
                    ) from None
                time.sleep(0.01)
