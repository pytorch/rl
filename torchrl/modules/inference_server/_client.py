# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from torchrl.modules.inference_server._transport import InferenceTransport


class _ImmediateFuture:
    def __init__(self, result: TensorDictBase | BaseException):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> TensorDictBase:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _ReleaseOnResultFuture:
    def __init__(self, future, release: Callable[[], None]):
        self._future = future
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def _release_once(self) -> None:
        with self._lock:
            if not self._released:
                self._released = True
                self._release()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> TensorDictBase:
        try:
            return self._future.result(timeout=timeout)
        finally:
            self._release_once()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._future, name)


class PolicyClientModule(TensorDictModuleBase):
    """TensorDict policy wrapper for remote inference-server clients.

    ``PolicyClientModule`` makes a transport client look like a TorchRL policy:
    it accepts a :class:`~tensordict.TensorDictBase`, submits it to an
    :class:`~torchrl.modules.inference_server.InferenceServer`, and returns the
    TensorDict produced by the remote policy. It can be passed anywhere a
    TensorDict policy module is expected.

    Args:
        client (Callable or InferenceTransport): actor-side inference client.
            If a transport is provided, ``transport.client()`` is called.

    Keyword Args:
        in_keys (sequence of NestedKey, optional): input keys advertised by the
            module. The full input TensorDict is still sent to the server.
        out_keys (sequence of NestedKey, optional): output keys advertised by
            the module.
        max_inflight (int, optional): maximum number of unresolved asynchronous
            requests submitted through this module. ``None`` means unbounded.
        target_policy_version (int, optional): expected latest policy version
            used for bounded-staleness checks.
        max_policy_lag (int, optional): maximum allowed
            ``target_policy_version - returned_policy_version``.
        policy_version_key (NestedKey, optional): key that contains the
            behavior policy version returned by the server. Defaults to
            ``"policy_version"``.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     PolicyClientModule,
        ...     ThreadingTransport,
        ... )
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> transport = ThreadingTransport()
        >>> server = InferenceServer(policy, transport).start()
        >>> remote_policy = PolicyClientModule(
        ...     transport, in_keys=["observation"], out_keys=["action"]
        ... )
        >>> td = remote_policy(TensorDict({"observation": torch.randn(4)}))
        >>> "action" in td.keys()
        True
        >>> server.shutdown()
    """

    def __init__(
        self,
        client: Callable[[TensorDictBase], TensorDictBase] | InferenceTransport,
        *,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        max_inflight: int | None = None,
        target_policy_version: int | None = None,
        max_policy_lag: int | None = None,
        policy_version_key: NestedKey = "policy_version",
    ) -> None:
        super().__init__()
        if isinstance(client, InferenceTransport):
            client = client.client()
        self.client = client
        self.in_keys = list(in_keys or [])
        self.out_keys = list(out_keys or [])
        self.max_inflight = max_inflight
        self.target_policy_version = target_policy_version
        self.max_policy_lag = max_policy_lag
        self.policy_version_key = policy_version_key
        self._inflight_sem = (
            threading.BoundedSemaphore(max_inflight)
            if max_inflight is not None
            else None
        )

    def _acquire_inflight(self) -> Callable[[], None]:
        if self._inflight_sem is None:
            return lambda: None
        self._inflight_sem.acquire()
        return self._inflight_sem.release

    def _check_policy_lag(self, tensordict: TensorDictBase) -> None:
        if self.target_policy_version is None or self.max_policy_lag is None:
            return
        version = tensordict.get(self.policy_version_key, default=None)
        if version is None:
            return
        if isinstance(version, torch.Tensor):
            version = int(version.max().item())
        else:
            version = int(version)
        lag = self.target_policy_version - version
        if lag > self.max_policy_lag:
            raise RuntimeError(
                f"Remote policy result is too stale: version={version}, "
                f"target_policy_version={self.target_policy_version}, "
                f"max_policy_lag={self.max_policy_lag}."
            )

    def submit(self, tensordict: TensorDictBase):
        """Submit a TensorDict request and return a future-like object.

        Args:
            tensordict (TensorDictBase): observation TensorDict to send to the
                remote policy.

        Returns:
            Future-like object whose ``result()`` method returns a TensorDict.
        """
        release = self._acquire_inflight()
        submit = getattr(self.client, "submit", None)
        if submit is None:
            try:
                result = self.client(tensordict)
                return _ReleaseOnResultFuture(_ImmediateFuture(result), release)
            except BaseException as exc:
                return _ReleaseOnResultFuture(_ImmediateFuture(exc), release)
        try:
            future = submit(tensordict)
        except BaseException:
            release()
            raise
        return _ReleaseOnResultFuture(future, release)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        result = self.submit(tensordict).result()
        self._check_policy_lag(result)
        return result


RemotePolicy = PolicyClientModule
