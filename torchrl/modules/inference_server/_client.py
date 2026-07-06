# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

import torch
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.nn.probabilistic import interaction_type
from tensordict.utils import NestedKey

from torchrl._utils import logger as torchrl_logger
from torchrl.modules.inference_server._transport import InferenceTransport

_REMOTE_INTERACTION_TYPE_KEY = "_torchrl_inference_interaction_type"
_INTERACTION_TYPE_TO_CODE = {
    "mode": 0,
    "median": 1,
    "mean": 2,
    "random": 3,
    "deterministic": 4,
}


class _ImmediateFuture:
    def __init__(self, result: TensorDictBase | BaseException):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> TensorDictBase:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _InflightGuardedFuture:
    """Future proxy that frees an inflight slot when the request *completes*.

    Callback-capable futures (:class:`concurrent.futures.Future`) release
    through ``add_done_callback``, so a dropped or cancelled future cannot
    leak its slot. Pull-based futures (e.g. queue transports) release when a
    completed result is first observed through :meth:`result` or
    :meth:`done`. In both cases a ``result(timeout=...)`` that times out does
    **not** free the slot -- the request is still running on the server, and
    releasing early would let the number of genuinely inflight requests
    exceed ``max_inflight``. Garbage collection of the proxy releases the
    slot as a last resort so an abandoned pull-based future cannot
    permanently exhaust the guard.
    """

    def __init__(self, future, release: Callable[[], None]) -> None:
        self.future = future
        self._release_cb = release
        self._released = False
        self._release_lock = threading.Lock()
        add_done_callback = getattr(future, "add_done_callback", None)
        if add_done_callback is not None:
            add_done_callback(lambda _fut: self._release_once())

    def _release_once(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._release_cb()

    def done(self) -> bool:
        is_done = self.future.done()
        if is_done:
            self._release_once()
        return is_done

    def result(self, timeout: float | None = None) -> TensorDictBase:
        try:
            result = self.future.result(timeout=timeout)
        except (queue.Empty, TimeoutError):
            # The request is still inflight on the server; keep the slot.
            raise
        except BaseException:
            self._release_once()
            raise
        self._release_once()
        return result

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for missing attributes; route through
        # __dict__ explicitly so a partially-initialised proxy (e.g. during
        # unpickling) raises AttributeError instead of recursing.
        try:
            future = object.__getattribute__(self, "future")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(future, name)

    def __del__(self) -> None:
        try:
            self._release_once()
        except Exception:
            pass


class PolicyClientModule(TensorDictModuleBase):
    """TensorDict policy wrapper for remote inference-server clients.

    ``PolicyClientModule`` makes a transport client look like a TorchRL policy:
    it accepts a :class:`~tensordict.TensorDictBase`, submits it to an
    :class:`~torchrl.modules.inference_server.InferenceServer`, and returns the
    TensorDict produced by the remote policy. It can be passed anywhere a
    TensorDict policy module is expected.

    This class is the reference implementation of TorchRL's service *client*
    contract: it duck-types the domain interface (a policy client IS a
    TensorDict policy, so consumer code cannot tell local from remote), it is
    cheap and picklable (it can be handed to spawned workers), and it carries
    no lifecycle rights -- clients can call the service but never start or
    shut it down; only the owner that constructed the server can.

    .. note::
        Unlike a local :class:`~tensordict.nn.TensorDictModule`, the result
        crosses a transport boundary, so :meth:`forward` returns a *new*
        TensorDict rather than writing the ``out_keys`` into the input
        TensorDict. Use the return value; do not rely on in-place updates of
        the input.

    Args:
        client (Callable or InferenceTransport): actor-side inference client.
            If a transport is provided, ``transport.client()`` is called.

    Keyword Args:
        in_keys (sequence of NestedKey, optional): input keys advertised by the
            module. The full input TensorDict is still sent to the server.
        out_keys (sequence of NestedKey, optional): output keys advertised by
            the module.
        max_inflight (int, optional): maximum number of unresolved
            asynchronous requests submitted through this module; further
            :meth:`submit` calls block until a slot frees up. A slot is
            freed when its request *completes* (including errors), not when
            ``result()`` is first called; a timed-out ``result()`` keeps the
            slot. Must be at least ``1``. ``None`` means unbounded.
        target_policy_version (int or Callable[[], int], optional): expected
            latest policy version used for bounded-staleness checks. A
            callable (e.g. ``lambda: server.policy_version``) is re-evaluated
            on every check, providing a live version source.
        max_policy_lag (int, optional): maximum allowed
            ``target_policy_version - returned_policy_version``. The lag is
            computed against the *oldest* version element in the result.
        policy_version_key (NestedKey or None, optional): key that contains
            the behavior policy version returned by the server. Must match the
            server's ``policy_version_key``; a mismatch triggers a one-time
            warning when the staleness guard is enabled. ``None`` disables the
            guard. Defaults to ``"policy_version"``.
        propagate_interaction_type (bool, optional): if ``True``, the active
            :func:`tensordict.nn.interaction_type` is attached to each request
            so the server can execute the remote policy under the caller's
            exploration context. Defaults to ``False``.

    .. note::
        Version tracking is an instance of the generic *service-stamped
        metadata* pattern: a service may stamp every response with metadata
        describing the state it was served from (here: the behavior-policy
        version), and clients may enforce freshness constraints on that
        metadata (here: bounded staleness through ``max_policy_lag``). Other
        services can reuse the same shape for their own response metadata.

    .. note::
        The server-side version counter is independent from the
        :class:`~torchrl.envs.llm.transforms.PolicyVersion` transform and the
        collectors' ``track_policy_version`` mechanism, but it uses the same
        default ``"policy_version"`` key. When combining an inference-server
        client with an env that carries the ``PolicyVersion`` transform, set
        distinct keys to avoid one writer overwriting the other.

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
        target_policy_version: int | Callable[[], int] | None = None,
        max_policy_lag: int | None = None,
        policy_version_key: NestedKey | None = "policy_version",
        propagate_interaction_type: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(client, InferenceTransport):
            client = client.client()
        if max_inflight is not None and max_inflight < 1:
            raise ValueError(
                f"max_inflight must be at least 1 (got {max_inflight}); "
                "use None to disable the guard."
            )
        self.client = client
        self.in_keys = list(in_keys or [])
        self.out_keys = list(out_keys or [])
        self.max_inflight = max_inflight
        self.target_policy_version = target_policy_version
        self.max_policy_lag = max_policy_lag
        self.policy_version_key = policy_version_key
        self._warned_missing_version = False
        self.propagate_interaction_type = bool(propagate_interaction_type)
        self._inflight_sem = (
            threading.BoundedSemaphore(max_inflight)
            if max_inflight is not None
            else None
        )

    def __getstate__(self):
        # Semaphores are not picklable; the guard is a per-process resource,
        # so a fresh one (with a full complement of slots) is rebuilt on
        # unpickling. This keeps clients picklable per the Client contract.
        state = super().__getstate__()
        state = dict(state)
        state["_inflight_sem"] = None
        return state

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if self.max_inflight is not None:
            self._inflight_sem = threading.BoundedSemaphore(self.max_inflight)

    def _acquire_inflight(self) -> Callable[[], None]:
        if self._inflight_sem is None:
            return lambda: None
        self._inflight_sem.acquire()
        return self._inflight_sem.release

    def _check_policy_lag(self, tensordict: TensorDictBase) -> None:
        if self.target_policy_version is None or self.max_policy_lag is None:
            return
        version = (
            tensordict.get(self.policy_version_key, default=None)
            if self.policy_version_key is not None
            else None
        )
        if version is None:
            # A user who configured the guard expects protection; a missing
            # key (server annotations disabled, or client/server key
            # mismatch) must not silently disable it.
            if not self._warned_missing_version:
                torchrl_logger.warning(
                    f"PolicyClientModule: max_policy_lag is set but the "
                    f"result carries no {self.policy_version_key!r} entry. "
                    f"The staleness guard is inactive; check that the server "
                    f"annotates versions and that policy_version_key matches "
                    f"on both sides."
                )
                self._warned_missing_version = True
            return
        if isinstance(version, torch.Tensor):
            # Bound the staleness of the worst-case (oldest) element.
            version = int(version.min().item())
        else:
            version = int(version)
        target = self.target_policy_version
        if callable(target):
            target = int(target())
        lag = target - version
        if lag > self.max_policy_lag:
            raise RuntimeError(
                f"Remote policy result is too stale: version={version}, "
                f"target_policy_version={target}, "
                f"max_policy_lag={self.max_policy_lag}."
            )

    def submit(self, tensordict: TensorDictBase) -> Future | _ImmediateFuture:
        """Submit a TensorDict request and return a future-like object.

        Args:
            tensordict (TensorDictBase): observation TensorDict to send to the
                remote policy.

        Returns:
            Future-like object whose ``result()`` method returns a TensorDict.
            When the wrapped client exposes ``submit`` this is the transport's
            :class:`~concurrent.futures.Future` and submission errors raise
            synchronously; for a plain callable client the call runs eagerly
            and errors are deferred to ``result()`` on a reduced future that
            only implements ``done()`` and ``result()``.
        """
        if self.propagate_interaction_type:
            current_interaction_type = interaction_type()
            if current_interaction_type is not None:
                tensordict = tensordict.clone(recurse=False)
                tensordict.set(
                    _REMOTE_INTERACTION_TYPE_KEY,
                    torch.full(
                        tensordict.batch_size,
                        _INTERACTION_TYPE_TO_CODE[current_interaction_type.value],
                        dtype=torch.int8,
                        device=tensordict.device or torch.device("cpu"),
                    ),
                )
        release = self._acquire_inflight()
        submit = getattr(self.client, "submit", None)
        if submit is None:
            # The plain-callable path runs eagerly, so the request has
            # already completed here: free the slot immediately.
            try:
                result = self.client(tensordict)
                return _ImmediateFuture(result)
            except Exception as exc:
                return _ImmediateFuture(exc)
            finally:
                release()
        try:
            future = submit(tensordict)
        except BaseException:
            release()
            raise
        return _InflightGuardedFuture(future, release)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        result = self.submit(tensordict).result()
        self._check_policy_lag(result)
        return result
