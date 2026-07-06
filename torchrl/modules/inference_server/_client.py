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

from torchrl.modules.inference_server._transport import InferenceTransport

_REMOTE_INTERACTION_TYPE_KEY = "_torchrl_inference_interaction_type"
# Code stamped when the caller has no active interaction context; keeping the
# key always present makes server batches homogeneous in key structure.
_NO_INTERACTION_TYPE_CODE = -1
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

    .. note::
        The caller's active :func:`tensordict.nn.interaction_type` is
        automatically attached to every transport request, and the server
        executes the remote policy under that exploration context -- exactly
        as a local policy would see it. In-process (plain callable) clients
        need no propagation since the caller's context is already active.

    .. note::
        Version tracking is an instance of the generic *service-stamped
        metadata* pattern: a service may stamp every response with metadata
        describing the state it was served from (here: the behavior-policy
        version), and the data pipeline may enforce freshness constraints on
        that metadata. Bounded-staleness enforcement lives in the replay
        buffer through :class:`~torchrl.envs.transforms.PolicyAgeFilter`,
        which silently drops too-old elements on extend or sample instead of
        raising in the consumer.

    .. note::
        The default ``"policy_version"`` key is shared on purpose with the
        :class:`~torchrl.envs.llm.transforms.PolicyVersion` transform and the
        collectors' ``track_policy_version`` mechanism: they stamp the same
        concept (the behavior-policy version that produced the data), so
        consumers such as
        :class:`~torchrl.envs.transforms.PolicyAgeFilter` can read it without
        caring which component wrote it. Both counters are driven by the same
        weight-update cascade (``update_policy_weights_``), so they agree when
        wired through a weight-sync scheme. Keep a single authoritative writer
        per data stream -- in a policy-server topology that is the server,
        which owns the weights; do not stack an independently-initialized
        ``PolicyVersion`` transform on top of server-stamped data.

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
        release = self._acquire_inflight()
        submit = getattr(self.client, "submit", None)
        if submit is not None:
            # Cross-boundary request: carry the caller's exploration context
            # so the server-side forward behaves like a local call. The key
            # is always attached (with a sentinel when no context is active)
            # so server batches stay homogeneous in key structure.
            current_interaction_type = interaction_type()
            code = (
                _INTERACTION_TYPE_TO_CODE[current_interaction_type.value]
                if current_interaction_type is not None
                else _NO_INTERACTION_TYPE_CODE
            )
            tensordict = tensordict.clone(recurse=False)
            tensordict.set(
                _REMOTE_INTERACTION_TYPE_KEY,
                torch.full(
                    tensordict.batch_size,
                    code,
                    dtype=torch.int8,
                    device=tensordict.device or torch.device("cpu"),
                ),
            )
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
        return self.submit(tensordict).result()
