# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import queue
import time

import torch
from tensordict.base import _is_leaf_nontensor, TensorDictBase
from tensordict.utils import NestedKey

from torchrl._comm import MailboxPeerClosedError, MailboxTransportError
from torchrl.modules.inference_server._client import (
    _NO_INTERACTION_TYPE_CODE,
    _REMOTE_INTERACTION_TYPE_KEY,
)
from torchrl.modules.inference_server._transport import InferenceTransport

_MISSING = object()
_PEER_CHECK_INTERVAL = 0.1


class _ProcessSlotFuture:
    """Future for one request in a fixed process-shared slot."""

    def __init__(self, client: _ProcessSlotClient):
        self._client = client
        self._outcome = _MISSING

    def done(self) -> bool:
        """Return whether the response is ready or the server has exited."""
        if self._outcome is not _MISSING:
            return True
        if self._client._response_event.is_set():
            return True
        peer_alive = self._client._peer_alive
        return peer_alive is not None and not peer_alive.is_set()

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Return the response, retaining the slot when a timeout elapses."""
        if self._outcome is _MISSING:
            self._outcome = self._client._receive(timeout)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _ProcessSlotClient:
    """Process-side client bound to one :class:`ProcessSlotTransport` slot."""

    def __init__(
        self,
        *,
        slot_id: int,
        request_slots: TensorDictBase,
        response_slots: TensorDictBase,
        request_keys: list[NestedKey],
        request_ready,
        submitted_at,
        response_status,
        response_event,
        exception_queue,
        work_semaphore,
        peer_alive,
        copy_result: bool,
    ):
        self._slot_id = slot_id
        self._request_slots = request_slots
        self._response_slots = response_slots
        self._request_keys = request_keys
        self._request_ready = request_ready
        self._submitted_at = submitted_at
        self._response_status = response_status
        self._response_event = response_event
        self._exception_queue = exception_queue
        self._work_semaphore = work_semaphore
        self._peer_alive = peer_alive
        self._copy_result = copy_result
        self._in_flight = False

    @property
    def client_id(self) -> int:
        """The fixed slot identifier assigned to this client."""
        return self._slot_id

    def submit(self, td: TensorDictBase) -> _ProcessSlotFuture:
        """Write a request into this client's slot and signal the server."""
        if self._in_flight:
            raise RuntimeError(
                "ProcessSlotTransport clients support one in-flight request. "
                "Wait for the current future before submitting another."
            )
        if self._peer_alive is not None and not self._peer_alive.is_set():
            raise MailboxPeerClosedError(
                "Inference server process closed before request submission."
            )
        request = td.select(*self._request_keys, strict=True)
        for key, value in request.items(include_nested=True, leaves_only=True):
            device = getattr(value, "device", None)
            if device is None or device.type != "cpu":
                raise ValueError(
                    "ProcessSlotTransport only accepts CPU tensors; got "
                    f"device {device} for key {key!r}. Keep environment workers "
                    "CPU-side and let the server move batches to the policy device."
                )

        slot = self._slot_id
        self._request_slots[slot].update_(request)
        interaction_code = td.get(_REMOTE_INTERACTION_TYPE_KEY, default=None)
        interaction_slot = self._request_slots[slot].get(_REMOTE_INTERACTION_TYPE_KEY)
        if interaction_code is None:
            interaction_slot.fill_(_NO_INTERACTION_TYPE_CODE)
        else:
            interaction_slot.copy_(interaction_code)
        self._response_event.clear()
        self._response_status[slot] = 0
        self._submitted_at[slot] = time.monotonic()
        self._request_ready[slot] = 1
        self._in_flight = True
        self._work_semaphore.release()
        return _ProcessSlotFuture(self)

    def __call__(
        self, td: TensorDictBase, timeout: float | None = None
    ) -> TensorDictBase:
        """Submit a request and block for its response."""
        return self.submit(td).result(timeout=timeout)

    def _receive(self, timeout: float | None) -> TensorDictBase | BaseException:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._response_event.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise queue.Empty(
                    f"Timeout waiting for process slot {self._slot_id}."
                ) from None
            wait_timeout = (
                _PEER_CHECK_INTERVAL
                if remaining is None
                else min(remaining, _PEER_CHECK_INTERVAL)
            )
            self._response_event.wait(timeout=wait_timeout)
            if self._response_event.is_set():
                break
            if self._peer_alive is not None:
                try:
                    peer_is_alive = self._peer_alive.is_set()
                except Exception as err:
                    raise MailboxTransportError(
                        "Failed to query the inference server's liveness."
                    ) from err
                if not peer_is_alive:
                    self._in_flight = False
                    raise MailboxPeerClosedError(
                        "Inference server process closed before replying to "
                        f"slot {self._slot_id}."
                    ) from None

        slot = self._slot_id
        try:
            if self._response_status[slot]:
                return self._exception_queue.get()
            result = self._response_slots[slot]
            return result.clone() if self._copy_result else result
        finally:
            self._response_event.clear()
            self._in_flight = False


class ProcessSlotTransport(InferenceTransport):
    """Fixed-slot shared-memory transport for environment worker processes.

    Each client owns one CPU shared-memory request/response slot. A worker
    copies an observation into its slot and releases a process-shared
    semaphore; the inference server sweeps ready slots in round-robin order,
    batches their tensor views, writes actions back, and wakes the matching
    workers. Only synchronization signals cross process boundaries on the
    inference hot path.

    This transport allows environment workers and a
    :class:`~torchrl.modules.inference_server.ProcessInferenceServer` to
    communicate without routing observations or actions through the driver.
    Each client permits one in-flight request, which naturally applies
    per-environment backpressure.

    Args:
        request_spec (TensorDictBase): representative request whose keys,
            shapes, dtypes and batch size define each request slot. Leaves
            must be CPU tensors.
        response_spec (TensorDictBase): representative response, including
            server-added keys such as ``"policy_version"``. Leaves must be
            CPU tensors.

    Keyword Args:
        num_slots (int): number of fixed slots and maximum number of clients.
        ctx (multiprocessing context, optional): context used for process
            synchronization primitives. Defaults to ``"spawn"``.
        copy_result (bool, optional): whether clients clone responses before
            returning them. Defaults to ``True``. If ``False``, a response is
            a borrowed view valid only until that client submits again.

    .. note::
        Create at most one client per environment worker. Unlike queue-based
        transports, clients do not need registration with the already-running
        server because every slot and signal is allocated at construction.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     ProcessSlotTransport,
        ... )
        >>> transport = ProcessSlotTransport(
        ...     TensorDict({"observation": torch.zeros(4)}),
        ...     TensorDict(
        ...         {
        ...             "action": torch.zeros(2),
        ...             "policy_version": torch.zeros((), dtype=torch.long),
        ...         }
        ...     ),
        ...     num_slots=4,
        ... )
        >>> client = transport.client()
        >>> policy = TensorDictModule(
        ...     torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> with InferenceServer(policy, transport, max_batch_size=4):
        ...     result = client(TensorDict({"observation": torch.randn(4)}))
        >>> result["action"].shape
        torch.Size([2])
    """

    _clients_require_registration = False

    def __init__(
        self,
        request_spec: TensorDictBase,
        response_spec: TensorDictBase,
        *,
        num_slots: int,
        ctx: mp.context.BaseContext | None = None,
        copy_result: bool = True,
    ):
        if isinstance(num_slots, bool) or not isinstance(num_slots, int):
            raise TypeError("num_slots must be an integer.")
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")
        self._num_slots = num_slots
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._copy_result = bool(copy_result)
        self._next_client_slot = 0
        self._next_slot = 0
        self._acquired_signals = 0

        request_keys = list(request_spec.keys(include_nested=True, leaves_only=True))
        self._request_keys = [
            key for key in request_keys if key != _REMOTE_INTERACTION_TYPE_KEY
        ]
        request_slot_spec = request_spec.clone(recurse=False)
        if _REMOTE_INTERACTION_TYPE_KEY not in request_keys:
            request_slot_spec.set(
                _REMOTE_INTERACTION_TYPE_KEY,
                torch.full(
                    request_spec.batch_size,
                    _NO_INTERACTION_TYPE_CODE,
                    dtype=torch.int8,
                ),
            )
        self._request_slots = self._make_slots(request_slot_spec, "request_spec")
        self._response_slots = self._make_slots(response_spec, "response_spec")
        self._response_keys = list(
            response_spec.keys(include_nested=True, leaves_only=True)
        )

        self._request_ready = self._ctx.Array("b", num_slots, lock=False)
        self._submitted_at = self._ctx.Array("d", num_slots, lock=False)
        self._response_status = self._ctx.Array("b", num_slots, lock=False)
        self._response_events = [self._ctx.Event() for _ in range(num_slots)]
        self._exception_queues = [self._ctx.SimpleQueue() for _ in range(num_slots)]
        self._work_semaphore = self._ctx.Semaphore(0)
        self._peer_alive = self._ctx.Event()
        self._peer_alive.set()

    def _make_slots(self, spec: TensorDictBase, argname: str) -> TensorDictBase:
        leaves = list(
            spec.items(
                include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor
            )
        )
        if not leaves:
            raise ValueError(f"{argname} must contain at least one tensor leaf.")
        for key, value in leaves:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "ProcessSlotTransport specs only support tensor leaves; "
                    f"{argname} has a {type(value).__name__} at key {key!r}."
                )
            if value.device.type != "cpu":
                raise ValueError(
                    "ProcessSlotTransport slots live in CPU shared memory; "
                    f"{argname} has a {value.device} tensor at key {key!r}."
                )
        return (
            spec.unsqueeze(0)
            .expand(self._num_slots, *spec.batch_size)
            .clone()
            .share_memory_()
        )

    def _set_peer_alive(self, alive_event) -> None:
        self._peer_alive = alive_event

    def client(self) -> _ProcessSlotClient:
        """Create a client bound to the next unused slot."""
        slot_id = self._next_client_slot
        self._next_client_slot += 1
        if slot_id >= self._num_slots:
            raise RuntimeError(
                f"ProcessSlotTransport has {self._num_slots} slots but client() "
                f"was called {slot_id + 1} times."
            )
        return _ProcessSlotClient(
            slot_id=slot_id,
            request_slots=self._request_slots,
            response_slots=self._response_slots,
            request_keys=self._request_keys,
            request_ready=self._request_ready,
            submitted_at=self._submitted_at,
            response_status=self._response_status,
            response_event=self._response_events[slot_id],
            exception_queue=self._exception_queues[slot_id],
            work_semaphore=self._work_semaphore,
            peer_alive=self._peer_alive,
            copy_result=self._copy_result,
        )

    def submit(self, td: TensorDictBase):
        """Reject unbound submissions; callers must first obtain a client."""
        raise NotImplementedError(
            "ProcessSlotTransport does not support submit(). Call client() "
            "to obtain a fixed-slot client."
        )

    def wait_for_work(self, timeout: float) -> None:
        """Wait until an environment worker marks a request slot ready."""
        if self._work_semaphore.acquire(timeout=timeout):
            self._acquired_signals += 1

    def drain(self, max_items: int) -> tuple[list[TensorDictBase], list[int]]:
        """Sweep ready slots in round-robin order."""
        items, callbacks, _submitted_at = self.drain_with_timing(max_items)
        return items, callbacks

    def drain_with_timing(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[int], list[float | None]]:
        """Sweep ready slots and return request submission timestamps."""
        items = []
        callbacks = []
        submitted_at = []
        start_slot = self._next_slot
        for offset in range(self._num_slots):
            slot = (start_slot + offset) % self._num_slots
            if not self._request_ready[slot]:
                continue
            self._request_ready[slot] = 0
            items.append(self._request_slots[slot].copy())
            callbacks.append(slot)
            submitted_at.append(self._submitted_at[slot])
            self._submitted_at[slot] = 0.0
            self._next_slot = (slot + 1) % self._num_slots
            if len(items) >= max_items:
                break

        signals_to_consume = len(items)
        acquired = min(signals_to_consume, self._acquired_signals)
        self._acquired_signals -= acquired
        signals_to_consume -= acquired
        for _ in range(signals_to_consume):
            self._work_semaphore.acquire(block=False)
        return items, callbacks, submitted_at

    def resolve(self, callback: int, result: TensorDictBase) -> None:
        """Copy a response into its slot and wake the owning worker."""
        self._response_slots[callback].update_(
            result.select(*self._response_keys, strict=True)
        )
        self._response_status[callback] = 0
        self._response_events[callback].set()

    def resolve_exception(self, callback: int, exc: BaseException) -> None:
        """Send a model exception to the owning worker and wake it."""
        self._exception_queues[callback].put(exc)
        self._response_status[callback] = 1
        self._response_events[callback].set()
