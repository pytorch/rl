# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import queue

import torch
from tensordict.base import _is_leaf_nontensor, TensorDictBase
from tensordict.utils import NestedKey

from torchrl._comm import MailboxClient, MailboxFuture
from torchrl.modules.inference_server._queue_transport import QueueBasedTransport

_MISSING = object()


class _SharedMemoryFuture:
    """Future for one in-flight :class:`SharedMemoryTransport` request.

    Wraps a :class:`~torchrl._comm.MailboxFuture` that resolves to the slot
    index; on success the response is read from the shared response slot and
    the slot is released back to the free-slot pool.
    """

    def __init__(
        self,
        inner: MailboxFuture,
        slot: int,
        response_slots: TensorDictBase,
        free_slots,
        copy_result: bool,
    ):
        self._inner = inner
        self._slot = slot
        self._response_slots = response_slots
        self._free_slots = free_slots
        self._copy_result = copy_result
        self._outcome = _MISSING

    def done(self) -> bool:
        """Return ``True`` when the result can be read without blocking."""
        if self._outcome is not _MISSING:
            return True
        return self._inner.done()

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Return the inference result or raise its remote exception.

        Raises :class:`queue.Empty` when *timeout* elapses before the server
        replies; the request stays in flight and the slot is kept, so
        ``result`` can be called again.
        """
        if self._outcome is _MISSING:
            try:
                self._inner.result(timeout=timeout)
            except queue.Empty:
                # Timeout: the request is still in flight and the server may
                # still write to the slot -- do not release it.
                raise
            except BaseException as exc:
                self._free_slots.put(self._slot)
                self._outcome = exc
            else:
                result = self._response_slots[self._slot]
                if self._copy_result:
                    result = result.clone()
                self._free_slots.put(self._slot)
                self._outcome = result
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _SharedMemorySlotClient:
    """Actor-side client for :class:`SharedMemoryTransport`.

    Writes request tensors into a shared-memory slot and submits only the
    slot index through the request queue.
    """

    def __init__(
        self,
        mailbox_client: MailboxClient,
        request_slots: TensorDictBase,
        response_slots: TensorDictBase,
        free_slots,
        request_keys: list[NestedKey],
        *,
        copy_result: bool,
    ):
        self._mailbox_client = mailbox_client
        self._request_slots = request_slots
        self._response_slots = response_slots
        self._free_slots = free_slots
        self._request_keys = request_keys
        self._copy_result = copy_result

    @property
    def client_id(self) -> int:
        """The identifier assigned by the owning transport."""
        return self._mailbox_client.client_id

    def submit(self, td: TensorDictBase) -> _SharedMemoryFuture:
        """Copy the request into a free slot and enqueue its header.

        Blocks until a slot is available when all ``num_slots`` slots hold
        in-flight requests (backpressure).
        """
        request = td.select(*self._request_keys, strict=True)
        for key, value in request.items(include_nested=True, leaves_only=True):
            device = getattr(value, "device", None)
            if device is None or device.type != "cpu":
                raise ValueError(
                    f"SharedMemoryTransport only accepts CPU tensors; got "
                    f"device {device} for key {key!r}. Keep env workers "
                    "CPU-side and let the server move batches to the policy "
                    "device."
                )
        slot = self._free_slots.get()
        try:
            self._request_slots[slot].update_(request)
            inner = self._mailbox_client.submit(slot)
        except BaseException:
            self._free_slots.put(slot)
            raise
        return _SharedMemoryFuture(
            inner,
            slot,
            self._response_slots,
            self._free_slots,
            self._copy_result,
        )

    def __call__(
        self, td: TensorDictBase, timeout: float | None = None
    ) -> TensorDictBase:
        """Submit a request and block for its result."""
        return self.submit(td).result(timeout=timeout)


class SharedMemoryTransport(QueueBasedTransport):
    """Cross-process transport backed by shared-memory TensorDict slots.

    Unlike :class:`~torchrl.modules.inference_server.MPTransport`, which
    pickles full request/response TensorDicts through multiprocessing queues,
    this transport preallocates two CPU shared-memory slot banks (one for
    requests, one for responses) and passes only slot indices through the
    queues. This removes per-request serialization of large payloads (e.g.
    image observations) from the hot path.

    A slot is owned by exactly one in-flight request: the client acquires a
    slot from a shared free-slot pool, copies the request tensors into it,
    and releases it once the response has been read. ``num_slots`` therefore
    bounds the number of concurrently in-flight requests; when all slots are
    busy, :meth:`~._SharedMemorySlotClient.submit` blocks until one is
    released.

    Device rules: slots live in CPU shared memory, clients must submit CPU
    tensors (a CUDA leaf raises a :class:`ValueError`), and the server owns
    all device transfers -- batches are moved to the policy device by
    :class:`~torchrl.modules.inference_server.InferenceServer` and results
    are copied back into the CPU response slots by :meth:`resolve`.

    Args:
        request_spec (TensorDictBase): a representative single request. Its
            keys, shapes, dtypes, and batch size define the request slot
            layout. All leaves must be CPU tensors.
        response_spec (TensorDictBase): a representative single response
            (including any server-added keys to forward, such as
            ``"policy_version"``). All leaves must be CPU tensors.

    Keyword Args:
        num_slots (int): number of preallocated slots, i.e. the maximum
            number of concurrently in-flight requests across all clients.
        ctx (multiprocessing context, optional): the multiprocessing context
            used for the control queues. Defaults to
            ``mp.get_context("spawn")``.
        copy_result (bool, optional): if ``True`` (default),
            ``Future.result()`` returns a clone of the response slot. If
            ``False``, it returns a view into the shared response slot that
            is only valid until the slot is reused by a later request;
            callers must consume (or copy) it before submitting again.

    .. note::
        Only the keys declared in the specs are transmitted: extra keys on
        submitted tensordicts and on model outputs are silently dropped, and
        a missing declared key raises a :class:`KeyError`. Non-tensor leaves
        are not supported; encode small metadata as tensors (e.g. static
        instruction ids) or use :class:`MPTransport`.

    .. note::
        As with :class:`MPTransport`, clients must be created with
        :meth:`client` in the owning process **before** spawning child
        processes, so that their response queues and the shared slot banks
        are inherited by the workers.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     SharedMemoryTransport,
        ... )
        >>> request_spec = TensorDict({"observation": torch.zeros(4)})
        >>> response_spec = TensorDict(
        ...     {
        ...         "action": torch.zeros(2),
        ...         "policy_version": torch.zeros((), dtype=torch.long),
        ...     }
        ... )
        >>> transport = SharedMemoryTransport(
        ...     request_spec, response_spec, num_slots=8
        ... )
        >>> client = transport.client()  # create before spawning workers
        >>> policy = TensorDictModule(
        ...     torch.nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> with InferenceServer(policy, transport, max_batch_size=4):
        ...     result = client(TensorDict({"observation": torch.randn(4)}))
        >>> assert result["action"].shape == (2,)
    """

    def __init__(
        self,
        request_spec: TensorDictBase,
        response_spec: TensorDictBase,
        *,
        num_slots: int,
        ctx: mp.context.BaseContext | None = None,
        copy_result: bool = True,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError(f"num_slots must be a positive integer, got {num_slots}.")
        self._num_slots = int(num_slots)
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._copy_result = bool(copy_result)

        self._request_slots = self._make_slots(request_spec, "request_spec")
        self._response_slots = self._make_slots(response_spec, "response_spec")
        self._request_keys = list(
            request_spec.keys(include_nested=True, leaves_only=True)
        )
        self._response_keys = list(
            response_spec.keys(include_nested=True, leaves_only=True)
        )

        self._free_slots = self._ctx.Queue()
        for slot in range(self._num_slots):
            self._free_slots.put(slot)

        self._request_queue = self._ctx.Queue()
        self._response_queues: dict[int, mp.Queue] = {}
        # Server-liveness flag baked into every client (see MPTransport): a
        # process-backed server owner clears it when the server process exits
        # so blocked clients raise MailboxPeerClosedError instead of hanging.
        peer_alive = self._ctx.Event()
        peer_alive.set()
        self._set_peer_alive(peer_alive)

    def _make_slots(self, spec: TensorDictBase, argname: str) -> TensorDictBase:
        # _is_leaf_nontensor surfaces NonTensorData leaves (excluded by the
        # default leaf iterator) so they are rejected instead of ignored.
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
                    f"SharedMemoryTransport specs only support tensor leaves; "
                    f"{argname} has a {type(value).__name__} at key {key!r}. "
                    "Encode small metadata as tensors or use MPTransport."
                )
            if value.device.type != "cpu":
                raise ValueError(
                    f"SharedMemoryTransport slots live in CPU shared memory; "
                    f"{argname} has a {value.device} tensor at key {key!r}."
                )
        return (
            spec.unsqueeze(0)
            .expand(self._num_slots, *spec.batch_size)
            .clone()
            .share_memory_()
        )

    def _make_response_queue(self) -> mp.Queue:
        return self._ctx.Queue()

    # -- actor API ------------------------------------------------------------

    def client(self) -> _SharedMemorySlotClient:
        """Create an actor-side client with a dedicated response queue.

        Must be called in the owning process **before** spawning children so
        that the response queue and the shared slot banks are inherited.

        Returns:
            A :class:`_SharedMemorySlotClient` that can be passed to a child
            process as an argument to :class:`multiprocessing.Process`.
        """
        inner = self._ensure_mailbox().client()
        return _SharedMemorySlotClient(
            inner,
            self._request_slots,
            self._response_slots,
            self._free_slots,
            self._request_keys,
            copy_result=self._copy_result,
        )

    # -- server API -----------------------------------------------------------

    def drain_with_timing(
        self, max_items: int
    ) -> tuple[
        list[TensorDictBase],
        list[tuple[tuple[int, int], int]],
        list[float | None],
    ]:
        """Dequeue request headers and return views into the request slots."""
        slots, mailbox_callbacks, submitted_at = self._ensure_mailbox().drain(max_items)
        # Shallow copies keep the leaf tensors zero-copy (shared storage) but
        # are unlocked, so the model can write its output keys into the
        # collated batch (share_memory_() locks the slot bank and its views).
        items = [self._request_slots[slot].copy() for slot in slots]
        callbacks = list(zip(mailbox_callbacks, slots))
        return items, callbacks, submitted_at

    def resolve(
        self, callback: tuple[tuple[int, int], int], result: TensorDictBase
    ) -> None:
        """Copy the result into the response slot and notify the client.

        Result tensors on a non-CPU device are copied back to the CPU slots
        leaf-by-leaf, so no CUDA tensor ever crosses a queue.
        """
        mailbox_callback, slot = callback
        self._response_slots[slot].update_(
            result.select(*self._response_keys, strict=True)
        )
        self._ensure_mailbox().resolve(mailbox_callback, slot)

    def resolve_exception(
        self, callback: tuple[tuple[int, int], int], exc: BaseException
    ) -> None:
        """Propagate an exception; the client releases the slot on receipt."""
        mailbox_callback, _slot = callback
        self._ensure_mailbox().reject(mailbox_callback, exc)
