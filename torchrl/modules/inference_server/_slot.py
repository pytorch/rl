# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import itertools
import threading

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._transport import InferenceTransport


class _SlotClient:
    """Actor-side handle for a :class:`SlotTransport` slot.

    Each client owns a single slot (identified by ``slot_id``).  Calling the
    client writes the observation into the slot and blocks until the server
    writes the action back.

    Args:
        transport: the parent :class:`SlotTransport`.
        slot_id: the slot this client owns.
    """

    def __init__(self, transport: SlotTransport, slot_id: int):
        self._transport = transport
        self._slot_id = slot_id

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Submit an observation and block until the action is ready."""
        self._transport._slot_submit(self._slot_id, td)
        return self._transport._slot_recv(self._slot_id)


class SlotTransport(InferenceTransport):
    """Lock-free, in-process transport using per-env slots.

    Each actor thread owns a dedicated *slot*.  Submitting an observation
    writes to the slot without any lock (each slot is accessed by exactly
    one writer thread).  The server sweeps slots to find ready ones, collects
    observations, runs the model, and writes actions back via per-slot events.

    This eliminates:

    * The shared ``threading.Lock`` that ``ThreadingTransport`` uses for
      every ``submit()`` and ``drain()``.
    * ``concurrent.futures.Future`` allocations (one per inference request).

    The trade-off is that the number of slots is fixed at construction time
    (equal to the number of environments).

    Args:
        num_slots (int): number of slots (one per environment / actor thread).

    Keyword Args:
        preallocate (bool, optional): if ``True``, a contiguous observation
            buffer of shape ``[num_slots, ...]`` is allocated on the first
            submit.  Subsequent submits copy into the buffer in-place
            (``update_``).  Defaults to ``False`` because the extra copy
            into the buffer is not currently compensated by the batching
            path (``lazy_stack`` still calls ``torch.stack``).

    .. note::
        This transport is only suitable for in-process threading scenarios
        (the default for :class:`~torchrl.collectors.AsyncBatchedCollector`
        with ``policy_backend="threading"``).
    """

    def __init__(self, num_slots: int, *, preallocate: bool = False):
        self._num_slots = num_slots
        self._preallocate = preallocate
        self._slot_counter = itertools.count()

        # Per-slot observation storage (written by env thread, read by server)
        self._obs: list[TensorDictBase | None] = [None] * num_slots

        # Per-slot readiness flag (True = observation ready for server)
        # Under CPython's GIL, bool assignment is atomic.
        self._obs_ready: list[bool] = [False] * num_slots

        # Per-slot action storage (written by server, read by env thread)
        self._actions: list[TensorDictBase | BaseException | None] = [None] * num_slots

        # Per-slot events: server sets after writing the action
        self._action_events: list[threading.Event] = [
            threading.Event() for _ in range(num_slots)
        ]

        # Condition variable: env threads notify, server waits.
        # Using a Condition instead of a bare Event avoids the race where
        # clear() in wait_for_work drops a signal set between wait() and
        # clear().
        self._work_cond = threading.Condition(threading.Lock())

        # Pre-allocated observation buffer (lazily initialised)
        self._obs_buffer: TensorDictBase | None = None

    # -- actor (env-thread) API -----------------------------------------------

    def _slot_submit(self, slot_id: int, td: TensorDictBase) -> None:
        """Write observation into the slot (no lock required)."""
        if self._obs_buffer is not None:
            # Copy into pre-allocated buffer (no new allocation)
            self._obs_buffer[slot_id].update_(td)
        else:
            self._obs[slot_id] = td
        self._obs_ready[slot_id] = True
        with self._work_cond:
            self._work_cond.notify()

    def _slot_recv(self, slot_id: int) -> TensorDictBase:
        """Block until the server writes an action into the slot."""
        self._action_events[slot_id].wait()
        self._action_events[slot_id].clear()
        result = self._actions[slot_id]
        self._actions[slot_id] = None
        if isinstance(result, BaseException):
            raise result
        return result

    # -- InferenceTransport interface -----------------------------------------

    def client(self) -> _SlotClient:
        """Create a slot-bound client for one actor thread."""
        slot_id = next(self._slot_counter)
        if slot_id >= self._num_slots:
            raise RuntimeError(
                f"SlotTransport has {self._num_slots} slots but "
                f"client() was called {slot_id + 1} times. "
                "Create a SlotTransport with more slots."
            )
        return _SlotClient(self, slot_id)

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to get a slot-bound callable."""
        raise NotImplementedError(
            "SlotTransport does not support submit(). "
            "Use client() to obtain a slot-bound callable."
        )

    def wait_for_work(self, timeout: float) -> None:
        """Block until at least one slot has a ready observation."""
        with self._work_cond:
            # Check if any slot is already ready before waiting
            if any(self._obs_ready):
                return
            self._work_cond.wait(timeout=timeout)

    def drain(self, max_items: int) -> tuple[list[TensorDictBase], list[int]]:
        """Sweep slots and return (observations, slot_ids) for ready ones."""
        # Lazily initialise the pre-allocated buffer on the first drain
        # that finds ready observations.
        if self._preallocate and self._obs_buffer is None:
            for i in range(self._num_slots):
                if self._obs_ready[i] and self._obs[i] is not None:
                    self._obs_buffer = (
                        self._obs[i]
                        .unsqueeze(0)
                        .expand(self._num_slots)
                        .clone()
                        .contiguous()
                    )
                    break

        items: list[TensorDictBase] = []
        slot_ids: list[int] = []
        for i in range(self._num_slots):
            if self._obs_ready[i]:
                self._obs_ready[i] = False
                if self._obs_buffer is not None:
                    # Flush first-time observations that arrived before the
                    # buffer existed into the buffer.
                    if self._obs[i] is not None:
                        self._obs_buffer[i].update_(self._obs[i])
                        self._obs[i] = None
                    items.append(self._obs_buffer[i])
                else:
                    items.append(self._obs[i])
                    self._obs[i] = None
                slot_ids.append(i)
                if len(slot_ids) >= max_items:
                    break
        return items, slot_ids

    def resolve(self, callback: int, result: TensorDictBase) -> None:
        """Write the action into the slot and wake the waiting env thread."""
        self._actions[callback] = result
        self._action_events[callback].set()

    def resolve_exception(self, callback: int, exc: BaseException) -> None:
        """Propagate an exception to the waiting env thread."""
        self._actions[callback] = exc
        self._action_events[callback].set()
