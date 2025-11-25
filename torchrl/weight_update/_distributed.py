from __future__ import annotations

import weakref

from typing import Any

import torch
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class DistributedWeightReceiver(WeightReceiver):
    """Weight receiver for torch.distributed systems.

    Receives weight updates from the main process via torch.distributed send/recv
    primitives and TCPStore signaling. This is typically instantiated and managed
    by :class:`DistributedWeightSyncScheme`.
    """

    _transport: DistributedTransport | None


class DistributedWeightSender(WeightSender):
    """Weight sender for torch.distributed systems.

    Sends weight updates to distributed workers via torch.distributed send/recv
    primitives and TCPStore signaling. This is typically instantiated and managed
    by :class:`DistributedWeightSyncScheme`.
    """

    _transport: DistributedTransport | None


class DistributedWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for torch.distributed.

    This scheme uses torch.distributed primitives (send/recv) to synchronize
    weights across distributed workers. Each worker gets its own transport,
    following the same pattern as multiprocess collectors.

    Args:
        backend (str): The distributed backend ("gloo", "nccl", etc.)
        sync (bool): Whether to use synchronous weight updates
    """

    _receiver_cls = DistributedWeightReceiver
    _sender_cls = DistributedWeightSender

    def __init__(self, backend: str = "gloo", sync: bool = True):
        super().__init__()
        self.backend = backend
        self.sync = sync

    def _init_on_sender_impl(
        self,
        *args,
        **kwargs,
    ) -> None:
        num_workers = kwargs.pop("num_workers")
        context = kwargs.pop("context")
        model_id = kwargs.pop("model_id")

        # Create and configure sender for this model
        sender = self.create_sender()
        sender._model_id = model_id

        # Attach context so the sender can resolve the model and prepare
        # weights on demand via scheme.prepare_weights().
        if context is not None:
            sender._set_context(context, model_id)

        # Store reference to source model for automatic extraction
        try:
            sender._source_model = _resolve_model(context, model_id)
        except (AttributeError, IndexError):
            pass

        # Create transports for each remote collector
        weights_buffer = self._get_weights_buffer_from_model(sender._source_model)
        for i in range(num_workers):
            rank = i + 1  # Workers are 1-indexed in distributed
            transport = self.create_transport(
                store=context._store, rank=rank, weights_buffer=weights_buffer
            )
            sender._transports[i] = transport

        # Expose sender through the base API
        self._sender = sender

    def _init_on_receiver_impl(self, *args, **kwargs) -> None:
        """Initialize scheme on the worker (receiver) side.

        Expected kwargs (as provided by collectors):
            - model_id: str              # e.g. "policy"
            - context: Any               # collector / inner collector
            - store: TCPStore | None     # distributed TCP store
            - rank: int | None           # worker rank (1-indexed)
        """
        context = kwargs.pop("context", None)
        model_id = kwargs.pop("model_id")
        store = kwargs.pop("store", None)
        rank = kwargs.pop("rank", None)

        if context is None:
            raise ValueError(
                "DistributedWeightSyncScheme.init_on_receiver requires a 'context' "
                "providing access to the model to be synchronized."
            )

        # Create receiver instance
        receiver = self._receiver_cls(self)
        receiver._model_id = model_id

        # Attach context so we can resolve string model refs like "policy"
        receiver._context_ref = weakref.ref(context)

        # Resolve the target model on this worker
        model = None
        # Prefer a collector-specific get_model if available, but fall back
        # gracefully to attribute resolution when no mapping exists.
        if hasattr(context, "get_model"):
            try:
                model = context.get_model(model_id)
            except (ValueError, AttributeError):
                model = None
        if model is None:
            model = _resolve_model(context, model_id)
        receiver._register_model(model)

        weights_buffer = self._get_weights_buffer_from_model(model)
        receiver._transport = self.create_transport(
            store=store, rank=rank, weights_buffer=weights_buffer
        )

        # Store receiver on scheme so get_receiver() works as expected
        self._receiver = receiver

    def create_transport(self, **kwargs) -> TransportBackend:
        """Create distributed transport for a specific worker."""
        if self._initialized_on_receiver:
            return DistributedTransport(**kwargs)
        elif self._initialized_on_sender:
            return DistributedTransport(**kwargs)
        else:
            raise RuntimeError(
                "DistributedWeightSyncScheme.create_transport must be called after initialization has been marked."
            )


class DistributedTransport:
    """torch.distributed transport for communicating with a single distributed worker.

    This transport handles weight updates for ONE specific distributed worker via
    torch.distributed send/recv. Multiple transports are created for multiple workers,
    following the same pattern as multiprocess collectors.
    """

    def __init__(
        self,
        *,
        weights_buffer: TensorDictBase,
        store: torch.distributed.Store = None,
        rank: int = None,
        sync: bool = True,
    ):
        """Initialize the DistributedTransport.

        Args:
            weights_buffer (TensorDictBase): a tensor buffer of weights.
            store (torch.distributed.Store): A (TCP)Store for communication.
            rank (int): Worker rank (1-indexed).
            sync (bool): Whether to use synchronous weight updates.
        """
        self._store = store
        self._rank = rank
        self._sync = sync
        self._weights_buffer = weights_buffer

    def send_weights(self, weights: Any) -> None:
        """Send weights to the distributed worker."""
        if self._store is None or self._rank is None:
            return

        # Instruct worker to expect weight update
        torchrl_logger.debug("RANK 0 -- Setting weight sync instructions to store")
        self._store.set(f"NODE_{self._rank}_in", b"update_weights")

        # Send weights via torch.distributed
        torchrl_logger.debug(f"RANK 0 -- Send {weights=} to rank {self._rank}")
        if self._sync:
            weights.send(self._rank)
        else:
            weights.isend(self._rank)

        # Wait for acknowledgment
        torchrl_logger.debug("RANK 0 -- Receiving acknowledgement from store")
        status = self._store.get(f"NODE_{self._rank}_out")
        if status != b"updated":
            raise RuntimeError(f"Expected 'updated' but got status {status}.")
        self._store.delete_key(f"NODE_{self._rank}_out")

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to distributed worker without waiting for acknowledgment.

        Use wait_ack() to wait for acknowledgment after sending to all workers.
        """
        if self._store is None or self._rank is None:
            return

        # Instruct worker to expect weight update
        torchrl_logger.info(
            f"RANK 0 -- Setting weight sync instructions to store for rank {self._rank}"
        )
        self._store.set(f"NODE_{self._rank}_in", b"update_weights")

        # Send weights via torch.distributed
        torchrl_logger.info(
            f"RANK 0 -- Send {weights=} to rank {self._rank} with sync={self._sync}"
        )
        if self._sync:
            weights.send(self._rank)
        else:
            weights.isend(self._rank)
        torchrl_logger.debug(f"RANK 0 -- Weights successfully sent to {self._rank}")

    def wait_ack(self) -> None:
        """Wait for acknowledgment from distributed worker."""
        if self._store is None or self._rank is None:
            return

        status = self._store.get(f"NODE_{self._rank}_out")
        if status != b"updated":
            raise RuntimeError(f"Expected 'updated' but got status {status}.")
        self._store.delete_key(f"NODE_{self._rank}_out")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        r"""Receive weights via torch.distributed.

        The surrounding collector loop is responsible for checking the TCPStore
        for the \"update_weights\" instruction. When this method is called we
        assume that a weight update has been requested and the sender has
        already performed the corresponding ``send()``.

        Args:
            timeout: Unused for now (kept for TransportBackend compatibility).

        Returns:
            Tuple of (model_id, weights) where model_id is currently always
            \"policy\".
        """
        if self._store is None or self._rank is None:
            return None

        # Receive weights via torch.distributed into the buffer
        if self._sync:
            self._weights_buffer.recv(src=0)
        else:
            # irecv() blocks until weights have been received
            self._weights_buffer.irecv(src=0)

        return ("policy", self._weights_buffer)

    def send_ack(self, message: str = "updated") -> None:
        """Send acknowledgment back to sender via TCPStore.

        Args:
            message: Acknowledgment message to send (default: "updated")
        """
        if self._store is None or self._rank is None:
            return

        self._store.set(f"NODE_{self._rank}_out", message.encode())

    def check_connection(self) -> bool:
        """Check if torch.distributed is initialized."""
        return torch.distributed.is_initialized()

    def synchronize_weights_on_sender(self) -> None:
        """No-op for DistributedTransport - weights are sent via send_weights()."""

    def synchronize_weights_on_receiver(self, worker_idx: int) -> Any:
        """No-op for DistributedTransport - weights are received via receive_weights()."""
        return None
