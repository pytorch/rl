from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class DistributedWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for torch.distributed.

    This scheme uses torch.distributed primitives (send/recv) to synchronize
    weights across distributed workers. Each worker gets its own transport,
    following the same pattern as multiprocess collectors.

    Args:
        backend (str): The distributed backend ("gloo", "nccl", etc.)
        sync (bool): Whether to use synchronous weight updates
    """

    def __init__(self, backend: str = "gloo", sync: bool = True):
        super().__init__()
        self.backend = backend
        self.sync = sync

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create distributed transport for a specific worker.

        Args:
            pipe_or_context: A tuple of (store, rank) for the worker.

        Returns:
            DistributedTransport configured for this specific worker.
        """
        if isinstance(pipe_or_context, tuple) and len(pipe_or_context) == 2:
            store, rank = pipe_or_context
            return DistributedTransport(store=store, rank=rank, sync=self.sync)
        # Fallback - shouldn't normally happen
        return DistributedTransport()


class DistributedTransport:
    """torch.distributed transport for communicating with a single distributed worker.

    This transport handles weight updates for ONE specific distributed worker via
    torch.distributed send/recv. Multiple transports are created for multiple workers,
    following the same pattern as multiprocess collectors.
    """

    def __init__(self, store=None, rank=None, sync=True):
        """Initialize the DistributedTransport.

        Args:
            store: TCPStore for communication.
            rank: Worker rank (1-indexed).
            sync: Whether to use synchronous weight updates.
        """
        self._store = store
        self._rank = rank
        self._sync = sync
        self._weights_buffer = None  # TensorDict buffer for receiving weights

    def send_weights(self, weights: Any) -> None:
        """Send weights to the distributed worker."""
        if self._store is None or self._rank is None:
            return

        # Instruct worker to expect weight update
        self._store.set(f"NODE_{self._rank}_in", b"update_weights")

        # Send weights via torch.distributed
        if self._sync:
            weights.send(self._rank)
        else:
            weights.isend(self._rank)

        # Wait for acknowledgment
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
        self._store.set(f"NODE_{self._rank}_in", b"update_weights")

        # Send weights via torch.distributed
        if self._sync:
            weights.send(self._rank)
        else:
            weights.isend(self._rank)

    def wait_ack(self) -> None:
        """Wait for acknowledgment from distributed worker."""
        if self._store is None or self._rank is None:
            return

        status = self._store.get(f"NODE_{self._rank}_out")
        if status != b"updated":
            raise RuntimeError(f"Expected 'updated' but got status {status}.")
        self._store.delete_key(f"NODE_{self._rank}_out")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Receive weights via torch.distributed, using TCPStore for signaling.

        This implements the RPC-like pattern:
        1. Check TCPStore for signal (non-blocking)
        2. If signal present, receive weights via torch.distributed
        3. Clean up signal and send acknowledgment

        Args:
            timeout: Timeout for receiving (currently not used for TCPStore check)

        Returns:
            Tuple of (model_id, weights) if weights were received, None otherwise.
        """
        if self._store is None or self._rank is None:
            return None

        try:
            # Non-blocking check of TCPStore "mailbox" for signal
            msg = self._store.get(f"NODE_{self._rank}_in")

            if msg == b"update_weights":
                # Initialize weights buffer on first use
                if self._weights_buffer is None:
                    self._weights_buffer = TensorDict()

                # Receive weights via torch.distributed
                # recv() and irecv() update the TensorDict in place
                if self._sync:
                    self._weights_buffer.recv(src=0)
                else:
                    # irecv() blocks until weights are received
                    self._weights_buffer.irecv(src=0)

                # Clean up the signal
                self._store.delete_key(f"NODE_{self._rank}_in")

                # Note: Acknowledgment is sent separately via send_ack() if transport supports it
                # This matches the pattern in WeightReceiver.receive()

                # Return model_id and received weights
                # For distributed transport, we use "policy" as default model_id
                return ("policy", self._weights_buffer)
            else:
                raise ValueError(f"Expected 'update_weights' but got {msg}")
        except KeyError:
            # No message in store - no weights available
            return None

        return None

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

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """No-op for DistributedTransport - weights are received via receive_weights()."""
        return None


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
