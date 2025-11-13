from __future__ import annotations

from typing import Any

from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class RPCWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for torch.distributed.rpc.

    This scheme uses RPC calls to synchronize weights across distributed
    workers. Each remote collector gets its own transport, following the
    same pattern as multiprocess collectors.
    """

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create RPC-based transport for a specific remote collector.

        Args:
            pipe_or_context: A tuple of (collector_info, collector_rref, collector_class)
                for the remote collector.

        Returns:
            RPCTransport configured for this specific remote collector.
        """
        if isinstance(pipe_or_context, tuple) and len(pipe_or_context) == 3:
            collector_info, collector_rref, collector_class = pipe_or_context
            return RPCTransport(
                collector_info=collector_info,
                collector_rref=collector_rref,
                collector_class=collector_class,
            )
        # If just passed the info directly
        return RPCTransport(collector_info=pipe_or_context)


class RPCTransport:
    """RPC transport for communicating with a single RPC remote collector.

    This transport handles weight updates for ONE specific remote collector via
    torch.distributed.rpc. Multiple transports are created for multiple collectors,
    following the same pattern as multiprocess collectors.
    """

    def __init__(self, collector_info=None, collector_rref=None, collector_class=None):
        self._collector_info = collector_info
        self._collector_rref = collector_rref
        self._collector_class = collector_class

    def send_weights(self, weights: Any) -> None:
        """Send weights to the remote collector via RPC."""
        if self._collector_info is None or self._collector_rref is None:
            return

        from torch.distributed import rpc

        # Send weights to the remote collector and wait for completion
        rpc.rpc_sync(
            self._collector_info,
            self._collector_class.update_policy_weights_,
            args=(self._collector_rref, weights),
        )

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to remote collector without waiting for completion.

        Use wait_ack() to wait for completion after sending to all workers.
        """
        if self._collector_info is None or self._collector_rref is None:
            return

        from torch.distributed import rpc

        # Send weights asynchronously
        self._pending_future = rpc.rpc_async(
            self._collector_info,
            self._collector_class.update_policy_weights_,
            args=(self._collector_rref, weights),
        )

    def wait_ack(self) -> None:
        """Wait for the RPC call to complete."""
        if hasattr(self, "_pending_future"):
            self._pending_future.wait()
            del self._pending_future

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """RPC workers typically don't receive weights through this transport."""
        return None

    def check_connection(self) -> bool:
        """Check if RPC is initialized."""
        from torch.distributed import rpc

        return rpc.is_initialized() if hasattr(rpc, "is_initialized") else True

    def synchronize_weights_on_sender(self) -> None:
        """No-op for RPCTransport - weights are sent via send_weights()."""

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """No-op for RPCTransport - weights are received via RPC calls."""
        return None


class RPCWeightReceiver(WeightReceiver):
    """Weight receiver for RPC-based distributed systems.

    Receives weight updates from the main process via torch.distributed.rpc.
    This is typically instantiated and managed by :class:`RPCWeightSyncScheme`.
    """


class RPCWeightSender(WeightSender):
    """Weight sender for RPC-based distributed systems.

    Sends weight updates to remote collectors via torch.distributed.rpc calls.
    This is typically instantiated and managed by :class:`RPCWeightSyncScheme`.
    """
