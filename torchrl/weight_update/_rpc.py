from __future__ import annotations

import time
import weakref
from typing import Any

from torchrl._utils import logger as torchrl_logger

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightStrategy,
    WeightSyncScheme,
)


class RPCWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for torch.distributed.rpc.

    This scheme uses RPC calls to synchronize weights across distributed
    workers. Each remote collector gets its own transport, following the
    same pattern as multiprocess collectors.
    """

    def _init_on_sender_impl(
        self,
        *,
        model_id: str,
        context: Any = None,
        num_workers: int,
    ) -> None:
        # Store model_id and context on scheme
        self.model_id = model_id
        if context is not None:
            self.context = context
        else:
            raise RuntimeError(f"Expected a context for {type(self).__name__}.")
        collector_infos = getattr(self.context, "collector_infos", None)
        collector_rrefs = getattr(self.context, "collector_rrefs", None)
        collector_class = getattr(self.context, "collector_class", None)
        if (
            collector_infos is None
            or collector_rrefs is None
            or collector_class is None
        ):
            raise RuntimeError(
                "RPCWeightSyncScheme requires a context with the following attributes: "
                "(context.collector_infos, context.collector_rrefs, context.collector_class)"
            )

        # Create transports for each remote collector
        # worker_rank is i+1 because rank 0 is the main/trainer process
        for i in range(num_workers):
            worker_rank = i + 1
            transport = self.create_transport(
                collector_info=collector_infos[i],
                collector_rref=collector_rrefs[i],
                collector_class=collector_class,
                worker_rank=worker_rank,
            )
            self._register_worker_sender(worker_idx=i, transport=transport)

    def _init_on_receiver_impl(
        self, *, model_id: str, context: Any = None, worker_idx: int | None = None
    ) -> None:
        """Initialize scheme on the worker (receiver) side.

        Expected kwargs (as provided by collectors):
            - model_id: str              # e.g. "policy"
            - context: Any               # collector / inner collector
            - worker_idx: int | None     # worker index (optional)
        """
        if context is None:
            raise ValueError(
                "RPCWeightSyncScheme.init_on_receiver requires a 'context' "
                "providing access to the model to be synchronized."
            )

        # Store model_id and context on scheme
        self.model_id = model_id
        self.worker_idx = worker_idx
        self.context = context
        # Access weights to set up missing elements
        self.weights  # noqa

        self._receiver_transport = RPCTransport(worker_rank=worker_idx)

    @property
    def model(self) -> Any | None:
        if self._model_ref is not None:
            return self._model_ref()
        if self._model_id is not None:
            model = _resolve_model(self.context, self._model_id)
            if model is None:
                if self._model_id == "policy":
                    torchrl_logger.debug(
                        f"Creating policy from factory and setting in collector {type(self.context)}"
                    )
                    model = self.context.policy_factory[0]()
                    self.context.policy = model
                    torchrl_logger.debug(f"{self.context.policy=}")
                else:
                    raise AttributeError(
                        f"Model {self._model_id} was `None` in context {self.context}"
                    )
            self._model_ref = weakref.ref(model)
            return model

    @model.setter
    def model(self, value: Any):
        if value is None:
            return
        self._model_ref = weakref.ref(value)

    def create_transport(
        self,
        *,
        collector_info=None,
        collector_rref=None,
        collector_class=None,
        worker_rank=None,
        **kwargs,
    ) -> TransportBackend:
        """Create RPC-based transport for a specific remote collector.

        Args:
            collector_info: RPC worker info for the remote collector.
            collector_rref: RPC remote reference to the collector.
            collector_class: Class of the remote collector.
            worker_rank: The torch.distributed rank of the remote worker.
            **kwargs: Additional transport configuration.

        Returns:
            RPCTransport configured for this specific remote collector.
        """
        return RPCTransport(
            collector_info=collector_info,
            collector_rref=collector_rref,
            collector_class=collector_class,
            worker_rank=worker_rank,
        )


class RPCTransport:
    """RPC transport for communicating with a single RPC remote collector.

    This transport handles weight updates for ONE specific remote collector via
    torch.distributed primitives (send/recv) with RPC used for signaling.
    Multiple transports are created for multiple collectors, following the same
    pattern as the DistributedDataCollector.
    """

    def __init__(
        self,
        collector_info=None,
        collector_rref=None,
        collector_class=None,
        worker_rank=None,
    ):
        self._collector_info = collector_info
        self._collector_rref = collector_rref
        self._collector_class = collector_class
        self._worker_rank = worker_rank  # The torch.distributed rank of this worker
        self._pending_future = None
        self._pending_send = None

    def send_weights(self, weights: Any) -> None:
        """Send weights to the remote collector using torch.distributed.

        Uses torch.distributed.send() for the actual weight transfer and RPC
        for signaling the remote collector to receive.

        Order is critical to avoid deadlock:
        1. Signal receiver via RPC to start recv() (non-blocking)
        2. Send weights via torch.distributed (blocking until recv completes)
        """
        if self._collector_info is None or self._collector_rref is None:
            return
        if self._worker_rank is None:
            raise RuntimeError("worker_rank must be set for RPC transport")

        # Step 1: Signal the remote collector via RPC to start receiving (async)
        # Use rref.rpc_async() to properly call the instance method on the remote object
        future = self._collector_rref.rpc_async()._receive_weights_scheme()

        # Step 2: Send weights via torch.distributed (blocks until receiver calls recv())
        weights.send(self._worker_rank)

        # Step 3: Wait for RPC to complete (receiver has applied weights)
        future.wait()

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to remote collector asynchronously.

        Uses torch.distributed.isend() for the actual weight transfer and RPC
        for signaling. Use wait_ack() to wait for completion.

        Order is critical to avoid deadlock:
        1. Signal receiver via RPC to start recv() (non-blocking)
        2. Send weights via torch.distributed.isend() (non-blocking)
        3. wait_ack() waits for both to complete
        """
        if self._collector_info is None or self._collector_rref is None:
            return
        if self._worker_rank is None:
            raise RuntimeError("worker_rank must be set for RPC transport")

        # Step 1: Signal the remote collector via RPC to start receiving (async)
        # Use rref.rpc_async() to properly call the instance method on the remote object
        self._pending_future = (
            self._collector_rref.rpc_async()._receive_weights_scheme()
        )

        # Step 2: Send weights asynchronously via torch.distributed
        # Store the Work handle for wait_ack()
        weights.isend(self._worker_rank)

    def wait_ack(self) -> None:
        """Wait for both the RPC call and the distributed send to complete."""
        # Wait for the RPC call to complete
        if hasattr(self, "_pending_future") and self._pending_future is not None:
            self._pending_future.wait()
            del self._pending_future

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> tuple[str, Any] | None:
        """Receive weights from sender using torch.distributed.

        Args:
            timeout: Maximum time to wait for weights (seconds). If None,
                blocks until weights are received.
            weights: Pre-allocated weight buffer to receive into.
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

        Returns:
            Tuple of (model_id, weights) where model_id is "policy", or None
            if timeout expires before weights are received.
        """
        if weights is None:
            return None

        if timeout is None:
            # Blocking receive
            weights.recv(0)
        else:
            # Non-blocking receive with timeout support
            futures = weights.irecv(src=0, return_premature=True)
            if futures:
                start_time = time.monotonic()
                while True:
                    # Check if all futures are complete
                    all_complete = all(f.is_completed() for f in futures)
                    if all_complete:
                        break
                    # Check timeout
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        # Timeout expired before receiving all weights
                        return None
                    # Small sleep to avoid busy-waiting
                    time.sleep(0.001)

        # Apply the received weights to the model
        if model is not None and strategy is not None:
            strategy.apply_weights(model, weights)

        return ("policy", weights)

    def check_connection(self) -> bool:
        """Check if both RPC and torch.distributed are initialized."""
        import torch.distributed
        from torch.distributed import rpc

        rpc_initialized = (
            rpc.is_initialized() if hasattr(rpc, "is_initialized") else True
        )
        dist_initialized = torch.distributed.is_initialized()
        return rpc_initialized and dist_initialized

    def setup_connection_and_weights_on_sender(self) -> None:
        """No-op for RPCTransport - weights are sent via send_weights()."""

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> Any:
        """No-op for RPCTransport - weights are received via receive()."""
        return None
