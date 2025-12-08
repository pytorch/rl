from __future__ import annotations

import socket
import threading
import time
from datetime import timedelta
from typing import Any

import torch
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger

from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightStrategy,
    WeightSyncScheme,
)


class DistributedWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for torch.distributed.

    This scheme uses torch.distributed primitives (send/recv) to synchronize
    weights across distributed workers. Each worker gets its own transport,
    following the same pattern as multiprocess collectors.

    The scheme can create its own TCPStore for coordination if one is not provided.
    Use `get_store_info()` after `init_on_sender()` to get connection details for workers.

    Args:
        backend (str): The distributed backend ("gloo", "nccl", etc.)
        sync (bool): If True, weight updates are synchronous (blocking receive).
            If False, a background thread monitors the store and applies weight
            updates automatically. Defaults to True.
        store (torch.distributed.Store, optional): Pre-existing store to use.
            If None, a TCPStore is created on init_on_sender.
        store_port (int, optional): Port for the created TCPStore.
            If None, a free port is automatically selected.
        timeout (float): Timeout in seconds for TCPStore operations.
            Defaults to 3600.0 (1 hour).
    """

    def __init__(
        self,
        backend: str = "gloo",
        sync: bool = True,
        store: torch.distributed.Store | None = None,
        store_port: int | None = None,
        timeout: float = 3600.0,
    ):
        super().__init__()
        self.backend = backend
        self.sync = sync
        self._provided_store = store
        self._store_port = store_port
        self._timeout = timeout
        self._store = None
        self._store_info = None
        self._num_workers = None

        # Background thread state (for async mode on receiver)
        self._background_thread = None
        self._stop_event = None

    def _init_on_sender_impl(
        self,
        *,
        model_id: str,
        context: Any = None,
        num_workers: int,
        model: Any = None,
        weights: Any = None,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self._num_workers = num_workers

        # Attach context so we can resolve the model and prepare
        # weights on demand via scheme.prepare_weights().
        weights_buffer = None
        if context is not None:
            self.context = context
        if weights is not None:
            self.weights = weights
            weights_buffer = weights
        if model is not None:
            self.model = model
        else:
            # resolve model
            try:
                model = self.model
            except (AttributeError, ValueError):
                pass

        if weights_buffer is None and model is not None:
            weights_buffer = self._get_weights_buffer_from_model(model)

        # Create TCPStore if not provided
        if self._provided_store is not None:
            self._store = self._provided_store
        elif hasattr(context, "_store") and context._store is not None:
            # Use context's store if available
            self._store = context._store
        else:
            # Create our own TCPStore as master
            self._store = self._make_store(is_master=True, num_workers=num_workers)

        for i in range(num_workers):
            rank = i + 1  # Workers are 1-indexed in distributed
            transport = self.create_transport(
                store=self._store,
                rank=rank,
                weights_buffer=weights_buffer,
                sync=self.sync,
            )
            self._register_worker_sender(worker_idx=i, transport=transport)

    def get_store_info(self) -> dict | None:
        """Return store connection info to pass to workers.

        Returns:
            Dictionary with 'host' and 'port' keys if store was created by this scheme,
            None if using a provided store.
        """
        return self._store_info

    def _make_store(
        self,
        is_master: bool,
        num_workers: int | None = None,
        store_info: dict | None = None,
    ) -> torch.distributed.TCPStore:
        """Create a TCPStore for weight synchronization.

        Args:
            is_master: If True, creates the store as master (server).
                If False, connects as client.
            num_workers: Number of workers (required for master).
            store_info: Dictionary with 'host' and 'port' keys (required for client).

        Returns:
            The created TCPStore.
        """
        if is_master:
            # Create as master (server)
            if num_workers is None:
                raise ValueError("num_workers is required when creating store as master")

            hostname = socket.gethostname()
            host = socket.gethostbyname(hostname)

            if self._store_port is None:
                # Find a free port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    self._store_port = s.getsockname()[1]

            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Creating TCPStore on {host}:{self._store_port}"
            )
            store = torch.distributed.TCPStore(
                host_name=host,
                port=self._store_port,
                world_size=num_workers + 1,  # workers + master
                is_master=True,
                timeout=timedelta(seconds=self._timeout),
            )
            self._store_info = {"host": host, "port": self._store_port}
        else:
            # Connect as client
            if store_info is None:
                raise ValueError("store_info is required when connecting as client")
            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Connecting to TCPStore at "
                f"{store_info['host']}:{store_info['port']}"
            )
            store = torch.distributed.TCPStore(
                host_name=store_info["host"],
                port=store_info["port"],
                is_master=False,
                timeout=timedelta(seconds=self._timeout),
            )
        return store

    def _init_on_receiver_impl(
        self,
        *,
        model_id: str,
        context: Any = None,
        store: torch.distributed.Store = None,
        store_info: dict | None = None,
        rank: int = None,
        **kwargs,
    ) -> None:
        """Initialize scheme on the worker (receiver) side.

        Expected kwargs (as provided by collectors):
            - model_id: str              # e.g. "policy"
            - context: Any               # collector / inner collector
            - store: TCPStore | None     # distributed TCP store
            - store_info: dict | None    # {"host": ..., "port": ...} to create store
            - rank: int | None           # worker rank (1-indexed)
        """
        if context is None:
            raise ValueError(
                "DistributedWeightSyncScheme.init_on_receiver requires a 'context' "
                "providing access to the model to be synchronized."
            )

        # Store model_id and context on scheme
        self.model_id = model_id
        self.context = context

        # Get or create store
        # Priority: provided store > provided store_info > self._store_info (from serialization)
        if store is not None:
            self._store = store
        elif store_info is not None or self._store_info is not None:
            # Connect to master's TCPStore as client
            info = store_info if store_info is not None else self._store_info
            self._store = self._make_store(is_master=False, store_info=info)
        else:
            raise ValueError(
                "DistributedWeightSyncScheme.init_on_receiver requires either 'store', "
                "'store_info', or the scheme must have been initialized on sender first."
            )

        if (model := getattr(self, "model", None)) is not None:
            self.model = model
            weights_buffer = self._get_weights_buffer_from_model(model)
        else:
            raise RuntimeError("Couldn't find weights")
        self._receiver_transport = self.create_transport(
            store=self._store, rank=rank, weights_buffer=weights_buffer, sync=self.sync
        )

        # Store worker_idx for synchronize_weights
        self._worker_idx = rank

        # For async mode, start background thread that monitors store for weight updates
        if not self.sync:
            self._start_background_receiver()

    def _start_background_receiver(self):
        """Start daemon thread that monitors store for weight updates."""
        self._stop_event = threading.Event()
        self._background_thread = threading.Thread(
            target=self._background_receive_loop,
            daemon=True,
            name=f"WeightReceiver-{self._worker_idx}",
        )
        self._background_thread.start()
        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Started background receiver thread for worker {self._worker_idx}"
        )

    def _background_receive_loop(self):
        """Monitor store for 'update_weights' instruction, receive and apply."""
        key = f"NODE_{self._worker_idx}_in"

        while not self._stop_event.is_set():
            try:
                # Check if there's an update instruction
                # TCPStore.get() blocks, so we use a polling approach with check()
                try:
                    # Try to get the key - this may block briefly
                    instruction = self._store.get(key)
                except RuntimeError:
                    # Key doesn't exist yet, continue polling
                    time.sleep(0.01)
                    continue

                if instruction == b"update_weights":
                    torchrl_logger.debug(
                        f"DistributedWeightSyncScheme: Worker {self._worker_idx} "
                        "received update_weights instruction"
                    )
                    self._store.delete_key(key)

                    # Receive weights via torch.distributed
                    weights = self._receiver_transport.receive_weights(
                        model=self.model,
                        strategy=self._strategy,
                    )

                    # Send acknowledgment
                    self._store.set(f"NODE_{self._worker_idx}_out", b"updated")

                    torchrl_logger.debug(
                        f"DistributedWeightSyncScheme: Worker {self._worker_idx} "
                        "received and applied weights"
                    )

            except Exception as e:
                if not self._stop_event.is_set():
                    torchrl_logger.warning(
                        f"DistributedWeightSyncScheme: Background receiver error: {e}"
                    )

            time.sleep(0.001)  # Small sleep to avoid busy-waiting

    def _setup_connection_and_weights_on_sender_impl(
        self, *, worker_idx: int | None = None, weights: Any | None = None
    ) -> None:
        """Send initial weights to all workers during connect().

        If the sender has a stateful model (weights available), send them
        to all workers so they start with the correct weights.

        Note: This uses direct torch.distributed send/recv without TCPStore
        signaling to avoid interfering with the main collection loop.
        """
        # Check if we have weights to send
        if weights is None and getattr(self, "model", None) is None:
            torchrl_logger.debug(
                "DistributedWeightSyncScheme: No model on sender, skipping initial weight sync"
            )
            self._store.set("STATELESS_MODEL", b"1")
            return

        self._store.set("STATELESS_MODEL", b"0")
        # Prepare weights from model
        weights = self._get_weights_buffer_from_model(self.model)
        if weights is None or weights.is_empty():
            torchrl_logger.debug(
                "DistributedWeightSyncScheme: Empty weights, skipping initial weight sync"
            )
            return

        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Sending initial weights to {self._num_workers} workers"
        )

        # Send to all workers using direct torch.distributed (no TCPStore signaling)
        for i, transport in enumerate(self._iterate_transports()):
            if worker_idx is not None and i != worker_idx:
                continue
            transport.send_initial_weights(weights)

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Receive initial weights from sender during connect().

        The receiver always has a model that needs weights, so we block
        waiting for the initial weights from the sender.
        """
        if self._receiver_transport is None:
            return
        stateless_model = self.receiver_transport._store.get("STATELESS_MODEL")
        if stateless_model not in (b"0", b"1"):
            raise RuntimeError(f"Invalid STATELESS_MODEL value: {stateless_model}")
        if stateless_model == b"1":
            torchrl_logger.debug(
                "DistributedWeightSyncScheme: Skipping initial weight sync on receiver because of stateless model."
            )
            return

        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = self._worker_idx

        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Worker {worker_idx} waiting for initial weights"
        )

        # Receive initial weights (blocking, no TCPStore coordination)
        weights = self._receiver_transport.receive_initial_weights()
        if weights is not None and self.model is not None:
            self._strategy.apply_weights(self.model, weights, inplace=False)
            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Worker {worker_idx} received and applied initial weights"
            )

    def shutdown(self) -> None:
        """Stop background receiver thread and clean up."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(timeout=5.0)
            if self._background_thread.is_alive():
                torchrl_logger.warning(
                    "DistributedWeightSyncScheme: Background thread did not stop gracefully"
                )
        self._background_thread = None
        self._stop_event = None

    def create_transport(self, **kwargs) -> TransportBackend:
        """Create distributed transport for a specific worker."""
        return DistributedTransport(**kwargs)


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
        torchrl_logger.debug(
            f"RANK 0 -- Setting weight sync instructions to store for rank {self._rank}"
        )
        self._store.set(f"NODE_{self._rank}_in", b"update_weights")

        # Send weights via torch.distributed
        torchrl_logger.debug(
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

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> Any | None:
        r"""Receive weights via torch.distributed and apply them to the model.

        The surrounding collector loop is responsible for checking the TCPStore
        for the \"update_weights\" instruction. When this method is called we
        assume that a weight update has been requested and the sender has
        already performed the corresponding ``send()``.

        Args:
            timeout: Maximum time to wait for weights (seconds). If None,
                blocks until weights are received.
            weights: Pre-allocated weight buffer to receive into.
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

        Returns:
            The received weights, or None if timeout expires.
        """
        if self._store is None or self._rank is None:
            return None

        # Use provided weights buffer or fallback to stored one
        weights_buffer = weights if weights is not None else self._weights_buffer

        # Receive weights via torch.distributed into the buffer
        if self._sync or timeout is None:
            # Blocking receive - no timeout support
            if self._sync:
                weights_buffer.recv(src=0)
            else:
                weights_buffer.irecv(src=0)
        else:
            # Non-blocking receive with timeout support
            futures = weights_buffer.irecv(src=0, return_premature=True)
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

        # Apply weights if model and strategy provided
        if model is not None and strategy is not None:
            strategy.apply_weights(model, weights_buffer)

        return weights_buffer

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

    def send_initial_weights(self, weights: Any) -> None:
        """Send initial weights during connect() without TCPStore signaling.

        This is used for the initial weight sync during connect() to avoid
        interfering with the main collection loop's TCPStore-based coordination.
        """
        if self._rank is None:
            return

        torchrl_logger.debug(
            f"DistributedTransport: Sending initial weights to rank {self._rank}"
        )
        if self._sync:
            weights.send(self._rank)
        else:
            weights.isend(self._rank)

    def receive_initial_weights(self) -> Any:
        """Receive initial weights during connect() without TCPStore signaling.

        This is used for the initial weight sync during connect() to avoid
        interfering with the main collection loop's TCPStore-based coordination.

        Returns:
            The received weights TensorDict.
        """
        torchrl_logger.debug(
            "DistributedTransport: Receiving initial weights from rank 0"
        )
        if self._sync:
            self._weights_buffer.recv(src=0)
        else:
            self._weights_buffer.irecv(src=0)
        return self._weights_buffer

    def setup_connection_and_weights_on_sender(self) -> None:
        """No-op for DistributedTransport - handled by scheme."""

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> Any:
        """No-op for DistributedTransport - handled by scheme."""
        return None
