from __future__ import annotations

import socket
import time
import weakref
from datetime import timedelta
from typing import Any

import torch
from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger

from torchrl.weight_update.utils import _resolve_model

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
        timeout (float): Timeout in seconds for TCPStore operations.
            Defaults to 3600.0 (1 hour).
    """

    def __init__(
        self,
        backend: str = "gloo",
        sync: bool = True,
        timeout: float = 3600.0,
    ):
        super().__init__()
        self.backend = backend
        self.sync = sync
        self._timeout = timeout
        self._store = None
        self._store_info = None
        self._num_workers = None

    def __getstate__(self):
        """Custom serialization - exclude non-picklable objects."""
        state = super().__getstate__()
        # TCPStore cannot be pickled - remove it but keep _store_info
        state["_store"] = None

        # Thread and Event cannot be pickled
        state["_background_thread"] = None
        state["_stop_event"] = None

        # Transports contain references to store/groups - exclude them
        # The receiver will create its own transport in init_on_receiver
        state["_sender_transports"] = {}
        state["_receiver_transport"] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization."""
        super().__setstate__(state)

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
        if kwargs:
            raise RuntimeError(f"Unexpected kwargs: {kwargs.keys()}")
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

        # Get base tcp_port from context if available to avoid port conflicts.
        # The DistributedDataCollector uses tcp_port for init and tcp_port+1 for its store,
        # so we use tcp_port+2 for the weight sync scheme's store.
        base_tcp_port = (
            getattr(context, "tcp_port", None) if context is not None else None
        )
        self._store = self._make_store(
            is_master=True, num_workers=num_workers, base_tcp_port=base_tcp_port
        )

        for i in range(num_workers):
            rank = i + 1  # Workers are 1-indexed in distributed
            transport = self.create_transport(
                store=self._store,
                rank=rank,
                weights_buffer=weights_buffer,
                sync=self.sync,
            )
            self._register_worker_sender(worker_idx=i, transport=transport)

    def _make_store(
        self,
        is_master: bool,
        num_workers: int | None = None,
        store_info: dict | None = None,
        base_tcp_port: int | str | None = None,
    ) -> torch.distributed.TCPStore:
        """Create a TCPStore for weight synchronization.

        Args:
            is_master: If True, creates the store as master (server).
                If False, connects as client.
            num_workers: Number of workers (required for master).
            store_info: Dictionary with 'host' and 'port' keys (required for client).
            base_tcp_port: Base TCP port from the collector. If provided, the store
                will use base_tcp_port + 2 to avoid conflicts with the collector's
                stores (which use base_tcp_port and base_tcp_port + 1).

        Returns:
            The created TCPStore.
        """
        if is_master:
            # Create as master (server)
            if num_workers is None:
                raise ValueError(
                    "num_workers is required when creating store as master"
                )

            hostname = socket.gethostname()
            host = socket.gethostbyname(hostname)

            # Use base_tcp_port + 2 if available (to avoid conflicts with collector's
            # tcp_port and tcp_port + 1), otherwise find a free port dynamically.
            if base_tcp_port is not None:
                self._store_port = int(base_tcp_port) + 2
            else:
                # Find a free port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    socketname = s.getsockname()
                    self._store_port = socketname[1]

            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Creating TCPStore on {host}:{self._store_port}"
            )
            store = torch.distributed.TCPStore(
                host_name=host,
                port=self._store_port,
                is_master=True,
                timeout=timedelta(seconds=self._timeout),
                wait_for_workers=False,  # Don't block - workers may not be started yet
            )
            self._store_info = {"host": host, "port": self._store_port}
            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: TCPStore info: {self._store_info}"
            )
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
        store_info: dict | None = None,
        worker_idx: int | None = None,
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
        if worker_idx is None:
            raise RuntimeError("rank was not provided.")
        if kwargs:
            raise RuntimeError(f"Unexpected kwargs: {kwargs.keys()}")

        # Store model_id and context on scheme
        self.model_id = model_id
        self.context = context

        # Get or create store
        # Priority: provided store > provided store_info > self._store_info (from serialization)
        # Connect to master's TCPStore as client
        info = self._store_info
        if info is None:
            raise RuntimeError(
                "TCPStore info not available. init_on_sender() must be called first on the sender side, before passing the scheme to the receiver."
            )
        self._store = self._make_store(is_master=False, store_info=info)

        if (model := getattr(self, "model", None)) is not None:
            self.model = model
            weights_buffer = self._get_weights_buffer_from_model(model)
        else:
            raise RuntimeError("Couldn't find weights")
        self._receiver_transport = self.create_transport(
            store=self._store,
            rank=worker_idx,
            weights_buffer=weights_buffer,
            sync=self.sync,
        )

        # Store worker_idx for synchronize_weights
        self._worker_idx = worker_idx
        # Note: Background thread for async mode is started in connect() after init_process_group

    def _wait_for_instruction(self, timeout: float | None = None) -> str | None:
        """Block until an instruction arrives via TCPStore.

        Args:
            timeout: Maximum time to wait for instruction (seconds).
                None means block indefinitely.

        Returns:
            The instruction string (e.g., "receive", "stop"), or None if
            stop event is set or timeout expires.
        """
        key = f"NODE_{self._worker_idx}_in"
        start_time = time.monotonic()

        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                return None

            try:
                instruction = self._store.get(key)
                self._store.delete_key(key)
                # Decode bytes to string
                return (
                    instruction.decode()
                    if isinstance(instruction, bytes)
                    else instruction
                )
            except RuntimeError:
                # Key doesn't exist yet, continue polling
                pass

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return None

            time.sleep(0.01)

    def _send_instruction(
        self,
        instruction: str = "receive",
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send instruction to receiver(s) via TCPStore.

        Args:
            instruction: The instruction to send (default: "receive").
            worker_ids: Which workers to send to (None = all workers).
        """
        if self._store is None:
            raise RuntimeError(
                "Store not initialized. init_on_sender() must be called first."
            )

        if worker_ids is None:
            target_workers = list(range(self._num_workers)) if self._num_workers else []
        elif isinstance(worker_ids, int):
            target_workers = [worker_ids]
        else:
            target_workers = list(worker_ids)

        # Map instruction to TCPStore format
        store_instruction = (
            b"update_weights" if instruction == "receive" else instruction.encode()
        )

        for worker_idx in target_workers:
            rank = worker_idx + 1  # Workers are 1-indexed in distributed
            self._store.set(f"NODE_{rank}_in", store_instruction)

    def _send_ack(self, message: str = "updated") -> None:
        """Send acknowledgment back to sender via TCPStore.

        Args:
            message: The acknowledgment message (default: "updated").
        """
        if self._store is None or self._worker_idx is None:
            return
        self._store.set(f"NODE_{self._worker_idx}_out", message.encode())

    def _wait_for_ack(
        self,
        worker_ids: int | list[int] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Wait for acknowledgment from receiver(s) via TCPStore.

        Args:
            worker_ids: Which workers to wait for (None = all workers).
            timeout: Maximum time to wait (seconds). None means block indefinitely.
        """
        if self._store is None:
            return

        if worker_ids is None:
            target_workers = list(range(self._num_workers)) if self._num_workers else []
        elif isinstance(worker_ids, int):
            target_workers = [worker_ids]
        else:
            target_workers = list(worker_ids)

        for worker_idx in target_workers:
            rank = worker_idx + 1
            key = f"NODE_{rank}_out"
            try:
                status = self._store.get(key)
                if status != b"updated":
                    torchrl_logger.warning(
                        f"Unexpected ack from worker {worker_idx}: {status}"
                    )
                self._store.delete_key(key)
            except Exception as e:
                torchrl_logger.warning(
                    f"Timeout waiting for ack from worker {worker_idx}: {e}"
                )

    def _background_receive_loop(self):
        """Background thread loop that waits for instructions and receives weights.

        This loop:
        1. Waits for an instruction via TCPStore
        2. Receives weights via torch.distributed
        3. Sends an acknowledgment back
        4. Repeats until stop event is set
        """
        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Background receiver started for worker {self._worker_idx}"
        )
        while not self._stop_event.is_set():
            try:
                instruction = self._wait_for_instruction()
                if instruction is None:
                    continue
                if instruction in ("receive", "update_weights"):
                    torchrl_logger.debug(
                        f"DistributedWeightSyncScheme: Worker {self._worker_idx} "
                        "received 'receive' instruction"
                    )

                    # Receive weights via torch.distributed
                    weights = self._receiver_transport.receive_weights(
                        model=self.model,
                        strategy=self._strategy,
                    )

                    if weights is not None:
                        # Cascade weight update to sub-collectors if context supports it
                        model_id = self._model_id or "policy"
                        if self.context is not None and hasattr(
                            self.context, "update_policy_weights_"
                        ):
                            torchrl_logger.debug(
                                f"DistributedWeightSyncScheme: Cascading weight update to sub-collectors for {model_id=}"
                            )
                            self.context.update_policy_weights_(
                                model_id=model_id, policy_or_weights=weights
                            )

                    # Send acknowledgment
                    self._send_ack("updated")

                    torchrl_logger.debug(
                        f"DistributedWeightSyncScheme: Worker {self._worker_idx} "
                        "received and applied weights"
                    )

                elif instruction == "stop":
                    torchrl_logger.debug(
                        f"DistributedWeightSyncScheme: Worker {self._worker_idx} received 'stop' instruction"
                    )
                    break
                else:
                    torchrl_logger.warning(
                        f"DistributedWeightSyncScheme: Unknown instruction: {instruction}"
                    )

            except Exception as e:
                if not self._stop_event.is_set():
                    torchrl_logger.warning(
                        f"DistributedWeightSyncScheme: Background receiver error: {e}"
                    )

        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Background receiver stopped for worker {self._worker_idx}"
        )

    def _setup_connection_and_weights_on_sender_impl(
        self, *, worker_idx: int | None = None, weights: Any | None = None
    ) -> None:
        """Send initial weights to all workers during connect().

        If the sender has a stateful model (weights available), send them
        to all workers so they start with the correct weights.

        Note: This uses direct torch.distributed send/recv without TCPStore
        signaling to avoid interfering with the main collection loop.
        """
        # Initialize torch.distributed process group if not already done
        # This is a collective operation - all workers must call it
        if not torch.distributed.is_initialized():
            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Initializing process group on sender "
                f"(world_size={self._num_workers + 1})"
            )
            torch.distributed.init_process_group(
                backend=self.backend,
                rank=0,  # Sender is always rank 0
                world_size=self._num_workers + 1,
                timeout=timedelta(seconds=self._timeout),
            )

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
        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = self._worker_idx

        # Initialize torch.distributed process group if not already done
        # This is a collective operation - sender and all workers must call it
        if not torch.distributed.is_initialized():
            torchrl_logger.debug(
                f"DistributedWeightSyncScheme: Initializing process group on worker {worker_idx} "
                f"(world_size={self._num_workers + 1})"
            )
            torch.distributed.init_process_group(
                backend=self.backend,
                rank=worker_idx,
                world_size=self._num_workers + 1,
                timeout=timedelta(seconds=self._timeout),
            )

        if self._receiver_transport is None:
            torchrl_logger.warning(
                "DistributedWeightSyncScheme: No receiver transport, skipping initial weight sync"
            )
            return

        torchrl_logger.debug(
            f"DistributedWeightSyncScheme: Worker {worker_idx} waiting for STATELESS_MODEL key"
        )
        stateless_model = self.receiver_transport._store.get("STATELESS_MODEL")
        if stateless_model not in (b"0", b"1"):
            raise RuntimeError(f"Invalid STATELESS_MODEL value: {stateless_model}")
        if stateless_model == b"1":
            torchrl_logger.debug(
                "DistributedWeightSyncScheme: Skipping initial weight sync on receiver because of stateless model."
            )
        else:
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

        # Start background receiver thread AFTER initial weight sync is complete
        # This prevents the background thread from consuming the initial sync messages
        if self._background_thread is None:
            self._start_background_receiver()

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

    @property
    def model(self) -> Any | None:
        """Get the model associated with this scheme.

        Returns:
            The model if set, None otherwise.
        """
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
        """Set the model for this scheme.

        Args:
            value: The model to set. If None, the setter is a no-op.
        """
        if value is None:
            return
        self._model_ref = weakref.ref(value)

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
        rank: int | None = None,
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
        torchrl_logger.debug(f"RANK 0 -- Send {type(weights)=} to rank {self._rank}")
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
            f"RANK 0 -- Send {type(weights)=} to rank {self._rank} with sync={self._sync}"
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
                torchrl_logger.debug(f"Rank {self._rank} -- calling recv")
                weights_buffer.recv(src=0)
            else:
                torchrl_logger.debug(f"Rank {self._rank} -- calling irecv")
                weights_buffer.irecv(src=0)
        else:
            # Non-blocking receive with timeout support
            torchrl_logger.debug(
                f"Rank {self._rank} -- calling irecv with premature return"
            )
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

        torchrl_logger.debug(f"Rank {self._rank} -- closing receive_weights")
        return weights_buffer

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
        # Note: No TCPStore signaling for initial sync - just direct send/recv
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
