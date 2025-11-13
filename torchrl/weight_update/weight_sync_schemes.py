# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import weakref
from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol

import torch
import torch.distributed

from tensordict import TensorDict, TensorDictBase

from torch import multiprocessing as mp, nn

__all__ = [
    "TransportBackend",
    "MPTransport",
    "SharedMemTransport",
    "RayTransport",
    "RayActorTransport",
    "RPCTransport",
    "DistributedTransport",
    "WeightStrategy",
    "WeightSender",
    "WeightReceiver",
    "RayModuleTransformSender",
    "RayModuleTransformReceiver",
    "WeightSyncScheme",
    "MultiProcessWeightSyncScheme",
    "SharedMemWeightSyncScheme",
    "NoWeightSyncScheme",
    "RayWeightSyncScheme",
    "RayModuleTransformScheme",
    "RPCWeightSyncScheme",
    "DistributedWeightSyncScheme",
]

# ============================================================================
# Transport Layer Abstraction
# ============================================================================


class TransportBackend(Protocol):
    """Abstract interface for different communication mechanisms."""

    def send_weights(self, weights: Any) -> None:
        """Send weights to the receiver."""
        ...

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Receive weights from the sender. Returns (model_id, weights) or None if timeout."""
        ...

    def check_connection(self) -> bool:
        """Check if the connection is still alive."""
        ...

    def synchronize_weights_on_sender(self) -> None:
        """Synchronize weights on sender side before collection starts.

        This is called once after workers are initialized to send the initial
        weights. This can be a no-op (weights are sent via
        send_weights).
        """
        ...

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """Synchronize weights on worker side before collection starts.

        This is called once in each worker after initialization to receive
        the initial weights. This is a no-op (weights are received via
        receive_weights).

        Args:
            worker_idx: The worker index.

        Returns:
            The received weights (for SharedMemTransport) or None.
        """
        ...


class MPTransport:
    """Multiprocessing transport using pipes.

    Args:
        pipe_connection (mp.Pipe): The pipe connection to use for communication.
        timeout (float): The timeout for waiting for acknowledgment. Default is 10 seconds.
    """

    def __init__(self, pipe_connection, timeout: float = 10.0):
        self.timeout = timeout
        self.pipe = pipe_connection

    def send_weights(self, weights: Any) -> None:
        """Send weights through the pipe.

        Sends weights and waits for acknowledgment to ensure delivery.
        """
        self.send_weights_async(weights)
        self.wait_ack()

    def send_weights_async(self, weights: Any) -> None:
        """Send weights through the pipe without waiting for acknowledgment.

        Use wait_ack() to wait for acknowledgment after sending to all workers.
        """
        self.pipe.send((weights, "update_weights"))

    def wait_ack(self) -> None:
        """Wait for acknowledgment from worker."""
        self.check_ack("updated")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Receive weights from the pipe (used in worker process).

        This method only handles weight update messages. Other messages
        (like "close", "continue", etc.) are ignored and should be handled
        by the main worker loop.

        Returns:
            Tuple of (model_id, weights) if weights were received, None if no data available
            or if a non-weight message was received.

        Note:
            model_id is returned as "policy" for backward compatibility, but transports
            are now bound to a single model during initialization.
        """
        if self.pipe.poll(timeout):
            data_in, msg = self.pipe.recv()
            if msg == "update_weights":
                weights = data_in
                return "policy", weights
            else:
                # Not a weight update message - put it back and return None
                # This allows the main worker loop to handle other messages
                # Note: We can't actually "put it back", so we'll just return None
                # and the message is lost. This is why receive() should only be called
                # when we're expecting weight updates, not in the main message loop.
                return None
        # No data available - return None instead of raising TimeoutError
        # This allows non-blocking checks in the worker loop
        return None

    def send_ack(self, message: str = "updated") -> None:
        """Send acknowledgment back to sender."""
        self.pipe.send((None, message))

    def check_ack(self, message: str = "updated") -> None:
        """Check for acknowledgment."""
        _, msg = self.pipe.recv()
        if msg != message:
            raise RuntimeError(f"Expected acknowledgment '{message}', got '{msg}'")

    def check_connection(self) -> bool:
        return not self.pipe.closed

    def synchronize_weights_on_sender(self) -> None:
        """No-op for MPTransport - weights are sent via send_weights()."""

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """No-op for MPTransport - weights are received via receive_weights()."""
        return None


class SharedMemTransport:
    """Shared memory transport for in-place weight updates.

    This transport uses queue-based buffer distribution for initialization, then
    updates shared memory tensors directly for subsequent weight updates.
    Workers automatically see weight updates without explicit communication.

    Initialization flow:
    - Shared memory buffers are created and sent to workers via per-worker queues
    - Workers receive the buffer reference and apply weights to their models
    - Subsequent updates are pure in-place shared memory (zero-copy)

    Both CPU and CUDA tensors maintain shared references when sent through mp.Queue.

    """

    def __init__(self):
        self._params_map = None  # a dict[worker_idx, TensorDictBase] map
        self._weight_queues = (
            None  # Dict of per-worker queues for distributing shared weights
        )

    def register_weights(
        self, params_map: dict[int, mp.Queue], init_queues: dict[int, mp.Queue]
    ) -> None:
        """Initialize per-worker queues for shared memory buffer distribution."""
        self._weight_queues = init_queues
        self._params_map = params_map
        # Create set of the unique weights
        self._unique_weights = []
        for weights in params_map.values():
            if id(weights) in [id(w) for w in self._unique_weights]:
                continue
            self._unique_weights.append(weights)

    def synchronize_weights_on_sender(self) -> None:
        """Send shared memory buffer reference to workers via their per-worker queues.

        Both CPU and CUDA tensors maintain shared references through queues.
        Each worker reads from its own dedicated queue, to avoid race conditions.

        """
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        for worker_idx, queue in self._weight_queues.items():
            weights = self._params_map[worker_idx]
            queue.put(weights)

    def synchronize_weights_on_worker(
        self, worker_idx: int, timeout: float = 10.0
    ) -> TensorDictBase:
        """Receive shared memory buffer reference from sender via their per-worker queues.

        Each worker reads from its own dedicated queue, to avoid race conditions.

        Args:
            worker_idx: The worker index.
            timeout: Timeout for reading from queue.

        Returns:
            The shared memory weights TensorDict.
        """
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        if worker_idx not in self._weight_queues:
            raise RuntimeError(f"Worker {worker_idx} not registered in queues.")

        # Read from dedicated queue for this worker
        worker_queue = self._weight_queues[worker_idx]
        weights = worker_queue.get(timeout=timeout)
        return weights

    def send_weights(self, weights: Any) -> None:
        """Update weights in-place in shared memory.

        Args:
            weights: New weights to send. Can be a TensorDictBase or dict.

        Raises:
            ValueError: If weights type is unsupported.
        """
        # Update shared memory in-place (workers see this automatically)
        if isinstance(weights, dict):
            weights = TensorDict(weights)
        if not isinstance(weights, TensorDictBase):
            raise ValueError(f"Unsupported weights type: {type(weights)}")
        # Unflatten if needed to match shared buffer structure
        weights_to_update = weights
        if any("." in key for key in weights.keys()):
            weights_to_update = weights.unflatten_keys(".")

        for buffer in self._unique_weights:
            buffer.update_(weights_to_update, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """No-op for shared memory - weights are already visible."""
        return None

    def send_ack(self, message: str = "updated") -> None:
        """No-op for shared memory - no acknowledgment needed."""

    def check_ack(self, message: str = "updated") -> None:
        """No-op for shared memory - no acknowledgment needed."""

    def check_connection(self) -> bool:
        """Shared memory is always 'connected'."""
        return True


class RayTransport:
    """Ray transport for communicating with a single Ray collector actor.

    This transport handles weight updates for ONE specific remote collector.
    Multiple transports are created for multiple collectors, following the
    same pattern as multiprocess collectors.
    """

    def __init__(
        self,
        remote_collector=None,
        tensor_transport: Literal["object_store", "nixl"] = "object_store",
    ):
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayTransport")
        self._remote_collector = remote_collector
        self._tensor_transport = tensor_transport

    def send_weights(self, weights: Any) -> None:
        """Send weights to the remote collector via Ray."""
        if self._remote_collector is None:
            return

        # Put weights in Ray's object store for efficient distribution
        # Ray will automatically deduplicate if the same weights are sent to multiple actors
        weights_ref = self.ray.put(weights, _tensor_transport=self._tensor_transport)

        # Send to the remote collector and wait for completion
        # This ensures weights are applied before we continue
        future = self._remote_collector.update_policy_weights_.remote(
            policy_or_weights=weights_ref
        )
        self.ray.wait([future], num_returns=1)

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to remote collector without waiting for completion.

        Use wait_ack() to wait for completion after sending to all workers.
        """
        if self._remote_collector is None:
            return

        weights_ref = self.ray.put(weights, _tensor_transport=self._tensor_transport)
        self._pending_future = self._remote_collector.update_policy_weights_.remote(
            policy_or_weights=weights_ref
        )

    def wait_ack(self) -> None:
        """Wait for the remote collector to finish applying weights."""
        if hasattr(self, "_pending_future"):
            self.ray.wait([self._pending_future], num_returns=1)
            del self._pending_future
        else:
            raise RuntimeError("No pending future. Did you call send_weights_async?")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Ray workers typically don't receive weights through this transport."""
        return None

    def check_connection(self) -> bool:
        """Check if Ray is initialized."""
        return self.ray.is_initialized()

    def synchronize_weights_on_sender(self) -> None:
        """No-op for RayTransport - weights are sent via send_weights()."""

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """No-op for RayTransport - weights are received via remote method calls."""
        return None


class RayActorTransport:
    """Ray transport for communicating with Ray actors (not collectors).

    This transport is designed for updating models hosted within Ray actors,
    such as RayModuleTransform instances. It directly calls the actor's
    update_weights method rather than going through collector update methods.
    """

    def __init__(
        self,
        actor_ref=None,
        update_method: str = "tensordict",
        tensor_transport: Literal["object_store", "nixl"] = "object_store",
    ):
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayActorTransport")

        self._actor_ref = actor_ref
        self._update_method = update_method
        self._tensor_transport = tensor_transport

    def set_actor(self, actor_ref):
        """Set the Ray actor reference to communicate with."""
        self._actor_ref = actor_ref

    def send_weights(self, weights: Any) -> None:
        """Send weights to the Ray actor."""
        if self._actor_ref is None:
            return

        weights_ref = self.ray.put(weights, _tensor_transport=self._tensor_transport)

        if self._update_method == "tensordict":
            self.ray.get(
                self._actor_ref._update_weights_tensordict.remote(params=weights_ref)
            )
        elif self._update_method == "state_dict":
            self.ray.get(
                self._actor_ref._update_weights_state_dict.remote(
                    state_dict=weights_ref
                )
            )
        else:
            raise ValueError(f"Unknown update method: {self._update_method}")

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to Ray actor without waiting for completion.

        Use wait_ack() to wait for completion after sending to all actors.
        """
        if self._actor_ref is None:
            return

        weights_ref = self.ray.put(weights, _tensor_transport=self._tensor_transport)

        if self._update_method == "tensordict":
            self._pending_future = self._actor_ref._update_weights_tensordict.remote(
                params=weights_ref
            )
        elif self._update_method == "state_dict":
            self._pending_future = self._actor_ref._update_weights_state_dict.remote(
                state_dict=weights_ref
            )
        else:
            raise ValueError(f"Unknown update method: {self._update_method}")

    def wait_ack(self) -> None:
        """Wait for Ray actor to finish applying weights."""
        if hasattr(self, "_pending_future"):
            self.ray.get(self._pending_future)
            del self._pending_future
        else:
            raise RuntimeError("No pending future. Did you call send_weights_async?")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Ray actor workers receive weights through direct method calls."""
        return None

    def send_ack(self, message: str = "updated") -> None:
        """No acknowledgment needed for Ray actors."""

    def check_ack(self, message: str = "updated") -> None:
        """No acknowledgment needed for Ray actors."""

    def check_connection(self) -> bool:
        """Check if Ray is initialized and actor exists."""
        if not self.ray.is_initialized():
            return False
        if self._actor_ref is None:
            return False
        return True

    def synchronize_weights_on_sender(self) -> None:
        """No-op for RayActorTransport - weights are sent via send_weights()."""

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """No-op for RayActorTransport - weights are received via remote method calls."""
        return None


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


# ============================================================================
# Weight Strategies
# ============================================================================


class WeightStrategy:
    """Unified strategy for weight transmission.

    This strategy handles both extraction and application of weights, supporting
    both TensorDict and state_dict formats.

    Args:
        extract_as (str): Format for extracting weights. Can be:
            - "tensordict" (default): Extract weights as TensorDict
            - "state_dict": Extract weights as PyTorch state_dict

    The application format is automatically detected based on the type of weights
    received (dict -> state_dict, TensorDict -> tensordict).
    """

    def __init__(self, extract_as: Literal["tensordict", "state_dict"] = "tensordict"):
        if extract_as not in ("tensordict", "state_dict"):
            raise ValueError(
                f"extract_as must be 'tensordict' or 'state_dict', got {extract_as}"
            )
        self.extract_as = extract_as

    def extract_weights(self, source: Any) -> Any:
        """Extract weights from source model in the specified format.

        Args:
            source: The model to extract weights from. Can be:
                - nn.Module: PyTorch module
                - TensorDictBase: TensorDict
                - dict: State dictionary

        Returns:
            Weights in the format specified by `extract_as` constructor argument.
        """
        if self.extract_as == "tensordict":
            # Extract as TensorDict
            if isinstance(source, nn.Module):
                return TensorDict.from_module(source)
            elif isinstance(source, TensorDictBase):
                return source
            elif isinstance(source, dict):
                # Convert state_dict to TensorDict
                return TensorDict(source, batch_size=[])
            else:
                raise ValueError(
                    f"Unsupported source type for TensorDict extraction: {type(source)}"
                )
        else:  # state_dict
            # Extract as state_dict
            if isinstance(source, nn.Module):
                return source.state_dict()
            elif isinstance(source, dict):
                return source
            elif isinstance(source, TensorDictBase):
                # Convert TensorDict to state_dict
                return source.to_dict()
            else:
                raise ValueError(
                    f"Unsupported source type for state_dict extraction: {type(source)}"
                )

    def apply_weights(self, destination: Any, weights: Any) -> None:
        """Apply weights to destination model.

        The format is automatically detected from the weights type:
        - dict -> state_dict format
        - TensorDictBase -> tensordict format

        Args:
            destination: The model to apply weights to. Can be:
                - nn.Module: PyTorch module
                - TensorDictBase: TensorDict
                - dict: State dictionary
            weights: The weights to apply (dict or TensorDictBase).
        """
        if weights is None:
            return

        # Auto-detect format from weights type
        if isinstance(weights, dict):
            weights = TensorDict(weights)
            if any("." in key for key in weights.keys()):
                weights = weights.unflatten_keys(".")
        if isinstance(destination, nn.Module):
            # Do not update in-place
            weights.to_module(destination)
            return
        elif isinstance(destination, dict):
            destination = TensorDict(destination)
            if any(isinstance(key, str) and "." in key for key in destination.keys()):
                destination = destination.unflatten_keys(".")

        if isinstance(weights, TensorDictBase):
            # Apply TensorDict format
            if isinstance(destination, TensorDictBase):
                try:
                    destination.data.update_(weights.data)
                except Exception as e:
                    raise KeyError(
                        f"Error updating destination: {e}. Destination keys: {destination.keys(True, True)}, weights keys: {weights.keys(True, True)}"
                    )
            else:
                raise ValueError(
                    f"Unsupported destination type for TensorDict: {type(destination)}"
                )
        else:
            raise ValueError(
                f"Unsupported weights type: {type(weights)}. Expected dict or TensorDictBase."
            )


def _get_strategy(strategy: Literal["tensordict", "state_dict"]) -> WeightStrategy:
    """Get strategy object from string name.

    Args:
        strategy: Either "tensordict" or "state_dict".

    Returns:
        WeightStrategy: Strategy configured with the specified extraction format.
    """
    if strategy not in ("tensordict", "state_dict"):
        raise ValueError(
            f"Unknown strategy: {strategy}. Must be 'tensordict' or 'state_dict'."
        )
    return WeightStrategy(extract_as=strategy)


# ============================================================================
# Sender (Trainer/Main Process Side)
# ============================================================================


class WeightSender:
    """Sends weights for ONE model to workers.

    A single sender can broadcast to all workers or send to specific workers.
    Created and managed by WeightSyncScheme. Users should not instantiate directly.
    """

    _transport: TransportBackend | None
    _transports: dict[int, TransportBackend]

    def __init__(self, scheme: WeightSyncScheme):
        self._scheme = scheme
        self._transports: dict[int, TransportBackend] = {}  # worker_idx -> transport
        self._transport: TransportBackend | None = None
        self._model_id = "policy"  # Default model ID
        self._strategy = _get_strategy(scheme.strategy)
        self._context_ref = None  # weakref to collector for model resolution
        self._pending_async = False  # Track if async send is pending

    def _set_context(self, context: Any, model_id: str | None = None) -> None:
        """Set the context object (collector) for model resolution (internal).

        This is now handled by init_on_sender(). Only kept for internal use.

        Args:
            context: The collector instance.
            model_id: Optional model identifier (for compatibility with RayModuleTransformSender).
        """
        self._context_ref = weakref.ref(context)
        if model_id is not None:
            self._model_id = model_id

    def _register_worker(self, worker_idx: int, pipe_or_context: Any) -> None:
        """Register a worker's communication pipe (internal).

        This is now handled by init_on_sender(). Only kept for internal use.

        Args:
            worker_idx: The worker index.
            pipe_or_context: The pipe connection for this worker.
        """
        if worker_idx not in self._transports:
            self._transports[worker_idx] = self._scheme.create_transport(
                pipe_or_context
            )

    def _iterate_transports(
        self, worker_ids: int | list[int] | None = None
    ) -> Iterator[TransportBackend]:
        """Iterate over transports for specified workers."""
        if worker_ids is None:
            # All workers
            if not self._transports:
                yield self._transport
            else:
                yield from self._transports.values()
        else:
            # Specific workers
            if isinstance(worker_ids, int):
                worker_ids = [worker_ids]
            for worker_id in worker_ids:
                if worker_id in self._transports:
                    yield self._transports[worker_id]
                else:
                    raise ValueError(f"Worker {worker_id} not registered")

    def send(
        self,
        weights: Any = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send weights synchronously to workers.

        This method:
        1. Prepares weights (extracts from model if weights=None)
        2. Sends to specified workers (or all if worker_ids=None)
        3. Waits for acknowledgments from those workers
        4. Returns when workers have applied the weights

        Args:
            weights: Weights to send. Can be:
                - None: Extract from model via context.get_model(model_id)
                - nn.Module: Extract weights from module
                - TensorDict: Use directly
                - dict: Convert to TensorDict
            worker_ids: Which workers to send to:
                - None: Send to all workers (default)
                - int: Send to single worker
                - list[int]: Send to specific workers

        Note: This is a blocking call that ensures specified workers are updated
        before returning.
        """
        if self._pending_async:
            raise RuntimeError(
                "Cannot call send() while an async send is pending. Call wait_async() first."
            )

        model_id = getattr(self, "_model_id", "policy")
        context = self._context_ref() if self._context_ref is not None else None

        # Let the scheme prepare the weights
        prepared_weights = self._scheme.prepare_weights(
            weights=weights,
            model_id=model_id,
            strategy=self._strategy,
            context=context,
        )

        transports = list(self._iterate_transports(worker_ids))

        # Send to all workers first (non-blocking if transport supports it)
        for transport in transports:
            if hasattr(transport, "send_weights_async"):
                transport.send_weights_async(prepared_weights)
            else:
                # Fallback for transports that don't support async send
                transport.send_weights(prepared_weights)

        # Wait for all acknowledgments
        for transport in transports:
            if hasattr(transport, "wait_ack"):
                transport.wait_ack()

    def send_async(
        self,
        weights: Any = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send weights asynchronously to workers (non-blocking).

        This initiates the send but returns immediately without waiting
        for workers to acknowledge. You must call wait_async() before
        the next send_async() or send() call.

        Args:
            weights: Same as send()
            worker_ids: Same as send()

        Raises:
            RuntimeError: If a previous send_async() is still pending
        """
        if self._pending_async:
            raise RuntimeError(
                "Cannot call send_async() again while a previous send is pending. Call wait_async() first."
            )

        model_id = getattr(self, "_model_id", "policy")
        context = self._context_ref() if self._context_ref is not None else None

        # Let the scheme prepare the weights
        prepared_weights = self._scheme.prepare_weights(
            weights=weights,
            model_id=model_id,
            strategy=self._strategy,
            context=context,
        )

        # Store transports for wait_async
        self._pending_transports = list(self._iterate_transports(worker_ids))

        # Send to all workers (non-blocking)
        for transport in self._pending_transports:
            if hasattr(transport, "send_weights_async"):
                transport.send_weights_async(prepared_weights)
            else:
                raise RuntimeError(
                    f"transport of type {type(transport)} does not support async send."
                )

        self._pending_async = True

    def wait_async(self) -> None:
        """Wait for a pending async send to complete.

        Blocks until all workers have acknowledged the previous send_async().
        This must be called after send_async() before any subsequent sends.

        Raises:
            RuntimeError: If no async send is pending
        """
        if not self._pending_async:
            raise RuntimeError("No async send is pending. Call send_async() first.")

        # Wait for all acknowledgments
        for transport in self._pending_transports:
            if hasattr(transport, "wait_ack"):
                transport.wait_ack()

        self._pending_async = False
        self._pending_transports = None

    def synchronize_weights(self) -> None:
        """Synchronize weights with workers before collection starts.

        This method is called once after workers are initialized to send
        the initial weights. For most transports this is a no-op (weights
        are sent via send()). For SharedMemTransport, this sends buffer
        references via queues.

        This is different from send() which is called during training to
        update weights.
        """
        # Iterate over all transports and call synchronize_weights_on_sender
        for transport in self._iterate_transports():
            if hasattr(transport, "synchronize_weights_on_sender"):
                transport.synchronize_weights_on_sender()

    def update_weights(self, weights: Any) -> None:
        """Send weights to ALL workers for this model.

        Args:
            weights: Weights to send (can be None, nn.Module, TensorDict, etc.).

        Note:
            Convenience method that calls send(weights=weights).
        """
        self.send(weights=weights)

    def __getstate__(self):
        """Pickle support: discard context weakref."""
        state = self.__dict__.copy()
        state["_context_ref"] = None
        state["_pending_async"] = False
        state["_pending_transports"] = None
        return state

    def __setstate__(self, state):
        """Pickle support: restore state without context."""
        self.__dict__.update(state)


# ============================================================================
# Receiver (Worker Process Side)
# ============================================================================


class WeightReceiver:
    """Receives weights for ONE model in ONE worker.

    Created and managed by WeightSyncScheme. Users should not instantiate directly.
    """

    def __init__(self, scheme: WeightSyncScheme):
        self._scheme = scheme
        self._context_ref = None  # weakref to inner_collector
        self._transport = None  # lazy
        self._model_ref = None
        self._strategy = _get_strategy(scheme.strategy)
        self._worker_idx = None  # Set by SharedMemWeightSyncScheme.init_on_worker()

    def _set_context(self, context: Any) -> None:
        """Set the context object (inner_collector) for resolving references (internal).

        This is now handled by init_on_worker(). Only kept for internal use.

        Args:
            context: The inner collector instance in the worker process.
        """
        self._context_ref = weakref.ref(context)

    def _register_model(self, model_ref: Any) -> None:
        """Register the model to apply weights to (internal).

        This is now handled by init_on_worker(). Only kept for internal use.

        Args:
            model_ref: Either a direct object reference or a string path like 'policy' or 'env.value_net'.
        """
        self._model_ref = model_ref

    def _register_worker_transport(self, pipe: Any) -> None:
        """Register this worker's communication pipe (internal).

        This is now handled by init_on_worker(). Only kept for internal use.

        Args:
            pipe: The pipe connection for this worker.
        """
        self._transport = self._scheme.create_transport(pipe)

    def receive(self, timeout: float = 0.001) -> bool:
        """Check for and apply new weights (non-blocking).

        This method is called in the worker's main loop to check if
        new weights have been sent. If weights are available, they
        are applied to the registered model immediately.

        Args:
            timeout: Maximum time to wait for weights (seconds).
                     Use 0 for immediate return.

        Returns:
            True if weights were received and applied
            False if no weights were available

        Note: For SharedMemWeightSyncScheme, this always returns False
        since workers automatically see updates via shared memory.
        """
        if self._transport is None:
            return False

        # Try to receive weights
        result = self._transport.receive_weights(timeout=timeout)
        if result is None:
            return False

        model_id, weights = result

        # Apply weights to the model
        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self._resolve_model_ref()
        self._strategy.apply_weights(model, weights)

        # Send acknowledgment if transport supports it
        if hasattr(self._transport, "send_ack"):
            self._transport.send_ack("updated")

        return True

    def synchronize_weights(self, worker_idx: int | None = None) -> None:
        """Synchronize weights with sender before collection starts.

        This method is called once after the worker is initialized to receive
        the initial weights. For most transports this is a no-op (weights are
        received via receive()). For SharedMemTransport, this receives the
        buffer reference via queue and applies it to the model.

        This is different from receive() which is called during collection
        to check for weight updates.

        Args:
            worker_idx: The worker index (required for SharedMemTransport).
                If not provided, uses the worker_idx stored during init_on_worker().
        """
        if self._transport is None:
            return

        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = getattr(self, "_worker_idx", None)

        # Call transport's synchronize method if available
        weights = self._transport.synchronize_weights_on_worker(worker_idx)

        # Apply weights to model if received (SharedMemTransport case)
        if weights is not None and self._model_ref is not None:
            model = self._resolve_model_ref()
            self._strategy.apply_weights(model, weights)
        else:
            raise ValueError("Failed to synchronize weights")

    def apply_weights(self, weights: Any) -> None:
        """Apply received weights to registered model.

        Args:
            weights: The weights to apply.

        Note:
            Convenience method. Normally weights are received and applied via receive() in the worker loop.
        """
        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self._resolve_model_ref()
        self._strategy.apply_weights(model, weights)

        # Send acknowledgment if transport supports it
        if hasattr(self._transport, "send_ack"):
            self._transport.send_ack("updated")

    def _resolve_model_ref(self) -> Any:
        """Resolve model reference to actual object."""
        if isinstance(self._model_ref, str):
            if self._context_ref is None:
                raise ValueError("Context is required to resolve string references")
            context = self._context_ref()
            if context is None:
                raise ValueError("Context has been garbage collected")
            return _resolve_model(context, self._model_ref)
        return self._model_ref

    def __getstate__(self):
        """Pickle support: discard context weakref."""
        state = self.__dict__.copy()
        state["_context_ref"] = None
        return state

    def __setstate__(self, state):
        """Pickle support: restore state without context."""
        self.__dict__.update(state)


class RayModuleTransformSender(WeightSender):
    """Specialized sender for :class:`~torchrl.envs.transforms.module.RayModuleTransform` actors.

    This sender handles weight updates for models hosted within Ray actors.
    Unlike the base WeightSender which uses pipes for multiprocessing,
    this sender directly communicates with Ray actors via their remote methods.

    For Ray actors, there is typically only one shared actor instance, so we
    store a single transport rather than per-worker transports.
    """

    def __init__(self, scheme: RayModuleTransformScheme):
        super().__init__(scheme)
        self._actor_ref = None
        self._single_transport = None
        self._context_ref = None
        self._model_id_str = None

    def _set_context(self, context: Any, model_id: str) -> None:
        """Set context for lazy actor resolution (internal).

        This is now handled by init_on_sender(). Only kept for internal use.

        Args:
            context: The collector instance.
            model_id: String path to the Ray actor (e.g., "env.transform[0]").
        """
        self._context_ref = weakref.ref(context)
        self._model_id_str = model_id

    def _register_worker(self, worker_idx: int, pipe_or_context: Any) -> None:
        """For Ray actors, worker registration is a no-op (internal).

        Ray actors are shared across all workers, so we don't need per-worker
        transports. The actor reference is resolved lazily on first use.
        """

    def update_weights(self, weights: Any) -> None:
        """Send weights to the Ray actor.

        Args:
            weights: Weights to send.
        """
        if self._single_transport is None:
            self._initialize_transport()

        if self._single_transport is not None:
            self._single_transport.send_weights(weights)

    def _initialize_transport(self) -> None:
        """Lazily initialize the transport by resolving the actor reference."""
        if self._context_ref is None or self._model_id_str is None:
            return

        context = self._context_ref()
        if context is None:
            return

        model = _resolve_model(context, self._model_id_str)
        if hasattr(model, "_actor"):
            self._actor_ref = model._actor
            self._single_transport = self._scheme.create_transport(model)
        elif type(model).__name__ == "ActorHandle":
            self._actor_ref = model
            self._single_transport = self._scheme.create_transport(model)


class RayModuleTransformReceiver(WeightReceiver):
    """Specialized receiver for RayModuleTransform actors.

    This receiver handles weight updates within Ray actors.
    Since Ray actors receive weights through direct method calls,
    this receiver primarily validates and applies weights locally.
    """

    def __init__(self, scheme: RayModuleTransformScheme):
        super().__init__(scheme)

    def _register_worker_transport(self, actor_or_context: Any) -> None:
        """Register the Ray actor's transport (internal).

        This is now handled by init_on_worker(). Only kept for internal use.

        Args:
            actor_or_context: Either a Ray actor reference or a context object.
        """
        self._transport = self._scheme.create_transport(actor_or_context)

    def apply_weights(self, weights: Any) -> None:
        """Apply received weights to registered model.

        For Ray actors, weights are applied directly to the module
        within the actor's process space.

        Args:
            weights: The weights to apply.
        """
        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self._resolve_model_ref()
        self._strategy.apply_weights(model, weights)


# ============================================================================
# Weight Synchronization Schemes
# ============================================================================


class WeightSyncScheme(metaclass=abc.ABCMeta):
    """Configuration for how to synchronize ONE model across workers.

    A scheme manages synchronization of ONE model across workers.
    The collector maintains a dict of {model_id: scheme} pairs.
    """

    def __init__(self, strategy: Literal["state_dict", "tensordict"] = "state_dict"):
        self.strategy = strategy
        self._sender = None
        self._receiver = None
        self._initialized_on_sender = False
        self._initialized_on_worker = False

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        This method is called once in the collector's _run_processes() method,
        after workers have been started and are ready to receive messages.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (e.g., collector) providing:
                - .pipes: list[mp.Connection]
                - .get_model(model_id: str) -> nn.Module
                - .get_cached_weights(model_id: str) -> TensorDict | None
                - .num_workers: int
            **kwargs: Alternative to context (pipes, num_workers, model, cached_weights, etc.)
        """
        raise NotImplementedError

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        This method is called once in each worker's initialization.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (e.g., inner collector) providing:
                - .pipe: mp.Connection
                - .get_model(model_id: str) -> nn.Module
            **kwargs: Alternative to context (pipe, model, etc.)
        """
        raise NotImplementedError

    def get_sender(self) -> WeightSender:
        """Get the sender instance.

        Returns:
            Sender instance for sending weights to workers

        Raises:
            RuntimeError: If init_on_sender() hasn't been called yet
        """
        if not self._initialized_on_sender or self._sender is None:
            raise RuntimeError(
                f"Must call init_on_sender() before get_sender() on {type(self).__name__}"
            )
        return self._sender

    def get_receiver(self) -> WeightReceiver:
        """Get the receiver instance.

        Returns:
            Receiver instance for receiving weights in this worker

        Raises:
            RuntimeError: If init_on_worker() hasn't been called yet
        """
        if not self._initialized_on_worker or self._receiver is None:
            raise RuntimeError(
                f"Must call init_on_worker() before get_receiver() on {type(self).__name__}"
            )
        return self._receiver

    def __getstate__(self):
        """Prepare the scheme for pickling by excluding non-serializable runtime state.

        Sender and receiver objects contain pipes, weak references, and other
        non-serializable resources that should not be pickled. These will be
        re-initialized when needed after unpickling.
        """
        state = self.__dict__.copy()
        # Remove non-serializable runtime state
        state["_sender"] = None
        state["_receiver"] = None
        state["_initialized_on_sender"] = False
        state["_initialized_on_worker"] = False
        return state

    def __setstate__(self, state):
        """Restore the scheme from pickling."""
        self.__dict__.update(state)

    @abc.abstractmethod
    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create transport for communication.

        Args:
            pipe_or_context: Either a pipe connection or context object to extract pipe from.

        Returns:
            A transport backend instance.

        Note:
            This is used internally by init_on_sender/init_on_worker.
        """
        ...

    def create_sender(self) -> WeightSender:
        """Create a sender for this scheme.

        Returns:
            WeightSender instance configured for this scheme.

        Note:
            Typically you should use init_on_sender() followed by get_sender() instead.
        """
        return WeightSender(self)

    def create_receiver(self) -> WeightReceiver:
        """Create a receiver for this scheme.

        Returns:
            WeightReceiver instance configured for this scheme.

        Note:
            Typically you should use init_on_worker() followed by get_receiver() instead.
        """
        return WeightReceiver(self)

    def prepare_weights(
        self,
        weights: Any,
        model_id: str,
        strategy: WeightStrategy,
        context: Any = None,
    ) -> Any:
        """Prepare weights for sending.

        This method handles weight extraction, conversion, and any scheme-specific
        preparation (e.g., cache lookups for SharedMemWeightSyncScheme).

        Args:
            weights: Raw weights input (can be None, nn.Module, TensorDict, dict, str reference, etc.)
            model_id: The model identifier (e.g., "policy")
            strategy: WeightStrategy for extracting/converting weights
            context: Optional context (e.g., collector) for model resolution

        Returns:
            Prepared weights ready to send via transport
        """
        # Default implementation: extract from model or pass through
        if weights is None and context is not None:
            # Try to resolve and extract from model in context
            try:
                model = _resolve_model(context, model_id)
                return strategy.extract_weights(model)
            except (AttributeError, KeyError):
                pass
            # Try fallback policy
            if model_id == "policy" and hasattr(context, "_fallback_policy"):
                if context._fallback_policy is not None:
                    return strategy.extract_weights(context._fallback_policy)
            return None

        if isinstance(weights, nn.Module):
            return strategy.extract_weights(weights)
        elif isinstance(weights, str):
            # String reference to model
            if context is not None:
                model = _resolve_model(context, weights)
                return strategy.extract_weights(model)
            raise ValueError(
                f"Cannot resolve string reference '{weights}' without context"
            )
        else:
            # Already extracted weights (TensorDict, dict, etc.)
            return weights


class MultiProcessWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for multiprocess operations using pipes.

    This scheme creates transports that communicate via multiprocessing pipes.
    """

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing pipes and num_workers
            **kwargs: Alternative to context (pipes, num_workers, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            pipes = getattr(context, "pipes", None)
            num_workers = getattr(context, "num_workers", None)
        else:
            pipes = kwargs.get("pipes")
            num_workers = kwargs.get("num_workers")

        if pipes is None:
            raise ValueError("pipes must be provided via context or kwargs")
        if num_workers is None:
            num_workers = len(pipes) if pipes else 0

        # Create sender and register all workers
        sender = WeightSender(self)
        sender._model_id = model_id
        if context is not None:
            sender._context_ref = weakref.ref(context)

        for worker_idx, pipe in enumerate(pipes):
            sender._register_worker(worker_idx, pipe)

        self._sender = sender
        self._initialized_on_sender = True

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing pipe and model
            **kwargs: Alternative to context (pipe, model, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            pipe = getattr(context, "pipe", None)
            if hasattr(context, "get_model"):
                model = context.get_model(model_id)
            else:
                model = None
        else:
            pipe = kwargs.get("pipe")
            model = kwargs.get("model")

        if pipe is None:
            raise ValueError("pipe must be provided via context or kwargs")

        # Create receiver and register model
        receiver = WeightReceiver(self)
        if context is not None:
            receiver._context_ref = weakref.ref(context)
        receiver._register_worker_transport(pipe)
        if model is not None:
            receiver._register_model(model)
        else:
            # Register by model_id for later resolution
            receiver._register_model(model_id)

        self._receiver = receiver
        self._initialized_on_worker = True

    def create_transport(self, pipe: Any) -> TransportBackend:
        """Create an MPTransport using the provided pipe.

        Note:
            This is used internally by init_on_sender/init_on_worker.
        """
        return MPTransport(pipe)


class SharedMemWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization using shared memory.

    This scheme uses shared memory for in-place weight updates. Workers
    automatically see weight updates without explicit message passing.

    Args:
        strategy: The weight transmission strategy (default: "tensordict").

    Example:
        >>> # Basic usage
        >>> scheme = SharedMemWeightSyncScheme()
        >>> # Weights are initialized via init_on_sender()
    """

    def __init__(
        self,
        strategy: str = "tensordict",
    ):
        super().__init__(strategy)
        # Create a single shared transport for all workers
        self._shared_transport = SharedMemTransport()
        # Create per-worker queues to avoid race conditions
        # Each worker gets its own queue for weight initialization
        self._weight_init_queues = {}  # worker_idx -> Queue
        # General message queue for coordination (if needed in future)
        self._message_queue = mp.Queue()

    def init_on_sender(
        self,
        model_id: str | None = None,
        context: Any = None,
        weights: TensorDictBase | None = None,
        model: nn.Module | None = None,
        params_map: dict[int, TensorDictBase] | None = None,
        devices: list[torch.device] | None = None,
        device_map_fn: Callable[[int, TensorDictBase], TensorDictBase] | None = None,
        num_workers: int | None = None,
    ) -> None:
        """Initialize on the main process (sender side).

        We create a map dict[worker_idx, weights_on_device]. Each model will be assigned a device. If two workers
        share the same device, the entry in the dict will be the same.
        To do this, we need to know the number of workers, their assigned device, and have access to the parameters.
        If a context is provided, we read the devices from it. If not, the dict[worker_idx, device] map must be provided
        explicitly.

        In some cases, the policy on the worker side will be on multiple devices which may or may not be the same as the
        devices on the main process. In this case, init_on_sender() needs to receive a mapping function as argument that
        will take as input the worker_idx and the parameters and return a new set of parameters on the desired devices.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing device_to_workers mapping and model access
            weights: Pre-extracted weights as TensorDict (for policy factory usage)
            model: Model to extract weights from
            params_map: Direct mapping of worker_idx to weights on device (most explicit)
            devices: List of devices for each worker
            device_map_fn: Custom function to map worker_idx and weights to device-specific weights
            num_workers: Number of workers (required with device_map_fn)

        Examples:
            Simple usage with collector context (stateful policy):

            >>> policy = make_stateful_policy()
            >>> scheme = SharedMemWeightSyncScheme(strategy="tensordict")
            >>> collector = MultiSyncDataCollector(
            ...     create_env_fn=[lambda: GymEnv("CartPole-v1")],
            ...     policy=policy,
            ...     frames_per_batch=100,
            ...     total_frames=1000,
            ...     weight_sync_schemes={"policy": scheme},
            ... )
            >>> # scheme.init_on_sender() is called automatically by collector

            Pre-initialized usage (policy factory):

            >>> policy_on_main = make_stateful_policy()
            >>> scheme = SharedMemWeightSyncScheme(strategy="tensordict")
            >>> # Must initialize before collector creation when using policy_factory
            >>> scheme.init_on_sender(
            ...     model_id="policy",
            ...     weights=TensorDict.from_module(policy_on_main),
            ...     devices=[torch.device("cuda:0"), torch.device("cuda:1")],
            ...     num_workers=2,
            ... )
            >>> collector = MultiSyncDataCollector(
            ...     create_env_fn=[lambda: GymEnv("CartPole-v1")],
            ...     policy_factory=[make_stateful_policy],
            ...     frames_per_batch=100,
            ...     total_frames=1000,
            ...     weight_sync_schemes={"policy": scheme},
            ... )

            Direct params_map usage (advanced):

            >>> weights_cpu = TensorDict.from_module(policy).share_memory_()
            >>> weights_cuda = weights_cpu.to("cuda").share_memory_()
            >>> scheme = SharedMemWeightSyncScheme(strategy="tensordict")
            >>> scheme.init_on_sender(
            ...     model_id="policy",
            ...     params_map={0: weights_cpu, 1: weights_cuda, 2: weights_cuda},
            ... )
        """
        # Plan: the goal of this init is to obtain a map dict[worker_idx, weights_on_device] that we can use to init
        #       the weights on the workers.
        #       Scenarios:
        #           - Easiest scenario: the user provides the map directly (params_map). Nothing to do other than creating
        #                 the transport and registering the workers etc.
        #           - The user provides a model or its params and a device map. We need to create the map from the params
        #                 explicitly.
        #           - The user provides a context (e.g. a Collector) and a model_id. Same as above, except that we need
        #                 to collect the model from the context.
        params_map = self._get_params_map(
            context=context,
            model_id=model_id,
            weights=weights,
            model=model,
            params_map=params_map,
            devices=devices,
            device_map_fn=device_map_fn,
            num_workers=num_workers,
        )

        # Create per-worker queues if not already created
        # Collect all unique worker indices
        all_workers = list(params_map.keys())

        for worker_idx in all_workers:
            if worker_idx not in self._weight_init_queues:
                self._weight_init_queues[worker_idx] = mp.Queue()

        # Set worker info in transport
        self._shared_transport.register_weights(params_map, self._weight_init_queues)

        # Create sender with the shared transport
        sender = WeightSender(self)
        sender._model_id = model_id
        sender._transport = self._shared_transport  # Use shared transport
        if context is not None:
            sender._context_ref = weakref.ref(context)

        self._sender = sender
        self._initialized_on_sender = True

    def synchronize_weights(self):
        """Method to be called once the workers have started.

        Triggers a rendez-vous for the workers to receive their copy of the weights.

        This is a convenience method that delegates to the sender's synchronize_weights().
        """
        if not self._initialized_on_sender or self._sender is None:
            raise RuntimeError(
                "Must call init_on_sender() before synchronize_weights() on SharedMemWeightSyncScheme"
            )
        self._sender.synchronize_weights()

    def _get_params_map(
        self,
        context: Any = None,
        model_id: str | None = None,
        weights: TensorDictBase | None = None,
        model: nn.Module | None = None,
        params_map: dict[int, TensorDictBase] | None = None,
        devices: list[torch.device] | None = None,
        device_map_fn: Callable[[int, TensorDictBase], TensorDictBase] | None = None,
        num_workers: int | None = None,
    ):
        """Get the params_map for init_on_sender()."""
        if params_map is not None:
            # Sanity check: params_map must be a dict[int, TensorDictBase]
            # All other args must be None
            if (
                not isinstance(params_map, dict)
                or not all(isinstance(v, int) for v in params_map.keys())
                or not all(isinstance(v, TensorDictBase) for v in params_map.values())
            ):
                raise ValueError("params_map must be a dict[int, TensorDictBase]")
            if model_id is not None or weights is not None or model is not None:
                raise ValueError(
                    "model_id, weights, and model cannot be provided if params_map is provided"
                )
            if context is not None:
                raise ValueError("context cannot be provided if params_map is provided")
            if devices is not None:
                raise ValueError("devices cannot be provided if params_map is provided")
            if device_map_fn is not None:
                raise ValueError(
                    "device_map_fn cannot be provided if params_map is provided"
                )
            if num_workers is not None:
                raise ValueError(
                    "num_workers cannot be provided if params_map is provided"
                )
            return params_map
        elif context is not None:
            if devices is not None:
                raise ValueError("devices cannot be provided if context is provided")
            # Sanity check: model_id must be provided if context is provided
            # All other args must be None
            if model_id is None:
                raise ValueError("model_id must be provided if context is provided")
            if model is not None:
                raise ValueError("model cannot be provided if context is provided")
            if weights is not None:
                raise ValueError("weights cannot be provided if context is provided")
            if device_map_fn is not None:
                raise ValueError(
                    "device_map_fn cannot be provided if context is provided"
                )
            # Get device map: the devices are stored as policy_device in the collector -- other contexts will be customized later
            devices = context.policy_device
            if num_workers is not None and num_workers != len(devices):
                raise ValueError(
                    "num_workers cannot be provided if context is provided"
                )
            # Get the weights
            model = _resolve_model(context, model_id)
            weights = TensorDict.from_module(model)
        elif model is not None:
            if weights is not None:
                raise ValueError("weights cannot be provided if model is provided")
            weights = TensorDict.from_module(model)
        # To make the map, we need the list of devices, or the map fn
        if devices is not None:
            # Import _cast locally to avoid circular imports
            from torchrl.collectors.utils import _cast

            # Get the unique devices
            devices_set = set(devices)
            weights_devices = {p.device for p in weights.values(True, True)}
            if len(weights_devices) == 1:
                weights_device = weights_devices.pop()
            else:
                weights_device = None

            # Create device map with proper Parameter handling using _cast
            # _cast ensures Parameters stay as Parameters (with requires_grad=False)
            device_map = {}
            for d in devices_set:
                if d != weights_device:
                    # Move to device and apply _cast to preserve Parameter/Buffer types
                    weights_on_device = weights.to(d)
                    weights_on_device = weights_on_device.apply(_cast, weights)
                    device_map[d] = weights_on_device
                else:
                    # Already on correct device, just apply _cast
                    device_map[d] = weights.apply(_cast, weights)

            # Create the map
            params_map = {
                worker_idx: device_map[device]
                for worker_idx, device in enumerate(devices)
            }
            return params_map
        if device_map_fn is not None:
            return {
                worker_idx: device_map_fn(worker_idx, weights)
                for worker_idx in range(num_workers)
            }
        raise ValueError(
            "Either params_map, model_id + context or model/weights + devices must be provided."
        )

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        model: Any = None,
        worker_idx: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Reads from the worker's dedicated queue to receive shared weights,
        then registers them in the transport. The receiver then applies these weights
        to the model.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing model and worker_idx
            model: Model being synchronized
            worker_idx: Worker index
            **kwargs: Alternative to context (model, worker_idx, timeout, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            if hasattr(context, "get_model"):
                model = context.get_model(model_id)
            elif model is None:
                model = _resolve_model(context, model_id)
            worker_idx = getattr(context, "worker_idx", worker_idx)

        # Create receiver with the shared transport
        receiver = WeightReceiver(self)
        if context is not None:
            receiver._context_ref = weakref.ref(context)
        receiver._transport = self._shared_transport  # Use shared transport

        # Register the model
        receiver._register_model(model)

        # Store worker_idx for synchronize_weights
        receiver._worker_idx = worker_idx

        self._receiver = receiver
        self._initialized_on_worker = True

    def get_weight_queues(self):
        """Get the per-worker weight initialization queues.

        Returns:
            Dict mapping worker_idx to Queue for receiving shared weight references.

        Raises:
            RuntimeError: If init_on_sender() hasn't been called yet.
        """
        if not self._weight_init_queues:
            raise RuntimeError("Queues not created. Call init_on_sender() first.")
        return self._weight_init_queues

    def get_message_queue(self):
        """Get the general message queue for coordination.

        Returns:
            The message queue for general coordination messages.
        """
        return self._message_queue

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create shared memory transport.

        Returns the shared transport instance that all workers will use.
        Since this is shared memory, there's only one transport shared by all workers.

        Note:
            This is used internally by init_on_sender/init_on_worker.
        """
        return self._shared_transport

    def prepare_weights(
        self,
        weights: Any,
        model_id: str,
        strategy: WeightStrategy,
        context: Any = None,
    ) -> Any:
        """Prepare weights for SharedMemWeightSyncScheme.

        For SharedMemWeightSyncScheme, we prioritize using cached shared memory weights
        from the context (collector) to avoid extracting fresh (non-shared) weights.

        Args:
            weights: Raw weights input
            model_id: The model identifier
            strategy: WeightStrategy for extracting/converting weights
            context: Optional context (e.g., collector) for cache lookup

        Returns:
            Shared memory weights ready to send
        """
        # If no weights provided, check for cached shared memory weights in collector
        if weights is None and context is not None:
            if model_id == "policy" and hasattr(context, "_policy_weights_dict"):
                policy_device = (
                    context.policy_device
                    if not isinstance(context.policy_device, (list, tuple))
                    else context.policy_device[0]
                )
                cached_weights = context._policy_weights_dict.get(policy_device)
                if cached_weights is not None:
                    return cached_weights

        # Fall back to default behavior
        return super().prepare_weights(weights, model_id, strategy, context)


class NoWeightSyncScheme(WeightSyncScheme):
    """No-op weight synchronization scheme.

    This scheme disables weight synchronization entirely.
    """

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (not used)
            **kwargs: Optional parameters (not used)
        """
        # Create a no-op sender
        sender = WeightSender(self)
        sender._model_id = model_id

        self._sender = sender
        self._initialized_on_sender = True

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (not used)
            **kwargs: Optional parameters (not used)
        """
        # Create a no-op receiver
        receiver = WeightReceiver(self)
        receiver._model_ref = model_id

        self._receiver = receiver
        self._initialized_on_worker = True

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create a no-op transport.

        Note:
            This is used internally by init_on_sender/init_on_worker.
        """
        # Return a dummy transport that does nothing
        class NoOpTransport:
            def send_weights(self, weights: Any) -> None:
                pass

            def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
                return None

            def check_connection(self) -> bool:
                return True

        return NoOpTransport()


class RayWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for Ray distributed computing.

    This scheme uses Ray's object store and remote calls to synchronize weights
    across distributed workers (Ray actors).

    Each remote collector gets its own transport, following the same pattern
    as multiprocess collectors.
    """

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create Ray-based transport for a specific remote collector.

        Args:
            pipe_or_context: The Ray actor handle for the remote collector.

        Returns:
            RayTransport configured for this specific remote collector.
        """
        return RayTransport(remote_collector=pipe_or_context)

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing remote_collectors
            **kwargs: Alternative to context (remote_collectors, source_model, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            remote_collectors = getattr(context, "remote_collectors", None)
            num_workers = getattr(context, "num_workers", None) or getattr(
                context, "num_collectors", None
            )
        else:
            remote_collectors = kwargs.get("remote_collectors")
            num_workers = kwargs.get("num_workers") or kwargs.get("num_collectors")

        if remote_collectors is None:
            raise ValueError("remote_collectors must be provided via context or kwargs")
        if num_workers is None:
            num_workers = len(remote_collectors) if remote_collectors else 0

        # Create sender and register all workers (Ray actors)
        sender = WeightSender(self)
        sender._model_id = model_id

        # Register each Ray actor - _register_worker will create the transport
        for worker_idx, remote_collector in enumerate(remote_collectors):
            sender._register_worker(worker_idx, remote_collector)

        # Set context with weak reference to avoid circular refs
        if context is not None:
            sender._set_context(weakref.ref(context), model_id)

        # Store source model reference if provided for automatic weight extraction
        source_model = kwargs.get("source_model")
        if source_model is not None:
            sender._source_model = source_model

        self._sender = sender
        self._initialized_on_sender = True

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        For Ray workers, weight updates are handled via remote method calls,
        so this is typically a no-op. The receiver is created but doesn't
        need special initialization.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (typically the remote collector)
            **kwargs: Optional parameters (pipe, model, etc.)
        """
        # Create receiver
        receiver = WeightReceiver(self)

        # Register model if provided
        model = kwargs.get("model") or (
            getattr(context, "policy", None) if context else None
        )
        if model is not None:
            receiver._register_model(model)

        # Set context if provided
        if context is not None:
            receiver._set_context(weakref.ref(context))

        self._receiver = receiver
        self._initialized_on_worker = True


class RayModuleTransformScheme(WeightSyncScheme):
    """Weight synchronization for RayModuleTransform actors.

    This scheme is designed specifically for updating models hosted within
    Ray actors, such as RayModuleTransform instances. It creates a transport
    that directly calls the actor's weight update methods.

    Args:
        strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
            Default is "tensordict".
    """

    def __init__(self, strategy: str = "tensordict"):
        super().__init__(strategy)

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create RayActorTransport for the given actor.

        Args:
            pipe_or_context: Either a Ray actor reference or a context object
                from which to extract the actor reference.

        Returns:
            RayActorTransport configured with the actor reference.
        """
        actor_ref = self._extract_actor_ref(pipe_or_context)
        return RayActorTransport(actor_ref=actor_ref, update_method=self.strategy)

    def _extract_actor_ref(self, pipe_or_context: Any) -> Any:
        """Extract the Ray actor reference from the context.

        Args:
            pipe_or_context: Either a direct actor reference or an object
                with an `_actor` attribute.

        Returns:
            The Ray actor reference.
        """
        if hasattr(pipe_or_context, "_actor"):
            return pipe_or_context._actor
        return pipe_or_context

    def create_sender(self) -> RayModuleTransformSender:
        """Create a specialized sender for Ray actor communication."""
        return RayModuleTransformSender(self)

    def create_receiver(self) -> RayModuleTransformReceiver:
        """Create a specialized receiver for Ray actor communication."""
        return RayModuleTransformReceiver(self)

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing actor references
            **kwargs: Alternative to context (actors, actor_refs, source_model, etc.)
        """
        # Extract actor references from context or kwargs
        if context is not None:
            # Could be actor_refs, actors, or remote_collectors
            actor_refs = (
                getattr(context, "actor_refs", None)
                or getattr(context, "actors", None)
                or getattr(context, "remote_collectors", None)
            )
        else:
            actor_refs = (
                kwargs.get("actor_refs")
                or kwargs.get("actors")
                or kwargs.get("remote_collectors")
            )

        if actor_refs is None:
            raise ValueError(
                "actor_refs (or actors) must be provided via context or kwargs"
            )

        # Create specialized sender
        sender = self.create_sender()
        sender._model_id = model_id

        # Register all actors - _register_worker will create the transport
        for worker_idx, actor_ref in enumerate(actor_refs):
            sender._register_worker(worker_idx, actor_ref)

        # Set context with weak reference
        if context is not None:
            sender._set_context(weakref.ref(context), model_id)

        # Store source model if provided
        source_model = kwargs.get("source_model")
        if source_model is not None:
            sender._source_model = source_model

        self._sender = sender
        self._initialized_on_sender = True

    def init_on_worker(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (typically the actor itself)
            **kwargs: Optional parameters (actor_ref, model, etc.)
        """
        # Create specialized receiver
        receiver = self.create_receiver()

        # Extract actor reference if needed
        actor_ref = kwargs.get("actor_ref") or context
        if actor_ref is not None:
            # Register the transport for this actor
            transport = self.create_transport(actor_ref)
            receiver._register_worker_transport(transport)

        # Register model if provided
        model = kwargs.get("model") or (
            getattr(context, "_actor_module", None) or getattr(context, "module", None)
            if context
            else None
        )
        if model is not None:
            receiver._register_model(model)

        # Set context if provided
        if context is not None:
            receiver._set_context(weakref.ref(context))

        self._receiver = receiver
        self._initialized_on_worker = True


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


# ============================================================================
# Helper Functions
# ============================================================================


def _resolve_model(context: Any, model_id: str) -> Any:
    """Resolve model_id like 'policy' or 'env.value_net' to actual object.

    Also processes getitem notation like 'env.transform[0]' to actual object.

    Args:
        context: The context object (collector or inner_collector).
        model_id: A string address like "policy" or "env.value_net".

    Returns:
        The object at the specified address.

    Examples:
        _resolve_model(collector, "policy")  # -> collector.policy
        _resolve_model(collector, "env.value_net")  # -> collector.env.value_net
    """
    parts = model_id.split(".")
    obj = context
    for i, part in enumerate(parts):
        if "[" in part:
            key, *indices = part.split("[")
            indices = [int(index[:-1]) for index in indices]
            try:
                obj = getattr(obj, key)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {key} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                )
            for index in indices:
                obj = obj[index]
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {part} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                )
    return obj
