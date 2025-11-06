# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import weakref
from collections.abc import Iterator
from typing import Any, Literal, Protocol

from tensordict import TensorDict, TensorDictBase

from torch import nn

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

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Send weights to the receiver."""
        ...

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Receive weights from the sender. Returns (model_id, weights) or None if timeout."""
        ...

    def check_connection(self) -> bool:
        """Check if the connection is still alive."""
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

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Send weights through the pipe.

        Sends weights and waits for acknowledgment to ensure delivery.
        """
        self.send_weights_async(model_id, weights)
        self.wait_ack()

    def send_weights_async(self, model_id: str, weights: Any) -> None:
        """Send weights through the pipe without waiting for acknowledgment.

        Use wait_ack() to wait for acknowledgment after sending to all workers.
        """
        self.pipe.send(((model_id, weights), "update_weights"))

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
        """
        if self.pipe.poll(timeout):
            data_in, msg = self.pipe.recv()
            if msg == "update_weights":
                model_id, weights = data_in
                return model_id, weights
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


class SharedMemTransport:
    """Shared memory transport for in-place weight updates.

    This transport updates shared memory tensors directly without message passing.
    Workers automatically see weight updates without explicit communication.

    The transport supports lazy registration with pipe-based buffer distribution:
    - On first weight send for a model, creates shared memory and sends buffer via pipes
    - Workers receive the buffer reference and update their local references
    - Subsequent updates are pure in-place shared memory (zero-copy)

    This hybrid approach solves the chicken-and-egg problem: workers can start before
    weights are available, and they'll receive the shared buffer references when ready.

    Args:
        policy_weights: Dictionary mapping model_id to shared TensorDict weights.
            Can be empty if using lazy registration.
        auto_register: Whether to automatically register models on first weight send.
            Default is True. Set to `False` to require explicit registration via
            register_weights().
    """

    def __init__(
        self,
        policy_weights: dict[str, TensorDictBase] | None = None,
        auto_register: bool = True,
    ):
        self._policy_weights = policy_weights if policy_weights is not None else {}
        self._auto_register = auto_register
        self._pipes = []  # List of pipes to send initial buffer references
        # Track which model_ids have been sent to workers
        self._registered_with_workers = set()

    def register_pipe(self, pipe: Any) -> None:
        """Register a pipe for sending buffer references on first weight send.

        Args:
            pipe: Pipe connection to a worker process.
        """
        if pipe not in self._pipes:
            self._pipes.append(pipe)

    def register_weights(self, model_id: str, weights: TensorDictBase) -> None:
        """Register a shared memory weights TensorDict for a model.

        This method allows explicit registration of shared weights. It's optional
        when auto_register=True (the default), but required when auto_register=False.

        If pipes are registered and this model hasn't been sent to workers yet,
        this will trigger sending the buffer reference to all workers. If pipes
        aren't registered yet, weights are stored and will be sent when pipes
        become available (during init_on_sender).
        """
        if not isinstance(weights, TensorDictBase):
            raise ValueError(f"Weights must be a TensorDictBase, got {type(weights)}")

        is_new_registration = model_id not in self._policy_weights
        if is_new_registration:
            self._policy_weights[model_id] = weights
        else:
            raise RuntimeError("Re-registering weights is not supported.")

        # If this is a new registration and we have pipes, send buffer to workers
        # If pipes aren't available yet, defer sending until init_on_sender is called
        if self._pipes:
            if model_id not in self._registered_with_workers:
                self._send_buffer_to_workers(model_id, weights)
            else:
                raise RuntimeError(
                    f"Model '{model_id}' has already been registered with workers."
                )

    def _send_buffer_to_workers(self, model_id: str, buffer: TensorDictBase) -> None:
        """Send shared memory buffer reference to all workers via pipes.

        This is called once per model_id when lazy registration occurs.
        Workers receive the buffer and update their local references.

        Note: We send buffer.data to avoid gradient tracking issues when crossing
        process boundaries. The .data attribute gives us the underlying tensors
        without autograd metadata.
        """
        for pipe in self._pipes:
            # Send special registration message with the shared buffer
            # Use .data to strip gradient information (can't serialize non-leaf tensors with requires_grad)
            pipe.send(((model_id, buffer.data), "register_shared_weights"))

        # Wait for acknowledgments from all workers
        for pipe in self._pipes:
            _, msg = pipe.recv()
            if msg != "registered":
                raise RuntimeError(f"Expected 'registered' acknowledgment, got '{msg}'")

        self._registered_with_workers.add(model_id)

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Update weights in-place in shared memory.

        If the model is not registered and auto_register=True, it will be automatically
        registered by creating a shared memory copy of the provided weights. The shared
        buffer reference is sent to all workers via pipes on first registration, then
        subsequent updates are pure in-place shared memory.

        Args:
            model_id: Identifier for the model whose weights to update.
            weights: New weights to send. Can be a TensorDictBase or dict.

        Raises:
            KeyError: If model is not registered and auto_register=False.
            ValueError: If weights type is unsupported for auto-registration.
        """
        if model_id not in self._policy_weights:
            if not self._auto_register:
                raise KeyError(
                    f"Model '{model_id}' not registered in SharedMemTransport. "
                    f"Available models: {list(self._policy_weights.keys())}. "
                    f"Either register the model using register_weights() or enable auto_register."
                )

            # Auto-register on first send
            if isinstance(weights, dict):
                weights = TensorDict(weights)
            if not isinstance(weights, TensorDictBase):
                raise ValueError(
                    f"Cannot auto-register model '{model_id}' with weights type: {type(weights)}. "
                    f"Supported types for auto-registration: TensorDictBase, dict. "
                    f"Please manually register shared weights using register_weights()."
                )
            # Unflatten keys if they're flat (e.g., 'module.0.weight' -> nested structure)
            # This is necessary for to_module() to work properly
            weights_to_share = weights
            # Check if keys are flattened by looking for dots in key names
            if any("." in key for key in weights_to_share.keys()):
                weights_to_share = weights_to_share.unflatten_keys(".")
            shared_buffer = weights_to_share.share_memory_()

            self._policy_weights[model_id] = shared_buffer

            # Send buffer reference to all workers if we have pipes
            if self._pipes and model_id not in self._registered_with_workers:
                self._send_buffer_to_workers(model_id, shared_buffer)

        shared_weights = self._policy_weights[model_id]

        # Update shared memory in-place (workers see this automatically)
        if isinstance(weights, dict):
            weights = TensorDict(weights)
        if not isinstance(weights, TensorDictBase):
            raise ValueError(f"Unsupported weights type: {type(weights)}")
        # Unflatten if needed to match shared buffer structure
        weights_to_update = weights
        if any("." in key for key in weights.keys()):
            weights_to_update = weights.unflatten_keys(".")
        shared_weights.data.update_(weights_to_update.data)

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

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Send weights to the remote collector via Ray.

        Note: We don't pass model_id to the remote collector because remote
        collectors don't have weight senders - they apply weights directly to
        their local policy.
        """
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

    def send_weights_async(self, model_id: str, weights: Any) -> None:
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

    def send_weights(self, model_id: str, weights: Any) -> None:
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

    def send_weights_async(self, model_id: str, weights: Any) -> None:
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

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Send weights to the remote collector via RPC.

        Note: We don't pass model_id to the remote collector because remote
        collectors don't have weight senders - they apply weights directly to
        their local policy.
        """
        if self._collector_info is None or self._collector_rref is None:
            return

        from torch.distributed import rpc

        # Send weights to the remote collector and wait for completion
        rpc.rpc_sync(
            self._collector_info,
            self._collector_class.update_policy_weights_,
            args=(self._collector_rref, weights),
        )

    def send_weights_async(self, model_id: str, weights: Any) -> None:
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

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Send weights to the distributed worker.

        Note: We don't pass model_id to the remote collector because remote
        collectors don't have weight senders - they apply weights directly to
        their local policy.
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

        # Wait for acknowledgment
        status = self._store.get(f"NODE_{self._rank}_out")
        if status != b"updated":
            raise RuntimeError(f"Expected 'updated' but got status {status}.")
        self._store.delete_key(f"NODE_{self._rank}_out")

    def send_weights_async(self, model_id: str, weights: Any) -> None:
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
        import torch.distributed

        return torch.distributed.is_initialized()


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
            destination = TensorDict.from_module(destination)
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
                transport.send_weights_async(model_id, prepared_weights)
            else:
                # Fallback for transports that don't support async send
                transport.send_weights(model_id, prepared_weights)

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
                transport.send_weights_async(model_id, prepared_weights)
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

    # Legacy method - kept for backward compatibility
    def update_weights(self, weights: Any) -> None:
        """Send weights to ALL workers for this model (legacy).

        Args:
            weights: Weights to send (can be None, nn.Module, TensorDict, etc.).

        Note:
            This is the legacy method. Use send() instead.
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

    def apply_weights(self, weights: Any) -> None:
        """Apply received weights to registered model (legacy).

        Args:
            weights: The weights to apply.

        Note:
            This is the legacy method. Use receive() in the worker loop instead.
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
            model_id = getattr(self, "_model_id", "policy")
            self._single_transport.send_weights(model_id, weights)

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

    # Legacy methods - kept for backward compatibility
    @abc.abstractmethod
    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create transport for communication.

        Args:
            pipe_or_context: Either a pipe connection or context object to extract pipe from.

        Returns:
            A transport backend instance.
        """
        ...

    def create_sender(self) -> WeightSender:
        """Create a sender for this scheme (legacy).

        Returns:
            WeightSender instance configured for this scheme.
        """
        return WeightSender(self)

    def create_receiver(self) -> WeightReceiver:
        """Create a receiver for this scheme (legacy).

        Returns:
            WeightReceiver instance configured for this scheme.
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
        """Create an MPTransport using the provided pipe (legacy)."""
        return MPTransport(pipe)


class SharedMemWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization using shared memory.

    This scheme mimics the old WeightUpdater behavior by using shared memory
    for in-place weight updates. Workers automatically see weight updates
    without explicit message passing.

    By default, this scheme uses lazy registration: models are automatically
    registered on the first weight send. This makes it seamless to use with
    configuration systems like Hydra where schemes are created before models
    are available.

    Args:
        policy_weights: Dictionary mapping model_id to shared TensorDict weights.
            Can be empty if using lazy registration (auto_register=True).
        strategy: The weight transmission strategy (default: "tensordict").
        auto_register: Whether to automatically register models on first weight send.
            Default is True. Set to False to require explicit registration via
            register_shared_weights().

    Example:
        >>> # With auto-registration (default) - works with Hydra configs
        >>> scheme = SharedMemWeightSyncScheme()
        >>> # Models are auto-registered on first weight send

        >>> # With explicit registration
        >>> scheme = SharedMemWeightSyncScheme(auto_register=False)
        >>> shared_weights = TensorDict.from_module(model).share_memory_()
        >>> scheme.register_shared_weights("policy", shared_weights)
    """

    def __init__(
        self,
        policy_weights: dict[str, TensorDictBase] | None = None,
        strategy: str = "tensordict",
        auto_register: bool = True,
    ):
        super().__init__(strategy)
        self.policy_weights = policy_weights if policy_weights is not None else {}
        self.auto_register = auto_register
        # Create a single shared transport for all workers
        self._shared_transport = SharedMemTransport(
            self.policy_weights, auto_register=auto_register
        )

    def register_shared_weights(self, model_id: str, weights: TensorDictBase) -> None:
        """Register shared memory weights for a model.

        This method allows explicit registration of shared weights. It's optional
        when auto_register=True (the default), but required when auto_register=False.

        Args:
            model_id: Identifier for the model.
            weights: Shared memory TensorDict containing the model's weights.
        """
        # Don't set self.policy_weights[model_id] here - register_weights does that
        # (self.policy_weights and transport._policy_weights are the same dict)
        self._shared_transport.register_weights(model_id, weights)

    def init_on_sender(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        For SharedMemWeightSyncScheme, this handles:
        1. Getting cached shared memory weights from context
        2. Pre-registering the weights with the transport
        3. Distributing buffer references to all workers (avoiding later deadlock)

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing pipes, cached_weights
            **kwargs: Alternative to context (pipes, cached_weights, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            pipes = getattr(context, "pipes", None)
            num_workers = getattr(context, "num_workers", None)
            # Try to get cached shared memory weights
            if hasattr(context, "get_cached_weights"):
                cached_weights = context.get_cached_weights(model_id)
            else:
                cached_weights = None
        else:
            pipes = kwargs.get("pipes")
            num_workers = kwargs.get("num_workers")
            cached_weights = kwargs.get("cached_weights")

        if pipes is None:
            raise ValueError("pipes must be provided via context or kwargs")
        if num_workers is None:
            num_workers = len(pipes) if pipes else 0

        # Register pipes with shared transport for lazy buffer distribution
        for pipe in pipes:
            self._shared_transport.register_pipe(pipe)

        # If we have cached shared memory weights, pre-register them
        if cached_weights is not None:
            # Check if already registered to avoid re-registration error
            if model_id not in self.policy_weights:
                self.register_shared_weights(model_id, cached_weights)

        # Send buffer references for any weights that were pre-registered
        # before pipes were available (e.g., via explicit register_shared_weights call)
        if model_id in self.policy_weights:
            if model_id not in self._shared_transport._registered_with_workers:
                self._shared_transport._send_buffer_to_workers(
                    model_id, self.policy_weights[model_id]
                )

        # Create sender with the shared transport
        sender = WeightSender(self)
        sender._model_id = model_id
        sender._transport = self._shared_transport  # Use shared transport
        if context is not None:
            sender._context_ref = weakref.ref(context)

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
            getattr(context, "pipe", None)
            if hasattr(context, "get_model"):
                model = context.get_model(model_id)
            else:
                model = None
        else:
            model = kwargs.get("model")

        # For shared memory, we don't need the pipe in the receiver
        # The transport is shared and workers see updates automatically

        # Create receiver with the shared transport
        receiver = WeightReceiver(self)
        if context is not None:
            receiver._context_ref = weakref.ref(context)
        receiver._transport = self._shared_transport  # Use shared transport
        if model is not None:
            receiver._register_model(model)
        else:
            # Register by model_id for later resolution
            receiver._register_model(model_id)

        self._receiver = receiver
        self._initialized_on_worker = True

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create shared memory transport and register pipe for lazy buffer distribution (legacy).

        For lazy registration to work, we register each worker's pipe with the transport.
        On first weight send, the transport will send buffer references via these pipes.

        Returns the shared transport instance that all workers will use.
        Since this is shared memory, there's only one transport shared by all workers.
        """
        # Register the pipe for lazy buffer distribution
        if pipe_or_context is not None:
            self._shared_transport.register_pipe(pipe_or_context)
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
        """Returns None as no transport is needed (legacy)."""
        # Return a dummy transport that does nothing
        class NoOpTransport:
            def send_weights(self, model_id: str, weights: Any) -> None:
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

        # Create transports for each Ray actor and register them
        for worker_idx, remote_collector in enumerate(remote_collectors):
            transport = self.create_transport(remote_collector)
            sender._register_worker(worker_idx, transport)

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

        # Register all actors as workers
        for worker_idx, actor_ref in enumerate(actor_refs):
            transport = self.create_transport(actor_ref)
            sender._register_worker(worker_idx, transport)

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
