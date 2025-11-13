# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import warnings
import weakref
from collections.abc import Iterator
from typing import Any, Literal, Protocol

from tensordict import TensorDict, TensorDictBase

from torch import nn

__all__ = [
    "TransportBackend",
    "WeightStrategy",
    "WeightSender",
    "WeightReceiver",
    "WeightSyncScheme",
]

from torchrl.weight_update.utils import _resolve_model


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
        if extract_as == "state_dict":
            warnings.warn(
                "state_dict strategy is experimental. Use tensordict strategy for safer weight updates.",
                UserWarning,
            )
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
        elif self.extract_as == "state_dict":  # state_dict
            # Extract as state_dict
            if isinstance(source, nn.Module):
                return source.state_dict()
            elif isinstance(source, dict):
                return source
            elif isinstance(source, TensorDictBase):
                # Convert TensorDict to state_dict
                return source.flatten_keys().to_dict()
            else:
                raise ValueError(
                    f"Unsupported source type for state_dict extraction: {type(source)}"
                )
        else:
            raise ValueError(
                f"Unknown extract_as: {self.extract_as}. Must be 'tensordict' or 'state_dict'."
            )

    def apply_weights(
        self, destination: Any, weights: Any, inplace: bool = True
    ) -> None:
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
            inplace: Whether to apply weights in place.
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
            if not inplace:
                weights.to_module(destination)
                return
            else:
                destination = TensorDict.from_module(destination)
        elif isinstance(destination, dict):
            if not inplace:
                raise ValueError("Cannot update state_dict out of place")
            destination = TensorDict(destination)
            if any(isinstance(key, str) and "." in key for key in destination.keys()):
                destination = destination.unflatten_keys(".")

        if not isinstance(weights, TensorDictBase) or not isinstance(
            destination, TensorDictBase
        ):
            raise ValueError(
                f"Unsupported weights or destination type: {type(weights)=} or {type(destination)=}. Expected TensorDictBase."
            )
            # Apply TensorDict format
        try:
            if not inplace:
                destination.update(weights)
            else:
                destination.data.update_(weights.data)
        except Exception as e:
            raise KeyError(
                f"Error updating destination. Destination keys: {destination.keys(True, True)}, weights keys: {weights.keys(True, True)}"
            ) from e
        return


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

        context = self._context_ref() if self._context_ref is not None else None

        # Let the scheme prepare the weights
        prepared_weights = self._scheme.prepare_weights(
            weights=weights,
            model_id=self._model_id,
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

        context = self._context_ref() if self._context_ref is not None else None

        # Let the scheme prepare the weights
        prepared_weights = self._scheme.prepare_weights(
            weights=weights,
            model_id=self._model_id,
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
        the initial weights. For SharedMemTransport, this sends buffer
        references via queues. For MultiProcessWeightSyncScheme (MPTransport),
        this extracts and sends initial weights via pipes.

        This is different from send() which is called during training to
        update weights.
        """
        # For other schemes (SharedMemWeightSyncScheme, etc.), use transport's method
        for transport in self._iterate_transports():
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
        # For other transports (MPTransport, etc.), weights is None and synchronization
        # happens later via receive(), so this is a no-op
        if weights is not None:
            if self._model_ref is not None:
                model = self._resolve_model_ref()
                self._strategy.apply_weights(model, weights, inplace=False)
            else:
                raise ValueError("Received weights but no model registered")

    def apply_weights(self, weights: Any, inplace: bool = True) -> None:
        """Apply received weights to registered model.

        Args:
            weights: The weights to apply.
            inplace: Whether to apply weights in place. Default is `True`.

        Note:
            Convenience method. Normally weights are received and applied via receive() in the worker loop.
        """
        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self._resolve_model_ref()
        self._strategy.apply_weights(model, weights, inplace=inplace)

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


# ============================================================================
# Weight Synchronization Schemes
# ============================================================================


class WeightSyncScheme(metaclass=abc.ABCMeta):
    """Configuration for how to synchronize ONE model across workers.

    A scheme manages synchronization of ONE model across workers.
    The collector maintains a dict of {model_id: scheme} pairs.
    """

    def __init__(self, strategy: Literal["state_dict", "tensordict"] = "tensordict"):
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
