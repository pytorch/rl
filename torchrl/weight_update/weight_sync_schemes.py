# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Literal, overload, Protocol

import torch

from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl._utils import logger as torchrl_logger

__all__ = [
    "TransportBackend",
    "WeightStrategy",
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

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> tuple[str, Any] | None:
        """Receive weights from the sender and apply them to the model.

        Args:
            timeout: Maximum time to wait for weights (seconds).
                     None means no timeout (blocking). Some transports may not
                     support timeout and will raise ValueError if specified.
            weights: Pre-allocated weight buffer to receive into.
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

        Returns:
            Tuple of (model_id, weights) if weights were received, None if timeout.
        """
        ...

    def check_connection(self) -> bool:
        """Check if the connection is still alive."""
        ...

    def setup_connection_and_weights_on_sender(self) -> None:
        """Synchronize weights on sender side before collection starts.

        This is called once after workers are initialized to send the initial
        weights. This can be a no-op (weights are sent via
        send_weights).
        """
        ...

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> Any:
        """Synchronize weights on worker side before collection starts.

        This is called once in each worker after initialization to receive
        the initial weights. This is a no-op (weights are received via
        receive_weights).

        Args:
            worker_idx: The worker index.
            weights: Pre-allocated weight buffer to receive into.
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

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

    def extract_weights(self, source: Any) -> TensorDictBase | dict | None:
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
                torchrl_logger.warning(
                    f"Unsupported source type for TensorDict extraction: {type(source)}"
                )
                return TensorDict(lock=True)
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
                torchrl_logger.warning(
                    f"Unsupported source type for TensorDict extraction: {type(source)}"
                )
                return {}
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

        if not isinstance(weights, TensorDictBase):
            raise ValueError(
                f"Unsupported weights type: {type(weights)}. Must be dict or TensorDictBase."
            )
        if not isinstance(destination, TensorDictBase):
            if not weights.is_empty():
                raise ValueError(
                    "Non-empty weights are associated with a non-dict, non-td, non-Module destination."
                )
            return

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
# Weight Synchronization Schemes
# ============================================================================


class WeightSyncScheme(metaclass=abc.ABCMeta):
    """Configuration for how to synchronize ONE model across workers.

    A scheme manages synchronization of ONE model across workers.
    The collector maintains a dict of {model_id: scheme} pairs.

    This class directly handles both sender and receiver functionality,
    with behavior determined by whether init_on_sender() or init_on_receiver()
    was called.
    """

    _model_id: str | None = None

    # Transport management
    _sender_transports: dict[int, TransportBackend] | None
    _receiver_transport: TransportBackend | None
    _shared_transport: TransportBackend | None

    # Context and model references
    _context_ref: weakref.ReferenceType[Any] | None
    _model_ref: weakref.ReferenceType[Any] | None

    # Strategy
    _strategy: WeightStrategy

    # Async state
    _pending_async: bool
    _pending_transports: list[TransportBackend] | None

    # Worker index (for receiver side)
    _worker_idx: int | None

    def __init__(self, strategy: Literal["state_dict", "tensordict"] = "tensordict"):
        self.strategy_str = strategy
        self._strategy = _get_strategy(strategy)
        self._initialized_on_sender = False
        self._initialized_on_receiver = False

        # Transport management
        self._sender_transports = None  # worker_idx -> transport
        self._receiver_transport = None
        self._shared_transport = None

        # Context and model references
        self._context_ref = None
        self._model_ref = None

        # Async state
        self._pending_async = False
        self._pending_transports = None

        # Worker index
        self._worker_idx = None

    # ========================================================================
    # Initialization
    # ========================================================================

    @property
    def strategy(self) -> WeightStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: WeightStrategy) -> None:
        self._strategy = value

    @overload
    def init_on_sender(
        self,
        *,
        model_id: str,
        context: Any,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        params_map: dict[int, TensorDictBase],
        model_id: str | None = None,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        params_map: dict[int, TensorDictBase],
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        weights: TensorDictBase,
        devices: list[torch.device],
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        weights: TensorDictBase,
        devices: list[torch.device],
        model_id: str | None = None,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        model: nn.Module,
        devices: list[torch.device],
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        model: nn.Module,
        devices: list[torch.device],
        model_id: str | None = None,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        weights: TensorDictBase,
        device_map_fn: Callable[[int, TensorDictBase], TensorDictBase],
        num_workers: int,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        *,
        model: nn.Module,
        device_map_fn: Callable[[int, TensorDictBase], TensorDictBase],
        num_workers: int,
        model_id: str | None = None,
    ) -> None:
        ...

    @overload
    def init_on_sender(self):
        ...

    def init_on_sender(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        This method is called once in the collector's _run_processes() method,
        after workers have been started and are ready to receive messages.
        """
        self._initialized_on_sender = True
        try:
            result = self._init_on_sender_impl(*args, **kwargs)
        except Exception:
            self._initialized_on_sender = False
            raise
        return result

    def _init_on_sender_impl(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def initialized_on_sender(self):
        return getattr(self, "_initialized_on_sender", False)

    @property
    def initialized_on_receiver(self):
        return getattr(self, "_initialized_on_receiver", False)

    @overload
    def init_on_receiver(
        self,
        model_id: str,
        context: Any,
        **kwargs,
    ) -> None:
        ...

    @overload
    def init_on_receiver(
        self,
        model_id: str,
        context: None = None,
        *,
        worker_idx: int = ...,
        model: Any | None = None,
        **kwargs,
    ) -> None:
        ...

    def init_on_receiver(
        self,
        *,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        This method is called once in each worker's initialization.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (e.g., inner collector)
            **kwargs: Alternative to context (model, etc.)
        """
        self._initialized_on_receiver = True
        try:
            result = self._init_on_receiver_impl(
                model_id=model_id, context=context, **kwargs
            )
        except Exception:
            self._initialized_on_receiver = False
            raise
        return result

    def _init_on_receiver_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    # ========================================================================
    # Context and Model Management
    # ========================================================================

    def _set_context(self, context: Any) -> None:
        """Set the context object (collector) for model resolution (internal).

        Args:
            context: The collector instance.
        """
        self._context_ref = weakref.ref(context)

    def _set_model(self, model: Any) -> None:
        """Set the model object for applying weights (internal).

        Args:
            model: The model object to apply weights to.
        """
        self._model_ref = weakref.ref(model)

    @property
    def context(self) -> Any | None:
        """Get the context object (e.g., collector), if available.

        Returns:
            The context object if available, None otherwise.
        """
        if self._context_ref is not None:
            return self._context_ref()
        return None

    @context.setter
    def context(self, context: Any) -> None:
        """Set the context object for resolving references.

        Args:
            context: The context object to resolve references from.
        """
        if context is not None:
            self._context_ref = weakref.ref(context)
        else:
            self._context_ref = None

    @property
    def model_id(self) -> str | None:
        """Get the model ID for this scheme.

        Returns:
            The model ID if set, None otherwise.
        """
        return self._model_id

    @model_id.setter
    def model_id(self, model_id: str) -> None:
        """Set the model ID for this scheme.

        Args:
            model_id: The model ID to set.
        """
        self._model_id = model_id

    @property
    def worker_idx(self) -> int | None:
        """Get the worker index for this scheme.

        Returns:
            The worker index if set, None otherwise.
        """
        return self._worker_idx

    @worker_idx.setter
    def worker_idx(self, worker_idx: int | None) -> None:
        """Set the worker index for this scheme.

        Args:
            worker_idx: The worker index to set.
        """
        if self.initialized_on_sender and worker_idx is not None:
            raise RuntimeError(
                "Worker index cannot be set after initialization on sender"
            )
        self._worker_idx = worker_idx

    @property
    def model(self) -> Any | None:
        """Get the model object, if available.

        Returns:
            The model object if available, None otherwise.
        """
        if self._model_ref is not None:
            return self._model_ref()
        if self._model_id is not None:
            model = _resolve_model(self.context, self._model_id)
            if model is None:
                raise AttributeError(
                    f"Model {self._model_id} was `None` in context {self.context}"
                )
            self._model_ref = weakref.ref(model)
            return model

    @model.setter
    def model(self, model: Any) -> None:
        """Set the model object for applying weights.

        Args:
            model: The model object to apply weights to.
        """
        if model is not None:
            self._model_ref = weakref.ref(model)
        else:
            self._model_ref = None

    @property
    def weights(self) -> Any | None:
        """Get the current weights, if available.

        Returns:
            The weights as TensorDict if available, None otherwise.
        """
        if (weights := getattr(self, "_weights", None)) is not None:
            return weights
        model = self.model
        if model is not None:
            return self._strategy.extract_weights(model)
        return None

    @weights.setter
    def weights(self, value: Any):
        self._weights = value

    def _get_weights_buffer_from_model(self, model: nn.Module | Any) -> TensorDictBase:
        from torchrl.collectors.utils import _cast

        if isinstance(model, torch.nn.Module):
            td = TensorDict.from_module(model)
            td = td.data.apply(_cast, td)
            return td
        # Return an empty TD
        return TensorDict()

    # ========================================================================
    # Transport Management
    # ========================================================================

    def _register_worker_sender(
        self,
        *,
        worker_idx: int,
        transport: TransportBackend | None = None,
        **transport_kwargs,
    ) -> None:
        """Register a worker's communication.

        Args:
            worker_idx: The worker index.
            transport: Optional pre-created transport.
            **transport_kwargs: Transport-specific configuration.
        """
        if self._sender_transports is None:
            if self._shared_transport is not None:
                raise RuntimeError(
                    "Cannot register transports on sender after shared transport is set"
                )
            self._sender_transports = {}
        if worker_idx not in self._sender_transports:
            if transport is not None:
                self._sender_transports[worker_idx] = transport
            else:
                self._sender_transports[worker_idx] = self.create_transport(
                    **transport_kwargs
                )

    def _register_transport_receiver(
        self, transport: TransportBackend | None = None, **transport_kwargs
    ) -> None:
        """Register a single transport (for receiver side).

        Args:
            transport: Optional pre-created transport.
            **transport_kwargs: Transport-specific configuration.
        """
        if transport is not None:
            self._receiver_transport = transport
        else:
            self._receiver_transport = self.create_transport(**transport_kwargs)

    def _iterate_transports(
        self, worker_ids: int | list[int] | None = None
    ) -> Iterator[TransportBackend]:
        """Iterate over transports for specified workers."""
        if worker_ids is None:
            # All workers
            if not self.sender_transports:
                if self.receiver_transport is not None:
                    yield self.receiver_transport
            else:
                # Make sure transports are sorted
                for k in sorted(self.sender_transports.keys()):
                    yield self.sender_transports[k]
        else:
            # Specific workers
            if isinstance(worker_ids, int):
                worker_ids = [worker_ids]
            for worker_id in worker_ids:
                if worker_id in self.sender_transports:
                    yield self.sender_transports[worker_id]
                else:
                    raise ValueError(f"Worker {worker_id} not registered")

    @abc.abstractmethod
    def create_transport(self, **kwargs) -> TransportBackend:
        """Create transport for communication.

        Args:
            **kwargs: Transport-specific configuration parameters.

        Returns:
            A transport backend instance.

        Note:
            This is used internally by init_on_sender/init_on_receiver.
        """
        ...

    @property
    def sender_transports(self) -> dict[int, TransportBackend]:
        """Get the sender transports.

        Returns:
            The sender transports.
        """
        if self._shared_transport is not None:
            return defaultdict(lambda: self._shared_transport)
        return self._sender_transports

    @property
    def receiver_transport(self) -> TransportBackend | None:
        """Get the receiver transport.

        Returns:
            The receiver transport.
        """
        if self._shared_transport is not None:
            return self._shared_transport
        return self._receiver_transport

    @property
    def shared_transport(self) -> TransportBackend | None:
        """Get the shared transport.

        Returns:
            The shared transport.
        """
        if self._receiver_transport is not None:
            raise RuntimeError(
                "Receiver transport and shared transport cannot be used together"
            )
        if self._sender_transports is not None:
            raise RuntimeError(
                "Sender transports and shared transport cannot be used together"
            )
        return self._shared_transport

    @shared_transport.setter
    def shared_transport(self, shared_transport: TransportBackend | None) -> None:
        """Set the shared transport.

        Args:
            shared_transport: The shared transport to set.
        """
        self._shared_transport = shared_transport

    # ========================================================================
    # Sending Weights (Sender Side)
    # ========================================================================

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
        if not self.initialized_on_sender:
            raise RuntimeError("Must be initialized on sender before sending weights")
        if not self.synchronized_on_sender:
            raise RuntimeError("Must be synchronized on sender before sending weights")

        if self._pending_async:
            raise RuntimeError(
                "Cannot call send() while an async send is pending. Call wait_async() first."
            )

        context = self.context

        # Let the scheme prepare the weights
        torchrl_logger.debug("Preparing weights")
        prepared_weights = self.prepare_weights(
            weights=weights,
            model_id=self._model_id,
            strategy=self._strategy,
            context=context,
        )

        transports = list(self._iterate_transports(worker_ids))

        if not transports:
            raise RuntimeError("No transports available.")

        # Send to all workers first (non-blocking if transport supports it)
        torchrl_logger.debug(f"Sending over transports {transports}")
        for transport in transports:
            if hasattr(transport, "send_weights_async"):
                torchrl_logger.debug(
                    f"Sending {type(prepared_weights)=} through {type(transport)=} asynchronously."
                )
                transport.send_weights_async(prepared_weights)
            else:
                # Fallback for transports that don't support async send
                torchrl_logger.debug(
                    f"Sending {type(prepared_weights)=} through {type(transport)=} synchronously."
                )
                transport.send_weights(prepared_weights)

        # Wait for all acknowledgments
        torchrl_logger.debug("Waiting for acknowledgement")
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
        if not self.initialized_on_sender:
            raise RuntimeError("Must be initialized on sender before sending weights")

        if self._pending_async:
            raise RuntimeError(
                "Cannot call send_async() again while a previous send is pending. Call wait_async() first."
            )

        context = self.context

        # Let the scheme prepare the weights
        prepared_weights = self.prepare_weights(
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

    def update_weights(self, weights: Any) -> None:
        """Send weights to ALL workers for this model.

        Args:
            weights: Weights to send (can be None, nn.Module, TensorDict, etc.).

        Note:
            Convenience method that calls send(weights=weights).
        """
        self.send(weights=weights)

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

    # ========================================================================
    # Receiving Weights (Receiver Side)
    # ========================================================================

    def receive(self, timeout: float | None = None) -> TensorDictBase | None:
        """Check for and apply new weights (non-blocking).

        This method is called in the worker's main loop to check if
        new weights have been sent. If weights are available, they
        are applied to the registered model immediately, and the update
        is cascaded to any sub-collectors via context.update_policy_weights_().

        Args:
            timeout: Maximum time to wait for weights (seconds).
                     None means no timeout (blocking). Some transports may not
                     support timeout and will raise ValueError if specified.

        Returns:
            The received weights if available, None otherwise.

        Note: For SharedMemWeightSyncScheme, this always returns None
        since workers automatically see updates via shared memory.
        """
        if not self.initialized_on_receiver:
            raise RuntimeError(
                "Must be initialized on receiver before receiving weights"
            )
        if not self.synchronized_on_receiver:
            raise RuntimeError(
                "Must be synchronized on receiver before receiving weights"
            )

        if self._receiver_transport is None:
            return None

        # Try to receive weights - transport handles receiving and applying
        torchrl_logger.debug(
            f"Calling receive_weights on transport {self.receiver_transport}"
        )
        result = self.receiver_transport.receive_weights(
            timeout=timeout,
            weights=self.weights,
            model=self.model,
            strategy=self._strategy,
        )
        if result is None:
            return None

        model_id, weights = result
        torchrl_logger.debug(f"Received weights for {model_id=}")

        # Cascade weight update to sub-collectors if context supports it
        if self.context is not None and hasattr(self.context, "update_policy_weights_"):
            torchrl_logger.debug(
                f"Cascading weight update to sub-collectors for {model_id=}"
            )
            self.context.update_policy_weights_(
                model_id=model_id, policy_or_weights=weights
            )

        # Send acknowledgment if transport supports it
        if hasattr(self.receiver_transport, "send_ack"):
            torchrl_logger.debug(f"Sending acknowledgement on {model_id=}")
            self.receiver_transport.send_ack("updated")

        return weights

    def apply_weights(self, weights: TensorDictBase, inplace: bool = True) -> None:
        """Apply weights to the model.

        Args:
            weights: The weights to apply.
            inplace: Whether to apply weights in place. Default is `True`.
        """
        if not self.initialized_on_receiver:
            if self.initialized_on_sender:
                raise RuntimeError("apply_weights() called on a sender side.")
            raise RuntimeError(
                "apply_weights() called before init_on_receiver has been called."
            )

        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self.model
        self._strategy.apply_weights(model, weights, inplace=inplace)

        # Send acknowledgment if transport supports it
        if self.receiver_transport is not None and hasattr(
            self.receiver_transport, "send_ack"
        ):
            self.receiver_transport.send_ack("updated")

    # ========================================================================
    # Synchronization
    # ========================================================================

    def is_sender(self):
        """Check if the current worker is the sender."""
        return self.initialized_on_sender

    def is_receiver(self):
        """Check if the current worker is the receiver."""
        return self.initialized_on_receiver

    @overload
    def connect(self, *, worker_idx: int | None = None) -> None:
        ...

    @overload
    def connect(self, *, weights: Any | None = None) -> None:
        ...

    def connect(
        self, *, worker_idx: int | None = None, weights: Any | None = None
    ) -> None:
        """Method to be called once the workers have started.

        Triggers a rendez-vous for the workers to receive their copy of the weights.

        Dispatches to _setup_connection_and_weights_on_sender_impl() or _setup_connection_and_weights_on_receiver_impl()
        based on which initialization was performed.
        """
        if self.synchronized_on_receiver or self.synchronized_on_sender:
            raise RuntimeError("Cannot synchronize weights on sender twice.")
        if self._initialized_on_sender:
            torchrl_logger.debug("Synchronizing weights on sender")
            if worker_idx is not None:
                # Safety check, we can consider removing this in the future.
                raise RuntimeError(
                    "Cannot specify worker_idx on sender side during synchronization."
                )
            self.synchronized_on_sender = True
            try:
                self._setup_connection_and_weights_on_sender_impl(weights=weights)
            except Exception:
                self.synchronized_on_sender = False
                raise
        elif self._initialized_on_receiver:
            torchrl_logger.debug(f"Synchronizing weights on receiver -- {worker_idx=}")
            if weights is not None:
                # safety check: weights are passed to sender, not receiver for initial sync
                raise RuntimeError(
                    "Cannot specify weights on receiver side during synchronization."
                )
            self.synchronized_on_receiver = True
            try:
                self._setup_connection_and_weights_on_receiver_impl(
                    worker_idx=worker_idx
                )
            except Exception:
                self.synchronized_on_receiver = False
                raise
        else:
            raise RuntimeError(
                "Neither init_on_sender nor init_on_receiver have been called."
            )

    def _setup_connection_and_weights_on_sender_impl(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any | None = None,
    ) -> None:
        """Synchronize weights on sender side.

        Default implementation uses transport's setup_connection_and_weights_on_sender().
        Subclasses may override for custom behavior.
        """
        if self._shared_transport is not None:
            # We only need to synchronize once
            self.shared_transport.setup_connection_and_weights_on_sender()
            return

        idx = -1
        for idx, transport in enumerate(self._iterate_transports()):
            if worker_idx is not None and idx != worker_idx:
                continue
            transport.setup_connection_and_weights_on_sender()
        if idx == -1:
            raise RuntimeError("No transports available.")

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Synchronize weights on receiver side.

        Default implementation uses transport's setup_connection_and_weights_on_receiver().
        Subclasses may override for custom behavior.
        """
        if self.receiver_transport is None:
            return

        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = self._worker_idx

        # Call transport's synchronize method with all relevant kwargs
        weights = self.receiver_transport.setup_connection_and_weights_on_receiver(
            worker_idx=worker_idx,
            weights=self.weights,
            model=self.model,
            strategy=self._strategy,
        )

        # Apply weights to model if received (SharedMemTransport case)
        # For other transports (MPTransport, etc.), weights is None and synchronization
        # happens later via receive(), so this is a no-op
        if weights is not None:
            model = self.model
            self._strategy.apply_weights(model, weights, inplace=False)

    @property
    def synchronized_on_sender(self):
        return getattr(self, "_synchronized_on_sender", False)

    @synchronized_on_sender.setter
    def synchronized_on_sender(self, value: bool):
        self._synchronized_on_sender = value

    @property
    def synchronized_on_receiver(self):
        return getattr(self, "_synchronized_on_receiver", False)

    @synchronized_on_receiver.setter
    def synchronized_on_receiver(self, value: bool):
        self._synchronized_on_receiver = value

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def check_weight_access(self) -> None:
        """Check if the weights are accessible.

        Raises:
            RuntimeError: If the scheme is not initialized or weights cannot be accessed.
        """
        try:
            weights = self.weights
            if weights is None:
                raise RuntimeError(
                    "Weights are not accessible. The scheme may not have been properly "
                    "initialized with a model or context that provides weights."
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot access weights: {e}. Ensure the scheme was initialized with "
                "either a context (collector), model, or params_map."
            ) from e

    def __getstate__(self):
        """Prepare the scheme for pickling by excluding non-serializable runtime state."""
        state = self.__dict__.copy()
        # Remove non-serializable runtime state
        state["_context_ref"] = None
        state["_model_ref"] = None

        state["_initialized_on_sender"] = False
        state["_initialized_on_receiver"] = False

        state["_synchronized_on_sender"] = False
        state["_synchronized_on_receiver"] = False

        state["_pending_async"] = False
        state["_pending_transports"] = None

        return state

    def __setstate__(self, state):
        """Restore the scheme from pickling."""
        self.__dict__.update(state)
