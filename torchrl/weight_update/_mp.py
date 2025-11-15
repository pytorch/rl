from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, overload

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class MultiProcessWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for multiprocess operations using pipes.

    This scheme creates transports that communicate via multiprocessing pipes.
    It follows a memory-efficient two-phase pattern similar to SharedMemWeightSyncScheme:

    1. **init_on_sender()**: Stores the recipe for creating device-specific weights
       (model reference, devices, mapping functions) without creating actual copies
    2. **synchronize_weights()**: Creates device-specific weight copies on-demand,
       sends them sequentially to workers via pipes, allowing garbage collection
       between workers to minimize memory usage

    This approach avoids holding multiple weight copies in memory simultaneously,
    which is especially beneficial for large models with many workers.

    Synchronization flow:
    - **init_on_sender()**: Store configuration and register worker pipes
    - **synchronize_weights()**: Create and send initial weights on-demand
    - **init_on_receiver()**: Create receiver that reads from pipe
    - **send()**: Extract and send weight updates, wait for acknowledgments

    Args:
        strategy: The weight transmission strategy (default: "tensordict").
            Can be "tensordict" or "state_dict".

    Example:
        >>> # Basic usage with collector
        >>> scheme = MultiProcessWeightSyncScheme()
        >>> collector = MultiSyncDataCollector(
        ...     create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
        ...     policy=policy,
        ...     frames_per_batch=100,
        ...     total_frames=1000,
        ...     weight_sync_schemes={"policy": scheme},
        ... )
        >>> # scheme.synchronize_weights() is called automatically by collector
        >>> # Weights are created on-demand and sent to workers efficiently

    Note:
        The on-demand weight creation means that synchronize_weights() will be
        slower than if weights were pre-computed, but memory usage is significantly
        reduced, especially when workers use different devices or when the model
        is large.
    """

    def synchronize_weights(self):
        """Send initial weights to all workers before collection starts.

        This method triggers the on-demand creation and distribution of device-specific
        weight copies to workers. Unlike pre-computing all weights during init_on_sender(),
        this approach creates each worker's weights sequentially, sends them via pipes,
        and allows garbage collection before creating the next worker's weights.

        This is a convenience method that delegates to the sender's synchronize_weights(),
        which handles the actual weight creation and distribution.

        Memory efficiency note:
            If all workers share the same device, only one weight copy is created and
            reused. If workers use different devices, weights are created and sent
            sequentially to minimize peak memory usage.

        Called automatically by:
            - MultiSyncDataCollector during initialization
            - MultiaSyncDataCollector during initialization

        Raises:
            RuntimeError: If init_on_sender() was not called first
        """
        if not self._initialized_on_sender or self._sender is None:
            raise RuntimeError(
                "Must call init_on_sender() before synchronize_weights() on MultiProcessWeightSyncScheme"
            )
        self._sender.synchronize_weights()

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

    def init_on_sender(
        self,
        *,
        model_id: str | None = None,
        context: Any = None,
        weights: TensorDictBase | None = None,
        model: nn.Module | None = None,
        params_map: dict[int, TensorDictBase] | None = None,
        devices: list[torch.device] | None = None,
        device_map_fn: Callable[[int, TensorDictBase], TensorDictBase] | None = None,
        num_workers: int | None = None,
        pipes: list[Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        This method stores the configuration needed to create device-specific weight
        copies during synchronization. Weight copies are created on-demand during
        `synchronize_weights()` to reduce memory usage.

        Similar to `SharedMemWeightSyncScheme`, this follows a two-phase pattern:
        1. `init_on_sender()`: Store the recipe for creating weights
        2. `synchronize_weights()`: Create and send weights on-demand

        Args:
            model_id: Identifier for the model being synchronized (e.g., "policy").
                Required when using context.
            context: Optional context object (e.g., collector) providing:
                - pipes: List of multiprocessing pipes for worker communication
                - num_workers: Number of worker processes
                - policy_device: List of devices for each worker
                When provided, model_id is used to resolve the model from context.
            weights: Pre-extracted weights as TensorDict. Mutually exclusive with
                model and context. Used when weights are already available.
            model: Model to extract weights from. Mutually exclusive with weights
                and context.
            params_map: Pre-computed mapping of worker_idx to device-specific weights.
                Most explicit option. When provided, all other parameters except pipes
                must be None.
            devices: List of devices for each worker. Used with weights or model to
                automatically create device-specific copies. Length must equal num_workers.
            device_map_fn: Custom function (worker_idx, weights) -> device_weights.
                Allows full control over device mapping. Requires num_workers.
            num_workers: Number of workers. Required with device_map_fn, inferred
                from devices length or pipes otherwise.
            pipes: List of multiprocessing pipes. Required unless provided via context.
            **kwargs: Alternative way to provide pipes (for backward compatibility).

        Examples:
            Simple usage with collector context (most common):

            >>> scheme = MultiProcessWeightSyncScheme()
            >>> collector = MultiSyncDataCollector(
            ...     create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
            ...     policy=policy,
            ...     frames_per_batch=100,
            ...     weight_sync_schemes={"policy": scheme},
            ... )
            >>> # scheme.init_on_sender() is called automatically by collector

            Direct initialization with explicit devices:

            >>> scheme = MultiProcessWeightSyncScheme()
            >>> weights = TensorDict.from_module(policy)
            >>> scheme.init_on_sender(
            ...     weights=weights,
            ...     devices=[torch.device("cpu"), torch.device("cuda:0")],
            ...     pipes=[pipe1, pipe2],
            ... )

            Advanced: Pre-computed params_map:

            >>> weights_cpu = TensorDict.from_module(policy)
            >>> weights_cuda = weights_cpu.to("cuda")
            >>> scheme.init_on_sender(
            ...     params_map={0: weights_cpu, 1: weights_cuda, 2: weights_cuda},
            ...     pipes=[pipe1, pipe2, pipe3],
            ... )
        """
        # Extract parameters from context or parameters/kwargs
        if context is not None:
            pipes = getattr(context, "pipes", None)
            num_workers = getattr(context, "num_workers", None)
        else:
            # Use the pipes parameter if provided, otherwise check kwargs
            if pipes is None:
                pipes = kwargs.get("pipes")

        if pipes is None:
            raise ValueError("pipes must be provided via context or kwargs")
        if num_workers is None:
            num_workers = len(pipes) if pipes else 0

        # Store the mapping recipe for later use in synchronize_weights
        # Don't compute params_map yet to save memory
        # Note: We don't store context directly to avoid pickle issues -
        # it's available via sender._context_ref
        self._device_mapping_info = {
            "model_id": model_id,
            "weights": weights,
            "model": model,
            "params_map": params_map,
            "devices": devices,
            "device_map_fn": device_map_fn,
            "num_workers": num_workers,
        }

        # Create sender with the shared transport
        sender = MPWeightSender(self)
        sender._model_id = model_id
        if context is not None:
            sender._context_ref = weakref.ref(context)

        for worker_idx, pipe in enumerate(pipes):
            sender._register_worker(worker_idx, pipe)

        self._sender = sender
        self._initialized_on_sender = True

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
        """Compute the params_map (dict[worker_idx, device_weights]) on-demand.

        This method creates device-specific weight copies based on the provided
        configuration. It's called during synchronize_weights() rather than
        init_on_sender() to reduce memory usage.

        The method supports several input patterns:
        1. Direct params_map: Returned as-is (already computed)
        2. Context + model_id: Extract model and devices from context
        3. Model/weights + devices: Create copies on specified devices
        4. Model/weights + device_map_fn: Apply custom mapping function

        Args:
            context: Context object (e.g., collector) to extract model and devices from
            model_id: Model identifier to resolve within context
            weights: Pre-extracted weights as TensorDict
            model: Model to extract weights from
            params_map: Pre-computed mapping (returned as-is if provided)
            devices: List of devices, one per worker
            device_map_fn: Custom mapping function (worker_idx, weights) -> device_weights
            num_workers: Number of workers (required with device_map_fn)

        Returns:
            dict[int, TensorDictBase]: Mapping from worker_idx to device-specific weights

        Raises:
            ValueError: If parameter combinations are invalid or mutually exclusive
        """
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
        pipe: Any = ...,
        model: Any | None = None,
        **kwargs,
    ) -> None:
        ...

    def init_on_receiver(
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
        receiver = MPWeightReceiver(self)
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
            This is used internally by init_on_sender/init_on_receiver.
        """
        return MPTransport(pipe)


class MPTransport:
    """Multiprocessing transport using pipes.

    This transport uses pipes for weight distribution and synchronization.
    Similar to SharedMemTransport's queue-based approach, MPTransport uses
    pipes to send initial weights to workers during synchronization.

    Initialization flow:
    - MPWeightSender.synchronize_weights() extracts weights and sends to all workers via pipes
    - Workers receive the initial weights via synchronize_weights_on_worker()
    - Subsequent updates use send_weights_async() followed by acknowledgments

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

    def send_weights_async(self, weights: Any, model_id: str = "policy") -> None:
        """Send weights through the pipe without waiting for acknowledgment.

        Use wait_ack() to wait for acknowledgment after sending to all workers.
        """
        # Send in format expected by worker loop: ((model_id, weights), "update_weights")
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

        Note:
            model_id is returned as "policy" for backward compatibility, but transports
            are now bound to a single model during initialization.
        """
        if self.pipe.poll(timeout):
            data_in, msg = self.pipe.recv()
            if msg == "update_weights":
                # data_in is now (model_id, weights)
                return data_in
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
        """No-op for MPTransport - weights are sent via MPWeightSender.synchronize_weights().

        The actual sending happens in MPWeightSender.synchronize_weights(), which:
        1. Extracts weights from the context (e.g., collector.policy)
        2. Calls send_weights_async() on all worker transports
        3. Sends initial weights through pipes to all workers

        This is similar to SharedMemTransport.synchronize_weights_on_sender() which
        sends shared memory buffer references via queues.
        """

    def synchronize_weights_on_worker(self, worker_idx: int) -> Any:
        """Receive initial weights from sender during worker initialization.

        This method blocks waiting for the initial weights to be sent from the main process
        via pipe. Similar to SharedMemTransport.synchronize_weights_on_worker() which receives
        shared memory buffer references via queues, this receives the actual weights via pipes.

        The received weights are then applied to the worker's model by MPWeightReceiver.synchronize_weights().

        Args:
            worker_idx: The worker index (used for logging/debugging).

        Returns:
            The received weights if available, None otherwise (weights will come later via receive()).
        """
        # Wait for initial weights (blocking)
        if self.pipe.poll(timeout=self.timeout):
            data_in, msg = self.pipe.recv()
            if msg == "update_weights":
                # data_in is (model_id, weights), extract just the weights
                _, weights = data_in
                return weights
        # If we don't receive weights, return None (weights will come later)
        return None


class MPWeightReceiver(WeightReceiver):
    """Weight receiver for multiprocess systems using pipes.

    Receives weight updates from the main process via multiprocessing pipes.
    This is typically instantiated and managed by :class:`MultiProcessWeightSyncScheme`.
    """

    _transport: MPTransport | None


class MPWeightSender(WeightSender):
    """Weight sender for multiprocess systems using pipes.

    Sends weight updates to worker processes via multiprocessing pipes.
    Supports both synchronous and asynchronous sending patterns.
    This is typically instantiated and managed by :class:`MultiProcessWeightSyncScheme`.
    """

    _transport: MPTransport | None
    _model_id: str
    _scheme: MultiProcessWeightSyncScheme

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

        model_id = self._model_id
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
                # For MPTransport, pass model_id; other transports don't need it
                transport.send_weights_async(prepared_weights, model_id=model_id)
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
                transport.send_weights_async(prepared_weights, model_id=self._model_id)
            else:
                raise RuntimeError(
                    f"transport of type {type(transport)} does not support async send."
                )

        self._pending_async = True

    def synchronize_weights(self) -> None:
        """Synchronize weights with workers before collection starts.

        Computes device-specific weight copies on-demand and sends them to workers
        sequentially via pipes. This is called once after workers are initialized
        but before they start collecting data.

        Unlike send(), this does not wait for acknowledgments since workers are still
        in their initialization phase.

        This approach creates weight copies on-demand and sends them sequentially,
        allowing garbage collection between workers to reduce memory usage.

        Raises:
            RuntimeError: If init_on_sender() was not called first.
        """
        # Get the device mapping info stored during init_on_sender
        if not hasattr(self._scheme, "_device_mapping_info"):
            raise RuntimeError(
                "MPWeightSender.synchronize_weights() requires a call to MultiProcessWeightSyncScheme.init_on_sender"
            )

        mapping_info = self._scheme._device_mapping_info

        # Get context from sender's weakref
        context = self._context_ref() if self._context_ref is not None else None

        # Compute params_map on-demand
        # Extract with explicit type casting for type checker
        model_id = mapping_info["model_id"]
        weights = mapping_info["weights"]
        model = mapping_info["model"]
        params_map_arg = mapping_info["params_map"]
        devices = mapping_info["devices"]
        device_map_fn = mapping_info["device_map_fn"]
        num_workers = mapping_info["num_workers"]

        params_map = self._scheme._get_params_map(
            context=context,
            model_id=model_id,
            weights=weights,
            model=model,
            params_map=params_map_arg,
            devices=devices,
            device_map_fn=device_map_fn,
            num_workers=num_workers,
        )

        # Send to workers sequentially via pipes (no ACK - workers are still initializing)
        # This allows GC to clean up each worker's weights before creating the next
        for i, transport in enumerate(self._iterate_transports()):
            worker_weights = params_map[i]
            if hasattr(transport, "send_weights_async"):
                transport.send_weights_async(worker_weights, model_id=self._model_id)  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    f"Transport {type(transport)} does not support async send for synchronization"
                )

        # Clean up the mapping info after synchronization
        delattr(self._scheme, "_device_mapping_info")
