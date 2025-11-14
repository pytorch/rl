from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, overload

import torch
import torch.distributed

from tensordict import TensorDict, TensorDictBase

from torch import multiprocessing as mp, nn

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightStrategy,
    WeightSyncScheme,
)


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
        self._unique_weights = None

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

        if self._unique_weights is None:
            raise RuntimeError("Unique weights not set. Call register_weights() first.")
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
        sender = SharedMemWeightSender(self)
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

    @overload
    def init_on_receiver(
        self,
        *,
        model_id: str,
        context: Any,
    ) -> None:
        ...

    @overload
    def init_on_receiver(
        self,
        *,
        model: Any,
        worker_idx: int,
    ) -> None:
        ...

    def init_on_receiver(
        self,
        *,
        model_id: str | None = None,
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
            if model_id is None:
                raise ValueError("model_id is required when context is provided")
            if hasattr(context, "get_model"):
                model = context.get_model(model_id)
            elif model is None:
                model = _resolve_model(context, model_id)
            worker_idx = getattr(context, "worker_idx", worker_idx)

        # Create receiver with the shared transport
        receiver = SharedMemWeightReceiver(self)
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
            This is used internally by init_on_sender/init_on_receiver.
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


class SharedMemWeightReceiver(WeightReceiver):
    """Weight receiver for shared memory systems.

    Receives weight updates via shared memory buffers. Workers automatically
    see weight updates without explicit message passing, providing zero-copy
    weight synchronization. This is typically instantiated and managed by
    :class:`SharedMemWeightSyncScheme`.
    """

    _transport: SharedMemTransport | None


class SharedMemWeightSender(WeightSender):
    """Weight sender for shared memory systems.

    Sends weight updates by writing directly to shared memory buffers.
    All workers automatically see updates without explicit communication,
    providing zero-copy weight synchronization. This is typically instantiated
    and managed by :class:`SharedMemWeightSyncScheme`.
    """

    _transport: SharedMemTransport | None
