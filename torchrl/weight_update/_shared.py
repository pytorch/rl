from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed

from tensordict import TensorDict, TensorDictBase

from torch import multiprocessing as mp, nn

from torchrl._utils import logger as torchrl_logger

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
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

    @property
    def unique_weights(self) -> list[TensorDictBase]:
        """Get the unique weights.

        Returns:
            The unique weights.
        """
        if self._unique_weights is None:
            raise RuntimeError("Unique weights not set. Call register_weights() first.")
        return self._unique_weights

    def register_weights(
        self, params_map: dict[int, mp.Queue], init_queues: dict[int, mp.Queue]
    ) -> None:
        """Initialize per-worker queues for shared memory buffer distribution."""
        from torchrl.collectors.utils import _cast

        self._weight_queues = init_queues
        self._params_map = params_map
        # Create set of the unique weights
        self._unique_weights = []
        for weights in params_map.values():
            if id(weights) in [id(w) for w in self._unique_weights]:
                continue
            weights = weights.data.apply(_cast, weights)
            self._unique_weights.append(weights)

    def setup_connection_and_weights_on_sender(self) -> None:
        """Send shared memory buffer reference to workers via their per-worker queues.

        Both CPU and CUDA tensors maintain shared references through queues.
        Each worker reads from its own dedicated queue, to avoid race conditions.

        """
        torchrl_logger.debug("Sending shared memory weights to workers.")
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        for worker_idx, queue in self._weight_queues.items():
            weights = self._params_map[worker_idx]
            queue.put(weights)

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any = None,
        model: Any = None,
        strategy: Any = None,
        timeout: float = 10.0,
    ) -> TensorDictBase:
        """Receive shared memory buffer reference from sender via their per-worker queues.

        Each worker reads from its own dedicated queue, to avoid race conditions.

        Args:
            worker_idx: The worker index.
            weights: Ignored (weights come from queue).
            model: Ignored.
            strategy: Ignored.
            timeout: Timeout for reading from queue.

        Returns:
            The shared memory weights TensorDict.
        """
        torchrl_logger.debug(
            f"Receiving shared memory weights from worker {worker_idx}."
        )
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        if worker_idx not in self._weight_queues:
            raise RuntimeError(f"Worker {worker_idx} not registered in queues.")

        # Read from dedicated queue for this worker
        worker_queue = self._weight_queues[worker_idx]
        received_weights = worker_queue.get(timeout=timeout)
        return received_weights

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
            raise ValueError(f"Unsupported weights type: {type(weights)=}")
        # Unflatten if needed to match shared buffer structure
        weights_to_update = weights
        if any("." in key for key in weights.keys()):
            weights_to_update = weights.unflatten_keys(".")

        # Detach weights to allow in-place updates (gradients are not needed for weight sync)
        weights_to_update = weights_to_update.detach()

        if self._unique_weights is None:
            raise RuntimeError("Unique weights not set. Call register_weights() first.")
        for buffer in self._unique_weights:
            if buffer.requires_grad:
                raise RuntimeError(
                    "Gradients should not be required for shared memory buffers."
                )
            if weights_to_update.requires_grad:
                raise RuntimeError("Gradients should not be required for weights.")
            buffer.update_(weights_to_update, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: Any = None,
    ) -> Any | None:
        """Apply shared memory weights to the model.

        For shared memory, weights are already available (passed via the weights arg).
        This method applies them to the model, matching the pattern of other transports.

        Args:
            timeout: Ignored (shared memory access is instant).
            weights: The shared memory buffer containing current weights.
            model: The model to apply weights to.
            strategy: Strategy for applying weights.

        Returns:
            The applied weights, or None if not applied.
        """
        # Apply weights to model if provided (same pattern as other transports)
        if model is not None and strategy is not None and weights is not None:
            torchrl_logger.debug(
                f"Applying shared memory weights {type(weights)=} to model {model} with {strategy=}."
            )
            strategy.apply_weights(model, weights)
            return weights
        torchrl_logger.debug(
            f"Not applying shared memory weights {type(weights)=} to model {model} with {strategy=}."
        )
        return None

    def send_ack(self, message: str = "updated") -> None:
        """No-op for shared memory - no acknowledgment needed."""


class SharedMemWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization using shared memory.

    This scheme uses shared memory for in-place weight updates. Workers
    automatically see weight updates without explicit message passing.

    A background thread on the receiver side listens for "receive" instructions
    from the sender. When an instruction arrives, the thread applies the current
    shared memory weights to the model and sends an acknowledgment.

    Args:
        strategy: The weight transmission strategy (default: "tensordict").
        sync: If True (default), send() blocks until receiver acknowledges.
            If False, send() returns immediately (use send_async/wait_async).

    Example:
        >>> # Basic usage
        >>> scheme = SharedMemWeightSyncScheme()
        >>> # Weights are initialized via init_on_sender()
    """

    def __init__(
        self,
        strategy: str = "tensordict",
        sync: bool = True,
    ):
        super().__init__(strategy)
        self.sync = sync
        # Create a single shared transport for all workers
        self.shared_transport = SharedMemTransport()

        # Create per-worker queues to avoid race conditions
        # Each worker gets its own queue for weight initialization
        self._weight_init_queues = {}  # worker_idx -> Queue

        # Instruction queues: sender puts "receive" instruction, receiver's background thread reads
        self._instruction_queues: dict[int, mp.Queue] = {}  # worker_idx -> Queue

        # Acknowledgment queues: receiver puts "updated" ack, sender reads for sync mode
        self._ack_queues: dict[int, mp.Queue] = {}  # worker_idx -> Queue

        # Receiver's instruction queue reference (set during init_on_receiver)
        self._receiver_instruction_queue: mp.Queue | None = None
        self._receiver_ack_queue: mp.Queue | None = None

    def _init_on_sender_impl(
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
            >>> collector = MultiSyncCollector(
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
            >>> collector = MultiSyncCollector(
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
            # Create instruction queues for background receiver
            if worker_idx not in self._instruction_queues:
                self._instruction_queues[worker_idx] = mp.Queue()
            # Create ack queues for synchronous mode
            if worker_idx not in self._ack_queues:
                self._ack_queues[worker_idx] = mp.Queue()

        # Set worker info in transport
        self.shared_transport.register_weights(params_map, self._weight_init_queues)

        # Store model_id and context on scheme
        self.model_id = model_id
        if context is not None:
            self.context = context

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
        # Import _cast locally to avoid circular imports
        from torchrl.collectors.utils import _cast

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
            if model is None:
                if model_id == "policy":
                    # we need to get a copy of the weights from the factory
                    model = context.policy_factory[0]()
            weights = TensorDict.from_module(model)
        elif model is not None:
            if weights is not None:
                raise ValueError("weights cannot be provided if model is provided")
            weights = TensorDict.from_module(model)
        if weights is not None:
            weights = weights.data.apply(_cast, weights)
        # To make the map, we need the list of devices, or the map fn
        if devices is not None:
            # Get the unique devices
            devices_set = set(devices)
            weights_devices = (
                {p.device for p in weights.values(True, True)}
                if weights is not None
                else set()
            )
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

    def _init_on_receiver_impl(
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

        # Store on scheme directly
        self.model_id = model_id
        if context is not None:
            self.context = context

        # Register the model
        if model is not None:
            self.model = model

        # Store worker_idx for synchronize_weights
        self.worker_idx = worker_idx

        # Store references to instruction and ack queues for this worker
        # These are created by init_on_sender and passed via pickle
        if worker_idx is not None:
            if worker_idx in self._instruction_queues:
                self._receiver_instruction_queue = self._instruction_queues[worker_idx]
            if worker_idx in self._ack_queues:
                self._receiver_ack_queue = self._ack_queues[worker_idx]

        self.create_transport()

    def _wait_for_instruction(self, timeout: float | None = None) -> str | None:
        """Block until an instruction arrives from the sender.

        Args:
            timeout: Maximum time to wait for instruction (seconds).
                None means block indefinitely.

        Returns:
            The instruction string (e.g., "receive", "stop"), or None if
            stop event is set or timeout expires.
        """
        if self._receiver_instruction_queue is None:
            raise RuntimeError(
                "Instruction queue not set. init_on_receiver() must be called first."
            )

        try:
            # Check stop event periodically while waiting
            while True:
                if self._stop_event is not None and self._stop_event.is_set():
                    return None
                try:
                    # Use short timeout to allow checking stop event
                    instruction = self._receiver_instruction_queue.get(timeout=0.1)
                    return instruction
                except Exception:
                    # Queue.Empty - continue waiting
                    if timeout is not None:
                        timeout -= 0.1
                        if timeout <= 0:
                            return None
        except Exception as e:
            torchrl_logger.warning(f"Error waiting for instruction: {e}")
            return None

    def _send_instruction(
        self,
        instruction: str = "receive",
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send instruction to receiver(s) to trigger weight reception.

        Args:
            instruction: The instruction to send (default: "receive").
            worker_ids: Which workers to send to (None = all workers).
        """
        if not self._instruction_queues:
            raise RuntimeError(
                "Instruction queues not created. init_on_sender() must be called first."
            )

        if worker_ids is None:
            target_workers = list(self._instruction_queues.keys())
        elif isinstance(worker_ids, int):
            target_workers = [worker_ids]
        else:
            target_workers = list(worker_ids)

        for worker_idx in target_workers:
            if worker_idx not in self._instruction_queues:
                raise ValueError(f"Worker {worker_idx} not registered")
            self._instruction_queues[worker_idx].put(instruction)

    def _send_ack(self, message: str = "updated") -> None:
        """Send acknowledgment back to sender after receiving weights.

        Args:
            message: The acknowledgment message (default: "updated").
        """
        if self._receiver_ack_queue is not None:
            self._receiver_ack_queue.put(message)

    def _wait_for_ack(
        self,
        worker_ids: int | list[int] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Wait for acknowledgment from receiver(s).

        Args:
            worker_ids: Which workers to wait for (None = all workers).
            timeout: Maximum time to wait (seconds). None means block indefinitely.
        """
        if not self._ack_queues:
            return  # No ack queues, nothing to wait for

        if worker_ids is None:
            target_workers = list(self._ack_queues.keys())
        elif isinstance(worker_ids, int):
            target_workers = [worker_ids]
        else:
            target_workers = list(worker_ids)

        for worker_idx in target_workers:
            if worker_idx not in self._ack_queues:
                raise ValueError(f"Worker {worker_idx} not registered")
            try:
                ack = self._ack_queues[worker_idx].get(timeout=timeout)
                if ack != "updated":
                    torchrl_logger.warning(
                        f"Unexpected ack from worker {worker_idx}: {ack}"
                    )
            except Exception as e:
                torchrl_logger.warning(
                    f"Timeout waiting for ack from worker {worker_idx}: {e}"
                )

    def create_transport(self, **kwargs) -> TransportBackend:
        """Create shared memory transport.

        Returns the shared transport instance that all workers will use.
        Since this is shared memory, there's only one transport shared by all workers.

        Note:
            This is used internally by init_on_sender/init_on_receiver.
        """
        return self.shared_transport

    def prepare_weights(
        self,
        weights: Any,
        model_id: str,
        strategy: WeightStrategy,
        context: Any = None,
    ) -> Any:
        """Prepare weights for SharedMemWeightSyncScheme.

        When weights=None, we extract fresh weights from the model and update
        the shared memory buffer in-place so workers see the change.

        Args:
            weights: Raw weights input
            model_id: The model identifier
            strategy: WeightStrategy for extracting/converting weights
            context: Optional context (e.g., collector) for cache lookup

        Returns:
            Shared memory weights ready to send
        """
        # If weights are explicitly provided, use them directly
        if weights is not None:
            fresh_weights = super().prepare_weights(
                weights, model_id, strategy, context
            )
        else:
            # Extract fresh weights from the model (base class handles this)
            fresh_weights = super().prepare_weights(None, model_id, strategy, context)

        if fresh_weights is None:
            return None

        # Update the shared memory buffer in-place so workers see the change
        if self._shared_transport is not None and self.shared_transport.unique_weights:
            torchrl_logger.debug("Updating shared memory buffer in-place")
            shared_weights = self.shared_transport.unique_weights[0]
            # In-place update of shared memory buffer with fresh weights
            shared_weights.data.update_(fresh_weights.data)
            return shared_weights

        torchrl_logger.debug("No shared transport, returning fresh weights")
        # If no shared transport, just return the fresh weights
        return fresh_weights

    def send(
        self,
        weights: Any = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send weights via shared memory (in-place update).

        For SharedMemWeightSyncScheme:
        1. prepare_weights() updates the shared memory buffer in-place
        2. _send_instruction() tells workers to apply the new weights
        3. If sync=True, waits for acknowledgments from all workers

        Args:
            weights: Weights to send (can be None to extract from model).
            worker_ids: Which workers to notify (None = all workers).
        """
        if not self.initialized_on_sender:
            raise RuntimeError("Must be initialized on sender before sending weights")
        if not self.synchronized_on_sender:
            raise RuntimeError("Must be synchronized on sender before sending weights")

        # prepare_weights updates the shared buffer in-place
        torchrl_logger.debug(
            "Sending weights via shared memory -- calling prepare_weights()"
        )
        self.prepare_weights(
            weights=weights,
            model_id=self._model_id,
            strategy=self._strategy,
            context=self.context,
        )

        # Send instruction to workers' background threads to apply the weights
        torchrl_logger.debug("Sending 'receive' instruction to workers")
        self._send_instruction(instruction="receive", worker_ids=worker_ids)

        # Wait for acknowledgments if in synchronous mode
        if self.sync:
            torchrl_logger.debug("Waiting for acknowledgments from workers")
            self._wait_for_ack(worker_ids=worker_ids)

    @property
    def weights(self) -> Any | None:
        """Get the current weights from shared memory.

        For SharedMemWeightSyncScheme:
        - On sender side: weights are in transport's _unique_weights
        - On receiver side: weights are in _receiver_shared_weights (stored during connect())

        Returns:
            The weights TensorDict if available, None otherwise.
        """
        # On receiver side, use the stored shared buffer reference
        if (
            hasattr(self, "_receiver_shared_weights")
            and self._receiver_shared_weights is not None
        ):
            return self._receiver_shared_weights

        # On sender side, get from the shared transport
        if self._shared_transport is not None and self.shared_transport.unique_weights:
            return self.shared_transport.unique_weights[0]

        # Fall back to parent implementation
        return super().weights

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Synchronize weights on receiver side for shared memory.

        Reads the shared memory buffer from the queue and applies it to the model.
        Then starts a background thread that listens for "receive" instructions
        from the sender and applies weights when instructed.

        If a receiver_transport is set (e.g., for MultiProcessWeightSyncScheme),
        defers to the base class implementation.
        """
        # If receiver_transport is set (e.g., MultiProcess subclass), use base behavior
        if self._receiver_transport is not None:
            return super()._setup_connection_and_weights_on_receiver_impl(
                worker_idx=worker_idx
            )

        # SharedMem-specific: use shared_transport
        if self._shared_transport is None:
            raise RuntimeError(
                "SharedMemWeightSyncScheme requires shared_transport to be set."
            )

        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = self.worker_idx

        if worker_idx is None:
            raise RuntimeError(
                "worker_idx must be provided for _setup_connection_and_weights_on_receiver_impl."
            )

        # Read shared memory buffer from queue
        weights = self._shared_transport.setup_connection_and_weights_on_receiver(
            worker_idx=worker_idx
        )

        # Store the shared buffer reference for later receive() calls
        # This is the actual shared memory buffer that the sender updates
        self._receiver_shared_weights = weights

        # Apply weights to model
        if weights is not None and self.model is not None:
            self._strategy.apply_weights(self.model, weights, inplace=False)

        # Start background receiver thread that listens for instructions
        self._start_background_receiver()

    def _background_receive_loop(self):
        """Background thread loop that waits for instructions and applies weights.

        This loop:
        1. Waits for a "receive" instruction from the sender
        2. Applies the current shared memory weights to the model
        3. Sends an acknowledgment back to the sender
        4. Repeats until stop event is set or "stop" instruction received
        """
        torchrl_logger.debug(
            f"SharedMemWeightSyncScheme: Background receiver started for worker {self._worker_idx}"
        )
        while not self._stop_event.is_set():
            try:
                instruction = self._wait_for_instruction()
                if instruction is None:
                    # Stop event was set or timeout
                    continue
                if instruction == "receive":
                    torchrl_logger.debug(
                        f"SharedMemWeightSyncScheme: Worker {self._worker_idx} received 'receive' instruction"
                    )
                    # Apply the current shared memory weights to the model
                    # The weights are already updated in shared memory by the sender
                    if (
                        self._receiver_shared_weights is not None
                        and self.model is not None
                    ):
                        self._strategy.apply_weights(
                            self.model, self._receiver_shared_weights, inplace=True
                        )
                        torchrl_logger.debug(
                            f"SharedMemWeightSyncScheme: Worker {self._worker_idx} applied weights"
                        )

                    # Cascade weight update to sub-collectors if context supports it
                    model_id = self._model_id or "policy"
                    if self.context is not None and hasattr(
                        self.context, "update_policy_weights_"
                    ):
                        torchrl_logger.debug(
                            f"SharedMemWeightSyncScheme: Cascading weight update to sub-collectors for {model_id=}"
                        )
                        self.context.update_policy_weights_(
                            model_id=model_id,
                            policy_or_weights=self._receiver_shared_weights,
                        )

                    # Send acknowledgment
                    self._send_ack("updated")
                elif instruction == "stop":
                    torchrl_logger.debug(
                        f"SharedMemWeightSyncScheme: Worker {self._worker_idx} received 'stop' instruction"
                    )
                    break
                else:
                    torchrl_logger.warning(
                        f"SharedMemWeightSyncScheme: Unknown instruction: {instruction}"
                    )
            except Exception as e:
                if not self._stop_event.is_set():
                    torchrl_logger.warning(
                        f"SharedMemWeightSyncScheme: Background receiver error: {e}"
                    )

        torchrl_logger.debug(
            f"SharedMemWeightSyncScheme: Background receiver stopped for worker {self._worker_idx}"
        )

    def __getstate__(self):
        """Prepare the scheme for pickling."""
        state = super().__getstate__()
        # mp.Queue objects can be pickled and shared across processes
        # Keep them in state so workers have access
        return state

    def shutdown(self) -> None:
        """Stop the background receiver thread and clean up."""
        # Signal all workers to stop
        if self._instruction_queues:
            for worker_idx in self._instruction_queues:
                try:
                    self._instruction_queues[worker_idx].put("stop")
                except Exception:
                    pass

        # Stop local background thread if running
        if self._stop_event is not None:
            self._stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(timeout=5.0)
            if self._background_thread.is_alive():
                torchrl_logger.warning(
                    "SharedMemWeightSyncScheme: Background thread did not stop gracefully"
                )
        self._background_thread = None
        self._stop_event = None
