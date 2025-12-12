from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tensordict import TensorDictBase
from torch import multiprocessing as mp, nn
from torchrl.weight_update._shared import SharedMemWeightSyncScheme
from torchrl.weight_update.utils import _resolve_model

from torchrl.weight_update.weight_sync_schemes import TransportBackend


class MultiProcessWeightSyncScheme(SharedMemWeightSyncScheme):
    """Weight synchronization for multiprocess operations using queues.

    This scheme creates transports that communicate via multiprocessing queues.
    Unlike the parent SharedMemWeightSyncScheme which uses shared memory for in-place
    updates, this scheme sends actual weight copies through queues to workers.

    A background thread on the receiver side listens for "receive" instructions
    from the sender. When an instruction arrives, the thread receives the weights
    from the weight queue and applies them to the model.

    It follows the same two-phase pattern as SharedMemWeightSyncScheme:

    1. **init_on_sender()**: Stores the recipe for creating device-specific weights
       (model reference, devices, mapping functions) without creating actual copies
    2. **synchronize_weights()**: Creates device-specific weight copies on-demand,
       sends them sequentially to workers via queues, allowing garbage collection
       between workers to minimize memory usage

    This approach avoids holding multiple weight copies in memory simultaneously,
    which is especially beneficial for large models with many workers.

    Synchronization flow:
    - **init_on_sender()**: Store configuration and register worker queues
    - **synchronize_weights()**: Create and send initial weights on-demand
    - **init_on_receiver()**: Create receiver that reads from queue
    - **send()**: Extract and send weight updates, wait for acknowledgments

    Args:
        strategy: The weight transmission strategy (default: "tensordict").
            Can be "tensordict" or "state_dict".
        sync: If True (default), send() blocks until receiver acknowledges.
            If False, send() returns immediately (use send_async/wait_async).

    Example:
        >>> # Basic usage with collector
        >>> scheme = MultiProcessWeightSyncScheme()
        >>> collector = MultiSyncCollector(
        ...     create_env_fn=[lambda: GymEnv("CartPole-v1")] * 3,
        ...     policy=policy,
        ...     frames_per_batch=100,
        ...     total_frames=1000,
        ...     weight_sync_schemes={"policy": scheme},
        ... )
        >>> # scheme.collect() is called automatically by collector
        >>> # Weights are created on-demand and sent to workers efficiently

    Note:
        The on-demand weight creation means that synchronize_weights() will be
        slower than if weights were pre-computed, but memory usage is significantly
        reduced, especially when workers use different devices or when the model
        is large.
    """

    def __init__(self, strategy: str = "tensordict", sync: bool = True):
        """Initialize the MultiProcessWeightSyncScheme.

        Args:
            strategy: The weight transmission strategy (default: "tensordict").
            sync: If True (default), send() blocks until receiver acknowledges.
        """
        super().__init__(strategy, sync=sync)
        # Override parent's shared transport - we don't use shared memory
        self._shared_transport = None

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
                - num_workers: Number of worker processes
                - policy_device: List of devices for each worker
                When provided, model_id is used to resolve the model from context.
            weights: Pre-extracted weights as TensorDict. Mutually exclusive with
                model and context. Used when weights are already available.
            model: Model to extract weights from. Mutually exclusive with weights
                and context.
            params_map: Pre-computed mapping of worker_idx to device-specific weights.
                Most explicit option. When provided, all other parameters must be None.
            devices: List of devices for each worker. Used with weights or model to
                automatically create device-specific copies. Length must equal num_workers.
            device_map_fn: Custom function (worker_idx, weights) -> device_weights.
                Allows full control over device mapping. Requires num_workers.
            num_workers: Number of workers. Required with device_map_fn, inferred
                from devices length otherwise.
            **kwargs: Reserved for future use.

        Examples:
            Simple usage with collector context (most common):

            >>> scheme = MultiProcessWeightSyncScheme()
            >>> collector = MultiSyncCollector(
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
            ...     num_workers=2,
            ... )

            Advanced: Pre-computed params_map:

            >>> weights_cpu = TensorDict.from_module(policy)
            >>> weights_cuda = weights_cpu.to("cuda")
            >>> scheme.init_on_sender(
            ...     params_map={0: weights_cpu, 1: weights_cuda, 2: weights_cuda},
            ...     num_workers=3,
            ... )
        """
        # Get params_map from parent class logic
        params_map_result = self._get_params_map(
            context=context,
            model_id=model_id,
            weights=weights,
            model=model,
            params_map=params_map,
            devices=devices,
            device_map_fn=device_map_fn,
            num_workers=num_workers,
        )

        # Store the mapping recipe for later use in synchronize_weights
        # Don't store params_map directly to save memory - we'll recompute on demand
        # Note: We don't store context directly to avoid pickle issues -
        # it's available via _context_ref
        self._device_mapping_info = {
            "model_id": model_id,
            "weights": weights,
            "model": model,
            "params_map": params_map,
            "devices": devices,
            "device_map_fn": device_map_fn,
            "num_workers": num_workers
            if num_workers is not None
            else len(params_map_result),
        }

        # Create per-worker queues for weight distribution
        # Each worker gets its own queue for receiving weights
        all_workers = list(params_map_result.keys())
        if not hasattr(self, "_weight_init_queues"):
            self._weight_init_queues = {}

        for worker_idx in all_workers:
            if worker_idx not in self._weight_init_queues:
                self._weight_init_queues[worker_idx] = mp.Queue()
            # Create instruction queues for background receiver
            if worker_idx not in self._instruction_queues:
                self._instruction_queues[worker_idx] = mp.Queue()
            # Create ack queues for synchronous mode
            if worker_idx not in self._ack_queues:
                self._ack_queues[worker_idx] = mp.Queue()

        # Store model_id and context on scheme
        self.model_id = model_id
        if context is not None:
            self.context = context

        # Register workers with their queues
        for worker_idx in all_workers:
            queue = self._weight_init_queues[worker_idx]
            ack_queue = self._ack_queues[worker_idx]
            # Create MPTransport for this worker with ack queue
            transport = MPTransport(weight_queue=queue, ack_queue=ack_queue)
            self._register_worker_sender(worker_idx=worker_idx, transport=transport)

    def _init_on_receiver_impl(
        self,
        *,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing worker_idx and model
            **kwargs: Alternative to context (worker_idx, model, etc.)
        """
        # Extract parameters from context or kwargs
        if context is not None:
            worker_idx = getattr(context, "worker_idx", None)
            if hasattr(context, "get_model"):
                model = context.get_model(model_id)
            else:
                model = _resolve_model(context, model_id)
        else:
            worker_idx = kwargs.get("worker_idx")
            model = kwargs.get("model")

        if worker_idx is None:
            raise ValueError("worker_idx must be provided via context or kwargs")

        # Get the queue for this worker
        if worker_idx not in self._weight_init_queues:
            raise ValueError(
                f"Worker {worker_idx} not registered. init_on_sender() must be called first."
            )

        queue = self._weight_init_queues[worker_idx]
        ack_queue = self._ack_queues.get(worker_idx)

        # Store on scheme directly
        self.model_id = model_id
        if context is not None:
            self.context = context

        # Store instruction and ack queue references for this worker
        if worker_idx in self._instruction_queues:
            self._receiver_instruction_queue = self._instruction_queues[worker_idx]
        if worker_idx in self._ack_queues:
            self._receiver_ack_queue = self._ack_queues[worker_idx]

        # Create transport with the worker's queue and ack queue
        transport = MPTransport(weight_queue=queue, ack_queue=ack_queue)
        self._register_transport_receiver(transport=transport)

        if model is not None:
            self.model = model

        # Store worker_idx for synchronize_weights
        self.worker_idx = worker_idx

    def send(
        self,
        weights: Any = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send weights synchronously to workers.

        This method:
        1. Prepares weights (extracts from model if weights=None)
        2. Sends weights to the weight queue
        3. Sends "receive" instruction to workers' background threads
        4. If sync=True, waits for acknowledgments from those workers

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

        Note: If sync=True (default), this is a blocking call that ensures
        specified workers are updated before returning.
        """
        from torchrl._utils import logger as torchrl_logger

        if not self.initialized_on_sender:
            raise RuntimeError("Must be initialized on sender before sending weights")
        if not self.synchronized_on_sender:
            raise RuntimeError("Must be synchronized on sender before sending weights")

        model_id = self.model_id
        context = self.context

        # Let the scheme prepare the weights
        prepared_weights = self.prepare_weights(
            weights=weights,
            model_id=model_id,
            strategy=self._strategy,
            context=context,
        )

        transports = list(self._iterate_transports(worker_ids))

        # Send weights to all workers first via queue (non-blocking)
        torchrl_logger.debug("Sending weights to queues")
        for transport in transports:
            if hasattr(transport, "send_weights_async"):
                # For MPTransport, pass model_id; other transports don't need it
                transport.send_weights_async(prepared_weights, model_id=model_id)
            else:
                # Fallback for transports that don't support async send
                transport.send_weights(prepared_weights)

        # Send instruction to workers' background threads to receive the weights
        torchrl_logger.debug("Sending 'receive' instruction to workers")
        self._send_instruction(instruction="receive", worker_ids=worker_ids)

        # Wait for all acknowledgments if in synchronous mode
        if self.sync:
            torchrl_logger.debug("Waiting for acknowledgments from workers")
            self._wait_for_ack(worker_ids=worker_ids)

    def _setup_connection_and_weights_on_sender_impl(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any | None = None,
    ) -> None:
        """Synchronize weights with workers before collection starts.

        Computes device-specific weight copies on-demand and sends them to workers
        sequentially via queues. This is called once after workers are initialized
        but before they start collecting data.

        Unlike send(), this does not wait for acknowledgments since workers are still
        in their initialization phase.

        This approach creates weight copies on-demand and sends them sequentially,
        allowing garbage collection between workers to reduce memory usage.

        Raises:
            RuntimeError: If init_on_sender() was not called first.
        """
        # Get the device mapping info stored during init_on_sender
        if not hasattr(self, "_device_mapping_info"):
            raise RuntimeError(
                "synchronize_weights() requires init_on_sender() to be called first"
            )

        mapping_info = self._device_mapping_info

        # Get context from weakref
        context = self.context

        # Compute params_map on-demand
        # Extract with explicit type casting for type checker
        model_id = mapping_info["model_id"]
        weights = mapping_info["weights"]
        model = mapping_info["model"]
        params_map_arg = mapping_info["params_map"]
        devices = mapping_info["devices"]
        device_map_fn = mapping_info["device_map_fn"]
        num_workers = mapping_info["num_workers"]

        params_map = self._get_params_map(
            context=context,
            model_id=model_id,
            weights=weights,
            model=model,
            params_map=params_map_arg,
            devices=devices,
            device_map_fn=device_map_fn,
            num_workers=num_workers,
        )

        # Send to workers sequentially via queues (no ACK - workers are still initializing)
        # This allows GC to clean up each worker's weights before creating the next
        for i, transport in enumerate(self._iterate_transports()):
            if worker_idx is not None and i != worker_idx:
                continue
            worker_weights = params_map[i]
            if hasattr(transport, "send_weights_async"):
                transport.send_weights_async(worker_weights, model_id=self._model_id)
            else:
                raise RuntimeError(
                    f"Transport {type(transport)} does not support async send for synchronization"
                )

        # Clean up the mapping info after synchronization
        delattr(self, "_device_mapping_info")

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Receive initial weights and start background receiver thread.

        This method:
        1. Receives initial weights from the sender via queue
        2. Applies them to the model
        3. Starts a background thread that listens for "receive" instructions

        Args:
            worker_idx: The worker index.
        """
        from torchrl._utils import logger as torchrl_logger

        # Use stored worker_idx if not provided
        if worker_idx is None:
            worker_idx = self._worker_idx

        if worker_idx is None:
            raise RuntimeError(
                "worker_idx must be provided for _setup_connection_and_weights_on_receiver_impl."
            )

        # Receive initial weights from queue via transport
        if self._receiver_transport is None:
            raise RuntimeError("Receiver transport not set.")

        weights = self._receiver_transport.setup_connection_and_weights_on_receiver(
            worker_idx=worker_idx,
            weights=self.weights,
            model=self.model,
            strategy=self._strategy,
        )

        # Store received weights for later use
        if weights is not None:
            self._receiver_weights = weights

        # Apply weights to model
        if weights is not None and self.model is not None:
            self._strategy.apply_weights(self.model, weights, inplace=False)
            torchrl_logger.debug(
                f"MultiProcessWeightSyncScheme: Worker {worker_idx} applied initial weights"
            )

        # Start background receiver thread
        self._start_background_receiver()

    def _background_receive_loop(self):
        """Background thread loop that waits for instructions and receives weights.

        This loop:
        1. Waits for a "receive" instruction from the sender
        2. Receives weights from the weight queue
        3. Applies them to the model
        4. Sends an acknowledgment back to the sender
        5. Repeats until stop event is set or "stop" instruction received
        """
        from torchrl._utils import logger as torchrl_logger

        torchrl_logger.debug(
            f"MultiProcessWeightSyncScheme: Background receiver started for worker {self._worker_idx}"
        )
        while not self._stop_event.is_set():
            try:
                instruction = self._wait_for_instruction()
                if instruction is None:
                    # Stop event was set or timeout
                    continue
                if instruction == "receive":
                    torchrl_logger.debug(
                        f"MultiProcessWeightSyncScheme: Worker {self._worker_idx} received 'receive' instruction"
                    )

                    # Receive weights from transport (blocking)
                    if self._receiver_transport is not None:
                        weights = self._receiver_transport.receive_weights(
                            model=self.model,
                            strategy=self._strategy,
                        )

                        if weights is not None:
                            torchrl_logger.debug(
                                f"MultiProcessWeightSyncScheme: Worker {self._worker_idx} received and applied weights"
                            )

                            # Cascade weight update to sub-collectors if context supports it
                            model_id = self._model_id or "policy"
                            if self.context is not None and hasattr(
                                self.context, "update_policy_weights_"
                            ):
                                torchrl_logger.debug(
                                    f"MultiProcessWeightSyncScheme: Cascading weight update to sub-collectors for {model_id=}"
                                )
                                self.context.update_policy_weights_(
                                    model_id=model_id, policy_or_weights=weights
                                )

                    # Send acknowledgment
                    self._send_ack("updated")

                elif instruction == "stop":
                    torchrl_logger.debug(
                        f"MultiProcessWeightSyncScheme: Worker {self._worker_idx} received 'stop' instruction"
                    )
                    break
                else:
                    torchrl_logger.warning(
                        f"MultiProcessWeightSyncScheme: Unknown instruction: {instruction}"
                    )
            except Exception as e:
                if not self._stop_event.is_set():
                    torchrl_logger.warning(
                        f"MultiProcessWeightSyncScheme: Background receiver error: {e}"
                    )

        torchrl_logger.debug(
            f"MultiProcessWeightSyncScheme: Background receiver stopped for worker {self._worker_idx}"
        )

    def create_transport(self, **kwargs) -> TransportBackend:
        """Create an MPTransport using the provided queue.

        Note:
            This is used internally by init_on_sender/init_on_receiver.
        """
        queue = kwargs.get("queue")
        return MPTransport(weight_queue=queue, ack_queue=None)


class MPTransport:
    """Multiprocessing transport using queues.

    This transport uses queues for weight distribution and synchronization.
    Similar to SharedMemTransport's queue-based approach, MPTransport uses
    queues to send initial weights to workers during synchronization.

    Initialization flow:
    - synchronize_weights() extracts weights and sends to all workers via queues
    - Workers receive the initial weights via setup_connection_and_weights_on_receiver()
    - Subsequent updates use send_weights_async() followed by acknowledgments

    Args:
        weight_queue (mp.Queue): The queue to use for sending weights.
        ack_queue (mp.Queue): The queue to use for receiving acknowledgments.
        timeout (float): The timeout for waiting for acknowledgment. Default is 10 seconds.
    """

    def __init__(self, weight_queue, ack_queue=None, timeout: float = 10.0):
        self.timeout = timeout
        self.weight_queue = weight_queue
        self.ack_queue = ack_queue

    def send_weights_async(self, weights: Any, model_id: str = "policy") -> None:
        """Send weights through the queue without waiting for acknowledgment.

        Use wait_ack() to wait for acknowledgment after sending to all workers.
        """
        # Send in format expected by worker loop: ((model_id, weights), "update_weights")
        self.weight_queue.put(((model_id, weights), "update_weights"))

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: Any = None,
    ) -> Any | None:
        """Receive weights from the queue (used in worker process).

        This method only handles weight update messages. Other messages
        (like "close", "continue", etc.) are ignored and should be handled
        by the main worker loop.

        Args:
            timeout: Maximum time to wait for weights (seconds).
                     None means use the transport's default timeout.
            weights: Ignored (weights come from queue).
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

        Returns:
            The received weights, or None if no data available.
        """
        # Use transport's default timeout if not specified
        if timeout is None:
            timeout = self.timeout
        data_in, msg = self.weight_queue.get(timeout=timeout)
        if msg == "update_weights":
            # data_in is (model_id, weights) - we ignore model_id, scheme knows it
            _model_id, received_weights = data_in

            # Apply weights to model if provided
            if model is not None and strategy is not None:
                strategy.apply_weights(model, received_weights)

            return received_weights
        else:
            raise ValueError(f"Expected 'update_weights' but got {msg}")

    def setup_connection_and_weights_on_sender(self) -> None:
        """No-op for MPTransport - weights are sent via scheme's synchronize_weights().

        The actual sending happens in MultiProcessWeightSyncScheme._setup_connection_and_weights_on_sender_impl(), which:
        1. Extracts weights from the context (e.g., collector.policy)
        2. Calls send_weights_async() on all worker transports
        3. Sends initial weights through queues to all workers

        This is similar to SharedMemTransport.setup_connection_and_weights_on_sender() which
        sends shared memory buffer references via queues.
        """

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int,
        weights: Any = None,
        model: Any = None,
        strategy: Any = None,
    ) -> Any:
        """Receive initial weights from sender during worker initialization.

        This method blocks waiting for the initial weights to be sent from the main process
        via queue. Similar to SharedMemTransport.setup_connection_and_weights_on_receiver() which receives
        shared memory buffer references via queues, this receives the actual weights via queues.

        The received weights are then applied to the worker's model by the scheme's synchronize_weights().

        Args:
            worker_idx: The worker index (used for logging/debugging).
            weights: Ignored (weights come from queue).
            model: Ignored.
            strategy: Ignored.

        Returns:
            The received weights if available, None otherwise (weights will come later via receive()).
        """
        # Wait for initial weights (blocking)
        data_in, msg = self.weight_queue.get(timeout=self.timeout)
        if msg == "update_weights":
            # data_in is (model_id, weights), extract just the weights
            _, received_weights = data_in
            return received_weights
        else:
            raise ValueError(f"Expected 'update_weights' but got {msg}")
