from __future__ import annotations

import weakref
from typing import Any, overload

from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class MultiProcessWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for multiprocess operations using pipes.

    This scheme creates transports that communicate via multiprocessing pipes.
    Similar to SharedMemWeightSyncScheme which uses queues for shared memory
    buffer distribution, MultiProcessWeightSyncScheme uses pipes to send
    weight copies to each worker.

    Synchronization flow:
    - init_on_sender() creates a MPWeightSender and registers all worker pipes
    - synchronize_weights() triggers the initial weight distribution via pipes
    - init_on_receiver() creates a MPWeightReceiver that receives from its pipe
    - Subsequent updates use send() which extracts, sends, and waits for ACKs

    Args:
        strategy: The weight transmission strategy (default: "tensordict").

    Example:
        >>> # Basic usage with collector
        >>> scheme = MultiProcessWeightSyncScheme()
        >>> collector = MultiSyncDataCollector(
        ...     create_env_fn=[lambda: GymEnv("CartPole-v1")],
        ...     policy=policy,
        ...     frames_per_batch=100,
        ...     total_frames=1000,
        ...     weight_sync_schemes={"policy": scheme},
        ... )
        >>> # scheme.synchronize_weights() is called automatically by collector
    """

    def synchronize_weights(self):
        """Method to be called once the workers have started.

        Triggers a rendez-vous for the workers to receive their copy of the weights.

        This is a convenience method that delegates to the sender's synchronize_weights().
        The sender will extract weights from the context and send them to all workers via pipes.
        """
        if not self._initialized_on_sender or self._sender is None:
            raise RuntimeError(
                "Must call init_on_sender() before synchronize_weights() on MultiProcessWeightSyncScheme"
            )
        self._sender.synchronize_weights()

    @overload
    def init_on_sender(
        self,
        model_id: str,
        context: Any,
        **kwargs,
    ) -> None:
        ...

    @overload
    def init_on_sender(
        self,
        model_id: str,
        context: None = None,
        *,
        pipes: list = ...,
        num_workers: int | None = None,
        **kwargs,
    ) -> None:
        ...

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
        sender = MPWeightSender(self)
        sender._model_id = model_id
        if context is not None:
            sender._context_ref = weakref.ref(context)

        for worker_idx, pipe in enumerate(pipes):
            sender._register_worker(worker_idx, pipe)

        self._sender = sender
        self._initialized_on_sender = True

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

        Extracts weights from the collector's policy and sends them to all workers
        via pipes. This is called once after workers are initialized but before they
        start collecting data.

        Unlike send(), this does not wait for acknowledgments since workers are still
        in their initialization phase.

        Raises:
            RuntimeError: If no context is available or context has no policy.
        """
        # Get context (collector)
        context = self._context_ref() if self._context_ref is not None else None
        if context is None or not hasattr(context, "policy"):
            raise RuntimeError(
                "MPWeightSender requires context with policy for synchronize_weights()"
            )

        # Extract and prepare weights from the policy
        prepared_weights = self._scheme.prepare_weights(
            weights=context.policy,
            model_id=self._model_id,
            strategy=self._strategy,
            context=context,
        )

        # Send to all workers via pipes (no ACK - workers are still initializing)
        for transport in self._iterate_transports():
            if hasattr(transport, "send_weights_async"):
                transport.send_weights_async(prepared_weights, model_id=self._model_id)  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    f"Transport {type(transport)} does not support async send for synchronization"
                )
