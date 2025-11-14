from __future__ import annotations

import weakref
from typing import Any, Literal, overload

from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


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
        remote_collectors: list = ...,
        num_workers: int | None = None,
        source_model: Any | None = None,
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
        actor_refs: list | None = None,
        actors: list | None = None,
        remote_collectors: list | None = None,
        source_model: Any | None = None,
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
        actor_ref: Any | None = None,
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

        This is now handled by init_on_receiver(). Only kept for internal use.

        Args:
            actor_or_context: Either a Ray actor reference or a context object.
        """
        self._transport = self._scheme.create_transport(actor_or_context)

    def apply_weights(self, weights: Any, inplace: bool = True) -> None:
        """Apply received weights to registered model.

        For Ray actors, weights are applied directly to the module
        within the actor's process space.

        Args:
            weights: The weights to apply.
            inplace: Whether to apply weights in place. Default is `True`.
        """
        if self._model_ref is None:
            raise ValueError("No model registered")

        model = self._resolve_model_ref()
        self._strategy.apply_weights(model, weights, inplace=inplace)


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
