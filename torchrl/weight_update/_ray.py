from __future__ import annotations

import socket

import time
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import torch
from tensordict import TensorDict
from tensordict.base import TensorDictBase

from torchrl._comm import RayRendezvous
from torchrl._utils import logger as torchrl_logger
from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    register_weight_sync_backend,
    TransportBackend,
    WeightStrategy,
    WeightSyncScheme,
)

# Default timeout for torch.distributed operations
_DIST_TIMEOUT = timedelta(seconds=60)


def _weight_tensors(
    weights: TensorDictBase | Mapping[str, torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(weights, TensorDictBase):
        keys = sorted(
            weights.keys(include_nested=True, leaves_only=True),
            key=lambda key: (key,) if isinstance(key, str) else tuple(key),
        )
        tensors = [weights.get(key) for key in keys]
    elif isinstance(weights, Mapping):
        tensors = [weights[key] for key in sorted(weights)]
    else:
        raise TypeError("Ray weight synchronization requires TensorDict or mapping.")
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("Ray weight synchronization only supports tensor leaves.")
    return tensors


def _transport_tensors(
    weights: TensorDictBase | Mapping[str, torch.Tensor], backend: str
) -> list[torch.Tensor]:
    tensors = _weight_tensors(weights)
    if backend == "gloo":
        return [tensor.detach().to("cpu") for tensor in tensors]
    return tensors


@dataclass
class ConnectionInfo:
    """Connection info for Ray distributed computing.

    Uses dataclass instead of UserDict to avoid Ray signature introspection
    issues with UserDict's __class_getitem__ in Python 3.11+
    (ValueError: no signature found for builtin type GenericAlias).
    """

    master_addr: str
    master_port: int
    world_size: int
    stateful_model: bool
    prefix: str

    def get(self, key: str, default: Any = None) -> Any:
        """Get a connection info value by key name.

        Args:
            key (str): The attribute name to retrieve.
            default: The default value if the attribute does not exist.
                Defaults to None.

        Returns:
            The value of the attribute, or the default if not found.
        """
        return getattr(self, key, default)


class RayTransport:
    """Ray transport for communicating with a single Ray actor.

    This transport handles weight updates for ONE specific remote actor
    using torch.distributed for efficient weight transfer. Ray is used for
    signaling/coordination, while the actual weight data is transferred via
    torch.distributed send/recv operations.

    Multiple transports are created for multiple actors, following the
    same pattern as multiprocess collectors.

    Args:
        remote_actor: The Ray actor handle for the remote collector/transform.
        worker_idx (int, optional): The worker index for this remote actor.
            Defaults to 0.
        backend (str): The torch.distributed backend to use ("gloo" or "nccl").
            Defaults to "gloo".
        connection_info_name (str): Name of the Ray actor storing connection info.
            Defaults to "connection_info".
        model_id (str, optional): The model identifier for weight synchronization.
    """

    def __init__(
        self,
        *,
        remote_actor=None,
        worker_idx: int | None = None,
        backend: str = "gloo",
        connection_info_name: str = "connection_info",
        model_id: str | None = None,
    ):
        """Initialize the RayTransport.

        Args:
            remote_actor: The Ray actor handle for the remote collector/transform.
            worker_idx (int, optional): The worker index for this remote actor.
                Defaults to 0.
            backend (str): The torch.distributed backend to use ("gloo" or "nccl").
                Defaults to "gloo".
            connection_info_name (str): Name of the Ray actor storing connection info.
                Defaults to "connection_info".
            model_id (str, optional): The model identifier for weight synchronization.
        """
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayTransport")
        self._remote_actor = remote_actor
        self._worker_idx = worker_idx if worker_idx is not None else 0
        self._backend = backend
        self._connection_info_name = connection_info_name
        self._model_id = model_id

        # Distributed state
        self._dist_initialized = False
        self._weights_buffer: TensorDictBase | None = None
        self._stateful_model: bool = True
        self._process_group = None
        self._store = None
        self._model_version: int | None = None

        # Async operation state
        self._pending_future = None
        self._pending_isend = None

        # Model reference (set by scheme on receiver side)
        self._model = None

    @property
    def _rank(self) -> int:
        """Get the torch.distributed rank for this worker.

        Returns:
            int: The rank (worker_idx + 1, since sender is rank 0).
        """
        return self._worker_idx + 1  # Sender is rank 0, workers are 1-indexed

    def set_model(self, model: Any) -> None:
        """Set the model for receiving weights.

        Args:
            model: The model to receive weights into.
        """
        self._model = model

    def set_process_group(self, process_group, store=None) -> None:
        """Attach the scheme-owned standalone process group."""
        self._process_group = process_group
        self._store = store

    def set_model_version(self, model_version: int | None) -> None:
        """Attach the semantic model version to the next publication."""
        self._model_version = model_version

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_process_group"] = None
        state["_store"] = None
        state["_pending_future"] = None
        state["_pending_isend"] = None
        return state

    # ========================================================================
    # Sending Weights (Sender Side)
    # ========================================================================

    def send_weights(self, weights: Any) -> None:
        """Send weights to the remote actor via torch.distributed.

        This method:
        1. Signals the remote actor to start receiving via Ray remote call
        2. Sends weights via torch.distributed.isend
        3. Waits for both to complete

        Args:
            weights: The weights to send (typically a TensorDict).
        """
        if self._remote_actor is None:
            return

        # Step 1: Signal the remote actor via Ray to start receiving (async)
        kwargs = (
            {}
            if self._model_version is None
            else {"model_version": self._model_version}
        )
        future = self._remote_actor._receive_weights_scheme.remote(**kwargs)

        # Step 2: Send weights via torch.distributed (async)
        for tag, tensor in enumerate(_transport_tensors(weights, self._backend)):
            self._process_group.send([tensor], self._rank, tag).wait()

        # Step 3: Wait for the Ray call to complete (receiver has applied weights)
        self.ray.get(future)

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to Ray actor without waiting for completion.

        Use :meth:`wait_ack` to wait for completion after sending to all actors.

        Args:
            weights: The weights to send (typically a TensorDict).
        """
        if self._remote_actor is None:
            return

        # Step 1: Signal the actor via Ray to start receiving (async)
        kwargs = (
            {}
            if self._model_version is None
            else {"model_version": self._model_version}
        )
        self._pending_future = self._remote_actor._receive_weights_scheme.remote(
            **kwargs
        )

        # Step 2: Send weights via torch.distributed (async)
        self._pending_isend = [
            self._process_group.send([tensor], self._rank, tag)
            for tag, tensor in enumerate(_transport_tensors(weights, self._backend))
        ]

    def wait_ack(self) -> None:
        """Wait for Ray actor to finish applying weights.

        Raises:
            RuntimeError: If no pending future exists (i.e., :meth:`send_weights_async`
                was not called before this method).
        """
        if self._pending_future is not None:
            self.ray.get(self._pending_future)
            if self._pending_isend is not None:
                for fut in self._pending_isend:
                    fut.wait()
            self._pending_future = None
            self._pending_isend = None
        else:
            raise RuntimeError("No pending future. Did you call send_weights_async?")

    # ========================================================================
    # Receiving Weights (Receiver Side)
    # ========================================================================

    def receive_weights(
        self,
        timeout: float | None = None,
        *,
        weights: Any = None,
        model: Any = None,
        strategy: WeightStrategy | None = None,
    ) -> Any | None:
        """Receive weights from sender via torch.distributed.

        Args:
            timeout: Maximum time to wait for weights (seconds). If None,
                blocks until weights are received.
            weights: Pre-allocated weight buffer to receive into.
            model: The model to apply weights to.
            strategy: Strategy for applying weights to the model.

        Returns:
            The received weights, or None if timeout expires.
        """
        from torchrl.collectors.utils import _cast

        # Use provided weights buffer or fallback to stored one
        weights_buffer = weights if weights is not None else self._weights_buffer
        if weights_buffer is None:
            if model is None:
                raise RuntimeError("No model available to receive weights")
            if isinstance(model, torch.nn.Module):
                weights_buffer = TensorDict.from_module(model)
                weights_buffer = weights_buffer.data.apply(_cast, weights_buffer)
            else:
                weights_buffer = TensorDict(lock=True)

        # Gloo transfers CPU tensors. Keep the communication buffer separate
        # from CUDA-backed module parameters and cast during strategy.apply_weights.
        if self._backend == "gloo":
            if isinstance(weights_buffer, TensorDictBase):
                weights_buffer = weights_buffer.to("cpu")
            elif isinstance(weights_buffer, Mapping):
                weights_buffer = {
                    key: value.to("cpu") for key, value in weights_buffer.items()
                }

        # Cache the weights buffer for future use
        if self._weights_buffer is None:
            self._weights_buffer = weights_buffer

        # Receive weights from rank 0
        works = [
            self._process_group.recv([tensor], 0, tag)
            for tag, tensor in enumerate(_weight_tensors(weights_buffer))
        ]
        try:
            for work in works:
                if timeout is None:
                    work.wait()
                else:
                    completed = work.wait(timedelta(seconds=timeout))
                    if completed is False:
                        return None
        except RuntimeError:
            if timeout is not None:
                return None
            raise

        # Apply weights to model
        if not isinstance(model, torch.nn.Module):
            if not weights_buffer.is_empty():
                raise RuntimeError(
                    f"Cannot cast weights to model type: {type(model)} with weights: {weights_buffer}."
                )
            return None

        if strategy is not None:
            strategy.apply_weights(model, weights_buffer)
        else:
            WeightStrategy(extract_as="tensordict").apply_weights(
                model, weights_buffer, inplace=False
            )

        return weights_buffer

    # ========================================================================
    # Connection Setup
    # ========================================================================

    def setup_connection_and_weights_on_sender(self) -> None:
        """Initialize torch.distributed on sender side for this worker's rank.

        This is called by the scheme after it has created the connection info
        Ray actor. The actual ``init_process_group`` happens in the scheme since
        it's a collective operation that needs to happen for rank 0.

        Note:
            This method exists for interface compatibility but the real work
            happens in the scheme's :meth:`_setup_distributed_connection_sender`.
        """
        # The scheme handles the collective init_process_group for rank 0.
        # This method exists for interface compatibility but the real work
        # happens in the scheme's _setup_distributed_connection_sender.

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int,
        strategy: WeightStrategy | None = None,
        model: Any | None = None,
        weights: Any | None = None,
    ) -> Any:
        """Join torch.distributed process group and receive initial weights.

        This method:
        1. Retrieves connection info from the shared Ray actor
        2. Initializes torch.distributed process group with rank=worker_idx+1
        3. Receives weights if model is stateful

        Args:
            worker_idx (int): The worker index for this transport.
            strategy (WeightStrategy, optional): The weight transmission strategy.
            model (nn.Module or compatible, optional): The model to receive weights for.
            weights (TensorDict, optional): Pre-allocated buffer for receiving weights.

        Returns:
            The received weights (TensorDict) if model is stateful, None otherwise.
        """
        if self._dist_initialized:
            # Already initialized, just receive weights if stateful
            if self._stateful_model:
                result = self.receive_weights(
                    weights=weights, model=model, strategy=strategy
                )
                return result[1] if result else None
            return None

        self._worker_idx = worker_idx
        rank = self._rank

        # Wait for connection info actor to be available
        i = 0
        while True:
            try:
                remote_connection_info = self.ray.get_actor(self._connection_info_name)
            except ValueError:
                i += 1
                time.sleep(0.1)
                continue
            break

        rendezvous = RayRendezvous(remote_connection_info)
        master_addr = rendezvous.read("master_addr")
        master_port = rendezvous.read("master_port")
        world_size = rendezvous.read("world_size")
        stateful_model = rendezvous.read("stateful_model")
        prefix = rendezvous.read("prefix")
        self._stateful_model = stateful_model
        store = torch.distributed.TCPStore(
            master_addr,
            master_port,
            world_size=None,
            is_master=False,
            timeout=_DIST_TIMEOUT,
        )
        prefix_store = torch.distributed.PrefixStore(prefix, store)
        if self._backend == "nccl":
            process_group = torch.distributed.ProcessGroupNCCL(
                prefix_store, rank, world_size
            )
        else:
            process_group = torch.distributed.ProcessGroupGloo(
                prefix_store, rank, world_size, _DIST_TIMEOUT
            )
        self.set_process_group(process_group, store)
        self._dist_initialized = True

        # Receive initial weights if model is stateful
        if self._stateful_model:
            return self.receive_weights(model=model, weights=weights, strategy=strategy)
        return None


@register_weight_sync_backend("ray")
class RayWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for Ray distributed computing.

    This scheme uses torch.distributed to synchronize weights across distributed
    workers (Ray actors). The process group is initialized during the first
    ``synchronize_weights()`` call, with the sender as rank 0 and workers as
    rank ``worker_idx + 1``.

    Each remote collector gets its own transport, following the same pattern
    as multiprocess collectors.

    Args:
        strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
            Defaults to "tensordict".
        backend (str): The torch.distributed backend to use ("gloo" or "nccl").
            Defaults to "gloo".
    """

    @property
    def connection_info_name(self) -> str:
        """Get the name of the Ray actor storing connection info.

        Returns a unique name based on model_id to avoid collisions when
        multiple schemes are used with different models.

        Returns:
            The connection info actor name.
        """
        model_id = self._model_id or "weights"
        return f"torchrl_weight_sync_{model_id}_{self._rendezvous_id}"

    def __init__(
        self,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
        backend: str = "gloo",
    ):
        """Initialize the RayWeightSyncScheme.

        Args:
            strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
                Defaults to "tensordict".
            backend (str): The torch.distributed backend to use ("gloo" or "nccl").
                Defaults to "gloo".
        """
        super().__init__(strategy)
        self._backend = backend
        self._dist_initialized = False
        self._remote_collectors: list | None = None
        self._num_workers: int = 0
        self._rendezvous_id = uuid.uuid4().hex
        self._process_group = None
        self._store = None
        self._manage_receiver_connect = False
        self._receiver_connect_futures = []
        self._model_version: int | None = None

    def _set_model_version(self, model_version: int) -> None:
        """Set the semantic version carried by the next publication."""
        self._model_version = model_version
        for transport in (self._sender_transports or {}).values():
            if isinstance(transport, RayTransport):
                transport.set_model_version(model_version)

    @property
    def model(self) -> Any | None:
        """Get the model associated with this scheme.

        Returns:
            The model if set, None otherwise.
        """
        if self._model_ref is not None:
            return self._model_ref()
        if self._model_id is not None:
            model = _resolve_model(self.context, self._model_id)
            if model is None:
                if self._model_id == "policy":
                    torchrl_logger.debug("Creating policy from factory.")
                    model = self.context.policy_factory[0]()
                    self.context.policy = model
                else:
                    raise AttributeError(
                        f"Model {self._model_id} was `None` in context {self.context}"
                    )
            self._model_ref = weakref.ref(model)
            return model

    @model.setter
    def model(self, value: Any):
        """Set the model for this scheme.

        Args:
            value: The model to set. If None, the setter is a no-op.
        """
        if value is None:
            return
        self._model_ref = weakref.ref(value)

    def create_transport(
        self,
        *,
        remote_actor=None,
        worker_idx: int | None = None,
        # Legacy parameter name for backwards compatibility
        remote_collector=None,
        **kwargs,
    ) -> TransportBackend:
        """Create Ray-based transport for a specific remote actor.

        Args:
            remote_actor: The Ray actor handle for the remote collector/transform.
            worker_idx: The worker index for this remote actor.
            remote_collector: Legacy alias for remote_actor.
            **kwargs: Additional transport configuration.

        Returns:
            RayTransport configured for this specific remote actor.
        """
        # Support legacy parameter name
        if remote_actor is None:
            remote_actor = remote_collector

        return RayTransport(
            remote_actor=remote_actor,
            worker_idx=worker_idx,
            backend=self._backend,
            connection_info_name=self.connection_info_name,
            model_id=self._model_id,
        )

    def _init_on_sender_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        This method sets up the torch.distributed connection info and shares it
        with all remote collectors so they can join the process group.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object providing remote_collectors
            **kwargs: Alternative to context (remote_collectors, source_model, etc.)
        """
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayWeightSyncScheme")

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

        # Store model_id and context on scheme
        self.model_id = model_id

        # Store remote collectors and num_workers for synchronize_weights
        self._remote_collectors = list(remote_collectors)
        self._num_workers = int(num_workers)

        # Register each Ray actor with explicit transport kwargs
        for worker_idx, remote_collector in enumerate(remote_collectors):
            transport = self.create_transport(
                remote_actor=remote_collector,
                worker_idx=worker_idx,
            )
            self._register_worker_sender(
                worker_idx=worker_idx,
                transport=transport,
            )

        # Set context with weak reference to avoid circular refs
        if context is not None:
            self.context = context

        # Store source model reference if provided for automatic weight extraction
        model = kwargs.get("model")
        if model is not None:
            self.model = model

        # Note: Distributed connection setup is deferred to synchronize_weights
        # because _receiver_schemes on workers won't exist until register_scheme_receiver is called

    def _init_on_receiver_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (typically the remote collector)
            **kwargs: Optional parameters (worker_idx, model, etc.)
        """
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayWeightSyncScheme")

        # Store model_id and context on scheme
        self.model_id = model_id
        self.context = context

        # Extract worker_idx from context or kwargs
        if context is not None:
            worker_idx = getattr(context, "worker_idx", None)
        else:
            worker_idx = kwargs.get("worker_idx")

        self._worker_idx = worker_idx

        # Resolve the target model on this worker
        model = kwargs.get("model")
        if model is not None:
            self.model = model
        # get the weights to possibly instantiate a copy of the model (policy factory with multi-collector)
        self.weights  # noqa

        # Create and register transport for receiver side
        # Note: create_transport returns TransportBackend but we know it's RayTransport
        transport = self.create_transport(
            remote_actor=None,  # Receiver doesn't need actor handle
            worker_idx=worker_idx,
        )
        if isinstance(transport, RayTransport):
            transport.set_model(model)
        self._register_transport_receiver(transport=transport)

    def _setup_distributed_connection_sender(
        self, timeout: float = 300.0, weights: Any | None = None
    ) -> None:
        """Set up torch.distributed connection info and share with remote collectors.

        This method:
        1. Gets master address and finds an available port
        2. Stores connection info in Ray's object store as a named actor
        3. Initializes torch.distributed process group with rank=0

        Args:
            timeout: Maximum time in seconds to wait for workers to be ready.
                Default is 300 seconds (5 minutes).
        """
        if self._dist_initialized:
            return

        if self._remote_collectors is None or self._num_workers == 0:
            raise RuntimeError(
                "_setup_distributed_connection() requires remote_collectors to be set"
            )

        # Get master address (hostname/IP)
        hostname = socket.gethostname()
        try:
            master_addr = socket.gethostbyname(hostname)
        except socket.gaierror:
            master_addr = "127.0.0.1"

        world_size = self._num_workers + 1  # +1 for the sender (rank 0)

        if weights is not None:
            self.weights = weights
            stateful_model = True
        else:
            try:
                stateful_model = self.weights is not None
            except (AttributeError, RuntimeError, ValueError):
                stateful_model = False
        self._stateful_model = stateful_model
        self._store = torch.distributed.TCPStore(
            master_addr,
            0,
            world_size=None,
            is_master=True,
            timeout=_DIST_TIMEOUT,
            wait_for_workers=False,
        )
        master_port = self._store.port
        prefix = f"weight-sync/{self._rendezvous_id}"

        # Connection info to share with workers via named Ray actor
        RemoteConnectionInfo = self.ray.remote(num_cpus=0)(ConnectionInfo).options(
            name=self.connection_info_name
        )
        self._connection_info_actor = RemoteConnectionInfo.remote(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            stateful_model=stateful_model,
            prefix=prefix,
        )
        if self._manage_receiver_connect:
            self._receiver_connect_futures = [
                actor._connect_weights_scheme.remote(model_version=self._model_version)
                for actor in self._remote_collectors
            ]
        prefix_store = torch.distributed.PrefixStore(prefix, self._store)
        if self._backend == "nccl":
            self._process_group = torch.distributed.ProcessGroupNCCL(
                prefix_store, 0, world_size
            )
        else:
            self._process_group = torch.distributed.ProcessGroupGloo(
                prefix_store, 0, world_size, _DIST_TIMEOUT
            )
        for transport in self.sender_transports.values():
            if isinstance(transport, RayTransport):
                transport.set_process_group(self._process_group, self._store)
                transport.set_model_version(self._model_version)
        self._dist_initialized = True

    def _setup_connection_and_weights_on_sender_impl(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any | None = None,
    ) -> None:
        """Set up distributed connection and send initial weights to all workers.

        This method:
        1. Sets up torch.distributed process group (waits for workers if needed)
        2. Sends initial weights to all workers via their transports

        The distributed setup is done here (not in ``init_on_sender``) because
        workers need to have ``register_scheme_receiver`` called first.

        Args:
            worker_idx (int, optional): Not used in this implementation.
            weights (optional): Not used in this implementation (weights are
                extracted from the model).
        """
        # Set up distributed connection (with wait for workers to be ready)
        if not self._dist_initialized:
            self._setup_distributed_connection_sender(weights=weights)

        # Send the initial weights
        if weights is not None:
            self.weights = weights
            self._stateful_model = True
        if self._stateful_model:
            self._send_weights_distributed(weights)
        if self._receiver_connect_futures:
            self.ray.get(self._receiver_connect_futures)
            self._receiver_connect_futures = []

    def _send_weights_distributed(self, weights=None) -> None:
        """Send weights to all workers via torch.distributed.

        Raises:
            RuntimeError: If no weights are available to send.
        """
        # Extract weights from model
        weights = self.weights if weights is None else weights
        if weights is None:
            raise RuntimeError("No weights available to send")

        # Send weights to each worker (ranks 1 to num_workers)
        futures = []
        tensors = _transport_tensors(weights, self._backend)
        for worker_idx in range(self._num_workers):
            rank = worker_idx + 1
            futures.extend(
                self._process_group.send([tensor], rank, tag)
                for tag, tensor in enumerate(tensors)
            )
        # Wait for all sends to complete
        for future in futures:
            future.wait()

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Join torch.distributed process group and receive initial weights.

        Delegates to the transport's :meth:`~RayTransport.setup_connection_and_weights_on_receiver`.

        Args:
            worker_idx (int, optional): The worker index. If None, uses the stored
                ``_worker_idx`` or defaults to 0.
        """
        if worker_idx is None:
            worker_idx = self._worker_idx
        if worker_idx is None:
            worker_idx = 0  # Default to worker 0

        transport = self.receiver_transport
        if transport is not None:
            # Transport handles joining process group and receiving weights
            transport.setup_connection_and_weights_on_receiver(
                worker_idx=worker_idx,
                model=self.model,
                weights=self.weights,
                strategy=self._strategy,
            )
            self._dist_initialized = True

    def shutdown(self) -> None:
        """Release only this scheme's standalone communication resources."""
        super().shutdown()
        actor = getattr(self, "_connection_info_actor", None)
        if actor is not None and hasattr(self, "ray"):
            try:
                self.ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self._connection_info_actor = None
        for transport in (self._sender_transports or {}).values():
            if isinstance(transport, RayTransport):
                transport.set_process_group(None, None)
        if isinstance(self._receiver_transport, RayTransport):
            self._receiver_transport.set_process_group(None, None)
        self._process_group = None
        self._store = None
        self._receiver_connect_futures = []
        self._dist_initialized = False

    def __getstate__(self):
        state = super().__getstate__()
        state["_process_group"] = None
        state["_store"] = None
        state["_connection_info_actor"] = None
        state["_dist_initialized"] = False
        state["_receiver_connect_futures"] = []
        return state


class RayModuleTransformScheme(RayWeightSyncScheme):
    """Weight synchronization for RayModuleTransform.

    This scheme uses torch.distributed to synchronize weights between
    a trainer/collector and a RayModuleTransform actor. The sender is rank 0,
    the transform's actor is rank 1.

    This enables updating the weights of a module running inside a RayModuleTransform
    from a parent collector or training loop.

    Args:
        strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
            Default is "tensordict".
        backend (str): The torch.distributed backend to use ("gloo" or "nccl").
            Default is "gloo".

    Example:
        >>> # Create scheme and transform
        >>> scheme = RayModuleTransformScheme()
        >>> transform = RayModuleTransform(module=my_module, weight_sync_scheme=scheme)
        >>>
        >>> # Create env with transform
        >>> env = TransformedEnv(base_env, transform)
        >>>
        >>> # Pass scheme to parent collector
        >>> collector = SomeCollector(
        ...     env, policy,
        ...     weight_sync_schemes={"transform_module": scheme}
        ... )
        >>>
        >>> # Update weights
        >>> collector.update_policy_weights_(model_id="transform_module")
    """

    def __init__(
        self,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
        backend: str = "gloo",
    ):
        """Initialize the RayModuleTransformScheme.

        Args:
            strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
                Defaults to "tensordict".
            backend (str): The torch.distributed backend to use ("gloo" or "nccl").
                Defaults to "gloo".
        """
        super().__init__(strategy, backend)
        self._ray_transform = None

    def _set_transform(self, ray_transform) -> None:
        """Store reference to the RayModuleTransform.

        Called by RayModuleTransform when the scheme is passed to it.

        Args:
            ray_transform: The RayModuleTransform instance.
        """
        self._ray_transform = ray_transform

    def _init_on_sender_impl(
        self,
        model_id: str | None = None,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Uses the stored transform reference (set via _set_transform) to
        create transport for the transform's actor.

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (typically the collector)
            **kwargs: Optional parameters (ray_transform, model, etc.)
        """
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayModuleTransformScheme")

        # Get transform reference - either stored via _set_transform or from kwargs
        ray_transform = self._ray_transform
        if ray_transform is None:
            ray_transform = kwargs.get("ray_transform")
        if ray_transform is None:
            raise ValueError(
                "ray_transform must be set via _set_transform() or provided in kwargs. "
                "Pass the scheme to RayModuleTransform constructor to set it automatically."
            )

        # Store model_id
        self.model_id = model_id

        # Single worker (the transform's actor)
        self._num_workers = 1
        self._remote_collectors = [ray_transform._actor]

        # Create transport for the transform's actor
        # The actor handle is ray_transform._actor
        transport = self.create_transport(
            remote_actor=ray_transform._actor,
            worker_idx=0,
        )
        self._register_worker_sender(
            worker_idx=0,
            transport=transport,
        )

        # Set context if provided
        if context is not None:
            self.context = context

        # Store source model reference if provided for automatic weight extraction
        model = kwargs.get("model")
        if model is not None:
            self.model = model

    def _init_on_receiver_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the transform's actor (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: The ModuleTransform instance (the actor's underlying class)
            **kwargs: Optional parameters (worker_idx, model, etc.)
        """
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayModuleTransformScheme")

        # Store model_id and context
        self.model_id = model_id
        self.context = context

        # Single transform actor is always worker_idx=0
        self._worker_idx = kwargs.get("worker_idx", 0)

        # Resolve the target model from context (ModuleTransform has a .module attribute)
        model = kwargs.get("model")
        if model is None and context is not None:
            model = getattr(context, "module", None)
        if model is not None:
            self.model = model

        # Create and register transport for receiver side
        # Note: create_transport returns TransportBackend but we know it's RayTransport
        transport = self.create_transport(
            remote_actor=None,
            worker_idx=self._worker_idx,
        )
        if isinstance(transport, RayTransport):
            transport.set_model(model)
        self._register_transport_receiver(transport=transport)

    def _setup_connection_and_weights_on_sender_impl(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any | None = None,
    ) -> None:
        """Set up distributed connection and send initial weights.

        Args:
            worker_idx (int, optional): The worker index. Not used for
                RayModuleTransformScheme as there is only one transform actor.
            weights (optional): Pre-extracted weights to send. If None, weights
                are extracted from the model.
        """
        receiver_future = self._ray_transform._actor._init_weight_sync_scheme.remote(
            scheme=self, model_id=self.model_id
        )

        super()._setup_connection_and_weights_on_sender_impl(weights=weights)

        self.ray.get(receiver_future)
