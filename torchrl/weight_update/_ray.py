from __future__ import annotations

import os
import socket

import time
from collections import UserDict
from datetime import timedelta
from typing import Any, Literal

import torch
from tensordict import TensorDict
from tensordict.base import TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import TransportBackend, WeightSyncScheme

# Default timeout for torch.distributed operations
_DIST_TIMEOUT = timedelta(seconds=60)


class ConnectionInfo(UserDict):
    ...


class RayWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization for Ray distributed computing.

    This scheme uses torch.distributed to synchronize weights across distributed
    workers (Ray actors). The process group is initialized during the first
    synchronize_weights() call, with the sender as rank 0 and workers as
    rank worker_idx+1.

    Each remote collector gets its own transport, following the same pattern
    as multiprocess collectors.

    Args:
        strategy (str): The weight transmission strategy ("state_dict" or "tensordict").
            Default is "tensordict".
        backend (str): The torch.distributed backend to use ("gloo" or "nccl").
            Default is "gloo".
    """

    def __init__(
        self,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
        backend: str = "gloo",
    ):
        super().__init__(strategy)
        self._backend = backend
        self._dist_initialized = False
        self._weights_buffer: TensorDictBase | None = None
        self._remote_collectors: list | None = None
        self._num_workers: int = 0

    def create_transport(
        self,
        *,
        remote_collector=None,
        worker_idx: int | None = None,
        **kwargs,
    ) -> TransportBackend:
        """Create Ray-based transport for a specific remote collector.

        Args:
            remote_collector: The Ray actor handle for the remote collector.
            worker_idx: The worker index for this remote collector.
            **kwargs: Additional transport configuration.

        Returns:
            RayTransport configured for this specific remote collector.
        """
        return RayTransport(
            remote_collector=remote_collector,
            worker_idx=worker_idx,
        )

    def _init_on_sender_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        This method se up the torch.distributed connection info and shares it
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
                remote_collector=remote_collector,
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
        if model is None and context is not None:
            model = _resolve_model(context, model_id)
        if model is not None:
            self.model = model

    def _setup_distributed_connection_sender(self, timeout: float = 300.0) -> None:
        """Set up torch.distributed connection info and share with remote collectors.

        This method:
        1. Waits for workers to have _receiver_schemes registered (with timeout)
        2. Gets master address and finds an available port
        3. Stores connection info in Ray's object store
        4. Shares connection info with all remote collectors via cascade_execute
        5. Initializes torch.distributed process group with rank=0

        This is called from synchronize_weights to ensure workers have had
        register_scheme_receiver called before we try to reach their schemes.

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

        # Find an available port
        master_port = self._find_free_port()
        world_size = self._num_workers + 1  # +1 for the sender (rank 0)

        torchrl_logger.debug(
            f"RayWeightSyncScheme: Setting up distributed connection with "
            f"master_addr={master_addr}, master_port={master_port}, world_size={world_size}"
        )

        try:
            self.weights
            stateful_model = True
        except (AttributeError, RuntimeError, ValueError):
            stateful_model = False
        self._stateful_model = stateful_model

        # Connection info to share with workers
        RemoteConnectionInfo = self.ray.remote(num_cpus=0)(ConnectionInfo).options(
            name="connection_info"
        )
        connection_info = RemoteConnectionInfo.remote(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            stateful_model=stateful_model,
        )

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Initialize process group on sender (rank 0)
        # Note: Workers will call init_process_group in _set_dist_connection_info
        # which is triggered by the remote calls above. The init_process_group is
        # a collective operation, so all ranks must call it together.
        torchrl_logger.debug(
            "RayWeightSyncScheme: Initializing process group on sender (rank 0) -- blocking."
        )
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=0,
            world_size=world_size,
            timeout=_DIST_TIMEOUT,
        )
        self._dist_initialized = True

        torchrl_logger.debug(
            "RayWeightSyncScheme: Distributed connection setup complete -- all workers at rendez-vous"
        )

    def _setup_distributed_connection_receiver(self):
        # Get connection info, if not existent wait
        worker_idx = self._worker_idx
        rank = worker_idx + 1  # Sender is rank 0, workers are 1-indexed
        i = 0
        while True:
            try:
                remote_connection_info = self.ray.get_actor("connection_info")
            except ValueError:
                i += 1
                time.sleep(0.1)
                if i % 50 == 0:
                    torchrl_logger.debug(
                        f"RayWeightSyncScheme: Waiting for connection info (attempt {i}) on {worker_idx=}/{rank=}"
                    )
                continue
            break

        master_addr = self.ray.get(remote_connection_info.get.remote("master_addr"))
        master_port = self.ray.get(remote_connection_info.get.remote("master_port"))
        world_size = self.ray.get(remote_connection_info.get.remote("world_size"))
        stateful_model = self.ray.get(
            remote_connection_info.get.remote("stateful_model")
        )
        self._stateful_model = stateful_model

        torchrl_logger.debug(
            f"RayWeightSyncScheme: Worker {worker_idx} joining process group with "
            f"rank={rank}, master_addr={master_addr}, master_port={master_port} -- blocking"
        )

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Initialize process group on receiver
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=rank,
            world_size=world_size,
        )
        torchrl_logger.debug(f"RayWeightSyncScheme: Worker {worker_idx} joined process group")
        self._dist_initialized = True

    def _setup_connection_and_weights_on_sender_impl(
        self, *, worker_idx: int | None = None, weights: Any | None = None,
    ) -> None:
        """Set up distributed connection and send initial weights to all workers.

        This method:
        1. Sets up torch.distributed process group (waits for workers if needed)
        2. Sends initial weights to all workers

        The distributed setup is done here (not in init_on_sender) because
        workers need to have register_scheme_receiver called first.
        """

        # Set up distributed connection (with wait for workers to be ready)
        if not self._dist_initialized:
            torchrl_logger.debug(
                "RayWeightSyncScheme: Setting up distributed connection (sender)"
            )
            self._setup_distributed_connection_sender()

        # Send the initial weights
        if self._stateful_model:
            self._send_weights_distributed()

    def _send_weights_distributed(self) -> None:
        """Send weights to all workers via torch.distributed."""
        # Extract weights from model
        weights = self.weights
        if weights is None:
            raise RuntimeError("No weights available to send")

        # Send weights to each worker (ranks 1 to num_workers)
        futures = []
        for worker_idx in range(self._num_workers):
            rank = worker_idx + 1
            torchrl_logger.debug(f"RayWeightSyncScheme: Sending weights to rank {rank}")
            futures.extend(weights.isend(dst=rank, return_early=True))
        # Wait for all sends to complete
        for future in futures:
            future.wait()

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Join torch.distributed process group and receive initial weights.

        This method:
        1. Retrieves connection info from the shared Ray object reference
        2. Initializes torch.distributed process group with rank=worker_idx+1
        3. Creates weights buffer from model
        4. Receives weights via irecv and applies them to model
        """
        # Set up distributed connection (with wait for workers to be ready)
        if not self._dist_initialized:
            torchrl_logger.debug(
                "RayWeightSyncScheme: Setting up distributed connection (sender)"
            )
            self._setup_distributed_connection_receiver()

        if self._stateful_model:
            # Already initialized, just receive weights
            self._receive_weights_distributed()
        return

    def receive(self, timeout: float = 0.001) -> TensorDict:
        self._receive_weights_distributed()
        return self._weights_buffer

    def _receive_weights_distributed(self) -> None:
        """Receive weights from sender via torch.distributed and apply to model."""
        from torchrl.collectors.utils import _cast

        # Create weights buffer from model if not already created
        if self._weights_buffer is None:
            model = self.model
            if model is None:
                raise RuntimeError("No model available to receive weights")
            if isinstance(model, torch.nn.Module):
                self._weights_buffer = TensorDict.from_module(model)
                self._weights_buffer = self._weights_buffer.data.apply(
                    _cast, self._weights_buffer
                )
            else:
                self._weights_buffer = TensorDict(lock=True)

        # Receive weights from rank 0
        torchrl_logger.debug(
            f"RayWeightSyncScheme: Receiving weights from rank 0: {self._weights_buffer=}"
        )
        self._weights_buffer.irecv(src=0)

        # Apply weights to model
        model = self.model
        if not isinstance(model, torch.nn.Module):
            if not self._weights_buffer.is_empty():
                raise RuntimeError(
                    f"Cannot cast weights to model type: {type(model)} with weights: {self._weights_buffer}."
                )
            torchrl_logger.debug("RayWeightSyncScheme: No weights to apply to model")
            return
        self._strategy.apply_weights(model, self._weights_buffer)
        torchrl_logger.debug("RayWeightSyncScheme: Weights applied to model")

    @staticmethod
    def _find_free_port() -> int:
        """Find a free port on the local machine."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _set_dist_connection_info(self, connection_info, worker_idx: int) -> None:
        """Set torch.distributed connection info and join the process group.

        This method is called remotely via cascade_execute to share connection info
        (master_addr, master_port, world_size) with this scheme instance. The worker
        joins the torch.distributed process group here so that the sender's
        init_process_group call can complete (it's a collective operation).

        Args:
            connection_info: Connection info dict (Ray auto-resolves ObjectRefs when
                passing to remote methods, so this is already a dict)
            worker_idx: The worker index for this scheme
        """
        # Store worker_idx
        self._worker_idx = worker_idx

        # connection_info is already a dict (Ray auto-resolves ObjectRefs)
        master_addr = connection_info["master_addr"]
        master_port = connection_info["master_port"]
        world_size = connection_info["world_size"]

        rank = worker_idx + 1  # Sender is rank 0, workers are 1-indexed

        torchrl_logger.debug(
            f"RayWeightSyncScheme: Worker {worker_idx} joining process group with "
            f"rank={rank}, master_addr={master_addr}, master_port={master_port}, world_size={world_size}"
        )

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Join the process group (rendezvous with sender)
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=rank,
            world_size=world_size,
            timeout=_DIST_TIMEOUT,
        )
        self._dist_initialized = True

        torchrl_logger.debug(
            f"RayWeightSyncScheme: Worker {worker_idx} joined process group as rank {rank}"
        )


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
        *,
        model_id: str | None=None,
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

        # Create transport for the transform's actor
        # The actor handle is ray_transform._actor
        transport = self.create_transport(
            remote_collector=ray_transform._actor,
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

    def _setup_distributed_connection_sender(self, timeout: float = 300.0) -> None:
        """Set up torch.distributed for the single transform actor.

        Overrides parent to work with a single RayModuleTransform instead of
        multiple remote collectors.
        """
        if self._dist_initialized:
            return

        if self._ray_transform is None:
            raise RuntimeError(
                "_setup_distributed_connection() requires ray_transform to be set. "
                "Did you pass the scheme to RayModuleTransform?"
            )

        # Get master address (hostname/IP)
        hostname = socket.gethostname()
        try:
            master_addr = socket.gethostbyname(hostname)
        except socket.gaierror:
            master_addr = "127.0.0.1"

        # Find an available port
        master_port = self._find_free_port()
        world_size = 2  # Sender (rank 0) + Transform (rank 1)

        torchrl_logger.debug(
            f"RayModuleTransformScheme: Setting up distributed connection with "
            f"master_addr={master_addr}, master_port={master_port}, world_size={world_size}"
        )

        # Check if model has weights
        try:
            w = self.weights
            stateful_model = w is not None
        except (AttributeError, RuntimeError, ValueError):
            stateful_model = False
        self._stateful_model = stateful_model

        # Connection info to share with the transform's actor
        RemoteConnectionInfo = self.ray.remote(num_cpus=0)(ConnectionInfo).options(
            name="connection_info_transform"
        )
        self._connection_info_actor = RemoteConnectionInfo.remote(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            stateful_model=stateful_model,
        )

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Now initialize process group on sender (rank 0)
        # The receiver is concurrently joining via the Ray call above
        torchrl_logger.debug(
            "RayModuleTransformScheme: Initializing process group on sender (rank 0) -- blocking."
        )
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=0,
            world_size=world_size,
            timeout=_DIST_TIMEOUT,
        )
        self._dist_initialized = True

        torchrl_logger.debug(
            "RayModuleTransformScheme: Distributed connection setup complete"
        )

    def _setup_distributed_connection_receiver(self) -> None:
        """Join torch.distributed process group on the transform's actor side."""
        worker_idx = self._worker_idx
        rank = worker_idx + 1  # Sender is rank 0, transform is rank 1
        i = 0
        while True:
            try:
                remote_connection_info = self.ray.get_actor("connection_info_transform")
            except ValueError:
                i += 1
                time.sleep(0.1)
                if i % 50 == 0:
                    torchrl_logger.debug(
                        f"RayModuleTransformScheme: Waiting for connection info "
                        f"(attempt {i}) on {worker_idx=}/{rank=}"
                    )
                continue
            break

        master_addr = self.ray.get(remote_connection_info.get.remote("master_addr"))
        master_port = self.ray.get(remote_connection_info.get.remote("master_port"))
        world_size = self.ray.get(remote_connection_info.get.remote("world_size"))
        stateful_model = self.ray.get(
            remote_connection_info.get.remote("stateful_model")
        )
        self._stateful_model = stateful_model

        torchrl_logger.debug(
            f"RayModuleTransformScheme: Transform actor joining process group with "
            f"rank={rank}, master_addr={master_addr}, master_port={master_port}"
        )

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Initialize process group on receiver
        torch.distributed.init_process_group(
            backend=self._backend,
            rank=rank,
            world_size=world_size,
        )
        self._dist_initialized = True

    def _setup_connection_and_weights_on_sender_impl(
        self, *, worker_idx: int | None = None, weights: Any | None = None,
    ) -> None:
        """Set up distributed connection (no initial weight send)."""

        torchrl_logger.debug(
            "RayModuleTransformScheme: Signaling receiver to join process group"
        )
        receiver_future = self._ray_transform._actor._init_weight_sync_scheme.remote(scheme=self, model_id=self.model_id)

        if not self._dist_initialized:
            torchrl_logger.debug(
                "RayModuleTransformScheme: Setting up distributed connection (sender)"
            )
            self._setup_distributed_connection_sender()

        if self._stateful_model:
            torchrl_logger.debug(
                "RayModuleTransformScheme: Sending first batch of weights (sender)"
            )
            self._send_weights_distributed(weights=weights)

        torchrl_logger.debug("Waiting for receiver to join process group...")
        self.ray.get(receiver_future)

    def _send_weights_distributed(self, weights: Any | None = None) -> None:
        """Send weights to the transform actor via torch.distributed."""
        if weights is None:
            weights = self.weights
        if weights is None:
            raise RuntimeError("No weights available to send")

        # Send weights to the transform (rank 1)
        torchrl_logger.debug("RayModuleTransformScheme: Sending weights to rank 1")
        futures = weights.isend(dst=1, return_early=True)
        for future in futures:
            future.wait()

    def _setup_connection_and_weights_on_receiver_impl(
        self, *, worker_idx: int | None = None
    ) -> None:
        """Receive weights on the RayModuleTransform actor."""
        # Set up distributed connection if not already done
        if not self._dist_initialized:
            torchrl_logger.debug(
                "RayModuleTransformScheme: Setting up distributed connection (receiver)"
            )
            self._setup_distributed_connection_receiver()

        # Receive weights if model has weights
        if getattr(self, "_stateful_model", True):
            torchrl_logger.debug(
                "RayModuleTransformScheme: Receiving first batch of weights (receiver)"
            )
            self._receive_weights_distributed()

    def _receive_weights_distributed(self) -> None:
        """Receive weights from sender via torch.distributed and apply to model."""
        weights = self.weights
        if weights is None:
            raise RuntimeError("No weights template available")

        # Receive weights from sender (rank 0)
        torchrl_logger.debug("RayModuleTransformScheme: Receiving weights from rank 0")
        weights.irecv(src=0)

        # Apply weights to model
        torchrl_logger.debug("RayModuleTransformScheme: Applying weights to model")
        weights.to_module(self.model)

    def create_transport(
        self,
        *,
        remote_collector=None,
        worker_idx: int | None = None,
        **kwargs,
    ) -> TransportBackend:
        """Create Ray-based transport for the transform's actor.

        Args:
            remote_collector: The Ray actor handle for the transform.
            worker_idx: The worker index (always 0 for single transform).
            **kwargs: Additional transport configuration.

        Returns:
            RayModuleTransformTransport configured for this transform.
        """
        return RayModuleTransformTransport(
            ray_actor=remote_collector,
            worker_idx=worker_idx,
        )


class RayTransport:
    """Ray transport for communicating with a single Ray collector actor.

    This transport handles weight updates for ONE specific remote collector
    using torch.distributed for efficient weight transfer. Ray is used for
    signaling/coordination, while the actual weight data is transferred via
    torch.distributed send/recv operations.

    Multiple transports are created for multiple collectors, following the
    same pattern as multiprocess collectors.
    """

    def __init__(
        self,
        remote_collector=None,
        worker_idx: int | None = None,
    ):
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayTransport")
        self._remote_collector = remote_collector
        self._worker_idx = worker_idx
        self._pending_future = None

    @property
    def _rank(self) -> int:
        """Get the torch.distributed rank for this worker."""
        if self._worker_idx is None:
            raise RuntimeError("worker_idx must be set before sending weights")
        return self._worker_idx + 1  # Sender is rank 0, workers are 1-indexed

    def send_weights(self, weights: Any) -> None:
        """Send weights to the remote collector via torch.distributed.

        This method:
        1. Signals the remote collector to start receiving via Ray remote call
        2. Sends weights via torch.distributed.isend
        3. Waits for both to complete
        """
        if self._remote_collector is None:
            return

        # Step 1: Signal the remote collector via Ray to start receiving (async)
        future = self._remote_collector._receive_weights_scheme.remote()

        # Step 2: Send weights via torch.distributed (async)
        torchrl_logger.debug(f"RayTransport: Sending weights to rank {self._rank}")
        weights.isend(dst=self._rank)

        # Step 3: Wait for the Ray call to complete (receiver has applied weights)
        self.ray.get(future)

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to Ray actor without waiting for completion.

        Use wait_ack() to wait for completion after sending to all actors.
        """
        if self._remote_collector is None:
            return

        # Step 1: Signal the actor via Ray to start receiving (async)
        torchrl_logger.debug(
            f"RayActorTransport: Sending weights async to rank {self._rank}"
        )
        self._pending_future = self._remote_collector._receive_weights_scheme.remote()

        # Step 2: Send weights via torch.distributed (async)
        torchrl_logger.debug(
            f"RayActorTransport: Sending weights async to rank {self._rank}"
        )
        self._pending_isend = weights.isend(dst=self._rank, return_early=True)
        torchrl_logger.debug(f"RayActorTransport: Async send initiated")

    def wait_ack(self) -> None:
        """Wait for Ray actor to finish applying weights."""
        if self._pending_future is not None:
            torchrl_logger.debug(
                f"RayActorTransport: Waiting for ack from rank {self._rank}"
            )
            self.ray.get(self._pending_future)
            torchrl_logger.debug(
                f"RayActorTransport: Ack received from rank {self._rank}. Waiting for isend to complete."
            )
            for fut in self._pending_isend:
                fut.wait()
            self._pending_future = None
        else:
            raise RuntimeError("No pending future. Did you call send_weights_async?")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Ray workers receive weights via torch.distributed in the scheme."""
        return None

    def check_connection(self) -> bool:
        """Check if Ray and torch.distributed are initialized."""
        return self.ray.is_initialized() and torch.distributed.is_initialized()

    def setup_connection_and_weights_on_sender(self) -> None:
        """No-op for RayTransport - synchronization is handled by the scheme."""

    def setup_connection_and_weights_on_receiver(self, worker_idx: int) -> Any:
        """No-op for RayTransport - synchronization is handled by the scheme."""
        return None


class RayModuleTransformTransport:
    """Transport for communicating with a RayModuleTransform actor.

    This transport handles weight updates for a RayModuleTransform actor
    using torch.distributed for efficient weight transfer. Ray is used for
    signaling/coordination, while the actual weight data is transferred via
    torch.distributed send/recv operations.
    """

    def __init__(
        self,
        ray_actor=None,
        worker_idx: int | None = None,
    ):
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is required for RayModuleTransformTransport")
        self._ray_actor = ray_actor
        self._worker_idx = worker_idx if worker_idx is not None else 0
        self._pending_future = None
        self._pending_isend = None

    @property
    def _rank(self) -> int:
        """Get the torch.distributed rank for the transform actor."""
        return self._worker_idx + 1  # Sender is rank 0, transform is rank 1

    def send_weights(self, weights: Any) -> None:
        """Send weights to the transform actor via torch.distributed.

        This method:
        1. Signals the transform actor to start receiving via Ray remote call
        2. Sends weights via torch.distributed.isend
        3. Waits for both to complete
        """
        if self._ray_actor is None:
            return

        # Step 1: Signal the actor via Ray to start receiving (async)
        future = self._ray_actor._receive_weights_scheme.remote()

        # Step 2: Send weights via torch.distributed (async)
        torchrl_logger.debug(
            f"RayModuleTransformTransport -- RANK 0: Sending weights to rank {self._rank}"
        )
        weights.isend(dst=self._rank)

        # Step 3: Wait for the Ray call to complete (receiver has applied weights)
        self.ray.get(future)

    def send_weights_async(self, weights: Any) -> None:
        """Send weights to transform actor without waiting for completion.

        Use wait_ack() to wait for completion after sending.
        """
        if self._ray_actor is None:
            return

        # Step 1: Signal the actor via Ray to start receiving (async)
        torchrl_logger.debug(
            f"RayModuleTransformTransport -- RANK 0: Sending weights async to rank {self._rank}"
        )
        self._pending_future = self._ray_actor._receive_weights_scheme.remote()

        # Step 2: Send weights via torch.distributed (async)
        self._pending_isend = weights.isend(dst=self._rank, return_early=True)
        torchrl_logger.debug("RayModuleTransformTransport -- RANK 0: Async send initiated")

    def wait_ack(self) -> None:
        """Wait for transform actor to finish applying weights."""
        if self._pending_future is not None:
            torchrl_logger.debug(
                f"RayModuleTransformTransport -- RANK 0: Waiting for ack from rank {self._rank}"
            )
            self.ray.get(self._pending_future)
            torchrl_logger.debug(
                f"RayModuleTransformTransport -- RANK 0: Ack received from rank {self._rank}. "
                "Waiting for isend to complete."
            )
            if self._pending_isend is not None:
                for fut in self._pending_isend:
                    fut.wait()
            self._pending_future = None
            self._pending_isend = None
        else:
            raise RuntimeError("No pending future. Did you call send_weights_async?")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Transform actors receive weights via torch.distributed in the scheme."""
        return None

    def check_connection(self) -> bool:
        """Check if Ray and torch.distributed are initialized."""
        return self.ray.is_initialized() and torch.distributed.is_initialized()

    def setup_connection_and_weights_on_sender(self) -> None:
        """No-op - synchronization is handled by the scheme."""

    def setup_connection_and_weights_on_receiver(self, worker_idx: int) -> Any:
        """No-op - synchronization is handled by the scheme."""
        return None


class RayActorTransport:
    ...
