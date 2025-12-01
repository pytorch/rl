# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM weight synchronization for the v2 API.

This module provides weight synchronization for vLLM engines using a two-layer
architecture:

**Architecture Overview**

The weight synchronization uses two separate layers:

1. **RPC Layer** (Coordination)
   - Signals workers when a collective operation will begin
   - Can be implemented with different backends (Ray, torch.distributed.rpc, etc.)
   - Tells vLLM workers: "prepare to receive weights via collective"
   - Currently supports Ray as the RPC backend

2. **Collective Layer** (Data Transfer)
   - Performs the actual weight broadcast using NCCL
   - High-bandwidth GPU-to-GPU communication
   - All ranks participate simultaneously in the collective

**Why Two Layers?**

Separating RPC and collectives provides:
- **Flexibility**: Swap RPC backends (Ray, RPC, gRPC) without changing collectives
- **Clarity**: Coordination logic separate from data transfer
- **Performance**: Use optimal transport for each (RPC for signals, NCCL for data)

**Flow Example (Ray Backend)**

.. code-block::

    Trainer (rank 0)                    vLLM Workers (ranks 1+)
    ================                    =======================

    # 1. RPC: Signal collective start
    trainer.update_weights() ---------> [Ray RPC] --------> receiver.init_all_workers_group()
                                                            "I'm ready for collective"

    # 2. Collective: Broadcast weights
    NCCL broadcast -------------------- [GPU-GPU] ---------> NCCL receive
    (high bandwidth)                                        (parallel)

    # 3. RPC: Confirmation (optional)
    "broadcast done" <----------------- [Ray RPC] --------- "weights applied"

**Extending to Other Backends**

To add a new RPC backend (e.g., torch.distributed.rpc):

1. Implement an RPC coordinator in the sender/receiver
2. Replace Ray remote calls with your RPC mechanism
3. Keep the collective layer unchanged (it's backend-agnostic)

.. rubric:: Example

.. code-block:: python

    class TorchRPCVLLMReceiver(VLLMWeightReceiver):
        def init_all_workers_group(self, metadata):
            # Use torch.distributed.rpc instead of Ray
            torch.distributed.rpc.rpc_sync(
                "trainer",
                lambda: "ready",
            )
            super().init_all_workers_group(metadata)  # Collective init

**Current Implementation (Ray Backend)**

The test suite in ``test_weightsync.py`` demonstrates the Ray-based RPC:

.. code-block:: python

    # Trainer actor (provides RPC endpoint)
    trainer = RayWorkerTransformer.as_remote().options(
        name="Trainer"  # Named for discovery
    ).remote(scheme_config)

    # Receiver actor (uses RPC to coordinate)
    receiver = RayWorkerVLLM.as_remote().remote(
        scheme_config, trainer_actor_name="Trainer"
    )

    # RPC Layer: Both actors call init() via Ray remote calls
    # This coordinates the collective handshake
    ray.get([trainer.init.remote(), receiver.init.remote()])

    # RPC Layer: Trigger update via Ray remote call
    # Collective Layer: NCCL broadcast happens automatically
    ray.get(trainer.update_weights.remote(modify_weights=True))

In this setup:
- **Ray provides RPC**: Named actors, ``remote()`` calls, ``ray.get()``
- **NCCL provides collectives**: GPU-GPU weight broadcast
- **Loose coupling**: Can replace Ray with any RPC mechanism
"""

from __future__ import annotations

import time

from typing import Any, Literal

import torch
import torch.distributed
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.modules.llm.backends import stateless_init_process_group
from torchrl.weight_update.weight_sync_schemes import (
    WeightReceiver,
    WeightSender,
    WeightStrategy,
    WeightSyncScheme,
)

# ============================================================================
# vLLM Transport using Collective Communication
# ============================================================================


class VLLMCollectiveTransport:
    """Transport for vLLM using collective communication (NCCL).

    **COLLECTIVE LAYER ONLY** - This class handles the data transfer layer.
    RPC coordination is handled separately by the caller (sender/receiver).

    This transport uses PyTorch distributed collectives to broadcast weights
    from a trainer (rank 0) to vLLM workers (ranks 1+).

    **Separation of Concerns:**
    - This class: NCCL collective operations (GPU-GPU data transfer)
    - Caller (sender/receiver): RPC coordination (when to start collective)

    Args:
        master_address: Address of the master node for distributed init.
        master_port: Port of the master node for distributed init.
        rank: Rank of this process (0 for trainer, 1+ for vLLM workers).
        world_size: Total number of processes (1 + num_replicas * gpus_per_replica).
        device: Device to use for communication (typically cuda:0).
        vllm_engine: Optional vLLM engine reference (for receiver side).

    Note:
        The RPC layer (e.g., Ray remote calls) must ensure all ranks call
        init_all_workers_group() simultaneously before any collective operations.
    """

    def __init__(
        self,
        master_address: str,
        master_port: int,
        rank: int | None,
        world_size: int,
        device: torch.device | str | int | None = None,
        vllm_engine: Any | None = None,
    ):
        self.master_address = master_address
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.vllm_engine = vllm_engine
        self._comm_group = None
        self._model_metadata = None

        # Ray sets CUDA_VISIBLE_DEVICES, so each actor sees only device 0
        # PyNcclCommunicator expects an integer device index
        if device is None:
            self.device = 0  # Default to device 0 (Ray convention)
        elif isinstance(device, str):
            # Extract device index from "cuda:X"
            self.device = int(device.split(":")[-1]) if ":" in device else 0
        elif isinstance(device, torch.device):
            # Extract index from torch.device
            self.device = device.index if device.index is not None else 0
        else:
            self.device = device

    def init_all_workers_group(
        self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]]
    ):
        """Initialize the collective communication group.

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
        """
        self._model_metadata = model_metadata

        if self.rank == 0:
            # Trainer side - initialize process group
            torchrl_logger.debug(
                f"Initializing trainer collective group: rank={self.rank}, world_size={self.world_size}, device={self.device}"
            )
            # Ray sets CUDA_VISIBLE_DEVICES, so we always use device 0
            # Set CUDA device before initializing NCCL to avoid segfaults
            torch.cuda.set_device(self.device)
            torchrl_logger.debug(f"Set CUDA device to {self.device}")

            self._comm_group = stateless_init_process_group(
                self.master_address,
                self.master_port,
                self.rank,
                self.world_size,
                device=self.device,
            )
            torchrl_logger.debug("Trainer collective group initialized successfully")
        else:
            # vLLM worker side - initialize through engine
            if self.vllm_engine is None:
                raise ValueError("vllm_engine must be provided for worker ranks")

            torchrl_logger.debug(
                "Initializing vLLM worker collective group through engine"
            )
            # Call vLLM engine's init method - it returns futures for all workers
            # Workers will start NCCL init in background threads and return immediately
            refs = self.vllm_engine.init_weight_update_group(
                master_address=self.master_address,
                master_port=self.master_port,
            )

            # Wait for RPCs to complete - ensures workers have dispatched their NCCL init threads
            import ray

            ray.get(refs)
            torchrl_logger.debug(
                f"All {len(refs)} vLLM workers have dispatched NCCL init RPCs"
            )

            # Small delay to ensure worker background threads have entered the NCCL collective
            # This prevents a race where the trainer starts NCCL before workers are ready

            time.sleep(0.2)

            self._comm_group = True  # Mark as initialized
            torchrl_logger.debug(
                "vLLM workers should now be blocked in NCCL collective, ready for trainer"
            )

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Broadcast weights to all workers using NCCL.

        This method follows AsyncVLLM's periodic-mono pattern:
        For each weight: RPC → NCCL broadcast → Wait for RPC completion

        This should only be called from rank 0 (trainer).

        Args:
            model_id: ID of the model (used for logging).
            weights: TensorDict or dict of weights to broadcast.
        """
        # This code is a duplicate from AsyncVLLM
        # We are waiting for vLLM server to accept tokens endpoints, at which point we will be
        # able to remove all dependencies on Ray for vllm distributed features.
        # This will allow a more natural integration with the sender/receiver API.

        import ray

        if self.rank != 0:
            raise RuntimeError("send_weights should only be called from rank 0")

        if self._comm_group is None:
            raise RuntimeError(
                "Communication group not initialized. Call init_all_workers_group first."
            )

        if self._model_metadata is None:
            raise RuntimeError("Model metadata not set")

        if self.vllm_engine is None:
            raise RuntimeError(
                "vllm_engine must be provided to sender for RPC coordination"
            )

        # Set CUDA device for this operation
        torch.cuda.set_device(self.device)

        # Convert to dict if needed
        if isinstance(weights, TensorDictBase):
            weights_dict = weights.to_dict()
        else:
            weights_dict = weights

        torchrl_logger.debug(
            f"Broadcasting {len(weights_dict)} weights for model '{model_id}'"
        )

        # Broadcast each weight using periodic-mono pattern (like AsyncVLLM)
        for name, (dtype, shape) in self._model_metadata.items():
            if name not in weights_dict:
                raise ValueError(
                    f"Weight '{name}' not found in weights. Weights keys: {list(weights_dict.keys())[:10]}..."
                )

            tensor = weights_dict[name].to(f"cuda:{self.device}")
            dtype_name = str(dtype).split(".")[-1]  # "torch.float16" -> "float16"

            # Step 1: Send RPC to workers for this weight
            futures = self.vllm_engine.collective_rpc(
                "update_weight", args=(name, dtype_name, tuple(shape))
            )

            # Step 2: Immediately broadcast this weight
            self._comm_group.broadcast(
                tensor,
                src=0,
                stream=torch.cuda.current_stream(),
            )

            # Step 3: Wait for workers to complete this weight
            ray.get(futures)
            del tensor

        torch.cuda.synchronize()
        torchrl_logger.debug(f"Broadcast complete for model '{model_id}'")

    def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
        """Receive weights from broadcaster.

        This should only be called from worker ranks (rank > 0).
        This method is called by vLLM engine internally through collective operations.

        Returns:
            None - vLLM handles weight application internally via collectives.
        """
        # vLLM handles this through its own collective operations
        # The weights are received and applied by the engine during broadcast
        return None

    def check_connection(self) -> bool:
        """Check if the communication group is initialized."""
        return self._comm_group is not None


# ============================================================================
# vLLM Weight Synchronization Components
# ============================================================================


class VLLMWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization scheme for vLLM engines.

    This scheme uses collective communication (NCCL) to broadcast weights from
    a trainer to vLLM inference workers with parallelism support.

    Args:
        master_address: Address of the master node. Defaults to "localhost".
        master_port: Port of the master node. If None, will auto-assign.
        gpus_per_replica: Number of GPUs per replica (tp_size × dp_size × pp_size).
        num_replicas: Number of vLLM engine replicas. Defaults to 1.
        strategy: Weight extraction strategy ("tensordict" or "state_dict").
        device: Device index to use for communication. Defaults to 0.
            Note: When using Ray, each actor sees only its assigned GPU as device 0
            due to CUDA_VISIBLE_DEVICES isolation. You should typically use 0.

    .. warning::
        Collective communication requires ALL ranks to participate simultaneously.
        Both the sender (trainer, rank 0) and all receivers (vLLM workers, ranks 1+)
        must call ``init_all_workers_group()`` at approximately the same time for the collective
        handshake to succeed. Do NOT wait for one init to complete before starting
        the other - start both and wait for both together.

    Note:
        The world_size for NCCL will be: 1 (trainer) + num_replicas × gpus_per_replica (vLLM workers)

    Example:
        >>> # Single replica with 2 GPUs (e.g., tp_size=2)
        >>> scheme = VLLMWeightSyncScheme(
        ...     master_port=12345,
        ...     gpus_per_replica=2,
        ...     num_replicas=1,
        ...     strategy="tensordict"
        ... )  # world_size = 1 + 1*2 = 3
        >>>
        >>> # Multiple replicas with 1 GPU each
        >>> scheme = VLLMWeightSyncScheme(
        ...     master_port=12345,
        ...     gpus_per_replica=1,
        ...     num_replicas=2,
        ...     strategy="tensordict"
        ... )  # world_size = 1 + 2*1 = 3
        >>>
        >>> # Multiple replicas with tp_size=2, dp_size=1, pp_size=1
        >>> scheme = VLLMWeightSyncScheme(
        ...     master_port=12345,
        ...     gpus_per_replica=2,  # 2*1*1
        ...     num_replicas=3,
        ...     strategy="tensordict"
        ... )  # world_size = 1 + 3*2 = 7
        >>>
        >>> # In trainer process (rank 0)
        >>> sender = VLLMWeightSender(scheme)
        >>> sender.register_model(policy)
        >>>
        >>> # In vLLM worker process (rank 1+)
        >>> receiver = VLLMWeightReceiver(scheme, vllm_engine)
        >>>
        >>> # IMPORTANT: Both must init simultaneously for collective handshake
        >>> # With Ray:
        >>> init_sender = sender_actor.init_all_workers_group.remote(metadata)
        >>> init_receiver = receiver_actor.init_all_workers_group.remote(metadata)
        >>> ray.get([init_sender, init_receiver])  # Wait for both together
        >>>
        >>> # After init, updates work normally
        >>> sender.update_weights()
        >>> # Weights are received automatically via collectives
    """

    def __init__(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
        gpus_per_replica: int = 1,
        num_replicas: int = 1,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
        device: torch.device | str | int = 0,
    ):
        self.master_address = (
            master_address if master_address is not None else "localhost"
        )
        self.master_port = master_port
        self.gpus_per_replica = gpus_per_replica
        self.num_replicas = num_replicas
        self.strategy_name = strategy
        # Ray sets CUDA_VISIBLE_DEVICES for each actor, so device 0 is typical
        self.device = device

        # Auto-assign port if not provided
        if self.master_port is None:
            try:
                from vllm.utils import get_open_port

                self.master_port = get_open_port()
            except ImportError:
                # Fallback if vLLM not available
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    self.master_port = s.getsockname()[1]

    def create_transport(self, **kwargs) -> VLLMCollectiveTransport:
        """Create transport for collective communication.

        For vLLM, this creates a transport but requires additional setup via init_all_workers_group().
        This method is required by the base class but transport creation for vLLM
        is more complex and typically handled by sender/receiver initialization.

        Args:
            **kwargs: Not used for vLLM (kept for API compatibility).

        Returns:
            A VLLMCollectiveTransport instance (needs init_all_workers_group() to be called).
        """
        # Return a transport with default rank 0 (trainer)
        # Actual initialization happens in sender/receiver
        world_size = 1 + self.num_replicas * self.gpus_per_replica
        return VLLMCollectiveTransport(
            master_address=self.master_address,
            master_port=self.master_port,
            rank=0,
            world_size=world_size,
            device=self.device,
        )

    def create_sender(self) -> VLLMWeightSender:
        """Create a weight sender for the trainer process."""
        return VLLMWeightSender(self)

    def create_receiver(self, vllm_engine) -> VLLMWeightReceiver:
        """Create a weight receiver for a vLLM worker process.

        Args:
            vllm_engine: The vLLM engine instance (must implement RLvLLMEngine interface).
        """
        return VLLMWeightReceiver(self, vllm_engine)


class VLLMWeightSender(WeightSender):
    """Sends weights to vLLM workers using collective communication.

    **RPC + Collective Implementation**

    This class implements both layers:

    1. **RPC Layer**: Currently uses Ray remote calls (implicit in test setup)
       - Can be extended to other RPC backends (torch.distributed.rpc, gRPC)
       - In the test, Ray actors provide the RPC mechanism

    2. **Collective Layer**: Uses VLLMCollectiveTransport for NCCL broadcast
       - Broadcasts weights from trainer (rank 0) to workers (ranks 1+)
       - High-bandwidth GPU-to-GPU transfer

    **Extending RPC Backends**

    To use a different RPC backend, subclass and override coordination:

    .. code-block:: python

        class TorchRPCVLLMSender(VLLMWeightSender):
            def update_weights(self, weights=None):
                # Custom RPC: Signal workers to prepare
                for worker in self.workers:
                    torch.distributed.rpc.rpc_async(worker, "prepare_receive")

                # Then do collective (unchanged)
                super().update_weights(weights)
    """

    def __init__(self, scheme: VLLMWeightSyncScheme):
        self._scheme = scheme
        self._strategy = WeightStrategy(extract_as=scheme.strategy_name)
        self._model_ref = None
        self._transport = None
        self._model_metadata = None

    def register_model(self, model: Any) -> None:
        """Register the model to extract weights from."""
        import weakref

        self._model_ref = weakref.ref(model)

    def init_all_workers_group(
        self,
        model_metadata: dict[str, tuple[torch.dtype, torch.Size]],
        vllm_engine: Any | None = None,
    ):
        """Initialize the collective communication group.

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
            vllm_engine: Optional vLLM engine for RPC coordination. Required for NCCL broadcasts.
        """
        self._model_metadata = model_metadata
        self._vllm_engine = vllm_engine

        # Create transport for trainer (rank 0)
        world_size = 1 + self._scheme.num_replicas * self._scheme.gpus_per_replica
        self._transport = VLLMCollectiveTransport(
            master_address=self._scheme.master_address,
            master_port=self._scheme.master_port,
            rank=0,  # Trainer is always rank 0
            world_size=world_size,
            device=self._scheme.device,
            vllm_engine=vllm_engine,
        )
        torchrl_logger.debug(
            f"Initializing transport from sender with world_size={world_size}"
        )
        self._transport.init_all_workers_group(model_metadata)

    def update_weights(self, weights: Any | None = None) -> None:
        """Extract and broadcast weights to vLLM workers.

        Args:
            weights: Optional weights to send. If None, extracts from registered model.
        """
        if self._transport is None:
            raise RuntimeError(
                "Transport not initialized. Call init_all_workers_group first."
            )

        # Extract weights if not provided
        if weights is None:
            model = self._model_ref()
            if model is None:
                raise RuntimeError("Model reference is dead")
            weights = self._strategy.extract_weights(model)
        else:
            # Ensure weights are in the right format
            if hasattr(weights, "state_dict"):
                # It's a module, extract
                weights = self._strategy.extract_weights(weights)

        # Send via transport
        self._transport.send_weights("vllm_model", weights)


class VLLMWeightReceiver(WeightReceiver):
    """Receives weights in a vLLM worker using collective communication.

    **RPC + Collective Implementation**

    This class implements both layers:

    1. **RPC Layer**: Currently uses Ray for coordination
       - `init()` in test uses Ray `ray.get_actor()` to find trainer
       - Fetches metadata via Ray remote call
       - Signals readiness to participate in collective

    2. **Collective Layer**: Participates in NCCL broadcast
       - Receives weights via collective operations
       - vLLM engine applies weights internally during broadcast

    **Extending RPC Backends**

    To use a different RPC backend:

    .. code-block:: python

        class TorchRPCVLLMReceiver(VLLMWeightReceiver):
            def init(self):
                # Custom RPC: Get metadata from trainer
                metadata = torch.distributed.rpc.rpc_sync(
                    "trainer",
                    lambda: get_metadata()
                )

                # Then init collective (unchanged)
                self.receiver.init_all_workers_group(metadata)

    Note:
        The RPC and collective layers are loosely coupled. The RPC layer
        ensures all ranks are ready before the collective starts, but the
        actual data transfer is independent of the RPC mechanism.
    """

    def __init__(self, scheme: VLLMWeightSyncScheme, vllm_engine):
        self._scheme = scheme
        self._strategy = WeightStrategy(extract_as=scheme.strategy_name)
        self._vllm_engine = vllm_engine
        self._transport = None

    def init_all_workers_group(
        self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]]
    ):
        """Initialize the collective communication group.

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
        """
        # For vLLM receiver, we use rank=1 as a placeholder
        # The engine handles actual rank assignment internally for all workers
        world_size = 1 + self._scheme.num_replicas * self._scheme.gpus_per_replica
        self._transport = VLLMCollectiveTransport(
            master_address=self._scheme.master_address,
            master_port=self._scheme.master_port,
            rank=None,  # Placeholder - engine assigns actual ranks
            world_size=world_size,
            device=self._scheme.device,
            vllm_engine=self._vllm_engine,
        )
        torchrl_logger.debug(
            f"Initializing transport from receiver with world_size={world_size}."
        )
        self._transport.init_all_workers_group(model_metadata)

    def apply_weights(self, weights: Any, inplace: bool = True) -> None:
        """Apply weights to vLLM engine.

        Args:
            weights: The weights to apply.
            inplace: Whether to apply weights in place. Default is `True`.

        Note: For vLLM, weights are applied automatically during the collective
        broadcast operation. This method is a no-op but kept for API consistency.
        """
        # vLLM handles weight application through its collective operations
        # The weights are already applied by the time broadcast completes

    def poll_and_apply(self, timeout: float = 0.1) -> bool:
        """Poll for and apply weights.

        Returns:
            False - vLLM uses push-based updates via collectives, not polling.
        """
        # vLLM uses collective broadcasts (push), not polling
        # This is handled by the engine's collective operations
        return False


# ============================================================================
# Helper Functions
# ============================================================================


def get_model_metadata(model) -> dict[str, tuple[torch.dtype, torch.Size]]:
    """Extract model metadata from a model.

    Args:
        model: A model with state_dict() or a model wrapper.

    Returns:
        Dict mapping parameter names to (dtype, shape) tuples.

    Note:
        This function must extract keys in the same format as WeightStrategy.extract_weights()
        to ensure consistency between metadata and actual weight keys during broadcasting.
    """
    # Extract state_dict directly from the model
    # This ensures keys match what extract_weights() will produce
    if hasattr(model, "state_dict"):
        if hasattr(model, "merge_and_unload"):
            # LoRA model
            sd = model.merge_and_unload().state_dict()
        else:
            sd = model.state_dict()
    else:
        raise TypeError(f"Cannot extract state_dict from {type(model)}")

    return {k: (v.dtype, v.shape) for k, v in sd.items()}
