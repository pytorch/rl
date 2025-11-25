# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM weight synchronization using double-buffered shared memory.

This module provides weight synchronization for vLLM engines using a double-buffer
approach with memory-mapped TensorDict storage.

**Architecture Overview**

The double-buffer synchronization uses a simpler architecture compared to NCCL:

1. **Sender (Trainer)**
   - Extracts weights from the training model
   - Writes weights to shared directory using TensorDict.memmap
   - No coordination needed - receiver pulls when ready

2. **Receiver (vLLM Worker)**
   - Uses RPC to tell all vLLM workers to load from shared directory
   - Each worker reads weights and calls model.load_weights()
   - Can trigger at any time (pull-based)

**Key Differences from NCCL**

- **Async vs Sync**: Double-buffer is asynchronous (no coordination required)
- **Push vs Pull**: Sender writes, receiver pulls when ready via RPC
- **Simplicity**: No NCCL collectives, uses file I/O
- **Storage**: Uses shared filesystem instead of GPU-GPU transfer

**RPC Pattern**

Like the NCCL implementation, this uses RPC to coordinate workers:
- RPC tells workers: "load weights from this directory"
- Workers read from shared storage independently
- Each worker calls `model_runner.model.load_weights()`

**Usage Example**

.. code-block:: python

    # Create scheme with shared directory
    scheme = VLLMDoubleBufferSyncScheme(
        remote_addr="/shared/weights",
        num_threads=4
    )

    # Sender side (trainer)
    sender = scheme.create_sender()
    sender.register_model(policy_model)
    sender.update_weights()  # Writes to /shared/weights

    # Receiver side (vLLM worker - AsyncVLLM)
    receiver = scheme.create_receiver(vllm_engine)
    receiver.poll_and_apply()  # RPC to workers -> load from /shared/weights

**Node-to-Node Transfer**

For distributed setups, you can use different addresses:
- Sender writes to local path
- Use NFS, rsync, or other file sync mechanisms
- Receiver reads from its local mount point
"""

from __future__ import annotations

from typing import Any, Literal

from tensordict import TensorDict, TensorDictBase
from torchrl._utils import logger
from torchrl.weight_update.weight_sync_schemes import (
    WeightReceiver,
    WeightSender,
    WeightStrategy,
    WeightSyncScheme,
)


class VLLMDoubleBufferTransport:
    """Transport for vLLM using double-buffered memory-mapped storage.

    This transport writes weights to a shared directory and reads them back
    using TensorDict's memory-mapping capabilities.

    Args:
        remote_addr: Directory path where sender writes weights.
        local_addr: Directory path where receiver reads weights.
            If None, uses same path as remote_addr (for local testing).
        num_threads: Number of threads for memmap operations.
    """

    def __init__(
        self, remote_addr: str, local_addr: str | None = None, num_threads: int = 1
    ):
        if local_addr is None:
            local_addr = remote_addr
        self.remote_addr = remote_addr
        self.local_addr = local_addr
        self.num_threads = num_threads

    def send_weights(self, model_id: str, weights: Any) -> None:
        """Writes the weights to a shared directory.

        Args:
            model_id: Identifier for the model (used for logging).
            weights: TensorDict or dict of weights to write.
        """
        if isinstance(weights, dict):
            weights = TensorDict(weights, batch_size=[])
        elif isinstance(weights, TensorDictBase):
            # Ensure it has a batch_size
            if weights.batch_size == ():
                weights = weights.clone()

        logger.info(f"Writing weights for model '{model_id}' to {self.remote_addr}")
        weights.memmap(self.remote_addr, num_threads=self.num_threads)
        logger.info(f"Weights written successfully to {self.remote_addr}")

    def receive_weights(self, timeout: float = 1.0) -> TensorDict:
        """Reads the weights from the shared directory.

        Args:
            timeout: Not used for file-based transport (kept for API compatibility).

        Returns:
            TensorDict with flattened keys containing the weights.
        """
        logger.info(f"Reading weights from {self.local_addr}")
        weights = TensorDict.load_memmap(self.local_addr)
        weights = weights.flatten_keys(".")
        logger.info(f"Weights read successfully from {self.local_addr}")
        return weights

    def check_connection(self) -> bool:
        """Check if the transport is ready.

        For file-based transport, always returns True.
        """
        return True


class VLLMDoubleBufferSyncScheme(WeightSyncScheme):
    """Weight synchronization scheme for vLLM using double-buffered storage.

    This scheme uses memory-mapped TensorDict storage to transfer weights from
    a trainer to vLLM inference workers. It's simpler than NCCL-based approaches
    and doesn't require process group coordination.

    Args:
        remote_addr: Directory path where sender writes weights.
        local_addr: Directory path where receiver reads weights.
            If None, uses same path as remote_addr (for local testing).
        num_threads: Number of threads for memmap operations. Defaults to 1.
        strategy: Weight extraction strategy ("tensordict" or "state_dict").

    Example:
        >>> # Local testing (same machine)
        >>> scheme = VLLMDoubleBufferSyncScheme(
        ...     remote_addr="/tmp/weights",
        ...     strategy="tensordict"
        ... )
        >>>
        >>> # Distributed setup (different machines)
        >>> # On trainer node:
        >>> scheme = VLLMDoubleBufferSyncScheme(
        ...     remote_addr="/mnt/shared/weights",  # NFS mount
        ...     num_threads=4
        ... )
        >>>
        >>> # On vLLM worker node:
        >>> scheme = VLLMDoubleBufferSyncScheme(
        ...     remote_addr="/mnt/shared/weights",  # Same NFS mount
        ...     num_threads=4
        ... )
    """

    def __init__(
        self,
        remote_addr: str,
        local_addr: str | None = None,
        num_threads: int = 1,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
    ):
        self.remote_addr = remote_addr
        self.local_addr = local_addr if local_addr is not None else remote_addr
        self.num_threads = num_threads
        self.strategy_name = strategy

    def create_transport(self, **kwargs) -> VLLMDoubleBufferTransport:
        """Create transport for double-buffered storage.

        Args:
            **kwargs: Not used for file-based transport (kept for API compatibility).

        Returns:
            A VLLMDoubleBufferTransport instance.
        """
        return VLLMDoubleBufferTransport(
            remote_addr=self.remote_addr,
            local_addr=self.local_addr,
            num_threads=self.num_threads,
        )

    def create_sender(self) -> VLLMDoubleBufferWeightSender:
        """Create a weight sender for the trainer process."""
        return VLLMDoubleBufferWeightSender(self)

    def create_receiver(self, vllm_engine) -> VLLMDoubleBufferWeightReceiver:
        """Create a weight receiver for a vLLM worker process.

        Args:
            vllm_engine: The vLLM engine instance (must have .llm_engine.model_executor attribute).
        """
        return VLLMDoubleBufferWeightReceiver(self, vllm_engine)


class VLLMDoubleBufferWeightSender(WeightSender):
    """Sends weights to vLLM workers using double-buffered storage.

    This sender extracts weights from a training model and writes them to
    a shared directory using TensorDict.memmap.

    Example:
        >>> sender = scheme.create_sender()
        >>> sender.register_model(policy_model)
        >>>
        >>> # During training loop
        >>> sender.update_weights()  # Writes current weights to shared storage
    """

    def __init__(self, scheme: VLLMDoubleBufferSyncScheme):
        self._scheme = scheme
        self._strategy = WeightStrategy(extract_as=scheme.strategy_name)
        self._model_ref = None
        self._transport = None

    def register_model(self, model: Any) -> None:
        """Register the model to extract weights from.

        Args:
            model: The model to extract weights from (e.g., TransformersWrapper).
        """
        import weakref

        self._model_ref = weakref.ref(model)

        # Create transport on registration
        self._transport = self._scheme.create_transport()
        logger.info(
            f"Registered model for double-buffer weight sync to {self._scheme.remote_addr}"
        )

    def update_weights(self, weights: Any | None = None) -> None:
        """Extract and write weights to shared storage.

        Args:
            weights: Optional weights to send. If None, extracts from registered model.
        """
        if self._transport is None:
            raise RuntimeError("Transport not initialized. Call register_model first.")

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


class VLLMDoubleBufferWeightReceiver(WeightReceiver):
    """Receives weights in a vLLM worker using double-buffered storage.

    This receiver reads weights from a shared directory and loads them into
    the vLLM engine using the engine's load_weights interface.

    Example:
        >>> receiver = scheme.create_receiver(vllm_engine)
        >>>
        >>> # Poll for new weights
        >>> if receiver.poll_and_apply():
        ...     print("Weights updated!")
    """

    def __init__(self, scheme: VLLMDoubleBufferSyncScheme, vllm_engine):
        self._scheme = scheme
        self._strategy = WeightStrategy(extract_as=scheme.strategy_name)
        self._vllm_engine = vllm_engine
        self._transport = scheme.create_transport()
        logger.info(
            f"Initialized double-buffer receiver reading from {self._scheme.local_addr}"
        )

    def apply_weights(self, weights: TensorDict, inplace: bool = True) -> None:
        """Apply weights to vLLM engine using RPC.

        This method uses RPC to tell all vLLM workers to load weights from
        the shared storage directory. Similar to how AsyncVLLM._update_weights_with_nccl_broadcast_simple
        uses collective_rpc to coordinate workers.

        Args:
            weights: TensorDict with flattened keys containing weights.
            inplace: Whether to apply weights in place. Default is `True`.
        """
        if not inplace:
            raise ValueError("Cannot apply weights out of place for vLLM double-buffer")
        logger.info("Applying weights to vLLM engine via RPC")

        # Convert TensorDict to list of (name, tensor) tuples
        weights_list = list(weights.items())

        # Check if this is an AsyncVLLM instance (uses RPC to coordinate workers)
        if hasattr(self._vllm_engine, "collective_rpc"):
            # AsyncVLLM path: use RPC to tell all workers to load weights
            logger.info(
                f"Using RPC to load {len(weights_list)} weights across all replicas"
            )

            # Call collective_rpc to tell workers to load from shared storage
            # The method 'load_weights_from_storage' will be called on each worker
            futures = self._vllm_engine.collective_rpc(
                method="load_weights_from_storage",
                args=(str(self._scheme.local_addr), self._transport.num_threads),
            )

            # Wait for all workers to complete
            import ray

            ray.get(futures)
            logger.info("Weights loaded successfully via RPC")
        else:
            # Direct path for local LLM (non-AsyncVLLM)
            logger.info("Using direct load for local LLM")
            engine = (
                self._vllm_engine.llm_engine
                if hasattr(self._vllm_engine, "llm_engine")
                else self._vllm_engine
            )
            worker = engine.model_executor.driver_worker
            model = worker.model_runner.model
            model.load_weights(weights_list)
            logger.info("Weights loaded successfully")

    def poll_and_apply(self, timeout: float = 180.0) -> bool:
        """Poll for and apply weights from shared storage.

        Args:
            timeout: Not used for file-based transport (kept for API compatibility).

        Returns:
            True if weights were successfully read and applied, False otherwise.
        """
        weights = self._transport.receive_weights(timeout=timeout)
        self.apply_weights(weights)
        return True
