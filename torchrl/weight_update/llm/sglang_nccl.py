# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang weight synchronization via NCCL.

This module provides weight synchronization for SGLang servers using a two-layer
architecture similar to the vLLM implementation:

**Architecture Overview**

1. **HTTP Layer** (Coordination)
   - Signals the SGLang server when a collective operation will begin
   - Uses SGLang's native /init_weights_update_group and /update_weights_from_distributed APIs
   - Tells SGLang workers: "prepare to receive weights via collective"

2. **Collective Layer** (Data Transfer)
   - Performs the actual weight broadcast using NCCL
   - High-bandwidth GPU-to-GPU communication
   - Trainer (rank 0) broadcasts, workers receive

**Flow Example**

.. code-block::

    Trainer (rank 0)                    SGLang Server (ranks 1+)
    ================                    ========================

    # 1. HTTP: Initialize NCCL group
    POST /init_weights_update_group --> "Joining NCCL group"

    # 2. Both enter NCCL collective
    NCCL init handshake <-------------> NCCL init handshake

    # 3. For each weight update:
    POST /update_weights_from_distributed --> "Ready to receive weight X"
    NCCL broadcast -----------------------> NCCL receive

Example:
    >>> from torchrl.weight_update.llm import SGLangWeightSyncScheme
    >>>
    >>> # Create scheme
    >>> scheme = SGLangWeightSyncScheme(
    ...     server_url="http://localhost:30000",
    ...     num_gpus=2,  # tp_size * dp_size
    ... )
    >>>
    >>> # Create sender for trainer
    >>> sender = scheme.create_sender()
    >>> sender.register_model(policy_model)
    >>>
    >>> # Initialize NCCL group (must be done before weight updates)
    >>> metadata = get_model_metadata(policy_model)
    >>> sender.init_all_workers_group(metadata)
    >>>
    >>> # Update weights
    >>> sender.update_weights()
"""

from __future__ import annotations

import time
from typing import Any, Literal

import requests
import torch

from torchrl._utils import logger as torchrl_logger
from torchrl.modules.llm.backends.sglang.sglang_utils import dtype_to_str, get_open_port
from torchrl.weight_update.weight_sync_schemes import WeightStrategy, WeightSyncScheme


def get_model_metadata(model: Any) -> dict[str, tuple[torch.dtype, torch.Size]]:
    """Extract model parameter metadata for weight broadcasting.

    Args:
        model: The model to extract metadata from. Can be a PyTorch model
               or a model with merge_and_unload() for LoRA.

    Returns:
        dict: Mapping of parameter names to (dtype, shape) tuples.
    """
    if hasattr(model, "merge_and_unload"):
        # LoRA model - merge first
        sd = model.merge_and_unload().state_dict()
    else:
        sd = model.state_dict()

    return {k: (v.dtype, v.shape) for k, v in sd.items()}


class SGLangCollectiveTransport:
    """Transport for SGLang using NCCL collective communication.

    This transport coordinates with SGLang servers via HTTP and performs
    weight transfer via NCCL broadcast.

    Args:
        server_url: URL of the SGLang server.
        master_address: Address for NCCL initialization.
        master_port: Port for NCCL initialization.
        rank: Rank of this process (0 for trainer).
        world_size: Total number of processes.
        device: Device to use for communication.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        server_url: str,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        device: torch.device | str | int | None = None,
        timeout: float = 300.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.master_address = master_address
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.timeout = timeout
        self._comm_group = None
        self._model_metadata = None

        # Handle device specification
        if device is None:
            self.device = 0
        elif isinstance(device, str):
            self.device = int(device.split(":")[-1]) if ":" in device else 0
        elif isinstance(device, torch.device):
            self.device = device.index if device.index is not None else 0
        else:
            self.device = device

    def init_all_workers_group(
        self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]]
    ) -> None:
        """Initialize the NCCL communication group.

        For the trainer (rank 0), this:
        1. Signals the SGLang server via HTTP to join the NCCL group
        2. Initializes the trainer's NCCL communicator

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
        """
        self._model_metadata = model_metadata

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 (trainer) should call init_all_workers_group"
            )

        # Step 1: Tell SGLang server to initialize its side of the NCCL group
        torchrl_logger.info(
            f"Requesting SGLang server to join NCCL group: "
            f"address={self.master_address}, port={self.master_port}"
        )

        init_data = {
            "master_address": self.master_address,
            "master_port": self.master_port,
            "rank_offset": 1,  # Server workers start from rank 1
            "world_size": self.world_size,
        }

        response = requests.post(
            f"{self.server_url}/init_weights_update_group",
            json=init_data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                f"SGLang server failed to initialize weight update group: "
                f"{result.get('message', 'Unknown error')}"
            )

        torchrl_logger.info(
            "SGLang server is joining NCCL group, initializing trainer side..."
        )

        # Small delay to ensure server has started NCCL init
        time.sleep(0.2)

        # Step 2: Initialize trainer's NCCL communicator
        torch.cuda.set_device(self.device)

        # Use SGLang's native NCCL utilities (no vLLM dependency)
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from sglang.srt.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=self.master_address,
            port=self.master_port,
            rank=0,
            world_size=self.world_size,
        )
        self._comm_group = PyNcclCommunicator(
            pg, device=torch.device(f"cuda:{self.device}")
        )

        torchrl_logger.info("NCCL group initialized successfully")

    def send_weights(
        self,
        model_id: str,
        weights: dict[str, torch.Tensor],
    ) -> None:
        """Broadcast weights to SGLang server via NCCL.

        Args:
            model_id: Identifier for the model (for logging).
            weights: Dict mapping parameter names to tensors.
        """
        if self.rank != 0:
            raise RuntimeError("send_weights should only be called from rank 0")

        if self._comm_group is None:
            raise RuntimeError(
                "Communication group not initialized. Call init_all_workers_group first."
            )

        if self._model_metadata is None:
            raise RuntimeError("Model metadata not set")

        torch.cuda.set_device(self.device)

        torchrl_logger.debug(
            f"Broadcasting {len(weights)} weights for model '{model_id}'"
        )

        for name, (dtype, shape) in self._model_metadata.items():
            if name not in weights:
                raise ValueError(
                    f"Weight '{name}' not found in weights. "
                    f"Available keys: {list(weights.keys())[:10]}..."
                )

            tensor = weights[name].to(f"cuda:{self.device}")

            # Step 1: Signal server to prepare for this weight
            update_data = {
                "name": name,
                "dtype": dtype_to_str(dtype),
                "shape": list(shape),
            }
            response = requests.post(
                f"{self.server_url}/update_weights_from_distributed",
                json=update_data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Step 2: Broadcast the weight via NCCL
            if hasattr(self._comm_group, "broadcast"):
                # PyNcclCommunicator interface
                self._comm_group.broadcast(
                    tensor,
                    src=0,
                    stream=torch.cuda.current_stream(),
                )
            else:
                # torch.distributed interface
                torch.distributed.broadcast(tensor, src=0, group=self._comm_group)

            del tensor

        torch.cuda.synchronize()
        torchrl_logger.debug(f"Broadcast complete for model '{model_id}'")

    def check_connection(self) -> bool:
        """Check if the communication group is initialized."""
        return self._comm_group is not None


class SGLangWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization scheme for SGLang servers.

    This scheme uses HTTP to coordinate with SGLang servers and NCCL to
    broadcast weights from a trainer to SGLang workers.

    Args:
        server_url: URL of the SGLang server.
        master_address: Address for NCCL initialization. Defaults to "localhost".
        master_port: Port for NCCL initialization. Auto-assigned if None.
        num_gpus: Number of GPUs used by the SGLang server (tp_size * dp_size).
        strategy: Weight extraction strategy ("tensordict" or "state_dict").
        device: Device index for the trainer. Defaults to 0.

    Example:
        >>> scheme = SGLangWeightSyncScheme(
        ...     server_url="http://localhost:30000",
        ...     num_gpus=2,
        ... )
        >>> sender = scheme.create_sender()
        >>> sender.register_model(policy)
        >>> metadata = get_model_metadata(policy)
        >>> sender.init_all_workers_group(metadata)
        >>> sender.update_weights()
    """

    def __init__(
        self,
        server_url: str,
        master_address: str | None = None,
        master_port: int | None = None,
        num_gpus: int = 1,
        strategy: Literal["tensordict", "state_dict"] = "tensordict",
        device: torch.device | str | int = 0,
    ):
        self.server_url = server_url.rstrip("/")
        self.master_address = master_address or "localhost"
        self.master_port = master_port or get_open_port()
        self.num_gpus = num_gpus
        self.strategy_name = strategy
        self.device = device

    @property
    def world_size(self) -> int:
        """Total world size: 1 (trainer) + num_gpus (workers)."""
        return 1 + self.num_gpus

    def create_transport(self, **kwargs) -> SGLangCollectiveTransport:
        """Create transport for collective communication."""
        return SGLangCollectiveTransport(
            server_url=self.server_url,
            master_address=self.master_address,
            master_port=self.master_port,
            rank=0,
            world_size=self.world_size,
            device=self.device,
        )

    def create_sender(self) -> SGLangWeightSender:
        """Create a weight sender for the trainer process."""
        return SGLangWeightSender(self)

    def create_receiver(self, *args, **kwargs) -> None:
        """Create a weight receiver.

        Note: For SGLang, receivers are managed by the SGLang server itself.
        This method is provided for API compatibility but returns None.
        """
        torchrl_logger.info(
            "SGLang receivers are managed by the SGLang server. "
            "No explicit receiver creation needed."
        )
        return None


class SGLangWeightSender:
    """Sends weights to SGLang workers using NCCL broadcast.

    Args:
        scheme: The SGLangWeightSyncScheme configuration.
    """

    def __init__(self, scheme: SGLangWeightSyncScheme):
        self._scheme = scheme
        self._model_ref = None
        self._model_metadata = None
        self._transport = None
        self._strategy = WeightStrategy(extract_as=scheme.strategy_name)

    def register_model(self, model: Any) -> None:
        """Register the model for weight extraction.

        Args:
            model: The PyTorch model to sync weights from.
        """
        import weakref

        self._model_ref = weakref.ref(model)

    def init_all_workers_group(
        self,
        model_metadata: dict[str, tuple[torch.dtype, torch.Size]],
    ) -> None:
        """Initialize the NCCL communication group.

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
        """
        self._model_metadata = model_metadata
        self._transport = self._scheme.create_transport()
        self._transport.init_all_workers_group(model_metadata)

    def update_weights(
        self,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Broadcast weights to SGLang workers.

        Args:
            weights: Optional dict of weights. If None, extracts from registered model.
        """
        if self._transport is None:
            raise RuntimeError(
                "Transport not initialized. Call init_all_workers_group first."
            )

        if weights is None:
            if self._model_ref is None:
                raise RuntimeError(
                    "No model registered and no weights provided. "
                    "Call register_model() first or provide weights explicitly."
                )
            model = self._model_ref()
            if model is None:
                raise RuntimeError("Model reference is no longer valid")
            weights = self._strategy.extract_weights(model)

        # Convert to dict if needed
        if hasattr(weights, "to_dict"):
            weights = weights.to_dict()

        self._transport.send_weights("sglang_model", weights)

    def flush_cache(self) -> bool:
        """Flush the SGLang server's radix cache after weight update.

        Returns:
            bool: True if cache was flushed successfully.
        """
        try:
            response = requests.post(
                f"{self._scheme.server_url}/flush_cache",
                timeout=30.0,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
