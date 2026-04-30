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
import uuid

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import requests
import torch

from torchrl._utils import logger as torchrl_logger
from torchrl.modules.llm.backends.sglang.sglang_utils import (
    dtype_to_str,
    get_local_ip_address,
    get_open_port,
)
from torchrl.weight_update.weight_sync_schemes import WeightStrategy, WeightSyncScheme


def _init_custom_process_group(
    backend: str = "nccl",
    init_method: str | None = None,
    world_size: int = -1,
    rank: int = -1,
    group_name: str = "default",
    timeout: float | None = None,
) -> torch.distributed.ProcessGroup:
    """Create a torch.distributed process group without requiring a default group.

    This mirrors SGLang's ``init_custom_process_group`` so that the trainer
    creates a process group compatible with what the SGLang server creates
    internally. Both sides use TCP rendezvous + ``_new_process_group_helper``
    to form the same NCCL collective.

    Adapted from SGLang (sglang.srt.distributed) and OpenRLHF.
    """
    from torch.distributed.distributed_c10d import (
        _new_process_group_helper,
        _world,
        Backend,
        default_pg_timeout,
        PrefixStore,
        rendezvous,
    )

    if init_method is None:
        init_method = "env://"

    backend = Backend(backend)

    if timeout is None:
        timeout = default_pg_timeout

    rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)

    store = PrefixStore(group_name, store)

    # PyTorch >= 2.6 renamed pg_options to backend_options
    pg_options_key = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_key: None},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


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
        self._group_name = None
        self._model_metadata = None
        self._http_executor = ThreadPoolExecutor(max_workers=1)

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
        1. Creates a torch.distributed process group via TCP rendezvous (rank 0 is master)
        2. Signals the SGLang server via HTTP to create a matching process group
        3. Both sides rendezvous via the TCP store and form an NCCL group

        The SGLang server uses ``init_custom_process_group`` internally which
        creates a ``torch.distributed`` process group (not SGLang's standalone
        ``StatelessProcessGroup`` + ``PyNcclCommunicator``). The trainer must
        use the same mechanism so both sides join the same NCCL collective.

        Args:
            model_metadata: Dict mapping param names to (dtype, shape) tuples.
        """
        self._model_metadata = model_metadata

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 (trainer) should call init_all_workers_group"
            )

        # Disable NCCL P2P/IPC transport. Ray may restrict CUDA_VISIBLE_DEVICES
        # for the train worker (e.g., only GPU 0 visible), while the SGLang
        # server sees all GPUs. This topology mismatch causes "Cuda failure 1
        # 'invalid argument'" during NCCL P2P/IPC channel setup. Disabling P2P
        # forces NCCL to use SHM/network transport which works with any device
        # visibility configuration.
        import os

        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"

        torch.cuda.set_device(self.device)

        # Use a unique group name per attempt to avoid "group already exists"
        # errors on SGLang server retries
        group_name = f"weight_update_group_{uuid.uuid4().hex[:8]}"

        torchrl_logger.info(
            f"Creating torch.distributed process group via TCP rendezvous on "
            f"{self.master_address}:{self.master_port} (group={group_name})"
        )

        # Run the trainer-side process group init in a background thread so it
        # can proceed concurrently with the SGLang server joining via HTTP.
        # Both sides must rendezvous at the TCP store simultaneously.
        pg_error = [None]
        pg_result = [None]

        def _trainer_pg_init():
            try:
                pg = _init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://{self.master_address}:{self.master_port}",
                    world_size=self.world_size,
                    rank=0,
                    group_name=group_name,
                )
                torchrl_logger.info(
                    "Trainer-side torch.distributed process group created successfully"
                )
                pg_result[0] = pg
            except Exception as e:
                pg_error[0] = e

        import threading

        pg_thread = threading.Thread(target=_trainer_pg_init, daemon=True)
        pg_thread.start()

        # Give the TCP store server a moment to start listening
        time.sleep(0.3)

        # NOW tell the SGLang server to connect to our TCP store
        # and join the same process group
        torchrl_logger.info(
            f"Requesting SGLang server to join NCCL group: "
            f"address={self.master_address}, port={self.master_port}"
        )

        init_data = {
            "master_address": self.master_address,
            "master_port": self.master_port,
            "rank_offset": 1,  # Server workers start from rank 1
            "world_size": self.world_size,
            "group_name": group_name,
        }

        init_future = self._http_executor.submit(
            requests.post,
            f"{self.server_url}/init_weights_update_group",
            json=init_data,
            timeout=self.timeout,
        )

        # Wait for both sides to complete
        torchrl_logger.info("Waiting for NCCL group initialization...")
        pg_thread.join(timeout=self.timeout)

        if pg_error[0] is not None:
            raise RuntimeError(
                f"Trainer process group init failed: {pg_error[0]}"
            ) from pg_error[0]

        if pg_result[0] is None:
            raise RuntimeError("Trainer process group init timed out")

        self._comm_group = pg_result[0]
        self._group_name = group_name

        response = init_future.result(timeout=self.timeout + 5.0)
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(
                f"SGLang server failed to initialize weight update group: "
                f"{result.get('message', 'Unknown error')}"
            )

        torchrl_logger.info("NCCL group initialized successfully")

    def send_weights(
        self,
        model_id: str,
        weights: dict[str, torch.Tensor],
    ) -> None:
        """Broadcast weights to SGLang server via NCCL.

        SGLang's ``/update_weights_from_distributed`` endpoint expects a single
        request with lists of all parameter names, dtypes, and shapes. The
        server then enters a broadcast-receive loop for each parameter in
        order. The trainer must broadcast each tensor in the same order,
        concurrently with the server receiving.

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

        # Build lists matching SGLang's UpdateWeightsFromDistributedReqInput
        names = []
        dtypes = []
        shapes = []
        for name, (dtype, shape) in self._model_metadata.items():
            if name not in weights:
                raise ValueError(
                    f"Weight '{name}' not found in weights. "
                    f"Available keys: {list(weights.keys())[:10]}..."
                )
            names.append(name)
            dtypes.append(dtype_to_str(dtype))
            shapes.append(list(shape))

        torchrl_logger.info(f"Broadcasting {len(names)} weights for model '{model_id}'")

        # Step 1: Send a single HTTP request with all weight metadata.
        # The server will enter a broadcast-receive loop for each parameter.
        update_data = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": self._group_name,
            "flush_cache": True,
        }
        update_future = self._http_executor.submit(
            requests.post,
            f"{self.server_url}/update_weights_from_distributed",
            json=update_data,
            timeout=self.timeout,
        )

        # Step 2: Broadcast each weight tensor via NCCL in the same order.
        # The server is concurrently receiving via broadcast on its side.
        for name in names:
            tensor = weights[name].to(f"cuda:{self.device}")
            torch.distributed.broadcast(tensor, src=0, group=self._comm_group)
            del tensor

        torch.cuda.synchronize()

        # Step 3: Wait for the HTTP response confirming server received all weights
        response = update_future.result(timeout=self.timeout + 5.0)
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(
                f"SGLang weight update failed: {result.get('message', 'Unknown error')}"
            )

        torchrl_logger.info(f"Broadcast complete for model '{model_id}'")

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
        self.master_address = master_address or get_local_ip_address()
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
        self._collectors: list = []
        self._post_hooks: list = []

    def register_collector(self, collector) -> None:
        """Register a collector for automatic policy version increment.

        After each :meth:`update_weights` call, ``collector.increment_version()``
        is called automatically.
        """
        self._collectors.append(collector)
        if len(self._post_hooks) == 0:
            self._post_hooks.append(self._increment_all_collector_versions)

    def _increment_all_collector_versions(self):
        for collector in self._collectors:
            try:
                collector.increment_version()
            except Exception as e:
                torchrl_logger.warning(
                    f"Failed to increment version for collector: {e}"
                )

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

        # Run post-hooks (e.g., increment collector versions)
        for hook in self._post_hooks:
            hook()

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
