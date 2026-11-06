from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from queue import Empty
from typing import Any

import torch
import torch.distributed
from tensordict import TensorDict, TensorDictBase
from torch import multiprocessing as mp, nn
from torchrl._utils import logger as torchrl_logger, WEIGHT_SYNC_TIMEOUT
from torchrl.weight_update.utils import _resolve_model
from torchrl.weight_update.weight_sync_schemes import (
    register_weight_sync_backend,
    TransportBackend,
    WeightStrategy,
    WeightSyncScheme,
)


def _close_mp_queue(queue: mp.Queue) -> None:
    """Close a multiprocessing Queue and wait for its feeder thread to exit."""
    queue.close()
    queue.join_thread()


def _tensor_summary(weights: TensorDictBase) -> tuple[set[str], int, int]:
    devices = set()
    numel = 0
    nbytes = 0
    for value in weights.values(True, True):
        if not torch.is_tensor(value):
            continue
        devices.add(str(value.device))
        numel += value.numel()
        nbytes += value.numel() * value.element_size()
    return devices, numel, nbytes


def _synchronize_cuda(weights: TensorDictBase) -> None:
    """Wait for all queued work on the CUDA devices that hold ``weights``.

    Another process, or a stream other than the one that queued a copy, only
    sees the result of that copy once the device has drained it.
    """
    devices = {
        value.device
        for value in weights.values(True, True)
        if torch.is_tensor(value) and value.is_cuda
    }
    for device in devices:
        torch.cuda.synchronize(device)


def _log_weight_sync(
    *,
    model_id: str | None,
    worker_idx: int | None,
    source: TensorDictBase,
    destination: TensorDictBase,
) -> None:
    source_devices, numel, nbytes = _tensor_summary(source)
    destination_devices, _, _ = _tensor_summary(destination)
    torchrl_logger.debug(
        "Synced weights model_id=%s worker=%s params=%s bytes=%s devices=%s -> %s",
        model_id,
        worker_idx,
        numel,
        nbytes,
        sorted(source_devices),
        sorted(destination_devices),
    )


class SharedMemTransport:
    """Shared memory transport for in-place weight updates.

    This transport uses queue-based buffer distribution for initialization, then
    updates shared memory tensors directly for subsequent weight updates.

    Initialization flow:
    - Shared memory buffers are created and sent to workers via per-worker queues
    - Workers receive the buffer reference and apply it to their models
    - Subsequent updates write the buffers in place

    By default the buffers are the sender's own weights, so workers that bind
    them as their parameters see every in-place change on the sender. With
    ``copy=True`` in :meth:`register_weights`, the buffers are staging copies
    that alias neither the sender's model nor the workers' parameters.

    Both CPU and CUDA tensors maintain shared references when sent through mp.Queue.

    .. note::
        Sharing CUDA tensors relies on CUDA IPC, which is only available on
        native Linux (not Windows or WSL2), and backends such as MPS cannot be
        shared across processes at all. On those platforms the schemes stage
        the weights through CPU shared memory instead (see
        :meth:`SharedMemWeightSyncScheme._get_params_map`), and each worker
        moves them back to its own device.

    """

    def __init__(self):
        self._weight_queues = None
        self._unique_weights = None
        self._worker_to_weights_idx = {}

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
        self,
        params_map: dict[int, TensorDictBase],
        init_queues: dict[int, mp.Queue],
        per_worker: bool = False,
        copy: bool = False,
    ) -> None:
        """Initialize per-worker queues for shared memory buffer distribution.

        Args:
            params_map: Mapping of worker_idx to weights TensorDict.
            init_queues: Mapping of worker_idx to initialization queues.
            per_worker: If True, each worker gets its own independent weight copy
                (for distinct policy factories). If False, workers on the same
                device share the same weight buffer (device-based deduplication).
            copy: If True, each unique entry of ``params_map`` is copied into a
                new shared memory staging buffer, so that writes to the tensors
                passed here (for instance an optimizer step on the sender's
                model) never reach the workers. If False (default), the buffers
                share storage with ``params_map``.
        """
        from torchrl.collectors.utils import _cast

        self._weight_queues = init_queues
        self._per_worker = per_worker
        self._unique_weights = []
        self._worker_to_weights_idx = {}
        source_to_idx = {}
        for worker_idx, weights in params_map.items():
            source = worker_idx if per_worker else id(weights)
            if source not in source_to_idx:
                source_to_idx[source] = len(self._unique_weights)
                if copy:
                    buffer = weights.data.clone().apply(_cast, weights)
                    buffer.share_memory_()
                else:
                    buffer = weights.data.apply(_cast, weights)
                self._unique_weights.append(buffer)
            self._worker_to_weights_idx[worker_idx] = source_to_idx[source]

    def setup_connection_and_weights_on_sender(self) -> None:
        """Send shared memory buffer reference to workers via their per-worker queues.

        Both CPU and CUDA tensors maintain shared references through queues.
        Each worker reads from its own dedicated queue, to avoid race conditions.

        """
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        for worker_idx, queue in self._weight_queues.items():
            queue.put(self._unique_weights[self._worker_to_weights_idx[worker_idx]])

    def setup_connection_and_weights_on_receiver(
        self,
        *,
        worker_idx: int | None = None,
        weights: Any = None,
        model: Any = None,
        strategy: Any = None,
        timeout: float = WEIGHT_SYNC_TIMEOUT,
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
        if self._weight_queues is None:
            raise RuntimeError("Queues not created yet. Call init_on_sender() first.")

        if worker_idx not in self._weight_queues:
            raise RuntimeError(f"Worker {worker_idx} not registered in queues.")

        # Read from dedicated queue for this worker
        worker_queue = self._weight_queues[worker_idx]
        received_weights = worker_queue.get(timeout=timeout)
        return received_weights

    def send_weights(
        self,
        weights: TensorDictBase | dict[int, TensorDictBase],
        worker_ids: list[int] | None = None,
    ) -> None:
        """Update weights in-place in shared memory.

        Args:
            weights: New weights to send. Can be:
                - TensorDictBase: Same weights broadcast to all workers (or subset via worker_ids)
                - dict[int, TensorDictBase]: Per-worker weights (keys are worker indices)
            worker_ids: Which workers to update (None = all workers). Only used when
                weights is a TensorDictBase (broadcast mode).

        Raises:
            ValueError: If weights type is unsupported.
        """
        if self._unique_weights is None:
            raise RuntimeError("Unique weights not set. Call register_weights() first.")

        # Check if per-worker weights dict is provided
        if isinstance(weights, dict) and all(
            isinstance(k, int) for k in weights.keys()
        ):
            # Per-worker weights: dict[int, TensorDictBase]
            if not getattr(self, "_per_worker", False):
                raise ValueError(
                    "Per-worker weights dict provided but register_weights() was not "
                    "called with per_worker=True. Cannot update individual worker weights."
                )
            for worker_idx, worker_weights in weights.items():
                if worker_idx not in self._worker_to_weights_idx:
                    raise ValueError(f"Worker {worker_idx} not registered.")
                self._update_single_worker(worker_idx, worker_weights)
        else:
            # Broadcast mode: same weights to all (or subset)
            if isinstance(weights, dict):
                weights = TensorDict(weights)
            if not isinstance(weights, TensorDictBase):
                raise ValueError(f"Unsupported weights type: {type(weights)=}")

            # Unflatten if needed to match shared buffer structure
            weights_to_update = weights
            if any("." in key for key in weights.keys()):
                weights_to_update = weights.unflatten_keys(".")

            # Detach weights to allow in-place updates
            weights_to_update = weights_to_update.detach()

            if getattr(self, "_per_worker", False) and worker_ids is not None:
                # Per-worker mode with specific worker_ids - update only those workers
                for worker_idx in worker_ids:
                    self._update_single_worker(worker_idx, weights_to_update)
            else:
                # Default: update all unique weight buffers
                for buffer in self._unique_weights:
                    if buffer.requires_grad:
                        raise RuntimeError(
                            "Gradients should not be required for shared memory buffers."
                        )
                    if weights_to_update.requires_grad:
                        raise RuntimeError(
                            "Gradients should not be required for weights."
                        )
                    _log_weight_sync(
                        model_id=None,
                        worker_idx=None,
                        source=weights_to_update,
                        destination=buffer,
                    )
                    buffer.update_(weights_to_update, non_blocking=True)

        for buffer in self._unique_weights:
            _synchronize_cuda(buffer)

    def _update_single_worker(self, worker_idx: int, weights: TensorDictBase) -> None:
        """Update weights for a single worker in per-worker mode.

        Args:
            worker_idx: The worker index.
            weights: The weights to update.
        """
        if not hasattr(self, "_worker_to_weights_idx"):
            raise RuntimeError("Per-worker mode not initialized.")

        weights_idx = self._worker_to_weights_idx[worker_idx]
        buffer = self._unique_weights[weights_idx]

        # Unflatten if needed
        weights_to_update = weights
        if any("." in key for key in weights.keys()):
            weights_to_update = weights.unflatten_keys(".")

        weights_to_update = weights_to_update.detach()

        if buffer.requires_grad:
            raise RuntimeError(
                "Gradients should not be required for shared memory buffers."
            )
        if weights_to_update.requires_grad:
            raise RuntimeError("Gradients should not be required for weights.")

        _log_weight_sync(
            model_id=None,
            worker_idx=worker_idx,
            source=weights_to_update,
            destination=buffer,
        )
        buffer.update_(weights_to_update, non_blocking=True)

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
            torchrl_logger.debug("Applying shared memory weights to model.")
            strategy.apply_weights(model, weights)
            return weights
        return None

    def send_ack(self, message: str = "updated") -> None:
        """No-op for shared memory - no acknowledgment needed."""


@register_weight_sync_backend("shared")
class SharedMemWeightSyncScheme(WeightSyncScheme):
    """Weight synchronization using shared memory.

    This scheme uses shared memory buffers for weight updates, so no weights
    are pickled or sent through a pipe. A background thread on the receiver
    side listens for "receive" instructions from the sender. When an
    instruction arrives, the thread applies the shared buffer to the model and
    sends an acknowledgment.

    Two modes are available, selected by ``atomic``:

    - ``atomic=False`` (the current behavior, also used when ``atomic`` is
      not passed): the shared buffers are the sender's own weights, and each
      worker binds them as its parameters. The workers see every in-place
      change on the sender, including an optimizer step, without any
      ``send()`` call. Because those writes land while a
      worker's policy is running, a policy call can read a mix of two weight
      versions.
    - ``atomic=True``: the sender copies the weights into a shared staging
      buffer on ``send()`` (one per device, or one per worker with
      ``per_worker_weights=True``), and each worker copies that buffer into
      its own private parameters while holding the lock that the receiving
      collector (or :class:`~torchrl.modules.InferenceServer`) holds around
      each policy call. A policy call therefore always reads a single, complete
      version of the weights, and workers only see new weights after
      ``send()`` (for instance through ``update_policy_weights_()``). Memory
      grows by one copy of the weights per worker plus the staging buffers,
      and each update costs one extra copy on the sender and on every
      worker.

    The default changes to ``atomic=True`` in v0.16. Code that trains the
    sender's policy and relies on the workers seeing the change without calling
    ``update_policy_weights_()`` must then pass ``atomic=False`` or call
    ``update_policy_weights_()`` after each update.

    Args:
        strategy (str, optional): The weight transmission strategy,
            ``"tensordict"`` or ``"state_dict"``. Defaults to ``"tensordict"``.
        sync (bool, optional): If ``True``, ``send()`` blocks until every
            receiver acknowledges. If ``False``, ``send()`` returns immediately;
            with ``atomic=True`` the next ``send()`` first waits for the
            previous acknowledgments, so that a staging buffer is never
            overwritten while a worker copies it. With ``atomic=True`` each
            wait lasts at most ``WEIGHT_SYNC_TIMEOUT`` seconds per worker, so a
            dead worker does not block the sender. Defaults to ``True``.
        per_worker_weights (bool, optional): If ``True``, each worker
            maintains independent weights (for distinct policy factories). If
            ``False``, workers on the same device share the same weight buffer.
            This flag is auto-set to ``True`` when distinct policy factories are
            detected. Defaults to ``False``.
        atomic (bool, optional): If ``True``, workers keep a private copy of the
            weights and apply each update between two policy calls, as
            described above. If ``False``, workers read the sender's weights in
            place. ``None`` behaves like ``False`` and emits a
            :class:`FutureWarning` when the scheme is initialized on the sender,
            since the default becomes ``True`` in v0.16. Defaults to ``None``.

    Example:
        >>> from torchrl.weight_update import SharedMemWeightSyncScheme
        >>> scheme = SharedMemWeightSyncScheme(atomic=True)
        >>> scheme.atomic
        True
        >>> # Pass it to a multiprocess collector, then push each new set of
        >>> # weights to the workers explicitly:
        >>> # collector = MultiSyncCollector(
        >>> #     [make_env] * 2, policy, frames_per_batch=64, total_frames=-1,
        >>> #     weight_sync_schemes={"policy": scheme},
        >>> # )
        >>> # for data in collector:
        >>> #     ...  # optimizer step on policy
        >>> #     collector.update_policy_weights_()
    """

    def __init__(
        self,
        strategy: str = "tensordict",
        sync: bool = True,
        per_worker_weights: bool = False,
        atomic: bool | None = None,
    ):
        super().__init__(strategy)
        self.sync = sync
        self.per_worker_weights = per_worker_weights
        self._warn_atomic_default = atomic is None
        self.atomic = bool(atomic)
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

        self._pending_acks: dict[int, int] = {}
        self._stalled_workers: set[int] = set()
        self._sender_weights_source: nn.Module | TensorDictBase | None = None

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
        ctx: Any = None,
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
            ctx: Multiprocessing context. Defaults to `mp.get_context()`.

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

        if ctx is None:
            ctx = mp.get_context()
        for worker_idx in all_workers:
            if worker_idx not in self._weight_init_queues:
                self._weight_init_queues[worker_idx] = ctx.Queue()
            # Create instruction queues for background receiver
            if worker_idx not in self._instruction_queues:
                self._instruction_queues[worker_idx] = ctx.Queue()
            # Create ack queues for synchronous mode
            if worker_idx not in self._ack_queues:
                self._ack_queues[worker_idx] = ctx.Queue()

        if self._warn_atomic_default:
            warnings.warn(
                "SharedMemWeightSyncScheme was created without `atomic`. Workers "
                "currently read the sender's weights in place, so they see every "
                "in-place update (such as an optimizer step) without "
                "update_policy_weights_(), and a policy call can read a mix of two "
                "weight versions. In torchrl v0.16 the default changes to "
                "atomic=True: workers keep a private copy and only see new weights "
                "after update_policy_weights_(). Pass "
                "weight_sync_schemes={'policy': SharedMemWeightSyncScheme(atomic=True)} "
                "to opt in now, or atomic=False to keep the current behavior.",
                FutureWarning,
                stacklevel=3,
            )

        # Set worker info in transport
        self.shared_transport.register_weights(
            params_map,
            self._weight_init_queues,
            per_worker=self.per_worker_weights,
            copy=self.atomic,
        )

        # Store model_id and context on scheme
        self.model_id = model_id
        if context is not None:
            self.context = context
        else:
            self._sender_weights_source = model if model is not None else weights

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
        from torchrl.collectors.utils import (
            _cast,
            _unshareable_devices,
            _warn_cpu_staging,
        )

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
                    # Check if policy factories are uniform or distinct
                    factories = context.policy_factory
                    has_uniform_factories = all(f is factories[0] for f in factories)

                    if has_uniform_factories:
                        # Uniform factories - single model for all workers
                        model = factories[0]()
                    else:
                        # Distinct factories - create per-worker weights directly
                        params_map = {}
                        staged_devices = []
                        for worker_idx, factory in enumerate(factories):
                            if factory is not None:
                                worker_model = factory()
                                if not isinstance(worker_model, nn.Module):
                                    continue
                                worker_weights = TensorDict.from_module(worker_model)
                                worker_weights = worker_weights.data.apply(
                                    _cast, worker_weights
                                )
                                # Move to appropriate device
                                device = None
                                if devices and worker_idx < len(devices):
                                    device = devices[worker_idx]
                                offending = _unshareable_devices(device, worker_weights)
                                if offending:
                                    staged_devices.extend(offending)
                                    worker_weights = worker_weights.to("cpu")
                                elif device is not None:
                                    worker_weights = worker_weights.to(device)
                                worker_weights = worker_weights.share_memory_()
                                params_map[worker_idx] = worker_weights
                        if staged_devices:
                            _warn_cpu_staging(staged_devices)
                        # Set per_worker_weights flag on the scheme
                        self.per_worker_weights = True
                        return params_map
            if not isinstance(model, nn.Module):
                raise TypeError(
                    f"SharedMemWeightSyncScheme requires an nn.Module policy, "
                    f"got {type(model)}. Non-Module policies (e.g. RandomPolicy) "
                    f"do not need weight synchronization."
                )
            weights = TensorDict.from_module(model)
        elif model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    f"SharedMemWeightSyncScheme requires an nn.Module model, "
                    f"got {type(model)}. Non-Module policies (e.g. RandomPolicy) "
                    f"do not need weight synchronization."
                )
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
            staged_weights = None
            staged_devices = []
            for d in devices_set:
                offending = _unshareable_devices(d, weights)
                if offending:
                    if staged_weights is None:
                        staged_weights = weights.to("cpu").apply(_cast, weights)
                        staged_weights.share_memory_()
                    staged_devices.extend(offending)
                    device_map[d] = staged_weights
                elif d != weights_device:
                    # Move to device and apply _cast to preserve Parameter/Buffer types
                    weights_on_device = weights.to(d)
                    weights_on_device = weights_on_device.apply(_cast, weights)
                    device_map[d] = weights_on_device
                else:
                    # Already on correct device, just apply _cast
                    device_map[d] = weights.apply(_cast, weights)
            if staged_devices:
                _warn_cpu_staging(staged_devices)

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

        if fresh_weights is None and self.atomic:
            source = self._sender_weights_source
            if isinstance(source, nn.Module):
                fresh_weights = strategy.extract_weights(source)
            elif source is not None:
                fresh_weights = source
            else:
                raise RuntimeError(
                    "SharedMemWeightSyncScheme(atomic=True).send() was called "
                    "without weights, and the scheme has no model to read them "
                    "from: pass the weights to send(), or initialize the scheme "
                    "with a context, a model or weights."
                )

        if fresh_weights is None:
            return None

        # Update the shared memory buffer(s) in-place so workers see the change.
        # When workers are on different devices, there are multiple unique weight
        # buffers (one per device). We must update ALL of them, not just the first.
        if self._shared_transport is not None and self.shared_transport.unique_weights:
            for shared_weights in self.shared_transport.unique_weights:
                _log_weight_sync(
                    model_id=model_id,
                    worker_idx=None,
                    source=fresh_weights,
                    destination=shared_weights,
                )
                shared_weights.data.update_(fresh_weights.data)
                _synchronize_cuda(shared_weights)
            return self.shared_transport.unique_weights[0]

        # If no shared transport, just return the fresh weights
        return fresh_weights

    def send(
        self,
        weights: TensorDictBase | dict[int, TensorDictBase] | None = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """Send weights via shared memory (in-place update).

        For SharedMemWeightSyncScheme:
        1. With ``atomic=True``, waits until every worker has finished
           copying the previous update, so that a staging buffer is never
           written while a worker reads it
        2. Updates the shared memory buffer(s) in-place
        3. _send_instruction() tells workers to apply the new weights (with
           ``atomic=True``, between two policy calls)
        4. If sync=True, waits for acknowledgments from all workers (with
           ``atomic=True``, at most ``WEIGHT_SYNC_TIMEOUT`` seconds per worker)

        Args:
            weights: Weights to send. Can be:
                - None: Extract weights from the model
                - TensorDictBase: Same weights broadcast to all workers
                - dict[int, TensorDictBase]: Per-worker weights (keys are worker indices)
            worker_ids: Which workers to notify (None = all workers).
        """
        if not self.initialized_on_sender:
            raise RuntimeError("Must be initialized on sender before sending weights")
        if not self.synchronized_on_sender:
            raise RuntimeError("Must be synchronized on sender before sending weights")

        self._wait_for_unacknowledged_workers()

        # Check if per-worker weights dict is provided
        is_per_worker_dict = (
            isinstance(weights, dict)
            and weights
            and all(isinstance(k, int) for k in weights.keys())
        )

        if is_per_worker_dict:
            # Per-worker weights: update each worker's buffer directly
            if not self.per_worker_weights:
                raise ValueError(
                    "Per-worker weights dict provided but scheme was not initialized "
                    "with per_worker_weights=True (or auto-detected from distinct factories)."
                )
            # Update shared memory buffers for each worker
            self.shared_transport.send_weights(weights, worker_ids=worker_ids)
            # Determine which workers to notify
            target_workers = list(weights.keys()) if worker_ids is None else worker_ids
        else:
            # Broadcast mode: prepare_weights updates the shared buffer in-place
            self.prepare_weights(
                weights=weights,
                model_id=self._model_id,
                strategy=self._strategy,
                context=self.context,
            )
            target_workers = worker_ids

        # Send instruction to workers' background threads to apply the weights
        self._send_instruction(instruction="receive", worker_ids=target_workers)

        if self.atomic:
            if target_workers is None:
                target_workers = list(self._instruction_queues)
            elif isinstance(target_workers, int):
                target_workers = [target_workers]
            for worker_idx in target_workers:
                self._pending_acks[worker_idx] = (
                    self._pending_acks.get(worker_idx, 0) + 1
                )
            if self.sync:
                self._wait_for_unacknowledged_workers()
        elif self.sync:
            self._wait_for_ack(worker_ids=target_workers)

    def _wait_for_unacknowledged_workers(self) -> None:
        """Collect the acknowledgments of the updates sent with ``atomic=True``.

        A worker acknowledges once it has copied the staging buffer into its
        parameters. Until then the buffer must not be overwritten. Every
        instruction is acknowledged once, so a late acknowledgment is counted
        against the update it belongs to rather than the next one.

        A worker gets ``WEIGHT_SYNC_TIMEOUT`` seconds. If it does not answer in
        time, a warning is logged and the worker is treated as stalled: later
        sends only collect whatever it has acknowledged in the meantime, without
        waiting, until it catches up. A dead worker therefore does not block
        the sender.
        """
        for worker_idx, pending in list(self._pending_acks.items()):
            stalled = worker_idx in self._stalled_workers
            while pending:
                try:
                    self._ack_queues[worker_idx].get(
                        timeout=0 if stalled else WEIGHT_SYNC_TIMEOUT
                    )
                except Empty:
                    if not stalled:
                        torchrl_logger.warning(
                            f"Worker {worker_idx} did not acknowledge a weight "
                            f"update within {WEIGHT_SYNC_TIMEOUT}s; its staging "
                            "buffer is overwritten while it may still be reading it."
                        )
                        self._stalled_workers.add(worker_idx)
                    break
                pending -= 1
            if pending:
                self._pending_acks[worker_idx] = pending
            else:
                del self._pending_acks[worker_idx]
                self._stalled_workers.discard(worker_idx)

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

        if weights is not None and self.model is not None:
            if self.atomic:
                weights = weights.clone()
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
        while not self._stop_event.is_set():
            try:
                instruction = self._wait_for_instruction()
                if instruction is None:
                    # Stop event was set or timeout
                    continue
                if instruction == "receive":
                    with self._weight_update_lock():
                        self._apply_shared_weights()
                    self._send_ack("updated")
                elif instruction == "stop":
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

    def receive(self, timeout: float | None = None) -> TensorDictBase | None:
        """Apply the shared buffer to the model, as the background thread does.

        The application holds the receiving context's weight-update lock, so a
        manual call cannot race with a policy call or with the background
        receiver thread.
        """
        with self._weight_update_lock():
            return super().receive(timeout=timeout)

    def _weight_update_lock(self) -> contextlib.AbstractContextManager:
        """Return the lock that the receiving context holds around policy calls.

        Copying the staging buffer into the model while holding it guarantees
        that a policy call reads either the previous or the new weights, never a
        mix of both. Collectors and :class:`~torchrl.modules.InferenceServer`
        expose this lock as ``_weight_update_lock``; receivers without such a
        context get a no-op context. With ``atomic=False`` the sender writes
        the workers' parameters directly, so the lock cannot help and is not
        taken.
        """
        if not self.atomic:
            return contextlib.nullcontext()
        return getattr(self.context, "_weight_update_lock", contextlib.nullcontext())

    def _apply_shared_weights(self) -> None:
        """Apply the shared buffer to the model and cascade the update.

        The update is then cascaded through the context's
        ``update_policy_weights_``, which also bumps the PolicyVersion transform
        of a leaf Collector. With ``atomic=True``, the cascade reads the
        model's freshly copied parameters rather than the staging buffer, so a
        context whose policy is the model itself does not copy the weights a
        second time.

        With ``atomic=True``, every copy is complete, on every stream, before
        this returns: the sender may overwrite the staging buffer as soon as it
        is acknowledged, and the next policy call may run on another CUDA
        stream.
        """
        weights = self._receiver_shared_weights
        model = self.model
        if weights is not None and model is not None:
            if not self.atomic:
                self._strategy.apply_weights(model, weights, inplace=True)
            else:
                destination = TensorDict.from_module(model)
                _synchronize_cuda(destination)
                self._strategy.apply_weights(model, weights, inplace=True)
                weights = destination

        model_id = self._model_id or "policy"
        if self.context is not None and hasattr(self.context, "update_policy_weights_"):
            self.context.update_policy_weights_(
                model_id=model_id,
                policy_or_weights=weights,
            )
        if self.atomic and weights is not None:
            _synchronize_cuda(weights)

    def __getstate__(self):
        """Prepare the scheme for pickling."""
        state = super().__getstate__()
        # mp.Queue objects can be pickled and shared across processes
        # Keep them in state so workers have access
        state["_sender_weights_source"] = None
        return state

    def shutdown(self) -> None:
        """Stop the background receiver thread and clean up."""
        # Check if already shutdown
        if getattr(self, "_is_shutdown", False):
            return
        self._is_shutdown = True

        # Signal all workers to stop
        instruction_queues = getattr(self, "_instruction_queues", None)
        if instruction_queues:
            for _, queue in instruction_queues.items():
                queue.put("stop")

        # Let base class handle background thread cleanup
        super().shutdown()

        # Close all multiprocessing queues created by the scheme.
        queues_to_close = []
        for name in ("_weight_init_queues", "_instruction_queues", "_ack_queues"):
            mapping = getattr(self, name, None)
            if not mapping:
                continue
            queues_to_close.extend(mapping.values())
            setattr(self, name, {})

        unique = {}
        for q in queues_to_close:
            unique[id(q)] = q
        for q in unique.values():
            _close_mp_queue(q)
