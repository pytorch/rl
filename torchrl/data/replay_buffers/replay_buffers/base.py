# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import contextlib
import json
import multiprocessing
import pickle
import textwrap
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

from functools import wraps
from typing import Literal, TYPE_CHECKING, TypeVar

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NestedKey,
    TensorClass,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.nn.utils import _set_dispatch_td_nn_modules
from torch import Tensor

try:
    from torch.utils._pytree import tree_leaves, tree_map
except ImportError:
    from torch.utils._pytree import tree_flatten, tree_map

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl._comm.replay_service import _DistributedReplayService, _extend_reply
from torchrl._utils import _RayServiceMetaClass, rl_warnings
from torchrl.data.replay_buffers.query import _query_source, Trajectory
from torchrl.data.replay_buffers.sample_units import SampleUnit
from torchrl.data.replay_buffers.samplers import (
    ConsumingSampler,
    PrioritizedSampler,
    RandomSampler,
    Sampler,
)
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    ListStorage,
    Storage,
    TensorStorage,
)
from torchrl.data.replay_buffers.utils import (
    _is_int,
    _to_numpy,
    INT_CLASSES,
    pin_memory_output,
)
from torchrl.data.replay_buffers.writers import RoundRobinWriter, Writer
from torchrl.envs.transforms.transforms import _InvertTransform, Compose, Transform

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


def _storage_index(index: Any, storage: Storage) -> Any:
    storage_device = getattr(storage, "device", None)
    if storage_device is None or storage_device == "auto":
        return index
    storage_device = torch.device(storage_device)

    def _maybe_to_storage_device(index):
        if isinstance(index, torch.Tensor) and index.device != storage_device:
            return index.to(storage_device)
        return index

    if isinstance(index, tuple):
        return tuple(_maybe_to_storage_device(item) for item in index)
    return _maybe_to_storage_device(index)


def _maybe_delay_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._delayed_init and not self.initialized:
            self._init()
        return func(self, *args, **kwargs)

    return wrapper


class ConditionalUpdateResult(TensorClass["nocast"]):
    """Result of :meth:`ReplayBuffer.update_if_present`.

    Attributes:
        updated (torch.Tensor): boolean mask aligned with the order of the
            indices passed to the update. ``True`` marks records that were
            still live and received the patch; ``False`` marks records that
            were not patched, either because their slot had been reused or
            emptied (stale) or because they were rejected by the version
            comparison. Non-patched records are left untouched.
        version_rejected (torch.Tensor, optional): boolean mask aligned like
            ``updated``. Only present (non-``None``) when the update was
            called with ``version_key``; ``True`` marks records that were
            generation-live but lost the version comparison (including
            duplicate handles on the same slot that did not carry the highest
            incoming version). Every input record lands in exactly one of
            updated / version_rejected / stale.

    Note that ``stale_count`` counts only generation-stale records: when a
    version comparison is active, records rejected by it are counted by
    ``version_rejected_count``, not by ``stale_count``.
    """

    updated: torch.Tensor
    version_rejected: torch.Tensor | None = None

    @property
    def updated_count(self) -> int:
        """Number of records that were live and patched."""
        return int(self.updated.sum().item())

    @property
    def version_rejected_count(self) -> int:
        """Number of records that were live but rejected by version comparison."""
        if self.version_rejected is None:
            return 0
        return int(self.version_rejected.sum().item())

    @property
    def stale_count(self) -> int:
        """Number of records that were stale and skipped."""
        base = int(self.updated.numel()) - self.updated_count
        if self.version_rejected is not None:
            base -= self.version_rejected_count
        return base


class ReplayBuffer(metaclass=_RayServiceMetaClass):
    """A generic, composable replay buffer class.

    See also :class:`~torchrl.trainers.algorithms.configs.ReplayBufferConfig`.

    Keyword Args:
        storage (Storage, Callable[[], Storage], optional): the storage to be used.
            If a callable is passed, it is used as constructor for the storage.
            If none is provided a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, Callable[[], Sampler], optional): the sampler to be used.
            If a callable is passed, it is used as constructor for the sampler.
            If none is provided, a default :class:`~torchrl.data.replay_buffers.RandomSampler`
            will be used.
        sample_unit (SampleUnit, optional): expands the anchors selected by
            the sampler into the records of the batch (see
            :class:`~torchrl.data.replay_buffers.SampleUnit`). ``None``
            (default) is equivalent to
            :class:`~torchrl.data.replay_buffers.Transition`: every anchor is
            one transition and classic behavior is preserved.
        writer (Writer, Callable[[], Writer], optional): the writer to be used.
            If a callable is passed, it is used as constructor for the writer.
            If none is provided a default :class:`~torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform or Callable[[Any], Any], optional): Transform to be executed when
            :meth:`sample` is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. A generic callable can also be passed if the replay buffer
            is used with PyTree structures (see example below).
            Unlike storages, writers and samplers, transform constructors must
            be passed as separate keyword argument :attr:`transform_factory`,
            as it is impossible to distinguish a constructor from a transform.
        transform_factory (Callable[[], Callable], optional): a factory for the
            transform. Exclusive with :attr:`transform`.
        batch_size (int, optional): the batch size to be used when sample() is
            called.

            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. important:: When using a collector with ``trajs_per_batch``,
                trajectories are written as flat 1-D sequences of variable
                length.  Do not set ``dim_extend > 0`` or ``ndim >= 2`` in
                this case — the storage must be 1-dimensional.

            .. note:: This argument has no effect on :meth:`add` and
                therefore should be used with caution when both :meth:`add`
                and :meth:`extend` are used in a codebase. For example:

                    >>> data = torch.zeros(3, 4)
                    >>> rb = ReplayBuffer(
                    ...     storage=LazyTensorStorage(10, ndim=2),
                    ...     dim_extend=1)
                    >>> # these two approaches are equivalent:
                    >>> for d in data.unbind(1):
                    ...     rb.add(d)
                    >>> rb.extend(data)

        generator (torch.Generator, optional): a generator to use for sampling.
            Using a dedicated generator for the replay buffer can allow a fine-grained control
            over seeding, for instance keeping the global seed different but the RB seed identical
            for distributed jobs.
            Defaults to ``None`` (global default generator).

            .. warning:: As of now, the generator has no effect on the transforms.
        consume_after_n_samples (int, optional): if provided, sampled items are
            removed from the sampleable set after they have been returned this
            many times. The default value of ``None`` keeps the standard replay
            buffer behavior. Passing ``1`` makes each item available for a
            single sample before it is consumed.
        shared (bool, optional): whether the buffer will be shared using multiprocessing or not.
            Defaults to ``False``.
        compilable (bool, optional): whether the writer is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.
        delayed_init (bool, optional): whether to initialize storage, writer, sampler and transform
            the first time the buffer is used rather than during construction.
            This is useful when the replay buffer needs to be pickled and sent to remote workers,
            particularly when using transforms with modules that require gradients.
            If not specified, defaults to ``True`` when ``transform_factory`` is provided,
            and ``False`` otherwise.
        service_backend (str): deployment backend, either ``"direct"`` or
            ``"ray"``. Defaults to ``"direct"``.
        service_backend_options (dict, optional): Ray initialization options.
            Accepted keys are ``ray_init_config`` and ``remote_config``.
        transport (str, optional): physical transport used by a remote replay
            owner. ``"auto"`` selects the backend default. Defaults to
            ``"auto"``.
        transport_options (dict, optional): options for the selected transport.
            For ``transport="distributed"``, ``backend`` selects ``"gloo"``
            or ``"nccl"``. TensorDict layouts are bound lazily on first use.

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import ReplayBuffer, ListStorage
        >>>
        >>> torch.manual_seed(0)
        >>> rb = ReplayBuffer(
        ...     storage=ListStorage(max_size=1000),
        ...     batch_size=5,
        ... )
        >>> # populate the replay buffer and get the item indices
        >>> data = range(10)
        >>> indices = rb.extend(data)
        >>> # sample will return as many elements as specified in the constructor
        >>> sample = rb.sample()
        >>> print(sample)
        tensor([4, 9, 3, 0, 3])
        >>> # Passing the batch-size to the sample method overrides the one in the constructor
        >>> sample = rb.sample(batch_size=3)
        >>> print(sample)
        tensor([9, 7, 3])
        >>> # one cans sample using the ``sample`` method or iterate over the buffer
        >>> for i, batch in enumerate(rb):
        ...     print(i, batch)
        ...     if i == 3:
        ...         break
        0 tensor([7, 3, 1, 6, 6])
        1 tensor([9, 8, 6, 6, 8])
        2 tensor([4, 3, 6, 9, 1])
        3 tensor([4, 4, 1, 9, 9])

    Replay buffers accept *any* kind of data. Not all storage types
    will work, as some expect numerical data only, but the default
    :class:`~torchrl.data.ListStorage` will:

    Examples:
        >>> torch.manual_seed(0)
        >>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
        >>> indices = buffer.extend(["a", 1, None])
        >>> buffer.sample(3)
        [None, 'a', None]

    The :class:`~torchrl.data.replay_buffers.TensorStorage`, :class:`~torchrl.data.replay_buffers.LazyMemmapStorage`
    and :class:`~torchrl.data.replay_buffers.LazyTensorStorage` also work
    with any PyTree structure (a PyTree is a nested structure of arbitrary depth made of dicts,
    lists or tuples where the leaves are tensors) provided that it only contains
    tensor data.

    Examples:
        >>> from torch.utils._pytree import tree_map
        >>> def transform(x):
        ...     # Zeros all the data in the pytree
        ...     return tree_map(lambda y: y * 0, x)
        >>> rb = ReplayBuffer(storage=LazyMemmapStorage(100), transform=transform)
        >>> data = {
        ...     "a": torch.randn(3),
        ...     "b": {"c": (torch.zeros(2), [torch.ones(1)])},
        ...     30: -torch.ones(()),
        ... }
        >>> rb.add(data)
        >>> # The sample has a similar structure to the data (with a leading dimension of 10 for each tensor)
        >>> s = rb.sample(10)
        >>> # let's check that our transform did its job:
        >>> def assert0(x):
        >>>     assert (x == 0).all()
        >>> tree_map(assert0, s)

    """

    _accepts_transport_backend = True

    @classmethod
    def _ServiceClass(
        cls,
        service_backend,
        *args,
        service_backend_options=None,
        **kwargs,
    ):
        if service_backend != "ray":
            raise ValueError(
                "ReplayBuffer supports service_backend='direct' or 'ray', "
                f"not {service_backend!r}."
            )
        from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer

        options = dict(service_backend_options or {})
        ray_init_config = options.pop("ray_init_config", None)
        remote_config = options.pop("remote_config", None)
        if options:
            raise TypeError(
                f"Unexpected Ray replay-buffer service options: {sorted(options)}"
            )
        return RayReplayBuffer(
            *args,
            replay_buffer_cls=cls,
            ray_init_config=ray_init_config,
            remote_config=remote_config,
            **kwargs,
        )

    def __init__(
        self,
        *,
        storage: Storage | Callable[[], Storage] | None = None,
        sampler: Sampler | Callable[[], Sampler] | None = None,
        sample_unit: SampleUnit | None = None,
        writer: Writer | Callable[[], Writer] | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: Transform | Callable | None = None,  # noqa-F821
        transform_factory: Callable[[], Transform | Callable]
        | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
        checkpointer: StorageCheckpointerBase  # noqa: F821
        | Callable[[], StorageCheckpointerBase]  # noqa: F821
        | None = None,  # noqa: F821
        generator: torch.Generator | None = None,
        consume_after_n_samples: int | None = None,
        shared: bool = False,
        compilable: bool | None = None,
        delayed_init: bool | None = None,
        service_backend: Literal["direct", "ray"] = "direct",
        service_backend_options: dict[str, Any] | None = None,
        transport: Literal["auto", "direct", "ray", "distributed"] = "auto",
        transport_options: dict[str, Any] | None = None,
    ) -> None:
        if service_backend == "direct" and transport not in ("auto", "direct"):
            raise ValueError(
                "A direct ReplayBuffer only supports transport='auto' or 'direct'."
            )
        if service_backend == "direct" and transport_options:
            raise ValueError(
                "transport_options are only valid for a remote ReplayBuffer."
            )
        del service_backend, service_backend_options, transport, transport_options
        if consume_after_n_samples is not None:
            if isinstance(consume_after_n_samples, bool) or not isinstance(
                consume_after_n_samples, INT_CLASSES
            ):
                raise TypeError("consume_after_n_samples must be a positive integer.")
            if consume_after_n_samples < 1:
                raise ValueError("consume_after_n_samples must be a positive integer.")
            consume_after_n_samples = int(consume_after_n_samples)

        self._delayed_init = delayed_init
        self._initialized = False
        self._service_shutdown = False

        # Store init parameters for potential delayed initialization
        self._init_storage = storage
        self._init_sampler = sampler
        self._init_writer = writer
        self._init_collate_fn = collate_fn
        self._init_transform = transform
        self._init_transform_factory = transform_factory
        self._init_checkpointer = checkpointer
        self._init_generator = generator
        self._init_compilable = compilable
        self._init_consume_after_n_samples = consume_after_n_samples
        self._consume_after_n_samples = consume_after_n_samples

        if transform is not None and transform_factory is not None:
            raise TypeError(
                f"transform and transform_factory are mutually exclusive. "
                f"Got transform={transform} and transform_factory={transform_factory}."
            )

        # Auto-detect delayed_init when transform_factory is provided
        if transform_factory is not None and delayed_init is None:
            delayed_init = True
        elif delayed_init is None:
            delayed_init = False

        # Update _delayed_init after auto-detection
        self._delayed_init = delayed_init

        if sample_unit is not None and not isinstance(sample_unit, SampleUnit):
            raise TypeError(
                f"sample_unit must be a SampleUnit instance, got {type(sample_unit).__name__}."
            )
        self._sample_unit = sample_unit
        self._pin_memory = pin_memory
        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        self._batch_size = batch_size

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if consume_after_n_samples is not None and prefetch:
            raise ValueError(
                "Prefetching is not supported when consume_after_n_samples is set."
            )

        if dim_extend is not None and dim_extend < 0:
            raise ValueError("dim_extend must be a positive value.")
        self._dim_extend = dim_extend

        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        if shared and prefetch:
            raise ValueError(
                "Cannot share prefetched replay buffers. Pass prefetch=0 or "
                "shared=False."
            )
        self.shared = shared
        self.share(self.shared)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()

        # If not delayed, initialize immediately
        if not self._delayed_init:
            self._init()

    def _init(self) -> None:
        """Initialize the replay buffer components.

        This method is called either immediately during __init__ (if delayed_init=False)
        or on first use of the buffer (if delayed_init=True).
        """
        if self._initialized:
            return

        self._initialized = True
        try:
            # Initialize storage
            self._storage = self._maybe_make_storage(
                self._init_storage, compilable=self._init_compilable
            )
            self._storage.attach(self)

            # Initialize sampler
            self._sampler = self._maybe_make_sampler(self._init_sampler)
            self._maybe_make_consuming_sampler()
            self._validate_consuming_sampler()

            # Initialize writer
            self._writer = self._maybe_make_writer(self._init_writer)
            self._writer.register_storage(self._storage)
            self._validate_consuming_writer()

            # Initialize collate function
            self._get_collate_fn(self._init_collate_fn)

            # Initialize transform
            self._transform = self._maybe_make_transform(
                self._init_transform, self._init_transform_factory
            )
            if self.shared:
                self._share_replay_buffer_transform()

            # Check batch_size compatibility with sampler
            if (
                self._batch_size is None
                and hasattr(self._sampler, "drop_last")
                and self._sampler.drop_last
            ):
                raise ValueError(
                    "Samplers with drop_last=True must work with a predictable batch-size. "
                    "Please pass the batch-size to the ReplayBuffer constructor."
                )

            # Set dim_extend properly now that storage is initialized
            if self._dim_extend is None:
                if self._storage is not None:
                    ndim = self._storage.ndim
                    self._dim_extend = ndim - 1
                else:
                    self._dim_extend = 1

            # Set checkpointer and generator
            self._storage.checkpointer = self._init_checkpointer
            self.set_rng(generator=self._init_generator)

            # Initialize prioritized sampler if needed
            self._initialize_prioritized_sampler()

            # Remove init parameters
            self._init_storage = None
            self._init_sampler = None
            self._init_writer = None
            self._init_collate_fn = None
            self._init_transform = None
            self._init_transform_factory = None
            self._init_checkpointer = None
            self._init_generator = None
            self._init_compilable = None
            self._init_consume_after_n_samples = None
        except Exception as e:
            self._initialized = False
            raise e

    @property
    def initialized(self) -> bool:
        """Whether the replay buffer has been initialized."""
        return self._initialized

    def start(self) -> Self:
        """Return this already-started direct replay buffer."""
        if self._service_shutdown:
            raise RuntimeError("A shut down replay buffer cannot be restarted.")
        return self

    @property
    def is_alive(self) -> bool:
        """Whether this direct replay buffer remains available."""
        return not self._service_shutdown

    @property
    def service_backend(self) -> str:
        """The canonical deployment backend for this replay buffer."""
        return "direct"

    def client(self) -> Self:
        """Return ``self`` for the zero-overhead direct backend."""
        return self

    def _start_distributed_service(self, transport_options: dict[str, Any]) -> None:
        """Start the private tensor transport inside a remote owner."""
        if hasattr(self, "_distributed_service"):
            raise RuntimeError("The distributed replay service is already running.")
        options = dict(transport_options)
        extend_spec = options.pop("extend_spec", None)
        sample_spec = options.pop("sample_spec", None)
        priority_spec = options.pop("priority_spec", None)
        self._distributed_service = _DistributedReplayService(
            self,
            extend_spec=extend_spec,
            sample_spec=sample_spec,
            priority_spec=priority_spec,
            **options,
        ).start()

    def _bootstrap_distributed_extend(self, data: TensorDictBase):
        """Bind the extend schema and apply exactly one first operation."""
        service = self._distributed_service
        with service._lock:
            if service.extend_transport is None:
                result = self.extend(data)
                reply = _extend_reply(result, service._wire_device)
                service.bind_extend(data, reply)
                return service.extend_client(), reply.get("result", None), True
            return service.extend_client(), None, False

    def _bootstrap_distributed_sample(self, batch_size: int):
        """Bind the sample schema and return exactly one first sample."""
        service = self._distributed_service
        with service._lock:
            if service.sample_transport is None:
                result = self.sample(batch_size).to(service._wire_device)
                service.bind_sample(result)
                return service.sample_client(), result, True, batch_size
            return (
                service.sample_client(),
                None,
                False,
                service._sample_batch_size,
            )

    def _bootstrap_distributed_priority(self, data: TensorDictBase):
        """Bind the priority schema and apply exactly one first update."""
        service = self._distributed_service
        with service._lock:
            if service.priority_transport is None:
                update = getattr(self, "update_tensordict_priority", None)
                if update is not None:
                    update(data)
                service.bind_priority(data)
                return service.priority_client(), True
            return service.priority_client(), False

    def _distributed_control_client(self):
        """Return a restricted control endpoint for length and write count."""
        return self._distributed_service.control_client()

    def _distributed_service_client(self):
        """Create an independently routed client for the private service."""
        service = getattr(self, "_distributed_service", None)
        if service is None:
            raise RuntimeError("The distributed replay service is not running.")
        return service.client()

    def _shutdown_distributed_service(self) -> None:
        """Stop the private tensor transport when the remote owner closes."""
        service = getattr(self, "_distributed_service", None)
        if service is not None:
            service.shutdown()
            del self._distributed_service

    def shutdown(self, timeout: float | None = None) -> None:
        """Mark this direct replay-buffer owner as shut down."""
        del timeout
        self._service_shutdown = True

    def _initialize_prioritized_sampler(self) -> None:
        """Initialize priority trees for existing data when using PrioritizedSampler.

        This method ensures that when a PrioritizedSampler is used with storage that
        already contains data, the priority trees are properly populated with default
        priorities for all existing entries.
        """
        if isinstance(self._sampler, PrioritizedSampler) and len(self._storage) > 0:
            # Set default priorities for all existing data
            device = getattr(self._storage, "device", None)
            if device == "auto":
                device = None
            indices = torch.arange(len(self._storage), dtype=torch.long, device=device)
            default_priorities = torch.full(
                (len(self._storage),),
                self._sampler.default_priority,
                dtype=torch.float,
                device=device,
            )
            self._sampler.update_priority(
                indices, default_priorities, storage=self._storage
            )

    def _maybe_make_storage(
        self, storage: Storage | Callable[[], Storage] | None, compilable
    ) -> Storage:
        if storage is None:
            return ListStorage(max_size=1_000, compilable=compilable)
        elif isinstance(storage, Storage):
            return storage
        elif callable(storage):
            storage = storage()
        if not isinstance(storage, Storage):
            raise TypeError(
                "storage must be either a Storage or a callable returning a storage instance."
            )
        return storage

    def _maybe_make_sampler(
        self, sampler: Sampler | Callable[[], Sampler] | None
    ) -> Sampler:
        if sampler is None:
            return RandomSampler()
        elif isinstance(sampler, Sampler):
            return sampler
        elif callable(sampler):
            sampler = sampler()
        if not isinstance(sampler, Sampler):
            raise TypeError(
                "sampler must be either a Sampler or a callable returning a sampler instance."
            )
        return sampler

    def _maybe_make_consuming_sampler(self) -> None:
        consume_after_n_samples = self._init_consume_after_n_samples
        if consume_after_n_samples is None:
            if isinstance(self._sampler, ConsumingSampler):
                self._consume_after_n_samples = self._sampler.max_sample_count
            return

        if isinstance(self._sampler, ConsumingSampler):
            if self._sampler.max_sample_count != consume_after_n_samples:
                raise ValueError(
                    "consume_after_n_samples conflicts with the provided "
                    "ConsumingSampler.max_sample_count."
                )
            return
        if not isinstance(self._sampler, RandomSampler):
            raise ValueError(
                "consume_after_n_samples only supports the default RandomSampler "
                "or an explicit ConsumingSampler. Prioritized, slice and "
                "without-replacement samplers are not supported."
            )
        self._sampler = ConsumingSampler(max_sample_count=consume_after_n_samples)

    def _validate_consuming_sampler(self) -> None:
        if not isinstance(self._sampler, ConsumingSampler):
            return
        if self._prefetch:
            raise ValueError("Prefetching is not supported with ConsumingSampler.")
        if self._storage.ndim != 1:
            raise ValueError(
                "ConsumingSampler only supports 1-dimensional storages. "
                f"Got storage.ndim={self._storage.ndim}."
            )
        if not isinstance(self._storage, (ListStorage, TensorStorage)):
            raise TypeError(
                "ConsumingSampler only supports ListStorage, TensorStorage, "
                "LazyTensorStorage and LazyMemmapStorage."
            )

    def _validate_consuming_writer(self) -> None:
        if not isinstance(self._sampler, ConsumingSampler):
            return
        if not callable(getattr(self._writer, "write_at", None)):
            raise TypeError(
                "ConsumingSampler requires a writer with a callable "
                "write_at(index, data) method."
            )

    def _maybe_make_writer(
        self, writer: Writer | Callable[[], Writer] | None
    ) -> Writer:
        if writer is None:
            return RoundRobinWriter()
        elif isinstance(writer, Writer):
            return writer
        elif callable(writer):
            writer = writer()
        if not isinstance(writer, Writer):
            raise TypeError(
                "writer must be either a Writer or a callable returning a writer instance."
            )
        return writer

    def _maybe_make_transform(
        self,
        transform: Transform | Callable[[], Transform] | None,
        transform_factory: Callable | None,
    ) -> Transform:
        from torchrl.envs.transforms.transforms import (
            _CallableTransform,
            Compose,
            Transform,
        )

        if transform_factory is not None:
            if transform is not None:
                raise TypeError(
                    "transform and transform_factory cannot be used simultaneously"
                )
            transform = transform_factory()
        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            if not isinstance(transform, Transform) and callable(transform):
                transform = _CallableTransform(transform)
            elif not isinstance(transform, Transform):
                raise RuntimeError(
                    "transform must be either a Transform instance or a callable."
                )
            transform = Compose(transform)
        transform.eval()
        return transform

    def _share_replay_buffer_transform(self) -> None:
        transform = getattr(self, "_transform", None)
        if transform is None:
            return
        self._share_transform_state(transform)

    @classmethod
    def _share_transform_state(cls, transform) -> None:
        if isinstance(transform, Compose):
            for subtransform in transform:
                cls._share_transform_state(subtransform)
            return
        share_memory = getattr(transform, "share_memory_", None)
        if callable(share_memory):
            share_memory()
            return
        if getattr(transform, "requires_shared_write_state", False):
            raise RuntimeError(
                f"{type(transform).__name__} keeps replay-buffer write state "
                "but does not implement share_memory_(). Use a centralized "
                "writer, a Ray-backed transform, or a transform that supports "
                "shared replay-buffer write state."
            )

    def share(self, shared: bool = True) -> Self:
        self.shared = shared
        if self.shared:
            self._write_lock = multiprocessing.Lock()
            if getattr(self, "_initialized", False):
                self._share_replay_buffer_transform()
        else:
            self._write_lock = contextlib.nullcontext()
        return self

    @_maybe_delay_init
    def set_rng(self, generator) -> None:
        self._rng = generator
        self._storage._rng = generator
        self._sampler._rng = generator
        self._writer._rng = generator

    @property
    def dim_extend(self):
        return self._dim_extend

    @property
    def batch_size(self):
        """The batch size of the replay buffer.

        The batch size can be overridden by setting the `batch_size` parameter in the :meth:`sample` method.

        It defines both the number of samples returned by :meth:`sample` and the number of samples that are
        yielded by the :class:`ReplayBuffer` iterator.
        """
        return self._batch_size

    @dim_extend.setter
    def dim_extend(self, value):
        if (
            hasattr(self, "_dim_extend")
            and self._dim_extend is not None
            and self._dim_extend != value
        ):
            raise RuntimeError(
                "dim_extend cannot be reset. Please create a new replay buffer."
            )

        if value is None:
            if self._initialized and self._storage is not None:
                ndim = self._storage.ndim
                value = ndim - 1
            else:
                value = 1

        self._dim_extend = value

    def _transpose(self, data):
        if is_tensor_collection(data):
            return data.transpose(self.dim_extend, 0)
        return tree_map(lambda x: x.transpose(self.dim_extend, 0), data)

    def _get_collate_fn(self, collate_fn):
        self._collate_fn = (
            collate_fn
            if collate_fn is not None
            else _get_default_collate(self._storage, _is_tensordict=self._is_tensordict)
        )

    @_maybe_delay_init
    def set_storage(self, storage: Storage, collate_fn: Callable | None = None):
        """Sets a new storage in the replay buffer and returns the previous storage.

        Args:
            storage (Storage): the new storage for the buffer.
            collate_fn (callable, optional): if provided, the collate_fn is set to this
                value. Otherwise it is reset to a default value.

        """
        prev_storage = self._storage
        self._storage = storage
        self._validate_consuming_sampler()
        self._get_collate_fn(collate_fn)

        return prev_storage

    @_maybe_delay_init
    def set_writer(self, writer: Writer):
        """Sets a new writer in the replay buffer and returns the previous writer."""
        prev_writer = self._writer
        self._writer = writer
        self._writer.register_storage(self._storage)
        return prev_writer

    @_maybe_delay_init
    def set_sampler(self, sampler: Sampler):
        """Sets a new sampler in the replay buffer and returns the previous sampler."""
        prev_sampler = self._sampler
        self._sampler = sampler
        if isinstance(sampler, ConsumingSampler):
            self._consume_after_n_samples = sampler.max_sample_count
        elif isinstance(prev_sampler, ConsumingSampler):
            self._consume_after_n_samples = None
        self._validate_consuming_sampler()
        return prev_sampler

    @_maybe_delay_init
    def __len__(self) -> int:
        with self._replay_lock:
            if isinstance(self._sampler, ConsumingSampler):
                return self._sampler._num_sampleable(self._storage)
            return len(self._storage)

    def _getattr(self, attr):
        # To access properties in remote settings, see RayReplayBuffer.write_count for instance
        return getattr(self, attr)

    def _setattr(self, attr, value):
        # To set properties in remote settings
        setattr(self, attr, value)
        return None  # explicit return for remote calls

    @property
    @_maybe_delay_init
    def write_count(self) -> int:
        """The total number of items written so far in the buffer through add and extend."""
        return self._writer._write_count

    def stats(self) -> dict[str, int | float | bool]:
        """Returns a cheap, serializable snapshot of the buffer's operational state.

        The snapshot only contains scalar counters and gauges. It never
        includes the storage content, does not modify the buffer state and is
        safe to call concurrently with writes and samples. Cumulative
        counters such as ``write_count`` are meant to be converted into rates
        by an external monitor such as
        :class:`~torchrl.record.loggers.monitoring.LoggerMonitor`.

        Calling this method on an uninitialized buffer does not trigger its
        initialization; an empty snapshot with ``initialized=False`` is
        returned instead (``capacity`` is still reported when the storage
        already advertises it).

        Returns:
            A dictionary with the following entries:

            - ``"size"``: current number of elements in the buffer (mirrors ``len(buffer)``);
            - ``"write_count"``: total number of items written through ``add`` and
              ``extend`` (``0`` for writers that do not track writes, such as
              :class:`~torchrl.data.replay_buffers.writers.ImmutableDatasetWriter`);
            - ``"prefetch_queue_size"``: number of pending prefetched batches;
            - ``"initialized"``: whether the buffer components are initialized;
            - ``"capacity"``: maximum number of elements the storage can hold
              (only present when the storage advertises a ``max_size``);
            - ``"utilization"``: ``size / capacity`` (only present alongside ``capacity``).

            Remote clients backed by the distributed transport report a subset
            of these entries (``size`` and ``write_count``).

        Examples:
            >>> import torch
            >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
            >>> rb = ReplayBuffer(storage=LazyTensorStorage(10))
            >>> rb.extend(torch.arange(5))
            >>> snapshot = rb.stats()
            >>> print(snapshot["size"], snapshot["write_count"], snapshot["capacity"])
            5 5 10
        """
        if not self.initialized:
            stats = {
                "size": 0,
                "write_count": 0,
                "prefetch_queue_size": 0,
                "initialized": False,
            }
            storage = getattr(self, "_storage", None) or getattr(
                self, "_init_storage", None
            )
            capacity = getattr(storage, "max_size", None)
            if isinstance(capacity, int):
                stats["capacity"] = capacity
                stats["utilization"] = 0.0
            return stats
        with self._replay_lock:
            size = len(self)
            capacity = getattr(self._storage, "max_size", None)
            write_count = getattr(self._writer, "_write_count", 0)
            prefetch_queue_size = len(self._prefetch_queue)
        stats = {
            "size": int(size),
            "write_count": int(write_count),
            "prefetch_queue_size": int(prefetch_queue_size),
            "initialized": True,
        }
        if capacity is not None:
            stats["capacity"] = int(capacity)
            stats["utilization"] = float(size) / capacity if capacity else 0.0
        return stats

    def update_if_present(
        self,
        *,
        index: torch.Tensor,
        generation: torch.Tensor,
        patch: Mapping[NestedKey, torch.Tensor] | TensorDictBase,
        version_key: NestedKey | None = None,
        version: int | torch.Tensor | None = None,
        require_newer: bool = False,
    ) -> ConditionalUpdateResult:
        """Conditionally updates stored records that are still live.

        Replay slots are recycled by round-robin writers, so a physical index
        captured at sampling time can point to a different record by the time
        an asynchronous computation writes back. This method applies ``patch``
        only to records whose ``(index, generation)`` pair still matches the
        writer's current slot generation, skipping records whose slot was
        reused or emptied since the handle was captured. Skipped records are
        never modified.

        The whole patch is validated (key existence, shape and dtype) before
        any write happens; a validation failure leaves the storage untouched.
        Updating a record refreshes its content, not its identity: the same
        handle keeps working until the slot is rewritten by ``add``,
        ``extend`` or ``empty``.

        Generation tracking is opt-in: the buffer must be constructed with a
        writer that tracks slot generations, e.g.
        ``RoundRobinWriter(track_generations=True)`` (see
        :ref:`ref_buffers_generations`). Calling this method on a buffer whose
        writer does not track generations raises a ``RuntimeError``.

        Keyword Args:
            index (torch.Tensor): storage indices, as returned by
                :meth:`extend` or found in the sample under ``"index"``.
            generation (torch.Tensor): slot generations captured with the
                indices, as found in the sample under ``"index_generation"``.
            patch (mapping of NestedKey to torch.Tensor, or TensorDictBase):
                the fields to overwrite for live records. Leading dimension
                must match the number of records addressed by ``index``.
            version_key (NestedKey, optional): a stored per-record scalar
                field holding each record's current version. When passed
                (together with ``version``), a generation-live record is only
                patched if the incoming version compares favorably against
                the stored one, and the accepted version is written into
                ``version_key`` atomically with the patch. ``version_key``
                may not appear in ``patch``. Nested keys must be passed in
                tuple form (``("nested", "version")``); dotted strings are
                rejected. Defaults to ``None`` (no version comparison).
            version (int or torch.Tensor, optional): the incoming version,
                either a scalar (broadcast to every record) or a tensor with
                one entry per record. Must be passed together with
                ``version_key``.
            require_newer (bool, optional): if ``True``, a record is only
                patched when ``version > stored``; if ``False``, ties are
                accepted (``version >= stored``). When the same slot is
                addressed several times in one call, only the row carrying
                the highest incoming version is applied (the last such row
                on ties); the losing rows are reported in
                ``version_rejected``. Defaults to ``False``.

        Returns:
            A :class:`ConditionalUpdateResult` whose ``updated`` mask is
            aligned with the input index order, with ``updated_count`` and
            ``stale_count`` conveniences. When ``version_key`` is passed, its
            ``version_rejected`` mask marks generation-live records that were
            rejected by the version comparison (``None`` otherwise).

        Raises:
            RuntimeError: if the storage does not support conditional updates
                (for example :class:`ListStorage`) or the writer does not
                track slot generations.
            KeyError: if a patch key (or ``version_key``) does not exist in
                the storage.
            ValueError: if a patch entry has an incompatible shape or dtype,
                if only one of ``version_key`` / ``version`` is passed, if
                ``version_key`` appears in ``patch`` or names a non-scalar
                field, or if it is a dotted string.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.data import (
            ...     LazyTensorStorage,
            ...     TensorDictReplayBuffer,
            ...     TensorDictRoundRobinWriter,
            ... )
            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(10),
            ...     writer=TensorDictRoundRobinWriter(track_generations=True),
            ...     batch_size=4,
            ... )
            >>> rb.extend(TensorDict({"obs": torch.zeros(10, 3)}, batch_size=[10]))
            >>> sample = rb.sample()
            >>> result = rb.update_if_present(
            ...     index=sample["index"],
            ...     generation=sample["index_generation"],
            ...     patch={"obs": torch.ones(4, 3)},
            ... )
            >>> print(result.updated_count, result.stale_count)
            4 0

            With a version comparison, outdated asynchronous writers lose
            deterministically:

            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(10),
            ...     writer=TensorDictRoundRobinWriter(track_generations=True),
            ...     batch_size=4,
            ... )
            >>> rb.extend(
            ...     TensorDict(
            ...         {
            ...             "obs": torch.zeros(10, 3),
            ...             "v": torch.full((10,), 5, dtype=torch.int64),
            ...         },
            ...         batch_size=[10],
            ...     )
            ... )
            >>> sample = rb.sample()
            >>> result = rb.update_if_present(
            ...     index=sample["index"],
            ...     generation=sample["index_generation"],
            ...     patch={"obs": torch.ones(4, 3)},
            ...     version_key="v",
            ...     version=4,
            ...     require_newer=True,
            ... )
            >>> print(result.updated_count, result.version_rejected_count)
            0 4
        """
        storage = self._storage
        if not getattr(storage, "supports_conditional_update", False) or not getattr(
            self._writer, "tracks_generations", False
        ):
            raise RuntimeError(
                f"Conditional updates are not supported by {type(storage).__name__} "
                f"with {type(self._writer).__name__}: the storage must support "
                "conditional updates and the writer must track slot generations."
            )
        index = torch.as_tensor(index, dtype=torch.long)
        dim0 = index[..., 0] if index.ndim > 1 else index.reshape(-1)
        generation = torch.as_tensor(generation, dtype=torch.long).reshape(-1)
        if generation.numel() != dim0.numel():
            raise ValueError(
                f"index and generation must address the same number of records, "
                f"got {dim0.numel()} indices and {generation.numel()} generations."
            )
        if (version_key is None) != (version is None):
            raise ValueError("version_key and version must be provided together.")

        if isinstance(patch, TensorDictBase):
            patch = dict(patch.items(include_nested=True, leaves_only=True))
        else:
            patch = dict(patch)
        # Normalize key spellings so ("v",) and "v" (and the patch's own keys)
        # cannot silently name the same field under different forms.
        patch = {unravel_key(key): value for key, value in patch.items()}

        version_leaf = None
        if version_key is not None:
            if isinstance(version_key, str) and "." in version_key:
                raise ValueError(
                    f"Dotted-string version keys are not supported: got "
                    f"{version_key!r}. Pass the nested key in tuple form, e.g. "
                    f"{tuple(version_key.split('.'))!r}."
                )
            version_key = unravel_key(version_key)
            if version_key in patch:
                raise ValueError(
                    f"version_key {version_key!r} may not appear in patch."
                )
            # Raises KeyError if the field does not exist; the version field
            # must hold one scalar per record (trailing singleton dims are
            # accepted and squeezed).
            version_leaf = storage._conditional_patch_leaf(version_key)
            n_coords = index.shape[-1] if index.ndim > 1 else 1
            feature_shape = version_leaf.shape[n_coords:]
            if any(dim != 1 for dim in feature_shape):
                raise ValueError(
                    f"version_key {version_key!r} must reference a per-record "
                    f"scalar field; the storage holds feature shape "
                    f"{tuple(feature_shape)}."
                )
            version_tensor = torch.as_tensor(version)
            while version_tensor.ndim > 1 and version_tensor.shape[-1] == 1:
                version_tensor = version_tensor.squeeze(-1)
            if version_tensor.ndim == 0:
                version_tensor = version_tensor.expand(dim0.shape)
            elif version_tensor.ndim != 1 or version_tensor.numel() != dim0.numel():
                raise ValueError(
                    f"version must be a scalar or hold one entry per record, "
                    f"got shape {tuple(torch.as_tensor(version).shape)} for "
                    f"{dim0.numel()} records."
                )
            patch[version_key] = version_tensor.reshape((dim0.numel(), *feature_shape))

        normalized = storage._validate_conditional_patch(index, patch)
        version_rejected = None

        with self._replay_lock, self._write_lock:
            # ``generations_of`` returns on the index device; align the captured
            # generations with it so the comparison never crosses devices
            # (index/generation/storage may live on CPU, CUDA or MPS).
            current = self._writer.generations_of(dim0)
            captured = generation.to(current.device)
            live = (current >= 0) & (captured >= 0) & (current == captured)
            if version_key is not None:
                version_rejected = torch.zeros_like(live)
            if live.any():
                if version_key is not None:
                    live_mask = live.to(version_leaf.device)
                    live_index = index.to(version_leaf.device)[live_mask]
                    # Read the stored versions the same way the write path
                    # addresses the storage, so ndim > 1 storages resolve
                    # coordinates instead of fancy-indexing dim 0.
                    if live_index.ndim > 1:
                        coords = tuple(live_index.unbind(-1))
                    else:
                        coords = (live_index,)
                    stored_version = version_leaf[coords].reshape(-1)
                    incoming_version = normalized[version_key][live_mask].reshape(-1)
                    if require_newer:
                        version_accepted = incoming_version > stored_version
                    else:
                        version_accepted = incoming_version >= stored_version

                    # A record addressed several times in one call would make
                    # the scatter write order-dependent (every row compares
                    # against the pre-call version, and the last write wins).
                    # Keep, per record, only the row carrying the highest
                    # incoming version -- the last such row on ties -- and
                    # reject the rest, so the result stays truthful. Records
                    # are identified by their full coordinates: for ndim > 1
                    # storages, cells sharing a dim-0 slot are distinct.
                    if live_index.ndim > 1:
                        slots, slot_ids = torch.unique(
                            live_index, dim=0, return_inverse=True
                        )
                        n_slots = slots.shape[0]
                    else:
                        slots, slot_ids = torch.unique(live_index, return_inverse=True)
                        n_slots = slots.numel()
                    if n_slots != slot_ids.numel():
                        if incoming_version.is_floating_point():
                            lowest = torch.finfo(incoming_version.dtype).min
                        else:
                            lowest = torch.iinfo(incoming_version.dtype).min
                        slot_max = torch.full(
                            (n_slots,),
                            lowest,
                            dtype=incoming_version.dtype,
                            device=version_leaf.device,
                        ).scatter_reduce(0, slot_ids, incoming_version, "amax")
                        is_max = incoming_version == slot_max[slot_ids]
                        row = torch.arange(slot_ids.numel(), device=version_leaf.device)
                        best_row = torch.full(
                            (n_slots,),
                            -1,
                            dtype=torch.int64,
                            device=version_leaf.device,
                        ).scatter_reduce(
                            0,
                            slot_ids,
                            torch.where(
                                is_max, row, row.new_full((), -1).expand_as(row)
                            ),
                            "amax",
                        )
                        version_accepted = version_accepted & (
                            row == best_row[slot_ids]
                        )

                    accepted = version_accepted.to(live.device)
                    version_rejected[live] = ~accepted
                    new_live = torch.zeros_like(live)
                    new_live[live] = accepted
                    live = new_live

                if live.any():
                    live_index = index[live.to(index.device)]
                    storage._apply_conditional_patch(
                        live_index,
                        {
                            key: value[live.to(value.device)]
                            for key, value in normalized.items()
                        },
                    )
        return ConditionalUpdateResult(
            updated=live,
            version_rejected=version_rejected,
            batch_size=live.shape,
        )

    def __repr__(self) -> str:
        from torchrl.envs.transforms import Compose

        storage = textwrap.indent(f"storage={getattr(self, '_storage', None)}", " " * 4)
        writer = textwrap.indent(f"writer={getattr(self, '_writer', None)}", " " * 4)
        sampler = textwrap.indent(f"sampler={getattr(self, '_sampler', None)}", " " * 4)
        if getattr(self, "_transform", None) is not None and not (
            isinstance(self._transform, Compose)
            and not len(getattr(self, "_transform", None))
        ):
            transform = textwrap.indent(
                f"transform={getattr(self, '_transform', None)}", " " * 4
            )
            transform = f"\n{self._transform}, "
        else:
            transform = ""
        batch_size = textwrap.indent(
            f"batch_size={getattr(self, '_batch_size', None)}", " " * 4
        )
        collate_fn = textwrap.indent(
            f"collate_fn={getattr(self, '_collate_fn', None)}", " " * 4
        )
        return f"{self.__class__.__name__}(\n{storage}, \n{sampler}, \n{writer}, {transform}\n{batch_size}, \n{collate_fn})"

    @_maybe_delay_init
    @pin_memory_output
    def __getitem__(self, index: int | torch.Tensor | NestedKey) -> Any:
        if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
            return self[:][index]
        if isinstance(index, tuple):
            if len(index) == 1:
                return self[index[0]]
            else:
                return self[:][index]
        index = _to_numpy(index)

        if self.dim_extend > 0:
            index = (slice(None),) * self.dim_extend + (index,)
            with self._replay_lock:
                data = self._storage[index]
            data = self._transpose(data)
        else:
            with self._replay_lock:
                data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        if self._transform is not None and len(self._transform):
            with data.unlock_() if is_tensor_collection(
                data
            ) else contextlib.nullcontext():
                data = self._transform(data)

        return data

    @_maybe_delay_init
    def __setitem__(self, index, value) -> None:
        if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
            self[:][index] = value
            return
        if isinstance(index, tuple):
            if len(index) == 1:
                self[index[0]] = value
            else:
                self[:][index] = value
            return
        index = _to_numpy(index)

        if self._transform is not None and len(self._transform):
            value = self._transform.inv(value)

        if self.dim_extend > 0:
            index = (slice(None),) * self.dim_extend + (index,)
            with self._replay_lock:
                self._storage[index] = self._transpose(value)
        else:
            with self._replay_lock:
                self._storage[index] = value
        return

    @_maybe_delay_init
    def read_all_in_order(self, end: int | None = None) -> Any:
        """Read storage contents in physical order.

        This is equivalent to ``rb[:]`` when ``end`` is ``None``.

        Args:
            end (int, optional): Number of leading storage entries to read.
                Defaults to the entire storage slice.

        Returns:
            A storage slice containing entries ``[:end]``.
        """
        if end is None:
            return self[:]
        return self[:end]

    @_maybe_delay_init
    def write_all(self, data: Any, end: int | None = None) -> None:
        """Write data back to storage in physical order.

        This is equivalent to ``rb[:end] = data``. If ``end`` is ``None``,
        ``end`` defaults to ``data.shape[0]`` for tensor collections and
        ``len(data)`` otherwise. If ``data`` spans the full storage, this is
        equivalent to ``rb[:] = data``.

        Args:
            data: Data to write to storage.
            end (int, optional): Number of leading storage entries to update.
                Defaults to ``data.shape[0]`` for tensor collections and
                ``len(data)`` otherwise.
        """
        if end is None:
            max_size = getattr(self._storage, "max_size", None)
            if max_size is not None and len(self) == max_size:
                self[:] = data
                return
            end = data.shape[0] if is_tensor_collection(data) else len(data)
        self[:end] = data

    @_maybe_delay_init
    def set_at_(self, key, value, index):
        """Sets the value of a key at specified indices in the replay buffer.

        Args:
            key (NestedKey): the key to set.
            value (torch.Tensor): the value to write.
            index: the indices where to write the value.

        Returns:
            self

        """
        index = _to_numpy(index)
        with self._replay_lock:
            self._storage[:].set_at_(key, value, index)
            self._storage._bump_mutation_revision()
        return self

    @_maybe_delay_init
    def set_(self, key, value):
        """Sets the value of a key across the entire replay buffer in-place.

        Args:
            key (NestedKey): the key to set.
            value (torch.Tensor): the value to write.

        Returns:
            self

        """
        with self._replay_lock:
            self._storage[:].set_(key, value)
            self._storage._bump_mutation_revision()
        return self

    @_maybe_delay_init
    def update_(self, input_dict_or_td, clone=False, *, keys_to_update=None):
        """Updates the replay buffer in-place with the given dict or TensorDict.

        Args:
            input_dict_or_td (dict or TensorDictBase): the data to update with.
            clone (bool, optional): whether to clone the values before writing.
                Defaults to ``False``.
            keys_to_update (sequence of NestedKey, optional): if provided, only
                these keys will be updated.

        Returns:
            self

        """
        with self._replay_lock:
            self._storage[:].update_(
                input_dict_or_td,
                clone=clone,
                keys_to_update=keys_to_update,
            )
            self._storage._bump_mutation_revision()
        return self

    def _clone_prefetch_result(
        self, result: tuple[Any, dict[str, Any]]
    ) -> tuple[Any, dict[str, Any]]:
        memo = {}

        def _clone_leaf(value: Any) -> Any:
            value_id = id(value)
            if value_id in memo:
                return memo[value_id]
            if isinstance(value, Tensor):
                # PyTorch deliberately rejects deepcopy for non-leaf tensors. A
                # checkpoint only needs their value and requires-grad property,
                # not the live autograd graph that produced the sample.
                cloned = value.detach().clone()
                if value.requires_grad:
                    cloned.requires_grad_()
            else:
                cloned = deepcopy(value, memo)
            memo[value_id] = cloned
            return cloned

        return tree_map(_clone_leaf, result)

    @contextlib.contextmanager
    def _capture_prefetch_state(
        self, *, clone_queue: bool = True
    ) -> Iterator[dict[str, Any]]:
        with self._futures_lock:
            results = (
                tuple(future.result() for future in self._prefetch_queue)
                if self._prefetch
                else ()
            )
            # Futures acquire the replay lock while sampling, so only take it
            # after every queued future has settled. Holding both locks while
            # the caller captures the remaining components keeps the queue,
            # storage, sampler, writer and RNG at one logical point in time.
            with self._replay_lock:
                queue = (
                    tuple(self._clone_prefetch_result(result) for result in results)
                    if clone_queue
                    else results
                )
                yield {
                    "version": 1,
                    "capacity": int(self._prefetch_cap),
                    "queue": queue,
                }

    def _clear_prefetch_queue_locked(self) -> None:
        futures = tuple(self._prefetch_queue)
        for future in futures:
            future.cancel()
        if futures:
            wait(futures)
        self._prefetch_queue.clear()

    def _validate_prefetch_state(self, prefetch_state: dict[str, Any] | None) -> None:
        if prefetch_state is None:
            return
        if not isinstance(prefetch_state, dict):
            raise TypeError("The prefetch state must be a dictionary.")
        if prefetch_state.get("version") != 1:
            raise RuntimeError(
                f"Unsupported prefetch state version: {prefetch_state.get('version')}."
            )
        capacity = prefetch_state.get("capacity")
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, INT_CLASSES)
            or capacity < 0
        ):
            raise ValueError(
                "The saved prefetch capacity must be a non-negative integer."
            )
        capacity = int(capacity)
        if capacity != int(self._prefetch_cap):
            raise RuntimeError(
                f"Cannot restore a prefetch queue with capacity {capacity} into a "
                f"replay buffer with capacity {self._prefetch_cap}."
            )
        queue = prefetch_state.get("queue")
        if not isinstance(queue, (list, tuple)):
            raise TypeError("The saved prefetch queue must be a list or tuple.")
        if len(queue) > capacity:
            raise ValueError(
                f"The saved prefetch queue contains {len(queue)} entries but its "
                f"capacity is {capacity}."
            )
        if queue and not self._prefetch:
            raise RuntimeError(
                "Cannot restore a non-empty prefetch queue when prefetching is disabled."
            )

    def _restore_prefetch_queue_locked(
        self,
        prefetch_state: dict[str, Any] | None,
        *,
        clone_queue: bool = True,
    ) -> None:
        self._validate_prefetch_state(prefetch_state)
        if prefetch_state is None:
            return
        queue = prefetch_state["queue"]
        for result in queue:
            future = Future()
            if clone_queue:
                result = self._clone_prefetch_result(result)
            future.set_result(result)
            self._prefetch_queue.append(future)

    def _dump_prefetch_state(self, path: Path, prefetch_state: dict[str, Any]) -> None:
        with open(path / "prefetch.pkl", "wb") as file:
            pickle.dump(prefetch_state, file)

    def _load_prefetch_state(self, path: Path) -> dict[str, Any] | None:
        prefetch_path = path / "prefetch.pkl"
        if not prefetch_path.exists():
            return None
        with open(prefetch_path, "rb") as file:
            return pickle.load(file)

    @_maybe_delay_init
    def state_dict(self) -> dict[str, Any]:
        with self._capture_prefetch_state() as prefetch_state:
            return {
                "_storage": self._storage.state_dict(),
                "_sampler": self._sampler.state_dict(),
                "_writer": self._writer.state_dict(),
                "_transforms": self._transform.state_dict(),
                "_batch_size": self._batch_size,
                "_consume_after_n_samples": self._consume_after_n_samples,
                "_rng": (self._rng.get_state().clone(), str(self._rng.device))
                if self._rng is not None
                else None,
                "_prefetch_state": prefetch_state,
            }

    @_maybe_delay_init
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        prefetch_state = state_dict.get("_prefetch_state")
        self._validate_prefetch_state(prefetch_state)
        with self._futures_lock:
            self._clear_prefetch_queue_locked()
            with self._replay_lock:
                self._storage.load_state_dict(state_dict["_storage"])
                self._sampler.load_state_dict(state_dict["_sampler"])
                self._writer.load_state_dict(state_dict["_writer"])
                self._transform.load_state_dict(state_dict["_transforms"])
                self._batch_size = state_dict["_batch_size"]
                self._consume_after_n_samples = state_dict.get(
                    "_consume_after_n_samples"
                )
                rng = state_dict.get("_rng")
                if rng is not None:
                    state, device = rng
                    rng = torch.Generator(device=device)
                    rng.set_state(state)
                    self.set_rng(generator=rng)
                self._restore_prefetch_queue_locked(prefetch_state)

    @_maybe_delay_init
    def dumps(self, path):
        """Saves the replay buffer on disk at the specified path.

        Args:
            path (Path or str): path where to save the replay buffer.

        Examples:
            >>> import tempfile
            >>> import tqdm
            >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
            >>> from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
            >>> import torch
            >>> from tensordict import TensorDict
            >>> # Build and populate the replay buffer
            >>> S = 1_000_000
            >>> sampler = PrioritizedSampler(S, 1.1, 1.0)
            >>> # sampler = RandomSampler()
            >>> storage = LazyMemmapStorage(S)
            >>> rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            >>>
            >>> for _ in tqdm.tqdm(range(100)):
            ...     td = TensorDict({"obs": torch.randn(100, 3, 4), "next": {"obs": torch.randn(100, 3, 4)}, "td_error": torch.rand(100)}, [100])
            ...     rb.extend(td)
            ...     sample = rb.sample(32)
            ...     rb.update_tensordict_priority(sample)
            >>> # save and load the buffer
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     rb.dumps(tmpdir)
            ...
            ...     sampler = PrioritizedSampler(S, 1.1, 1.0)
            ...     # sampler = RandomSampler()
            ...     storage = LazyMemmapStorage(S)
            ...     rb_load = TensorDictReplayBuffer(storage=storage, sampler=sampler)
            ...     rb_load.loads(tmpdir)
            ...     assert len(rb) == len(rb_load)

        """
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        # The queue cannot be consumed while this context is active, so the
        # pickle stream can read it directly without allocating another full
        # queue-sized copy first.
        with self._capture_prefetch_state(clone_queue=False) as prefetch_state:
            self._storage.dumps(path / "storage")
            self._sampler.dumps(path / "sampler")
            self._writer.dumps(path / "writer")
            if self._rng is not None:
                rng_state = TensorDict(
                    rng_state=self._rng.get_state().clone(),
                    device=self._rng.device,
                )
                rng_state.memmap(path / "rng_state")

            # fall back on state_dict for transforms
            transform_sd = self._transform.state_dict()
            if transform_sd:
                torch.save(transform_sd, path / "transform.t")
            with open(path / "buffer_metadata.json", "w") as file:
                json.dump(
                    {
                        "batch_size": self._batch_size,
                        "consume_after_n_samples": self._consume_after_n_samples,
                    },
                    file,
                )
            self._dump_prefetch_state(path, prefetch_state)

    @_maybe_delay_init
    def loads(self, path):
        """Loads a replay buffer state at the given path.

        The buffer should have matching components and be saved using :meth:`dumps`.

        Args:
            path (Path or str): path where the replay buffer was saved.

        See :meth:`dumps` for more info.

        """
        path = Path(path).absolute()
        prefetch_state = self._load_prefetch_state(path)
        self._validate_prefetch_state(prefetch_state)
        with self._futures_lock:
            self._clear_prefetch_queue_locked()
            with self._replay_lock:
                self._storage.loads(path / "storage")
                self._sampler.loads(path / "sampler")
                self._writer.loads(path / "writer")
                if (path / "rng_state").exists():
                    rng_state = TensorDict.load_memmap(path / "rng_state")
                    rng = torch.Generator(device=rng_state.device)
                    rng.set_state(rng_state["rng_state"])
                    self.set_rng(rng)
                # fall back on state_dict for transforms
                if (path / "transform.t").exists():
                    self._transform.load_state_dict(torch.load(path / "transform.t"))
                with open(path / "buffer_metadata.json") as file:
                    metadata = json.load(file)
                self._batch_size = metadata["batch_size"]
                self._consume_after_n_samples = metadata.get("consume_after_n_samples")
                # This method owns the freshly unpickled queue, so moving its
                # results into completed futures avoids a redundant deep copy.
                self._restore_prefetch_queue_locked(prefetch_state, clone_queue=False)

    @_maybe_delay_init
    def save(self, *args, **kwargs):
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    @_maybe_delay_init
    def dump(self, *args, **kwargs):
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    @_maybe_delay_init
    def load(self, *args, **kwargs):
        """Alias for :meth:`loads`."""
        return self.loads(*args, **kwargs)

    def _torchrl_checkpoint_detach_from_load_path(self):
        detach = getattr(self._storage.checkpointer, "_detach_from_load_path", None)
        if detach is not None:
            detach(self._storage)

    @_maybe_delay_init
    def register_save_hook(self, hook: Callable[[Any], Any]):
        """Registers a save hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.

        """
        self._storage.register_save_hook(hook)

    @_maybe_delay_init
    def register_load_hook(self, hook: Callable[[Any], Any]):
        """Registers a load hook for the storage.

        .. note:: Hooks are currently not serialized when saving a replay buffer: they must
            be manually re-initialized every time the buffer is created.

        """
        self._storage.register_load_hook(hook)

    @_maybe_delay_init
    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                make_none = False
                # Transforms usually expect a time batch dimension when called within a RB, so we unsqueeze the data temporarily
                is_tc = is_tensor_collection(data)
                cm = data.unsqueeze(-1) if is_tc else contextlib.nullcontext(data)
                new_data = None
                with cm as data_unsq:
                    data_unsq_r = self._transform.inv(data_unsq)
                    if is_tc and data_unsq_r is not None:
                        # this is a no-op whenever the result matches the input
                        new_data = data_unsq_r.squeeze(-1)
                    else:
                        make_none = data_unsq_r is None
                data = new_data if new_data is not None else data
                if make_none:
                    data = None
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        if rl_warnings() and is_tensor_collection(data) and data.ndim:
            warnings.warn(
                f"Using `add()` with a TensorDict that has batch_size={data.batch_size}. "
                f"Use `extend()` to add multiple elements, or `add()` with a single element (batch_size=torch.Size([])). "
                "You can silence this warning by setting the `RL_WARNINGS` environment variable to `'0'`."
            )

        return self._add(data)

    def _is_consuming(self) -> bool:
        return isinstance(self._sampler, ConsumingSampler)

    def _get_batch_size(self, data) -> int:
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            return len(data)
        if isinstance(data, list):
            return len(data)
        return len(tree_leaves(data)[0])

    def _cat_write_indices(self, first, second):
        if _is_int(first):
            first = torch.as_tensor([first], dtype=torch.long)
        if _is_int(second):
            second = torch.as_tensor([second], dtype=torch.long)
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return torch.cat([first.reshape(-1), second.to(first.device).reshape(-1)])
        raise RuntimeError(
            "Cannot concatenate write indices with different structures in "
            "a consuming replay buffer."
        )

    def _cursor_write_indices(
        self, data, batch_size: int, skip_index: torch.Tensor
    ) -> torch.Tensor:
        device = data.device if hasattr(data, "device") else skip_index.device
        max_size = self._storage._max_size_along_dim0(batched_data=data)
        skip = set(skip_index.cpu().tolist())
        cursor = self._writer._cursor
        write_indices = []
        scanned = 0
        while len(write_indices) < batch_size:
            if cursor not in skip or scanned >= max_size:
                write_indices.append(cursor)
            cursor = (cursor + 1) % max_size
            scanned += 1
        self._writer._cursor = cursor
        return torch.as_tensor(write_indices, dtype=torch.long, device=device)

    def _add(self, data):
        with self._replay_lock, self._write_lock:
            if self._is_consuming():
                consumed_index = self._sampler._pop_consumed_indices(self._storage, 1)
                if consumed_index.numel():
                    index = self._writer.write_at(int(consumed_index.item()), data)
                    self._sampler.add(index)
                    return index
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence, *, update_priority: bool = True) -> torch.Tensor:
        is_comp = is_compiling()
        nc = contextlib.nullcontext()
        with self._replay_lock if not is_comp else nc, self._write_lock if not is_comp else nc:
            if self.dim_extend > 0:
                data = self._transpose(data)
            if self._is_consuming():
                batch_size = self._get_batch_size(data)
                consumed_index = self._sampler._pop_consumed_indices(
                    self._storage, batch_size
                )
                consumed_batch_size = consumed_index.numel()
                if consumed_batch_size:
                    if consumed_batch_size < batch_size:
                        cursor_index = self._cursor_write_indices(
                            data, batch_size - consumed_batch_size, consumed_index
                        )
                        index = self._cat_write_indices(consumed_index, cursor_index)
                    else:
                        index = consumed_index
                    index = self._writer.write_at(index, data)
                    self._sampler.extend(index)
                    return index
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    @_maybe_delay_init
    def extend(
        self, data: Sequence, *, update_priority: bool | None = None
    ) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Keyword Args:
            update_priority (bool, optional): Whether to update the priority of the data. Defaults to True.
                Without effect in this class. See :meth:`~torchrl.data.TensorDictReplayBuffer.extend` for more details.

        Returns:
            Indices of the data added to the replay buffer.

        .. warning:: :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` can have an
          ambiguous signature when dealing with lists of values, which should be interpreted
          either as PyTree (in which case all elements in the list will be put in a slice
          in the stored PyTree in the storage) or a list of values to add one at a time.
          To solve this, TorchRL makes the clear-cut distinction between list and tuple:
          a tuple will be viewed as a PyTree, a list (at the root level) will be interpreted
          as a stack of values to add one at a time to the buffer.
          For :class:`~torchrl.data.replay_buffers.ListStorage` instances, only
          unbound elements can be provided (no PyTrees).

        """
        if update_priority is not None:
            raise NotImplementedError(
                "update_priority is not supported in this class. See :meth:`~torchrl.data.TensorDictReplayBuffer.extend` for more details."
            )
        if self._transform is not None and len(self._transform):
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)
        return self._extend(data, update_priority=update_priority)

    @_maybe_delay_init
    def update_priority(
        self,
        index: int | torch.Tensor | tuple[torch.Tensor],
        priority: int | torch.Tensor,
    ) -> None:
        if isinstance(index, tuple):
            index = torch.stack(index, -1)
        priority = torch.as_tensor(priority)
        if self.dim_extend > 0 and priority.ndim > 1:
            priority = self._transpose(priority).flatten()
            # priority = priority.flatten()
        with self._replay_lock, self._write_lock:
            self._sampler.update_priority(index, priority, storage=self.storage)

    @pin_memory_output
    def _sample(self, batch_size: int) -> tuple[Any, dict]:
        is_comp = is_compiling()
        nc = contextlib.nullcontext()
        with self._replay_lock if not is_comp else nc, self._write_lock if not is_comp else nc:
            index, info = self._sampler.sample(self._storage, batch_size)
            if self._sample_unit is not None:
                index, info = self._sample_unit.expand(index, info, self._storage)
            info["index"] = index
            if self._writer.tracks_generations:
                info["index_generation"] = self._writer.generations_of(index)
            data = self._storage.get(_storage_index(index, self._storage))
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = is_tensor_collection(data)
            with data.unlock_() if is_td else contextlib.nullcontext(), _set_dispatch_td_nn_modules(
                is_td
            ):
                data = self._transform(data)

        return data, info

    @_maybe_delay_init
    def empty(self, empty_write_count: bool = True):
        """Empties the replay buffer and reset cursor to 0.

        Args:
            empty_write_count (bool, optional): Whether to empty the write_count attribute. Defaults to `True`.
        """
        self._writer._empty(empty_write_count=empty_write_count)
        self._sampler._empty()
        self._storage._empty()

    @_maybe_delay_init
    def sample(self, batch_size: int | None = None, return_info: bool = False) -> Any:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
        if (
            batch_size is not None
            and self._batch_size is not None
            and batch_size != self._batch_size
        ):
            warnings.warn(
                f"Got conflicting batch_sizes in constructor ({self._batch_size}) "
                f"and `sample` ({batch_size}). Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments. "
                "The batch-size provided to the sample method "
                "will prevail."
            )
        elif batch_size is None and self._batch_size is not None:
            batch_size = self._batch_size
        elif batch_size is None:
            raise RuntimeError(
                "batch_size not specified. You can specify the batch_size when "
                "constructing the replay buffer, or pass it to the sample method. "
                "Refer to the ReplayBuffer documentation "
                "for a proper usage of the batch-size arguments."
            )
        if not self._prefetch:
            result = self._sample(batch_size)
        else:
            with self._futures_lock:
                if len(self._prefetch_queue):
                    result = self._prefetch_queue.popleft().result()
                else:
                    result = self._sample(batch_size)
                while (
                    len(self._prefetch_queue)
                    < min(self._sampler._remaining_batches, self._prefetch_cap)
                    and not self._sampler.ran_out
                ):
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)

        if return_info:
            out, info = result
            if getattr(self.storage, "device", None) is not None:
                device = self.storage.device
                info = tree_map(lambda x: x.to(device) if hasattr(x, "to") else x, info)
            return out, info
        return result[0]

    @_maybe_delay_init
    def query(
        self,
        predicate: Callable[[Trajectory], bool] | None = None,
        *,
        trajectory_key: NestedKey | None = None,
    ) -> list[Trajectory]:
        """Filters the stored trajectories with a query predicate.

        Splits the buffer content into trajectories (see
        :func:`~torchrl.data.replay_buffers.query.iter_trajectories`) and
        returns those matching the predicate as
        :class:`~torchrl.data.replay_buffers.query.Trajectory` views.

        Args:
            predicate (Callable[[Trajectory], bool], optional): a
                :class:`~torchrl.data.replay_buffers.query.TrajectoryPredicate`
                built from :data:`~torchrl.data.replay_buffers.query.traj`, or
                any callable mapping a trajectory to a boolean. Defaults to
                None (return all trajectories).

        Keyword Args:
            trajectory_key (NestedKey, optional): entry holding
                per-transition trajectory ids. Defaults to None
                (auto-detection from ``("collector", "traj_ids")``,
                ``"traj_ids"``, ``"episode"`` or the done/terminated/truncated
                flags).

        Returns:
            A list of matching trajectory views, ordered chronologically
            (oldest trajectory first; for multi-dimensional storages, grouped
            by batch coordinate).

        The trajectory boundaries are computed from the stored (untransformed)
        data with the same machinery
        :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` uses, so
        samplers and queries always agree on where trajectories start and
        stop. This includes storages with ``ndim > 1`` (e.g.
        ``LazyTensorStorage(..., ndim=2)`` holding ``[B, T]`` batches), whose
        trajectories are recovered per batch coordinate.

        Predicates built from :data:`~torchrl.data.replay_buffers.query.traj`
        report the keys they read via
        :meth:`~torchrl.data.replay_buffers.query.TrajectoryPredicate.required_keys`;
        evaluation then only fetches those entries from the storage and only
        runs the transforms that can affect them. Matching trajectories are
        extracted in full with the complete transform chain applied, so
        predicates and results see the same values a sampler would produce.
        Opaque callables are evaluated against the fully transformed content.

        .. note::
            Once the buffer has wrapped around (it is at capacity and older
            entries have been overwritten), the oldest trajectory may have
            lost its first transitions to overwriting and will appear
            truncated at the front. A trajectory written across the wrap
            point is followed through it and returned whole, in time order.

        Examples:
            >>> from torchrl.data import traj
            >>> good_trajs = rb.query((traj.reward.sum() > 100) & (traj.length >= 50))
            >>> observations = good_trajs[0].observation
        """
        storage = self._storage
        if not len(storage):
            return []
        with self._replay_lock:
            source = storage[:]
        if isinstance(source, (list, tuple)):
            if not source:
                return []
            if not all(is_tensor_collection(item) for item in source):
                raise TypeError(
                    "ReplayBuffer.query requires a tensordict-backed storage, "
                    f"got items of type {type(source[0])}."
                )
            if any(item.batch_dims for item in source):
                raise TypeError(
                    "ReplayBuffer.query on a list-based storage expects "
                    "single-transition (scalar) items."
                )
            source = LazyStackedTensorDict.lazy_stack(list(source))
        elif not is_tensor_collection(source):
            raise TypeError(
                "ReplayBuffer.query requires a tensordict-backed storage, "
                f"got content of type {type(source)}."
            )
        if self._transform is not None and len(self._transform):
            transforms = list(self._transform.transforms)
        else:
            transforms = []
        return _query_source(
            source,
            transforms=transforms,
            predicate=predicate,
            trajectory_key=trajectory_key,
            at_capacity=bool(storage._is_full),
            cursor=getattr(storage, "_last_cursor_index", None),
        )

    @_maybe_delay_init
    def mark_update(self, index: int | torch.Tensor) -> None:
        self._sampler.mark_update(index, storage=self._storage)

    @_maybe_delay_init
    def append_transform(
        self, transform: Transform, *, invert: bool = False  # noqa-F821
    ) -> ReplayBuffer:  # noqa: D417
        """Appends transform at the end.

        Transforms are applied in order when `sample` is called.

        Args:
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        Example:
            >>> rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
            >>> data = TensorDict({"a": torch.zeros(10)}, [10])
            >>> def t(data):
            ...     data += 1
            ...     return data
            >>> rb.append_transform(t, invert=True)
            >>> rb.extend(data)
            >>> assert (data == 1).all()

        """
        from torchrl.envs.transforms.transforms import _CallableTransform, Transform

        if not isinstance(transform, Transform) and callable(transform):
            transform = _CallableTransform(transform)
        if invert:
            transform = _InvertTransform(transform)
        transform.eval()
        if self.shared:
            self._share_transform_state(transform)
        self._transform.append(transform)
        return self

    @_maybe_delay_init
    def insert_transform(
        self,
        index: int,
        transform: Transform,  # noqa-F821
        *,
        invert: bool = False,
    ) -> ReplayBuffer:  # noqa: D417
        """Inserts transform.

        Transforms are executed in order when `sample` is called.

        Args:
            index (int): Position to insert the transform.
            transform (Transform): The transform to be appended

        Keyword Args:
            invert (bool, optional): if ``True``, the transform will be inverted (forward calls will be called
                during writing and inverse calls during reading). Defaults to ``False``.

        """
        transform.eval()
        if invert:
            transform = _InvertTransform(transform)
        if self.shared:
            self._share_transform_state(transform)
        self._transform.insert(index, transform)
        return self

    _is_tensordict = False
    _iterator = None

    @_maybe_delay_init
    def next(self):
        """Returns the next item in the replay buffer.

        This method is used to iterate over the replay buffer in contexts where __iter__ is not available,
        such as :class:`~torchrl.data.replay_buffers.RayReplayBuffer`.
        """
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            if out is not None and (out.device != "cpu"):
                out = out.copy().clear_device_()
            return out
        except StopIteration:
            self._iterator = None
            return None

    @_maybe_delay_init
    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out or (
            self._prefetch and len(self._prefetch_queue)
        ):
            yield self.sample()

    @_maybe_delay_init
    def __getstate__(self) -> dict[str, Any]:
        with self._capture_prefetch_state() as prefetch_state:
            state = self.__dict__.copy()
            if getattr(self, "_rng", None) is not None:
                rng_state = TensorDict(
                    rng_state=self._rng.get_state().clone(),
                    device=self._rng.device,
                )
                state["_rng"] = rng_state
            _replay_lock = state.pop("_replay_lock", None)
            _futures_lock = state.pop("_futures_lock", None)
            if _replay_lock is not None:
                state["_replay_lock_placeholder"] = None
            if _futures_lock is not None:
                state["_futures_lock_placeholder"] = None
            _prefetch_queue = state.pop("_prefetch_queue", None)
            _prefetch_executor = state.pop("_prefetch_executor", None)
            if _prefetch_queue is not None:
                state["_prefetch_queue_placeholder"] = None
            if _prefetch_executor is not None:
                state["_prefetch_executor_placeholder"] = None
            state["_prefetch_state"] = prefetch_state
            return state

    def __setstate__(self, state: dict[str, Any]):
        prefetch_state = state.pop("_prefetch_state", None)
        rngstate = None
        if "_rng" in state:
            rngstate = state["_rng"]
            if rngstate is not None:
                rng = torch.Generator(device=rngstate.device)
                rng.set_state(rngstate["rng_state"])

        if "_replay_lock_placeholder" in state:
            state.pop("_replay_lock_placeholder")
            _replay_lock = threading.RLock()
            state["_replay_lock"] = _replay_lock
        if "_futures_lock_placeholder" in state:
            state.pop("_futures_lock_placeholder")
            _futures_lock = threading.RLock()
            state["_futures_lock"] = _futures_lock
        # Recreate prefetch objects after unpickling if they were present
        if "_prefetch_queue_placeholder" in state:
            state.pop("_prefetch_queue_placeholder")
            state["_prefetch_queue"] = collections.deque()
        if "_prefetch_executor_placeholder" in state:
            state.pop("_prefetch_executor_placeholder")
            state["_prefetch_executor"] = ThreadPoolExecutor(
                max_workers=state["_prefetch_cap"]
            )
        self.__dict__.update(state)
        if rngstate is not None:
            self.set_rng(rng)
        with self._futures_lock:
            # __setstate__ owns prefetch_state after popping it from the pickle
            # payload, so queue entries can be transferred without cloning.
            self._restore_prefetch_queue_locked(prefetch_state, clone_queue=False)

    @property
    @_maybe_delay_init
    def sampler(self) -> Sampler:
        """The sampler of the replay buffer.

        The sampler must be an instance of :class:`~torchrl.data.replay_buffers.Sampler`.

        """
        return self._sampler

    @property
    @_maybe_delay_init
    def writer(self) -> Writer:
        """The writer of the replay buffer.

        The writer must be an instance of :class:`~torchrl.data.replay_buffers.Writer`.

        """
        return self._writer

    @property
    @_maybe_delay_init
    def storage(self) -> Storage:
        """The storage of the replay buffer.

        The storage must be an instance of :class:`~torchrl.data.replay_buffers.Storage`.

        """
        return self._storage

    @property
    @_maybe_delay_init
    def transform(self) -> Transform:
        """The transform of the replay buffer.

        The transform must be an instance of :class:`~torchrl.envs.transforms.Transform`.
        """
        return self._transform
