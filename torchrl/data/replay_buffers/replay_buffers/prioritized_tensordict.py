# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
from collections.abc import Callable
from multiprocessing.context import get_spawning_popen
from typing import Any

import torch

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

from typing import Literal, TYPE_CHECKING, TypeVar

from tensordict import is_tensor_collection, NestedKey, TensorDictBase
from tensordict.nn.utils import _set_dispatch_td_nn_modules

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.sample_units import SampleUnit
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import _reduce, INT_CLASSES, pin_memory_output
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import Transform

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .base import _maybe_delay_init, _storage_index, ReplayBuffer
from .tensordict import TensorDictReplayBuffer


class TensorDictPrioritizedReplayBuffer(TensorDictReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.PrioritizedReplayBuffer` class.

    This class returns tensordicts with a new key ``"index"`` that represents
    the index of each element in the replay buffer. It also provides the
    :meth:`~.update_tensordict_priority` method that only requires for the
    tensordict to be passed to it with its new priority value.

    Keyword Args:
        alpha (:obj:`float`): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (:obj:`float`): importance sampling negative exponent.
        eps (:obj:`float`): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, Callable[[], Storage], optional): the storage to be used.
            If a callable is passed, it is used as constructor for the storage.
            If none is provided a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
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
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

        priority_key (NestedKey, optional): the key at which priority is assumed to
            be stored within TensorDicts added to this ReplayBuffer.
            This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.
            Defaults to ``"td_error"``.
        sampler_device (torch.device or str, optional): device where the
            priority sampler trees will be stored. Defaults to ``None``, in
            which case CUDA storage selects CUDA sampling and CPU storage
            selects CPU sampling.
        sync (bool, optional): whether the priority sampler is synchronized with
            writes. If ``True``, this class uses the standard
            :class:`~torchrl.data.PrioritizedSampler` write path. If ``False``,
            writer processes use a shareable :class:`~torchrl.data.RandomSampler`
            and the learner owns a local priority sampler that catches up from
            ``write_count`` before sampling. Defaults to ``True``.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectories). Can be one of "max", "min",
            "median" or "mean".
        dim_extend (int, optional): indicates the dim to consider for
            extension when calling :meth:`~.extend`. Defaults to ``storage.ndim-1``.
            When using ``dim_extend > 0``, we recommend using the ``ndim``
            argument in the storage instantiation if that argument is
            available, to let storages know that the data is
            multi-dimensional and keep consistent notions of storage-capacity
            and batch-size during sampling.

            .. note:: This argument has no effect on :meth:`~.add` and
                therefore should be used with caution when both :meth:`~.add`
                and :meth:`~.extend` are used in a codebase. For example:

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
        transport (str, optional): physical transport used by a remote replay
            owner. ``"auto"`` selects the backend default. Defaults to
            ``"auto"``.
        transport_options (dict, optional): options for the selected transport.
            For ``transport="distributed"``, ``backend`` selects ``"gloo"``
            or ``"nccl"``. TensorDict layouts are bound lazily on first use.

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
        >>> from tensordict import TensorDict
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, storage=LazyTensorStorage(10), batch_size=5)
        >>> data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 3, 1)}, [10])
        >>> rb.extend(data)
        >>> print("len of rb", len(rb))
        len of rb 10
        >>> sample = rb.sample(5)
        >>> print(sample)
        TensorDict(
            fields={
                priority_weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print("index", sample["index"])
        index tensor([9, 5, 2, 2, 7])
        >>> # give a high priority to these samples...
        >>> sample.set("td_error", 100*torch.ones(sample.shape))
        >>> # and update priority
        >>> rb.update_tensordict_priority(sample)
        >>> # the new sample should have a high overlap with the previous one
        >>> sample = rb.sample(5)
        >>> print(sample)
        TensorDict(
            fields={
                priority_weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print("index", sample["index"])
        index tensor([2, 5, 5, 9, 7])

    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        priority_key: NestedKey = "td_error",
        eps: float = 1e-8,
        storage: Storage | None = None,
        sample_unit: SampleUnit | None = None,
        sampler_device: DEVICE_TYPING | None = None,
        sync: bool = True,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: Transform | None = None,  # noqa-F821
        reduction: str = "max",
        batch_size: int | None = None,
        dim_extend: int | None = None,
        generator: torch.Generator | None = None,
        shared: bool = False,
        compilable: bool = False,
        transport: Literal["auto", "direct", "ray", "distributed"] = "auto",
        transport_options: dict[str, Any] | None = None,
    ) -> None:
        storage = self._maybe_make_storage(storage, compilable=compilable)
        self._sync = sync
        self._prioritized_sampler = None
        self._prioritized_sampler_write_count = 0
        prioritized_sampler = PrioritizedSampler(
            storage.max_size,
            alpha,
            beta,
            eps,
            reduction=reduction,
            device=sampler_device,
        )
        if sync:
            sampler = prioritized_sampler
        else:
            if storage.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__} only supports 1-D storages when sync=False, "
                    f"got storage.ndim={storage.ndim}."
                )
            self._prioritized_sampler = prioritized_sampler
            sampler = RandomSampler()
        super().__init__(
            priority_key=priority_key,
            storage=storage,
            sampler=sampler,
            sample_unit=sample_unit,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
            dim_extend=dim_extend,
            generator=generator,
            shared=shared,
            compilable=compilable,
            transport=transport,
            transport_options=transport_options,
        )

    @property
    def prioritized_sampler(self) -> PrioritizedSampler:
        """The sampler that owns the priority tree."""
        if self._sync:
            return self._sampler
        sampler = self._prioritized_sampler
        if sampler is None:
            raise RuntimeError(
                f"{type(self).__name__} cannot sample with prioritized replay in a worker process."
            )
        return sampler

    def _catch_up_prioritized_sampler(self) -> None:
        sampler = self.prioritized_sampler
        write_count = int(self.write_count)
        if write_count < self._prioritized_sampler_write_count:
            sampler._empty()
            self._prioritized_sampler_write_count = 0
        delta = write_count - self._prioritized_sampler_write_count
        if delta <= 0:
            return
        max_size = self._storage.max_size
        if delta >= max_size:
            index = torch.arange(max_size, dtype=torch.long)
        else:
            index = torch.arange(
                self._prioritized_sampler_write_count,
                write_count,
                dtype=torch.long,
            ).remainder_(max_size)
        sampler.mark_update(index, storage=self._storage)
        self._prioritized_sampler_write_count = write_count

    @_maybe_delay_init
    def add(self, data: TensorDictBase) -> int:
        if self._sync:
            return super().add(data)
        if self._transform is not None:
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = ReplayBuffer._add(self, data)
        if index is not None and is_tensor_collection(data):
            self._set_index_in_td(data, index)
        return index

    @_maybe_delay_init
    def extend(
        self, tensordicts: TensorDictBase, *, update_priority: bool | None = None
    ) -> torch.Tensor:
        if self._sync:
            return super().extend(tensordicts, update_priority=update_priority)
        if update_priority:
            raise RuntimeError(
                f"{type(self).__name__}.extend does not support updating priorities "
                "from writer processes. Call update_tensordict_priority from the "
                "learner process instead."
            )
        if not isinstance(tensordicts, TensorDictBase):
            raise ValueError(
                f"{self.__class__.__name__} only accepts TensorDictBase subclasses. "
                "tensorclasses and other types are not compatible with that class. "
                "Please use a regular `ReplayBuffer` instead."
            )
        if self._transform is not None:
            tensordicts = self._transform.inv(tensordicts)
        if tensordicts is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = ReplayBuffer._extend(self, tensordicts)
        self._set_index_in_td(tensordicts, index)
        return index

    def _get_priority_item(self, tensordict: TensorDictBase) -> float:
        if self._sync:
            return super()._get_priority_item(tensordict)
        sampler = self.prioritized_sampler
        priority = tensordict.get(self.priority_key, None)
        if priority is None:
            return sampler.default_priority
        try:
            if priority.numel() > 1:
                priority = _reduce(priority, sampler.reduction)
            else:
                priority = priority.item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size"
                f" {tensordict.get(self.priority_key).shape} but expected "
                f"scalar value"
            )
        return priority

    def _get_priority_vector(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self._sync:
            return super()._get_priority_vector(tensordict)
        sampler = self.prioritized_sampler
        priority = tensordict.get(self.priority_key, None)
        if priority is None:
            return torch.tensor(
                sampler.default_priority,
                dtype=torch.float,
                device=tensordict.device,
            ).expand(tensordict.shape[0])

        priority = priority.reshape(priority.shape[0], -1)
        return _reduce(priority, sampler.reduction, dim=1)

    @pin_memory_output
    def _sample(self, batch_size: int) -> tuple[Any, dict]:
        if self._sync:
            return super()._sample(batch_size)
        self._catch_up_prioritized_sampler()
        is_comp = is_compiling()
        nc = contextlib.nullcontext()
        with (
            self._replay_lock if not is_comp else nc,
            self._write_lock if not is_comp else nc,
        ):
            index, info = self.prioritized_sampler.sample(self._storage, batch_size)
            if self._sample_unit is not None:
                index, info = self._sample_unit.expand(index, info, self._storage)
            info["index"] = index
            if self._writer.tracks_generations:
                info["index_generation"] = self._writer.generations_of(index)
            data = self._storage.get(_storage_index(index, self._storage))
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            with data.unlock_(), _set_dispatch_td_nn_modules(True):
                data = self._transform(data)
        return data, info

    @_maybe_delay_init
    def update_priority(
        self,
        index: int | torch.Tensor | tuple[torch.Tensor],
        priority: int | torch.Tensor,
    ) -> None:
        if self._sync:
            return super().update_priority(index, priority)
        if isinstance(index, tuple):
            index = torch.stack(index, -1)
        priority = torch.as_tensor(priority)
        if self.dim_extend > 0 and priority.ndim > 1:
            priority = self._transpose(priority).flatten()
        with self._replay_lock, self._write_lock:
            self.prioritized_sampler.update_priority(
                index, priority, storage=self.storage
            )

    @_maybe_delay_init
    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        if self._sync:
            return super().update_tensordict_priority(data)
        if data.ndim:
            priority = self._get_priority_vector(data)
            anchored = self._anchor_reduced_priority(data, priority)
            if anchored is not None:
                return self.update_priority(*anchored)
        else:
            priority = torch.as_tensor(self._get_priority_item(data))
        index = data.get("index")
        while index.shape != priority.shape:
            index = index[..., 0]
        return self.update_priority(index, priority)

    @_maybe_delay_init
    def empty(self, empty_write_count: bool = True):
        super().empty(empty_write_count=empty_write_count)
        if not self._sync:
            self.prioritized_sampler._empty()
            self._prioritized_sampler_write_count = 0

    @_maybe_delay_init
    def set_rng(self, generator) -> None:
        super().set_rng(generator)
        if not self._sync and getattr(self, "_prioritized_sampler", None) is not None:
            self._prioritized_sampler._rng = generator

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        if not self._sync and get_spawning_popen() is not None:
            state["_prioritized_sampler"] = None
        return state
