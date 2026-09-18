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

from tensordict import is_tensor_collection
from tensordict.nn.utils import _set_dispatch_td_nn_modules

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.sample_units import SampleUnit
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    Sampler,
)
from torchrl.data.replay_buffers.storages import ListStorage, Storage
from torchrl.data.replay_buffers.utils import INT_CLASSES, pin_memory_output
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.transforms.transforms import Transform

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .base import _maybe_delay_init, _storage_index, ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer.

    All arguments are keyword-only arguments.

    Presented in "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
    Prioritized experience replay." (https://arxiv.org/abs/1511.05952)

    Args:
        alpha (:obj:`float`): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (:obj:`float`): importance sampling negative exponent.
        eps (:obj:`float`): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided,
            a default :class:`~torchrl.data.replay_buffers.PrioritizedSampler` with
            ``alpha``, ``beta``, and ``eps`` will be created.
        sampler_device (torch.device or str, optional): device where the
            priority sampler trees will be stored. Defaults to ``None``, in
            which case CUDA storage selects CUDA sampling and CPU storage
            selects CPU sampling. Cannot be used together with ``sampler``.
        sync (bool, optional): whether the priority sampler is synchronized with
            writes. If ``True``, this class uses the standard
            :class:`~torchrl.data.PrioritizedSampler` write path. If ``False``,
            writer processes use a shareable :class:`~torchrl.data.RandomSampler`
            and the learner owns a local priority sampler that catches up from
            ``write_count`` before sampling. Defaults to ``True``.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset. The default value will be decided
            based on the storage type.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading. Defaults to None (no prefetching).
        transform (Transform, optional): Transform to be executed when
            sample() is called.
            To chain transforms use the :class:`~torchrl.envs.Compose` class.
            Transforms should be used with :class:`tensordict.TensorDict`
            content. If used with other structures, the transforms should be
            encoded with a ``"data"`` leading key that will be used to
            construct a tensordict from the non-tensordict content.
        batch_size (int, optional): the batch size to be used when sample() is
            called.

            .. note:: The batch-size can be specified at construction time via the
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

    .. note::
        Generic prioritized replay buffers (ie. non-tensordict backed) require
        calling :meth:`~.sample` with the ``return_info`` argument set to
        ``True`` to have access to the indices, and hence update the priority.
        Using :class:`tensordict.TensorDict` and the related
        :class:`~torchrl.data.TensorDictPrioritizedReplayBuffer` simplifies this
        process.

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import ListStorage, PrioritizedReplayBuffer
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(10))
        >>> data = range(10)
        >>> rb.extend(data)
        >>> sample = rb.sample(3)
        >>> print(sample)
        tensor([1, 0, 1])
        >>> # get the info to find what the indices are
        >>> sample, info = rb.sample(5, return_info=True)
        >>> print(sample, info)
        tensor([2, 7, 4, 3, 5]) {'priority_weight': array([1., 1., 1., 1., 1.], dtype=float32), 'index': array([2, 7, 4, 3, 5])}
        >>> # update priority
        >>> priority = torch.ones(5) * 5
        >>> rb.update_priority(info["index"], priority)
        >>> # and now a new sample, the weights should be updated
        >>> sample, info = rb.sample(5, return_info=True)
        >>> print(sample, info)
        tensor([2, 5, 2, 2, 5]) {'priority_weight': array([0.36278465, 0.36278465, 0.36278465, 0.36278465, 0.36278465],
              dtype=float32), 'index': array([2, 5, 2, 2, 5])}

    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        storage: Storage | None = None,
        sampler: Sampler | None = None,
        sample_unit: SampleUnit | None = None,
        sampler_device: DEVICE_TYPING | None = None,
        sync: bool = True,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: Transform | None = None,  # noqa-F821
        batch_size: int | None = None,
        dim_extend: int | None = None,
        delayed_init: bool = False,
        transport: Literal["auto", "direct", "ray", "distributed"] = "auto",
        transport_options: dict[str, Any] | None = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        self._sync = sync
        self._prioritized_sampler = None
        self._prioritized_sampler_write_count = 0
        if sampler is None:
            prioritized_sampler = PrioritizedSampler(
                storage.max_size, alpha, beta, eps, dtype, device=sampler_device
            )
        elif sampler_device is not None:
            raise TypeError("sampler_device cannot be passed when sampler is provided.")
        else:
            prioritized_sampler = sampler
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
            storage=storage,
            sampler=sampler,
            sample_unit=sample_unit,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
            dim_extend=dim_extend,
            delayed_init=delayed_init,
            transport=transport,
            transport_options=transport_options,
        )

    @property
    def prioritized_sampler(self) -> Sampler:
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
            is_td = is_tensor_collection(data)
            with data.unlock_() if is_td else contextlib.nullcontext(), _set_dispatch_td_nn_modules(
                is_td
            ):
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
