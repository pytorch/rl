# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import math
import warnings
from typing import Any

import torch

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

from functools import partial
from typing import TYPE_CHECKING, TypeVar

from tensordict import is_tensor_collection, is_tensorclass, TensorDictBase
from tensordict.nn.utils import _set_dispatch_td_nn_modules
from tensordict.utils import expand_as_right

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.sample_units import Sequence as SequenceSampleUnit
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.utils import (
    _is_int,
    _reduce,
    _to_torch,
    INT_CLASSES,
    pin_memory_output,
)
from torchrl.data.replay_buffers.writers import TensorDictRoundRobinWriter

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .base import _maybe_delay_init, _storage_index, ReplayBuffer


class TensorDictReplayBuffer(ReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.ReplayBuffer` class.

    See also :class:`~torchrl.trainers.algorithms.configs.TensorDictReplayBufferConfig`.

    Keyword Args:
        storage (Storage, Callable[[], Storage], optional): the storage to be used.
            If a callable is passed, it is used as constructor for the storage.
            If none is provided a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, Callable[[], Sampler], optional): the sampler to be used.
            If a callable is passed, it is used as constructor for the sampler.
            If none is provided, a default :class:`~torchrl.data.replay_buffers.RandomSampler`
            will be used.
        writer (Writer, Callable[[], Writer], optional): the writer to be used.
            If a callable is passed, it is used as constructor for the writer.
            If none is provided a default :class:`~torchrl.data.replay_buffers.TensorDictRoundRobinWriter`
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
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

        priority_key (str, optional): the key at which priority is assumed to
            be stored within TensorDicts added to this ReplayBuffer.
            This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.
            Defaults to ``"td_error"``.
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

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from tensordict import TensorDict
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=5)
        >>> data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 1, 1)}, [10])
        >>> rb.extend(data)
        >>> sample = rb.sample(3)
        >>> # samples keep track of the index
        >>> print(sample)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)
        >>> # we can iterate over the buffer
        >>> for i, data in enumerate(rb):
        ...     print(i, data)
        ...     if i == 2:
        ...         break
        0 TensorDict(
            fields={
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        1 TensorDict(
            fields={
                a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)

    """

    _is_tensordict = True

    def __init__(self, *, priority_key: str = "td_error", **kwargs) -> None:
        writer = kwargs.get("writer", None)
        if writer is None:
            kwargs["writer"] = partial(
                TensorDictRoundRobinWriter, compilable=kwargs.get("compilable")
            )
        super().__init__(**kwargs)
        self.priority_key = priority_key

    def _get_priority_item(self, tensordict: TensorDictBase) -> float:
        priority = tensordict.get(self.priority_key, None)
        if self._storage.ndim > 1:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)
        if priority is None:
            return self._sampler.default_priority
        try:
            if priority.numel() > 1:
                priority = _reduce(priority, self._sampler.reduction)
            else:
                priority = priority.item()
        except ValueError:
            raise ValueError(
                f"Found a priority key of size"
                f" {tensordict.get(self.priority_key).shape} but expected "
                f"scalar value"
            )

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    def _get_priority_vector(self, tensordict: TensorDictBase) -> torch.Tensor:
        priority = tensordict.get(self.priority_key, None)
        if priority is None:
            return torch.tensor(
                self._sampler.default_priority,
                dtype=torch.float,
                device=tensordict.device,
            ).expand(tensordict.shape[0])
        if self._storage.ndim > 1 and priority.ndim >= self._storage.ndim:
            # We have to flatten the priority otherwise we'll be aggregating
            # the priority across batches
            priority = priority.flatten(0, self._storage.ndim - 1)

        priority = priority.reshape(priority.shape[0], -1)
        priority = _reduce(priority, self._sampler.reduction, dim=1)

        if self._storage.ndim > 1:
            priority = priority.unflatten(0, tensordict.shape[: self._storage.ndim])

        return priority

    @_maybe_delay_init
    def add(self, data: TensorDictBase) -> int:
        if self._transform is not None:
            with _set_dispatch_td_nn_modules(is_tensor_collection(data)):
                data = self._transform.inv(data)
        if data is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._add(data)
        if index is not None:
            if is_tensor_collection(data):
                self._set_index_in_td(data, index)

            self.update_tensordict_priority(data)
        return index

    @_maybe_delay_init
    def extend(
        self, tensordicts: TensorDictBase, *, update_priority: bool | None = None
    ) -> torch.Tensor:
        """Extends the replay buffer with a batch of data.

        Args:
            tensordicts (TensorDictBase): The data to extend the replay buffer with.

        Keyword Args:
            update_priority (bool, optional): Whether to update the priority of the data. Defaults to True.

        Returns:
            The indices of the data that were added to the replay buffer.
        """
        if not isinstance(tensordicts, TensorDictBase):
            raise ValueError(
                f"{self.__class__.__name__} only accepts TensorDictBase subclasses. tensorclasses "
                f"and other types are not compatible with that class. "
                "Please use a regular `ReplayBuffer` instead."
            )
        if self._transform is not None:
            tensordicts = self._transform.inv(tensordicts)
        if tensordicts is None:
            return torch.zeros((0, self._storage.ndim), dtype=torch.long)

        index = super()._extend(tensordicts)

        # TODO: to be usable directly, the indices should be flipped but the issue
        #  is that just doing this results in indices that are not sorted like the original data
        #  so the actually indices will have to be used on the _storage directly (not on the buffer)
        self._set_index_in_td(tensordicts, index)
        if update_priority is None:
            update_priority = True
        if update_priority:
            try:
                vector = tensordicts.get(self.priority_key)
                if vector is not None:
                    self.update_priority(index, vector)
            except Exception as e:
                raise RuntimeError(
                    "Failed to update priority of extended data. You can try to set update_priority=False in the extend method and update the priority manually."
                ) from e
        return index

    def _set_index_in_td(self, tensordict, index):
        if index is None:
            return
        if _is_int(index):
            index = torch.as_tensor(index, device=tensordict.device)
        elif index.ndim == 2 and index.shape[:1] != tensordict.shape[:1]:
            for dim in range(tensordict.ndim, 1, -1):
                if index.shape[:1].numel() == tensordict.shape[:dim].numel():
                    # if index has 2 dims and is in a non-zero format
                    index = index.unflatten(0, tensordict.shape[:dim])
                    break
            else:
                raise RuntimeError(
                    f"could not find how to reshape index with shape {index.shape} to fit in tensordict with shape {tensordict.shape}"
                )
            tensordict.set("index", index)
            return
        tensordict.set("index", expand_as_right(index, tensordict))

    def _anchor_reduced_priority(
        self, data: TensorDictBase, priority: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Reduces per-record priorities of a sample-unit expansion to anchors.

        When a sample unit expanded the sampled anchors into windows of
        records (e.g. :class:`~torchrl.data.replay_buffers.Sequence`), the
        sample carries a per-record ``"anchor_index"`` entry while
        ``"index"`` holds the expanded per-record storage indices. Priorities
        are per-anchor quantities: this reduces the per-record priorities
        with a max over each anchor's valid records (``"validity_mask"``)
        and returns the unique anchors with their reduced priorities, making
        the update well-defined when the same anchor appears in several
        windows. Returns ``None`` when no expansion metadata is present.
        """
        if not isinstance(self._sample_unit, SequenceSampleUnit):
            return None
        anchor = data.get("anchor_index", None)
        if anchor is None:
            return None
        validity = data.get("validity_mask", None)
        if self._storage.ndim > 1:
            if anchor.ndim < 2 or anchor.shape[-1] != self._storage.ndim:
                return None
            anchor = anchor.reshape(-1, self._storage.ndim)
            priority = priority.reshape(-1)
            if anchor.shape[0] != priority.shape[0]:
                return None
            if validity is not None:
                validity = validity.reshape(-1)
                anchor = anchor[validity]
                priority = priority[validity]
            shape = tuple(self._storage.shape)
            stride = anchor.new_tensor(
                [math.prod(shape[dim + 1 :]) for dim in range(len(shape))]
            )
            flat_anchor = (anchor * stride).sum(-1)
            unique, inverse = torch.unique(flat_anchor, return_inverse=True)
            reduced = torch.zeros_like(unique, dtype=priority.dtype)
            reduced.scatter_reduce_(
                0, inverse, priority, reduce="amax", include_self=False
            )
            coordinate = (unique.unsqueeze(-1) // stride) % anchor.new_tensor(shape)
            return coordinate, reduced
        while anchor.shape != priority.shape and anchor.ndim > priority.ndim:
            anchor = anchor[..., 0]
            if validity is not None:
                validity = validity[..., 0]
        if anchor.shape != priority.shape:
            return None
        anchor = anchor.reshape(-1)
        priority = priority.reshape(-1)
        if validity is not None:
            validity = validity.reshape(-1)
            # every anchor's own record is always valid, so masking cannot
            # drop an anchor from the update
            anchor = anchor[validity]
            priority = priority[validity]
        unique, inverse = torch.unique(anchor, return_inverse=True)
        reduced = torch.zeros_like(unique, dtype=priority.dtype)
        reduced.scatter_reduce_(0, inverse, priority, reduce="amax", include_self=False)
        return unique, reduced

    @_maybe_delay_init
    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        if not isinstance(self._sampler, PrioritizedSampler):
            return
        if data.ndim:
            priority = self._get_priority_vector(data)
            anchored = self._anchor_reduced_priority(data, priority)
            if anchored is not None:
                return self.update_priority(*anchored)
        else:
            priority = torch.as_tensor(self._get_priority_item(data))
        index = data.get("index")
        if self._storage.ndim > 1 and index.ndim == 2:
            index = index.unbind(-1)
        else:
            while index.shape != priority.shape:
                # reduce index
                index = index[..., 0]
        return self.update_priority(index, priority)

    def sample(
        self,
        batch_size: int | None = None,
        return_info: bool = False,
        include_info: bool | None = None,
    ) -> TensorDictBase:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.
            include_info (bool, optional): deprecated alias for ``return_info``.

        Returns:
            A tensordict containing a batch of data selected in the replay buffer.
            A tuple containing this tensordict and info if return_info flag is set to True.
        """
        if include_info is not None:
            warnings.warn(
                "include_info is going to be deprecated soon."
                "The default behavior has changed to `include_info=True` "
                "to avoid bugs linked to wrongly preassigned values in the "
                "output tensordict."
            )

        data, info = super().sample(batch_size, return_info=True)
        is_tc = is_tensor_collection(data)
        if is_tc and not is_tensorclass(data) and include_info in (True, None):
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            for key, val in info.items():
                if key == "index" and isinstance(val, tuple):
                    val = torch.stack(val, -1)
                try:
                    val = _to_torch(val, data.device)
                    if val.ndim < data.ndim:
                        val = expand_as_right(val, data)
                    data.set(key, val)
                except RuntimeError:
                    raise RuntimeError(
                        "Failed to set the metadata (e.g., indices or weights) in the sampled tensordict within TensorDictReplayBuffer.sample. "
                        "This is probably caused by a shape mismatch (one of the transforms has probably modified "
                        "the shape of the output tensordict). "
                        "You can always recover these items from the `sample` method from a regular ReplayBuffer "
                        "instance with the 'return_info' flag set to True."
                    )
            if is_locked:
                data.lock_()
        elif not is_tc and include_info in (True, None):
            raise RuntimeError("Cannot include info in non-tensordict data")
        if return_info:
            return data, info
        return data

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
            with data.unlock_(), _set_dispatch_td_nn_modules(True):
                data = self._transform(data)
        return data, info
