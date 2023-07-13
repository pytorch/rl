# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from tensordict import is_tensorclass
from tensordict.tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import expand_as_right

from torchrl._utils import accept_remote_rref_udf_invocation

from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    Sampler,
)
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    ListStorage,
    Storage,
)
from torchrl.data.replay_buffers.utils import (
    _to_numpy,
    _to_torch,
    INT_CLASSES,
    pin_memory_output,
)
from torchrl.data.replay_buffers.writers import (
    RoundRobinWriter,
    TensorDictRoundRobinWriter,
    Writer,
)

from torchrl.data.utils import DEVICE_TYPING


class ReplayBuffer:
    """A generic, composable replay buffer class.

    Keyword Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided,
            a default :class:`~torchrl.data.replay_buffers.RandomSampler`
            will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
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
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

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
    :class:`torchrl.data.ListStorage` will:

    Examples:
        >>> torch.manual_seed(0)
        >>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
        >>> indices = buffer.extend(["a", 1, None])
        >>> buffer.sample(3)
        [None, 'a', None]
    """

    def __init__(
        self,
        *,
        storage: Optional[Storage] = None,
        sampler: Optional[Sampler] = None,
        writer: Optional[Writer] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
        batch_size: Optional[int] = None,
    ) -> None:
        self._storage = storage if storage is not None else ListStorage(max_size=1_000)
        self._storage.attach(self)
        self._sampler = sampler if sampler is not None else RandomSampler()
        self._writer = writer if writer is not None else RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._collate_fn = (
            collate_fn
            if collate_fn is not None
            else _get_default_collate(
                self._storage, _is_tensordict=isinstance(self, TensorDictReplayBuffer)
            )
        )
        self._pin_memory = pin_memory

        self._prefetch = bool(prefetch)
        self._prefetch_cap = prefetch or 0
        self._prefetch_queue = collections.deque()
        if self._prefetch_cap:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=self._prefetch_cap)

        self._replay_lock = threading.RLock()
        self._futures_lock = threading.RLock()
        from torchrl.envs.transforms.transforms import Compose

        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            transform = Compose(transform)
        transform.eval()
        self._transform = transform

        if batch_size is None and prefetch:
            raise ValueError(
                "Dynamic batch-size specification is incompatible "
                "with multithreaded sampling. "
                "When using prefetch, the batch-size must be specified in "
                "advance. "
            )
        if (
            batch_size is None
            and hasattr(self._sampler, "drop_last")
            and self._sampler.drop_last
        ):
            raise ValueError(
                "Samplers with drop_last=True must work with a predictible batch-size. "
                "Please pass the batch-size to the ReplayBuffer constructor."
            )
        self._batch_size = batch_size

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"storage={self._storage}, "
            f"sampler={self._sampler}, "
            f"writer={self._writer}"
            ")"
        )

    @pin_memory_output
    def __getitem__(self, index: Union[int, torch.Tensor]) -> Any:
        index = _to_numpy(index)
        with self._replay_lock:
            data = self._storage[index]

        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)

        if self._transform is not None and len(self._transform):
            is_td = True
            if not is_tensor_collection(data):
                data = TensorDict({"data": data}, [])
                is_td = False
            data = self._transform(data)
            if not is_td:
                data = data["data"]

        return data

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_sampler": self._sampler.state_dict(),
            "_writer": self._writer.state_dict(),
            "_batch_size": self._batch_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._sampler.load_state_dict(state_dict["_sampler"])
        self._writer.load_state_dict(state_dict["_writer"])
        self._batch_size = state_dict["_batch_size"]

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        with self._replay_lock:
            index = self._writer.add(data)
            self._sampler.add(index)
        return index

    def _extend(self, data: Sequence) -> torch.Tensor:
        with self._replay_lock:
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        If present, the inverse transforms will be called.`

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data added to the replay buffer.
        """
        if self._transform is not None and is_tensor_collection(data):
            data = self._transform.inv(data)
        elif self._transform is not None and len(self._transform):
            data = self._transform.inv(data)
        return self._extend(data)

    def update_priority(
        self,
        index: Union[int, torch.Tensor],
        priority: Union[int, torch.Tensor],
    ) -> None:
        with self._replay_lock:
            self._sampler.update_priority(index, priority)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, dict]:
        with self._replay_lock:
            index, info = self._sampler.sample(self._storage, batch_size)
            info["index"] = index
            data = self._storage[index]
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        if self._transform is not None and len(self._transform):
            is_td = True
            if not is_tensor_collection(data):
                data = TensorDict({"data": data}, [])
                is_td = False
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            data = self._transform(data)
            if is_locked:
                data.lock_()
            if not is_td:
                data = data["data"]

        return data, info

    def empty(self):
        """Empties the replay buffer and reset cursor to 0."""
        self._writer._empty()
        self._sampler._empty()
        self._storage._empty()

    def sample(
        self, batch_size: Optional[int] = None, return_info: bool = False
    ) -> Any:
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
            ret = self._sample(batch_size)
        else:
            if len(self._prefetch_queue) == 0:
                ret = self._sample(batch_size)
            else:
                with self._futures_lock:
                    ret = self._prefetch_queue.popleft().result()

            with self._futures_lock:
                while len(self._prefetch_queue) < self._prefetch_cap:
                    fut = self._prefetch_executor.submit(self._sample, batch_size)
                    self._prefetch_queue.append(fut)

        if return_info:
            return ret
        return ret[0]

    def mark_update(self, index: Union[int, torch.Tensor]) -> None:
        self._sampler.mark_update(index)

    def append_transform(self, transform: "Transform") -> None:  # noqa-F821
        """Appends transform at the end.

        Transforms are applied in order when `sample` is called.

        Args:
            transform (Transform): The transform to be appended
        """
        transform.eval()
        self._transform.append(transform)

    def insert_transform(self, index: int, transform: "Transform") -> None:  # noqa-F821
        """Inserts transform.

        Transforms are executed in order when `sample` is called.

        Args:
            index (int): Position to insert the transform.
            transform (Transform): The transform to be appended
        """
        transform.eval()
        self._transform.insert(index, transform)

    def __iter__(self):
        if self._sampler.ran_out:
            self._sampler.ran_out = False
        if self._batch_size is None:
            raise RuntimeError(
                "Cannot iterate over the replay buffer. "
                "Batch_size was not specified during construction of the replay buffer."
            )
        while not self._sampler.ran_out:
            data = self.sample()
            yield data


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer.

    All arguments are keyword-only arguments.

    Presented in
        "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
        Prioritized experience replay."
        (https://arxiv.org/abs/1511.05952)

    Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
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
            .. note::
              The batch-size can be specified at construction time via the
              ``batch_size`` argument, or at sampling time. The former should
              be preferred whenever the batch-size is consistent across the
              experiment. If the batch-size is likely to change, it can be
              passed to the :meth:`~.sample` method. This option is
              incompatible with prefetching (since this requires to know the
              batch-size in advance) as well as with samplers that have a
              ``drop_last`` argument.

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
        tensor([2, 7, 4, 3, 5]) {'_weight': array([1., 1., 1., 1., 1.], dtype=float32), 'index': array([2, 7, 4, 3, 5])}
        >>> # update priority
        >>> priority = torch.ones(5) * 5
        >>> rb.update_priority(info["index"], priority)
        >>> # and now a new sample, the weights should be updated
        >>> sample, info = rb.sample(5, return_info=True)
        >>> print(sample, info)
        tensor([2, 5, 2, 2, 5]) {'_weight': array([0.36278465, 0.36278465, 0.36278465, 0.36278465, 0.36278465],
              dtype=float32), 'index': array([2, 5, 2, 2, 5])}

    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        storage: Optional[Storage] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
        batch_size: Optional[int] = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        sampler = PrioritizedSampler(storage.max_size, alpha, beta, eps, dtype)
        super(PrioritizedReplayBuffer, self).__init__(
            storage=storage,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
        )


class TensorDictReplayBuffer(ReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.ReplayBuffer` class.

    Keyword Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.RoundRobinWriter`
            will be used.
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

    def __init__(self, *, priority_key: str = "td_error", **kw) -> None:
        writer = kw.get("writer", None)
        if writer is None:
            kw["writer"] = TensorDictRoundRobinWriter()

        super().__init__(**kw)
        self.priority_key = priority_key

    def _get_priority(self, tensordict: TensorDictBase) -> Optional[torch.Tensor]:
        if "_data" in tensordict.keys():
            tensordict = tensordict.get("_data")
        if self.priority_key not in tensordict.keys():
            return self._sampler.default_priority
        try:
            priority = tensordict.get(self.priority_key)
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
        return priority

    def add(self, data: TensorDictBase) -> int:
        if is_tensor_collection(data):
            data_add = TensorDict(
                {
                    "_data": data,
                },
                batch_size=[],
            )
            if data.batch_size:
                data_add["_rb_batch_size"] = torch.tensor(data.batch_size)

        else:
            data_add = data
        index = super().add(data_add)
        if is_tensor_collection(data_add):
            data_add.set("index", index)

        # priority = self._get_priority(data)
        # if priority:
        self.update_tensordict_priority(data_add)
        return index

    def extend(self, tensordicts: Union[List, TensorDictBase]) -> torch.Tensor:
        if is_tensor_collection(tensordicts):
            tensordicts = TensorDict(
                {"_data": tensordicts}, batch_size=tensordicts.batch_size[:1]
            )
            if tensordicts.batch_dims > 1:
                # we want the tensordict to have one dimension only. The batch size
                # of the sampled tensordicts can be changed thereafter
                if not isinstance(tensordicts, LazyStackedTensorDict):
                    tensordicts = tensordicts.clone(recurse=False)
                else:
                    tensordicts = tensordicts.contiguous()
                # we keep track of the batch size to reinstantiate it when sampling
                if "_rb_batch_size" in tensordicts.keys():
                    raise KeyError(
                        "conflicting key '_rb_batch_size'. Consider removing from data."
                    )
                shape = torch.tensor(tensordicts.batch_size[1:]).expand(
                    tensordicts.batch_size[0], tensordicts.batch_dims - 1
                )
                tensordicts.set("_rb_batch_size", shape)
            tensordicts.set(
                "index",
                torch.zeros(
                    tensordicts.shape, device=tensordicts.device, dtype=torch.int
                ),
            )

        if not is_tensor_collection(tensordicts):
            stacked_td = torch.stack(tensordicts, 0)
        else:
            stacked_td = tensordicts

        if self._transform is not None:
            stacked_td.set("_data", self._transform.inv(stacked_td.get("_data")))

        index = super()._extend(stacked_td)
        # stacked_td.set(
        #     "index",
        #     torch.tensor(index, dtype=torch.int, device=stacked_td.device),
        #     inplace=True,
        # )
        self.update_tensordict_priority(stacked_td)
        return index

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        if not isinstance(self._sampler, PrioritizedSampler):
            return
        if data.ndim:
            priority = torch.tensor(
                [self._get_priority(td) for td in data],
                dtype=torch.float,
                device=data.device,
            )
        else:
            priority = self._get_priority(data)
        index = data.get("index")
        while index.shape != priority.shape:
            # reduce index
            index = index[..., 0]
        self.update_priority(index, priority)

    def sample(
        self,
        batch_size: Optional[int] = None,
        return_info: bool = False,
        include_info: bool = None,
    ) -> TensorDictBase:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int, optional): size of data to be collected. If none
                is provided, this method will sample a batch-size as indicated
                by the sampler.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A tensordict containing a batch of data selected in the replay buffer.
            A tuple containing this tensordict and info if return_info flag is set to True.
        """
        if include_info is not None:
            warnings.warn(
                "include_info is going to be deprecated soon."
                "The default behaviour has changed to `include_info=True` "
                "to avoid bugs linked to wrongly preassigned values in the "
                "output tensordict."
            )

        data, info = super().sample(batch_size, return_info=True)
        if not is_tensorclass(data) and include_info in (True, None):
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            for k, v in info.items():
                data.set(k, expand_as_right(_to_torch(v, data.device), data))
            if is_locked:
                data.lock_()
        if return_info:
            return data, info
        return data


class TensorDictPrioritizedReplayBuffer(TensorDictReplayBuffer):
    """TensorDict-specific wrapper around the :class:`~torchrl.data.PrioritizedReplayBuffer` class.

    This class returns tensordicts with a new key ``"index"`` that represents
    the index of each element in the replay buffer. It also provides the
    :meth:`~.update_tensordict_priority` method that only requires for the
    tensordict to be passed to it with its new priority value.

    Keyword Args:
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        storage (Storage, optional): the storage to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.ListStorage` with
            ``max_size`` of ``1_000`` will be created.
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
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectories). Can be one of "max", "min",
            "median" or "mean".

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
                _weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
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
                _weight: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
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
        priority_key: str = "td_error",
        eps: float = 1e-8,
        storage: Optional[Storage] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
        reduction: Optional[str] = "max",
        batch_size: Optional[int] = None,
    ) -> None:
        if storage is None:
            storage = ListStorage(max_size=1_000)
        sampler = PrioritizedSampler(
            storage.max_size, alpha, beta, eps, reduction=reduction
        )
        super(TensorDictPrioritizedReplayBuffer, self).__init__(
            priority_key=priority_key,
            storage=storage,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
        )


@accept_remote_rref_udf_invocation
class RemoteTensorDictReplayBuffer(TensorDictReplayBuffer):
    """A remote invocation friendly ReplayBuffer class. Public methods can be invoked by remote agents using `torch.rpc` or called locally as normal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        batch_size: Optional[int] = None,
        include_info: bool = None,
        return_info: bool = False,
    ) -> TensorDictBase:
        return super().sample(
            batch_size=batch_size, include_info=include_info, return_info=return_info
        )

    def add(self, data: TensorDictBase) -> int:
        return super().add(data)

    def extend(self, tensordicts: Union[List, TensorDictBase]) -> torch.Tensor:
        return super().extend(tensordicts)

    def update_priority(
        self, index: Union[int, torch.Tensor], priority: Union[int, torch.Tensor]
    ) -> None:
        return super().update_priority(index, priority)

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        return super().update_tensordict_priority(data)


class InPlaceSampler:
    """A sampler to write tennsordicts in-place.

    To be used cautiously as this may lead to unexpected behaviour (i.e. tensordicts
    overwritten during execution).

    """

    def __init__(self, device: Optional[DEVICE_TYPING] = None):
        self.out = None
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

    def __call__(self, list_of_tds):
        if self.out is None:
            self.out = torch.stack(list_of_tds, 0).contiguous()
            if self.device is not None:
                self.out = self.out.to(self.device)
        else:
            torch.stack(list_of_tds, 0, out=self.out)
        return self.out


def _reduce(tensor: torch.Tensor, reduction: str):
    """Reduces a tensor given the reduction method."""
    if reduction == "max":
        return tensor.max().item()
    elif reduction == "min":
        return tensor.min().item()
    elif reduction == "mean":
        return tensor.mean().item()
    elif reduction == "median":
        return tensor.median().item()
    raise NotImplementedError(f"Unknown reduction method {reduction}")


def stack_tensors(list_of_tensor_iterators: List) -> Tuple[torch.Tensor]:
    """Zips a list of iterables containing tensor-like objects and stacks the resulting lists of tensors together.

    Args:
        list_of_tensor_iterators (list): Sequence containing similar iterators,
            where each element of the nested iterator is a tensor whose
            shape match the tensor of other iterators that have the same index.

    Returns:
         Tuple of stacked tensors.

    Examples:
         >>> list_of_tensor_iterators = [[torch.ones(3), torch.zeros(1,2)]
         ...     for _ in range(4)]
         >>> stack_tensors(list_of_tensor_iterators)
         (tensor([[1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.]]), tensor([[[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]]]))

    """
    return tuple(torch.stack(tensors, 0) for tensors in zip(*list_of_tensor_iterators))
