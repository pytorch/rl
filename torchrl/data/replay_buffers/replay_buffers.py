# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from tensordict.tensordict import LazyStackedTensorDict, TensorDictBase
from tensordict.utils import expand_right

from torchrl.data.utils import DEVICE_TYPING

from .samplers import PrioritizedSampler, RandomSampler, Sampler
from .storages import _get_default_collate, ListStorage, Storage
from .utils import _to_numpy, accept_remote_rref_udf_invocation, INT_CLASSES
from .writers import RoundRobinWriter, Writer


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


def _pin_memory(output: Any) -> Any:
    if hasattr(output, "pin_memory") and output.device == torch.device("cpu"):
        return output.pin_memory()
    else:
        return output


def pin_memory_output(fun) -> Callable:
    """Calls pin_memory on outputs of decorated function if they have such method."""

    def decorated_fun(self, *args, **kwargs):
        output = fun(self, *args, **kwargs)
        if self._pin_memory:
            _tuple_out = True
            if not isinstance(output, tuple):
                _tuple_out = False
                output = (output,)
            output = tuple(_pin_memory(_output) for _output in output)
            if _tuple_out:
                return output
            return output[0]
        return output

    return decorated_fun


class ReplayBuffer:
    """A generic, composable replay buffer class.

    Args:
        storage (Storage, optional): the storage to be used. If none is provided
            a default ListStorage with max_size of 1_000 will be created.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default RoundRobinWriter() will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :obj:`Compose` class.
    """

    def __init__(
        self,
        storage: Optional[Storage] = None,
        sampler: Optional[Sampler] = None,
        writer: Optional[Writer] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
    ) -> None:
        self._storage = storage if storage is not None else ListStorage(max_size=1_000)
        self._storage.attach(self)
        self._sampler = sampler if sampler is not None else RandomSampler()
        self._writer = writer if writer is not None else RoundRobinWriter()
        self._writer.register_storage(self._storage)

        self._collate_fn = (
            collate_fn
            if collate_fn is not None
            else _get_default_collate(self._storage)
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

        return data

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": self._storage.state_dict(),
            "_sampler": self._sampler.state_dict(),
            "_writer": self._writer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._storage.load_state_dict(state_dict["_storage"])
        self._sampler.load_state_dict(state_dict["_sampler"])
        self._writer.load_state_dict(state_dict["_writer"])

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

    def extend(self, data: Sequence) -> torch.Tensor:
        """Extends the replay buffer with one or more elements contained in an iterable.

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data aded to the replay buffer.
        """
        data = self._transform.inv(data)
        with self._replay_lock:
            index = self._writer.extend(data)
            self._sampler.extend(index)
        return index

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
            data = self._storage[index]
        if not isinstance(index, INT_CLASSES):
            data = self._collate_fn(data)
        data = self._transform(data)
        return data, info

    def sample(self, batch_size: int, return_info: bool = False) -> Any:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int): size of data to be collected.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A batch of data selected in the replay buffer.
            A tuple containing this batch and info if return_info flag is set to True.
        """
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer.

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
        dtype (torch.dtype): type of the data. Can be torch.float or torch.double.
        storage (Storage, optional): the storage to be used. If none is provided
            a default ListStorage with max_size of 1_000 will be created.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :obj:`Compose` class.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        storage: Optional[Storage] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
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
        )


class TensorDictReplayBuffer(ReplayBuffer):
    """TensorDict-specific wrapper around the ReplayBuffer class.

    Args:
        priority_key (str): the key at which priority is assumed to be stored
            within TensorDicts added to this ReplayBuffer.
    """

    def __init__(self, priority_key: str = "td_error", **kw) -> None:
        super().__init__(**kw)
        self.priority_key = priority_key

    def _get_priority(self, tensordict: TensorDictBase) -> Optional[torch.Tensor]:
        if self.priority_key not in tensordict.keys():
            return self._sampler.default_priority
        if tensordict.batch_dims:
            tensordict = tensordict.clone(recurse=False)
            tensordict.batch_size = []
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
        index = super().add(data)
        data.set("index", index)

        priority = self._get_priority(data)
        if priority:
            self.update_priority(index, priority)
        return index

    def extend(self, tensordicts: Union[List, TensorDictBase]) -> torch.Tensor:
        if isinstance(tensordicts, TensorDictBase):
            if tensordicts.batch_dims > 1:
                # we want the tensordict to have one dimension only. The batch size
                # of the sampled tensordicts can be changed thereafter
                if not isinstance(tensordicts, LazyStackedTensorDict):
                    tensordicts = tensordicts.clone(recurse=False)
                else:
                    tensordicts = tensordicts.contiguous()
                # we keep track of the batch size to reinstantiate it when sampling
                if "_batch_size" in tensordicts.keys():
                    raise KeyError(
                        "conflicting key '_batch_size'. Consider removing from data."
                    )
                shape = torch.tensor(tensordicts.batch_size[1:]).expand(
                    tensordicts.batch_size[0], tensordicts.batch_dims - 1
                )
                tensordicts.batch_size = tensordicts.batch_size[:1]
                tensordicts.set("_batch_size", shape)
            tensordicts.set(
                "index",
                torch.zeros(
                    tensordicts.shape, device=tensordicts.device, dtype=torch.int
                ),
            )

        if not isinstance(tensordicts, TensorDictBase):
            stacked_td = torch.stack(tensordicts, 0)
        else:
            stacked_td = tensordicts

        index = super().extend(stacked_td)
        stacked_td.set(
            "index",
            torch.tensor(index, dtype=torch.int, device=stacked_td.device),
            inplace=True,
        )
        self.update_tensordict_priority(stacked_td)
        return index

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        priority = torch.tensor(
            [self._get_priority(td) for td in data],
            dtype=torch.float,
            device=data.device,
        )
        # if the index shape does not match the priority shape, we have expanded it.
        # we just take the first value
        index = data.get("index")
        while index.shape != priority.shape:
            # reduce index
            index = index[..., 0]
        self.update_priority(index, priority)

    def sample(
        self, batch_size: int, include_info: bool = False, return_info: bool = False
    ) -> TensorDictBase:
        """Samples a batch of data from the replay buffer.

        Uses Sampler to sample indices, and retrieves them from Storage.

        Args:
            batch_size (int): size of data to be collected.
            include_info (bool): whether to add info to the returned tensordict.
            return_info (bool): whether to return info. If True, the result
                is a tuple (data, info). If False, the result is the data.

        Returns:
            A tensordict containing a batch of data selected in the replay buffer.
            A tuple containing this tensordict and info if return_info flag is set to True.
        """
        data, info = super().sample(batch_size, return_info=True)
        if include_info:
            for k, v in info.items():
                data.set(k, torch.tensor(v, device=data.device), inplace=True)
        if "_batch_size" in data.keys():
            # we need to reset the batch-size
            shape = data.pop("_batch_size")
            shape = shape[0]
            shape = torch.Size([data.shape[0], *shape])
            # we may need to update some values in the data
            for key, value in data.items():
                if value.ndim >= len(shape):
                    continue
                value = expand_right(value, shape)
                data.set(key, value)
            data.batch_size = shape
        if return_info:
            return data, info
        return data


class TensorDictPrioritizedReplayBuffer(TensorDictReplayBuffer):
    """TensorDict-specific wrapper around the PrioritizedReplayBuffer class.

    This class returns tensordicts with a new key "index" that represents
    the index of each element in the replay buffer. It also provides the
    'update_tensordict_priority' method that only requires for the
    tensordict to be passed to it with its new priority value.

    Args:
        alpha (float): exponent α determines how much prioritization is
            used, with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        priority_key (str, optional): key where the priority value can be
            found in the stored tensordicts. Default is :obj:`"td_error"`
        eps (float, optional): delta added to the priorities to ensure that the
            buffer does not contain null priorities.
        dtype (torch.dtype): type of the data. Can be torch.float or torch.double.
        storage (Storage, optional): the storage to be used. If none is provided
            a default ListStorage with max_size of 1_000 will be created.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched loading
            from a map-style dataset.
        pin_memory (bool, optional): whether pin_memory() should be called on
            the rb samples. Default is :obj:`False`.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :obj:`Compose` class.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (ie stored trajectories). Can be one of "max", "min",
            "median" or "mean".
    """

    def __init__(
        self,
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
        )


@accept_remote_rref_udf_invocation
class RemoteTensorDictReplayBuffer(TensorDictReplayBuffer):
    """A remote invocation friendly ReplayBuffer class. Public methods can be invoked by remote agents using `torch.rpc` or called locally as normal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self, batch_size: int, include_info: bool = False, return_info: bool = False
    ) -> TensorDictBase:
        return super().sample(batch_size, include_info, return_info)

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
