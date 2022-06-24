import abc
from typing import Any, Sequence, Union

__all__ = ["Storage", "ListStorage", "LazyMemmapStorage"]

import torch

from torchrl.data.replay_buffers.utils import INT_CLASSES
from torchrl.data.tensordict.memmap import MemmapTensor
from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    @abc.abstractmethod
    def set(self, cursor: int, data: Any):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        return self.set(index, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class ListStorage(Storage):
    def __init__(self):
        self._storage = []

    def set(self, cursor: Union[int, Sequence[int], slice], data: Any):
        if not isinstance(cursor, INT_CLASSES):
            if isinstance(cursor, slice):
                self._storage[cursor] = data
                return
            for _cursor, _data in zip(cursor, data):
                self.set(_cursor, _data)
            return
        else:
            if cursor >= len(self._storage):
                if cursor != len(self._storage):
                    raise RuntimeError(
                        "Cannot append data located more than one item away from"
                        f"the storage size: the storage size is {len(self)} and the"
                        f"index of the item to be set is {cursor}."
                    )
                self._storage.append(data)
            else:
                self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if isinstance(index, (INT_CLASSES, slice)):
            return self._storage[index]
        else:
            return [self._storage[i] for i in index]

    def __len__(self):
        return len(self._storage)


class LazyMemmapStorage(Storage):
    def __init__(self, size):
        self.size = size
        self.initialized = False

    def _init(self, data: Union[_TensorDict, torch.Tensor]) -> None:
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemmapTensor of the desired shape, device and dtype
            data = MemmapTensor(
                self.size, *data.shape, device=data.device, dtype=data.dtype
            )
        else:
            data = TensorDict(
                {
                    key: MemmapTensor(
                        self.size,
                        *tensor.shape,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    for key, tensor in data.items()
                },
                [self.size, *data.shape],
            )
        self._storage = data
        self.initialized = True

    def set(
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Union[_TensorDict, torch.Tensor],
    ):
        if not self.initialized:
            if not isinstance(cursor, INT_CLASSES):
                self._init(data[0])
            else:
                self._init(data)
        self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if not self.initialized:
            raise RuntimeError(
                "Cannot get an item from an unitialized LazyMemmapStorage"
            )
        return self._storage[index]

    def __len__(self):
        return self.size
