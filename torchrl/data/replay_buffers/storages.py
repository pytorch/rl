import abc
import os
from typing import Any, Sequence, Union

import torch

from torchrl.data.replay_buffers.utils import INT_CLASSES
from torchrl.data.tensordict.memmap import MemmapTensor
from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict

__all__ = ["Storage", "ListStorage", "LazyMemmapStorage", "LazyTensorStorage"]


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


class LazyTensorStorage(Storage):
    """A pre-allocated tensor storage for tensors and tensordicts.

    Args:
        size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is `torch.device("cpu")`.
    """

    def __init__(self, size, scratch_dir=None, device=None):
        self.size = int(size)
        self.initialized = False
        self.device = device if device else torch.device("cpu")

    def _init(self, data: Union[_TensorDict, torch.Tensor]) -> None:
        print("Creating a MemmapStorage...")
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemmapTensor of the desired shape, device and dtype
            out = torch.empty(
                self.size,
                *data.shape,
                device=self.device,
                dtype=data.dtype,
            )
        else:
            out = TensorDict({}, [self.size, *data.shape])
            print("The storage is being created: ")
            for key, tensor in data.items():
                out[key] = torch.empty(
                    self.size,
                    *tensor.shape,
                    device=self.device,
                    dtype=tensor.dtype,
                )

        self._storage = out
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
        out = self._storage[index]
        return out

    def __len__(self):
        return self.size


class LazyMemmapStorage(LazyTensorStorage):
    """A memory-mapped storage for tensors and tensordicts.

    Args:
        size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        scratch_dir (str or path): directory where memmap-tensors will be written.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is `torch.device("cpu")`.
    """

    def __init__(self, size, scratch_dir=None, device=None):
        self.size = int(size)
        self.initialized = False
        self.scratch_dir = None
        if scratch_dir is not None:
            self.scratch_dir = str(scratch_dir)
            if self.scratch_dir[-1] != "/":
                self.scratch_dir += "/"
        self.device = device if device else torch.device("cpu")

    def _init(self, data: Union[_TensorDict, torch.Tensor]) -> None:
        print("Creating a MemmapStorage...")
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemmapTensor of the desired shape, device and dtype
            out = MemmapTensor(
                self.size, *data.shape, device=self.device, dtype=data.dtype
            )
            filesize = os.path.getsize(out.filename) / 1024 / 1024
            print(
                f"The storage was created in {out.filename} and occupies {filesize} Mb of storage."
            )
        else:
            out = TensorDict({}, [self.size, *data.shape])
            print("The storage is being created: ")
            for key, tensor in data.items():
                out[key] = _value = MemmapTensor(
                    self.size,
                    *tensor.shape,
                    device=self.device,
                    dtype=tensor.dtype,
                    prefix=self.scratch_dir,
                )
                filesize = os.path.getsize(_value.filename) / 1024 / 1024
                print(
                    f"\t{key}: {_value.filename}, {filesize} Mb of storage (size: {[self.size, *tensor.shape]})."
                )

        self._storage = out
        self.initialized = True
