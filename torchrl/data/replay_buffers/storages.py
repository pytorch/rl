# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
from typing import Any, Sequence, Union

import torch

from torchrl.data.replay_buffers.utils import INT_CLASSES
from torchrl.data.tensordict.memmap import MemmapTensor
from torchrl.data.tensordict.tensordict import TensorDictBase, TensorDict

__all__ = ["Storage", "ListStorage", "LazyMemmapStorage", "LazyTensorStorage"]


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)

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
    def __init__(self, max_size: int):
        super().__init__(max_size)
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
            if cursor > len(self._storage):
                raise RuntimeError(
                    "Cannot append data located more than one item away from "
                    f"the storage size: the storage size is {len(self)} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor >= self.max_size:
                raise RuntimeError(
                    f"Cannot append data to the list storage: "
                    f"maximum capacity is {self.max_size} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor == len(self._storage):
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

    def __init__(self, max_size, scratch_dir=None, device=None):
        super().__init__(max_size)
        self.initialized = False
        self.device = device if device else torch.device("cpu")
        self._len = 0

    def _init(self, data: Union[TensorDictBase, torch.Tensor]) -> None:
        print("Creating a TensorStorage...")
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemmapTensor of the desired shape, device and dtype
            out = torch.empty(
                self.max_size,
                *data.shape,
                device=self.device,
                dtype=data.dtype,
            )
        else:
            out = TensorDict({}, [self.max_size, *data.shape])
            print("The storage is being created: ")
            for key, tensor in data.items():
                if isinstance(tensor, TensorDictBase):
                    out[key] = (
                        tensor.expand(self.max_size).clone().to(self.device).zero_()
                    )
                else:
                    out[key] = torch.empty(
                        self.max_size,
                        *tensor.shape,
                        device=self.device,
                        dtype=tensor.dtype,
                    )

        self._storage = out
        self.initialized = True

    def set(
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Union[TensorDictBase, torch.Tensor],
    ):
        if isinstance(cursor, INT_CLASSES):
            self._len = max(self._len, cursor + 1)
        else:
            self._len = max(self._len, max(cursor) + 1)

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
        return self._len


class LazyMemmapStorage(LazyTensorStorage):
    """A memory-mapped storage for tensors and tensordicts.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        scratch_dir (str or path): directory where memmap-tensors will be written.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is `torch.device("cpu")`.
    """

    def __init__(self, max_size, scratch_dir=None, device=None):
        super().__init__(max_size)
        self.initialized = False
        self.scratch_dir = None
        if scratch_dir is not None:
            self.scratch_dir = str(scratch_dir)
            if self.scratch_dir[-1] != "/":
                self.scratch_dir += "/"
        self.device = device if device else torch.device("cpu")
        self._len = 0

    def _init(self, data: Union[TensorDictBase, torch.Tensor]) -> None:
        print("Creating a MemmapStorage...")
        if isinstance(data, torch.Tensor):
            # if Tensor, we just create a MemmapTensor of the desired shape, device and dtype
            out = MemmapTensor(
                self.max_size, *data.shape, device=self.device, dtype=data.dtype
            )
            filesize = os.path.getsize(out.filename) / 1024 / 1024
            print(
                f"The storage was created in {out.filename} and occupies {filesize} Mb of storage."
            )
        else:
            out = TensorDict({}, [self.max_size, *data.shape])
            print("The storage is being created: ")
            for key, tensor in data.items():
                if isinstance(tensor, TensorDictBase):
                    out[key] = (
                        tensor.expand(self.max_size)
                        .clone()
                        .zero_()
                        .memmap_(prefix=self.scratch_dir)
                        .to(self.device)
                    )
                else:
                    out[key] = _value = MemmapTensor(
                        self.max_size,
                        *tensor.shape,
                        device=self.device,
                        dtype=tensor.dtype,
                        prefix=self.scratch_dir,
                    )
                filesize = os.path.getsize(_value.filename) / 1024 / 1024
                print(
                    f"\t{key}: {_value.filename}, {filesize} Mb of storage (size: {[self.max_size, *tensor.shape]})."
                )
        self._storage = out
        self.initialized = True
