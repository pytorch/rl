# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
from collections import OrderedDict
from copy import copy
from typing import Any, Dict, Sequence, Union

import torch
from tensordict.memmap import MemmapTensor
from tensordict.prototype import is_tensorclass
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl._utils import _CKPT_BACKEND
from torchrl.data.replay_buffers.utils import INT_CLASSES

try:
    from torchsnapshot.serialization import tensor_from_memoryview

    _has_ts = True
except ImportError:
    _has_ts = False


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)
        # Prototype feature. RBs that use a given instance of Storage should add
        # themselves to this set.
        self._attached_entities = set()

    @abc.abstractmethod
    def set(self, cursor: int, data: Any):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        raise NotImplementedError

    def attach(self, buffer: Any) -> None:
        """This function attaches a sampler to this storage.

        Buffers that read from this storage must be included as an attached
        entity by calling this method. This guarantees that when data
        in the storage changes, components are made aware of changes even if the storage
        is shared with other buffers (eg. Priority Samplers).

        Args:
            buffer: the object that reads from this storage.
        """
        self._attached_entities.add(buffer)

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        ret = self.set(index, value)
        for ent in self._attached_entities:
            ent.mark_update(index)
        return ret

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError


class ListStorage(Storage):
    """A storage stored in a list.

    Args:
        max_size (int): the maximum number of elements stored in the storage.

    """

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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": [
                elt if not hasattr(elt, "state_dict") else elt.state_dict()
                for elt in self._storage
            ]
        }

    def load_state_dict(self, state_dict):
        _storage = state_dict["_storage"]
        self._storage = []
        for elt in _storage:
            if isinstance(elt, torch.Tensor):
                self._storage.append(elt)
            elif isinstance(elt, (dict, OrderedDict)):
                self._storage.append(TensorDict({}, []).load_state_dict(elt))
            else:
                raise TypeError(
                    f"Objects of type {type(elt)} are not supported by ListStorage.load_state_dict"
                )


class LazyTensorStorage(Storage):
    """A pre-allocated tensor storage for tensors and tensordicts.

    Args:
        size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._storage = None
        return super().__new__(cls)

    def __init__(self, max_size, scratch_dir=None, device=None):
        super().__init__(max_size)
        self.initialized = False
        self.device = device if device else torch.device("cpu")
        self._len = 0

    def state_dict(self) -> Dict[str, Any]:
        _storage = self._storage
        if isinstance(_storage, torch.Tensor):
            pass
        elif isinstance(_storage, TensorDictBase):
            _storage = _storage.state_dict()
        elif _storage is None:
            _storage = {}
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by LazyTensorStorage.state_dict"
            )
        return {
            "_storage": _storage,
            "initialized": self.initialized,
            "_len": self._len,
        }

    def load_state_dict(self, state_dict):
        _storage = copy(state_dict["_storage"])
        if isinstance(_storage, torch.Tensor):
            if isinstance(self._storage, torch.Tensor):
                self._storage.copy_(_storage)
            elif self._storage is None:
                self._storage = _storage
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if isinstance(self._storage, TensorDictBase):
                self._storage.load_state_dict(_storage)
            elif self._storage is None:
                batch_size = _storage.pop("__batch_size")
                device = _storage.pop("__device")
                self._storage = TensorDict(
                    _storage, batch_size=batch_size, device=device
                )
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by ListStorage.load_state_dict"
            )
        self.initialized = state_dict["initialized"]
        self._len = state_dict["_len"]

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
        elif is_tensorclass(data):
            out = (
                data.expand(self.max_size, *data.shape).clone().zero_().to(self.device)
            )
        else:
            out = (
                data.expand(self.max_size, *data.shape)
                .to_tensordict()
                .zero_()
                .clone()
                .to(self.device)
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
            stored and sent. Default is :obj:`torch.device("cpu")`.
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

    def state_dict(self) -> Dict[str, Any]:
        _storage = self._storage
        if isinstance(_storage, torch.Tensor):
            _storage = _mem_map_tensor_as_tensor(_storage)
        elif isinstance(_storage, TensorDictBase):
            _storage = _storage.apply(_mem_map_tensor_as_tensor).state_dict()
        elif _storage is None:
            _storage = {}
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by LazyTensorStorage.state_dict"
            )
        return {
            "_storage": _storage,
            "initialized": self.initialized,
            "_len": self._len,
        }

    def load_state_dict(self, state_dict):
        _storage = copy(state_dict["_storage"])
        if isinstance(_storage, torch.Tensor):
            if isinstance(self._storage, torch.Tensor):
                _mem_map_tensor_as_tensor(self._storage).copy_(_storage)
            elif self._storage is None:
                self._storage = MemmapTensor(_storage)
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if isinstance(self._storage, TensorDictBase):
                self._storage.load_state_dict(_storage)
                self._storage.memmap_()
            elif self._storage is None:
                batch_size = _storage.pop("__batch_size")
                device = _storage.pop("__device")
                self._storage = TensorDict(
                    _storage, batch_size=batch_size, device=device
                )
                self._storage.memmap_()
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by ListStorage.load_state_dict"
            )
        self.initialized = state_dict["initialized"]
        self._len = state_dict["_len"]

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
        elif is_tensorclass(data):
            out = (
                data.clone()
                .expand(self.max_size, *data.shape)
                .memmap_like(prefix=self.scratch_dir)
                .to(self.device)
            )
            for key, tensor in sorted(
                out.items(include_nested=True, leaves_only=True), key=str
            ):
                filesize = os.path.getsize(tensor.filename) / 1024 / 1024
                print(
                    f"\t{key}: {tensor.filename}, {filesize} Mb of storage (size: {tensor.shape})."
                )
        else:
            # out = TensorDict({}, [self.max_size, *data.shape])
            print("The storage is being created: ")
            out = (
                data.clone()
                .expand(self.max_size, *data.shape)
                .memmap_like(prefix=self.scratch_dir)
                .to(self.device)
            )
            for key, tensor in sorted(
                out.items(include_nested=True, leaves_only=True), key=str
            ):
                filesize = os.path.getsize(tensor.filename) / 1024 / 1024
                print(
                    f"\t{key}: {tensor.filename}, {filesize} Mb of storage (size: {tensor.shape})."
                )
        self._storage = out
        self.initialized = True


# Utils
def _mem_map_tensor_as_tensor(mem_map_tensor: MemmapTensor) -> torch.Tensor:
    if _CKPT_BACKEND == "torchsnapshot" and not _has_ts:
        raise ImportError(
            "the checkpointing backend is set to torchsnapshot but the library is not installed. Consider installing the library or switch to another backend. "
            f"Supported backends are {_CKPT_BACKEND.backends}"
        )
    if isinstance(mem_map_tensor, torch.Tensor):
        return mem_map_tensor
    if _CKPT_BACKEND == "torchsnapshot":
        # TorchSnapshot doesn't know how to stream MemmapTensor, so we view MemmapTensor
        # as a Tensor for saving and loading purposes. This doesn't incur any copy.
        return tensor_from_memoryview(
            dtype=mem_map_tensor.dtype,
            shape=list(mem_map_tensor.shape),
            mv=memoryview(mem_map_tensor._memmap_array),
        )
    elif _CKPT_BACKEND == "torch":
        return mem_map_tensor._tensor


def _collate_list_tensordict(x):
    out = torch.stack(x, 0)
    if isinstance(out, TensorDictBase):
        return out.to_tensordict()
    return out


def _collate_list_tensors(*x):
    return tuple(torch.stack(_x, 0) for _x in zip(*x))


def _collate_contiguous(x):
    if isinstance(x, TensorDictBase):
        return x.to_tensordict()
    return x.clone()


def _get_default_collate(storage, _is_tensordict=True):
    if isinstance(storage, ListStorage):
        if _is_tensordict:
            return _collate_list_tensordict
        else:
            return _collate_list_tensors
    elif isinstance(storage, (LazyTensorStorage, LazyMemmapStorage)):
        return _collate_contiguous
