# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from multiprocessing.context import get_spawning_popen
from typing import Any

import numpy as np
import tensordict
import torch
from tensordict import is_tensor_collection, lazy_stack, TensorDict, TensorDictBase
from tensordict.utils import _zip_strict
from torch.utils._pytree import tree_map

from torchrl.data.replay_buffers.checkpointers import (
    CompressedListStorageCheckpointer,
    ListStorageCheckpointer,
)
from torchrl.data.replay_buffers.utils import INT_CLASSES

from .base import Storage


class ListStorage(Storage):
    """A storage stored in a list.

    This class cannot be extended with PyTrees, the data provided during calls to
    :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` should be iterables
    (like lists, tuples, tensors or tensordicts with non-empty batch-size).

    Args:
        max_size (int, optional): the maximum number of elements stored in the storage.
            If not provided, an unlimited storage is created.

    Keyword Args:
        compilable (bool, optional): if ``True``, the storage will be made compatible with :func:`~torch.compile` at
            the cost of being executable in multiprocessed settings.
        device (str, optional): the device to use for the storage. Defaults to `None` (inputs are not moved to the device).

    """

    _default_checkpointer = ListStorageCheckpointer

    def __init__(
        self,
        max_size: int | None = None,
        *,
        compilable: bool = False,
        device: torch.device | str | int | None = None,
    ):
        if max_size is None:
            max_size = torch.iinfo(torch.int64).max
        super().__init__(max_size, compilable=compilable)
        self._storage = []
        self.device = device

    def _to_device(self, data: Any) -> Any:
        """Utility method to move data to the device."""
        if self.device is not None:
            if hasattr(data, "to"):
                data = data.to(self.device)
            else:
                data = tree_map(
                    lambda x: x.to(self.device) if hasattr(x, "to") else x, data
                )
        return data

    def set(
        self,
        cursor: int | Sequence[int] | slice,
        data: Any,
        *,
        set_cursor: bool = True,
    ):
        if not isinstance(cursor, INT_CLASSES):
            if (isinstance(cursor, torch.Tensor) and cursor.ndim == 0) or (
                isinstance(cursor, np.ndarray) and cursor.ndim == 0
            ):
                self.set(int(cursor), data, set_cursor=set_cursor)
                return
            if isinstance(cursor, slice):
                data = self._to_device(data)
                self._set_slice(cursor, data)
                self._bump_mutation_revision()
                return
            if isinstance(
                data,
                (
                    list,
                    tuple,
                    torch.Tensor,
                    TensorDictBase,
                    *tensordict.base._ACCEPTED_CLASSES,
                    range,
                    set,
                    np.ndarray,
                ),
            ):
                for _cursor, _data in _zip_strict(cursor, data):
                    self.set(_cursor, _data, set_cursor=set_cursor)
            else:
                raise TypeError(
                    f"Cannot extend a {type(self)} with data of type {type(data)}. "
                    f"Provide a list, tuple, set, range, np.ndarray, tensor or tensordict subclass instead."
                )
            return
        else:
            if cursor > len(self._storage):
                raise RuntimeError(
                    "Cannot append data located more than one item away from "
                    f"the storage size: the storage size is {len(self._storage)} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor >= self.max_size:
                raise RuntimeError(
                    f"Cannot append data to the list storage: "
                    f"maximum capacity is {self.max_size} "
                    f"and the index of the item to be set is {cursor}."
                )
            data = self._to_device(data)
            self._set_item(cursor, data)
            self._bump_mutation_revision()

    def _set_item(self, cursor: int, data: Any) -> None:
        """Set a single item in the storage."""
        if cursor == len(self._storage):
            self._storage.append(data)
        else:
            self._storage[cursor] = data

    def _set_slice(self, cursor: slice, data: Any) -> None:
        """Set a slice in the storage."""
        self._storage[cursor] = data

    def get(self, index: int | Sequence[int] | slice) -> Any:
        if isinstance(index, INT_CLASSES):
            return self._get_item(index)
        elif isinstance(index, slice):
            return self._get_slice(index)
        elif isinstance(index, tuple):
            if len(index) > 1:
                raise RuntimeError(
                    f"{type(self).__name__} can only be indexed with one-length tuples."
                )
            return self.get(index[0])
        else:
            if isinstance(index, torch.Tensor) and index.device.type != "cpu":
                index = index.cpu().tolist()
            return self._get_list(index)

    def _get_item(self, index: int) -> Any:
        """Get a single item from the storage."""
        return self._storage[index]

    def _get_slice(self, index: slice) -> Any:
        """Get a slice from the storage."""
        return self._storage[index]

    def _get_list(self, index: list) -> list:
        """Get a list of items from the storage."""
        return [self._storage[i] for i in index]

    def __len__(self):
        """Get the length of the storage."""
        return len(self._storage)

    def state_dict(self) -> dict[str, Any]:
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
            # clone to decouple the storage from the caller's tensors (which may
            # e.g. be mmap-backed views over a checkpoint file)
            if isinstance(elt, torch.Tensor):
                self._storage.append(elt.clone())
            elif isinstance(elt, (dict, OrderedDict)):
                self._storage.append(
                    TensorDict().load_state_dict(elt, strict=False).clone()
                )
            else:
                raise TypeError(
                    f"Objects of type {type(elt)} are not supported by ListStorage.load_state_dict"
                )
        self._bump_mutation_revision()

    def _empty(self):
        self._storage = []
        self._bump_mutation_revision()

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processes."
            )
        state = super().__getstate__()
        return state

    def __repr__(self):
        storage = getattr(self, "_storage", [None])
        if not storage:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(items=[{storage[0]}, ...])"

    def contains(self, item):
        if isinstance(item, int):
            if item < 0:
                item += len(self._storage)
            return self._contains_int(item)
        if isinstance(item, torch.Tensor):
            return torch.tensor(
                [self.contains(elt) for elt in item.tolist()],
                dtype=torch.bool,
                device=item.device,
            ).reshape_as(item)
        raise NotImplementedError(f"type {type(item)} is not supported yet.")

    def _contains_int(self, item: int) -> bool:
        """Check if an integer index is contained in the storage."""
        return 0 <= item < len(self._storage)


class LazyStackStorage(ListStorage):
    """A ListStorage that returns LazyStackTensorDict instances.

    This storage allows for heterougeneous structures to be indexed as a single `TensorDict` representation.
    It uses :class:`~tensordict.LazyStackedTensorDict` which operates on non-contiguous lists of tensordicts,
    lazily stacking items when queried.
    This means that this storage is going to be fast to sample but data access may be slow (as it requires a stack).
    Tensors of heterogeneous shapes can also be stored within the storage and stacked together.
    Because the storage is represented as a list, the number of tensors to store in memory will grow linearly with
    the size of the buffer.

    If possible, nested tensors can also be created via :meth:`~tensordict.LazyStackedTensorDict.densify`
    (see :mod:`~torch.nested`).

    Args:
        max_size (int, optional): the maximum number of elements stored in the storage.
            If not provided, an unlimited storage is created.

    Keyword Args:
        compilable (bool, optional): if ``True``, the storage will be made compatible with :func:`~torch.compile` at
            the cost of being executable in multiprocessed settings.
        stack_dim (int, optional): the stack dimension in terms of TensorDict batch sizes. Defaults to `0`.
        device (str, optional): the device to use for the storage. Defaults to `None` (inputs are not moved to the device).

    Examples:
        >>> import torch
        >>> from torchrl.data import ReplayBuffer, LazyStackStorage
        >>> from tensordict import TensorDict
        >>> _ = torch.manual_seed(0)
        >>> rb = ReplayBuffer(storage=LazyStackStorage(max_size=1000, stack_dim=-1))
        >>> data0 = TensorDict(a=torch.randn((10,)), b=torch.rand(4), c="a string!")
        >>> data1 = TensorDict(a=torch.randn((11,)), b=torch.rand(4), c="another string!")
        >>> _ = rb.add(data0)
        >>> _ = rb.add(data1)
        >>> rb.sample(10)
        LazyStackedTensorDict(
            fields={
                a: Tensor(shape=torch.Size([10, -1]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                c: NonTensorStack(
                    ['another string!', 'another string!', 'another st...,
                    batch_size=torch.Size([10]),
                    device=None)},
            exclusive_fields={
            },
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False,
            stack_dim=0)
    """

    def __init__(
        self,
        max_size: int | None = None,
        *,
        compilable: bool = False,
        stack_dim: int = 0,
        device: torch.device | str | int | None = None,
    ):
        super().__init__(max_size=max_size, compilable=compilable, device=device)
        self.stack_dim = stack_dim

    def get(self, index: int | Sequence[int] | slice) -> Any:
        out = super().get(index=index)
        if isinstance(out, list):
            stack_dim = self.stack_dim
            if stack_dim < 0:
                stack_dim = out[0].ndim + 1 + stack_dim
            out = lazy_stack(list(out), stack_dim)
            return out
        return out


class CompressedListStorage(ListStorage):
    """A storage that compresses and decompresses data.

    This storage compresses data when storing and decompresses when retrieving.
    It's particularly useful for storing raw sensory observations like images
    that can be compressed significantly to save memory.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.
        compression_fn (callable, optional): function to compress data. Should take
            a tensor and return a compressed byte tensor. Defaults to zstd compression.
        decompression_fn (callable, optional): function to decompress data. Should take
            a compressed byte tensor and return the original tensor. Defaults to zstd decompression.
        compression_level (int, optional): compression level (1-22 for zstd) when using the default compression function.
            Defaults to 3.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
        compilable (bool, optional): whether the storage is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from torchrl.data import CompressedListStorage, ReplayBuffer
        >>> from tensordict import TensorDict
        >>>
        >>> # Create a compressed storage for image data
        >>> storage = CompressedListStorage(max_size=1000, compression_level=3)
        >>> rb = ReplayBuffer(storage=storage, batch_size=5)
        >>>
        >>> # Add some image data
        >>> images = torch.randn(10, 3, 84, 84)  # Atari-like frames
        >>> data = TensorDict({"obs": images}, batch_size=[10])
        >>> rb.extend(data)
        >>>
        >>> # Sample and verify data is decompressed correctly
        >>> sample = rb.sample(3)
        >>> print(sample["obs"].shape)  # torch.Size([3, 3, 84, 84])

    """

    _default_checkpointer = CompressedListStorageCheckpointer

    def __init__(
        self,
        max_size: int,
        *,
        compression_fn: Callable | None = None,
        decompression_fn: Callable | None = None,
        compression_level: int = 3,
        device: torch.device = "cpu",
        compilable: bool = False,
    ):
        super().__init__(max_size, compilable=compilable, device=device)
        self.compression_level = compression_level

        # Set up compression functions
        if compression_fn is None:
            self.compression_fn = self._default_compression_fn
        else:
            self.compression_fn = compression_fn

        if decompression_fn is None:
            self.decompression_fn = self._default_decompression_fn
        else:
            self.decompression_fn = decompression_fn

        # Store compressed data and metadata
        self._storage = []
        self._metadata = []  # Store shape, dtype, device info for each item

    def _default_compression_fn(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default compression using zstd."""
        if sys.version_info >= (3, 14):
            from compression import zstd

            compressor_fn = zstd.compress

        else:
            import zlib

            compressor_fn = zlib.compress

        # Convert tensor to bytes
        tensor_bytes = self.to_bytestream(tensor)

        # Compress with zstd
        compressed_bytes = compressor_fn(tensor_bytes, level=self.compression_level)

        # Convert to tensor
        return torch.frombuffer(bytearray(compressed_bytes), dtype=torch.uint8)

    def _default_decompression_fn(
        self, compressed_tensor: torch.Tensor, metadata: dict
    ) -> torch.Tensor:
        """Default decompression using zstd."""
        if sys.version_info >= (3, 14):
            from compression import zstd

            decompressor_fn = zstd.decompress

        else:
            import zlib

            decompressor_fn = zlib.decompress

        # Convert tensor to bytes
        compressed_bytes = self.to_bytestream(compressed_tensor.cpu())

        # Decompress with zstd
        decompressed_bytes = decompressor_fn(compressed_bytes)

        # Convert back to tensor
        tensor = torch.frombuffer(
            bytearray(decompressed_bytes), dtype=metadata["dtype"]
        )
        tensor = tensor.reshape(metadata["shape"])
        tensor = tensor.to(metadata["device"])

        return tensor

    def _compress_item(self, item: Any) -> tuple[torch.Tensor, dict]:
        """Compress a single item and return compressed data with metadata."""
        if isinstance(item, torch.Tensor):
            metadata = {
                "type": "tensor",
                "shape": item.shape,
                "dtype": item.dtype,
                "device": item.device,
            }
            compressed = self.compression_fn(item)
        elif is_tensor_collection(item):
            # For TensorDict, compress each tensor field
            compressed_fields = {}
            metadata = {"type": "tensordict", "fields": {}}

            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    compressed_fields[key] = self.compression_fn(value)
                    metadata["fields"][key] = {
                        "type": "tensor",
                        "shape": value.shape,
                        "dtype": value.dtype,
                        "device": value.device,
                    }
                else:
                    # For non-tensor data, store as-is
                    compressed_fields[key] = value
                    metadata["fields"][key] = {"type": "non_tensor", "value": value}

            compressed = compressed_fields
        else:
            # For other types, store as-is
            compressed = item
            metadata = {"type": "other", "value": item}

        return compressed, metadata

    def _decompress_item(self, compressed_data: Any, metadata: dict) -> Any:
        """Decompress a single item using its metadata."""
        if metadata["type"] == "tensor":
            return self.decompression_fn(compressed_data, metadata)
        elif metadata["type"] == "tensordict":
            # Reconstruct TensorDict
            result = TensorDict({}, batch_size=metadata.get("batch_size", []))

            for key, field_metadata in metadata["fields"].items():
                if field_metadata["type"] == "non_tensor":
                    result[key] = field_metadata["value"]
                else:
                    # Decompress tensor field
                    result[key] = self.decompression_fn(
                        compressed_data[key], field_metadata
                    )

            return result
        else:
            # Return as-is for other types
            return metadata["value"]

    def _set_item(self, cursor: int, data: Any) -> None:
        """Set a single item in the compressed storage."""
        # Ensure we have enough space
        while len(self._storage) <= cursor:
            self._storage.append(None)
            self._metadata.append(None)

        # Compress and store
        compressed_data, metadata = self._compress_item(data)
        self._storage[cursor] = compressed_data
        self._metadata[cursor] = metadata

    def _set_slice(self, cursor: slice, data: Any) -> None:
        """Set a slice in the compressed storage."""
        # Handle slice assignment
        if not hasattr(data, "__iter__"):
            data = [data]
        start, stop, step = cursor.indices(len(self._storage))
        indices = list(range(start, stop, step))

        for i, value in zip(indices, data):
            self._set_item(i, value)

    def _get_item(self, index: int) -> Any:
        """Get a single item from the compressed storage."""
        if index >= len(self._storage) or self._storage[index] is None:
            raise IndexError(f"Index {index} out of bounds or not set")

        compressed_data = self._storage[index]
        metadata = self._metadata[index]
        return self._decompress_item(compressed_data, metadata)

    def _get_slice(self, index: slice) -> list:
        """Get a slice from the compressed storage."""
        start, stop, step = index.indices(len(self._storage))
        results = []
        for i in range(start, stop, step):
            if i < len(self._storage) and self._storage[i] is not None:
                results.append(self._get_item(i))
        return results

    def _get_list(self, index: list) -> list:
        """Get a list of items from the compressed storage."""
        if isinstance(index, torch.Tensor) and index.device.type != "cpu":
            index = index.cpu().tolist()

        results = []
        for i in index:
            if i >= len(self._storage) or self._storage[i] is None:
                raise IndexError(f"Index {i} out of bounds or not set")
            results.append(self._get_item(i))
        return results

    def __len__(self) -> int:
        """Get the length of the compressed storage."""
        return len([item for item in self._storage if item is not None])

    def _contains_int(self, item: int) -> bool:
        """Check if an integer index is contained in the compressed storage."""
        return 0 <= item < len(self._storage) and self._storage[item] is not None

    def _empty(self):
        """Empty the storage."""
        self._storage = []
        self._metadata = []
        self._bump_mutation_revision()

    def state_dict(self) -> dict[str, Any]:
        """Save the storage state."""
        return {
            "_storage": self._storage,
            "_metadata": self._metadata,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the storage state."""
        # clone tensors and copy containers to decouple the storage from the
        # caller's objects
        self._storage = [
            elt.clone() if isinstance(elt, torch.Tensor) else elt
            for elt in state_dict["_storage"]
        ]
        self._metadata = deepcopy(state_dict["_metadata"])
        self._bump_mutation_revision()

    def to_bytestream(self, data_to_bytestream: torch.Tensor | np.array | Any) -> bytes:
        """Convert data to a byte stream."""
        if isinstance(data_to_bytestream, torch.Tensor):
            byte_stream = data_to_bytestream.cpu().numpy().tobytes()

        elif isinstance(data_to_bytestream, np.array):
            byte_stream = bytes(data_to_bytestream.tobytes())

        else:
            import io
            import pickle

            buffer = io.BytesIO()
            pickle.dump(data_to_bytestream, buffer)
            buffer.seek(0)
            byte_stream = bytes(buffer.read())

        return byte_stream

    def bytes(self):
        """Return the number of bytes in the storage."""

        def compressed_size_from_list(data: Any) -> int:
            if data is None:
                return 0
            elif isinstance(data, (bytes,)):
                return len(data)
            elif isinstance(data, (np.ndarray,)):
                return data.nbytes
            elif isinstance(data, (torch.Tensor)):
                return compressed_size_from_list(data.cpu().numpy())
            elif isinstance(data, (tuple, list, Sequence)):
                return sum(compressed_size_from_list(item) for item in data)
            elif isinstance(data, Mapping) or is_tensor_collection(data):
                return sum(compressed_size_from_list(value) for value in data.values())
            else:
                return 0

        compressed_size_estimate = compressed_size_from_list(self._storage)
        if compressed_size_estimate == 0:
            if len(self._storage) > 0:
                raise RuntimeError(
                    "Compressed storage is not empty but the compressed size is 0. This is a bug."
                )
            warnings.warn("Compressed storage is empty, returning 0 bytes.")

        return compressed_size_estimate
