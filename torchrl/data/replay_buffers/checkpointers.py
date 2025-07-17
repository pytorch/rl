# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from tensordict import (
    is_tensor_collection,
    lazy_stack,
    NonTensorData,
    PersistentTensorDict,
    TensorDict,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import _zip_strict
from torch.utils._pytree import tree_map
from torchrl._utils import _STRDTYPE2DTYPE

from torchrl.data.replay_buffers.utils import (
    _save_pytree,
    Flat2TED,
    H5Combine,
    H5Split,
    Nested2TED,
    TED2Flat,
    TED2Nested,
)


class StorageCheckpointerBase:
    """Public base class for storage checkpointers.

    Each storage checkpointer must implement a `save` and `load` method that take as input a storage and a
    path.

    """

    @abc.abstractmethod
    def dumps(self, storage, path):
        ...

    @abc.abstractmethod
    def loads(self, storage, path):
        ...


class ListStorageCheckpointer(StorageCheckpointerBase):
    """A storage checkpointer for ListStoage.

    Currently not implemented.

    """

    @staticmethod
    def dumps(storage, path):
        raise NotImplementedError(
            "ListStorage doesn't support serialization via `dumps` - `loads` API."
        )

    @staticmethod
    def loads(storage, path):
        raise NotImplementedError(
            "ListStorage doesn't support serialization via `dumps` - `loads` API."
        )


class CompressedListStorageCheckpointer(StorageCheckpointerBase):
    """A storage checkpointer for CompressedListStorage.

    This checkpointer saves compressed data and metadata using memory-mapped storage
    for efficient disk I/O and memory usage.

    """

    def dumps(self, storage, path):
        """Save compressed storage to disk using memory-mapped storage.

        Args:
            storage: The CompressedListStorage instance to save
            path: Directory path where to save the storage
        """
        path = Path(path)
        path.mkdir(exist_ok=True)

        if not hasattr(storage, "_storage") or len(storage._storage) == 0:
            raise RuntimeError(
                "Cannot save an empty or non-initialized CompressedListStorage."
            )

        # Get state dict from storage
        state_dict = storage.state_dict()
        compressed_data = state_dict["_storage"]
        metadata = state_dict["_metadata"]

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Process compressed data for memmap storage
            processed_data = []
            for item in compressed_data:
                if item is None:
                    processed_data.append(None)
                    continue

                if isinstance(item, torch.Tensor):
                    # For tensor data, create a TensorDict with the tensor
                    processed_item = TensorDict({"data": item}, batch_size=[])
                elif isinstance(item, dict):
                    # For dict data (tensordict fields), convert to TensorDict
                    processed_item = TensorDict(item, batch_size=[])
                else:
                    # For other types, wrap in TensorDict
                    processed_item = TensorDict({"data": item}, batch_size=[])

                processed_data.append(processed_item)

            # Stack all non-None items into a single TensorDict for memmap
            non_none_data = [item for item in processed_data if item is not None]
            if non_none_data:
                # Use lazy_stack to handle heterogeneous structures
                stacked_data = lazy_stack(non_none_data)

                # Save to memmap
                stacked_data.memmap_(tmp_path / "compressed_data")

                # Create index mapping for None values
                data_indices = []
                current_idx = 0
                for item in processed_data:
                    if item is None:
                        data_indices.append(None)
                    else:
                        data_indices.append(current_idx)
                        current_idx += 1
            else:
                # No data to save
                data_indices = []

            # Process metadata for JSON serialization
            def is_leaf(item):
                return isinstance(
                    item,
                    (
                        torch.Size,
                        torch.dtype,
                        torch.device,
                        str,
                        int,
                        float,
                        bool,
                        torch.Tensor,
                        NonTensorData,
                    ),
                )

            def map_to_json_serializable(item):
                if isinstance(item, torch.Size):
                    return {"__type__": "torch.Size", "value": list(item)}
                elif isinstance(item, torch.dtype):
                    return {"__type__": "torch.dtype", "value": str(item)}
                elif isinstance(item, torch.device):
                    return {"__type__": "torch.device", "value": str(item)}
                elif isinstance(item, torch.Tensor):
                    return {"__type__": "torch.Tensor", "value": item.tolist()}
                elif isinstance(item, NonTensorData):
                    return {"__type__": "NonTensorData", "value": item.data}
                return item

            serializable_metadata = tree_map(
                map_to_json_serializable, metadata, is_leaf=is_leaf
            )

            # Save metadata and indices
            metadata_file = tmp_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(serializable_metadata, f, indent=2)

            indices_file = tmp_path / "data_indices.json"
            with open(indices_file, "w") as f:
                json.dump(data_indices, f, indent=2)

            # Copy all files from temp directory to final destination
            import shutil

            for item in tmp_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, path / item.name, dirs_exist_ok=True)

    def loads(self, storage, path):
        """Load compressed storage from disk.

        Args:
            storage: The CompressedListStorage instance to load into
            path: Directory path where the storage was saved
        """
        path = Path(path)

        # Load metadata
        metadata_file = path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

        with open(metadata_file) as f:
            serializable_metadata = json.load(f)

        # Load data indices
        indices_file = path / "data_indices.json"
        if not indices_file.exists():
            raise FileNotFoundError(f"Data indices file not found at {indices_file}")

        with open(indices_file) as f:
            data_indices = json.load(f)

        # Convert serializable metadata back to original format
        def is_leaf(item):
            return isinstance(item, dict) and "__type__" in item

        def map_from_json_serializable(item):
            if isinstance(item, dict) and "__type__" in item:
                if item["__type__"] == "torch.Size":
                    return torch.Size(item["value"])
                elif item["__type__"] == "torch.dtype":
                    # Handle torch.dtype conversion
                    dtype_str = item["value"]
                    if hasattr(torch, dtype_str.replace("torch.", "")):
                        return getattr(torch, dtype_str.replace("torch.", ""))
                    else:
                        # Handle cases like 'torch.float32' -> torch.float32
                        return eval(dtype_str)
                elif item["__type__"] == "torch.device":
                    return torch.device(item["value"])
                elif item["__type__"] == "torch.Tensor":
                    return torch.tensor(item["value"])
                elif item["__type__"] == "NonTensorData":
                    return NonTensorData(item["value"])
            return item

        metadata = tree_map(
            map_from_json_serializable, serializable_metadata, is_leaf=is_leaf
        )

        # Load compressed data from memmap
        compressed_data = []
        memmap_path = path / "compressed_data"

        if memmap_path.exists():
            # Load the memmapped data
            stacked_data = TensorDict.load_memmap(memmap_path)
            compressed_data = stacked_data.tolist()
            if len(compressed_data) != len(data_indices):
                raise ValueError(
                    f"Length of compressed data ({len(compressed_data)}) does not match length of data indices ({len(data_indices)})"
                )
            for i, (data, mtdt) in enumerate(_zip_strict(compressed_data, metadata)):
                if mtdt["type"] == "tensor":
                    compressed_data[i] = data["data"]
                else:
                    compressed_data[i] = data

        else:
            # No data to load
            compressed_data = [None] * len(data_indices)

        # Load into storage
        storage._storage = compressed_data
        storage._metadata = metadata


class TensorStorageCheckpointer(StorageCheckpointerBase):
    """A storage checkpointer for TensorStorages.

    This class supports TensorDict-based storages as well as pytrees.

    This class will call save and load hooks if provided. These hooks should take as input the
    data being transformed as well as the path where the data should be saved.

    """

    _save_hooks = []
    _load_hooks = []

    def dumps(self, storage, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        if not storage.initialized:
            raise RuntimeError("Cannot save a non-initialized storage.")
        metadata = {}
        _storage = storage._storage
        for hook in self._save_hooks:
            _storage = hook(_storage, path=path)
        if is_tensor_collection(_storage):
            if (
                _storage.is_memmap()
                and _storage.saved_path
                and Path(_storage.saved_path).absolute() == Path(path).absolute()
            ):
                _storage.memmap_refresh_()
            else:
                # try to load the path and overwrite.
                _storage.memmap(
                    path,
                    copy_existing=True,  # num_threads=torch.get_num_threads()
                )
            is_pytree = False
        else:
            _save_pytree(_storage, metadata, path)
            is_pytree = True

        with open(path / "storage_metadata.json", "w") as file:
            json.dump(
                {
                    "metadata": metadata,
                    "is_pytree": is_pytree,
                    "len": storage._len,
                },
                file,
            )

    def loads(self, storage, path):
        with open(path / "storage_metadata.json") as file:
            metadata = json.load(file)
        is_pytree = metadata["is_pytree"]
        _len = metadata["len"]
        if is_pytree:
            if self._load_hooks:
                raise RuntimeError(
                    "Loading hooks are not compatible with PyTree storages."
                )
            path = Path(path)
            for local_path, md in metadata["metadata"].items():
                # load tensor
                local_path_dot = local_path.replace(".", "/")
                total_tensor_path = path / (local_path_dot + ".memmap")
                shape = torch.Size(md["shape"])
                dtype = _STRDTYPE2DTYPE[md["dtype"]]
                tensor = MemoryMappedTensor.from_filename(
                    filename=total_tensor_path, shape=shape, dtype=dtype
                )
                # split path
                local_path = local_path.split(".")
                # replace potential dots
                local_path = [_path.replace("_<dot>_", ".") for _path in local_path]
                if storage.initialized:
                    # copy in-place
                    _storage_tensor = storage._storage
                    # in this case there is a single tensor, so we skip
                    if local_path != ["_-single-tensor-_"]:
                        for _path in local_path:
                            if _path.isdigit():
                                _path_attempt = int(_path)
                                try:
                                    _storage_tensor = _storage_tensor[_path_attempt]
                                    continue
                                except IndexError:
                                    pass
                            _storage_tensor = _storage_tensor[_path]
                    _storage_tensor.copy_(tensor)
                else:
                    raise RuntimeError(
                        "Cannot fill a non-initialized pytree-based TensorStorage."
                    )
        else:
            _storage = TensorDict.load_memmap(path)
            if storage.initialized:
                dest = storage._storage
            else:
                # TODO: This could load the RAM a lot, maybe try to catch this within the hook and use memmap instead
                dest = None
            for hook in self._load_hooks:
                _storage = hook(_storage, out=dest)
            if not storage.initialized:
                from torchrl.data.replay_buffers.storages import LazyMemmapStorage

                if (
                    isinstance(storage, LazyMemmapStorage)
                    and storage.scratch_dir
                    and Path(storage.scratch_dir).absolute() == Path(path).absolute()
                ):
                    storage._storage = TensorDict.load_memmap(path)
                    storage.initialized = True
                else:
                    # this should not be reached if is_pytree=True
                    storage._init(_storage[0])
                    storage._storage.update_(_storage)
            elif (
                storage._storage.is_memmap()
                and storage._storage.saved_path
                and Path(storage._storage.saved_path).absolute()
                == Path(path).absolute()
            ):
                # If the storage is already where it should be, we don't need to load anything.
                storage._storage.memmap_refresh_()

            else:
                storage._storage.copy_(_storage)
        storage._len = _len


class FlatStorageCheckpointer(TensorStorageCheckpointer):
    """Saves the storage in a compact form, saving space on the TED format.

    This class explicitly assumes and does NOT check that:

      - done states (including terminated and truncated) at the root are always False;
      - observations in the "next" tensordict are shifted by one step in the future (this
        is not the case when a multi-step transform is used for instance) unless `done` is `True`
        in which case the observation in `("next", key)` at time `t` and the one in `key` at time
        `t+1` should not match.

    .. seealso: The full list of arguments can be found in :class:`~torchrl.data.TED2Flat`.

    """

    def __init__(self, done_keys=None, reward_keys=None):
        kwargs = {}
        if done_keys is not None:
            kwargs["done_keys"] = done_keys
        if reward_keys is not None:
            kwargs["reward_keys"] = reward_keys
        self._save_hooks = [TED2Flat(**kwargs)]
        self._load_hooks = [Flat2TED(**kwargs)]

    def _save_shift_is_full(self, storage):
        is_full = storage._is_full
        last_cursor = storage._last_cursor
        for hook in self._save_hooks:
            if hasattr(hook, "is_full"):
                hook.is_full = is_full
        if last_cursor is None:
            warnings.warn(
                "las_cursor is None. The replay buffer "
                "may not be saved properly in this setting. To solve this issue, make "
                "sure the storage updates the _las_cursor value during calls to `set`."
            )
        shift = self._get_shift_from_last_cursor(last_cursor)
        for hook in self._save_hooks:
            if hasattr(hook, "shift"):
                hook.shift = shift

    def dumps(self, storage, path):
        self._save_shift_is_full(storage)
        return super().dumps(storage, path)

    def _get_shift_from_last_cursor(self, last_cursor):
        if isinstance(last_cursor, slice):
            return last_cursor.stop + 1
        if isinstance(last_cursor, int):
            return last_cursor + 1
        if isinstance(last_cursor, torch.Tensor):
            return last_cursor.reshape(-1)[-1].item() + 1
        if isinstance(last_cursor, np.ndarray):
            return last_cursor.reshape(-1)[-1].item() + 1
        raise ValueError(f"Unrecognised last_cursor type {type(last_cursor)}.")


class NestedStorageCheckpointer(FlatStorageCheckpointer):
    """Saves the storage in a compact form, saving space on the TED format and using memory-mapped nested tensors.

    This class explicitly assumes and does NOT check that:

      - done states (including terminated and truncated) at the root are always False;
      - observations in the "next" tensordict are shifted by one step in the future (this
        is not the case when a multi-step transform is used for instance).

    .. seealso: The full list of arguments can be found in :class:`~torchrl.data.TED2Flat`.

    """

    def __init__(self, done_keys=None, reward_keys=None, **kwargs):
        kwargs = {}
        if done_keys is not None:
            kwargs["done_keys"] = done_keys
        if reward_keys is not None:
            kwargs["reward_keys"] = reward_keys
        self._save_hooks = [TED2Nested(**kwargs)]
        self._load_hooks = [Nested2TED(**kwargs)]


class H5StorageCheckpointer(NestedStorageCheckpointer):
    """Saves the storage in a compact form, saving space on the TED format and using H5 format to save the data.

    This class explicitly assumes and does NOT check that:

      - done states (including terminated and truncated) at the root are always False;
      - observations in the "next" tensordict are shifted by one step in the future (this
        is not the case when a multi-step transform is used for instance).

    Keyword Args:
        checkpoint_file: the filename where to save the checkpointed data.
            This will be ignored iff the path passed to dumps / loads ends with the ``.h5``
            suffix. Defaults to ``"checkpoint.h5"``.
        h5_kwargs (Dict[str, Any] or Tuple[Tuple[str, Any], ...]): kwargs to be
            passed to :meth:`h5py.File.create_dataset`.

    .. note:: To prevent out-of-memory issues, the data of the H5 file will be temporarily written
        on memory-mapped tensors stored in shared file system. The physical memory usage may increase
        during loading as a consequence.

    .. seealso: The full list of arguments can be found in :class:`~torchrl.data.TED2Flat`. Note that this class only
        supports keyword arguments.

    """

    def __init__(
        self,
        *,
        checkpoint_file: str = "checkpoint.h5",
        done_keys=None,
        reward_keys=None,
        h5_kwargs=None,
        **kwargs,
    ):
        ted2_kwargs = kwargs
        if done_keys is not None:
            ted2_kwargs["done_keys"] = done_keys
        if reward_keys is not None:
            ted2_kwargs["reward_keys"] = reward_keys
        self._save_hooks = [TED2Nested(**ted2_kwargs), H5Split()]
        self._load_hooks = [H5Combine(), Nested2TED(**ted2_kwargs)]
        self.kwargs = {} if h5_kwargs is None else dict(h5_kwargs)
        self.checkpoint_file = checkpoint_file

    def dumps(self, storage, path):
        path = self._get_path(path)

        self._save_shift_is_full(storage)

        if not storage.initialized:
            raise RuntimeError("Cannot save a non-initialized storage.")
        _storage = storage._storage
        length = storage._len
        for hook in self._save_hooks:
            # we don't pass a path here since we're not reusing the tensordict
            _storage = hook(_storage)
        if is_tensor_collection(_storage):
            # try to load the path and overwrite.
            data = PersistentTensorDict.from_dict(_storage, path, **self.kwargs)
            data["_len"] = NonTensorData(data=length)
        else:
            raise ValueError("Only tensor collections are supported.")

    def loads(self, storage, path):
        path = self._get_path(path)
        data = PersistentTensorDict.from_h5(path)
        if storage.initialized:
            dest = storage._storage
        else:
            # TODO: This could load the RAM a lot, maybe try to catch this within the hook and use memmap instead
            dest = None
        _len = data["_len"]
        for hook in self._load_hooks:
            data = hook(data, out=dest)
        if not storage.initialized:
            # this should not be reached if is_pytree=True
            storage._init(data[0])
            storage._storage.update_(data)
        else:
            storage._storage.copy_(data)
        storage._len = _len

    def _get_path(self, path):
        path = Path(path)
        if path.suffix == ".h5":
            return str(path.absolute())
        try:
            path.mkdir(exist_ok=True)
        except Exception:
            raise RuntimeError(f"Failed to create the checkpoint directory {path}.")
        path = path / self.checkpoint_file
        return str(path.absolute())


class StorageEnsembleCheckpointer(StorageCheckpointerBase):
    """Checkpointer for ensemble storages."""

    @staticmethod
    def dumps(storage, path: Path):
        path = Path(path).absolute()
        storages = storage._storages
        for i, storage in enumerate(storages):
            storage.dumps(path / str(i))
        if storage._transforms is not None:
            for i, transform in enumerate(storage._transforms):
                torch.save(transform.state_dict(), path / f"{i}_transform.pt")

    @staticmethod
    def loads(storage, path: Path):
        path = Path(path).absolute()
        for i, _storage in enumerate(storage._storages):
            _storage.loads(path / str(i))
        if storage._transforms is not None:
            for i, transform in enumerate(storage._transforms):
                transform.load_state_dict(torch.load(path / f"{i}_transform.pt"))
