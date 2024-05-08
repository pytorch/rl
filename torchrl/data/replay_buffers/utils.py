# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tree
from __future__ import annotations

import math
import os
import typing
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
from tensordict import MemoryMappedTensor, NonTensorData, TensorDict, TensorDictBase

from torch import Tensor
from torch.utils._pytree import LeafSpec, tree_flatten, tree_unflatten

from torchrl._utils import implement_for, logger as torchrl_logger

SINGLE_TENSOR_BUFFER_NAME = os.environ.get(
    "SINGLE_TENSOR_BUFFER_NAME", "_-single-tensor-_"
)


INT_CLASSES_TYPING = Union[int, np.integer]
if hasattr(typing, "get_args"):
    INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)
else:
    # python 3.7
    INT_CLASSES = (int, np.integer)


def _to_numpy(data: Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data


def _to_torch(
    data: Tensor, device, pin_memory: bool = False, non_blocking: bool = False
) -> torch.Tensor:
    if isinstance(data, np.generic):
        return torch.as_tensor(data, device=device)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif not isinstance(data, Tensor):
        data = torch.as_tensor(data, device=device)

    if pin_memory:
        data = data.pin_memory()
    if device is not None:
        data = data.to(device, non_blocking=non_blocking)

    return data


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


def _pin_memory(output: Any) -> Any:
    if hasattr(output, "pin_memory") and output.device == torch.device("cpu"):
        return output.pin_memory()
    else:
        return output


def _reduce(
    tensor: torch.Tensor, reduction: str, dim: int | None = None
) -> Union[float, torch.Tensor]:
    """Reduces a tensor given the reduction method."""
    if reduction == "max":
        result = tensor.max(dim=dim)
    elif reduction == "min":
        result = tensor.min(dim=dim)
    elif reduction == "mean":
        result = tensor.mean(dim=dim)
    elif reduction == "median":
        result = tensor.median(dim=dim)
    elif reduction == "sum":
        result = tensor.sum(dim=dim)
    else:
        raise NotImplementedError(f"Unknown reduction method {reduction}")
    if isinstance(result, tuple):
        result = result[0]
    return result.item() if dim is None else result


def _is_int(index):
    if isinstance(index, INT_CLASSES):
        return True
    if isinstance(index, (np.ndarray, torch.Tensor)):
        return index.ndim == 0
    return False


class TED2Flat:
    """A storage saving hook to serialize TED data in a compact format.

    Examples:
        >>> import tempfile
        >>>
        >>> from tensordict import TensorDict
        >>>
        >>> from torchrl.collectors import SyncDataCollector
        >>> from torchrl.data import ReplayBuffer, TED2Flat, LazyMemmapStorage
        >>> from torchrl.envs import GymEnv
        >>> import torch
        >>>
        >>> env = GymEnv("CartPole-v1")
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> collector = SyncDataCollector(env, policy=env.rand_step, total_frames=200, frames_per_batch=200)
        >>> rb = ReplayBuffer(storage=LazyMemmapStorage(200))
        >>> rb.register_save_hook(TED2Flat())
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for i, data in enumerate(collector):
        ...         rb.extend(data)
        ...         rb.dumps(tmpdir)
        ...     # load the data to represent it
        ...     td = TensorDict.load(tmpdir + "/storage/")
        ...     print(td)
        TensorDict(
            fields={
                action: MemoryMappedTensor(shape=torch.Size([200, 2]), device=cpu, dtype=torch.int64, is_shared=True),
                collector: TensorDict(
                    fields={
                        traj_ids: MemoryMappedTensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=True)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                done: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=True),
                observation: MemoryMappedTensor(shape=torch.Size([220, 4]), device=cpu, dtype=torch.float32, is_shared=True),
                reward: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=True),
                terminated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=True),
                truncated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=True)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

    """

    _shift: int = None
    _is_full: bool = None

    def __init__(
        self, done_key=("next", "done"), shift_key="shift", is_full_key="is_full"
    ):
        self.done_key = done_key
        self.shift_key = shift_key
        self.is_full_key = is_full_key

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value: int):
        self._shift = value

    @property
    def is_full(self):
        return self._is_full

    @is_full.setter
    def is_full(self, value: int):
        self._is_full = value

    def __call__(self, data: TensorDictBase):
        # Get the done state
        shift = self.shift
        is_full = self.is_full
        done = data.get(self.done_key)
        done = done.squeeze(-1).clone()
        if not is_full:
            # shift is the cursor place
            done[..., shift - 1] = True
        else:
            done = done.roll(-shift, dims=0)
            done[..., -1] = True

        # capture for each item in data where the observation should be written
        idx = torch.arange(data.shape[0])
        idx_done = (idx + done.cumsum(0))[done]
        idx += torch.nn.functional.pad(done, [1, 0])[:-1].cumsum(0)

        # Get the keys that require extra storage
        keys_to_expand = set(data.get("next").keys(True, True)) - {
            "terminated",
            "done",
            "truncated",
            "reward",
        }

        # Create an output storage
        output = TensorDict({}, [])
        total_keys = data.exclude("next").keys(True, True)
        total_keys = set(total_keys).union(set(data.get("next").keys(True, True)))
        for key in total_keys:
            if key in ("done", "truncated", "terminated", "reward"):
                entry = data.get(("next", key))
            else:
                entry = data.get(key)
            if is_full:
                entry = entry.roll(-shift, dims=0)

            if key in keys_to_expand:
                shape = torch.Size([idx.max() + 2, *entry.shape[1:]])
                dtype = entry.dtype
                empty = MemoryMappedTensor.zeros(shape=shape, dtype=dtype)
                empty[idx] = entry
                shifted_next = data.get(("next", key))
                if is_full:
                    shifted_next = shifted_next.roll(-shift, dims=0)
                empty[idx_done] = shifted_next[done]
                entry = empty
            output.set(key, entry)
        output.set_non_tensor(self.is_full_key, is_full)
        output.set_non_tensor(self.shift_key, shift)
        return output


class Flat2TED:
    """A storage loading hook to deserialize flattened TED data to TED format.

    Examples:
        >>> import tempfile
        >>>
        >>> from tensordict import TensorDict
        >>>
        >>> from torchrl.collectors import SyncDataCollector
        >>> from torchrl.data import ReplayBuffer, TED2Flat, LazyMemmapStorage, Flat2TED
        >>> from torchrl.envs import GymEnv
        >>> import torch
        >>>
        >>> env = GymEnv("CartPole-v1")
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> collector = SyncDataCollector(env, policy=env.rand_step, total_frames=200, frames_per_batch=200)
        >>> rb = ReplayBuffer(storage=LazyMemmapStorage(200))
        >>> rb.register_save_hook(TED2Flat())
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for i, data in enumerate(collector):
        ...         rb.extend(data)
        ...         rb.dumps(tmpdir)
        ...     # load the data to represent it
        ...     td = TensorDict.load(tmpdir + "/storage/")
        ...
        ...     rb_load = ReplayBuffer(storage=LazyMemmapStorage(200))
        ...     rb_load.register_load_hook(Flat2TED())
        ...     rb_load.load(tmpdir)
        ...     print("storage after loading", rb_load[:])
        ...     assert (rb[:] == rb_load[:]).all()
        storage after loading TensorDict(
            fields={
                action: MemoryMappedTensor(shape=torch.Size([200, 2]), device=cpu, dtype=torch.int64, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: MemoryMappedTensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: MemoryMappedTensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: MemoryMappedTensor(shape=torch.Size([200, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: MemoryMappedTensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)


    """

    def __init__(self, done_key="done", shift_key="shift", is_full_key="is_full"):
        self.done_key = done_key
        self.shift_key = shift_key
        self.is_full_key = is_full_key

    def __call__(self, data):
        done = data.get(self.done_key)
        done = done.clone()
        shift = data.get_non_tensor(self.shift_key, default=None)
        is_full = data.get_non_tensor(self.is_full_key, default=None)

        nsteps = done.shape[0]

        # capture for each item in data where the observation should be written
        idx = torch.arange(done.shape[0])
        root_idx = idx + torch.nn.functional.pad(done.squeeze(-1), [1, 0])[:-1].cumsum(
            0
        )
        next_idx = root_idx + 1

        out = TensorDict({}, [nsteps])

        def maybe_roll(entry):
            if is_full and shift is not None:
                return entry.roll(shift, dims=0)
            return entry

        root_idx = maybe_roll(root_idx)
        next_idx = maybe_roll(next_idx)

        for key, entry in data.items(True, True):
            if entry.shape[0] == nsteps:
                if key in ("done", "terminated", "truncated", "reward"):
                    out["next", key] = maybe_roll(entry)
                    if key != "reward":
                        out[key] = torch.zeros_like(entry)
                else:
                    # action and similar
                    out[key] = maybe_roll(entry)
            else:
                root_entry = entry[root_idx]
                next_entry = entry[next_idx]
                out["next", key] = next_entry
                out[key] = root_entry
        return out


class TED2Nested(TED2Flat):
    """Converts a TED-formatted dataset into a tensordict populated with nested tensors where each row is a trajectory."""

    _shift: int = None
    _is_full: bool = None

    def __call__(self, data: TensorDictBase):
        data = super().__call__(data)

        shift = self.shift
        is_full = self.is_full

        done = data.get("done")
        done = done.squeeze(-1).clone()
        if not is_full:
            done[..., shift - 1] = True
        else:
            done[..., -1] = True

        ntraj = done.sum()

        nz = done.nonzero()[:, 0]
        traj_lengths = torch.cat([nz[:1] + 1, nz.diff()])
        if not is_full:
            traj_lengths = torch.cat(
                [traj_lengths, (done.shape[0] - traj_lengths.sum()).unsqueeze(0)]
            )

        keys_to_expand, keys_to_keep = zip(
            *[
                (key, None) if val.shape[0] != done.shape[0] else (None, key)
                for key, val in data.items(True, True)
            ]
        )
        keys_to_expand = [key for key in keys_to_expand if key is not None]
        keys_to_keep = [key for key in keys_to_keep if key is not None]

        out = TensorDict({}, batch_size=[ntraj + (not is_full)])
        for key in keys_to_expand:
            val = data.get(key)
            shape = torch.cat(
                [
                    traj_lengths.unsqueeze(-1) + 1,
                    torch.tensor(val.shape[1:], dtype=torch.long).repeat(
                        traj_lengths.numel(), 1
                    ),
                ],
                -1,
            )
            val = MemoryMappedTensor.from_storage(
                val.untyped_storage(), dtype=val.dtype, shape=shape
            )
            out[key] = val
            # non_nt_shape = torch.Size([idx.max() + 2, *entry.shape[1:]])
        for key in keys_to_keep:
            val = data.get(key)
            shape = torch.cat(
                [
                    traj_lengths.unsqueeze(-1),
                    torch.tensor(val.shape[1:], dtype=torch.long).repeat(
                        traj_lengths.numel(), 1
                    ),
                ],
                -1,
            )
            val = MemoryMappedTensor.from_storage(
                val.untyped_storage(), dtype=val.dtype, shape=shape
            )
            # non_nt_shape = val.shape
            out[key] = val

        out.set_non_tensor(
            self.shift_key, data.get_non_tensor(self.shift_key, default=None)
        )
        out.set_non_tensor(
            self.is_full_key, data.get_non_tensor(self.is_full_key, default=None)
        )
        return out


class Nested2TED(Flat2TED):
    """Converts a nested tensordict where each row is a trajectory into the TED format."""

    def __call__(self, data):
        # Get a flat representation of data
        def flatten_het_dim(tensor):
            shape = [tensor.size(i) for i in range(2, tensor.ndim)]
            tensor = torch.tensor(tensor.untyped_storage(), dtype=tensor.dtype).view(
                -1, *shape
            )
            return tensor

        data = data.apply(flatten_het_dim, batch_size=[])
        data.auto_batch_size_()
        return super().__call__(data)


class H5Split(TED2Flat):
    """Splits a dataset prepared with TED2Nested into a TensorDict where each trajectory is stored as views on their parent nested tensors."""

    _shift: int = None
    _is_full: bool = None

    def __call__(self, data):
        nzeros = int(math.ceil(math.log10(data.shape[0])))

        is_full = data.pop(self.is_full_key)
        shift = data.pop(self.shift_key)

        return TensorDict(
            {
                f"traj_{str(i).zfill(nzeros)}": _data
                for i, _data in enumerate(data.unbind(0))
            }
        ).update({"is_full": is_full, "shift": shift})


class H5Combine:
    """Combines trajectories in a persistent tensordict into a single standing tensordict stored in filesystem."""

    def __call__(self, data):
        values = [val for key, val in data.items() if key.startswith("traj")]
        metadata_keys = [key for key in data.keys() if not key.startswith("traj")]
        result = TensorDict(
            {key: NonTensorData(data.get(key).item()) for key in metadata_keys}
        )
        result = result.update(
            values[0].apply(
                lambda *x: torch.nested.nested_tensor(list(x)),
                *values[1:],
                batch_size=[],
            )
        )
        return result


@implement_for("torch", "2.3", None)
def _path2str(path, default_name=None):
    # Uses the Keys defined in pytree to build a path
    from torch.utils._pytree import MappingKey, SequenceKey

    if default_name is None:
        default_name = SINGLE_TENSOR_BUFFER_NAME
    if not path:
        return default_name
    if isinstance(path, tuple):
        return "/".join([_path2str(_sub, default_name=default_name) for _sub in path])
    if isinstance(path, MappingKey):
        if not isinstance(path.key, (int, str, bytes)):
            raise ValueError("Values must be of type int, str or bytes in PyTree maps.")
        result = str(path.key)
        if result == default_name:
            raise RuntimeError(
                "A tensor had the same identifier as the default name used when the buffer contains "
                f"a single tensor (name={default_name}). This behaviour is not allowed. Please rename your "
                f"tensor in the map/dict or set a new default name with the environment variable SINGLE_TENSOR_BUFFER_NAME."
            )
        return result
    if isinstance(path, SequenceKey):
        return str(path.idx)


@implement_for("torch", None, "2.3")
def _path2str(path, default_name=None):  # noqa: F811
    raise RuntimeError


def _save_pytree_common(tensor_path, path, tensor, metadata):
    if "." in tensor_path:
        tensor_path.replace(".", "_<dot>_")
    total_tensor_path = path / (tensor_path + ".memmap")
    if os.path.exists(total_tensor_path):
        MemoryMappedTensor.from_filename(
            shape=tensor.shape,
            filename=total_tensor_path,
            dtype=tensor.dtype,
        ).copy_(tensor)
    else:
        os.makedirs(total_tensor_path.parent, exist_ok=True)
        MemoryMappedTensor.from_tensor(
            tensor,
            filename=total_tensor_path,
            copy_existing=True,
            copy_data=True,
        )
    key = tensor_path.replace("/", ".")
    if key in metadata:
        raise KeyError(
            "At least two values have conflicting representations in "
            f"the data structure to be serialized: {key}."
        )
    metadata[key] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


@implement_for("torch", "2.3", None)
def _save_pytree(_storage, metadata, path):
    from torch.utils._pytree import tree_map_with_path

    def save_tensor(
        tensor_path: tuple, tensor: torch.Tensor, metadata=metadata, path=path
    ):
        tensor_path = _path2str(tensor_path)
        _save_pytree_common(tensor_path, path, tensor, metadata)

    tree_map_with_path(save_tensor, _storage)


@implement_for("torch", None, "2.3")
def _save_pytree(_storage, metadata, path):  # noqa: F811

    flat_storage, storage_specs = tree_flatten(_storage)
    storage_paths = _get_paths(storage_specs)

    def save_tensor(
        tensor_path: str, tensor: torch.Tensor, metadata=metadata, path=path
    ):
        _save_pytree_common(tensor_path, path, tensor, metadata)

    for tensor, tensor_path in zip(flat_storage, storage_paths):
        save_tensor(tensor_path, tensor)


def _get_paths(spec, cumulpath=""):
    # alternative way to build a path without the keys
    if isinstance(spec, LeafSpec):
        yield cumulpath if cumulpath else SINGLE_TENSOR_BUFFER_NAME

    contexts = spec.context
    children_specs = spec.children_specs
    if contexts is None:
        contexts = range(len(children_specs))

    for context, spec in zip(contexts, children_specs):
        cpath = "/".join((cumulpath, str(context))) if cumulpath else str(context)
        yield from _get_paths(spec, cpath)


def _init_pytree_common(tensor_path, scratch_dir, max_size_fn, tensor):
    if "." in tensor_path:
        tensor_path.replace(".", "_<dot>_")
    if scratch_dir is not None:
        total_tensor_path = Path(scratch_dir) / (tensor_path + ".memmap")
        if os.path.exists(total_tensor_path):
            raise RuntimeError(
                f"The storage of tensor {total_tensor_path} already exists. "
                f"To load an existing replay buffer, use storage.loads. "
                f"Choose a different path to store your buffer or delete the existing files."
            )
        os.makedirs(total_tensor_path.parent, exist_ok=True)
    else:
        total_tensor_path = None
    out = MemoryMappedTensor.empty(
        shape=max_size_fn(tensor.shape),
        filename=total_tensor_path,
        dtype=tensor.dtype,
    )
    try:
        filesize = os.path.getsize(tensor.filename) / 1024 / 1024
        torchrl_logger.debug(
            f"The storage was created in {out.filename} and occupies {filesize} Mb of storage."
        )
    except (RuntimeError, AttributeError):
        pass
    return out


@implement_for("torch", "2.3", None)
def _init_pytree(scratch_dir, max_size_fn, data):
    from torch.utils._pytree import tree_map_with_path

    # If not a tensorclass/tensordict, it must be a tensor(-like) or a PyTree
    # if Tensor, we just create a MemoryMappedTensor of the desired shape, device and dtype
    def save_tensor(tensor_path: tuple, tensor: torch.Tensor):
        tensor_path = _path2str(tensor_path)
        return _init_pytree_common(tensor_path, scratch_dir, max_size_fn, tensor)

    out = tree_map_with_path(save_tensor, data)
    return out


@implement_for("torch", None, "2.3")
def _init_pytree(scratch_dir, max_size, data):  # noqa: F811

    flat_data, data_specs = tree_flatten(data)
    data_paths = _get_paths(data_specs)
    data_paths = list(data_paths)

    # If not a tensorclass/tensordict, it must be a tensor(-like) or a PyTree
    # if Tensor, we just create a MemoryMappedTensor of the desired shape, device and dtype
    def save_tensor(tensor_path: str, tensor: torch.Tensor):
        return _init_pytree_common(tensor_path, scratch_dir, max_size, tensor)

    out = []
    for tensor, tensor_path in zip(flat_data, data_paths):
        out.append(save_tensor(tensor_path, tensor))

    return tree_unflatten(out, data_specs)
