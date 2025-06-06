# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tree
from __future__ import annotations

import contextlib
import itertools
import math
import operator
import os
import typing
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
from tensordict import (
    lazy_stack,
    MemoryMappedTensor,
    NonTensorData,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from torch import Tensor
from torch.nn import functional as F
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
) -> float | torch.Tensor:
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

    Args:
        done_key (NestedKey, optional): the key where the done states should be read.
            Defaults to ``("next", "done")``.
        shift_key (NestedKey, optional): the key where the shift will be written.
            Defaults to "shift".
        is_full_key (NestedKey, optional): the key where the is_full attribute will be written.
            Defaults to "is_full".
        done_keys (Tuple[NestedKey], optional): a tuple of nested keys indicating the done entries.
            Defaults to ("done", "truncated", "terminated")
        reward_keys (Tuple[NestedKey], optional): a tuple of nested keys indicating the reward entries.
            Defaults to ("reward",)


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

    _shift: int | None = None
    _is_full: bool | None = None

    def __init__(
        self,
        done_key=("next", "done"),
        shift_key="shift",
        is_full_key="is_full",
        done_keys=("done", "truncated", "terminated"),
        reward_keys=("reward",),
    ):
        self.done_key = done_key
        self.shift_key = shift_key
        self.is_full_key = is_full_key
        self.done_keys = {unravel_key(key) for key in done_keys}
        self.reward_keys = {unravel_key(key) for key in reward_keys}

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

    def __call__(self, data: TensorDictBase, path: Path = None):
        # Get the done state
        shift = self.shift
        is_full = self.is_full

        # Create an output storage
        output = TensorDict()
        output.set_non_tensor(self.is_full_key, is_full)
        output.set_non_tensor(self.shift_key, shift)
        output.set_non_tensor("_storage_shape", tuple(data.shape))
        output.memmap_(path)

        # Preallocate the output
        done = data.get(self.done_key).squeeze(-1).clone()
        if not is_full:
            # shift is the cursor place
            done[shift - 1] = True
        else:
            done = done.roll(-shift, dims=0)
            done[-1] = True
        ntraj = done.sum()

        # Get the keys that require extra storage
        keys_to_expand = set(data.get("next").keys(True, True)) - (
            self.done_keys.union(self.reward_keys)
        )

        total_keys = data.exclude("next").keys(True, True)
        total_keys = set(total_keys).union(set(data.get("next").keys(True, True)))

        len_with_offset = data.numel() + ntraj  # + done[0].numel()
        for key in total_keys:
            if key in (self.done_keys.union(self.reward_keys)):
                entry = data.get(("next", key))
            else:
                entry = data.get(key)

            if key in keys_to_expand:
                shape = torch.Size([len_with_offset, *entry.shape[data.ndim :]])
                dtype = entry.dtype
                output.make_memmap(key, shape=shape, dtype=dtype)
            else:
                shape = torch.Size([data.numel(), *entry.shape[data.ndim :]])
                output.make_memmap(key, shape=shape, dtype=entry.dtype)

        if data.ndim == 1:
            return self._call(
                data=data,
                output=output,
                is_full=is_full,
                shift=shift,
                done=done,
                total_keys=total_keys,
                keys_to_expand=keys_to_expand,
            )

        with data.flatten(1, -1) if data.ndim > 2 else contextlib.nullcontext(
            data
        ) as data_flat:
            if data.ndim > 2:
                done = done.flatten(1, -1)
            traj_per_dim = done.sum(0)
            nsteps = data_flat.shape[0]

            start = 0
            start_with_offset = start
            stop_with_offset = 0
            stop = 0
            for data_slice, done_slice, traj_for_dim in zip(
                data_flat.unbind(1), done.unbind(1), traj_per_dim
            ):
                stop_with_offset = stop_with_offset + nsteps + traj_for_dim
                cur_slice_offset = slice(start_with_offset, stop_with_offset)
                start_with_offset = stop_with_offset

                stop = stop + data.shape[0]
                cur_slice = slice(start, stop)
                start = stop

                def _index(
                    key,
                    val,
                    keys_to_expand=keys_to_expand,
                    cur_slice=cur_slice,
                    cur_slice_offset=cur_slice_offset,
                ):
                    if key in keys_to_expand:
                        return val[cur_slice_offset]
                    return val[cur_slice]

                out_slice = output.named_apply(_index, nested_keys=True)
                self._call(
                    data=data_slice,
                    output=out_slice,
                    is_full=is_full,
                    shift=shift,
                    done=done_slice,
                    total_keys=total_keys,
                    keys_to_expand=keys_to_expand,
                )
        return output

    def _call(self, *, data, output, is_full, shift, done, total_keys, keys_to_expand):
        # capture for each item in data where the observation should be written
        idx = torch.arange(data.shape[0])
        idx_done = (idx + done.cumsum(0))[done]
        idx += torch.nn.functional.pad(done, [1, 0])[:-1].cumsum(0)

        for key in total_keys:
            if key in (self.done_keys.union(self.reward_keys)):
                entry = data.get(("next", key))
            else:
                entry = data.get(key)

            if key in keys_to_expand:
                mmap = output.get(key)
                shifted_next = data.get(("next", key))
                if is_full:
                    _roll_inplace(entry, shift=-shift, out=mmap, index_dest=idx)
                    _roll_inplace(
                        shifted_next,
                        shift=-shift,
                        out=mmap,
                        index_dest=idx_done,
                        index_source=done,
                    )
                else:
                    mmap[idx] = entry
                    mmap[idx_done] = shifted_next[done]
            elif is_full:
                mmap = output.get(key)
                _roll_inplace(entry, shift=-shift, out=mmap)
            else:
                mmap = output.get(key)
                mmap.copy_(entry)
        return output


class Flat2TED:
    """A storage loading hook to deserialize flattened TED data to TED format.

    Args:
        done_key (NestedKey, optional): the key where the done states should be read.
            Defaults to ``("next", "done")``.
        shift_key (NestedKey, optional): the key where the shift will be written.
            Defaults to "shift".
        is_full_key (NestedKey, optional): the key where the is_full attribute will be written.
            Defaults to "is_full".
        done_keys (Tuple[NestedKey], optional): a tuple of nested keys indicating the done entries.
            Defaults to ("done", "truncated", "terminated")
        reward_keys (Tuple[NestedKey], optional): a tuple of nested keys indicating the reward entries.
            Defaults to ("reward",)

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

    def __init__(
        self,
        done_key="done",
        shift_key="shift",
        is_full_key="is_full",
        done_keys=("done", "truncated", "terminated"),
        reward_keys=("reward",),
    ):
        self.done_key = done_key
        self.shift_key = shift_key
        self.is_full_key = is_full_key
        self.done_keys = {unravel_key(key) for key in done_keys}
        self.reward_keys = {unravel_key(key) for key in reward_keys}

    def __call__(self, data: TensorDictBase, out: TensorDictBase = None):
        _storage_shape = data.get_non_tensor("_storage_shape", default=None)
        if isinstance(_storage_shape, int):
            _storage_shape = torch.Size([_storage_shape])
        shift = data.get_non_tensor(self.shift_key, default=None)
        is_full = data.get_non_tensor(self.is_full_key, default=None)
        done = (
            data.get("done")
            .reshape((*_storage_shape[1:], -1))
            .contiguous()
            .permute(-1, *range(0, len(_storage_shape) - 1))
            .clone()
        )
        if not is_full:
            # shift is the cursor place
            done[shift - 1] = True
        else:
            # done = done.roll(-shift, dims=0)
            done[-1] = True

        if _storage_shape is not None and len(_storage_shape) > 1:
            # iterate over data and allocate
            if out is None:
                # out = TensorDict(batch_size=_storage_shape)
                # for i in range(out.ndim):
                #     if i >= 2:
                #         # FLattening the lazy stack will make the data unavailable - we need to find a way to make this
                #         # possible.
                #         raise RuntimeError(
                #             "Checkpointing an uninitialized buffer with more than 2 dimensions is currently not supported. "
                #             "Please file an issue on GitHub to ask for this feature!"
                #         )
                #     out = LazyStackedTensorDict(*out.unbind(i), stack_dim=i)
                out = TensorDict(batch_size=_storage_shape)
                for i in range(1, out.ndim):
                    if i >= 2:
                        # FLattening the lazy stack will make the data unavailable - we need to find a way to make this
                        # possible.
                        raise RuntimeError(
                            "Checkpointing an uninitialized buffer with more than 2 dimensions is currently not supported. "
                            "Please file an issue on GitHub to ask for this feature!"
                        )
                    out_list = [
                        out._get_sub_tensordict((slice(None),) * i + (j,))
                        for j in range(out.shape[i])
                    ]
                    out = lazy_stack(out_list, i)

            # Create a function that reads slices of the input data
            with out.flatten(1, -1) if out.ndim > 2 else contextlib.nullcontext(
                out
            ) as out_flat:
                nsteps = done.shape[0]
                n_elt_batch = done.shape[1:].numel()
                traj_per_dim = done.sum(0)

                start = 0
                start_with_offset = start
                stop_with_offset = 0
                stop = 0

                for out_unbound, traj_for_dim in zip(out_flat.unbind(-1), traj_per_dim):
                    stop_with_offset = stop_with_offset + nsteps + traj_for_dim
                    cur_slice_offset = slice(start_with_offset, stop_with_offset)
                    start_with_offset = stop_with_offset

                    stop = stop + nsteps
                    cur_slice = slice(start, stop)
                    start = stop

                    def _index(
                        key,
                        val,
                        cur_slice=cur_slice,
                        nsteps=nsteps,
                        n_elt_batch=n_elt_batch,
                        cur_slice_offset=cur_slice_offset,
                    ):
                        if val.shape[0] != (nsteps * n_elt_batch):
                            return val[cur_slice_offset]
                        return val[cur_slice]

                    data_slice = data.named_apply(
                        _index, nested_keys=True, batch_size=[]
                    )
                    self._call(
                        data=data_slice,
                        out=out_unbound,
                        is_full=is_full,
                        shift=shift,
                        _storage_shape=_storage_shape,
                    )
            return out
        return self._call(
            data=data,
            out=out,
            is_full=is_full,
            shift=shift,
            _storage_shape=_storage_shape,
        )

    def _call(self, *, data, out, _storage_shape, shift, is_full):
        done = data.get(self.done_key)
        done = done.clone()

        nsteps = done.shape[0]

        # capture for each item in data where the observation should be written
        idx = torch.arange(done.shape[0])
        padded_done = F.pad(done.squeeze(-1), [1, 0])
        root_idx = idx + padded_done[:-1].cumsum(0)
        next_idx = root_idx + 1

        if out is None:
            out = TensorDict(batch_size=[nsteps])

        def maybe_roll(entry, out=None):
            if is_full and shift is not None:
                if out is not None:
                    _roll_inplace(entry, shift=shift, out=out)
                    return
                else:
                    return entry.roll(shift, dims=0)
            if out is not None:
                out.copy_(entry)
                return
            return entry

        root_idx = maybe_roll(root_idx)
        next_idx = maybe_roll(next_idx)
        if not is_full:
            next_idx = next_idx[:-1]

        for key, entry in data.items(True, True):
            if entry.shape[0] == nsteps:
                if key in (self.done_keys.union(self.reward_keys)):
                    if key != "reward" and key not in out.keys(True, True):
                        # Create a done state at the root full of 0s
                        out.set(key, torch.zeros_like(entry), inplace=True)
                    entry = maybe_roll(entry, out=out.get(("next", key), None))
                    if entry is not None:
                        out.set(("next", key), entry, inplace=True)
                else:
                    # action and similar
                    entry = maybe_roll(entry, out=out.get(key, default=None))
                    if entry is not None:
                        # then out is not locked
                        out.set(key, entry, inplace=True)
            else:
                dest_next = out.get(("next", key), None)
                if dest_next is not None:
                    if not is_full:
                        dest_next = dest_next[:-1]
                    dest_next.copy_(entry[next_idx])
                else:
                    if not is_full:
                        val = entry[next_idx]
                        val = torch.cat([val, torch.zeros_like(val[:1])])
                        out.set(("next", key), val, inplace=True)
                    else:
                        out.set(("next", key), entry[next_idx], inplace=True)

                dest = out.get(key, None)
                if dest is not None:
                    dest.copy_(entry[root_idx])
                else:
                    out.set(key, entry[root_idx], inplace=True)
        return out


class TED2Nested(TED2Flat):
    """Converts a TED-formatted dataset into a tensordict populated with nested tensors where each row is a trajectory."""

    _shift: int | None = None
    _is_full: bool | None = None

    def __init__(self, *args, **kwargs):
        if not hasattr(torch, "_nested_compute_contiguous_strides_offsets"):
            raise ValueError(
                f"Unsupported torch version {torch.__version__}. "
                f"torch>=2.4 is required for {type(self).__name__} to be used."
            )
        return super().__init__(*args, **kwargs)

    def __call__(self, data: TensorDictBase, path: Path = None):
        data = super().__call__(data, path=path)

        shift = self.shift
        is_full = self.is_full
        storage_shape = data.get_non_tensor("_storage_shape", (-1,))
        # place time at the end
        storage_shape = (*storage_shape[1:], storage_shape[0])

        done = data.get("done")
        done = done.squeeze(-1).clone()
        if not is_full:
            done.view(storage_shape)[..., shift - 1] = True
        # else:
        done.view(storage_shape)[..., -1] = True

        ntraj = done.sum()

        nz = done.nonzero(as_tuple=True)[0]
        traj_lengths = torch.cat([nz[:1] + 1, nz.diff()])
        # if not is_full:
        #     traj_lengths = torch.cat(
        #         [traj_lengths, (done.shape[0] - traj_lengths.sum()).unsqueeze(0)]
        #     )

        keys_to_expand, keys_to_keep = zip(
            *[
                (key, None) if val.shape[0] != done.shape[0] else (None, key)
                for key, val in data.items(True, True)
            ]
        )
        keys_to_expand = [key for key in keys_to_expand if key is not None]
        keys_to_keep = [key for key in keys_to_keep if key is not None]

        out = TensorDict(batch_size=[ntraj])
        out.update(dict(data.non_tensor_items()))

        out.memmap_(path)

        traj_lengths = traj_lengths.unsqueeze(-1)
        if not is_full:
            # Increment by one only the trajectories that are not terminal
            traj_lengths_expand = traj_lengths + (
                traj_lengths.cumsum(0) % storage_shape[-1] != 0
            )
        else:
            traj_lengths_expand = traj_lengths + 1
        for key in keys_to_expand:
            val = data.get(key)
            shape = torch.cat(
                [
                    traj_lengths_expand,
                    torch.tensor(val.shape[1:], dtype=torch.long).repeat(
                        traj_lengths.numel(), 1
                    ),
                ],
                -1,
            )
            # This works because the storage location is the same as the previous one - no copy is done
            # but a new shape is written
            out.make_memmap_from_storage(
                key, val.untyped_storage(), dtype=val.dtype, shape=shape
            )
        for key in keys_to_keep:
            val = data.get(key)
            shape = torch.cat(
                [
                    traj_lengths,
                    torch.tensor(val.shape[1:], dtype=torch.long).repeat(
                        traj_lengths.numel(), 1
                    ),
                ],
                -1,
            )
            out.make_memmap_from_storage(
                key, val.untyped_storage(), dtype=val.dtype, shape=shape
            )
        return out


class Nested2TED(Flat2TED):
    """Converts a nested tensordict where each row is a trajectory into the TED format."""

    def __call__(self, data, out: TensorDictBase = None):
        # Get a flat representation of data
        def flatten_het_dim(tensor):
            shape = [tensor.size(i) for i in range(2, tensor.ndim)]
            tensor = torch.tensor(tensor.untyped_storage(), dtype=tensor.dtype).view(
                -1, *shape
            )
            return tensor

        data = data.apply(flatten_het_dim, batch_size=[])
        data.auto_batch_size_()
        return super().__call__(data, out=out)


class H5Split(TED2Flat):
    """Splits a dataset prepared with TED2Nested into a TensorDict where each trajectory is stored as views on their parent nested tensors."""

    _shift: int | None = None
    _is_full: bool | None = None

    def __call__(self, data):
        nzeros = int(math.ceil(math.log10(data.shape[0])))

        result = TensorDict(
            {
                f"traj_{str(i).zfill(nzeros)}": _data
                for i, _data in enumerate(data.filter_non_tensor_data().unbind(0))
            }
        ).update(dict(data.non_tensor_items()))

        return result


class H5Combine:
    """Combines trajectories in a persistent tensordict into a single standing tensordict stored in filesystem."""

    def __call__(self, data, out=None):
        # TODO: this load the entire H5 in memory, which can be problematic
        # Ideally we would want to load it on a memmap tensordict
        # We currently ignore out in this call but we should leverage that
        values = [val for key, val in data.items() if key.startswith("traj")]
        metadata_keys = [key for key in data.keys() if not key.startswith("traj")]
        result = TensorDict({key: NonTensorData(data[key]) for key in metadata_keys})

        # Create a memmap in file system (no files associated)
        result.memmap_()

        # Create each entry
        def initialize(key, *x):
            result.make_memmap(
                key,
                shape=torch.stack([torch.tensor(_x.shape) for _x in x]),
                dtype=x[0].dtype,
            )
            return

        values[0].named_apply(
            initialize,
            *values[1:],
            nested_keys=True,
            batch_size=[],
            filter_empty=True,
        )

        # Populate the entries
        def populate(key, *x):
            dest = result.get(key)
            for i, _x in enumerate(x):
                dest[i].copy_(_x)

        values[0].named_apply(
            populate,
            *values[1:],
            nested_keys=True,
            batch_size=[],
            filter_empty=True,
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
                f"a single tensor (name={default_name}). This behavior is not allowed. Please rename your "
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


def _roll_inplace(tensor, shift, out, index_dest=None, index_source=None):
    # slice 0
    source0 = tensor[:-shift]
    if index_source is not None:
        source0 = source0[index_source[shift:]]

    slice0_shift = source0.shape[0]
    if index_dest is not None:
        out[index_dest[-slice0_shift:]] = source0
    else:
        slice0 = out[-slice0_shift:]
        slice0.copy_(source0)

    # slice 1
    source1 = tensor[-shift:]
    if index_source is not None:
        source1 = source1[index_source[:shift]]
    if index_dest is not None:
        out[index_dest[:-slice0_shift]] = source1
    else:
        slice1 = out[:-slice0_shift]
        slice1.copy_(source1)
    return out


# Copy-paste of unravel-index for PT 2.0
def _unravel_index(
    indices: Tensor, shape: int | typing.Sequence[int] | torch.Size
) -> tuple[Tensor, ...]:
    res_tensor = _unravel_index_impl(indices, shape)
    return res_tensor.unbind(-1)


def _unravel_index_impl(indices: Tensor, shape: int | typing.Sequence[int]) -> Tensor:
    if isinstance(shape, (int, torch.SymInt)):
        shape = torch.Size([shape])
    else:
        shape = torch.Size(shape)

    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )
    return indices.unsqueeze(-1).floor_divide(
        torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)


@implement_for("torch", None, "2.2")
def unravel_index(indices, shape):
    """A version-compatible wrapper around torch.unravel_index."""
    return _unravel_index(indices, shape)


@implement_for("torch", "2.2")
def unravel_index(indices, shape):  # noqa: F811
    """A version-compatible wrapper around torch.unravel_index."""
    return torch.unravel_index(indices, shape)


@implement_for("torch", None, "2.3")
def tree_iter(pytree):
    """A version-compatible wrapper around tree_iter."""
    flat_tree, _ = torch.utils._pytree.tree_flatten(pytree)
    yield from flat_tree


@implement_for("torch", "2.3", "2.4")
def tree_iter(pytree):  # noqa: F811
    """A version-compatible wrapper around tree_iter."""
    yield from torch.utils._pytree.tree_leaves(pytree)


@implement_for("torch", "2.4")
def tree_iter(pytree):  # noqa: F811
    """A version-compatible wrapper around tree_iter."""
    yield from torch.utils._pytree.tree_iter(pytree)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.mps.is_available():
        return torch.device("mps:0")
    return torch.device("cpu")
