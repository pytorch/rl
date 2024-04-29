# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tree
from __future__ import annotations

import typing
from typing import Any, Callable, Union

import numpy as np
import torch
from tensordict import MemoryMappedTensor, TensorDict, TensorDictBase

from torch import Tensor

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

    """

    def __init__(self, done_key=("next", "done")):
        self.done_key = done_key

    def __call__(self, data: TensorDictBase):
        # Get the done state
        done = data.get(self.done_key).clone()
        done = done.squeeze(-1)
        done[..., -1] = True
        # capture for each item in data where the observation should be written
        idx = torch.arange(data.shape[0])
        idx_done = (idx + done.cumsum(0))[done]
        idx += torch.nn.functional.pad(done, [1, 0])[:-1].cumsum(0)
        # data["custom", "idx_obs"] = idx

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

            if key in keys_to_expand:
                shape = torch.Size([idx.max() + 2, *entry.shape[1:]])
                dtype = entry.dtype
                empty = MemoryMappedTensor.empty(shape=shape, dtype=dtype)
                empty[idx] = entry
                empty[idx_done] = data.get(("next", key))[done]
                entry = empty
            output.set(key, entry)
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
        ...     print(td)
        ...
        ...     rb_load = ReplayBuffer(storage=LazyMemmapStorage(200))
        ...     rb_load.register_load_hook(Flat2TED())
        ...     rb_load.load(tmpdir)
        ...     print("storage after loading", rb_load[:])
        ...     assert (rb[:] == rb_load[:]).all()

    """

    def __init__(self, done_key="done"):
        self.done_key = done_key

    def __call__(self, data):
        done = data.get(self.done_key)
        nsteps = done.shape[0]

        # capture for each item in data where the observation should be written
        idx = torch.arange(done.shape[0])
        root_idx = idx + torch.nn.functional.pad(done.squeeze(-1), [1, 0])[:-1].cumsum(
            0
        )
        next_idx = root_idx + 1

        out = TensorDict({}, [nsteps])
        for key, entry in data.items(True, True):
            if entry.shape[0] == nsteps:
                if key in ("done", "terminated", "truncated", "reward"):
                    out["next", key] = entry
                    if key != "reward":
                        out[key] = torch.zeros_like(entry)
                else:
                    # action and similar
                    out[key] = entry
            else:
                root_entry = entry[root_idx]
                next_entry = entry[next_idx]
                out["next", key] = next_entry
                out[key] = root_entry
        return out
