# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from tensordict.tensordict import pad, TensorDictBase
from torch import multiprocessing as mp
from torch.multiprocessing import queues


def _stack_output(fun) -> Callable:
    def stacked_output_fun(*args, **kwargs):
        out = fun(*args, **kwargs)
        return tuple(torch.stack(_o, 0) for _o in out)

    return stacked_output_fun


def _stack_output_zip(fun) -> Callable:
    def stacked_output_fun(*args, **kwargs):
        out = fun(*args, **kwargs)
        return tuple(torch.stack(_o, 0) for _o in zip(*out))

    return stacked_output_fun


def split_trajectories(
    rollout_tensordict: TensorDictBase, prefix=None
) -> TensorDictBase:
    """A util function for trajectory separation.

    Takes a tensordict with a key traj_ids that indicates the id of each trajectory.

    From there, builds a B x T x ... zero-padded tensordict with B batches on max duration T

    Args:
        rollout_tensordict (TensorDictBase): a rollout with adjacent trajectories
            along the last dimension.
        prefix (str or tuple of str, optional): the prefix used to read and write meta-data,
            such as ``"traj_ids"`` (the optional integer id of each trajectory)
            and the ``"mask"`` entry indicating which data are valid and which
            aren't. Defaults to ``None`` (no prefix).
    """
    sep = ".-|-."

    if isinstance(prefix, str):
        traj_ids_key = (prefix, "traj_ids")
        mask_key = (prefix, "mask")
    elif isinstance(prefix, tuple):
        traj_ids_key = (*prefix, "traj_ids")
        mask_key = (*prefix, "mask")
    elif prefix is None:
        traj_ids_key = "traj_ids"
        mask_key = "mask"
    else:
        raise NotImplementedError(f"Unknown key type {type(prefix)}.")

    traj_ids = rollout_tensordict.get(traj_ids_key, None)
    done = rollout_tensordict.get(("next", "done"))
    truncated = rollout_tensordict.get(
        ("next", "truncated"), torch.zeros((), device=done.device, dtype=torch.bool)
    )
    done = done | truncated
    if traj_ids is None:
        traj_ids = done.cumsum(rollout_tensordict.ndim - 1)
        if rollout_tensordict.ndim > 1:
            for i in range(1, rollout_tensordict.shape[0]):
                traj_ids[i] += traj_ids[i - 1].max()
        rollout_tensordict.set(traj_ids_key, traj_ids)

    splits = traj_ids.view(-1)
    splits = [(splits == i).sum().item() for i in splits.unique_consecutive()]
    # if all splits are identical then we can skip this function
    if len(set(splits)) == 1 and splits[0] == traj_ids.shape[-1]:
        rollout_tensordict.set(
            mask_key,
            torch.ones(
                rollout_tensordict.shape,
                device=rollout_tensordict.device,
                dtype=torch.bool,
            ),
        )
        if rollout_tensordict.ndimension() == 1:
            rollout_tensordict = rollout_tensordict.unsqueeze(0).to_tensordict()
        return rollout_tensordict.unflatten_keys(sep)
    out_splits = rollout_tensordict.view(-1).split(splits, 0)

    for out_split in out_splits:
        out_split.set(
            mask_key,
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.get(("next", "done")).device,
            ),
        )
    if len(out_splits) > 1:
        MAX = max(*[out_split.shape[0] for out_split in out_splits])
    else:
        MAX = out_splits[0].shape[0]
    td = torch.stack(
        [pad(out_split, [0, MAX - out_split.shape[0]]) for out_split in out_splits], 0
    ).contiguous()
    # td = td.unflatten_keys(sep)
    return td


# from https://github.com/vterron/lemon/blob/d60576bec2ad5d1d5043bcb3111dff1fcb58a8d6/methods.py#L536-L573
class SharedCounter(object):
    """A synchronized shared counter.

    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/

    """

    def __init__(self, n=0):
        self.count = mp.Value("i", n)

    def increment(self, n=1):
        """Increment the counter by n (default = 1)."""
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """Return the value of the counter."""
        return self.count.value


class Queue(queues.Queue):
    """A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    """

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """Reliable implementation of qsize."""
        return self.size.value

    def empty(self):
        """Reliable implementation of empty."""
        return not self.qsize()

    def clear(self):
        """Remove all elements from the Queue."""
        while not self.empty():
            self.get()
