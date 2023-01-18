# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from tensordict.tensordict import pad, TensorDictBase


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


def split_trajectories(rollout_tensordict: TensorDictBase) -> TensorDictBase:
    """A util function for trajectory separation.

    Takes a tensordict with a key traj_ids that indicates the id of each trajectory.

    From there, builds a B x T x ... zero-padded tensordict with B batches on max duration T
    """
    # TODO: incorporate tensordict.split once it's implemented
    sep = ".-|-."
    rollout_tensordict = rollout_tensordict.flatten_keys(sep)
    traj_ids = rollout_tensordict.get("traj_ids")
    splits = traj_ids.view(-1)
    splits = [(splits == i).sum().item() for i in splits.unique_consecutive()]
    # if all splits are identical then we can skip this function
    if len(set(splits)) == 1 and splits[0] == traj_ids.shape[-1]:
        rollout_tensordict.set(
            "mask",
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
            "mask",
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.get("done").device,
            ),
        )
    MAX = max(*[out_split.shape[0] for out_split in out_splits])
    td = torch.stack(
        [pad(out_split, [0, MAX - out_split.shape[0]]) for out_split in out_splits], 0
    ).contiguous()
    td = td.unflatten_keys(sep)
    return td
