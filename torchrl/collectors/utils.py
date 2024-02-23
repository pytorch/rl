# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Callable

import torch

from tensordict import NestedKey, pad, set_lazy_legacy, TensorDictBase


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


@set_lazy_legacy(False)
def split_trajectories(
    rollout_tensordict: TensorDictBase,
    *,
    prefix=None,
    trajectory_key: NestedKey | None = None,
    done_key: NestedKey | None = None,
) -> TensorDictBase:
    """A util function for trajectory separation.

    Takes a tensordict with a key traj_ids that indicates the id of each trajectory.

    From there, builds a B x T x ... zero-padded tensordict with B batches on max duration T

    Args:
        rollout_tensordict (TensorDictBase): a rollout with adjacent trajectories
            along the last dimension.
        prefix (NestedKey, optional): the prefix used to read and write meta-data,
            such as ``"traj_ids"`` (the optional integer id of each trajectory)
            and the ``"mask"`` entry indicating which data are valid and which
            aren't. Defaults to ``"collector"`` if the input has a ``"collector"``
            entry, ``()`` (no prefix) otherwise.
            ``prefix`` is kept as a legacy feature and will be deprecated eventually.
            Prefer ``trajectory_key`` or ``done_key`` whenever possible.
        trajectory_key (NestedKey, optional): the key pointing to the trajectory
            ids. Supersedes ``done_key`` and ``prefix``. If not provided, defaults
            to ``(prefix, "traj_ids")``.
        done_key (NestedKey, optional): the key pointing to the ``"done""`` signal,
            if the trajectory could not be directly recovered. Defaults to ``"done"``.

    """
    mask_key = None
    if trajectory_key is not None:
        from torchrl.envs.utils import _replace_last

        traj_ids_key = trajectory_key
        mask_key = _replace_last(trajectory_key, "mask")
    else:
        if prefix is None and "collector" in rollout_tensordict.keys():
            prefix = "collector"
        if prefix is None:
            traj_ids_key = "traj_ids"
            mask_key = "mask"
        else:
            traj_ids_key = (prefix, "traj_ids")
            mask_key = (prefix, "mask")

    rollout_tensordict = rollout_tensordict.copy()
    traj_ids = rollout_tensordict.get(traj_ids_key, None)
    if traj_ids is None:
        if done_key is None:
            done_key = "done"
        done_key = ("next", done_key)
        done = rollout_tensordict.get(done_key)
        idx = (slice(None),) * (rollout_tensordict.ndim - 1) + (slice(None, -1),)
        done_sel = done[idx]
        pads = [1, 0]
        pads = [0, 0] * (done.ndim - rollout_tensordict.ndim) + pads
        done_sel = torch.nn.functional.pad(done_sel, pads)
        if done_sel.shape != done.shape:
            raise RuntimeError(
                f"done and done_sel have different shape {done.shape} - {done_sel.shape} "
            )
        traj_ids = done_sel.cumsum(rollout_tensordict.ndim - 1)
        traj_ids = traj_ids.squeeze(-1)
        if rollout_tensordict.ndim > 1:
            for i in range(1, rollout_tensordict.shape[0]):
                traj_ids[i] += traj_ids[i - 1].max() + 1
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
            rollout_tensordict = rollout_tensordict.unsqueeze(0)
        return rollout_tensordict

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
    return td
