# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence

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
    The input tensordict has batch_size = B x *other_dims

    From there, builds a B / T x *other_dims x T x ... zero-padded tensordict with B / T batches on max duration T
    """
    # TODO: incorporate tensordict.split once it's implemented
    env_batch_size_unmasked = rollout_tensordict.batch_size[1:]
    mask = torch.ones(
        rollout_tensordict.batch_size,
        device=rollout_tensordict.device,
        dtype=torch.bool,
    )
    for dim in range(1, len(rollout_tensordict.batch_size)):
        mask.index_fill_(dim, torch.arange(1, rollout_tensordict.batch_size[dim]), 0)

    sep = ".-|-."
    rollout_tensordict = rollout_tensordict.flatten_keys(sep)
    traj_ids = rollout_tensordict.get("traj_ids")[mask]
    splits = traj_ids.view(-1)
    splits = [(splits == i).sum().item() for i in splits.unique_consecutive()]
    # if all splits are identical then we can skip this function
    if len(set(splits)) == 1:
        rollout_tensordict.set(
            "mask",
            torch.ones(
                rollout_tensordict.shape,
                device=rollout_tensordict.device,
                dtype=torch.bool,
            ),
        )
        rollout_tensordict = rollout_tensordict.reshape(
            -1, *env_batch_size_unmasked, splits[0]
        )
        return rollout_tensordict.unflatten_keys(sep)
    out_splits = rollout_tensordict.view(-1, *env_batch_size_unmasked).split(splits, 0)

    for i in range(len(out_splits)):
        assert (
            out_splits[i]["traj_ids"]
            == rollout_tensordict.get("traj_ids")[mask].unique_consecutive()[i]
        ).all()

    MAX = max(*[out_split.shape[0] for out_split in out_splits])
    for i, out_split in enumerate(out_splits):
        out_split.set(
            "mask",
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.get("done").device,
            ),
        )
        out_splits[i] = pad(out_split, [0, MAX - out_split.shape[0]])
        out_splits[i] = out_splits[i].permute(
            -1, *range(len(out_splits[i].batch_size) - 1)
        )

    td = torch.stack(out_splits, 0).contiguous()
    td = td.unflatten_keys(sep)
    return td


def get_batch_size_masked(
    batch_size: torch.Size, mask: Optional[Sequence[bool]] = None
) -> torch.Size:
    """Returns a size with the masked dimensions equal to 1."""
    if mask is None:
        return batch_size
    if mask is not None and len(mask) != len(batch_size):
        raise RuntimeError(
            f"Batch size mask and env batch size have different lengths: mask={mask}, env.batch_size={batch_size}"
        )
    return torch.Size(
        [
            (dim if is_in else 1)
            for dim, is_in in zip(
                batch_size,
                mask,
            )
        ]
    )


def bring_forward_and_squash_batch_sizes(
    tensordict: TensorDictBase,
    permute: Sequence[int],
    batch_size_unmasked: Sequence[int],
) -> TensorDictBase:
    # Bring all batch dimensions to the front (only performs computation if it is not already the case)
    tensordict = tensordict.permute(permute)
    # Flatten all batch dimensions into first one and leave unmasked dimensions untouched
    tensordict = tensordict.reshape(-1, *batch_size_unmasked)
    return tensordict
