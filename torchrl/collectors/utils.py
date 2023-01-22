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
    mask = torch.ones(
        rollout_tensordict.batch_size,
        device=rollout_tensordict.device,
        dtype=torch.bool,
    )
    for dim in range(1, len(rollout_tensordict.batch_size)):
        mask.index_fill_(dim, torch.arange(1, rollout_tensordict.batch_size[dim]), 0)

    sep = ".-|-."
    rollout_tensordict = rollout_tensordict.flatten_keys(sep)
    traj_ids = rollout_tensordict.get("traj_ids")[mask].view(-1)
    unique_traj_ids = traj_ids.unique()
    MAX = max([(traj_ids == i).count_nonzero() for i in unique_traj_ids])

    out_splits = []
    for i in unique_traj_ids:
        out_split = rollout_tensordict[traj_ids == i]
        out_split.set(
            "mask",
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.get("done").device,
            ),
        )
        out_split = pad(out_split, [0, MAX - out_split.shape[0]])
        out_split = out_split.permute(-1, *range(len(out_split.batch_size) - 1))
        out_splits.append(out_split)

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
    """Permutes the batch dimesnions attording to the permute indeces and then squashes all leadning dimesnions apart from batch_size_unmasked."""
    # Bring all batch dimensions to the front (only performs computation if it is not already the case)
    tensordict = tensordict.permute(permute)
    # Flatten all batch dimensions into first one and leave unmasked dimensions untouched
    tensordict = tensordict.reshape(-1, *batch_size_unmasked)
    return tensordict
