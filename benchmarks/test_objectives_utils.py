# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from torchrl.objectives.value.functional import (
    _get_num_per_traj,
    _split_and_pad_sequence,
)


def _split_and_pad_sequence_old(tensor, splits):
    """Previous version of _split_and_pad_sequence"""
    tensor = tensor.flatten(0, -1)
    max_val = max(splits)
    mask = torch.zeros(len(splits), max_val, dtype=torch.bool, device=tensor.device)
    mask.scatter_(
        index=max_val - splits.unsqueeze(-1),
        dim=1,
        value=1,
    )
    mask = mask.cumsum(-1).flip(-1).bool()

    def _fill_tensor(tensor):
        empty_tensor = torch.zeros(
            len(splits),
            max_val,
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        empty_tensor[mask] = tensor
        return empty_tensor

    tensor = _fill_tensor(tensor)

    return tensor


@pytest.mark.parametrize(
    "split_fn", [_split_and_pad_sequence, _split_and_pad_sequence_old]
)
@pytest.mark.parametrize("batches", [1, 32, 256])
@pytest.mark.parametrize("timesteps", [20, 200, 2000])
def test_split_and_pad_sequence(benchmark, split_fn, batches, timesteps):
    size = (batches, timesteps, 1)
    print(size)

    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.device_count() else "cpu"

    traj = torch.rand(*size, device=device)
    done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli(0.2)
    splits = _get_num_per_traj(done)

    benchmark(
        split_fn,
        tensor=traj,
        splits=splits,
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
