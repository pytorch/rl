# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import operator

import pytest
import torch

from _utils_internal import get_available_devices  # noqa

from tensordict.tensordict import TensorDict
from torchrl.objectives.value.functional import _get_num_per_traj_init
from torchrl.objectives.value.utils import (
    _get_num_per_traj,
    _inv_pad_sequence,
    _split_and_pad_sequence,
)


class TestUtils:
    @pytest.mark.parametrize("B", [None, (1, ), (4, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [1, 10])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_get_num_per_traj_no_stops(self, B, T, device):
        """check _get_num_per_traj when input contains no stops"""
        size = (*B, T) if B else (T,)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1) if B else 1
        res = torch.full((count,), T, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_get_num_per_traj(self, B, T, device):
        """check _get_num_per_traj where input contains a stop at half of each trace"""
        size = (*B, T)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        done[..., T // 2] = True
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1)
        res = [T - (T + 1) // 2 + 1, (T + 1) // 2 - 1] * count
        res = torch.as_tensor(res, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_reverse(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it"""
        torch.manual_seed(42)

        size = (*B, T)
        traj = torch.rand(*size, device=device)
        done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli(0.2)
        splits = _get_num_per_traj(done)

        splitted = _split_and_pad_sequence(traj, splits)
        reversed = _inv_pad_sequence(splitted, splits).reshape(traj.shape)

        torch.testing.assert_close(traj, reversed)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_no_stops(self, B, T, device):
        """_split_and_pad_sequence on trajectories without stops should not change input but flatten it along batch dimension"""
        size = (*B, T)
        count = functools.reduce(operator.mul, size, 1)

        traj = torch.arange(0, count, device=device).reshape(size)
        done = torch.zeros(*size, dtype=torch.bool, device=device)

        splits = _get_num_per_traj(done)
        splitted = _split_and_pad_sequence(traj, splits)

        traj_flat = traj.flatten(0, -2)
        torch.testing.assert_close(traj_flat, splitted)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_manual(self, device):
        """handcrafted example to test _split_and_pad_seqeunce"""

        traj = torch.as_tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
        splits = torch.as_tensor([3, 2, 1, 4], device=device)
        res = torch.as_tensor(
            [[0, 1, 2, 0], [3, 4, 0, 0], [5, 0, 0, 0], [6, 7, 8, 9]], device=device
        )

        splitted = _split_and_pad_sequence(traj, splits)
        torch.testing.assert_close(res, splitted)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_reverse_tensordict(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it on tensordict input"""
        torch.manual_seed(42)

        td = TensorDict(
            {
                "observation": torch.arange(T, dtype=torch.float32, device=device)
                .unsqueeze(-1)
                .expand(*B, T, 3),
                "is_init": torch.zeros(
                    *B, T, 1, dtype=torch.bool, device=device
                ).bernoulli(0.3),
            },
            [*B, T],
        )

        is_init = td.get("is_init").squeeze(-1)
        splits = _get_num_per_traj_init(is_init)
        splitted = _split_and_pad_sequence(
            td.select("observation", strict=False), splits
        )

        reversed = _inv_pad_sequence(splitted, splits)
        reversed = reversed.reshape(td.shape)
        torch.testing.assert_close(td["observation"], reversed["observation"])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
