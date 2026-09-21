# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchrl.trainers.algorithms import SimplicialNormalization


def test_simplicial_normalization_normalizes_each_group():
    module = SimplicialNormalization(2)
    output = module(torch.zeros(3, 4))

    assert output.shape == (3, 4)
    torch.testing.assert_close(
        output.reshape(3, 2, 2).sum(-1),
        torch.ones(3, 2),
    )


def test_simplicial_normalization_rejects_non_divisible_features():
    with pytest.raises(ValueError, match="divisible"):
        SimplicialNormalization(3)(torch.zeros(2, 4))
