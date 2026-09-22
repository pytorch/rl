# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchrl.modules import SimplicialNormalization


class TestSimplicialNormalization:
    def test_groups(self):
        module = SimplicialNormalization(2)
        output = module(torch.zeros(3, 4))

        assert output.shape == (3, 4)
        torch.testing.assert_close(
            output.reshape(3, 2, 2).sum(-1),
            torch.ones(3, 2),
        )

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="divisible"):
            SimplicialNormalization(3)(torch.zeros(2, 4))


if __name__ == "__main__":
    pytest.main()
