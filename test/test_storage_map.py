# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import importlib.util

import pytest

import torch

from torchrl.data.map import BinaryToDecimal, RandomProjectionHash, SipHash

_has_gym = importlib.util.find_spec("gymnasium", None) or importlib.util.find_spec(
    "gym", None
)


class TestHash:
    def test_binary_to_decimal(self):
        binary_to_decimal = BinaryToDecimal(
            num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
        )
        binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
        decimal = binary_to_decimal(binary)

        assert decimal.shape == (2,)
        assert (decimal == torch.Tensor([3, 2])).all()

    def test_sip_hash(self):
        a = torch.rand((3, 2))
        b = a.clone()
        hash_module = SipHash(as_tensor=True)
        hash_a = torch.tensor(hash_module(a))
        hash_b = torch.tensor(hash_module(b))
        assert (hash_a == hash_b).all()

    @pytest.mark.parametrize("n_components", [None, 14])
    @pytest.mark.parametrize("scale", [0.001, 0.01, 1, 100, 1000])
    def test_randomprojection_hash(self, n_components, scale):
        torch.manual_seed(0)
        r = RandomProjectionHash(n_components=n_components)
        x = torch.randn(10000, 100).mul_(scale)
        y = r(x)
        if n_components is None:
            assert r.n_components == r._N_COMPONENTS_DEFAULT
        else:
            assert r.n_components == n_components

        assert y.shape == (10000,)
        assert y.unique().numel() == y.numel()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
