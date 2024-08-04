# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import importlib.util

import pytest

import torch

from tensordict import TensorDict
from torchrl.data.map import BinaryToDecimal, QueryModule, RandomProjectionHash, SipHash

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


class TestQuery:
    def test_query_construct(self):
        query_module = QueryModule(
            in_keys=[(("key1",),), (("another",), "key2")],
            index_key=("some", ("_index",)),
            hash_module=SipHash(),
            clone=False,
        )
        assert not query_module.clone
        assert query_module.in_keys == ["key1", ("another", "key2")]
        assert query_module.index_key == ("some", "_index")
        assert isinstance(query_module.hash_module, dict)
        assert isinstance(
            query_module.aggregator,
            type(query_module.hash_module[query_module.in_keys[0]]),
        )
        query_module = QueryModule(
            in_keys=[(("key1",),), (("another",), "key2")],
            index_key=("some", ("_index",)),
            hash_module=SipHash(),
            clone=False,
            aggregator=SipHash(),
        )
        # assert not isinstance(query_module.aggregator is not query_module.hash_module[0]
        assert isinstance(query_module.aggregator, SipHash)
        query_module = QueryModule(
            in_keys=[(("key1",),), (("another",), "key2")],
            index_key=("some", ("_index",)),
            hash_module=[SipHash(), SipHash()],
            clone=False,
        )
        # assert query_module.aggregator is not query_module.hash_module[0]
        assert isinstance(query_module.aggregator, SipHash)

    @pytest.mark.parametrize("index_key", ["index", ("another", "index")])
    @pytest.mark.parametrize("clone", [True, False])
    def test_query(self, clone, index_key):
        query_module = QueryModule(
            in_keys=["key1", "key2"],
            index_key=index_key,
            hash_module=SipHash(),
            clone=clone,
        )

        query = TensorDict(
            {
                "key1": torch.Tensor([[1], [1], [1], [2]]),
                "key2": torch.Tensor([[3], [3], [2], [3]]),
            },
            batch_size=(4,),
        )
        res = query_module(query)
        if clone:
            assert res is not query
        else:
            assert res is query
        assert index_key in res

        assert res[index_key][0] == res[index_key][1]
        for i in range(1, 3):
            assert res[index_key][i].item() != res[index_key][i + 1].item()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
