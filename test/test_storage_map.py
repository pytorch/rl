# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import importlib.util

import pytest

import torch

from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ListStorage
from torchrl.data.map import (
    BinaryToDecimal,
    QueryModule,
    RandomProjectionHash,
    SipHash,
    TensorDictMap,
)
from torchrl.envs import GymEnv

_has_gym = importlib.util.find_spec("gymnasium", None) or importlib.util.find_spec(
    "gym", None
)

# def test_embedding_memory():
#     embedding_storage = FixedStorage(
#         torch.nn.Embedding(num_embeddings=10, embedding_dim=2),
#         lambda x: torch.nn.init.constant_(x, 0),
#     )
#
#     index = torch.Tensor([1, 2]).long()
#     assert len(embedding_storage) == 0
#     assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()
#
#     embedding_storage[index] = torch.ones(size=(2, 2))
#     assert torch.sum(embedding_storage.contains(index)).item() == 2
#
#     assert (embedding_storage[index] == torch.ones(size=(2, 2))).all()
#
#     assert len(embedding_storage) == 2
#     embedding_storage.clear()
#     assert len(embedding_storage) == 0
#     assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()


# def test_dynamic_storage():
#     storage = DynamicStorage(default_tensor=torch.zeros((1,)))
#     index = torch.randn((3,))
#     value = torch.rand((3, 1))
#     storage[index] = value
#     assert len(storage) == 3
#     assert (storage[index.clone()] == value).all()


def test_binary_to_decimal():
    binary_to_decimal = BinaryToDecimal(
        num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
    )
    binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
    decimal = binary_to_decimal(binary)

    assert decimal.shape == (2,)
    assert (decimal == torch.Tensor([3, 2])).all()


class TestHash:
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


def test_query():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    query = TensorDict(
        {
            "key1": torch.Tensor([[1], [1], [1], [2]]),
            "key2": torch.Tensor([[3], [3], [2], [3]]),
        },
        batch_size=(4,),
    )
    res = query_module(query)

    assert res["index"][0] == res["index"][1]
    for i in range(1, 3):
        assert res["index"][i].item() != res["index"][i + 1].item(), (
            f"{i} = ({query[i]['key1']}, {query[i]['key2']}) s index and {i + 1} = ({query[i + 1]['key1']}, "
            f"{query[i + 1]['key2']})'s index are the same!"
        )


def test_query_module():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    embedding_storage = LazyTensorStorage(23)

    tensor_dict_storage = TensorDictMap(
        query_module=query_module,
        storage=embedding_storage,
    )

    index = TensorDict(
        {
            "key1": torch.Tensor([[-1], [1], [3], [-3]]),
            "key2": torch.Tensor([[0], [2], [4], [-4]]),
        },
        batch_size=(4,),
    )

    value = TensorDict(
        {"index": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
    )

    tensor_dict_storage[index] = value
    assert torch.sum(tensor_dict_storage.contains(index)).item() == 4

    new_index = index.clone(True)
    new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
    retrieve_value = tensor_dict_storage[new_index]

    assert (retrieve_value["index"] == value["index"]).all()


def test_storage():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    embedding_storage = ListStorage()

    tensor_dict_storage = TensorDictMap(
        query_module=query_module,
        storage=embedding_storage,
    )

    index = TensorDict(
        {
            "key1": torch.Tensor([[-1], [1], [3], [-3]]),
            "key2": torch.Tensor([[0], [2], [4], [-4]]),
        },
        batch_size=(4,),
    )

    value = TensorDict(
        {"index": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
    )

    tensor_dict_storage[index] = value
    assert torch.sum(tensor_dict_storage.contains(index)).item() == 4

    new_index = index.clone(True)
    new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
    retrieve_value = tensor_dict_storage[new_index]

    assert (retrieve_value["index"] == value["index"]).all()


@pytest.mark.skipif(not _has_gym, reason="gym not installed")
def test_rollout():
    torch.manual_seed(0)
    env = GymEnv("CartPole-v1")
    env.set_seed(0)
    rollout = env.rollout(100)
    source, dest = rollout.exclude("next"), rollout.get("next")
    storage = TensorDictMap.from_tensordict_pair(
        source,
        dest,
        in_keys=["observation", "action"],
    )
    storage_indices = TensorDictMap.from_tensordict_pair(
        source,
        dest,
        in_keys=["observation"],
        out_keys=["_index"],
    )
    # maps the (obs, action) tuple to a corresponding next state
    storage[source] = dest
    storage_indices[source] = source
    contains = storage.contains(source)
    assert len(contains) == rollout.shape[-1]
    assert contains.all()
    contains = storage.contains(torch.cat([source, source + 1]))
    assert len(contains) == rollout.shape[-1] * 2
    assert contains[: rollout.shape[-1]].all()
    assert not contains[rollout.shape[-1] :].any()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
