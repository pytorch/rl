# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import importlib.util
from typing import Tuple

import pytest

import torch

from tensordict import assert_close, TensorDict
from torchrl.data import LazyTensorStorage, ListStorage, MCTSForest
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

    def test_query_module(self):
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


class TesttTensorDictMap:
    @pytest.mark.parametrize(
        "storage_type",
        [
            functools.partial(ListStorage, 1000),
            functools.partial(LazyTensorStorage, 1000),
        ],
    )
    def test_map(self, storage_type):
        query_module = QueryModule(
            in_keys=["key1", "key2"],
            index_key="index",
            hash_module=SipHash(),
        )

        embedding_storage = storage_type()

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
        assert not hasattr(tensor_dict_storage, "out_keys")

        tensor_dict_storage[index] = value
        if isinstance(embedding_storage, LazyTensorStorage):
            assert hasattr(tensor_dict_storage, "out_keys")
        else:
            assert not hasattr(tensor_dict_storage, "out_keys")
        assert tensor_dict_storage._has_lazy_out_keys()
        assert torch.sum(tensor_dict_storage.contains(index)).item() == 4

        new_index = index.clone(True)
        new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
        retrieve_value = tensor_dict_storage[new_index]

        assert (retrieve_value["index"] == value["index"]).all()

    @pytest.mark.skipif(not _has_gym, reason="gym not installed")
    def test_map_rollout(self):
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


class TestMCTSForest:
    def dummy_rollouts(self) -> Tuple[TensorDict, ...]:
        """
        ├── 0
        │   ├── 16
        │   ├── 17
        │   ├── 18
        │   ├── 19
        │   └── 20
        ├── 1
        ├── 2
        ├── 3
        │   ├── 6
        │   ├── 7
        │   ├── 8
        │   ├── 9
        │   └── 10
        ├── 4
        │   ├── 11
        │   ├── 12
        │   ├── 13
        │   │   ├── 21
        │   │   ├── 22
        │   │   ├── 23
        │   │   ├── 24
        │   │   └── 25
        │   ├── 14
        │   └── 15
        └── 5

        """

        states0 = torch.arange(6)
        actions0 = torch.full((5,), 0)

        states1 = torch.cat([torch.tensor([3]), torch.arange(6, 11)])
        actions1 = torch.full((5,), 1)

        states2 = torch.cat([torch.tensor([4]), torch.arange(11, 16)])
        actions2 = torch.full((5,), 2)

        states3 = torch.cat([torch.tensor([0]), torch.arange(16, 21)])
        actions3 = torch.full((5,), 3)

        states4 = torch.cat([torch.tensor([13]), torch.arange(21, 26)])
        actions4 = torch.full((5,), 4)

        return (
            self._make_td(states0, actions0),
            self._make_td(states1, actions1),
            self._make_td(states2, actions2),
            self._make_td(states3, actions3),
            self._make_td(states4, actions4),
        )

    def _state0(self) -> TensorDict:
        return self.dummy_rollouts()[0][0]

    @staticmethod
    def _make_td(state: torch.Tensor, action: torch.Tensor) -> TensorDict:
        done = torch.zeros_like(action, dtype=torch.bool).unsqueeze(-1)
        reward = action.clone()

        return TensorDict(
            {
                "observation": state[:-1],
                "action": action,
                "done": torch.zeros_like(done),
                "next": {
                    "observation": state[1:],
                    "done": done,
                    "reward": reward,
                },
            }
        ).auto_batch_size_()

    def _make_forest(self) -> MCTSForest:
        r0, r1, r2, r3, r4 = self.dummy_rollouts()
        assert r0.shape
        forest = MCTSForest()
        forest.extend(r0)
        forest.extend(r1)
        forest.extend(r2)
        forest.extend(r3)
        forest.extend(r4)
        return forest

    def _make_forest_intersect(self) -> MCTSForest:
        """
        ├── 0
        │   ├── 16
        │   ├── 17
        │   ├── 18
        │   ├── 19───────│
        │   │    └── 26  │
        │   └── 20       │
        ├── 1            │
        ├── 2            │
        ├── 3            │
        │   ├── 6        │
        │   ├── 7        │
        │   ├── 8        │
        │   ├── 9        │
        │   └── 10       │
        ├── 4            │
        │   ├── 11       │
        │   ├── 12       │
        │   ├── 13       │
        │   │   ├── 21   │
        │   │   ├── 22   │
        │   │   ├── 23   │
        │   │   ├── 24 ──│
        │   │   └── 25
        │   ├── 14
        │   └── 15
        └── 5
        """
        forest = self._make_forest()
        states5 = torch.cat([torch.tensor([24]), torch.tensor([19, 26])])
        actions5 = torch.full((2,), 5)
        rollout5 = self._make_td(states5, actions5)
        forest.extend(rollout5)
        return forest

    @staticmethod
    def make_labels(tree):
        if tree.rollout is not None:
            s = torch.cat(
                [
                    tree.rollout["observation"][:1],
                    tree.rollout["next", "observation"],
                ]
            )
            a = tree.rollout["action"].tolist()
            s = s.tolist()
            return f"node {tree.node_id}: states {s}, actions {a}"
        return f"node {tree.node_id}"

    def test_forest_build(self):
        r0, *_ = self.dummy_rollouts()
        forest = self._make_forest()
        tree = forest.get_tree(r0[0])
        for leaf in tree.vertices().values():
            assert leaf in tree
        # tree.plot(make_labels=self.make_labels)

    def test_forest_vertices(self):
        r0, *_ = self.dummy_rollouts()
        forest = self._make_forest()

        tree = forest.get_tree(r0[0])
        assert tree.num_vertices() == 9  # (0, 20, 3, 10, 4, 13, 25, 15, 5)

        tree = forest.get_tree(r0[0], compact=False)
        assert tree.num_vertices() == 26

    def test_forest_rebuild_rollout(self):
        r0, r1, r2, r3, r4 = self.dummy_rollouts()
        forest = self._make_forest()

        tree = forest.get_tree(r0[0])
        assert_close(tree.rollout_from_path((0, 0, 0)), r0, intersection=True)
        assert_close(tree.rollout_from_path((0, 1))[-5:], r1, intersection=True)
        assert_close(tree.rollout_from_path((0, 0, 1, 0))[-5:], r2, intersection=True)
        assert_close(tree.rollout_from_path((1,))[-5:], r3, intersection=True)
        assert_close(tree.rollout_from_path((0, 0, 1, 1))[-5:], r4, intersection=True)

    def test_forest_check_hashes(self):
        r0, *_ = self.dummy_rollouts()
        forest = self._make_forest()
        tree = forest.get_tree(r0[0])
        nodes = range(tree.num_vertices())
        hashes = set()
        for n in nodes:
            vertex = tree.get_vertex_by_id(n)
            node_hash = vertex.hash
            if node_hash is not None:
                assert isinstance(node_hash, int)
                hashes.add(node_hash)
            else:
                assert vertex is tree
        assert len(hashes) == tree.num_vertices() - 1

    def test_forest_check_ids(self):
        r0, *_ = self.dummy_rollouts()
        forest = self._make_forest()
        tree = forest.get_tree(r0[0])
        nodes = range(tree.num_vertices())
        for n in nodes:
            vertex = tree.get_vertex_by_id(n)
            node_id = vertex.node_id
            assert isinstance(node_id, int)
            assert node_id == n

    # Ideally, we'd like to have only views but because we index the storage with a tensor
    #  we actually get regular, single-storage tensors
    # def test_forest_view(self):
    #     import tensordict.base
    #     r0, *_ = self.dummy_rollouts()
    #     forest = self._make_forest()
    #     tree = forest.get_tree(r0[0])
    #     dataptr = set()
    #     # Check that all tensors point to the same storage (ie, that we only have views)
    #     for k, v in tree.items(True, True, is_leaf=tensordict.base._NESTED_TENSORS_AS_LISTS):
    #         if isinstance(k, tuple) and "rollout" in k:
    #             dataptr.add(v.storage().data_ptr())
    #             assert len(dataptr) == 1, k

    def test_forest_intersect(self):
        state0 = self._state0()
        forest = self._make_forest_intersect()
        tree = forest.get_tree(state0)
        subtree = forest.get_tree(TensorDict(observation=19))

        # subtree.plot(make_labels=make_labels)
        # tree.plot(make_labels=make_labels)
        assert tree.get_vertex_by_id(2).num_children == 2
        assert tree.get_vertex_by_id(1).num_children == 2
        assert tree.get_vertex_by_id(3).num_children == 2
        assert tree.get_vertex_by_id(8).num_children == 2
        assert tree.get_vertex_by_id(10).num_children == 2
        assert tree.get_vertex_by_id(12).num_children == 2

        # Test contains
        assert subtree in tree

    def test_forest_intersect_vertices(self):
        state0 = self._state0()
        forest = self._make_forest_intersect()
        tree = forest.get_tree(state0)
        assert len(tree.vertices(key_type="path")) > len(tree.vertices(key_type="hash"))
        assert len(tree.vertices(key_type="id")) == len(tree.vertices(key_type="hash"))
        with pytest.raises(ValueError, match="key_type must be"):
            tree.vertices(key_type="another key type")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
