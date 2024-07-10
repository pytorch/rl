# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from abc import abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar

import torch

import torch.nn as nn

from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.nn.common import TensorDictModuleBase

from torchrl.data import LazyTensorStorage, Storage
from torchrl.data.map import QueryModule, RandomProjectionHash

K = TypeVar("K")
V = TypeVar("V")


class TensorMap(abc.ABC, Generic[K, V]):
    """An Abstraction for implementing different storage.

    This class is for internal use, please use derived classes instead.
    """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: K) -> V:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def contains(self, item: K) -> torch.Tensor:
        raise NotImplementedError

    def __contains__(self, item):
        return self.contains(item)


class TensorDictMap(
    TensorDictModuleBase, TensorMap[TensorDictModuleBase, TensorDictModuleBase]
):
    """A Map-Storage for TensorDict.

    This module resembles a storage. It takes a tensordict as its input and
    returns another tensordict as output similar to TensorDictModuleBase. However,
    it provides additional functionality like python map:

    Args:
        query_module (TensorDictModuleBase): a query module, typically an instance of
            :class:`~tensordict.nn.QueryModule`, used to map a set of tensordict
            entries to a hash key.
        key_to_storage (Dict[NestedKey, TensorMap[torch.Tensor, torch.Tensor]]):
            a dictionary representing the map from an index key to a tensor storage.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from typing import cast
        >>> from torchrl.data import LazyTensorStorage
        >>> query_module = QueryModule(
        ...     in_keys=["key1", "key2"],
        ...     index_key="index",
        ... )
        >>> embedding_storage = LazyTensorStorage(1000)
        >>> tensor_dict_storage = TensorDictMap(
        ...     query_module=query_module,
        ...     key_to_storage={"out": embedding_storage},
        ... )
        >>> index = TensorDict(
        ...     {
        ...         "key1": torch.Tensor([[-1], [1], [3], [-3]]),
        ...         "key2": torch.Tensor([[0], [2], [4], [-4]]),
        ...     },
        ...     batch_size=(4,),
        ... )
        >>> value = TensorDict(
        ...     {"out": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
        ... )
        >>> tensor_dict_storage[index] = value
        >>> tensor_dict_storage[index]
        TensorDict(
            fields={
                out: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False)
        >>> assert torch.sum(tensor_dict_storage.contains(index)).item() == 4
        >>> new_index = index.clone(True)
        >>> new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
        >>> retrieve_value = tensor_dict_storage[new_index]
        >>> assert cast(torch.Tensor, retrieve_value["index"] == value["index"]).all()
    """

    def __init__(
        self,
        *,
        query_module: QueryModule,
        key_to_storage: Dict[NestedKey, TensorMap[torch.Tensor, torch.Tensor]],
    ):
        self.in_keys = query_module.in_keys
        self.out_keys = list(key_to_storage.keys())

        super().__init__()

        self.query_module = query_module
        self.index_key = query_module.index_key
        self.key_to_storage = key_to_storage
        self.batch_added = False

    @classmethod
    def from_tensordict_pair(
        cls,
        source,
        dest,
        in_keys: List[NestedKey],
        out_keys: List[NestedKey] | None = None,
        storage_type: type = lambda: LazyTensorStorage(1000),
        hash_module: Callable | None = None,
    ):
        """Creates a new TensorDictStorage from a pair of tensordicts (source and dest) using pre-defined rules of thumb.

        Args:
            source (TensorDict): An example of source tensordict, used as index in the storage.
            dest (TensorDict): An example of dest tensordict, used as data in the storage.
            in_keys (List[NestedKey]): a list of keys to use in the map.
            out_keys (List[NestedKey]): a list of keys to return in the output tensordict.
                All keys absent from out_keys, even if present in ``dest``, will not be stored
                in the storage. Defaults to ``None`` (all keys are registered).
            storage_type (type, optional): a type of tensor storage.
                Defaults to :class:`~tensordict.nn.storage.LazyDynamicStorage`.
                Other options include :class:`~tensordict.nn.storage.FixedStorage`.
            hash_module (Callable, optional): a hash function to use in the :class:`~tensordict.nn.storage.QueryModule`.
                Defaults to :class:`SipHash` for low-dimensional inputs, and :class:`~tensordict.nn.storage.RandomProjectionHash`
                for larger inputs.

        Examples:
            >>> # The following example requires torchrl and gymnasium to be installed
            >>> from torchrl.envs import GymEnv
            >>> torch.manual_seed(0)
            >>> env = GymEnv("CartPole-v1")
            >>> env.set_seed(0)
            >>> rollout = env.rollout(100)
            >>> source, dest = rollout.exclude("next"), rollout.get("next")
            >>> storage = TensorDictMap.from_tensordict_pair(
            ...     source, dest,
            ...     in_keys=["observation", "action"],
            ... )
            >>> # maps the (obs, action) tuple to a corresponding next state
            >>> storage[source] = dest
            >>> print(source["_index"])
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])
            >>> storage[source]
            TensorDict(
                fields={
                    done: Tensor(shape=torch.Size([14, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    observation: Tensor(shape=torch.Size([14, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    reward: Tensor(shape=torch.Size([14, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([14, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([14, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([14]),
                device=None,
                is_shared=False)

        """
        # Build query module
        if hash_module is None:
            # Count the features, if they're greater than RandomProjectionHash._N_COMPONENTS_DEFAULT
            #  use that module to project them to that dimensionality.
            n_feat = 0
            for in_key in in_keys:
                n_feat += source[in_key].shape[-1]
            if n_feat > RandomProjectionHash._N_COMPONENTS_DEFAULT:
                hash_module = RandomProjectionHash()
        query_module = QueryModule(in_keys, hash_module=hash_module)

        # Build key_to_storage
        if out_keys is None:
            out_keys = list(dest.keys(True, True))
        key_to_storage = {}
        for key in out_keys:
            key_to_storage[key] = storage_type()
        return cls(query_module=query_module, key_to_storage=key_to_storage)

    def clear(self) -> None:
        for mem in self.key_to_storage.values():
            mem.clear()

    def _to_index(self, item: TensorDictBase, extend: bool) -> torch.Tensor:
        item = self.query_module(item, extend=extend)
        return item[self.index_key]

    def _maybe_add_batch(
        self, item: TensorDictBase, value: TensorDictBase | None
    ) -> TensorDictBase:
        self.batch_added = False
        if len(item.batch_size) == 0:
            self.batch_added = True

            item = item.unsqueeze(dim=0)
            if value is not None:
                value = value.unsqueeze(dim=0)

        return item, value

    def _maybe_remove_batch(self, item: TensorDictBase) -> TensorDictBase:
        if self.batch_added:
            item = item.squeeze(dim=0)
        return item

    def __getitem__(self, item: TensorDictBase) -> TensorDictBase:
        item, _ = self._maybe_add_batch(item, None)

        index = self._to_index(item, extend=False)

        res = TensorDict({}, batch_size=item.batch_size)
        for k in self.out_keys:
            storage: Storage = self.key_to_storage[k]
            res.set(k, storage[index])

        res = self._maybe_remove_batch(res)
        return res

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        item, value = self._maybe_add_batch(item, value)

        index = self._to_index(item, extend=True)
        for k in self.out_keys:
            storage: Storage = self.key_to_storage[k]
            storage.set(index, value[k])

    def __len__(self):
        return len(next(iter(self.key_to_storage.values())))

    def contains(self, item: TensorDictBase) -> torch.Tensor:
        item, _ = self._maybe_add_batch(item, None)
        index = self._to_index(item, extend=False)

        res = next(iter(self.key_to_storage.values())).contains(index)
        res = self._maybe_remove_batch(res)
        return res
