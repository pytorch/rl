# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import functools
from abc import abstractmethod
from typing import Any, Callable, Generic, TypeVar

import torch
from tensordict import is_tensor_collection, NestedKey, TensorDictBase
from tensordict.nn.common import TensorDictModuleBase

from torchrl.data.map.hash import RandomProjectionHash, SipHash
from torchrl.data.map.query import QueryModule
from torchrl.data.replay_buffers.storages import (
    _get_default_collate,
    LazyTensorStorage,
    TensorStorage,
)

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

    Keyword Args:
        query_module (TensorDictModuleBase): a query module, typically an instance of
            :class:`~tensordict.nn.QueryModule`, used to map a set of tensordict
            entries to a hash key.
        storage (Dict[NestedKey, TensorMap[torch.Tensor, torch.Tensor]]):
            a dictionary representing the map from an index key to a tensor storage.
        collate_fn (callable, optional): a function to use to collate samples from the
            storage. Defaults to a custom value for each known storage type (stack for
            :class:`~torchrl.data.ListStorage`, identity for :class:`~torchrl.data.TensorStorage`
            subtypes and others).

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
        ...     storage={"out": embedding_storage},
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
        storage: dict[NestedKey, TensorMap[torch.Tensor, torch.Tensor]],
        collate_fn: Callable[[Any], Any] | None = None,
        out_keys: list[NestedKey] | None = None,
        write_fn: Callable[[Any, Any], Any] | None = None,
    ):
        super().__init__()

        self.in_keys = query_module.in_keys
        if out_keys is not None:
            self.out_keys = out_keys

        self.query_module = query_module
        self.index_key = query_module.index_key
        self.storage = storage
        self.batch_added = False
        if collate_fn is None:
            collate_fn = _get_default_collate(self.storage)
        self.collate_fn = collate_fn
        self.write_fn = write_fn

    @property
    def max_size(self):
        return self.storage.max_size

    @property
    def out_keys(self) -> list[NestedKey]:
        out_keys = self.__dict__.get("_out_keys_and_lazy")
        if out_keys is not None:
            return out_keys[0]
        storage = self.storage
        if isinstance(storage, TensorStorage) and is_tensor_collection(
            storage._storage
        ):
            out_keys = list(storage._storage.keys(True, True))
            self._out_keys_and_lazy = (out_keys, True)
            return self.out_keys
        raise AttributeError(
            f"No out-keys found in the storage of type {type(storage)}"
        )

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys_and_lazy = (value, False)

    def _has_lazy_out_keys(self):
        _out_keys_and_lazy = self.__dict__.get("_out_keys_and_lazy")
        if _out_keys_and_lazy is None:
            return True
        return self._out_keys_and_lazy[1]

    @classmethod
    def from_tensordict_pair(
        cls,
        source,
        dest,
        in_keys: list[NestedKey],
        out_keys: list[NestedKey] | None = None,
        max_size: int = 1000,
        storage_constructor: type | None = None,
        hash_module: Callable | None = None,
        collate_fn: Callable[[Any], Any] | None = None,
        write_fn: Callable[[Any, Any], Any] | None = None,
        consolidated: bool | None = None,
    ) -> TensorDictMap:
        """Creates a new TensorDictStorage from a pair of tensordicts (source and dest) using pre-defined rules of thumb.

        Args:
            source (TensorDict): An example of source tensordict, used as index in the storage.
            dest (TensorDict): An example of dest tensordict, used as data in the storage.
            in_keys (List[NestedKey]): a list of keys to use in the map.
            out_keys (List[NestedKey]): a list of keys to return in the output tensordict.
                All keys absent from out_keys, even if present in ``dest``, will not be stored
                in the storage. Defaults to ``None`` (all keys are registered).
            max_size (int, optional): the maximum number of elements in the storage. Ignored if the
                ``storage_constructor`` is passed. Defaults to ``1000``.
            storage_constructor (Type, optional): a type of tensor storage.
                Defaults to :class:`~tensordict.nn.storage.LazyDynamicStorage`.
                Other options include :class:`~tensordict.nn.storage.FixedStorage`.
            hash_module (Callable, optional): a hash function to use in the :class:`~torchrl.data.map.QueryModule`.
                Defaults to :class:`SipHash` for low-dimensional inputs, and :class:`~torchrl.data.map.RandomProjectionHash`
                for larger inputs.
            collate_fn (callable, optional): a function to use to collate samples from the
                storage. Defaults to a custom value for each known storage type (stack for
                :class:`~torchrl.data.ListStorage`, identity for :class:`~torchrl.data.TensorStorage`
                subtypes and others).
            consolidated (bool, optional): whether to consolidate the storage in a single storage tensor.
                Defaults to ``False``.

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
            hash_module = []
            for in_key in in_keys:
                entry = source[in_key]
                if entry.ndim == source.ndim:
                    # this is a good example of why td/tc are useful - carrying metadata
                    # allows us to know if there's a feature dim or not
                    n_feat = 0
                else:
                    n_feat = entry.shape[-1]
                if n_feat > RandomProjectionHash._N_COMPONENTS_DEFAULT:
                    _hash_module = RandomProjectionHash()
                else:
                    _hash_module = SipHash()
                hash_module.append(_hash_module)
        query_module = QueryModule(in_keys, hash_module=hash_module)

        # Build key_to_storage
        if storage_constructor is None:
            storage_constructor = functools.partial(
                LazyTensorStorage, max_size, consolidated=bool(consolidated)
            )
        elif consolidated is not None:
            storage_constructor = functools.partial(
                storage_constructor, consolidated=consolidated
            )
        storage = storage_constructor()
        result = cls(
            query_module=query_module,
            storage=storage,
            collate_fn=collate_fn,
            out_keys=out_keys,
            write_fn=write_fn,
        )
        return result

    def clear(self) -> None:
        for mem in self.storage.values():
            mem.clear()

    def _to_index(
        self, item: TensorDictBase, extend: bool, clone: bool | None = None
    ) -> torch.Tensor:
        item = self.query_module(item, extend=extend, clone=clone)
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
        item = item.copy()
        item, _ = self._maybe_add_batch(item, None)

        index = self._to_index(item, extend=False, clone=False)

        res = self.storage[index]
        res = self.collate_fn(res)
        res = self._maybe_remove_batch(res)
        return res

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        if not self._has_lazy_out_keys():
            # TODO: make this work with pytrees and avoid calling select if keys match
            value = value.select(*self.out_keys, strict=False)
        item, value = self._maybe_add_batch(item, value)
        index = self._to_index(item, extend=True)
        if index.unique().numel() < index.numel():
            # If multiple values point to the same place in the storage, we cannot process them by batch
            # There could be a better way to deal with this, using unique ids.
            vals = []
            for it, val in zip(item.split(1), value.split(1)):
                self[it] = val
                vals.append(val)
            # __setitem__ may affect the content of the input data
            value.update(TensorDictBase.lazy_stack(vals))
            return
        if self.write_fn is not None:
            # We use this block in the following context: the value written in the storage is already present,
            # but it needs to be updated.
            # We first check if the value is already there using `contains`. If so, we pass the new value and the
            # previous one to write_fn. The values that are not present are passed alone.
            if len(self):
                modifiable = self.contains(item)
                if modifiable.any():
                    to_modify = (value[modifiable], self[item[modifiable]])
                    v1 = self.write_fn(*to_modify)
                    result = value.empty()
                    result[modifiable] = v1
                    result[~modifiable] = self.write_fn(value[~modifiable])
                    value = result
                else:
                    value = self.write_fn(value)
            else:
                value = self.write_fn(value)
        self.storage.set(index, value)

    def __len__(self):
        return len(self.storage)

    def contains(self, item: TensorDictBase) -> torch.Tensor:
        item, _ = self._maybe_add_batch(item, None)
        index = self._to_index(item, extend=False, clone=True)

        res = self.storage.contains(index)
        res = self._maybe_remove_batch(res)
        return res
