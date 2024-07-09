# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from abc import abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar, Mapping, Any

import torch

import torch.nn as nn

from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn.common import TensorDictModuleBase

from torchrl.data.td2td import SipHash

K = TypeVar("K")
V = TypeVar("V")

class HashToInt(nn.Module):
    def __init__(self):
        self._index_to_index = {}

    def __call__(self, key: torch.Tensor) -> torch.Tensor:
        result = []
        for _item in key.tolist():
            result.append(
                self._index_to_index.setdefault(_item, len(self._index_to_index))
            )
        return torch.tensor(result, device=key.device, dtype=key.dtype)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        values = torch.tensor(self._index_to_index.values())
        keys = torch.tensor(self._index_to_index.keys())
        return {"keys": keys, "values": values}
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        keys = state_dict['keys']
        values=  state_dict["values"]
        self._index_to_index = {key: val for key, val in zip(keys.tolist(), values.tolist())}

class QueryModule(TensorDictModuleBase):
    """A Module to generate compatible indices for storage.

    A module that queries a storage and return required index of that storage.
    Currently, it only outputs integer indices (torch.int64).

    Args:
        in_keys (list of NestedKeys): keys of the input tensordict that
            will be used to generate the hash value.
        index_key (NestedKey): the output key where the hash value will be written.
            Defaults to ``"_index"``.

    Keyword Args:
        hash_module (nn.Module or Callable[[torch.Tensor], torch.Tensor]): a hash
            module similar to :class:`~tensordict.nn.SipHash` (default).
        aggregation_module (torch.nn.Module or Callable[[torch.Tensor], torch.Tensor]): a
            method to aggregate the hash values. Defaults to the value of ``hash_module``.
            If only one ``in_Keys`` is provided, this module will be ignored.
        clone (bool, optional): if ``True``, a shallow clone of the input TensorDict will be
            returned. Defaults to ``False``.

    Examples:
        >>> query_module = QueryModule(
        ...     in_keys=["key1", "key2"],
        ...     index_key="index",
        ...     hash_module=SipHash(),
        ... )
        >>> query = TensorDict(
        ...     {
        ...         "key1": torch.Tensor([[1], [1], [1], [2]]),
        ...         "key2": torch.Tensor([[3], [3], [2], [3]]),
        ...         "other": torch.randn(4),
        ...     },
        ...     batch_size=(4,),
        ... )
        >>> res = query_module(query)
        >>> # The first two pairs of key1 and key2 match
        >>> assert res["index"][0] == res["index"][1]
        >>> # The last three pairs of key1 and key2 have at least one mismatching value
        >>> assert res["index"][1] != res["index"][2]
        >>> assert res["index"][2] != res["index"][3]
    """

    def __init__(
        self,
        in_keys: List[NestedKey],
        index_key: NestedKey = "_index",
        *,
        hash_module: torch.nn.Module | None = None,
        hash_to_int: Callable[[int], int] | None=None,
        clone: bool = False,
    ):
        self.in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        if len(in_keys) == 0:
            raise ValueError("`in_keys` cannot be empty.")
        self.out_keys = [index_key]

        super().__init__()

        if hash_module is None:
            hash_module = SipHash()
        if hash_to_int is None:
            hash_to_int = HashToInt()

        self.hash_module = hash_module
        self.hash_to_int = hash_to_int

        self.index_key = index_key
        self.clone = clone

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        hash_values = []

        i = -1  # to make linter happy
        for k in self.in_keys:
            hash_values.append(self.hash_module(tensordict.get(k)))

        if i > 0:
            td_hash_value = self.hash_to_int(
                torch.stack(
                    hash_values,
                    dim=-1,
                ),
            )
        else:
            td_hash_value = self.hash_to_int(hash_values[0])

        if self.clone:
            output = tensordict.copy()
        else:
            output = tensordict

        output.set(self.index_key, td_hash_value)
        return output
