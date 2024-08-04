# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, TypeVar

import torch
import torch.nn as nn
from tensordict import NestedKey, TensorDictBase
from tensordict.nn.common import TensorDictModuleBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.map import SipHash

K = TypeVar("K")
V = TypeVar("V")


class HashToInt(nn.Module):
    """Converts a hash value to an integer that can be used for indexing a contiguous storage."""

    def __init__(self):
        super().__init__()
        self._index_to_index = {}

    def __call__(self, key: torch.Tensor, extend: bool = False) -> torch.Tensor:
        result = []
        if extend:
            for _item in key.tolist():
                result.append(
                    self._index_to_index.setdefault(_item, len(self._index_to_index))
                )
        else:
            for _item in key.tolist():
                result.append(
                    self._index_to_index.get(_item, len(self._index_to_index))
                )
        return torch.tensor(result, device=key.device, dtype=key.dtype)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        values = torch.tensor(self._index_to_index.values())
        keys = torch.tensor(self._index_to_index.keys())
        return {"keys": keys, "values": values}

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        keys = state_dict["keys"]
        values = state_dict["values"]
        self._index_to_index = {
            key: val for key, val in zip(keys.tolist(), values.tolist())
        }


class QueryModule(TensorDictModuleBase):
    """A Module to generate compatible indices for storage.

    A module that queries a storage and return required index of that storage.
    Currently, it only outputs integer indices (torch.int64).

    Args:
        in_keys (list of NestedKeys): keys of the input tensordict that
            will be used to generate the hash value.
        index_key (NestedKey): the output key where the index value will be written.
            Defaults to ``"_index"``.

    Keyword Args:
        hash_key (NestedKey): the output key where the hash value will be written.
            Defaults to ``"_hash"``.
        hash_module (Callable[[Any], int] or a list of these, optional): a hash
            module similar to :class:`~tensordict.nn.SipHash` (default).
            If a list of callables is provided, its length must equate the number of in_keys.
        hash_to_int (Callable[[int], int], optional): a stateful function that
            maps a hash value to a non-negative integer corresponding to an index in a
            storage. Defaults to :class:`~torchrl.data.map.HashToInt`.
        aggregator (Callable[[int], int], optional): a hash function to group multiple hashes
            together. This argument should only be passed when there is more than one ``in_keys``.
            If a single ``hash_module`` is provided but no aggregator is passed, it will take
            the value of the hash_module. If no ``hash_module`` or a list of ``hash_modules`` is
            provided but no aggregator is passed, it will default to ``SipHash``.
       clone (bool, optional): if ``True``, a shallow clone of the input TensorDict will be
            returned. This can be used to retrieve the integer index within the storage,
            corresponding to a given input tensordict.
            Defaults to ``False``.
    d
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
        hash_key: NestedKey = "_hash",
        *,
        hash_module: Callable[[Any], int] | List[Callable[[Any], int]] | None = None,
        hash_to_int: Callable[[int], int] | None = None,
        aggregator: Callable[[Any], int] = None,
        clone: bool = False,
    ):
        if len(in_keys) == 0:
            raise ValueError("`in_keys` cannot be empty.")
        in_keys = in_keys if isinstance(in_keys, List) else [in_keys]

        super().__init__()
        in_keys = self.in_keys = in_keys
        self.out_keys = [index_key, hash_key]
        index_key = self.out_keys[0]
        self.hash_key = self.out_keys[1]

        if aggregator is not None and len(self.in_keys) == 1:
            torchrl_logger.warn(
                "An aggregator was provided but there is only one in-key to be read. "
                "This module will be ignored."
            )
        elif aggregator is None:
            if hash_module is not None and not isinstance(hash_module, list):
                aggregator = hash_module
            else:
                aggregator = SipHash()
        if hash_module is None:
            hash_module = [SipHash() for _ in range(len(self.in_keys))]
        elif not isinstance(hash_module, list):
            try:
                hash_module = [
                    deepcopy(hash_module) if len(self.in_keys) > 1 else hash_module
                    for _ in range(len(self.in_keys))
                ]
            except Exception as err:
                raise RuntimeError(
                    "failed to deepcopy the hash module. Please provide a list of hash modules instead."
                ) from err
        elif len(hash_module) != len(self.in_keys):
            raise ValueError(
                "The number of hash_modules must match the number of in_keys. "
                f"Got {len(hash_module)} hash modules but {len(in_keys)} in_keys."
            )
        if hash_to_int is None:
            hash_to_int = HashToInt()

        self.aggregator = aggregator
        self.hash_module = dict(zip(self.in_keys, hash_module))
        self.hash_to_int = hash_to_int

        self.index_key = index_key
        self.clone = clone

    def forward(
        self,
        tensordict: TensorDictBase,
        extend: bool = True,
        write_hash: bool = True,
    ) -> TensorDictBase:
        hash_values = []

        for k in self.in_keys:
            hash_values.append(self.hash_module[k](tensordict.get(k)))
        if len(self.in_keys) > 1:
            hash_values = torch.stack(
                hash_values,
                dim=-1,
            )
            hash_values = self.aggregator(hash_values)
        else:
            hash_values = hash_values[0]

        td_hash_value = self.hash_to_int(hash_values, extend=extend)

        if self.clone:
            output = tensordict.copy()
        else:
            output = tensordict

        output.set(self.index_key, td_hash_value)
        if write_hash:
            output.set(self.hash_key, hash_values)
        return output
