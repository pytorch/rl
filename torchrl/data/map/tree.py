# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import LazyStackedTensorDict, tensorclass, TensorDict
from torchrl.data import ListStorage, TensorDictMap


@tensorclass
class Node:
    data: TensorDict
    children: Node | None = None


class StateTree:
    def __init__(self, data_storage=None, storage_indices=None):

        self.data_storage = data_storage

        self.storage_indices = storage_indices

    @staticmethod
    def _write_fn(new, old=None):
        if old is None:
            result = new.apply(lambda x: x.unsqueeze(-1))
        else:

            def cat(x, y):
                return torch.cat([x.view(-1), y.view(-1)], -1).unique().view(-1)

            result = old.apply(cat, new)
        return result

    def _make_storage(self, source, dest):
        self.data_storage = TensorDictMap.from_tensordict_pair(
            source,
            dest,
            in_keys=["observation", "action"],
        )

    def _make_storage_indices(self, source, dest):
        self.storage_indices = TensorDictMap.from_tensordict_pair(
            source,
            dest,
            in_keys=["observation"],
            out_keys=["_index"],
            storage_constructor=ListStorage,
            collate_fn=lambda x: TensorDict.lazy_stack(x),
            write_fn=self._write_fn,
        )

    def extend(self, rollout):
        source, dest = rollout.exclude("next"), rollout.get("next")
        if self.data_storage is None:
            self._make_storage(source, dest)
        dest[:-1] = source[1:]
        self.data_storage[source] = dest
        value = source
        if self.storage_indices is None:
            self._make_storage_indices(source, dest)
        self.storage_indices[source] = TensorDict.lazy_stack(value.unbind(0))

    def get_child(self, root):
        return self.data_storage[root]

    def get_tree(
        self,
        root,
        inplace: bool = False,
        recurse: bool = True,
        max_depth: int | None = None,
        as_tensordict: bool = False,
    ):
        if root not in self.storage_indices:
            return Node(root)
        indices = self.storage_indices[root]["_index"]
        children = self.data_storage.storage[indices]
        if not inplace:
            root = root.copy()
        if recurse:
            children = children.unbind(0)
            children = tuple(
                self.get_tree(
                    child,
                    inplace=inplace,
                    max_depth=max_depth - 1 if isinstance(max_depth, int) else None,
                )
                for child in children
            )
            children = LazyStackedTensorDict(*(child._tensordict for child in children))
            if not as_tensordict:
                children = Node.from_tensordict(children)
        if not as_tensordict:
            return Node(root, children=children)
        return TensorDict({"root": root, "children": children})

    def __len__(self):
        return len(self.data_storage)
