# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import tensorclass, TensorDict


@tensorclass(autocast=True)
class MCTSNode:
    prior_action: torch.Tensor
    parent: MCTSNode | None
    children_values: torch.Tensor
    children_priors: torch.Tensor
    children_visits: torch.Tensor
    score: torch.Tensor
    children: MCTSNode
    children_ids: torch.Tensor
    state: TensorDict
    terminated: torch.Tensor

    def __init__(
        self,
        action: torch.Tensor,
        parent: MCTSNode | None,
    ):
        self.prior_action = action
        self.parent = parent
        self.children_ids = torch.tensor([], dtype=torch.int32)

    @property
    def visits(self) -> torch.Tensor:
        assert self.parent != None
        return self.parent.children_visits[self.prior_action]

    @visits.setter
    def visits(self, x) -> None:
        assert self.parent != None
        self.parent.children_visits[self.prior_action] = x

    @property
    def value(self) -> torch.Tensor:
        assert self.parent != None
        return self.parent.children_values[self.prior_action]

    @value.setter
    def value(self, x) -> None:
        assert self.parent != None
        self.parent.children_values[self.prior_action] = x

    @property
    def expanded(self) -> bool:
        return self.children_ids.numel() > 0

    def get_child(self, action: torch.Tensor) -> MCTSNode:
        idx = (self.children_ids == action).all(-1)
        return self.children[idx]  # type: ignore

    @classmethod
    def root(cls) -> MCTSNode:
        return cls(torch.Tensor(-1), None)
