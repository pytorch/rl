# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import tensorclass, TensorDict

@tensorclass(autocast=True)
class MCTSChildren:
    vals: torch.Tensor
    priors: torch.Tensor
    visits: torch.Tensor
    ids: torch.Tensor | None = None
    nodes: MCTSNode | None = None

    @classmethod
    def init_from_prob(cls, probs):
        vals = torch.zeros_like(probs)
        visits = torch.zeros_like(probs)
        return cls(vals=vals, priors=probs, visits=visits)


@tensorclass(autocast=True)
class MCTSNode:
    prior_action: torch.Tensor
    _children: MCTSChildren | None = None
    score: torch.Tensor | None = None
    state: TensorDict | None = None
    terminated: torch.Tensor | None = None
    parent: MCTSNode | None = None

    @classmethod
    def from_action(
        cls,
        action: torch.Tensor,
        parent: MCTSNode | None,
    ):
        return cls(prior_action=action, parent=parent)

    @property
    def children(self) -> MCTSChildren:
        children = self._children
        if children is None:
            return MCTSChildren(*[torch.zeros((), device=self.device) for _ in range(4)])
        return children

    @children.setter
    def children(self, value):
        self._children = value

    @property
    def visits(self) -> torch.Tensor:
        assert self.parent is not None
        return self.parent.children.visits[self.prior_action]

    @visits.setter
    def visits(self, x) -> None:
        assert self.parent is not None
        self.parent.children.visits[self.prior_action] = x

    @property
    def value(self) -> torch.Tensor:
        assert self.parent is not None
        return self.parent.children.vals[self.prior_action]

    @value.setter
    def value(self, x) -> None:
        assert self.parent is not None
        self.parent.children.vals[self.prior_action] = x

    @property
    def expanded(self) -> bool:
        return self.children.ids.numel() > 0

    def get_child(self, action: torch.Tensor) -> MCTSNode:
        idx = (self.children.ids == action).all(-1)
        return self.children.nodes[idx]  # type: ignore

    @classmethod
    def root(cls) -> MCTSNode:
        return cls(torch.Tensor(-1), None)

    @classmethod
    def dummy(cls):
        """Creates a 'dummy' MCTSNode that can be used to explore TorchRL's MCTS API."""
        children_values = stuff
        children_priors = stuff
        children_visits = stuff
        children_ids = stuff
        children_nodes = stuff
        children = MCTSChildren(
            values = children_values,
            priors = children_priors,
            visits = children_visits,
            ids = children_ids,
            nodes = children_nodes,
        )
        prior_action = stuff
        score = stuff
        state = stuff
        terminated = stuff
        parent = None
        return cls(
            prior_action=prior_action,
            children=children,
            score=score,
            state=state,
            terminated=terminated,
            parent=parent,
        )
