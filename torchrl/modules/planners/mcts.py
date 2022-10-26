# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math

import torch

from typing import Dict

from torchrl.data import TensorDict, DEVICE_TYPING
from torchrl.envs import EnvBase

# This file is inspired by https://github.com/FelixOpolka/Single-Player-MCTS
# Refactoring involves: renaming, torch and tensordict compatibility and integration in torchrl's API

class MCTSPlanner():
    pass

class _MCTSNode:
    """Represents a node in the Monte-Carlo search tree. Each node holds a single environment state.

    Reference: https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf

    Args:
        state (TensorDict): A tensordict representing the state of the node.
        n_actions (int): number of actions available at that stage.
        env (EnvBase): a stateless environment reading a state and an action
            through a step method.
        parent (_MCTSNode): a parent node.
        prev_action (int): the action that lead to this node.
        exploration_factor (float, optional): Exploration constant. Default: :obj:`1.38`.
        d_noise_alpha (float, optional): Dirichlet noise alpha parameter. Default: :obj:`0.03`.
        temp_threshold (int, optional): Number of steps into the episode after
            which we always select the action with highest action probability
            rather than selecting randomly.

    """

    def __init__(
        self,
        state: TensorDict,
        n_actions: int,
        env: EnvBase,
        parent: _MCTSNode,
        prev_action: int,
        exploration_factor: float=1.38,
        d_noise_alpha: float = 0.03,
        temp_threshold: int = 5,
        device: DEVICE_TYPING = "cpu",
    ):
        self.state = state
        self.n_actions = n_actions
        self.env = env
        self.parent = parent
        self.children = {}
        self.prev_action = prev_action
        self.exploration_factor = exploration_factor
        self.d_noise_alpha = d_noise_alpha
        self.temp_threshold = temp_threshold

        self._device = device
        self._is_expanded = False
        self._n_vlosses = 0  # Number of virtual losses on this node
        self._child_visit_count: torch.Tensor = torch.zeros([n_actions], dtype=torch.long, device=self.device)
        self._child_total_value: torch.Tensor = torch.zeros([n_actions], device=self.device)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self._original_prior = torch.zeros([n_actions], device=self.device)
        self._child_prior = torch.zeros([n_actions], device=self.device)

    @property
    def is_expanded(self):
        return self._is_expanded

    @property
    def device(self):
        return self._device

    @property
    def visit_count(self) -> int:
        return self.parent._child_visit_count[self.prev_action]

    @visit_count.setter
    def visit_count(self, value):
        self.parent._child_visit_count[self.prev_action] = value

    @property
    def total_value(self):
        return self.parent._child_total_value[self.prev_action]

    @total_value.setter
    def total_value(self, value):
        self.parent._child_total_value[self.prev_action] = value

    @property
    def action_value(self):
        return self.total_value / (1 + self.visit_count)

    @property
    def exploration_credit(self):
        """Exploration boost factor: gives a high credit to moves with few simulations."""
        return (self.exploration_factor * math.sqrt(1 + self.visit_count) *
                self._child_prior / (1 + self._child_visit_count))

    @property
    def exploitation_credit(self):
        """Exploration credit: gives a high credit to moves with a high average value."""
        return self._child_total_value / (1 + self._child_visit_count)

    @property
    def action_score(self):
        """Action score.

        Proposed in: An Adaptive Sampling Algorithm for Solving Markov Decision
        Processes. Hyeong Soo Chang, Michael C. Fu, Jiaqiao Hu, Steven I. Marcus,
        https://doi.org/10.1287/opre.1040.0145

        """
        return self.exploitation_credit + self.exploration_credit

    def select_leaf(self):
        """Finds a leaf in the MCT rooted in the current node.

        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        _child_visit_count and _child_total_value values but not as a dedicated
        node in the parent's children-mapping). Nodes are selected according to
        their action_score.

        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.

        """
        current = self
        while True:
            current.visit_count += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Choose action with highest score.
            best_move = current.child_action_score.argmax(-1)
            current = current.maybe_add_child(best_move)
        return current

    def incorporate_estimates(self, action_probs: torch.Tensor, value: torch.Tensor, up_to: int):
        """

        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.

        Args:
            action_probs (torch.Tensor): Action probabilities for the current
                node's state predicted by the neural network.
            value (torch.Tensor): Value of the current node's state predicted
                by the neural network.
            up_to (int): The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self._is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self._child_total_value = torch.ones([self.n_actions], dtype=torch.float32, device=self.device) * value
        self.backup_value(value, up_to=up_to)

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)
