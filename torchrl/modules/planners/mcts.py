# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math

import torch

from typing import Dict

from torchrl.data import TensorDict
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

        self._is_expanded = False
        self._n_vlosses = 0  # Number of virtual losses on this node
        self._child_visit_count: torch.Tensor = torch.zeros([n_actions], dtype=torch.long)
        self._child_total_value: torch.Tensor = torch.zeros([n_actions])
        # Save copy of original prior before it gets mutated by dirichlet noise
        self._original_prior = torch.zeros([n_actions])
        self._child_prior = torch.zeros([n_actions])

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

