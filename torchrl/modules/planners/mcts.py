# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class MCTSPlanner():
    pass

class _MCTSNode:
    """Represents a node in the Monte-Carlo search tree. Each node holds a single environment state.

    Args:
        state (TensorDict): A tensordict representing the state of the node.
        n_actions (int): number of actions available at that stage.
        env (EnvBase): a stateless environment reading a state and an action
            through a step method.
        parent (_MCTSNode): a parent node.
        prev_action (int): the action that lead to this node.

    """

    def __init__(
        self,
        state: TensorDict,
        n_actions: int,
        env: EnvBase,
        parent: _MCTSNode,
        prev_action: int
                 ):
        self.state = state
        self.n_actions = n_actions
        self.env = env
        self.parent = parent
        self.children = {}
        self.prev_action = prev_action

        self._is_expanded = False
        self._n_vlosses = 0  # Number of virtual losses on this node
        self._child_N = torch.zeros([n_actions])
        self._child_W = torch.zeros([n_actions])
        # Save copy of original prior before it gets mutated by dirichlet noise
        self._original_prior = torch.zeros([n_actions])
        self._child_prior = torch.zeros([n_actions])

    def count(self):
        return self.parent._child_N[action]