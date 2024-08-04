# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum

from tensordict.nn import TensorDictModuleBase

from torchrl.data.map.tree import MCTSForest
from torchrl.envs.common import EnvBase

from torchrl.modules.mcts.scores import MCTSScores

class ExpansionStrategies(Enum):
    Exhaustive = "Exhaustive"
    Sampling = "Sampling"
    Embedding = "Embedding"


class MCTSPolicy(TensorDictModuleBase):
    rollout_kwargs = {"break_when_any_done": False}
    num_sim = 1

    def __init__(
        self,
        simulation_env: EnvBase,
        *,
        forest: MCTSForest | None = None,
        expansion_strategy: ExpansionStrategies = ExpansionStrategies.Exhaustive,
        selection_criterion: MCTSScores = MCTSScores.PUCT,
    ):
        super().__init__()
        self.env = simulation_env
        if forest is None:
            forest = MCTSForest()
        self.forest = forest
        self.expansion_strategy = expansion_strategy
        self.selection_criterion = selection_criterion

    def forward(self, node):
        # 1. Selection
        selected_node = self.select_node(node)

        # 2. Expansion: generate new child nodes for all possible responses to this move
        actions = self.get_possible_actions()

        # 3. Simulation
        node_with_actions = self.set_actions(
            selected_node, actions
        )  # Expands child to make all possible moves

        # we may want to expand the children_with_node to do more than one simulation
        if self.num_sim > 1:
            node_with_actions = node_with_actions.expand(
                self.num_sim, *node_with_actions.shape
            )
        # Get init state of rollouts (new children)
        _, reset_nodes = self.env.step_and_maybe_reset(node_with_actions)

        # Get the rollouts
        rollouts = self.env.rollout(
            max_steps=100, tensordict=reset_nodes, auto_reset=False, **self.rollout_kwargs
        )
        print(rollouts)
        # Update stats of the child_with_move
        self.update_stats(node_with_actions, rollouts)

        # 4. Backprop

    def select_node(self, node):
        return self.forest.select_node(node, criterion=self.selection_criterion)

    def get_possible_actions(self):
        if self.expansion_strategy == ExpansionStrategies.Exhaustive:
            # lists the possible moves at the node
            return self.env.full_action_spec.enumerate()
        elif self.expansion_strategy == ExpansionStrategies.Sampling:
            raise NotImplementedError
        elif self.expansion_strategy == ExpansionStrategies.Embedding:
            raise NotImplementedError
        else:
            raise NotImplementedError
    def set_actions(self, node, actions):
        return node.expand(actions.shape[0], *node.shape).update(actions)
