# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchrl
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data.map import MCTSForest, Tree
from torchrl.envs import EnvBase


class MCTS(nn.Module):
    """Monte-Carlo tree search.

    Attributes:
        num_traversals (int): Number of times to traverse the tree.
        rollout_max_steps (int): Maximum number of steps for each rollout.

    Methods:
        forward: Runs the tree search.
    """

    def __init__(
        self,
        num_traversals: int,
        rollout_max_steps: int | None = None,
    ):
        super().__init__()
        self.num_traversals = num_traversals
        self.rollout_max_steps = rollout_max_steps

    def forward(
        self,
        forest: MCTSForest,
        root: TensorDictBase,
        env: EnvBase,
    ) -> Tree:
        """Performs Monte-Carlo tree search in an environment.

        Args:
            forest (MCTSForest): Forest of the tree to update. If the tree does not
                exist yet, it is added.
            root (TensorDict): The root step of the tree to update.
            env (EnvBase): Environment to performs actions in.
        """
        for action in env.all_actions(root):
            td = env.step(env.reset(root.clone()).update(action))
            forest.extend(td.unsqueeze(0))

        tree = forest.get_tree(root)

        tree.wins = torch.zeros_like(td["next", env.reward_key])
        for subtree in tree.subtree:
            subtree.wins = torch.zeros_like(td["next", env.reward_key])

        for _ in range(self.num_traversals):
            self._traverse_MCTS_one_step(forest, tree, env, self.rollout_max_steps)

        return tree

    def _traverse_MCTS_one_step(self, forest, tree, env, rollout_max_steps):
        done = False
        trees_visited = [tree]

        while not done:
            if tree.subtree is None:
                td_tree = tree.rollout[-1]["next"].clone()

                if (tree.visits > 0 or tree.parent is None) and not td_tree["done"]:
                    actions = env.all_actions(td_tree)
                    subtrees = []

                    for action in actions:
                        td = env.step(env.reset(td_tree).update(action))
                        new_node = torchrl.data.Tree(
                            rollout=td.unsqueeze(0),
                            node_data=td["next"].select(*forest.node_map.in_keys),
                            count=torch.tensor(0),
                            wins=torch.zeros_like(td["next", env.reward_key]),
                        )
                        subtrees.append(new_node)

                    # NOTE: This whole script runs about 2x faster with lazy stack
                    # versus eager stack.
                    tree.subtree = TensorDict.lazy_stack(subtrees)
                    chosen_idx = torch.randint(0, len(subtrees), ()).item()
                    rollout_state = subtrees[chosen_idx].rollout[-1]["next"]

                else:
                    rollout_state = td_tree

                if rollout_state["done"]:
                    rollout_reward = rollout_state[env.reward_key]
                else:
                    rollout = env.rollout(
                        max_steps=rollout_max_steps,
                        tensordict=rollout_state,
                    )
                    rollout_reward = rollout[-1]["next", env.reward_key]
                done = True

            else:
                priorities = self._traversal_priority_UCB1(tree)
                chosen_idx = torch.argmax(priorities).item()
                tree = tree.subtree[chosen_idx]
                trees_visited.append(tree)

        for tree in trees_visited:
            tree.visits += 1
            tree.wins += rollout_reward

    # TODO: Allow user to specify different priority functions with PR #2358
    def _traversal_priority_UCB1(self, tree):
        subtree = tree.subtree
        visits = subtree.visits
        reward_sum = subtree.wins

        # If it's black's turn, flip the reward, since black wants to optimize for
        # the lowest reward, not highest.
        # TODO: Need a more generic way to do this, since not all use cases of MCTS
        # will be two player turn based games.
        if not subtree.rollout[0, 0]["turn"]:
            reward_sum = -reward_sum

        parent_visits = tree.visits
        reward_sum = reward_sum.squeeze(-1)
        C = 2.0**0.5
        priority = (reward_sum + C * torch.sqrt(torch.log(parent_visits))) / visits
        priority[visits == 0] = float("inf")
        return priority
