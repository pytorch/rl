# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchrl
from tensordict import TensorDict

pgn_or_fen = "fen"

env = torchrl.envs.ChessEnv(
    include_pgn=False,
    include_fen=True,
    include_hash=True,
    include_hash_inv=True,
    include_san=True,
    stateful=True,
    mask_actions=True,
)


def transform_reward(td):
    if "reward" not in td:
        return td
    reward = td["reward"]
    if reward == 0.5:
        td["reward"] = 0
    elif reward == 1 and td["turn"]:
        td["reward"] = -td["reward"]
    return td


# ChessEnv sets the reward to 0.5 for a draw and 1 for a win for either player.
# Need to transform the reward to be:
#   white win = 1
#   draw = 0
#   black win = -1
env.append_transform(transform_reward)

forest = torchrl.data.MCTSForest()
forest.reward_keys = env.reward_keys + ["_visits", "_reward_sum"]
forest.done_keys = env.done_keys
forest.action_keys = env.action_keys
forest.observation_keys = [f"{pgn_or_fen}_hash", "turn", "action_mask"]

C = 2.0**0.5


def traversal_priority_UCB1(tree):
    if tree.rollout[-1]["next", "_visits"] == 0:
        res = float("inf")
    else:
        if tree.parent.rollout is None:
            parent_visits = 0
            for child in tree.parent.subtree:
                parent_visits += child.rollout[-1]["next", "_visits"]
        else:
            parent_visits = tree.parent.rollout[-1]["next", "_visits"]
            assert parent_visits > 0

        value_avg = (
            tree.rollout[-1]["next", "_reward_sum"]
            / tree.rollout[-1]["next", "_visits"]
        )

        # If it's black's turn, flip the reward, since black wants to optimize
        # for the lowest reward.
        if not tree.rollout[0]["turn"]:
            value_avg = -value_avg

        res = (
            value_avg
            + C
            * torch.sqrt(torch.log(parent_visits) / tree.rollout[-1]["next", "_visits"])
        ).item()

    return res


def _traverse_MCTS_one_step(forest, tree, env, max_rollout_steps):
    done = False
    trees_visited = []

    while not done:
        if tree.subtree is None:
            td_tree = tree.rollout[-1]["next"]

            if (td_tree["_visits"] > 0 or tree.parent is None) and not td_tree["done"]:
                actions = env.all_actions(td_tree)
                subtrees = []

                for action in actions:
                    td = env.step(env.reset(td_tree.clone()).update(action)).update(
                        TensorDict(
                            {
                                ("next", "_visits"): 0,
                                ("next", "_reward_sum"): env.reward_spec.zeros(),
                            }
                        )
                    )

                    new_node = torchrl.data.Tree(
                        rollout=td.unsqueeze(0),
                        node_data=td["next"].select(*forest.node_map.in_keys),
                    )
                    subtrees.append(new_node)

                tree.subtree = TensorDict.lazy_stack(subtrees)
                chosen_idx = torch.randint(0, len(subtrees), ()).item()
                rollout_state = subtrees[chosen_idx].rollout[-1]["next"]

            else:
                rollout_state = td_tree

            if rollout_state["done"]:
                rollout_reward = rollout_state["reward"]
            else:
                rollout = env.rollout(
                    max_steps=max_rollout_steps,
                    tensordict=rollout_state,
                )
                rollout_reward = rollout[-1]["next", "reward"]
            done = True

        else:
            priorities = torch.tensor(
                [traversal_priority_UCB1(subtree) for subtree in tree.subtree]
            )
            chosen_idx = torch.argmax(priorities).item()
            tree = tree.subtree[chosen_idx]
            trees_visited.append(tree)

    for tree in trees_visited:
        td = tree.rollout[-1]["next"]
        td["_visits"] += 1
        td["_reward_sum"] += rollout_reward


def traverse_MCTS(forest, root, env, num_steps, max_rollout_steps):
    """Performs Monte-Carlo tree search in an environment.

    Args:
        forest (MCTSForest): Forest of the tree to update. If the tree does not
            exist yet, it is added.
        root (TensorDict): The root step of the tree to update.
        env (EnvBase): Environment to performs actions in.
        num_steps (int): Number of iterations to traverse.
        max_rollout_steps (int): Maximum number of steps for each rollout.
    """
    if root not in forest:
        for action in env.all_actions(root.clone()):
            td = env.step(env.reset(root.clone()).update(action)).update(
                TensorDict(
                    {
                        ("next", "_visits"): 0,
                        ("next", "_reward_sum"): env.reward_spec.zeros(),
                    }
                )
            )
            forest.extend(td.unsqueeze(0))

    tree = forest.get_tree(root)

    for _ in range(num_steps):
        _traverse_MCTS_one_step(forest, tree, env, max_rollout_steps)

    return tree


def tree_format_fn(tree):
    td = tree.rollout[-1]["next"]
    return [
        td["san"],
        td[pgn_or_fen].split("\n")[-1],
        td["_reward_sum"].item(),
        td["_visits"].item(),
    ]


def get_best_move(fen, mcts_steps, rollout_steps):
    root = env.reset(TensorDict({"fen": fen}))
    tree = traverse_MCTS(forest, root, env, mcts_steps, rollout_steps)

    # print('------------------------------')
    # print(tree.to_string(tree_format_fn))
    # print('------------------------------')

    moves = []

    for subtree in tree.subtree:
        san = subtree.rollout[0]["next", "san"]
        reward_sum = subtree.rollout[-1]["next", "_reward_sum"]
        visits = subtree.rollout[-1]["next", "_visits"]
        value_avg = (reward_sum / visits).item()
        if not subtree.rollout[0]["turn"]:
            value_avg = -value_avg
        moves.append((value_avg, san))

    moves = sorted(moves, key=lambda x: -x[0])

    print("------------------")
    for value_avg, san in moves:
        print(f" {value_avg:0.02f} {san}")
    print("------------------")

    return moves[0][1]


# White has M1, best move Rd8#. Any other moves lose to M2 or M1.
fen0 = "7k/6pp/7p/7K/8/8/6q1/3R4 w - - 0 1"
assert get_best_move(fen0, 100, 10) == "Rd8#"

# Black has M1, best move Qg6#. Other moves give rough equality or worse.
fen1 = "6qk/2R4p/7K/8/8/8/8/4R3 b - - 1 1"
assert get_best_move(fen1, 100, 10) == "Qg6#"

# White has M2, best move Rxg8+. Any other move loses.
fen2 = "2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1"
assert get_best_move(fen2, 1000, 10) == "Rxg8+"
