# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torchrl
import torchrl.envs
from tensordict import TensorDict

start_time = time.time()

pgn_or_fen = "fen"
mask_actions = True

env = torchrl.envs.ChessEnv(
    include_pgn=False,
    include_fen=True,
    include_hash=True,
    include_hash_inv=True,
    include_san=True,
    stateful=True,
    mask_actions=mask_actions,
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
env = env.append_transform(transform_reward)

forest = torchrl.data.MCTSForest()
forest.reward_keys = env.reward_keys
forest.done_keys = env.done_keys
forest.action_keys = env.action_keys

if mask_actions:
    forest.observation_keys = [f"{pgn_or_fen}_hash", "turn", "action_mask"]
else:
    forest.observation_keys = [f"{pgn_or_fen}_hash", "turn"]

C = 2.0**0.5


def traversal_priority_UCB1(tree):
    subtree = tree.subtree
    visits = subtree.visits
    reward_sum = subtree.wins

    # If it's black's turn, flip the reward, since black wants to
    # optimize for the lowest reward, not highest.
    if not subtree.rollout[0, 0]["turn"]:
        reward_sum = -reward_sum

    parent_visits = tree.visits
    reward_sum = reward_sum.squeeze(-1)
    priority = (reward_sum + C * torch.sqrt(torch.log(parent_visits))) / visits
    priority[visits == 0] = float("inf")
    return priority


def _traverse_MCTS_one_step(forest, tree, env, max_rollout_steps):
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
                        wins=torch.zeros_like(td["next"]["reward"]),
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
                rollout_reward = rollout_state["reward"]
            else:
                rollout = env.rollout(
                    max_steps=max_rollout_steps,
                    tensordict=rollout_state,
                )
                rollout_reward = rollout[-1]["next", "reward"]
            done = True

        else:
            priorities = traversal_priority_UCB1(tree)
            chosen_idx = torch.argmax(priorities).item()
            tree = tree.subtree[chosen_idx]
            trees_visited.append(tree)

    for tree in trees_visited:
        tree.visits += 1
        tree.wins += rollout_reward


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
        for action in env.all_actions(root):
            td = env.step(env.reset(root.clone()).update(action))
            forest.extend(td.unsqueeze(0))

    tree = forest.get_tree(root)
    tree.wins = torch.zeros_like(td["next", "reward"])
    for subtree in tree.subtree:
        subtree.wins = torch.zeros_like(td["next", "reward"])

    for _ in range(num_steps):
        _traverse_MCTS_one_step(forest, tree, env, max_rollout_steps)

    return tree


def tree_format_fn(tree):
    td = tree.rollout[-1]["next"]
    return [
        td["san"],
        td[pgn_or_fen].split("\n")[-1],
        tree.wins,
        tree.visits,
    ]


def get_best_move(fen, mcts_steps, rollout_steps):
    root = env.reset(TensorDict({"fen": fen}))
    tree = traverse_MCTS(forest, root, env, mcts_steps, rollout_steps)
    moves = []

    for subtree in tree.subtree:
        san = subtree.rollout[0]["next", "san"]
        reward_sum = subtree.wins
        visits = subtree.visits
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

end_time = time.time()
total_time = end_time - start_time

print(f"Took {total_time} s")
