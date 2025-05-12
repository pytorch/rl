# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torchrl
import torchrl.envs
import torchrl.modules.mcts
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


class TransformReward:
    def __init__(self):
        self.first_turn = None

    def __call__(self, td):
        if self.first_turn is None and "turn" in td:
            self.first_turn = td["turn"]
            print(f"first turn: {self.first_turn}")

        if "reward" not in td:
            return td
        reward = td["reward"]
        if reward == 0.5:
            reward = 0
        # elif reward == 1 and td["turn"] == env.lib.WHITE:
        elif reward == 1 and td["turn"] == self.first_turn:
            reward = -reward

        td["reward"] = reward
        return td

    def reset(self, td):
        self.first_turn = None


# ChessEnv sets the reward to 0.5 for a draw and 1 for a win for either player.
# Need to transform the reward to be:
#   white win = 1
#   draw = 0
#   black win = -1
env = env.append_transform(TransformReward())

forest = torchrl.data.MCTSForest()
forest.reward_keys = env.reward_keys
forest.done_keys = env.done_keys
forest.action_keys = env.action_keys

if mask_actions:
    forest.observation_keys = [f"{pgn_or_fen}_hash", "turn", "action_mask"]
else:
    forest.observation_keys = [f"{pgn_or_fen}_hash", "turn"]


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
    tree = torchrl.modules.mcts.MCTS(forest, root, env, mcts_steps, rollout_steps)
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

    # print(tree.to_string(tree_format_fn))

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
