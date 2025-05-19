# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torchrl
import torchrl.envs
import torchrl.modules.mcts
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs import Transform

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


class TurnBasedChess(Transform):
    def transform_observation_spec(self, obsspec):
        obsspec["agent0", "turn"] = Unbounded(dtype=torch.bool, shape=())
        obsspec["agent1", "turn"] = Unbounded(dtype=torch.bool, shape=())
        return obsspec

    def transform_reward_spec(self, reward_spec):
        reward = reward_spec["reward"].clone()
        del reward_spec["reward"]
        return Composite(
            agent0=Composite(reward=reward),
            agent1=Composite(reward=reward),
        )

    def _reset(self, _td, td):
        td["agent0", "turn"] = td["turn"]
        td["agent1", "turn"] = ~td["turn"]
        return td

    def _step(self, td, td_next):
        td_next["agent0", "turn"] = td_next["turn"]
        td_next["agent1", "turn"] = ~td_next["turn"]

        reward = td_next["reward"]
        turn = td["turn"]

        if reward == 0.5:
            reward = 0
        elif reward == 1:
            if not turn:
                reward = -reward

        td_next["agent0", "reward"] = reward
        td_next["agent1", "reward"] = -reward
        del td_next["reward"]

        return td_next


env = env.append_transform(TurnBasedChess())
env.rollout(3)

forest = torchrl.data.MCTSForest()
forest.reward_keys = env.reward_keys
forest.done_keys = env.done_keys
forest.action_keys = env.action_keys

if mask_actions:
    forest.observation_keys = [
        f"{pgn_or_fen}_hash",
        "turn",
        "action_mask",
        ("agent0", "turn"),
        ("agent1", "turn"),
    ]
else:
    forest.observation_keys = [
        f"{pgn_or_fen}_hash",
        "turn",
        ("agent0", "turn"),
        ("agent1", "turn"),
    ]


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
    agent_keys = ["agent0", "agent1"]
    mcts = torchrl.modules.mcts.MCTS(mcts_steps, rollout_steps, agent_keys=agent_keys)
    tree = mcts(forest, root, env)
    moves = []

    for subtree in tree.subtree:
        td = subtree.rollout[0]
        san = td["next", "san"]
        active_agent = agent_keys[
            torch.stack([td[agent]["turn"] for agent in agent_keys]).nonzero()
        ]
        reward_sum = subtree.wins[active_agent, "reward"]
        visits = subtree.visits
        value_avg = (reward_sum / visits).item()
        moves.append((value_avg, san))

    moves = sorted(moves, key=lambda x: -x[0])

    # print(tree.to_string(tree_format_fn))

    print("------------------")
    for value_avg, san in moves:
        print(f" {value_avg:0.02f} {san}")
    print("------------------")

    return moves[0][1]


for idx in range(3):
    print("==========")
    print(idx)
    print("==========")
    torch.manual_seed(idx)

    start_time = time.time()

    # White has M1, best move Rd8#. Any other moves lose to M2 or M1.
    fen0 = "7k/6pp/7p/7K/8/8/6q1/3R4 w - - 0 1"
    assert get_best_move(fen0, 40, 10) == "Rd8#"

    # Black has M1, best move Qg6#. Other moves give rough equality or worse.
    fen1 = "6qk/2R4p/7K/8/8/8/8/4R3 b - - 1 1"
    assert get_best_move(fen1, 40, 10) == "Qg6#"

    # White has M2, best move Rxg8+. Any other move loses.
    fen2 = "2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1"
    assert get_best_move(fen2, 600, 10) == "Rxg8+"

    # Black has M2, best move Rxg1+. Any other move loses.
    fen3 = "2r5/5R2/8/8/8/7k/5P1P/2r3QK b - - 0 1"
    assert get_best_move(fen3, 600, 10) == "Rxg1+"

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Took {total_time} s")
