# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import math
from abc import abstractmethod
from enum import Enum

import torch

from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch import nn


class MCTSScore(TensorDictModuleBase):
    @abstractmethod
    def forward(self, node):
        pass


class PUCTScore(MCTSScore):
    c: float

    def __init__(
        self,
        *,
        c: float,
        win_count_key: NestedKey = "win_count",
        visits_key: NestedKey = "visits",
        total_visits_key: NestedKey = "total_visits",
        prior_prob_key: NestedKey = "prior_prob",
        score_key: NestedKey = "score",
    ):
        super().__init__()
        self.c = c
        self.win_count_key = win_count_key
        self.visits_key = visits_key
        self.total_visits_key = total_visits_key
        self.prior_prob_key = prior_prob_key
        self.score_key = score_key
        self.in_keys = [
            self.win_count_key,
            self.prior_prob_key,
            self.total_visits_key,
            self.visits_key,
        ]
        self.out_keys = [self.score_key]

    def forward(self, node: TensorDictBase) -> TensorDictBase:
        win_count = node.get(self.win_count_key)
        visits = node.get(self.visits_key)
        n_total = node.get(self.total_visits_key)
        prior_prob = node.get(self.prior_prob_key)
        node.set(
            self.score_key,
            (win_count / visits) + self.c * prior_prob * n_total.sqrt() / (1 + visits),
        )
        return node


class UCBScore(MCTSScore):
    c: float

    def __init__(
        self,
        *,
        c: float,
        win_count_key: NestedKey = "win_count",
        visits_key: NestedKey = "visits",
        total_visits_key: NestedKey = "total_visits",
        score_key: NestedKey = "score",
    ):
        super().__init__()
        self.c = c
        self.win_count_key = win_count_key
        self.visits_key = visits_key
        self.total_visits_key = total_visits_key
        self.score_key = score_key
        self.in_keys = [self.win_count_key, self.total_visits_key, self.visits_key]
        self.out_keys = [self.score_key]

    def forward(self, node: TensorDictBase) -> TensorDictBase:
        win_count = node.get(self.win_count_key)
        visits = node.get(self.visits_key)
        n_total = node.get(self.total_visits_key)
        node.set(
            self.score_key,
            (win_count / visits) + self.c * n_total.sqrt() / (1 + visits),
        )
        return node


class EXP3Score(MCTSScore):
    def __init__(
        self,
        *,
        gamma: float = 0.1,
        weights_key: NestedKey = "weights",
        action_prob_key: NestedKey = "action_prob",
        reward_key: NestedKey = "reward",
        score_key: NestedKey = "score",
        num_actions_key: NestedKey = "num_actions",
    ):
        super().__init__()
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
        self.gamma = gamma
        self.weights_key = weights_key
        self.action_prob_key = action_prob_key
        self.reward_key = reward_key
        self.score_key = score_key
        self.num_actions_key = num_actions_key

        self.in_keys = [self.weights_key, self.num_actions_key]
        self.out_keys = [self.score_key]

    def forward(self, node: TensorDictBase) -> TensorDictBase:
        num_actions = node.get(self.num_actions_key)

        if self.weights_key not in node.keys(include_nested=True):
            batch_size = node.batch_size
            if isinstance(num_actions, torch.Tensor) and num_actions.numel() == 1:
                k = int(num_actions.item())
            elif isinstance(num_actions, int):
                k = num_actions
            else:
                raise ValueError(
                    f"'{self.num_actions_key}' ('num_actions') must be an integer or a scalar tensor."
                )
            weights_shape = (*batch_size, k)
            weights = torch.ones(weights_shape, device=node.device)
            node.set(self.weights_key, weights)
        else:
            weights = node.get(self.weights_key)

        k = weights.shape[-1]
        if isinstance(num_actions, torch.Tensor) and num_actions.numel() == 1:
            if k != num_actions.item():
                raise ValueError(
                    f"Shape of weights {weights.shape} implies {k} actions."
                    f"but num_actions is {num_actions.item()}"
                )
        elif isinstance(num_actions, int):
            if k != num_actions:
                raise ValueError(
                    f"Shape of weights {weights.shape} implies {k} actions, "
                    f"but num_actions is {num_actions}."
                )

        sum_weights = torch.sum(weights, dim=-1, keepdim=True)
        sum_weights = torch.where(
            sum_weights == 0, torch.ones_like(sum_weights), sum_weights
        )

        p_i = (1 - self.gamma) * (weights / sum_weights) + (self.gamma / k)
        node.set(self.score_key, p_i)
        if self.action_prob_key != self.score_key:
            node.set(self.action_prob_key, p_i)
        return node

    def update_weights(
        self, node: TensorDictBase, action_idx: int, reward: float
    ) -> None:
        if not (0 <= reward <= 1):
            ValueError(
                f"Reward {reward} is outside the expected [0, 1] range for EXP3."
            )

        weights = node.get(self.weights_key)
        action_probs = node.get(self.score_key)
        k = weights.shape[-1]

        if weights.ndim == 1:
            current_weight = weights[action_idx]
            prob_i = action_probs[action_idx]
        elif weights.ndim > 1:
            current_weight = weights[..., action_idx]
            prob_i = action_probs[..., action_idx]
        else:
            raise ValueError(f"Invalid weights dimensions: {weights.ndim}")

        if torch.any(prob_i <= 0):
            ValueError(
                f"Probability p_i(t) for action {action_idx} is {prob_i}, which is <= 0."
                "This might lead to issues in weight update."
            )
            prob_i = torch.clamp(prob_i, min=1e-9)

        reward_tensor = torch.as_tensor(
            reward, device=current_weight.device, dtype=current_weight.dtype
        )
        exponent = (self.gamma / k) * (reward_tensor / prob_i)
        new_weight = current_weight * torch.exp(exponent)

        if weights.ndim == 1:
            weights[action_idx] = new_weight
        else:
            weights[..., action_idx] = new_weight
        node.set(self.weights_key, weights)


class UCB1TunedScore(MCTSScore):
    def __init__(
        self,
        *,
        win_count_key: NestedKey = "win_count",
        visits_key: NestedKey = "visits",
        total_visits_key: NestedKey = "total_visits",
        sum_squared_rewards_key: NestedKey = "sum_squared_rewards",
        score_key: NestedKey = "score",
        exploration_constant: float = 2.0,
    ):
        super().__init__()
        self.win_count_key = win_count_key
        self.visits_key = visits_key
        self.total_visits_key = total_visits_key
        self.sum_squared_rewards_key = sum_squared_rewards_key
        self.score_key = score_key
        self.exploration_constant = exploration_constant

        self.in_keys = [
            self.win_count_key,
            self.visits_key,
            self.total_visits_key,
            self.sum_squared_rewards_key,
        ]
        self.out_keys = [self.score_key]

    def forward(self, node: TensorDictBase) -> TensorDictBase:
        q_sum_i = node.get(self.win_count_key)
        n_i = node.get(self.visits_key)
        n_parent = node.get(self.total_visits_key)
        sum_sq_rewards_i = node.get(self.sum_squared_rewards_key)

        if n_parent.ndim > 0 and n_parent.ndim < q_sum_i.ndim:
            n_parent_expanded = n_parent.unsqueeze(-1)
        else:
            n_parent_expanded = n_parent

        safe_n_parent_for_log = torch.clamp(n_parent_expanded, min=1.0)
        log_n_parent = torch.log(safe_n_parent_for_log)

        scores = torch.zeros_like(q_sum_i, device=q_sum_i.device)

        visited_mask = n_i > 0

        if torch.any(visited_mask):
            q_sum_i_v = q_sum_i[visited_mask]
            n_i_v = n_i[visited_mask]
            sum_sq_rewards_i_v = sum_sq_rewards_i[visited_mask]

            log_n_parent_v = log_n_parent.expand_as(n_i)[visited_mask]

            avg_reward_i_v = q_sum_i_v / n_i_v

            empirical_variance_v = (sum_sq_rewards_i_v / n_i_v) - avg_reward_i_v.pow(2)
            bias_correction_v = (
                self.exploration_constant * log_n_parent_v / n_i_v
            ).sqrt()

            v_i_v = empirical_variance_v + bias_correction_v
            v_i_v.clamp(min=0)

            min_variance_term_v = torch.min(torch.full_like(v_i_v, 0.25), v_i_v)
            exploration_component_v = (
                log_n_parent_v / n_i_v * min_variance_term_v
            ).sqrt()

            scores[visited_mask] = avg_reward_i_v + exploration_component_v

        unvisited_mask = ~visited_mask
        if torch.any(unvisited_mask):
            scores[unvisited_mask] = torch.finfo(scores.dtype).max / 10.0

        node.set(self.score_key, scores)
        return node


class MCTSScores(Enum):
    PUCT = functools.partial(PUCTScore, c=5)  # AlphaGo default value
    UCB = functools.partial(UCBScore, c=math.sqrt(2))  # default from Auer et al. 2002
    UCB1_TUNED = functools.partial(
        UCB1TunedScore, exploration_constant=2.0
    )  # Auer et al. (2002) C=2 for rewards in [0,1]
    EXP3 = functools.partial(EXP3Score, gamma=0.1)
    PUCT_VARIANT = "PUCT-Variant"
