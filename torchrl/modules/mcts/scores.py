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

class MCTSScores(Enum):
    PUCT = functools.partial(PUCTScore, c=5)  # AlphaGo default value
    UCB = functools.partial(UCBScore, c=math.sqrt(2))  # default from Auer et al. 2002
    UCB1_TUNED = "UCB1-Tuned"
    EXP3 = functool.partial(EXP3Score, gamma=0.1)
    PUCT_VARIANT = "PUCT-Variant"
