# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import math
from abc import abstractmethod
from enum import Enum

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


class MCTSScores(Enum):
    PUCT = functools.partial(PUCTScore, c=5)  # AlphaGo default value
    UCB = functools.partial(UCBScore, c=math.sqrt(2))  # default from Auer et al. 2002
    UCB1_TUNED = "UCB1-Tuned"
    EXP3 = "EXP3"
    PUCT_VARIANT = "PUCT-Variant"
