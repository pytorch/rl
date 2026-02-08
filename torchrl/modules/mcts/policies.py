# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from abc import abstractmethod
import torch

from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.modules.mcts.scores import MCTSScore, PUCTScore

class MCTSPolicyBase(TensorDictModuleBase):
    """Abstract base class for MCTS policies.

    A policy consumes a node tensorclass/tensordict, computes a score for each action,
    and writes the selected action index in the output.

    Args:
        score_module (MCTSScore): score module used to evaluate available actions.
        action_mask_key (NestedKey, optional): key for an optional boolean mask where
            `True` marks valid actions. If absent, all actions are considered valid.
            Defaults to "action_mask".
        score_key (NestedKey, optional): key where the score module stores action scores.
            Defaults to "score".
        action_key (NestedKey, optional): key where selected action index is written.
            Defaults to "action".
    """

    def __init__(
            self,
            *,
            score_module: MCTSScore,
            action_mask_key: NestedKey = "action_mask",
            score_key: NestedKey = "score",
            action_key: NestedKey = "action",
    ) -> None:
        super().__init__()
        self.score_module = score_module
        self.action_mask_key = action_mask_key
        self.score_key = score_key
        self.action_key = action_key
        self.in_keys = list(getattr(score_module, "in_keys", []))
        if self.action_mask_key not in self.in_keys:
            self.in_keys.append(self.action_mask_key)
        self.out_keys = [self.action_key, self.score_key]

    @abstractmethod
    def select_action(self, score: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Selects one action per batch based on score and optional mask.

        Args:
            score (torch.Tensor): The scores for each action, shape (..., num_actions).
            mask (torch.Tensor | None): Optional boolean mask for valid actions, shape (..., num_actions).

        Returns:
            torch.Tensor: The selected action indices, shape (...).
        """

    def forward(self, node: TensorDictBase) -> TensorDictBase:
        """Computes scores and selects an action for the given node.

        Args:
            node (TensorDictBase): The input node containing state and action information.

        Returns:
            TensorDictBase: The updated node with the selected action and scores.
        """
        node = self.score_module(node)
        score = node.get(self.score_key)
        mask = node.get(self.action_mask_key, None)
        action = self.select_action(score, mask=mask)
        node.set(self.action_key, action)
        return node
            
class MCTSPolicy(MCTSPolicyBase):
    """Standard MCTS policy that selects the action with the highest score.

    This policy uses a score module to evaluate actions and selects the one with the maximum score,
    optionally respecting an action mask.

    Args:
        score_module (MCTSScore): Score module for action evaluation.
        action_mask_key (NestedKey, optional): Key for the action mask. Defaults to "action_mask".
        score_key (NestedKey, optional): Key for storing scores. Defaults to "score".
        action_key (NestedKey, optional): Key for the selected action. Defaults to "action".
    """

    def select_action(self, score: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.to(torch.bool)
            while mask.ndim > score.ndim and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if mask.shape != score.shape:
                mask, score = torch.broadcast_tensors(mask, score)
            if not mask.any(dim=-1).all():
                raise ValueError("action_mask must allow at least one valid action per node.")
            score = torch.where(mask, score, torch.full_like(score, -torch.inf))
        action = score.argmax(dim=-1)
        return action

class AlphaGoPolicy(MCTSPolicy):
    """AlphaGo-style MCTS policy using PUCT scoring.

    This policy implements the selection mechanism from AlphaGo, using PUCT (Predictor + Upper Confidence Bound)
    to balance exploration and exploitation.

    Args:
        c (float, optional): Exploration constant for PUCT. Defaults to 5.0.
        win_count_key (NestedKey, optional): Key for win counts. Defaults to "win_count".
        visits_key (NestedKey, optional): Key for visit counts. Defaults to "visits".
        total_visits_key (NestedKey, optional): Key for total visits. Defaults to "total_visits".
        prior_prob_key (NestedKey, optional): Key for prior probabilities. Defaults to "prior_prob".
        action_mask_key (NestedKey, optional): Key for the action mask. Defaults to "action_mask".
        score_key (NestedKey, optional): Key for storing scores. Defaults to "score".
        action_key (NestedKey, optional): Key for the selected action. Defaults to "action".
    """
    def __init__(self, *, c: float = 5.0, win_count_key: NestedKey = "win_count", visits_key: NestedKey = "visits", total_visits_key: NestedKey = "total_visits", prior_prob_key: NestedKey = "prior_prob", action_mask_key: NestedKey = "action_mask", score_key: NestedKey = "score", action_key: NestedKey = "action") -> None:
        score_module = PUCTScore(
            c=c,
            win_count_key=win_count_key,
            visits_key=visits_key,
            total_visits_key=total_visits_key,
            prior_prob_key=prior_prob_key,
            score_key=score_key,
        )

        super().__init__(
            score_module=score_module,
            action_mask_key=action_mask_key,
            score_key=score_key,
            action_key=action_key,
        )

class AlphaStarPolicy(AlphaGoPolicy):
    """AlphaStar-style MCTS policy with a lower exploration constant.

    This policy is similar to AlphaGo but uses a smaller exploration constant (c=1.0) for potentially
    more exploitative behavior.

    Args:
        c (float, optional): Exploration constant. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to AlphaGoPolicy.
    """
    def __init__(self, *, c: float = 1.0, **kwargs) -> None:
        super().__init__(c=c, **kwargs)

class MuZeroPolicy(AlphaGoPolicy):
    """MuZero-style MCTS policy with a specific exploration constant.

    This policy implements the selection from MuZero, using PUCT with c=1.25.

    Args:
        c (float, optional): Exploration constant. Defaults to 1.25.
        **kwargs: Additional keyword arguments passed to AlphaGoPolicy.
    """
    def __init__(self, *, c: float = 1.25, **kwargs) -> None:
        super().__init__(c=c, **kwargs)
