# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import math
import warnings
from abc import abstractmethod
from enum import Enum

import torch

from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase
try:
    from enum import member as _enum_member
except ImportError:  # Python < 3.11
    def _enum_member(value):
        return value

class MCTSScore(TensorDictModuleBase):
    """Abstract base class for MCTS score computation modules."""

    @abstractmethod
    def forward(self, node: TensorDictBase) -> TensorDictBase:
        pass


class PUCTScore(MCTSScore):
    """Computes the PUCT (Polynomial Upper Confidence Trees) score for MCTS.

    PUCT is a widely used score in MCTS algorithms, notably in AlphaGo and AlphaZero,
    to balance exploration and exploitation. It incorporates prior probabilities from a
    policy network, encouraging exploration of actions deemed promising by the policy,
    while also considering visit counts and accumulated rewards.

    The formula used is:
    `score = (win_count / visits) + c * prior_prob * sqrt(total_visits) / (1 + visits)`

    Where:
    - `win_count`: Sum of rewards (or win counts) for the action.
    - `visits`: Visit count for the action.
    - `total_visits`: Visit count of the parent node (N).
    - `prior_prob`: Prior probability of selecting the action (e.g., from a policy network).
    - `c`: The exploration constant, controlling the trade-off between exploitation
      (first term) and exploration (second term).

    Args:
        c (float): The exploration constant.
        win_count_key (NestedKey, optional): Key for the tensor in the input `TensorDictBase`
            containing the sum of rewards (or win counts) for each action.
            Defaults to "win_count".
        visits_key (NestedKey, optional): Key for the tensor containing the visit
            count for each action. Defaults to "visits".
        total_visits_key (NestedKey, optional): Key for the tensor (or scalar)
            representing the visit count of the parent node (N). Defaults to "total_visits".
        prior_prob_key (NestedKey, optional): Key for the tensor containing the
            prior probabilities for each action. Defaults to "prior_prob".
        score_key (NestedKey, optional): Key where the calculated PUCT scores
            will be stored in the output `TensorDictBase`. Defaults to "score".

    Input Keys:
        - `win_count_key` (torch.Tensor): Tensor of shape (..., num_actions)
          or matching `visits_key`.
        - `visits_key` (torch.Tensor): Tensor of shape (..., num_actions). If an action
          has zero visits, its exploitation term (win_count / visits) will result in NaN
          if win_count is also zero, or +/-inf if win_count is non-zero. The exploration
          term will still be valid due to `(1 + visits)`.
        - `total_visits_key` (torch.Tensor): Scalar or tensor broadcastable to other inputs,
          representing the parent node's visit count.
        - `prior_prob_key` (torch.Tensor): Tensor of shape (..., num_actions) containing
          prior probabilities.

    Output Keys:
        - `score_key` (torch.Tensor): Tensor of the same shape as `visits_key`, containing
          the calculated PUCT scores.

    Example:
        ```python
        from tensordict import TensorDict
        from torchrl.modules.mcts.scores import PUCTScore

        # Create a PUCTScore instance
        puct = PUCTScore(c=1.5)

        # Define a TensorDict with required keys
        node = TensorDict(
            {
                "win_count": torch.tensor([10.0, 20.0]),
                "visits": torch.tensor([5.0, 10.0]),
                "total_visits": torch.tensor(50.0),
                "prior_prob": torch.tensor([0.6, 0.4]),
            },
            batch_size=[],
        )

        # Compute the PUCT scores
        result = puct(node)
        print(result["score"])  # Output: Tensor with PUCT scores
        ```
    """

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
        # Handle broadcasting for batched inputs
        if n_total.ndim > 0 and n_total.ndim < visits.ndim:
            n_total = n_total.unsqueeze(-1)
        node.set(
            self.score_key,
            (win_count / visits) + self.c * prior_prob * n_total.sqrt() / (1 + visits),
        )
        return node


class UCBScore(MCTSScore):
    """Computes the UCB (Upper Confidence Bound) score, specifically UCB1, for MCTS.

    UCB1 is a classic algorithm for the multi-armed bandit problem that balances
    exploration and exploitation. In MCTS, it's used to select which action to
    explore from a given node. The score encourages trying actions with high
    empirical rewards and actions that have been visited less frequently.

    The formula used is:
    `score = (win_count / visits) + c * sqrt(total_visits) / (1 + visits)`

    Args:
        c (float): The exploration constant. A common value is `sqrt(2)`.
        win_count_key (NestedKey, optional): Key for the tensor in the input `TensorDictBase`
            containing the sum of rewards (or win counts) for each action.
            Defaults to "win_count".
        visits_key (NestedKey, optional): Key for the tensor containing the visit
            count for each action. Defaults to "visits".
        total_visits_key (NestedKey, optional): Key for the tensor (or scalar)
            representing the visit count of the parent node (N). This is used in the
            exploration term. Defaults to "total_visits".
        score_key (NestedKey, optional): Key where the calculated UCB scores
            will be stored in the output `TensorDictBase`. Defaults to "score".

    Input Keys:
        - `win_count_key` (torch.Tensor): Tensor of shape (..., num_actions).
        - `visits_key` (torch.Tensor): Tensor of shape (..., num_actions).
        - `total_visits_key` (torch.Tensor): Scalar or tensor broadcastable to other inputs.

    Output Keys:
        - `score_key` (torch.Tensor): Tensor of the same shape as `visits_key`, containing
          the calculated UCB scores.

    Example:
        ```python
        from tensordict import TensorDict
        from torchrl.modules.mcts.scores import UCBScore

        # Create a UCBScore instance
        ucb = UCBScore(c=1.414)

        # Define a TensorDict with required keys
        node = TensorDict(
            {
                "win_count": torch.tensor([15.0, 25.0]),
                "visits": torch.tensor([10.0, 20.0]),
                "total_visits": torch.tensor(100.0),
            },
            batch_size=[],
        )

        # Compute the UCB scores
        result = ucb(node)
        print(result["score"])  # Output: Tensor with UCB scores
        ```
    """

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
        # Handle broadcasting for batched inputs
        if n_total.ndim > 0 and n_total.ndim < visits.ndim:
            n_total = n_total.unsqueeze(-1)
        node.set(
            self.score_key,
            (win_count / visits) + self.c * n_total.sqrt() / (1 + visits),
        )
        return node


class EXP3Score(MCTSScore):
    """Computes action selection probabilities for the EXP3 algorithm in MCTS.

    EXP3 (Exponential-weight algorithm for Exploration and Exploitation) is a bandit
    algorithm that performs well in adversarial or non-stationary environments.
    It maintains weights for each action and adjusts them based on received rewards.

    Args:
        gamma (float, optional): Exploration factor, balancing uniform exploration
            and exploitation of current weights. Must be in [0, 1]. Defaults to 0.1.
        weights_key (NestedKey, optional): Key in the input `TensorDictBase` for
            the tensor containing current action weights. Defaults to "weights".
        action_prob_key (NestedKey, optional): Key to store the calculated action
            probabilities. Defaults to "action_prob".
        score_key (NestedKey, optional): Key where the calculated action probabilities
            will be stored. Defaults to "score".
        num_actions_key (NestedKey, optional): Key for the number of available
            actions (K). Defaults to "num_actions".

    Input Keys:
        - `weights_key` (torch.Tensor): Tensor of shape (..., num_actions).
        - `num_actions_key` (int or torch.Tensor): Scalar representing K, the number of actions.

    Output Keys:
        - `score_key` (torch.Tensor): Tensor of shape (..., num_actions) containing
          the calculated action probabilities.

    Example:
        ```python
        from tensordict import TensorDict
        from torchrl.modules.mcts.scores import EXP3Score

        # Create an EXP3Score instance
        exp3 = EXP3Score(gamma=0.1)

        # Define a TensorDict with required keys
        node = TensorDict(
            {
                "weights": torch.tensor([1.0, 1.0]),
                "num_actions": torch.tensor(2),
            },
            batch_size=[],
        )

        # Compute the action probabilities
        result = exp3(node)
        print(result["score"])  # Output: Tensor with action probabilities
        ```
    """

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

        # Extract scalar value from num_actions (handles batched tensors too)
        if isinstance(num_actions, torch.Tensor):
            # For batched tensors, take the first element (all should be same)
            k = int(num_actions.flatten()[0].item())
        elif isinstance(num_actions, int):
            k = num_actions
        else:
            raise ValueError(
                f"'{self.num_actions_key}' ('num_actions') must be an integer or a tensor."
            )

        if self.weights_key not in node.keys(include_nested=True):
            batch_size = node.batch_size
            weights_shape = (*batch_size, k)
            weights = torch.ones(weights_shape, device=node.device)
            node.set(self.weights_key, weights)
        else:
            weights = node.get(self.weights_key)

        k_from_weights = weights.shape[-1]
        if k_from_weights != k:
            raise ValueError(
                f"Shape of weights {weights.shape} implies {k_from_weights} actions, "
                f"but num_actions is {k}."
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
        """Updates the weight of the chosen action based on the reward.

        This method updates the weight of the selected action using the EXP3 algorithm.
        The weight update formula is:
        `w_i(t+1) = w_i(t) * exp((gamma / K) * (reward / p_i(t)))`

        Args:
            node (TensorDictBase): The node containing the current weights and probabilities.
                Must include the keys specified by `weights_key` and `score_key`.
            action_idx (int): The index of the action that was selected.
            reward (float): The reward received for the selected action. Must be in the range [0, 1].

        Raises:
            ValueError: If the reward is not in the range [0, 1].
            ValueError: If the probability of the selected action is less than or equal to 0.

        Example:
            ```python
            from tensordict import TensorDict
            from torchrl.modules.mcts.scores import EXP3Score

            # Create an EXP3Score instance
            exp3 = EXP3Score(gamma=0.1)

            # Define a TensorDict with required keys
            node = TensorDict(
                {
                    "weights": torch.tensor([1.0, 1.0]),
                    "num_actions": torch.tensor(2),
                },
                batch_size=[],
            )

            # Compute the action probabilities
            result = exp3(node)
            print(result["score"])  # Output: Tensor with action probabilities

            # Update the weights based on the reward for action 0
            exp3.update_weights(node, action_idx=0, reward=0.8)
            print(node["weights"])  # Updated weights
            ```
        """
        if not (0 <= reward <= 1):
            warnings.warn(
                f"Reward {reward} is outside the expected [0,1] range for EXP3.",
                UserWarning,
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
            prob_i_val = prob_i.item() if prob_i.numel() == 1 else prob_i
            warnings.warn(
                f"Probability p_i(t) for action {action_idx} is {prob_i_val}. "
                "Weight will not be updated for zero probability actions.",
                UserWarning,
            )
            # Don't update weights for zero probability - just return
            return

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
    """Computes the UCB1-Tuned score for MCTS, using variance estimation.

    UCB1-Tuned is an enhancement of the UCB1 algorithm that incorporates an estimate
    of the variance of rewards for each action. This allows for a more refined
    balance between exploration and exploitation, potentially leading to better
    performance, especially when reward variances differ significantly across actions.

    The score for an action `i` is calculated as:
    `score_i = avg_reward_i + sqrt(log(N) / N_i * min(0.25, V_i))`

    The variance estimate `V_i` for action `i` is calculated as:
    `V_i = (sum_squared_rewards_i / N_i) - avg_reward_i^2 + sqrt(exploration_constant * log(N) / N_i)`

    Where:
    - `avg_reward_i`: Average reward obtained from action `i`.
    - `N_i`: Number of times action `i` has been visited.
    - `N`: Total number of times the parent node has been visited.
    - `sum_squared_rewards_i`: Sum of the squares of rewards received from action `i`.
    - `exploration_constant`: A constant used in the bias correction term of `V_i`.
      Auer et al. (2002) suggest a value of 2.0 for rewards in the range [0,1].
    - The term `min(0.25, V_i)` implies that rewards are scaled to `[0, 1]`, as 0.25 is
      the maximum variance for a distribution in this range (e.g., Bernoulli(0.5)).

    Reference: "Finite-time Analysis of the Multiarmed Bandit Problem"
    (Auer, Cesa-Bianchi, Fischer, 2002).

    Args:
        exploration_constant (float, optional): The constant `C` used in the bias
            correction term for the variance estimate `V_i`. Defaults to `2.0`,
            as suggested for rewards in `[0,1]`.
        win_count_key (NestedKey, optional): Key for the tensor in the input `TensorDictBase`
            containing the sum of rewards for each action (Q_i * N_i). Defaults to "win_count".
        visits_key (NestedKey, optional): Key for the tensor containing the visit
            count for each action (N_i). Defaults to "visits".
        total_visits_key (NestedKey, optional): Key for the tensor (or scalar)
            representing the visit count of the parent node (N). Defaults to "total_visits".
        sum_squared_rewards_key (NestedKey, optional): Key for the tensor containing
            the sum of squared rewards received for each action. This is crucial for
            calculating the empirical variance. Defaults to "sum_squared_rewards".
        score_key (NestedKey, optional): Key where the calculated UCB1-Tuned scores
            will be stored in the output `TensorDictBase`. Defaults to "score".

    Input Keys:
        - `win_count_key` (torch.Tensor): Sum of rewards for each action.
        - `visits_key` (torch.Tensor): Visit counts for each action (N_i).
        - `total_visits_key` (torch.Tensor): Parent node's visit count (N).
        - `sum_squared_rewards_key` (torch.Tensor): Sum of squared rewards for each action.

    Output Keys:
        - `score_key` (torch.Tensor): Calculated UCB1-Tuned scores for each action.

    Important Notes:
        - **Unvisited Nodes**: Actions with zero visits (`visits_key` is 0) are assigned a
          very large positive score to ensure they are selected for exploration.
        - **Reward Range**: The `min(0.25, V_i)` term is theoretically most sound when
          rewards are normalized to the range `[0, 1]`.
        - **Logarithm of N**: `log(N)` (log of parent visits) is calculated using `torch.log(torch.clamp(N, min=1.0))`
          to prevent issues with `N=0` or `N` between 0 and 1.
    """

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
            v_i_v = v_i_v.clamp(min=0)

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
    PUCT = _enum_member(functools.partial(PUCTScore, c=5))
    UCB = _enum_member(functools.partial(UCBScore, c=math.sqrt(2)))
    UCB1_TUNED = _enum_member(
        functools.partial(UCB1TunedScore, exploration_constant=2.0)
    )
    EXP3 = _enum_member(functools.partial(EXP3Score, gamma=0.1))
