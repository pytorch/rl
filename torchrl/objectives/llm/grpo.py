# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import importlib.util
import multiprocessing as mp
import os

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from math import comb
from typing import Literal, TypeVar

import torch
from tensordict import (
    is_tensor_collection,
    lazy_stack,
    NestedKey,
    TensorClass,
    TensorDict,
    TensorDictBase,
)
from tensordict.base import _is_leaf_nontensor
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
)
from tensordict.utils import expand_as_right
from torch import distributions as d
from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.envs.transforms.ray_service import _maybe_clear_device, _maybe_to_device
from torchrl.envs.transforms.transforms import Transform
from torchrl.modules.llm import LLMWrapperBase
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _sum_td_features, _validate_clip_epsilon

_has_ray = importlib.util.find_spec("ray") is not None


class LLMLossOutput(TensorClass["nocast"]):
    """Base class for LLM loss outputs.

    This base class defines the common structure for all LLM-based policy optimization
    loss outputs (GRPO, DAPO, CISPO, etc.).
    """

    loss_objective: torch.Tensor
    clip_fraction: torch.Tensor
    kl_approx: torch.Tensor
    ESS: torch.Tensor
    entropy: torch.Tensor | None = None
    loss_entropy: torch.Tensor | None = None
    loss_kl_to_ref: torch.Tensor | None = None
    kl_to_ref: torch.Tensor | None = None
    loss_kl_to_inference: torch.Tensor | None = None
    kl_to_inference: torch.Tensor | None = None


LLMOutputType = TypeVar("LLMOutputType", bound=LLMLossOutput)

_MCADVANTAGE_STATS = (
    "completed_trajectories",
    "completed_decisions",
    "trajectory_return_sum",
    "trajectory_return_max",
    "successful_trajectories",
    "completed_groups",
    "written_groups",
    "dropped_groups",
    "rescued_groups",
    "selected_trajectories",
    "unselected_trajectories",
)


def _make_mcadvantage_queue(candidate_group_size: int) -> deque:
    return deque(maxlen=candidate_group_size)


def _get_ray():
    if not _has_ray:
        raise RuntimeError(
            "RayMCAdvantage requires ray, but ray is not installed. "
            "Install it with `pip install ray`."
        )
    import ray

    return ray


class _MCAdvantageSharedQueues:
    def __init__(
        self,
        manager,
        candidate_group_size: int,
        initial_queues: dict | None = None,
    ) -> None:
        self.candidate_group_size = int(candidate_group_size)
        self._queues = manager.dict()
        if initial_queues is not None:
            for group, queue in initial_queues.items():
                self._queues[group] = list(queue)[-self.candidate_group_size :]

    def append(self, group, tensordict: TensorDictBase) -> None:
        queue = list(self._queues.get(group, ()))
        queue.append(tensordict)
        self._queues[group] = queue[-self.candidate_group_size :]

    def values(self) -> list[list[TensorDictBase]]:
        return [list(queue) for queue in self._queues.values()]

    def list(self, group) -> list[TensorDictBase]:
        return list(self._queues[group])

    def clear(self) -> None:
        self._queues.clear()

    def __bool__(self) -> bool:
        return bool(len(self))

    def __contains__(self, group) -> bool:
        return group in self._queues

    def __delitem__(self, group) -> None:
        del self._queues[group]

    def __getitem__(self, group) -> list[TensorDictBase]:
        return self.list(group)

    def __len__(self) -> int:
        return len(self._queues)


class GRPOLossOutput(LLMLossOutput):
    """GRPO Loss Output."""


class DAPOLossOutput(LLMLossOutput):
    """DAPO Loss Output."""


class CISPOLossOutput(LLMLossOutput):
    """CISPO Loss Output."""


class MCAdvantageSelector:
    """Select trajectories from an oversampled Monte-Carlo advantage group.

    ``MCAdvantage`` can collect more candidate trajectories for a group than
    the number used for the GRPO update. This selector chooses the subset that
    should be written to storage and trained on. The default ``"balanced"``
    strategy keeps the historical behavior when there is no oversampling, and
    when oversampling is enabled it tries to pick a subset whose mean return
    lies inside the dynamic-sampling bounds.

    Args:
        strategy (str, optional): Selection strategy. ``"first"`` selects the
            first ``group_size`` candidates, matching the non-oversampled
            behavior. ``"uniform"`` sorts candidates by return and samples
            roughly uniformly across that order. ``"balanced"`` searches for
            a subset that passes ``keep_return_bounds`` and is closest to the
            middle of the accepted interval. Defaults to ``"balanced"``.
        max_combinations (int, optional): Maximum exact combinations to score
            for ``"balanced"`` selection. Larger candidate pools fall back to
            a deterministic greedy strategy. Defaults to ``100_000``.
        in_keys (list of NestedKey, optional): Candidate keys consumed by the
            selector. Defaults to ``["return"]``. ``MCAdvantage`` passes a
            candidate-level tensordict with one entry per candidate trajectory,
            containing ``"return"`` and a lazy-stacked ``"trajectories"``
            tensordict with the full candidate trajectories. Subclasses can
            set this argument and override :meth:`select` to implement custom
            metadata- or trajectory-based selection.

    Examples:
        >>> import torch
        >>> from torchrl.objectives.llm import MCAdvantageSelector
        >>> from tensordict import TensorDict
        >>> selector = MCAdvantageSelector()
        >>> selector.select(
        ...     TensorDict({"return": torch.tensor([0.0, 0.0, 0.0, 1.0])}, [4]),
        ...     group_size=2,
        ...     keep_return_bounds=(0.1, 0.9),
        ... )
        [0, 3]
    """

    def __init__(
        self,
        strategy: Literal["first", "uniform", "balanced"] = "balanced",
        *,
        max_combinations: int = 100_000,
        in_keys: list[NestedKey] | None = None,
    ) -> None:
        if strategy not in ("first", "uniform", "balanced"):
            raise ValueError(
                "strategy must be one of 'first', 'uniform' or 'balanced', "
                f"got {strategy!r}."
            )
        if max_combinations < 1:
            raise ValueError(
                f"max_combinations must be strictly positive, got {max_combinations}."
            )
        if in_keys is None:
            in_keys = ["return"]
        elif not in_keys:
            raise ValueError("in_keys must contain at least one key.")
        self.strategy = strategy
        self.max_combinations = int(max_combinations)
        self.in_keys = list(in_keys)

    def select(
        self,
        candidates: TensorDictBase,
        *,
        group_size: int,
        keep_return_bounds: tuple[float, float] | None = None,
    ) -> list[int] | None:
        """Select candidate indices.

        Args:
            candidates (TensorDictBase): Candidate-level tensordict with one
                entry per candidate trajectory. The default selector reads the
                first ``in_keys`` entry as a scalar value per candidate.
            group_size (int): Number of trajectories to select.
            keep_return_bounds (tuple of float, optional): Accepted exclusive
                mean-return interval. If supplied and no valid subset is found,
                ``None`` is returned.

        Returns:
            list[int] or None: Selected candidate indices, or ``None`` when the
                candidate group should be skipped.
        """
        values = self._values(candidates)
        candidate_count = len(values)
        if candidate_count < group_size:
            raise ValueError(
                f"Need at least {group_size} candidates, got {candidate_count}."
            )
        if candidate_count == group_size:
            indices = list(range(candidate_count))
            return (
                indices if self._accepted(values, indices, keep_return_bounds) else None
            )
        if self.strategy == "first":
            indices = list(range(group_size))
            return (
                indices if self._accepted(values, indices, keep_return_bounds) else None
            )
        if self.strategy == "uniform" or keep_return_bounds is None:
            indices = self._uniform_indices(values, group_size)
            return (
                indices if self._accepted(values, indices, keep_return_bounds) else None
            )
        return self._balanced_indices(values, group_size, keep_return_bounds)

    @staticmethod
    def _as_float_list(values: torch.Tensor) -> list[float]:
        return [float(value) for value in values.detach().reshape(-1).cpu()]

    def _values(self, candidates: TensorDictBase) -> list[float]:
        key = self.in_keys[0]
        values = candidates.select(key, strict=True).get(key)
        candidate_count = candidates.numel()
        if values.numel() != candidate_count:
            raise ValueError(
                "The built-in MCAdvantageSelector strategies expect one scalar "
                f"value per candidate under {key!r}, got {values.shape=} for "
                f"{candidates.batch_size=}."
            )
        return self._as_float_list(values)

    @staticmethod
    def _accepted(
        values: list[float],
        indices: list[int],
        keep_return_bounds: tuple[float, float] | None,
    ) -> bool:
        if keep_return_bounds is None:
            return True
        low, high = keep_return_bounds
        mean_return = sum(values[index] for index in indices) / len(indices)
        return low < mean_return < high

    @staticmethod
    def _uniform_indices(values: list[float], group_size: int) -> list[int]:
        ordered = sorted(range(len(values)), key=values.__getitem__)
        if group_size == 1:
            return [ordered[len(ordered) // 2]]
        last = len(ordered) - 1
        return [
            ordered[round(position * last / (group_size - 1))]
            for position in range(group_size)
        ]

    def _balanced_indices(
        self,
        values: list[float],
        group_size: int,
        keep_return_bounds: tuple[float, float],
    ) -> list[int] | None:
        low, high = keep_return_bounds
        target = 0.5 * (low + high)
        candidate_count = len(values)
        if comb(candidate_count, group_size) <= self.max_combinations:
            best_combo = None
            best_score = None
            for combo in combinations(range(candidate_count), group_size):
                selected = [values[index] for index in combo]
                mean_return = sum(selected) / group_size
                if low < mean_return < high:
                    spread = max(selected) - min(selected)
                    score = (abs(mean_return - target), -spread)
                    if best_score is None or score < best_score:
                        best_combo = combo
                        best_score = score
            return list(best_combo) if best_combo is not None else None

        # Deterministic fallback for large candidate pools: start with values
        # spread across the sorted returns, then swap greedily until the mean
        # enters the accepted interval.
        indices = self._uniform_indices(values, group_size)
        if self._accepted(values, indices, keep_return_bounds):
            return indices
        selected = set(indices)
        ordered = sorted(range(candidate_count), key=values.__getitem__)
        current = sum(values[index] for index in indices) / group_size
        if current <= low:
            replacements = reversed(ordered)
            replace_order = sorted(indices, key=values.__getitem__)
        else:
            replacements = iter(ordered)
            replace_order = sorted(indices, key=values.__getitem__, reverse=True)
        for new_index in replacements:
            if new_index in selected:
                continue
            if not replace_order:
                break
            old_index = replace_order.pop(0)
            selected.remove(old_index)
            selected.add(new_index)
            indices[indices.index(old_index)] = new_index
            if self._accepted(values, indices, keep_return_bounds):
                return indices
        return None


class GRPOLoss(LossModule):
    """GRPO loss.

    The clipped importance weighted loss is computed as follows::

        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (LLMWrapperBase): policy operator.

    .. note::
        It is critical to keep your model in eval mode during GRPO training to ensure deterministic behavior and correct
        importance sampling. A mismatch between train and eval modes is a common cause of instability or failure to learn
        in RL post-training.

    .. note::
        The Effective Sample Size (ESS) is a key diagnostic metric in GRPO. ESS measures the effective number of samples
        in the batch, computed as the inverse of the sum of the squared importance weights.
        A value of 1 indicates that all importance weights are equal (ideal case). If ESS drops or increases significantly,
        it usually indicates a problem with the model configuration, such as a train/eval mode mismatch or a large policy update.

    .. note::
        The masking_strategy parameter is crucial for LLM training scenarios. It determines which tokens are included
        in the loss computation:
        - "sft": Only response tokens (excludes prompt tokens) - suitable for single-turn conversations
        - "rlhf": Only assistant tokens (excludes user/system tokens) - suitable for multi-turn conversations
        - "generic": All valid tokens (excludes padding tokens) - suitable for generic scenarios

        The masking strategy must match the strategy used for advantage computation to avoid shape mismatches.

    Keyword Args:
        clip_epsilon (float | tuple[float, float], optional): clipping threshold(s) for the clipped surrogate.
            - float x: symmetric clipping [1 - x, 1 + x] (default: 0.2)
            - tuple (eps_low, eps_high): asymmetric clipping [1 - eps_low, 1 + eps_high] as in DAPO Clip-Higher
              recommended defaults from DAPO: (0.20, 0.28); see Eq. (10) in the paper.
        kl_mask_threshold (float | None, optional): enable token-wise trust-region filtering (KL-Mask).
            When set, tokens with 0.5 * (log(pi_theta/pi_ref))^2 > kl_mask_threshold are masked out from the loss.
            This stabilizes updates by skipping tokens that drifted too far from the reference distribution
            (see table and description; enables per-token trust region).
        aggregation (str, optional): loss aggregation strategy for the policy objective.
            - "token_mean": global masked token mean (weights long sequences more). Default.
            - "prompt_mean": per-sample masked mean over tokens, then mean across samples (equal sample weight).
            - "none": return per-token loss (mask applied, no aggregation). Useful for downstream custom reductions.
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coeff (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction with respect to the input tensordict value estimate and use it to
            calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
            helping stabilize training and preventing large updates. However, it will have no impact if the value
            estimate was done by the current version of the value estimator. If instead ``True`` is provided, the
            ``clip_epsilon`` parameter will be used as the clipping threshold. If not provided or ``False``, no
            clipping will be performed. Defaults to ``False``.
        kl_to_ref_coeff (float, optional): coefficient for the KL divergence to the reference policy. Defaults to ``None`` (no KL divergence).
        kl_to_inference_coeff (float, optional): coefficient for the KL divergence to the inference policy. Defaults to ``None`` (no KL divergence).
        device (torch.device, optional): device of the buffers. Defaults to ``None``.
        masking_strategy (Literal["sft", "rlhf", "generic"], optional): The masking strategy to use for distribution creation.
            - "sft": Use prompt masking (response tokens only, suitable for single-turn)
            - "rlhf": Use assistant masking (assistant tokens only, suitable for multi-turn)
            - "generic": Use attention masking (all valid tokens)
            Defaults to "sft" since we can't guarantee assistant masks are available.

            .. note:: Parameters and buffers from the policy / critic will not be cast to that device to ensure that
                the storages match the ones that are passed to other components, such as data collectors.

    .. note:: For non-symmetric clipping thresholds, see the `DAPO <https://arxiv.org/html/2503.14476>`_ paper.

    """

    actor_network: LLMWrapperBase
    output_type: type[LLMLossOutput] = GRPOLossOutput

    @dataclass
    class _AcceptedKeys(LossModule._AcceptedKeys):
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values
        """

        advantage: NestedKey = "advantage"
        action: NestedKey = ("tokens", "full")
        sample_log_prob: NestedKey = ("log_probs", "full")
        ref_log_probs: NestedKey = ("next", "ref_log_probs", "full")

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        """Access the tensordict key configuration for this loss.

        This property provides access to the configurable keys used by the loss module
        to read tensors from input TensorDicts. These keys include:

        - ``advantage``: key for the advantage values
        - ``action``: key for the action tokens (default: ``("tokens", "full")``)
        - ``sample_log_prob``: key for the log probabilities from the reference policy (default: ``("log_probs", "full")``)
        - ``ref_log_probs``: key for the reference policy log probabilities (default: ``("next", "ref_log_probs", "full")``)

        To modify these keys, use the :meth:`~.set_keys` method.

        Examples:
            >>> loss = GRPOLoss(actor_network)
            >>> # Access current keys
            >>> print(loss.tensor_keys.advantage)  # "advantage"
            >>> # Modify keys
            >>> loss.set_keys(advantage="my_advantage_key")
            >>> print(loss.tensor_keys.advantage)  # "my_advantage_key"

        Returns:
            An instance of _AcceptedKeys containing all configurable tensordict keys.
        """
        return self._tensor_keys

    _schedulable_buffers = frozenset({"clip_epsilon_low", "clip_epsilon_high"})

    def __init__(
        self,
        actor_network: LLMWrapperBase | None = None,
        *,
        clip_epsilon: float | tuple[float, float] = 0.2,
        kl_mask_threshold: float | None = None,
        aggregation: str | None = "token_mean",
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coeff: float = 0.01,
        gamma: float | None = None,
        reduction: str | None = None,
        clip_value: bool | float | None = None,
        kl_to_ref_coeff: float | None = None,
        kl_to_inference_coeff: float | None = None,
        device: torch.device | None = None,
        masking_strategy: Literal["sft", "rlhf", "generic"] = "sft",
        **kwargs,
    ):
        super().__init__()
        # Core modules and hyper-parameters
        self.actor_network = actor_network
        self.entropy_bonus = entropy_bonus
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_coeff = entropy_coeff
        self.reduction = reduction if reduction is not None else "mean"
        self.kl_mask_threshold = kl_mask_threshold
        self.aggregation = aggregation or "token_mean"

        # Determine device and register clip epsilon as buffer
        if device is None:
            try:
                device = next(self.parameters()).device
            except (AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()
        # Accept symmetric or asymmetric thresholds
        eps_low, eps_high = _validate_clip_epsilon(clip_epsilon)
        # Register buffers
        self.register_buffer("clip_epsilon_low", torch.tensor(eps_low, device=device))
        self.register_buffer("clip_epsilon_high", torch.tensor(eps_high, device=device))

        self.masking_strategy = masking_strategy
        # Defaults for keys
        self.set_keys(sample_log_prob=("log_probs", "full"), action=("tokens", "full"))
        # KL coefficients
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self.kl_to_inference_coeff = kl_to_inference_coeff
        # Prepare IO keys
        self._set_in_keys()

    @property
    def _clip_bounds(self):
        # Returns (log(1 - eps_low), log(1 + eps_high)) for clamping log-weight
        return (
            (-self.clip_epsilon_low).log1p(),
            self.clip_epsilon_high.log1p(),
        )

    def _set_in_keys(self):
        keys = []
        if getattr(self, "actor_network", None) is not None and hasattr(
            self.actor_network, "in_keys"
        ):
            in_keys = self.actor_network.in_keys
            if isinstance(in_keys, (list, tuple)):
                keys.extend(in_keys)
        keys.append(self.tensor_keys.action)
        keys.append(self.tensor_keys.sample_log_prob)
        keys.append(self.tensor_keys.advantage)
        keys.append(self.tensor_keys.ref_log_probs)
        self._in_keys = list(dict.fromkeys(keys))

    @property
    def in_keys(self):
        if getattr(self, "_in_keys", None) is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if getattr(self, "_out_keys", None) is None:
            keys = ["loss_objective", "clip_fraction", "ESS", "kl_approx"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            keys.extend(
                [
                    "loss_kl_to_ref",
                    "kl_to_ref",
                    "loss_kl_to_inference",
                    "kl_to_inference",
                ]
            )
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        # No value estimator in GRPO; simply refresh input keys
        self._set_in_keys()

    def _get_cur_log_prob(self, tensordict):
        """Override to use LLM-specific distribution with explicit masking strategy.

        This ensures that the loss is computed with the correct masking strategy,
        and provides helpful error messages when there are shape mismatches.
        """
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # Use the specified masking strategy
            #  dists are always defined over the whole sequence, so we can reuse the mask as the dist will always
            #  be a MaskedCategorical
            # TODO: eventually, we want to always use `get_dist` and just pass the key of the mask
            #  Masks should contain: prompt and response masks, assistant, and attention.
            #  Additionally, we should make sure that the masks are properly updated when log-probs is called (using vllm and transformers)
            #  because in some instances it looks like they can be overwritten with None values.
            if self.masking_strategy == "sft" and hasattr(
                self.actor_network, "_get_sft_dist"
            ):
                dist = self.actor_network._get_sft_dist(tensordict)
            elif self.masking_strategy == "rlhf" and hasattr(
                self.actor_network, "_get_rlhf_dist"
            ):
                dist = self.actor_network._get_rlhf_dist(tensordict)
            elif self.masking_strategy == "generic" and hasattr(
                self.actor_network, "_get_generic_dist"
            ):
                dist = self.actor_network._get_generic_dist(tensordict)
            elif hasattr(self.actor_network, "get_dist"):
                # Fallback to generic distribution method
                dist = self.actor_network.get_dist(
                    tensordict,
                    logits_key="logits",
                )
            else:
                raise NotImplementedError(
                    f"Actor network must have get_dist method or the appropriate method for "
                    f"masking strategy '{self.masking_strategy}'."
                )

            action = tensordict.get(
                self.tensor_keys.action,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=-100,
            )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs (to compute "
                "the PPO objective) or the distribution (for the PPO entropy), please augment "
                f"the {type(self).__class__} by implementing your own logic in _get_cur_log_prob."
            )
        return log_prob, dist, False

    def forward(self, tensordict: TensorDictBase) -> LLMOutputType:
        # Some sanity checks and housekeeping:
        # - We may not have the tokens yet. If not, we will use the tokenizer of the actor to tokenize the text.
        #   We default to history rather than text because the history will account for multiturn, or multimodal inputs.
        if self.tensor_keys.action not in tensordict:
            raise ValueError(f"Action key {self.tensor_keys.action} not in tensordict.")

        tensordict = tensordict.copy()
        advantage = tensordict.get(
            self.tensor_keys.advantage, None, as_padded_tensor=True
        )
        if advantage is None:
            raise ValueError(
                f"Advantage key {self.tensor_keys.advantage} not in tensordict."
            )
        log_weight, dist, kl_approx = self._log_weight(
            tensordict, adv_shape=advantage.shape[:-1]
        )
        mask = dist.mask

        # Optional per-token trust-region filtering (KL-Mask) vs reference policy
        if self.kl_mask_threshold is not None and self.kl_mask_threshold > 0:
            try:
                inference_log_prob = tensordict.get(
                    self.tensor_keys.sample_log_prob,
                    as_padded_tensor=True,
                    padding_side="left",
                    padding_value=0.0,
                )
            except KeyError:
                inference_log_prob = None
            cur_log_prob = tensordict.get("_cur_log_prob", None)
            if (inference_log_prob is not None) and (cur_log_prob is not None):
                # Align to valid tokens only (safety)
                cur_log_prob_masked = torch.where(
                    expand_as_right(mask, cur_log_prob), cur_log_prob, 0.0
                )
                inference_log_prob_masked = torch.where(
                    expand_as_right(mask, inference_log_prob), inference_log_prob, 0.0
                )
                log_is_ref = cur_log_prob_masked - inference_log_prob_masked
                kl_token = 0.5 * (log_is_ref**2)
                tr_mask = kl_token <= self.kl_mask_threshold
                # Combine with attention mask
                mask = mask & tr_mask
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still, it can give an idea of the weights'
            # dispersion.
            lw = log_weight.squeeze(-1)[mask]
            batch = mask.sum()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()

        if advantage.ndim != log_weight.ndim:
            raise ValueError(
                f"advantage and log_weight must have the same number of dimensions, got {advantage.ndim=} and {log_weight.ndim=}"
            )
        loss_objective, clip_fraction = self._compute_policy_objective(
            log_weight, advantage
        )
        td_out = TensorDict({"loss_objective": loss_objective})
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging

        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                # Reports the entropy of each action head.
                td_out.set("composite_entropy", entropy.detach())
                entropy = _sum_td_features(entropy)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        td_out.set("ESS", ess / batch)
        # Aggregate loss terms according to aggregation strategy
        for key in list(td_out.keys()):
            if isinstance(key, tuple) or not isinstance(key, str):
                continue
            if key.startswith("loss_"):
                val = td_out.get(key)
                td_out.set(
                    key, self._aggregate_loss_value(val, mask, tensordict=tensordict)
                )
        if self.kl_to_ref_coeff is not None and self.kl_to_ref_coeff > 0:
            # FIXME: parameterize this
            loss_kl, kl_penalty = self._kl_to_ref(
                tensordict,
                mask=mask,
                dist=dist,
                ref_log_prob=tensordict.get(
                    self.tensor_keys.ref_log_probs,
                    as_padded_tensor=True,
                    padding_side="left",
                    padding_value=0.0,
                ),
            )
            td_out["loss_kl_to_ref"] = loss_kl
            td_out["kl_to_ref"] = kl_penalty.detach()
        if self.kl_to_inference_coeff is not None:
            loss_kl, kl_penalty = self._kl_to_ref(
                tensordict,
                key=self.tensor_keys.sample_log_prob,
                coeff=self.kl_to_inference_coeff,
                mask=mask,
                dist=dist,
            )
            td_out["loss_kl_to_inference"] = loss_kl
            td_out["kl_to_inference"] = kl_penalty.detach()
        del tensordict["_cur_log_prob"]
        return self.output_type.from_tensordict(td_out)

    def _compute_policy_objective(
        self, log_weight: torch.Tensor, advantage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Default GRPO objective: PPO-style min between unclipped and clipped ratios.

        Returns (loss_objective, clip_fraction).
        """
        gain1 = log_weight.exp() * advantage
        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values
        return -gain, clip_fraction

    def _aggregate_loss_value(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> torch.Tensor:
        """Aggregate a per-token loss tensor using the configured strategy.

        Supports:
            - token_mean: masked mean across all tokens (default)
            - prompt_mean: per-sample masked mean over tokens, then mean across batch
            - none: return per-token loss with masked-out tokens set to 0

        The input `value` is expected to have shape [..., T, 1] where T is the token dimension,
        and `mask` has shape [..., T].
        """
        if self.aggregation == "none" or self.reduction == "none":
            mask_exp = expand_as_right(mask, value)
            return torch.where(mask_exp, value, value.new_zeros(()).expand_as(value))

        if self.aggregation == "prompt_mean":
            # Mean over valid tokens per sample, then mean across batch
            mask_exp = expand_as_right(mask, value).to(value.dtype)
            token_sum = (value * mask_exp).sum(dim=-2, keepdim=False)
            token_count = mask_exp.sum(dim=-2, keepdim=False).clamp_min(1.0)
            sample_mean = token_sum / token_count
            return sample_mean.mean(dim=0, keepdim=False)

        # token_mean (global masked mean)
        mask_exp = expand_as_right(mask, value)
        return self._reduce_loss(
            value, tensordict=tensordict, reduction="mean", mask=mask_exp
        ).squeeze(-1)

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        try:
            entropy = dist.entropy()
            if not entropy.isfinite().all():
                del entropy
                if VERBOSE:
                    torchrl_logger.info(
                        "Entropy is not finite. Using Monte Carlo sampling."
                    )
                raise NotImplementedError
        except NotImplementedError:
            if VERBOSE:
                torchrl_logger.warning(
                    f"Entropy not implemented for {type(dist)} or is not finite. Using Monte Carlo sampling."
                )
            if getattr(dist, "has_rsample", False):
                x = dist.rsample((self.samples_mc_entropy,))
            else:
                x = dist.sample((self.samples_mc_entropy,))
            with set_composite_lp_aggregate(False) if isinstance(
                dist, CompositeDistribution
            ) else contextlib.nullcontext():
                log_prob = dist.log_prob(x)
                if is_tensor_collection(log_prob):
                    if isinstance(self.tensor_keys.sample_log_prob, NestedKey):
                        log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
                    else:
                        log_prob = log_prob.select(*self.tensor_keys.sample_log_prob)
            entropy = -log_prob.mean(0)
        if is_tensor_collection(entropy) and entropy.batch_size != adv_shape:
            entropy.batch_size = adv_shape
        return entropy.unsqueeze(-1)

    def _kl_to_ref(
        self,
        tensordict: TensorDictBase,
        key: NestedKey = ("next", "ref_log_probs"),
        ref_log_prob: torch.Tensor | None = None,
        coeff: float | None = None,
        mask: torch.Tensor | None = None,
        dist: d.Distribution | None = None,
    ):
        if coeff is None:
            coeff = self.kl_to_ref_coeff
        # TODO: customize this
        if ref_log_prob is None:
            ref_log_prob = tensordict.get(
                key,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=0.0,
            )
            if ref_log_prob is None:
                raise KeyError(
                    f"Couldn't find the ref log-prob {key} in the input data ({tensordict.keys(True)=})."
                )
            ref_log_prob = ref_log_prob.squeeze(-1)
        cur_log_prob = tensordict.get("_cur_log_prob")
        # TODO: remove this
        if cur_log_prob.shape != ref_log_prob.shape:
            raise ValueError(
                f"cur_log_prob and ref_log_prob must have the same shape, got {cur_log_prob.shape=} and {ref_log_prob.shape=}"
            )
        if mask is not None:
            ref_log_prob = torch.where(
                expand_as_right(mask, ref_log_prob), ref_log_prob, 0.0
            )
            cur_log_prob = torch.where(
                expand_as_right(mask, cur_log_prob), cur_log_prob, 0.0
            )
        diff = ref_log_prob - cur_log_prob
        kl_penalty = (diff.expm1() - diff).mean()
        return coeff * kl_penalty, kl_penalty

    def _log_weight(
        self, tensordict: TensorDictBase, adv_shape: torch.Size
    ) -> tuple[torch.Tensor, d.Distribution, torch.Tensor]:

        cur_log_prob, dist, is_composite = self._get_cur_log_prob(tensordict)

        prev_log_prob = tensordict.get(
            self.tensor_keys.sample_log_prob,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0.0,
        )

        if prev_log_prob is None:
            raise KeyError(
                f"Couldn't find the log-prob {self.tensor_keys.sample_log_prob} in the input data."
            )
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        # Check for shape mismatches and provide helpful error messages
        if cur_log_prob.shape != prev_log_prob.shape:
            # Try to provide helpful debugging information
            error_msg = (
                f"Shape mismatch detected in GRPOLoss: current log-prob shape {cur_log_prob.shape} "
                f"!= previous log-prob shape {prev_log_prob.shape}. "
                f"This usually indicates a mismatch between the masking strategy used for "
                f"advantage computation and the masking strategy used for loss computation.\n"
                f"Current masking strategy: '{self.masking_strategy}'\n"
                f"Possible solutions:\n"
                f"1. If using RLHF (multi-turn conversations), set masking_strategy='rlhf'\n"
                f"2. If using SFT (single-turn conversations), set masking_strategy='sft'\n"
                f"3. If using generic scenarios, set masking_strategy='generic'\n"
                f"4. Ensure the advantage was computed with the same masking strategy as the loss"
            )
            raise ValueError(error_msg)

        attention_mask = dist.mask
        cur_log_prob = torch.where(
            expand_as_right(attention_mask, cur_log_prob), cur_log_prob, 0.0
        )
        prev_log_prob = torch.where(
            expand_as_right(attention_mask, prev_log_prob), prev_log_prob, 0.0
        )

        if is_composite:
            raise NotImplementedError
        log_weight = (cur_log_prob - prev_log_prob).unsqueeze(-1)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(adv_shape).unsqueeze(-1)

        kl_approx = (prev_log_prob - cur_log_prob).unsqueeze(-1)
        if is_tensor_collection(kl_approx):
            kl_approx = _sum_td_features(kl_approx)

        tensordict.set("_cur_log_prob", cur_log_prob)

        return log_weight, dist, kl_approx


class DAPO(GRPOLoss):
    """DAPO (Clip-Higher over GRPO).

    Validates asymmetric clip thresholds; recommended (0.20, 0.28), see Eq. (10) in
    the `DAPO <https://arxiv.org/html/2503.14476>`_ paper.
    """

    output_type: type[LLMLossOutput] = DAPOLossOutput

    def __init__(
        self,
        tensordict: TensorDictBase,
        key: NestedKey = ("next", "ref_log_prob"),
        ref_log_prob: torch.Tensor | None = None,
        coeff: float | None = None,
        mask: torch.Tensor | None = None,
        dist: d.Distribution | None = None,
    ):
        if coeff is None:
            coeff = self.kl_to_ref_coeff
        # TODO: customize this
        if ref_log_prob is None:
            ref_log_prob = tensordict.get(
                key,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=0.0,
            )
            if ref_log_prob is None:
                raise KeyError(
                    f"Couldn't find the ref log-prob {key} in the input data ({tensordict.keys(True)=})."
                )
            ref_log_prob = ref_log_prob.squeeze(-1)
        cur_log_prob = tensordict.get("_cur_log_prob")
        # TODO: remove this
        if cur_log_prob.shape != ref_log_prob.shape:
            raise ValueError(
                f"cur_log_prob and ref_log_prob must have the same shape, got {cur_log_prob.shape=} and {ref_log_prob.shape=}"
            )
        if mask is not None:
            ref_log_prob = torch.where(
                expand_as_right(mask, ref_log_prob), ref_log_prob, 0.0
            )
            cur_log_prob = torch.where(
                expand_as_right(mask, cur_log_prob), cur_log_prob, 0.0
            )
        diff = ref_log_prob - cur_log_prob
        kl_penalty = (diff.expm1() - diff).mean()
        return coeff * kl_penalty, kl_penalty


class CISPOLoss(GRPOLoss):
    """CISPO (Clipped Importance Sampling Policy Optimization).

    Inherits the GRPO pipeline (masking, ESS, entropy, optional KL penalties) but
    replaces the PPO-style min with a clipped-importance objective::

        loss = - clip(weight, [1 - eps_low, 1 + eps_high]) * advantage

    See the `MiniMax-M1 (CISPO) <https://arxiv.org/html/2506.13585>`_ paper.
    """

    output_type: type[LLMLossOutput] = CISPOLossOutput

    def _compute_policy_objective(
        self, log_weight: torch.Tensor, advantage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CISPO: use clipped importance weights directly
        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain = ratio * advantage
        return -gain, clip_fraction


class MCAdvantage(Transform):
    """Monte-Carlo advantage computation engine.

    When writing on a replay buffer, this transform keeps track of the existing trajectories sharing
    a group identifier (e.g. the initial prompt, or an explicit group id stamped by the collector)
    and holds a queue for that particular group in memory.
    When that queue hits a certain length, the group-relative advantage is computed and the whole
    group is written to the buffer.

    Two normalization semantics are available, selected with ``trajectory_return``:

    - ``trajectory_return=None`` (default): per-step rewards are normalized across all the steps
      of all the group's trajectories. This is the original LLM GRPO behavior, suited to dense
      per-step rewards.
    - ``trajectory_return="sum"`` / ``"max"`` / ``"mean"``: each trajectory is first reduced to a
      scalar return, the returns are normalized across the group's ``grpo_size`` trajectories
      (``(R_i - mean) / std``), and each trajectory's advantage is broadcast to all of its steps.
      This is the group-relative advantage used for RL fine-tuning over sparse trajectory-level
      rewards (e.g. a binary success signal), as in SimpleVLA-RL
      (`arXiv:2509.09674 <https://arxiv.org/abs/2509.09674>`_). It expects dense, same-shaped
      per-step reward entries within each trajectory.

    With trajectory-level returns, ``keep_return_bounds`` additionally enables DAPO-style dynamic
    sampling: a group whose mean return falls outside the exclusive ``(low, high)`` bounds (e.g.
    every rollout failed, or every rollout succeeded) carries no learning signal and is dropped
    wholesale instead of being written to the buffer. ``candidate_group_size`` can oversample more
    trajectories than ``grpo_size`` for the same group id, then select ``grpo_size`` trajectories
    whose mean return lies inside the bounds before writing them. By default,
    selection is attempted as soon as ``grpo_size`` candidates have arrived:
    if that subset is not useful yet, the transform keeps queueing candidates
    until either a useful subset is found or ``candidate_group_size`` is reached.

    This transform assumes that :meth:`~torchrl.data.ReplayBuffer.add` and :meth:`~torchrl.data.ReplayBuffer.extend`
    are executed with completed trajectories (i.e., trajectories that end up with a done state). If this is not the
    case, an exception is raised.

    When the transform is attached to a replay buffer shared across several
    writer processes, :meth:`share_memory_` is called automatically by
    :meth:`~torchrl.data.ReplayBuffer.share`. This centralizes the incomplete
    group queues and counters so that trajectories produced by different
    writers can complete one logical group. Ray replay buffers already execute
    their transform inside the replay-buffer actor; use :class:`RayMCAdvantage`
    when only the grouping transform state should be centralized in Ray.

    .. note:: Setting the environment variable
        ``TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES=1`` turns :meth:`share_memory_`
        into a no-op: the grouping queues and counters stay process-local
        instead of being moved to a multiprocessing manager. Use this to skip
        the manager round-trips when a single process writes to the replay
        buffer, or when every trajectory of a group is guaranteed to be
        produced by the same writer process. Caveat: with multiple writer
        processes, a GRPO group whose trajectories span several workers can
        never complete -- each worker only sees its local fraction of the
        group, so those trajectories are held in memory forever and never
        written to the buffer.

    .. warning:: This transform will flatten the input tensordicts and therefore is not compatible yet with replay
        buffers hosting storages of more than one dimension.

    Args:
        grpo_size (int): Number of trajectories to keep in memory for the advantage computation.
        prompt_key (NestedKey): Key to the group identifier in the tensordict. May point to a
            string (e.g. the prompt) or a tensor (e.g. an integer group id); tensor identifiers
            are grouped by value. Defaults to ``"query"``.
        rewards_key (NestedKey): Key to the rewards in the tensordict. Defaults to ("next", "reward").
        advantage_key (NestedKey): Key to the advantage in the tensordict. Defaults to "advantage".
        done_key (NestedKey): Key to the done state in the tensordict. Defaults to ("next", "done").
        verbose (bool): Whether to print verbose information. Defaults to `False`.

    Keyword Args:
        trajectory_return (str, optional): if set, reduces each trajectory's rewards to a scalar
            return (``"sum"``, ``"max"`` or ``"mean"``), normalizes the returns across the group
            and broadcasts each trajectory's advantage to all of its steps. ``None`` (default)
            keeps the per-step normalization.
        keep_return_bounds (tuple of float, optional): exclusive ``(low, high)`` bounds on the
            group's mean return outside of which the whole group is dropped (dynamic sampling).
            Requires ``trajectory_return``. Defaults to ``None`` (no filtering).
        candidate_group_size (int, optional): Number of candidate trajectories to collect for each
            group id before dropping the group if no useful subset is found. Defaults to
            ``grpo_size``. Values greater than ``grpo_size`` require ``trajectory_return`` and
            select ``grpo_size`` candidates before writing to storage.
        candidate_selection_min_size (int, optional): Number of candidates required before
            trying to select ``grpo_size`` trajectories. Defaults to ``grpo_size``. Values larger
            than ``grpo_size`` force the transform to wait for more candidates before attempting
            selection, up to ``candidate_group_size``.
        candidate_selector (MCAdvantageSelector, optional): Strategy used to select ``grpo_size``
            trajectories from the candidate group. Defaults to ``MCAdvantageSelector()``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.objectives.llm import MCAdvantage
        >>> def traj(group_id, rewards):
        ...     T = len(rewards)
        ...     return TensorDict(
        ...         group_id=torch.full((T,), group_id),
        ...         next=TensorDict(
        ...             reward=torch.tensor(rewards).reshape(T, 1),
        ...             done=torch.tensor([False] * (T - 1) + [True]).reshape(T, 1),
        ...             batch_size=[T],
        ...         ),
        ...         batch_size=[T],
        ...     )
        >>> t = MCAdvantage(grpo_size=2, prompt_key="group_id", trajectory_return="sum")
        >>> t.inv(traj(7, [0.0, 0.0, 0.0])) is None  # waits for the full group
        True
        >>> out = t.inv(traj(7, [0.0, 0.0, 1.0]))  # group of 2 complete
        >>> out["advantage"].squeeze(-1)
        tensor([-0.7071, -0.7071, -0.7071,  0.7071,  0.7071,  0.7071])
        >>> # dynamic sampling: an all-failed group is dropped wholesale
        >>> t = MCAdvantage(
        ...     grpo_size=2,
        ...     prompt_key="group_id",
        ...     trajectory_return="sum",
        ...     keep_return_bounds=(0.1, 0.9),
        ... )
        >>> t.inv(traj(0, [0.0, 0.0])) is None
        True
        >>> t.inv(traj(0, [0.0, 0.0])) is None
        True

    """

    requires_shared_write_state = True

    def __init__(
        self,
        grpo_size: int,
        prompt_key: NestedKey = "query",
        rewards_key: NestedKey = ("next", "reward"),
        advantage_key: NestedKey = "advantage",
        done_key: NestedKey = ("next", "done"),
        verbose: bool = False,
        *,
        trajectory_return: Literal["sum", "max", "mean"] | None = None,
        keep_return_bounds: tuple[float, float] | None = None,
        candidate_group_size: int | None = None,
        candidate_selection_min_size: int | None = None,
        candidate_selector: MCAdvantageSelector | None = None,
    ):
        super().__init__()
        if trajectory_return not in (None, "sum", "max", "mean"):
            raise ValueError(
                "trajectory_return must be one of 'sum', 'max', 'mean' or None, "
                f"got {trajectory_return!r}."
            )
        if trajectory_return is not None and grpo_size < 2:
            raise ValueError(
                "trajectory_return requires grpo_size >= 2: the group-relative "
                "normalization (std over the group's returns) is undefined for "
                f"a single trajectory, got grpo_size={grpo_size}."
            )
        candidate_group_size = (
            grpo_size if candidate_group_size is None else int(candidate_group_size)
        )
        if candidate_group_size < grpo_size:
            raise ValueError(
                "candidate_group_size must be greater than or equal to grpo_size, "
                f"got {candidate_group_size=} and {grpo_size=}."
            )
        if candidate_group_size > grpo_size and trajectory_return is None:
            raise ValueError(
                "candidate_group_size > grpo_size requires trajectory_return so "
                "candidate trajectories can be selected by return."
            )
        candidate_selection_min_size = (
            grpo_size
            if candidate_selection_min_size is None
            else int(candidate_selection_min_size)
        )
        if candidate_selection_min_size < grpo_size:
            raise ValueError(
                "candidate_selection_min_size must be greater than or equal to "
                "grpo_size, got "
                f"{candidate_selection_min_size=} and {grpo_size=}."
            )
        if candidate_selection_min_size > candidate_group_size:
            raise ValueError(
                "candidate_selection_min_size must be less than or equal to "
                "candidate_group_size, got "
                f"{candidate_selection_min_size=} and {candidate_group_size=}."
            )
        if keep_return_bounds is not None:
            if trajectory_return is None:
                raise ValueError(
                    "keep_return_bounds (dynamic sampling) filters on trajectory-level "
                    "returns: set trajectory_return to 'sum', 'max' or 'mean'."
                )
            if (
                len(keep_return_bounds) != 2
                or not keep_return_bounds[0] < keep_return_bounds[1]
            ):
                raise ValueError(
                    "keep_return_bounds must be an increasing (low, high) pair, "
                    f"got {keep_return_bounds}."
                )
            keep_return_bounds = (
                float(keep_return_bounds[0]),
                float(keep_return_bounds[1]),
            )
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.out_keys = [advantage_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.advantage_key = advantage_key
        self.done_key = done_key
        self.grpo_size = grpo_size
        self.candidate_group_size = candidate_group_size
        self.candidate_selection_min_size = candidate_selection_min_size
        self.queues = defaultdict(
            partial(_make_mcadvantage_queue, candidate_group_size)
        )
        self.candidate_selector = (
            MCAdvantageSelector() if candidate_selector is None else candidate_selector
        )
        self.verbose = verbose
        self.trajectory_return = trajectory_return
        self.keep_return_bounds = keep_return_bounds
        self._stats = {}
        self._state_lock = contextlib.nullcontext()
        self._shared_manager = None
        self.reset_stats()

    @property
    def is_shared(self) -> bool:
        """Whether grouping queues and counters are backed by shared state."""
        return isinstance(self.queues, _MCAdvantageSharedQueues)

    def share_memory_(self) -> MCAdvantage:
        """Move replay-buffer write state to multiprocessing-shared objects."""
        if self.is_shared:
            return self
        if os.environ.get("TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES") == "1":
            torchrl_logger.info(
                "MCAdvantage.share_memory_ skipped because "
                "TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES=1: grouping queues stay "
                "process-local, so GRPO groups whose trajectories span several "
                "writer processes will never complete."
            )
            return self
        manager = mp.Manager()
        queues = {group: list(queue) for group, queue in self.queues.items()}
        self.queues = _MCAdvantageSharedQueues(
            manager, self.candidate_group_size, queues
        )
        self._stats = manager.dict(dict(self._stats))
        self._state_lock = manager.RLock()
        self._shared_manager = manager
        return self

    def __getstate__(self):
        state = super().__getstate__()
        state["_shared_manager"] = None
        return state

    def _get_stat(self, name: str):
        return self._stats[name]

    def _set_stat(self, name: str, value) -> None:
        self._stats[name] = value

    def reset_stats(self) -> None:
        """Reset counters tracking replay-buffer write decisions."""
        with self._state_lock:
            self.completed_trajectories = 0
            self.completed_decisions = 0
            self.trajectory_return_sum = 0.0
            self.trajectory_return_max = float("-inf")
            self.successful_trajectories = 0
            self.completed_groups = 0
            self.written_groups = 0
            self.dropped_groups = 0
            self.rescued_groups = 0
            self.selected_trajectories = 0
            self.unselected_trajectories = 0

    def clear_queues(self) -> None:
        """Clear incomplete Monte-Carlo trajectory groups."""
        with self._state_lock:
            self.queues.clear()

    def get_stats(self) -> dict[str, float | int]:
        """Return a serializable snapshot of counters and pending queues."""
        with self._state_lock:
            stats = {name: self._get_stat(name) for name in _MCADVANTAGE_STATS}
            stats.update(
                queued_groups=self.queued_groups,
                queued_trajectories=self.queued_trajectories,
                max_queued_trajectories_per_group=(
                    self.max_queued_trajectories_per_group
                ),
            )
            return stats

    @property
    def queued_groups(self) -> int:
        """Number of incomplete groups currently held in memory."""
        return len(self.queues)

    @property
    def queued_trajectories(self) -> int:
        """Number of incomplete trajectories currently held in memory."""
        return sum(len(queue) for queue in self.queues.values())

    @property
    def max_queued_trajectories_per_group(self) -> int:
        """Largest number of incomplete trajectories queued for one group."""
        return max((len(queue) for queue in self.queues.values()), default=0)

    def _queue_append(self, group, tensordict: TensorDictBase) -> None:
        if isinstance(self.queues, _MCAdvantageSharedQueues):
            self.queues.append(group, tensordict)
        else:
            self.queues[group].append(tensordict)

    def _queue_list(self, group) -> list[TensorDictBase]:
        if isinstance(self.queues, _MCAdvantageSharedQueues):
            return self.queues.list(group)
        return list(self.queues[group])

    def _queue_delete(self, group) -> None:
        del self.queues[group]

    def forward(self, tensordict: TensorDictBase) -> GRPOLossOutput:
        return tensordict

    @staticmethod
    def _concrete_if_possible(tensordict: TensorDictBase) -> TensorDictBase:
        """Materialize lazy inputs unless their tensor leaves are ragged.

        Ragged leaves (e.g. variable-length token sequences in a lazy stack)
        cannot be stacked into a contiguous tensordict; returning the lazy
        input unchanged is always safe, so any ``RuntimeError`` raised by
        ``contiguous()`` falls back to it rather than pattern-matching the
        (version-dependent) error message.
        """
        try:
            out = tensordict.contiguous()
        except RuntimeError as err:
            torchrl_logger.debug(
                f"MCAdvantage: keeping lazy tensordict; contiguous() failed with: {err}"
            )
            return tensordict
        # contiguous() can silently turn stacked NonTensorData entries (e.g.
        # string prompts) into empty TensorDicts. Keep the lazy input whenever
        # materialization would lose entries.
        lazy_keys = set(tensordict.keys(True, True, is_leaf=_is_leaf_nontensor))
        concrete_keys = set(out.keys(True, True, is_leaf=_is_leaf_nontensor))
        if lazy_keys - concrete_keys:
            torchrl_logger.debug(
                "MCAdvantage: keeping lazy tensordict; contiguous() dropped "
                f"entries {lazy_keys - concrete_keys}"
            )
            return tensordict
        return out

    @staticmethod
    def _cat_tensordicts(
        tensordicts: list[TensorDictBase], dim: int = 0
    ) -> TensorDictBase:
        """Concatenate trajectory TensorDicts with a concrete output type.

        ``MCAdvantage`` may receive view/lazy TensorDicts from replay storage and
        plain TensorDicts from direct collector writes. Queued and returned
        trajectory groups should nevertheless have a consistent concrete
        representation, both to avoid backend-specific lazy-stack assumptions
        and to keep replay-buffer storage writes predictable.
        """
        return torch.cat(
            [MCAdvantage._concrete_if_possible(td) for td in tensordicts], dim
        )

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        if self.verbose:
            torchrl_logger.info(
                f"Invoking MCAdvantage.\nData size: {tensordict.shape}.\nCurrent queue size: {len(self.queues)}.\nTotal queue content: {sum(len(q) for q in self.queues.values())}"
            )
        # Tensordict can be any number of dims, but it must contain entire trajectories
        if tensordict.ndim == 1:
            # Check how many done states we have
            num_done = tensordict[self.done_key].sum()
            if num_done > 1:
                done_idx = tensordict[self.done_key].nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                # TensorDict.split accepts int or list of ints, not tensors
                tensordicts = tensordict.split(splits.tolist())
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return self._cat_tensordicts(tensordicts) if tensordicts else None
            # Then we have a single trajectory
            with self._state_lock:
                return self._inv_call_single(tensordict)
        elif tensordict.ndim > 2:
            # keep the time dim at the end
            tensordict = tensordict.flatten(0, -2)
        trajs = tensordict.unbind(0)
        # Iterate over the trajectories
        result = []
        for traj in trajs:
            td_out = self._inv_call(traj)
            if td_out is None:
                continue
            result.append(td_out)
        if result:
            return self._cat_tensordicts(result, 0)
        return

    def _inv_call_single(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        """Process a single complete trajectory.

        This method is called under ``self._state_lock`` so queue mutation,
        candidate selection and stat updates are serialized across shared
        replay-buffer writers.
        """
        if tensordict.ndim != 1:
            raise RuntimeError(
                f"Expected a single flat trajectory, got {tensordict.shape}."
            )
        if not tensordict[-1][self.done_key].all():
            raise RuntimeError("Expected the trajectory to be done.")
        tensordict = self._concrete_if_possible(tensordict)
        group = tensordict[0][self.prompt_key]
        if isinstance(group, torch.Tensor):
            # tensor group identifiers (e.g. an integer group id stamped
            # by the collector) are grouped by value
            group = (
                group.item()
                if group.numel() == 1
                else tuple(group.reshape(-1).tolist())
            )
        elif not isinstance(group, str):
            raise TypeError(
                f"Expected a string or tensor as group identifier, got {type(group)=}"
            )
        self.completed_trajectories += 1
        self.completed_decisions += tensordict.numel()
        reward = None
        if self.trajectory_return is not None:
            reward = tensordict.get(self.rewards_key, None)
        if reward is not None:
            trajectory_return = float(reward.sum())
            self.trajectory_return_sum += trajectory_return
            self.trajectory_return_max = max(
                self.trajectory_return_max, trajectory_return
            )
            self.successful_trajectories += int(trajectory_return > 0.0)
        self._queue_append(group, tensordict)
        queue_len = len(self.queues[group])
        if self.trajectory_return is not None:
            if queue_len < self.candidate_selection_min_size:
                return
            if self.verbose:
                torchrl_logger.info(
                    "Trying trajectory-level advantage for %s with %d/%d "
                    "candidate trajectories.",
                    group,
                    queue_len,
                    self.candidate_group_size,
                )
            trajs = self._queue_list(group)
            tds = self._trajectory_advantage(trajs)
            if tds is not None:
                self._queue_delete(group)
                self.completed_groups += 1
                self.written_groups += 1
                return tds
            if queue_len == self.candidate_group_size:
                self._queue_delete(group)
                self.completed_groups += 1
                self.dropped_groups += 1
            return
        if queue_len == self.candidate_group_size:
            if self.verbose:
                torchrl_logger.info(f"Computing advantage for {group=}")
            trajs = self._queue_list(group)
            self._queue_delete(group)
            self.completed_groups += 1
            # Cat is the most robust way to combine the trajs
            tds = self._cat_tensordicts(trajs, -1)
            # Collect rewards. Same-length trajectories concatenate into a
            # regular strided tensor rather than a nested one.
            reward = tds.get(self.rewards_key, as_nested_tensor=True)
            reward_values = reward.values() if reward.is_nested else reward
            reward_mean = reward_values.mean()
            reward_scale = reward_values.std()
            advantage = (reward - reward_mean) / reward_scale.clamp_min(1e-6)
            if self.verbose:
                torchrl_logger.info(f"Advantage: {reward_mean=} {reward_scale=}")
            tds.set(self.advantage_key, advantage)
            self.written_groups += 1
            return tds

    def _trajectory_returns(self, rewards: list[torch.Tensor]) -> torch.Tensor:
        if self.trajectory_return == "sum":
            return torch.stack([reward.sum() for reward in rewards])
        if self.trajectory_return == "max":
            return torch.stack([reward.max() for reward in rewards])
        return torch.stack([reward.mean() for reward in rewards])

    def _trajectory_advantage(
        self, trajs: list[TensorDictBase]
    ) -> TensorDictBase | None:
        # Reduce each trajectory to a scalar return, normalize across the
        # selected group and broadcast each trajectory's advantage to all of
        # its steps.
        rewards = [traj.get(self.rewards_key) for traj in trajs]
        returns = self._trajectory_returns(rewards)
        candidates = TensorDict(
            {
                "return": returns,
                "trajectories": lazy_stack(trajs, 0),
            },
            batch_size=[len(trajs)],
        )
        selected_indices = self.candidate_selector.select(
            candidates.select(*self.candidate_selector.in_keys, strict=True),
            group_size=self.grpo_size,
            keep_return_bounds=self.keep_return_bounds,
        )
        if selected_indices is None:
            if self.keep_return_bounds is not None and self.verbose:
                low, high = self.keep_return_bounds
                torchrl_logger.info(
                    "Dropping candidate group: no subset of %d/%d trajectories "
                    "has mean return inside (%s, %s).",
                    self.grpo_size,
                    len(trajs),
                    low,
                    high,
                )
            return None
        candidate_count = len(trajs)
        if candidate_count > self.grpo_size:
            first_indices = list(range(self.grpo_size))
            first_selection_kept = self.candidate_selector._accepted(
                MCAdvantageSelector._as_float_list(returns),
                first_indices,
                self.keep_return_bounds,
            )
            rescued = not first_selection_kept
            self.rescued_groups += int(rescued)
            self.unselected_trajectories += candidate_count - self.grpo_size
        trajs = [trajs[index] for index in selected_indices]
        rewards = [rewards[index] for index in selected_indices]
        returns = returns[selected_indices]
        self.selected_trajectories += len(trajs)
        if self.keep_return_bounds is not None:
            low, high = self.keep_return_bounds
            mean_return = float(returns.mean())
            if not low < mean_return < high:
                # dynamic sampling: a degenerate group (e.g. all failed or
                # all succeeded) carries no learning signal
                if self.verbose:
                    torchrl_logger.info(
                        f"Dropping group: mean return {mean_return} outside ({low}, {high})."
                    )
                return None
        advantage = (returns - returns.mean()) / returns.std().clamp_min(1e-6)
        if self.verbose:
            torchrl_logger.info(f"Group returns: {returns=} {advantage=}")
        for traj, reward, adv in zip(trajs, rewards, advantage.unbind(0)):
            traj.set(self.advantage_key, adv.expand(reward.shape).clone())
        return self._cat_tensordicts(trajs, -1)


class _MCAdvantageRayActor:
    def __init__(self, *args, **kwargs) -> None:
        self.transform = MCAdvantage(*args, **kwargs)

    def _inv_call(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        if tensordict is None:
            return None
        return self.transform._inv_call(tensordict)

    def inv(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        if tensordict is None:
            return None
        return self.transform.inv(tensordict)

    def forward(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        if tensordict is None:
            return None
        return self.transform.forward(tensordict)

    def clear_queues(self) -> None:
        self.transform.queues.clear()

    def queued_groups(self) -> int:
        return self.transform.queued_groups

    def queued_trajectories(self) -> int:
        return self.transform.queued_trajectories

    def max_queued_trajectories_per_group(self) -> int:
        return self.transform.max_queued_trajectories_per_group

    def queues_values(self) -> list[list[TensorDictBase]]:
        return [list(queue) for queue in self.transform.queues.values()]

    def reset_stats(self) -> None:
        self.transform.reset_stats()

    def get_stat(self, name: str):
        return getattr(self.transform, name)

    def set_stat(self, name: str, value) -> None:
        setattr(self.transform, name, value)

    def __repr__(self) -> str:
        return repr(self.transform)


class _RayMCAdvantageQueues:
    def __init__(self, transform: RayMCAdvantage) -> None:
        self._transform = transform

    def clear(self) -> None:
        self._transform._ray.get(self._transform._actor.clear_queues.remote())

    def values(self) -> list[list[TensorDictBase]]:
        return self._transform._ray.get(self._transform._actor.queues_values.remote())

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return self._transform.queued_groups


class RayMCAdvantage(Transform):
    """Ray actor-backed :class:`MCAdvantage`.

    ``RayMCAdvantage`` mirrors :class:`MCAdvantage` but stores grouping queues
    and counters in a Ray actor. This is useful when multiple writers should
    share only the Monte-Carlo advantage grouping state. If the whole replay
    buffer is already a :class:`~torchrl.data.RayReplayBuffer`, prefer passing
    ``transform_factory=MCAdvantage`` to the replay buffer: the transform then
    already runs inside the replay-buffer actor and is centralized there.

    Args:
        grpo_size (int): Number of trajectories to keep in memory for the
            advantage computation.
        prompt_key (NestedKey): Key to the group identifier in the tensordict.
            Defaults to ``"query"``.
        rewards_key (NestedKey): Key to the rewards in the tensordict.
            Defaults to ``("next", "reward")``.
        advantage_key (NestedKey): Key to the advantage in the tensordict.
            Defaults to ``"advantage"``.
        done_key (NestedKey): Key to the done state in the tensordict.
            Defaults to ``("next", "done")``.
        verbose (bool): Whether to log verbose information. Defaults to
            ``False``.

    Keyword Args:
        trajectory_return (str, optional): See :class:`MCAdvantage`.
        keep_return_bounds (tuple of float, optional): See :class:`MCAdvantage`.
        candidate_group_size (int, optional): See :class:`MCAdvantage`.
        candidate_selection_min_size (int, optional): See :class:`MCAdvantage`.
        candidate_selector (MCAdvantageSelector, optional): See
            :class:`MCAdvantage`.
        ray_init_config (dict, optional): Keyword arguments for
            :func:`ray.init` when Ray is not initialized yet.
        remote_config (dict, optional): Keyword arguments for
            :func:`ray.remote` when creating the actor. Defaults to
            ``{"num_cpus": 1}``.
        actor_name (str, optional): Ray actor name. If an actor with this name
            already exists, it is reused.

    Examples:
        >>> import importlib.util
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.objectives.llm import RayMCAdvantage
        >>> def traj(group_id, rewards):
        ...     T = len(rewards)
        ...     return TensorDict(
        ...         group_id=torch.full((T,), group_id),
        ...         next=TensorDict(
        ...             reward=torch.tensor(rewards).reshape(T, 1),
        ...             done=torch.tensor([False] * (T - 1) + [True]).reshape(T, 1),
        ...             batch_size=[T],
        ...         ),
        ...         batch_size=[T],
        ...     )
        >>> if importlib.util.find_spec("ray") is not None:
        ...     t = RayMCAdvantage(
        ...         grpo_size=2,
        ...         prompt_key="group_id",
        ...         trajectory_return="sum",
        ...         ray_init_config={"ignore_reinit_error": True},
        ...     )
        ...     t.inv(traj(0, [0.0])) is None
        ...     out = t.inv(traj(0, [1.0]))
        ...     t.close()
        ...     out["advantage"].squeeze(-1)
        tensor([-0.7071,  0.7071])

    """

    requires_shared_write_state = True

    @property
    def _ray(self):
        ray = self.__dict__.get("_ray_module", None)
        if ray is None:
            ray = _get_ray()
            self.__dict__["_ray_module"] = ray
        return ray

    @_ray.setter
    def _ray(self, value) -> None:
        self.__dict__["_ray_module"] = value

    def __init__(
        self,
        grpo_size: int,
        prompt_key: NestedKey = "query",
        rewards_key: NestedKey = ("next", "reward"),
        advantage_key: NestedKey = "advantage",
        done_key: NestedKey = ("next", "done"),
        verbose: bool = False,
        *,
        trajectory_return: Literal["sum", "max", "mean"] | None = None,
        keep_return_bounds: tuple[float, float] | None = None,
        candidate_group_size: int | None = None,
        candidate_selection_min_size: int | None = None,
        candidate_selector: MCAdvantageSelector | None = None,
        ray_init_config: dict | None = None,
        remote_config: dict | None = None,
        actor_name: str | None = None,
    ) -> None:
        super().__init__()
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.out_keys = [advantage_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.advantage_key = advantage_key
        self.done_key = done_key
        self.grpo_size = grpo_size
        self.candidate_group_size = (
            grpo_size if candidate_group_size is None else int(candidate_group_size)
        )
        self.candidate_selection_min_size = (
            grpo_size
            if candidate_selection_min_size is None
            else int(candidate_selection_min_size)
        )
        self.candidate_selector = candidate_selector
        self.verbose = verbose
        self.trajectory_return = trajectory_return
        self.keep_return_bounds = keep_return_bounds
        self._actor_name = actor_name
        self._owns_actor = actor_name is None
        self._remote_config = (
            {"num_cpus": 1} if remote_config is None else dict(remote_config)
        )
        self._ray = _get_ray()
        if not self._ray.is_initialized():
            self._ray.init(**({} if ray_init_config is None else ray_init_config))
        self._actor = self._make_actor()

    @property
    def is_shared(self) -> bool:
        """Whether grouping queues and counters are backed by shared state."""
        return True

    @property
    def queues(self) -> _RayMCAdvantageQueues:
        """Proxy exposing queue inspection and clearing on the Ray actor."""
        return _RayMCAdvantageQueues(self)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_ray_module", None)
        return state

    def _make_actor(self):
        args, kwargs = self._mcadvantage_args
        if self._actor_name is not None:
            try:
                return self._ray.get_actor(self._actor_name)
            except ValueError:
                self._owns_actor = False
        remote_actor = self._ray.remote(**self._remote_config)(_MCAdvantageRayActor)
        if self._actor_name is None:
            return remote_actor.remote(*args, **kwargs)
        return remote_actor.options(name=self._actor_name).remote(*args, **kwargs)

    @property
    def _mcadvantage_args(self):
        return (
            self.grpo_size,
            self.prompt_key,
            self.rewards_key,
            self.advantage_key,
            self.done_key,
            self.verbose,
        ), {
            "trajectory_return": self.trajectory_return,
            "keep_return_bounds": self.keep_return_bounds,
            "candidate_group_size": self.candidate_group_size,
            "candidate_selection_min_size": self.candidate_selection_min_size,
            "candidate_selector": self.candidate_selector,
        }

    def _remote_call(self, method: str, tensordict: TensorDictBase | None):
        if tensordict is None:
            return None
        device = tensordict.device
        tensordict = _maybe_clear_device(tensordict)
        result = self._ray.get(getattr(self._actor, method).remote(tensordict))
        if device is not None:
            return _maybe_to_device(result, device)
        return _maybe_clear_device(result)

    def share_memory_(self) -> RayMCAdvantage:
        """Return ``self`` because Ray already centralizes the write state."""
        return self

    def reset_stats(self) -> None:
        """Reset counters tracking replay-buffer write decisions."""
        self._ray.get(self._actor.reset_stats.remote())

    @property
    def queued_groups(self) -> int:
        """Number of incomplete groups currently held by the Ray actor."""
        return self._ray.get(self._actor.queued_groups.remote())

    @property
    def queued_trajectories(self) -> int:
        """Number of incomplete trajectories currently held by the Ray actor."""
        return self._ray.get(self._actor.queued_trajectories.remote())

    @property
    def max_queued_trajectories_per_group(self) -> int:
        """Largest number of incomplete trajectories queued for one group."""
        return self._ray.get(self._actor.max_queued_trajectories_per_group.remote())

    def _get_stat(self, name: str):
        return self._ray.get(self._actor.get_stat.remote(name))

    def _set_stat(self, name: str, value) -> None:
        self._ray.get(self._actor.set_stat.remote(name, value))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        return self._remote_call("_inv_call", tensordict)

    def close(self) -> None:
        """Terminate the anonymous Ray actor owned by this transform."""
        actor = getattr(self, "_actor", None)
        if actor is not None and self._owns_actor:
            try:
                self._ray.kill(actor)
            except (RuntimeError, ValueError):
                pass
        self._actor = None

    def __repr__(self) -> str:
        if getattr(self, "_actor", None) is None:
            return "RayMCAdvantage(actor=None)"
        return self._ray.get(self._actor.__repr__.remote())


def _mcadvantage_stat_property(name: str) -> property:
    def getter(self):
        return self._get_stat(name)

    def setter(self, value) -> None:
        self._set_stat(name, value)

    return property(getter, setter)


for _mcadvantage_stat_name in _MCADVANTAGE_STATS:
    setattr(
        MCAdvantage,
        _mcadvantage_stat_name,
        _mcadvantage_stat_property(_mcadvantage_stat_name),
    )
    setattr(
        RayMCAdvantage,
        _mcadvantage_stat_name,
        _mcadvantage_stat_property(_mcadvantage_stat_name),
    )
del _mcadvantage_stat_name
