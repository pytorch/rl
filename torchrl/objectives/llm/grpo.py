# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
from tensordict import (
    is_tensor_collection,
    NestedKey,
    TensorClass,
    TensorDict,
    TensorDictBase,
)
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
)
from tensordict.utils import expand_as_right
from torch import distributions as d
from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.envs.transforms.transforms import Transform
from torchrl.modules.llm import LLMWrapperBase
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce, _sum_td_features


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


class GRPOLossOutput(LLMLossOutput):
    """GRPO Loss Output."""


class DAPOLossOutput(LLMLossOutput):
    """DAPO Loss Output."""


class CISPOLossOutput(LLMLossOutput):
    """CISPO Loss Output."""


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
        if isinstance(clip_epsilon, (tuple, list)):
            if len(clip_epsilon) != 2:
                raise ValueError(
                    f"clip_epsilon tuple must have length 2, got {clip_epsilon}."
                )
            eps_low, eps_high = clip_epsilon
        else:
            eps_low = float(clip_epsilon)
            eps_high = float(clip_epsilon)
        # Basic validation
        if eps_low < 0 or eps_high < 0:
            raise ValueError(
                f"clip_epsilon values must be non-negative, got ({eps_low}, {eps_high})."
            )
        if eps_low >= 1.0:
            raise ValueError(
                f"clip_epsilon low must be < 1 (to keep 1 - eps_low > 0), got {eps_low}."
            )
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
            #  dists are always defined over the whole sequence, so we can re-use the mask as the dist will always
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

        td_out.set("ESS", _reduce(ess / batch, self.reduction))
        # Aggregate loss terms according to aggregation strategy
        for key in list(td_out.keys()):
            if isinstance(key, tuple) or not isinstance(key, str):
                continue
            if key.startswith("loss_"):
                val = td_out.get(key)
                td_out.set(key, self._aggregate_loss_value(val, mask))
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
        self, value: torch.Tensor, mask: torch.Tensor
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
        return _reduce(value, reduction="mean", mask=mask).squeeze(-1)

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

    When writing on a replay buffer, this transform keeps track of the existing trajectories with a similar
    initial prompt and holds a queue for that particular prompt in memory.
    When that queue hits a certain length, the advantage is computed by normalizing the rewards across all the
    steps of all the trajectories.

    This transform assumes that :meth:`~torchrl.data.ReplayBuffer.add` and :meth:`~torchrl.data.ReplayBuffer.extend`
    are executed with completed trajectories (i.e., trajectories that end up with a done state). If this is not the
    case, an exception is raised.

    .. warning:: This transform will flatten the input tensordicts and therefore is not compatible yet with replay
        buffers hosting storages of more than one dimension.

    Args:
        grpo_size (int): Number of trajectories to keep in memory for the advantage computation.
        prompt_key (NestedKey): Key to the prompt in the tensordict. Defaults to ("text", "prompt").
        rewards_key (NestedKey): Key to the rewards in the tensordict. Defaults to ("next", "reward").
        advantage_key (NestedKey): Key to the advantage in the tensordict. Defaults to "advantage".
        done_key (NestedKey): Key to the done state in the tensordict. Defaults to ("next", "done").
        verbose (bool): Whether to print verbose information. Defaults to `False`.

    """

    def __init__(
        self,
        grpo_size: int,
        prompt_key: NestedKey = "query",
        rewards_key: NestedKey = ("next", "reward"),
        advantage_key: NestedKey = "advantage",
        done_key: NestedKey = ("next", "done"),
        verbose: bool = False,
    ):
        super().__init__()
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.out_keys = [advantage_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.advantage_key = advantage_key
        self.done_key = done_key
        self.queues = defaultdict(lambda: deque(maxlen=grpo_size))
        self.grpo_size = grpo_size
        self.verbose = verbose

    def forward(self, tensordict: TensorDictBase) -> GRPOLossOutput:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
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
                tensordicts = tensordict.split(splits)
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts) if tensordicts else None
            # Then we have a single trajectory
            if not tensordict[-1][self.done_key].all():
                raise RuntimeError("Expected the trajectory to be done.")
            prompt = tensordict[0][self.prompt_key]
            if not isinstance(prompt, str):
                raise TypeError(f"Expected a string as prompt, got {type(prompt)=}")
            self.queues[prompt].append(tensordict)
            if len(self.queues[prompt]) == self.grpo_size:
                if self.verbose:
                    torchrl_logger.info(f"Computing advantage for {prompt=}")
                # Cat is the most robust way to combine the trajs
                tds = torch.cat(list(self.queues[prompt]), -1)
                del self.queues[prompt]
                # Collect rewards
                reward = tds.get(self.rewards_key, as_nested_tensor=True)
                reward_mean = reward.values().mean()
                reward_scale = reward.values().std()
                advantage = (reward - reward_mean) / reward_scale.clamp_min(1e-6)
                if self.verbose:
                    torchrl_logger.info(f"Advantage: {reward_mean=} {reward_scale=}")
                tds.set(self.advantage_key, advantage)
                return tds
            return
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
            return torch.cat(result, 0)
        return
