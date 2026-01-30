# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Literal

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
from torchrl.modules.llm import LLMWrapperBase
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce, _sum_td_features


class SDPOLossOutput(TensorClass["nocast"]):
    """SDPO Loss Output.

    This class defines the output structure for Self-Distillation Policy Optimization
    (SDPO) loss computation.

    Attributes:
        loss_objective: The main policy objective loss.
        divergence: The divergence between student and teacher distributions.
        kl_approx: Approximate KL divergence for logging.
        entropy: Policy entropy (if entropy_bonus is enabled).
        loss_entropy: Entropy loss term (if entropy_bonus is enabled).
        loss_kl_to_ref: KL divergence loss to reference policy (if kl_to_ref_coeff is set).
        kl_to_ref: KL divergence to reference policy for logging.
    """

    loss_objective: torch.Tensor
    divergence: torch.Tensor
    kl_approx: torch.Tensor
    entropy: torch.Tensor | None = None
    loss_entropy: torch.Tensor | None = None
    loss_kl_to_ref: torch.Tensor | None = None
    kl_to_ref: torch.Tensor | None = None


class SDPOLoss(LossModule):
    """Self-Distillation Policy Optimization (SDPO) loss.

    SDPO is an on-policy algorithm that uses self-distillation for credit assignment
    in LLM post-training. Instead of relying on sparse scalar rewards, SDPO uses the
    model itself as a self-teacher by conditioning it on rich feedback (runtime errors,
    successful solutions, etc.).

    The core idea is that the same model can be used in two roles:
    - **Student**: The policy generating responses without seeing feedback
    - **Self-Teacher**: The same policy conditioned on feedback, which can retrospectively
      identify mistakes and assign dense, per-token credit

    The loss minimizes the KL divergence (or Jensen-Shannon divergence) between the
    student and self-teacher distributions::

        L_SDPO = sum_t KL(pi_theta(·|x, y_{<t}) || stopgrad(pi_theta(·|x, f, y_{<t})))

    where `f` is the rich feedback (environment output + optionally a successful rollout).

    Reference: "Reinforcement Learning via Self-Distillation" (Hübotter et al., 2025)
    https://arxiv.org/abs/2601.20802

    Args:
        actor_network (LLMWrapperBase): The policy network (used as both student and teacher).

    Keyword Args:
        divergence_type (str, optional): Type of divergence to use. Options:
            - "kl": KL divergence KL(student || teacher)
            - "reverse_kl": Reverse KL divergence KL(teacher || student)
            - "js": Jensen-Shannon divergence (default, recommended for stability)
        topk (int | None, optional): If set, only compute divergence over top-K logits
            from the student for memory efficiency. Defaults to ``None`` (use all logits).
        use_ema_teacher (bool, optional): Whether to use an Exponential Moving Average
            of parameters for the teacher. Helps stabilize training. Defaults to ``False``.
        ema_decay (float, optional): Decay rate for EMA teacher. Defaults to ``0.99``.
        trust_region_alpha (float | None, optional): If set, use trust-region regularization
            by interpolating between reference and current teacher log-probs.
            The teacher becomes: (1-alpha)*log_ref + alpha*log_current. Defaults to ``None``.
        entropy_bonus (bool, optional): If ``True``, add an entropy bonus to encourage
            exploration. Defaults to ``True``.
        samples_mc_entropy (int, optional): Number of samples for Monte Carlo entropy
            estimation if closed-form is unavailable. Defaults to ``1``.
        entropy_coeff (float, optional): Coefficient for the entropy bonus. Defaults to ``0.01``.
        kl_to_ref_coeff (float | None, optional): Coefficient for KL divergence penalty
            to a reference policy. Defaults to ``None`` (no KL penalty).
        aggregation (str, optional): Loss aggregation strategy:
            - "token_mean": Global masked token mean (default)
            - "prompt_mean": Per-sample mean, then mean across samples
            - "none": No aggregation
        reduction (str, optional): Final reduction: "mean", "sum", or "none".
            Defaults to ``"mean"``.
        masking_strategy (str, optional): Masking strategy for distributions:
            - "sft": Response tokens only (single-turn)
            - "rlhf": Assistant tokens only (multi-turn)
            - "generic": All valid tokens
            Defaults to ``"sft"``.
        device (torch.device | None, optional): Device for buffers. Defaults to ``None``.

    Examples:
        Basic usage:

        >>> from torchrl.objectives.llm import SDPOLoss
        >>> from torchrl.modules.llm import TransformersWrapper
        >>>
        >>> # Wrap model
        >>> actor = TransformersWrapper(model, tokenizer=tokenizer, generate=False, pad_output=True)
        >>>
        >>> # Create SDPO loss with Jensen-Shannon divergence and EMA teacher
        >>> loss_fn = SDPOLoss(
        ...     actor_network=actor,
        ...     divergence_type="js",
        ...     topk=100,
        ...     use_ema_teacher=True,
        ...     ema_decay=0.99,
        ... )
        >>>
        >>> # Compute loss (tensordict must include teacher_context)
        >>> loss_output = loss_fn(tensordict)
        >>> loss_output.loss_objective.backward()

        Complete training loop with feedback:

        >>> from torchrl.envs.llm.transforms import AddFeedbackContext
        >>> from torchrl.data import ReplayBuffer, LazyStackStorage
        >>>
        >>> # Set up replay buffer with feedback transform
        >>> rb = ReplayBuffer(storage=LazyStackStorage(1000))
        >>> rb.append_transform(AddFeedbackContext(grpo_size=4))
        >>>
        >>> # Training loop
        >>> for epoch in range(num_epochs):
        ...     # Collect rollouts with environment feedback
        ...     rollouts = collector.collect()  # Should contain "env_feedback" key
        ...     rb.extend(rollouts)
        ...
        ...     # Sample batch and compute loss
        ...     batch = rb.sample(batch_size)
        ...     loss_output = loss_fn(batch)
        ...
        ...     # Backward pass
        ...     optimizer.zero_grad()
        ...     loss_output.loss_objective.backward()
        ...     optimizer.step()
        ...
        ...     # Update EMA teacher
        ...     loss_fn.update_ema_teacher()

    Note:
        The input tensordict should contain a "teacher_context" key with the feedback-augmented
        context for the self-teacher. This can be prepared using the ``AddFeedbackContext``
        transform from ``torchrl.envs.llm.transforms``.
    """

    actor_network: LLMWrapperBase
    output_type: type[SDPOLossOutput] = SDPOLossOutput

    @dataclass
    class _AcceptedKeys(LossModule._AcceptedKeys):
        """Maintains default values for all configurable tensordict keys.

        Attributes:
            action: Key for action tokens. Defaults to ``("tokens", "full")``.
            sample_log_prob: Key for student log probabilities. Defaults to ``("log_probs", "full")``.
            teacher_context: Key for the self-teacher context (feedback-augmented prompt).
                Defaults to ``"teacher_context"``.
            ref_log_probs: Key for reference policy log probabilities (for KL penalty).
                Defaults to ``("next", "ref_log_probs", "full")``.
        """

        action: NestedKey = ("tokens", "full")
        sample_log_prob: NestedKey = ("log_probs", "full")
        teacher_context: NestedKey = "teacher_context"
        ref_log_probs: NestedKey = ("next", "ref_log_probs", "full")

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        """Access the tensordict key configuration for this loss."""
        return self._tensor_keys

    def __init__(
        self,
        actor_network: LLMWrapperBase | None = None,
        *,
        divergence_type: Literal["kl", "reverse_kl", "js"] = "js",
        topk: int | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.99,
        trust_region_alpha: float | None = None,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coeff: float = 0.01,
        kl_to_ref_coeff: float | None = None,
        aggregation: str = "token_mean",
        reduction: str | None = None,
        masking_strategy: Literal["sft", "rlhf", "generic"] = "sft",
        device: torch.device | None = None,
    ):
        super().__init__()

        self.actor_network = actor_network
        self.divergence_type = divergence_type
        self.topk = topk
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.trust_region_alpha = trust_region_alpha
        self.entropy_bonus = entropy_bonus
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_coeff = entropy_coeff
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self.aggregation = aggregation
        self.reduction = reduction if reduction is not None else "mean"
        self.masking_strategy = masking_strategy

        # Determine device
        if device is None:
            try:
                device = next(self.parameters()).device
            except (AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()
        self._device = device

        # Initialize EMA teacher parameters if requested
        if use_ema_teacher and actor_network is not None:
            self._init_ema_teacher()
        else:
            self._ema_teacher_params = None

        # Store reference teacher log-probs for trust-region (populated on first forward)
        self._ref_teacher_logprobs = None

        # Set default keys
        self.set_keys(
            sample_log_prob=("log_probs", "full"),
            action=("tokens", "full"),
        )
        self._set_in_keys()

    def _init_ema_teacher(self):
        """Initialize EMA teacher parameters as a copy of actor network parameters."""
        if hasattr(self.actor_network, "parameters"):
            # Create a deep copy of parameters for EMA
            self._ema_teacher_params = {
                name: param.detach().clone()
                for name, param in self.actor_network.named_parameters()
            }
        else:
            self._ema_teacher_params = None

    def update_ema_teacher(self):
        """Update EMA teacher parameters with current actor parameters.

        Should be called after each optimization step when using EMA teacher.
        The update rule is: theta_ema = decay * theta_ema + (1 - decay) * theta
        """
        if self._ema_teacher_params is None:
            return

        with torch.no_grad():
            for name, param in self.actor_network.named_parameters():
                if name in self._ema_teacher_params:
                    self._ema_teacher_params[name].mul_(self.ema_decay).add_(
                        param.data, alpha=1 - self.ema_decay
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
        keys.append(self.tensor_keys.teacher_context)
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
            keys = ["loss_objective", "divergence", "kl_approx"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.kl_to_ref_coeff is not None:
                keys.extend(["loss_kl_to_ref", "kl_to_ref"])
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def _get_student_log_prob(self, tensordict: TensorDictBase):
        """Get log probabilities from the student (policy without feedback)."""
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # Use the specified masking strategy
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
                dist = self.actor_network.get_dist(tensordict, logits_key="logits")
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
            # Also get logits for divergence computation
            logits = tensordict.get("logits", None)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported."
            )
        return log_prob, dist, logits

    def _get_teacher_log_prob(
        self, tensordict: TensorDictBase, teacher_context: TensorDictBase
    ):
        """Get log probabilities from the self-teacher (policy with feedback context).

        The teacher sees the same response but with additional context (feedback).
        """
        # Build teacher input by incorporating feedback context
        teacher_td = tensordict.clone(False)

        # Merge teacher context into the tensordict
        # The teacher_context should contain the reprompted history/tokens
        if teacher_context is not None:
            teacher_td.update(teacher_context)

        # Use EMA parameters if available
        if self._ema_teacher_params is not None:
            # Temporarily swap parameters
            original_params = {}
            for name, param in self.actor_network.named_parameters():
                original_params[name] = param.data.clone()
                param.data.copy_(self._ema_teacher_params[name])

        try:
            # Get teacher distribution
            if isinstance(
                self.actor_network,
                (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
            ) or hasattr(self.actor_network, "get_dist"):
                if self.masking_strategy == "sft" and hasattr(
                    self.actor_network, "_get_sft_dist"
                ):
                    teacher_dist = self.actor_network._get_sft_dist(teacher_td)
                elif self.masking_strategy == "rlhf" and hasattr(
                    self.actor_network, "_get_rlhf_dist"
                ):
                    teacher_dist = self.actor_network._get_rlhf_dist(teacher_td)
                elif self.masking_strategy == "generic" and hasattr(
                    self.actor_network, "_get_generic_dist"
                ):
                    teacher_dist = self.actor_network._get_generic_dist(teacher_td)
                elif hasattr(self.actor_network, "get_dist"):
                    teacher_dist = self.actor_network.get_dist(
                        teacher_td, logits_key="logits"
                    )
                else:
                    raise NotImplementedError(
                        "Actor network must have get_dist method."
                    )

                action = tensordict.get(
                    self.tensor_keys.action,
                    as_padded_tensor=True,
                    padding_side="left",
                    padding_value=-100,
                )
                teacher_log_prob = teacher_dist.log_prob(action)
                teacher_logits = teacher_td.get("logits", None)
            else:
                raise NotImplementedError(
                    "Only probabilistic modules from tensordict.nn are currently supported."
                )
        finally:
            # Restore original parameters if we swapped them
            if self._ema_teacher_params is not None:
                for name, param in self.actor_network.named_parameters():
                    param.data.copy_(original_params[name])

        return teacher_log_prob, teacher_dist, teacher_logits

    def _compute_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute divergence between student and teacher distributions.

        Args:
            student_logits: Logits from student [batch, seq_len, vocab_size]
            teacher_logits: Logits from teacher [batch, seq_len, vocab_size]
            mask: Attention mask [batch, seq_len]

        Returns:
            Per-token divergence [batch, seq_len]
        """
        # Apply top-K filtering if specified
        if self.topk is not None:
            divergence = self._topk_divergence(student_logits, teacher_logits)
        else:
            divergence = self._full_divergence(student_logits, teacher_logits)

        # Apply mask if provided
        if mask is not None:
            divergence = torch.where(
                expand_as_right(mask, divergence), divergence, divergence.new_zeros(())
            )

        return divergence

    def _full_divergence(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute full divergence over all vocabulary tokens."""
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        if self.divergence_type == "kl":
            # KL(student || teacher) = sum_i p_student(i) * log(p_student(i) / p_teacher(i))
            divergence = (student_probs * (student_log_probs - teacher_log_probs)).sum(
                -1
            )
        elif self.divergence_type == "reverse_kl":
            # KL(teacher || student) = sum_i p_teacher(i) * log(p_teacher(i) / p_student(i))
            divergence = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(
                -1
            )
        elif self.divergence_type == "js":
            # Jensen-Shannon divergence
            # JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = 0.5 * (p + q)
            m_log_probs = torch.logaddexp(
                student_log_probs, teacher_log_probs
            ) - math.log(2)
            kl_student_m = (student_probs * (student_log_probs - m_log_probs)).sum(-1)
            kl_teacher_m = (teacher_probs * (teacher_log_probs - m_log_probs)).sum(-1)
            divergence = 0.5 * (kl_student_m + kl_teacher_m)
        else:
            raise ValueError(f"Unknown divergence type: {self.divergence_type}")

        return divergence

    def _topk_divergence(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence over top-K logits only for memory efficiency.

        This approximates the full divergence by only considering the top-K most
        likely tokens according to the student, plus a tail term capturing the
        remaining probability mass.
        """
        k = self.topk

        # Get top-K indices from student
        student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
        topk_log_probs, topk_idx = student_log_probs_full.topk(k, dim=-1)

        # Gather corresponding teacher log-probs
        teacher_log_probs_full = torch.log_softmax(teacher_logits, dim=-1)
        teacher_topk_log_probs = teacher_log_probs_full.gather(-1, topk_idx)

        # Convert to probabilities for top-K
        student_topk_probs = topk_log_probs.exp()
        teacher_topk_probs = teacher_topk_log_probs.exp()

        # Compute tail probabilities
        student_tail_prob = 1.0 - student_topk_probs.sum(-1, keepdim=True)
        teacher_tail_prob = 1.0 - teacher_topk_probs.sum(-1, keepdim=True)

        # Clamp to avoid log(0)
        student_tail_prob = student_tail_prob.clamp(min=1e-10)
        teacher_tail_prob = teacher_tail_prob.clamp(min=1e-10)

        if self.divergence_type == "kl":
            # KL over top-K
            kl_topk = (
                student_topk_probs * (topk_log_probs - teacher_topk_log_probs)
            ).sum(-1)
            # Tail term
            kl_tail = student_tail_prob.squeeze(-1) * (
                student_tail_prob.log().squeeze(-1)
                - teacher_tail_prob.log().squeeze(-1)
            )
            divergence = kl_topk + kl_tail

        elif self.divergence_type == "reverse_kl":
            kl_topk = (
                teacher_topk_probs * (teacher_topk_log_probs - topk_log_probs)
            ).sum(-1)
            kl_tail = teacher_tail_prob.squeeze(-1) * (
                teacher_tail_prob.log().squeeze(-1)
                - student_tail_prob.log().squeeze(-1)
            )
            divergence = kl_topk + kl_tail

        elif self.divergence_type == "js":
            # JS over top-K
            m_topk_log_probs = torch.logaddexp(
                topk_log_probs, teacher_topk_log_probs
            ) - math.log(2)
            kl_student_m = (
                student_topk_probs * (topk_log_probs - m_topk_log_probs)
            ).sum(-1)
            kl_teacher_m = (
                teacher_topk_probs * (teacher_topk_log_probs - m_topk_log_probs)
            ).sum(-1)

            # JS tail term
            m_tail_prob = 0.5 * (student_tail_prob + teacher_tail_prob)
            m_tail_log_prob = m_tail_prob.log()
            kl_s_tail = student_tail_prob * (student_tail_prob.log() - m_tail_log_prob)
            kl_t_tail = teacher_tail_prob * (teacher_tail_prob.log() - m_tail_log_prob)

            divergence = 0.5 * (kl_student_m + kl_teacher_m) + 0.5 * (
                kl_s_tail + kl_t_tail
            ).squeeze(-1)
        else:
            raise ValueError(f"Unknown divergence type: {self.divergence_type}")

        return divergence

    def _apply_trust_region(
        self, teacher_log_probs: torch.Tensor, ref_teacher_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply trust-region regularization to teacher log-probs.

        Interpolates between reference and current teacher:
            q*(y) ∝ exp((1-α)*log q_ref + α*log q_current)
        """
        if self.trust_region_alpha is None:
            return teacher_log_probs

        alpha = self.trust_region_alpha
        # Interpolate in log-space
        interpolated = (1 - alpha) * ref_teacher_log_probs + alpha * teacher_log_probs
        # Renormalize (log-sum-exp trick)
        interpolated = interpolated - interpolated.logsumexp(dim=-1, keepdim=True)
        return interpolated

    def _get_entropy(
        self, dist: d.Distribution, adv_shape: torch.Size
    ) -> torch.Tensor | TensorDict:
        """Compute entropy of the policy distribution."""
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
        mask: torch.Tensor | None = None,
        dist: d.Distribution | None = None,
    ):
        """Compute KL divergence to reference policy."""
        ref_log_prob = tensordict.get(
            self.tensor_keys.ref_log_probs,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0.0,
        )
        if ref_log_prob is None:
            raise KeyError(
                f"Couldn't find the ref log-prob {self.tensor_keys.ref_log_probs} in the input data."
            )
        ref_log_prob = ref_log_prob.squeeze(-1)

        cur_log_prob = tensordict.get("_cur_log_prob")
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
        return self.kl_to_ref_coeff * kl_penalty, kl_penalty

    def _aggregate_loss_value(
        self, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate a per-token loss tensor using the configured strategy."""
        if self.aggregation == "none" or self.reduction == "none":
            mask_exp = expand_as_right(mask, value)
            return torch.where(mask_exp, value, value.new_zeros(()).expand_as(value))

        if self.aggregation == "prompt_mean":
            mask_exp = expand_as_right(mask, value).to(value.dtype)
            token_sum = (value * mask_exp).sum(dim=-2, keepdim=False)
            token_count = mask_exp.sum(dim=-2, keepdim=False).clamp_min(1.0)
            sample_mean = token_sum / token_count
            return sample_mean.mean(dim=0, keepdim=False)

        # token_mean (global masked mean)
        return _reduce(value, reduction="mean", mask=mask).squeeze(-1)

    def forward(self, tensordict: TensorDictBase) -> SDPOLossOutput:
        """Compute the SDPO loss.

        Args:
            tensordict: Input data containing:
                - action tokens (self.tensor_keys.action)
                - student log probabilities (self.tensor_keys.sample_log_prob)
                - teacher context (self.tensor_keys.teacher_context)

        Returns:
            SDPOLossOutput containing loss_objective, divergence, and other metrics.
        """
        tensordict = tensordict.copy()

        # Get teacher context (feedback-augmented prompt)
        teacher_context = tensordict.get(self.tensor_keys.teacher_context, None)

        # Run forward pass to get student logits
        with torch.no_grad() if not self.training else contextlib.nullcontext():
            self.actor_network(tensordict)

        # Get student distribution and log-probs
        student_log_prob, student_dist, student_logits = self._get_student_log_prob(
            tensordict
        )
        tensordict.set("_cur_log_prob", student_log_prob)

        # Get teacher log-probs (with stopgrad on teacher)
        with torch.no_grad():
            teacher_log_prob, teacher_dist, teacher_logits = self._get_teacher_log_prob(
                tensordict, teacher_context
            )

        # Get mask from distribution
        mask = student_dist.mask

        # Compute divergence loss
        if student_logits is not None and teacher_logits is not None:
            # Use logit-level divergence
            divergence = self._compute_divergence(student_logits, teacher_logits, mask)
        else:
            # Fall back to log-prob based approximation
            # This is less accurate but works when logits aren't available
            divergence = (student_log_prob - teacher_log_prob).abs()
            if mask is not None:
                divergence = torch.where(
                    expand_as_right(mask, divergence),
                    divergence,
                    divergence.new_zeros(()),
                )

        # Compute kl_approx for logging (log-prob level)
        kl_approx = (student_log_prob - teacher_log_prob).unsqueeze(-1)

        # Build output
        td_out = TensorDict(
            {
                "loss_objective": divergence.unsqueeze(-1),
                "divergence": divergence.detach().mean(),
                "kl_approx": kl_approx.detach().mean(),
            }
        )

        # Add entropy bonus if requested
        if self.entropy_bonus:
            entropy = self._get_entropy(student_dist, adv_shape=student_log_prob.shape)
            if is_tensor_collection(entropy):
                td_out.set("composite_entropy", entropy.detach())
                entropy = _sum_td_features(entropy)
            td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)

        # Add KL to reference if requested
        if self.kl_to_ref_coeff is not None and self.kl_to_ref_coeff > 0:
            loss_kl, kl_penalty = self._kl_to_ref(
                tensordict, mask=mask, dist=student_dist
            )
            td_out["loss_kl_to_ref"] = loss_kl
            td_out["kl_to_ref"] = kl_penalty.detach()

        # Aggregate loss terms
        for key in list(td_out.keys()):
            if isinstance(key, tuple) or not isinstance(key, str):
                continue
            if key.startswith("loss_"):
                val = td_out.get(key)
                td_out.set(key, self._aggregate_loss_value(val, mask))

        # Clean up
        del tensordict["_cur_log_prob"]

        return self.output_type.from_tensordict(td_out)
