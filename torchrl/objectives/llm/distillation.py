# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch
from tensordict import NestedKey, TensorClass, TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.utils import _zip_strict
from torchrl.data import History
from torchrl.modules.llm.policies.transformers_wrapper import TransformersWrapper
from torchrl.objectives.common import LossModule

if TYPE_CHECKING:
    import transformers


def reverse_kl_token_estimate(
    cur_log_prob: torch.Tensor, teacher_log_prob: torch.Tensor
) -> torch.Tensor:
    r"""Per-token low-variance estimate of the reverse KL divergence to the teacher.

    Computes the per-token k3 estimator (Schulman,
    `"Approximating KL Divergence" <http://joschu.net/blog/kl-approx.html>`_) of
    :math:`\mathrm{KL}(\pi_\theta \,\|\, \pi_T)` using tokens drawn from the
    student policy :math:`\pi_\theta`:

    .. math::
        \widehat{\mathrm{kl}}_t = e^{\Delta_t} - 1 - \Delta_t,
        \qquad \Delta_t = \log \pi_T(a_t) - \log \pi_\theta(a_t)

    The estimator is non-negative and equals zero exactly when the two log-probs
    match. With ``a_t`` sampled from the student, summing over a sequence
    estimates the sequence-level reverse KL :math:`\mathrm{KL}(\pi_\theta\,\|\,\pi_T)`.

    Args:
        cur_log_prob (torch.Tensor): student log-probabilities of the sampled tokens.
        teacher_log_prob (torch.Tensor): teacher log-probabilities of the same tokens.

    Returns:
        torch.Tensor: the per-token KL estimate, with the same shape as the inputs.
    """
    if cur_log_prob.shape != teacher_log_prob.shape:
        raise ValueError(
            "Student and teacher log-probabilities have different shapes: "
            f"{cur_log_prob.shape=} vs {teacher_log_prob.shape=}."
        )
    diff = teacher_log_prob - cur_log_prob
    return diff.expm1() - diff


def distillation_loss(
    per_sequence_kl: torch.Tensor, reduction: Literal["mean", "sum", "none"]
) -> torch.Tensor:
    """Reduce the per-sequence distillation KL into the final loss value.

    Args:
        per_sequence_kl (torch.Tensor): a one-dimensional tensor holding one
            (already token-reduced) KL value per sequence in the batch.
        reduction (Literal["mean", "sum", "none"]): the batch reduction to apply.

    Returns:
        torch.Tensor: the reduced loss (a scalar for ``"mean"``/``"sum"``, the
        per-sequence tensor for ``"none"``).
    """
    if reduction == "mean":
        return per_sequence_kl.mean()
    if reduction == "sum":
        return per_sequence_kl.sum()
    if reduction == "none":
        return per_sequence_kl
    raise ValueError(f"Invalid reduction: {reduction}.")


class DistillationLossOutput(TensorClass["nocast"]):
    """Output of :class:`~torchrl.objectives.llm.DistillationLoss`.

    Attributes:
        loss_distill (torch.Tensor): the coefficient-scaled on-policy reverse-KL
            distillation loss. This is the only differentiable field.
        kl_to_teacher (torch.Tensor | None): the unscaled mean per-token KL to the
            teacher, detached, for logging.

    .. note::
        Only ``loss_distill`` carries a gradient; ``kl_to_teacher`` is detached.
        Summing the output (``loss_fn(td).sum(reduce=True)``) therefore yields the
        total differentiable loss, mirroring
        :class:`~torchrl.objectives.llm.SFTLossOutput`.
    """

    loss_distill: torch.Tensor
    kl_to_teacher: torch.Tensor | None = None


class DistillationLoss(LossModule):
    r"""On-policy knowledge-distillation loss for LLM policies.

    Trains a student policy to imitate a (typically larger or stronger) teacher
    by minimizing the reverse KL divergence
    :math:`\mathrm{KL}(\pi_\theta \,\|\, \pi_T)` on the student's own generations.
    Because the expectation is taken over the student's distribution, this is the
    *on-policy* distillation regime: the student generates the sequences and the
    teacher scores those sequences. The teacher's per-token log-probabilities are
    expected to have been written to the input tensordict by an upstream
    :class:`~torchrl.envs.llm.transforms.RetrieveLogProb` transform pointed at the
    teacher model, exactly as reference log-probabilities are produced for
    :class:`~torchrl.objectives.llm.SFTLoss` and
    :class:`~torchrl.objectives.llm.GRPOLoss`.

    The divergence is estimated per token with the low-variance k3 estimator (see
    :func:`~torchrl.objectives.llm.distillation.reverse_kl_token_estimate`),
    restricted to assistant (response) tokens, then reduced per sequence and over
    the batch.

    .. note::
        This is the *reverse-KL, token-log-probability* form of distillation. It
        is a single-sample estimate of the sequence-level reverse KL on the
        student's own samples, and its gradient is taken through the student's
        re-scored log-probabilities only (the dependence of the sampling
        distribution on the parameters is not differentiated, as is standard for
        on-policy distillation surrogates). The exact per-token KL and the
        forward-KL / generalized-JSD variants require the teacher's full
        vocabulary distribution and are not covered here.

    Args:
        actor_network (TensorDictModule): the student network. Usually a
            :class:`~torchrl.modules.llm.TransformersWrapper` with
            ``generate=False`` so that its log-probabilities are differentiable.
        tokenizer (`Tokenizer`, optional): the tokenizer used to recompute the
            assistant mask from the chat history when it is not already present in
            the input tensordict. If not provided, it is taken from
            ``actor_network``.
        tokenizer_kwargs (dict, optional): keyword arguments forwarded to
            :meth:`~torchrl.data.llm.chat.History.apply_chat_template`.

    Keyword Args:
        distill_coeff (float, optional): scaling applied to the distillation loss.
            Defaults to ``1.0``.
        reduction (Literal["mean", "sum", "none"], optional): the batch reduction.
            Defaults to ``"mean"``.
        normalize_by_seq_length (bool, optional): whether to normalize the
            per-sequence KL by the number of assistant tokens. Defaults to ``True``.
        device (torch.device, optional): the device used when tokenizing the
            input. Defaults to ``None``.

    .. note::
        The input tensordict is expected to contain the following keys by default:
            - ``("history", "full")``: the chat history (used to run the student
              and to recompute the assistant mask if needed).
            - ``("next", "teacher_log_probs", "full")``: the teacher per-token
              log-probabilities, produced upstream by
              :class:`~torchrl.envs.llm.transforms.RetrieveLogProb`.

        These keys can be customized using the ``set_keys()`` method, e.g.
        ``loss.set_keys(teacher_log_prob=("next", "my_teacher", "full"))``.

    .. seealso:: :class:`~torchrl.envs.llm.transforms.RetrieveLogProb` for the
        teacher log-probability computation, and
        :class:`~torchrl.objectives.llm.SFTLoss` for the supervised-fine-tuning
        loss this module mirrors.

    References:
        - Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, et al., 2023.
          `"On-Policy Distillation of Language Models: Learning from Self-Generated
          Mistakes" (GKD) <https://arxiv.org/abs/2306.13649>`_
        - Yoon Kim, Alexander M. Rush, 2016.
          `"Sequence-Level Knowledge Distillation" <https://arxiv.org/abs/1606.07947>`_

    Examples:
        >>> from torchrl.data.llm.chat import History, _CHAT_TEMPLATES
        >>> from torchrl.envs.llm.transforms import RetrieveLogProb
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.objectives.llm import DistillationLoss
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, lazy_stack
        >>> import torch
        >>>
        >>> chats = [
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Hello, how are you?"},
        ...         {"role": "assistant", "content": "I'm doing well, thank you!"},
        ...     ],
        ... ]
        >>> history = History.from_chats(chats)
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]
        >>> # A small student and a separate (frozen) teacher.
        >>> student = TransformersWrapper(
        ...     OPTForCausalLM(OPTConfig()).eval(),
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     chat_template_name="qwen",
        ... )
        >>> teacher = TransformersWrapper(
        ...     OPTForCausalLM(OPTConfig()).eval(),
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     return_log_probs=True,
        ...     chat_template_name="qwen",
        ... )
        >>> # Score the student's tokens under the teacher.
        >>> transform = RetrieveLogProb(
        ...     teacher,
        ...     assistant_only=True,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ...     tokenizer=tokenizer,
        ...     log_probs_full_key=("teacher_log_probs", "full"),
        ... )
        >>> text = history[:, :-1].apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        ... )
        >>> text_response = history.apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        ... )
        >>> text_response = [
        ...     txt[len(txt_start):] for txt, txt_start in zip(text_response, text)
        ... ]
        >>> td = TensorDict(
        ...     text=text,
        ...     text_response=text_response,
        ...     history=history,
        ...     next=TensorDict(
        ...         reward=torch.randn(1, 1),
        ...         done=torch.zeros(1, dtype=torch.bool),
        ...         history=history,
        ...     ),
        ...     batch_size=(1,),
        ... )
        >>> data = lazy_stack(list(td.unbind(0)))
        >>> with torch.no_grad():
        ...     data = transform(data)
        >>> loss = DistillationLoss(
        ...     actor_network=student,
        ...     tokenizer=tokenizer,
        ...     distill_coeff=1.0,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ... )
        >>> loss_vals = loss(data)
        >>> print(f"distillation loss: {loss_vals.loss_distill.item():.4f}")
        >>> loss_vals.sum(reduce=True).backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using
        ``.set_keys(key_name=key_value)`` and their default values.

        Attributes:
            history (NestedKey): The input tensordict key where the chat history is
                expected. Defaults to ``("history", "full")``.
            teacher_log_prob (NestedKey): The input tensordict key where the teacher
                per-token log-probabilities are expected (produced upstream by
                :class:`~torchrl.envs.llm.transforms.RetrieveLogProb`). Defaults to
                ``("next", "teacher_log_probs", "full")``.
            log_probs (NestedKey): The key under which the student log-probabilities
                are read from the actor output. Defaults to ``("log_probs", "full")``.
        """

        history: NestedKey = ("history", "full")
        teacher_log_prob: NestedKey = ("next", "teacher_log_probs", "full")
        log_probs: NestedKey = ("log_probs", "full")

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        actor_network: TensorDictModule | TransformersWrapper,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        tokenizer_kwargs: dict | None = None,
        *,
        distill_coeff: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        normalize_by_seq_length: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_keys = []
        self.actor_network = actor_network
        if tokenizer is None:
            tokenizer = getattr(actor_network, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        self.tokenizer = tokenizer
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("return_tensors", "pt")
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("add_generation_prompt", False)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.distill_coeff = distill_coeff
        self.reduction = reduction
        self.normalize_by_seq_length = normalize_by_seq_length
        self.device = device
        self._set_in_keys()

    def _set_in_keys(self) -> None:
        """Sets the input keys for the loss module."""
        self.in_keys = [self.tensor_keys.history, self.tensor_keys.teacher_log_prob]
        self.out_keys = []  # Loss modules typically don't have out_keys

    def _get_assistant_masks(
        self, tensordict: TensorDictBase, history: History
    ) -> tuple[list[torch.Tensor], object]:
        """Return per-sequence assistant masks, recomputing them if necessary.

        Mirrors :class:`~torchrl.objectives.llm.SFTLoss`: prefer the masks already
        stored on the tensordict, otherwise recompute them from the chat history.
        """
        token_struct = None
        assistant_masks = tensordict.get(("masks", "all_assistant_mask"), as_list=True)
        attention_mask = tensordict.get(("masks", "all_attention_mask"), as_list=True)
        if assistant_masks is None:
            with torch.device(
                self.device
            ) if self.device is not None else contextlib.nullcontext():
                token_struct = history.apply_chat_template(
                    tokenizer=self.tokenizer, **self.tokenizer_kwargs
                )
            if "assistant_masks" not in token_struct:
                raise ValueError(
                    f"Assistant masks are not present in the token structure: {token_struct=}."
                )
            assistant_masks = token_struct.get("assistant_masks", as_list=True)
            attention_mask = token_struct.get("attention_mask", as_list=True)
        assistant_masks = [mask.bool() for mask in assistant_masks]
        attention_mask = [mask.bool() for mask in attention_mask]
        assistant_masks = [
            mask & a_mask for mask, a_mask in zip(assistant_masks, attention_mask)
        ]
        if not all(mask.any(-1).all() for mask in assistant_masks):
            raise ValueError("Some inputs have no valid assistant masks.")
        return assistant_masks, token_struct

    def forward(self, tensordict: TensorDictBase) -> DistillationLossOutput:
        history: History = tensordict[self.tensor_keys.history]
        assistant_masks, token_struct = self._get_assistant_masks(tensordict, history)

        # Re-run the student so its log-probabilities are differentiable.
        input_loss = tensordict.select(self.tensor_keys.history)
        with torch.device(
            self.device
        ) if self.device is not None else contextlib.nullcontext():
            output_loss = self.actor_network(input_loss)
        log_probs = output_loss.get(self.tensor_keys.log_probs, as_list=True)

        # Teacher log-probabilities are computed upstream and arrive detached.
        teacher_log_probs = tensordict.get(
            self.tensor_keys.teacher_log_prob,
            default=None,
            as_list=True,
        )
        if teacher_log_probs is None:
            raise ValueError(
                f"Teacher log-probs not found in tensordict at key "
                f"{self.tensor_keys.teacher_log_prob}. Did you run a RetrieveLogProb "
                f"transform pointed at the teacher? Existing keys: "
                f"{set(tensordict.keys(include_nested=True, leaves_only=True))}"
            )

        if not all(
            mask.shape == lp.shape
            for mask, lp in _zip_strict(assistant_masks, log_probs)
        ):
            if token_struct is not None:
                suffix = (
                    "Tokens from current template: "
                    f"{[inp.shape for inp in token_struct.get('input_ids', as_padded_tensor=True)]}"
                )
            else:
                suffix = ""
            raise ValueError(
                f"Assistant masks and log_probs have different shapes: "
                f"{[mask.shape for mask in assistant_masks]} vs "
                f"{[lp.shape for lp in log_probs]}. {suffix}"
            )

        reduce_dim = tensordict.ndim - 1
        per_sequence_kl = []
        per_token_kl_mean = []
        for student_lp, teacher_lp, mask in _zip_strict(
            log_probs, teacher_log_probs, assistant_masks
        ):
            teacher_lp = teacher_lp.detach().to(student_lp.device)
            mask = mask.to(student_lp.device)
            kl_token = reverse_kl_token_estimate(student_lp, teacher_lp)
            kl_token = kl_token.masked_fill(~mask, 0.0)
            summed = kl_token.sum(reduce_dim)
            count = mask.sum(reduce_dim).clamp(min=1)
            per_token_kl_mean.append(summed / count)
            per_sequence_kl.append(
                summed / count if self.normalize_by_seq_length else summed
            )

        per_sequence_kl = torch.stack(per_sequence_kl)
        loss = self.distill_coeff * distillation_loss(per_sequence_kl, self.reduction)
        kl_to_teacher = torch.stack(per_token_kl_mean).mean().detach()
        return DistillationLossOutput(loss_distill=loss, kl_to_teacher=kl_to_teacher)
