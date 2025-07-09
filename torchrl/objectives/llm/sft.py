# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import warnings

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


def sft_loss(summed_log_probs: torch.Tensor, reduction: str) -> torch.Tensor:
    """Compute the SFT loss."""
    if reduction == "mean":
        loss = -summed_log_probs.mean()
    elif reduction == "sum":
        loss = -summed_log_probs.sum()
    elif reduction == "none":
        loss = -summed_log_probs
    else:
        raise ValueError(f"Invalid reduction: {reduction}.")
    return loss


def minor_sft_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    beta: float,
    reduction: str,
) -> torch.Tensor:
    """Compute the MinorSFT loss.

    This loss is inspired by DPO and is designed to be less aggressive than standard SFT.
    It computes ``-log_sigmoid(beta * (log_probs - ref_log_probs))``.

    Args:
        log_probs (torch.Tensor): The log probabilities from the model being trained.
        ref_log_probs (torch.Tensor): The log probabilities from the reference model.
        beta (float): The beta parameter from DPO.
        reduction (str): The reduction to apply to the loss.

    Returns:
        The MinorSFT loss.

    References:
        - Shiming Xie, Hong Chen, Fred Yu, Zeye Sun, Xiuyu Wu, 2024.
          `"Minor SFT loss for LLM fine-tune to increase performance and reduce model deviation" <https://arxiv.org/abs/2408.10642>`_
    """
    if log_probs.shape != ref_log_probs.shape:
        raise ValueError(
            f"Current log probabilities and reference log probabilities have different shapes: {log_probs.shape=} vs {ref_log_probs.shape=}."
        )
    loss = -torch.nn.functional.logsigmoid(beta * (log_probs - ref_log_probs))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")


class SFTLossOutput(TensorClass["nocast"]):
    """SFT Loss Output.

    Attributes:
        loss_sft (torch.Tensor): The loss for the SFT objective.
        loss_kl_to_ref (torch.Tensor | None): The loss for the KL divergence to the reference model.
        kl_to_ref (torch.Tensor | None): The KL divergence to the reference model.

    .. note::
        The loss components are kept separate to allow for logging and visualization.
        Before backpropagation, the loss components are to be summed together. Since non-loss components are not differentiable
        when the loss is constructed via :class:`~torchrl.objectives.llm.sft.SFTLoss`, summing
        the :class:`~torchrl.objectives.llm.sft.SFTLossOutput` directly is a proper way of obtaining the total loss.

            >>> loss_fn = SFTLoss(...)
            >>> loss_output = loss_fn(td)
            >>> loss = loss_output.loss_sft + loss_output.loss_kl_to_ref
            >>> loss.backward()
            >>> # or equivalently
            >>> loss = loss_fn(td)
            >>> loss.sum(reduce=True).backward()
    """

    loss_sft: torch.Tensor
    loss_kl_to_ref: torch.Tensor | None = None
    kl_to_ref: torch.Tensor | None = None


class SFTLoss(LossModule):
    r"""Supervised fine-tuning loss.

    Args:
        actor_network (TensorDictModule): the actor network. Usually a :class:`~torchrl.modules.llm.TransformersWrapper` instance,
            with `return_log_prob=True` and `from_text=True`.
        tokenizer (`Tokenizer`): the tokenizer to be used to tokenize the input and compute the assitant mask. If not provided, the tokenizer will be inferred from the `actor_network`.
        tokenizer_kwargs (dict, optional): keyword arguments to pass to the tokenizer during :meth:`~torchrl.data.llm.chat.History.apply_chat_template`.
            This can be used to override arguments such as the `chat_template` or `chat_template_name`.
        reduction (Literal["mean", "sum", "none"], optional): the reduction to apply to the loss. Defaults to `"mean"`.
        normalize_by_seq_length (bool, optional): whether to normalize the loss by the sequence length. Defaults to `True`.
        kl_to_ref_coeff (float | None, optional): coefficient for KL divergence to reference model. Defaults to `None`.
        loss_function (Literal["sft", "minor_sft"], optional): The loss function to use. Defaults to `"sft"`.
        beta (float, optional): The beta parameter for MinorSFT loss. This is only used when `loss_function` is `"minor_sft"`.
            Higher values of beta make the loss more aggressive (pushes the model to generate responses further from the reference model):

            .. math::
                \text{loss} = -\log\sigma(\beta \cdot (\text{log_probs} - \text{ref_log_probs}))

            Defaults to `0.1`.
        device (torch.device | None, optional): the device to use for the loss, when tokenizing the input. Defaults to `None`.

    .. note::
        The input tensordict is expected to contain the following keys by default:
            - ``("next", "history")``: The chat history
            - ``("next", "ref_log_prob")`` (optional): Reference model log probabilities, required if kl_to_ref_coeff is set

        These keys can be customized using the ``set_keys()`` method.

    .. seealso:: :class:`~torchrl.envs.llm.transforms.RetrieveLogProb` for the KL divergence computation.

    References:
        - Shiming Xie, Hong Chen, Fred Yu, Zeye Sun, Xiuyu Wu, 2024.
          `"Minor SFT loss for LLM fine-tune to increase performance and reduce model deviation" <https://arxiv.org/abs/2408.10642>`_

    Examples:
        >>> from torchrl.data.llm.chat import History, _CHAT_TEMPLATES
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.objectives.llm.sft import SFTLoss
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, lazy_stack
        >>> import torch
        >>>
        >>> # Create chat data
        >>> chats = [
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Hello, how are you?"},
        ...         {"role": "assistant", "content": "I'm doing well, thank you!"},
        ...     ],
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What's the weather like?"},
        ...         {"role": "assistant", "content": "I can't check the weather for you."},
        ...     ],
        ... ]
        >>> history = History.from_chats(chats)
        >>>
        >>> # Setup tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]
        >>> model = OPTForCausalLM(OPTConfig()).eval()
        >>>
        >>> # Create training and reference policies
        >>> policy_train = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     from_text=True,
        ...     chat_template_name="qwen",
        ... )
        >>> policy_ref = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     from_text=True,
        ...     return_log_probs=True,
        ...     chat_template_name="qwen",
        ... )
        >>>
        >>> # Create the RetrieveLogProb transform
        >>> transform = RetrieveLogProb(
        ...     policy_ref,
        ...     assistant_only=True,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ...     tokenizer=tokenizer,
        ... )
        >>>
        >>> # Prepare data
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
        ...         reward=torch.randn(2, 1),
        ...         done=torch.zeros(2, dtype=torch.bool),
        ...         history=history,
        ...     ),
        ...     batch_size=(2,),
        ... )
        >>> data = lazy_stack(list(td.unbind(0)))
        >>>
        >>> # Apply the transform to get reference log probabilities
        >>> data = transform(data)
        >>> assert "ref_log_prob" in data["next"].keys()
        >>>
        >>> # Use with SFTLoss for KL regularization
        >>> loss = SFTLoss(
        ...     actor_network=policy_train,
        ...     tokenizer=tokenizer,
        ...     reduction="mean",
        ...     normalize_by_seq_length=True,
        ...     kl_to_ref_coeff=0.1,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ...     loss_function="sft",
        ... )
        >>> loss_vals = loss(data)
        >>> print(f"SFT Loss: {loss_vals.loss_sft.item():.4f}")
        >>> print(f"KL to Reference Loss: {loss_vals.loss_kl_to_ref.item():.4f}")

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            history (NestedKey): The input tensordict key where the chat history is expected.
                Defaults to ``("next", "history")``.
            ref_log_prob (NestedKey): The input tensordict key where the reference model log probabilities are expected.
                Only used when kl_to_ref_coeff is set. Defaults to ``("next", "ref_log_prob")``.
            log_probs (NestedKey): The output tensordict key where the model's log probabilities will be written.
                Defaults to ``"log_probs"``.
        """

        history: NestedKey = ("history", "full")
        ref_log_prob: NestedKey = ("next", "ref_log_probs", "full")
        log_probs: NestedKey = ("log_probs", "full")

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        actor_network: TensorDictModule | TransformersWrapper,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        tokenizer_kwargs: dict | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        normalize_by_seq_length: bool = True,
        kl_to_ref_coeff: float | None = None,
        loss_function: Literal["sft", "minor_sft"] = "sft",
        beta: float = 0.1,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_keys = []
        self.actor_network = actor_network
        if tokenizer is None:
            tokenizer = actor_network.tokenizer
        self.tokenizer = tokenizer
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("return_tensors", "pt")
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("add_generation_prompt", False)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.reduction = reduction
        self.normalize_by_seq_length = normalize_by_seq_length
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self.loss_function = loss_function
        if self.loss_function == "minor_sft" and kl_to_ref_coeff:
            warnings.warn(
                "kl_to_ref_coeff should not be set when using minor_sft loss, as KL regularization is implicit. Setting kl_to_ref_coeff to 0.0."
            )
            self.kl_to_ref_coeff = 0.0
        self.beta = beta
        self._set_in_keys()
        self.device = device

    def _set_in_keys(self) -> None:
        """Sets the input keys for the loss module."""
        in_keys = [self.tensor_keys.history]
        if self.kl_to_ref_coeff is not None or self.loss_function == "minor_sft":
            in_keys.append(self.tensor_keys.ref_log_prob)
        self.in_keys = in_keys
        self.out_keys = []  # Loss modules typically don't have out_keys

    def _kl_to_ref(
        self,
        cur_log_prob: list[torch.Tensor],
        ref_log_prob: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence to reference model.

        Args:
            cur_log_prob (List[torch.Tensor]): Log probabilities from current model. Must have shape [T] where T is the number of tokens in the assistant response.
            ref_log_prob (List[torch.Tensor]): Log probabilities from reference model. Must have shape [T] where T is the number of tokens in the assistant response.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (KL loss term, KL penalty for logging)
        """
        # Apply mask
        ref_log_prob = torch.cat(ref_log_prob)
        cur_log_prob = torch.cat(cur_log_prob)
        # ref_log_prob = ref_log_prob[mask]
        # cur_log_prob = cur_log_prob[mask].squeeze()
        if cur_log_prob.shape != ref_log_prob.shape:
            raise ValueError(
                f"Current log probabilities and reference log probabilities have different shapes: {cur_log_prob.shape=} vs {ref_log_prob.shape=}."
            )
        # Compute KL using same approximation as GRPO
        diff = ref_log_prob - cur_log_prob

        kl_penalty = (diff.expm1() - diff).mean()
        return self.kl_to_ref_coeff * kl_penalty, kl_penalty

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather history
        history: History = tensordict[self.tensor_keys.history]

        # Try to get mask from td
        token_struct = None
        assistant_masks = tensordict.get(("masks", "all_assistant_mask"), as_list=True)
        attention_mask = tensordict.get(("masks", "all_attention_mask"), as_list=True)
        if assistant_masks is None:
            # Apply tokenizer to history and gather mask
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
            assistant_masks = token_struct.get(
                "assistant_masks",
                as_list=True,
            )
            attention_mask = token_struct.get("attention_mask", as_list=True)
        assistant_masks = [mask.bool() for mask in assistant_masks]
        attention_mask = [mask.bool() for mask in attention_mask]
        assistant_masks = [
            mask & a_mask for mask, a_mask in zip(assistant_masks, attention_mask)
        ]

        if not any(mask.any(-1).all() for mask in assistant_masks):
            raise ValueError("Some inputs have no valid assistant masks.")

        input_loss = tensordict.select(self.tensor_keys.history)

        with torch.device(
            self.device
        ) if self.device is not None else contextlib.nullcontext():
            output_loss = self.actor_network(input_loss)

        # get log-probs
        log_probs = output_loss.get(
            self.tensor_keys.log_probs,
            as_list=True,
        )

        # apply mask
        if not all(
            mask.shape == lp.shape
            for mask, lp in _zip_strict(assistant_masks, log_probs)
        ):
            if token_struct is not None:
                suffix = f"Tokens from current template: {[inp.shape for inp in token_struct.get('input_ids', as_padded_tensor=True)]}"
            else:
                suffix = ""
            raise ValueError(
                f"Assistant masks and log_probs have different shapes: {[mask.shape for mask in assistant_masks]} vs "
                f"{[lp.shape for lp in log_probs]}. {suffix}"
            )

        log_probs_masked = [
            lp.masked_fill(~mask, 0.0)
            for lp, mask in _zip_strict(log_probs, assistant_masks)
        ]

        # Sum log probs, optionally normalize by sequence length
        summed_log_probs = torch.stack(
            [lp.sum(tensordict.ndim - 1) for lp in log_probs_masked]
        )
        seq_lengths = torch.stack(
            [mask.sum(tensordict.ndim - 1) for mask in assistant_masks]
        )
        if self.normalize_by_seq_length:
            # Compute sequence lengths for normalization (number of assistant tokens)
            summed_log_probs = summed_log_probs / seq_lengths.clamp(min=1)

        # Compute main loss
        if self.loss_function == "sft":
            loss = sft_loss(summed_log_probs, self.reduction)
            # Add KL divergence loss if reference model is provided
            if self.kl_to_ref_coeff is not None:
                ref_log_probs = tensordict.get(
                    self.tensor_keys.ref_log_prob,
                    default=None,
                    as_list=True,
                )
                if ref_log_probs is None:
                    raise ValueError(
                        f"Reference log probs not found in tensordict at key {self.tensor_keys.ref_log_prob} but kl_to_ref_coeff was set. "
                        f"Existing keys in tensordict: {set(tensordict.keys(include_nested=True, leaves_only=True))}"
                    )

                log_probs_masked = [
                    lp.masked_fill(~mask, 0.0)
                    for lp, mask in _zip_strict(log_probs, assistant_masks)
                ]

                loss_kl, kl_penalty = self._kl_to_ref(
                    log_probs_masked,
                    ref_log_probs,
                )
                output = SFTLossOutput(
                    loss_sft=loss,
                    loss_kl_to_ref=loss_kl,
                    kl_to_ref=kl_penalty.detach(),
                )
            else:
                output = SFTLossOutput(loss_sft=loss)
        elif self.loss_function == "minor_sft":
            ref_log_probs = tensordict.get(self.tensor_keys.ref_log_prob, as_list=True)
            if ref_log_probs is None:
                raise ValueError(
                    f"Reference log probs not found at {self.tensor_keys.ref_log_prob=} in tensordict with keys {tensordict.keys()} but loss_function is 'minor_sft'"
                )

            # we need to re-sum ref_log_probs as they are not summed per-sequence
            summed_ref_log_probs = torch.stack([lp.sum() for lp in ref_log_probs]).to(
                summed_log_probs.device
            )
            if self.normalize_by_seq_length:
                summed_ref_log_probs = summed_ref_log_probs / seq_lengths.clamp(min=1)
            loss = minor_sft_loss(
                summed_log_probs, summed_ref_log_probs, self.beta, self.reduction
            )
            if self.kl_to_ref_coeff is not None:
                with torch.no_grad():
                    log_probs_masked = [
                        lp.masked_fill(~mask, 0.0)
                        for lp, mask in _zip_strict(log_probs, assistant_masks)
                    ]
                    loss_kl, kl_penalty = self._kl_to_ref(
                        log_probs_masked,
                        ref_log_probs,
                    )
                output = SFTLossOutput(
                    loss_sft=loss,
                    loss_kl_to_ref=loss_kl,
                    kl_to_ref=kl_penalty.detach(),
                )
            else:
                output = SFTLossOutput(loss_sft=loss)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_function}")

        return output
