# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import warnings

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import NestedKey, TensorClass, TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.utils import _zip_strict
from torchrl.data import History
from torchrl.modules.llm.policies.transformers_wrapper import TransformersWrapper
from torchrl.objectives.common import LossModule


def dpo_loss(
    policy_chosen_logprob: torch.Tensor,
    policy_rejected_logprob: torch.Tensor,
    reference_chosen_logprob: torch.Tensor,
    reference_rejected_logprob: torch.Tensor,
    beta: float,
    reduction: Literal["mean", "sum", "none"],
) -> torch.Tensor:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps (torch.Tensor): Log probabilities of chosen responses from the policy model.
        policy_rejected_logps (torch.Tensor): Log probabilities of rejected responses from the policy model.
        reference_chosen_logps (torch.Tensor): Log probabilities of chosen responses from the reference model.
        reference_rejected_logps (torch.Tensor): Log probabilities of rejected responses from the reference model.
        beta (float): The beta parameter controlling the strength of the preference optimization.
        reduction (str): The reduction to apply to the loss.

    Returns:
        torch.Tensor: The DPO loss.

    References:
        - Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
          `"Direct Preference Optimization: Your Language Model is Secretly a Reward Model" <https://arxiv.org/abs/2305.18290>`_
    """
    chosen_rewards = beta * (policy_chosen_logprob - reference_chosen_logprob)
    rejected_rewards = beta * (policy_rejected_logprob - reference_rejected_logprob)
    
    losses = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards)
    
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class DPOLossOutput(TensorClass["nocast"]):
    """DPO Loss Output.

    Attributes:
        loss_dpo (torch.Tensor): The loss for the DPO objective.
        loss_kl_to_ref (torch.Tensor | None): The loss for the KL divergence to the reference model.
        kl_to_ref (torch.Tensor | None): The KL divergence to the reference model.
        chosen_rewards (torch.Tensor): The rewards for chosen responses.
        rejected_rewards (torch.Tensor): The rewards for rejected responses.
        accuracy (torch.Tensor): The accuracy of preference prediction.

    .. note::
        The loss components are kept separate to allow for logging and visualization.
        Before backpropagation, the loss components are to be summed together. Since non-loss components are not differentiable
        when the loss is constructed via :class:`~torchrl.objectives.llm.dpo.DPOLoss`, summing
        the :class:`~torchrl.objectives.llm.dpo.DPOLossOutput` directly is a proper way of obtaining the total loss.

            >>> loss_fn = DPOLoss(...)
            >>> loss_output = loss_fn(td)
            >>> loss = loss_output.loss_dpo + loss_output.loss_kl_to_ref
            >>> loss.backward()
            >>> # or equivalently
            >>> loss = loss_fn(td)
            >>> loss.sum(reduce=True).backward()
    """

    loss_dpo: torch.Tensor
    loss_kl_to_ref: torch.Tensor | None = None
    kl_to_ref: torch.Tensor | None = None
    chosen_rewards: torch.Tensor | None = None
    rejected_rewards: torch.Tensor | None = None
    accuracy: torch.Tensor | None = None


class DPOLoss(LossModule):
    r"""Direct Preference Optimization loss.

    Args:
        actor_network (TensorDictModule): the actor network. Usually a :class:`~torchrl.modules.llm.TransformersWrapper` instance,
            with `return_log_prob=True` and `from_text=True`.
        tokenizer (`Tokenizer`): the tokenizer to be used to tokenize the input and compute the assistant mask. If not provided, the tokenizer will be inferred from the `actor_network`.
        tokenizer_kwargs (dict, optional): keyword arguments to pass to the tokenizer during :meth:`~torchrl.data.llm.chat.History.apply_chat_template`.
            This can be used to override arguments such as the `chat_template` or `chat_template_name`.
        beta (float): The beta parameter controlling the strength of the preference optimization. Higher values make the optimization more aggressive.
        reduction (Literal["mean", "sum", "none"], optional): the reduction to apply to the loss. Defaults to `"mean"`.
        normalize_by_seq_length (bool, optional): whether to normalize the loss by the sequence length. Defaults to `True`.
        kl_to_ref_coeff (float | None, optional): coefficient for KL divergence to reference model. Defaults to `None`.
        device (torch.device | None, optional): the device to use for the loss, when tokenizing the input. Defaults to `None`.

    .. note::
        The input tensordict is expected to contain the following keys by default:
            - ``("next", "history")``: The chat history
            - ``("next", "is_chosen")``: Boolean tensor indicating which response is chosen (True) vs rejected (False)
            - ``("next", "ref_log_prob")`` (optional): Reference model log probabilities, required if kl_to_ref_coeff is set

        These keys can be customized using the ``set_keys()`` method.

    .. seealso:: :class:`~torchrl.envs.llm.transforms.RetrieveLogProb` for the KL divergence computation.

    References:
        - Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
          `"Direct Preference Optimization: Your Language Model is Secretly a Reward Model" <https://arxiv.org/abs/2305.18290>`_

    Examples:
        >>> from torchrl.data.llm.chat import History, _CHAT_TEMPLATES
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.objectives.llm.dpo import DPOLoss
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, lazy_stack
        >>> import torch
        >>>
        >>> # Create preference data
        >>> chats = [
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What's 2+2?"},
        ...         {"role": "assistant", "content": "2+2 equals 4."},  # chosen
        ...         {"role": "assistant", "content": "I don't know."},  # rejected
        ...     ],
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Explain quantum physics."},
        ...         {"role": "assistant", "content": "Quantum physics is complex."},  # chosen
        ...         {"role": "assistant", "content": "It's magic."},  # rejected
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
        >>> # Prepare data with preference labels
        >>> text = history[:, :-2].apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        ... )
        >>> text_chosen = history[:, -2:-1].apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        ... )
        >>> text_rejected = history[:, -1:].apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        ... )
        >>> 
        >>> # Create preference labels (True for chosen, False for rejected)
        >>> is_chosen = torch.tensor([True, False, True, False]).reshape(2, 2)
        >>> 
        >>> td = TensorDict(
        ...     text=text,
        ...     text_chosen=text_chosen,
        ...     text_rejected=text_rejected,
        ...     history=history,
        ...     next=TensorDict(
        ...         is_chosen=is_chosen,
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
        >>> # Use with DPOLoss
        >>> loss = DPOLoss(
        ...     actor_network=policy_train,
        ...     tokenizer=tokenizer,
        ...     beta=0.1,
        ...     reduction="mean",
        ...     normalize_by_seq_length=True,
        ...     kl_to_ref_coeff=0.1,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ... )
        >>> loss_vals = loss(data)
        >>> print(f"DPO Loss: {loss_vals.loss_dpo.item():.4f}")
        >>> print(f"KL to Reference Loss: {loss_vals.loss_kl_to_ref.item():.4f}")
        >>> print(f"Accuracy: {loss_vals.accuracy.item():.4f}")

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            history (NestedKey): The input tensordict key where the chat history is expected.
                Defaults to ``("next", "history")``.
            is_chosen (NestedKey): The input tensordict key where the preference labels are expected.
                Defaults to ``("next", "is_chosen")``.
            ref_log_prob (NestedKey): The input tensordict key where the reference model log probabilities are expected.
                Only used when kl_to_ref_coeff is set. Defaults to ``("next", "ref_log_prob")``.
            log_probs (NestedKey): The output tensordict key where the model's log probabilities will be written.
                Defaults to ``"log_probs"``.
        """

        history: NestedKey = ("next", "history")
        is_chosen: NestedKey = ("next", "is_chosen")
        ref_log_prob: NestedKey = ("next", "ref_log_prob")
        log_probs: NestedKey = "log_probs"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        actor_network: TensorDictModule | TransformersWrapper,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        tokenizer_kwargs: dict | None = None,
        beta: float = 0.1,
        reduction: Literal["mean", "sum", "none"] = "mean",
        normalize_by_seq_length: bool = True,
        kl_to_ref_coeff: float | None = None,
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
        self.beta = beta
        self.reduction = reduction
        self.normalize_by_seq_length = normalize_by_seq_length
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self._set_in_keys()
        self.device = device

    def _set_in_keys(self) -> None:
        """Sets the input keys for the loss module."""
        in_keys = [self.tensor_keys.history, self.tensor_keys.is_chosen]
        if self.kl_to_ref_coeff is not None:
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
        if cur_log_prob.shape != ref_log_prob.shape:
            raise ValueError(
                f"Current log probabilities and reference log probabilities have different shapes: {cur_log_prob.shape=} vs {ref_log_prob.shape=}."
            )
        # Compute KL using same approximation as GRPO
        diff = ref_log_prob - cur_log_prob

        kl_penalty = (diff.expm1() - diff).mean()
        return self.kl_to_ref_coeff * kl_penalty, kl_penalty

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather history and preference labels
        history: History = tensordict[self.tensor_keys.history]
        is_chosen: torch.Tensor = tensordict[self.tensor_keys.is_chosen]

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
        assistant_masks = [mask.bool() for mask in assistant_masks]
        attention_mask = token_struct.get("attention_mask", as_list=True)
        attention_mask = [mask.bool() for mask in attention_mask]
        assistant_masks = [
            mask & a_mask for mask, a_mask in zip(assistant_masks, attention_mask)
        ]

        if not any(mask.any(-1).all() for mask in assistant_masks):
            raise ValueError("Some inputs have no valid assistant masks.")

        input_loss = tensordict.select(self.tensor_keys.history)
        if (
            isinstance(self.tensor_keys.history, tuple)
            and self.tensor_keys.history[0] == "next"
        ):
            input_loss = input_loss["next"]

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
            raise ValueError(
                f"Assistant masks and log_probs have different shapes: {[mask.shape for mask in assistant_masks]} vs {[lp.shape for lp in log_probs]}. Tokens from current template: {[inp.shape for inp in token_struct.get('input_ids', as_padded_tensor=True)]}"
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

        # Split log probs into chosen and rejected based on preference labels
        chosen_mask = is_chosen.bool()
        rejected_mask = ~is_chosen.bool()
        
        if not chosen_mask.any() or not rejected_mask.any():
            raise ValueError("Both chosen and rejected responses must be present in the batch.")
        
        policy_chosen_logps = summed_log_probs[chosen_mask]
        policy_rejected_logps = summed_log_probs[rejected_mask]
        
        # Get reference log probabilities if available
        if self.kl_to_ref_coeff is not None:
            ref_log_probs = tensordict.get(
                self.tensor_keys.ref_log_prob,
                default=None,
                as_list=True,
            )
            if ref_log_probs is None:
                raise ValueError(
                    "Reference log probs not found in tensordict but kl_to_ref_coeff was set"
                )
            
            # Sum reference log probs similarly to policy log probs
            summed_ref_log_probs = torch.stack([lp.sum() for lp in ref_log_probs]).to(
                summed_log_probs.device
            )
            if self.normalize_by_seq_length:
                summed_ref_log_probs = summed_ref_log_probs / seq_lengths.clamp(min=1)
            
            reference_chosen_logps = summed_ref_log_probs[chosen_mask]
            reference_rejected_logps = summed_ref_log_probs[rejected_mask]
        else:
            # If no reference model, use zeros (equivalent to no reference model in DPO)
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # Compute DPO loss
        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            self.beta,
            self.reduction,
        )

        # Compute additional metrics for logging
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        # Add KL divergence loss if reference model is provided
        if self.kl_to_ref_coeff is not None:
            loss_kl, kl_penalty = self._kl_to_ref(
                [lp[mask] for lp, mask in _zip_strict(log_probs, assistant_masks)],
                ref_log_probs,
            )
            output = DPOLossOutput(
                loss_dpo=loss,
                loss_kl_to_ref=loss_kl,
                kl_to_ref=kl_penalty.detach(),
                chosen_rewards=chosen_rewards.detach(),
                rejected_rewards=rejected_rewards.detach(),
                accuracy=accuracy,
            )
        else:
            output = DPOLossOutput(
                loss_dpo=loss,
                chosen_rewards=chosen_rewards.detach(),
                rejected_rewards=rejected_rewards.detach(),
                accuracy=accuracy,
            )

        return output 