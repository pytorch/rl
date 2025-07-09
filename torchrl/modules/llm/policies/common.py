# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref
from typing import Any, Literal, overload

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.tensorclass import TensorClass
from tensordict.utils import _zip_strict
from torch import distributions as D
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torchrl.data.llm import History
from torchrl.data.tensor_specs import Unbounded
from torchrl.modules.distributions.discrete import LLMMaskedCategorical

# TODOs:
# - [ ] Remove the useless view(-1) calls when num_samples is not > 1
# - [ ] Remove as_list=True and use a context manager to handle that
# - [ ] Make sure tensordict can handle nested lazy tds that have a get(key, as_list=True) - I think it breaks atm
# - [ ] Handle packing


class Tokens(TensorClass["nocast"]):
    """A Tokens container.

    Args:
        prompt (torch.Tensor | None): The prompt tokens.
        response (torch.Tensor | None): The response tokens.
        assistant (torch.Tensor | None): The assistant tokens.
        full (torch.Tensor | None): The tokens across prompt and response.
        padded (bool | None): Whether the tokens are padded.

    Shapes:
        - prompt: (batch_size, prompt_length). If padded, padded on the left.
        - response: (batch_size, response_length). If padded, padded on the right.
        - full: (batch_size, prompt_length + response_length). If padded, padded on the left and/or right.
        - padded: bool.

    """

    prompt: torch.Tensor | None = None
    response: torch.Tensor | None = None
    full: torch.Tensor | None = None
    padded: bool | None = None

    @classmethod
    def default_spec(
        cls,
        shape=(-1,),
        keys: list[Literal["prompt", "response", "full"]] | None = None,
    ):
        """A default spec to use in transforms / envs that return Tokens objects."""
        from torchrl.data import Composite, NonTensor

        if keys is None:
            keys = ["prompt", "response", "full"]

        defaults = {k: Unbounded(shape=shape + (-1,)) for k in keys}
        defaults["padded"] = NonTensor(shape=shape, example_data=False)

        return Composite(defaults, shape=shape[:-1], data_cls=cls, step_mdp_static=True)


class Masks(TensorClass["nocast"]):
    """A Masks container.

    Args:
        all_attention_mask (torch.Tensor | None): The attention mask across all tokens. The attention mask represents
            the tokens that are not masked. and that the model can attend to.
        all_assistant_mask (torch.Tensor | None): The assistant mask across all tokens, i.e. the tokens that
            are produced by the assistant.
            This is recovered from the the `assistant_masks` output of :meth:`~torchrl.data.llm.History.apply_chat_template`,
            if the chat template supports it.
        padded (bool | None): Whether the masks are padded.

    The masks always have the same shape as the `full` tensor in :class:`~torchrl.modules.llm.policies.common.Tokens`,
    and :class:`~torchrl.modules.llm.policies.common.LogProbs`.

    """

    all_attention_mask: torch.Tensor | None = None
    all_assistant_mask: torch.Tensor | None = None
    padded: bool | None = None

    @classmethod
    def default_spec(
        cls,
        shape=(-1,),
        keys: list[Literal["all_attention_mask", "all_assistant_mask"]] | None = None,
    ):
        """A default spec to use in transforms / envs that return Masks objects."""
        from torchrl.data import Composite, NonTensor

        if keys is None:
            keys = ["all_attention_mask", "all_assistant_mask"]

        defaults = {k: Unbounded(shape=shape + (-1,)) for k in keys}
        defaults["padded"] = NonTensor(shape=shape, example_data=False)

        return Composite(defaults, shape=shape[:-1], data_cls=cls, step_mdp_static=True)


class ChatHistory(TensorClass["nocast"]):
    """A chat history container for managing conversation data in LLM environments.

    This class serves as a structured container for chat history data, similar to how
    :class:`~torchrl.modules.llm.policies.Text` and :class:`~torchrl.modules.llm.policies.Tokens`
    are used for text and token data respectively.

    **Recent Changes:**
    - **Modular Design**: ChatHistory is now used consistently across LLM wrappers and environments
      to represent conversation state in a structured way.
    - **Integration with Wrappers**: Both vLLMWrapper and TransformersWrapper now use ChatHistory
      objects when `input_mode="history"` is specified.
    - **Environment Support**: ChatEnv and related environments use ChatHistory for state management.

    Args:
        prompt (History | None): The prompt history stack containing the conversation up to the current point.
        response (History | None): The response history items (typically generated by the LLM).
        full (History | None): The complete history across prompt and response.

    Example:
        >>> from torchrl.data.llm import History
        >>> from torchrl.modules.llm.policies import ChatHistory
        >>>
        >>> # Create a conversation history
        >>> history = History.from_chats([[
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]])
        >>>
        >>> # Create ChatHistory object for LLM wrapper input
        >>> chat_history = ChatHistory(prompt=history)
        >>>
        >>> # Use with LLM wrapper
        >>> result = wrapper(TensorDict(history=chat_history, batch_size=(1,)))
        >>> print(result["history"].response)  # New response from LLM
        >>> print(result["history"].full)      # Complete conversation

    .. seealso::
        :class:`~torchrl.modules.llm.policies.Text`: Container for text data.
        :class:`~torchrl.modules.llm.policies.Tokens`: Container for token data.
        :class:`~torchrl.data.llm.History`: The underlying History class for conversation data.
    """

    prompt: History | None = None
    response: History | None = None
    full: History | None = None

    @classmethod
    def default_spec(
        cls,
        shape=(-1,),
        keys: list[Literal["prompt", "response", "full"]] | None = None,
    ):
        """A default spec to use in transforms / envs that return ChatHistory objects."""
        from torchrl.data import Composite

        if keys is None:
            keys = ["prompt", "response", "full"]
        return Composite(
            {k: History.default_spec(shape=shape + (-1,)) for k in keys},
            shape=shape[:-1],
            data_cls=cls,
            step_mdp_static=True,
        )


class LogProbs(TensorClass["nocast"]):
    """A log-probability container.

    Args:
        prompt (torch.Tensor | None): The prompt log-probabilities.
        response (torch.Tensor | None): The response log-probabilities.
        assistant (torch.Tensor | None): The assistant log-probabilities.
        full (torch.Tensor | None): The log-probabilities across prompt and response.
        padded (bool | None): Whether the log-probabilities are padded.

    Shapes:
        - prompt: (batch_size, prompt_length). If padded, padded on the left.
        - response: (batch_size, response_length). If padded, padded on the right.
        - full: (batch_size, prompt_length + response_length). If padded, padded on the left and/or right.
        - padded: bool.

    """

    prompt: torch.Tensor | None = None
    response: torch.Tensor | None = None
    full: torch.Tensor | None = None
    padded: bool | None = None

    @classmethod
    def default_spec(
        cls,
        shape=(-1,),
        keys: list[Literal["prompt", "response", "full"]] | None = None,
    ):
        """A default spec to use in transforms / envs that return LogProbs objects."""
        from torchrl.data import Composite, NonTensor

        if keys is None:
            keys = ["prompt", "response", "full"]

        defaults = {k: Unbounded(shape=shape + (-1,)) for k in keys}
        defaults["padded"] = NonTensor(shape=shape, example_data=False)

        return Composite(defaults, shape=shape[:-1], data_cls=cls, step_mdp_static=True)


class Text(TensorClass["nocast"]):
    """A text container.

    Args:
        prompt (str | None): The prompt text.
        response (str | None): The response text.
        full (str | None): The text across prompt and response.
    """

    prompt: str | None = None
    response: str | None = None
    full: str | None = None

    @classmethod
    def default_spec(
        cls,
        shape=(-1,),
        keys: list[Literal["prompt", "response", "full"]] | None = None,
    ):
        """A default spec to use in transforms / envs that return Text objects."""
        from torchrl.data import Composite, NonTensor

        if keys is None:
            keys = ["prompt", "response", "full"]

        defaults = {k: NonTensor(shape=shape, example_data="a string") for k in keys}

        return Composite(defaults, shape=shape[:-1], data_cls=cls, step_mdp_static=True)


class LogProbDistribution(D.Distribution):
    """A distribution that works directly with log-probabilities.

    This is useful when we have pre-computed log-probabilities (e.g., from vLLM)
    and want to compute log_prob() without having access to the original logits.
    """

    def __init__(self, log_probs: torch.Tensor, mask: torch.Tensor | None = None):
        """Initialize with log-probabilities.

        Args:
            log_probs: Tensor of shape [batch, seq_len] containing log-probabilities
            mask: Optional mask of shape [batch, seq_len] indicating valid positions
        """
        self.log_probs = log_probs
        self.mask = mask
        batch_shape = log_probs.shape[:-1] if log_probs.dim() > 1 else log_probs.shape
        event_shape = log_probs.shape[-1:] if log_probs.dim() > 1 else torch.Size([])
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log-probability for the given tokens.

        Args:
            value: Tensor of shape [batch, seq_len] containing token indices

        Returns:
            Tensor of shape [batch, seq_len] containing log-probabilities
        """
        # For log-prob distributions, we just return the pre-computed log-probs
        # at the positions specified by the value tensor
        if value.shape != self.log_probs.shape:
            raise ValueError(
                f"Value shape {value.shape} must match log_probs shape {self.log_probs.shape}"
            )

        result = self.log_probs.clone()

        # Apply mask if provided
        if self.mask is not None:
            result = torch.where(
                self.mask,
                result,
                torch.tensor(0.0, device=result.device, dtype=result.dtype),
            )

        return result

    def sample(self, sample_shape: tuple | torch.Size | None = None) -> torch.Tensor:
        """Sample from the distribution.

        Note: This is not implemented for log-prob distributions since we don't have
        the full probability distribution, only the log-probs for specific tokens.
        """
        raise NotImplementedError("Sampling not supported for LogProbDistribution")

    def entropy(self) -> torch.Tensor:
        """Compute entropy.

        Note: This is not implemented for log-prob distributions since we don't have
        the full probability distribution.
        """
        raise NotImplementedError("Entropy not supported for LogProbDistribution")


class LLMWrapperBase(TensorDictModuleBase):
    r"""A LLM wrapper base class.

    This class provides a consistent interface for LLM wrappers with the following features:
    - Support for different input modalities (history, text, tokens)
    - Consistent output structure using TensorClass objects (Text, Tokens, Masks, LogProbs)
    - Configurable generation and log-probability computation

    Args:
        model: The underlying model to wrap.

    Keyword Args:
        tokenizer: The tokenizer to use for encoding and decoding text.
        input_mode: The input modality to use. Must be one of "history", "text", or "tokens".
        input_key: The key for the input data. If None, defaults to the input_mode name.
        attention_mask_key: The key for attention masks (used in "tokens" mode).
        generate: Whether to enable text generation.
        generate_kwargs: Additional arguments to pass to the model's generate method.
        tokenizer_kwargs: Additional arguments to pass to the tokenizer.
        pad_output: Whether to pad the output sequences to a uniform length.
        inplace: Determines how the module should handle in-place operations.
        device: The device to use for computation.
        layout: The layout to use for the output tensors when pad_output=False.
        num_samples: The number of samples to generate.
        log_probs_key (NestedKey | None, optional): The key for the log probabilities :class:`~torchrl.modules.llm.policies.LogProbs` object. Defaults to `"log_probs"`.
        text_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Text` object. Defaults to `"text"`.
        tokens_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Tokens` object. Defaults to `"tokens"`.
        masks_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Masks` object. Defaults to `"masks"`.

    Attributes:
        collector: The collector associated with the module, if it exists.

    .. seealso::
        - :class:`~torchrl.modules.llm.policies.TransformersWrapper` (see :ref:`ref_transformers_wrapper`)
        - :class:`~torchrl.modules.llm.policies.vLLMWrapper` (see :ref:`ref_vllm_wrapper`)
    """

    generate: bool
    pad_output: bool
    text_key: NestedKey
    tokens_key: NestedKey
    masks_key: NestedKey
    log_probs_key: NestedKey
    in_keys: list[NestedKey]
    out_keys: list[NestedKey]
    inplace: bool
    device: torch.device | None
    layout: torch.layout | None
    num_samples: int | None

    @overload
    def __init__(
        self,
        model: Any | str,
        *,
        tokenizer: callable | str | None = None,  # type: ignore
        input_mode: str = "history",
        input_key: NestedKey | None = None,
        attention_mask_key: str = "attention_mask",
        generate: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        inplace: Literal[True, False, "empty"] | None = None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        chat_template_name: Literal["chatml_format", "qwen"] | None = None,
        chat_template: str | None = None,
        return_log_probs: bool | None = None,
        history_key: NestedKey | None = "history",
        text_key: NestedKey | None = "text",
        tokens_key: NestedKey | None = "tokens",
        masks_key: NestedKey | None = "masks",
        log_probs_key: NestedKey | None = "log_probs",
    ):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_new_version(self, **kwargs):
        """Returns a new version of the module with altered parameters.

        For instance, the generate parameter can be altered to enable text generation or log-probabilities computation.
        This is especially useful when one wants to avoid re-initializing the module with a new set of parameters, when the
        same parameters could be used to gather log-probs.

        Positional arguments are not supported.

        See the class constructor for more details about the parameters.
        """
        raise NotImplementedError

    _collector: weakref.ReferenceType[
        LLMCollector  # noqa: F821 # type: ignore
    ] | None = None

    def register_collector(self, collector: LLMCollector):  # noqa: F821 # type: ignore
        """Registers a weak reference to the container collector.

        This is automatically called by the :class:`~torchrl.collectors.llm.LLMCollector` class.
        """
        self._collector = weakref.ref(collector)

    @property
    def collector(self) -> LLMCollector | None:  # noqa: F821 # type: ignore
        """Returns the collector associated with the module, if it exists."""
        return self._collector() if self._collector is not None else None

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        logits_key: NestedKey = "logits",
        mask_key: NestedKey | None = None,
        as_padded_tensor: bool | None = None,
        as_nested_tensor: bool | None = None,
        padding_value: float | None = None,
        padding_side: str = "left",
        layout: torch.layout | None = None,
        **kwargs,
    ) -> D.Distribution:
        """Get distribution from logits/log-probs with optional masking.

        Args:
            tensordict: Input tensordict
            tensordict_out: Output tensordict (optional)
            logits_key: Key for logits/log-probs
            mask_key: Key for mask (optional).
            as_padded_tensor: Whether to return padded tensor. Default is False.
            as_nested_tensor: Whether to return nested tensor. Default is False.
            padding_value: Value for padding. Default is 0.0 for logits and False for masks.
            padding_side: Side for padding. Default is left by convention.
            layout: Tensor layout
            **kwargs: Additional arguments

        Returns:
            Distribution (Categorical or LLMMaskedCategorical)
        """
        if self.generate:
            raise NotImplementedError(
                "get_dist is not implemented for generate=True. "
                "You can create a new version of this wrapper using the `get_new_version` method."
            )

        td_out = self(tensordict.copy())

        # Get logits/log-probs
        if as_padded_tensor is None:
            as_padded_tensor = as_nested_tensor is not True
            if padding_value is None:
                padding_value = 0.0
        if as_nested_tensor is None:
            as_nested_tensor = False

        logits = td_out.get(
            logits_key,
            as_padded_tensor=as_padded_tensor,
            as_nested_tensor=as_nested_tensor,
            padding_value=padding_value,
            padding_side=padding_side,
            layout=layout,
        )

        # Get mask if provided
        mask = None
        if mask_key is not None:
            mask = td_out.get(
                mask_key,
                as_padded_tensor=as_padded_tensor,
                as_nested_tensor=as_nested_tensor,
                padding_value=False,
                padding_side=padding_side,
                layout=layout,
            )
        elif as_padded_tensor:
            # Default mask for padded tensors
            mask = logits != padding_value

        if mask is not None:
            dist = LLMMaskedCategorical(
                logits=logits,
                mask=mask,
            )
            if not dist._position_level_masking:
                raise ValueError(
                    "Mask is not a position-level mask. "
                    "This is likely because the mask is not a position-level mask."
                )
            return dist
        return Categorical(logits)

    def _get_dist_with_prompt_mask(
        self,
        tensordict: TensorDictBase,
        tokens_key: NestedKey = ("tokens", "prompt"),
        logits_key: NestedKey = "logits",
        # TODO: add a prompt_mask and response_mask in Masks
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        padding_side: str = "left",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include response tokens (exclude prompt).

        This is suitable for single-turn scenarios where we want to compute loss
        only on the generated response, not the input prompt.

        Note: If prompt tokens are not available (e.g., when using history input),
        this method falls back to using the assistant mask.

        Padding side is left by convention.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        if self.generate:
            raise NotImplementedError(
                "get_dist_with_prompt_mask is not implemented for generate=True. "
                "You can create a new version of this wrapper using the `get_new_version` method."
            )
        td_out = self(tensordict.copy())

        # Try to get prompt tokens first
        if self.pad_output:
            prompt_tokens = tensordict.get(
                tokens_key,
                as_padded_tensor=True,
                padding_value=-100,
                padding_side=padding_side,
            )
            logits = td_out.get(
                logits_key,
                as_padded_tensor=True,
                padding_value=0.0,
                padding_side=padding_side,
            )
            attention_mask = tensordict.get(
                attention_mask_key,
                as_padded_tensor=True,
                padding_value=False,
                padding_side=padding_side,
            )
            assistant_mask = tensordict.get(
                assistant_mask_key,
                as_padded_tensor=True,
                padding_value=False,
                padding_side=padding_side,
            )
        else:
            prompt_tokens = tensordict.get(tokens_key, as_list=True)
            logits = td_out.get(logits_key, as_list=True)
            attention_mask = td_out.get(attention_mask_key, as_list=True)
            assistant_mask = td_out.get(assistant_mask_key, as_list=True)

        if prompt_tokens is None:
            if assistant_mask is None:
                raise ValueError(
                    f"Assistant mask not found in tensordict at key {assistant_mask_key} (keys: {td_out.keys()})"
                )
            if self.pad_output:
                response_mask = assistant_mask.clone()
            else:
                response_mask = [am.clone() for am in assistant_mask]
        else:
            if self.pad_output:
                response_mask = attention_mask.clone()
                response_mask[..., : prompt_tokens.shape[-1]] = False
            else:
                response_mask = []
                for am, p in _zip_strict(attention_mask, prompt_tokens):
                    am = am.clone()
                    am[..., : p.size(-1)] = False
                    response_mask.append(am)

        if logits is None:
            raise ValueError(
                f"Logits not found in tensordict at key {logits_key} (keys: {td_out.keys()})"
            )

        # Make the response mask using prompt tokens
        if not self.pad_output:
            # Check that the lengths of the mask is the same as the logits
            for m, lg in _zip_strict(response_mask, logits):
                if m.shape[-1] != lg.shape[-2]:
                    raise ValueError(
                        f"Mask and logits have different lengths: {m.shape[-1]} != {lg.shape[-2]}.\n"
                        f"All the logits shapes: {[lg.shape for lg in logits]}, all the mask shapes: {[m.shape for m in response_mask]}"
                    )
            logits = pad_sequence(
                logits, batch_first=True, padding_value=0.0, padding_side=padding_side
            )
            response_mask = pad_sequence(
                response_mask,
                batch_first=True,
                padding_value=False,
                padding_side=padding_side,
            )

        dist = LLMMaskedCategorical(
            logits=logits,
            mask=response_mask.bool(),
        )
        if not dist._position_level_masking:
            raise ValueError(
                "Mask is not a position-level mask. "
                "This is likely because the mask is not a position-level mask."
            )
        return dist

    def _get_dist_with_assistant_mask(
        self,
        tensordict: TensorDictBase,
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        logits_key: NestedKey = "logits",
        padding_side: str = "left",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include assistant tokens.

        This is suitable for multi-turn scenarios where we want to compute loss
        only on assistant-generated tokens across the entire conversation.

        Padding side is left by convention.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        if self.generate:
            raise NotImplementedError(
                "get_dist_with_assistant_mask is not implemented for generate=True. "
                "You can create a new version of this wrapper using the `get_new_version` method."
            )
        td_out = self(tensordict.copy())
        # Update the tokens key to reflect the tokenized history when querying the log-probs
        tensordict.update(
            td_out,
            keys_to_update=[
                ("tokens", "full"),
            ],
        )

        if self.pad_output:
            logits = td_out.get(logits_key)
            assistant_mask = td_out.get(assistant_mask_key)
        else:
            logits = td_out.get(
                logits_key,
                as_padded_tensor=True,
                padding_value=0.0,
                padding_side=padding_side,
            )
            assistant_mask = td_out.get(
                assistant_mask_key,
                as_padded_tensor=True,
                padding_value=False,
                padding_side=padding_side,
            )
        if logits is None:
            raise ValueError(f"Logits not found in tensordict at key {logits_key}")
        if assistant_mask is None:
            if self.input_mode != "history":
                post_msg = "This is likely because the input_mode is not 'history'."
            else:
                post_msg = ""
            raise ValueError(
                f"Assistant mask not found in tensordict at key {assistant_mask_key}. {post_msg}"
            )

        dist = LLMMaskedCategorical(
            logits=logits,
            mask=assistant_mask,
        )
        if not dist._position_level_masking:
            raise ValueError(
                "Assistant mask is not a position-level mask. "
                "This is likely because the assistant mask is not a position-level mask."
            )
        return dist

    def _get_dist_with_attention_mask(
        self,
        tensordict: TensorDictBase,
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        logits_key: NestedKey = "logits",
        padding_side: str = "left",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked using attention mask.

        This is suitable for generic scenarios where we want to compute loss
        on all valid tokens (non-padding tokens).

        Padding side is left by convention.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        if self.generate:
            raise NotImplementedError(
                "get_dist_with_attention_mask is not implemented for generate=True. "
                "You can create a new version of this wrapper using the `get_new_version` method."
            )
        td_out = self(tensordict.copy())
        if self.pad_output:
            logits = td_out.get(logits_key)
            attention_mask = td_out.get(attention_mask_key)
        else:
            logits = td_out.get(
                logits_key,
                as_padded_tensor=True,
                padding_value=0.0,
                padding_side=padding_side,
            )
            attention_mask = td_out.get(
                attention_mask_key,
                as_padded_tensor=True,
                padding_value=False,
                padding_side=padding_side,
            )

        if logits is None:
            raise ValueError(f"Logits not found in tensordict at key {logits_key}")
        if attention_mask is None:
            raise ValueError(
                f"Attention mask not found in tensordict at key {attention_mask_key}"
            )

        dist = LLMMaskedCategorical(
            logits=logits,
            mask=attention_mask,
        )
        if not dist._position_level_masking:
            raise ValueError(
                "Attention mask is not a position-level mask. "
                "This is likely because the attention mask is not a position-level mask."
            )
        return dist

    def _get_dist_with_custom_mask(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor,
        logits_key: NestedKey = "logits",
        padding_side: str = "left",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution with custom mask.

        This allows for completely custom masking logic.

        Padding side is left by convention.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        if self.generate:
            raise NotImplementedError(
                "get_dist_with_custom_mask is not implemented for generate=True. "
                "You can create a new version of this wrapper using the `get_new_version` method."
            )
        td_out = self(tensordict.copy())
        if self.pad_output:
            logits = td_out.get(logits_key)
        else:
            logits = td_out.get(
                logits_key,
                as_padded_tensor=True,
                padding_value=0.0,
                padding_side=padding_side,
            )

        if logits is None:
            raise ValueError(f"Logits not found in tensordict at key {logits_key}")

        dist = LLMMaskedCategorical(
            logits=logits,
            mask=mask,
        )
        if not dist._position_level_masking:
            raise ValueError(
                "Custom mask is not a position-level mask. "
                "This is likely because the custom mask is not a position-level mask."
            )
        return dist

    # Convenience methods for common LLM training scenarios
    def _get_sft_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for SFT loss (response tokens only).

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        return self._get_dist_with_prompt_mask(tensordict, **kwargs)

    def _get_rlhf_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for RLHF loss (assistant tokens only).

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        return self._get_dist_with_assistant_mask(tensordict, **kwargs)

    def _get_generic_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for generic losses (all tokens).

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        return self._get_dist_with_attention_mask(tensordict, **kwargs)

    # Sampling is taken care of by the sub-modules
    forward = TensorDictSequential.forward

    def _check_padded(self, val: torch.Tensor) -> torch.Tensor:
        """Check that a value is a padded tensor."""
        assert isinstance(
            val, torch.Tensor
        ), f"val must be torch.Tensor, got {type(val)}"
        if not isinstance(val, torch.Tensor):
            raise ValueError("Not a padded tensor")
        return val

    def _check_not_padded(
        self, val: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor] | torch.Tensor:
        """Check that a value is not a padded tensor (i.e., a list of tensors)."""
        if isinstance(val, torch.Tensor):
            raise ValueError("Expected a list of tensors - not padded, got a tensor")
        return val

    @property
    def log_prob_keys(self) -> list[NestedKey]:
        return getattr(self, "_log_prob_keys", ["log_probs"])

    @log_prob_keys.setter
    def log_prob_keys(self, value: list[NestedKey]):
        self._log_prob_keys = value

    @property
    def dist_params_keys(self) -> list[NestedKey]:
        raise NotImplementedError

    @property
    def dist_sample_keys(self) -> list[NestedKey]:
        return ["tokens_response"]

    def log_prob(self, data: TensorDictBase, **get_kwargs) -> TensorDictBase:
        if not self.generate:
            data = self(data)
            return data.get((self.log_prob_key, "response"), **get_kwargs)
        raise RuntimeError("log_prob not callable when generate=True.")


def _extract_responses_from_full_histories(
    text_full: list[str],
    prompt_histories,
    chat_template_name: str | None = None,
    tokenizer=None,
) -> History:
    """Extract response histories from full text histories.

    This function parses the full text back to history objects and extracts
    the response portions (everything after the prompt).

    Args:
        text_full: List of full text strings to parse
        prompt_histories: The original prompt histories
        chat_template_name: Optional chat template name for parsing
        tokenizer: Optional tokenizer for template detection

    Returns:
        Stacked History object with response portions

    Raises:
        RuntimeError: If full history is shorter than prompt history
        RuntimeError: If parsing produces inconsistent batch shapes
    """
    import torch
    from tensordict.utils import _zip_strict
    from torchrl.data.llm import History

    # Extract response portions by processing each element individually
    # This avoids the stacking issue when different batch elements produce
    # different numbers of responses
    response_histories = []
    full_histories = History.from_text(
        text_full, chat_template_name=chat_template_name, tokenizer=tokenizer
    )
    for h_prompt, h_full in _zip_strict(
        prompt_histories.unbind(0), full_histories.unbind(0)
    ):
        if h_full.shape[0] <= h_prompt.shape[0]:
            raise RuntimeError(
                f"Full history is shorter than prompt history: {h_full.shape} <= {h_prompt.shape}"
            )
        # Note: there can be more than one response, so the response has the same number of dims as prompt
        response_histories.append(h_full[h_prompt.shape[0] :])

    # Check if all responses have the same shape
    shapes = [r.shape for r in response_histories]
    if len(set(shapes)) > 1:
        # Different shapes detected - pad to the same length
        max_length = max(r.shape[0] for r in response_histories)
        padded_responses = []
        for response in response_histories:
            if response.shape[0] < max_length:
                # Pad with empty messages using "<none>" role
                padding_needed = max_length - response.shape[0]
                padding_history = History(
                    role="<none>", content="", batch_size=(padding_needed,)
                )
                padded_response = response.extend(padding_history, inplace=False)
                padded_responses.append(padded_response)
            else:
                padded_responses.append(response)
        return torch.stack(padded_responses)

    return torch.stack(response_histories)
