# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref

from typing import Any, overload

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.tensorclass import TensorClass
from torch import distributions as D
from torch.distributions import Categorical
from torchrl.modules import MaskedCategorical

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


class Text(TensorClass["nocast"]):
    """A text container.

    Args:
        prompt (str | None): The prompt text.
        response (str | None): The response text.
        full (str | None): The text across prompt and response.
        padded (bool | None): Whether the text is padded.
    """

    prompt: str | None = None
    response: str | None = None
    full: str | None = None
    padded: bool | None = None


class CategoricalSequential(TensorDictModuleBase):
    """A ProbabilisticTensorDictSequential subclass meant to work with LLMs.

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
        return_log_probs: Whether to return log probabilities.
        return_text: Whether to return text outputs.
        return_tokens: Whether to return token outputs.
        return_masks: Whether to return mask outputs.
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

    .. seealso:: :class:`~tensordict.nn.ProbabilisticTensorDictSequential` class.
    """

    generate: bool
    pad_output: bool
    log_prob_key: NestedKey
    text_key: NestedKey
    tokens_key: NestedKey
    masks_key: NestedKey
    in_keys: list[NestedKey]
    out_keys: list[NestedKey]
    inplace: bool
    device: torch.device | None
    layout: torch.layout | None
    num_samples: int | None

    @overload
    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        input_mode: str = "history",
        input_key: str | None = None,
        attention_mask_key: str = "attention_mask",
        generate: bool = True,
        return_log_probs: bool = False,
        return_text: bool = True,
        return_tokens: bool = True,
        return_masks: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        inplace=None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        log_probs_key: NestedKey | None = ("log_probs", "response"),
        text_key: NestedKey | None = "text",
        tokens_key: NestedKey | None = "tokens",
        masks_key: NestedKey | None = "masks",
    ):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overload
    def get_new_version(
        self,
        model: Any | None = None,
        *,
        tokenizer=None,
        input_mode: str | None = None,
        input_key: str | None = None,
        attention_mask_key: str | None = None,
        generate: bool | None = None,
        return_log_probs: bool | None = None,
        return_text: bool | None = None,
        return_tokens: bool | None = None,
        return_masks: bool | None = None,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool | None = None,
        inplace: bool | None = None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        log_probs_key: NestedKey | None = None,
        text_key: NestedKey | None = None,
        tokens_key: NestedKey | None = None,
        masks_key: NestedKey | None = None,
        **kwargs,
    ):
        ...

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
        as_padded_tensor: bool | None = None,
        as_nested_tensor: bool | None = None,
        padding_value: float | None = None,
        padding_side: str = "right",
        layout: torch.layout | None = None,
        **kwargs,
    ) -> D.Distribution:
        td_out = self(tensordict.copy())
        # By default, pad and use masked categorical
        if as_padded_tensor is None:
            as_padded_tensor = as_nested_tensor is not True
            if padding_value is None:
                padding_value = 0.0
        if as_nested_tensor is None:
            as_nested_tensor = False
        logits = td_out.get(
            "logits",
            as_padded_tensor=as_padded_tensor,
            as_nested_tensor=as_nested_tensor,
            padding_value=padding_value,
            padding_side=padding_side,
            layout=layout,
        )
        if as_padded_tensor:
            # We can use MaskedCategorical
            dist = MaskedCategorical(
                logits=logits,
                mask=logits != padding_value,
                use_cross_entropy=True,
            )
            return dist
        return Categorical(logits)

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
    def log_prob_key(self) -> NestedKey:
        return self.log_prob_keys[0]

    @log_prob_key.setter
    def log_prob_key(self, value: NestedKey) -> None:
        self.log_prob_keys[0] = value

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
