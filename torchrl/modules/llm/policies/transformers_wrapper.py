# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib

from contextlib import nullcontext
from copy import copy
from typing import Literal

import torch
from tensordict import (
    lazy_stack,
    MetaData,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import _zip_strict, NestedKey
from torch import distributions as D
from torch.nn.utils.rnn import pad_sequence

from torchrl.modules.llm.policies.common import (
    _extract_responses_from_full_histories,
    ChatHistory,
    LLMWrapperBase,
    LogProbs,
    Masks,
    Text,
    Tokens,
)
from torchrl.modules.utils.utils import _unpad_tensors


class TransformersWrapper(LLMWrapperBase):
    """A wrapper class for Hugging Face Transformers models, providing a consistent interface for text generation and log probability computation.

    This class is a subclass of :class:`~torchrl.modules.llm.policies.LLMWrapperBase` and provides a unified API for handling different input modalities
    (history, text, tokens) with consistent output structure using :class:`~tensordict.TensorClass` objects.

    Args:
        model (transformers.AutoModelForCausalLM | str): The Hugging Face Transformers model to wrap.
            If a string, it will be passed to `transformers.AutoModelForCausalLM.from_pretrained`.

    Keyword Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | str | None, optional): The tokenizer to use for
            encoding and decoding text. If `None`, the tokenizer associated with the model will be used.
            If a string, it will be passed to `transformers.AutoTokenizer.from_pretrained`. Defaults to `None`.
        input_mode (str, optional): The input modality to use. Must be one of `"history"`, `"text"`, or `"tokens"`.
            Defaults to `"history"`.
        input_key (str | None, optional): The key for the input data. If `None`, defaults to
            - `("history", "prompt")` for `"history"` when `generate=True`, `("history", "full")` for `"history"` when `generate=False`
            - `("text", "prompt")` for `"text"` when `generate=True`, `("text", "full")` for `"text"` when `generate=False`
            - `("tokens", "prompt")` for `"tokens"` when `generate=True`, `("tokens", "full")` for `"tokens"` when `generate=False`
        attention_mask_key (str, optional): The key for attention masks (used in `"tokens"` mode). Defaults to `"attention_mask"`.

                    .. warning:: This argument is under development and may change in the future.

        generate (bool, optional): Whether to enable text generation. If `True`, the model will generate text based on the input.
            If `False`, only log probabilities will be computed. Defaults to `True`.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `False`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to the model's generate method. Defaults to `None`.
        tokenizer_kwargs (dict | None, optional): Additional arguments to pass to the tokenizer. Defaults to `None`.
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. Transformers require `pad_output=True`, and the output
            sequences will be padded and represented as tensors. Defaults to `False`.
        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place operations. Defaults to `True`.
        device (torch.device | None, optional): The device to use for computation. Defaults to `None`.
        layout (torch.layout | None, optional): The layout to use for the output tensors when `pad_output=False`. Defaults to `torch.strided`.
        num_samples (int | None, optional): The number of samples to generate. Defaults to `None` (one sample, and no batch-dimension for it).
            Can also be set via the `generate_kwargs["num_return_sequences"] = value` argument. Requires the "do_sample" argument to be set to `True` in `generate_kwargs`.
        chat_template_name (Literal["chatml_format", "qwen"] | None, optional): The name of the chat template to use when applying the chat
            template to the history. Defaults to `None`. For `input_mode="history"` only.
        chat_template (str | None, optional): The chat template to use when applying the chat template to the history.
            Defaults to `None`. For `input_mode="history"` only.
        log_probs_key (NestedKey | None, optional): The key for the log probabilities :class:`~torchrl.modules.llm.policies.LogProbs` object. Defaults to `"log_probs"`.
        text_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Text` object. Defaults to `"text"`.
        tokens_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Tokens` object. Defaults to `"tokens"`.
        masks_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Masks` object. Defaults to `"masks"`.
        history_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.ChatHistory` object. Defaults to `"history"`.

    Input Keys:
        The input key depends on both `input_mode` and `generate`:
        - If `input_mode="history"` and `generate=True`: `input_key` (defaults to `("history", "prompt")`)
        - If `input_mode="history"` and `generate=False`: `input_key` (defaults to `("history", "full")`)
        - If `input_mode="text"` and `generate=True`: `input_key` (defaults to `("text", "prompt")`)
        - If `input_mode="text"` and `generate=False`: `input_key` (defaults to `("text", "full")`)
        - If `input_mode="tokens"` and `generate=True`: `input_key` (defaults to `("tokens", "prompt")`)
        - If `input_mode="tokens"` and `generate=False`: `input_key` (defaults to `("tokens", "full")`)

    Output Keys:
        The output keys are automatically determined based on the input_mode:
        - **Tokens**: Always returned (`tokens_key`, defaults to `"tokens"`)
        - **Text**: Returned for `"text"` and `"history"` modes (`text_key`, defaults to `"text"`)
        - **History**: Returned only for `"history"` mode (`history_key`, defaults to `"history"`)
        - **Masks**: Always returned (`masks_key`, defaults to `"masks"`)
        - **Log Probs**: Returned when `return_log_probs=True` (`log_probs_key`, defaults to `"log_probs"`)

        Example output structure for `input_mode="history"`:
        ```
        TensorDict(
            text=Text(prompt=..., response=..., full=...),
            masks=Masks(all_attention_mask=..., all_assistant_mask=...),
            tokens=Tokens(prompt=..., response=..., full=...),
            log_probs=LogProbs(prompt=..., response=..., full=...),
            history=ChatHistory(prompt=..., response=..., full=...)
        )
        ```

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from torchrl.data.llm import History
        >>> from torchrl.modules.llm.policies import ChatHistory
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # History input (recommended for RL environments)
        >>> wrapper = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     input_mode="history",
        ...     generate=True,
        ...     return_log_probs=True
        ... )
        >>>
        >>> history = History.from_chats([[
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]])
        >>> chat_history = ChatHistory(prompt=history)
        >>> result = wrapper(TensorDict(history=chat_history, batch_size=(1,)))
        >>> print(result["text"].response)  # Generated text
        >>> print(result["log_probs"].response)  # Log probabilities
        >>> print(result["history"].response)  # History with response

    Attributes:
        collector: The collector associated with the module, if it exists.

    .. seealso::
        - :class:`~torchrl.modules.llm.policies.LLMWrapperBase` (see :ref:`ref_categorical_sequential`)
        - :class:`~torchrl.modules.llm.policies.vLLMWrapper` (see :ref:`ref_vllm_wrapper`)
    """

    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        input_mode: str = "history",
        input_key: str | None = None,
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
        super().__init__()

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model)

        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # Validate input_mode
        if input_mode not in ["history", "text", "tokens"]:
            raise ValueError(
                f"input_mode must be one of 'history', 'text', 'tokens'. Got '{input_mode}'"
            )

        self.model = model
        self.input_mode = input_mode
        self.attention_mask_key = attention_mask_key
        self.generate = generate

        # Auto-determine what to return based on input mode
        self.return_history = input_mode in ("history",)
        self.return_text = input_mode in ("text", "history")
        self.return_tokens = input_mode in ("tokens", "history", "text")
        self.return_masks = True
        if return_log_probs is False and not generate:
            raise ValueError("return_log_probs must be True when generate=False.")
        return_log_probs = (
            True
            if (return_log_probs is None and generate) or (not generate)
            else bool(return_log_probs)
        )
        self.return_log_probs = return_log_probs

        self.history_key = history_key
        self.text_key = text_key
        self.tokens_key = tokens_key
        self.masks_key = masks_key
        self.log_probs_key = log_probs_key
        if not isinstance(pad_output, bool):
            raise ValueError("pad_output must be a boolean")
        self.pad_output = pad_output
        self._device = device
        if not pad_output and layout is None:
            layout = torch.strided
        self.layout = layout
        padding_value = None

        # Auto-determine input_key if not provided

        # Set input keys based on mode and generate parameter
        if input_mode == "history":
            if generate:
                self.in_keys = [
                    ("history", "prompt") if input_key is None else input_key
                ]
            else:
                self.in_keys = [("history", "full") if input_key is None else input_key]
        elif input_mode == "text":
            if generate:
                self.in_keys = [("text", "prompt") if input_key is None else input_key]
            else:
                self.in_keys = [("text", "full") if input_key is None else input_key]
        elif input_mode == "tokens":
            if generate:
                self.in_keys = [
                    ("tokens", "prompt") if input_key is None else input_key
                ]
            else:
                self.in_keys = [("tokens", "full") if input_key is None else input_key]
        self.input_key = self.in_keys[0]

        # Set output keys based on auto-determined return flags
        self.out_keys = []
        if self.return_text:
            self.out_keys.append(self.text_key)
        if self.return_masks:
            self.out_keys.append(self.masks_key)
        if self.return_tokens:
            self.out_keys.append(self.tokens_key)
        if self.return_log_probs:
            self.out_keys.append(self.log_probs_key)
        if self.return_history:
            self.out_keys.append(self.history_key)

        # Tokenizer setup
        if not tokenizer_kwargs:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = dict(tokenizer_kwargs)
        if not tokenizer_kwargs.setdefault("return_attention_mask", True):
            raise RuntimeError("return_attention_mask must be True")

        # We always pad, so we always return tensors
        return_tensors = "pt"
        tokenizer_kwargs.setdefault("padding", True)
        if return_tensors:
            if (
                tokenizer_kwargs.setdefault("return_tensors", return_tensors)
                != return_tensors
            ):
                raise RuntimeError

        # We always pad atm
        if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
            raise RuntimeError

        self.tokenizer_kwargs = tokenizer_kwargs

        # Get tokenizer if needed
        if (
            pad_output or (input_mode in ["text", "history"] and not generate)
        ) and tokenizer is None:
            tokenizer = model.get_tokenizer()
        self.tokenizer = tokenizer

        if self.tokenizer is not None and (
            not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer is not None:
            padding_value = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.padding_value = padding_value

        # Generate kwargs setup
        if generate_kwargs is None:
            generate_kwargs = {}
        else:
            generate_kwargs = dict(generate_kwargs)

        self.num_samples = num_samples
        if (
            generate_kwargs.get("num_return_sequences", 1) > 1
            or num_samples is not None
        ):
            if inplace in (True, "empty"):
                raise ValueError(
                    "inplace must be False (or None) when generating more than one sample."
                )
            if inplace is None:
                inplace = False
            if (
                generate_kwargs.get("num_return_sequences", 1) > 1
                and num_samples is not None
                and generate_kwargs.get("num_return_sequences", 1) != num_samples
            ):
                raise ValueError("num_samples differs from generate_kwargs['n'].")
            elif num_samples is None:
                self.num_samples = generate_kwargs.get("num_return_sequences", 1)
            generate_kwargs["num_return_sequences"] = self.num_samples
        elif inplace is None:
            inplace = True

        self.inplace = inplace

        if not generate:
            # We want only the log-probs, we generate a single token (that we then discard)
            # and retrieve the prompt log-probs
            generate_kwargs["max_tokens"] = 1

        generate_kwargs.setdefault("tokenizer", self.tokenizer)
        generate_kwargs.setdefault("output_logits", self.return_log_probs)
        generate_kwargs.setdefault("return_dict_in_generate", True)

        self.generate_kwargs = generate_kwargs

        # Additional transformers-specific settings
        self.chat_template_name = chat_template_name
        self.chat_template = chat_template

        # Flag to track when we're in a get_dist call
        self._in_get_dist_call = False

    def get_new_version(self, **kwargs):
        """Returns a new version of the module with altered parameters.

        For instance, the generate parameter can be altered to enable text generation or log-probabilities computation.
        This is especially useful when one wants to avoid re-initializing the module with a new set of parameters, when the
        same parameters could be used to gather log-probs.

        Positional arguments are not supported.

        See the class constructor for more details about the parameters.
        """
        # Build the constructor arguments by using current values for missing parameters
        constructor_kwargs = {}

        # Model is always required
        constructor_kwargs["model"] = kwargs.get("model", self.model)

        # Check for each parameter and use current value if not provided
        if "tokenizer" in kwargs:
            constructor_kwargs["tokenizer"] = kwargs["tokenizer"]
        elif hasattr(self, "tokenizer"):
            constructor_kwargs["tokenizer"] = self.tokenizer

        if "input_mode" in kwargs:
            constructor_kwargs["input_mode"] = kwargs["input_mode"]
        elif hasattr(self, "input_mode"):
            constructor_kwargs["input_mode"] = self.input_mode

        if "input_key" in kwargs:
            constructor_kwargs["input_key"] = kwargs["input_key"]
        elif hasattr(self, "input_key"):
            constructor_kwargs["input_key"] = self.input_key

        if "attention_mask_key" in kwargs:
            constructor_kwargs["attention_mask_key"] = kwargs["attention_mask_key"]
        elif hasattr(self, "attention_mask_key"):
            constructor_kwargs["attention_mask_key"] = self.attention_mask_key

        if "generate" in kwargs:
            constructor_kwargs["generate"] = kwargs["generate"]
        elif hasattr(self, "generate"):
            constructor_kwargs["generate"] = self.generate

        if "generate_kwargs" in kwargs:
            constructor_kwargs["generate_kwargs"] = kwargs["generate_kwargs"]
        elif hasattr(self, "generate_kwargs"):
            constructor_kwargs["generate_kwargs"] = self.generate_kwargs

        if "pad_output" in kwargs:
            constructor_kwargs["pad_output"] = kwargs["pad_output"]
        elif hasattr(self, "pad_output"):
            constructor_kwargs["pad_output"] = self.pad_output

        if "tokenizer_kwargs" in kwargs:
            constructor_kwargs["tokenizer_kwargs"] = kwargs["tokenizer_kwargs"]
        elif hasattr(self, "tokenizer_kwargs"):
            constructor_kwargs["tokenizer_kwargs"] = self.tokenizer_kwargs
            if (
                "pad_output" in kwargs
                and kwargs.get("pad_output")
                != constructor_kwargs["tokenizer_kwargs"]["padding"]
            ):
                constructor_kwargs["tokenizer_kwargs"]["padding"] = kwargs.get(
                    "pad_output"
                )

        if "inplace" in kwargs:
            constructor_kwargs["inplace"] = kwargs["inplace"]
        elif hasattr(self, "inplace"):
            constructor_kwargs["inplace"] = self.inplace

        if "device" in kwargs:
            constructor_kwargs["device"] = kwargs["device"]
        elif hasattr(self, "_device"):
            constructor_kwargs["device"] = self._device

        if "layout" in kwargs:
            constructor_kwargs["layout"] = kwargs["layout"]
        elif hasattr(self, "layout"):
            constructor_kwargs["layout"] = self.layout

        if "num_samples" in kwargs:
            constructor_kwargs["num_samples"] = kwargs["num_samples"]
        elif hasattr(self, "num_samples"):
            constructor_kwargs["num_samples"] = self.num_samples

        if "chat_template_name" in kwargs:
            constructor_kwargs["chat_template_name"] = kwargs["chat_template_name"]
        elif hasattr(self, "chat_template_name"):
            constructor_kwargs["chat_template_name"] = self.chat_template_name

        if "chat_template" in kwargs:
            constructor_kwargs["chat_template"] = kwargs["chat_template"]
        elif hasattr(self, "chat_template"):
            constructor_kwargs["chat_template"] = self.chat_template

        if "text_key" in kwargs:
            constructor_kwargs["text_key"] = kwargs["text_key"]
        elif hasattr(self, "text_key"):
            constructor_kwargs["text_key"] = self.text_key

        if "tokens_key" in kwargs:
            constructor_kwargs["tokens_key"] = kwargs["tokens_key"]
        elif hasattr(self, "tokens_key"):
            constructor_kwargs["tokens_key"] = self.tokens_key

        if "masks_key" in kwargs:
            constructor_kwargs["masks_key"] = kwargs["masks_key"]
        elif hasattr(self, "masks_key"):
            constructor_kwargs["masks_key"] = self.masks_key

        if "log_probs_key" in kwargs:
            constructor_kwargs["log_probs_key"] = kwargs["log_probs_key"]
        elif hasattr(self, "log_probs_key"):
            constructor_kwargs["log_probs_key"] = self.log_probs_key

        # Create and return new instance
        return type(self)(**constructor_kwargs)

    @set_list_to_stack(True)
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if not tensordict.ndim:
            # unsqueeze - squeeze the input
            try:
                return self(lazy_stack([tensordict])).squeeze(0)
            except Exception as e:
                raise RuntimeError(
                    f"Unsqueeze/squeeze failed. Inputs to {type(self).__name__} should ideally be 1 dimensional."
                ) from e
        elif tensordict.ndim > 1:
            return self(tensordict.reshape(-1)).view(tensordict.shape)

        _source_device = None
        if self._device:
            _source_device = tensordict.device
        if tensordict.device:
            tensordict = tensordict.copy().clear_device_()

        if kwargs:
            from transformers import GenerationConfig

            cfg = GenerationConfig(**kwargs)
        else:
            cfg = None

        if self.num_samples is not None:
            out = (
                TensorDict(
                    device=tensordict.device,
                    batch_size=(
                        tensordict.batch_size[0],
                        self.num_samples,
                        *tensordict.batch_size[1:],
                    ),
                )
                .to_lazystack(1)
                .to_lazystack(0)
            )
        else:
            out = TensorDict(
                device=tensordict.device, batch_size=tensordict.batch_size
            ).to_lazystack(0)

        if self.input_mode == "history":
            if self.generate:
                out = self._from_transformers_generate_history(tensordict, cfg, out)
            else:
                out = self._from_transformers_logprobs_history(tensordict, cfg, out)
        elif self.input_mode == "text":
            if self.generate:
                out = self._from_transformers_generate_text(tensordict, cfg, out)
            else:
                out = self._from_transformers_logprobs_text(tensordict, cfg, out)
        elif self.input_mode == "tokens":
            if self.generate:
                out = self._from_transformers_generate_tokens(tensordict, cfg, out)
            else:
                out = self._from_transformers_logprobs_tokens(tensordict, cfg, out)

        if _source_device:
            out = out.to(_source_device)

        if tensordict_out is None:
            if self.inplace is True:
                # The output is the input
                tensordict_out = tensordict
            elif self.inplace is False:
                # The output is the new structure
                tensordict_out = out
            elif self.inplace == "empty":
                # The output is empty
                tensordict_out = tensordict.empty()

        if tensordict_out is not None and tensordict_out is not out:
            result = tensordict_out.exclude(*self.out_keys, inplace=True)
            result.update(out, keys_to_update=self.out_keys)
        elif tensordict_out is out:
            result = out.select(*self.out_keys)
        elif self.inplace:
            result = out
            keys = list(set(self.out_keys + list(tensordict.keys(True, True))))
            result = tensordict.exclude(*self.out_keys, inplace=True).update(
                result, keys_to_update=keys
            )
        else:
            result = out
        return result

    def _from_transformers_generate_history(self, td, cfg, out) -> TensorDictBase:
        """Generate text from history input."""
        from torchrl.data.llm import History

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for history input mode, "
                f"but found keys: {list(td.keys())}"
            )

        history = td.get(self.input_key)
        if not isinstance(history, History):
            raise TypeError(
                f"Expected History object for '{self.input_key}', got {type(history)}"
            )

        # Apply chat template
        tokenizer_kwargs = {}
        if self.chat_template_name is not None:
            tokenizer_kwargs.setdefault("chat_template_name", self.chat_template_name)
        if self.chat_template is not None:
            tokenizer_kwargs.setdefault("chat_template", self.chat_template)
        tokenizer_kwargs.setdefault("add_generation_prompt", True)
        text_prompt = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )
        if not isinstance(text_prompt, list):
            raise ValueError(
                f"Expected list of text for history input, got {type(text_prompt)}"
            )
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", False)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("return_dict", True)
        response_struct = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )
        tokens_prompt_padded = response_struct.get(
            "input_ids",
            as_padded_tensor=True,
            padding_value=self.padding_value,
            padding_side="left",
        )
        attention_mask_prompt_padded = response_struct.get(
            "attention_mask",
            as_padded_tensor=True,
            padding_value=0,
            padding_side="left",
        )

        if attention_mask_prompt_padded is None:
            attention_mask_prompt_padded = (
                tokens_prompt_padded != self.tokenizer.pad_token_id
            )

        result = self._generate_from_tokens(
            tokens_prompt_padded, attention_mask_prompt_padded, cfg, out
        )

        # Generate using text path
        if self.pad_output:
            result[(self.tokens_key, "prompt")] = (
                tokens_prompt_padded
                if not self.num_samples
                else tokens_prompt_padded.unsqueeze(1).repeat(1, self.num_samples, 1)
            )
        else:
            tokens_prompt_unpadded = response_struct.get(
                "input_ids",
                as_nested_tensor=True,
            )
            if not self.num_samples:
                result[(self.tokens_key, "prompt")] = tokens_prompt_unpadded
            else:
                for r in result.unbind(1):
                    r[(self.tokens_key, "prompt")] = tokens_prompt_unpadded

        text_result = Text._from_tensordict(result.empty())
        result.set(self.text_key, text_result)
        if not self.num_samples:
            text_result.prompt = text_prompt
        else:
            for r in result.unbind(1):
                r[self.text_key, "prompt"] = text_prompt
        with result.view(-1) as result_flat:
            if self.pad_output:
                tokens_full_padded = result_flat.get(
                    (self.tokens_key, "full"),
                    as_padded_tensor=True,
                    padding_side="right",
                    padding_value=self.padding_value,
                )
                if tokens_full_padded is None:
                    raise ValueError("tokens_full_padded is None")
                text_full = self.tokenizer.batch_decode(
                    tokens_full_padded, skip_special_tokens=False
                )
            else:
                tokens_full_unpadded = result_flat.get(
                    (self.tokens_key, "full"), as_list=True
                )
                if tokens_full_unpadded is None:
                    raise ValueError("tokens_full_unpadded is None")
                text_full = self.tokenizer.batch_decode(
                    tokens_full_unpadded, skip_special_tokens=False
                )
            text_prompt = result_flat[self.text_key, "prompt"]
            text_response = [
                txt[len(prompt) :]
                for txt, prompt in _zip_strict(text_full, text_prompt)
            ]
            result_flat.set((self.text_key, "full"), text_full)
            result_flat.set((self.text_key, "response"), text_response)
        # Now parse the full text back to a history object, and use the extra history objects
        # as response
        history_chat = ChatHistory._from_tensordict(result.empty())
        if self.num_samples is None:
            history_chat.prompt = history
        else:
            for h in history_chat.unbind(1):
                h.prompt = history
        with history_chat.view(-1) as history_chat_flat:
            prompt_histories = history_chat_flat.prompt
            # Extract response histories from full text
            h_responses = _extract_responses_from_full_histories(
                text_full, prompt_histories, self.chat_template_name, self.tokenizer
            )
            history_chat_flat.response = h_responses
        result.set(self.history_key, history_chat)
        return result

    def _from_transformers_logprobs_history(self, td, cfg, out):
        """Compute log-probs from history input."""
        from torchrl.data.llm import History

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for history input mode, "
                f"but found keys: {list(td.keys())}"
            )

        history = td.get(self.input_key)
        if not isinstance(history, History):
            raise TypeError(
                f"Expected History object for '{self.input_key}', got {type(history)}"
            )

        # Apply chat template
        tokenizer_kwargs = {}
        if self.chat_template_name is not None:
            tokenizer_kwargs.setdefault("chat_template_name", self.chat_template_name)
        if self.chat_template is not None:
            tokenizer_kwargs.setdefault("chat_template", self.chat_template)
        tokenizer_kwargs.setdefault("add_generation_prompt", False)
        text_full = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )

        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("return_dict", True)

        with torch.device(self._device) if self._device is not None else nullcontext():
            response_tokens = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )
        if not isinstance(response_tokens, TensorDictBase):
            raise ValueError(
                f"Expected TensorDictBase for history input, got {type(response_tokens)}"
            )
        result = self._logprobs_from_history_tokens(response_tokens, cfg, out)
        text_result = Text._from_tensordict(result.empty())
        result.set(self.text_key, text_result)
        result[self.text_key, "full"] = text_full
        result.set(self.history_key, ChatHistory(full=history))
        return result

    def _cat_text(self, text, response_text):
        """Concatenate text and response text."""
        if isinstance(text, list):
            return [self._cat_text(t, t_) for t, t_ in _zip_strict(text, response_text)]
        else:
            return text + response_text

    def _generate_from_text(self, text, cfg, out) -> TensorDictBase:
        """Generate text from text input."""
        pad_val = self.tokenizer.pad_token_id

        # Convert text to list format
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = text.tolist()

        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        tokenizer_kwargs.setdefault("padding", True)

        with torch.device(
            self._device
        ) if self._device is not None else contextlib.nullcontext():
            tokens_in = self.tokenizer(text, **tokenizer_kwargs)
        if self._device is not None:
            tokens_in = tokens_in.to(self._device)
        # We are going to map this tokens_in to a tensordict to facilitate the padding in case we need it
        tokens_in = dict(tokens_in)
        for k, v in dict(tokens_in).items():
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    v = torch.nested.nested_tensor(v)
                else:
                    v = torch.nested.nested_tensor([torch.tensor(t) for t in v])
            tokens_in[k] = v
        tokens_in = (
            TensorDict(batch_size=tokens_in["input_ids"].size(0))
            .to_lazystack(0)
            .update(tokens_in)
        )
        tokens_prompt_padded = tokens_in.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        attention_mask_prompt_padded = tokens_in.get(
            "attention_mask",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        tokens_out = self.model.generate(
            input_ids=tokens_prompt_padded,
            attention_mask=attention_mask_prompt_padded,
            **kwargs,
        )
        tokens_full_padded = tokens_out["sequences"]
        tokens_response_padded = tokens_full_padded[
            ..., tokens_prompt_padded.shape[-1] :
        ]

        attention_mask_response_padded = tokens_response_padded != pad_val
        if self.num_samples:
            attention_mask_full_padded = torch.cat(
                [
                    attention_mask_prompt_padded.repeat_interleave(
                        self.num_samples, dim=0
                    ),
                    attention_mask_response_padded,
                ],
                dim=-1,
            )
        else:
            attention_mask_full_padded = torch.cat(
                [attention_mask_prompt_padded, attention_mask_response_padded], dim=-1
            )
        tokens_response_unpadded = _unpad_tensors(
            tokens_response_padded, attention_mask_response_padded, as_nested=False
        )

        if self.return_log_probs:
            # These are only for the new tokens, not for the prompt - to get that, we'd need to run the forward pass again
            logits = torch.stack(list(tokens_out["logits"]), 1)
            log_probs, logits = self._log_probs_generate(
                tokens_response_padded, logits, pad_val=-100, pad=False
            )

        response_text = self.tokenizer.batch_decode(
            tokens_response_unpadded, skip_special_tokens=False
        )

        # Build output TensorClass objects
        if self.num_samples is not None:
            text = [txt for txt in text for _ in range(self.num_samples)]
        text_obj = Text._from_tensordict(out.empty())
        with text_obj.view(-1) as text_obj_flat:
            text_obj_flat.prompt = text
            text_obj_flat.response = response_text
            text_obj_flat.full = self._cat_text(text, response_text)
        out.set(self.text_key, text_obj)

        tokens_obj = Tokens._from_tensordict(out.empty())
        if self.pad_output:
            prompt = tokens_prompt_padded
        else:
            prompt = _unpad_tensors(
                tokens_prompt_padded, attention_mask_prompt_padded, as_nested=False
            )
        if tokens_obj.ndim == 2:
            for i in range(self.num_samples):
                tokens_obj[:, i].prompt = prompt
        else:
            tokens_obj.prompt = prompt
        with tokens_obj.view(-1) as tokens_obj_flat:
            if not self.pad_output:
                tokens_obj_flat.response = tokens_response_unpadded
                tokens_full_unpadded = _unpad_tensors(
                    tokens_full_padded, attention_mask_full_padded, as_nested=False
                )
                tokens_obj_flat.full = tokens_full_unpadded
            else:
                tokens_obj_flat.response = tokens_response_padded
                tokens_obj_flat.full = tokens_full_padded
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        masks_obj = Masks._from_tensordict(out.empty())
        if out.ndim == 2:
            attention_mask_full_padded = attention_mask_full_padded.unflatten(
                0, (-1, self.num_samples)
            )
        if self.pad_output:
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
        else:
            if out.ndim == 2:
                with tokens_obj.view(-1) as tokens_obj_flat, masks_obj.view(
                    -1
                ) as masks_obj_flat:
                    attention_mask_full_unpadded = attention_mask_full_padded.flatten(
                        0, 1
                    )
                    attention_mask_full_unpadded = _unpad_tensors(
                        attention_mask_full_unpadded.bool(),
                        attention_mask_full_padded.flatten(0, 1),
                        as_nested=False,
                    )
                    masks_obj_flat.all_attention_mask = attention_mask_full_unpadded
            else:
                attention_mask_full_unpadded = _unpad_tensors(
                    attention_mask_full_padded.bool(),
                    attention_mask_full_padded,
                    as_nested=False,
                )
                masks_obj.all_attention_mask = attention_mask_full_unpadded
        masks_obj.all_assistant_mask = None
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            with log_probs_obj.view(-1) as log_probs_obj_flat:
                # Unfortunate but we only have the log-probs for the new tokens, not for the prompt - to get that, we'd need to run the forward pass again
                if self.pad_output:
                    log_probs_obj_flat.prompt = None
                    log_probs_obj_flat.response = log_probs
                    log_probs_obj_flat.full = None
                else:
                    log_probs_unpadded = _unpad_tensors(
                        log_probs, attention_mask_response_padded, as_nested=False
                    )
                    log_probs_obj_flat.prompt = None
                    log_probs_obj_flat.response = log_probs_unpadded
                    log_probs_obj_flat.full = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set(self.log_probs_key, log_probs_obj)

        # Add logits to output if we're in a get_dist call
        if self._in_get_dist_call:
            if self.pad_output:
                out.set("logits", logits)
            else:
                logits_full_unpadded = _unpad_tensors(
                    logits, attention_mask_full_padded, as_nested=False
                )
                out.set("logits", logits_full_unpadded)

        return out

    def _cat_tensors(
        self,
        tokens: torch.Tensor | list[torch.Tensor],
        response_tokens: torch.Tensor | list[torch.Tensor],
        cast: torch.dtype | None = None,
    ):
        """Concatenate tokens and response tokens."""
        if isinstance(tokens, list) or isinstance(response_tokens, list):
            return [
                self._cat_tensors(t, t_, cast=cast)
                for t, t_ in _zip_strict(tokens, response_tokens)
            ]
        else:
            result = torch.cat([tokens, response_tokens], dim=-1)
            if cast is not None:
                result = result.to(cast)
            return result

    def _logprobs_from_history_tokens(self, response_tokens, cfg, out):
        """Compute log-probs from history tokens."""
        pad_val = self.tokenizer.pad_token_id

        # unfortunately HF wants us to use padded tensors
        tokens_full_padded = response_tokens.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        if not isinstance(tokens_full_padded, torch.Tensor):
            raise ValueError(
                f"Expected Tensor for tokens_full_padded, got {type(tokens_full_padded)}"
            )
        attention_mask_full_padded = response_tokens.get(
            "attention_mask",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )
        if not isinstance(attention_mask_full_padded, torch.Tensor):
            raise ValueError(
                f"Expected Tensor for attention_mask_full_padded, got {type(attention_mask_full_padded)}"
            )

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        tokens_out_struct = self.model(
            tokens_full_padded, attention_mask=attention_mask_full_padded, **kwargs
        )

        (
            log_probs_full_padded,
            logits_full_padded,
        ) = self._compute_log_probs_from_model_output(
            tokens_out_struct,
            tokens_full_padded,
            attention_mask_full_padded,
            pad_val,
        )

        # Build output TensorClass objects
        text_obj = Text._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        text_obj.prompt = None
        text_obj.response = None
        text_obj.full = None
        out.set(self.text_key, text_obj)

        all_assistant_mask_padded = response_tokens.get(
            "assistant_masks",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )
        if all_assistant_mask_padded is not None:
            all_assistant_mask_padded = all_assistant_mask_padded.bool()
        masks_obj = Masks._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
            if all_assistant_mask_padded is not None:
                masks_obj.all_assistant_mask = all_assistant_mask_padded
        else:
            masks_obj.all_attention_mask = _unpad_tensors(
                attention_mask_full_padded.bool(),
                attention_mask_full_padded,
                as_nested=False,
            )
            if all_assistant_mask_padded is not None:
                masks_obj.all_assistant_mask = _unpad_tensors(
                    all_assistant_mask_padded,
                    attention_mask_full_padded,
                    as_nested=False,
                )
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        tokens_obj = Tokens._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            tokens_obj.full = tokens_full_padded
        else:
            input_ids_full_unpadded = _unpad_tensors(
                tokens_full_padded, attention_mask_full_padded, as_nested=False
            )
            tokens_obj.full = input_ids_full_unpadded
        tokens_obj.response = None
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        log_probs_obj = LogProbs._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            log_probs_obj.full = log_probs_full_padded
        else:
            log_probs_full_unpadded = _unpad_tensors(
                log_probs_full_padded, attention_mask_full_padded, as_nested=False
            )
            log_probs_obj.full = log_probs_full_unpadded
        log_probs_obj.response = None
        log_probs_obj.padded = MetaData(self.pad_output)
        out.set(self.log_probs_key, log_probs_obj)

        # Add logits to output if we're in a get_dist call
        if self._in_get_dist_call:
            if self.pad_output:
                out.set("logits", logits_full_padded)
            else:
                logits_full_unpadded = _unpad_tensors(
                    logits_full_padded, attention_mask_full_padded, as_nested=False
                )
                out.set("logits", logits_full_unpadded)

        return out

    def _from_transformers_generate_text(self, td, cfg, out) -> TensorDictBase:
        """Generate text from text input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for text input mode, "
                f"but found keys: {list(td.keys())}"
            )

        text = td.get(self.input_key)
        if text is None:
            raise ValueError(f"Expected '{self.input_key}' key for text input mode")
        if isinstance(text, NonTensorStack):
            text = text.tolist()
        if not isinstance(text, list):
            raise ValueError(f"Expected list of text for text input, got {type(text)}")
        return self._generate_from_text(text, cfg, out)

    def _from_transformers_logprobs_text(self, td, cfg, out):
        """Compute log-probs from text input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for text input mode, "
                f"but found keys: {list(td.keys())}"
            )

        text = td.get(self.input_key)
        if isinstance(text, NonTensorStack):
            text = text.tolist()
        if text is None:
            raise ValueError(f"Expected '{self.input_key}' key for text input mode")
        if not isinstance(text, list):
            raise ValueError(f"Expected list of text for text input, got {type(text)}")
        # Tokenize the text
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is required for log-probs computation with text input"
            )

        # Convert text to list format
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = text.tolist()

        # Tokenize the text
        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        with torch.device(
            self._device
        ) if self._device is not None else contextlib.nullcontext():
            tokens_in = self.tokenizer(text, **tokenizer_kwargs)

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        # We are going to map this tokens_in to a tensordict to facilitate the padding in case we need it
        tokens_in = (
            TensorDict(batch_size=len(tokens_in["input_ids"]))
            .to_lazystack(0)
            .update(dict(tokens_in))
        )
        input_ids_full_padded = tokens_in.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=self.padding_value,
        )
        attention_mask_full_padded = tokens_in.get(
            "attention_mask",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )

        tokens_out_struct = self.model(
            input_ids_full_padded, attention_mask=attention_mask_full_padded, **kwargs
        )

        # Compute log-probs for the input tokens
        (
            log_probs_full_padded,
            logits_full_padded,
        ) = self._compute_log_probs_from_model_output(
            tokens_out_struct,
            input_ids_full_padded,
            attention_mask_full_padded,
            self.tokenizer.pad_token_id,
        )

        # Build output TensorClass objects
        text_obj = Text._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        text_obj.prompt = None
        text_obj.response = None
        text_obj.full = text
        out.set(self.text_key, text_obj)

        tokens_obj = Tokens._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            tokens_obj.full = input_ids_full_padded
        else:
            input_ids_full_unpadded = _unpad_tensors(
                input_ids_full_padded, attention_mask_full_padded, as_nested=False
            )
            tokens_obj.full = input_ids_full_unpadded
        tokens_obj.response = None
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        masks_obj = Masks._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
            masks_obj.all_assistant_mask = td.get(("masks", "all_assistant_mask"))
        else:
            attention_mask_full_unpadded = _unpad_tensors(
                attention_mask_full_padded.bool(),
                attention_mask_full_padded,
                as_nested=False,
            )
            masks_obj.all_attention_mask = attention_mask_full_unpadded
            masks_obj.all_assistant_mask = td.get(
                ("masks", "all_assistant_mask"), as_list=True
            )
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        log_probs_obj = LogProbs._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            log_probs_obj.full = log_probs_full_padded
        else:
            log_probs_full_unpadded = _unpad_tensors(
                log_probs_full_padded, attention_mask_full_padded, as_nested=False
            )
            log_probs_obj.full = log_probs_full_unpadded
        log_probs_obj.response = None
        log_probs_obj.padded = MetaData(self.pad_output)
        out.set(self.log_probs_key, log_probs_obj)

        # Add logits to output if we're in a get_dist call
        if self._in_get_dist_call:
            if self.pad_output:
                out.set("logits", logits_full_padded)
            else:
                logits_full_unpadded = _unpad_tensors(
                    logits_full_padded, attention_mask_full_padded, as_nested=False
                )
                out.set("logits", logits_full_unpadded)

        return out

    def _from_transformers_generate_tokens(
        self, td: TensorDictBase, cfg: dict | None, out: TensorDictBase
    ) -> TensorDictBase:
        """Generate text from tokens input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys())}"
            )

        pad_val = self.tokenizer.pad_token_id

        input_ids_prompt_padded = td.get(
            self.input_key,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        attention_mask_prompt_padded = td.get(
            ("masks", "all_attention_mask"),
            as_padded_tensor=True,
            padding_side="left",
            padding_value=False,
        )
        if attention_mask_prompt_padded is None:
            attention_mask_prompt_padded = td.get(
                self.attention_mask_key,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=False,
            )
            if attention_mask_prompt_padded is None:
                attention_mask_prompt_padded = input_ids_prompt_padded != pad_val
        return self._generate_from_tokens(
            input_ids_prompt_padded, attention_mask_prompt_padded, cfg, out
        )

    def _generate_from_tokens(
        self,
        tokens_prompt_padded: torch.Tensor,
        attention_mask_prompt_padded: torch.Tensor,
        cfg: dict | None,
        out: TensorDictBase,
    ) -> TensorDictBase:
        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        tokens_out_struct = self.model.generate(
            input_ids=tokens_prompt_padded,
            attention_mask=attention_mask_prompt_padded,
            **kwargs,
        )
        tokens_full_padded = tokens_out_struct["sequences"]
        tokens_response_padded = tokens_full_padded[:, tokens_prompt_padded.shape[-1] :]
        pad_val = getattr(self.tokenizer, "pad_token_id", None)
        if pad_val is None:
            pad_val = self.padding_value
        attention_mask_reponse_padded = tokens_response_padded != pad_val
        attention_mask_full_padded = tokens_full_padded != pad_val
        tokens_response_unpadded = _unpad_tensors(
            tokens_response_padded, attention_mask_reponse_padded, as_nested=False
        )

        if self.return_log_probs:
            # These are only for the new tokens, not for the prompt - to get that, we'd need to run the forward pass again
            logits_response_padded = tokens_out_struct["logits"]
            logits_response_padded = torch.stack(list(logits_response_padded), 1)
            (
                log_probs_response_padded,
                logits_response_padded,
            ) = self._log_probs_generate(
                tokens_response_padded,
                logits_response_padded,
                pad_val=pad_val,
                pad=False,
            )

        response_text = self.tokenizer.batch_decode(
            tokens_response_unpadded, skip_special_tokens=False
        )

        # Build output TensorClass objects
        text_obj = Text._from_tensordict(out.empty())
        text_obj.prompt = None  # We don't have text in tokens mode
        with text_obj.view(-1) as text_obj_flat:
            text_obj_flat.response = response_text
        text_obj.full = None  # we don't have text in tokens mode so no all_text either
        out.set(self.text_key, text_obj)

        tokens_obj = Tokens._from_tensordict(out.empty())
        if not self.pad_output:
            input_ids_prompt_unpadded = _unpad_tensors(
                tokens_prompt_padded,
                attention_mask_prompt_padded,
                as_nested=False,
            )
        if self.num_samples is not None:
            # replicate tokens
            for i in range(self.num_samples):
                tokens_obj[:, i].prompt = (
                    input_ids_prompt_unpadded
                    if not self.pad_output
                    else tokens_prompt_padded
                )
        else:
            tokens_obj.prompt = (
                input_ids_prompt_unpadded
                if not self.pad_output
                else tokens_prompt_padded
            )
        with tokens_obj.view(-1) as tokens_obj_flat:
            if self.pad_output:
                tokens_obj_flat.response = tokens_response_padded
                tokens_obj_flat.full = tokens_full_padded
            else:
                tokens_obj_flat.response = tokens_response_unpadded
                tokens_full_unpadded = _unpad_tensors(
                    tokens_full_padded, attention_mask_full_padded, as_nested=False
                )
                tokens_obj_flat.full = tokens_full_unpadded
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        masks_obj = Masks._from_tensordict(out.empty())
        if out.ndim == 2:
            attention_mask_full_padded = attention_mask_full_padded.unflatten(
                0, (-1, self.num_samples)
            )
        if self.pad_output:
            # Get "real" attention masks
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
        else:
            # Get "real" attention masks
            # We can use select to avoid batch-size problems
            _td = torch.ones_like(
                out.select(("tokens", "full"))
                .copy()
                .rename_key_(("tokens", "full"), "all_attention_mask")
            ).bool()
            del _td["tokens"]
            masks_obj.update(_td)
        masks_obj.all_assistant_mask = None
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            if self.num_samples is None:
                if self.pad_output:
                    log_probs_obj.response = log_probs_response_padded
                else:
                    log_probs_response_unpadded = _unpad_tensors(
                        log_probs_response_padded,
                        attention_mask_reponse_padded,
                        as_nested=False,
                    )
                    log_probs_obj.response = log_probs_response_unpadded
            else:
                with log_probs_obj.view(-1) as log_probs_obj_flat:
                    if self.pad_output:
                        log_probs_obj_flat.response = log_probs_response_padded
                    else:
                        log_probs_response_unpadded = _unpad_tensors(
                            log_probs_response_padded,
                            attention_mask_reponse_padded,
                            as_nested=False,
                        )
                        log_probs_obj_flat.response = log_probs_response_unpadded
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set(self.log_probs_key, log_probs_obj)

        return out

    def _from_transformers_logprobs_tokens(
        self, td: TensorDictBase, cfg: dict | None, out: TensorDictBase
    ) -> TensorDictBase:
        """Compute log-probs from tokens input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys(isinstance(self.input_key, tuple)))}"
            )

        pad_val = self.tokenizer.pad_token_id

        input_ids_full_padded = td.get(
            self.input_key,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        # Attention mask: try first the regular entry, then the key provided in the constructor, finally fallback on eager attention mask
        attention_mask_full_padded = td.get(
            ("masks", "all_attention_mask"),
            as_padded_tensor=True,
            padding_side="left",
            padding_value=False,
        )
        if attention_mask_full_padded is None:
            attention_mask_full_padded = td.get(
                self.attention_mask_key,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=False,
            )
            if attention_mask_full_padded is None:
                attention_mask_full_padded = input_ids_full_padded != pad_val

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        tokens_out_struct = self.model(
            input_ids_full_padded, attention_mask=attention_mask_full_padded, **kwargs
        )

        # Compute log-probs for the input tokens
        (
            log_probs_full_padded,
            logits_full_padded,
        ) = self._compute_log_probs_from_model_output(
            tokens_out_struct,
            input_ids_full_padded,
            attention_mask_full_padded,
            self.tokenizer.pad_token_id,
        )

        # Build output TensorClass objects
        text_obj = Text._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        text_obj.prompt = None
        text_obj.response = None
        text_obj.full = None
        out.set(self.text_key, text_obj)

        tokens_obj = Tokens._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if not self.pad_output:
            input_ids_full_unpadded = _unpad_tensors(
                input_ids_full_padded, attention_mask_full_padded, as_nested=False
            )
            tokens_obj.full = input_ids_full_unpadded
        else:
            tokens_obj.full = input_ids_full_padded
        tokens_obj.response = None
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        masks_obj = Masks._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
            masks_obj.all_assistant_mask = td.get(("masks", "all_assistant_mask"))
        else:
            masks_obj.all_attention_mask = _unpad_tensors(
                attention_mask_full_padded.bool(),
                attention_mask_full_padded,
                as_nested=False,
            )
            masks_obj.all_assistant_mask = td.get(
                ("masks", "all_assistant_mask"), as_list=True
            )

        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        log_probs_obj = LogProbs._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            log_probs_obj.full = log_probs_full_padded
        else:
            log_probs_full_unpadded = _unpad_tensors(
                log_probs_full_padded, attention_mask_full_padded, as_nested=False
            )
            log_probs_obj.full = log_probs_full_unpadded
        log_probs_obj.response = None
        log_probs_obj.padded = MetaData(self.pad_output)
        out.set(self.log_probs_key, log_probs_obj)

        # Add logits to output if we're in a get_dist call
        if self._in_get_dist_call:
            if self.pad_output:
                out.set("logits", logits_full_padded)
            else:
                logits_full_unpadded = _unpad_tensors(
                    logits_full_padded, attention_mask_full_padded, as_nested=False
                )
                out.set("logits", logits_full_unpadded)
        return out

    @classmethod
    def _log_probs_generate(cls, tokens, logits, pad_val=-100, pad: bool = True):
        if pad:
            tokens = pad_sequence(
                tokens,
                padding_value=pad_val,
                batch_first=True,
                padding_side="left",
            )
            logits = pad_sequence(
                logits,
                padding_value=0.0,
                batch_first=True,
                padding_side="left",
            )

        # logits = logits.log_softmax(dim=-1)
        # log_probs = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        td = TensorDict(logits=logits, tokens=tokens).auto_batch_size_()
        with td.flatten() as tdflat:
            tdflat["log_probs"] = -torch.nn.functional.cross_entropy(
                tdflat["logits"], tdflat["tokens"], reduce=False, ignore_index=pad_val
            )
        td["log_probs"][:, 0] = 0
        log_probs = td["log_probs"]
        return log_probs, logits

    def _compute_log_probs_from_model_output(
        self, model_output, input_ids, attention_mask, pad_val
    ):
        """Compute log-probs from model output without modifying original tensors.

        Args:
            model_output: Output from the model containing logits
            input_ids: Original input token ids
            attention_mask: Original attention mask
            pad_val: Padding token value to ignore in loss computation

        Returns:
            tuple: (log_probs, shifted_logits) where log_probs are the computed log probabilities
                   and shifted_logits are the logits shifted to align with tokens
        """
        logits = model_output["logits"]

        # Create shifted versions for log-prob computation without modifying originals
        shifted_logits = logits[:, :-1, :]
        # shifted_logits = shifted_logits - shifted_logits.logsumexp(dim=-1, keepdim=True)
        shifted_logits = torch.cat(
            [torch.zeros_like(shifted_logits[:, :1]), shifted_logits], 1
        )

        shifted_input_ids = input_ids[:, 1:]
        shifted_input_ids = torch.cat(
            [torch.zeros_like(shifted_input_ids[:, :1]), shifted_input_ids], 1
        )

        # Check that the shape is correct
        if shifted_logits.shape[-2] != shifted_input_ids.shape[-1]:
            raise ValueError(
                f"The logits shape {shifted_logits.shape} does not match the input ids shape {shifted_input_ids.shape}"
            )

        # Compute log-probs
        td = TensorDict(
            logits=shifted_logits, tokens=shifted_input_ids
        ).auto_batch_size_()
        with td.flatten() as tdflat:
            tdflat["log_probs"] = -torch.nn.functional.cross_entropy(
                tdflat["logits"],
                tdflat["tokens"],
                reduce=False,
                ignore_index=pad_val,
            )
        # For consistency with vllm, we set the log-probs of the first token to 0
        #  However, the first element may not be the first - we want the first of the attention mask,
        #  i.e, the first element that is true on the left
        attention_mask = attention_mask.bool()
        attention_mask_first_left = ~attention_mask[:, :-1] & attention_mask[:, 1:]
        attention_mask_first_left = torch.cat(
            [
                torch.zeros_like(attention_mask_first_left[..., :1]),
                attention_mask_first_left,
            ],
            -1,
        )
        attention_mask_first_left[~(attention_mask_first_left.any(-1)), 0] = True
        assert attention_mask_first_left.any(-1).all()
        attention_mask_first_left = attention_mask_first_left | ~attention_mask
        td["log_probs"][attention_mask_first_left] = 0

        return td["log_probs"], shifted_logits

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        logits_key: NestedKey = "logits",
        mask_key: NestedKey | None = None,
        as_padded_tensor: bool | None = None,
        as_nested_tensor: bool | None = None,
        padding_value: float | None = None,
        padding_side: str = "right",
        layout: torch.layout | None = None,
        **kwargs,
    ) -> D.Distribution:
        """Get distribution from logits/log-probs with optional masking.

        This method enables logits computation for distribution creation.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super().get_dist(
                tensordict,
                tensordict_out,
                logits_key,
                mask_key,
                as_padded_tensor,
                as_nested_tensor,
                padding_value,
                padding_side,
                layout,
                **kwargs,
            )
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_dist_with_prompt_mask(
        self,
        tensordict: TensorDictBase,
        tokens_key: NestedKey = ("tokens", "prompt"),
        logits_key: NestedKey = "logits",
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include response tokens (exclude prompt).

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_dist_with_prompt_mask(
                tensordict,
                tokens_key,
                logits_key,
                assistant_mask_key,
                attention_mask_key,
                **kwargs,
            )
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_dist_with_assistant_mask(
        self,
        tensordict: TensorDictBase,
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include assistant tokens.

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_dist_with_assistant_mask(
                tensordict, assistant_mask_key, logits_key, **kwargs
            )
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_dist_with_attention_mask(
        self,
        tensordict: TensorDictBase,
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked using attention mask.

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_dist_with_attention_mask(
                tensordict, attention_mask_key, logits_key, **kwargs
            )
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_dist_with_custom_mask(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor,
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution with custom mask.

        This method enables logits computation for distribution creation.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_dist_with_custom_mask(
                tensordict, mask, logits_key, **kwargs
            )
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    # Convenience methods for common LLM training scenarios
    def _get_sft_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for SFT loss (response tokens only).

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_sft_dist(tensordict, **kwargs)
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_rlhf_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for RLHF loss (assistant tokens only).

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_rlhf_dist(tensordict, **kwargs)
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")

    def _get_generic_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for generic losses (all tokens).

        This method enables logits computation for distribution creation.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        self._in_get_dist_call = True
        self.out_keys += ["logits"]
        try:
            return super()._get_generic_dist(tensordict, **kwargs)
        finally:
            self._in_get_dist_call = False
            self.out_keys.remove("logits")
