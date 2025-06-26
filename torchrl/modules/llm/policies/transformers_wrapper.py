# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from typing import Literal

import torch

from tensordict import (
    lazy_stack,
    MetaData,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import _zip_strict
from torch.nn.utils.rnn import pad_sequence

from torchrl.modules.llm.policies.common import (
    CategoricalSequential,
    LogProbs,
    Masks,
    Text,
    Tokens,
)
from torchrl.modules.utils.utils import _unpad_tensors

# TODOs:
# - [ ] Remove the useless view(-1) calls when num_samples is not > 1
# - [ ] Remove as_list=True and use a context manager to handle that
# - [ ] Make sure tensordict can handle nested lazy tds that have a get(key, as_list=True) - I think it breaks atm


class TransformersWrapper(CategoricalSequential):
    """A wrapper class for Hugging Face Transformers models, providing a consistent interface for text generation and log probability computation.

    This class provides a unified API for handling different input modalities (history, text, tokens) with consistent
    output structure using :class:`~tensordict.TensorClass` objects.

    Args:
        model (transformers.AutoModelForCausalLM | str): The Hugging Face Transformers model to wrap.
            If a string, it will be passed to `transformers.AutoModelForCausalLM.from_pretrained`.

    Keyword Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | str | None, optional): The tokenizer to use for
            encoding and decoding text. If `None`, the tokenizer associated with the model will be used.
            If a string, it will be passed to `transformers.AutoTokenizer.from_pretrained`. Defaults to `None`.
        input_mode (str, optional): The input modality to use. Must be one of `"history"`, `"text"`, or `"tokens"`.
            Defaults to `"history"`.
        input_key (str | None, optional): The key for the input data. If `None`, defaults to the `input_mode` name.
            Defaults to `None`.
        attention_mask_key (str, optional): The key for attention masks (used in `"tokens"` mode). Defaults to `"attention_mask"`.
        generate (bool, optional): Whether to enable text generation. If `True`, the model will generate text based on
            the input. If `False`, only log probabilities will be computed. Defaults to `True`.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `False`.
        return_text (bool, optional): Whether to return text outputs. Defaults to `True`.
        return_tokens (bool, optional): Whether to return token outputs. Defaults to `True`.
        return_masks (bool, optional): Whether to return mask outputs. Defaults to `True`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to the model's generate method. Defaults to `None`.
        tokenizer_kwargs (dict | None, optional): Additional arguments to pass to the tokenizer. Defaults to `None`.
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. Transformers require
            `pad_output=True`, and the output sequences will be padded and represented as tensors. Defaults to `True`.
        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place
            operations. Defaults to `True` when generating a single sample, `False` otherwise.
        device (torch.device | None, optional): The device to use for computation. Defaults to `None`.
        layout (torch.layout | None, optional): The layout to use for the output tensors when `pad_output=False`. Defaults to `torch.strided`.
        num_samples (int | None, optional): The number of samples to generate. Defaults to `None` (one sample, and no batch-dimension for it).
            Can also be set via the `generate_kwargs["num_return_sequences"] = value` argument.
        chat_template_name (Literal["chatml_format", "qwen"] | None, optional): The name of the chat template to use when
            applying the chat template to the history. Defaults to `None`.
            For `input_mode="history"` only.
        chat_template (str | None, optional): The chat template to use when applying the chat template to the history.
            Defaults to `None`.
            For `input_mode="history"` only.

    Input Keys:
        - If `input_mode="history"`: `input_key` (defaults to `"history"`)
        - If `input_mode="text"`: `input_key` (defaults to `"text"`)
        - If `input_mode="tokens"`: `input_key` (defaults to `"tokens"`)

    Output Keys:
        Always returns a TensorDict with the following structure:
        ```
        TensorDict(
            text=Text(...),      # if return_text=True
            masks=Masks(...),    # if return_masks=True
            tokens=Tokens(...),  # if return_tokens=True
            log_probs=LogProbs(...)  # if return_log_probs=True
        )
        ```

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from torchrl.data.llm import History
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
        >>> result = wrapper(TensorDict(history=history, batch_size=(1,)))
        >>> print(result["text"].response)
        >>> print(result["log_probs"].response)

    .. seealso:: :class:`~torchrl.modules.llm.vLLMWrapper` for a similar interface using vLLM.
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
        return_log_probs: bool = False,
        return_text: bool = True,
        return_tokens: bool = True,
        return_masks: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = True,
        inplace: Literal[True, False, "empty"] | None = None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        chat_template_name: Literal["chatml_format", "qwen"] | None = None,
        chat_template: str | None = None,
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
        self.return_log_probs = return_log_probs
        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_masks = return_masks
        if not isinstance(pad_output, bool):
            raise ValueError("pad_output must be a boolean")
        if return_masks and not return_tokens:
            raise ValueError("return_masks cannot be True if return_tokens is False")
        self.pad_output = pad_output
        self._device = device
        if not pad_output and layout is None:
            layout = torch.strided
        self.layout = layout
        padding_value = None

        # Auto-determine input_key if not provided
        if input_key is None:
            input_key = input_mode
        self.input_key = input_key

        # Set input keys based on mode
        if input_mode == "history":
            self.in_keys = [input_key]
        elif input_mode == "text":
            self.in_keys = [input_key]
        elif input_mode == "tokens":
            self.in_keys = [input_key]

        # Set output keys based on return flags
        self.out_keys = []
        if return_text:
            self.out_keys.append("text")
        if return_masks:
            self.out_keys.append("masks")
        if return_tokens:
            self.out_keys.append("tokens")
        if return_log_probs:
            self.out_keys.append("log_probs")

        # Tokenizer setup
        if not tokenizer_kwargs:
            tokenizer_kwargs = {}
        if not tokenizer_kwargs.setdefault("return_attention_mask", True):
            raise RuntimeError("return_attention_mask must be True")

        # If we don't pad, we use lists
        return_tensors = "pt" if self.pad_output else False
        if return_tensors:
            if (
                tokenizer_kwargs.setdefault("return_tensors", return_tensors)
                != return_tensors
            ):
                raise RuntimeError
        if tokenizer_kwargs.setdefault("padding", self.pad_output) not in (
            self.pad_output,
        ):
            raise RuntimeError
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
            if not return_log_probs:
                raise ValueError("return_log_probs must be True when generate=False.")

        generate_kwargs.setdefault("tokenizer", self.tokenizer)
        generate_kwargs.setdefault("output_logits", self.return_log_probs)
        generate_kwargs.setdefault("return_dict_in_generate", True)

        self.generate_kwargs = generate_kwargs

        # Additional transformers-specific settings
        self.chat_template_name = chat_template_name
        self.chat_template = chat_template

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
            result = tensordict_out
            result.update(out, keys_to_update=self.out_keys)
        elif tensordict_out is out:
            return out.select(*self.out_keys)
        elif self.inplace:
            result = out
            keys = list(set(self.out_keys + list(tensordict.keys(True, True))))
            return tensordict.update(result, keys_to_update=keys)
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
        text = history.apply_chat_template(tokenizer=self.tokenizer, **tokenizer_kwargs)

        # Generate using text path
        return self._generate_from_text(text, cfg, out)

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
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("return_dict", True)

        with torch.device(self._device) if self._device is not None else nullcontext():
            response_tokens = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )
        return self._logprobs_from_history_tokens(response_tokens, cfg, out)

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

        tokens_in = self.tokenizer(text, **self.tokenizer_kwargs)
        if self._device is not None:
            tokens_in = tokens_in.to(self._device)
        # We are going to map this tokens_in to a tensordict to facilitate the padding in case we need it
        tokens_in = (
            TensorDict(batch_size=len(tokens_in["input_ids"]))
            .to_lazystack(0)
            .update(dict(tokens_in))
        )
        input_ids = tokens_in.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        attention_mask = tokens_in.get(
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
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        full_sequences = tokens_out["sequences"]
        sequences = full_sequences[..., input_ids.shape[-1] :]

        mask_sequences = sequences != pad_val
        sequences_unpadded = _unpad_tensors(sequences, mask_sequences, as_nested=False)

        if self.return_log_probs:
            # These are only for the new tokens, not for the prompt - to get that, we'd need to run the forward pass again
            logits = torch.stack(list(tokens_out["logits"]), 1)
            log_probs, logits = self._log_probs_generate(
                sequences, logits, pad_val=-100, pad=False
            )

        response_text = self.tokenizer.batch_decode(
            sequences_unpadded, skip_special_tokens=False
        )

        # Build output TensorClass objects
        if self.return_text:
            if self.num_samples is not None:
                text = [txt for txt in text for _ in range(self.num_samples)]
            text_obj = Text._from_tensordict(out.empty())
            with text_obj.view(-1) as text_obj_flat:
                text_obj_flat.prompt = text
                text_obj_flat.response = response_text
                text_obj_flat.full = self._cat_text(text, response_text)
            text_obj.padded = MetaData(self.pad_output)
            out.set("text", text_obj)

        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(out.empty())
            if self.pad_output:
                prompt = input_ids
            else:
                prompt = _unpad_tensors(input_ids, attention_mask, as_nested=False)
            if tokens_obj.ndim == 2:
                for i in range(self.num_samples):
                    tokens_obj[:, i].prompt = prompt
            else:
                tokens_obj.prompt = prompt
            with tokens_obj.view(-1) as tokens_obj_flat:
                if not self.pad_output:
                    response = tokens_obj_flat.response = sequences_unpadded
                else:
                    response = tokens_obj_flat.response = sequences
                tokens_obj_flat.full = self._cat_tensors(
                    tokens_obj_flat.prompt, response
                )
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_masks:
            masks_obj = Masks._from_tensordict(out.empty())
            if out.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).expand(
                    attention_mask.shape[0], self.num_samples, *attention_mask.shape[1:]
                )
            if self.pad_output:
                response = tokens_obj.get(
                    "response",
                    as_padded_tensor=True,
                    padding_side="right",
                    padding_value=0,
                )
                masks_obj.all_attention_mask = self._cat_tensors(
                    attention_mask, response, cast=torch.bool
                )  # _unpad_tensors(attention_mask, attention_mask, as_nested=False)
            else:
                if out.ndim == 2:
                    with tokens_obj.view(-1) as tokens_obj_flat, masks_obj.view(
                        -1
                    ) as masks_obj_flat:
                        masks_obj_flat.all_attention_mask = self._cat_tensors(
                            attention_mask.flatten(0, 1),
                            tokens_obj_flat.get("response", as_list=True),
                            cast=torch.bool,
                        )
                else:
                    masks_obj.all_attention_mask = self._cat_tensors(
                        attention_mask,
                        tokens_obj.get("response", as_list=True),
                        cast=torch.bool,
                    )  # _unpad_tensors(attention_mask, attention_mask, as_nested=False)
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

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
                        log_probs, mask_sequences, as_nested=False
                    )
                    log_probs_obj_flat.prompt = None
                    log_probs_obj_flat.response = log_probs_unpadded
                    log_probs_obj_flat.full = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

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
        total_input_ids = response_tokens.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        total_attention_mask = response_tokens.get(
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

        total_tokens_out = self.model(
            total_input_ids, attention_mask=total_attention_mask, **kwargs
        )

        log_probs, logits = self._compute_log_probs_from_model_output(
            total_tokens_out, total_input_ids, total_attention_mask, pad_val
        )

        # Build output TensorClass objects
        if self.return_text:
            text_obj = Text._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            text_obj.prompt = None
            text_obj.response = None
            text_obj.full = None
            text_obj.padded = MetaData(self.pad_output)
            out.set("text", text_obj)

        all_assistant_mask = response_tokens.get(
            "assistant_masks",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )
        if all_assistant_mask is not None:
            all_assistant_mask = all_assistant_mask.bool()
        if self.return_masks:
            masks_obj = Masks._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                masks_obj.all_attention_mask = total_attention_mask
                if all_assistant_mask is not None:
                    masks_obj.all_assistant_mask = all_assistant_mask
            else:
                masks_obj.all_attention_mask = _unpad_tensors(
                    total_attention_mask, total_attention_mask, as_nested=False
                )
                if all_assistant_mask is not None:
                    masks_obj.all_assistant_mask = _unpad_tensors(
                        all_assistant_mask, total_attention_mask, as_nested=False
                    )
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                tokens_obj.full = tokens_obj.prompt = total_input_ids
                if all_assistant_mask is not None:
                    tokens_obj.assistant = pad_sequence(
                        [
                            t[mask]
                            for t, mask in _zip_strict(
                                total_input_ids, all_assistant_mask
                            )
                        ],
                        padding_value=self.padding_value,
                        batch_first=True,
                        padding_side="left",
                    )
            else:
                tokens_obj.full = tokens_obj.prompt = _unpad_tensors(
                    total_input_ids, total_attention_mask, as_nested=False
                )
                if all_assistant_mask is not None:
                    tokens_obj.assistant = _unpad_tensors(
                        total_input_ids, all_assistant_mask, as_nested=False
                    )
            tokens_obj.response = None
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                log_probs_obj.full = log_probs_obj.prompt = log_probs
                # We can set the log-probs of the assistant
                if all_assistant_mask is not None:
                    log_probs_obj.assistant = pad_sequence(
                        [
                            lp[mask]
                            for lp, mask in _zip_strict(log_probs, all_assistant_mask)
                        ],
                        padding_value=0,
                        batch_first=True,
                        padding_side="left",
                    )
            else:
                log_probs_obj.full = log_probs_obj.prompt = _unpad_tensors(
                    log_probs, total_attention_mask, as_nested=False
                )
                if all_assistant_mask is not None:
                    log_probs_obj.assistant = _unpad_tensors(
                        log_probs, all_assistant_mask, as_nested=False
                    )
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

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
        if text is None:
            raise ValueError(f"Expected '{self.input_key}' key for text input mode")

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
        print("text", text)
        tokens_in = self.tokenizer(text, **self.tokenizer_kwargs)

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
        input_ids = tokens_in.get(
            "input_ids",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=self.padding_value,
        )
        attention_mask = tokens_in.get(
            "attention_mask",
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0,
        )

        total_tokens_out = self.model(
            input_ids, attention_mask=attention_mask, **kwargs
        )

        # Compute log-probs for the input tokens
        log_probs, logits = self._compute_log_probs_from_model_output(
            total_tokens_out, input_ids, attention_mask, self.tokenizer.pad_token_id
        )

        # Build output TensorClass objects
        if self.return_text:
            text_obj = Text._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            text_obj.prompt = text
            text_obj.response = None
            text_obj.full = text
            text_obj.padded = MetaData(self.pad_output)
            out.set("text", text_obj)

        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                tokens_obj.full = tokens_obj.prompt = input_ids
            else:
                tokens_obj.full = tokens_obj.prompt = _unpad_tensors(
                    input_ids, attention_mask, as_nested=False
                )
            tokens_obj.response = None
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_masks:
            masks_obj = Masks._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                masks_obj.all_attention_mask = attention_mask
            else:
                masks_obj.all_attention_mask = _unpad_tensors(
                    attention_mask, attention_mask, as_nested=False
                )
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            print("log_probs", log_probs.shape)
            if self.pad_output:
                log_probs_obj.full = log_probs_obj.prompt = log_probs
            else:
                log_probs_obj.full = log_probs_obj.prompt = _unpad_tensors(
                    log_probs, attention_mask, as_nested=False
                )
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

        return out

    def _from_transformers_generate_tokens(self, td, cfg, out) -> TensorDictBase:
        """Generate text from tokens input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys())}"
            )

        pad_val = self.tokenizer.pad_token_id

        input_ids = td.get(
            self.input_key,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        attention_mask = (input_ids != pad_val).to(torch.int64)

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        tokens_out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        full_sequences = tokens_out["sequences"]
        sequences = full_sequences[:, input_ids.shape[-1] :]
        mask_sequences = sequences != pad_val
        sequences_unpadded = _unpad_tensors(sequences, mask_sequences, as_nested=False)

        if self.return_log_probs:
            # These are only for the new tokens, not for the prompt - to get that, we'd need to run the forward pass again
            logits = tokens_out["logits"]
            logits = torch.stack(list(logits), 1)
            # logits = _unpad_tensors(logits, mask_sequences, as_nested=False)
            print("logits", logits.shape)
            log_probs, logits = self._log_probs_generate(
                sequences, logits, pad_val=pad_val, pad=False
            )
            print("log_probs", log_probs.shape, "logits", logits.shape)

        response_text = self.tokenizer.batch_decode(
            sequences_unpadded, skip_special_tokens=False
        )

        # Build output TensorClass objects
        if self.return_text:
            text_obj = Text._from_tensordict(out.empty())
            text_obj.prompt = None  # We don't have text in tokens mode
            with text_obj.view(-1) as text_obj_flat:
                text_obj_flat.response = response_text
            text_obj.full = (
                None  # we don't have text in tokens mode so no all_text either
            )
            text_obj.padded = MetaData(self.pad_output)
            out.set("text", text_obj)

        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(out.empty())
            if not self.pad_output:
                input_ids = td.get(self.input_key, as_list=True)
            if self.num_samples is not None:
                # replicate tokens
                for i in range(self.num_samples):
                    tokens_obj[:, i].prompt = input_ids
            else:
                tokens_obj.prompt = input_ids
            with tokens_obj.view(-1) as tokens_obj_flat:
                if self.pad_output:
                    tokens_obj_flat.response = sequences
                    tokens_obj_flat.full = self._cat_tensors(
                        tokens_obj_flat.prompt, tokens_obj_flat.response
                    )
                else:
                    tokens_obj_flat.response = sequences_unpadded
                    tokens_obj_flat.full = self._cat_tensors(
                        tokens_obj_flat.prompt,
                        tokens_obj_flat.get("response", as_list=True),
                    )
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_masks:
            masks_obj = Masks._from_tensordict(out.empty())
            # self.return_tokens must be True
            if self.pad_output:
                # Get "real" attention masks
                response_attention_masks = tokens_obj.get("full") != self.padding_value
                masks_obj.all_attention_mask = response_attention_masks
            else:
                # Get "real" attention masks
                # We can use select to avoid batch-size problems
                _td = torch.ones_like(
                    out.select(("tokens", "full"))
                    .copy()
                    .rename_key_(("tokens", "full"), "all_attention_mask")
                )
                del _td["tokens"]
                masks_obj.update(_td)
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            with log_probs_obj.view(-1) as log_probs_obj_flat:
                if self.pad_output:
                    log_probs_obj_flat.response = log_probs
                else:
                    log_probs_obj_flat.response = _unpad_tensors(
                        log_probs, mask_sequences, as_nested=False
                    )
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

        return out

    def _from_transformers_logprobs_tokens(self, td, cfg, out):
        """Compute log-probs from tokens input."""
        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys())}"
            )

        pad_val = self.tokenizer.pad_token_id

        input_ids = td.get(
            self.input_key,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=pad_val,
        )
        attention_mask = (input_ids != pad_val).to(torch.int64)

        if cfg is not None:
            kwargs = copy(self.generate_kwargs)
            kwargs["generation_config"] = cfg
        else:
            kwargs = self.generate_kwargs

        total_tokens_out = self.model(
            input_ids, attention_mask=attention_mask, **kwargs
        )

        # Compute log-probs for the input tokens
        log_probs, logits = self._compute_log_probs_from_model_output(
            total_tokens_out, input_ids, attention_mask, self.tokenizer.pad_token_id
        )

        # Build output TensorClass objects
        if self.return_text:
            text_obj = Text._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            text_obj.prompt = None
            text_obj.response = None
            text_obj.full = None
            text_obj.padded = MetaData(self.pad_output)
            out.set("text", text_obj)

        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if not self.pad_output:
                tokens = td.get(self.input_key, as_list=True)
            else:
                tokens = input_ids
            tokens_obj.prompt = tokens
            tokens_obj.response = None
            tokens_obj.full = tokens
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_masks:
            masks_obj = Masks._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            masks_obj.all_attention_mask = None
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                log_probs_obj.full = log_probs_obj.prompt = log_probs
            else:
                log_probs_obj.full = log_probs_obj.prompt = _unpad_tensors(
                    log_probs, attention_mask, as_nested=False
                )
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

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
        td["log_probs"][:, 0] = 0

        return td["log_probs"], shifted_logits
