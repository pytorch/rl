# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Literal

import torch

from tensordict import LazyStackedTensorDict, NestedKey, TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorStack
from tensordict.utils import _zip_strict

from torchrl.modules.llm.common import CategoricalSequential


class TransformersWrapper(CategoricalSequential):
    """A wrapper class for Hugging Face Transformers models, providing a consistent interface for text generation and log probability computation.

    This class handles both text and token inputs, enabling text generation and log probability computation based on
    the specified configuration. Unlike vLLM, Transformers require padded tensors for input and output sequences.

    Args:
        model (transformers.LLM): The Hugging Face Transformers model to wrap.

    Keyword Args:
        return_log_probs (bool | None, optional): Whether to return log probabilities of the generated tokens.
            Defaults to `None`.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | None, optional): The tokenizer to use for
            encoding and decoding text. If `None`, the tokenizer associated with the model will be used. Defaults to
            `None`.
        from_text (bool, optional): Indicates whether the input is in text format. If `True`, the input is expected to
            be text that will be tokenized. If `False`, the input is expected to be token sequences. Defaults to `False`.
        device (torch.device | None, optional): The device to use for computation. If `None`, the default device will
            be used. Defaults to `None`.
        generate (bool, optional): Whether to enable text generation. If `True`, the model will generate text based on
            the input. If `False`, only log probabilities will be computed for the response tokens/text. Defaults to `True`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to the model's generate method. These
            arguments can control aspects of the generation process, such as temperature and top-k sampling. Defaults
            to `None`.
        tokenizer_kwargs (dict | None, optional): Additional arguments to pass to the tokenizer. These arguments can
            control aspects of the tokenization process, such as padding and truncation. Defaults to `None`.
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. Transformers require
            `pad_output=True`, and the output sequences will be padded and represented as tensors. Defaults to `True`.
        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place
            operations. If `True`, operations will be performed in-place. If `False`, a new TensorDict instance will be
            created. If `"empty"`, the output data structure will be initialized with `input.empty()` (i.e., it will
            conserve type, batch-size, and device). Defaults to `True`.

    .. note:: The tokenizer is used when `from_text` is `True` to convert input text into token sequences. It is also
        required (or retrieved) when `pad_output` is `True` or when using text inputs with `generate=False` to ensure proper
        tokenization and padding.

    Input Keys:

    - If `from_text` is `True`:

        - `"text"`: The input text to be tokenized.
        - `"text_response"`: the response text (if `generate=False` as the log probabilities are computed for the response.)

    - If `from_text` is `False`:

        - "tokens": The input token sequences.
        - "attention_mask": The attention mask for the tokens.
        - "tokens_response": The response token sequences (if `generate=False` as the log probabilities are
          computed for the response.)

    Output Keys:

        - `"tokens_response"`: The generated token sequences.
        - `"log_probs"`: The log probabilities of the generated tokens (if `return_log_probs` is `True`).
        - `"text_response"`: The generated text (if `from_text` is `True` and `generate` is `True`).

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> wrapper = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     from_text=True,
        ...     generate=True
        ... )
        >>> input_data = TensorDict({"text": ["Hello, world!", "This is another text"]}, batch_size=1)
        >>> output_data = wrapper(input_data)
        >>> print(output_data["text_response"])

    .. seealso:: :func:`~torchrl.modules.from_hf_transformers` for a similar interface using the Hugging Face
        Transformers library.
    """

    text_key: NestedKey = ("text",)
    token_key: NestedKey = ("tokens",)
    token_response_key: NestedKey = ("tokens_response",)
    text_response_key: NestedKey = ("text_response",)
    attention_mask_key: NestedKey = ("attention_mask",)

    def __init__(
        self,
        model: transformers.LLM,  # noqa
        # noqa
        *,
        return_log_probs: bool | None = None,
        tokenizer: transformers.tokenization_utils.PreTrainedTokenizer  # noqa
        | None = None,
        # noqa
        from_text: bool = False,
        device: torch.device | None = None,
        generate: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = True,
        inplace: Literal[True, False, "empty"] | None = True,
    ):
        super().__init__()

        self.model = model
        self.from_text = from_text
        self._device = device
        self.generate = generate
        self.inplace = inplace
        self.pad_output = pad_output
        padding_value = None

        if not tokenizer_kwargs:
            tokenizer_kwargs = {}
        if not tokenizer_kwargs.setdefault("return_attention_mask", True):
            raise RuntimeError

        # If we don't pad, we use lists
        if not self.pad_output:
            raise NotImplementedError("transformers requires `pad_output=True`.")
        if tokenizer_kwargs.setdefault("return_tensors", "pt") != "pt":
            raise RuntimeError
        if tokenizer_kwargs.setdefault("padding", self.pad_output) not in (
            self.pad_output,
        ):
            raise RuntimeError
        if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
            raise RuntimeError

        self.tokenizer_kwargs = tokenizer_kwargs
        if (pad_output or (from_text and not generate)) and tokenizer is None:
            # We need a tokenizer if we pad or when using text inputs with generate=False
            #  The latter case is due to the fact that we want the log-probs for the response only,
            #  but if the response is presented as a text we have to tokenize the whole prompt + response and
            #  identify where the prompt ends and where the response starts.
            tokenizer = model.get_tokenizer()
        self.tokenizer = tokenizer
        if tokenizer is not None and (
            not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer is not None:
            padding_value = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.padding_value = padding_value

        if generate_kwargs is None:
            generate_kwargs = {}
        else:
            generate_kwargs = dict(generate_kwargs)

        if not generate:
            # TODO
            if return_log_probs in (None, True):
                return_log_probs = True
            else:
                raise ValueError(
                    "return_log_probs must be True or None when generate=False."
                )
        elif return_log_probs in (None, False):
            return_log_probs = False
        self.return_log_probs = return_log_probs

        generate_kwargs.setdefault("tokenizer", self.tokenizer)
        generate_kwargs.setdefault("output_logits", self.return_log_probs)
        generate_kwargs.setdefault("return_dict_in_generate", True)

        self.generate_kwargs = generate_kwargs

        if from_text:
            self.in_keys = [self.text_key]
        else:
            self.in_keys = [self.token_key, self.attention_mask_key]
        self.out_keys = [self.token_response_key]
        if from_text:
            self.out_keys += [self.text_response_key, self.token_key]
        if self.return_log_probs:
            self.out_keys += ["log_probs", "logits"]

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        _source_device = None
        if self._device:
            _source_device = tensordict.device
        if tensordict.device:
            tensordict = tensordict.copy().clear_device_()

        out = LazyStackedTensorDict(
            *[
                TensorDict(
                    device=tensordict.device, batch_size=tensordict.batch_size[1:]
                )
                for _ in range(tensordict.shape[0])
            ]
        )
        if self.from_text:
            if self.generate:
                out = self._from_transformers_generate_text(tensordict, out=out)
            else:
                out = self._from_transformers_logprobs_text(tensordict, out=out)
        else:
            if self.generate:
                out = self._from_transformers_generate_tokens(tensordict, out=out)
            else:
                out = self._from_transformers_logprobs_tokens(tensordict, out=out)
        if _source_device:
            out = out.to(_source_device)

        if tensordict_out is None:
            if self.inplace is True:
                tensordict_out = tensordict
            elif self.inplace is False:
                tensordict_out = TensorDict()
            elif self.inplace == "empty":
                tensordict_out = tensordict.empty()

        if tensordict_out is not None:
            result = tensordict_out
            result.update(out, keys_to_update=self.out_keys)
        else:
            result = out
            keys = list(set(self.out_keys + list(tensordict.keys(True, True))))
            return tensordict.update(result, keys_to_update=keys)
        return result

    def _from_transformers_generate_text(self, td, out):
        text = td.get(self.text_key)
        if not isinstance(text, (list, str)):
            text = text.tolist()
        tokens_in = self.tokenizer(text, **self.tokenizer_kwargs)
        input_ids = tokens_in["input_ids"]
        attention_mask = tokens_in["attention_mask"]
        tokens_out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **self.generate_kwargs
        )
        sequences = tokens_out["sequences"]
        sequences = sequences[..., input_ids.shape[-1] :]
        if self.return_log_probs:
            logits = tokens_out["logits"]
            log_probs, logits = self._log_probs_generate(sequences, logits)
        response_text = self.tokenizer.batch_decode(sequences)
        out.set("tokens_response", sequences)
        out.set("tokens", input_ids)
        out.set("attention_mask", attention_mask)
        out.set("text_response", NonTensorStack(*response_text))
        if self.return_log_probs:
            out.set("log_probs", log_probs)
            out.set("logits", logits)
        return out

    def _from_transformers_generate_tokens(self, td, out):
        input_ids = td.get(self.token_key)
        attention_mask = td.get(self.attention_mask_key)
        tokens_out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **self.generate_kwargs
        )
        sequences = tokens_out["sequences"]
        sequences = sequences[..., input_ids.shape[-1] :]
        if self.return_log_probs:
            logits = tokens_out["logits"]
            log_probs, logits = self._log_probs_generate(sequences, logits)
        out.set("tokens_response", sequences)
        out.set("tokens", input_ids)
        out.set("attention_mask", attention_mask)
        if self.return_log_probs:
            out.set("log_probs", log_probs)
            out.set("logits", logits)
        return out

    def _from_transformers_logprobs_text(self, td, out):
        prompt_txt = td.get(self.text_key)
        if not isinstance(prompt_txt, (list, str)):
            prompt_txt = prompt_txt.tolist()
        response_txt = td.get(self.text_response_key)
        if not isinstance(response_txt, (list, str)):
            response_txt = response_txt.tolist()
        total_txt = [x + y for x, y in _zip_strict(prompt_txt, response_txt)]
        total_tokens_in = self.tokenizer(total_txt, **self.tokenizer_kwargs)
        prompt_tokens_in = self.tokenizer(prompt_txt, **self.tokenizer_kwargs)

        total_input_ids = total_tokens_in["input_ids"]
        total_attention_mask = total_tokens_in["attention_mask"]
        prompt_input_ids = prompt_tokens_in["input_ids"]
        # prompt_attention_mask = prompt_tokens_in["attention_mask"]
        response_input_ids = total_input_ids[:, prompt_input_ids.shape[-1] :]
        # response_attention_mask = total_attention_mask[
        #     :, prompt_attention_mask.shape[-1] :
        # ]

        total_tokens_out = self.model(
            total_input_ids, attention_mask=total_attention_mask, **self.generate_kwargs
        )
        log_probs, logits = self._log_probs_from_logits(
            total_tokens_out, response_input_ids
        )
        out.set("logits", logits)
        out.set("log_probs", log_probs)
        out.set(self.token_response_key, response_input_ids)
        return out

    def _from_transformers_logprobs_tokens(self, td, out):
        prompt_input_ids = td[self.token_key]
        response_input_ids = td[self.token_response_key]
        prompt_attention_mask = td[self.attention_mask_key]
        total_input_ids = torch.cat([prompt_input_ids, response_input_ids], -1)

        if prompt_attention_mask is not None:
            total_attention_mask = torch.cat(
                [
                    prompt_attention_mask,
                    prompt_attention_mask.new_ones(response_input_ids.shape),
                ],
                -1,
            )

        total_tokens_out = self.model(
            total_input_ids, attention_mask=total_attention_mask, **self.generate_kwargs
        )
        log_probs, logits = self._log_probs_from_logits(
            total_tokens_out, response_input_ids
        )
        out.set("logits", logits)
        out.set("log_probs", log_probs)
        return out

    @classmethod
    def _log_probs_from_logits(cls, total_tokens_out, response_input_ids):
        seq_len = response_input_ids.shape[-1]
        logits = total_tokens_out["logits"]
        logits = logits.log_softmax(dim=-1)
        logits = logits[..., -seq_len - 1 : -1, :]
        log_probs = logits.gather(-1, response_input_ids.unsqueeze(-1))
        return log_probs, logits

    @classmethod
    def _log_probs_generate(cls, sequences, logits):
        seq_len = sequences.shape[1]

        # logits = dict(logits.items())
        logits = torch.stack(list(logits), 1)
        logits = logits.log_softmax(dim=-1)
        logits = logits[..., -seq_len:, :]

        tokens = sequences[..., -seq_len:]  # shape (B, seq-len)
        log_probs = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return log_probs, logits
