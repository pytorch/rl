# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
from typing import Literal

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    NestedKey,
    TensorDict,
    TensorDictBase,
)
from tensordict.tensorclass import from_dataclass, NonTensorStack, TensorClass
from tensordict.utils import _zip_strict, expand_as_right

from torchrl.envs.utils import _classproperty

from torchrl.modules.llm.policies.common import CategoricalSequential


class vLLMWrapper(CategoricalSequential):
    """A wrapper class for vLLM models, providing a consistent interface for text generation and log probability computation, similar to the Hugging Face Transformers interface.

    This class allows for handling both text and token inputs, enabling text generation and log probability
    computation based on the specified configuration. Unlike Transformers, vLLM can handle both padded and non-padded tensors.

    Args:
        model (vllm.LLM): The vLLM model to wrap.

    Keyword Args:
        return_log_probs (bool | None, optional): Whether to return log probabilities of the generated tokens.
            Defaults to `None`.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | None, optional): The tokenizer to use for
            encoding and decoding text. If `None`, the tokenizer associated with the model will be used. Defaults to
            `None`.
        from_text (bool, optional): Indicates whether the input is in text format. If `True`, the input is expected to
            be text that will be tokenized. If `False`, the input is expected to be token sequences. Defaults to `True`.
        use_history (bool, NestedKey, optional): Whether to use the history key as input. If `True`, the input is expected to
            be a history object (under the `"history"` key). If `False`, the input is expected to be a text string, with
            the `"text"` and `"text_response"` key pair. If a `NestedKey` is passed, it will be used as the key for the
            history object. Defaults to `None`: first attempts to use the `("next", "history")` key, then the `"history"` key,
            then finally the `"text"` and `"text_response"` key pair.
        from_text (bool, optional): Indicates whether the input is in text format. If `True`, the input is expected to
            be text that will be tokenized. If `False`, the input is expected to be token sequences. Defaults to `True`.

            .. note:: If `from_text` is `True`, the input text can be provided in the `"text"` key or in the `"history"` key.
                If using the `"history"` key, the history will be parsed from a :class:`~torchrl.data.llm.History` object to a
                text string using the tokenizer.
                See the `use_history` keyword argument for more details.

        device (torch.device | None, optional): The device to use for computation. If `None`, the default device will
            be used. Defaults to `None`.
        generate (bool, optional): Whether to enable text generation. If `True`, the model will generate text based on
            the input. If `False`, only log probabilities will be computed for the response tokens/text. Defaults to `True`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to the model's generate method. These
            arguments can control aspects of the generation process, such as temperature and top-k sampling. Defaults
            to `None`.

            .. note:: Sampling params can be overwritten at runtime using the kwargs of the forward method.

        tokenizer_kwargs (dict | None, optional): Additional arguments to pass to the tokenizer. These arguments can
            control aspects of the tokenization process, such as padding and truncation. Defaults to `None`.
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. If `True`, the output
            sequences will be padded and represented as tensors. If `False`, lists of tokens will be used without
            padding. Defaults to `False`.

            .. warning:: The default value of `pad_output` differs from :func:`~torchrl.modules.TransformersWrapper`
                which does not handle non-padded inputs.

        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place
            operations. If `True`, operations will be performed in-place. If `False`, a new TensorDict instance will be
            created. If `"empty"`, the output data structure will be initialized with `input.empty()` (i.e., it will
            conserve type, batch-size, and device). Defaults to `True` when generating a single sample, `False`
            otherwise.

        chat_template_name (str | None, optional): The name of the chat template to use for the history. Defaults to `None`.
        chat_template (str | None, optional): The chat template to use for the history. Defaults to `None`.
        assistant_only (bool, optional): Whether to only use the assistant tokens in the history when computing the log-probs.
            Defaults to `True`.
            Without effect if the history is not used.

            .. note:: This presumes that the chat format supports the `return_assistant_tokens_mask` argument, which is not the case for all chat formats.

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
        >>> from vllm import LLM
        >>> from transformers import AutoTokenizer
        >>> model = LLM("gpt2")
        >>> wrapper = vLLMWrapper(
        ...     model,
        ...     from_text=True,
        ...     generate=True
        ... )
        >>> input_data = LLMData(text=NonTensorStack("Hello, world!", "This is another text"), batch_size=1)
        >>> output_data = wrapper(input_data)
        >>> print(output_data.text_response)

    .. seealso:: :func:`~torchrl.modules.TransformersWrapper` for a similar interface using the Hugging Face
        Transformers library.
    """

    text_key: NestedKey = ("text",)
    token_key: NestedKey = ("tokens",)
    token_response_key: NestedKey = ("tokens_response",)
    text_response_key: NestedKey = ("text_response",)
    attention_mask_key: NestedKey = ("attention_mask",)
    history_key: NestedKey = ("history",)

    def __init__(
        self,
        model: vllm.LLM,  # noqa
        # noqa
        *,
        return_log_probs: bool | None = None,
        tokenizer: transformers.tokenization_utils.PreTrainedTokenizer  # noqa
        | None = None,
        # noqa
        from_text: bool = True,
        device: torch.device | None = None,
        generate: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        inplace: Literal[True, False, "empty"] | None = None,
        chat_template_name: str | None = None,
        chat_template: str | None = None,
        use_history: bool | NestedKey | None = None,
        assistant_only: bool = True,
    ):
        super().__init__()

        import vllm
        from vllm import SamplingParams

        self.model = model
        self._remote_calls = not isinstance(model, vllm.LLM)
        self.from_text = from_text
        self._device = device
        self.generate = generate
        self.pad_output = pad_output
        self.chat_template_name = chat_template_name
        self.chat_template = chat_template
        self.assistant_only = assistant_only
        padding_value = None

        if isinstance(use_history, (str, tuple)):  # equivalent to NestedKey but faster
            self.history_key = use_history
            self.use_history = True
        else:
            self.history_key = None
            self.use_history = use_history  # None means "maybe"

        if not tokenizer_kwargs:
            tokenizer_kwargs = {}
        if not tokenizer_kwargs.setdefault("return_attention_mask", True):
            raise RuntimeError

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

        self.num_samples = None
        if generate_kwargs.get("n", 1) > 1:
            if inplace in (True, "empty"):
                raise ValueError(
                    "inplace must be False (or None) when generating more than one sample."
                )
            if inplace is None:
                inplace = False
            self.num_samples = generate_kwargs.get("n", 1)
        elif inplace is None:
            inplace = True

        self.inplace = inplace

        prompt_logprobs = False

        if not generate:
            # We want only the log-probs, we generate a single token (that we then discard)
            #  and retrieve the prompt log-probs
            generate_kwargs["max_tokens"] = 1
            prompt_logprobs = True
            if return_log_probs in (None, True):
                return_log_probs = True
            else:
                raise ValueError(
                    "return_log_probs must be True or None when generate=False."
                )
        elif return_log_probs in (None, False):
            return_log_probs = False
        self.return_log_probs = return_log_probs

        generate_kwargs.setdefault("detokenize", not pad_output)
        generate_kwargs.setdefault("prompt_logprobs", prompt_logprobs)
        generate_kwargs.setdefault("logprobs", return_log_probs)
        generate_kwargs.setdefault("include_stop_str_in_output", True)
        generate_kwargs.setdefault("skip_special_tokens", False)

        sampling_params = SamplingParams(**generate_kwargs)
        self.sampling_params = sampling_params

        if from_text:
            self.in_keys = [self.text_key]
        else:
            self.in_keys = [self.token_key, self.attention_mask_key]
        self.out_keys = [self.token_response_key]
        if from_text:
            self.out_keys += [self.text_response_key, self.token_key]
        if self.return_log_probs:
            self.out_keys += [self.log_prob_key]

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

        if kwargs:
            sampling_params = self.sampling_params.clone()
            for key, val in kwargs.items():
                setattr(sampling_params, key, val)
        else:
            sampling_params = self.sampling_params

        _source_device = None
        if self._device:
            _source_device = tensordict.device
        if tensordict.device:
            tensordict = tensordict.copy().clear_device_()

        if self.generate and self.num_samples is not None:
            n = self.num_samples
            out = LazyStackedTensorDict(
                *[
                    TensorDict(
                        device=tensordict.device,
                        batch_size=(n,) + tensordict.batch_size[1:],
                    ).to_lazystack()
                    for _ in range(tensordict.shape[0])
                ]
            )
        else:
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
                out = self._from_vllm_generate_text(
                    tensordict, sampling_params=sampling_params, out=out
                )
            else:
                out = self._from_vllm_logprobs_text(
                    tensordict, sampling_params=sampling_params, out=out
                )
        else:
            if self.generate:
                out = self._from_vllm_generate_tokens(
                    tensordict, sampling_params=sampling_params, out=out
                )
            else:
                out = self._from_vllm_logprobs_tokens(
                    tensordict, sampling_params=sampling_params, out=out
                )
        if _source_device:
            out = out.to(_source_device)

        if tensordict_out is None:
            if self.inplace is True:
                tensordict_out = tensordict
            elif self.inplace is False:
                tensordict_out = out
            elif self.inplace == "empty":
                tensordict_out = tensordict.empty()

        if tensordict_out is not None and tensordict_out is not out:
            result = tensordict_out
            result.update(out, keys_to_update=self.out_keys)
        elif tensordict_out is not out:
            result = out
            keys = list(set(self.out_keys + list(tensordict.keys(True, True))))
            return tensordict.update(result, keys_to_update=keys)
        else:
            result = out
        return result

    def _from_vllm_generate_text(self, td, sampling_params, out) -> TensorDictBase:
        kwargs = {"sampling_params": sampling_params}
        args = ()
        input_ids = None
        attention_mask = None

        # Should we use the history?
        if self.use_history is None or (
            self.use_history is True and self.history_key is None
        ):
            # eagerly attempt to use the history key
            history_key = self.history_key
            if history_key is None:
                history_key = ("next", "history")
                if history_key not in td:
                    history_key = "history"
            use_history = bool(self.use_history) or history_key in td
        else:
            use_history = self.use_history
            history_key = self.history_key

        if use_history:
            # Fallback on history parsing
            if history_key is None:
                history_key = ("next", "history")
                if history_key not in td:
                    history_key = "history"
            history = td.get(history_key)
            if history is None:
                raise ValueError("No text or history provided to the vLLMWrapper.")
            tokenizer_kwargs = {}
            if self.chat_template_name is not None:
                tokenizer_kwargs["chat_template_name"] = self.chat_template_name
            if self.chat_template is not None:
                tokenizer_kwargs["chat_template"] = self.chat_template
            text = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )
        else:
            text = td.get(self.text_key)
            if text is None:
                raise ValueError(
                    "No text or history provided to the vLLMWrapper and the 'text' key is not present."
                )

        if self.pad_output:
            tokenizer_kwargs = self.tokenizer_kwargs
            if not isinstance(text, (list, str)):
                text = text.tolist()
            tokens_in = TensorDict.from_dict(self.tokenizer(text, **tokenizer_kwargs))
            # out.set("tokens_in", tokens_in)
            input_ids, attention_mask = (
                tokens_in["input_ids"],
                tokens_in["attention_mask"],
            )
            prompt_token_ids = self._to_list(input_ids, attention_mask)
            kwargs["prompt_token_ids"] = prompt_token_ids
        else:
            if not isinstance(text, (list, str)):
                text = text.tolist()
            args = (text,)

        if not self._remote_calls:
            tokens_out = self.model.generate(*args, **kwargs)
        else:
            import ray

            tokens_out = ray.get(self.model.generate.remote(*args, **kwargs))

        tokens_out = self._get_output_tokens_and_log_probs(tokens_out)
        if self.pad_output:
            tokens_out.set(
                self.text_response_key,
                NonTensorStack(
                    *self.tokenizer.batch_decode(tokens_out[self.token_response_key])
                ),
            )
        in_keys = [
            self.log_prob_key,
            self.token_response_key,
            self.text_response_key,
            self.token_key,
            self.attention_mask_key,
        ]
        print("tokens_out", tokens_out)
        print("out", out)
        out = out.update(tokens_out.select(*in_keys, strict=False))
        # We might already have the tokens
        if input_ids is not None and self.token_key not in out:
            out[self.token_key] = input_ids
        if attention_mask is not None and self.attention_mask_key not in out:
            out[self.attention_mask_key] = attention_mask
        inputs = td.select(*self.in_keys, strict=False)
        if inputs.ndim < out.ndim:
            # This happens when n > 1
            inputs = inputs.unsqueeze(-1).expand(out.shape)
        out.update(inputs)
        return out

    def _from_vllm_logprobs_text(self, td, sampling_params, out):
        # Should we use the history?
        if self.use_history is None or (
            self.use_history is True and self.history_key is None
        ):
            # eagerly attempt to use the history key
            history_key = self.history_key
            if history_key is None:
                history_key = ("next", "history")
                if history_key not in td:
                    history_key = "history"
            use_history = bool(self.use_history) or history_key in td
        else:
            use_history = self.use_history
            history_key = self.history_key

        if use_history:
            # Fallback on history parsing
            if history_key is None:
                history_key = ("next", "history")
                if history_key not in td:
                    history_key = "history"
            history = td.get(history_key)
            if history is None:
                raise ValueError("No text or history provided to the vLLMWrapper.")
            tokenizer_kwargs = {}
            if self.chat_template_name is not None:
                tokenizer_kwargs.setdefault(
                    "chat_template_name", self.chat_template_name
                )
            if self.chat_template is not None:
                tokenizer_kwargs.setdefault("chat_template", self.chat_template)
            tokenizer_kwargs.setdefault("add_generation_prompt", False)
            tokenizer_kwargs.setdefault(
                "return_assistant_tokens_mask", self.assistant_only
            )
            tokenizer_kwargs.setdefault("tokenize", True)
            tokenizer_kwargs.setdefault("padding", False)
            tokenizer_kwargs.setdefault("return_dict", True)

            response_tokens = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )
            # unfortunately vLLM wants us to use padded tensors
            total_input_ids = response_tokens.get(
                "input_ids",
                as_padded_tensor=True,
                padding_side="left",
                padding_value=self.padding_value,
            )
            total_attention_mask = response_tokens.get(
                "attention_mask",
                as_padded_tensor=True,
                padding_side="left",
                padding_value=0,
            )
            if self.assistant_only:
                assistant_masks = response_tokens.get(
                    "assistant_masks",
                    as_padded_tensor=True,
                    padding_side="left",
                    padding_value=0,
                ).bool()
            else:
                assistant_masks = None
        else:
            text_prompt = td.get(self.text_key)
            text_response = td.get(self.text_response_key)
            if text_response is None or text_prompt is None:
                raise ValueError(
                    "No text or history provided to the vLLMWrapper and the text keys are not present."
                )

            if not isinstance(text_prompt, list):
                text_prompt = text_prompt.tolist()
            if not isinstance(text_response, list):
                text_response = text_response.tolist()
            text = [_x + _y for _x, _y in _zip_strict(text_prompt, text_response)]

            tokenized_total = self.tokenizer(text, **self.tokenizer_kwargs)
            tokenized_prompt_only = self.tokenizer(text_prompt, **self.tokenizer_kwargs)

            total_input_ids = tokenized_total["input_ids"]
            total_attention_mask = tokenized_total["attention_mask"]
            assistant_masks = None

        input_ids_total = self._to_list(total_input_ids, total_attention_mask)
        kwargs = {"sampling_params": sampling_params}
        if self.tokenizer is not None:
            kwargs.update({"prompt_token_ids": input_ids_total})
            args = ()
        else:
            # TODO: this is unreachable as of now - but ultimately we may want to pass the text directly
            args = (td[self.text_key],)
        if not self._remote_calls:
            tokens_out = self.model.generate(*args, **kwargs)
        else:
            import ray

            tokens_out = ray.get(self.model.generate.remote(*args, **kwargs))

        tokens_out = _RequestOutput_tc.from_request_output(tokens_out)
        tokens_out = tokens_out.select(
            "prompt_token_ids", "prompt_logprobs", strict=False
        )._tensordict

        # we disregard the tokens from the prompt to focus on those of the response
        if self.pad_output:
            lps = tokens_out.get(
                "prompt_logprobs", as_padded_tensor=True, padding_side="left"
            )
            if not use_history:
                # For text/text_response mode, we need to extract response tokens
                tokenized_prompt_only = self.tokenizer(
                    text_prompt, **self.tokenizer_kwargs
                )
                input_ids_prompt = tokenized_prompt_only["input_ids"]
                input_ids_response = total_input_ids[:, input_ids_prompt.shape[1] :]
                lps = lps[..., -input_ids_response.shape[1] :]
                padded = input_ids_response == self.padding_value
                lps = torch.where(~padded, lps, 0.0)
                input_ids_response_final = input_ids_response
            else:
                # For history mode, we take all tokens
                lps = lps[..., -total_input_ids.shape[1] :]
                padded = total_input_ids == self.padding_value
                lps = torch.where(~padded, lps, 0.0)
                input_ids_response_final = total_input_ids

                # Apply assistant mask if needed
                if self.assistant_only and assistant_masks is not None:
                    # Select only assistant tokens
                    lps = [lp[am] for lp, am in _zip_strict(lps, assistant_masks)]
                    input_ids_response_final = [
                        ids[am]
                        for ids, am in _zip_strict(total_input_ids, assistant_masks)
                    ]
        else:
            lps = tokens_out.get(
                "prompt_logprobs",
                as_list=True,
            )
            if not use_history:
                # For text/text_response mode
                tokenized_prompt_only = self.tokenizer(
                    text_prompt, **self.tokenizer_kwargs
                )
                input_ids_prompt = tokenized_prompt_only["input_ids"]
                input_ids_response = []
                for token_total, token_prompt in zip(total_input_ids, input_ids_prompt):
                    input_ids_response.append(token_total[len(token_prompt) :])
                # We use a nested tensor as it will be unbound during writing
                lps = torch.nested.nested_tensor(
                    [lp[..., -len(tr) :] for lp, tr in zip(lps, input_ids_response)]
                )
                input_ids_response_final = input_ids_response
            else:
                # For history mode
                lps = torch.nested.nested_tensor(
                    [lp[..., -len(tr) :] for lp, tr in zip(lps, total_input_ids)]
                )
                input_ids_response_final = total_input_ids

                # Apply assistant mask if needed
                if self.assistant_only and assistant_masks is not None:
                    # Select only assistant tokens
                    lps = [lp[am] for lp, am in _zip_strict(lps, assistant_masks)]
                    input_ids_response_final = [
                        ids[am]
                        for ids, am in _zip_strict(total_input_ids, assistant_masks)
                    ]

        out = out.update(tokens_out.empty(recurse=True))
        if isinstance(input_ids_response_final, list):
            input_ids_response_final = torch.nested.nested_tensor(
                input_ids_response_final
            )
        out["tokens_response"] = input_ids_response_final
        out[self.log_prob_key] = lps
        inputs = td.select(*self.in_keys, strict=False)
        if inputs.ndim < out.ndim:
            # This happens when n > 1
            inputs = inputs.unsqueeze(-1).expand(out.shape)
        out.update(inputs)
        return out

    def _from_vllm_generate_tokens(self, td, sampling_params, out):
        input_ids = td.get(self.token_key)
        attention_mask = td.get(self.attention_mask_key)
        input_ids_list = self._to_list(input_ids, attention_mask)
        args = ()
        kwargs = {
            "sampling_params": sampling_params,
            "prompt_token_ids": input_ids_list,
        }
        if not self._remote_calls:
            tokens_out = self.model.generate(*args, **kwargs)
        else:
            import ray

            tokens_out = ray.get(self.model.generate.remote(*args, **kwargs))
        tokens_out = _RequestOutput_tc.from_request_output(tokens_out)
        # When not generate, we don't want to overwrite this
        tokens_response_td = tokens_out.outputs._tensordict.select(
            "token_ids", "logprobs", strict=False
        )
        if self.pad_output:
            tokens_response_td = tokens_response_td.densify(
                layout=torch.strided
            ).to_padded_tensor(padding=self.padding_value)
        tokens_response_td.rename_key_("token_ids", "tokens_response")
        if self.return_log_probs:
            tokens_response_td.rename_key_("logprobs", self.log_prob_key)
            if self.pad_output:
                padded_values = (
                    tokens_response_td["tokens_response"] == self.padding_value
                )
                if padded_values.any():
                    lps = tokens_response_td[self.log_prob_key]
                    lps = torch.where(expand_as_right(~padded_values, lps), lps, 0.0)
                    tokens_response_td[self.log_prob_key] = lps
        out = out.update(tokens_response_td.empty(recurse=True))
        out.update(
            tokens_response_td,
            keys_to_update=(self.token_response_key, self.log_prob_key),
        )
        inputs = td.select(*self.in_keys, strict=False)
        if inputs.ndim < out.ndim:
            # This happens when n > 1
            inputs = inputs.unsqueeze(-1).expand(out.shape)
        out.update(inputs)
        return out

    def _from_vllm_logprobs_tokens(self, td, sampling_params, out):

        tokens = td.get(self.token_key)
        tokens_response = td.get(self.token_response_key)
        attention_mask = td.get(self.attention_mask_key)

        tokens = torch.cat([tokens, tokens_response], -1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(tokens_response.shape)], -1
            )
        input_ids_list = self._to_list(tokens, attention_mask)
        args = ()
        kwargs = {
            "sampling_params": sampling_params,
            "prompt_token_ids": input_ids_list,
        }
        if not self._remote_calls:
            tokens_out = self.model.generate(*args, **kwargs)
        else:
            import ray

            tokens_out = ray.get(self.model.generate.remote(*args, **kwargs))
        tokens_out = _RequestOutput_tc.from_request_output(tokens_out)
        prompt_logprobs = tokens_out.prompt_logprobs
        prompt_logprobs = prompt_logprobs[..., -tokens_response.shape[-1] :]
        padded = tokens_response == self.padding_value
        prompt_logprobs = torch.where(~padded, prompt_logprobs, 0.0)
        out = out.update(tokens_out._tensordict.empty(recurse=True))
        out.set(self.log_prob_key, prompt_logprobs)
        out.set(self.token_response_key, tokens_response)
        inputs = td.select(*self.in_keys, strict=False)
        if inputs.ndim < out.ndim:
            # This happens when n > 1
            inputs = inputs.unsqueeze(-1).expand(out.shape)
        out.update(inputs)
        return out

    def _get_output_tokens_and_log_probs(self, tokens_out):
        padding_value = self.padding_value
        tokens_out = _RequestOutput_tc.from_request_output(tokens_out)

        # When not generate, we don't want to overwrite this
        tokens_response_td = tokens_out.outputs._tensordict.select(
            "text", "token_ids", "logprobs", strict=False
        )
        if self.pad_output:
            tokens_response_td = tokens_response_td.densify(
                layout=torch.strided
            ).to_padded_tensor(padding=padding_value)
        tokens_response_td.rename_key_("token_ids", "tokens_response")
        tokens_response_td.rename_key_("text", "text_response")
        if not self.pad_output:
            # Then we can safely move the input tokens, but otherwise they
            #  may need padding
            tokens_out = tokens_out.select("prompt_token_ids")
            if tokens_out.ndim < tokens_response_td.ndim:
                tokens_out = tokens_out.unsqueeze(1).expand(tokens_response_td.shape)
            tokens_response_td.update(tokens_out).rename_key_(
                "prompt_token_ids", self.token_key
            )

        if self.return_log_probs or "logprobs" in tokens_response_td:
            tokens_response_td.rename_key_("logprobs", self.log_prob_key)
            if self.pad_output:
                padded_values = tokens_response_td["tokens_response"] == padding_value
                if padded_values.any():
                    lps = tokens_response_td[self.log_prob_key]
                    lps = torch.where(expand_as_right(~padded_values, lps), lps, 0.0)
                    tokens_response_td[self.log_prob_key] = lps
        return tokens_response_td

    def _to_list(self, tokens, attention_mask):
        """Converts a tensor of integer in a masked list (of lists) of integers."""
        if isinstance(tokens, torch.Tensor):
            # TODO: make this an ND NonTensorStack
            parent = []
            queue = collections.deque()
            if attention_mask is None:
                attention_mask = torch.ones_like(tokens)
            queue.append((tokens, attention_mask.bool(), parent))
            while queue:
                token, amask, _parent = queue.popleft()
                if token.ndim == 1:
                    _parent.extend(token[amask].tolist())
                else:
                    _parent.extend([[] for _ in range(token.shape[0])])
                    queue.extend(
                        [
                            (t, m, local_parent)
                            for t, m, local_parent in zip(token, amask, _parent)
                        ]
                    )
            tokens = parent
        return tokens

    @_classproperty
    def CompletionOutput_tc(cls):
        import vllm

        if hasattr(cls, "_CompletionOutput_tc"):
            return cls._CompletionOutput_tc
        CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)
        cls._CompletionOutput_tc = CompletionOutput_tc
        return CompletionOutput_tc


class _RequestOutput_tc(TensorClass["nocast"]):
    request_id: str
    prompt: str
    prompt_token_ids: str
    prompt_logprobs: str
    outputs: str
    finished: str
    metrics: str
    lora_request: str
    encoder_prompt: str
    encoder_prompt_token_ids: str
    num_cached_tokens: str

    def __post_init__(self):
        CompletionOutput_tc = vLLMWrapper.CompletionOutput_tc

        def postproc(output):
            def get_logprob(output):
                t = []
                for v, tid in zip(output.logprobs, output.token_ids):
                    t.append(
                        v[int(tid)]["logprob"]
                        if v[tid].get("logprob") is not None
                        else 0.0
                    )
                return torch.tensor(t)

            if output.logprobs:
                output.logprobs = get_logprob(output)
            output.token_ids = torch.as_tensor(output.token_ids)
            return output

        if isinstance(self.outputs, list):
            outputs = self.outputs
            outputs = [
                postproc(from_dataclass(output, dest_cls=CompletionOutput_tc))
                for output in outputs
            ]
            if len(outputs) == 1:
                self.outputs = outputs[0]
            else:
                self.outputs = maybe_dense_stack(outputs)
            if self.prompt_logprobs is not None:
                self.prompt_logprobs = torch.tensor(
                    [
                        v[int(tid)].logprob if v is not None else 0.0
                        for v, tid in _zip_strict(
                            self.prompt_logprobs, self.prompt_token_ids
                        )
                    ]
                )
            self.prompt_token_ids = torch.as_tensor(self.prompt_token_ids)
            self.num_cached_tokens = torch.as_tensor(self.num_cached_tokens)

    @classmethod
    def from_request_output(cls, requests):
        out = lazy_stack(
            [
                cls(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_logprobs=request.prompt_logprobs,
                    outputs=request.outputs,
                    finished=request.finished,
                    metrics=request.metrics,
                    lora_request=request.lora_request,
                    encoder_prompt=request.encoder_prompt,
                    encoder_prompt_token_ids=request.encoder_prompt_token_ids,
                    num_cached_tokens=request.num_cached_tokens,
                )
                for request in requests
            ]
        )
        return out
