# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import warnings
from typing import Any, Literal

import torch
from tensordict import (
    lazy_stack,
    MetaData,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.tensorclass import from_dataclass, TensorClass
from tensordict.utils import _zip_strict
from torch.nn.utils.rnn import pad_sequence

from torchrl.envs.utils import _classproperty
from torchrl.modules.llm.policies.common import (
    CategoricalSequential,
    LogProbs,
    Masks,
    Text,
    Tokens,
)
from torchrl.modules.utils.utils import _unpad_tensors

# Type imports
try:
    import transformers
    import vllm
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
except ImportError:
    vllm = None
    transformers = None
    SamplingParams = Any  # type: ignore
    RequestOutput = Any  # type: ignore


class vLLMWrapper(CategoricalSequential):
    """A wrapper class for vLLM models, providing a consistent interface for text generation and log probability computation.

    This class provides a simplified API for handling different input modalities (history, text, tokens) with consistent
    output structure using :class:`~tensordict.TensorClass` objects.

    Args:
        model (vllm.LLM | str): The vLLM model to wrap. If a string, it will be passed to `vllm.LLM`.

    Keyword Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | str | None, optional): The tokenizer to use for
            encoding and decoding text. If `None`, the tokenizer associated with the model will be used.
            If a string, it will be passed to `transformers.AutoTokenizer.from_pretrained`. Defaults to `None`.
        input_mode (str, optional): The input modality to use. Must be one of `"history"`, `"text"`, or `"tokens"`.
            Defaults to `"history"`.
        input_key (str | None, optional): The key for the input data. If `None`, defaults to "history" for `"history"`, 
            `("text", "prompt")` for `"text"`, and `("tokens", "prompt")` for `"tokens"`.
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
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. Defaults to `False`.
        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place
            operations. Defaults to `True` when generating a single sample, `False` otherwise.
        device (torch.device | None, optional): The device to use for computation. Defaults to `None`.
        layout (torch.layout | None, optional): The layout to use for the output tensors when `pad_output=False`. Defaults to `torch.strided`.
        chat_template_name (Literal["chatml_format", "qwen"] | None, optional): The name of the chat template to use when
            applying the chat template to the history. Defaults to `None`.
            For `input_mode="history"` only.
        chat_template (str | None, optional): The chat template to use when applying the chat template to the history.
            Defaults to `None`.
            For `input_mode="history"` only.
        num_samples (int | None, optional): The number of samples to generate. Defaults to `None` (one sample, and no batch-dimension for it).
            Can also be set via the `generate_kwargs["n"] = value` argument.

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
        >>> from vllm import LLM
        >>> from transformers import AutoTokenizer
        >>> from torchrl.data.llm import History
        >>>
        >>> model = LLM("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # History input (recommended for RL environments)
        >>> wrapper = vLLMWrapper(
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

    .. seealso:: :class:`~torchrl.modules.llm.TransformersWrapper` for a similar interface using the Hugging Face
        Transformers library.
    """

    def __init__(
        self,
        model: vllm.LLM | str,
        *,
        tokenizer: callable | str | None = None,  # type: ignore
        input_mode: str = "history",
        input_key: NestedKey | None = None,
        attention_mask_key: str = "attention_mask",
        generate: bool = True,
        return_log_probs: bool = False,
        return_text: bool = True,
        return_tokens: bool = True,
        return_masks: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        inplace: Literal[True, False, "empty"] | None = None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        chat_template_name: Literal["chatml_format", "qwen"] | None = None,
        chat_template: str | None = None,
    ):
        super().__init__()

        if vllm is None:
            raise ImportError("vllm is required for vLLMWrapper")
        if transformers is None:
            raise ImportError("transformers is required for vLLMWrapper")

        if isinstance(model, str):
            model = vllm.LLM(model)

        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        from vllm import SamplingParams

        # Validate input_mode
        if input_mode not in ["history", "text", "tokens"]:
            raise ValueError(
                f"input_mode must be one of 'history', 'text', 'tokens'. Got '{input_mode}'"
            )

        self.model = model
        self._remote_calls = not isinstance(model, vllm.LLM)
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

        # Set input keys based on mode
        if input_mode == "history":
            self.in_keys = ["history" if input_key is None else input_key]
        elif input_mode == "text":
            self.in_keys = [("text", "prompt") if input_key is None else input_key]
        elif input_mode == "tokens":
            self.in_keys = [("tokens", "prompt") if input_key is None else input_key]
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")
        self.input_key = self.in_keys[0]

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
        if tokenizer is None:
            try:
                tokenizer = model.get_tokenizer()
            except AttributeError:
                warnings.warn("No tokenizer provided and no tokenizer found in model.")
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
        if generate_kwargs.get("n", 1) > 1 or num_samples is not None:
            if inplace in (True, "empty"):
                raise ValueError(
                    "inplace must be False (or None) when generating more than one sample."
                )
            if inplace is None:
                inplace = False
            if (
                generate_kwargs.get("n", 1) > 1
                and num_samples is not None
                and generate_kwargs.get("n", 1) != num_samples
            ):
                raise ValueError("num_samples differs from generate_kwargs['n'].")
            elif num_samples is None:
                self.num_samples = generate_kwargs.get("n", 1)
            generate_kwargs["n"] = self.num_samples
        elif inplace is None:
            inplace = True

        self.inplace = inplace

        prompt_logprobs = return_log_probs

        if not generate:
            # We want only the log-probs, we generate a single token (that we then discard)
            # and retrieve the prompt log-probs
            generate_kwargs["max_tokens"] = 1
            if not return_log_probs:
                raise ValueError("return_log_probs must be True when generate=False.")

        generate_kwargs.setdefault("detokenize", not pad_output)
        generate_kwargs.setdefault("prompt_logprobs", prompt_logprobs)
        generate_kwargs.setdefault("logprobs", return_log_probs)
        generate_kwargs.setdefault("include_stop_str_in_output", True)
        generate_kwargs.setdefault("skip_special_tokens", False)

        sampling_params = SamplingParams(**generate_kwargs)
        self.sampling_params = sampling_params

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
            from vllm import SamplingParams

            sampling_params = SamplingParams(**kwargs)
        else:
            sampling_params = self.sampling_params

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
                out = self._from_vllm_generate_history(tensordict, sampling_params, out)
            else:
                out = self._from_vllm_logprobs_history(tensordict, sampling_params, out)
        elif self.input_mode == "text":
            if self.generate:
                out = self._from_vllm_generate_text(tensordict, sampling_params, out)
            else:
                out = self._from_vllm_logprobs_text(tensordict, sampling_params, out)
        elif self.input_mode == "tokens":
            if self.generate:
                out = self._from_vllm_generate_tokens(tensordict, sampling_params, out)
            else:
                out = self._from_vllm_logprobs_tokens(tensordict, sampling_params, out)

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
            result = tensordict.exclude(*self.out_keys, inplace=True).update(result, keys_to_update=keys)
        else:
            result = out
        return result

    def _from_vllm_generate_history(
        self,
        tensordict_input: TensorDictBase,
        sampling_params: SamplingParams,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate text from history input."""
        from torchrl.data.llm import History

        assert isinstance(
            tensordict_input, TensorDictBase
        ), f"tensordict_input must be TensorDictBase, got {type(tensordict_input)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Validate input
        if self.input_key not in tensordict_input:
            raise ValueError(
                f"Expected '{self.input_key}' key for history input mode, "
                f"but found keys: {list(tensordict_input.keys())}"
            )

        history = tensordict_input.get(self.input_key)
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
        return self._generate_from_text(text, sampling_params, out)

    def _from_vllm_logprobs_history(
        self,
        tensordict_input: TensorDictBase,
        sampling_params: SamplingParams,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Compute log-probs from history input."""
        assert isinstance(
            tensordict_input, TensorDictBase
        ), f"tensordict_input must be TensorDictBase, got {type(tensordict_input)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        from torchrl.data.llm import History

        # Validate input
        if self.input_key not in tensordict_input:
            raise ValueError(
                f"Expected '{self.input_key}' key for history input mode, "
                f"but found keys: {list(tensordict_input.keys())}"
            )

        history = tensordict_input.get(self.input_key)
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
        response_struct = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )

        return self._logprobs_from_tokens(
            response_struct=response_struct, sampling_params=sampling_params, out=out
        )

    def _from_vllm_generate_text(
        self, td: TensorDictBase, sampling_params: SamplingParams, out: TensorDictBase
    ) -> TensorDictBase:
        """Generate text from text input."""
        # Type assertions
        assert isinstance(
            td, TensorDictBase
        ), f"td must be TensorDictBase, got {type(td)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for text input mode, "
                f"but found keys: {list(td.keys())}"
            )

        text = td.get(self.input_key)
        if text is None:
            raise ValueError(f"Expected '{self.input_key}' key for text input mode")

        return self._generate_from_text(text, sampling_params, out)

    def _from_vllm_logprobs_text(
        self, td: TensorDictBase, sampling_params: SamplingParams, out: TensorDictBase
    ) -> TensorDictBase:
        """Compute log-probs from text input."""
        # Type assertions
        assert isinstance(
            td, TensorDictBase
        ), f"td must be TensorDictBase, got {type(td)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for text input mode, "
                f"but found keys: {list(td.keys())}"
            )

        text = td.get(self.input_key)
        if text is None:
            raise ValueError(f"Expected '{self.input_key}' key for text input mode")

        return self._logprobs_from_text(text, sampling_params, out)

    def _from_vllm_generate_tokens(
        self, td: TensorDictBase, sampling_params: SamplingParams, out: TensorDictBase
    ) -> TensorDictBase:
        """Generate text from tokens input."""
        # Type assertions
        assert isinstance(
            td, TensorDictBase
        ), f"td must be TensorDictBase, got {type(td)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys())}"
            )

        tokens_prompt_padded = None
        tokens_prompt_unpadded = None
        if self.pad_output:
            tokens_prompt_padded = td.get(self.input_key)
        else:
            tokens_prompt_unpadded = list(td.get(self.input_key, as_list=True))
            # make sure we remove the padding tokens
            tokens_prompt_unpadded = [
                tokens[tokens != self.padding_value]
                for tokens in tokens_prompt_unpadded
            ]

        return self._generate_from_tokens(
            tokens_prompt_unpadded=tokens_prompt_unpadded,
            tokens_prompt_padded=tokens_prompt_padded,
            sampling_params=sampling_params,
            out=out,
        )

    def _from_vllm_logprobs_tokens(
        self, td: TensorDictBase, sampling_params: SamplingParams, out: TensorDictBase
    ) -> TensorDictBase:
        """Compute log-probs from tokens input."""
        # Type assertions
        assert isinstance(
            td, TensorDictBase
        ), f"td must be TensorDictBase, got {type(td)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Validate input
        if self.input_key not in td:
            raise ValueError(
                f"Expected '{self.input_key}' key for tokens input mode, "
                f"but found keys: {list(td.keys())}"
            )

        tokens_full_padded = None
        tokens_full_unpadded = None
        if self.pad_output:
            tokens_full_padded = td.get(self.input_key)
        else:
            tokens_full_unpadded = list(td.get(self.input_key, as_list=True))
            # make sure we remove the padding tokens
            tokens_full_unpadded = [
                tokens[tokens != self.padding_value] for tokens in tokens_full_unpadded
            ]

        return self._logprobs_from_tokens(
            response_struct=None,
            tokens_full_unpadded=tokens_full_unpadded,
            tokens_full_padded=tokens_full_padded,
            sampling_params=sampling_params,
            out=out,
        )

    def _cat_text(
        self, text: str | list[str], response_text: str | list[str]
    ) -> str | list[str]:
        """Concatenate text and response text."""
        assert isinstance(
            text, (str, list)
        ), f"text must be str or list, got {type(text)}"
        assert isinstance(
            response_text, (str, list)
        ), f"response_text must be str or list, got {type(response_text)}"

        if isinstance(text, list):
            return [self._cat_text(t, t_) for t, t_ in _zip_strict(text, response_text)]
        else:
            return text + response_text

    def _generate_from_text(
        self,
        text: str | list[str] | NonTensorStack,
        sampling_params: SamplingParams,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate text from text input."""
        # Convert text to list format
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = text.tolist()

        assert isinstance(
            text, (str, list)
        ), f"text must be str or list, got {type(text)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        generate_kwargs = {"sampling_params": sampling_params}
        args = ()

        # Convert text to list format
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = text.tolist()

        if not self._remote_calls:
            request_output = self.model.generate(text, *args, **generate_kwargs)
        else:
            import ray

            request_output = ray.get(
                self.model.generate.remote(text, *args, **generate_kwargs)
            )

        request_output_tc = _RequestOutput_tc.from_request_output(request_output)

        # Extract response tokens and text
        outputs = (
            request_output_tc.outputs.view(-1)
            if self.num_samples is not None
            else request_output_tc.outputs
        )
        if self.pad_output:
            response_tokens_padded = outputs.view(-1).get(
                "token_ids",
                as_padded_tensor=self.pad_output,
                padding_value=self.padding_value,
                padding_side="right",
            )
        response_tokens_list = outputs.view(-1).get(
            "token_ids",
            as_list=True,
        )
        self._check_not_padded(response_tokens_list)
        if self.tokenizer is not None:
            response_text = self.tokenizer.batch_decode(
                response_tokens_list, skip_special_tokens=False
            )
        else:
            response_text = None

        # Build output TensorClass objects

        if self.return_masks:
            masks_obj = Masks._from_tensordict(out.empty())
            masks_obj.all_attention_mask = None
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

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
            with tokens_obj.view(-1) as tokens_obj_flat:
                tokens_obj_flat.prompt = (
                    None  # We don't have prompt tokens in this path
                )
                if self.pad_output:
                    tokens_obj_flat.response = response_tokens_padded
                    self._check_padded(response_tokens_padded)
                else:
                    tokens_obj_flat.response = response_tokens_list
                    self._check_not_padded(response_tokens_list)
                tokens_obj_flat.full = None  # we don't have prompt tokens in this path so no all_tokens either
            tokens_obj.padded = self.pad_output
            out.set("tokens", tokens_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            with log_probs_obj.view(-1) as log_probs_obj_flat:
                if self.pad_output:
                    log_probs_padded = outputs.get(
                        "logprobs",
                        as_padded_tensor=self.pad_output,
                        padding_value=self.padding_value,
                        padding_side="right",
                    )
                    self._check_padded(log_probs_padded)
                    log_probs_obj_flat.response = log_probs_padded
                    log_probs_obj_flat.full = log_probs_padded
                else:
                    log_probs_list = outputs.get(
                        "logprobs",
                        as_list=True,
                    )
                    self._check_not_padded(log_probs_list)
                    log_probs_obj_flat.response = log_probs_list
                    log_probs_obj_flat.full = log_probs_list
                log_probs_obj_flat.prompt = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

        return out

    def _logprobs_from_text(
        self,
        text: str | list[str] | NonTensorStack,
        sampling_params: SamplingParams,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Compute log-probs from text input."""
        # Convert text to list format
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            text = text.tolist()

        assert isinstance(
            text, (str, list)
        ), f"text must be str or list, got {type(text)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        # Tokenize the text
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is required for log-probs computation with text input"
            )

        # Tokenize the text
        tokenized_output = self.tokenizer(text, **self.tokenizer_kwargs)
        if self.pad_output:
            tokens_prompt_full_padded = tokenized_output["input_ids"]
            attention_mask_full_padded = tokenized_output["attention_mask"]
            tokens_prompt_full_list = self._to_list(
                tokens_prompt_full_padded, attention_mask_full_padded
            )
        else:
            tokens_prompt_full_unpadded = tokenized_output["input_ids"]
            tokens_prompt_full_list = self._to_list(tokens_prompt_full_unpadded, None)
            attention_mask_full_unpadded = tokenized_output["attention_mask"]
            attention_mask_full_unpadded = [
                am.bool()
                if isinstance(am, torch.Tensor)
                else torch.tensor(am, dtype=torch.bool)
                for am in attention_mask_full_unpadded
            ]
            self._check_not_padded(attention_mask_full_unpadded)

        # Convert to list format for vLLM
        generate_kwargs = {
            "sampling_params": sampling_params,
            "prompt_token_ids": tokens_prompt_full_list,
        }

        # Generate with vLLM to get prompt_logprobs
        if not self._remote_calls:
            request_output = self.model.generate(**generate_kwargs)
        else:
            import ray

            request_output = ray.get(self.model.generate.remote(**generate_kwargs))

        request_output_tc = _RequestOutput_tc.from_request_output(request_output)

        # Extract log-probs from prompt_logprobs
        if self.pad_output:
            # For padded case, use all prompt_logprobs
            log_probs_full_padded = request_output_tc.get(
                "prompt_logprobs",
                as_padded_tensor=True,
                padding_value=0,
                padding_side="left",
            )

            # Mask out padding
            log_probs_full_padded = torch.where(
                attention_mask_full_padded.bool(), log_probs_full_padded, 0.0
            )
        else:
            log_probs_full_unpadded = request_output_tc.get(
                "prompt_logprobs", as_list=True
            )
            self._check_not_padded(log_probs_full_unpadded)

        # Build output TensorClass objects

        if self.return_masks:
            masks_obj = Masks._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                self._check_padded(attention_mask_full_padded)
                masks_obj.all_attention_mask = attention_mask_full_padded.bool()
            else:
                self._check_not_padded(attention_mask_full_unpadded)
                masks_obj.all_attention_mask = attention_mask_full_unpadded
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

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
                self._check_padded(tokens_prompt_full_padded)
                tokens_obj.full = tokens_obj.prompt = tokens_prompt_full_padded
            else:
                self._check_not_padded(tokens_prompt_full_unpadded)
                tokens_obj.full = tokens_obj.prompt = tokens_prompt_full_unpadded
            tokens_obj.response = None
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                self._check_padded(log_probs_full_padded)
                log_probs_obj.full = log_probs_obj.prompt = log_probs_full_padded
            else:
                self._check_not_padded(log_probs_full_unpadded)
                log_probs_obj.full = log_probs_obj.prompt = log_probs_full_unpadded
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

        return out

    def _cat_tensors(
        self,
        tokens: list[torch.Tensor] | torch.Tensor,
        response_tokens: list[torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Concatenate tokens and response tokens."""
        if isinstance(tokens, list) or isinstance(response_tokens, list):
            return [
                self._cat_tensors(t, t_)
                for t, t_ in _zip_strict(tokens, response_tokens)
            ]
        else:
            return torch.cat([tokens, response_tokens], dim=-1)

    def _generate_from_tokens(
        self,
        tokens_prompt_unpadded: list[torch.Tensor] | None,
        tokens_prompt_padded: torch.Tensor | None,
        sampling_params: SamplingParams,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate text from tokens input."""
        assert isinstance(
            tokens_prompt_padded, (torch.Tensor, type(None))
        ), f"tokens_prompt_padded must be torch.Tensor or None, got {type(tokens_prompt_padded)}"
        assert isinstance(
            tokens_prompt_unpadded, (list, type(None))
        ), f"tokens_prompt_unpadded must be list or None, got {type(tokens_prompt_unpadded)}"
        assert isinstance(
            sampling_params, SamplingParams
        ), f"sampling_params must be SamplingParams, got {type(sampling_params)}"
        assert isinstance(
            out, TensorDictBase
        ), f"out must be TensorDictBase, got {type(out)}"

        generate_kwargs = {"sampling_params": sampling_params}
        args = ()

        if tokens_prompt_unpadded is None:
            # TODO: To be on the safe side, we may do this even in the unpadded case since we're not sure
            #  the user passed an unpadded tensor in the first place.
            tokens_prompt_list = self._to_list(
                tokens_prompt_padded, tokens_prompt_padded != self.padding_value
            )
        else:
            tokens_prompt_list = self._to_list(tokens_prompt_unpadded, None)
        generate_kwargs.update({"prompt_token_ids": tokens_prompt_list})

        if not self._remote_calls:
            request_output = self.model.generate(*args, **generate_kwargs)
        else:
            import ray

            request_output = ray.get(
                self.model.generate.remote(*args, **generate_kwargs)
            )

        request_output_tc = _RequestOutput_tc.from_request_output(request_output)

        # Extract response tokens and text
        outputs = (
            request_output_tc.outputs.view(-1)
            if self.num_samples is not None
            else request_output_tc.outputs
        )
        if self.pad_output:
            tokens_response_padded = outputs.get(
                "token_ids",
                as_padded_tensor=self.pad_output,
                padding_value=self.padding_value,
                padding_side="right",
            )
            self._check_padded(tokens_response_padded)
        tokens_response_unpadded = outputs.get(
            "token_ids",
            as_list=True,
        )
        self._check_not_padded(tokens_response_unpadded)
        if self.tokenizer is not None:
            response_text = self.tokenizer.batch_decode(
                tokens_response_unpadded, skip_special_tokens=False
            )
        else:
            response_text = None

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
            if self.pad_output:
                self._check_padded(tokens_response_padded)
                self._check_padded(tokens_prompt_padded)
            else:
                self._check_not_padded(tokens_response_unpadded)
                self._check_not_padded(tokens_prompt_unpadded)

            if self.num_samples is not None:
                # replicate tokens
                for i in range(self.num_samples):
                    tokens_obj[:, i].prompt = (
                        tokens_prompt_unpadded
                        if not self.pad_output
                        else tokens_prompt_padded
                    )
            else:
                tokens_obj.prompt = (
                    tokens_prompt_unpadded
                    if not self.pad_output
                    else tokens_prompt_padded
                )
            with tokens_obj.view(-1) as tokens_obj_flat:
                if self.pad_output:
                    tokens_obj_flat.response = tokens_response_padded
                    tokens_full_padded = self._cat_tensors(
                        tokens_obj_flat.prompt, tokens_response_padded
                    )
                    tokens_obj_flat.full = tokens_full_padded
                else:
                    tokens_obj_flat.response = tokens_response_unpadded
                    tokens_full_unpadded = self._cat_tensors(
                        tokens_obj_flat.get("prompt", as_list=True),
                        tokens_response_unpadded,
                    )
                    tokens_obj_flat.full = tokens_full_unpadded
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_masks:
            masks_obj = Masks._from_tensordict(out.empty())
            # self.return_tokens must be True
            if self.pad_output:
                # Get "real" attention masks
                full_attention_mask_padded = (
                    tokens_obj.get("full") != self.padding_value
                )
                masks_obj.all_attention_mask = full_attention_mask_padded.bool()
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
            out.set("masks", masks_obj)

        if self.return_log_probs:
            if self.pad_output:
                log_probs_padded = outputs.get(
                    "logprobs",
                    as_padded_tensor=self.pad_output,
                    padding_value=self.padding_value,
                    padding_side="right",
                )
            else:
                log_probs_list = outputs.get(
                    "logprobs",
                    as_list=True,
                )
                self._check_not_padded(log_probs_list)
            if self.num_samples is None:
                # TODO: this is not correct, we should use the prompt_logprobs
                #  but they're not returned by vLLM
                if self.pad_output:
                    prompt_logprobs_padded = request_output_tc.get(
                        "prompt_logprobs",
                        as_padded_tensor=self.pad_output,
                        padding_value=self.padding_value,
                        padding_side="right",
                    )
                else:
                    prompt_logprobs_list = request_output_tc.get(
                        "prompt_logprobs",
                        as_list=True,
                    )
                    self._check_not_padded(prompt_logprobs_list)
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            if self.pad_output:
                self._check_padded(log_probs_padded)
                if self.num_samples is None:
                    self._check_padded(prompt_logprobs_padded)
                    log_probs_obj.prompt = prompt_logprobs_padded
            else:
                self._check_not_padded(log_probs_list)
                if self.num_samples is None:
                    self._check_not_padded(prompt_logprobs_list)
                    log_probs_obj.prompt = prompt_logprobs_list
            with log_probs_obj.view(-1) as log_probs_obj_flat:
                log_probs_obj_flat.response = (
                    log_probs_padded if self.pad_output else log_probs_list
                )
                if self.num_samples is None:
                    if self.pad_output:
                        log_probs_obj_flat.full = self._cat_tensors(
                            log_probs_obj_flat.prompt, log_probs_padded
                        )
                    else:
                        log_probs_obj_flat.full = self._cat_tensors(
                            log_probs_obj_flat.get("prompt", as_list=True),
                            log_probs_list,
                        )
                else:
                    log_probs_obj_flat.full = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)
        return out

    def _logprobs_from_tokens(
        self,
        *,
        response_struct: TensorDictBase | None = None,
        tokens_full_unpadded: list[torch.Tensor] | None = None,
        tokens_full_padded: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
        out: TensorDictBase | None = None,
    ) -> TensorDictBase:
        """Compute log-probs from tokens input."""
        assert isinstance(
            response_struct, (TensorDictBase, type(None))
        ), f"response_struct must be TensorDictBase or None, got {type(response_struct)}"
        assert isinstance(
            tokens_full_unpadded, (list, type(None))
        ), f"tokens_full_unpadded must be list or None, got {type(tokens_full_unpadded)}"
        assert isinstance(
            tokens_full_padded, (torch.Tensor, type(None))
        ), f"tokens_full_padded must be torch.Tensor or None, got {type(tokens_full_padded)}"
        assert isinstance(
            sampling_params, (SamplingParams, type(None))
        ), f"sampling_params must be SamplingParams or None, got {type(sampling_params)}"
        assert isinstance(
            out, (TensorDictBase, type(None))
        ), f"out must be TensorDictBase or None, got {type(out)}"

        # Convert to list format for vLLM
        if response_struct is not None:
            tokens_full_padded = response_struct.get(
                "input_ids",
                as_padded_tensor=True,
                padding_value=self.padding_value,
                padding_side="left",
            )
            attention_mask_full_padded = response_struct.get(
                "attention_mask",
                as_padded_tensor=True,
                padding_value=False,
                padding_side="left",
            ).bool()
            attention_mask_full_unpadded = _unpad_tensors(
                attention_mask_full_padded, attention_mask_full_padded, as_nested=False
            )
        elif tokens_full_unpadded is not None:
            tokens_full_padded = pad_sequence(
                tokens_full_unpadded,
                padding_value=self.padding_value,
                batch_first=True,
                padding_side="left",
            )
            attention_mask_full_unpadded = [
                t != self.padding_value for t in tokens_full_unpadded
            ]
            attention_mask_full_padded = pad_sequence(
                attention_mask_full_unpadded,
                padding_value=False,
                batch_first=True,
                padding_side="left",
            )
        elif tokens_full_padded is not None:
            attention_mask_full_padded = tokens_full_padded != self.padding_value
        else:
            raise ValueError("Either response_struct or tokens must be provided")

        assert isinstance(tokens_full_padded, torch.Tensor)
        assert isinstance(attention_mask_full_padded, torch.Tensor)
        if tokens_full_unpadded is None:
            tokens_full_list = self._to_list(
                tokens_full_padded, attention_mask_full_padded
            )
        else:
            tokens_full_list = self._to_list(tokens_full_unpadded, None)

        generate_kwargs = {
            "sampling_params": sampling_params,
            "prompt_token_ids": tokens_full_list,
        }

        # Generate with vLLM to get prompt_logprobs
        if not self._remote_calls:
            tokens_out_stuct = self.model.generate(**generate_kwargs)
        else:
            import ray

            tokens_out_stuct = ray.get(self.model.generate.remote(**generate_kwargs))

        request_output_tc = _RequestOutput_tc.from_request_output(tokens_out_stuct)

        # Extract log-probs from prompt_logprobs
        if self.pad_output:
            # For padded case, use all prompt_logprobs
            log_probs_full_padded = request_output_tc.get(
                "prompt_logprobs",
                as_padded_tensor=True,
                padding_value=0,
                padding_side="left",
            )

            # Mask out padding
            attention_mask_full_padded = tokens_full_padded != self.padding_value
            log_probs_full_padded = torch.where(
                attention_mask_full_padded, log_probs_full_padded, 0.0
            )
        else:
            # For unpadded case, extract from each sequence
            log_probs_full_unpadded = request_output_tc.get(
                "prompt_logprobs", as_list=True
            )

        assistant_mask_full_padded = None
        if response_struct is not None:
            assistant_mask_full_padded = response_struct.get(
                "assistant_masks",
                as_padded_tensor=True,
                padding_side="left",
                padding_value=0,
            )
        if assistant_mask_full_padded is not None:
            assistant_mask_full_padded = assistant_mask_full_padded.bool()
            if not self.pad_output:
                assistant_mask_full_unpadded = _unpad_tensors(
                    assistant_mask_full_padded,
                    attention_mask_full_padded,
                    as_nested=False,
                )
            else:
                assistant_mask_full_unpadded = None
        else:
            assistant_mask_full_unpadded = None

        if self.return_masks:
            masks_obj = Masks._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                masks_obj.all_attention_mask = attention_mask_full_padded.bool()
                if assistant_mask_full_padded is not None:
                    masks_obj.all_assistant_mask = assistant_mask_full_padded
            else:
                masks_obj.all_attention_mask = attention_mask_full_unpadded
                if assistant_mask_full_unpadded is not None:
                    masks_obj.all_assistant_mask = assistant_mask_full_unpadded
            masks_obj.padded = MetaData(self.pad_output)
            out.set("masks", masks_obj)

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
            if self.pad_output:
                self._check_padded(tokens_full_padded)
                tokens_obj.full = tokens_obj.prompt = tokens_full_padded
            else:
                tokens_obj.full = tokens_obj.prompt = tokens_full_unpadded
            tokens_obj.response = None
            tokens_obj.padded = MetaData(self.pad_output)
            out.set("tokens", tokens_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                log_probs_obj.full = log_probs_obj.prompt = log_probs_full_padded
            else:
                self._check_not_padded(log_probs_full_unpadded)
                log_probs_obj.full = log_probs_obj.prompt = log_probs_full_unpadded
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set("log_probs", log_probs_obj)

        return out

    def _to_list(
        self,
        tokens_padded: torch.Tensor | list[torch.Tensor],
        attention_mask_padded: torch.Tensor | None,
    ) -> list[list[int]]:
        """Converts a tensor of integers into a masked list (of lists) of integers."""
        if isinstance(tokens_padded, torch.Tensor):
            parent = []
            queue = collections.deque()
            if attention_mask_padded is None:
                attention_mask_padded = torch.ones_like(tokens_padded)
            queue.append((tokens_padded, attention_mask_padded.bool(), parent))
            while queue:
                token_tensor, attention_mask_bool, _parent = queue.popleft()
                if token_tensor.ndim == 1:
                    _parent.extend(token_tensor[attention_mask_bool].tolist())
                else:
                    _parent.extend([[] for _ in range(token_tensor.shape[0])])
                    queue.extend(
                        [
                            (t, m, local_parent)
                            for t, m, local_parent in zip(
                                token_tensor, attention_mask_bool, _parent
                            )
                        ]
                    )
            tokens_list = parent
        elif isinstance(tokens_padded, list):
            parent = []
            queue = collections.deque()
            queue.append((tokens_padded, parent))
            while queue:
                tokens_list, _parent = queue.popleft()
                if isinstance(tokens_list, list) and isinstance(
                    tokens_list[0], (list, torch.Tensor)
                ):
                    _parent.extend([[] for _ in tokens_list])
                    queue.extend(
                        [
                            (t, local_parent)
                            for t, local_parent in zip(tokens_list, _parent)
                        ]
                    )
                    continue
                elif isinstance(tokens_list, torch.Tensor):
                    tokens_list = tokens_list.tolist()
                _parent.extend(tokens_list)
            tokens_list = parent

        return tokens_list

    @_classproperty
    def CompletionOutput_tc(cls):
        if vllm is None:
            raise ImportError("vllm is required for CompletionOutput_tc")

        if hasattr(cls, "_CompletionOutput_tc"):
            return cls._CompletionOutput_tc
        CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)  # type: ignore
        cls._CompletionOutput_tc = CompletionOutput_tc
        return CompletionOutput_tc


class _RequestOutput_tc(TensorClass["nocast"]):
    """TensorClass wrapper for vLLM RequestOutput."""

    request_id: str
    prompt: str
    prompt_token_ids: torch.Tensor
    prompt_logprobs: torch.Tensor
    outputs: list  # type: ignore
    finished: str
    metrics: str
    lora_request: str
    encoder_prompt: str
    encoder_prompt_token_ids: str
    num_cached_tokens: torch.Tensor

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
                # Check if we can stack the outputs (they should have the same shape)
                try:
                    self.outputs = lazy_stack(outputs)
                except RuntimeError:
                    # If stacking fails (different sizes), keep as list
                    self.outputs = outputs

    @classmethod
    def from_request_output(
        cls, requests: RequestOutput | list[RequestOutput]
    ) -> _RequestOutput_tc | list[_RequestOutput_tc]:
        """Create _RequestOutput_tc from vLLM RequestOutput."""
        # Type assertions
        assert isinstance(
            requests, (RequestOutput, list)
        ), f"requests must be RequestOutput or list, got {type(requests)}"

        # Check if we can stack the outputs
        try:
            out = lazy_stack(
                [
                    cls(
                        request_id=request.request_id,
                        prompt=request.prompt,
                        prompt_token_ids=torch.as_tensor(request.prompt_token_ids),
                        prompt_logprobs=torch.tensor(
                            [
                                v[int(tid)].logprob if v is not None else 0.0
                                for v, tid in _zip_strict(
                                    request.prompt_logprobs, request.prompt_token_ids
                                )
                            ]
                        )
                        if request.prompt_logprobs is not None
                        else torch.tensor([]),
                        outputs=request.outputs,
                        finished=request.finished,
                        metrics=request.metrics,
                        lora_request=request.lora_request,
                        encoder_prompt=request.encoder_prompt,
                        encoder_prompt_token_ids=request.encoder_prompt_token_ids,
                        num_cached_tokens=torch.as_tensor(request.num_cached_tokens),
                    )
                    for request in requests
                ]
            )
            return out
        except RuntimeError:
            # If stacking fails, return a list of individual _RequestOutput_tc objects
            return [
                cls(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    prompt_token_ids=torch.as_tensor(request.prompt_token_ids),
                    prompt_logprobs=torch.tensor(
                        [
                            v[int(tid)].logprob if v is not None else 0.0
                            for v, tid in _zip_strict(
                                request.prompt_logprobs, request.prompt_token_ids
                            )
                        ]
                    )
                    if request.prompt_logprobs is not None
                    else torch.tensor([]),
                    outputs=request.outputs,
                    finished=request.finished,
                    metrics=request.metrics,
                    lora_request=request.lora_request,
                    encoder_prompt=request.encoder_prompt,
                    encoder_prompt_token_ids=request.encoder_prompt_token_ids,
                    num_cached_tokens=torch.as_tensor(request.num_cached_tokens),
                )
                for request in requests
            ]
