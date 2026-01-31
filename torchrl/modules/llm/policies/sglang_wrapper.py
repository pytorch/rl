# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SGLang wrapper for TorchRL LLM policies.

This module provides SGLangWrapper, a policy wrapper that interfaces with
SGLang servers for text generation in RL training workflows.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    MetaData,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import NestedKey

from torchrl.modules.llm.policies.common import (
    _batching,
    ChatHistory,
    LLMWrapperBase,
    LogProbs,
    Masks,
    Text,
    Tokens,
)

_HAS_SGLANG = importlib.util.find_spec("sglang") is not None
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

if TYPE_CHECKING:
    from torchrl.modules.llm.backends.sglang import AsyncSGLang


def _require_transformers() -> None:
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for SGLangWrapper. Please install it with `pip install transformers`."
        )


class SGLangWrapper(LLMWrapperBase):
    """A wrapper class for SGLang models, providing a consistent interface for text generation.

    This class is a subclass of :class:`~torchrl.modules.llm.policies.LLMWrapperBase` and provides
    a unified API for handling different input modalities (history, text, tokens) with consistent
    output structure using :class:`~tensordict.TensorClass` objects.

    The wrapper interfaces with SGLang servers via HTTP for generation and uses the same
    output structures as vLLMWrapper for compatibility.

    Args:
        model (AsyncSGLang | str): The SGLang backend to wrap.
            - If a string URL, connects to an existing SGLang server
            - If an AsyncSGLang instance, uses it directly

    Keyword Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | str | None, optional):
            The tokenizer to use for encoding and decoding text. If `None`, attempts to load
            from the model. Defaults to `None`.
        input_mode (str, optional): The input modality to use. Must be one of `"history"`,
            `"text"`, or `"tokens"`. Defaults to `"history"`.
        input_key (str | None, optional): The key for the input data. If `None`, defaults based
            on input_mode and generate flag.
        generate (bool, optional): Whether to enable text generation. Defaults to `True`.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `True`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to generation.
            Supports standardized parameters like `max_new_tokens`, `temperature`, `top_p`, etc.
        pad_output (bool, optional): Whether to pad output sequences. Defaults to `False`.
        inplace (Literal[True, False, "empty"] | None, optional): In-place operation mode.
        device (torch.device | None, optional): Device for computation.
        num_samples (int | None, optional): Number of samples to generate.
        chat_template_name (str | None, optional): Chat template name for history mode.
        chat_template (str | None, optional): Custom chat template string.
        text_key (NestedKey | None, optional): Key for Text output. Defaults to `"text"`.
        tokens_key (NestedKey | None, optional): Key for Tokens output. Defaults to `"tokens"`.
        masks_key (NestedKey | None, optional): Key for Masks output. Defaults to `"masks"`.
        log_probs_key (NestedKey | None, optional): Key for LogProbs output. Defaults to `"log_probs"`.
        history_key (NestedKey | None, optional): Key for ChatHistory output. Defaults to `"history"`.

    Example:
        >>> from torchrl.modules.llm.backends.sglang import AsyncSGLang
        >>> from torchrl.modules.llm.policies import SGLangWrapper
        >>>
        >>> # Connect to existing server
        >>> backend = AsyncSGLang.connect("http://localhost:30000")
        >>> wrapper = SGLangWrapper(backend, input_mode="text", generate=True)
        >>>
        >>> # Or launch managed server
        >>> backend = AsyncSGLang.from_pretrained("Qwen/Qwen2.5-3B")
        >>> wrapper = SGLangWrapper(backend, input_mode="history")
        >>>
        >>> # Generate text
        >>> from tensordict import TensorDict
        >>> td = TensorDict({"text": {"prompt": ["Hello, how are you?"]}}, batch_size=[1])
        >>> result = wrapper(td)
        >>> print(result["text"]["response"])

    .. seealso::
        - :class:`~torchrl.modules.llm.policies.LLMWrapperBase`
        - :class:`~torchrl.modules.llm.policies.vLLMWrapper`
        - :class:`~torchrl.modules.llm.backends.sglang.AsyncSGLang`
    """

    def __init__(
        self,
        model: AsyncSGLang | str,
        *,
        tokenizer: callable | str | None = None,
        input_mode: str = "history",
        input_key: NestedKey | None = None,
        generate: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        inplace: Literal[True, False, "empty"] | None = None,
        device: torch.device | None = None,
        layout: torch.layout | None = None,
        num_samples: int | None = None,
        chat_template_name: str | None = None,
        chat_template: str | None = None,
        return_log_probs: bool | None = None,
        history_key: NestedKey | None = "history",
        text_key: NestedKey | None = "text",
        tokens_key: NestedKey | None = "tokens",
        masks_key: NestedKey | None = "masks",
        log_probs_key: NestedKey | None = "log_probs",
        batching: bool | None = None,
        min_batch_size: int | None = None,
        max_batch_size: int | None = None,
        batching_timeout: float = 10.0,
    ):
        super().__init__()

        _require_transformers()

        # Handle model initialization
        if isinstance(model, str):
            # Assume it's a server URL
            from torchrl.modules.llm.backends.sglang import AsyncSGLang

            model = AsyncSGLang.connect(model)

        # Validate input_mode
        if input_mode not in ["history", "text", "tokens"]:
            raise ValueError(
                f"input_mode must be one of 'history', 'text', 'tokens'. Got '{input_mode}'"
            )

        self.model = model
        self.input_mode = input_mode
        self.generate = generate
        self.pad_output = pad_output
        self._device = device
        self.layout = layout if not pad_output else None
        self.num_samples = num_samples
        self.chat_template_name = chat_template_name
        self.chat_template = chat_template

        # Batching setup
        if batching and min_batch_size is None:
            min_batch_size = 1
        elif (min_batch_size is not None or max_batch_size is not None) and (
            batching is False
        ):
            raise ValueError(
                "min_batch_size and max_batch_size must be None if batching is False."
            )
        self._min_batch_size = min_batch_size
        self._max_batch_size = max_batch_size
        self._batching_timeout = batching_timeout
        self._batch_queue = []
        self._futures = []

        if self.batching:
            import threading

            self._batching_lock = threading.Lock()
        else:
            self._batching_lock = None

        # Return flags
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

        # Output keys
        self.history_key = history_key
        self.log_probs_key = log_probs_key
        self.masks_key = masks_key
        self.text_key = text_key
        self.tokens_key = tokens_key

        # Set input keys based on mode and generate parameter
        if input_mode == "history":
            self.in_keys = (
                [("history", "prompt") if input_key is None else input_key]
                if generate
                else [("history", "full") if input_key is None else input_key]
            )
        elif input_mode == "text":
            self.in_keys = (
                [("text", "prompt") if input_key is None else input_key]
                if generate
                else [("text", "full") if input_key is None else input_key]
            )
        elif input_mode == "tokens":
            self.in_keys = (
                [("tokens", "prompt") if input_key is None else input_key]
                if generate
                else [("tokens", "full") if input_key is None else input_key]
            )
        self.input_key = self.in_keys[0]

        # Set output keys
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
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            # Try to get from model info
            model_path = getattr(model, "_model_path", None)
            if model_path:
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception as e:
                    warnings.warn(f"Could not load tokenizer from {model_path}: {e}")

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if (
                not hasattr(self.tokenizer, "pad_token")
                or self.tokenizer.pad_token is None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.padding_value = self.tokenizer(self.tokenizer.pad_token)["input_ids"][
                0
            ]
        else:
            self.padding_value = None

        # Tokenizer kwargs
        if not tokenizer_kwargs:
            tokenizer_kwargs = {}
        tokenizer_kwargs.setdefault("return_attention_mask", True)
        tokenizer_kwargs.setdefault("padding", self.pad_output)
        tokenizer_kwargs.setdefault("padding_side", "left")
        self.tokenizer_kwargs = tokenizer_kwargs

        # Generation kwargs - standardize and convert to SGLang format
        if generate_kwargs is None:
            generate_kwargs = {}
        else:
            generate_kwargs = dict(generate_kwargs)

        generate_kwargs = self._standardize_generate_kwargs(generate_kwargs)
        self.generate_kwargs = self._convert_to_sglang_params(generate_kwargs)

        # Inplace handling
        if num_samples is not None and num_samples > 1:
            if inplace in (True, "empty"):
                raise ValueError(
                    "inplace must be False (or None) when generating more than one sample."
                )
            inplace = False
        elif inplace is None:
            inplace = True
        self.inplace = inplace

    def _convert_to_sglang_params(self, generate_kwargs: dict) -> dict:
        """Convert standardized parameters to SGLang format."""
        sglang_params = {}

        param_mapping = {
            "max_new_tokens": "max_new_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "repetition_penalty": "repetition_penalty",
            "stop_sequences": "stop",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
        }

        for std_name, sglang_name in param_mapping.items():
            if std_name in generate_kwargs:
                sglang_params[sglang_name] = generate_kwargs[std_name]

        # Handle do_sample
        if generate_kwargs.get("do_sample") is False:
            sglang_params["temperature"] = 0.0

        return sglang_params

    @set_list_to_stack(True)
    @_batching
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        tensordict_out: TensorDictBase | None = None,
        logits_only: bool = False,
        **kwargs,
    ) -> TensorDictBase:
        """Forward pass for the SGLang policy.

        Args:
            tensordict: Input tensordict containing prompts
            tensordict_out: Optional output tensordict
            logits_only: Whether to return only logits (not supported for SGLang)
            **kwargs: Additional generation parameters

        Returns:
            TensorDictBase with generation results
        """
        tensordict_orig = tensordict

        if not tensordict.ndim:
            return self.forward(lazy_stack([tensordict]), logits_only=logits_only)[0]
        elif tensordict.ndim > 1:
            return self.forward(tensordict.reshape(-1), logits_only=logits_only).view(
                tensordict.shape
            )

        if not isinstance(tensordict, LazyStackedTensorDict):
            tensordict = tensordict.to_lazystack(0)

        # Prepare output
        out = TensorDict(
            device=tensordict.device, batch_size=tensordict.batch_size
        ).to_lazystack(0)

        if self.input_mode == "history":
            out = self._generate_from_history(tensordict, out)
        elif self.input_mode == "text":
            out = self._generate_from_text(tensordict, out)
        elif self.input_mode == "tokens":
            out = self._generate_from_tokens(tensordict, out)

        # Handle inplace
        if tensordict_out is None:
            if self.inplace is True:
                tensordict_out = tensordict_orig
            elif self.inplace is False:
                tensordict_out = out
            elif self.inplace == "empty":
                tensordict_out = tensordict.empty()

        if tensordict_out is not None and tensordict_out is not out:
            result = tensordict_out.exclude(*self.out_keys, inplace=True)
            result.update(out, keys_to_update=self.out_keys)
        else:
            result = out

        return result

    def _generate_from_history(
        self,
        tensordict: TensorDictBase,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate from history input mode."""
        from torchrl.data.llm import History

        history = tensordict.get(self.input_key)
        if not isinstance(history, History):
            raise TypeError(
                f"Expected History object for '{self.input_key}', got {type(history)}"
            )

        # Apply chat template to get text prompts
        tokenizer_kwargs = {}
        if self.chat_template_name is not None:
            tokenizer_kwargs["chat_template_name"] = self.chat_template_name
        if self.chat_template is not None:
            tokenizer_kwargs["chat_template"] = self.chat_template
        tokenizer_kwargs["add_generation_prompt"] = True

        text_prompts = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )

        # Generate via SGLang
        results = self.model.generate(
            text_prompts,
            sampling_params=self.generate_kwargs,
            return_logprobs=self.return_log_probs,
        )

        # Process results
        return self._process_generation_results(
            results, text_prompts, out, history=history
        )

    def _generate_from_text(
        self,
        tensordict: TensorDictBase,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate from text input mode."""
        text_prompts = tensordict.get(self.input_key)

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        elif not isinstance(text_prompts, list):
            text_prompts = list(text_prompts)

        # Generate via SGLang
        results = self.model.generate(
            text_prompts,
            sampling_params=self.generate_kwargs,
            return_logprobs=self.return_log_probs,
        )

        return self._process_generation_results(results, text_prompts, out)

    def _generate_from_tokens(
        self,
        tensordict: TensorDictBase,
        out: TensorDictBase,
    ) -> TensorDictBase:
        """Generate from tokens input mode."""
        tokens_prompt = tensordict.get(self.input_key, as_list=True)

        # Decode tokens to text for SGLang
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for tokens input mode")

        text_prompts = self.tokenizer.batch_decode(
            tokens_prompt, skip_special_tokens=False
        )

        # Generate via SGLang
        results = self.model.generate(
            text_prompts,
            sampling_params=self.generate_kwargs,
            return_logprobs=self.return_log_probs,
        )

        return self._process_generation_results(
            results, text_prompts, out, tokens_prompt=tokens_prompt
        )

    def _process_generation_results(
        self,
        results: list[dict[str, Any]],
        text_prompts: list[str],
        out: TensorDictBase,
        history: Any = None,
        tokens_prompt: list | None = None,
    ) -> TensorDictBase:
        """Process SGLang generation results into output tensordicts."""
        # Extract generated text and tokens
        response_texts = []
        response_token_ids = []
        log_probs_list = []

        for result in results:
            response_texts.append(result.get("text", ""))
            response_token_ids.append(
                torch.tensor(result.get("output_ids", []), dtype=torch.long)
            )
            # Extract log probs if available
            meta_info = result.get("meta_info", {})
            if "logprobs" in meta_info:
                log_probs_list.append(torch.tensor(meta_info["logprobs"]))
            else:
                log_probs_list.append(None)

        # Build Text output
        if self.return_text:
            text_obj = Text._from_tensordict(out.empty())
            text_obj.prompt = text_prompts
            text_obj.response = response_texts
            text_obj.full = [p + r for p, r in zip(text_prompts, response_texts)]
            out.set(self.text_key, text_obj)

        # Build Tokens output
        if self.return_tokens:
            tokens_obj = Tokens._from_tensordict(out.empty())
            if tokens_prompt is not None:
                tokens_obj.prompt = tokens_prompt
            tokens_obj.response = response_token_ids
            tokens_obj.padded = MetaData(self.pad_output)
            out.set(self.tokens_key, tokens_obj)

        # Build Masks output
        if self.return_masks:
            masks_obj = Masks._from_tensordict(out.empty())
            masks_obj.all_attention_mask = None
            masks_obj.all_assistant_mask = None
            masks_obj.padded = MetaData(self.pad_output)
            out.set(self.masks_key, masks_obj)

        # Build LogProbs output
        if self.return_log_probs and any(lp is not None for lp in log_probs_list):
            log_probs_obj = LogProbs._from_tensordict(out.empty())
            log_probs_obj.response = log_probs_list
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set(self.log_probs_key, log_probs_obj)

        # Build History output
        if self.return_history and history is not None:
            from torchrl.data.llm import History

            chat_history = ChatHistory._from_tensordict(out.empty())
            chat_history.prompt = history

            # Parse response text back to history
            response_histories = []
            for resp_text in response_texts:
                resp_history = History.from_text(
                    [resp_text],
                    chat_template_name=self.chat_template_name,
                    tokenizer=self.tokenizer,
                )
                response_histories.append(resp_history)

            chat_history.response = lazy_stack(response_histories)
            chat_history.full = history.extend(
                chat_history.response, inplace=False, dim=-1
            )
            out.set(self.history_key, chat_history)

        return out

    def get_new_version(self, **kwargs):
        """Returns a new version of the module with altered parameters."""
        constructor_kwargs = {
            "model": kwargs.get("model", self.model),
            "tokenizer": kwargs.get("tokenizer", self.tokenizer),
            "input_mode": kwargs.get("input_mode", self.input_mode),
            "generate": kwargs.get("generate", self.generate),
            "generate_kwargs": kwargs.get("generate_kwargs", self.generate_kwargs),
            "pad_output": kwargs.get("pad_output", self.pad_output),
            "inplace": kwargs.get("inplace", self.inplace),
            "device": kwargs.get("device", self._device),
            "layout": kwargs.get("layout", self.layout),
            "num_samples": kwargs.get("num_samples", self.num_samples),
            "chat_template_name": kwargs.get(
                "chat_template_name", self.chat_template_name
            ),
            "chat_template": kwargs.get("chat_template", self.chat_template),
            "return_log_probs": kwargs.get("return_log_probs", self.return_log_probs),
            "history_key": kwargs.get("history_key", self.history_key),
            "text_key": kwargs.get("text_key", self.text_key),
            "tokens_key": kwargs.get("tokens_key", self.tokens_key),
            "masks_key": kwargs.get("masks_key", self.masks_key),
            "log_probs_key": kwargs.get("log_probs_key", self.log_probs_key),
        }
        return type(self)(**constructor_kwargs)

    def get_dist(self, *args, **kwargs):
        """Get distribution from logits/log-probs.

        SGLang does not return logits, so this method is not supported.
        """
        raise NotImplementedError(
            "SGLang does not return logits, so get_dist is not supported"
        )
