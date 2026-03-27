# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections

import importlib.util
import threading
import warnings
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    MetaData,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.tensorclass import from_dataclass, TensorClass
from tensordict.utils import _zip_strict, NestedKey
from torch import distributions as D
from torch.nn.utils.rnn import pad_sequence

from torchrl.envs.utils import _classproperty
from torchrl.modules.llm.policies.common import (
    _batching,
    _extract_responses_from_full_histories,
    ChatHistory,
    LLMWrapperBase,
    LogProbs,
    Masks,
    Text,
    Tokens,
)
from torchrl.modules.utils.utils import _unpad_tensors


_HAS_VLLM = importlib.util.find_spec("vllm") is not None
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt  # type: ignore[import-not-found]
    from vllm.outputs import RequestOutput  # type: ignore[import-not-found]
    from vllm.sampling_params import SamplingParams  # type: ignore[import-not-found]
elif _HAS_VLLM:
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        # Fallback for older vLLM versions
        TokensPrompt = None
else:
    SamplingParams = None  # Will error at usage if vLLM not available
    RequestOutput = None
    TokensPrompt = None


def _require_transformers() -> None:
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for vLLMWrapper. Please install it with `pip install transformers`."
        )


def _require_vllm():
    """Import vLLM lazily.

    We intentionally avoid importing vLLM at module import time because importing vLLM can
    load native extensions that may hard-crash the interpreter on some platforms.
    """
    if not _HAS_VLLM:
        raise ImportError(
            "vllm is required for vLLMWrapper. Please install it with `pip install vllm`."
        )
    import vllm as _vllm  # local import is intentional / required

    return _vllm


# Import async vLLM engines


class vLLMWrapper(LLMWrapperBase):
    """A wrapper class for vLLM models, providing a consistent interface for text generation and log probability computation.

    This class is a subclass of :class:`~torchrl.modules.llm.policies.LLMWrapperBase` and provides a unified API for handling different input
    modalities (history, text, tokens) with consistent output structure using :class:`~tensordict.TensorClass` objects.

    The wrapper supports both synchronous (vllm.LLM) and asynchronous (:class:`~torchrl.modules.llm.backends.AsyncVLLM`) vLLM engines.

    .. note::
        **Recommended: Use AsyncVLLM for better performance**

        For distributed inference and better resource utilization, we recommend using
        :class:`~torchrl.modules.llm.backends.AsyncVLLM` instead of the synchronous vllm.LLM:

        >>> from torchrl.modules.llm.backends import AsyncVLLM
        >>> from torchrl.modules.llm import vLLMWrapper
        >>>
        >>> # Recommended approach
        >>> async_engine = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B", num_replicas=2)
        >>> wrapper = vLLMWrapper(async_engine, input_mode="history", generate=True)

        AsyncVLLM provides:
        - Better GPU utilization through Ray-based distribution
        - Multiple replicas for higher throughput
        - Native vLLM batching for optimal performance
        - Automatic resource management and cleanup

    Args:
        model (vllm.LLM | AsyncVLLM | Ray Actor | str): The vLLM model to wrap.
            - If a string, it will be converted to an AsyncVLLM instance (recommended)
            - If a vllm.LLM instance, uses synchronous generation via `model.generate()`
            - If an AsyncVLLM instance, uses async generation via `model.generate()`
            - If a Ray actor with generate method, uses remote calls via `ray.get(model.generate.remote())`

    Keyword Args:
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer | str | None, optional): The tokenizer to use for encoding and decoding text.
            If `None`, the tokenizer associated with the model will be used. If a string, it will be passed to `transformers.AutoTokenizer.from_pretrained`.
            Defaults to `None`.
        input_mode (str, optional): The input modality to use. Must be one of `"history"`, `"text"`, or `"tokens"`. Defaults to `"history"`.
        input_key (str | None, optional): The key for the input data. If `None`, defaults to
            - `("history", "prompt")` for `"history"` when `generate=True`, `("history", "full")` for `"history"` when `generate=False`
            - `("text", "prompt")` for `"text"` when `generate=True`, `("text", "full")` for `"text"` when `generate=False`
            - `("tokens", "prompt")` for `"tokens"` when `generate=True`, `("tokens", "full")` for `"tokens"` when `generate=False`
        attention_mask_key (str, optional): The key for attention masks (used in `"tokens"` mode). Defaults to `"attention_mask"`.

            .. warning:: This argument is under development and may change in the future.

        generate (bool, optional): Whether to enable text generation. If `True`, the model will generate text based on the input.
            If `False`, only log probabilities will be computed. Defaults to `True`.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `True`.
        generate_kwargs (dict | None, optional): Additional arguments to pass to the model's generate method. Defaults to `None`.

            **Standardized Parameters (cross-backend compatible):**

            * **max_new_tokens** (int): Maximum number of new tokens to generate (maps to vLLM's max_tokens)
            * **num_return_sequences** (int): Number of sequences to return (maps to vLLM's n)
            * **temperature** (float): Sampling temperature (0.0 = deterministic, higher = more random)
            * **top_p** (float): Nucleus sampling parameter (0.0-1.0)
            * **top_k** (int): Top-k sampling parameter
            * **repetition_penalty** (float): Penalty for repeating tokens
            * **do_sample** (bool): Whether to use sampling vs greedy decoding
            * **num_beams** (int): Number of beams for beam search
            * **length_penalty** (float): Penalty for sequence length
            * **early_stopping** (bool): Whether to stop early in beam search
            * **stop_sequences** (list): Sequences that stop generation (maps to vLLM's stop)
            * **skip_special_tokens** (bool): Whether to skip special tokens in output
            * **logprobs** (bool): Whether to return log probabilities

                .. warning:: Usage of this parameter is discouraged as it may conflict with the `generate` parameter
                    of the class.

            **vLLM-Specific Parameters:**

            * **presence_penalty** (float): Penalty for token presence
            * **frequency_penalty** (float): Penalty for token frequency
            * **ignore_eos** (bool): Whether to ignore EOS token
            * **prompt_logprobs** (bool): Whether to return prompt log probabilities
            * **detokenize** (bool): Whether to detokenize output
            * **include_stop_str_in_output** (bool): Whether to include stop strings in output
            * **spaces_between_special_tokens** (bool): Whether to add spaces between special tokens
            * **sampling_type** (str): Type of sampling to use
            * **temperature_last** (bool): Whether to apply temperature only to last token
            * **top_p_last** (bool): Whether to apply top_p only to last token
            * **top_k_last** (bool): Whether to apply top_k only to last token

            **Legacy Parameter Support:**

            * **max_tokens** (int): Automatically converted to max_new_tokens
            * **n** (int): Automatically converted to num_return_sequences

            **Parameter Conflict Resolution:**

            When both legacy (vLLM-specific) and standardized parameter names are provided,
            a :exc:`ValueError` is raised to prevent confusion. For example:

            * If both ``max_tokens`` and ``max_new_tokens`` are passed, an error is raised
            * If both ``n`` and ``num_return_sequences`` are passed, an error is raised

            This ensures clear parameter usage and prevents unexpected behavior.

        tokenizer_kwargs (dict | None, optional): Additional arguments to pass to the tokenizer. Defaults to `None`.
        pad_output (bool, optional): Whether to pad the output sequences to a uniform length. Defaults to `False`.
        pad_model_input (bool, optional): Whether to pad the model input sequences to a uniform length.
            This is not supported by vLLM.
        inplace (Literal[True, False, "empty"] | None, optional): Determines how the module should handle in-place operations. Defaults to `True`.
        device (torch.device | None, optional): The device to use for computation. Defaults to `None`.
        layout (torch.layout | None, optional): The layout to use for the output tensors when `pad_output=False`. Defaults to `torch.strided`.
        chat_template_name (Literal["chatml_format", "qwen"] | None, optional): The name of the chat template to use when applying the chat template to the history.
            Defaults to `None`. For `input_mode="history"` only.
        chat_template (str | None, optional): The chat template to use when applying the chat template to the history. Defaults to `None`.
            For `input_mode="history"` only.
        num_samples (int | None, optional): The number of samples to generate. Defaults to `None` (one sample, and no batch-dimension for it).
            Can also be set via the `generate_kwargs["n"] = value` argument.
        log_probs_key (NestedKey | None, optional): The key for the log probabilities :class:`~torchrl.modules.llm.policies.LogProbs` object. Defaults to `"log_probs"`.
        text_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Text` object. Defaults to `"text"`.
        tokens_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Tokens` object. Defaults to `"tokens"`.
        masks_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.Masks` object. Defaults to `"masks"`.
        history_key (NestedKey | None, optional): The key for the action :class:`~torchrl.modules.llm.policies.ChatHistory` object. Defaults to `"history"`.
        batching (bool, optional): Whether to enable batching. Defaults to `False`. See `Batching`_ below for more details.
        min_batch_size (int | None, optional): The minimum batch size to use for batching. See `Batching`_ below for more details.
        max_batch_size (int | None, optional): The maximum batch size to use for batching. See `Batching`_ below for more details.
        batching_timeout (float, optional): The timeout for batching. See `Batching`_ below for more details.
        prefer_tokens (bool, optional): If ``True`` and ``tokens.prompt`` exists in the input tensordict,
            use those tokens directly instead of re-tokenizing from history. This enables KV cache
            consistency when used with :class:`~torchrl.envs.llm.ChatEnv` with ``with_tokenizer=True``
            or :class:`~torchrl.envs.llm.transforms.IncrementalTokenizer`. Defaults to ``False``.

    .. _Batching:

    **Batching**

    Batching is a feature that allows the module to process multiple inputs in a single call.
        It is designed to work in a multi-threaded environment.
        To enable batching, it suffices to set `batching=True` which will set `min_batch_size` to 1 if not provided.
        If you want to set a different value for `min_batch_size` or `max_batch_size` for a fine-grained control,
        you can to set `batching=True` and then set `min_batch_size` or `max_batch_size` to a value greater or equal to 1.
        The way batching works is as follows:
        - If `min_batch_size` is not provided but `max_batch_size` is, `min_batch_size` is set to 1.
        - If `max_batch_size` is not provided but `min_batch_size` is, `max_batch_size` is set to the number of inputs in the queue.
        - When the model is called, a check is performed to see if the number of inputs in the queue is greater or equal to `min_batch_size`.
          If it is, the batch is processed immediately, while waiting for the previous batch to be processed if the model is busy.
          Otherwise, the input is added to the queue and the function waits for the batch to be completed.
          While waiting for the batch to be completed, a timeout is set to `batching_timeout` seconds such that if the batch is not
          completed after `batching_timeout` seconds, the remaining items to process are processed as is and the function returns after
          at most `batching_timeout` seconds (plus the time to finish processing the previous and current batch).

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

        Example output structure for `input_mode="history"`::

            TensorDict(
                text=Text(prompt=..., response=..., full=...),
                masks=Masks(all_attention_mask=..., all_assistant_mask=...),
                tokens=Tokens(prompt=..., response=..., full=...),
                log_probs=LogProbs(prompt=..., response=..., full=...),
                history=ChatHistory(prompt=..., response=..., full=...)
            )

    Example:
        >>> from vllm import LLM
        >>> from transformers import AutoTokenizer
        >>> from torchrl.data.llm import History
        >>> from torchrl.modules.llm.policies import ChatHistory
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
        ...     return_log_probs=True,
        ...     generate_kwargs={
        ...         "max_new_tokens": 50,  # Standardized parameter
        ...         "temperature": 0.7,
        ...         "top_p": 0.9,
        ...         "do_sample": True,
        ...     }
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
        - :class:`~torchrl.modules.llm.policies.LLMWrapperBase`
        - :class:`~torchrl.modules.llm.policies.TransformersWrapper`
    """

    def __init__(
        self,
        model: Any,  # vllm.LLM | AsyncVLLMEngineService | AsyncLLMEngineExtended | str
        *,
        tokenizer: callable | str | None = None,  # type: ignore
        input_mode: str = "history",
        input_key: NestedKey | None = None,
        attention_mask_key: str = "attention_mask",
        generate: bool = True,
        generate_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        pad_output: bool = False,
        pad_model_input: bool | None = None,
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
        batching: bool | None = None,
        min_batch_size: int | None = None,
        max_batch_size: int | None = None,
        batching_timeout: float = 10.0,
        prefer_tokens: bool = False,
    ):
        super().__init__()
        self.prefer_tokens = prefer_tokens

        if batching and min_batch_size is None:
            min_batch_size = 1
        elif (min_batch_size is not None or max_batch_size is not None) and (
            batching is False
        ):
            raise ValueError(
                "min_batch_size and max_batch_size must be None if batching is False."
            )

        # Validate that min_batch_size <= max_batch_size when both are specified
        if min_batch_size is not None and max_batch_size is not None:
            if min_batch_size > max_batch_size:
                raise ValueError(
                    f"min_batch_size ({min_batch_size}) must be <= max_batch_size ({max_batch_size})"
                )

        self._min_batch_size = min_batch_size
        self._max_batch_size = max_batch_size
        self._batching_timeout = batching_timeout
        self._batch_queue = []
        self._futures = []
        if self.batching:
            self._batching_lock = threading.Lock()
        else:
            self._batching_lock = None

        _require_transformers()

        # Detect and initialize model
        if isinstance(model, str):
            # Import lazily to avoid importing vLLM backends unless actually needed.
            from torchrl.modules.llm.backends.vllm import (  # local import is intentional / required
                AsyncVLLM,
            )

            model = AsyncVLLM.from_pretrained(model)

        # Validate model type
        model_type = type(model)
        model_module = getattr(model_type, "__module__", "")
        model_name = getattr(model_type, "__name__", "")
        if model_name == "AsyncVLLM" and model_module.startswith(
            "torchrl.modules.llm.backends.vllm"
        ):
            self._model_type = "async_vllm"
        elif model_name == "LLM" and model_module.startswith("vllm"):
            self._model_type = "sync_vllm"
        elif hasattr(model, "generate") and hasattr(model, "remote"):
            # Ray actor with generate method
            self._model_type = "ray_actor"
        else:
            raise ValueError(
                f"model must be a string, vllm.LLM, AsyncVLLM, or Ray actor. Got {type(model)}"
            )

        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Import vLLM lazily: only needed if we are going to interact with vLLM types.
        # (This keeps importing this module safe even if vLLM hard-crashes on import.)
        if self._model_type in ("sync_vllm",):
            _require_vllm()

        # Validate input_mode
        if input_mode not in ["history", "text", "tokens"]:
            raise ValueError(
                f"input_mode must be one of 'history', 'text', 'tokens'. Got '{input_mode}'"
            )

        self.model = model
        self.input_mode = input_mode
        self.attention_mask_key = attention_mask_key
        self.generate = generate
        if pad_model_input is not None:
            raise ValueError("pad_model_input is not supported by vLLMWrapper.")

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
        self.log_probs_key = log_probs_key
        self.masks_key = masks_key
        self.text_key = text_key
        self.tokens_key = tokens_key

        if not isinstance(pad_output, bool):
            raise ValueError("pad_output must be a boolean")
        self.pad_output = pad_output
        self._device = device
        if not pad_output and layout is None:
            layout = torch.strided
        self.layout = layout
        padding_value = None

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
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")
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
                if hasattr(model, "get_tokenizer"):
                    tokenizer = model.get_tokenizer()
                else:
                    # Try to extract model name and load tokenizer as fallback
                    model_name = self._extract_model_name(model)
                    if model_name:
                        warnings.warn(
                            f"No tokenizer provided. Attempting to load tokenizer from model name: {model_name}"
                        )
                        from transformers import AutoTokenizer

                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                        except Exception as tokenizer_error:
                            warnings.warn(
                                f"Failed to load tokenizer from {model_name}: {tokenizer_error}"
                            )
                    else:
                        warnings.warn(
                            "No tokenizer provided and no tokenizer found in model."
                        )
            except Exception as e:
                warnings.warn(f"Could not get tokenizer from model: {e}")
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

        # Standardize common parameters
        generate_kwargs = self._standardize_generate_kwargs(generate_kwargs)

        # Extract wrapper-specific parameters
        vllm_specific_kwargs = self._get_wrapper_specific_kwargs(
            generate_kwargs, "vllm"
        )

        # Convert common parameters back to vLLM format
        vllm_kwargs = {}
        for key, value in generate_kwargs.items():
            if key in self.COMMON_GENERATION_PARAMS:
                # Convert common names to vLLM names
                if key == "max_new_tokens":
                    vllm_kwargs["max_tokens"] = value
                elif key == "num_return_sequences":
                    vllm_kwargs["n"] = value
                elif key == "stop_sequences":
                    vllm_kwargs["stop"] = value
                elif key == "logprobs":
                    # vLLM expects int for logprobs, not bool
                    if isinstance(value, bool):
                        value = 1 if value else None
                    vllm_kwargs["logprobs"] = value
                elif key == "do_sample":
                    # do_sample is handled through the sampling parameters
                    # If do_sample=False, we use greedy decoding (temperature=0)
                    # If do_sample=True, we use the provided sampling parameters
                    if not value:
                        vllm_kwargs["temperature"] = 0.0
                    # If do_sample=True, we keep the existing temperature/top_p/top_k values
                elif key in ["length_penalty", "early_stopping", "num_beams"]:
                    # These parameters are not supported by vLLM, skip them
                    pass
                else:
                    # Direct mapping for other common parameters
                    vllm_kwargs[key] = value

        # Add vLLM-specific parameters
        vllm_kwargs.update(vllm_specific_kwargs)

        self.num_samples = num_samples
        if vllm_kwargs.get("n", 1) > 1 or num_samples is not None:
            if inplace in (True, "empty"):
                raise ValueError(
                    "inplace must be False (or None) when generating more than one sample."
                )
            if inplace is None:
                inplace = False
            if (
                vllm_kwargs.get("n", 1) > 1
                and num_samples is not None
                and vllm_kwargs.get("n", 1) != num_samples
            ):
                raise ValueError("num_samples differs from generate_kwargs['n'].")
            elif num_samples is None:
                self.num_samples = vllm_kwargs.get("n", 1)
            vllm_kwargs["n"] = self.num_samples
        elif inplace is None:
            inplace = True

        self.inplace = inplace

        # vLLM expects int for logprobs, not bool. Use 1 if True, None if False.
        prompt_logprobs = 1 if return_log_probs else None

        if not generate:
            # We want only the log-probs, we generate a single token (that we then discard)
            # and retrieve the prompt log-probs
            vllm_kwargs["max_tokens"] = 1
            if not return_log_probs:
                raise ValueError("return_log_probs must be True when generate=False.")

        vllm_kwargs.setdefault("detokenize", not pad_output)
        vllm_kwargs.setdefault("prompt_logprobs", prompt_logprobs)
        vllm_kwargs.setdefault("logprobs", 1 if return_log_probs else None)
        vllm_kwargs.setdefault("include_stop_str_in_output", True)
        vllm_kwargs.setdefault("skip_special_tokens", False)

        sampling_params = SamplingParams(**vllm_kwargs)
        self.sampling_params = sampling_params

        # Additional transformers-specific settings
        self.chat_template_name = chat_template_name
        self.chat_template = chat_template

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
        # Since the input_key is dynamically determined, we don't want to set it here
        # elif hasattr(self, "input_key"):
        # constructor_kwargs["input_key"] = self.input_key

        if "attention_mask_key" in kwargs:
            constructor_kwargs["attention_mask_key"] = kwargs["attention_mask_key"]
        elif hasattr(self, "attention_mask_key"):
            constructor_kwargs["attention_mask_key"] = self.attention_mask_key

        if "generate" in kwargs:
            constructor_kwargs["generate"] = kwargs["generate"]
        elif hasattr(self, "generate"):
            constructor_kwargs["generate"] = self.generate

        if "return_log_probs" in kwargs:
            constructor_kwargs["return_log_probs"] = kwargs["return_log_probs"]
        elif not constructor_kwargs.get("generate", True):
            # if we are not generating, we want to return log-probs
            constructor_kwargs["return_log_probs"] = True
        elif hasattr(self, "return_log_probs"):
            constructor_kwargs["return_log_probs"] = self.return_log_probs

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
            constructor_kwargs["tokenizer_kwargs"] = dict(self.tokenizer_kwargs)
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

        if "history_key" in kwargs:
            constructor_kwargs["history_key"] = kwargs["history_key"]
        elif hasattr(self, "history_key"):
            constructor_kwargs["history_key"] = self.history_key

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

        if "prefer_tokens" in kwargs:
            constructor_kwargs["prefer_tokens"] = kwargs["prefer_tokens"]
        elif hasattr(self, "prefer_tokens"):
            constructor_kwargs["prefer_tokens"] = self.prefer_tokens

        # Create and return new instance
        return type(self)(**constructor_kwargs)

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for the wrapper. Useful for async engines where tokenizer retrieval is deferred."""
        self.tokenizer = tokenizer
        if self.tokenizer is not None and (
            not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer is not None:
            padding_value = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        else:
            padding_value = None
        self.padding_value = padding_value

    def _extract_model_name(self, model) -> str | None:
        """Extract model name from different model types for tokenizer fallback."""
        try:
            # For AsyncVLLM, try to get the model name from engine_args
            if hasattr(model, "engine_args") and hasattr(model.engine_args, "model"):
                return model.engine_args.model

            # For vllm.LLM, try to get the model name
            elif hasattr(model, "llm_engine") and hasattr(
                model.llm_engine, "model_config"
            ):
                return getattr(model.llm_engine.model_config, "model", None)

            # For Ray actors, try to get model name via remote call
            elif hasattr(model, "remote") and hasattr(model, "get_model_name"):
                import ray

                try:
                    return ray.get(model.get_model_name.remote())
                except Exception:
                    pass

            # Try common attributes that might contain model name
            for attr in ["model_name", "model", "model_path", "_model_name"]:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    if isinstance(value, str):
                        return value

            return None
        except Exception:
            return None

    def _call_generate(self, *args, **kwargs):
        """Call generate method based on model type.

        In vLLM 0.14+, prompt_token_ids should be passed as TokensPrompt objects
        rather than as a keyword argument.
        """
        # Convert prompt_token_ids to TokensPrompt format for vLLM 0.14+ compatibility
        prompt_token_ids = kwargs.pop("prompt_token_ids", None)
        if prompt_token_ids is not None and TokensPrompt is not None:
            # Convert list of token ID lists to TokensPrompt objects
            if isinstance(prompt_token_ids, list) and len(prompt_token_ids) > 0:
                if isinstance(prompt_token_ids[0], list):
                    # List of token ID lists -> list of TokensPrompt
                    prompts = [
                        TokensPrompt(prompt_token_ids=tids) for tids in prompt_token_ids
                    ]
                else:
                    # Single token ID list -> single TokensPrompt
                    prompts = TokensPrompt(prompt_token_ids=prompt_token_ids)
                # Insert prompts as the first positional argument
                args = (prompts,) + args
        elif prompt_token_ids is not None:
            # Fallback for older vLLM versions that still support prompt_token_ids kwarg
            kwargs["prompt_token_ids"] = prompt_token_ids

        if self._model_type == "ray_actor":
            import ray

            return ray.get(self.model.generate.remote(*args, **kwargs))
        else:
            # Both sync_vllm and async_vllm have direct generate methods
            return self.model.generate(*args, **kwargs)

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
        tensordict_orig = tensordict
        if not tensordict.ndim:
            if tensordict_out is not None:
                raise ValueError(
                    "tensordict_out must not be provided when tensordict.ndim == 0. If this is needed, "
                    "please submit an issue on github."
                )
            # unsqueeze - squeeze the input
            return self.forward(lazy_stack([tensordict]), logits_only=logits_only)[0]
        elif tensordict.ndim > 1:
            if tensordict_out is not None:
                raise ValueError(
                    "tensordict_out must not be provided when tensordict.ndim > 1. If this is needed, "
                    "please submit an issue on github."
                )
            return self.forward(tensordict.reshape(-1), logits_only=logits_only).view(
                tensordict.shape
            )

        if not isinstance(tensordict, LazyStackedTensorDict):
            tensordict = tensordict.to_lazystack(0)

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
                tensordict_out = tensordict_orig
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

    def _from_vllm_generate_history(
        self,
        tensordict_input: TensorDictBase,
        sampling_params: Any,
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

        # Check for existing tokens when prefer_tokens=True
        # This enables token-first inference for KV cache consistency
        existing_tokens = None
        if self.prefer_tokens:
            # Primary: tokens.prompt (from IncrementalTokenizer)
            existing_tokens = tensordict_input.get((self.tokens_key, "prompt"), None)
            if existing_tokens is None:
                # Fallback: tokens.full (for backward compatibility)
                existing_tokens = tensordict_input.get((self.tokens_key, "full"), None)

        tokens_prompt_padded = None
        tokens_prompt_unpadded = None

        if existing_tokens is not None:
            # Use existing tokens directly - skip tokenization for KV cache consistency
            # Handle different token storage formats:
            # - list: from manual construction or as_list=True retrieval
            # - nested tensor: from IncrementalTokenizer (torch.nested.as_nested_tensor)
            # - regular tensor: padded tensor
            if isinstance(existing_tokens, list):
                tokens_list = existing_tokens
            elif (
                isinstance(existing_tokens, torch.Tensor) and existing_tokens.is_nested
            ):
                # Unbind nested tensor to get list of tensors
                tokens_list = list(existing_tokens.unbind(0))
            else:
                # Already a padded tensor - extract non-padded sequences
                tokens_list = [
                    tokens[tokens != self.padding_value] for tokens in existing_tokens
                ]

            if self.pad_output:
                tokens_prompt_padded = pad_sequence(
                    tokens_list,
                    batch_first=True,
                    padding_value=self.padding_value,
                    padding_side="left",
                )
            else:
                tokens_prompt_unpadded = tokens_list

            # Still need text_prompt for output, but we can derive it from tokens
            text_prompt = self.tokenizer.batch_decode(
                tokens_list,
                skip_special_tokens=False,
            )
        else:
            # Fall back to tokenizing from history (original behavior)
            # Apply chat template
            tokenizer_kwargs = {}
            if self.chat_template_name is not None:
                tokenizer_kwargs.setdefault(
                    "chat_template_name", self.chat_template_name
                )
            if self.chat_template is not None:
                tokenizer_kwargs.setdefault("chat_template", self.chat_template)
            tokenizer_kwargs.setdefault("add_generation_prompt", True)
            text_prompt = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )

            tokenizer_kwargs.setdefault("return_assistant_tokens_mask", False)
            tokenizer_kwargs.setdefault("tokenize", True)
            tokenizer_kwargs.setdefault("padding", False)
            tokenizer_kwargs.setdefault("return_dict", True)
            response_struct = history.apply_chat_template(
                tokenizer=self.tokenizer, **tokenizer_kwargs
            )
            if self.pad_output:
                tokens_prompt_padded = response_struct.get(
                    "input_ids",
                    as_padded_tensor=True,
                    padding_value=self.padding_value,
                    padding_side="left",
                )
            else:
                tokens_prompt_unpadded = response_struct.get("input_ids", as_list=True)

        result = self._generate_from_tokens(
            tokens_prompt_padded=tokens_prompt_padded,
            tokens_prompt_unpadded=tokens_prompt_unpadded,
            sampling_params=sampling_params,
            out=out,
        )

        # Generate using text path
        if self.pad_output:
            result[(self.tokens_key, "prompt")] = (
                tokens_prompt_padded
                if not self.num_samples
                else tokens_prompt_padded.unsqueeze(1).repeat(1, self.num_samples, 1)
            )
        else:
            tokens_prompt_nested = torch.nested.as_nested_tensor(tokens_prompt_unpadded)
            if not self.num_samples:
                result[(self.tokens_key, "prompt")] = tokens_prompt_nested
            else:
                for r in result.unbind(1):
                    r[(self.tokens_key, "prompt")] = tokens_prompt_nested

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
                # print("shapes of assistant masks", [t.shape for t in result_flat.get(("masks", "all_assistant_mask"), as_list=True)])
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
            history_chat_flat.full = history_chat_flat.prompt.extend(
                h_responses, inplace=False, dim=-1
            )
        result.set(self.history_key, history_chat)
        return result

    def _from_vllm_logprobs_history(
        self,
        tensordict_input: TensorDictBase,
        sampling_params: Any,
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
        text_full = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("return_dict", True)
        response_struct = history.apply_chat_template(
            tokenizer=self.tokenizer, **tokenizer_kwargs
        )

        result = self._logprobs_from_tokens(
            response_struct=response_struct, sampling_params=sampling_params, out=out
        )
        text_result = Text._from_tensordict(result.empty())
        result.set(self.text_key, text_result)
        result[self.text_key, "full"] = text_full
        result.set(self.history_key, ChatHistory(full=history))
        return result

    def _from_vllm_generate_text(
        self, td: TensorDictBase, sampling_params: Any, out: TensorDictBase
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
        self, td: TensorDictBase, sampling_params: Any, out: TensorDictBase
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
        self, td: TensorDictBase, sampling_params: Any, out: TensorDictBase
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
        self, td: TensorDictBase, sampling_params: Any, out: TensorDictBase
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
        self, text: str | list[str], response_text: str | list[str] | None
    ) -> str | list[str]:
        """Concatenate text and response text."""
        assert isinstance(
            text, (str, list)
        ), f"text must be str or list, got {type(text)}"

        # Handle None response_text (when tokenizer is not available)
        if response_text is None:
            raise RuntimeError(
                "response_text is None, likely due to missing tokenizer. "
                "Cannot decode vLLM response without a tokenizer. "
                "Please provide a tokenizer explicitly or ensure the model has one available."
            )

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
        sampling_params: Any,
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

        # Call generate based on model type
        request_output = self._call_generate(text, *args, **generate_kwargs)

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

        masks_obj = Masks._from_tensordict(out.empty())
        masks_obj.all_attention_mask = None
        masks_obj.all_assistant_mask = None
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        if self.num_samples is not None:
            text = [txt for txt in text for _ in range(self.num_samples)]
        text_obj = Text._from_tensordict(out.empty())
        with text_obj.view(-1) as text_obj_flat:
            text_obj_flat.prompt = text
            text_obj_flat.response = response_text
            text_obj_flat.full = self._cat_text(text, response_text)
        out.set(self.text_key, text_obj)

        tokens_obj = Tokens._from_tensordict(out.empty())
        with tokens_obj.view(-1) as tokens_obj_flat:
            tokens_obj_flat.prompt = None  # We don't have prompt tokens in this path
            if self.pad_output:
                tokens_obj_flat.response = response_tokens_padded
                self._check_padded(response_tokens_padded)
            else:
                tokens_obj_flat.response = response_tokens_list
                self._check_not_padded(response_tokens_list)
            tokens_obj_flat.full = (
                None  # we don't have prompt tokens in this path so no all_tokens either
            )
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

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
            out.set(self.log_probs_key, log_probs_obj)

        return out

    def _logprobs_from_text(
        self,
        text: str | list[str] | NonTensorStack,
        sampling_params: Any,
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
            tokens_full_padded = tokenized_output["input_ids"]
            attention_mask_full_padded = tokenized_output["attention_mask"]
            tokens_full_list = self._to_list(
                tokens_full_padded, attention_mask_full_padded
            )
        else:
            tokens_full_unpadded = tokenized_output["input_ids"]
            tokens_full_list = self._to_list(tokens_full_unpadded, None)
            attention_mask_full_unpadded = tokenized_output["attention_mask"]
            attention_mask_full_unpadded = [
                am.bool()
                if isinstance(am, torch.Tensor)
                else torch.tensor(am, dtype=torch.bool)
                for am in attention_mask_full_unpadded
            ]

        # Convert to list format for vLLM
        generate_kwargs = {
            "sampling_params": sampling_params,
            "prompt_token_ids": tokens_full_list,
        }

        # Generate with vLLM to get prompt_logprobs
        request_output = self._call_generate(**generate_kwargs)

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
            attention_mask_full_padded = tokens_full_padded != self.padding_value
            log_probs_full_padded = torch.where(
                attention_mask_full_padded, log_probs_full_padded, 0.0
            )
        else:
            # For unpadded case, extract from each sequence
            log_probs_full_unpadded = request_output_tc.get(
                "prompt_logprobs", as_list=True
            )
            self._check_not_padded(log_probs_full_unpadded)

        masks_obj = Masks._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            self._check_padded(attention_mask_full_padded)
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
        else:
            self._check_not_padded(attention_mask_full_unpadded)
            masks_obj.all_attention_mask = attention_mask_full_unpadded
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

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
            self._check_padded(tokens_full_padded)
            tokens_obj.full = tokens_full_padded
        else:
            tokens_obj.full = tokens_full_unpadded
        tokens_obj.response = None
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        if self.return_log_probs:
            log_probs_obj = LogProbs._from_tensordict(
                TensorDict(batch_size=out.batch_size).to_lazystack(0)
            )
            if self.pad_output:
                self._check_padded(log_probs_full_padded)
                log_probs_obj.full = log_probs_full_padded
            else:
                self._check_not_padded(log_probs_full_unpadded)
                log_probs_obj.full = log_probs_full_unpadded
            log_probs_obj.response = None
            log_probs_obj.padded = MetaData(self.pad_output)
            out.set(self.log_probs_key, log_probs_obj)

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
        sampling_params: Any,
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
        empirical_attention_mask = None

        if tokens_prompt_unpadded is None:
            # TODO: To be on the safe side, we may do this even in the unpadded case since we're not sure
            #  the user passed an unpadded tensor in the first place.
            empirical_attention_mask = tokens_prompt_padded != self.padding_value
            tokens_prompt_list = self._to_list(
                tokens_prompt_padded, empirical_attention_mask
            )
        else:
            tokens_prompt_list = self._to_list(tokens_prompt_unpadded, None)
        generate_kwargs.update({"prompt_token_ids": tokens_prompt_list})

        # Call generate based on model type
        request_output = self._call_generate(*args, **generate_kwargs)

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
                tokens_prompt_unpadded if not self.pad_output else tokens_prompt_padded
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
        out.set(self.tokens_key, tokens_obj)

        masks_obj = Masks._from_tensordict(out.empty())
        # self.return_tokens must be True
        if self.pad_output:
            # Get "real" attention masks
            full_attention_mask_padded = tokens_obj.get("full") != self.padding_value
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
        out.set(self.masks_key, masks_obj)

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
                    if (
                        prompt_logprobs_padded.shape[-1]
                        != tokens_prompt_padded.shape[-1]
                    ):
                        tshape = tokens_prompt_padded.shape
                        oshape = prompt_logprobs_padded.shape
                        # it could be that the input was padded already - padding again then
                        prompt_logprobs_padded = torch.cat(
                            [
                                prompt_logprobs_padded.new_zeros(
                                    tshape[:-1] + (tshape[-1] - oshape[-1],)
                                ),
                                prompt_logprobs_padded,
                            ],
                            -1,
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
            out.set(self.log_probs_key, log_probs_obj)
        return out

    def _logprobs_from_tokens(
        self,
        *,
        response_struct: TensorDictBase | None = None,
        tokens_full_unpadded: list[torch.Tensor] | None = None,
        tokens_full_padded: torch.Tensor | None = None,
        sampling_params: Any | None = None,
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
        tokens_out_stuct = self._call_generate(**generate_kwargs)

        request_output_tc = _RequestOutput_tc.from_request_output(tokens_out_stuct)

        # For unpadded case, extract from each sequence
        log_probs_full_unpadded = request_output_tc.get("prompt_logprobs", as_list=True)

        # Extract log-probs from prompt_logprobs
        if self.pad_output:
            # For padded case, use all prompt_logprobs
            if attention_mask_full_padded is not None:
                attention_mask_full_padded = tokens_full_padded != self.padding_value
            log_probs_full_padded = torch.zeros_like(
                tokens_full_padded, dtype=torch.get_default_dtype()
            )
            log_probs_full_padded[attention_mask_full_padded] = torch.cat(
                log_probs_full_unpadded, -1
            )
        else:
            self._check_not_padded(log_probs_full_unpadded)

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

        masks_obj = Masks._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            self._check_padded(attention_mask_full_padded)
            masks_obj.all_attention_mask = attention_mask_full_padded.bool()
            if assistant_mask_full_padded is not None:
                masks_obj.all_assistant_mask = assistant_mask_full_padded
        else:
            self._check_not_padded(attention_mask_full_unpadded)
            masks_obj.all_attention_mask = attention_mask_full_unpadded
            if assistant_mask_full_unpadded is not None:
                masks_obj.all_assistant_mask = assistant_mask_full_unpadded
        masks_obj.padded = MetaData(self.pad_output)
        out.set(self.masks_key, masks_obj)

        tokens_obj = Tokens._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            self._check_padded(tokens_full_padded)
            tokens_obj.full = tokens_full_padded
        else:
            tokens_obj.full = tokens_full_unpadded
        tokens_obj.response = None
        tokens_obj.padded = MetaData(self.pad_output)
        out.set(self.tokens_key, tokens_obj)

        log_probs_obj = LogProbs._from_tensordict(
            TensorDict(batch_size=out.batch_size).to_lazystack(0)
        )
        if self.pad_output:
            self._check_padded(log_probs_full_padded)
            log_probs_obj.full = log_probs_full_padded
        else:
            self._check_not_padded(log_probs_full_unpadded)
            log_probs_obj.full = log_probs_full_unpadded
        log_probs_obj.response = None
        log_probs_obj.padded = MetaData(self.pad_output)
        out.set(self.log_probs_key, log_probs_obj)

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
        _vllm = _require_vllm()

        if hasattr(cls, "_CompletionOutput_tc"):
            return cls._CompletionOutput_tc
        CompletionOutput_tc = from_dataclass(_vllm.outputs.CompletionOutput)  # type: ignore
        cls._CompletionOutput_tc = CompletionOutput_tc
        return CompletionOutput_tc

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

        vLLM does not return logits, so this method is not supported.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_dist is not supported"
        )

    def get_dist_with_prompt_mask(
        self,
        tensordict: TensorDictBase,
        tokens_key: NestedKey = ("tokens", "full"),
        logits_key: NestedKey = "logits",
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include response tokens (exclude prompt).

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_dist_with_prompt_mask is not supported"
        )

    def _get_dist_with_assistant_mask(
        self,
        tensordict: TensorDictBase,
        assistant_mask_key: NestedKey = ("masks", "all_assistant_mask"),
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked to only include assistant tokens.

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_dist_with_assistant_mask is not supported"
        )

    def _get_dist_with_attention_mask(
        self,
        tensordict: TensorDictBase,
        attention_mask_key: NestedKey = ("masks", "all_attention_mask"),
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution masked using attention mask.

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_dist_with_attention_mask is not supported"
        )

    def _get_dist_with_custom_mask(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor,
        logits_key: NestedKey = "logits",
        **kwargs,
    ) -> D.Distribution:
        """Get distribution with custom mask.

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_dist_with_custom_mask is not supported"
        )

    # Convenience methods for common LLM training scenarios
    def _get_sft_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for SFT loss (response tokens only).

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_sft_dist is not supported"
        )

    def _get_rlhf_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for RLHF loss (assistant tokens only).

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_rlhf_dist is not supported"
        )

    def _get_generic_dist(self, tensordict: TensorDictBase, **kwargs) -> D.Distribution:
        """Get distribution suitable for generic losses (all tokens).

        vLLM does not return logits, so this method is not supported.

        This is a provisional method that will be replaced by the `get_dist` method once we have a better masking strategy.
        """
        raise NotImplementedError(
            "vLLM does not return logits, so get_generic_dist is not supported"
        )


class _RequestOutput_tc(TensorClass["nocast"]):
    """TensorClass wrapper for vLLM RequestOutput."""

    request_id: str
    prompt: str
    prompt_token_ids: torch.Tensor
    prompt_logprobs: torch.Tensor
    outputs: Any
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
                token_ids = output.token_ids
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                for v, tid in zip(output.logprobs, token_ids):
                    t.append(
                        v[tid]["logprob"] if v[tid].get("logprob") is not None else 0.0
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
