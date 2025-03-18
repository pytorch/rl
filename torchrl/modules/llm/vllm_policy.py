# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import importlib.util

import torch
from tensordict import (
    from_dataclass,
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    NestedKey,
    NonTensorData,
    NonTensorStack,
    TensorClass,
    TensorDict,
)
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
    WrapModule,
)
from tensordict.utils import _zip_strict, expand_as_right

from torchrl.data import LLMData

_has_vllm = importlib.util.find_spec("vllm")

CompletionOutput_tc = None


def _maybe_clear_device(td):
    if td.device is None:
        return td
    return td.set(NonTensorData("_source_device"), td.device).clear_device_()


def _maybe_set_device(td):
    device = td.pop("_source_device", None)
    if device is None:
        return td
    elif isinstance(device, NonTensorData):
        device: torch.device = device.data
    return td.to(device)


def from_vllm(
    model: vllm.LLM,  # noqa
    *,
    return_log_probs: bool = False,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer  # noqa
    | None = None,  # noqa
    from_text: bool = False,
    device: torch.device | None = None,
    generate: bool = True,
    generate_kwargs: dict | None = None,
    tokenizer_kwargs: dict | None = None,
    pad_output: bool = True,
) -> TensorDictModuleBase:
    """Creates a TensorDictModule from a vLLM model.

    This function provides a consistent interface across various LLM engines.

    It supports text generation and log probability computation, similar to the Hugging Face Transformers interface.

    Args:
        model (LLM): The vLLM model to wrap.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `False`.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer, optional): The tokenizer to use. Defaults to `None`.
        from_text (bool, optional): Whether the input is text. Defaults to `False`.
        device (torch.device, optional): The device to use for computation. Defaults to `None`.
        generate (bool, optional): Whether to generate text. Defaults to `True`.
        generate_kwargs (dict, optional): Additional arguments for the model's generate method. Defaults to `None`.
        tokenizer_kwargs (dict, optional): Additional arguments for the tokenizer. Defaults to `None`.

    Returns:
        TensorDictModuleBase: A configured TensorDictModule for the specified model.

    Input Keys:

        - If `from_text` is `True`:

            - "text": The input text to be tokenized.

        - If `from_text` is False:

            - "tokens": The input token sequences.
            - "attention_mask": The attention mask for the tokens.

    Output Keys:

        - "tokens_response": The generated token sequences.
        - "log_probs": The log probabilities of the generated tokens (if `return_log_probs` is True).
        - "text_response": The generated text (if `from_text` is True and `generate` is True).

    Example:
        >>> from vllm import LLM
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = LLM(model="facebook/opt-125m")
        >>> module = from_vllm(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     from_text=True,
        ...     generate=True
        ... )
        >>> input_data = LLMData(text=NonTensorStack("Hello, world!"), batch_size=1)
        >>> output_data = module(input_data)
        >>> print(output_data.text_response)

    .. seealso:: :func:`~torchrl.modules.from_hf_transformers` for a similar interface using the Hugging Face
        Transformers library.

    """

    # TODO: Seq should have a return_log_prob and be of ProbabilisticTDSequential type for instance checks
    if tokenizer is None:
        tokenizer = model.get_tokenizer()

    # retrieve the padding value - we use this to make the log-probs of pad token = 1
    padding_value = tokenizer(tokenizer.pad_token)["input_ids"][0]

    if from_text:
        if generate:
            func = _from_vllm_generate_text
        else:
            func = _from_vllm_logprobs_text
    else:
        if generate:
            func = _from_vllm_generate_tokens
        else:
            func = _from_vllm_logprobs_tokens
    module_dict = func(
        tokenizer=tokenizer,
        model=model,
        device=device,
        padding_value=padding_value,
        generate_kwargs=generate_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        return_log_probs=return_log_probs,
        pad_output=pad_output,
    )
    return Seq(module_dict, inplace=True)


def to_list(tokens, attention_mask):
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
    return NonTensorStack(*tokens)


def _from_vllm_generate_text(
    *,
    tokenizer,
    model,
    device,
    padding_value,
    generate_kwargs,
    tokenizer_kwargs,
    return_log_probs,
    pad_output,
):
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Please install `vllm` to use `from_vllm`.")

    text_key: NestedKey = ("text",)
    token_key: NestedKey = ("tokens",)
    attention_mask_key: NestedKey = ("attention_mask",)

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    if not tokenizer_kwargs:
        tokenizer_kwargs = {}
    if not tokenizer_kwargs.setdefault("return_attention_mask", True):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("return_tensors", "pt") != "pt":
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding", True) not in (True,):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
        raise RuntimeError

    def tokenize(td):
        out = TensorDict(batch_size=td.batch_size, device=td.device)
        tokens_in = TensorDict.from_dict(
            tokenizer(td.get(text_key), **tokenizer_kwargs)
        )
        out.set("tokens_in", tokens_in)
        return out

    module_dict["encode"] = WrapModule(
        tokenize,
        in_keys=[text_key],
        out_keys=["tokens_in"],
    )

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        out_keys=[("tokens_in", "input_ids_list")],
        strict=False,
    )

    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs.setdefault("detokenize", False)
    generate_kwargs.setdefault("prompt_logprobs", False)
    generate_kwargs.setdefault("logprobs", return_log_probs)
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids_list"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td, padding_value=padding_value):
        td["tokens_out"] = _RequestOutput_tc.from_request_output(td["tokens_out"])
        if pad_output and td.ndim and not isinstance(td, LazyStackedTensorDict):
            td = lazy_stack(list(td.unbind(0)))
        # When not generate, we don't want to overwrite this
        tokens_response_td = td["tokens_out"].outputs._tensordict.select(
            "token_ids", "logprobs", strict=False
        )
        if pad_output:
            tokens_response_td = tokens_response_td.densify(
                layout=torch.strided
            ).to_padded_tensor(padding=padding_value)
        tokens_response_td.rename_key_("token_ids", "tokens_response")
        if return_log_probs:
            padded_values = tokens_response_td["tokens_response"] == padding_value
            tokens_response_td.rename_key_("logprobs", "log_probs")
            if padded_values.any():
                lps = tokens_response_td["log_probs"]
                lps = torch.where(expand_as_right(~padded_values, lps), lps, 0.0)
                tokens_response_td["log_probs"] = lps
        td.update(tokens_response_td)
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    module_dict["decode"] = Mod(
        tokenizer.batch_decode,
        in_keys=["tokens_response"],
        out_keys=["text_response"],
    )

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    in_keys = [
        "log_probs",
        "tokens_response",
        ("tokens_in", "input_ids"),
        ("tokens_in", "attention_mask"),
        "text_response",
    ]
    out_keys = [
        "log_probs",
        "tokens_response",
        token_key,
        attention_mask_key,
        "text_response",
    ]

    def format_td(td):
        td = td.select(*in_keys, strict=False)
        td.rename_key_(("tokens_in", "input_ids"), token_key)
        td.rename_key_(("tokens_in", "attention_mask"), attention_mask_key)
        del td["tokens_in"]
        return td

    module_dict["format"] = WrapModule(
        format_td,
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return module_dict


def _from_vllm_generate_tokens(
    *,
    tokenizer,
    model,
    device,
    padding_value,
    generate_kwargs,
    tokenizer_kwargs,
    return_log_probs,
    pad_output,
):
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Please install `vllm` to use `from_vllm`.")

    token_key: NestedKey = ("tokens",)
    attention_mask_key: NestedKey = ("attention_mask",)

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device

    def move_input(td):
        result = TensorDict(batch_size=td.batch_size, device=td.device)
        result["tokens_in"] = result.new_empty()
        result["tokens_in", "input_ids"] = td.get("tokens")
        result["tokens_in", "attention_mask"] = td.get("attention_mask")
        return result

    module_dict["move_inputs"] = WrapModule(
        move_input,
        in_keys=["tokens", "attention_mask"],
        out_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
    )

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        out_keys=[("tokens_in", "input_ids_list")],
        strict=False,
    )

    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs.setdefault("detokenize", False)
    generate_kwargs.setdefault("prompt_logprobs", False)
    generate_kwargs.setdefault("logprobs", return_log_probs)
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids_list"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td, padding_value=padding_value):
        td["tokens_out"] = _RequestOutput_tc.from_request_output(td["tokens_out"])
        if pad_output and td.ndim and not isinstance(td, LazyStackedTensorDict):
            td = lazy_stack(list(td.unbind(0)))
        # When not generate, we don't want to overwrite this
        tokens_response_td = td["tokens_out"].outputs._tensordict.select(
            "token_ids", "logprobs", strict=False
        )
        if pad_output:
            tokens_response_td = tokens_response_td.densify(
                layout=torch.strided
            ).to_padded_tensor(padding=padding_value)
        tokens_response_td.rename_key_("token_ids", "tokens_response")
        if return_log_probs:
            padded_values = tokens_response_td["tokens_response"] == padding_value
            tokens_response_td.rename_key_("logprobs", "log_probs")
            if padded_values.any():
                lps = tokens_response_td["log_probs"]
                lps = torch.where(expand_as_right(~padded_values, lps), lps, 0.0)
                tokens_response_td["log_probs"] = lps
        td.update(tokens_response_td)
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    in_keys = [
        "log_probs",
        "tokens_response",
        ("tokens_in", "input_ids"),
        ("tokens_in", "attention_mask"),
        "text_response",
    ]
    out_keys = [
        "log_probs",
        "tokens_response",
        token_key,
        attention_mask_key,
        "text_response",
    ]

    def format_td(td):
        td = td.select(*in_keys, strict=False)
        td.rename_key_(("tokens_in", "input_ids"), token_key)
        td.rename_key_(("tokens_in", "attention_mask"), attention_mask_key)
        del td["tokens_in"]
        return td

    module_dict["format"] = WrapModule(
        format_td,
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return module_dict


def _from_vllm_logprobs_text(
    *,
    tokenizer,
    model,
    device,
    padding_value,
    generate_kwargs,
    tokenizer_kwargs,
    return_log_probs,
    pad_output,
):
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Please install `vllm` to use `from_vllm`.")

    text_key: NestedKey = ("text",)

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    if not tokenizer_kwargs:
        tokenizer_kwargs = {}
    if not tokenizer_kwargs.setdefault("return_attention_mask", True):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("return_tensors", "pt") != "pt":
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding", True) not in (True,):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
        raise RuntimeError

    def tokenize(td):
        out = TensorDict(batch_size=td.batch_size, device=td.device)
        text_prompt = td.get(text_key)
        text_response = td.get("text_response")
        tokens_in = tokenizer(
            [_x + _y for _x, _y in zip(text_prompt, text_response)], **tokenizer_kwargs
        )
        tokens_prompt = tokenizer(text_prompt, **tokenizer_kwargs)
        tokens_in = TensorDict.from_dict(tokens_in)
        out["tokens_in"] = tokens_in
        tokens_response = tokens_in.apply(
            lambda total_tokens, input_tokens: total_tokens[:, input_tokens.shape[1] :],
            TensorDict.from_dict(tokens_prompt),
        )
        out["tokens_response"] = tokens_response
        return out

    module_dict["encode"] = WrapModule(
        # TODO: make this work with many strings
        tokenize,
        in_keys=[text_key, "text_response"],
        out_keys=["tokens_in", "tokens_response"],
    )

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        out_keys=[("tokens_in", "input_ids_list")],
        strict=False,
    )

    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs.setdefault("detokenize", False)
    generate_kwargs.setdefault("prompt_logprobs", True)
    generate_kwargs.setdefault("logprobs", return_log_probs)
    generate_kwargs["max_tokens"] = 1
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids_list"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td, padding_value=padding_value):
        td["tokens_out"] = _RequestOutput_tc.from_request_output(td["tokens_out"])
        if pad_output and td.ndim and not isinstance(td, LazyStackedTensorDict):
            td = lazy_stack(list(td.unbind(0)))
        td["prompt_logprobs"] = td["tokens_out"].prompt_logprobs
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    def translate_lps(tokens_response, x):
        # we disregard the tokens from the prompt to focus on those of the response
        padded = tokens_response == padding_value
        lps = x[..., -tokens_response.shape[-1] :]
        lps = torch.where(~padded, lps, 0.0)
        return lps

    module_dict["translate_lps"] = Mod(
        translate_lps,
        in_keys=[("tokens_response", "input_ids"), "prompt_logprobs"],
        out_keys=["log_probs"],
    )

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    module_dict["format"] = Mod(
        lambda *x: x,
        in_keys=["log_probs", ("tokens_response", "input_ids")],
        out_keys=["log_probs", "tokens_response"],
        strict=False,
        inplace="empty",
    )

    return module_dict


def _from_vllm_logprobs_tokens(
    *,
    tokenizer,
    model,
    device,
    padding_value,
    generate_kwargs,
    tokenizer_kwargs,
    return_log_probs,
    pad_output,
):
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Please install `vllm` to use `from_vllm`.")

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device

    def stack_for_logprobs(td):
        tokens = td.get("tokens")
        tokens_response = td.get("tokens_response")
        attention_mask = td.get("attention_mask")

        tokens = torch.cat([tokens, tokens_response], -1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(tokens_response.shape)], -1
            )
        result = TensorDict(batch_size=td.batch_size, device=td.device)
        result.set(("tokens_in", "input_ids"), tokens)
        result.set(("tokens_response", "input_ids"), tokens_response)
        if attention_mask is not None:
            result.set(("tokens_in", "attention_mask"), attention_mask)
        return result

    module_dict["stack_response"] = WrapModule(
        stack_for_logprobs,
        in_keys=["tokens", "tokens_response", "attention_mask"],
        out_keys=[
            ("tokens_in", "input_ids"),
            ("tokens_response", "input_ids"),
            ("tokens_in", "attention_mask"),
        ],
    )

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        out_keys=[("tokens_in", "input_ids_list")],
        strict=False,
    )

    if generate_kwargs is None:
        generate_kwargs = {}
    generate_kwargs.setdefault("detokenize", False)
    generate_kwargs.setdefault("prompt_logprobs", True)
    generate_kwargs.setdefault("logprobs", return_log_probs)
    generate_kwargs["max_tokens"] = 1
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids_list"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td, padding_value=padding_value):
        td["tokens_out"] = _RequestOutput_tc.from_request_output(td["tokens_out"])
        if pad_output and td.ndim and not isinstance(td, LazyStackedTensorDict):
            td = lazy_stack(list(td.unbind(0)))
        td["prompt_logprobs"] = td["tokens_out"].prompt_logprobs
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    def translate_lps(tokens_response, lps):
        # we disregard the tokens from the prompt to focus on those of the response
        padded = tokens_response == padding_value
        lps = lps[..., -tokens_response.shape[-1] :]
        lps = torch.where(~padded, lps, 0.0)
        return lps

    module_dict["translate_lps"] = Mod(
        translate_lps,
        in_keys=[("tokens_response", "input_ids"), "prompt_logprobs"],
        out_keys=["log_probs"],
    )

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    module_dict["format"] = Mod(
        lambda *x: x,
        in_keys=["log_probs", "tokens_response"],
        out_keys=["log_probs", "tokens_response"],
        strict=False,
        inplace="empty",
    )

    return module_dict


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
        global CompletionOutput_tc
        if CompletionOutput_tc is None:
            import vllm

            CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)

        def postproc(output):
            def get_logprob(output):
                t = []
                for v, tid in zip(output.logprobs, output.token_ids):
                    t.append(
                        v[tid]["logprob"] if v[tid].get("logprob") is not None else 0.0
                    )
                return torch.tensor(t)

            if output.logprobs:
                output.logprobs = get_logprob(output)
            output.token_ids = torch.tensor(output.token_ids)
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
                self.outputs = torch.stack(outputs)
            self.prompt_logprobs = torch.tensor(
                [
                    v[tid].logprob if v is not None else 0.0
                    for v, tid in _zip_strict(
                        self.prompt_logprobs, self.prompt_token_ids
                    )
                ]
            )
            self.prompt_token_ids = torch.tensor(self.prompt_token_ids)
            self.num_cached_tokens = torch.tensor(self.num_cached_tokens)

    @classmethod
    def from_request_output(cls, requests):
        out = maybe_dense_stack(
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


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts, sampling_params)
    m = from_vllm(llm, from_text=True)

    td = m(LLMData(text=NonTensorStack("a text"), batch_size=1))

    td = m(LLMData(text=NonTensorData("a text"), batch_size=()))

    td = m(LLMData(text=NonTensorStack("a text"), batch_size=1))
    m = from_vllm(llm, from_text=True, generate=False)
    assert td.copy().text == ["a text"]
    td_lp = LLMData(
        text=NonTensorStack("a text"),
        text_response=NonTensorStack(*td.text_response),
        batch_size=(1,),
    )
    td_lp = m(td_lp)
    # torch.testing.assert_close(td.log_probs, td_lp.log_probs)

    m = from_vllm(llm, from_text=True, generate=True)
    td = m(LLMData(text=NonTensorStack("a text", "another text here"), batch_size=2))
