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
    maybe_dense_stack,
    NestedKey,
    NonTensorData,
    NonTensorStack,
    TensorClass,
)
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from tensordict.utils import _zip_strict

from torchrl.data import LLMData

_has_vllm = importlib.util.find_spec("vllm")

if _has_vllm:
    import vllm

    CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)
else:
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
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Please install `vllm` to use `from_vllm`.")

    text_key: NestedKey = ("text",)
    token_key: NestedKey = ("tokens",)
    attention_mask_key: NestedKey = ("attention_mask",)

    # TODO: Seq should have a return_log_prob and be of ProbabilisticTDSequential type for instance checks
    if tokenizer is None:
        tokenizer = model.get_tokenizer()
    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    if from_text:
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

        if generate:
            module_dict["encode"] = Mod(
                tokenizer,
                in_keys=[text_key],
                out_keys=["tokens_in"],
                method_kwargs=tokenizer_kwargs,
                strict=True,
                inplace=False,
            )
        else:
            module_dict["encode"] = Mod(
                # TODO: make this work with many strings
                # Tokenize both strings, and only the first
                lambda x, y: (
                    tokenizer([_x + _y for _x, _y in zip(x, y)], **tokenizer_kwargs),
                    tokenizer(x, **tokenizer_kwargs),
                ),
                in_keys=[text_key, "text_response"],
                out_keys=["tokens_in", "tokens_response"],
                strict=True,
                inplace=False,
            )

            def select(x, y):
                return x.apply(lambda _x, _y: _x[..., _y.shape[-1] :], y)

            module_dict["stack_response"] = Mod(
                # Remove the init from the total tokens to get only the response tokens
                select,
                in_keys=["tokens_in", "tokens_response"],
                out_keys=["tokens_response"],
                strict=True,
            )
    elif not generate:

        def stack_for_logprobs(tokens, tokens_response, attention_mask=None):
            tokens = torch.cat([tokens, tokens_response], -1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(tokens_response.shape)], -1
                )
            return tokens, tokens_response, attention_mask

        module_dict["stack_response"] = Mod(
            stack_for_logprobs,
            in_keys=["tokens", "tokens_response", "attention_mask"],
            out_keys=[
                ("tokens_in", "input_ids"),
                ("tokens_response", "input_ids"),
                ("tokens_in", "attention_mask"),
            ],
            strict=False,
            inplace=False,
        )
    else:
        module_dict["move_inputs"] = Mod(
            lambda *x: x,
            in_keys=["tokens", "attention_mask"],
            out_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
            # It's ok if there's no mask
            strict=False,
            inplace=False,
        )

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

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        out_keys=[("tokens_in", "input_ids_list")],
        strict=False,
    )

    if generate_kwargs is None:
        generate_kwargs = {
            "detokenize": False,
            "prompt_logprobs": not generate,
            "logprobs": return_log_probs,
        }
    if not generate:
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

    def get_output_tokens_and_log_probs(td):
        td["tokens_out"] = _RequestOutput_tc.from_request_output(td["tokens_out"])
        if generate:
            # When not generate, we don't want to overwrite this
            td["tokens_response"] = td["tokens_out"].outputs.token_ids
            if return_log_probs:
                td["log_probs"] = td["tokens_out"].outputs.logprobs.unsqueeze(-1)
        elif not generate:
            td["prompt_logprobs"] = td["tokens_out"].prompt_logprobs.unsqueeze(-1)
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    if not generate:

        def translate_lps(tokens_response, x):
            # we disregard the tokens from the prompt to focus on those of the response
            return x[..., -tokens_response.shape[-1] :, :]

        module_dict["translate_lps"] = Mod(
            translate_lps,
            in_keys=[("tokens_response", "input_ids"), "prompt_logprobs"],
            out_keys=["log_probs"],
        )
    elif from_text:
        module_dict["decode"] = Mod(
            tokenizer.batch_decode,
            in_keys=["tokens_response"],
            out_keys=["text_response"],
        )

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    if generate:
        module_dict["format"] = Mod(
            lambda *x: x,
            in_keys=[
                "log_probs",
                "tokens_response",
                ("tokens_in", "input_ids"),
                ("tokens_in", "attention_mask"),
                "text_response",
            ],
            out_keys=[
                "log_probs",
                "tokens_response",
                token_key,
                attention_mask_key,
                "text_response",
            ],
            strict=False,
            inplace=False,
        )
    else:
        module_dict["format"] = Mod(
            lambda *x: x,
            in_keys=["log_probs", "tokens_response"],
            out_keys=["log_probs", "tokens_response"],
            strict=False,
            inplace=False,
        )

    return Seq(module_dict, inplace=True)


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
