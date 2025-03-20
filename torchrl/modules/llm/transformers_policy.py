# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch

from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule as Mod, TensorDictModuleBase, WrapModule
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.data.llm import LLMData
from torchrl.modules.llm.common import CategoricalSequential


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


def log_probs_generate(td: TensorDictBase) -> TensorDictBase:
    """Computes the log_probs from a Transformer formatted TensorDict.

    Required keys in tensordict:

    - "tokens_out": containing

        - "scores": logits of shape (B, seq-len, vocab_size)
        - "sequences": token sequences of shape (B, seq-len)

    Written keys in tensordict:

    - "logits": normalized scores of shape (B, seq-len, vocab_size)
    - "log_probs": log probabilities of shape (B, seq-len, 1)

    Note: The following keys will be deleted from the tensordict:

    - "tokens_out", "past_key_values"
    - "tokens_out", "scores"

    """
    tokens_out = td["tokens_out", "sequences"]
    seq_len = tokens_out.shape[1]

    del td["tokens_out", "past_key_values"]

    logits = dict(td["tokens_out", "logits"].items())
    logits = torch.stack(
        [logits[str(k)] for k in range(len(logits))], 1
    )  # shape (B, seq-len, vocab_size)
    # logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    logits = logits.log_softmax(dim=-1)
    torch.testing.assert_close(
        logits.logsumexp(-1), torch.zeros_like(logits.logsumexp(-1))
    )
    td["logits"] = logits[..., -seq_len:, :]
    del td["tokens_out", "logits"]

    tokens = tokens_out[..., -seq_len:]  # shape (B, seq-len)
    log_probs = logits.gather(-1, tokens.unsqueeze(-1))
    td["log_probs"] = log_probs.squeeze(-1)
    return td


def log_probs_from_logits(td: TensorDictBase) -> TensorDictBase:
    """Computes the log_probs from a Transformer formatted TensorDict.

    Required keys in tensordict:

    - "forward": containing
        - "logits": logits of shape (B, seq-len, vocab_size)
    - "tokens_in": containing
        - "input_ids": token sequences of shape (B, seq-len)

    Written keys in tensordict:

    - "logits": normalized scores of shape (B, seq-len, vocab_size)
    - "log_probs": log probabilities of shape (B, seq-len, 1)

    Note: The following keys will be deleted from the tensordict:
    - "forward", "past_key_values"
    - "forward"
    """
    tokens = td["tokens_response", "input_ids"]
    seq_len = tokens.shape[-1]

    logits = td["forward", "logits"]
    # logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    logits = logits.log_softmax(dim=-1)
    torch.testing.assert_close(
        logits.logsumexp(-1), torch.zeros_like(logits.logsumexp(-1))
    )
    logits = logits[..., -seq_len - 1 : -1, :]
    td["logits"] = logits
    del td["forward"]
    td["tokens_response"] = td["tokens_response", "input_ids"]

    log_probs = logits.gather(-1, tokens.unsqueeze(-1))
    td["log_probs"] = log_probs.squeeze(-1)

    return td


def from_hf_transformers(
    model: transformers.modeling_utils.PreTrainedModel,  # noqa
    *,
    generate: bool = True,
    return_log_probs: bool = True,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
    | None = None,  # noqa
    from_text: bool = False,
    device: torch.device | None = None,
    kwargs: dict | None = None,
    tokenizer_kwargs: dict | None = None,
) -> TensorDictModuleBase:
    """Creates a TensorDictModule from a Hugging Face Transformers model.

    This allows for a consistent interface across various LLM engines.
    This function facilitates text generation and log probability computation.

    Args:
        model (transformers.modeling_utils.PreTrainedModel): The Hugging Face model to wrap.
        generate (bool, optional): Whether to generate text. Defaults to `True`.
        return_log_probs (bool, optional): Whether to return log probabilities. Defaults to `True`.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer, optional): The tokenizer to use. Defaults to `None`.
        from_text (bool, optional): Whether the input is text. Defaults to `False`.
        device (torch.device, optional): The device to use for computation. Defaults to `None`.
        kwargs (dict, optional): Additional arguments for the model's generate method. Defaults to `None`.
        tokenizer_kwargs (dict, optional): Additional arguments for the tokenizer. Defaults to `None`.

    Returns:
        TensorDictModuleBase: A configured TensorDictModule for the specified model.

    Input Keys:

        - If `from_text` is `True`:

            - "text": The input text to be tokenized.

        - If `from_text` is `False`:

            - "tokens": The input token sequences.
            - "attention_mask": The attention mask for the tokens.

    Output Keys:

        - "tokens_response": The generated token sequences.
        - "log_probs": The log probabilities of the generated tokens (if `return_log_probs` is `True`).
        - "logits": The logits of the generated tokens (if applicable).
        - "text_response": The generated text (if `from_text` is `True` and `generate` is `True`).

    Example:
        >>> from tensordict.tensorclass import NonTensorStack
        >>> from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
        >>>
        >>> from torchrl.data import LLMData
        >>> from torchrl.modules import from_hf_transformers
        >>>
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = GPT2LMHeadModel(GPT2Config())
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>>
        >>> module = from_hf_transformers(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     from_text=True,
        ...     generate=True
        ... )
        >>> input_data = LLMData(text=NonTensorStack("Hello, world!"), batch_size=1)
        >>> output_data = module(input_data)
        >>> print(output_data.text_response)
        [' heritageillon rooft rooft Pear Tes grantingalde 58ocrocrocrocrcubecubecubecubecubecubecube']

    .. seealso:: :func:`~torchrl.modules.from_vllm` for a similar interface using the vLLM library.

    """
    # TODO: Seq should have a return_log_prob and be of ProbabilisticTDSequential type for instance checks

    # Keys:
    text_key: NestedKey = "text"
    if from_text:
        if generate:
            func = _from_hf_generate_text
        else:
            func = _from_hf_lp_text
    else:
        if generate:
            func = _from_hf_generate_tokens
        else:
            func = _from_hf_lp_tokens
    module_dict = func(
        device=device,
        tokenizer_kwargs=tokenizer_kwargs,
        text_key=text_key,
        tokenizer=tokenizer,
        kwargs=kwargs,
        return_log_probs=return_log_probs,
        model=model,
    )
    return CategoricalSequential(module_dict, inplace=True)


def _from_hf_generate_text(
    *, device, tokenizer_kwargs, text_key, tokenizer, kwargs, return_log_probs, model
) -> dict:
    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    if not tokenizer_kwargs:
        tokenizer_kwargs = {}
    if not tokenizer_kwargs.setdefault("return_attention_mask", True):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("return_tensors", "pt") != "pt":
        raise RuntimeError
    # TODO: add other paddings
    if tokenizer_kwargs.setdefault("padding", True) not in (True,):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
        raise RuntimeError

    module_dict["encode"] = Mod(
        tokenizer,
        in_keys=[text_key],
        out_keys=["tokens_in"],
        method_kwargs=tokenizer_kwargs,
        strict=True,
        # We don't need the text after this
        inplace=False,
    )

    if device:
        module_dict["to_dest_device"] = Mod(
            lambda tensor: tensor.to(device),
            in_keys=["tokens_in"],
            out_keys=["tokens_in"],
            strict=True,
        )

    if not kwargs:
        kwargs = {}
    if return_log_probs:
        if not kwargs.setdefault("output_logits", True):
            raise RuntimeError
    if not kwargs.setdefault("return_dict_in_generate", True):
        raise RuntimeError
    if (
        kwargs.setdefault("tokenizer", tokenizer) is not tokenizer
        and tokenizer is not None
    ):
        raise RuntimeError

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs=kwargs,
        in_keys={
            "input_ids": ("tokens_in", "input_ids"),
            "attention_mask": ("tokens_in", "attention_mask"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    # Keep only the new tokens
    def remove_input_seq(tokens_in, tokens_out, tokenizer=tokenizer):
        result = tokens_out[..., tokens_in.shape[-1] :]
        return result

    module_dict["remove_input_seq"] = Mod(
        remove_input_seq,
        in_keys=[("tokens_in", "input_ids"), ("tokens_out", "sequences")],
        out_keys=[("tokens_out", "sequences")],
        strict=True,
    )
    if return_log_probs:
        module_dict["extract_log_probs"] = WrapModule(
            log_probs_generate,
            in_keys=[("tokens_out", "sequences"), ("tokens_out", "scores")],
            out_keys=["logits", "log_probs"],
        )
    module_dict["decode"] = Mod(
        tokenizer.batch_decode,
        in_keys=[("tokens_out", "sequences")],
        out_keys=["text_out"],
        strict=True,
    )
    if device:
        module_dict["to_source_device"] = _maybe_set_device

    module_dict["rebuild"] = Mod(
        lambda *x: x,
        in_keys=[
            ("tokens_out", "sequences"),
            ("tokens_in", "input_ids"),
            ("tokens_in", "attention_mask"),
            "text_out",
            "log_probs",
            "logits",
        ],
        out_keys=[
            "tokens_response",
            "tokens",
            "attention_mask",
            "text_response",
            "log_probs",
            "logits",
        ],
        # There may not be log_probs and logits
        strict=False,
        inplace=False,
    )
    return module_dict


def _from_hf_generate_tokens(
    *, device, tokenizer_kwargs, text_key, tokenizer, kwargs, return_log_probs, model
) -> dict:
    # Keys:
    token_key: NestedKey = "tokens"
    attention_mask_key: NestedKey = "attention_mask"

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    module_dict["format"] = Mod(
        lambda *x: x,
        in_keys=[token_key, attention_mask_key],
        out_keys=[("tokens_in", "input_ids"), ("tokens_in", "attention_mask")],
        strict=False,
        # We don't need the text after this
        inplace=False,
    )

    if device:
        module_dict["to_dest_device"] = Mod(
            lambda tensor: tensor.to(device),
            in_keys=["tokens_in"],
            out_keys=["tokens_in"],
            strict=True,
        )

    if not kwargs:
        kwargs = {}
    if return_log_probs:
        # if not kwargs.setdefault("output_scores", True):
        #     raise RuntimeError
        if not kwargs.setdefault("output_logits", True):
            raise RuntimeError
    if not kwargs.setdefault("return_dict_in_generate", True):
        raise RuntimeError
    if (
        kwargs.setdefault("tokenizer", tokenizer) is not tokenizer
        and tokenizer is not None
    ):
        raise RuntimeError

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs=kwargs,
        in_keys={
            "input_ids": ("tokens_in", "input_ids"),
            "attention_mask": ("tokens_in", "attention_mask"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    # Keep only the new tokens
    def remove_input_seq(tokens_in, tokens_out):
        # TODO: remove this assert
        assert (tokens_out[..., : tokens_in.shape[-1]] == tokens_in).all()
        result = tokens_out[..., tokens_in.shape[-1] :]
        return result

    module_dict["remove_input_seq"] = Mod(
        remove_input_seq,
        in_keys=[("tokens_in", "input_ids"), ("tokens_out", "sequences")],
        out_keys=[("tokens_out", "sequences")],
        strict=True,
    )
    if return_log_probs:
        module_dict["extract_log_probs"] = WrapModule(
            log_probs_generate,
            in_keys=[("tokens_out", "sequences"), ("tokens_out", "logits")],
            out_keys=["logits", "log_probs"],
        )
    if device:
        module_dict["to_source_device"] = _maybe_set_device
    module_dict["rebuild"] = Mod(
        lambda *x: x,
        in_keys=[("tokens_out", "sequences"), "log_probs", "logits"],
        out_keys=["tokens_response", "log_probs", "logits"],
        inplace=False,
    )
    return CategoricalSequential(module_dict, inplace=True)


def _from_hf_lp_text(
    *, device, tokenizer_kwargs, text_key, tokenizer, kwargs, return_log_probs, model
) -> dict:
    # Keys:
    text_key: NestedKey = "text"

    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device
    if not tokenizer_kwargs:
        tokenizer_kwargs = {}
    if not tokenizer_kwargs.setdefault("return_attention_mask", True):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("return_tensors", "pt") != "pt":
        raise RuntimeError
    # TODO: add other paddings
    if tokenizer_kwargs.setdefault("padding", True) not in (True,):
        raise RuntimeError
    if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
        raise RuntimeError

    def encode(input_txt: list[str], output_txt: list[str]):
        input_tokens = TensorDict.from_dict(
            tokenizer(
                [_x + _y for _x, _y in zip(input_txt, output_txt)],
                **tokenizer_kwargs,
            )
        )
        # TODO: if not generating, we should use the tokens generates before
        #  because encode-decode isn't strictly a bijection so encoding the response
        #  only may lead to surprises
        input_only_tokens = TensorDict.from_dict(
            tokenizer(input_txt, **tokenizer_kwargs)
        )
        output_tokens = input_tokens.apply(
            lambda x, y: x[:, y.shape[1] :], input_only_tokens
        )
        return input_tokens, output_tokens

    module_dict["encode"] = Mod(
        # TODO: make this work with many strings
        # Tokenize both strings, and only the first
        encode,
        in_keys=[text_key, "text_response"],
        out_keys=["tokens_in", "tokens_response"],
        strict=True,
        inplace=False,
    )

    if device:
        module_dict["to_dest_device"] = Mod(
            lambda tensor: tensor.to(device),
            in_keys=["tokens_in"],
            out_keys=["tokens_in"],
            strict=True,
        )

    if not kwargs:
        kwargs = {}
    if not kwargs.setdefault("return_dict", True):
        raise RuntimeError
    if return_log_probs not in (True, None):
        raise RuntimeError(
            "return_log_probs should be True or None when not generating."
        )
    module_dict["get_logprobs"] = Mod(
        model,
        method_kwargs=kwargs,
        in_keys={
            "input_ids": ("tokens_in", "input_ids"),
            "attention_mask": ("tokens_in", "attention_mask"),
        },
        out_keys=["forward"],
        out_to_in_map=True,
        strict=True,
    )
    module_dict["extract_log_probs"] = WrapModule(
        log_probs_from_logits,
        in_keys=[
            ("tokens_in", "input_ids"),
            ("forward", "logits"),
            ("tokens_response", "logits"),
        ],
        out_keys=["logits", "log_probs"],
    )
    if device:
        module_dict["to_source_device"] = _maybe_set_device
    module_dict["rebuild"] = Mod(
        lambda *x: x,
        in_keys=["log_probs", "logits", "tokens_response"],
        out_keys=["log_probs", "logits", "tokens_response"],
        inplace=False,
        strict=True,
    )
    return module_dict


def _from_hf_lp_tokens(
    *, device, tokenizer_kwargs, text_key, tokenizer, kwargs, return_log_probs, model
) -> dict:
    module_dict = {}
    if device:
        module_dict["clear_device"] = _maybe_clear_device

    def stack_for_logprobs(tokens, tokens_response, attention_mask=None):
        tokens = torch.cat([tokens, tokens_response], -1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(tokens_response.shape)], -1
            )
        # We need to carry over tokens_response to retrieve the shape of the response
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
        # We can also work with lists, just ask for lists through as_list=True
        get_kwargs={
            "as_padded_tensor": True,
            "padding_side": "left",
        },
    )

    if device:
        module_dict["to_dest_device"] = Mod(
            lambda tensor: tensor.to(device),
            in_keys=["tokens_in"],
            out_keys=["tokens_in"],
            strict=True,
        )

    if not kwargs:
        kwargs = {}
    if not kwargs.setdefault("return_dict", True):
        raise RuntimeError
    if return_log_probs not in (True, None):
        raise RuntimeError(
            "return_log_probs should be True or None when not generating."
        )
    module_dict["get_logprobs"] = Mod(
        model,
        method_kwargs=kwargs,
        in_keys={
            "input_ids": ("tokens_in", "input_ids"),
            "attention_mask": ("tokens_in", "attention_mask"),
            # "labels": ("tokens_in", "input_ids"),
        },
        out_keys=["forward"],
        out_to_in_map=True,
        strict=True,
    )
    module_dict["extract_log_probs"] = WrapModule(
        log_probs_from_logits,
        in_keys=[
            ("tokens_in", "input_ids"),
            ("forward", "logits"),
            ("tokens_response", "logits"),
        ],
        out_keys=["logits", "log_probs"],
    )
    if device:
        module_dict["to_source_device"] = _maybe_set_device
    module_dict["rebuild"] = Mod(
        lambda *x: x,
        in_keys=["log_probs", "logits"],
        out_keys=["log_probs", "logits"],
        inplace=False,
    )

    return module_dict


if __name__ == "__main__":
    import transformers
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    max_seq_length = 50000

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2Config()).eval()

    tokenizer.padding_side = "left"

    m = from_hf_transformers(model, tokenizer=tokenizer, from_text=True, generate=True)
    td = m(LLMData(text=NonTensorStack("a text"), batch_size=1))

    m = from_hf_transformers(model, tokenizer=tokenizer, from_text=True, generate=False)
    td = m(LLMData(text=NonTensorStack("a text"), batch_size=1))

    m = from_hf_transformers(model, tokenizer=tokenizer, from_text=False, generate=True)
    td = m(
        LLMData(
            tokens=torch.randint(1024, (1, 10)),
            attention_mask=torch.ones(1, 10, dtype=torch.int64),
            batch_size=1,
        )
    )

    m = from_hf_transformers(model, tokenizer=tokenizer, from_text=False, generate=True)
    td = m(LLMData(tokens=torch.randint(1024, (1, 10)), batch_size=1))
