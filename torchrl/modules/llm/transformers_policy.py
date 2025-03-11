# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: lazy imports

import torch

import transformers
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
    WrapModule,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.data.llm import LLMData
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


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


def log_probs_from_scores(td: TensorDictBase) -> TensorDictBase:
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
    # TODO: how do we avoid getting these?
    del td["tokens_out", "past_key_values"]
    scores = dict(td["tokens_out", "scores"].items())
    scores = torch.stack(
        [scores[str(k)] for k in range(len(scores))], 1
    )  # shape (B, seq-len, vocab_size)
    logits = scores - scores.logsumexp(dim=-1, keepdim=True)
    td["logits"] = scores
    del td["tokens_out", "scores"]
    seq_len = scores.shape[1]
    tokens = td["tokens_out", "sequences"][..., -seq_len:]  # shape (B, seq-len)
    log_probs = logits.gather(-1, tokens.unsqueeze(-1))
    td["log_probs"] = log_probs
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
    # TODO: how do we avoid getting these?
    del td["forward", "past_key_values"]
    scores = td["forward", "logits"]
    logits = scores - scores.logsumexp(dim=-1, keepdim=True)
    td["logits"] = scores
    del td["forward"]
    scores.shape[1]
    tokens = td["tokens_in", "input_ids"]
    log_probs = logits.gather(-1, tokens.unsqueeze(-1))
    td["log_probs"] = log_probs
    return td


def from_hf_transformers(
    model: transformers.modeling_utils.PreTrainedModel,
    *,
    generate: bool = True,
    return_log_probs: bool = True,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer | None = None,
    from_text: bool = False,
    device: torch.device | None = None,
    # Keys:
    text_key: NestedKey = "text",
    token_key: NestedKey = "tokens",
    attention_mask_key: NestedKey = "attention_mask",
    kwargs: dict | None = None,
    tokenizer_kwargs: dict | None = None,
) -> TensorDictModuleBase:
    # TODO: Seq should have a return_log_prob and be of ProbabilisticTDSequential type for instance checks


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
    else:
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

    if generate:
        if not kwargs:
            kwargs = {}
        if return_log_probs:
            if not kwargs.setdefault("output_scores", True):
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
        if return_log_probs:
            module_dict["extract_log_probs"] = WrapModule(
                log_probs_from_scores,
                in_keys=[("tokens_out", "sequences"), ("tokens_out", "scores")],
                out_keys=["logits", "log_probs"],
            )
        if from_text:
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
                strict=True,
                inplace=False,
            )
        else:
            if device:
                module_dict["to_source_device"] = _maybe_set_device
            module_dict["rebuild"] = Mod(
                lambda *x: x,
                in_keys=[("tokens_out", "sequences"), "log_probs", "logits"],
                out_keys=["tokens_response", "log_probs", "logits"],
                inplace=False,
            )
    else:
        if not kwargs:
            kwargs = {}
        if not kwargs.setdefault("return_dict", True):
            raise RuntimeError
        if not return_log_probs:
            raise RuntimeError
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
            in_keys=[("tokens_in", "input_ids"), ("forward", "logits")],
            out_keys=["logits", "log_probs"],
        )
        if device:
            module_dict["to_source_device"] = _maybe_set_device
        if from_text:
            module_dict["rebuild"] = Mod(
                lambda *x: x,
                in_keys=["log_probs", "logits", ("tokens_in", "attention_mask")],
                out_keys=["log_probs", "logits", "attention_mask"],
                inplace=False,
            )
        else:
            module_dict["rebuild"] = Mod(
                lambda *x: x,
                in_keys=["log_probs", "logits"],
                out_keys=["log_probs", "logits"],
                inplace=False,
            )

    return Seq(module_dict, inplace=True)


if __name__ == "__main__":
    max_seq_length = 50000

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel(GPT2Config())

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
