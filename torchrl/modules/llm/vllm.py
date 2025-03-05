# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import transformers
from tensordict import NestedKey, NonTensorData, NonTensorStack, TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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
    model: LLM,
    return_log_probs: bool = False,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer | None = None,
    from_text: bool = False,
    device: torch.device | None = None,
    text_key: NestedKey = "text",
    generate_kwargs: dict | None = None,
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
        if tokenizer_kwargs.setdefault("padding", True) not in (True,):
            raise RuntimeError
        if tokenizer_kwargs.setdefault("padding_side", "left") != "left":
            raise RuntimeError
        module_dict["encode"] = Mod(
            tokenizer,
            in_keys=[text_key],
            out_keys=["tokens_in"],  # method_kwargs=tokenizer_kwargs,
            strict=True,
        )

    # FIXME: this is not great!
    def f(td):
        td["tokens_in", "input_ids"] = NonTensorStack(
            *td["tokens_in", "input_ids"].tolist()
        )
        print("td['tokens_in', 'input_ids']", td["tokens_in", "input_ids"])
        return td

    module_dict["to_list"] = f

    if generate_kwargs is None:
        generate_kwargs = {
            "detokenize": False,
            "prompt_logprobs": return_log_probs,
            "logprobs": return_log_probs,
        }
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids"),
            # "attention_mask": ("tokens_in", "attention_mask"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td):
        # FIXME: shouldn't have to be doing 0 index here to make sure this works with batches
        td["output_tokens"] = td["tokens_out"][0].outputs[0].token_ids
        # FIXME: this is not in a tensor form yet but uses their own LogProb object
        td["log_probs"] = td["tokens_out"][0].outputs[0].logprobs
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    # module_dict["extract_log_probs"] = WrapModule(log_probs_from_logits, in_keys=[("tokens_in", "sequences"), ("tokens_in", "scores")], out_keys=["logits", "log_probs"])
    if from_text:
        module_dict["decode"] = Mod(
            tokenizer.batch_decode,
            in_keys=["output_tokens"],  # in_keys=["tokens_out", "sequences"],
            out_keys=["action"],  # strict=True,
        )
    if device:
        module_dict["to_source_device"] = _maybe_set_device

    return Seq(module_dict)


if __name__ == "__main__":
    max_seq_length = 50000
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = LLM(model_name, skip_tokenizer_init=True, device="cuda:0")
    model.llm_engine.model_executor.driver_worker.worker.model_runner.model.sampler.include_gpu_probs_tensor = (
        True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device="cuda:0")
    # tokenizer.padding_side = "left"
    m = from_vllm(model, tokenizer=tokenizer, from_text=True, device="cuda:0")
    print(m(TensorDict(text="a text is a text")))
