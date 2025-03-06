# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import transformers
import vllm.outputs
from tensordict import (
    from_dataclass,
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
)
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

CompletionOutput_tc = from_dataclass(vllm.outputs.CompletionOutput)


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
            out_keys=["tokens_in"],
            # method_kwargs=tokenizer_kwargs,
            strict=True,
        )

    def to_list(tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        print("tokens", tokens)
        return NonTensorStack(*tokens)

    module_dict["to_list"] = Mod(
        to_list,
        in_keys=[("tokens_in", "input_ids")],
        out_keys=[("tokens_in", "input_ids_list")],
    )

    if generate_kwargs is None:
        generate_kwargs = {"detokenize": False, "prompt_logprobs": 1, "logprobs": 1}
    sampling_params = SamplingParams(**generate_kwargs)

    module_dict["generate"] = Mod(
        model,
        method="generate",
        method_kwargs={"sampling_params": sampling_params},
        in_keys={
            "prompt_token_ids": ("tokens_in", "input_ids_list"),
            # "attention_mask": ("tokens_in", "attention_mask"),
        },
        out_keys=["tokens_out"],
        out_to_in_map=True,
        strict=True,
    )

    def get_output_tokens_and_log_probs(td):
        td["tokens_out"] = RequestOutput_tc.from_request_output(td["tokens_out"])
        td["output_tokens"] = td["tokens_out"].outputs.token_ids
        td["log_probs"] = td["tokens_out"].outputs.token_ids
        return td

    module_dict["get_output_tokens_and_log_probs"] = get_output_tokens_and_log_probs

    if from_text:
        module_dict["to_list_decode"] = Mod(
            to_list, in_keys=[("output_tokens")], out_keys=[("output_tokens_list")]
        )
        module_dict["decode"] = Mod(
            tokenizer.batch_decode,
            in_keys=["output_tokens_list"],
            out_keys=["action"],
        )

    if device:
        module_dict["to_source_device"] = _maybe_set_device

    return Seq(module_dict)


class RequestOutput_tc(TensorClass["nocast"]):
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
            print("local", output)

            def get_logprob(output):
                t = []
                for v, tid in zip(output.logprobs, output.token_ids):
                    t.append(
                        v[tid]["logprob"] if v[tid].get("logprob") is not None else 0.0
                    )
                return torch.tensor(t)

            output.logprobs = get_logprob(output)
            print("token ids before transform", output.token_ids)
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
                    for v, tid in zip(self.prompt_logprobs, self.prompt_token_ids)
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
        print("result of from_request_output", out)
        return out


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
    print(m(TensorDict(text=NonTensorStack("a text is a text", "another text"))))
