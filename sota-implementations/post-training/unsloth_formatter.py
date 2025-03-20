# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# from transformers import AutoTokenizer
from __future__ import annotations

import gc
import os
import pickle
import re
import tempfile
from typing import Literal

import psutil
import torch
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear as Peft_Linear, Linear4bit as Peft_Linear4bit
from tensordict import TensorDict
from torchrl._utils import logger
from unsloth.kernels import fast_dequantize, get_lora_parameters_bias

LLAMA_LAYERNORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
LLAMA_WEIGHTS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _getattr(layer, name):
    name = name.split(".")
    obj = None
    for n in name:
        obj = getattr(layer, n)
        layer = obj
    return obj


def _merge_lora(layer, name):

    bias = getattr(layer, "bias", None)
    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
        # Is LoRA so we need to merge!
        W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)
        if quant_state is not None:
            dtype = (
                quant_state.dtype if type(quant_state) is not list else quant_state[2]
            )
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype
        W = W.to(torch.float32).t()
        # W = W.t()

        if A is not None:
            # sAB = (A.t().to(torch.float32) @ (s * B.t().to(torch.float32)))
            # W += sAB
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha=s)
            # W.addmm_(A.t().to(W.dtype), B.t().to(W.dtype), alpha = s)
            # if not torch.isfinite(W).all():
            maximum_element = torch.max(W.min().abs(), W.max())
            if not torch.isfinite(maximum_element).item():
                raise ValueError(
                    f"Unsloth: Merge failed.\n{name} has some elements = infinity."
                )
        W = W.t().to(dtype)
    else:
        W = layer.weight
    return W, bias


@torch.inference_mode
def unsloth_state_dict(
    model,
    tokenizer: transformers.AutoTokenizer | None = None,  # noqa
    state_dict: dict | None = None,
    max_shard_size: int | str = "5GB",
    variant: str | None = None,
    save_peft_format: bool = True,
    # Push to hub
    use_temp_dir: bool | None = None,
    commit_message: str | None = "Trained with Unsloth",
    private: bool | None = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: list[str] = None,
    # Our functions
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
    mem_location: Literal["cuda", "memmap", "ram"] = "ram",
):
    save_method = "merged_16bit"

    if commit_description is None:
        commit_description = "Upload model trained with Unsloth 2x faster"
    elif "Unsloth 2x faster" not in commit_description:
        commit_description += " (Trained with Unsloth 2x faster)"

    assert maximum_memory_usage > 0 and maximum_memory_usage <= 0.95

    # Clean memory up first
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()

    save_method = save_method.lower().replace(" ", "_")
    if (
        save_method != "lora"
        and save_method != "merged_16bit"
        and save_method != "merged_4bit"
    ):
        raise RuntimeError(
            "Unsloth: You must select one of 3 options when saving models:\n"
            '"lora"         ==> This is the fastest and easiet. Just saves LoRA modules.\n'
            '"merged_16bit" ==> This merges LoRA weights and saves to float16. Needed for llama.cpp / GGUF.\n'
            '"merged_4bit"  ==> This merges LoRA weights and saves to 4bit. Useful for DPO / inference.'
        )

    if tags is not None:
        assert isinstance(tags, (list, tuple))
        tags = list(tags) + [
            "unsloth",
        ]
    else:
        tags = [
            "unsloth",
        ]

    # Check if PEFT Model or not - if yes, 3 levels. If not 2 levels.
    from peft import PeftModelForCausalLM

    if isinstance(model, PeftModelForCausalLM):
        internal_model = model.model
    else:
        internal_model = model

    logger.info("Unsloth: Merging 4bit and LoRA weights to 16bit...")

    # Determine max RAM usage minus sharding
    max_ram = psutil.virtual_memory().available
    sharded_ram_usage = 5 * 1024 * 1024 * 1024
    if type(max_shard_size) is str:
        gb_found = re.match(
            r"([0-9]{1,})[\s]{0,}GB", max_shard_size, flags=re.IGNORECASE
        )
        mb_found = re.match(
            r"([0-9]{1,})[\s]{0,}MB", max_shard_size, flags=re.IGNORECASE
        )
        if gb_found:
            sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024
        elif mb_found:
            sharded_ram_usage = int(mb_found.group(1)) * 1024 * 1024
    elif type(max_shard_size) is int:
        sharded_ram_usage = sharded_ram_usage

    # Switch to our fast saving modules if it's a slow PC!
    n_cpus = psutil.cpu_count(logical=False)
    if n_cpus is None:
        n_cpus = psutil.cpu_count()
    if n_cpus is None:
        n_cpus = 1

    max_ram -= sharded_ram_usage * 0.25  # Uses much less

    max_ram = int(max(0, max_ram) * maximum_memory_usage)
    logger.info(
        f"Unsloth: Will use up to "
        f"{round(max_ram/1024/1024/1024, 2)} out of "
        f"{round(psutil.virtual_memory().total/1024/1024/1024, 2)} RAM for saving."
    )

    # Max directory for disk saving
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    # HF also uses a OrderedDict
    from collections import OrderedDict

    state_dict = OrderedDict()

    torch_dtype = internal_model.config.torch_dtype
    if type(torch_dtype) is str:
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16

    # Check modules to save float32 dtype
    state_dict[
        "model.embed_tokens.weight"
    ] = internal_model.model.embed_tokens.weight.data.to(torch_dtype)

    max_vram = int(
        torch.cuda.get_device_properties(0).total_memory * maximum_memory_usage
    )

    logger.info("Unsloth: Saving model... This might take 5 minutes ...")

    from tqdm import tqdm as ProgressBar
    if mem_location == "memmap":
        tmpdir = tempfile.mkdtemp()
        td = TensorDict().memmap_(tmpdir)

    for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = _getattr(layer, item)
            name = f"model.layers.{j}.{item}.weight"
            W, bias = _merge_lora(proj, name)

            # Bias term
            if bias is not None:
                state_dict[f"model.layers.{j}.{item}.bias"] = bias

            if mem_location == "cuda": # torch.cuda.memory_allocated() + W.nbytes) < max_vram:
                # Save to GPU memory
                state_dict[name] = W
            # [TODO] Saving to RAM seems to leak memory???
            elif mem_location == "ram":  # (max_ram - W.nbytes) > 0:
                # Save to CPU memory
                logger.warning_once("We will save to RAM and not VRAM now.")
                state_dict[name] = W.to("cpu", non_blocking=True, copy=True)
                max_ram = max(max_ram - W.nbytes, 0)
            elif mem_location == "memmap":
                td.make_memmap_from_tensor(name, W)
            else:
                # Save to Disk
                logger.warning_once("\nWe will save to Disk and not RAM now.")
                filename = os.path.join(temporary_location, f"{name}.pt")
                torch.save(
                    W,
                    filename,
                    pickle_module=pickle,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )
                # weights_only = True weirdly fails?
                state_dict[name] = torch.load(
                    filename, map_location="cpu", mmap=True, weights_only=False
                )
        for item in LLAMA_LAYERNORMS:
            try:
                # TODO: Skip for Gemma 2
                state_dict[f"model.layers.{j}.{item}.weight"] = _getattr(
                    layer, f"{item}.weight.data"
                )
            except AttributeError:
                pass

    state_dict["model.norm.weight"] = internal_model.model.norm.weight.data

    if mem_location == "memmap":
        state_dict = td.to_dict()
    # Check for modules_to_save float32 dtype

    # Check for tied weights
    if (
        internal_model.model.embed_tokens.weight.data_ptr()
        != internal_model.lm_head.weight.data_ptr()
    ):
        state_dict["lm_head.weight"] = internal_model.lm_head.weight.data.to(
            torch_dtype
        )

    # All tensors MUST be type torch.Tensor and not torch.nn.parameter.Parameter
    for key, value in state_dict.items():
        if hasattr(value, "data"):
            state_dict[key] = value = value.data
        if type(value) is not torch.Tensor:
            logger.warning_once(f"Unsloth: {key} is not a Tensor but a {type(value)}.")

    # commit_description does not seem to work?
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(
            [
                "unsloth",
            ]
        )
    return state_dict
