# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from tensordict.nn import TensorDictModule
from torchrl._utils import logger as torchrl_logger
from transformers import GPT2LMHeadModel


def init_transformer(
    name_or_path,
    dropout,
    device,
    compile_model,
    as_tensordictmodule=True,
    inference=False,
):
    model_kwargs = {
        "resid_pdrop": dropout,
        "embd_pdrop": dropout,
        "attn_pdrop": dropout,
        "summary_first_dropout": dropout,
    }
    model = GPT2LMHeadModel.from_pretrained(
        name_or_path, return_dict=False, **model_kwargs
    )
    model.to(device)

    if compile_model:
        torchrl_logger.info("Compiling transformer model...")
        model = torch.compile(model)

    if as_tensordictmodule:
        model = TensorDictModule(
            model,
            in_keys={
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels": "labels",
            },
            out_keys=["logits"] if inference else ["loss", "logits"],
        )
    return model
