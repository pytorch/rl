from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel

HERE = Path(__file__).parent


def init_transformer(
    name_or_path,
    dropout,
    device,
    compile_,
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

    if compile_:
        # TODO: logging instead of printing?
        print("Compiling transformer model...")
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
