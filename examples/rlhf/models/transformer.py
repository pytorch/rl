from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel

HERE = Path(__file__).parent


def init_transformer(
    config, as_tensordictmodule=True, skip_compilation=False, inference=False
):
    model_kwargs = {
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
        "summary_first_dropout": config["dropout"],
    }
    model = GPT2LMHeadModel.from_pretrained(
        config["transformer_name_or_path"], return_dict=False, **model_kwargs
    )
    model.to(config["device"])

    # compile the model
    if not skip_compilation and config["compile"]:
        # TODO: logging instead of printing?
        print("Compiling transformer model...")
        model = torch.compile(model)

    if as_tensordictmodule:
        model = TensorDictModule(
            model,
            in_keys={
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "labels": "labels"
            },
            out_keys=["logits"] if inference else ["loss", "logits"],
        )
    return model
