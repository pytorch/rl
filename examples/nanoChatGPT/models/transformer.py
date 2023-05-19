from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel

from .utils import crop_block_size, print_trainable_parameters

HERE = Path(__file__).parent

DEFAULT_VOCAB_SIZE = 50_304


def forward_wrap(out_forward=GPT2LMHeadModel.forward):
    """Return a wrapped instance method"""

    def forward(self, *args):
        if len(args) == 2:
            return_value = out_forward(self, input_ids=args[0], labels=args[1])
            return return_value.logits, return_value.loss
        else:
            return_value = out_forward(self, input_ids=args[0])
            return return_value.logits

    return forward


GPT2LMHeadModel.forward = forward_wrap()


def init_transformer(config, as_tensordictmodule=True, skip_compilation=False):
    model_kwargs = {
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
    }
    assert config["base_model"].startswith("gpt2")

    if config["init_base_from"] in ["scratch", "pretrained"]:
        model = GPT2LMHeadModel.from_pretrained(config["base_model"], **model_kwargs)
        if config["init_base_from"] == "scratch":
            model.post_init()
    elif config["init_base_from"] == "resume":
        model = GPT2LMHeadModel.from_pretrained(config["out_dir"], **model_kwargs)
    else:
        raise ValueError(f"option {config['init_base_from']=} not recognised")

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.n_positions:
        print(
            f"cropping model from block_size {model.config.n_positions} to {config['block_size']}"
        )
        crop_block_size(model, config["block_size"])
    print_trainable_parameters(model)

    model.to(config["device"])
    # compile the model
    if not skip_compilation and config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    if as_tensordictmodule:
        model = TensorDictModule(
            model, in_keys=["prompt", "target"], out_keys=["logits", "loss"]
        )
    return model
