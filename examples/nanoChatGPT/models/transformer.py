from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel

from .utils import crop_block_size, print_trainable_parameters

HERE = Path(__file__).parent

DEFAULT_VOCAB_SIZE = 50_304


class GPT2(nn.Module):
    def __init__(self, model_path="gpt2", **kwargs):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_path, **kwargs)

    @property
    def config(self):
        return self.gpt2.config

    @property
    def transformer(self):
        return self.gpt2.transformer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        if labels is not None:
            return output.logits, output.loss
        return output.logits

    def generate(
        self, input_ids, attention_mask, generation_config, logits_processor=None
    ):
        return self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

    @classmethod
    def from_pretrained(cls, path):
        return cls(path)

    def save_pretrained(self, path):
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        self.gpt2.save_pretrained(save_dir)


def init_transformer(config, as_tensordictmodule=True, skip_compilation=False):
    model_kwargs = {
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
        "summary_first_dropout": config["dropout"],
        # "n_positions": 1024,
    }

    # TODO: do we need to support "scratch"
    # TODO: init_base_from redundant? replace with transformer_path which can either
    # be "gpt2" or a path to a checkpoint
    if config["init_base_from"] in ["scratch", "pretrained"]:
        model = GPT2(config["base_model"], **model_kwargs)
        if config["init_base_from"] == "scratch":
            model.gpt2.post_init()
    elif config["init_base_from"] == "resume":
        model = GPT2.from_pretrained(config["out_dir"])
    else:
        raise ValueError(f"option {config['init_base_from']=} not recognised")

    # crop down the model block size if desired, using model surgery
    # if config["block_size"] < model.config.n_positions:
    #     print(
    #         f"cropping model from block_size {model.config.n_positions} to {config['block_size']}"
    #     )
    #     crop_block_size(model, config["block_size"])
    # print_trainable_parameters(model)

    model.to(config["device"])
    # compile the model
    if not skip_compilation and config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    if as_tensordictmodule:
        model = TensorDictModule(
            model,
            in_keys=[
                ("transformer_data", "input_ids"),
                ("transformer_data", "attention_mask"),
                ("transformer_data", "labels"),
            ],
            out_keys=[("transformer_data", "logits"), "loss"],
        )
    return model
