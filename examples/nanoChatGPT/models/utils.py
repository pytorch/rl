from pathlib import Path

import torch
import torch.nn as nn


def load_checkpoint(checkpoint_dir, device):
    ckpt_path = Path(checkpoint_dir) / "ckpt.pt"
    return torch.load(ckpt_path, map_location=device)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def crop_block_size(model, block_size):
    # model surgery to decrease the block size if necessary
    # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    # but want to use a smaller block size for some smaller, simpler model
    # model.transformer.wpe.weight = nn.Parameter(
    #     model.transformer.wpe.weight[:block_size]
    # )
    for block in model.transformer.h:
        if hasattr(block.attn, "bias"):
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
