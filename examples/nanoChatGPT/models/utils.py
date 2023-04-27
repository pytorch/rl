from pathlib import Path

import torch


def _remove_state_dict_prefixes(state_dict, unwanted_prefixes=("_orig_mod.", "model.")):
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    for unwanted_prefix in unwanted_prefixes:
        for k in list(state_dict):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)


def load_checkpoint(checkpoint_dir, device):
    ckpt_path = Path(checkpoint_dir) / "ckpt.pt"
    return torch.load(ckpt_path, map_location=device)
