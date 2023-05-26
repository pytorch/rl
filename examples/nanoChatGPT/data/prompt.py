# download and prepare the openai_summarize_tldr dataset for fine tuning transformers
# adapted from
# https://github.com/sanjeevanahilan/nanoChatGPT/blob/3cde2746c7ea8b0bd32edd44c76ead581bbda5d5/data/openai_summarize_tldr/prepare.py
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tensordict import tensorclass
from torch.utils.data import Dataset


HERE = Path(__file__).parent


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PromptDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitems__(self, idx):
        idx = torch.tensor(idx).unsqueeze(1) + torch.arange(self.block_size).unsqueeze(
            0
        )

        return Data(
            prompt=torch.from_numpy(self._memmap[idx[:]].astype(np.int64)).view_as(idx),
            target=torch.from_numpy(self._memmap[idx[:] + 1].astype(np.int64)).view_as(
                idx
            ),
            batch_size=[],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size
