from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import tensorclass
from torch.utils.data import Dataset

from .utils import create_infinite_dataloader

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0).contiguous()
        batch.batch_size = []
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PairedDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitem__(self, idx):
        return Data(
            prompt=torch.from_numpy(
                self._memmap[idx : idx + self.block_size].astype(np.int64)
            ),
            target=torch.from_numpy(
                self._memmap[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            ),
            batch_size=[self.block_size],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size


def create_datasets(config):
    data_dir = HERE.parent / "models" / "nanoGPT" / "data" / config["dataset"]
    train_data = PairedDataset(data_dir / "train.bin", block_size=config["block_size"])
    val_data = PairedDataset(data_dir / "val.bin", block_size=config["block_size"])

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
