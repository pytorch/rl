# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch

from torchrl.data import (
    DEVICE_TYPING,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage

__all__ = ["make_replay_buffer"]



class collate_fn:
    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        return x.to(self.device)


def make_replay_buffer(device: DEVICE_TYPING, cfg: "DictConfig") -> ReplayBuffer:
    """Builds a replay buffer using the config built from ReplayArgsConfig."""
    device = torch.device(device)
    if not cfg.prb:
        buffer = TensorDictReplayBuffer(
            cfg.buffer_size,
            collate_fn=collate_fn(device),
            pin_memory=device != torch.device("cpu"),
            prefetch=cfg.buffer_prefetch,
            storage=LazyTensorStorage(
                cfg.buffer_size,
                scratch_dir=cfg.buffer_scratch_dir,
            ),
        )
    else:
        buffer = TensorDictPrioritizedReplayBuffer(
            cfg.buffer_size,
            alpha=0.7,
            beta=0.5,
            collate_fn=collate_fn(device),
            pin_memory=device != torch.device("cpu"),
            prefetch=cfg.buffer_prefetch,
            storage=LazyTensorStorage(
                cfg.buffer_size,
                scratch_dir=cfg.buffer_scratch_dir,
            ),
        )
    return buffer


@dataclass
class ReplayArgsConfig:
    buffer_size: int = 1000000
    # buffer size, in number of frames stored. Default=1e6
    prb: bool = False
    # whether a Prioritized replay buffer should be used instead of a more basic circular one.
    buffer_scratch_dir: Optional[str] = None
    # directory where the buffer data should be stored. If none is passed, they will be placed in /tmp/
    buffer_prefetch: int = 10
    # prefetching queue length for the replay buffer
