# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from torchrl.data import (
    DEVICE_TYPING,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)

__all__ = ["make_replay_buffer"]


def make_replay_buffer(device: DEVICE_TYPING, args: DictConfig) -> ReplayBuffer:
    """Builds a replay buffer using the arguments build from the parser returned by ReplayArgsConfig."""
    device = torch.device(device)
    if not args.prb:
        buffer = TensorDictReplayBuffer(
            args.buffer_size,
            # collate_fn=InPlaceSampler(device),
            pin_memory=device != torch.device("cpu"),
            prefetch=3,
        )
    else:
        buffer = TensorDictPrioritizedReplayBuffer(
            args.buffer_size,
            alpha=0.7,
            beta=0.5,
            # collate_fn=InPlaceSampler(device),
            pin_memory=device != torch.device("cpu"),
            prefetch=3,
        )
    return buffer


@dataclass
class ReplayArgsConfig:
    buffer_size: int = 1000000
    # buffer size, in number of frames stored. Default=1e6
    prb: bool = False
    # whether a Prioritized replay buffer should be used instead of a more basic circular one.
