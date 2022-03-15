from argparse import ArgumentParser, Namespace

__all__ = ["parser_replay_args", "make_replay_buffer"]

import torch

from torchrl.data import (
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    DEVICE_TYPING,
)
from torchrl.data.replay_buffers.replay_buffers import InPlaceSampler


def make_replay_buffer(device: DEVICE_TYPING, args: Namespace) -> ReplayBuffer:
    device = torch.device(device)
    if not args.prb:
        buffer = ReplayBuffer(
            args.buffer_size,
            collate_fn=InPlaceSampler(device),
            pin_memory=device != torch.device("cpu"),
        )
    else:
        buffer = TensorDictPrioritizedReplayBuffer(
            args.buffer_size,
            alpha=0.7,
            beta=0.5,
            collate_fn=InPlaceSampler(device),
            pin_memory=device != torch.device("cpu"),
        )
    return buffer


def parser_replay_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000000,
        help="buffer size, in number of frames stored. Default=1e6",
    )
    parser.add_argument(
        "--prb",
        action="store_true",
        help="whether a Prioritized replay buffer should be used instead of a more basic circular one.",
    )
    return parser
