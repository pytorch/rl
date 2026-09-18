# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .base import (
    _maybe_delay_init as _maybe_delay_init,
    _storage_index as _storage_index,
    ConditionalUpdateResult,
    ReplayBuffer,
)
from .blocking import BlockingReplayBuffer
from .deprecated import InPlaceSampler, stack_tensors
from .ensemble import ReplayBufferEnsemble
from .prioritized import PrioritizedReplayBuffer
from .prioritized_tensordict import TensorDictPrioritizedReplayBuffer
from .rate_limited import RateLimitedReplayBuffer
from .remote import RemoteTensorDictReplayBuffer
from .tensordict import TensorDictReplayBuffer

__all__ = [
    "BlockingReplayBuffer",
    "ConditionalUpdateResult",
    "InPlaceSampler",
    "PrioritizedReplayBuffer",
    "RateLimitedReplayBuffer",
    "RemoteTensorDictReplayBuffer",
    "ReplayBuffer",
    "ReplayBufferEnsemble",
    "stack_tensors",
    "TensorDictPrioritizedReplayBuffer",
    "TensorDictReplayBuffer",
]

for _export in (
    _storage_index,
    _maybe_delay_init,
    ConditionalUpdateResult,
    ReplayBuffer,
    BlockingReplayBuffer,
    PrioritizedReplayBuffer,
    RateLimitedReplayBuffer,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    RemoteTensorDictReplayBuffer,
    InPlaceSampler,
    stack_tensors,
    ReplayBufferEnsemble,
):
    _export.__module__ = __name__
del _export
