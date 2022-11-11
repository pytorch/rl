# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .postprocs import MultiStep
from .replay_buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    Storage,
    ListStorage,
    LazyMemmapStorage,
    LazyTensorStorage,
)
from .tensor_specs import (
    TensorSpec,
    BoundedTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    DiscreteTensorSpec,
    CompositeSpec,
    DEVICE_TYPING,
)
from .tensordict import (
    MemmapTensor,
    set_transfer_ownership,
    TensorDict,
    SubTensorDict,
    merge_tensordicts,
    LazyStackedTensorDict,
    SavedTensorDict,
)
