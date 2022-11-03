# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .memmap import MemmapTensor, set_transfer_ownership
from .tensordict import (
    TensorDict,
    SubTensorDict,
    merge_tensordicts,
    LazyStackedTensorDict,
    SavedTensorDict,
)
