# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import TensorDictBase
from torch import nn


def _runtime_device(
    module: nn.Module,
    tensordict: TensorDictBase,
    *values: object,
    fallback: torch.Tensor,
) -> torch.device:
    """Resolve the device relevant to the current loss invocation."""
    if tensordict.device is not None:
        return tensordict.device
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item.device
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return fallback.device
