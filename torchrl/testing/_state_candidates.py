# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared, private fixtures for the recurrent-state comparison and benchmarks."""
from __future__ import annotations

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict


class _GRUTC(TensorClass):
    carry: torch.Tensor


class _GRUTTD(TypedTensorDict):
    carry: torch.Tensor


class _GTrXLTC(TensorClass):
    memory: torch.Tensor
    valid: torch.Tensor


class _GTrXLTTD(TypedTensorDict):
    memory: torch.Tensor
    valid: torch.Tensor


_STATE_CLASSES = {
    "gru": {"td": TensorDict, "tc": _GRUTC, "ttd": _GRUTTD},
    "gtrxl": {"td": TensorDict, "tc": _GTrXLTC, "ttd": _GTrXLTTD},
}
