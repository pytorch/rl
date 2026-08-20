# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .ensemble import SamplerEnsemble
from .llm import PromptGroupSampler
from .prioritized import (
    CudaMinSegmentTreeFp32,
    CudaMinSegmentTreeFp64,
    CudaSumSegmentTreeFp32,
    CudaSumSegmentTreeFp64,
    MinSegmentTreeFp32,
    MinSegmentTreeFp64,
    PrioritizedSampler,
    SumSegmentTreeFp32,
    SumSegmentTreeFp64,
)
from .prioritized_slice import PrioritizedSliceSampler
from .random import (
    _default_staleness_weight,
    ConsumingSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
from .base import _REPLACEMENT_DISPATCH, _SamplerMeta, Sampler
from .slice import SliceSampler
from .slice_without_replacement import SliceSamplerWithoutReplacement
from .staleness import StalenessAwareSampler

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."

_REPLACEMENT_DISPATCH.update(
    {
        RandomSampler: SamplerWithoutReplacement,
        SliceSampler: SliceSamplerWithoutReplacement,
    }
)

__all__ = [
    "ConsumingSampler",
    "PrioritizedSampler",
    "PrioritizedSliceSampler",
    "PromptGroupSampler",
    "RandomSampler",
    "Sampler",
    "SamplerEnsemble",
    "SamplerWithoutReplacement",
    "SliceSampler",
    "SliceSamplerWithoutReplacement",
    "StalenessAwareSampler",
]
