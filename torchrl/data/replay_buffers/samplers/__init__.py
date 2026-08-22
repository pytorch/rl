# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .base import (
    _REPLACEMENT_DISPATCH as _REPLACEMENT_DISPATCH,
    _SamplerMeta as _SamplerMeta,
    Sampler,
)
from .ensemble import SamplerEnsemble
from .geometric import GeometricTrajectoryWindowSampler
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
    _default_staleness_weight as _default_staleness_weight,
    ConsumingSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
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
    "CudaMinSegmentTreeFp32",
    "CudaMinSegmentTreeFp64",
    "CudaSumSegmentTreeFp32",
    "CudaSumSegmentTreeFp64",
    "GeometricTrajectoryWindowSampler",
    "MinSegmentTreeFp32",
    "MinSegmentTreeFp64",
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
    "SumSegmentTreeFp32",
    "SumSegmentTreeFp64",
]

for _export in (
    _SamplerMeta,
    Sampler,
    RandomSampler,
    ConsumingSampler,
    SamplerWithoutReplacement,
    _default_staleness_weight,
    StalenessAwareSampler,
    PrioritizedSampler,
    SliceSampler,
    SliceSamplerWithoutReplacement,
    PrioritizedSliceSampler,
    GeometricTrajectoryWindowSampler,
    PromptGroupSampler,
    SamplerEnsemble,
):
    _export.__module__ = __name__
del _export
