# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.nn import NormalParamExtractor
from torch import distributions as torch_dist

from .continuous import (
    Delta,
    IndependentNormal,
    NormalParamWrapper,
    TanhDelta,
    TanhNormal,
    TruncatedNormal,
)
from .discrete import (
    MaskedCategorical,
    MaskedOneHotCategorical,
    OneHotCategorical,
    OneHotOrdinal,
    Ordinal,
    ReparamGradientStrategy,
)

distributions_maps = {
    str(dist).lower(): dist
    for dist in (
        Delta,
        IndependentNormal,
        TanhDelta,
        TanhNormal,
        TruncatedNormal,
        MaskedCategorical,
        MaskedOneHotCategorical,
        OneHotCategorical,
        Ordinal,
        OneHotOrdinal,
    )
}

HAS_ENTROPY = {
    Delta: False,
    IndependentNormal: True,
    TanhDelta: False,
    TanhNormal: False,
    TruncatedNormal: False,
    MaskedCategorical: False,
    MaskedOneHotCategorical: False,
    OneHotCategorical: True,
    torch_dist.Categorical: True,
    torch_dist.Normal: True,
}

__all__ = [
    "NormalParamExtractor",
    "distributions",
    "Delta",
    "IndependentNormal",
    "NormalParamWrapper",
    "TanhDelta",
    "TanhNormal",
    "TruncatedNormal",
    "MaskedCategorical",
    "MaskedOneHotCategorical",
    "OneHotCategorical",
    "OneHotOrdinal",
    "Ordinal",
    "ReparamGradientStrategy",
]
