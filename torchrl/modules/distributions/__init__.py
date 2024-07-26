# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.nn import NormalParamExtractor

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
    )
}
