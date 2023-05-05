# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .continuous import (
    __all__ as _all_continuous,
    Delta,
    IndependentNormal,
    NormalParamWrapper,
    TanhDelta,
    TanhNormal,
    TruncatedNormal,
)
from .discrete import __all__ as _all_discrete, MaskedCategorical, OneHotCategorical

distributions_maps = {
    distribution_class.lower(): eval(distribution_class)
    for distribution_class in _all_continuous + _all_discrete
}
