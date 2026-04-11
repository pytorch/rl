# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The VecNorm class is to be deprecated in favor of"
    ),
]
