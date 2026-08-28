# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:The current behavior of MLP when not providing `num_cells` is that the number"
    ),
    pytest.mark.filterwarnings(
        "ignore:dep_util is Deprecated. Use functions from setuptools instead"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings("ignore:unclosed event loop:ResourceWarning"),
    pytest.mark.filterwarnings("ignore:unclosed.*socket:ResourceWarning"),
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.script` is deprecated:DeprecationWarning"
    ),
]
