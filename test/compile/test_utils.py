# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for torch.compile compatibility of utility functions."""
from __future__ import annotations

import sys

import pytest
import torch
from packaging import version

from torchrl.testing import capture_log_records

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
    ),
]


# Check that 'capture_log_records' captures records emitted when torch
# recompiles a function.
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="torch.compile is not supported on Python 3.14+",
)
def test_capture_log_records_recompile():
    torch.compiler.reset()

    # This function recompiles each time it is called with a different string
    # input.
    @torch.compile
    def str_to_tensor(s):
        return bytes(s, "utf8")

    str_to_tensor("a")

    try:
        torch._logging.set_logs(recompiles=True)
        records = []
        capture_log_records(records, "torch._dynamo", "recompiles")
        str_to_tensor("b")

    finally:
        torch._logging.set_logs()

    assert len(records) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
