# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Testing utilities for TorchRL.

This module provides helper classes and utilities for testing TorchRL functionality,
particularly for distributed and Ray-based tests that require importable classes.
"""

from torchrl.testing.llm_mocks import (
    MockTransformerConfig,
    MockTransformerModel,
    MockTransformerOutput,
)
from torchrl.testing.ray_helpers import (
    WorkerTransformerDoubleBuffer,
    WorkerTransformerNCCL,
    WorkerVLLMDoubleBuffer,
    WorkerVLLMNCCL,
)

__all__ = [
    "WorkerVLLMNCCL",
    "WorkerTransformerNCCL",
    "WorkerVLLMDoubleBuffer",
    "WorkerTransformerDoubleBuffer",
    "MockTransformerConfig",
    "MockTransformerModel",
    "MockTransformerOutput",
]
