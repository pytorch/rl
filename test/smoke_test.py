# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from pathlib import Path

import torch
import torchrl._torchrl as ext
from torchrl.data import PrioritizedReplayBuffer


def test_imports():
    PrioritizedReplayBuffer(alpha=1.1, beta=1.1)


def test_cuda_segment_tree_extension_available_on_cuda():
    if not torch.cuda.is_available():
        return

    ext_path = Path(ext.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if (
        repo_root in ext_path.parents
        and os.getenv("TORCHRL_SMOKE_REQUIRE_CUDA_EXT") != "1"
    ):
        return

    assert hasattr(ext, "CudaSumSegmentTreeFp32")
