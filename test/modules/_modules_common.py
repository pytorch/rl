# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import sys

import torch
from packaging import version

from torchrl._utils import _triton_version_at_least

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"


# Mirror of the triton-availability check inside the RNN backend: Triton
# >= 2.2 must be installed and CUDA must be available. Older Triton
# installations are routed to scan/pad backends, so the triton-specific
# tests are skipped there.
_has_triton = _triton_version_at_least("2.2") and torch.cuda.is_available()
_triton_skip_reason = "requires triton (>= 2.2) and CUDA"

_has_functorch = (
    hasattr(torch, "vmap") or importlib.util.find_spec("functorch") is not None
)
