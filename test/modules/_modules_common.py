# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import sys

import torch
from packaging import version

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"


def _has_triton_backend() -> bool:
    """Mirror of the triton-availability check inside the RNN backend.

    Triton must be installed, CUDA must be available, and the Triton build
    must expose the ``triton.language.extra.libdevice`` submodule
    (Triton >= 2.2). Older Triton installations are routed to scan/pad
    backends, so the triton-specific tests are skipped there.
    """
    if importlib.util.find_spec("triton") is None or not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("triton.language.extra.libdevice") is not None


_has_triton = _has_triton_backend()
_triton_skip_reason = "requires triton (>= 2.2) and CUDA"

_has_functorch = False
try:
    try:
        from torch import vmap as vmap  # noqa: F401
    except ImportError:
        from functorch import vmap as vmap  # noqa: F401

    _has_functorch = True
except ImportError:
    pass
