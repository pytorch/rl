# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import os

import torch


@contextlib.contextmanager
def _cuda_visible_devices(devices: list[torch.device | int]):
    devices = [torch.device(d).index if not isinstance(d, int) else d for d in devices]
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    yield
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    else:
        os.unsetenv("CUDA_VISIBLE_DEVICES")
