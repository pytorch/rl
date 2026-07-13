# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmarks for OpenVLA-style image preprocessing.

Run with:

    python -m pytest benchmarks/test_vla_preprocessing_benchmark.py \
        --benchmark-group-by=func
"""
from __future__ import annotations

import argparse
import importlib.util
from typing import Literal

import pytest
import torch

from torchrl.data.vla import OpenVLAImagePreprocessor

_has_pil = importlib.util.find_spec("PIL") is not None
_has_torchvision = importlib.util.find_spec("torchvision") is not None


def _make_images(batch_size: int, height: int, width: int) -> torch.Tensor:
    return torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.uint8)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("height,width", [(224, 224), (256, 256), (480, 640)])
@pytest.mark.parametrize("backend", ["pil", "torchvision", "torch_reference"])
def test_openvla_preprocessing_throughput(
    benchmark,
    batch_size: int,
    height: int,
    width: int,
    backend: Literal["pil", "torchvision", "torch_reference"],
):
    if backend == "pil" and not _has_pil:
        pytest.skip("Pillow not found")
    if backend in ("torchvision", "torch_reference") and not _has_torchvision:
        pytest.skip("torchvision not found")
    images = _make_images(batch_size, height, width)
    preprocessor = OpenVLAImagePreprocessor(backend=backend, center_crop=True)
    benchmark(preprocessor, images)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
