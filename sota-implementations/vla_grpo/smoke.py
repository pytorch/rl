# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end environment readiness check for the LIBERO VLA recipe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

import moviepy
import timm
import torch
from tensordict import TensorDictBase

from torchrl.data.vla import OpenVLAImagePreprocessor
from torchrl.envs import LiberoEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=0)
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    if gpu_count < args.expected_gpus:
        raise RuntimeError(
            f"Expected at least {args.expected_gpus} CUDA devices, found {gpu_count}."
        )

    preprocessor = OpenVLAImagePreprocessor(
        size=32, backend="torch_reference", center_crop=True
    )
    image = torch.arange(3 * 24 * 40, dtype=torch.int64).reshape(1, 3, 24, 40)
    processed = preprocessor(image.remainder(256).to(torch.uint8))
    if processed.shape != (1, 3, 32, 32) or processed.dtype != torch.uint8:
        raise RuntimeError(
            "The OpenVLA torch_reference preprocessor returned an invalid result."
        )

    env = LiberoEnv(
        "libero_spatial",
        task_id=0,
        camera_height=32,
        camera_width=32,
        max_episode_steps=2,
        settle_steps=2,
    )
    try:
        reset = env.reset()
        if not isinstance(reset, TensorDictBase):
            raise RuntimeError("LiberoEnv.reset() did not return a TensorDict.")
        if reset["observation", "image"].shape != (3, 32, 32):
            raise RuntimeError("LiberoEnv returned an unexpected image shape.")
        instruction = reset["language_instruction"]
        if not isinstance(instruction, str) or not instruction:
            raise RuntimeError("LiberoEnv did not return a language instruction.")
    finally:
        env.close(raise_if_closed=False)

    payload = {
        "cuda_device_count": gpu_count,
        "libero_reset": True,
        "moviepy": importlib.metadata.version("moviepy"),
        "moviepy_module": moviepy.__name__,
        "preprocessor": "torch_reference",
        "timm": timm.__version__,
    }
    sys.stdout.write("TORCHRL_RECIPE_READY=" + json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
