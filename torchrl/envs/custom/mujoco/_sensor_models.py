# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Small sensor encoders used by MicroDuck prior models."""

from __future__ import annotations

import torch
from torch import nn


class _MicroDuckSensorEncoder(nn.Module):
    """Encode only declared proprioception and optional raw RGB inputs."""

    def __init__(self, hidden_size: int, *, vision: bool, device="cpu"):
        super().__init__()
        self.vision = vision
        self.proprioception = nn.Linear(53, hidden_size, device=device)
        if vision:
            self.camera = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, device=device),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, device=device),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, device=device),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(128, hidden_size, device=device),
            )
            self.availability = nn.Linear(2, hidden_size, device=device)

    def forward(self, proprioception, pixels=None, age=None, valid=None):
        features = self.proprioception(proprioception)
        if self.vision:
            batch = pixels.shape[:-3]
            images = (
                pixels.reshape(-1, *pixels.shape[-3:]).movedim(-1, 1).to(features.dtype)
                / 255
            )
            image_features = self.camera(images).reshape(*batch, -1)
            available = valid.to(features.dtype)
            features = (
                features
                + image_features * available
                + self.availability(torch.cat((age, available), -1))
            )
        return torch.tanh(features)
