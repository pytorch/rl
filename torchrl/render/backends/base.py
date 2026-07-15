# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, Protocol

from tensordict import TensorDictBase

from torchrl.render.config import FrameBundle, RenderConfig

__all__ = ["RenderBackend"]


class RenderBackend(Protocol):
    """Protocol implemented by rlrender frame-capture backends.

    Examples:
        >>> from torchrl.render.backends import NullRenderBackend
        >>> backend = NullRenderBackend()
        >>> backend.name
        'null'
    """

    name: str

    def supports(self, env: Any, config: RenderConfig) -> bool:
        """Returns whether the backend can capture from this environment."""

    def capture(
        self,
        env: Any,
        tensordict: TensorDictBase,
        config: RenderConfig,
        *,
        step: int,
        trajectory_index: int,
    ) -> FrameBundle | None:
        """Captures frames for one rollout step."""

    def close(self) -> None:
        """Closes backend resources."""
