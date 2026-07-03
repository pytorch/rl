# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from tensordict import TensorDictBase

from torchrl.render.config import FrameBundle, RenderConfig

__all__ = ["NullRenderBackend"]


class NullRenderBackend:
    """Fallback backend used when no RGB renderer is available.

    Examples:
        >>> from torchrl.render.backends import NullRenderBackend
        >>> NullRenderBackend().capture(None, None, None, step=0, trajectory_index=0) is None
        True
    """

    name = "null"

    def supports(self, env: Any, config: RenderConfig) -> bool:
        return True

    def capture(
        self,
        env: Any,
        tensordict: TensorDictBase,
        config: RenderConfig,
        *,
        step: int,
        trajectory_index: int,
    ) -> FrameBundle | None:
        return None

    def close(self) -> None:
        return None
