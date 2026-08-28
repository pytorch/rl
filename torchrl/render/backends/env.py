# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from tensordict import TensorDictBase

from torchrl.render.config import FrameBundle, RenderConfig
from torchrl.render.video import normalize_frame_output

__all__ = ["EnvRenderBackend"]


class EnvRenderBackend:
    """Captures frames by calling ``env.render()``.

    Examples:
        >>> from torchrl.render.backends import EnvRenderBackend
        >>> EnvRenderBackend().name
        'env'
    """

    name = "env"

    def supports(self, env: Any, config: RenderConfig) -> bool:
        render = getattr(env, "render", None)
        return callable(render)

    def capture(
        self,
        env: Any,
        tensordict: TensorDictBase,
        config: RenderConfig,
        *,
        step: int,
        trajectory_index: int,
    ) -> FrameBundle | None:
        render = getattr(env, "render", None)
        if not callable(render):
            return None
        output = self._render(render, config)
        frames = normalize_frame_output(output)
        if not frames:
            return None
        return FrameBundle(
            frames=frames,
            step=step,
            trajectory_index=trajectory_index,
            metadata={"backend": self.name},
        )

    def close(self) -> None:
        return None

    def _render(self, render: Any, config: RenderConfig) -> Any:
        if config.render_mode is not None:
            try:
                return render(mode=config.render_mode)
            except TypeError:
                return render()
        return render()
