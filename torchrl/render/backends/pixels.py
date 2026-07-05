# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from tensordict import TensorDictBase

from torchrl.render.config import FrameBundle, RenderConfig
from torchrl.render.video import normalize_frame_output

__all__ = ["TensorDictPixelsBackend"]


class TensorDictPixelsBackend:
    """Captures frames from pixel entries already present in a TensorDict.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.render import RenderConfig
        >>> from torchrl.render.backends import TensorDictPixelsBackend
        >>> td = TensorDict({"pixels": torch.zeros(2, 2, 3, dtype=torch.uint8)}, [])
        >>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=1)
        >>> TensorDictPixelsBackend().capture(None, td, cfg, step=0, trajectory_index=0).step
        0
    """

    name = "pixels"

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
        for key in _pixel_key_candidates(config):
            try:
                value = tensordict.get(key)
            except Exception:
                continue
            frames = normalize_frame_output(value)
            if frames:
                return FrameBundle(
                    frames=frames,
                    step=step,
                    trajectory_index=trajectory_index,
                    metadata={"backend": self.name, "pixel_key": _key_to_metadata(key)},
                )
        return None

    def close(self) -> None:
        return None


def _pixel_key_candidates(config: RenderConfig) -> list[Any]:
    # The "next" entry holds the post-step frame; the root entry is the
    # pre-step observation, so it is only used as a fallback (e.g. for
    # reset tensordicts that carry no "next" entry).
    key = config.pixel_key
    if isinstance(key, tuple):
        candidates = [("next", *key), key]
    else:
        candidates = [("next", key), key]
    candidates.extend([("next", "pixels"), "pixels"])
    out = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def _key_to_metadata(key: Any) -> str:
    if isinstance(key, tuple):
        return ".".join(str(item) for item in key)
    return str(key)
