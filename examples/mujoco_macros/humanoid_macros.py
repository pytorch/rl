# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""HumanoidEnv example for generic MuJoCo macro actions with rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensordict import TensorDictBase
from torchrl.data import TensorSpec
from torchrl.envs import HumanoidEnv, MacroAction, MacroPrimitiveTransform
from torchrl.record import VideoRecorder

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos" / "humanoid_macros.html"
_BACKENDS = ("mujoco-torch", "mjx", "mujoco")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco-torch", choices=_BACKENDS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--video-interval-ms", type=int, default=60)
    return parser.parse_args()


def _save_animation(
    recorder: VideoRecorder,
    output: Path,
    *,
    title: str,
    interval: int,
) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    animation = recorder.to_animation(title=title, interval=interval, clear=True)
    if output.suffix.lower() == ".html":
        output.write_text(animation.to_jshtml(), encoding="utf-8")
    else:
        animation.save(output)
    return output


def _target(action_spec: TensorSpec, values: list[float]) -> torch.Tensor:
    target = torch.zeros_like(action_spec.rand())
    values_tensor = torch.as_tensor(values, dtype=target.dtype, device=target.device)
    action_dim = min(target.shape[-1], values_tensor.numel())
    target[..., :action_dim] = values_tensor[:action_dim]
    return action_spec.project(target)


class HumanoidPosePolicy:
    """Scripted policy that emits one macro action per requested pose."""

    def __init__(self, action_spec: TensorSpec) -> None:
        # The values are low-level control destinations, not a pre-expanded
        # sequence. ``MacroPrimitiveTransform(execute=True)`` will interpolate
        # between the previous action and each destination and execute the whole
        # short manoeuver through ``MultiAction``.
        self.actions = [
            MacroAction.reach_action(
                _target(action_spec, [0.16, -0.14, 0.10, -0.10, 0.08, -0.08]),
                steps=24,
                settle_steps=8,
            ),
            MacroAction.reach_action(
                _target(action_spec, [-0.14, 0.16, -0.10, 0.10, -0.08, 0.08]),
                steps=24,
                settle_steps=8,
            ),
            MacroAction.reach_action(
                _target(action_spec, [0.08, 0.08, -0.14, -0.14, 0.10, 0.10]),
                steps=24,
                settle_steps=8,
            ),
            MacroAction.reach_action(
                _target(action_spec, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                steps=28,
                settle_steps=10,
            ),
        ]
        self.index = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set("action", self.actions[self.index])
        self.index += 1
        return tensordict


def main() -> None:
    args = _parse_args()
    env = HumanoidEnv(
        seed=0,
        backend=args.backend,
        max_episode_steps=512,
        from_pixels=True,
        pixels_only=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    low_level_action_spec = env.action_spec.clone()
    primitive_control = MacroPrimitiveTransform(
        action_dim=low_level_action_spec.shape[-1],
        execute=True,
        stack_rewards=True,
        stack_observations=False,
    )
    recorder = VideoRecorder(
        logger=None,
        tag="humanoid_macros",
        skip=2,
        make_grid=False,
    )
    env = env.append_transform(recorder)
    env = env.append_transform(primitive_control)

    try:
        env.rollout(
            max_steps=4,
            policy=HumanoidPosePolicy(low_level_action_spec),
            break_when_any_done=False,
        )
        output = _save_animation(
            recorder,
            args.output,
            title="Humanoid macro control-pose sequence",
            interval=args.video_interval_ms,
        )
        print(f"Saved rendered macro animation to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
