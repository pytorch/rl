# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SatelliteEnv example for generic MuJoCo macro actions with rendering."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import TensorSpec
from torchrl.envs import MacroAction, MacroPrimitiveTransform, SatelliteEnv
from torchrl.record import VideoRecorder

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos" / "satellite_macros.html"
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


def _project_action(action_spec: TensorSpec, action: torch.Tensor) -> torch.Tensor:
    return action_spec.project(
        action.to(dtype=action_spec.dtype, device=action_spec.device)
    )


class SatelliteSlewPolicy:
    """Scripted macro policy that slews toward a fixed target attitude."""

    def __init__(self, action_spec: TensorSpec) -> None:
        self.action_spec = action_spec
        self.macro_count = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        # ``SatelliteEnv`` exposes ``quat_err``: the logarithmic attitude error
        # between the current bus attitude and the reset-time target attitude.
        # This tiny steering law maps that 3D error to a destination gimbal-rate
        # command, then wraps it in ``MacroAction``. The transform appended to the
        # env expands this single action into a smooth low-level manoeuver.
        quat_err = tensordict["quat_err"]
        target = torch.zeros_like(self.action_spec.rand())
        gains = (0.55, 0.40, 0.25, 0.12, 0.0)
        gain = gains[min(self.macro_count, len(gains) - 1)]
        target[..., :3] = gain * quat_err.clamp(-1.0, 1.0)
        target[..., 3:] = -0.5 * target[..., :1]
        target = _project_action(self.action_spec, target)
        tensordict.set(
            "action",
            MacroAction.reach_action(target, steps=28, settle_steps=8),
        )
        self.macro_count += 1
        return tensordict


def main() -> None:
    args = _parse_args()

    # Ask the satellite to slew from identity to a 90 degree yaw target. The
    # policy above never writes low-level action sequences itself: it only places
    # one explicit ``MacroAction`` destination under ``td["action"]`` at each
    # macro step, and TorchRL's transform stack does the expansion and execution.
    target_quat = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    env = SatelliteEnv(
        num_cmgs=4,
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
        tag="satellite_macros",
        skip=2,
        make_grid=False,
    )
    env = env.append_transform(recorder)
    env = env.append_transform(primitive_control)

    reset = TensorDict(
        {
            "target_quat": target_quat.to(dtype=env.dtype, device=env.device),
            "init_bus_quat": init_quat.to(dtype=env.dtype, device=env.device),
        },
        batch_size=[1],
        device=env.device,
    )
    td = env.reset(reset)
    try:
        env.rollout(
            max_steps=8,
            policy=SatelliteSlewPolicy(low_level_action_spec),
            auto_reset=False,
            break_when_any_done=False,
            tensordict=td,
        )
        output = _save_animation(
            recorder,
            args.output,
            title="Satellite macro slew to target attitude",
            interval=args.video_interval_ms,
        )
        print(f"Saved rendered macro animation to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
