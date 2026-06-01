# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CubeBowlEnv example for UR-style MuJoCo macro actions with rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tensordict import TensorDictBase
from torchrl.envs import CubeBowlEnv, EnvBase, RobotAction
from torchrl.record import VideoRecorder

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos" / "cube_bowl_macros.html"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--menagerie-path",
        type=Path,
        default=None,
        help=(
            "Path to a MuJoCo Menagerie checkout. Defaults to "
            f"${CubeBowlEnv.MENAGERIE_ENV_VAR}."
        ),
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--render-width", type=int, default=360)
    parser.add_argument("--render-height", type=int, default=270)
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


def _offset_like(reference: torch.Tensor, xyz: list[float]) -> torch.Tensor:
    return torch.as_tensor(xyz, dtype=reference.dtype, device=reference.device).view(
        1, 3
    )


def _run_macro(
    env: EnvBase,
    tensordict: TensorDictBase,
    action: RobotAction,
) -> TensorDictBase:
    # The env has ``URScriptPrimitiveTransform(execute=True)`` appended, so one
    # call to ``step_and_maybe_reset`` runs a complete high-level command. We do
    # not manually unbind or step through the low-level sequence here: the
    # transform expands ``RobotAction`` and ``MultiAction`` executes it.
    tensordict.set("action", action)
    _, tensordict = env.step_and_maybe_reset(tensordict)
    return tensordict


def main() -> None:
    args = _parse_args()
    menagerie_path = args.menagerie_path
    if menagerie_path is None:
        menagerie_env = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
        if menagerie_env is None:
            raise RuntimeError(
                f"Set {CubeBowlEnv.MENAGERIE_ENV_VAR} to a MuJoCo Menagerie checkout."
            )
        menagerie_path = Path(menagerie_env)

    env = CubeBowlEnv(
        menagerie_path=menagerie_path,
        seed=0,
        max_episode_steps=1024,
        from_pixels=True,
        pixels_only=False,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    recorder = VideoRecorder(
        logger=None,
        tag="cube_bowl_macros",
        skip=2,
        make_grid=False,
    )
    primitive_control = env.make_urscript_transform(
        macro_steps=28,
        settle_steps=8,
        execute=True,
        ik_kwargs={
            "iterations": 160,
            "orientation_weight": 1.0,
            "step_size": 0.7,
            "damping": 1e-4,
        },
        stack_rewards=True,
        stack_observations=False,
    )
    env = env.append_transform(recorder)
    env = env.append_transform(primitive_control)

    td = env.reset()
    gripper_quat = td["pinch_quat"].clone()
    home_qpos = torch.as_tensor(
        env.robot_home_qpos,
        dtype=td["robot_qpos"].dtype,
        device=td["robot_qpos"].device,
    ).view(1, 6)
    close_command = env.gripper_ctrl_for_width(2 * env.OBJECT_HALF_SIZE - 0.001)
    try:
        # The manoeuver is intentionally readable: open, hover above the cube,
        # descend, close the gripper, lift, carry over the bowl, release, then
        # return to the home posture. Every item is an explicit ``RobotAction``
        # destination placed under ``td["action"]``.
        cube = td["cube_pos"].clone()
        td = _run_macro(env, td, RobotAction.open_gripper(steps=18, settle_steps=4))
        td = _run_macro(
            env,
            td,
            RobotAction.reach_pose(
                position=cube + _offset_like(cube, [0.0, 0.0, 0.12]),
                quaternion=gripper_quat,
                gripper="open",
                steps=32,
                settle_steps=6,
            ),
        )
        td = _run_macro(
            env,
            td,
            RobotAction.reach_pose(
                position=cube + _offset_like(cube, [0.0, 0.0, 0.02]),
                quaternion=gripper_quat,
                gripper="open",
                steps=32,
                settle_steps=6,
            ),
        )
        td = _run_macro(
            env,
            td,
            RobotAction.close_gripper(command=close_command, steps=24, settle_steps=12),
        )

        cube = td["cube_pos"].clone()
        pinch_to_cube = td["pinch_pos"].clone() - cube
        td = _run_macro(
            env,
            td,
            RobotAction.reach_pose(
                position=cube + pinch_to_cube + _offset_like(cube, [0.0, 0.0, 0.20]),
                quaternion=gripper_quat,
                gripper="closed",
                gripper_command=close_command,
                steps=36,
                settle_steps=8,
            ),
        )

        bowl = td["bowl_pos"].clone()
        cube = td["cube_pos"].clone()
        pinch_to_cube = td["pinch_pos"].clone() - cube
        td = _run_macro(
            env,
            td,
            RobotAction.reach_pose(
                position=bowl + pinch_to_cube + _offset_like(bowl, [0.0, 0.0, 0.16]),
                quaternion=gripper_quat,
                gripper="closed",
                gripper_command=close_command,
                steps=36,
                settle_steps=8,
            ),
        )
        td = _run_macro(
            env,
            td,
            RobotAction.reach_pose(
                position=bowl + pinch_to_cube + _offset_like(bowl, [0.0, 0.0, 0.06]),
                quaternion=gripper_quat,
                gripper="closed",
                gripper_command=close_command,
                steps=28,
                settle_steps=6,
            ),
        )
        td = _run_macro(env, td, RobotAction.open_gripper(steps=24, settle_steps=8))
        td = _run_macro(
            env,
            td,
            RobotAction.home(joints=home_qpos, gripper="open", steps=36),
        )
        output = _save_animation(
            recorder,
            args.output,
            title="CubeBowl pick-carry-release macro sequence",
            interval=args.video_interval_ms,
        )
        print(f"Saved rendered macro animation to {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
