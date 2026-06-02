# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CubeBowlEnv example for UR-style MuJoCo macro actions in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from _viewer import (
    add_rollout_video_args,
    dump_video,
    ensure_mjpython_for_passive_viewer,
    maybe_add_video_recorder,
    MujocoViewerLoop,
    ViewerClosed,
)
from tensordict import TensorDictBase
from torchrl.envs import CubeBowlEnv, EnvBase, RobotMacroAction


_VIDEO_TAG = "cube_bowl_macros"


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
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause-between-rollouts", type=float, default=0.5)
    add_rollout_video_args(parser)
    return parser.parse_args()


def _offset_like(reference: torch.Tensor, xyz: list[float]) -> torch.Tensor:
    return torch.as_tensor(xyz, dtype=reference.dtype, device=reference.device).view(
        1, 3
    )


def _run_macro(
    env: EnvBase,
    tensordict: TensorDictBase,
    action: RobotMacroAction,
) -> TensorDictBase:
    # The env has ``URScriptPrimitiveTransform(execute=True)`` appended, so one
    # call to ``step_and_maybe_reset`` runs a complete high-level command. We do
    # not manually unbind or step through the low-level sequence here: the
    # transform expands ``RobotMacroAction`` and ``MultiAction`` executes it.
    tensordict.set("action", action)
    _, tensordict = env.step_and_maybe_reset(tensordict)
    return tensordict


def _run_pick_carry_release(
    env: EnvBase,
    base_env: CubeBowlEnv,
    td: TensorDictBase,
) -> TensorDictBase:
    gripper_quat = td["pinch_quat"].clone()
    home_qpos = torch.as_tensor(
        base_env.robot_home_qpos,
        dtype=td["robot_qpos"].dtype,
        device=td["robot_qpos"].device,
    ).view(1, 6)
    close_command = base_env.gripper_ctrl_for_width(
        2 * base_env.OBJECT_HALF_SIZE - 0.001
    )

    # The manoeuver is intentionally readable: open, hover above the cube,
    # descend, close the gripper, lift, carry over the bowl, release, then
    # return to the home posture. Every item is an explicit ``RobotMacroAction``
    # destination placed under ``td["action"]``.
    cube = td["cube_pos"].clone()
    td = _run_macro(env, td, RobotMacroAction.open_gripper(steps=18, settle_steps=4))
    td = _run_macro(
        env,
        td,
        RobotMacroAction.reach_pose(
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
        RobotMacroAction.reach_pose(
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
        RobotMacroAction.close_gripper(
            command=close_command, steps=24, settle_steps=12
        ),
    )

    cube = td["cube_pos"].clone()
    pinch_to_cube = td["pinch_pos"].clone() - cube
    td = _run_macro(
        env,
        td,
        RobotMacroAction.reach_pose(
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
        RobotMacroAction.reach_pose(
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
        RobotMacroAction.reach_pose(
            position=bowl + pinch_to_cube + _offset_like(bowl, [0.0, 0.0, 0.06]),
            quaternion=gripper_quat,
            gripper="closed",
            gripper_command=close_command,
            steps=28,
            settle_steps=6,
        ),
    )
    td = _run_macro(env, td, RobotMacroAction.open_gripper(steps=24, settle_steps=8))
    return _run_macro(
        env,
        td,
        RobotMacroAction.home(joints=home_qpos, gripper="open", steps=36),
    )


def main() -> None:
    args = _parse_args()
    ensure_mjpython_for_passive_viewer()
    menagerie_path = args.menagerie_path
    if menagerie_path is None:
        menagerie_env = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
        if menagerie_env is None:
            raise RuntimeError(
                f"Set {CubeBowlEnv.MENAGERIE_ENV_VAR} to a MuJoCo Menagerie checkout."
            )
        menagerie_path = Path(menagerie_env)

    base_env = CubeBowlEnv(
        menagerie_path=menagerie_path,
        seed=0,
        backend="mujoco",
        max_episode_steps=1024,
    )
    env = base_env.append_transform(
        base_env.make_urscript_transform(
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
    )
    env, recorder, logger = maybe_add_video_recorder(env, args, tag=_VIDEO_TAG)

    # Loop forever: reset the scene, execute the full pick/carry/release macro
    # sequence, pause, and reset to replay it in the MuJoCo viewer.
    rollout_count = 0
    with MujocoViewerLoop(base_env, speed=args.speed) as viewer:
        while viewer.is_running() and (
            args.max_rollouts is None or rollout_count < args.max_rollouts
        ):
            td = env.reset()
            try:
                _run_pick_carry_release(env, base_env, td)
            except ViewerClosed:
                break
            dump_video(recorder, logger, tag=_VIDEO_TAG, step=rollout_count)
            rollout_count += 1
            if args.max_rollouts is None or rollout_count < args.max_rollouts:
                time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
