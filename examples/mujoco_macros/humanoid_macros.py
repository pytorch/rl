# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""HumanoidEnv example for actuator-control macros in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time

import torch

from _viewer import (
    add_rollout_video_args,
    dump_video,
    ensure_mjpython_for_passive_viewer,
    maybe_add_video_recorder,
    MujocoViewerLoop,
    ViewerClosed,
)
from torchrl.data import TensorSpec
from torchrl.envs import HumanoidEnv, HumanoidMacroAction


_VIDEO_TAG = "humanoid_macros"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause-between-rollouts", type=float, default=0.5)
    add_rollout_video_args(parser)
    return parser.parse_args()


def _target(action_spec: TensorSpec, values: list[float]) -> torch.Tensor:
    target = action_spec.zero()
    values_tensor = torch.as_tensor(values, dtype=target.dtype, device=target.device)
    action_dim = min(target.shape[-1], values_tensor.numel())
    target[..., :action_dim] = values_tensor[:action_dim]
    return action_spec.project(target)


def pose_macros(action_spec: TensorSpec) -> list[HumanoidMacroAction]:
    """Return a short scripted sequence of macro control-pose destinations.

    Each entry is a low-level actuator-control destination, not a pre-expanded
    action sequence. ``HumanoidEnv.make_control_transform(execute=True)``
    interpolates toward each destination and ``MultiAction`` executes the
    resulting sequence through the env.
    """
    return [
        HumanoidMacroAction.reach_control(
            _target(action_spec, [0.16, -0.14, 0.10, -0.10, 0.08, -0.08]),
            steps=24,
            settle_steps=8,
        ),
        HumanoidMacroAction.reach_control(
            _target(action_spec, [-0.14, 0.16, -0.10, 0.10, -0.08, 0.08]),
            steps=24,
            settle_steps=8,
        ),
        HumanoidMacroAction.reach_control(
            _target(action_spec, [0.08, 0.08, -0.14, -0.14, 0.10, 0.10]),
            steps=24,
            settle_steps=8,
        ),
        HumanoidMacroAction.reach_control(
            _target(action_spec, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            steps=28,
            settle_steps=10,
        ),
    ]


def main() -> None:
    args = _parse_args()
    ensure_mjpython_for_passive_viewer()
    base_env = HumanoidEnv(
        seed=0,
        backend="mujoco",
        max_episode_steps=512,
    )
    low_level_action_spec = base_env.action_spec.clone()
    env = base_env.append_transform(
        base_env.make_control_transform(
            execute=True,
            stack_rewards=True,
            stack_observations=False,
        )
    )
    env, recorder, logger = maybe_add_video_recorder(env, args, tag=_VIDEO_TAG)

    # Loop forever: reset the humanoid, execute a short sequence of macro
    # control-pose destinations, pause, and reset again. The macros are an
    # open-loop list of ``HumanoidMacroAction`` objects, so we hand them straight
    # to ``env.rollout(actions=...)`` -- no bespoke stepping loop needed.
    rollout_count = 0
    with MujocoViewerLoop(base_env, speed=args.speed) as viewer:
        while viewer.is_running() and (
            args.max_rollouts is None or rollout_count < args.max_rollouts
        ):
            actions = pose_macros(low_level_action_spec)
            try:
                env.rollout(
                    max_steps=len(actions),
                    actions=actions,
                    break_when_any_done=False,
                )
            except ViewerClosed:
                break
            dump_video(recorder, logger, tag=_VIDEO_TAG, step=rollout_count)
            rollout_count += 1
            if args.max_rollouts is None or rollout_count < args.max_rollouts:
                time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
