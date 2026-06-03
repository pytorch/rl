# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SatelliteEnv example for attitude-target macros in the MuJoCo viewer."""

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
from tensordict import TensorDict
from torchrl.envs import SatelliteEnv, SatelliteMacroAction, TerminateTransform
from torchrl.envs.custom.mujoco._math import quat_mul, random_unit_quat


_VIDEO_TAG = "satellite_macros"
_IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_FIXED_TARGET_QUAT = torch.tensor([[0.79335334, 0.24229610, 0.42401818, 0.36344416]])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause-between-rollouts", type=float, default=0.5)
    add_rollout_video_args(parser)
    parser.add_argument(
        "--max-macros",
        type=int,
        default=32,
        help="Maximum number of high-level macro actions before resetting.",
    )
    parser.add_argument(
        "--target-error-threshold",
        type=float,
        default=0.06,
        help="Stop the current replay once ||quat_err|| is below this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the scripted sequence of start and target attitudes.",
    )
    parser.add_argument(
        "--fixed-attitudes",
        action="store_true",
        help="Replay the same start and target attitudes instead of randomizing.",
    )
    parser.add_argument(
        "--target-angle-min-deg",
        type=float,
        default=25.0,
        help="Minimum initial attitude error for randomized target frames.",
    )
    parser.add_argument(
        "--target-angle-max-deg",
        type=float,
        default=50.0,
        help="Maximum initial attitude error for randomized target frames.",
    )
    args = parser.parse_args()
    if args.target_angle_min_deg > args.target_angle_max_deg:
        raise ValueError("--target-angle-min-deg must be <= --target-angle-max-deg.")
    return args


def _relative_angle_deg(relative_quat: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation angle (degrees) of a relative unit quaternion."""
    return torch.rad2deg(
        2.0
        * torch.atan2(
            relative_quat[..., 1:].norm(dim=-1),
            relative_quat[..., 0].abs().clamp(max=1.0),
        )
    )


def _sample_attitudes(
    args: argparse.Namespace,
    base_env: SatelliteEnv,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick a start attitude and a target frame within a reachable slew band.

    The demo controller is a compact proportional-derivative CMG steering law,
    not a full guidance stack, so very large slews are not reliably reached
    within ``--max-macros``. We keep the sampling simple (random start, random
    relative rotation) but reject relative rotations whose angle falls outside
    ``[--target-angle-min-deg, --target-angle-max-deg]`` so the reset -> target
    -> stop loop generally completes while still varying across rollouts.
    """
    if args.fixed_attitudes:
        return (
            _IDENTITY_QUAT.to(dtype=base_env.dtype, device=base_env.device),
            _FIXED_TARGET_QUAT.to(dtype=base_env.dtype, device=base_env.device),
        )
    init_quat = random_unit_quat(
        (1,), generator=generator, device=base_env.device, dtype=base_env.dtype
    )
    for _ in range(1000):
        relative = random_unit_quat(
            (1,), generator=generator, device=base_env.device, dtype=base_env.dtype
        )
        angle = float(_relative_angle_deg(relative).item())
        if args.target_angle_min_deg <= angle <= args.target_angle_max_deg:
            break
    target_quat = quat_mul(init_quat, relative)
    return (
        init_quat.to(dtype=base_env.dtype, device=base_env.device),
        target_quat.to(dtype=base_env.dtype, device=base_env.device),
    )


def main() -> None:
    args = _parse_args()
    ensure_mjpython_for_passive_viewer()

    # The viewer examples use the official MuJoCo C-bindings backend so the
    # passive viewer can display the live ``mjData`` object. We repeatedly reset
    # the satellite to a sampled start attitude, ask it to slew to a different
    # sampled target frame, and run high-level macro actions until the attitude
    # error is small. ``reset_noise_scale=0.0`` keeps the scripted attitudes
    # visually clean; the sequence still changes across rollouts unless
    # ``--fixed-attitudes`` is passed.
    base_env = SatelliteEnv(
        num_cmgs=4,
        seed=args.seed,
        backend="mujoco",
        max_episode_steps=2048,
        reset_noise_scale=0.0,
    )

    def aligned(td: TensorDict) -> bool:
        return bool(td["quat_err"].norm(dim=-1).max() < args.target_error_threshold)

    env = base_env.append_transform(
        base_env.make_attitude_transform(
            execute=True,
            stack_rewards=True,
            stack_observations=False,
        )
    )
    # ``TerminateTransform`` flips the done flag once the attitude error is
    # small, so the open-loop ``rollout(actions=...)`` below stops as soon as the
    # satellite is aligned (``break_when_any_done=True``).
    env = env.append_transform(TerminateTransform(stop=aligned))
    env, recorder, logger = maybe_add_video_recorder(env, args, tag=_VIDEO_TAG)
    generator = torch.Generator(device=base_env.device)
    generator.manual_seed(int(args.seed))

    rollout_count = 0
    with MujocoViewerLoop(base_env, speed=args.speed) as viewer:
        while viewer.is_running() and (
            args.max_rollouts is None or rollout_count < args.max_rollouts
        ):
            init_quat, target_quat = _sample_attitudes(args, base_env, generator)
            reset_td = TensorDict(
                {"target_quat": target_quat, "init_bus_quat": init_quat},
                batch_size=[1],
                device=base_env.device,
            )
            # The whole policy-side command is a desired attitude frame. The
            # appended ``SatelliteAttitudeTransform`` reads the current bus
            # attitude, angular velocity and CMG gimbal angles, then expands the
            # target attitude into a low-level gimbal-rate action sequence. The
            # macro action is the same every step -- "move toward target" -- so
            # we replay it up to ``max_macros`` times and let
            # ``TerminateTransform`` stop the rollout once aligned.
            action = SatelliteMacroAction.slew_attitude(target_quat)
            try:
                env.rollout(
                    max_steps=args.max_macros,
                    actions=[action] * args.max_macros,
                    tensordict=reset_td,
                    auto_reset=True,
                    break_when_any_done=True,
                )
            except ViewerClosed:
                break
            dump_video(recorder, logger, tag=_VIDEO_TAG, step=rollout_count)
            rollout_count += 1
            if args.max_rollouts is None or rollout_count < args.max_rollouts:
                time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
