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
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase, SatelliteEnv, SatelliteMacroAction
from torchrl.envs.custom.mujoco._math import quat_mul, random_unit_quat


_VIDEO_TAG = "satellite_macros"
_FIXED_TARGET_QUAT = torch.tensor([[0.79335334, 0.24229610, 0.42401818, 0.36344416]])
_IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_RELATIVE_TARGET_BANK = torch.tensor(
    [
        [0.85553485, -0.14270805, 0.15409711, -0.47323212],
        [0.85325271, -0.10478908, 0.19274925, -0.47310328],
        [0.84678650, -0.39844781, 0.21106993, -0.28220820],
        [0.88125467, 0.27627289, 0.26118225, -0.28079763],
        [0.74100631, -0.25096110, 0.33191326, 0.52703112],
        [0.79964513, -0.37387171, -0.02581860, -0.46917057],
        [0.71658695, 0.52446926, 0.04953621, -0.45714468],
        [0.84455746, 0.05724995, -0.33791912, -0.41140702],
        [0.82524592, 0.37116331, 0.15097792, 0.39801091],
        [0.84306234, -0.21775915, -0.45735148, 0.18071111],
    ]
)


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
        "--fixed-start-attitude",
        action="store_true",
        help="Keep the satellite starting at identity attitude while randomizing targets.",
    )
    parser.add_argument(
        "--target-angle-min-deg",
        type=float,
        default=55.0,
        help="Minimum initial attitude error for randomized target frames.",
    )
    parser.add_argument(
        "--target-angle-max-deg",
        type=float,
        default=95.0,
        help="Maximum initial attitude error for randomized target frames.",
    )
    return parser.parse_args()


class SatelliteSlewPolicy:
    """Scripted macro policy that slews toward the reset-time target attitude."""

    def __init__(
        self,
        target_quat: torch.Tensor,
        *,
        macro_steps: int = 36,
        settle_steps: int = 8,
    ) -> None:
        self.target_quat = target_quat
        self.macro_steps = int(macro_steps)
        self.settle_steps = int(settle_steps)

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        # The policy only states the desired attitude. The appended
        # ``SatelliteAttitudeTransform`` reads the current bus attitude, angular
        # velocity and CMG gimbal angles, computes a gimbal-rate destination, and
        # expands it into a low-level action sequence.
        target_quat = self.target_quat.to(
            dtype=tensordict["bus_quat"].dtype,
            device=tensordict["bus_quat"].device,
        )
        if target_quat.ndim == 1:
            target_quat = target_quat.unsqueeze(0)
        target_quat = target_quat.expand(tensordict.batch_size + (4,))
        tensordict.set(
            "action",
            SatelliteMacroAction.slew_attitude(
                target_quat,
                steps=self.macro_steps,
                settle_steps=self.settle_steps,
            ),
        )
        return tensordict


def _run_until_aligned(
    env: EnvBase,
    tensordict: TensorDictBase,
    policy: SatelliteSlewPolicy,
    *,
    max_macros: int,
    target_error_threshold: float,
) -> TensorDictBase:
    for _ in range(max_macros):
        if bool(tensordict["quat_err"].norm(dim=-1).max() < target_error_threshold):
            break
        tensordict = policy(tensordict)
        _, tensordict = env.step_and_maybe_reset(tensordict)
    return tensordict


def _sample_relative_target(
    args: argparse.Namespace,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # The scripted CMG feedback controller is intentionally simple, so the demo
    # samples from a small bank of reliable relative slews instead of arbitrary
    # SO(3) targets. The absolute start attitude is sampled independently below,
    # which still makes the target frame and visible manoeuver change on every
    # replay while keeping the "stop when aligned" behaviour reliable.
    target_bank = _RELATIVE_TARGET_BANK.to(device=device, dtype=dtype)
    target_angles = torch.rad2deg(
        2
        * torch.atan2(
            target_bank[..., 1:].norm(dim=-1),
            target_bank[..., 0].clamp(-1.0, 1.0),
        )
    )
    mask = (target_angles >= args.target_angle_min_deg) & (
        target_angles <= args.target_angle_max_deg
    )
    if not bool(mask.any()):
        raise ValueError(
            "No demonstrated target attitudes match the requested angle range. "
            "Use bounds overlapping approximately 56 to 88 degrees."
        )
    target_bank = target_bank[mask]
    index = torch.randint(
        target_bank.shape[0],
        (1,),
        generator=generator,
        device=device,
    )
    return target_bank[index]


def _make_reset_tensordict(
    args: argparse.Namespace,
    base_env: SatelliteEnv,
    generator: torch.Generator,
) -> TensorDict:
    if args.fixed_attitudes:
        init_quat = _IDENTITY_QUAT
        target_quat = _FIXED_TARGET_QUAT
    else:
        if args.fixed_start_attitude:
            init_quat = _IDENTITY_QUAT.to(dtype=base_env.dtype, device=base_env.device)
        else:
            init_quat = random_unit_quat(
                (1,),
                generator=generator,
                device=base_env.device,
                dtype=base_env.dtype,
            )
        relative_target = _sample_relative_target(
            args,
            generator=generator,
            device=base_env.device,
            dtype=base_env.dtype,
        )
        target_quat = quat_mul(init_quat, relative_target)

    return TensorDict(
        {
            "target_quat": target_quat.to(dtype=base_env.dtype, device=base_env.device),
            "init_bus_quat": init_quat.to(dtype=base_env.dtype, device=base_env.device),
        },
        batch_size=[1],
        device=base_env.device,
    )


def main() -> None:
    args = _parse_args()
    if args.target_angle_min_deg > args.target_angle_max_deg:
        raise ValueError(
            "--target-angle-min-deg must be less than or equal to "
            "--target-angle-max-deg."
        )
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
    env = base_env.append_transform(
        base_env.make_attitude_transform(
            execute=True,
            stack_rewards=True,
            stack_observations=False,
        )
    )
    env, recorder, logger = maybe_add_video_recorder(env, args, tag=_VIDEO_TAG)
    generator = torch.Generator(device=base_env.device)
    generator.manual_seed(int(args.seed))
    rollout_count = 0
    with MujocoViewerLoop(base_env, speed=args.speed) as viewer:
        while viewer.is_running() and (
            args.max_rollouts is None or rollout_count < args.max_rollouts
        ):
            reset = _make_reset_tensordict(args, base_env, generator)
            td = env.reset(reset)
            try:
                _run_until_aligned(
                    env,
                    td,
                    SatelliteSlewPolicy(td["target_quat"]),
                    max_macros=args.max_macros,
                    target_error_threshold=args.target_error_threshold,
                )
            except ViewerClosed:
                break
            dump_video(recorder, logger, tag=_VIDEO_TAG, step=rollout_count)
            rollout_count += 1
            if args.max_rollouts is None or rollout_count < args.max_rollouts:
                time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
