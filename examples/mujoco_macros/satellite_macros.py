# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SatelliteEnv example for generic MuJoCo macro actions in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time

import torch

from _viewer import ensure_mjpython_for_passive_viewer, MujocoViewerLoop, ViewerClosed
from tensordict import TensorDict, TensorDictBase
from torchrl.data import TensorSpec
from torchrl.envs import EnvBase, MacroAction, MacroPrimitiveTransform, SatelliteEnv
from torchrl.envs.custom.mujoco._math import cmg_jacobian, pyramid_4cmg_geometry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause-between-rollouts", type=float, default=0.5)
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
    return parser.parse_args()


def _project_action(action_spec: TensorSpec, action: torch.Tensor) -> torch.Tensor:
    return action_spec.project(
        action.to(dtype=action_spec.dtype, device=action_spec.device)
    )


class SatelliteSlewPolicy:
    """Scripted macro policy that slews toward a fixed target attitude."""

    def __init__(
        self,
        action_spec: TensorSpec,
        action_scale: float,
        *,
        attitude_gain: float = 5.0,
        angular_rate_gain: float = 8.0,
        macro_steps: int = 36,
        settle_steps: int = 8,
    ) -> None:
        self.action_spec = action_spec
        self.action_scale = float(action_scale)
        self.attitude_gain = float(attitude_gain)
        self.angular_rate_gain = float(angular_rate_gain)
        self.macro_steps = int(macro_steps)
        self.settle_steps = int(settle_steps)
        self.gimbal_axes, self.rotor_axes_ref = pyramid_4cmg_geometry(
            device=action_spec.device,
            dtype=action_spec.dtype,
        )

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        # ``SatelliteEnv`` exposes ``quat_err``: the logarithmic attitude error
        # from the current bus attitude to the reset-time target attitude. This
        # small feedback controller asks for a bus angular acceleration that
        # reduces the attitude error while damping the current angular velocity,
        # then inverts the local CMG Jacobian to obtain a destination gimbal-rate
        # command. The destination is still a normal TorchRL action placed under
        # ``td["action"]``; the appended transform expands and executes it.
        quat_err = tensordict["quat_err"]
        bus_omega = tensordict["bus_omega"]
        n_gimbals = self.action_spec.shape[-1]
        gimbal_obs = tensordict["gimbal_angles"]
        gimbal_angles = torch.atan2(
            gimbal_obs[..., :n_gimbals],
            gimbal_obs[..., n_gimbals:],
        )
        jacobian = cmg_jacobian(
            gimbal_angles,
            self.gimbal_axes.to(device=quat_err.device, dtype=quat_err.dtype),
            self.rotor_axes_ref.to(device=quat_err.device, dtype=quat_err.dtype),
            1.0,
        )
        desired_bus_accel = (
            self.attitude_gain * quat_err - self.angular_rate_gain * bus_omega
        )
        gimbal_rate = -torch.linalg.pinv(jacobian).matmul(
            desired_bus_accel.unsqueeze(-1)
        )
        target = gimbal_rate.squeeze(-1) / self.action_scale
        target = _project_action(self.action_spec, target)
        tensordict.set(
            "action",
            MacroAction.reach_action(
                target,
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


def main() -> None:
    args = _parse_args()
    ensure_mjpython_for_passive_viewer()

    # The viewer examples use the official MuJoCo C-bindings backend so the
    # passive viewer can display the live ``mjData`` object. We repeatedly reset
    # the satellite to identity attitude, ask it to slew to a tilted target
    # frame, and run high-level macro actions until the attitude error is small.
    # ``reset_noise_scale=0.0`` keeps each viewer replay on the same manoeuver;
    # the default SatelliteEnv reset remains stochastic for training.
    target_quat = torch.tensor([[0.79335334, 0.24229610, 0.42401818, 0.36344416]])
    init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    base_env = SatelliteEnv(
        num_cmgs=4,
        seed=0,
        backend="mujoco",
        max_episode_steps=2048,
        reset_noise_scale=0.0,
    )
    low_level_action_spec = base_env.action_spec.clone()
    env = base_env.append_transform(
        MacroPrimitiveTransform(
            action_dim=low_level_action_spec.shape[-1],
            execute=True,
            stack_rewards=True,
            stack_observations=False,
        )
    )

    reset = TensorDict(
        {
            "target_quat": target_quat.to(dtype=base_env.dtype, device=base_env.device),
            "init_bus_quat": init_quat.to(dtype=base_env.dtype, device=base_env.device),
        },
        batch_size=[1],
        device=base_env.device,
    )
    with MujocoViewerLoop(base_env, speed=args.speed) as viewer:
        while viewer.is_running():
            td = env.reset(reset)
            try:
                _run_until_aligned(
                    env,
                    td,
                    SatelliteSlewPolicy(
                        low_level_action_spec,
                        action_scale=base_env.action_scale,
                    ),
                    max_macros=args.max_macros,
                    target_error_threshold=args.target_error_threshold,
                )
            except ViewerClosed:
                break
            time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
