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
from torchrl.envs import MacroAction, MacroPrimitiveTransform, SatelliteEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause-between-rollouts", type=float, default=0.5)
    return parser.parse_args()


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
        # from the current bus attitude to the reset-time target attitude. The
        # policy maps that error to one destination gimbal-rate command and puts
        # the destination under ``td["action"]`` as a ``MacroAction``. The
        # appended transform expands and executes the manoeuver.
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
    ensure_mjpython_for_passive_viewer()

    # The viewer examples use the official MuJoCo C-bindings backend so the
    # passive viewer can display the live ``mjData`` object. We repeatedly reset
    # the satellite to identity attitude, ask it to slew to a 90 degree yaw
    # target, and let ``rollout`` call the policy for each high-level macro.
    target_quat = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    base_env = SatelliteEnv(
        num_cmgs=4,
        seed=0,
        backend="mujoco",
        max_episode_steps=512,
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
                env.rollout(
                    max_steps=8,
                    policy=SatelliteSlewPolicy(low_level_action_spec),
                    auto_reset=False,
                    break_when_any_done=False,
                    tensordict=td,
                )
            except ViewerClosed:
                break
            time.sleep(args.pause_between_rollouts)


if __name__ == "__main__":
    main()
