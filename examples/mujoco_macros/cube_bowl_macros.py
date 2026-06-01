# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CubeBowlEnv example for UR-style MuJoCo macro actions."""

from __future__ import annotations

import os

import torch
from torchrl.envs import CubeBowlEnv, RobotAction, step_mdp


def main() -> None:
    menagerie_path = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR)
    if menagerie_path is None:
        raise RuntimeError(
            f"Set {CubeBowlEnv.MENAGERIE_ENV_VAR} to a MuJoCo Menagerie checkout."
        )
    env = CubeBowlEnv(menagerie_path=menagerie_path, seed=0, max_episode_steps=64)
    td = env.reset()
    transform = env.make_urscript_transform(macro_steps=8, execute=False)
    above_cube = td["cube_pos"] + torch.tensor([[0.0, 0.0, 0.08]], dtype=env.dtype)
    td["action"] = RobotAction.reach_pose(
        position=above_cube,
        quaternion=td["pinch_quat"],
        gripper="open",
        steps=8,
    )
    for action in transform.inv(td)["action"].unbind(-2):
        td = step_mdp(env.step(td.set("action", action)))
    env.close()


if __name__ == "__main__":
    main()
