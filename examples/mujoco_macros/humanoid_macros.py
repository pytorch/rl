# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""HumanoidEnv example for generic MuJoCo macro actions."""

from __future__ import annotations

import torch
from torchrl.envs import HumanoidEnv, MacroPrimitive, MacroPrimitiveTransform, step_mdp


def main() -> None:
    env = HumanoidEnv(seed=0, max_episode_steps=32)
    transform = MacroPrimitiveTransform(
        action_dim=env.action_spec.shape[-1],
        macro_steps=6,
        settle_steps=2,
    )
    td = env.reset()
    target = torch.zeros_like(env.action_spec.rand())
    target[..., :2] = torch.tensor([0.1, -0.1], dtype=target.dtype)
    sequence = transform.action_sequence(td, MacroPrimitive.MOVEJ, target_qpos=target)
    for action in sequence.unbind(-2):
        td = step_mdp(env.step(td.set("action", action)))
    env.close()


if __name__ == "__main__":
    main()
