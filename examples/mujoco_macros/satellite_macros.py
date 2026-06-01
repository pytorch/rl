# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SatelliteEnv example for generic MuJoCo macro actions."""

from __future__ import annotations

import torch
from torchrl.envs import MacroPrimitive, MacroPrimitiveTransform, SatelliteEnv, step_mdp


def main() -> None:
    env = SatelliteEnv(num_cmgs=4, seed=0, max_episode_steps=32)
    transform = MacroPrimitiveTransform(
        action_dim=env.action_spec.shape[-1],
        macro_steps=8,
        settle_steps=2,
    )
    td = env.reset()
    target = torch.zeros_like(env.action_spec.rand())
    target[..., 0] = 0.25
    sequence = transform.action_sequence(td, MacroPrimitive.MOVEJ, target_qpos=target)
    for action in sequence.unbind(-2):
        td = step_mdp(env.step(td.set("action", action)))
    env.close()


if __name__ == "__main__":
    main()
