# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchrl.envs import GymLikeEnv


class IsaacLabEnv(GymLikeEnv):
    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        categorical_action_encoding=False,
        allow_done_after_reset=True,
        convert_actions_to_numpy=False,
        **kwargs
    ):
        """
        Here we are setting some parameters that are what we need for IsaacLab.
        """
        super().__init__(
            env,
            device=torch.device("cuda:0"),
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs
        )

    @classmethod
    def from_cfg(
        cls,
        cfg,
        categorical_action_encoding=False,
        allow_done_after_reset=True,
        convert_actions_to_numpy=False,
        **kwargs
    ):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg=cfg)
        return cls(
            env=env,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs
        )

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        """
        We discovered the IsaacLab will modify the `terminated` and `truncated` tensors
        in place. Clone them here to make sure data doesn't inadvertently get modified.

        This is a PR in torchRL:
        Once we update to the version with this PR, we can delete this.
        """
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        done = terminated | truncated
        reward = reward.unsqueeze(-1)  # to get to (num_envs, 1)
        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )


if __name__ == "__main__":
    from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

    env = IsaacLabEnv.from_cfg(CartpoleEnvCfg())
    env.check_env_specs(break_when_any_done="both")