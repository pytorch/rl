# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchrl.envs.libs.gym import GymWrapper


class IsaacLabWrapper(GymWrapper):
    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        categorical_action_encoding=False,
        allow_done_after_reset=True,
        convert_actions_to_numpy=False,
        device=torch.device("cuda:0"),
        **kwargs,
    ):
        """
        Here we are setting some parameters that are what we need for IsaacLab.
        """
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
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
    import argparse

    from isaaclab.app import AppLauncher
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

    if __name__ == "__main__":
        # import isaaclab_tasks

        env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        env = IsaacLabWrapper(env)

        import tqdm

        # env.check_env_specs(break_when_any_done="both")
        # env.check_env_specs(break_when_any_done="both")
        from torchrl.collectors import SyncDataCollector
        from torchrl.record.loggers.wandb import WandbLogger

        logger = WandbLogger(exp_name="test_isaac")
        col = SyncDataCollector(
            env, env.rand_action, frames_per_batch=1000, total_frames=100_000_000
        )
        for d in tqdm.tqdm(col):
            logger.log_scalar("frames", col._frames)
