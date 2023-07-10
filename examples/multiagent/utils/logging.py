# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import wandb
from tensordict import TensorDictBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.record.loggers.wandb import WandbLogger


def log_training(
    logger: WandbLogger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )

    logger.experiment.log(
        {
            f"train/learner/{key}": value.mean().item()
            for key, value in training_td.items()
        },
        commit=False,
    )
    if "info" in sampling_td.get("agents").keys():
        logger.experiment.log(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td.get(("agents", "info")).items()
            },
            commit=False,
        )

    reward = sampling_td.get(("next", "agents", "reward"))
    logger.experiment.log(
        {
            "train/reward/reward_min": reward.mean(-2).min().item(),  # Mean over agents
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.mean(-2).max().item(),  # Mean over agents
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        },
        commit=False,
    )


def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]
    vid = np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2))
    logger.experiment.log(
        {
            "eval/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
        },
        commit=False,
    )

    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts]
    logger.experiment.log(
        {
            "eval/episode_reward_min": min(rewards),
            "eval/episode_reward_max": max(rewards),
            "eval/episode_reward_mean": sum(rewards) / len(rollouts),
            "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
            / len(rollouts),
            "eval/evaluation_time": evaluation_time,
        },
        commit=False,
    )
