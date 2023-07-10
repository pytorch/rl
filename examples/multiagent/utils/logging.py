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
        sampling_td["next", "agents", "reward"] = (
            sampling_td["next", "reward"]
            .expand(sampling_td["agents"].shape)
            .unsqueeze(-1)
        )

    logger.experiment.log(
        {
            f"train/learner/{key}": value.mean().item()
            for key, value in training_td.items()
        },
        commit=False,
    )
    if "info" in sampling_td["agents"].keys():
        logger.experiment.log(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td["agents", "info"].items()
            },
            commit=False,
        )

    logger.experiment.log(
        {
            "train/reward/reward_min": sampling_td["next", "agents", "reward"]
            .mean(-2)  # Agents
            .min()
            .item(),
            "train/reward/reward_mean": sampling_td["next", "agents", "reward"]
            .mean()
            .item(),
            "train/reward/reward_max": sampling_td["next", "agents", "reward"]
            .mean(-2)  # Agents
            .max()
            .item(),
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
        next_done = r["next"]["done"].sum(
            tuple(range(r.batch_dims, r["next", "done"].ndim)),
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
    ),

    logger.experiment.log(
        {
            "eval/episode_reward_min": min(
                [td["next", "agents", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_max": max(
                [td["next", "agents", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_mean": sum(
                [td["next", "agents", "reward"].sum(0).mean() for td in rollouts]
            )
            / len(rollouts),
            "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
            / len(rollouts),
            "eval/evaluation_time": evaluation_time,
        },
        commit=False,
    )
