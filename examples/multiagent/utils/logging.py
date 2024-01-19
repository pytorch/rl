# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.record.loggers.wandb import WandbLogger


def init_logging(cfg, model_name: str):
    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=os.getcwd(),
        experiment_name=generate_exp_name(cfg.env.scenario_name, model_name),
        wandb_kwargs={
            "group": cfg.logger.group_name or model_name,
            "project": cfg.logger.project_name
            or f"torchrl_example_{cfg.env.scenario_name}",
        },
    )
    logger.log_hparams(cfg)
    return logger


def log_training(
    logger: Logger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
    step: int,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )
    if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "episode_reward"),
            sampling_td.get(("next", "episode_reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )

    to_log = {
        f"train/learner/{key}": value.mean().item()
        for key, value in training_td.items()
    }

    if "info" in sampling_td.get("agents").keys():
        to_log.update(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td.get(("agents", "info")).items()
            }
        )

    reward = sampling_td.get(("next", "agents", "reward")).mean(-2)  # Mean over agents
    done = sampling_td.get(("next", "done"))
    if done.ndim > reward.ndim:
        done = done[..., 0, :]  # Remove expanded agent dim
    episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[
        done
    ]
    to_log.update(
        {
            "train/reward/reward_min": reward.min().item(),
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.max().item(),
            "train/reward/episode_reward_min": episode_reward.min().item(),
            "train/reward/episode_reward_mean": episode_reward.mean().item(),
            "train/reward/episode_reward_max": episode_reward.max().item(),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        }
    )
    if isinstance(logger, WandbLogger):
        logger.experiment.log(to_log, commit=False)
    else:
        for key, value in to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)

    return to_log


def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
    step: int,
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

    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts]
    to_log = {
        "eval/episode_reward_min": min(rewards),
        "eval/episode_reward_max": max(rewards),
        "eval/episode_reward_mean": sum(rewards) / len(rollouts),
        "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
        / len(rollouts),
        "eval/evaluation_time": evaluation_time,
    }

    vid = torch.tensor(
        np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
        dtype=torch.uint8,
    ).unsqueeze(0)

    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log(to_log, commit=False)
        logger.experiment.log(
            {
                "eval/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
            },
            commit=False,
        )
    else:
        for key, value in to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)
        logger.log_video("eval_video", vid, step=step)
