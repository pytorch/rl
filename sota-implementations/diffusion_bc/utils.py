# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from torch import nn, optim

from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import DiffusionActor
from torchrl.objectives import DiffusionBCLoss
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps):
    transformed_env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger=None):
    """Make environments for evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.logger.eval_envs,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    eval_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)
    return eval_env


# ====================================================================
# Replay buffer
# ---------------------------


def make_offline_replay_buffer(rb_cfg, device):
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
        direct_download=True,
    )

    data.append_transform(DoubleToFloat())
    data.append_transform(lambda td: td.to(device))

    return data


# ====================================================================
# Model
# -----


def make_diffusion_actor(cfg, eval_env, device):
    """Make DiffusionActor from config."""
    action_spec = eval_env.action_spec_unbatched.to(device)
    obs_dim = eval_env.observation_spec["observation"].shape[-1]
    action_dim = action_spec.shape[-1]

    score_network = make_score_network(cfg, obs_dim, action_dim, device)

    actor = DiffusionActor(
        action_dim=action_dim,
        score_network=score_network,
        num_steps=cfg.network.num_steps,
        beta_start=cfg.network.beta_start,
        beta_end=cfg.network.beta_end,
        spec=action_spec,
    )

    # Init lazy layers
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict().to(device)
        actor(td)

    return actor


def make_score_network(cfg, obs_dim, action_dim, device):
    """Build the score (noise-prediction) network."""
    input_dim = action_dim + obs_dim + 1  # noisy_action || obs || t
    hidden_sizes = cfg.network.hidden_sizes
    activation_class = get_activation(cfg)

    layers = []
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h, device=device))
        layers.append(activation_class())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, action_dim, device=device))

    return nn.Sequential(*layers)


# ====================================================================
# Loss
# ---------


def make_loss_module(actor):
    """Make DiffusionBCLoss module."""
    loss_module = DiffusionBCLoss(actor)
    return loss_module


# ====================================================================
# Optimizer
# ---------


def make_optimizer(cfg, loss_module):
    optimizer = optim.Adam(
        loss_module.actor_network_params.flatten_keys().values(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    return optimizer


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    logger.log_metrics(metrics, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    elif cfg.network.activation == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError(f"Unknown activation {cfg.network.activation}")


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
