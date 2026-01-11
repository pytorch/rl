# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

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
from torchrl.modules import AdditiveGaussianModule, MLP, TanhModule, ValueOperator

from torchrl.objectives import SoftUpdate
from torchrl.objectives.td3_bc import TD3BCLoss
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
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.logger.eval_envs,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)
    return train_env


# ====================================================================
# Replay buffer
# ---------------------------


def make_offline_replay_buffer(rb_cfg, device):
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        # drop_last for compile
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


def make_td3_agent(cfg, train_env, device):
    """Make TD3 agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec_unbatched.to(device)

    actor_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=action_spec.shape[-1],
        activation_class=get_activation(cfg),
        device=device,
    )

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=["param"],
    )
    actor = TensorDictSequential(
        actor_module,
        TanhModule(
            in_keys=["param"],
            out_keys=["action"],
            spec=action_spec,
        ),
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=1,
        activation_class=get_activation(cfg),
        device=device,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue])

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = train_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    del td

    # Exploration wrappers:
    actor_model_explore = TensorDictSequential(
        model[0],
        AdditiveGaussianModule(
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=0.1,
            spec=action_spec,
            device=device,
        ),
    )
    return model, actor_model_explore


# ====================================================================
# TD3 Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create TD3 loss
    loss_module = TD3BCLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.loss_function,
        delay_actor=True,
        delay_qvalue=True,
        action_spec=model[0][1].spec,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        alpha=cfg.alpha,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)
    return loss_module, target_net_updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.values(True, True))
    actor_params = list(loss_module.actor_network_params.values(True, True))

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.adam_eps,
    )
    return optimizer_actor, optimizer_critic


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
