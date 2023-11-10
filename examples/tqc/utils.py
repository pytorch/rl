# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from contextlib import nullcontext
from typing import Tuple

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ActorCriticWrapper, MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, TQCLoss
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


# ====================================================================
# Environment utils
# -----------------


def env_maker(task, device="cpu"):
    with set_gym_backend("gym"):
        return GymEnv(
            task,
            device=device,
        )


def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg):
    """Make environments for training and evaluation."""
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda: env_maker(task=cfg.env.name)),
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(lambda: env_maker(task=cfg.env.name)),
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.collector_device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1_000_000,
    buffer_scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model architecture for critic
# -----------------------------


class TQC_Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nets = []
        qvalue_net_kwargs = {
            "num_cells": cfg.network.critic_hidden_sizes,
            "out_features": cfg.network.n_quantiles,
            "activation_class": get_activation(cfg),
        }
        for i in range(cfg.network.n_nets):
            net = MLP(**qvalue_net_kwargs)
            self.add_module(f"critic_net_{i}", net)
            self.nets.append(net)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        quantiles = torch.stack(
            tuple(net(*inputs) for net in self.nets), dim=-2
        )  # batch x n_nets x n_quantiles
        return quantiles


# ====================================================================
# Model
# -----


def make_tqc_agent(cfg, train_env, eval_env, device):
    """Make TQC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.actor_hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,  # can be omitted since this is default value
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    # Define Critic Network
    qvalue_net = TQC_Critic(cfg)

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]


# ====================================================================
# TQC Loss
# --------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create TQC loss
    top_quantiles_to_drop = (
        cfg.network.top_quantiles_to_drop_per_net * cfg.network.n_nets
    )
    loss_module = TQCLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        top_quantiles_to_drop=top_quantiles_to_drop,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(
        value_type=ValueEstimators.TD0, gamma=cfg.optim.gamma
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_tqc_optimizer(cfg, loss_module):
    critic_params = list(loss_module.critic_params.flatten_keys().values())
    actor_params = list(loss_module.actor_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# -------------


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
