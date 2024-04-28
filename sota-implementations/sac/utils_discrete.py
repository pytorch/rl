# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import tempfile
from contextlib import nullcontext

import torch
from tensordict.nn import InteractionType, TensorDictModule

from torch import nn, optim
from torchrl._utils import _append_last
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    CompositeSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    DTypeCastTransform,
    EnvCreator,
    FlattenObservation,
    InitTracker,
    JumanjiEnv,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, SafeModule
from torchrl.modules.distributions import OneHotCategorical

from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name, device=device, from_pixels=from_pixels, pixels_only=False
            )
    if lib.lower() == "jumanji":
        env = JumanjiEnv(
            cfg.env.name,
            device=device,
            from_pixels=from_pixels,
            batch_size=[],
            categorical_action_encoding=False,
        )
        env.set_seed(0)
        keys = set(env.observation_spec.keys(include_nested=True, leaves_only=True)) - {
            "pixels"
        }
        keys = [
            key for key in keys if not (isinstance(key, tuple) and key[0] == "state")
        ]
        ndim = env.ndim

        entries_to_flatten, dims = zip(
            *[
                (key, env.observation_spec[key].ndim)
                for key in keys
                if env.observation_spec[key].ndim >= ndim + 2
            ]
        )
        entries_to_flatten_suffix = {
            key: _append_last(key, "_flat") for key in entries_to_flatten
        }
        for dim, (key0, key1) in zip(dims, entries_to_flatten_suffix.items()):
            # 2 -> -2
            # 3 -> -3
            env = env.append_transform(
                FlattenObservation(
                    first_dim=-dim, last_dim=-1, in_keys=[key0], out_keys=[key1]
                )
            )
        keys = {
            key if key not in entries_to_flatten else entries_to_flatten_suffix[key]
            for key in keys
        }

        entries_to_unsqueeze = [
            key for key in keys if env.observation_spec[key].ndim == ndim
        ]
        entries_to_unsqueeze_suffix = {
            key: _append_last(key, "_unsq") for key in entries_to_unsqueeze
        }
        env = env.append_transform(
            UnsqueezeTransform(
                unsqueeze_dim=-1,
                in_keys=entries_to_unsqueeze,
                out_keys=list(entries_to_unsqueeze_suffix.values()),
            )
        )
        keys = {
            key if key not in entries_to_unsqueeze else entries_to_unsqueeze_suffix[key]
            for key in keys
        }

        env = env.append_transform(
            CatTensors(in_keys=list(keys), out_key="observation")
        )
        env = env.append_transform(
            DTypeCastTransform(
                env.observation_spec["observation"].dtype,
                torch.float32,
                in_keys=["observation"],
            )
        )
        return env
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
    maker = functools.partial(env_maker, cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(
        parallel_env, max_episode_steps=cfg.env.max_episode_steps
    )

    maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        device=device,
        storing_device="cpu",
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    with (
        tempfile.TemporaryDirectory()
        if scratch_dir is None
        else nullcontext(scratch_dir)
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
# Model
# -----


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make discrete SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    # Define Actor Network
    in_keys = ["observation"]

    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    actor_module = SafeModule(
        module=actor_net,
        in_keys=in_keys,
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        spec=CompositeSpec(action=eval_env.action_spec),
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        distribution_kwargs={},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }
    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = TensorDictModule(
        in_keys=in_keys,
        out_keys=["action_value"],
        module=qvalue_net,
    )

    model = torch.nn.ModuleList([actor, qvalue]).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model


# ====================================================================
# Discrete SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create discrete SAC loss
    loss_module = DiscreteSACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_actions=model[0].spec["action"].space.n,
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        target_entropy_weight=cfg.optim.target_entropy_weight,
        delay_qvalue=True,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


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
