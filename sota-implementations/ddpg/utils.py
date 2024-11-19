# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools

import torch

from tensordict.nn import TensorDictModule, TensorDictSequential

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
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
from torchrl.modules import (
    AdditiveGaussianModule,
    MLP,
    OrnsteinUhlenbeckProcessModule,
    TanhModule,
    ValueOperator,
)

from torchrl.objectives import SoftUpdate
from torchrl.objectives.ddpg import DDPGLoss
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
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


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


def make_environment(cfg, logger):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg, from_pixels=False)
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
            cfg.logger.num_eval_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    eval_env.set_seed(0)
    if cfg.logger.video:
        eval_env = eval_env.append_transform(
            VideoRecorder(logger, tag="rendered", in_keys=["pixels"])
        )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(
    cfg,
    train_env,
    actor_model_explore,
    compile=False,
    compile_mode=None,
    cudagraph=False,
    device: torch.device | None = None,
):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode, "fullgraph": True} if compile else False,
        cudagraph_policy=cudagraph,
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


def make_ddpg_agent(cfg, train_env, eval_env, device):
    """Make DDPG agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.single_action_spec
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

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
        ),
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

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

    # Exploration wrappers:
    if cfg.network.noise_type == "ou":
        actor_model_explore = TensorDictSequential(
            model[0],
            OrnsteinUhlenbeckProcessModule(
                spec=action_spec,
                annealing_num_steps=1_000_000,
                device=device,
                safe=False,
            ),
        )
    elif cfg.network.noise_type == "gaussian":
        actor_model_explore = TensorDictSequential(
            model[0],
            AdditiveGaussianModule(
                spec=action_spec,
                sigma_end=1.0,
                sigma_init=1.0,
                mean=0.0,
                std=0.1,
                device=device,
                safe=False,
            ),
        )
    else:
        raise NotImplementedError

    return model, actor_model_explore


# ====================================================================
# DDPG Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create DDPG loss
    loss_module = DDPGLoss(
        actor_network=model[0],
        value_network=model[1],
        loss_function=cfg.optim.loss_function,
        delay_actor=True,
        delay_value=True,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_optimizer(cfg, loss_module):
    critic_params = list(loss_module.value_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
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
