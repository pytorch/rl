# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
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
from torchrl.objectives.td3 import TD3Loss
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


def make_environment(cfg, logger, device):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
        device=device,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    partial = functools.partial(env_maker, cfg=cfg, from_pixels=cfg.logger.video)
    trsf_clone = train_env.transform.clone()
    if cfg.logger.video:
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            1,
            EnvCreator(partial),
            serial_for_single=True,
            device=device,
        ),
        trsf_clone,
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode, device):
    """Make collector."""
    collector_device = cfg.collector.device
    if collector_device in ("", None):
        collector_device = device
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        reset_at_each_iter=cfg.collector.reset_at_each_iter,
        device=collector_device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size: int,
    prb: bool = False,
    buffer_size: int = 1000000,
    scratch_dir: str | None = None,
    device: torch.device = "cpu",
    prefetch: int = 3,
    compile: bool = False,
):
    if compile:
        prefetch = 0
    if scratch_dir in ("", None):
        ctx = nullcontext(None)
    elif scratch_dir == "temp":
        ctx = tempfile.TemporaryDirectory()
    else:
        ctx = nullcontext(scratch_dir)
    with ctx as scratch_dir:
        storage_cls = (
            functools.partial(LazyTensorStorage, device=device, compilable=compile)
            if not scratch_dir
            else functools.partial(
                LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir
            )
        )

        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=storage_cls(buffer_size),
                batch_size=batch_size,
                compilable=compile,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=storage_cls(buffer_size),
                batch_size=batch_size,
                compilable=compile,
            )
        if scratch_dir:
            replay_buffer.append_transform(lambda td: td.to(device))
        return replay_buffer


# ====================================================================
# Model
# -----


def make_td3_agent(cfg, train_env, eval_env, device):
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
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    # Exploration wrappers:
    actor_model_explore = TensorDictSequential(
        actor,
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
    loss_module = TD3Loss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=True,
        delay_qvalue=True,
        action_spec=model[0][1].spec,
        policy_noise=cfg.optim.policy_noise,
        noise_clip=cfg.optim.noise_clip,
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
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
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
