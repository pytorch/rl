# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.minari_data import MinariExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    EGreedyModule,
    MLP,
    ProbabilisticActor,
    QValueActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import CQLLoss, DiscreteCQLLoss, SoftUpdate
from torchrl.record import VideoRecorder

from torchrl.trainers.helpers.models import ACTIVATIONS

# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.backend
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


def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, train_num_envs=1, eval_num_envs=1, logger=None):
    """Make environments for training and evaluation."""
    maker = functools.partial(env_maker, cfg)
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(maker),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env)

    maker = functools.partial(env_maker, cfg, from_pixels=cfg.logger.video)
    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
            EnvCreator(maker),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    eval_env.set_seed(0)
    if cfg.logger.video:
        eval_env = eval_env.insert_transform(
            0, VideoRecorder(logger=logger, tag="rendered", in_keys=["pixels"])
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
):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile else False,
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


def make_offline_replay_buffer(rb_cfg):
    data = MinariExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
        download=True,
    )

    data.append_transform(DoubleToFloat())

    return data


def make_offline_discrete_replay_buffer(rb_cfg):
    import gymnasium as gym
    import minari
    from minari import DataCollector

    # Create custom minari dataset from environment

    env = gym.make(rb_cfg.env)
    env = DataCollector(env)

    for _ in range(rb_cfg.episodes):
        env.reset(seed=123)
        while True:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.create_dataset(
        dataset_id=rb_cfg.dataset,
        algorithm_name="Random-Policy",
        code_permalink="https://github.com/Farama-Foundation/Minari",
        author="Farama",
        author_email="contact@farama.org",
    )

    data = MinariExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        load_from_local_minari=True,
        sampler=SamplerWithoutReplacement(drop_last=True),
        prefetch=4,
    )

    data.append_transform(DoubleToFloat())

    # Clean up
    minari.delete_dataset(rb_cfg.dataset)

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_cql_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model

    action_spec = train_env.action_spec_unbatched

    actor_net, q_net = make_cql_modules_state(model_cfg, eval_env)
    in_keys = ["observation"]
    out_keys = ["loc", "scale"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

    # We use a ProbabilisticActor to make sure that we map the
    # network output to the right space using a TanhDelta
    # distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        # Wrapping the kwargs in a TensorDictParams such that these items are
        #  send to device when necessary - not compatible with compile yet
        # distribution_kwargs=TensorDictParams(
        #     TensorDict(
        #         {
        #             "low": torch.as_tensor(action_spec.space.low, device=device),
        #             "high": torch.as_tensor(action_spec.space.high, device=device),
        #             "tanh_loc": NonTensorData(False),
        #         }
        #     ),
        #     no_convert=True,
        # ),
        distribution_kwargs={
            "low": action_spec.space.low.to(device),
            "high": action_spec.space.high.to(device),
            "tanh_loc": False,
        },
        default_interaction_type=ExplorationType.RANDOM,
    )

    in_keys = ["observation", "action"]

    out_keys = ["state_action_value"]
    qvalue = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=q_net,
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


def make_discretecql_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model

    action_spec = train_env.action_spec

    actor_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": action_spec.shape[-1],
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    actor_net = MLP(**actor_net_kwargs)
    qvalue_module = QValueActor(
        module=actor_net,
        spec=Composite(action=action_spec),
        in_keys=["observation"],
    )
    qvalue_module = qvalue_module.to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        qvalue_module(td)

    del td
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=action_spec,
    )
    model_explore = TensorDictSequential(
        qvalue_module,
        greedy_module,
    ).to(device)
    return qvalue_module, model_explore


def make_cql_modules_state(model_cfg, proof_environment):
    action_spec = proof_environment.action_spec_unbatched

    actor_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    actor_net = MLP(**actor_net_kwargs)
    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{model_cfg.default_policy_scale}",
        scale_lb=model_cfg.scale_lb,
    )
    actor_net = torch.nn.Sequential(actor_net, actor_extractor)

    qvalue_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": 1,
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }

    q_net = MLP(**qvalue_net_kwargs)

    return actor_net, q_net


# ====================================================================
# CQL Loss
# ---------


def make_continuous_loss(loss_cfg, model, device: torch.device | None = None):
    loss_module = CQLLoss(
        model[0],
        model[1],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        min_q_weight=loss_cfg.min_q_weight,
        max_q_backup=loss_cfg.max_q_backup,
        deterministic_backup=loss_cfg.deterministic_backup,
        num_random=loss_cfg.num_random,
        with_lagrange=loss_cfg.with_lagrange,
        lagrange_thresh=loss_cfg.lagrange_thresh,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma, device=device)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_discrete_loss(loss_cfg, model, device: torch.device | None = None):

    if "action_space" in loss_cfg:  # especify action space
        loss_module = DiscreteCQLLoss(
            model,
            loss_function=loss_cfg.loss_function,
            action_space=loss_cfg.action_space,
            delay_value=True,
        )
    else:
        loss_module = DiscreteCQLLoss(
            model,
            loss_function=loss_cfg.loss_function,
            delay_value=True,
        )

    loss_module.make_value_estimator(gamma=loss_cfg.gamma, device=device)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_discrete_cql_optimizer(cfg, loss_module):
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    return optim


def make_continuous_cql_optimizer(cfg, loss_module):
    critic_params = loss_module.qvalue_network_params.flatten_keys().values()
    actor_params = loss_module.actor_network_params.flatten_keys().values()
    actor_optim = torch.optim.Adam(
        actor_params,
        lr=cfg.optim.actor_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    critic_optim = torch.optim.Adam(
        critic_params,
        lr=cfg.optim.critic_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    alpha_optim = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=cfg.optim.actor_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    if loss_module.with_lagrange:
        alpha_prime_optim = torch.optim.Adam(
            [loss_module.log_alpha_prime],
            lr=cfg.optim.critic_lr,
        )
    else:
        alpha_prime_optim = None
    return actor_optim, critic_optim, alpha_optim, alpha_prime_optim


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
