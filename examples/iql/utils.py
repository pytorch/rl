# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ParallelEnv,
    RewardSum,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import IQLLoss, SoftUpdate

from torchrl.trainers.helpers.models import ACTIVATIONS


# ====================================================================
# Environment utils
# -----------------


def env_maker(task, device="cpu", from_pixels=False):
    with set_gym_backend("gym"):
        return GymEnv(task, device=device, from_pixels=from_pixels)


def apply_env_transforms(
    env,
):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, train_num_envs=1, eval_num_envs=1):
    """Make environments for training and evaluation."""
    parallel_env = ParallelEnv(
        train_num_envs,
        EnvCreator(lambda: env_maker(task=cfg.env.name)),
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env)

    eval_env = TransformedEnv(
        ParallelEnv(
            eval_num_envs,
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
        frames_per_batch=cfg.collector.frames_per_batch,
        init_random_frames=cfg.collector.init_random_frames,
        max_frames_per_traj=cfg.collector.max_frames_per_traj,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.collector_device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir=None,
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
                scratch_dir=buffer_scratch_dir,
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
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


def make_offline_replay_buffer(rb_cfg):
    data = D4RLExperienceReplay(
        name=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
    )

    data.append_transform(
        DoubleToFloat(
            in_keys=["observation", ("next", "observation")],
            in_keys_inv=[],
        )
    )

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_iql_model(cfg, train_env, eval_env, device="cpu"):
    model_cfg = cfg.model

    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net, q_net, value_net = make_iql_modules_state(model_cfg, eval_env)

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
        distribution_kwargs={
            "min": action_spec.space.minimum,
            "max": action_spec.space.maximum,
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
    in_keys = ["observation"]
    out_keys = ["state_value"]
    value_net = ValueOperator(
        in_keys=in_keys,
        out_keys=out_keys,
        module=value_net,
    )
    model = torch.nn.ModuleList([actor, qvalue, value_net]).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model


def make_iql_modules_state(model_cfg, proof_environment):
    action_spec = proof_environment.action_spec

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

    # Define Value Network
    value_net_kwargs = {
        "num_cells": model_cfg.hidden_sizes,
        "out_features": 1,
        "activation_class": ACTIVATIONS[model_cfg.activation],
    }
    value_net = MLP(**value_net_kwargs)

    return actor_net, q_net, value_net


# ====================================================================
# IQL Loss
# ---------


def make_loss(loss_cfg, model):
    loss_module = IQLLoss(
        model[0],
        model[1],
        value_network=model[2],
        loss_function=loss_cfg.loss_function,
        temperature=loss_cfg.temperature,
        expectile=loss_cfg.expectile,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma)
    target_net_updater = SoftUpdate(loss_module, tau=loss_cfg.tau)

    return loss_module, target_net_updater


def make_iql_optimizer(optim_cfg, loss_module):
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optim


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)
