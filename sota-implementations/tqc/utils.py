# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim

from torchrl.collectors import Collector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, TQCLoss
from torchrl.record import VideoRecorder


ACTIVATIONS = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def env_maker(cfg, device="cpu", from_pixels=False):
    if cfg.env.library in ("gym", "gymnasium"):
        with set_gym_backend(cfg.env.library):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    if cfg.env.library == "dm_control":
        env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            from_pixels=from_pixels,
            pixels_only=False,
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    raise NotImplementedError(f"Unknown environment library {cfg.env.library}.")


def apply_env_transforms(env, max_episode_steps=1000):
    return TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )


def make_environment(cfg, logger=None):
    env_factory = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(env_factory),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)
    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    eval_factory = functools.partial(env_maker, cfg=cfg, from_pixels=cfg.logger.video)
    eval_transforms = train_env.transform.clone()
    if cfg.logger.video:
        eval_transforms.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(eval_factory),
            serial_for_single=True,
        ),
        eval_transforms,
    )
    return train_env, eval_env


def make_collector(cfg, train_env, exploration_policy, compile_mode):
    device = cfg.collector.device
    if device in ("", None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    collector = Collector(
        train_env,
        exploration_policy,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(cfg, device):
    storage = (
        LazyTensorStorage(cfg.replay_buffer.size, device=device)
        if not cfg.replay_buffer.scratch_dir
        else LazyMemmapStorage(
            cfg.replay_buffer.size,
            device="cpu",
            scratch_dir=cfg.replay_buffer.scratch_dir,
        )
    )
    replay_buffer_class = (
        TensorDictPrioritizedReplayBuffer
        if cfg.replay_buffer.prb
        else TensorDictReplayBuffer
    )
    replay_buffer_kwargs = {}
    if cfg.replay_buffer.prb:
        replay_buffer_kwargs.update(alpha=0.7, beta=0.5)
    replay_buffer = replay_buffer_class(
        storage=storage,
        batch_size=cfg.optim.batch_size,
        prefetch=3,
        **replay_buffer_kwargs,
    )
    if cfg.replay_buffer.scratch_dir:
        replay_buffer.append_transform(lambda data: data.to(device))
    return replay_buffer


def make_tqc_agent(cfg, train_env, eval_env, device):
    action_spec = train_env.action_spec_unbatched.to(device)
    activation = ACTIVATIONS[cfg.network.activation]
    actor_net = nn.Sequential(
        MLP(
            num_cells=cfg.network.actor_hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=activation,
            device=device,
        ),
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
            scale_lb=cfg.network.scale_lb,
        ),
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )
    critic = ValueOperator(
        in_keys=["action", "observation"],
        module=MLP(
            num_cells=cfg.network.critic_hidden_sizes,
            out_features=cfg.loss.num_quantiles,
            activation_class=activation,
            device=device,
        ),
    )
    model = nn.ModuleList([actor, critic])
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        data = eval_env.fake_tensordict().to(device)
        for network in model:
            network(data)
    return model, actor


def make_loss_module(cfg, model):
    loss_module = TQCLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=cfg.loss.num_qvalue_nets,
        top_quantiles_to_drop_per_net=cfg.loss.top_quantiles_to_drop_per_net,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)
    target_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_updater


def make_optimizers(cfg, loss_module):
    actor_parameters = loss_module.actor_network_params.flatten_keys().values()
    critic_parameters = loss_module.qvalue_network_params.flatten_keys().values()
    actor_optimizer = optim.Adam(
        actor_parameters,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    critic_optimizer = optim.Adam(
        critic_parameters,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    alpha_optimizer = optim.Adam([loss_module.log_alpha], lr=cfg.optim.lr)
    return actor_optimizer, critic_optimizer, alpha_optimizer


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
