# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import Compose, DoubleToFloat, EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.data import CompositeSpec
from torchrl.objectives.common import LossModule
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Tuple


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
        buffer_scratch_dir="/tmp/",
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
            self.add_module(f'critic_net_{i}', net)
            self.nets.append(net)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        quantiles = torch.stack(tuple(net(*inputs) for net in self.nets), dim=-2)  # batch x n_nets x n_quantiles
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
# Quantile Huber Loss
# -------------------


def quantile_huber_loss_f(quantiles, samples):
    """
    Quantile Huber loss from the original PyTorch TQC implementation.
    See: https://github.com/SamsungLabs/tqc_pytorch/blob/master/tqc/functions.py

    quantiles is assumed to be of shape [batch size, n_nets, n_quantiles]
    samples is assumed to be of shape [batch size, n_samples]
    Arbitrary batch sizes are allowed.
    """
    pairwise_delta = samples[..., None, None, :] - quantiles[..., None]  # batch x n_nets x n_quantiles x n_samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)
    n_quantiles = quantiles.shape[-1]
    tau = torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[..., None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


# ====================================================================
# TQC Loss
# --------

class TQCLoss(LossModule):
    def __init__(
            self,
            actor_network,
            qvalue_network,
            gamma,
            top_quantiles_to_drop,
            alpha_init,
            device
    ):
        super(type(self), self).__init__()
        super().__init__()

        self.convert_to_functional(
            actor_network,
            "actor",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )

        self.convert_to_functional(
            qvalue_network,
            "critic",
            create_target_params=True  # Create a target critic network
        )

        self.device = device
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True, device=self.device)
        self.gamma = gamma
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # Compute target entropy
        action_spec = getattr(self.actor, "spec", None)
        if action_spec is None:
            print("Could not deduce action spec from actor network.")
        if not isinstance(action_spec, CompositeSpec):
            action_spec = CompositeSpec({"action": action_spec})
        action_container_len = len(action_spec.shape)
        self.target_entropy = -float(action_spec["action"].shape[action_container_len:].numel())

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_next = tensordict.get("next")
        reward = td_next.get("reward")
        not_done = tensordict.get("done").logical_not()

        alpha = torch.exp(self.log_alpha)

        # Q-loss
        with torch.no_grad():
            # get policy action
            self.actor(td_next, params=self.actor_params)
            self.critic(td_next, params=self.target_critic_params)

            next_log_pi = td_next.get("sample_log_prob")
            next_log_pi = torch.unsqueeze(next_log_pi, dim=-1)

            # compute and cut quantiles at the next state
            next_z = td_next.get("state_action_value")
            sorted_z, _ = torch.sort(next_z.reshape(*tensordict.batch_size, -1))
            sorted_z_part = sorted_z[..., :-self.top_quantiles_to_drop]

            # compute target
            target = reward + not_done * self.gamma * (sorted_z_part - alpha * next_log_pi)

        self.critic(tensordict, params=self.critic_params)
        cur_z = tensordict.get("state_action_value")
        critic_loss = quantile_huber_loss_f(cur_z, target)

        # --- Policy and alpha loss ---
        self.actor(tensordict, params=self.actor_params)
        self.critic(tensordict, params=self.critic_params)
        new_log_pi = tensordict.get("sample_log_prob")
        alpha_loss = -self.log_alpha * (new_log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * new_log_pi - tensordict.get("state_action_value").mean(-1).mean(-1, keepdim=True)).mean()

        # --- Entropy ---
        with set_exploration_type(ExplorationType.RANDOM):
            dist = self.actor.get_dist(
                tensordict,
                params=self.actor_params,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm).detach()
        entropy = -log_prob.mean()

        return TensorDict(
            {
                "loss_critic": critic_loss,
                "loss_actor": actor_loss,
                "loss_alpha": alpha_loss,
                "alpha": alpha,
                "entropy": entropy,
            },
            batch_size=[]
        )


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create TQC loss
    loss_module = TQCLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        device=cfg.network.device,
        gamma=cfg.optim.gamma,
        top_quantiles_to_drop=cfg.network.top_quantiles_to_drop_per_net * cfg.network.n_nets,
        alpha_init=cfg.optim.alpha_init
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
