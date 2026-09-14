# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CartPole environments, policy construction and fixed-window replay for GTrXL PPO."""
from __future__ import annotations

import functools as ft
import math

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.envs import (
    Compose,
    GymEnv,
    InitTracker,
    ObservationTransform,
    RewardSum,
    SerialEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import GTrXL, ProbabilisticActor, TransformerModule, ValueOperator


class GTrXLState(TypedTensorDict):
    """Example schema; container selection remains configurable in this experiment."""

    memory: torch.Tensor
    valid: torch.Tensor


class GTrXLTensorClass(TensorClass):
    """Equivalent tensorclass schema for the container comparison."""

    memory: torch.Tensor
    valid: torch.Tensor


class HideVelocity(ObservationTransform):
    """Expose only cart position and pole angle, making CartPole partially observed."""

    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _apply_transform(self, observation):
        return observation[..., ::2]

    def transform_observation_spec(self, observation_spec):
        observation_spec["observation"] = observation_spec["observation"][..., ::2]
        return observation_spec


def make_env(num_envs, seed):
    """Create seeded CPU CartPole streams with velocity hidden and reset tracking."""
    env = TransformedEnv(
        SerialEnv(
            num_envs,
            ft.partial(GymEnv, "CartPole-v1", categorical_action_encoding=True),
        ),
        Compose(HideVelocity(), InitTracker(), RewardSum(), StepCounter()),
    )
    env.set_seed(seed)
    return env


def make_policy(cfg, device):
    """Share a GTrXL encoder between a categorical actor and scalar value head."""
    state_cls = {"td": TensorDict, "tc": GTrXLTensorClass, "ttd": GTrXLState}[
        cfg.container
    ]
    transformer = TransformerModule(
        transformer=GTrXL(
            2,
            cfg.hidden_size,
            cfg.num_layers,
            num_heads=cfg.num_heads,
            memory_len=cfg.memory_len,
            state_cls=state_cls,
            device=device,
        ),
        in_keys=["observation", "state"],
        out_keys=["features", ("next", "state")],
    )
    actor = ProbabilisticActor(
        TensorDictModule(
            nn.Linear(cfg.hidden_size, 2, device=device),
            in_keys=["features"],
            out_keys=["logits"],
        ),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    critic = ValueOperator(
        nn.Linear(cfg.hidden_size, 1, device=device), in_keys=["features"]
    )
    # Standard PPO head initialization starts near a uniform action distribution.
    nn.init.orthogonal_(actor[0].module.weight, gain=0.01)
    nn.init.zeros_(actor[0].module.bias)
    nn.init.orthogonal_(critic.module.weight, gain=1.0)
    nn.init.zeros_(critic.module.bias)
    nn.init.orthogonal_(transformer.transformer.embedding.weight, gain=math.sqrt(2))
    nn.init.zeros_(transformer.transformer.embedding.bias)
    policy = TensorDictSequential(transformer, actor, critic)
    return policy, transformer, actor, critic


def make_window_records(data, window_length):
    """Pack dense [env, time] transitions into records with one starting carry."""
    num_envs, length = data.batch_size
    initial_state = data["state"][:, ::window_length].clone().flatten(0, 1)
    # Drop both copies of the large state before padding or copying transitions.
    transitions = data.select(
        "observation",
        "is_init",
        "action",
        "action_log_prob",
        "advantage",
        "value_target",
        "state_value",
        ("next", "reward"),
        ("next", "done"),
        ("next", "terminated"),
    )
    transitions.set(
        ("collector", "mask"),
        torch.ones(*data.batch_size, dtype=torch.bool, device=data.device),
    )
    padding = -length % window_length
    if padding:
        zeros = transitions[:, :1].apply(torch.zeros_like)
        transitions = torch.cat((transitions, zeros.expand(num_envs, padding)), 1)
    transitions = transitions.reshape(-1, window_length)
    return TensorDict(
        {"initial_state": initial_state, "transitions": transitions},
        [transitions.shape[0]],
        device=data.device,
    )
