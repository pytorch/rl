# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for the Isaac Lab recurrent PPO example.

Isaac Lab is only imported lazily inside :func:`make_env`, so the
top-level main script can run without ``isaaclab`` on the Python path
and only the worker subprocesses (where ``make_env`` is called) pay the
import cost.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl.envs import ExplorationType
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

RnnBackend = Literal["pad", "scan", "triton"]


def make_env(task: str, num_envs: int, max_episode_steps: int, device: str):
    """Build an Isaac Lab env. Imports ``isaaclab`` lazily (worker-only)."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    if task == "Isaac-Ant-v0":
        cfg = AntEnvCfg()
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = num_envs
        if hasattr(cfg, "sim") and hasattr(cfg.sim, "device"):
            cfg.sim.device = device
        if hasattr(cfg, "device"):
            cfg.device = device
        if (
            hasattr(cfg, "episode_length_s")
            and hasattr(cfg, "sim")
            and hasattr(cfg.sim, "dt")
        ):
            cfg.episode_length_s = max_episode_steps * cfg.sim.dt
        env = gym.make(task, cfg=cfg)
    else:
        env = gym.make(task)
    return IsaacLabWrapper(env, device=torch.device(device))


def make_models(
    *,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: RnnBackend,
    device: torch.device,
) -> tuple[ProbabilisticActor, ValueOperator, TensorDictSequential]:
    """Build the actor, critic head, and the full value module sharing the backbone.

    The actor is a :class:`~torchrl.modules.ProbabilisticActor` wrapping a
    shared embed + LSTM backbone and a per-action Gaussian head. The value
    module reuses the same backbone (so a single GAE pass amortises the
    recurrent compute across the policy and the value head).
    """
    embed = TensorDictModule(
        nn.Linear(obs_dim, hidden_size, device=device),
        in_keys=["policy"],
        out_keys=["embed"],
    )
    lstm = LSTMModule(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        in_keys=["embed", "recurrent_state_h", "recurrent_state_c"],
        out_keys=[
            "lstm_out",
            ("next", "recurrent_state_h"),
            ("next", "recurrent_state_c"),
        ],
        recurrent_backend=rnn_backend,
        device=device,
    )
    backbone = TensorDictSequential(embed, lstm)

    actor_head = TensorDictModule(
        nn.Sequential(
            MLP(
                in_features=hidden_size,
                out_features=action_dim,
                num_cells=[],
                activation_class=nn.Tanh,
                device=device,
            ),
            AddStateIndependentNormalScale(action_dim, scale_lb=1e-4).to(device),
        ),
        in_keys=["lstm_out"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=TensorDictSequential(backbone, actor_head),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # The value module re-uses the backbone (shared params, single LSTM call
    # per GAE/update pass). An identity TDM caches the lstm_out under a
    # distinct key so the critic head and the actor head don't fight over
    # write semantics on a shared key.
    value_feature = TensorDictModule(
        nn.Identity(), in_keys=["lstm_out"], out_keys=["value_lstm_out"]
    )
    critic = ValueOperator(
        nn.Linear(hidden_size, 1, device=device),
        in_keys=["value_lstm_out"],
    ).to(device)
    full_value = TensorDictSequential(backbone, value_feature, critic)
    return actor, critic, full_value
