# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Building blocks of the FQL recipe.

``make_data_and_envs`` loads an OGBench dataset into a replay buffer and builds
the training and evaluation environments, ``make_agent`` assembles the flow
policy, one-step student, twin critics, :class:`~torchrl.objectives.FQLLoss`
and target updater, and ``evaluate`` scores the student on complete episodes.
"""
from __future__ import annotations

import importlib.util

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn

from torchrl.checkpoint import StopOnSignal
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import (
    ActionScaling,
    Compose,
    default_info_dict_reader,
    DoubleToFloat,
    EnvBase,
    GymWrapper,
    StepCounter,
    TransformedEnv,
)
from torchrl.modules import FlowMatchingPolicy, MLP, OneStepPolicy, ValueOperator
from torchrl.objectives import FQLLoss, SoftUpdate


_has_ogbench = importlib.util.find_spec("ogbench") is not None
if _has_ogbench:
    import ogbench


REPLAY_KEYS = [
    "observation",
    "action",
    ("next", "observation"),
    ("next", "reward"),
    ("next", "done"),
    ("next", "terminated"),
    ("next", "truncated"),
]


class SuccessReader(default_info_dict_reader):
    """Copy OGBench's ``info["success"]`` flag into the tensordict for evaluation."""

    def __call__(self, info, tensordict):
        return tensordict.set(
            "success", torch.tensor(info.get("success", 0.0), dtype=torch.float32)
        )


def wrap_environment(raw_env, cfg):
    """Wrap a Gymnasium env with action scaling, float32 observations and a step cap."""
    env = GymWrapper(raw_env, device="cpu")
    if cfg.dataset.name:
        env.set_info_dict_reader(SuccessReader(["success"]))
    # Replay actions stay normalized; only the environment sees physical bounds.
    return TransformedEnv(
        env,
        Compose(
            ActionScaling(), DoubleToFloat(), StepCounter(cfg.env.max_episode_steps)
        ),
    )


def make_data_and_envs(cfg):
    """Return a replay buffer holding the offline data, a training env and an eval env.

    Without ``cfg.dataset.name``, a random rollout of ``cfg.dataset.random_frames``
    steps on ``cfg.env.name`` stands in for the dataset, for smoke tests only.
    """
    if cfg.dataset.name:
        if not _has_ogbench:
            raise ImportError("Install ogbench to load an OGBench dataset.")

        raw_env, dataset, _ = ogbench.make_env_and_datasets(
            cfg.dataset.name, dataset_dir=cfg.dataset.root
        )
        raw_eval = ogbench.make_env_and_datasets(cfg.dataset.name, env_only=True)
        # Masks control bootstrapping; episode boundaries can also be truncations.
        terminated = torch.as_tensor(1 - dataset["masks"], dtype=torch.bool).unsqueeze(
            -1
        )
        boundary = torch.as_tensor(dataset["terminals"], dtype=torch.bool).unsqueeze(-1)
        data = TensorDict(
            {
                "observation": torch.as_tensor(
                    dataset["observations"], dtype=torch.float32
                ),
                "action": torch.as_tensor(
                    dataset["actions"], dtype=torch.float32
                ).clamp(-1 + 1e-5, 1 - 1e-5),
                "next": {
                    "observation": torch.as_tensor(
                        dataset["next_observations"], dtype=torch.float32
                    ),
                    "reward": torch.as_tensor(
                        dataset["rewards"], dtype=torch.float32
                    ).unsqueeze(-1),
                    "done": boundary | terminated,
                    "terminated": terminated,
                    "truncated": boundary & ~terminated,
                },
            },
            batch_size=[len(terminated)],
        )
    else:
        raw_env = gym.make(cfg.env.name)
        raw_eval = gym.make(cfg.env.name)
        data = None

    env = wrap_environment(raw_env, cfg)
    eval_env = wrap_environment(raw_eval, cfg)
    env.set_seed(cfg.seed)
    eval_env.set_seed(cfg.seed + 1)
    if data is None:
        # Random data exercises the pipeline; it is not an offline benchmark.
        data = env.rollout(cfg.dataset.random_frames, break_when_any_done=False).select(
            *REPLAY_KEYS
        )
    # Keep all offline transitions while uniformly mixing in incoming online data.
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(data.numel() + cfg.optim.online_steps),
        batch_size=cfg.optim.batch_size,
    )
    replay.extend(data)
    return replay, env, eval_env


def make_network(in_features, out_features, cfg, device, layer_norm=False):
    """Build an MLP initialized and normalized like the reference FQL networks."""

    def activation():
        # FQL puts normalization after GELU; MLP's norm_class puts it before.
        return (
            nn.Sequential(
                nn.GELU(approximate="tanh"), nn.LayerNorm(cfg.network.width, eps=1e-6)
            )
            if layer_norm
            else nn.GELU(approximate="tanh")
        )

    model = MLP(
        in_features,
        out_features,
        num_cells=[cfg.network.width] * cfg.network.depth,
        activation_class=activation,
        device=device,
    )
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def make_agent(cfg, env, device):
    """Return the student policy, the FQL loss and its target updater."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    flow = FlowMatchingPolicy(
        make_network(obs_dim + action_dim + 1, action_dim, cfg, device),
        action_dim,
        cfg.network.num_steps,
    )
    student = OneStepPolicy(
        make_network(obs_dim + action_dim, action_dim, cfg, device), action_dim
    )
    critics = [
        ValueOperator(
            make_network(obs_dim + action_dim, 1, cfg, device, layer_norm=True),
            in_keys=["observation", "action"],
        )
        for _ in range(2)
    ]
    loss = FQLLoss(
        flow,
        student,
        critics,
        alpha=cfg.loss.alpha,
        q_aggregation=cfg.loss.q_aggregation,
        normalize_q_loss=cfg.loss.normalize_q_loss,
    )
    loss.make_value_estimator(gamma=cfg.loss.gamma)
    updater = SoftUpdate(loss, tau=cfg.loss.tau)
    return student, loss, updater


@torch.no_grad()
def evaluate(
    policy: TensorDictModuleBase,
    env: EnvBase,
    cfg: DictConfig,
    stop: StopOnSignal | None = None,
) -> TensorDict:
    """Average complete evaluation episodes without advancing training RNG streams."""
    episodes = []
    devices = list(
        {parameter.device for parameter in policy.parameters() if parameter.is_cuda}
    )
    with torch.random.fork_rng(devices=devices):
        for _ in range(cfg.evaluation.episodes):
            if stop is not None and stop.requested:
                return TensorDict({}, [])
            rollout = env.rollout(
                cfg.env.max_episode_steps, policy, auto_cast_to_device=True
            )
            metrics = TensorDict(
                {"evaluation/return": rollout["next", "reward"].sum()}, []
            )
            if cfg.dataset.name:
                metrics["evaluation/success"] = rollout["next", "success"].max()
            episodes.append(metrics)
    if stop is not None and stop.requested:
        return TensorDict({}, [])
    return torch.stack(episodes).mean(0)
