# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Building blocks of the Transitive RL recipe.

``make_replay_buffer`` loads an OGBench dataset into a replay buffer sampling
:class:`~torchrl.data.replay_buffers.SubgoalTriplet` records, ``make_agent``
builds the networks and :class:`~torchrl.objectives.TransitiveQLoss`, and
``evaluate`` runs the OGBench evaluation goals.
"""
from __future__ import annotations

import importlib.util
import math

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SubgoalTriplet
from torchrl.envs import (
    default_info_dict_reader,
    DoubleToFloat,
    GymWrapper,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.objectives import SoftUpdate, TransitiveQLoss

_has_ogbench = importlib.util.find_spec("ogbench") is not None
if _has_ogbench:
    import ogbench


class SuccessReader(default_info_dict_reader):
    """Copy OGBench's ``info["success"]`` flag into the tensordict for evaluation."""

    def __call__(self, info, tensordict):
        return tensordict.set(
            "success", torch.tensor(info.get("success", 0.0), dtype=torch.float32)
        )


class GoalObservation(gym.Wrapper):
    """Expose the OGBench evaluation goal, reported in the reset info, next to a float32 observation.

    Envs that report no goal use their reset observation, which only serves the
    synthetic smoke-test setup.
    """

    def __init__(self, env: gym.Env, goal_dim: int):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    -np.inf, np.inf, env.observation_space.shape, np.float32
                ),
                "goal": gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
            }
        )
        self._goal = np.zeros(goal_dim, np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._goal = np.asarray(info.get("goal", obs), np.float32)
        return {"observation": obs.astype(np.float32), "goal": self._goal}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            {"observation": obs.astype(np.float32), "goal": self._goal},
            reward,
            terminated,
            truncated,
            info,
        )


def load_dataset(cfg: DictConfig) -> tuple[TensorDict, gym.Env]:
    """Return the offline data and the raw evaluation env.

    Without ``cfg.dataset.name``, a random rollout of ``cfg.dataset.random_frames``
    steps on ``cfg.env.name`` stands in for the dataset and its observations for
    the goals, for smoke tests only.
    """
    if cfg.dataset.name:
        if not _has_ogbench:
            raise ImportError("Install ogbench to load an OGBench dataset.")
        raw_env, dataset, _ = ogbench.make_env_and_datasets(
            cfg.dataset.name, compact_dataset=True
        )
        observation = torch.as_tensor(dataset["observations"], dtype=torch.float32)
        data = TensorDict(
            {
                "observation": observation,
                "oracle_goal": torch.as_tensor(
                    dataset.get("oracle_reps", dataset["observations"]),
                    dtype=torch.float32,
                ),
                "action": torch.as_tensor(dataset["actions"], dtype=torch.float32),
                ("next", "done"): torch.as_tensor(
                    dataset["terminals"], dtype=torch.bool
                ).unsqueeze(-1),
            },
            batch_size=[observation.shape[0]],
        )
    else:
        raw_env = gym.make(cfg.env.name, max_episode_steps=cfg.env.max_episode_steps)
        source = TransformedEnv(
            GymWrapper(
                gym.make(cfg.env.name, max_episode_steps=cfg.env.max_episode_steps)
            ),
            DoubleToFloat(),
        )
        source.set_seed(cfg.seed)
        rollout = source.rollout(cfg.dataset.random_frames, break_when_any_done=False)
        data = rollout.select("observation", "action", ("next", "done"))
        data["oracle_goal"] = data["observation"]
        data["next", "done"][-1] = True
        source.close()
    data["action"] = data["action"].clamp(-1 + 1e-5, 1 - 1e-5)
    return data, raw_env


def make_replay_buffer(cfg: DictConfig, device: torch.device):
    """Return the replay buffer, the evaluation env and the policy-goal dimension."""
    data, raw_env = load_dataset(cfg)
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(data.numel(), device=device),
        batch_size=cfg.optim.batch_size,
        sample_unit=SubgoalTriplet(
            goal_discount=cfg.loss.discount if cfg.sampling.geometric_goal else None,
            policy_goal_discount=(
                cfg.loss.discount if cfg.sampling.geometric_policy_goal else None
            ),
        ),
    )
    replay.extend(data)
    goal_dim = data["oracle_goal"].shape[-1]
    env = GymWrapper(GoalObservation(raw_env, goal_dim), device="cpu")
    env.set_info_dict_reader(SuccessReader(["success"]))
    return replay, env, goal_dim


def mlp(
    in_features: int,
    hidden: list[int],
    out_features: int,
    *,
    device: torch.device,
    final_scale: float = 1.0,
) -> nn.Sequential:
    """Dense -> GELU -> LayerNorm blocks with fan-average uniform init, as in the reference."""
    layers = []
    for width in hidden:
        linear = nn.Linear(in_features, width, device=device)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers += [
            linear,
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(width, eps=1e-6, device=device),
        ]
        in_features = width
    linear = nn.Linear(in_features, out_features, device=device)
    nn.init.xavier_uniform_(linear.weight, gain=math.sqrt(final_scale))
    nn.init.zeros_(linear.bias)
    layers.append(linear)
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Goal-conditioned critic logit ``Q(s, a, g)`` on concatenated inputs."""

    def __init__(self, in_features: int, hidden: list[int], device: torch.device):
        super().__init__()
        self.net = mlp(in_features, hidden, 1, device=device)

    def forward(self, observation, goal, action):
        return self.net(torch.cat([observation, goal, action], -1))


class Actor(nn.Module):
    """Gaussian policy head with a constant unit standard deviation."""

    def __init__(
        self, in_features: int, hidden: list[int], action_dim: int, device: torch.device
    ):
        super().__init__()
        self.net = mlp(in_features, hidden, action_dim, device=device, final_scale=1e-2)

    def forward(self, observation, goal):
        loc = self.net(torch.cat([observation, goal], -1))
        return loc, torch.ones_like(loc)


def make_agent(cfg: DictConfig, replay, goal_dim: int, device: torch.device):
    """Return the policy, the TRL loss and its target updater."""
    sample = replay.storage[:2]
    obs_dim = sample["observation"].shape[-1]
    action_dim = sample["action"].shape[-1]
    hidden = list(cfg.network.hidden)
    oracle = cfg.loss.oracle_distill
    if not oracle and goal_dim != obs_dim:
        raise ValueError(
            "loss.oracle_distill=false needs goals in the observation space, but "
            f"{cfg.dataset.name} provides {goal_dim}-dimensional oracle goals; "
            "set loss.oracle_distill=true."
        )
    critic_keys = ["observation", "goal", "action"]

    def critics(goal_features):
        return [
            TensorDictModule(
                Critic(obs_dim + goal_features + action_dim, hidden, device),
                in_keys=critic_keys,
                out_keys=["state_action_value"],
            )
            for _ in range(2)
        ]

    actor = ProbabilisticActor(
        TensorDictModule(
            Actor(obs_dim + goal_dim, hidden, action_dim, device),
            in_keys=["observation", "goal"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=Bounded(-1.0, 1.0, (action_dim,), device=device),
        safe=True,
        distribution_class=IndependentNormal,
        default_interaction_type=ExplorationType.DETERMINISTIC,
    )
    loss = TransitiveQLoss(
        actor,
        critics(obs_dim),
        oracle_qvalue_network=critics(goal_dim) if oracle else None,
        discount=cfg.loss.discount,
        expectile=cfg.loss.expectile,
        distance_weight=cfg.loss.distance_weight,
        alpha=cfg.loss.alpha,
    )
    loss.set_keys(policy_goal="oracle_goal")
    updater = SoftUpdate(loss, tau=cfg.loss.tau)
    return actor, loss, updater


@torch.no_grad()
def evaluate(policy, env, cfg: DictConfig) -> TensorDict:
    """Success rate on each OGBench evaluation task and over all tasks.

    Each task runs ``cfg.evaluation.episodes`` deterministic episodes; an
    episode succeeds if the env reports success on its last step.
    """
    success = TensorDict(batch_size=[cfg.evaluation.episodes])
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for task_id in range(1, cfg.evaluation.num_tasks + 1):
            success[f"evaluation/task{task_id}"] = torch.stack(
                [
                    env.rollout(
                        cfg.evaluation.max_steps,
                        policy,
                        auto_reset=False,
                        tensordict=env.reset(options={"task_id": task_id}),
                        auto_cast_to_device=True,
                    )["next", "success"][-1].float()
                    for _ in range(cfg.evaluation.episodes)
                ]
            )
    success = success.mean(0)
    success["evaluation/overall"] = torch.stack(list(success.values())).mean()
    return success
