# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.envs.utils import _classproperty

_has_lbforaging = importlib.util.find_spec("lbforaging") is not None


def _get_envs() -> list[str]:
    if not _has_lbforaging:
        raise ImportError("lbforaging is not installed in your virtual environment.")
    import gymnasium
    import lbforaging  # noqa: F401 - registers the Foraging-* Gymnasium environments.

    return sorted(k for k in gymnasium.envs.registry if k.startswith("Foraging-"))


class LBForagingWrapper(_EnvWrapper):
    """Level-Based Foraging environment wrapper.

    `Level-Based Foraging <https://github.com/semitable/lb-foraging>`__ is a
    fully-cooperative, sparse-reward Gymnasium environment for evaluating
    multi-agent credit assignment: a variable number of agents, each with a
    randomly assigned level, must coordinate to collect food items that also
    have levels, and a food item is only collected when the agents currently
    adjacent to it have levels summing to at least its own.

    ``lbforaging`` exposes a single ``gymnasium.Env`` whose observation,
    action and reward spaces are ``gymnasium.spaces.Tuple`` instances, one
    entry per agent, and whose ``terminated``/``truncated`` flags are shared
    by the whole team (LBF episodes end when either all food is collected or
    a step limit is reached, for every agent at once). This wrapper exposes
    that structure the way every other TorchRL multi-agent wrapper does:
    per-agent entries nested under a single ``"agents"`` group, and the
    shared ``done``/``terminated``/``truncated`` at the root.

    Args:
        env (gymnasium.Env): a Level-Based Foraging environment, i.e. the
            result of ``gymnasium.make("Foraging-<...>-v3")`` after
            ``import lbforaging``.

    Keyword Args:
        categorical_actions (bool, optional): whether discrete actions
            should be provided as categorical indices or one-hot encodings.
            Defaults to ``True``.
        seed (int, optional): the seed to use to reset the environment on
            the first call to :meth:`~.reset`. Defaults to ``None``.

    Examples:
        >>> import gymnasium
        >>> import lbforaging
        >>> from torchrl.envs.libs.lbforaging import LBForagingWrapper
        >>> base_env = gymnasium.make("Foraging-8x8-2p-3f-v3")
        >>> env = LBForagingWrapper(base_env, categorical_actions=False)
        >>> env.rollout(3)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([3, 2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 2, 15]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 2]),
                    device=None,
                    is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                observation: Tensor(shape=torch.Size([3, 2, 15]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([3, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([3, 2]),
                            device=None,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
    """

    git_url = "https://github.com/semitable/lb-foraging"
    libname = "lbforaging"

    @_classproperty
    def available_envs(cls):
        if not _has_lbforaging:
            return []
        return _get_envs()

    def __init__(
        self,
        env=None,
        *,
        categorical_actions: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        if env is not None:
            kwargs["env"] = env
        self.categorical_actions = categorical_actions
        self._seed = seed
        super().__init__(**kwargs)

    @property
    def lib(self):
        import lbforaging

        return lbforaging

    def _check_kwargs(self, kwargs: dict):
        import gymnasium
        from lbforaging.foraging.environment import ForagingEnv

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, gymnasium.Env) or not isinstance(
            env.unwrapped, ForagingEnv
        ):
            raise TypeError("env is not a Level-Based Foraging gymnasium.Env.")

    def _build_env(self, env, **kwargs):
        if len(self.batch_size):
            raise RuntimeError(
                f"LBForaging does not support custom batch_size {self.batch_size}."
            )
        return env

    def _make_specs(self, env) -> None:
        self.n_agents = env.unwrapped.n_agents
        self.group_map = {"agents": [str(i) for i in range(self.n_agents)]}

        observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space, device=self.device
        )
        action_spec = _gym_to_torchrl_spec_transform(
            env.action_space,
            categorical_action_encoding=self.categorical_actions,
            device=self.device,
        )
        self.full_observation_spec = Composite(
            {
                "agents": Composite(
                    {"observation": observation_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        self.full_action_spec = Composite(
            {
                "agents": Composite(
                    {"action": action_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        self.full_reward_spec = Composite(
            {
                "agents": Composite(
                    {
                        "reward": Unbounded(
                            shape=torch.Size((self.n_agents, 1)), device=self.device
                        )
                    },
                    shape=torch.Size((self.n_agents,)),
                )
            }
        )
        self.full_done_spec = Composite(
            {
                key: Categorical(
                    n=2, shape=torch.Size((1,)), dtype=torch.bool, device=self.device
                )
                for key in ("done", "terminated", "truncated")
            }
        )

    def _init_env(self) -> None:
        pass

    def _set_seed(self, seed: int | None) -> None:
        self._seed = seed

    def _stack_obs(self, observations) -> torch.Tensor:
        return torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=self.device
        )

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        observations, _ = self._env.reset(seed=self._seed)
        self._seed = None

        agents_td = TensorDict(
            {"observation": self._stack_obs(observations)},
            batch_size=torch.Size((self.n_agents,)),
            device=self.device,
        )
        return TensorDict(
            {"agents": agents_td}, batch_size=(), device=self.device
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(("agents", "action"))
        action_np = self.full_action_spec[self.action_key].to_numpy(action)
        observations, rewards, terminated, truncated, _info = self._env.step(
            [int(a) for a in action_np]
        )

        agents_td = TensorDict(
            {
                "observation": self._stack_obs(observations),
                "reward": torch.tensor(
                    rewards, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
            },
            batch_size=torch.Size((self.n_agents,)),
            device=self.device,
        )
        terminated_t = torch.tensor(
            [bool(terminated)], dtype=torch.bool, device=self.device
        )
        truncated_t = torch.tensor(
            [bool(truncated)], dtype=torch.bool, device=self.device
        )
        return TensorDict(
            {
                "agents": agents_td,
                "done": terminated_t | truncated_t,
                "terminated": terminated_t,
                "truncated": truncated_t,
            },
            batch_size=(),
            device=self.device,
        )

    def close(self, *, raise_if_closed: bool = True) -> None:
        self._env.close()


class LBForagingEnv(LBForagingWrapper):
    """Level-Based Foraging environment wrapper, constructed from an environment name.

    See :class:`~torchrl.envs.libs.lbforaging.LBForagingWrapper` for a
    description of the environment and the tensordict layout it produces.

    Args:
        env_name (str): the name of a registered Level-Based Foraging
            Gymnasium environment, e.g. ``"Foraging-8x8-2p-3f-v3"`` (an 8x8
            grid, 2 players, 3 food items). See
            :attr:`~.available_envs` for the full list.

    Keyword Args:
        categorical_actions (bool, optional): whether discrete actions
            should be provided as categorical indices or one-hot encodings.
            Defaults to ``True``.
        seed (int, optional): the seed to use to reset the environment on
            the first call to :meth:`~.reset`. Defaults to ``None``.
        **kwargs: forwarded to ``gymnasium.make``.

    Examples:
        >>> from torchrl.envs.libs.lbforaging import LBForagingEnv
        >>> env = LBForagingEnv("Foraging-8x8-2p-3f-v3", categorical_actions=False)
        >>> rollout = env.rollout(3)
    """

    def __init__(
        self,
        env_name: str,
        *,
        categorical_actions: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        if not _has_lbforaging:
            raise ImportError(
                f"lbforaging python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
        kwargs["env_name"] = env_name
        kwargs["categorical_actions"] = categorical_actions
        kwargs["seed"] = seed
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ):
        import gymnasium
        import lbforaging  # noqa: F401 - registers the Foraging-* Gymnasium environments.

        # LBF returns a per-agent list of rewards, which Gymnasium's passive
        # env checker flags as invalid (it expects a single scalar) even
        # though it is the documented, intended behaviour for this env.
        env = gymnasium.make(env_name, disable_env_checker=True, **kwargs)
        return super()._build_env(env)
