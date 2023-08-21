# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper

IMPORT_ERR = None
try:
    import smacv2
    from smacv2.env.starcraft2.maps import smac_maps

    _has_smacv2 = True
except ImportError as err:
    _has_smacv2 = False
    IMPORT_ERR = err


def _get_envs():
    if not _has_smacv2:
        return []
    return smac_maps.get_smac_map_registry().keys()


class SMACv2Wrapper(_EnvWrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    Examples:
        >>> env = smac.env.StarCraft2Env("8m")
        >>> env = SMACv2Wrapper(env)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([8, 14]), dtype=torch.int64),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        obs: Tensor(torch.Size([8, 80]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                obs: Tensor(torch.Size([8, 80]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', ...]
    """

    git_url = "https://github.com/oxwhirl/smacv2"
    libname = "smacv2"
    available_envs = _get_envs()

    def __init__(
        self,
        env: "smacv2.env.StarCraft2Env" = None,
        categorical_actions: bool = False,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
        self.categorical_actions = categorical_actions

        super().__init__(**kwargs)

    @property
    def lib(self):
        return smacv2

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, smacv2.env.StarCraft2Env):
            raise TypeError("env is not of type 'smacv2.env.StarCraft2Env'.")

    def _build_env(
        self,
        env: "smacv2.env.StarCraft2Env",
    ):
        if len(self.batch_size):
            raise RuntimeError(
                f"SMACv2 does not support custom batch_size {self.batch_size}."
            )

        return env

    def _make_specs(self, env: "smacv2.env.StarCraft2Env") -> None:
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1,)),
            device=self.device,
        )
        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )
        self.action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()

    def _init_env(self) -> None:
        self._env.reset()
        self.update_action_mask()

    def _make_action_spec(self) -> CompositeSpec:
        if self.categorical_actions:
            action_spec = DiscreteTensorSpec(
                self.n_actions,
                shape=torch.Size((self.n_agents,)),
                device=self.device,
                dtype=torch.long,
            )
        else:
            action_spec = OneHotDiscreteTensorSpec(
                self.n_actions,
                shape=torch.Size((self.n_agents, self.n_actions)),
                device=self.device,
                dtype=torch.long,
            )
        spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": action_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        return spec

    def _make_observation_spec(self) -> CompositeSpec:
        obs_spec = BoundedTensorSpec(
            minimum=-1.0,
            maximum=1.0,
            shape=torch.Size([self.n_agents, self.get_obs_size()]),
            device=self.device,
            dtype=torch.float32,
        )
        info_spec = CompositeSpec(
            {
                "battle_won": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device
                ),
                "episode_limit": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device
                ),
                "dead_allies": BoundedTensorSpec(
                    minimum=0,
                    maximum=self.n_agents,
                    dtype=torch.long,
                    device=self.device,
                    shape=(),
                ),
                "dead_enemies": BoundedTensorSpec(
                    minimum=0,
                    maximum=self.n_enemies,
                    dtype=torch.long,
                    device=self.device,
                    shape=(),
                ),
            }
        )
        mask_spec = DiscreteTensorSpec(
            2,
            torch.Size([self.n_agents, self.n_actions]),
            device=self.device,
            dtype=torch.bool,
        )
        spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"obs": obs_spec, "action_mask": mask_spec},
                    shape=torch.Size((self.n_agents,)),
                ),
                "state": BoundedTensorSpec(
                    minimum=-1.0,
                    maximum=1.0,
                    shape=torch.Size((self.get_state_size(),)),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "info": info_spec,
            }
        )
        return spec

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            raise NotImplementedError(
                "Seed cannot be changed once environment was created."
            )

    def get_obs(self):
        obs = self._env.get_obs()
        return self._to_tensor(obs)

    def get_state(self):
        state = self._env.get_state()
        return self._to_tensor(state)

    def _to_tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        obs, state = self._env.reset()

        # collect outputs
        obs = self._to_tensor(obs)
        state = self._to_tensor(state)
        info = self.observation_spec["info"].zero()

        mask = self.update_action_mask()

        # build results
        agents_td = TensorDict(
            {"obs": obs, "action_mask": mask}, batch_size=(self.n_agents,)
        )
        tensordict_out = TensorDict(
            source={"agents": agents_td, "state": state, "info": info},
            batch_size=(),
            device=self.device,
        )

        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # perform actions
        action = tensordict.get(("agents", "action"))
        action_np = self.action_spec.to_numpy(action)

        # Actions are validated by the environment.
        reward, done, info = self._env.step(action_np)

        # collect outputs
        obs = self.get_obs()
        state = self.get_state()
        info = self.observation_spec["info"].encode(info)
        if "episode_limit" not in info.keys():
            info["episode_limit"] = self.observation_spec["info"][
                "episode_limit"
            ].zero()

        reward = torch.tensor(
            reward, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        done = torch.tensor(done, device=self.device, dtype=torch.bool).unsqueeze(-1)

        mask = self.update_action_mask()

        # build results
        agents_td = TensorDict(
            {"obs": obs, "action_mask": mask}, batch_size=(self.n_agents,)
        )

        tensordict_out = TensorDict(
            source={
                "next": {
                    "agents": agents_td,
                    "state": state,
                    "info": info,
                    "reward": reward,
                    "done": done,
                }
            },
            batch_size=(),
            device=self.device,
        )

        return tensordict_out

    def update_action_mask(self):
        mask = torch.tensor(
            self.get_avail_actions(), dtype=torch.bool, device=self.device
        )
        self.action_spec.update_mask(mask)
        return mask

    def close(self):
        # Closes StarCraft II
        self._env.close()

    def get_agent_type(self, agent_index: int) -> str:
        """Get the agent type string.

        Given the agent index, get its unit type name.

        Args:
            agent_index (int): the index of the agent to get the type of

        """
        if agent_index < 0 or agent_index >= self.n_agents:
            raise ValueError(f"Agent index out of range, {self.n_agents} available")

        agent_info = self.agents[agent_index]
        if agent_info.unit_type == self.marine_id:
            agent_type = "marine"
        elif agent_info.unit_type == self.marauder_id:
            agent_type = "marauder"
        elif agent_info.unit_type == self.medivac_id:
            agent_type = "medivac"
        elif agent_info.unit_type == self.hydralisk_id:
            agent_type = "hydralisk"
        elif agent_info.unit_type == self.zergling_id:
            agent_type = "zergling"
        elif agent_info.unit_type == self.baneling_id:
            agent_type = "baneling"
        elif agent_info.unit_type == self.stalker_id:
            agent_type = "stalker"
        elif agent_info.unit_type == self.colossus_id:
            agent_type = "colossus"
        elif agent_info.unit_type == self.zealot_id:
            agent_type = "zealot"
        else:
            raise AssertionError(f"Agent type {agent_info.unit_type} unidentified")

        return agent_type


class SMACv2Env(SMACv2Wrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    Examples:
        >>> env = SMACv2Env(map_name="8m")
        >>> print(env.available_envs)
        ['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', ...]
    """

    def __init__(
        self,
        map_name: str,
        capability_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        categorical_actions: bool = False,
        **kwargs,
    ):
        if not _has_smacv2:
            raise ImportError(
                f"smacv2 python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            ) from IMPORT_ERR
        kwargs["map_name"] = map_name
        kwargs["capability_config"] = capability_config
        kwargs["seed"] = seed
        kwargs["categorical_actions"] = categorical_actions

        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "map_name" not in kwargs:
            raise TypeError("Expected 'map_name' to be part of kwargs")

    def _build_env(
        self,
        map_name: str,
        capability_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "smacv2.env.StarCraft2Env":

        if capability_config is not None:
            env = smacv2.env.StarCraftCapabilityEnvWrapper(
                capability_config=capability_config, map_name=map_name, seed=seed
            )
        else:
            env = smacv2.env.StarCraft2Env(map_name=map_name, seed=seed)

        return super()._build_env(env)
