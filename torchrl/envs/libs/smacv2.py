# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import re

from typing import Dict, Optional

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Bounded, Categorical, Composite, OneHot, Unbounded
from torchrl.envs.common import _EnvWrapper

from torchrl.envs.utils import _classproperty, ACTION_MASK_ERROR

_has_smacv2 = importlib.util.find_spec("smacv2") is not None


def _get_envs():
    if not _has_smacv2:
        raise ImportError("SMAC-v2 is not installed in your virtual environment.")
    from smacv2.env.starcraft2.maps import smac_maps

    return list(smac_maps.get_smac_map_registry().keys())


class SMACv2Wrapper(_EnvWrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    To install the environment follow the following `guide <https://github.com/oxwhirl/smacv2#getting-started>`__.

    Examples:
        >>> from torchrl.envs.libs.smacv2 import SMACv2Wrapper
        >>> import smacv2
        >>> print(SMACv2Wrapper.available_envs)
        ['10gen_terran', '10gen_zerg', '10gen_protoss', '3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m',
         '27m_vs_30m', 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '3s_vs_4z', '3s_vs_5z', '1c3s5z',
          '2m_vs_1z', 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane', '2c_vs_64zg']
        >>> # You can use old SMAC maps
        >>> env = SMACv2Wrapper(smacv2.env.StarCraft2Env(map_name="MMM2"), categorical_actions=False)
        >>> print(env.rollout(5))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5, 10]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                info: TensorDict(
                    fields={
                        battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
                        dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                        dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                        episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
                                observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([5, 10]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
                                dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                                dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                                episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([5]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> # Or the new features for procedural generation
        >>> distribution_config = {
        ...     "n_units": 5,
        ...     "n_enemies": 6,
        ...     "team_gen": {
        ...         "dist_type": "weighted_teams",
        ...         "unit_types": ["marine", "marauder", "medivac"],
        ...         "exception_unit_types": ["medivac"],
        ...         "weights": [0.5, 0.2, 0.3],
        ...         "observe": True,
        ...     },
        ...     "start_positions": {
        ...         "dist_type": "surrounded_and_reflect",
        ...         "p": 0.5,
        ...         "n_enemies": 5,
        ...         "map_x": 32,
        ...         "map_y": 32,
        ...     },
        ... }
        >>> env = SMACv2Wrapper(
        ...     smacv2.env.StarCraft2Env(
        ...         map_name="10gen_terran",
        ...         capability_config=distribution_config,
        ...     )
        ... )
        >>> print(env.rollout(4))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4, 5]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                info: TensorDict(
                    fields={
                        battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                        dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                        episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                                observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([4, 5]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                                dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                                dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                                episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([4]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4]),
            device=cpu,
            is_shared=False)
    """

    git_url = "https://github.com/oxwhirl/smacv2"
    libname = "smacv2"

    @_classproperty
    def available_envs(cls):
        if not _has_smacv2:
            return []
        return list(_get_envs())

    def __init__(
        self,
        env: "smacv2.env.StarCraft2Env" = None,  # noqa: F821
        categorical_actions: bool = True,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
        self.categorical_actions = categorical_actions

        super().__init__(**kwargs)

    @property
    def lib(self):
        import smacv2

        return smacv2

    def _check_kwargs(self, kwargs: Dict):
        import smacv2

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, smacv2.env.StarCraft2Env):
            raise TypeError("env is not of type 'smacv2.env.StarCraft2Env'.")

    def _build_env(
        self,
        env: "smacv2.env.StarCraft2Env",  # noqa: F821
    ):
        if len(self.batch_size):
            raise RuntimeError(
                f"SMACv2 does not support custom batch_size {self.batch_size}."
            )

        return env

    def _make_specs(self, env: "smacv2.env.StarCraft2Env") -> None:  # noqa: F821
        self.group_map = {"agents": [str(i) for i in range(self.n_agents)]}
        self.reward_spec = Unbounded(
            shape=torch.Size((1,)),
            device=self.device,
        )
        self.done_spec = Categorical(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )
        self.full_action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()

    def _init_env(self) -> None:
        self._env.reset()
        self._update_action_mask()

    def _make_action_spec(self) -> Composite:
        if self.categorical_actions:
            action_spec = Categorical(
                self.n_actions,
                shape=torch.Size((self.n_agents,)),
                device=self.device,
                dtype=torch.long,
            )
        else:
            action_spec = OneHot(
                self.n_actions,
                shape=torch.Size((self.n_agents, self.n_actions)),
                device=self.device,
                dtype=torch.long,
            )
        spec = Composite(
            {
                "agents": Composite(
                    {"action": action_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        return spec

    def _make_observation_spec(self) -> Composite:
        obs_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=torch.Size([self.n_agents, self.get_obs_size()]),
            device=self.device,
            dtype=torch.float32,
        )
        info_spec = Composite(
            {
                "battle_won": Categorical(2, dtype=torch.bool, device=self.device),
                "episode_limit": Categorical(2, dtype=torch.bool, device=self.device),
                "dead_allies": Bounded(
                    low=0,
                    high=self.n_agents,
                    dtype=torch.long,
                    device=self.device,
                    shape=(),
                ),
                "dead_enemies": Bounded(
                    low=0,
                    high=self.n_enemies,
                    dtype=torch.long,
                    device=self.device,
                    shape=(),
                ),
            }
        )
        mask_spec = Categorical(
            2,
            torch.Size([self.n_agents, self.n_actions]),
            device=self.device,
            dtype=torch.bool,
        )
        spec = Composite(
            {
                "agents": Composite(
                    {"observation": obs_spec, "action_mask": mask_spec},
                    shape=torch.Size((self.n_agents,)),
                ),
                "state": Bounded(
                    low=-1.0,
                    high=1.0,
                    shape=torch.Size((self.get_state_size(),)),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "info": info_spec,
            }
        )
        return spec

    def _set_seed(self, seed: Optional[int]) -> None:
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

        mask = self._update_action_mask()

        # build results
        agents_td = TensorDict(
            {"observation": obs, "action_mask": mask}, batch_size=(self.n_agents,)
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
        action_np = self.full_action_spec[self.action_key].to_numpy(action)

        # Actions are validated by the environment.
        try:
            reward, done, info = self._env.step(action_np)
        except AssertionError as err:
            if re.match(r"Agent . cannot perform action .", str(err)):
                raise ACTION_MASK_ERROR
            else:
                raise err

        # collect outputs
        obs = self.get_obs()
        state = self.get_state()
        info = self.observation_spec["info"].encode(info)
        actual_keys = info.keys()
        for expected_key, spec in self.observation_spec["info"].items():
            if expected_key not in actual_keys:
                info[expected_key] = spec.zero()

        reward = torch.tensor(
            reward, device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        done = torch.tensor(done, device=self.device, dtype=torch.bool).unsqueeze(-1)

        mask = self._update_action_mask()

        # build results
        agents_td = TensorDict(
            {"observation": obs, "action_mask": mask}, batch_size=(self.n_agents,)
        )

        tensordict_out = TensorDict(
            source={
                "agents": agents_td,
                "state": state,
                "info": info,
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=(),
            device=self.device,
        )

        return tensordict_out

    def _update_action_mask(self):
        mask = torch.tensor(
            self.get_avail_actions(), dtype=torch.bool, device=self.device
        )
        self.full_action_spec[self.action_key].update_mask(mask)
        return mask

    def close(self, *, raise_if_closed: bool = True):
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
            return "marine"
        elif agent_info.unit_type == self.marauder_id:
            return "marauder"
        elif agent_info.unit_type == self.medivac_id:
            return "medivac"
        elif agent_info.unit_type == self.hydralisk_id:
            return "hydralisk"
        elif agent_info.unit_type == self.zergling_id:
            return "zergling"
        elif agent_info.unit_type == self.baneling_id:
            return "baneling"
        elif agent_info.unit_type == self.stalker_id:
            return "stalker"
        elif agent_info.unit_type == self.colossus_id:
            return "colossus"
        elif agent_info.unit_type == self.zealot_id:
            return "zealot"
        else:
            raise AssertionError(f"Agent type {agent_info.unit_type} unidentified")

    # This patches the bug in https://github.com/oxwhirl/smacv2/issues/33
    def render(self, mode: str = "human"):
        import smacv2

        if isinstance(self._env, smacv2.env.StarCraftCapabilityEnvWrapper):
            return self._env.env.render(mode=mode)
        else:
            return self._env.render(mode=mode)


class SMACv2Env(SMACv2Wrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    To install the environment follow the following `guide <https://github.com/oxwhirl/smacv2#getting-started>`__.

    Examples:
        >>> from torchrl.envs.libs.smacv2 import SMACv2Env
        >>> print(SMACv2Env.available_envs)
        ['10gen_terran', '10gen_zerg', '10gen_protoss', '3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m',
         '27m_vs_30m', 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '3s_vs_4z', '3s_vs_5z', '1c3s5z',
          '2m_vs_1z', 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane', '2c_vs_64zg']
        >>> # You can use old SMAC maps
        >>> env = SMACv2Env(map_name="MMM2")
        >>> print(env.rollout(5)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([5, 10]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                info: TensorDict(
                    fields={
                        battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
                        dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                        dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                        episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([5, 10, 18]), device=cpu, dtype=torch.bool, is_shared=False),
                                observation: Tensor(shape=torch.Size([5, 10, 176]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([5, 10]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                battle_won: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False),
                                dead_allies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                                dead_enemies: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int64, is_shared=False),
                                episode_limit: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([5]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                state: Tensor(shape=torch.Size([5, 322]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> # Or the new features for procedural generation
        >>> distribution_config = {
        ...     "n_units": 5,
        ...     "n_enemies": 6,
        ...     "team_gen": {
        ...         "dist_type": "weighted_teams",
        ...         "unit_types": ["marine", "marauder", "medivac"],
        ...         "exception_unit_types": ["medivac"],
        ...         "weights": [0.5, 0.2, 0.3],
        ...         "observe": True,
        ...     },
        ...     "start_positions": {
        ...         "dist_type": "surrounded_and_reflect",
        ...         "p": 0.5,
        ...         "n_enemies": 5,
        ...         "map_x": 32,
        ...         "map_y": 32,
        ...     },
        ... }
        >>> env = SMACv2Env(
        ...     map_name="10gen_terran",
        ...     capability_config=distribution_config,
        ...     categorical_actions=False,
        ... )
        >>> print(env.rollout(4))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4, 5]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                info: TensorDict(
                    fields={
                        battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                        dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                        dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                        episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                action_mask: Tensor(shape=torch.Size([4, 5, 12]), device=cpu, dtype=torch.bool, is_shared=False),
                                observation: Tensor(shape=torch.Size([4, 5, 88]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([4, 5]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                battle_won: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False),
                                dead_allies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                                dead_enemies: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                                episode_limit: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([4]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                state: Tensor(shape=torch.Size([4, 131]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4]),
            device=cpu,
            is_shared=False)
    """

    def __init__(
        self,
        map_name: str,
        capability_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        categorical_actions: bool = True,
        **kwargs,
    ):
        if not _has_smacv2:
            raise ImportError(
                f"smacv2 python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
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
    ) -> "smacv2.env.StarCraft2Env":  # noqa: F821
        import smacv2.env

        if capability_config is not None:
            env = smacv2.env.StarCraftCapabilityEnvWrapper(
                capability_config=capability_config,
                map_name=map_name,
                seed=seed,
                **kwargs,
            )
        else:
            env = smacv2.env.StarCraft2Env(map_name=map_name, seed=seed, **kwargs)

        return super()._build_env(env)
