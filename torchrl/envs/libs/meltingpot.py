# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib

from typing import Dict, List, Mapping, Sequence

import torch

from tensordict import TensorDict, TensorDictBase

from torchrl.data import CompositeSpec, DiscreteTensorSpec, TensorSpec
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.dm_control import _dmcontrol_to_torchrl_spec_transform
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

_has_meltingpot = importlib.util.find_spec("meltingpot") is not None

PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."


def _get_envs():
    if not _has_meltingpot:
        raise ImportError("meltingpot is not installed in your virtual environment.")
    from meltingpot.configs import substrates as substrate_configs

    return list(substrate_configs.SUBSTRATES)


def _filter_global_state_from_dict(obs_dict: Dict, world: bool) -> Dict:  # noqa
    return {
        key: value
        for key, value in obs_dict.items()
        if ((_WORLD_PREFIX not in key) if not world else (_WORLD_PREFIX in key))
    }


def _remove_world_observations_from_obs_spec(
    observation_spec: Sequence[Mapping[str, "dm_env.specs.Array"]],  # noqa
) -> Sequence[Mapping[str, "dm_env.specs.Array"]]:  # noqa
    return [
        _filter_global_state_from_dict(agent_obs, world=False)
        for agent_obs in observation_spec
    ]


def _global_state_spec_from_obs_spec(
    observation_spec: Sequence[Mapping[str, "dm_env.specs.Array"]]  # noqa
) -> Mapping[str, "dm_env.specs.Array"]:  # noqa
    # We only look at agent 0 since world entries are the same for all agents
    world_entries = _filter_global_state_from_dict(observation_spec[0], world=True)
    if len(world_entries) != 1 and _WORLD_PREFIX + "RGB" not in world_entries:
        raise ValueError(
            f"Expected only one world entry named {_WORLD_PREFIX}RGB in observation_spec, but got {world_entries}"
        )
    return _remove_world_prefix(world_entries)


def _remove_world_prefix(world_entries: Dict) -> Dict:
    return {key[len(_WORLD_PREFIX) :]: value for key, value in world_entries.items()}


class MeltingpotWrapper(_EnvWrapper):
    """Meltingpot environment wrapper.

    GitHub: https://github.com/google-deepmind/meltingpot

    Paper: https://arxiv.org/abs/2211.13746

    Melting Pot assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
    and has been designed to test a broad range of social interactions such as: cooperation, competition, deception,
    reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a set of over 50 multi-agent
    reinforcement learning substrates (multi-agent games) on which to train agents, and over 256 unique test scenarios
    on which to evaluate these trained agents.

    Args:
        env (``meltingpot.utils.substrates.substrate.Substrate``): the meltingpot substrate to wrap.

    Keyword Args:
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon).
            Each Meltingpot substrate can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated (and the ``"terminated"`` flag is set) whenever this horizon is reached.
            Unlike gym's ``TimeLimit`` transform or torchrl's :class:`~torchrl.envs.transforms.StepCounter`,
            this argument will not set the ``"truncated"`` entry in the tensordict.
        categorical_actions (bool, optional): if the environment actions are discrete, whether to transform
            them to categorical or one-hot. Defaults to ``True``.
        group_map (MarlGroupMapType or Dict[str, List[str]], optional): how to group agents in tensordicts for
            input/output. By default, they will be all put
            in one group named ``"agents"``.
            Otherwise, a group map can be specified or selected from some premade options.
            See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.

    Attributes:
        group_map (Dict[str, List[str]]): how to group agents in tensordicts for
            input/output. See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        agent_names (list of str): names of the agent in the environment
        agent_names_to_indices_map (Dict[str, int]): dictionary mapping agent names to their index in the environment
        available_envs (List[str]): the list of the scenarios available to build.

    .. warning::
        Meltingpot returns a single ``done`` flag which does not distinguish between
        when the env reached ``max_steps`` and termination.
        If you deem the ``truncation`` signal necessary, set ``max_steps`` to
        ``None`` and use a :class:`~torchrl.envs.transforms.StepCounter` transform.

    Examples:
        >>> from meltingpot import substrate
        >>> from torchrl.envs.libs.meltingpot import MeltingpotWrapper
        >>> substrate_config = substrate.get_config("commons_harvest__open")
        >>> mp_env = substrate.build_from_config(
        ...     substrate_config, roles=substrate_config.default_player_roles
        ... )
        >>> env_torchrl = MeltingpotWrapper(env=mp_env)
        >>> print(env_torchrl.rollout(max_steps=5))
        TensorDict(
            fields={
                RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([5, 7]), device=cpu, dtype=torch.int64, is_shared=False),
                        observation: TensorDict(
                            fields={
                                COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
                            batch_size=torch.Size([5, 7]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([5, 7]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        agents: TensorDict(
                            fields={
                                observation: TensorDict(
                                    fields={
                                        COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                        READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                        RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
                                    batch_size=torch.Size([5, 7]),
                                    device=cpu,
                                    is_shared=False),
                                reward: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False)},
                            batch_size=torch.Size([5, 7]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)

    """

    git_url = "https://github.com/google-deepmind/meltingpot"
    libname = "melitingpot"

    @property
    def lib(self):
        import meltingpot

        return meltingpot

    @_classproperty
    def available_envs(cls):
        if not _has_meltingpot:
            return []
        return _get_envs()

    def __init__(
        self,
        env: "meltingpot.utils.substrates.substrate.Substrate" = None,  # noqa
        categorical_actions: bool = True,
        group_map: MarlGroupMapType
        | Dict[str, List[str]] = MarlGroupMapType.ALL_IN_ONE_GROUP,
        max_steps: int = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
        self.group_map = group_map
        self.categorical_actions = categorical_actions
        self.max_steps = max_steps
        self.num_cycles = 0
        super().__init__(**kwargs)

    def _build_env(
        self,
        env: "meltingpot.utils.substrates.substrate.Substrate",  # noqa
    ):
        return env

    def _make_group_map(self):
        if isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

    def _make_specs(
        self, env: "meltingpot.utils.substrates.substrate.Substrate"  # noqa
    ) -> None:
        mp_obs_spec = self._env.observation_spec()  # List of dict of arrays
        mp_obs_spec_no_world = _remove_world_observations_from_obs_spec(
            mp_obs_spec
        )  # List of dict of arrays
        mp_global_state_spec = _global_state_spec_from_obs_spec(
            mp_obs_spec
        )  # Dict of arrays
        mp_act_spec = self._env.action_spec()  # List of discrete arrays
        mp_rew_spec = self._env.reward_spec()  # List of arrays

        torchrl_agent_obs_specs = [
            _dmcontrol_to_torchrl_spec_transform(agent_obs_spec)
            for agent_obs_spec in mp_obs_spec_no_world
        ]
        torchrl_agent_act_specs = [
            _dmcontrol_to_torchrl_spec_transform(
                agent_act_spec, categorical_discrete_encoding=self.categorical_actions
            )
            for agent_act_spec in mp_act_spec
        ]
        torchrl_state_spec = _dmcontrol_to_torchrl_spec_transform(mp_global_state_spec)
        torchrl_rew_spec = [
            _dmcontrol_to_torchrl_spec_transform(agent_rew_spec)
            for agent_rew_spec in mp_rew_spec
        ]

        # Create and check group map
        _num_players = len(torchrl_rew_spec)
        self.agent_names = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(_num_players)
        ]
        self.agent_names_to_indices_map = {
            agent_name: i for i, agent_name in enumerate(self.agent_names)
        }
        self._make_group_map()

        action_spec = CompositeSpec()
        observation_spec = CompositeSpec()
        reward_spec = CompositeSpec()

        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
            ) = self._make_group_specs(
                group,
                torchrl_agent_obs_specs,
                torchrl_agent_act_specs,
                torchrl_rew_spec,
            )
            action_spec[group] = group_action_spec
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec

        observation_spec.update(torchrl_state_spec)
        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    n=2, shape=torch.Size((1,)), dtype=torch.bool
                ),
            },
        )
        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

    def _make_group_specs(
        self,
        group: str,
        torchrl_agent_obs_specs: List[TensorSpec],
        torchrl_agent_act_specs: List[TensorSpec],
        torchrl_rew_spec: List[TensorSpec],
    ):
        # Agent specs
        action_specs = []
        observation_specs = []
        reward_specs = []

        for agent_name in self.group_map[group]:
            agent_index = self.agent_names_to_indices_map[agent_name]
            action_specs.append(
                CompositeSpec(
                    {
                        "action": torchrl_agent_act_specs[
                            agent_index
                        ]  # shape = (n_actions_per_agent,)
                    },
                )
            )
            observation_specs.append(
                CompositeSpec(
                    {
                        "observation": torchrl_agent_obs_specs[
                            agent_index
                        ]  # shape = (n_obs_per_agent,)
                    },
                )
            )
            reward_specs.append(
                CompositeSpec({"reward": torchrl_rew_spec[agent_index]})  # shape = (1,)
            )

        # Create multi-agent specs
        group_action_spec = torch.stack(
            action_specs, dim=0
        )  # shape = (n_agents_in_group, n_actions_per_agent)
        group_observation_spec = torch.stack(
            observation_specs, dim=0
        )  # shape = (n_agents_in_group, n_obs_per_agent)
        group_reward_spec = torch.stack(
            reward_specs, dim=0
        )  # shape = (n_agents_in_group, 1)
        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
        )

    def _check_kwargs(self, kwargs: Dict):
        meltingpot = self.lib

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, meltingpot.utils.substrates.substrate.Substrate):
            raise TypeError(
                "env is not of type 'meltingpot.utils.substrates.substrate.Substrate'."
            )

    def _init_env(self):
        # Caching
        self.cached_full_done_spec_zero = self.full_done_spec.zero()

    def _set_seed(self, seed: int | None):
        raise NotImplementedError(
            "It is currently unclear how to set a seed in Meltingpot"
            "see https://github.com/google-deepmind/meltingpot/issues/129 to track the issue."
        )

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        self.num_cycles = 0
        timestep = self._env.reset()
        obs = timestep.observation

        td = self.cached_full_done_spec_zero.clone()

        for group, agent_names in self.group_map.items():
            agent_tds = []
            for index_in_group, agent_name in enumerate(agent_names):
                global_index = self.agent_names_to_indices_map[agent_name]
                agent_obs = self.observation_spec[group, "observation"][
                    index_in_group
                ].encode(_filter_global_state_from_dict(obs[global_index], world=False))
                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )

                agent_tds.append(agent_td)
            agent_tds = torch.stack(agent_tds, dim=0)
            td.set(group, agent_tds)

        # Global state
        td.update(
            _remove_world_prefix(_filter_global_state_from_dict(obs[0], world=True))
        )

        tensordict_out = TensorDict(
            source=td,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action_dict = {}
        for group, agents in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_np = self.full_action_spec[group, "action"].to_numpy(
                group_action
            )
            for index, agent in enumerate(agents):
                action_dict[agent] = group_action_np[index]

        actions = [action_dict[agent] for agent in self.agent_names]
        timestep = self._env.step(actions)
        self.num_cycles += 1

        rewards = timestep.reward
        done = timestep.last() or (
            (self.num_cycles >= self.max_steps) if self.max_steps is not None else False
        )
        obs = timestep.observation

        td = TensorDict(
            {
                "done": self.full_done_spec["done"].encode(done),
                "terminated": self.full_done_spec["terminated"].encode(done),
            },
            batch_size=self.batch_size,
        )
        # Global state
        td.update(
            _remove_world_prefix(_filter_global_state_from_dict(obs[0], world=True))
        )

        for group, agent_names in self.group_map.items():
            agent_tds = []
            for index_in_group, agent_name in enumerate(agent_names):
                global_index = self.agent_names_to_indices_map[agent_name]
                agent_obs = self.observation_spec[group, "observation"][
                    index_in_group
                ].encode(_filter_global_state_from_dict(obs[global_index], world=False))
                agent_reward = self.full_reward_spec[group, "reward"][
                    index_in_group
                ].encode(rewards[global_index])
                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                        "reward": agent_reward,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )

                agent_tds.append(agent_td)
            agent_tds = torch.stack(agent_tds, dim=0)
            td.set(group, agent_tds)

        return td

    def get_rgb_image(self) -> torch.Tensor:
        """Returns an RGB image of the environment.

        Returns:
            a ``torch.Tensor`` containing image in format WHC.

        """
        return torch.from_numpy(self._env.observation()[0][_WORLD_PREFIX + "RGB"])


class MeltingpotEnv(MeltingpotWrapper):
    """Meltingpot environment wrapper.

    GitHub: https://github.com/google-deepmind/meltingpot

    Paper: https://arxiv.org/abs/2211.13746

    Melting Pot assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
    and has been designed to test a broad range of social interactions such as: cooperation, competition, deception,
    reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a set of over 50 multi-agent
    reinforcement learning substrates (multi-agent games) on which to train agents, and over 256 unique test scenarios
    on which to evaluate these trained agents.

    Args:
        substrate(str or ml_collections.config_dict.ConfigDict): the meltingpot substrate to build.
            Can be a string from :attr:`~.available_envs` or a ConfigDict for the substrate

    Keyword Args:
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon).
            Each Meltingpot substrate can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated (and the ``"terminated"`` flag is set) whenever this horizon is reached.
            Unlike gym's ``TimeLimit`` transform or torchrl's :class:`~torchrl.envs.transforms.StepCounter`,
            this argument will not set the ``"truncated"`` entry in the tensordict.
        categorical_actions (bool, optional): if the environment actions are discrete, whether to transform
            them to categorical or one-hot. Defaults to ``True``.
        group_map (MarlGroupMapType or Dict[str, List[str]], optional): how to group agents in tensordicts for
            input/output. By default, they will be all put
            in one group named ``"agents"``.
            Otherwise, a group map can be specified or selected from some premade options.
            See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.


    Attributes:
        group_map (Dict[str, List[str]]): how to group agents in tensordicts for
            input/output. See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        agent_names (list of str): names of the agent in the environment
        agent_names_to_indices_map (Dict[str, int]): dictionary mapping agent names to their index in the enviornment
        available_envs (List[str]): the list of the scenarios available to build.

    .. warning::
        Meltingpot returns a single ``done`` flag which does not distinguish between
        when the env reached ``max_steps`` and termination.
        If you deem the ``truncation`` signal necessary, set ``max_steps`` to
        ``None`` and use a :class:`~torchrl.envs.transforms.StepCounter` transform.

    Examples:
        >>> from torchrl.envs.libs.meltingpot import MeltingpotEnv
        >>> env_torchrl = MeltingpotEnv("commons_harvest__open")
        >>> print(env_torchrl.rollout(max_steps=5))
        TensorDict(
            fields={
                RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([5, 7]), device=cpu, dtype=torch.int64, is_shared=False),
                        observation: TensorDict(
                            fields={
                                COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
                            batch_size=torch.Size([5, 7]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([5, 7]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        agents: TensorDict(
                            fields={
                                observation: TensorDict(
                                    fields={
                                        COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                        READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                                        RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
                                    batch_size=torch.Size([5, 7]),
                                    device=cpu,
                                    is_shared=False),
                                reward: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False)},
                            batch_size=torch.Size([5, 7]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)


    """

    def __init__(
        self,
        substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa
        *,
        max_steps: int | None = None,
        categorical_actions: bool = True,
        group_map: MarlGroupMapType
        | Dict[str, List[str]] = MarlGroupMapType.ALL_IN_ONE_GROUP,
        **kwargs,
    ):
        if not _has_meltingpot:
            raise ImportError(
                f"meltingpot python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
        super().__init__(
            substrate=substrate,
            max_steps=max_steps,
            categorical_actions=categorical_actions,
            group_map=group_map,
            **kwargs,
        )

    def _check_kwargs(self, kwargs: Dict):
        if "substrate" not in kwargs:
            raise TypeError("Could not find environment key 'substrate' in kwargs.")

    def _build_env(
        self,
        substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa
    ) -> "meltingpot.utils.substrates.substrate.Substrate":  # noqa
        from meltingpot import substrate as mp_substrate

        if isinstance(substrate, str):
            substrate_config = mp_substrate.get_config(substrate)
        else:
            substrate_config = substrate

        return super()._build_env(
            env=mp_substrate.build_from_config(
                substrate_config, roles=substrate_config.default_player_roles
            )
        )
