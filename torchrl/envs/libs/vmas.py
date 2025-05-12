# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import warnings

import torch
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    DEVICE_TYPING,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    StackedComposite,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import gym_backend, set_gym_backend
from torchrl.envs.utils import (
    _classproperty,
    _selective_unsqueeze,
    check_marl_grouping,
    MarlGroupMapType,
)

_has_vmas = importlib.util.find_spec("vmas") is not None


__all__ = ["VmasWrapper", "VmasEnv"]


def _get_envs():
    if not _has_vmas:
        raise ImportError("VMAS is not installed in your virtual environment.")
    import vmas

    all_scenarios = vmas.scenarios + vmas.mpe_scenarios + vmas.debug_scenarios

    return all_scenarios


@set_gym_backend("gym")
def _vmas_to_torchrl_spec_transform(
    spec,
    device,
    categorical_action_encoding,
) -> TensorSpec:
    gym_spaces = gym_backend("spaces")
    if isinstance(spec, gym_spaces.discrete.Discrete):
        action_space_cls = Categorical if categorical_action_encoding else OneHot
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )
        return action_space_cls(spec.n, device=device, dtype=dtype)
    elif isinstance(spec, gym_spaces.multi_discrete.MultiDiscrete):
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )
        return (
            MultiCategorical(spec.nvec, device=device, dtype=dtype)
            if categorical_action_encoding
            else MultiOneHot(spec.nvec, device=device, dtype=dtype)
        )
    elif isinstance(spec, gym_spaces.Box):
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        dtype = numpy_to_torch_dtype_dict[spec.dtype]
        low = torch.tensor(spec.low, device=device, dtype=dtype)
        high = torch.tensor(spec.high, device=device, dtype=dtype)
        is_unbounded = low.isinf().all() and high.isinf().all()
        return (
            Unbounded(shape, device=device, dtype=dtype)
            if is_unbounded
            else Bounded(
                low,
                high,
                shape,
                dtype=dtype,
                device=device,
            )
        )
    elif isinstance(spec, gym_spaces.Dict):
        spec_out = {}
        for key in spec.keys():
            spec_out[key] = _vmas_to_torchrl_spec_transform(
                spec[key],
                device=device,
                categorical_action_encoding=categorical_action_encoding,
            )
        # the batch-size must be set later
        return Composite(spec_out, device=device)
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for vmas"
        )


class VmasWrapper(_EnvWrapper):
    """Vmas environment wrapper.

    GitHub: https://github.com/proroklab/VectorizedMultiAgentSimulator

    Paper: https://arxiv.org/abs/2207.03530

    Args:
        env (``vmas.simulator.environment.environment.Environment``): the vmas environment to wrap.

    Keyword Args:
        num_envs (int): Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (torch.device, optional): Device for simulation. Defaults to the default device. All the tensors created by VMAS
            will be placed on this device.
        continuous_actions (bool, optional): Whether to use continuous actions. Defaults to ``True``. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario.
            See the VMAS repository for more info.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated (and the ``"terminated"`` flag is set) whenever this horizon is reached.
            Unlike gym's ``TimeLimit`` transform or torchrl's :class:`~torchrl.envs.transforms.StepCounter`,
            this argument will not set the ``"truncated"`` entry in the tensordict.
        categorical_actions (bool, optional): if the environment actions are discrete, whether to transform
            them to categorical or one-hot. Defaults to ``True``.
        group_map (MarlGroupMapType or Dict[str, List[str]], optional): how to group agents in tensordicts for
            input/output. By default, if the agent names follow the ``"<name>_<int>"``
            convention, they will be grouped by ``"<name>"``. If they do not follow this convention, they will be all put
            in one group named ``"agents"``.
            Otherwise, a group map can be specified or selected from some premade options.
            See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.

    Attributes:
        group_map (Dict[str, List[str]]): how to group agents in tensordicts for
            input/output. See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        agent_names (list of str): names of the agent in the environment
        agent_names_to_indices_map (Dict[str, int]): dictionary mapping agent names to their index in the environment
        unbatched_action_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_observation_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_reward_spec (TensorSpec): version of the spec without the vectorized dimension
        het_specs (bool): whether the environment has any lazy spec
        het_specs_map (Dict[str, bool]): dictionary mapping each group to a flag representing of the group has lazy specs
        available_envs (List[str]): the list of the scenarios available to build.

    .. warning::
        VMAS returns a single ``done`` flag which does not distinguish between
        when the env reached ``max_steps`` and termination.
        If you deem the ``truncation`` signal necessary, set ``max_steps`` to
        ``None`` and use a :class:`~torchrl.envs.transforms.StepCounter` transform.

    Examples:
        >>>  env = VmasWrapper(
        ...      vmas.make_env(
        ...          scenario="flocking",
        ...          num_envs=32,
        ...          continuous_actions=True,
        ...          max_steps=200,
        ...          device="cpu",
        ...          seed=None,
        ...          # Scenario kwargs
        ...          n_agents=5,
        ...      )
        ...  )
        >>>  print(env.rollout(10))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([32, 10, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                        info: TensorDict(
                            fields={
                                agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 10, 5]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                info: TensorDict(
                                    fields={
                                        agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                                    batch_size=torch.Size([32, 10, 5]),
                                    device=cpu,
                                    is_shared=False),
                                observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32, 10]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([32, 10]),
            device=cpu,
            is_shared=False)
    """

    git_url = "https://github.com/proroklab/VectorizedMultiAgentSimulator"
    libname = "vmas"

    @property
    def lib(self):
        import vmas

        return vmas

    @_classproperty
    def available_envs(cls):
        if not _has_vmas:
            return []
        return list(_get_envs())

    def __init__(
        self,
        env: vmas.simulator.environment.environment.Environment = None,  # noqa
        categorical_actions: bool = True,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
            if "device" in kwargs.keys() and kwargs["device"] != str(env.device):
                raise TypeError("Env device is different from vmas device")
            kwargs["device"] = str(env.device)
        self.group_map = group_map
        self.categorical_actions = categorical_actions
        super().__init__(**kwargs, allow_done_after_reset=True)

    def _build_env(
        self,
        env: vmas.simulator.environment.environment.Environment,  # noqa
        from_pixels: bool = False,
        pixels_only: bool = False,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        # TODO pixels
        if self.from_pixels:
            raise NotImplementedError("vmas rendering not yet implemented")

        # Adjust batch size
        if len(self.batch_size) == 0:
            # Batch size not set
            self.batch_size = torch.Size((env.num_envs,))
        elif len(self.batch_size) == 1:
            # Batch size is set
            if not self.batch_size[0] == env.num_envs:
                raise TypeError(
                    "Batch size used in constructor does not match vmas batch size."
                )
        else:
            raise TypeError(
                "Batch size used in constructor is not compatible with vmas."
            )

        return env

    def _get_default_group_map(self, agent_names: list[str]):
        # This function performs the default grouping in vmas.
        # Agents with names "<name>_<int>" will be grouped in group name "<name>".
        # If any of the agents does not follow the naming convention, we fall back
        # back on having all agents in one group named "agents".
        group_map = {}
        follows_convention = True
        for agent_name in agent_names:
            # See if the agent follows the convention "<name>_<int>"
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            follows_convention = follows_convention and agent_name_split[-1].isdigit()

            if not follows_convention:
                break

            # Group it with other agents that follow the same convention
            group_name = "_".join(agent_name_split[:-1])
            if group_name in group_map:
                group_map[group_name].append(agent_name)
            else:
                group_map[group_name] = [agent_name]

        if not follows_convention:
            group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)

        # For BC-compatibility rename the "agent" group to "agents"
        if "agent" in group_map and len(group_map) == 1:
            agent_group = group_map["agent"]
            group_map["agents"] = agent_group
            del group_map["agent"]
        return group_map

    def _make_specs(
        self, env: vmas.simulator.environment.environment.Environment  # noqa
    ) -> None:
        # Create and check group map
        self.agent_names = [agent.name for agent in self.agents]
        self.agent_names_to_indices_map = {
            agent.name: i for i, agent in enumerate(self.agents)
        }
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.agent_names)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

        full_action_spec_unbatched = Composite(device=self.device)
        full_observation_spec_unbatched = Composite(device=self.device)
        full_reward_spec_unbatched = Composite(device=self.device)

        self.het_specs = False
        self.het_specs_map = {}
        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_info_spec,
            ) = self._make_unbatched_group_specs(group)
            full_action_spec_unbatched[group] = group_action_spec
            full_observation_spec_unbatched[group] = group_observation_spec
            full_reward_spec_unbatched[group] = group_reward_spec
            if group_info_spec is not None:
                full_observation_spec_unbatched[(group, "info")] = group_info_spec
            group_het_specs = isinstance(
                group_observation_spec, StackedComposite
            ) or isinstance(group_action_spec, StackedComposite)
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs

        full_done_spec_unbatched = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )

        self.full_action_spec_unbatched = full_action_spec_unbatched
        self.full_observation_spec_unbatched = full_observation_spec_unbatched
        self.full_reward_spec_unbatched = full_reward_spec_unbatched
        self.full_done_spec_unbatched = full_done_spec_unbatched

    @property
    def unbatched_action_spec(self):
        warnings.warn(
            "unbatched_action_spec is deprecated and will be removed in v0.9. "
            "Please use full_action_spec_unbatched instead."
        )
        return self.full_action_spec_unbatched

    @property
    def unbatched_observation_spec(self):
        warnings.warn(
            "unbatched_observation_spec is deprecated and will be removed in v0.9. "
            "Please use full_observation_spec_unbatched instead."
        )
        return self.full_observation_spec_unbatched

    @property
    def unbatched_reward_spec(self):
        warnings.warn(
            "unbatched_reward_spec is deprecated and will be removed in v0.9. "
            "Please use full_reward_spec_unbatched instead."
        )
        return self.full_reward_spec_unbatched

    @property
    def unbatched_done_spec(self):
        warnings.warn(
            "unbatched_done_spec is deprecated and will be removed in v0.9. "
            "Please use full_done_spec_unbatched instead."
        )
        return self.full_done_spec_unbatched

    def _make_unbatched_group_specs(self, group: str):
        # Agent specs
        action_specs = []
        observation_specs = []
        reward_specs = []
        info_specs = []
        for agent_name in self.group_map[group]:
            agent_index = self.agent_names_to_indices_map[agent_name]
            agent = self.agents[agent_index]
            action_specs.append(
                Composite(
                    {
                        "action": _vmas_to_torchrl_spec_transform(
                            self.action_space[agent_index],
                            categorical_action_encoding=self.categorical_actions,
                            device=self.device,
                        )  # shape = (n_actions_per_agent,)
                    },
                )
            )
            observation_specs.append(
                Composite(
                    {
                        "observation": _vmas_to_torchrl_spec_transform(
                            self.observation_space[agent_index],
                            device=self.device,
                            categorical_action_encoding=self.categorical_actions,
                        )  # shape = (n_obs_per_agent,)
                    },
                )
            )
            reward_specs.append(
                Composite(
                    {
                        "reward": Unbounded(
                            shape=torch.Size((1,)),
                            device=self.device,
                        )  # shape = (1,)
                    }
                )
            )
            agent_info = self.scenario.info(agent)
            if len(agent_info):
                info_specs.append(
                    Composite(
                        {
                            key: Unbounded(
                                shape=_selective_unsqueeze(
                                    value, batch_size=self.batch_size
                                ).shape[1:],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            for key, value in agent_info.items()
                        },
                    ).to(self.device)
                )

        # Create multi-agent specs
        group_action_spec = torch.stack(
            action_specs, dim=0
        )  # shape = (n_agents, n_actions_per_agent)
        group_observation_spec = torch.stack(
            observation_specs, dim=0
        )  # shape = (n_agents, n_obs_per_agent)
        group_reward_spec = torch.stack(reward_specs, dim=0)  # shape = (n_agents, 1)
        group_info_spec = None
        if len(info_specs):
            group_info_spec = torch.stack(info_specs, dim=0)

        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_info_spec,
        )

    def _check_kwargs(self, kwargs: dict):
        vmas = self.lib

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, vmas.simulator.environment.Environment):
            raise TypeError(
                "env is not of type 'vmas.simulator.environment.Environment'."
            )

    def _init_env(self) -> int | None:
        pass

    def _set_seed(self, seed: int | None) -> None:
        self._env.seed(seed)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            envs_to_reset = _reset.squeeze(-1)
            if envs_to_reset.all():
                self._env.reset(return_observations=False)
            else:
                for env_index, to_reset in enumerate(envs_to_reset):
                    if to_reset:
                        self._env.reset_at(env_index, return_observations=False)
        else:
            self._env.reset(return_observations=False)

        obs, dones, infos = self._env.get_from_scenario(
            get_observations=True,
            get_infos=True,
            get_rewards=False,
            get_dones=True,
        )
        dones = self.read_done(dones)

        source = {"done": dones, "terminated": dones.clone()}
        for group, agent_names in self.group_map.items():
            agent_tds = []
            for agent_name in agent_names:
                i = self.agent_names_to_indices_map[agent_name]

                agent_obs = self.read_obs(obs[i])
                agent_info = self.read_info(infos[i])
                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                if agent_info is not None:
                    agent_td.set("info", agent_info)
                agent_tds.append(agent_td)

            agent_tds = LazyStackedTensorDict.maybe_dense_stack(agent_tds, dim=1)
            if not self.het_specs_map[group]:
                agent_tds = agent_tds.to_tensordict()
            source.update({group: agent_tds})

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        agent_indices = {}
        action_list = []
        n_agents = 0
        for group, agent_names in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_list = list(self.read_action(group_action, group=group))
            agent_indices.update(
                {
                    self.agent_names_to_indices_map[agent_name]: i + n_agents
                    for i, agent_name in enumerate(agent_names)
                }
            )
            n_agents += len(agent_names)
            action_list += group_action_list
        action = [action_list[agent_indices[i]] for i in range(self.n_agents)]

        obs, rews, dones, infos = self._env.step(action)

        dones = self.read_done(dones)

        source = {"done": dones, "terminated": dones.clone()}
        for group, agent_names in self.group_map.items():
            agent_tds = []
            for agent_name in agent_names:
                i = self.agent_names_to_indices_map[agent_name]

                agent_obs = self.read_obs(obs[i])
                agent_rew = self.read_reward(rews[i])
                agent_info = self.read_info(infos[i])

                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                        "reward": agent_rew,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                if agent_info is not None:
                    agent_td.set("info", agent_info)
                agent_tds.append(agent_td)

            agent_tds = LazyStackedTensorDict.maybe_dense_stack(agent_tds, dim=1)
            if not self.het_specs_map[group]:
                agent_tds = agent_tds.to_tensordict()
            source.update({group: agent_tds})

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def read_obs(self, observations: dict | torch.Tensor) -> dict | torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return _selective_unsqueeze(observations, batch_size=self.batch_size)
        return TensorDict(
            source={key: self.read_obs(value) for key, value in observations.items()},
            batch_size=self.batch_size,
        )

    def read_info(self, infos: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(infos) == 0:
            return None
        infos = TensorDict(
            source={
                key: _selective_unsqueeze(
                    value.to(torch.float32), batch_size=self.batch_size
                )
                for key, value in infos.items()
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return infos

    def read_done(self, done):
        done = _selective_unsqueeze(done, batch_size=self.batch_size)
        return done

    def read_reward(self, rewards):
        rewards = _selective_unsqueeze(rewards, batch_size=self.batch_size)
        return rewards

    def read_action(self, action, group: str = "agents"):
        if not self.continuous_actions and not self.categorical_actions:
            action = self.full_action_spec_unbatched[group, "action"].to_categorical(
                action
            )
        agent_actions = action.unbind(dim=1)
        return agent_actions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_envs={self.num_envs}, n_agents={self.n_agents},"
            f" batch_size={self.batch_size}, device={self.device})"
        )

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self._env.to(device)
        return super().to(device)


class VmasEnv(VmasWrapper):
    """Vmas environment wrapper.

    GitHub: https://github.com/proroklab/VectorizedMultiAgentSimulator

    Paper: https://arxiv.org/abs/2207.03530

    Args:
        scenario (str or vmas.simulator.scenario.BaseScenario): the vmas scenario to build.
            Must be one of :attr:`~.available_envs`. For a description and rendering of available scenarios see
            `the README <https://github.com/proroklab/VectorizedMultiAgentSimulator/tree/VMAS-1.3.3?tab=readme-ov-file#main-scenarios>`__.


    Keyword Args:
        num_envs (int): Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (torch.device, optional): Device for simulation. Defaults to the defaultt device. All the tensors created by VMAS
            will be placed on this device.
        continuous_actions (bool, optional): Whether to use continuous actions. Defaults to ``True``. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario.
            See the VMAS repository for more info.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated (and the ``"terminated"`` flag is set) whenever this horizon is reached.
            Unlike gym's ``TimeLimit`` transform or torchrl's :class:`~torchrl.envs.transforms.StepCounter`,
            this argument will not set the ``"truncated"`` entry in the tensordict.
        categorical_actions (bool, optional): if the environment actions are discrete, whether to transform
            them to categorical or one-hot. Defaults to ``True``.
        group_map (MarlGroupMapType or Dict[str, List[str]], optional): how to group agents in tensordicts for
            input/output. By default, if the agent names follow the ``"<name>_<int>"``
            convention, they will be grouped by ``"<name>"``. If they do not follow this convention, they will be all put
            in one group named ``"agents"``.
            Otherwise, a group map can be specified or selected from some premade options.
            See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        **kwargs (Dict, optional): These are additional arguments that can be passed to the VMAS scenario constructor.
            (e.g., number of agents, reward sparsity). The available arguments will vary based on the chosen scenario.
            To see the available arguments for a specific scenario, see the constructor in its file from
            `the scenario folder <https://github.com/proroklab/VectorizedMultiAgentSimulator/tree/VMAS-1.3.3/vmas/scenarios>`__.


    Attributes:
        group_map (Dict[str, List[str]]): how to group agents in tensordicts for
            input/output. See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        agent_names (list of str): names of the agent in the environment
        agent_names_to_indices_map (Dict[str, int]): dictionary mapping agent names to their index in the environment
        unbatched_action_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_observation_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_reward_spec (TensorSpec): version of the spec without the vectorized dimension
        het_specs (bool): whether the environment has any lazy spec
        het_specs_map (Dict[str, bool]): dictionary mapping each group to a flag representing of the group has lazy specs
        available_envs (List[str]): the list of the scenarios available to build.

    .. warning::
        VMAS returns a single ``done`` flag which does not distinguish between
        when the env reached ``max_steps`` and termination.
        If you deem the ``truncation`` signal necessary, set ``max_steps`` to
        ``None`` and use a :class:`~torchrl.envs.transforms.StepCounter` transform.

    Examples:
        >>>  env = VmasEnv(
        ...      scenario="flocking",
        ...      num_envs=32,
        ...      continuous_actions=True,
        ...      max_steps=200,
        ...      device="cpu",
        ...      seed=None,
        ...      # Scenario kwargs
        ...      n_agents=5,
        ...  )
        >>>  print(env.rollout(10))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([32, 10, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                        info: TensorDict(
                            fields={
                                agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 10, 5]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                info: TensorDict(
                                    fields={
                                        agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                                    batch_size=torch.Size([32, 10, 5]),
                                    device=cpu,
                                    is_shared=False),
                                observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32, 10]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([32, 10]),
            device=cpu,
            is_shared=False)
    """

    def __init__(
        self,
        scenario: str | vmas.simulator.scenario.BaseScenario,  # noqa
        *,
        num_envs: int,
        continuous_actions: bool = True,
        max_steps: int | None = None,
        categorical_actions: bool = True,
        seed: int | None = None,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        **kwargs,
    ):
        if not _has_vmas:
            raise ImportError(
                f"vmas python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
        super().__init__(
            scenario=scenario,
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            max_steps=max_steps,
            seed=seed,
            categorical_actions=categorical_actions,
            group_map=group_map,
            **kwargs,
        )

    def _check_kwargs(self, kwargs: dict):
        if "scenario" not in kwargs:
            raise TypeError("Could not find environment key 'scenario' in kwargs.")
        if "num_envs" not in kwargs:
            raise TypeError("Could not find environment key 'num_envs' in kwargs.")

    def _build_env(
        self,
        scenario: str | vmas.simulator.scenario.BaseScenario,  # noqa
        num_envs: int,
        continuous_actions: bool,
        max_steps: int | None,
        seed: int | None,
        **scenario_kwargs,
    ) -> vmas.simulator.environment.environment.Environment:  # noqa
        vmas = self.lib

        self.scenario_name = scenario
        from_pixels = scenario_kwargs.pop("from_pixels", False)
        pixels_only = scenario_kwargs.pop("pixels_only", False)

        return super()._build_env(
            env=vmas.make_env(
                scenario=scenario,
                num_envs=num_envs,
                device=self.device
                if self.device is not None
                else getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )(),
                continuous_actions=continuous_actions,
                max_steps=max_steps,
                seed=seed,
                wrapper=None,
                **scenario_kwargs,
            ),
            pixels_only=pixels_only,
            from_pixels=from_pixels,
        )

    def __repr__(self):
        return f"{super().__repr__()} (scenario={self.scenario_name})"
