# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    BoundedContinuous,
    Categorical,
    Composite,
    MultiCategorical,
    MultiOneHot,
    Unbounded,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

_has_unity_mlagents = importlib.util.find_spec("mlagents_envs") is not None


def _get_registered_envs():
    if not _has_unity_mlagents:
        raise ImportError(
            "mlagents_envs not found. Consider downloading and installing "
            f"mlagents from {UnityMLAgentsWrapper.git_url}."
        )

    from mlagents_envs.registry import default_registry

    return list(default_registry.keys())


class UnityMLAgentsWrapper(_EnvWrapper):
    """Unity ML-Agents environment wrapper.

    GitHub: https://github.com/Unity-Technologies/ml-agents

    Documentation: https://unity-technologies.github.io/ml-agents/Python-LLAPI/

    Args:
        env (mlagents_envs.environment.UnityEnvironment): the ML-Agents
            environment to wrap.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``None``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to
            group agents in tensordicts for input/output. See
            :class:`~torchrl.envs.utils.MarlGroupMapType` for more info. If not
            specified, agents are grouped according to the group ID given by the
            Unity environment. Defaults to ``None``.
        categorical_actions (bool, optional): if ``True``, categorical specs
            will be converted to the TorchRL equivalent
            (:class:`torchrl.data.Categorical`), otherwise a one-hot encoding
            will be used (:class:`torchrl.data.OneHot`).  Defaults to ``False``.

    Attributes:
        available_envs: list of registered environments available to build

    Examples:
        >>> from mlagents_envs.environment import UnityEnvironment
        >>> base_env = UnityEnvironment()
        >>> from torchrl.envs import UnityMLAgentsWrapper
        >>> env = UnityMLAgentsWrapper(base_env)
        >>> td = env.reset()
        >>> td = env.step(td.update(env.full_action_spec.rand()))
    """

    git_url = "https://github.com/Unity-Technologies/ml-agents"
    libname = "mlagents_envs"
    _lib = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib

        import mlagents_envs.environment

        cls._lib = mlagents_envs
        return mlagents_envs

    def __init__(
        self,
        env=None,
        *,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        categorical_actions: bool = False,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        self.group_map = group_map
        self.categorical_actions = categorical_actions
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        mlagents_envs = self.lib
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, mlagents_envs.environment.UnityEnvironment):
            raise TypeError(
                "env is not of type 'mlagents_envs.environment.UnityEnvironment'"
            )

    def _build_env(self, env, requires_grad: bool = False, **kwargs):
        self.requires_grad = requires_grad
        return env

    def _init_env(self):
        self._update_action_mask()

    # Creates a group map where agents are grouped by the group_id given by the
    # Unity environment.
    def _collect_agents(self, env):
        agent_name_to_behavior_map = {}
        agent_name_to_group_id_map = {}

        for steps_idx in [0, 1]:
            for behavior in env.behavior_specs.keys():
                steps = env.get_steps(behavior)[steps_idx]
                agent_ids = steps.agent_id
                group_ids = steps.group_id

                for agent_id, group_id in zip(agent_ids, group_ids):
                    agent_name = f"agent_{agent_id}"
                    if agent_name in agent_name_to_behavior_map:
                        # Sometimes in an MLAgents environment, an agent may
                        # show up in both the decision steps and the terminal
                        # steps. When that happens, just skip the duplicate.
                        continue
                    agent_name_to_behavior_map[agent_name] = behavior
                    agent_name_to_group_id_map[agent_name] = group_id

        return (
            agent_name_to_behavior_map,
            agent_name_to_group_id_map,
        )

    # Creates a group map where agents are grouped by their group_id.
    def _make_default_group_map(self, agent_name_to_group_id_map):
        group_map = {}
        for agent_name, group_id in agent_name_to_group_id_map.items():
            group_name = f"group_{group_id}"
            if group_name not in group_map:
                group_map[group_name] = []
            group_map[group_name].append(agent_name)
        return group_map

    def _make_group_map(self, group_map, agent_name_to_group_id_map):
        if group_map is None:
            group_map = self._make_default_group_map(agent_name_to_group_id_map)
        elif isinstance(group_map, MarlGroupMapType):
            group_map = group_map.get_group_map(agent_name_to_group_id_map.keys())
        check_marl_grouping(group_map, agent_name_to_group_id_map.keys())
        agent_name_to_group_name_map = {}
        for group_name, agents in group_map.items():
            for agent_name in agents:
                agent_name_to_group_name_map[agent_name] = group_name
        return group_map, agent_name_to_group_name_map

    def _make_specs(
        self, env: mlagents_envs.environment.UnityEnvironment  # noqa: F821
    ) -> None:
        # NOTE: We need to reset here because mlagents only initializes the
        # agents and behaviors after reset. In order to build specs, we make the
        # following assumptions about the mlagents environment:
        #   * all behaviors are defined on the first step
        #   * all agents request an action on the first step
        # However, mlagents allows you to break these assumptions, so we probably
        # will need to detect changes to the behaviors and agents on each step.
        env.reset()
        (
            self.agent_name_to_behavior_map,
            self.agent_name_to_group_id_map,
        ) = self._collect_agents(env)

        (self.group_map, self.agent_name_to_group_name_map) = self._make_group_map(
            self.group_map, self.agent_name_to_group_id_map
        )

        action_spec = {}
        observation_spec = {}
        reward_spec = {}
        done_spec = {}

        for group_name, agents in self.group_map.items():
            group_action_spec = {}
            group_observation_spec = {}
            group_reward_spec = {}
            group_done_spec = {}
            for agent_name in agents:
                behavior = self.agent_name_to_behavior_map[agent_name]
                behavior_spec = env.behavior_specs[behavior]

                # Create action spec
                agent_action_spec = Composite()
                env_action_spec = behavior_spec.action_spec
                discrete_branches = env_action_spec.discrete_branches
                continuous_size = env_action_spec.continuous_size
                if len(discrete_branches) > 0:
                    discrete_action_spec_cls = (
                        MultiCategorical if self.categorical_actions else MultiOneHot
                    )
                    agent_action_spec["discrete_action"] = discrete_action_spec_cls(
                        discrete_branches,
                        dtype=torch.int32,
                        device=self.device,
                    )
                if continuous_size > 0:
                    # In mlagents, continuous actions can take values between -1
                    # and 1 by default:
                    # https://github.com/Unity-Technologies/ml-agents/blob/22a59aad34ef46a5de05469735426feed758f8f5/ml-agents-envs/mlagents_envs/base_env.py#L395
                    agent_action_spec["continuous_action"] = BoundedContinuous(
                        -1, 1, (continuous_size,), self.device, torch.float32
                    )
                group_action_spec[agent_name] = agent_action_spec

                # Create observation spec
                agent_observation_spec = Composite()
                for obs_idx, env_observation_spec in enumerate(
                    behavior_spec.observation_specs
                ):
                    if len(env_observation_spec.name) == 0:
                        obs_name = f"observation_{obs_idx}"
                    else:
                        obs_name = env_observation_spec.name
                    agent_observation_spec[obs_name] = Unbounded(
                        env_observation_spec.shape,
                        dtype=torch.float32,
                        device=self.device,
                    )
                group_observation_spec[agent_name] = agent_observation_spec

                # Create reward spec
                agent_reward_spec = Composite()
                agent_reward_spec["reward"] = Unbounded(
                    (1,),
                    dtype=torch.float32,
                    device=self.device,
                )
                agent_reward_spec["group_reward"] = Unbounded(
                    (1,),
                    dtype=torch.float32,
                    device=self.device,
                )
                group_reward_spec[agent_name] = agent_reward_spec

                # Create done spec
                agent_done_spec = Composite()
                for done_key in ["done", "terminated", "truncated"]:
                    agent_done_spec[done_key] = Categorical(
                        2, (1,), dtype=torch.bool, device=self.device
                    )
                group_done_spec[agent_name] = agent_done_spec

            action_spec[group_name] = group_action_spec
            observation_spec[group_name] = group_observation_spec
            reward_spec[group_name] = group_reward_spec
            done_spec[group_name] = group_done_spec

        self.action_spec = Composite(action_spec)
        self.observation_spec = Composite(observation_spec)
        self.reward_spec = Composite(reward_spec)
        self.done_spec = Composite(done_spec)

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            raise NotImplementedError("This environment has no seed.")

    def _check_agent_exists(self, agent_name, group_id):
        if agent_name not in self.agent_name_to_group_id_map:
            raise RuntimeError(
                "Unity environment added a new agent. This is not yet "
                "supported in torchrl."
            )
        if self.agent_name_to_group_id_map[agent_name] != group_id:
            raise RuntimeError(
                "Unity environment changed the group of an agent. This "
                "is not yet supported in torchrl."
            )

    def _update_action_mask(self):
        for behavior, behavior_spec in self._env.behavior_specs.items():
            env_action_spec = behavior_spec.action_spec
            discrete_branches = env_action_spec.discrete_branches

            if len(discrete_branches) > 0:
                steps = self._env.get_steps(behavior)[0]
                env_action_mask = steps.action_mask
                if env_action_mask is not None:
                    combined_action_mask = torch.cat(
                        [
                            torch.tensor(m, device=self.device, dtype=torch.bool)
                            for m in env_action_mask
                        ],
                        dim=-1,
                    ).logical_not()

                    for agent_id, group_id, agent_action_mask in zip(
                        steps.agent_id, steps.group_id, combined_action_mask
                    ):
                        agent_name = f"agent_{agent_id}"
                        self._check_agent_exists(agent_name, group_id)
                        group_name = self.agent_name_to_group_name_map[agent_name]
                        self.full_action_spec[
                            group_name, agent_name, "discrete_action"
                        ].update_mask(agent_action_mask)

    def _make_td_out(self, tensordict_in, is_reset=False):
        source = {}
        for behavior, behavior_spec in self._env.behavior_specs.items():
            for idx, steps in enumerate(self._env.get_steps(behavior)):
                is_terminal = idx == 1
                for steps_idx, (agent_id, group_id) in enumerate(
                    zip(steps.agent_id, steps.group_id)
                ):
                    agent_name = f"agent_{agent_id}"
                    self._check_agent_exists(agent_name, group_id)
                    group_name = self.agent_name_to_group_name_map[agent_name]
                    if group_name not in source:
                        source[group_name] = {}
                    if agent_name not in source[group_name]:
                        source[group_name][agent_name] = {}

                    # Add observations
                    for obs_idx, (
                        behavior_observation,
                        env_observation_spec,
                    ) in enumerate(zip(steps.obs, behavior_spec.observation_specs)):
                        observation = torch.tensor(
                            behavior_observation[steps_idx],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        if len(env_observation_spec.name) == 0:
                            obs_name = f"observation_{obs_idx}"
                        else:
                            obs_name = env_observation_spec.name
                        source[group_name][agent_name][obs_name] = observation

                    # Add rewards
                    if not is_reset:
                        source[group_name][agent_name]["reward"] = torch.tensor(
                            [steps.reward[steps_idx]],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        source[group_name][agent_name]["group_reward"] = torch.tensor(
                            [steps.group_reward[steps_idx]],
                            device=self.device,
                            dtype=torch.float32,
                        )

                    # Add done
                    done = is_terminal and not is_reset
                    source[group_name][agent_name]["done"] = torch.tensor(
                        done, device=self.device, dtype=torch.bool
                    )
                    source[group_name][agent_name]["truncated"] = torch.tensor(
                        done and steps.interrupted[steps_idx],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    source[group_name][agent_name]["terminated"] = torch.tensor(
                        done and not steps.interrupted[steps_idx],
                        device=self.device,
                        dtype=torch.bool,
                    )

        if tensordict_in is not None:
            # In MLAgents, a given step will only contain information for agents
            # which either terminated or requested a decision during the step.
            # Some agents may have neither terminated nor requested a decision,
            # so we need to fill in their information from the previous step.
            for group_name, agents in self.group_map.items():
                for agent_name in agents:
                    if group_name not in source.keys():
                        source[group_name] = {}
                    if agent_name not in source[group_name].keys():
                        agent_dict = {}
                        agent_behavior = self.agent_name_to_behavior_map[agent_name]
                        behavior_spec = self._env.behavior_specs[agent_behavior]
                        td_agent_in = tensordict_in[group_name, agent_name]

                        # Add observations
                        for env_observation_spec in behavior_spec.observation_specs:
                            if len(env_observation_spec.name) == 0:
                                obs_name = f"observation_{obs_idx}"
                            else:
                                obs_name = env_observation_spec.name
                            agent_dict[obs_name] = td_agent_in[obs_name]

                        # Add rewards
                        if not is_reset:
                            # Since the agent didn't request an decision, the
                            # reward is 0
                            agent_dict["reward"] = torch.zeros(
                                (1,), device=self.device, dtype=torch.float32
                            )
                            agent_dict["group_reward"] = torch.zeros(
                                (1,), device=self.device, dtype=torch.float32
                            )

                        # Add done
                        agent_dict["done"] = torch.tensor(
                            False, device=self.device, dtype=torch.bool
                        )
                        agent_dict["terminated"] = torch.tensor(
                            False, device=self.device, dtype=torch.bool
                        )
                        agent_dict["truncated"] = torch.tensor(
                            False, device=self.device, dtype=torch.bool
                        )

                        source[group_name][agent_name] = agent_dict

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

        return tensordict_out

    def _get_action_from_tensor(self, tensor):
        if not self.categorical_actions:
            action = torch.argmax(tensor, dim=-1)
        else:
            action = tensor
        return action

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Apply actions
        for behavior, behavior_spec in self._env.behavior_specs.items():
            env_action_spec = behavior_spec.action_spec
            steps = self._env.get_steps(behavior)[0]

            for agent_id, group_id in zip(steps.agent_id, steps.group_id):
                agent_name = f"agent_{agent_id}"
                self._check_agent_exists(agent_name, group_id)
                group_name = self.agent_name_to_group_name_map[agent_name]

                agent_action_spec = self.full_action_spec[group_name, agent_name]
                action_tuple = self.lib.base_env.ActionTuple()
                discrete_branches = env_action_spec.discrete_branches
                continuous_size = env_action_spec.continuous_size

                if len(discrete_branches) > 0:
                    discrete_spec = agent_action_spec["discrete_action"]
                    discrete_action = tensordict[
                        group_name, agent_name, "discrete_action"
                    ]
                    if not self.categorical_actions:
                        discrete_action = discrete_spec.to_categorical(discrete_action)
                    action_tuple.add_discrete(discrete_action[None, ...].numpy())

                if continuous_size > 0:
                    continuous_action = tensordict[
                        group_name, agent_name, "continuous_action"
                    ]
                    action_tuple.add_continuous(continuous_action[None, ...].numpy())

                self._env.set_action_for_agent(behavior, agent_id, action_tuple)

        self._env.step()
        self._update_action_mask()
        return self._make_td_out(tensordict)

    def _to_tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        self._env.reset()
        return self._make_td_out(tensordict, is_reset=True)

    def close(self, *, raise_if_closed: bool = True):
        self._env.close()

    @_classproperty
    def available_envs(cls):
        if not _has_unity_mlagents:
            return []
        return _get_registered_envs()


class UnityMLAgentsEnv(UnityMLAgentsWrapper):
    """Unity ML-Agents environment wrapper.

    GitHub: https://github.com/Unity-Technologies/ml-agents

    Documentation: https://unity-technologies.github.io/ml-agents/Python-LLAPI/

    This class can be provided any of the optional initialization arguments that
    :class:`mlagents_envs.environment.UnityEnvironment` class provides. For a
    list of these arguments, see:
    https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#__init__

    If both ``file_name`` and ``registered_name`` are given, an error is raised.

    If neither ``file_name`` nor``registered_name`` are given, the environment
    setup waits on a localhost port, and the user must execute a Unity ML-Agents
    environment binary for to connect to it.

    Args:
        file_name (str, optional): if provided, the path to the Unity
            environment binary. Defaults to ``None``.
        registered_name (str, optional): if provided, the Unity environment
            binary is loaded from the default ML-Agents registry. The list of
            registered environments is in :attr:`~.available_envs`. Defaults to
            ``None``.

    Keyword Args:
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``None``.
        batch_size (torch.Size, optional): the batch size of the environment.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to
            group agents in tensordicts for input/output. See
            :class:`~torchrl.envs.utils.MarlGroupMapType` for more info. If not
            specified, agents are grouped according to the group ID given by the
            Unity environment. Defaults to ``None``.
        categorical_actions (bool, optional): if ``True``, categorical specs
            will be converted to the TorchRL equivalent
            (:class:`torchrl.data.Categorical`), otherwise a one-hot encoding
            will be used (:class:`torchrl.data.OneHot`).  Defaults to ``False``.

    Attributes:
        available_envs: list of registered environments available to build

    Examples:
        >>> from torchrl.envs import UnityMLAgentsEnv
        >>> env = UnityMLAgentsEnv(registered_name='3DBall')
        >>> td = env.reset()
        >>> td = env.step(td.update(env.full_action_spec.rand()))
        >>> td
        TensorDict(
            fields={
                group_0: TensorDict(
                    fields={
                        agent_0: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_10: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_11: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_1: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_2: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_3: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_4: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_5: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_6: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_7: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_8: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        agent_9: TensorDict(
                            fields={
                                VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
                                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        group_0: TensorDict(
                            fields={
                                agent_0: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_10: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_11: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_1: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_2: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_3: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_4: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_5: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_6: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_7: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_8: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False),
                                agent_9: TensorDict(
                                    fields={
                                        VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
                                        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                                        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                                    batch_size=torch.Size([]),
                                    device=None,
                                    is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        file_name: str | None = None,
        registered_name: str | None = None,
        *,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        categorical_actions=False,
        **kwargs,
    ):
        kwargs["file_name"] = file_name
        kwargs["registered_name"] = registered_name
        super().__init__(
            group_map=group_map,
            categorical_actions=categorical_actions,
            **kwargs,
        )

    def _build_env(
        self,
        file_name: str | None,
        registered_name: str | None,
        **kwargs,
    ) -> mlagents_envs.environment.UnityEnvironment:  # noqa: F821
        if not _has_unity_mlagents:
            raise ImportError(
                "mlagents_envs not found, unable to create environment. "
                "Consider downloading and installing mlagents from "
                f"{self.git_url}"
            )
        if file_name is not None and registered_name is not None:
            raise ValueError(
                "Both `file_name` and `registered_name` were specified, which "
                "is not allowed. Specify one of them or neither."
            )
        elif registered_name is not None:
            from mlagents_envs.registry import default_registry

            env = default_registry[registered_name].make(**kwargs)
        else:
            env = self.lib.environment.UnityEnvironment(file_name, **kwargs)
        requires_grad = kwargs.pop("requires_grad", False)
        return super()._build_env(
            env,
            requires_grad=requires_grad,
        )

    @property
    def file_name(self):
        return self._constructor_kwargs["file_name"]

    @property
    def registered_name(self):
        return self._constructor_kwargs["registered_name"]

    def _check_kwargs(self, kwargs: dict):
        pass

    def __repr__(self) -> str:
        if self.registered_name is not None:
            env_name = self.registered_name
        else:
            env_name = self.file_name
        return f"{self.__class__.__name__}(env={env_name}, batch_size={self.batch_size}, device={self.device})"
