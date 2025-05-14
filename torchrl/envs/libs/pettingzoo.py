# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import importlib
import warnings
from typing import Dict

import numpy as np
import packaging
import torch
from tensordict import TensorDictBase

from torchrl.data.tensor_specs import Categorical, Composite, OneHot, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None


def _get_envs():
    if not _has_pettingzoo:
        raise ImportError("PettingZoo is not installed in your virtual environment.")
    try:
        from pettingzoo.utils.all_modules import all_environments
    except ModuleNotFoundError as err:
        warnings.warn(
            f"PettingZoo failed to load all modules with error message {err}, trying to load individual modules."
        )
        all_environments = _load_available_envs()

    return list(all_environments.keys())


def _load_available_envs() -> dict:
    all_environments = {}
    try:
        from pettingzoo.mpe.all_modules import mpe_environments

        all_environments.update(mpe_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"MPE environments failed to load with error message {err}.")
    try:
        from pettingzoo.sisl.all_modules import sisl_environments

        all_environments.update(sisl_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"SISL environments failed to load with error message {err}.")
    try:
        from pettingzoo.classic.all_modules import classic_environments

        all_environments.update(classic_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"Classic environments failed to load with error message {err}.")
    try:
        from pettingzoo.atari.all_modules import atari_environments

        all_environments.update(atari_environments)
    except ModuleNotFoundError as err:
        warnings.warn(f"Atari environments failed to load with error message {err}.")
    try:
        from pettingzoo.butterfly.all_modules import butterfly_environments

        all_environments.update(butterfly_environments)
    except ModuleNotFoundError as err:
        warnings.warn(
            f"Butterfly environments failed to load with error message {err}."
        )
    return all_environments


def _extract_nested_with_index(data: np.ndarray | dict[str, np.ndarray], index: int):
    if isinstance(data, np.ndarray):
        return data[index]
    elif isinstance(data, dict):
        return {
            key: _extract_nested_with_index(value, index) for key, value in data.items()
        }
    else:
        raise NotImplementedError(f"Invalid type of data {data}")


class PettingZooWrapper(_EnvWrapper):
    """PettingZoo environment wrapper.

    To install petting zoo follow the guide `here <https://github.com/Farama-Foundation/PettingZoo#installation>__`.

    This class is a general torchrl wrapper for all PettingZoo environments.
    It is able to wrap both ``pettingzoo.AECEnv`` and ``pettingzoo.ParallelEnv``.

    Let's see how more in details:

    In wrapped ``pettingzoo.ParallelEnv`` all agents will step at each environment step.
    If the number of agents during the task varies, please set ``use_mask=True``.
    ``"mask"`` will be provided
    as an output in each group and should be used to mask out dead agents.
    The environment will be reset as soon as one agent is done (unless ``done_on_any`` is ``False``).

    In wrapped ``pettingzoo.AECEnv``, at each step only one agent will act.
    For this reason, it is compulsory to set ``use_mask=True`` for this type of environment.
    ``"mask"`` will be provided as an output for each group and can be used to mask out non-acting agents.
    The environment will be reset only when all agents are done (unless ``done_on_any`` is ``True``).

    If there are any unavailable actions for an agent,
    the environment will also automatically update the mask of its ``action_spec`` and output an ``"action_mask"``
    for each group to reflect the latest available actions. This should be passed to a masked distribution during
    training.

    As a feature of torchrl multiagent, you are able to control the grouping of agents in your environment.
    You can group agents together (stacking their tensors) to leverage vectorization when passing them through the same
    neural network. You can split agents in different groups where they are heterogenous or should be processed by
    different neural networks. To group, you just need to pass a ``group_map`` at env constructiuon time.

    By default, agents in pettingzoo will be grouped by name.
    For example, with agents ``["agent_0","agent_1","agent_2","adversary_0"]``, the tensordicts will look like:

        >>> print(env.rand_action(env.reset()))
        TensorDict(
            fields={
                agent: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]))},
                adversary: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([1, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([1]))},
            batch_size=torch.Size([]))
        >>> print(env.group_map)
        {"agent": ["agent_0", "agent_1", "agent_2"], "adversary": ["adversary_0"]}

    Otherwise, a group map can be specified or selected from some premade options.
    See :class:`torchrl.envs.utils.MarlGroupMapType` for more info.
    For example, you can provide ``MarlGroupMapType.ONE_GROUP_PER_AGENT``, telling that each agent should
    have its own tensordict (similar to the pettingzoo parallel API).

    Grouping is useful for leveraging vectorization among agents whose data goes through the same
    neural network.

    Args:
        env (``pettingzoo.utils.env.ParallelEnv`` or ``pettingzoo.utils.env.AECEnv``): the pettingzoo environment to wrap.
        return_state (bool, optional): whether to return the global state from pettingzoo
            (not available in all environments). Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to group agents in tensordicts for
            input/output. By default, agents will be grouped by their name. Otherwise, a group map can be specified
            or selected from some premade options. See :class:`torchrl.envs.utils.MarlGroupMapType` for more info.
        use_mask (bool, optional): whether the environment should output a ``"mask"``. This is compulsory in
            wrapped ``pettingzoo.AECEnv`` to mask out non-acting agents and should be also used
            for ``pettingzoo.ParallelEnv`` when the number of agents can vary. Defaults to ``False``.
        categorical_actions (bool, optional): if the environments actions are discrete, whether to transform
            them to categorical or one-hot.
        seed (int, optional): the seed. Defaults to ``None``.
        done_on_any (bool, optional): whether the environment's done keys are set by aggregating the agent keys
            using ``any()`` (when ``True``) or ``all()`` (when ``False``). Default (``None``) is to use ``any()`` for
            parallel environments and ``all()`` for AEC ones.

    Examples:
        >>> # Parallel env
        >>> from torchrl.envs.libs.pettingzoo import PettingZooWrapper
        >>> from pettingzoo.butterfly import pistonball_v6
        >>> kwargs = {"n_pistons": 21, "continuous": True}
        >>> env = PettingZooWrapper(
        ...     env=pistonball_v6.parallel_env(**kwargs),
        ...     return_state=True,
        ...     group_map=None, # Use default for parallel (all pistons grouped together)
        ... )
        >>> print(env.group_map)
        ... {'piston': ['piston_0', 'piston_1', ..., 'piston_20']}
        >>> env.rollout(10)
        >>> # AEC env
        >>> from pettingzoo.classic import tictactoe_v3
        >>> from torchrl.envs.libs.pettingzoo import PettingZooWrapper
        >>> from torchrl.envs.utils import MarlGroupMapType
        >>> env = PettingZooWrapper(
        ...     env=tictactoe_v3.env(),
        ...     use_mask=True, # Must use it since one player plays at a time
        ...     group_map=None # # Use default for AEC (one group per player)
        ... )
        >>> print(env.group_map)
        ... {'player_1': ['player_1'], 'player_2': ['player_2']}
        >>> env.rollout(10)
    """

    git_url = "https://github.com/Farama-Foundation/PettingZoo"
    libname = "pettingzoo"

    @_classproperty
    def available_envs(cls):
        if not _has_pettingzoo:
            return []
        return list(_get_envs())

    def __init__(
        self,
        env: (
            pettingzoo.utils.env.ParallelEnv  # noqa: F821
            | pettingzoo.utils.env.AECEnv  # noqa: F821
        ) = None,
        return_state: bool = False,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        use_mask: bool = False,
        categorical_actions: bool = True,
        seed: int | None = None,
        done_on_any: bool | None = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        self.group_map = group_map
        self.return_state = return_state
        self.seed = seed
        self.use_mask = use_mask
        self.categorical_actions = categorical_actions
        self.done_on_any = done_on_any

        super().__init__(**kwargs, allow_done_after_reset=True)

    def _get_default_group_map(self, agent_names: list[str]):
        # This function performs the default grouping in pettingzoo
        if not self.parallel:
            # In AEC envs we will have one group per agent by default
            group_map = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(agent_names)
        else:
            # In parallel envs, by default
            # Agents with names "str_int" will be grouped in group name "str"
            group_map = {}
            for agent_name in agent_names:
                # See if the agent follows the convention "name_int"
                follows_convention = True
                agent_name_split = agent_name.split("_")
                if len(agent_name_split) == 1:
                    follows_convention = False
                try:
                    int(agent_name_split[-1])
                except ValueError:
                    follows_convention = False

                # If not, just put it in a single group
                if not follows_convention:
                    group_map[agent_name] = [agent_name]
                # Otherwise, group it with other agents that follow the same convention
                else:
                    group_name = "_".join(agent_name_split[:-1])
                    if group_name in group_map:
                        group_map[group_name].append(agent_name)
                    else:
                        group_map[group_name] = [agent_name]

        return group_map

    @property
    def lib(self):
        import pettingzoo

        return pettingzoo

    def _build_env(
        self,
        env: (
            pettingzoo.utils.env.ParallelEnv  # noqa: F821
            | pettingzoo.utils.env.AECEnv  # noqa: F821
        ),
    ):
        import pettingzoo

        if packaging.version.parse(pettingzoo.__version__).base_version != "1.24.3":
            warnings.warn(
                "PettingZoo in TorchRL is tested using version == 1.24.3 , "
                "If you are using a different version and are experiencing compatibility issues,"
                "please raise an issue in the TorchRL github."
            )

        self.parallel = isinstance(env, pettingzoo.utils.env.ParallelEnv)
        if not self.parallel and not self.use_mask:
            raise ValueError("For AEC environments you need to set use_mask=True")
        if len(self.batch_size):
            raise RuntimeError(
                f"PettingZoo does not support custom batch_size {self.batch_size}."
            )

        return env

    @set_gym_backend("gymnasium")
    def _make_specs(
        self,
        env: (
            pettingzoo.utils.env.ParallelEnv  # noqa: F821
            | pettingzoo.utils.env.AECEnv  # noqa: F821
        ),
    ) -> None:
        # Set default for done on any or all
        if self.done_on_any is None:
            self.done_on_any = self.parallel

        # Create and check group map
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.possible_agents)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.possible_agents)
        check_marl_grouping(self.group_map, self.possible_agents)
        self.has_action_mask = {group: False for group in self.group_map.keys()}

        action_spec = Composite()
        observation_spec = Composite()
        reward_spec = Composite()
        done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )
        for group, agents in self.group_map.items():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_done_spec,
            ) = self._make_group_specs(group_name=group, agent_names=agents)
            action_spec[group] = group_action_spec
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec
            done_spec[group] = group_done_spec

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

    def _make_group_specs(self, group_name: str, agent_names: list[str]):
        n_agents = len(agent_names)
        action_specs = []
        observation_specs = []
        for agent in agent_names:
            action_specs.append(
                Composite(
                    {
                        "action": _gym_to_torchrl_spec_transform(
                            self.action_space(agent),
                            remap_state_to_observation=False,
                            categorical_action_encoding=self.categorical_actions,
                            device=self.device,
                        )
                    },
                )
            )
            observation_specs.append(
                Composite(
                    {
                        "observation": _gym_to_torchrl_spec_transform(
                            self.observation_space(agent),
                            remap_state_to_observation=False,
                            device=self.device,
                        )
                    }
                )
            )
        group_action_spec = torch.stack(action_specs, dim=0)
        group_observation_spec = torch.stack(observation_specs, dim=0)

        # Sometimes the observation spec contains an action mask.
        # Or sometimes the info spec contains an action mask.
        # We uniform this by removing it from both places and optionally set it in a standard location.
        group_observation_inner_spec = group_observation_spec["observation"]
        if (
            isinstance(group_observation_inner_spec, Composite)
            and "action_mask" in group_observation_inner_spec.keys()
        ):
            self.has_action_mask[group_name] = True
            del group_observation_inner_spec["action_mask"]
            group_observation_spec["action_mask"] = Categorical(
                n=2,
                shape=group_action_spec["action"].shape
                if not self.categorical_actions
                else group_action_spec["action"].to_one_hot_spec().shape,
                dtype=torch.bool,
                device=self.device,
            )

        if self.use_mask:
            group_observation_spec["mask"] = Categorical(
                n=2,
                shape=torch.Size((n_agents,)),
                dtype=torch.bool,
                device=self.device,
            )

        group_reward_spec = Composite(
            {
                "reward": Unbounded(
                    shape=torch.Size((n_agents, 1)),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            shape=torch.Size((n_agents,)),
        )
        group_done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Categorical(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=torch.Size((n_agents,)),
        )
        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_done_spec,
        )

    def _check_kwargs(self, kwargs: dict):
        import pettingzoo

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(
            env, (pettingzoo.utils.env.ParallelEnv, pettingzoo.utils.env.AECEnv)
        ):
            raise TypeError("env is not of type expected.")

    def _init_env(self):
        # Add info
        if self.parallel:
            _, info_dict = self._reset_parallel(seed=self.seed)
        else:
            _, info_dict = self._reset_aec(seed=self.seed)

        for group, agents in self.group_map.items():
            info_specs = []
            for agent in agents:
                info_specs.append(
                    Composite(
                        {
                            "info": Composite(
                                {
                                    key: Unbounded(
                                        shape=torch.as_tensor(value).shape,
                                        device=self.device,
                                    )
                                    for key, value in info_dict[agent].items()
                                }
                            )
                        },
                        device=self.device,
                    )
                )
            info_specs = torch.stack(info_specs, dim=0)
            if ("info", "action_mask") in info_specs.keys(True, True):
                if not self.has_action_mask[group]:
                    self.has_action_mask[group] = True
                    group_action_spec = self.input_spec[
                        "full_action_spec", group, "action"
                    ]
                    self.observation_spec[group]["action_mask"] = Categorical(
                        n=2,
                        shape=group_action_spec.shape
                        if not self.categorical_actions
                        else group_action_spec.to_one_hot_spec().shape,
                        dtype=torch.bool,
                        device=self.device,
                    )
                group_inner_info_spec = info_specs["info"]
                del group_inner_info_spec["action_mask"]

            if len(info_specs["info"].keys()):
                self.observation_spec[group].update(info_specs)

        if self.return_state:
            try:
                state_spec = _gym_to_torchrl_spec_transform(
                    self.state_space,
                    remap_state_to_observation=False,
                    device=self.device,
                )
            except AttributeError:
                state_example = torch.as_tensor(self.state(), device=self.device)
                state_spec = Unbounded(
                    shape=state_example.shape,
                    dtype=state_example.dtype,
                    device=self.device,
                )
            self.observation_spec["state"] = state_spec

        # Caching
        self.cached_reset_output_zero = self.observation_spec.zero()
        self.cached_reset_output_zero.update(self.output_spec["full_done_spec"].zero())

        self.cached_step_output_zero = self.observation_spec.zero()
        self.cached_step_output_zero.update(self.output_spec["full_reward_spec"].zero())
        self.cached_step_output_zero.update(self.output_spec["full_done_spec"].zero())

    def _set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.reset(seed=self.seed)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None:
            _reset = tensordict.get("_reset", None)
            if _reset is not None and not _reset.all():
                raise RuntimeError(
                    f"An attempt to call {type(self)}._reset was made when no "
                    f"reset signal could be found. Expected '_reset' entry to "
                    f"be `tensor(True)` or `None` but got `{_reset}`."
                )
        if self.parallel:
            # This resets when any is done
            observation_dict, info_dict = self._reset_parallel(**kwargs)
        else:
            # This resets when all are done
            observation_dict, info_dict = self._reset_aec(**kwargs)

        # We start with zeroed data and fill in the data for alive agents
        tensordict_out = self.cached_reset_output_zero.clone()
        # Update the "mask" for non-acting agents
        self._update_agent_mask(tensordict_out)
        # Update the "action_mask" for non-available actions
        observation_dict, info_dict = self._update_action_mask(
            tensordict_out, observation_dict, info_dict
        )

        # Now we get the data (obs and info)
        for group, agent_names in self.group_map.items():
            group_observation = tensordict_out.get((group, "observation"))
            group_info = tensordict_out.get((group, "info"), None)

            for index, agent in enumerate(agent_names):
                group_observation[index] = self.observation_spec[group, "observation"][
                    index
                ].encode(observation_dict[agent])
                if group_info is not None:
                    agent_info_dict = info_dict[agent]
                    for agent_info, value in agent_info_dict.items():
                        group_info.get(agent_info)[index] = torch.as_tensor(
                            value, device=self.device
                        )

        return tensordict_out

    def _reset_aec(self, **kwargs) -> tuple[dict, dict]:
        self._env.reset(**kwargs)

        observation_dict = {
            agent: self._env.observe(agent) for agent in self.possible_agents
        }
        info_dict = self._env.infos
        return observation_dict, info_dict

    def _reset_parallel(self, **kwargs) -> tuple[dict, dict]:
        return self._env.reset(**kwargs)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        if self.parallel:
            (
                observation_dict,
                rewards_dict,
                terminations_dict,
                truncations_dict,
                info_dict,
            ) = self._step_parallel(tensordict)
        else:
            (
                observation_dict,
                rewards_dict,
                terminations_dict,
                truncations_dict,
                info_dict,
            ) = self._step_aec(tensordict)

        # We start with zeroed data and fill in the data for alive agents
        tensordict_out = self.cached_step_output_zero.clone()
        # Update the "mask" for non-acting agents
        self._update_agent_mask(tensordict_out)
        # Update the "action_mask" for non-available actions
        observation_dict, info_dict = self._update_action_mask(
            tensordict_out, observation_dict, info_dict
        )

        # Now we get the data
        for group, agent_names in self.group_map.items():
            group_observation = tensordict_out.get((group, "observation"))
            group_reward = tensordict_out.get((group, "reward"))
            group_done = tensordict_out.get((group, "done"))
            group_terminated = tensordict_out.get((group, "terminated"))
            group_truncated = tensordict_out.get((group, "truncated"))
            group_info = tensordict_out.get((group, "info"), None)

            for index, agent in enumerate(agent_names):
                if agent in observation_dict:  # Live agents
                    group_observation[index] = self.observation_spec[
                        group, "observation"
                    ][index].encode(observation_dict[agent])
                    group_reward[index] = torch.tensor(
                        rewards_dict[agent],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    group_done[index] = torch.tensor(
                        terminations_dict[agent] or truncations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    group_truncated[index] = torch.tensor(
                        truncations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    group_terminated[index] = torch.tensor(
                        terminations_dict[agent],
                        device=self.device,
                        dtype=torch.bool,
                    )

                    if group_info is not None:
                        agent_info_dict = info_dict[agent]
                        for agent_info, value in agent_info_dict.items():
                            group_info.get(agent_info)[index] = torch.tensor(
                                value, device=self.device
                            )

                elif self.use_mask:
                    if agent in self.agents:
                        raise ValueError(
                            f"Dead agent {agent} not found in step observation but still available in {self.agents}"
                        )
                    # Dead agent
                    terminated = (
                        terminations_dict[agent] if agent in terminations_dict else True
                    )
                    truncated = (
                        truncations_dict[agent] if agent in truncations_dict else True
                    )
                    done = terminated or truncated
                    group_done[index] = done
                    group_terminated[index] = terminated
                    group_truncated[index] = truncated

                else:
                    # Dead agent, if we are not masking it out, this is not allowed
                    raise ValueError(
                        "Dead agents found in the environment,"
                        " you need to set use_mask=True to allow this."
                    )

        # set done values
        done, terminated, truncated = self._aggregate_done(
            tensordict_out, use_any=self.done_on_any
        )

        tensordict_out.set("done", done)
        tensordict_out.set("terminated", terminated)
        tensordict_out.set("truncated", truncated)
        return tensordict_out

    def _aggregate_done(self, tensordict_out, use_any):
        done = False if use_any else True
        truncated = False if use_any else True
        terminated = False if use_any else True
        for key in self.done_keys:
            if isinstance(key, tuple):  # Only look at group keys
                if use_any:
                    if key[-1] == "done":
                        done = done | tensordict_out.get(key).any()
                    if key[-1] == "terminated":
                        terminated = terminated | tensordict_out.get(key).any()
                    if key[-1] == "truncated":
                        truncated = truncated | tensordict_out.get(key).any()
                    if done and terminated and truncated:
                        # no need to proceed further, all values are flipped
                        break
                else:
                    if key[-1] == "done":
                        done = done & tensordict_out.get(key).all()
                    if key[-1] == "terminated":
                        terminated = terminated & tensordict_out.get(key).all()
                    if key[-1] == "truncated":
                        truncated = truncated & tensordict_out.get(key).all()
                    if not done and not terminated and not truncated:
                        # no need to proceed further, all values are flipped
                        break
        return (
            torch.tensor([done], device=self.device),
            torch.tensor([terminated], device=self.device),
            torch.tensor([truncated], device=self.device),
        )

    def _step_parallel(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[dict, dict, dict, dict, dict]:
        action_dict = {}
        for group, agents in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_np = self.input_spec[
                "full_action_spec", group, "action"
            ].to_numpy(group_action)
            for index, agent in enumerate(agents):
                # group_action_np can be a dict or an array. We need to recursively index it
                action = _extract_nested_with_index(group_action_np, index)
                action_dict[agent] = action

        return self._env.step(action_dict)

    def _step_aec(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[dict, dict, dict, dict, dict]:
        for group, agents in self.group_map.items():
            if self.agent_selection in agents:
                agent_index = agents.index(self._env.agent_selection)
                group_action = tensordict.get((group, "action"))
                group_action_np = self.input_spec[
                    "full_action_spec", group, "action"
                ].to_numpy(group_action)
                # group_action_np can be a dict or an array. We need to recursively index it
                action = _extract_nested_with_index(group_action_np, agent_index)
                break

        self._env.step(action)
        terminations_dict = self._env.terminations
        truncations_dict = self._env.truncations
        info_dict = self._env.infos
        rewards_dict = self._env.rewards
        observation_dict = {
            agent: self._env.observe(agent) for agent in self.possible_agents
        }
        return (
            observation_dict,
            rewards_dict,
            terminations_dict,
            truncations_dict,
            info_dict,
        )

    def _update_action_mask(self, td, observation_dict, info_dict):
        # Since we remove the action_mask keys we need to copy the data
        observation_dict = copy.deepcopy(observation_dict)
        info_dict = copy.deepcopy(info_dict)
        # In AEC only one agent acts, in parallel env self.agents contains the agents alive
        agents_acting = self.agents if self.parallel else [self.agent_selection]

        for group, agents in self.group_map.items():
            if self.has_action_mask[group]:
                group_mask = td.get((group, "action_mask"))
                group_mask += True
                for index, agent in enumerate(agents):
                    agent_obs = observation_dict[agent]
                    agent_info = info_dict[agent]
                    if isinstance(agent_obs, Dict) and "action_mask" in agent_obs:
                        if agent in agents_acting:
                            group_mask[index] = torch.tensor(
                                agent_obs["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                        del agent_obs["action_mask"]
                    elif isinstance(agent_info, Dict) and "action_mask" in agent_info:
                        if agent in agents_acting:
                            group_mask[index] = torch.tensor(
                                agent_info["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                        del agent_info["action_mask"]

                group_action_spec = self.input_spec["full_action_spec", group, "action"]
                if isinstance(group_action_spec, (Categorical, OneHot)):
                    # We update the mask for available actions
                    group_action_spec.update_mask(group_mask.clone())

        return observation_dict, info_dict

    def _update_agent_mask(self, td):
        if self.use_mask:
            # In AEC only one agent acts, in parallel env self.agents contains the agents alive
            agents_acting = self.agents if self.parallel else [self.agent_selection]
            for group, agents in self.group_map.items():
                group_mask = td.get((group, "mask"))
                group_mask += True

                # We now add dead agents to the mask
                for index, agent in enumerate(agents):
                    if agent not in agents_acting:
                        group_mask[index] = False

    def close(self, *, raise_if_closed: bool = True) -> None:
        self._env.close()


class PettingZooEnv(PettingZooWrapper):
    """PettingZoo Environment.

    To install petting zoo follow the guide `here <https://github.com/Farama-Foundation/PettingZoo#installation>__`.

    This class is a general torchrl wrapper for all PettingZoo environments.
    It is able to wrap both ``pettingzoo.AECEnv`` and ``pettingzoo.ParallelEnv``.

    Let's see how more in details:

    For wrapping ``pettingzoo.ParallelEnv`` provide the name of your petting zoo task (in the ``task`` argument)
    and specify ``parallel=True``. This will construct the ``pettingzoo.ParallelEnv`` version of that task
    (if it is supported in pettingzoo) and wrap it for torchrl.
    In wrapped ``pettingzoo.ParallelEnv`` all agents will step at each environment step.
    If the number of agents during the task varies, please set ``use_mask=True``.
    ``"mask"`` will be provided
    as an output in each group and should be used to mask out dead agents.
    The environment will be reset as soon as one agent is done (unless ``done_on_any`` is ``False``).

    For wrapping ``pettingzoo.AECEnv`` provide the name of your petting zoo task (in the ``task`` argument)
    and specify ``parallel=False``. This will construct the ``pettingzoo.AECEnv`` version of that task
    and wrap it for torchrl.
    In wrapped ``pettingzoo.AECEnv``, at each step only one agent will act.
    For this reason, it is compulsory to set ``use_mask=True`` for this type of environment.
    ``"mask"`` will be provided as an output for each group and can be used to mask out non-acting agents.
    The environment will be reset only when all agents are done (unless ``done_on_any`` is ``True``).

    If there are any unavailable actions for an agent,
    the environment will also automatically update the mask of its ``action_spec`` and output an ``"action_mask"``
    for each group to reflect the latest available actions. This should be passed to a masked distribution during
    training.

    As a feature of torchrl multiagent, you are able to control the grouping of agents in your environment.
    You can group agents together (stacking their tensors) to leverage vectorization when passing them through the same
    neural network. You can split agents in different groups where they are heterogenous or should be processed by
    different neural networks. To group, you just need to pass a ``group_map`` at env constructiuon time.

    By default, agents in pettingzoo will be grouped by name.
    For example, with agents ``["agent_0","agent_1","agent_2","adversary_0"]``, the tensordicts will look like:

        >>> print(env.rand_action(env.reset()))
        TensorDict(
            fields={
                agent: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]))},
                adversary: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                        action_mask: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([1, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([1]))},
            batch_size=torch.Size([]))
        >>> print(env.group_map)
        {"agent": ["agent_0", "agent_1", "agent_2"], "adversary": ["adversary_0"]}

    Otherwise, a group map can be specified or selected from some premade options.
    See :class:`torchrl.envs.utils.MarlGroupMapType` for more info.
    For example, you can provide ``MarlGroupMapType.ONE_GROUP_PER_AGENT``, telling that each agent should
    have its own tensordict (similar to the pettingzoo parallel API).

    Grouping is useful for leveraging vectorization among agents whose data goes through the same
    neural network.

    Args:
        task (str): the name of the pettingzoo task to create in the "<env>/<task>" format (for example, "sisl/multiwalker_v9")
            or "<task>" format (for example, "multiwalker_v9").
        parallel (bool): if to construct the ``pettingzoo.ParallelEnv`` version of the task or the ``pettingzoo.AECEnv``.
        return_state (bool, optional): whether to return the global state from pettingzoo
            (not available in all environments).  Defaults to ``False``.
        group_map (MarlGroupMapType or Dict[str, List[str]]], optional): how to group agents in tensordicts for
            input/output. By default, agents will be grouped by their name. Otherwise, a group map can be specified
            or selected from some premade options. See :class:`torchrl.envs.utils.MarlGroupMapType` for more info.
        use_mask (bool, optional): whether the environment should output an ``"mask"``. This is compulsory in
            wrapped ``pettingzoo.AECEnv`` to mask out non-acting agents and should be also used
            for ``pettingzoo.ParallelEnv`` when the number of agents can vary. Defaults to ``False``.
        categorical_actions (bool, optional): if the environments actions are discrete, whether to transform
            them to categorical or one-hot.
        seed (int, optional): the seed.  Defaults to ``None``.
        done_on_any (bool, optional): whether the environment's done keys are set by aggregating the agent keys
            using ``any()`` (when ``True``) or ``all()`` (when ``False``). Default (``None``) is to use ``any()`` for
            parallel environments and ``all()`` for AEC ones.

    Examples:
        >>> # Parallel env
        >>> from torchrl.envs.libs.pettingzoo import PettingZooEnv
        >>> kwargs = {"n_pistons": 21, "continuous": True}
        >>> env = PettingZooEnv(
        ...     task="pistonball_v6",
        ...     parallel=True,
        ...     return_state=True,
        ...     group_map=None, # Use default (all pistons grouped together)
        ...     **kwargs,
        ... )
        >>> print(env.group_map)
        ... {'piston': ['piston_0', 'piston_1', ..., 'piston_20']}
        >>> env.rollout(10)
        >>> # AEC env
        >>> from torchrl.envs.libs.pettingzoo import PettingZooEnv
        >>> from torchrl.envs.utils import MarlGroupMapType
        >>> env = PettingZooEnv(
        ...     task="tictactoe_v3",
        ...     parallel=False,
        ...     use_mask=True, # Must use it since one player plays at a time
        ...     group_map=None # # Use default for AEC (one group per player)
        ... )
        >>> print(env.group_map)
        ... {'player_1': ['player_1'], 'player_2': ['player_2']}
        >>> env.rollout(10)
    """

    def __init__(
        self,
        task: str,
        parallel: bool,
        return_state: bool = False,
        group_map: MarlGroupMapType | dict[str, list[str]] | None = None,
        use_mask: bool = False,
        categorical_actions: bool = True,
        seed: int | None = None,
        done_on_any: bool | None = None,
        **kwargs,
    ):
        if not _has_pettingzoo:
            raise ImportError(
                f"pettingzoo python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
        kwargs["task"] = task
        kwargs["parallel"] = parallel
        kwargs["return_state"] = return_state
        kwargs["group_map"] = group_map
        kwargs["use_mask"] = use_mask
        kwargs["categorical_actions"] = categorical_actions
        kwargs["seed"] = seed
        kwargs["done_on_any"] = done_on_any

        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        if "task" not in kwargs:
            raise TypeError("Could not find environment key 'task' in kwargs.")
        if "parallel" not in kwargs:
            raise TypeError("Could not find environment key 'parallel' in kwargs.")

    def _build_env(
        self,
        task: str,
        parallel: bool,
        **kwargs,
    ) -> (
        pettingzoo.utils.env.ParallelEnv  # noqa: F821
        | pettingzoo.utils.env.AECEnv  # noqa: F821
    ):
        self.task_name = task

        try:
            from pettingzoo.utils.all_modules import all_environments
        except ModuleNotFoundError as err:
            warnings.warn(
                f"PettingZoo failed to load all modules with error message {err}, trying to load individual modules."
            )
            all_environments = _load_available_envs()

        if task not in all_environments:
            # Try looking at the literal translation of values
            task_module = None
            for value in all_environments.values():
                if value.__name__.split(".")[-1] == task:
                    task_module = value
                    break
            if task_module is None:
                raise RuntimeError(
                    f"Specified task not in available environments {all_environments}"
                )
        else:
            task_module = all_environments[task]

        if parallel:
            petting_zoo_env = task_module.parallel_env(**kwargs)
        else:
            petting_zoo_env = task_module.env(**kwargs)

        return super()._build_env(env=petting_zoo_env)
