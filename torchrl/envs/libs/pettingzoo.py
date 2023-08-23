from typing import Dict, List, Optional, Union

import torch
from tensordict.tensordict import TensorDictBase

from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend
from torchrl.envs.libs.utils import _check_marl_grouping, MarlGroupMapType

IMPORT_ERR = None
try:
    import pettingzoo

    _has_pettingzoo = True

except ImportError as err:
    _has_pettingzoo = False
    IMPORT_ERR = err

__all__ = ["PettingZooWrapper", "PettingZooEnv"]


all_environments = {}
if _has_pettingzoo:
    # TODO change this when new petting zoo version lands
    # from pettingzoo.utlis.all_modules import all_environments
    from pettingzoo.atari import (
        basketball_pong_v3,
        boxing_v2,
        combat_plane_v2,
        combat_tank_v2,
        double_dunk_v3,
        entombed_competitive_v3,
        entombed_cooperative_v3,
        flag_capture_v2,
        foozpong_v3,
        ice_hockey_v2,
        joust_v3,
        mario_bros_v3,
        maze_craze_v3,
        othello_v3,
        pong_v3,
        quadrapong_v4,
        space_invaders_v2,
        space_war_v2,
        surround_v2,
        tennis_v3,
        video_checkers_v4,
        volleyball_pong_v3,
        warlords_v3,
        wizard_of_wor_v3,
    )
    from pettingzoo.butterfly import (
        cooperative_pong_v5,
        knights_archers_zombies_v10,
        pistonball_v6,
    )
    from pettingzoo.classic import (
        chess_v6,
        connect_four_v3,
        gin_rummy_v4,
        go_v5,
        hanabi_v5,
        leduc_holdem_v4,
        rps_v2,
        texas_holdem_no_limit_v6,
        texas_holdem_v4,
        tictactoe_v3,
    )
    from pettingzoo.mpe import (
        simple_adversary_v3,
        simple_crypto_v3,
        simple_push_v3,
        simple_reference_v3,
        simple_speaker_listener_v4,
        simple_spread_v3,
        simple_tag_v3,
        simple_v3,
        simple_world_comm_v3,
    )
    from pettingzoo.sisl import multiwalker_v9, pursuit_v4, waterworld_v4

    all_environments = {
        "atari/basketball_pong_v3": basketball_pong_v3,
        "atari/boxing_v2": boxing_v2,
        "atari/combat_tank_v2": combat_tank_v2,
        "atari/combat_plane_v2": combat_plane_v2,
        "atari/double_dunk_v3": double_dunk_v3,
        "atari/entombed_competitive_v3": entombed_competitive_v3,
        "atari/entombed_cooperative_v3": entombed_cooperative_v3,
        "atari/flag_capture_v2": flag_capture_v2,
        "atari/foozpong_v3": foozpong_v3,
        "atari/joust_v3": joust_v3,
        "atari/ice_hockey_v2": ice_hockey_v2,
        "atari/maze_craze_v3": maze_craze_v3,
        "atari/mario_bros_v3": mario_bros_v3,
        "atari/othello_v3": othello_v3,
        "atari/pong_v3": pong_v3,
        "atari/quadrapong_v4": quadrapong_v4,
        "atari/space_invaders_v2": space_invaders_v2,
        "atari/space_war_v2": space_war_v2,
        "atari/surround_v2": surround_v2,
        "atari/tennis_v3": tennis_v3,
        "atari/video_checkers_v4": video_checkers_v4,
        "atari/volleyball_pong_v3": volleyball_pong_v3,
        "atari/wizard_of_wor_v3": wizard_of_wor_v3,
        "atari/warlords_v3": warlords_v3,
        "classic/chess_v6": chess_v6,
        "classic/rps_v2": rps_v2,
        "classic/connect_four_v3": connect_four_v3,
        "classic/tictactoe_v3": tictactoe_v3,
        "classic/leduc_holdem_v4": leduc_holdem_v4,
        "classic/texas_holdem_v4": texas_holdem_v4,
        "classic/texas_holdem_no_limit_v6": texas_holdem_no_limit_v6,
        "classic/gin_rummy_v4": gin_rummy_v4,
        "classic/go_v5": go_v5,
        "classic/hanabi_v5": hanabi_v5,
        "butterfly/knights_archers_zombies_v10": knights_archers_zombies_v10,
        "butterfly/pistonball_v6": pistonball_v6,
        "butterfly/cooperative_pong_v5": cooperative_pong_v5,
        "mpe/simple_adversary_v3": simple_adversary_v3,
        "mpe/simple_crypto_v3": simple_crypto_v3,
        "mpe/simple_push_v3": simple_push_v3,
        "mpe/simple_reference_v3": simple_reference_v3,
        "mpe/simple_speaker_listener_v4": simple_speaker_listener_v4,
        "mpe/simple_spread_v3": simple_spread_v3,
        "mpe/simple_tag_v3": simple_tag_v3,
        "mpe/simple_world_comm_v3": simple_world_comm_v3,
        "mpe/simple_v3": simple_v3,
        "sisl/multiwalker_v9": multiwalker_v9,
        "sisl/waterworld_v4": waterworld_v4,
        "sisl/pursuit_v4": pursuit_v4,
    }


def _get_envs() -> List[str]:
    if not _has_pettingzoo:
        return []
    return list(all_environments.keys())


class PettingZooWrapper(_EnvWrapper):
    """PettingZoo wrapper."""

    git_url = "https://github.com/Farama-Foundation/PettingZoo"
    libname = "pettingzoo"
    available_envs = _get_envs()

    def __init__(
        self,
        env: Union[
            "pettingzoo.utils.env.ParallelEnv", "pettingzoo.utils.env.AECEnv"
        ] = None,
        return_state: Optional[bool] = False,
        group_map: Optional[Union[MarlGroupMapType, Dict[str, List[str]]]] = None,
        use_action_mask: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        self.group_map = group_map
        self.return_state = return_state
        self.seed = seed
        self.use_action_mask = use_action_mask

        super().__init__(**kwargs)

    def _get_default_group_map(self, agent_names: List[str]):
        map = {}
        for agent_name in agent_names:
            # See if the agent follows the convention "name_int"
            follows_convention = True
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            try:
                agent_name_split[-1] = int(agent_name_split[-1])
            except ValueError:
                follows_convention = False

            # If not, just put it in a single group
            if not follows_convention:
                map[agent_name] = [agent_name]
            # Otherwise, group it with other agents that follow the same convention
            else:
                group_name = "_".join(agent_name_split[:-1])
                if group_name in map:
                    map[group_name].append(agent_name)
                else:
                    map[group_name] = [agent_name]
        for group_name in list(map.keys()):
            agent_names = map[group_name]
            # If there are groups with one agent only, rename them to the agent's name
            if len(agent_names) == 1 and group_name != agent_names[0]:
                map[agent_names[0]] = agent_names
                del map[group_name]

        return map

    @property
    def lib(self):
        return pettingzoo

    def _build_env(
        self,
        env: Union["pettingzoo.utils.env.ParallelEnv", "pettingzoo.utils.env.AECEnv"],
    ):
        self.parallel = isinstance(env, pettingzoo.utils.env.ParallelEnv)
        if len(self.batch_size):
            raise RuntimeError(
                f"PettingZoo does not support custom batch_size {self.batch_size}."
            )

        return env

    @set_gym_backend("gymnasium")
    def _make_specs(
        self,
        env: Union["pettingzoo.utils.env.ParallelEnv", "pettingzoo.utils.env.AECEnv"],
    ) -> None:

        # Create and check group map
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.possible_agents)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.possible_agents)
        _check_marl_grouping(self.group_map, self.possible_agents)

        action_spec = CompositeSpec()
        observation_spec = CompositeSpec()
        reward_spec = CompositeSpec()
        done_spec = CompositeSpec()
        for group, agents in self.group_map.items():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_done_spec,
            ) = self._make_group_specs(agent_names=agents)
            action_spec[group] = group_action_spec
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec
            done_spec[group] = group_done_spec

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

    def _make_group_specs(self, agent_names: List[str]):
        n_agents = len(agent_names)
        action_specs = []
        observation_specs = []
        for agent in agent_names:
            action_specs.append(
                CompositeSpec(
                    {
                        "action": _gym_to_torchrl_spec_transform(
                            self.action_space(agent),
                            remap_state_to_observation=False,
                            categorical_action_encoding=False,  # Always one hot
                            device=self.device,
                        )
                    },
                )
            )
            observation_specs.append(
                CompositeSpec(
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
        group_observation_inner_spec = group_observation_spec["observation"]
        if (
            isinstance(group_observation_inner_spec, CompositeSpec)
            and "action_mask" in group_observation_inner_spec.keys()
        ):
            del group_observation_inner_spec["action_mask"]
        if self.use_action_mask:
            group_observation_spec["action_mask"] = DiscreteTensorSpec(
                n=2,
                shape=group_action_spec["action"].shape,
                dtype=torch.bool,
                device=self.device,
            )

        group_reward_spec = CompositeSpec(
            {
                "reward": UnboundedContinuousTensorSpec(
                    shape=torch.Size((n_agents, 1)),
                    device=self.device,
                    dtype=torch.float32,
                )
            },
            shape=torch.Size((n_agents,)),
        )
        group_done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    n=2,
                    shape=torch.Size((n_agents, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=torch.Size((n_agents,)),
        )
        if n_agents == 1:
            group_observation_spec = group_observation_spec.squeeze(0)
            group_action_spec = group_action_spec.squeeze(0)
            group_reward_spec = group_reward_spec.squeeze(0)
            group_done_spec = group_done_spec.squeeze(0)
        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_done_spec,
        )

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(
            env, (pettingzoo.utils.env.ParallelEnv, pettingzoo.utils.env.AECEnv)
        ):
            raise TypeError("env is not of type expected.")

    def _init_env(self) -> Optional[int]:
        # Add info
        _, info_dict = self._env.reset()

        for group, agents in self.group_map.items():
            info_specs = []
            for agent in agents:
                info_specs.append(
                    CompositeSpec(
                        {
                            "info": CompositeSpec(
                                {
                                    key: UnboundedContinuousTensorSpec(
                                        shape=torch.tensor(value).shape
                                    )
                                    for key, value in info_dict[agent].items()
                                    if key != "action_mask"
                                }
                            )
                        },
                        device=self.device,
                    )
                )
            info_specs = torch.stack(info_specs, dim=0)
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
                state_example = torch.tensor(self.state(), device=self.device)
                state_spec = UnboundedContinuousTensorSpec(
                    shape=state_example.shape,
                    dtype=state_example.dtype,
                    device=self.device,
                )
            self.observation_spec["state"] = state_spec

    def _set_seed(self, seed: Optional[int]):
        self.seed = seed

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        if self.parallel:
            return self._reset_parallel()
        else:
            return self._reset_aec()

    def _reset_parallel(
        self,
    ) -> TensorDictBase:
        self._env.reset(seed=self.seed)

        observation_dict, info_dict = self._env.reset(seed=self.seed)
        tensordict_out = self.observation_spec.zero()
        observation_dict, info_dict = self._update_action_mask(
            tensordict_out, observation_dict, info_dict
        )

        for group, agent_names in self.group_map.items():
            group_observation = tensordict_out.get((group, "observation"))
            group_info = tensordict_out.get((group, "info"), None)

            for i, agent in enumerate(agent_names):
                index = (
                    i if len(agent_names) > 1 else Ellipsis
                )  # If group has one agent we index with '...'
                group_observation[index] = self.observation_spec[group, "observation"][
                    index
                ].encode(observation_dict[agent])
                if group_info is not None:
                    agent_info_dict = info_dict[agent]
                    for agent_info, value in agent_info_dict.items():
                        group_info.get(agent_info)[index] = torch.tensor(
                            value, device=self.device
                        )

        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action_dict = {}
        for group, agents in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_np = self.input_spec["_action_spec", group, "action"].to_numpy(
                group_action
            )
            for i, agent in enumerate(agents):
                if agent in self.agents:
                    if len(agents) > 1:
                        action_dict[agent] = group_action_np[i]
                    else:
                        action_dict[agent] = group_action_np
                else:
                    raise ValueError("Provided action for dead agent")

        (
            observation_dict,
            rewards_dict,
            terminations_dict,
            truncations_dict,
            info_dict,
        ) = self._env.step(action_dict)
        tensordict_out = self.observation_spec.zero()
        tensordict_out.update(self.output_spec["_reward_spec"].zero())
        tensordict_out.update(self.output_spec["_done_spec"].zero())
        observation_dict, info_dict = self._update_action_mask(
            tensordict_out, observation_dict, info_dict
        )

        for group, agent_names in self.group_map.items():
            group_observation = tensordict_out.get((group, "observation"))
            group_reward = tensordict_out.get((group, "reward"))
            group_done = tensordict_out.get((group, "done"))
            group_info = tensordict_out.get((group, "info"), None)

            for i, agent in enumerate(agent_names):
                if agent in observation_dict:  # Live agent
                    index = (
                        i if len(agent_names) > 1 else Ellipsis
                    )  # If group has one agent we index with '...'
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

                    if group_info is not None:
                        agent_info_dict = info_dict[agent]
                        for agent_info, value in agent_info_dict.items():
                            group_info.get(agent_info)[index] = torch.tensor(
                                value, device=self.device
                            )

                elif not self.use_action_mask:
                    # Dead agent, if we are not masking it out, this is not allowed
                    raise ValueError(
                        "Dead agents found in the environment,"
                        " you need to set use_action_mask=True to allow this."
                    )

        return tensordict_out.select().set("next", tensordict_out)

    def _update_action_mask(self, td, observation_dict, info_dict):
        if self.use_action_mask:
            for group, agents in self.group_map.items():
                group_mask = td.get((group, "action_mask"))
                group_mask += True
                for i, agent in enumerate(agents):
                    index = (
                        i if len(agents) > 1 else Ellipsis
                    )  # If group has one agent we index with '...'
                    if agent in observation_dict:  # Live agents
                        agent_obs = observation_dict[agent]
                        agent_info = info_dict[agent]
                        if isinstance(agent_obs, Dict) and "action_mask" in agent_obs:
                            group_mask[index] = torch.tensor(
                                agent_obs["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                            del agent_obs["action_mask"]
                        elif (
                            isinstance(agent_info, Dict) and "action_mask" in agent_info
                        ):
                            group_mask[index] = torch.tensor(
                                agent_info["action_mask"],
                                device=self.device,
                                dtype=torch.bool,
                            )
                    else:  # Dead agent
                        group_mask[index] = False
                group_action_spec = self.input_spec["_action_spec", group, "action"]
                if isinstance(
                    group_action_spec, (DiscreteTensorSpec, OneHotDiscreteTensorSpec)
                ):
                    group_action_spec.update_mask(group_mask)
        return observation_dict, info_dict

    def close(self) -> None:
        self._env.close()


class PettingZooEnv(PettingZooWrapper):
    """PettingZooEnv."""

    def __init__(
        self,
        task: str,
        parallel: bool,
        return_state: Optional[bool] = False,
        group_map: Optional[Union[MarlGroupMapType, Dict[str, List[str]]]] = None,
        use_action_mask: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if not _has_pettingzoo:
            raise ImportError(
                f"pettingzoo python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            ) from IMPORT_ERR
        kwargs["task"] = task
        kwargs["parallel"] = parallel
        kwargs["return_state"] = return_state
        kwargs["group_map"] = group_map
        kwargs["use_action_mask"] = use_action_mask
        kwargs["seed"] = seed
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "task" not in kwargs:
            raise TypeError("Could not find environment key 'task' in kwargs.")
        if "parallel" not in kwargs:
            raise TypeError("Could not find environment key 'parallel' in kwargs.")

    def _build_env(
        self,
        task: str,
        parallel: bool,
        **kwargs,
    ) -> Union["pettingzoo.utils.env.ParallelEnv", "pettingzoo.utils.env.AECEnv"]:
        self.task_name = task

        if task not in all_environments:
            # Try looking at the literal translation of values
            task_module = None
            for value in all_environments.values():
                if value.__name__.split(".")[-1] == task:
                    task_module = value
                    break
            if task_module is None:
                raise RuntimeError(f"Specified task not in {_get_envs()}")
        else:
            task_module = all_environments[task]

        if parallel:
            petting_zoo_env = task_module.parallel_env(**kwargs)
        else:
            petting_zoo_env = task_module.env(**kwargs)

        return super()._build_env(env=petting_zoo_env)
