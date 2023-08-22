from typing import Dict, List, Optional, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend
from torchrl.envs.libs.utils import _check_marl_grouping, MarlGroupMapType

IMPORT_ERR = None
try:
    import pettingzoo

    _has_pettingzoo = True

except ImportError as err:
    _has_pettingzoo = False
    IMPORT_ERR = err

# __all__ = ["VmasWrapper", "VmasEnv"]


def _get_envs() -> List[str]:
    return [
        "atari/basketball_pong_v3",
        "atari/boxing_v2",
        "atari/combat_tank_v2",
        "atari/combat_plane_v2",
        "atari/double_dunk_v3",
        "atari/entombed_competitive_v3",
        "atari/entombed_cooperative_v3",
        "atari/flag_capture_v2",
        "atari/foozpong_v3",
        "atari/joust_v3",
        "atari/ice_hockey_v2",
        "atari/maze_craze_v3",
        "atari/mario_bros_v3",
        "atari/othello_v3",
        "atari/pong_v3",
        "atari/quadrapong_v4",
        "atari/space_invaders_v2",
        "atari/space_war_v2",
        "atari/surround_v2",
        "atari/tennis_v3",
        "atari/video_checkers_v4",
        "atari/volleyball_pong_v3",
        "atari/wizard_of_wor_v3",
        "atari/warlords_v3",
        "classic/chess_v6",
        "classic/rps_v2",
        "classic/connect_four_v3",
        "classic/tictactoe_v3",
        "classic/leduc_holdem_v4",
        "classic/texas_holdem_v4",
        "classic/texas_holdem_no_limit_v6",
        "classic/gin_rummy_v4",
        "classic/go_v5",
        "classic/hanabi_v5",
        "butterfly/knights_archers_zombies_v10",
        "butterfly/pistonball_v6",
        "butterfly/cooperative_pong_v5",
        "mpe/simple_adversary_v3",
        "mpe/simple_crypto_v3",
        "mpe/simple_push_v3",
        "mpe/simple_reference_v3",
        "mpe/simple_speaker_listener_v4",
        "mpe/simple_spread_v3",
        "mpe/simple_tag_v3",
        "mpe/simple_world_comm_v3",
        "mpe/simple_v3",
        "sisl/multiwalker_v9",
        "sisl/waterworld_v4",
        "sisl/pursuit_v4",
    ]


class PettingZooWrapper(_EnvWrapper):

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
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
        self.group_map = group_map
        self.return_state = return_state

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
        for group_name, agent_names in map.items():
            # If there are groups with one agent only, rename them to the agent's name
            if len(agent_names) == 1 and group_name != agent_names[0]:
                map[agent_names[0]] = agent_name
                del map[group_name]

        return map

    @property
    def lib(self):
        return pettingzoo

    def _build_env(
        self,
        env: Union["pettingzoo.utils.env.ParallelEnv", "pettingzoo.utils.env.AECEnv"],
    ):
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
        if self.return_state:
            self._env.reset()
            state_example = torch.tensor(self.state(), device=self.device)
            self.observation_spec["state"] = UnboundedContinuousTensorSpec(
                shape=state_example.shape, dtype=state_example.dtype, device=self.device
            )

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            raise NotImplementedError(
                "Seed cannot be changed once environment was created."
            )

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        pass

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        pass


# class _ParallelEnvPettingZooWrapper(PettingZooWrapper):
#     pass
#
#
# class _AECEnvPettingZooWrapper(PettingZooWrapper):
#     pass


# class PettingZooEnv(VmasWrapper):
#     def __init__(
#         self,
#         scenario: Union[str, "vmas.simulator.scenario.BaseScenario"],
#         num_envs: int,
#         continuous_actions: bool = True,
#         max_steps: Optional[int] = None,
#         seed: Optional[int] = None,
#         **kwargs,
#     ):
#         if not _has_pettingzoo:
#             raise ImportError(
#                 f"vmas python package was not found. Please install this dependency. "
#                 f"More info: {self.git_url}."
#             ) from IMPORT_ERR
#         kwargs["scenario"] = scenario
#         kwargs["num_envs"] = num_envs
#         kwargs["continuous_actions"] = continuous_actions
#         kwargs["max_steps"] = max_steps
#         kwargs["seed"] = seed
#         super().__init__(**kwargs)
#
#     def _check_kwargs(self, kwargs: Dict):
#         if "scenario" not in kwargs:
#             raise TypeError("Could not find environment key 'scenario' in kwargs.")
#         if "num_envs" not in kwargs:
#             raise TypeError("Could not find environment key 'num_envs' in kwargs.")
#
#     def _build_env(
#         self,
#         scenario: Union[str, "vmas.simulator.scenario.BaseScenario"],
#         num_envs: int,
#         continuous_actions: bool,
#         max_steps: Optional[int],
#         seed: Optional[int],
#         **scenario_kwargs,
#     ) -> "vmas.simulator.environment.environment.Environment":
#         self.scenario_name = scenario
#         from_pixels = scenario_kwargs.pop("from_pixels", False)
#         pixels_only = scenario_kwargs.pop("pixels_only", False)
#
#         return super()._build_env(
#             env=vmas.make_env(
#                 scenario=scenario,
#                 num_envs=num_envs,
#                 device=self.device,
#                 continuous_actions=continuous_actions,
#                 max_steps=max_steps,
#                 seed=seed,
#                 wrapper=None,
#                 **scenario_kwargs,
#             ),
#             pixels_only=pixels_only,
#             from_pixels=from_pixels,
#         )
