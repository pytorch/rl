from typing import Dict, List, Optional, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    CompositeSpec,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend
from torchrl.envs.utils import _selective_unsqueeze

IMPORT_ERR = None
try:
    import vmas

    _has_vmas = True

except ImportError as err:
    _has_vmas = False
    IMPORT_ERR = err

__all__ = ["VmasWrapper", "VmasEnv"]


def _get_envs() -> List:
    if not _has_vmas:
        return []
    all_scenarios = vmas.scenarios + vmas.mpe_scenarios + vmas.debug_scenarios
    # TODO heterogenous spaces
    # For now torchrl does not support heterogenous spaces (Tple(Box)) so many OpenAI MPE scenarios do not work
    heterogenous_spaces_scenarios = [
        "simple_adversary",
        "simple_crypto",
        "simple_push",
        "simple_speaker_listener",
        "simple_tag",
        "simple_world_comm",
    ]

    return [
        scenario
        for scenario in all_scenarios
        if scenario not in heterogenous_spaces_scenarios
    ]


class VmasWrapper(_EnvWrapper):
    """Vmas environment wrapper.

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
                action: Tensor(shape=torch.Size([32, 10, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                info: TensorDict(
                    fields={
                        agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 10, 5]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 10]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([32, 10]),
            device=cpu,
            is_shared=False)

    """

    git_url = "https://github.com/proroklab/VectorizedMultiAgentSimulator"
    libname = "vmas"
    available_envs = _get_envs()

    def __init__(
        self, env: "vmas.simulator.environment.environment.Environment" = None, **kwargs
    ):
        if env is not None:
            kwargs["env"] = env
            if "device" in kwargs.keys() and kwargs["device"] != str(env.device):
                raise TypeError("Env device is different from vmas device")
            kwargs["device"] = str(env.device)
        super().__init__(**kwargs)

    @property
    def lib(self):
        return vmas

    def _build_env(
        self,
        env: "vmas.simulator.environment.environment.Environment",
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

    @set_gym_backend("gym")
    def _make_specs(
        self, env: "vmas.simulator.environment.environment.Environment"
    ) -> None:
        # TODO heterogenous spaces

        # Agent specs
        action_specs = []
        observation_specs = []
        reward_specs = []
        info_specs = []
        for agent_index, agent in enumerate(self.agents):
            action_specs.append(
                _gym_to_torchrl_spec_transform(
                    self.action_space[agent_index],
                    categorical_action_encoding=True,
                    device=self.device,
                )
            )  # shape = (n_actions_per_agent,)
            observation_specs.append(
                _gym_to_torchrl_spec_transform(
                    self.observation_space[agent_index],
                    device=self.device,
                )
            )  # shape = (n_obs_per_agent,)
            reward_specs.append(
                UnboundedContinuousTensorSpec(
                    shape=torch.Size((1,)),
                    device=self.device,
                )
            )  # shape = (1,)
            agent_info = self.scenario.info(agent)
            if len(agent_info):
                info_specs.append(
                    CompositeSpec(
                        {
                            key: UnboundedContinuousTensorSpec(
                                shape=_selective_unsqueeze(
                                    value, batch_size=torch.Size((self.num_envs,))
                                ).shape[1:],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            for key, value in agent_info.items()
                        },
                    ).to(self.device)
                )

        # Create multi-agent specs
        multi_agent_action_spec = torch.stack(
            action_specs, dim=0
        )  # UnboundedContinuousTensorSpec with shape = (n_agents, n_actions_per_agent)
        multi_agent_observation_spec = torch.stack(
            observation_specs, dim=0
        )  # UnboundedContinuousTensorSpec with shape = (n_agents, n_obs_per_agent)
        multi_agent_reward_spec = torch.stack(
            reward_specs, dim=0
        )  # UnboundedContinuousTensorSpec with shape = (n_agents, 1)

        done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )  # shape = (1,)

        self.unbatched_action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": multi_agent_action_spec}, shape=(self.n_agents,)
                )
            }
        )
        self.unbatched_observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": multi_agent_observation_spec},
                    shape=(self.n_agents,),
                )
            }
        )
        if len(info_specs):
            multi_agent_info_spec = torch.stack(info_specs, dim=0)
            self.unbatched_observation_spec[("agents", "info")] = multi_agent_info_spec

        self.unbatched_reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"reward": multi_agent_reward_spec}, shape=(self.n_agents,)
                )
            }
        )
        self.unbatched_done_spec = done_spec

        self.action_spec = self.unbatched_action_spec.expand(
            *self.batch_size, *self.unbatched_action_spec.shape
        )
        self.observation_spec = self.unbatched_observation_spec.expand(
            *self.batch_size, *self.unbatched_observation_spec.shape
        )
        self.reward_spec = self.unbatched_reward_spec.expand(
            *self.batch_size, *self.unbatched_reward_spec.shape
        )
        self.done_spec = self.unbatched_done_spec.expand(
            *self.batch_size, *self.unbatched_done_spec.shape
        )

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, vmas.simulator.environment.Environment):
            raise TypeError(
                "env is not of type 'vmas.simulator.environment.Environment'."
            )

    def _init_env(self) -> Optional[int]:
        pass

    def _set_seed(self, seed: Optional[int]):
        self._env.seed(seed)

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
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

        agent_tds = []
        for i in range(self.n_agents):
            agent_obs = self.read_obs(obs[i])
            agent_info = self.read_info(infos[i])

            agent_td = TensorDict(
                source={
                    "agents": {
                        "observation": agent_obs,
                    },
                },
                batch_size=(self.num_envs,),
                device=self.device,
            )
            if agent_info is not None:
                agent_td.set(("agents", "info"), agent_info)
            agent_tds.append(agent_td)

        tensordict_out = torch.stack(agent_tds, dim=1).to_tensordict()
        tensordict_out.batch_size = self.batch_size
        tensordict_out.set("done", dones)
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict.get(("agents", "action"))
        action = self.read_action(action)

        obs, rews, dones, infos = self._env.step(action)

        dones = self.read_done(dones)

        agent_tds = []
        for i in range(self.n_agents):
            agent_obs = self.read_obs(obs[i])
            agent_rew = self.read_reward(rews[i])
            agent_info = self.read_info(infos[i])

            agent_td = TensorDict(
                source={
                    "agents": {
                        "observation": agent_obs,
                        "reward": agent_rew,
                    },
                },
                batch_size=(self.num_envs,),
                device=self.device,
            )
            if agent_info is not None:
                agent_td.set(("agents", "info"), agent_info)
            agent_tds.append(agent_td)

        tensordict_out = torch.stack(agent_tds, dim=1).to_tensordict()
        tensordict_out.batch_size = self.batch_size
        tensordict_out.set("done", dones)

        return tensordict_out.select().set("next", tensordict_out)

    def read_obs(self, observations: torch.Tensor) -> torch.Tensor:
        observations = _selective_unsqueeze(
            observations, batch_size=torch.Size((self.num_envs,))
        )
        return observations

    def read_info(self, infos: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(infos) == 0:
            return None
        infos = TensorDict(
            source={
                key: _selective_unsqueeze(
                    value.to(torch.float32), batch_size=torch.Size((self.num_envs,))
                )
                for key, value in infos.items()
            },
            batch_size=torch.Size((self.num_envs,)),
            device=self.device,
        )

        return infos

    def read_done(self, done):
        done = _selective_unsqueeze(done, batch_size=torch.Size((self.num_envs,)))
        return done

    def read_reward(self, rewards):
        rewards = _selective_unsqueeze(rewards, batch_size=torch.Size((self.num_envs,)))
        return rewards

    def read_action(self, action):
        agent_actions = []
        for i in range(self.n_agents):
            agent_actions.append(action[:, i, ...])
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
                action: Tensor(torch.Size([5, 32, 10, 2]), dtype=torch.float64),
                done: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.bool),
                info: TensorDict(
                    fields={
                        cohesion_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                        collision_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                        separation_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                        velocity_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32)},
                    batch_size=torch.Size([5, 32, 10]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        info: TensorDict(
                            fields={
                                cohesion_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                                collision_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                                separation_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32),
                                velocity_rew: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32)},
                            batch_size=torch.Size([5, 32, 10]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(torch.Size([5, 32, 10, 18]), dtype=torch.float32)},
                    batch_size=torch.Size([5, 32, 10]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([5, 32, 10, 18]), dtype=torch.float32),
                reward: Tensor(torch.Size([5, 32, 10, 1]), dtype=torch.float32)},
            batch_size=torch.Size([5, 32, 10]),
            device=cpu,
            is_shared=False)
    """

    def __init__(
        self,
        scenario: Union[str, "vmas.simulator.scenario.BaseScenario"],
        num_envs: int,
        continuous_actions: bool = True,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if not _has_vmas:
            raise ImportError(
                f"vmas python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            ) from IMPORT_ERR
        kwargs["scenario"] = scenario
        kwargs["num_envs"] = num_envs
        kwargs["continuous_actions"] = continuous_actions
        kwargs["max_steps"] = max_steps
        kwargs["seed"] = seed
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "scenario" not in kwargs:
            raise TypeError("Could not find environment key 'scenario' in kwargs.")
        if "num_envs" not in kwargs:
            raise TypeError("Could not find environment key 'num_envs' in kwargs.")

    def _build_env(
        self,
        scenario: Union[str, "vmas.simulator.scenario.BaseScenario"],
        num_envs: int,
        continuous_actions: bool,
        max_steps: Optional[int],
        seed: Optional[int],
        **scenario_kwargs,
    ) -> "vmas.simulator.environment.environment.Environment":
        self.scenario_name = scenario
        from_pixels = scenario_kwargs.pop("from_pixels", False)
        pixels_only = scenario_kwargs.pop("pixels_only", False)

        return super()._build_env(
            env=vmas.make_env(
                scenario=scenario,
                num_envs=num_envs,
                device=self.device,
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
