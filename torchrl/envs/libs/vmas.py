from typing import Dict, List, Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import CompositeSpec, DEVICE_TYPING, UnboundedContinuousTensorSpec
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.envs.utils import _selective_unsqueeze

try:
    import vmas

    _has_vmas = True

except ImportError as err:

    _has_vmas = False
    IMPORT_ERR = str(err)

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
        ...          scenario_name="flocking",
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
        self.batch_size = torch.Size([env.n_agents, *self.batch_size])

        return env

    def _make_specs(
        self, env: "vmas.simulator.environment.environment.Environment"
    ) -> None:
        # TODO heterogenous spaces
        # For now the wrapper assumes all agent spaces to be homogenous, thus let's use agent0
        agent0 = self.agents[0]
        agent0_index = 0

        self.input_spec = CompositeSpec(
            action=(
                _gym_to_torchrl_spec_transform(
                    self.action_space[agent0_index],
                    categorical_action_encoding=True,
                    device=self.device,
                )
            )
        ).expand(self.batch_size)

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1,)),
            device=self.device,
        ).expand([*self.batch_size, 1])

        self.observation_spec = CompositeSpec(
            observation=(
                _gym_to_torchrl_spec_transform(
                    self.observation_space[agent0_index],
                    device=self.device,
                )
            ),
            info=CompositeSpec(
                {
                    key: UnboundedContinuousTensorSpec(
                        shape=_selective_unsqueeze(
                            value, batch_size=torch.Size((self.num_envs,))
                        ).shape[1:],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for key, value in self.scenario.info(agent0).items()
                },
            ).to(self.device),
        ).expand(self.batch_size)

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
            envs_to_reset = _reset.any(dim=0)
            for env_index, to_reset in enumerate(envs_to_reset):
                if to_reset:
                    self._env.reset_at(env_index)
            done = _selective_unsqueeze(self._env.done(), batch_size=(self.num_envs,))
            obs = []
            infos = []
            dones = []
            for agent in self.agents:
                obs.append(self.scenario.observation(agent))
                infos.append(self.scenario.info(agent))
                dones.append(done.clone())

        else:
            obs, infos = self._env.reset(return_info=True)
            dones = None

        agent_tds = []
        for i in range(self.n_agents):
            agent_obs = self.read_obs(obs[i])
            agent_info = self.read_info(infos[i])

            agent_td = TensorDict(
                source={
                    "observation": agent_obs,
                },
                batch_size=(self.num_envs,),
                device=self.device,
            )

            if infos is not None:
                agent_td.set("info", agent_info)
            if dones is not None:
                agent_td.set("done", dones[i])
            agent_tds.append(agent_td)

        tensordict_out = torch.stack(agent_tds, dim=0)

        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:

        action = tensordict.get("action")
        action = self.read_action(action)

        obs, rews, dones, infos = self._env.step(action)

        dones = self.read_done(dones)

        agent_tds = []
        for i in range(self.n_agents):
            agent_obs = self.read_obs(obs[i])
            agent_rew = self.read_reward(rews[i])
            agent_done = dones.clone()
            agent_info = self.read_info(infos[i])

            agent_td = TensorDict(
                source={
                    "observation": agent_obs,
                    "done": agent_done,
                    "reward": agent_rew,
                },
                batch_size=(self.num_envs,),
                device=self.device,
            )

            if infos is not None:
                agent_td.set("info", agent_info)
            agent_tds.append(agent_td)

        tensordict_out = torch.stack(agent_tds, dim=0)

        return tensordict_out

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
            agent_actions.append(action[i, :, ...])
        return agent_actions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, num_envs={self.num_envs}, n_agents={self.n_agents},"
            f" batch_size={self.batch_size}, device={self.device})"
        )

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self._env.to(device)
        return super().to(device)


class VmasEnv(VmasWrapper):
    """Vmas environment wrapper.

    Examples:
        >>>  env = VmasEnv(
        ...      scenario_name="flocking",
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
        scenario_name: str,
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
        kwargs["scenario_name"] = scenario_name
        kwargs["num_envs"] = num_envs
        kwargs["continuous_actions"] = continuous_actions
        kwargs["max_steps"] = max_steps
        kwargs["seed"] = seed
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "scenario_name" not in kwargs:
            raise TypeError("Could not find environment key 'scenario_name' in kwargs.")
        if "num_envs" not in kwargs:
            raise TypeError("Could not find environment key 'num_envs' in kwargs.")

    def _build_env(
        self,
        scenario_name: str,
        num_envs: int,
        continuous_actions: bool,
        max_steps: Optional[int],
        seed: Optional[int],
        **scenario_kwargs,
    ) -> "vmas.simulator.environment.environment.Environment":
        self.scenario_name = scenario_name
        from_pixels = scenario_kwargs.pop("from_pixels", False)
        pixels_only = scenario_kwargs.pop("pixels_only", False)

        return super()._build_env(
            env=vmas.make_env(
                scenario_name=scenario_name,
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
        return f"{super().__repr__()} (scenario_name={self.scenario_name})"
