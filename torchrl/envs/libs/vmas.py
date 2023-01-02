from typing import Dict, List, Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

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
    return vmas.scenarios + vmas.mpe_scenarios + vmas.debug_scenarios


def _selective_unsqueeze(tensor: torch.Tensor, batch_size: torch.Size, dim: int = -1):
    shape_len = len(tensor.shape)

    assert shape_len >= len(batch_size)
    assert tensor.shape[: len(batch_size)] == batch_size

    if shape_len == len(batch_size):
        return tensor.unsqueeze(dim=dim)
    return tensor


class VmasWrapper(_EnvWrapper):

    git_url = "https://github.com/proroklab/VectorizedMultiAgentSimulator"
    libname = "vmas"
    available_envs = _get_envs()

    def __init__(
        self, env: "vmas.simulator.environment.environment.Environment"= None, **kwargs
    ):
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)

        assert self.device == self._env.device
        if len(self.batch_size) == 0:
            # Batch size not set
            self.batch_size = torch.Size((self.num_envs,))
        elif len(self.batch_size) == 1:
            # Batch size is set
            if not self.batch_size[0] == self.num_envs:
                raise TypeError("Batch size used in constructor does not match vmas batch size.")
        else:
            raise TypeError(
                "Batch size used in constructor is not compatible with vmas."
            )
        self.batch_size = torch.Size([*self.batch_size, self.n_agents])

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

        # TODO rendering
        if self.from_pixels:
            raise NotImplementedError("vmas rendiering not yet implemented")

        return env

    def _make_specs(
        self, env: "vmas.simulator.environment.environment.Environment"
    ) -> None:
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
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1,)),
            device=self.device,
        )

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
                    )
                    for key, value in self.scenario.info(agent0).items()
                },
            ).to(self.device),
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

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        obs, info = self._env.reset(return_info=True)

        obs = self.read_obs(obs)
        info = self.read_info(info)

        tensordict_out = TensorDict(
            source={"observation": obs},
            batch_size=self.batch_size,
            device=self.device,
        )

        if info is not None:
            tensordict_out.set("info", info)

        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict.get("action")
        action = self.read_action(action)

        obs, rews, dones, infos = self._env.step(action)

        obs = self.read_obs(obs)
        rews = self.read_reward(rews)
        dones = self.read_done(dones)
        infos = self.read_info(infos)

        tensordict_out = TensorDict(
            source={
                "observation": obs,
                "done": dones,
                "reward": rews,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        if infos is not None:
            tensordict_out.set("info", infos)

        return tensordict_out

    def read_obs(self, observations: List[torch.Tensor]) -> torch.Tensor:
        observations = torch.stack(observations, dim=1)
        observations = _selective_unsqueeze(observations, batch_size=self.batch_size)
        return observations

    def read_info(self, infos: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        infos = [
            TensorDict(
                source={
                    key: _selective_unsqueeze(value, batch_size=self.batch_size[:1])
                    for key, value in agent_info.items()
                },
                batch_size=torch.Size((self.num_envs,)),
                device=self.device,
            )
            for agent_info in infos
            if len(agent_info) > 0
        ]
        infos = torch.stack(infos, dim=1).to_tensordict() if len(infos) > 0 else None

        return infos

    def read_done(self, done):
        done = _selective_unsqueeze(done, batch_size=self.batch_size[:1])
        done = done.repeat(1, self.n_agents)
        done = _selective_unsqueeze(done, batch_size=self.batch_size)
        return done

    def read_reward(self, rewards):
        rewards = torch.stack(rewards, dim=1)
        rewards = _selective_unsqueeze(rewards, batch_size=self.batch_size)
        return rewards

    def read_action(self, action):
        if isinstance(self.action_spec, MultOneHotDiscreteTensorSpec):
            vals = action.split([space.n for space in self.action_spec.space], dim=-1)
            action = torch.stack([val.argmax(-1) for val in vals], -1)

        agent_actions = []
        for i in range(self.n_agents):
            agent_actions.append(action[:, i, ...])
        return agent_actions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, num_envs={self.num_envs}, n_agents={self.n_agents},"
            f" batch_size={self.batch_size}, device={self.device})"
        )


class VmasEnv(VmasWrapper):
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
                f"More info: {self.git_url} (ImportError: {IMPORT_ERR})"
            )
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
