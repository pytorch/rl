# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import torch

from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper

IMPORT_ERR = None
try:
    import smacv2
    from smacv2.env.starcraft2.maps import smac_maps

    _has_smacv2 = True
except ImportError as err:
    _has_smacv2 = False
    IMPORT_ERR = err


def _get_envs():
    if not _has_smacv2:
        return []
    return smac_maps.get_smac_map_registry().keys()


class SMACv2Wrapper(_EnvWrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    Examples:
        >>> env = smac.env.StarCraft2Env("8m")
        >>> env = SMACv2Wrapper(env)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([8, 14]), dtype=torch.int64),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        observation: Tensor(torch.Size([8, 80]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([8, 80]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', ...]
    """

    git_url = "https://github.com/oxwhirl/smacv2"
    libname = "smacv2"
    available_envs = _get_envs()

    def __init__(
        self,
        env: "smacv2.env.StarCraft2Env" = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        super().__init__(**kwargs)

    @property
    def lib(self):
        return smacv2

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, smacv2.env.StarCraft2Env):
            raise TypeError("env is not of type 'smacv2.env.StarCraft2Env'.")

    def _build_env(
        self,
        env: "smacv2.env.StarCraft2Env",
    ):
        if len(self.batch_size):
            raise RuntimeError(
                f"SMACv2 does not support custom batch_size {self.batch_size}."
            )

        return env

    def _make_specs(self, env: "smacv2.env.StarCraft2Env") -> None:
        # Extract specs from definition.
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1,)),
            device=self.device,
        )
        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
            device=self.device,
        )

        # Specs that require initialized environment are built in _init_env.

    def _init_env(self) -> None:
        self._env.reset()

        # Before extracting environment specific specs, env.reset() must be executed.
        self.action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()

    def _make_action_spec(self) -> CompositeSpec:
        # TODO masking
        # mask = torch.tensor(env.get_avail_actions(), dtype=torch.bool, device=self.device)
        action_spec = MultiOneHotDiscreteTensorSpec(
            [self.n_actions],
            shape=torch.Size([self.n_agents, self.n_actions]),
            device=self.device,
        )
        spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": action_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        return spec

    def _make_observation_spec(self) -> CompositeSpec:
        obs_spec = UnboundedContinuousTensorSpec(
            torch.Size([self.n_agents, self.get_obs_size()]), device=self.device
        )
        spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": obs_spec}, shape=torch.Size((self.n_agents,))
                ),
                "state": UnboundedContinuousTensorSpec(
                    torch.Size([self.n_agents, self.get_state_size()]),
                    device=self.device,
                ),
            }
        )
        return spec

    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError(
            "Seed cannot be changed once environment was created."
        )

    # def _action_transform(self, action: torch.Tensor):
    #     action_np = self.action_spec.to_numpy(action)
    #     return action_np
    #
    # def _read_state(self, state: np.ndarray) -> torch.Tensor:
    #     return self.state_spec.encode(
    #         torch.Tensor(state, device=self.device).expand(*self.state_spec.shape)
    #     )
    #
    #
    # def _reset(
    #     self, tensordict: Optional[TensorDictBase] = None, **kwargs
    # ) -> TensorDictBase:
    #     env: smac.env.StarCraft2Env = self._env
    #     obs, state = env.reset()
    #
    #     # collect outputs
    #     obs_dict = self.read_obs(obs)
    #     state = self._read_state(state)
    #     self._is_done = torch.zeros(self.batch_size, dtype=torch.bool)
    #
    #     # build results
    #     tensordict_out = TensorDict(
    #         source=obs_dict,
    #         batch_size=self.batch_size,
    #         device=self.device,
    #     )
    #     tensordict_out.set("done", self._is_done)
    #     tensordict_out["state"] = state
    #
    #     self.input_spec = self._make_input_spec(env)
    #
    #     return tensordict_out
    #
    # def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     env: smac.env.StarCraft2Env = self._env
    #
    #     # perform actions
    #     action = tensordict.get("action")  # this is a list of actions for each agent
    #     action_np = self._action_transform(action)
    #
    #     # Actions are validated by the environment.
    #     reward, done, info = env.step(action_np)
    #
    #     # collect outputs
    #     obs_dict = self.read_obs(env.get_obs())
    #     # TODO: add centralized flag?
    #     state = self._read_state(env.get_state())
    #
    #     reward = self._to_tensor(reward, dtype=self.reward_spec.dtype).expand(
    #         self.batch_size
    #     )
    #     done = self._to_tensor(done, dtype=torch.bool).expand(self.batch_size)
    #
    #     # build results
    #     tensordict_out = TensorDict(
    #         source=obs_dict,
    #         batch_size=tensordict.batch_size,
    #         device=self.device,
    #     )
    #     tensordict_out.set("reward", reward)
    #     tensordict_out.set("done", done)
    #     tensordict_out["state"] = state
    #
    #     # Update available actions mask.
    #     self.input_spec = self._make_input_spec(env)
    #
    #     return tensordict_out


class SMACv2Env(SMACv2Wrapper):
    """SMACv2 (StarCraft Multi-Agent Challenge v2) environment wrapper.

    Examples:
        >>> env = SMACv2Env(map_name="8m")
        >>> print(env.available_envs)
        ['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', ...]
    """

    def __init__(
        self,
        map_name: str,
        capability_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if not _has_smacv2:
            raise ImportError(
                f"smacv2 python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            ) from IMPORT_ERR
        kwargs["map_name"] = map_name
        kwargs["capability_config"] = capability_config
        kwargs["seed"] = seed

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
    ) -> "smacv2.env.StarCraft2Env":

        if capability_config is not None:
            env = smacv2.env.StarCraftCapabilityEnvWrapper(
                capability_config=capability_config, map_name=map_name, seed=seed
            )
        else:
            env = smacv2.env.StarCraft2Env(
                capability_config=capability_config, map_name=map_name, seed=seed
            )

        return super()._build_env(env)
