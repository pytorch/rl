from typing import Dict, Optional

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    CompositeSpec,
    MultiOneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import GymLikeEnv

try:
    import smac
    import smac.env
    from smac.env.starcraft2.maps import get_map_params, smac_maps

    _has_smac = True
except ImportError as err:
    _has_smac = False
    IMPORT_ERR = str(err)


def _get_envs():
    if not _has_smac:
        return []
    return [map_name for map_name, _ in smac_maps.get_smac_map_registry().items()]


class SC2Wrapper(GymLikeEnv):
    """SMAC (StarCraft Multi-Agent Challenge) environment wrapper.

    Examples:
        >>> env = smac.env.StarCraft2Env("8m", seed=42) # Seed cannot be changed once environment was created.
        >>> env = SC2Wrapper(env)
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

    git_url = "https://github.com/oxwhirl/smac"
    available_envs = _get_envs()
    libname = "smac"

    def __init__(
        self,
        env: "smac.env.StarCraft2Env" = None,
        batch_size: Optional[torch.Size] = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
            if batch_size is None:
                batch_size = torch.Size([env.n_agents])

        kwargs["batch_size"] = batch_size
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (smac.env.StarCraft2Env,)):
            raise TypeError("env is not of type 'smac.env.StarCraft2Env'.")

    def _build_env(self, env, **kwargs) -> smac.env.StarCraft2Env:
        return env

    def _make_specs(self, env: "smac.env.StarCraft2Env") -> None:
        # Extract specs from definition.
        self.reward_spec = self._make_reward_spec()

        # Specs that require initialized environment are built in _init_env.

    def _init_env(self) -> None:
        self._env.reset()

        # Before extracting environment specific specs, env.reset() must be executed.
        self.input_spec = self._make_input_spec(self._env)
        self.observation_spec = self._make_observation_spec(self._env)
        self.state_spec = self._make_state_spec(self._env)

    def _make_reward_spec(self) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            shape=torch.Size(
                [
                    *self.batch_size,
                    1,
                ]
            ),
            device=self.device,
        )

    def _make_input_spec(self, env: "smac.env.StarCraft2Env") -> TensorSpec:
        mask = torch.tensor(
            env.get_avail_actions(), dtype=torch.bool, device=self.device
        )

        action_spec = MultiOneHotDiscreteTensorSpec(
            [env.n_actions],
            shape=torch.Size([env.n_agents, env.n_actions]),
            device=self.device,
            mask=mask,
        )
        return CompositeSpec(action=action_spec, shape=self.batch_size)

    def _make_observation_spec(self, env: "smac.env.StarCraft2Env") -> TensorSpec:
        obs_spec = UnboundedContinuousTensorSpec(
            torch.Size([env.n_agents, env.get_obs_size()]), device=self.device
        )
        return CompositeSpec(observation=obs_spec, shape=self.batch_size)

    def _make_state_spec(self, env: "smac.env.StarCraft2Env") -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            torch.Size([env.n_agents, env.get_state_size()]), device=self.device
        )

    def _action_transform(self, action: torch.Tensor):
        action_np = self.action_spec.to_numpy(action)
        return action_np

    def _read_state(self, state: np.ndarray) -> torch.Tensor:
        return self.state_spec.encode(
            torch.Tensor(state, device=self.device).expand(*self.state_spec.shape)
        )

    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError(
            "Seed cannot be changed once environment was created."
        )

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        env: smac.env.StarCraft2Env = self._env
        obs, state = env.reset()

        # collect outputs
        obs_dict = self.read_obs(obs)
        state = self._read_state(state)
        self._is_done = torch.zeros(self.batch_size, dtype=torch.bool)

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.set("done", self._is_done)
        tensordict_out["state"] = state

        self.input_spec = self._make_input_spec(env)

        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        env: smac.env.StarCraft2Env = self._env

        # perform actions
        action = tensordict.get("action")  # this is a list of actions for each agent
        action_np = self._action_transform(action)

        # Actions are validated by the environment.
        reward, done, info = env.step(action_np)

        # collect outputs
        obs_dict = self.read_obs(env.get_obs())
        # TODO: add centralized flag?
        state = self._read_state(env.get_state())

        reward = self._to_tensor(reward, dtype=self.reward_spec.dtype).expand(
            self.batch_size
        )
        done = self._to_tensor(done, dtype=torch.bool).expand(self.batch_size)

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        tensordict_out["state"] = state

        # Update available actions mask.
        self.input_spec = self._make_input_spec(env)

        return tensordict_out

    def get_seed(self) -> Optional[int]:
        return self._env.seed()


class SC2Env(SC2Wrapper):
    """SMAC (StarCraft Multi-Agent Challenge) environment wrapper.

    Examples:
        >>> env = SC2Env(map_name="8m", seed=42)
        >>> td = env.rand_step()
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
                reward: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', ...]
    """

    def __init__(
        self,
        map_name: str,
        batch_size: Optional[torch.Size] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        kwargs["map_name"] = map_name

        if batch_size is None:
            map_info = get_map_params(map_name)
            batch_size = torch.Size([map_info["n_agents"]])
        kwargs["batch_size"] = batch_size

        if seed is not None:
            kwargs["seed"] = seed

        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "map_name" not in kwargs:
            raise TypeError("Expected 'map_name' to be part of kwargs")

    def _build_env(
        self,
        map_name: str,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "smac.env.StarCraft2Env":
        if not _has_smac:
            raise RuntimeError(
                f"smac not found, unable to create smac.env.StarCraft2Env. "
                f"Consider installing smac. More info:"
                f" {self.git_url}. (Original error message during import: {IMPORT_ERR})."
            )

        self.wrapper_frame_skip = 1
        env = smac.env.StarCraft2Env(map_name, seed=seed, **kwargs)

        return super()._build_env(env)
