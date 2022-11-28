import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Dict, Optional

from torchrl.data import (
    CompositeSpec,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    NdBoundedTensorSpec,
    CustomNdOneHotDiscreteTensorSpec,
    NdUnboundedContinuousTensorSpec,
    UnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
)
from torchrl.envs import GymLikeEnv

try:
    import smac
    from smac.env import StarCraft2Env

    _has_smac = True
except ImportError as err:
    _has_smac = False
    IMPORT_ERR = str(err)


# TODO: discuss with Vincent if separation to ..Wrapper and ..Env classes makes sense here.
class SC2Wrapper(GymLikeEnv):
    """TODO: comments
    """
    git_url = "https://github.com/oxwhirl/smac"

    def __init__(self, map_name: str = None, **kwargs):
        if map_name is not None:
            kwargs["map_name"] = map_name
        # TODO: process seed?
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        pass

    def _init_env(self) -> Optional[int]:
        # TODO: verify that isn't required.
        pass

    def _build_env(self, env, seed: Optional[int] = None, **kwargs) -> "smac.env.StarCraft2Env":
        # TODO: if required
        # self.from_pixels = from_pixels
        # self.pixels_only = pixels_only

        # if from_pixels:
        #     raise NotImplementedError("TODO")
        return env

    def _make_state_example(self, env):
        # TODO
        pass
        # key = jax.random.PRNGKey(0)
        # keys = jax.random.split(key, self.batch_size.numel())
        # state, _ = jax.vmap(env.reset)(jnp.stack(keys))
        # state = self._reshape(state)
        # return state

    def _make_state_spec(self, env) -> TensorSpec:
        # TODO
        pass
        # key = jax.random.PRNGKey(0)
        # state, _ = env.reset(key)
        # state_dict = _object_to_tensordict(state, self.device, batch_size=())
        # state_spec = _torchrl_data_to_spec_transform(state_dict)
        # return state_spec

    def _make_input_spec(self, env: StarCraft2Env) -> TensorSpec:
        action_spec = CustomNdOneHotDiscreteTensorSpec(
            torch.tensor(env.get_avail_actions()),
            device=self.device
        )
        return CompositeSpec(action=action_spec)

    def _make_observation_spec(self, env: StarCraft2Env) -> TensorSpec:
        info = env.get_env_info()
        size = torch.Size(info["n_agents"], info["obs_shape"])
        return NdUnboundedContinuousTensorSpec(size, device=self.device)

    def _make_reward_spec(self) -> TensorSpec:
        return UnboundedContinuousTensorSpec(device=self.device)

    def _make_specs(self, env: StarCraft2Env) -> None:
        # extract specs from definition
        self._reward_spec = self._make_reward_spec()

        # extract specs from instance
        self._input_spec = self._make_input_spec(env)
        self._observation_spec = self._make_observation_spec(env)
        self._state_spec = self._make_state_spec(env)
        self._input_spec["state"] = self._state_spec

        # TODO: build state example for data conversion
        self._state_example = self._make_state_example(env)

    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError("Seed can be set only when creating environment.")

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        env: smac.env.StarCraft2Env = self._env
        obs, state = env.reset()

        # reshape batch size from vector
        # TODO
        state = self._reshape(state)
        obs = self._reshape(obs)

        # collect outputs
        state_dict = self.read_state(state)
        obs_dict = self.read_obs(obs)
        done = torch.zeros(self.batch_size, dtype=torch.bool)

        self._is_done = done

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.set("done", done)
        tensordict_out["state"] = state_dict

        return tensordict_out


class SC2Env(SC2Wrapper):
    """TODO: comments
    """

    def __init__(self, map_name: str, seed: Optional[int] = None, **kwargs):
        kwargs["map_name"] = map_name
        if seed is not None:
            kwargs["seed"] = map_name

        super().__init__(**kwargs)

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
        # TODO: check if those are required
        # from_pixels = kwargs.pop("from_pixels", False)
        # pixels_only = kwargs.pop("pixels_only", True)

        # TODO: check if this is required
        # self.wrapper_frame_skip = 1
        env = smac.env.StarCraft2Env(map_name, seed, **kwargs)

        # TODO: return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)
        return super()._build_env(env)
