# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Union

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs import GymLikeEnv

try:
    import jax
    import jumanji
    from jax import numpy as jnp
    from torchrl.envs.libs.jax_utils import (
        _extract_spec,
        _ndarray_to_tensor,
        _object_to_tensordict,
        _tensordict_to_object,
        _tree_flatten,
        _tree_reshape,
    )

    _has_jumanji = True
    IMPORT_ERR = ""
except ImportError as err:
    _has_jumanji = False
    IMPORT_ERR = str(err)


def _get_envs():
    if not _has_jumanji:
        return []
    return jumanji.registered_environments()


def _jumanji_to_torchrl_spec_transform(
    spec,
    dtype: Optional[torch.dtype] = None,
    device: DEVICE_TYPING = None,
    categorical_action_encoding: bool = True,
) -> TensorSpec:
    if isinstance(spec, jumanji.specs.DiscreteArray):
        action_space_cls = (
            DiscreteTensorSpec
            if categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return action_space_cls(spec.num_values, dtype=dtype, device=device)
    elif isinstance(spec, jumanji.specs.BoundedArray):
        shape = spec.shape
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return BoundedTensorSpec(
            shape=shape,
            minimum=np.asarray(spec.minimum),
            maximum=np.asarray(spec.maximum),
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, jumanji.specs.Array):
        shape = spec.shape
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        if dtype in (torch.float, torch.double, torch.half):
            return UnboundedContinuousTensorSpec(
                shape=shape, dtype=dtype, device=device
            )
        else:
            return UnboundedDiscreteTensorSpec(shape=shape, dtype=dtype, device=device)
    elif isinstance(spec, jumanji.specs.Spec) and hasattr(spec, "__dict__"):
        new_spec = {}
        for key, value in spec.__dict__.items():
            if isinstance(value, jumanji.specs.Spec):
                if key.endswith("_obs"):
                    key = key[:-4]
                if key.endswith("_spec"):
                    key = key[:-5]
                new_spec[key] = _jumanji_to_torchrl_spec_transform(
                    value, dtype, device, categorical_action_encoding
                )
        return CompositeSpec(**new_spec)
    else:
        raise TypeError(f"Unsupported spec type {type(spec)}")


class JumanjiWrapper(GymLikeEnv):
    """Jumanji environment wrapper.

    Examples:
        >>> env = jumanji.make("Snake-6x6-v0")
        >>> env = JumanjiWrapper(env)
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td1)
        TensorDict(
            fields={
                action: Tensor(torch.Size([1]), dtype=torch.int32),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        observation: Tensor(torch.Size([6, 6, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([6, 6, 5]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32),
                state: TensorDict(...)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['Snake-6x6-v0', 'Snake-12x12-v0', 'TSP50-v0', 'TSP100-v0', ...]
    """

    git_url = "https://github.com/instadeepai/jumanji"
    available_envs = _get_envs()
    libname = "jumanji"

    @property
    def lib(self):
        return jumanji

    def __init__(self, env: "jumanji.env.Environment" = None, **kwargs):
        if not _has_jumanji:
            raise ImportError(
                "jumanji is not installed or importing it failed. Consider checking your installation."
            ) from IMPORT_ERR
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)

    def _build_env(
        self,
        env,
        _seed: Optional[int] = None,
        from_pixels: bool = False,
        render_kwargs: Optional[dict] = None,
        pixels_only: bool = False,
        camera_id: Union[int, str] = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        if from_pixels:
            raise NotImplementedError("TODO")
        return env

    def _make_state_example(self, env):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state, _ = jax.vmap(env.reset)(jnp.stack(keys))
        state = _tree_reshape(state, self.batch_size)
        return state

    def _make_state_spec(self, env) -> TensorSpec:
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict)
        return state_spec

    def _make_action_spec(self, env) -> TensorSpec:
        action_spec = _jumanji_to_torchrl_spec_transform(
            env.action_spec(), device=self.device
        )
        action_spec = action_spec.expand(*self.batch_size, *action_spec.shape)
        return action_spec

    def _make_observation_spec(self, env) -> TensorSpec:
        spec = env.observation_spec()
        new_spec = _jumanji_to_torchrl_spec_transform(spec, device=self.device)
        if isinstance(spec, jumanji.specs.Array):
            return CompositeSpec(observation=new_spec).expand(self.batch_size)
        elif isinstance(spec, jumanji.specs.Spec):
            return CompositeSpec(**{k: v for k, v in new_spec.items()}).expand(
                self.batch_size
            )
        else:
            raise TypeError(f"Unsupported spec type {type(spec)}")

    def _make_reward_spec(self, env) -> TensorSpec:
        reward_spec = _jumanji_to_torchrl_spec_transform(
            env.reward_spec(), device=self.device
        )
        if not len(reward_spec.shape):
            reward_spec.shape = torch.Size([1])
        return reward_spec.expand([*self.batch_size, *reward_spec.shape])

    def _make_specs(self, env: "jumanji.env.Environment") -> None:  # noqa: F821

        # extract spec from jumanji definition
        self.action_spec = self._make_action_spec(env)
        self.observation_spec = self._make_observation_spec(env)
        self.reward_spec = self._make_reward_spec(env)

        # extract state spec from instance
        state_spec = self._make_state_spec(env).expand(self.batch_size)
        self.state_spec["state"] = state_spec
        self.observation_spec["state"] = state_spec.clone()

        # build state example for data conversion
        self._state_example = self._make_state_example(env)

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (jumanji.env.Environment,)):
            raise TypeError("env is not of type 'jumanji.env.Environment'.")

    def _init_env(self):
        pass

    def _set_seed(self, seed):
        if seed is None:
            raise Exception("Jumanji requires an integer seed.")
        self.key = jax.random.PRNGKey(seed)

    def read_state(self, state):
        state_dict = _object_to_tensordict(state, self.device, self.batch_size)
        return self.state_spec["state"].encode(state_dict)

    def read_obs(self, obs):
        if isinstance(obs, (list, jnp.ndarray, np.ndarray)):
            obs_dict = _ndarray_to_tensor(obs).to(self.device)
        else:
            obs_dict = _object_to_tensordict(obs, self.device, self.batch_size)
        return super().read_obs(obs_dict)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:

        # prepare inputs
        state = _tensordict_to_object(tensordict.get("state"), self._state_example)
        action = self.read_action(tensordict.get("action"))
        reward = self.reward_spec.zero()

        # flatten batch size into vector
        state = _tree_flatten(state, self.batch_size)
        action = _tree_flatten(action, self.batch_size)

        # jax vectorizing map on env.step
        state, timestep = jax.vmap(self._env.step)(state, action)

        # reshape batch size from vector
        state = _tree_reshape(state, self.batch_size)
        timestep = _tree_reshape(timestep, self.batch_size)

        # collect outputs
        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
        reward = self.read_reward(reward, np.asarray(timestep.reward))
        done = timestep.step_type == self.lib.types.StepType.LAST
        done = _ndarray_to_tensor(done).view(torch.bool).to(self.device)

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        tensordict_out["state"] = state_dict

        return tensordict_out.select().set("next", tensordict_out)

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        # generate random keys
        self.key, *keys = jax.random.split(self.key, self.numel() + 1)

        # jax vectorizing map on env.reset
        state, timestep = jax.vmap(self._env.reset)(jnp.stack(keys))

        # reshape batch size from vector
        state = _tree_reshape(state, self.batch_size)
        timestep = _tree_reshape(timestep, self.batch_size)

        # collect outputs
        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
        done = self.done_spec.zero()

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.set("done", done)
        tensordict_out["state"] = state_dict

        return tensordict_out


class JumanjiEnv(JumanjiWrapper):
    """Jumanji environment wrapper.

    Examples:
        >>> env = JumanjiEnv(env_name="Snake-6x6-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)
    """

    def __init__(self, env_name, **kwargs):
        kwargs["env_name"] = env_name
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "jumanji.env.Environment":
        if not _has_jumanji:
            raise RuntimeError(
                f"jumanji not found, unable to create {env_name}. "
                f"Consider installing jumanji. More info:"
                f" {self.git_url}."
            ) from IMPORT_ERR
        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        if kwargs:
            raise ValueError(f"Extra kwargs are not supported by {type(self)}.")
        self.wrapper_frame_skip = 1
        env = self.lib.make(env_name, **kwargs)
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"
