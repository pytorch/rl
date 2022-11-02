import collections
from typing import Optional, Dict, Union, Tuple

import numpy as np
import torch

from torchrl.data import (
    DEVICE_TYPING,
    TensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs import GymLikeEnv

try:
    import jax
    import jumanji

    _has_jumanji = True
except ImportError:
    _has_jumanji = False


def _jumanji_to_torchrl_spec_transform(
    spec,
    dtype: Optional[torch.dtype] = None,
    device: DEVICE_TYPING = None,
    categorical_action_encoding: bool = True,
) -> TensorSpec:
    if isinstance(spec, collections.OrderedDict):
        spec = {
            "next_" + k: _jumanji_to_torchrl_spec_transform(item, device=device)
            for k, item in spec.items()
        }
        return CompositeSpec(**spec)
    elif isinstance(spec, jumanji.specs.DiscreteArray):
        action_space_cls = (
            DiscreteTensorSpec
            if categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        return action_space_cls(spec.num_values, device=device)
    elif isinstance(spec, jumanji.specs.BoundedArray):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            shape=spec.shape,
            minimum=spec.minimum,
            maximum=spec.maximum,
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, jumanji.specs.Array):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        if dtype in (torch.float, torch.double, torch.half):
            return NdUnboundedContinuousTensorSpec(
                shape=spec.shape, dtype=dtype, device=device
            )
        else:
            return NdUnboundedDiscreteTensorSpec(
                shape=spec.shape, dtype=dtype, device=device
            )
    else:
        raise NotImplementedError(type(spec))


class JumanjiWrapper(GymLikeEnv):
    git_url = "https://github.com/instadeepai/jumanji"

    @property
    def lib(self):
        return jumanji

    def __init__(self, env: "jumanji.env.Environment" = None, **kwargs):
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

    def _make_specs(self, env: "jumanji.env.Environment") -> None:  # noqa: F821
        obs_spec = env.observation_spec()
        self._observation_spec = CompositeSpec(next_observation=obs_spec)
        action_spec = env.action_spec()
        self._input_spec = CompositeSpec(action=action_spec)
        reward_spec = env.reward_spec()
        self._reward_spec = reward_spec

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (jumanji.env.Environment,)):
            raise TypeError("env is not of type 'jumanji.env.Environment'.")

    def _set_seed(self, seed):
        random_key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(random_key)
        self._key1 = key1
        self._key2 = key2

    def _reset(self, tensordict):
        # Sketch of functionality:
        keys = jax.random.split(self._key1, *self.batch_size)
        state, timestep = jax.vmap(self.env.reset)(keys)
        obs_dict = self.read_obs(state)
        tensordict.update(obs_dict)
        return tensordict

    def _read_state(self, tensordict):
        """Reads a tensordict, and maps it onto a State"""
        raise NotImplementedError

    def _step(self, tensordict):
        def step_fn(state, key):
            action = jax.random.randint(
                key=key, minval=0, maxval=self.frame_skip, shape=()
            )
            new_state, timestep = self.env.step(state, action)
            return new_state, timestep

        # the state is the input to the stateless env. It should be contained in the tensordict.
        state = self._read_state(tensordict)
        random_keys = jax.random.split(self._key2, 1)
        state, _ = jax.lax.scan(step_fn, state, random_keys)
        obs_dict = self.read_obs(state)
        tensordict.update(obs_dict)
        return tensordict

    def _output_transform(
        self, timestep_tuple: Tuple["TimeStep"]  # noqa: F821
    ) -> Tuple[np.ndarray, float, bool]:
        # Copy-paste from dm_control, should hold but does not account for the state,
        # only the timestamp
        if type(timestep_tuple) is not tuple:
            timestep_tuple = (timestep_tuple,)
        reward = timestep_tuple[0].reward

        done = False  # dm_control envs are non-terminating
        observation = timestep_tuple[0].observation
        return observation, reward, done


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
    ) -> "gym.core.Env":
        if not _has_jumanji:
            raise RuntimeError(
                f"jumanji not found, unable to create {env_name}. "
                f"Consider installing jumanji. More info:"
                f" {self.git_url}"
            )
        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        assert not kwargs
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
