import collections
from typing import Optional, Dict, Union, Tuple

import numpy as np
import torch

from torchrl.data import (
    DEVICE_TYPING,
    TensorDict,
    TensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
)
from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs import GymLikeEnv

try:
    import jax
    from jax import numpy as jnp
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
        self._input_spec = CompositeSpec(
            action=_jumanji_to_torchrl_spec_transform(
                self._env.action_spec(), device=self.device
            )
        )
        self._observation_spec = CompositeSpec(
            next_observation=_jumanji_to_torchrl_spec_transform(
                self._env.observation_spec(), device=self.device
            )
        )
        self._reward_spec = _jumanji_to_torchrl_spec_transform(
            self._env.reward_spec(), device=self.device
        )

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, (jumanji.env.Environment,)):
            raise TypeError("env is not of type 'jumanji.env.Environment'.")

    def _init_env(self, seed: Optional[int] = None) -> Optional[int]:
        seed = self.set_seed(seed)
        return seed

    def _set_seed(self, seed):
        # TODO: when seed is None, what should happen?
        if seed is None:
            # jax.random.PRNGKey requires an integer seed.
            seed = int.from_bytes(np.random.bytes(8), byteorder="big", signed=True)
        self.key = jax.random.PRNGKey(seed)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:

        state = self._decode_state(tensordict)
        action = self.read_action(tensordict.get("action"))
        reward = self.reward_spec.zero(self.batch_size)

        state = self._flatten(state)
        action = self._flatten(action)
        state, timestep = jax.vmap(self._env.step)(state, action)
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        state_dict = self._encode_state(state)
        obs_dict = self.read_obs(np.asarray(timestep.observation))
        reward = self.read_reward(reward, np.asarray(timestep.reward))
        done = torch.tensor(np.asarray(
            timestep.step_type == self.lib.types.StepType.LAST))

        self._is_done = done

        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        tensordict_out.update(state_dict)

        return tensordict_out

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        self.key, *keys = jax.random.split(self.key, self.numel() + 1)
        state, timestep = jax.vmap(self._env.reset)(jnp.stack(keys))
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        state_dict = self._encode_state(state)
        obs_dict = self.read_obs(np.asarray(timestep.observation))
        done = torch.zeros(self.batch_size, dtype=torch.bool)

        self._is_done = done

        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.set("done", done)
        tensordict_out.update(state_dict)

        return tensordict_out

    def _reshape(self, x):
        shape, n = self.batch_size, 1
        return jax.tree_util.tree_map(
            lambda x: x.reshape(shape + x.shape[n:]), x
        )

    def _flatten(self, x):
        shape, n = (self.batch_size.numel(),), len(self.batch_size)
        return jax.tree_util.tree_map(
            lambda x: x.reshape(shape + x.shape[n:]), x
        )

    def _encode_state(self, state):
        # TODO: encode state to tensordict
        self._state = state
        return TensorDict({}, batch_size=self.batch_size, device=self.device)

    def _decode_state(self, tensordict):
        # TODO: decode state from tensordict
        return self._state


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
