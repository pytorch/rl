import dataclasses
from typing import Dict, Optional, Union

import numpy as np
import torch
from tensordict.tensordict import make_tensordict, TensorDict, TensorDictBase

from torchrl.data import (
    CompositeSpec,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    NdUnboundedDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs import GymLikeEnv

try:
    import jax
    import jumanji
    from jax import numpy as jnp

    _has_jumanji = True
except ImportError as err:
    _has_jumanji = False
    IMPORT_ERR = str(err)


def _ndarray_to_tensor(value: Union["jnp.ndarray", np.ndarray], device) -> torch.Tensor:
    # tensor doesn't support conversion from jnp.ndarray.
    if isinstance(value, jnp.ndarray):
        value = np.asarray(value)
    # tensor doesn't support unsigned dtypes.
    if value.dtype == np.uint16:
        value = value.astype(np.int16)
    elif value.dtype == np.uint32:
        value = value.astype(np.int32)
    elif value.dtype == np.uint64:
        value = value.astype(np.int64)
    # convert to tensor.
    return torch.tensor(value).to(device)


def _object_to_tensordict(obj: Union, device, batch_size) -> TensorDictBase:
    """Converts a namedtuple or a dataclass to a TensorDict."""
    t = {}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # named tuple
        _iter = obj._fields
    elif dataclasses.is_dataclass(obj):
        _iter = (field.name for field in dataclasses.fields(obj))
    else:
        raise NotImplementedError(f"unsupported data type {type(obj)}")
    for name in _iter:
        value = getattr(obj, name)
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            t[name] = _ndarray_to_tensor(value, device=device)
        else:
            t[name] = _object_to_tensordict(value, device, batch_size)
    return make_tensordict(**t, device=device, batch_size=batch_size)


def _tensordict_to_object(tensordict: TensorDictBase, object_example):
    """Converts a TensorDict to a namedtuple or a dataclass."""
    object_type = type(object_example)
    t = {}
    for name in tensordict.keys():
        value = tensordict[name]
        if isinstance(value, TensorDictBase):
            t[name] = _tensordict_to_object(value, getattr(object_example, name))
        else:
            example = getattr(object_example, name)
            t[name] = (
                value.detach().numpy().reshape(example.shape).astype(example.dtype)
            )
    return object_type(**t)


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
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            shape=spec.shape,
            minimum=np.asarray(spec.minimum),
            maximum=np.asarray(spec.maximum),
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


def _torchrl_data_to_spec_transform(data) -> TensorSpec:
    if isinstance(data, torch.Tensor):
        if data.dtype in (torch.float, torch.double, torch.half):
            return NdUnboundedContinuousTensorSpec(
                shape=data.shape, dtype=data.dtype, device=data.device
            )
        else:
            return NdUnboundedDiscreteTensorSpec(
                shape=data.shape, dtype=data.dtype, device=data.device
            )
    elif isinstance(data, TensorDict):
        return CompositeSpec(
            **{
                key: _torchrl_data_to_spec_transform(value)
                for key, value in data.items()
            }
        )
    else:
        raise TypeError(f"Unsupported data type {type(data)}")


class JumanjiWrapper(GymLikeEnv):
    """Jumanji environment wrapper.

    Examples:
        >>> env = jumanju.make("Snake-6x6-v0")
        >>> env = JumanjiWrapper(env)
        >>> td0 = env.reset()
        >>> print(td0)
        >>> td1 = env.rand_step(td0)
        >>> print(td1)
        >>> print(env.available_envs)
    """

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

    def _make_state_example(self, env):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state, _ = jax.vmap(env.reset)(jnp.stack(keys))
        state = self._reshape(state)
        return state

    def _make_state_spec(self, env) -> TensorSpec:
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _torchrl_data_to_spec_transform(state_dict)
        return state_spec

    def _make_input_spec(self, env) -> TensorSpec:
        return CompositeSpec(
            action=_jumanji_to_torchrl_spec_transform(
                env.action_spec(), device=self.device
            ),
        )

    def _make_observation_spec(self, env) -> TensorSpec:
        spec = env.observation_spec()
        new_spec = _jumanji_to_torchrl_spec_transform(spec, device=self.device)
        if isinstance(spec, jumanji.specs.Array):
            return CompositeSpec(observation=new_spec)
        elif isinstance(spec, jumanji.specs.Spec):
            return CompositeSpec(**{k: v for k, v in new_spec.items()})
        else:
            raise TypeError(f"Unsupported spec type {type(spec)}")

    def _make_reward_spec(self, env) -> TensorSpec:
        return _jumanji_to_torchrl_spec_transform(env.reward_spec(), device=self.device)

    def _make_specs(self, env: "jumanji.env.Environment") -> None:  # noqa: F821

        # extract spec from jumanji definition
        self._input_spec = self._make_input_spec(env)
        self._observation_spec = self._make_observation_spec(env)
        self._reward_spec = self._make_reward_spec(env)

        # extract state spec from instance
        self._state_spec = self._make_state_spec(env)
        self._input_spec["state"] = self._state_spec

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
        return self._state_spec.encode(state_dict)

    def read_obs(self, obs):
        if isinstance(obs, (list, jnp.ndarray, np.ndarray)):
            obs_dict = _ndarray_to_tensor(obs, self.device)
        else:
            obs_dict = _object_to_tensordict(obs, self.device, self.batch_size)
        return super().read_obs(obs_dict)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:

        # prepare inputs
        state = _tensordict_to_object(tensordict.get("state"), self._state_example)
        action = self.read_action(tensordict.get("action"))
        reward = self.reward_spec.zero(self.batch_size)

        # flatten batch size into vector
        state = self._flatten(state)
        action = self._flatten(action)

        # jax vectorizing map on env.step
        state, timestep = jax.vmap(self._env.step)(state, action)

        # reshape batch size from vector
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        # collect outputs
        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
        reward = self.read_reward(reward, np.asarray(timestep.reward))
        done = torch.tensor(
            np.asarray(timestep.step_type == self.lib.types.StepType.LAST)
        )

        self._is_done = done

        # build results
        tensordict_out = TensorDict(
            source=obs_dict,
            batch_size=tensordict.batch_size,
            device=self.device,
        )
        tensordict_out.set("reward", reward)
        tensordict_out.set("done", done)
        tensordict_out["state"] = state_dict

        return tensordict_out

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:

        # generate random keys
        self.key, *keys = jax.random.split(self.key, self.numel() + 1)

        # jax vectorizing map on env.reset
        state, timestep = jax.vmap(self._env.reset)(jnp.stack(keys))

        # reshape batch size from vector
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        # collect outputs
        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
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

    def _reshape(self, x):
        shape, n = self.batch_size, 1
        return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[n:]), x)

    def _flatten(self, x):
        shape, n = (self.batch_size.numel(),), len(self.batch_size)
        return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[n:]), x)


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
                f" {self.git_url}. (Original error message during import: {IMPORT_ERR})."
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
