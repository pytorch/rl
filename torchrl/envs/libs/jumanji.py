from typing import Optional, Dict, Union

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

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
    from jax import numpy as jnp

    _has_jumanji = True
except ImportError as err:
    _has_jumanji = False
    IMPORT_ERR = str(err)


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


def _data_to_spec_transform(state) -> TensorSpec:
    if isinstance(state, torch.Tensor):
        if state.dtype in (torch.float, torch.double, torch.half):
            return NdUnboundedContinuousTensorSpec(
                shape=state.shape, dtype=state.dtype, device=state.device
            )
        else:
            return NdUnboundedDiscreteTensorSpec(
                shape=state.shape, dtype=state.dtype, device=state.device
            )
    elif isinstance(state, TensorDict):
        return CompositeSpec(
            **{key: _data_to_spec_transform(value) for key, value in state.items()}
        )
    else:
        raise TypeError(f"Unsupported state type {type(state)}")


def _jumanji_to_torchrl_data_transform(val, device, batch_size):
    if isinstance(val, (jax.Array, np.ndarray)):
        if isinstance(val, jax.Array):
            val = np.array(val)
        if val.dtype == np.uint16:
            val = val.astype(np.int16)
        if val.dtype == np.uint32:
            val = val.astype(np.int32)
        if val.dtype == np.uint64:
            val = val.astype(np.int64)
        return torch.tensor(val, device=device)
    elif isinstance(val, tuple) and hasattr(val, "_fields"):  # named tuples
        return TensorDict(
            {
                k: _jumanji_to_torchrl_data_transform(v, device, batch_size)
                for k, v in zip(val._fields, val)
            },
            device=device,
            batch_size=batch_size,
        )
    elif hasattr(val, "__dict__"):
        return TensorDict(
            {
                k: _jumanji_to_torchrl_data_transform(v, device, batch_size)
                for k, v in val.__dict__.items()
            },
            device=device,
            batch_size=batch_size,
        )
    else:
        raise TypeError(f"Unsupported data type {type(val)}")


def _torchrl_to_jumanji_state_transform(tensordict: TensorDict, env):
    if isinstance(env, jumanji.environments.games.snake.env.Snake):
        return jumanji.environments.games.snake.types.State(
            key=tensordict.get("key").numpy().astype(np.uint32),
            body_state=tensordict.get("body_state").numpy(),
            head_pos=jumanji.environments.games.snake.types.Position(
                tensordict.get("head_pos").get("row").squeeze(-1).numpy(),
                tensordict.get("head_pos").get("col").squeeze(-1).numpy(),
            ),
            fruit_pos=jumanji.environments.games.snake.types.Position(
                tensordict.get("fruit_pos").get("row").squeeze(-1).numpy(),
                tensordict.get("fruit_pos").get("col").squeeze(-1).numpy(),
            ),
            length=tensordict.get("length").squeeze(-1).numpy(),
            step=tensordict.get("step").squeeze(-1).numpy(),
        )
    if isinstance(env, jumanji.environments.games.connect4.env.Connect4):
        return jumanji.environments.games.connect4.types.State(
            current_player=tensordict.get("current_player").squeeze(-1).numpy(),
            board=tensordict.get("board").numpy(),
        )
    if isinstance(env, jumanji.environments.combinatorial.tsp.env.TSP):
        return jumanji.environments.combinatorial.tsp.types.State(
            problem=tensordict.get("problem").numpy(),
            position=tensordict.get("position").squeeze(-1).numpy(),
            visited_mask=tensordict.get("visited_mask").numpy(),
            order=tensordict.get("order").numpy(),
            num_visited=tensordict.get("num_visited").squeeze(-1).numpy(),
        )
    if isinstance(env, jumanji.environments.combinatorial.knapsack.env.Knapsack):
        return jumanji.environments.combinatorial.knapsack.types.State(
            problem=tensordict.get("problem").numpy(),
            last_item=tensordict.get("last_item").squeeze(-1).numpy(),
            first_item=tensordict.get("first_item").squeeze(-1).numpy(),
            used_mask=tensordict.get("used_mask").numpy(),
            num_steps=tensordict.get("num_steps").squeeze(-1).numpy(),
            remaining_budget=tensordict.get("remaining_budget").squeeze(-1).numpy(),
        )
    if isinstance(env, jumanji.environments.combinatorial.binpack.env.BinPack):

        def get_fields(tensordict, fields):
            return {key: tensordict.get(key).numpy() for key in fields}

        def get_squeezed_fields(tensordict, fields):
            return {key: tensordict.get(key).squeeze(-1).numpy() for key in fields}

        return jumanji.environments.combinatorial.binpack.types.State(
            container=jumanji.environments.combinatorial.binpack.space.Space(
                **get_squeezed_fields(
                    tensordict.get("container"), ["x1", "x2", "y1", "y2", "z1", "z2"]
                )
            ),
            ems=jumanji.environments.combinatorial.binpack.space.Space(
                **get_fields(
                    tensordict.get("ems"), ["x1", "x2", "y1", "y2", "z1", "z2"]
                )
            ),
            ems_mask=tensordict.get("ems_mask").numpy(),
            items=jumanji.environments.combinatorial.binpack.types.Item(
                **get_fields(tensordict.get("items"), ["x_len", "y_len", "z_len"])
            ),
            items_mask=tensordict.get("items_mask").numpy(),
            items_placed=tensordict.get("items_placed").numpy(),
            items_location=jumanji.environments.combinatorial.binpack.types.Location(
                **get_fields(tensordict.get("items_location"), ["x", "y", "z"])
            ),
            action_mask=tensordict.get("action_mask").numpy(),
            sorted_ems_indexes=tensordict.get("sorted_ems_indexes").numpy(),
            key=tensordict.get("key").numpy(),
        )
    if isinstance(env, jumanji.environments.combinatorial.routing.env.Routing):
        return jumanji.environments.combinatorial.routing.types.State(
            key=tensordict.get("key").numpy().astype(np.uint32),
            grid=tensordict.get("grid").numpy(),
            step=tensordict.get("step").squeeze(-1).numpy(),
            finished_agents=tensordict.get("finished_agents").numpy(),
        )
    raise NotImplementedError(f"Unsupported Jumanji environment {type(env)}")


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

    def _make_state_spec(self, env) -> TensorSpec:
        # generate a sample state object to build state spec from.
        key = jax.random.PRNGKey(0)
        state, _ = env.reset(key)

        state_dict = _jumanji_to_torchrl_data_transform(
            state, self.device, batch_size=()
        )
        state_spec = _data_to_spec_transform(state_dict)
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
        self._input_spec = self._make_input_spec(env)
        self._observation_spec = self._make_observation_spec(env)
        self._reward_spec = self._make_reward_spec(env)

        state_spec = self._make_state_spec(env)
        self._input_spec["state"] = state_spec
        self._observation_spec["state"] = state_spec

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
        state = _jumanji_to_torchrl_data_transform(
            state, device=self.device, batch_size=self.batch_size
        )
        state = self.input_spec["state"].encode(state)
        return state

    def read_obs(self, obs):
        obs = _jumanji_to_torchrl_data_transform(
            obs, device=self.device, batch_size=self.batch_size
        )
        return super().read_obs(obs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:

        state = _torchrl_to_jumanji_state_transform(tensordict.get("state"), self._env)
        action = self.read_action(tensordict.get("action"))
        reward = self.reward_spec.zero(self.batch_size)

        state = self._flatten(state)
        action = self._flatten(action)
        state, timestep = jax.vmap(self._env.step)(state, action)
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
        reward = self.read_reward(reward, np.asarray(timestep.reward))
        done = torch.tensor(
            np.asarray(timestep.step_type == self.lib.types.StepType.LAST)
        )

        self._is_done = done

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

        self.key, *keys = jax.random.split(self.key, self.numel() + 1)
        state, timestep = jax.vmap(self._env.reset)(jnp.stack(keys))
        state = self._reshape(state)
        timestep = self._reshape(timestep)

        state_dict = self.read_state(state)
        obs_dict = self.read_obs(timestep.observation)
        done = torch.zeros(self.batch_size, dtype=torch.bool)

        self._is_done = done

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
