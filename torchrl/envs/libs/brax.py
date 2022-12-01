from typing import Dict, Optional, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, NdBoundedTensorSpec, NdUnboundedContinuousTensorSpec
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.jumanji import _object_to_tensordict, _tensordict_to_object, _torchrl_data_to_spec_transform

try:
    import brax
    import brax.envs
    import brax.io.torch
    from brax import jumpy as jp
    import jax
    import numpy as np

    _has_brax = True
except ImportError as err:
    _has_brax = False
    IMPORT_ERR = str(err)


class BraxWrapper(_EnvWrapper):
    """Google Brax environment wrapper.

    Examples:
        >>> env = brax.envs.get_environment("ant")
        >>> env = BraxWrapper(env)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    git_url = "https://github.com/google/brax"
    
    @property
    def lib(self):
        return brax

    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, brax.envs.env.Env):
            raise TypeError("env is not of type 'brax.envs.env.Env'.")

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

    def _make_state_spec(self, env: "brax.envs.env.Env"):
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _torchrl_data_to_spec_transform(state_dict)
        return state_spec

    def _make_specs(self, env: "brax.envs.env.Env") -> None:  # noqa: F821
        self._input_spec = CompositeSpec(
            action=NdBoundedTensorSpec(
                minimum=-1,
                maximum=1,
                shape=(env.action_size,),
                device=self.device
            )
        )
        self._reward_spec = NdUnboundedContinuousTensorSpec(
            shape=(),
            device=self.device
        )
        self._observation_spec = CompositeSpec(
            observation=NdUnboundedContinuousTensorSpec(
                shape=(env.observation_size,),
                device=self.device
            )
        )
        # extract state spec from instance
        self._state_spec = self._make_state_spec(env)
        self._input_spec["state"] = self._state_spec

    def _make_state_example(self):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))
        state = self._reshape(state)
        return state

    def _init_env(self) -> Optional[int]:
        self._key = None
        self._vmap_jit_env_reset = jp.vmap(jax.jit(self._env.reset))
        self._vmap_jit_env_step = jp.vmap(jax.jit(self._env.step))
        self._state_example = self._make_state_example()

    def _set_seed(self, seed: int):
        if seed is None:
            raise Exception("Brax requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        self._key, *keys = jax.random.split(self._key, 1 + self.numel())
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))
        state = self._reshape(state)
        state = _object_to_tensordict(state, self.device, self.batch_size)

        tensordict_out = TensorDict(
            source={
                "observation": state.get("obs"),
                "reward": state.get("reward"),
                "done": state.get("done").bool(),
                "state": state,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out


    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:

        state = _tensordict_to_object(tensordict.get("state"), self._state_example)
        action = tensordict.get("action").numpy()

        state = self._flatten(state)
        action = self._flatten(action)
        state = self._vmap_jit_env_step(state, action)
        state = self._reshape(state)
        state = _object_to_tensordict(state, self.device, self.batch_size)

        tensordict_out = TensorDict(
            source={
                "observation": state.get("obs"),
                "reward": state.get("reward"),
                "done": state.get("done").bool(),
                "state": state,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _reshape(self, x):
        shape, n = self.batch_size, 1
        return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[n:]), x)

    def _flatten(self, x):
        shape, n = (self.batch_size.numel(),), len(self.batch_size)
        return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[n:]), x)


class BraxEnv(BraxWrapper):
    """Google Brax environment wrapper.

    Examples:
        >>> env = BraxEnv(env_name="ant")
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
    ) -> "brax.envs.env.Env":
        if not _has_brax:
            raise RuntimeError(
                f"brax not found, unable to create {env_name}. "
                f"Consider downloading and installing brax from"
                f" {self.git_url}"
            )
        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        assert not kwargs
        self.wrapper_frame_skip = 1
        env = self.lib.envs.get_environment(env_name, **kwargs)
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"
