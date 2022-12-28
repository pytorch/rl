from typing import Dict, Optional, Union

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    CompositeSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper

try:
    import brax
    import brax.envs
    import jax
    from torchrl.envs.libs.jax_utils import (
        _extract_spec,
        _ndarray_to_tensor,
        _object_to_tensordict,
        _tensor_to_ndarray,
        _tensordict_to_object,
        _tree_flatten,
        _tree_reshape,
    )

    _has_brax = True
    IMPORT_ERR = ""
except ImportError as err:
    _has_brax = False
    IMPORT_ERR = str(err)


def _get_envs():
    if not _has_brax:
        return []
    return list(brax.envs._envs.keys())


class BraxWrapper(_EnvWrapper):
    """Google Brax environment wrapper.

    Examples:
        >>> env = brax.envs.get_environment("ant")
        >>> env = BraxWrapper(env)
        >>> env.set_seed(0)
        >>> td = env.reset()
        >>> td["action"] = env.action_spec.rand()
        >>> td = env.step(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([8]), dtype=torch.float32),
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                next: TensorDict(
                    fields={
                        observation: Tensor(torch.Size([87]), dtype=torch.float32)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(torch.Size([87]), dtype=torch.float32),
                reward: Tensor(torch.Size([1]), dtype=torch.float32),
                state: TensorDict(...)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> print(env.available_envs)
        ['acrobot', 'ant', 'fast', 'fetch', ...]
    """

    git_url = "https://github.com/google/brax"
    available_envs = _get_envs()
    libname = "brax"

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
        requires_grad: bool = False,
        camera_id: Union[int, str] = 0,
        **kwargs,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.requires_grad = requires_grad

        if from_pixels:
            raise NotImplementedError("TODO")
        return env

    def _make_state_spec(self, env: "brax.envs.env.Env"):
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict)
        return state_spec

    def _make_specs(self, env: "brax.envs.env.Env") -> None:  # noqa: F821
        self.input_spec = CompositeSpec(
            action=NdBoundedTensorSpec(
                minimum=-1, maximum=1, shape=(env.action_size,), device=self.device
            )
        )
        self.reward_spec = NdUnboundedContinuousTensorSpec(
            shape=[
                1,
            ],
            device=self.device,
        )
        self.observation_spec = CompositeSpec(
            observation=NdUnboundedContinuousTensorSpec(
                shape=(env.observation_size,), device=self.device
            )
        )
        # extract state spec from instance
        self.state_spec = self._make_state_spec(env)
        self.input_spec["state"] = self.state_spec

    def _make_state_example(self):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))
        state = _tree_reshape(state, self.batch_size)
        return state

    def _init_env(self) -> Optional[int]:
        self._key = None
        self._vmap_jit_env_reset = jax.vmap(jax.jit(self._env.reset))
        self._vmap_jit_env_step = jax.vmap(jax.jit(self._env.step))
        self._state_example = self._make_state_example()

    def _set_seed(self, seed: int):
        if seed is None:
            raise Exception("Brax requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        # generate random keys
        self._key, *keys = jax.random.split(self._key, 1 + self.numel())

        # call env reset with jit and vmap
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))

        # reshape batch size
        state = _tree_reshape(state, self.batch_size)
        state = _object_to_tensordict(state, self.device, self.batch_size)

        # build result
        tensordict_out = TensorDict(
            source={
                "observation": state.get("obs"),
                "reward": state.get("reward"),
                "done": state.get("done").bool(),
                "state": state,
            },
            batch_size=self.batch_size,
            device=self.device,
            _run_checks=False,
        )
        return tensordict_out

    def _step_without_grad(self, tensordict: TensorDictBase):

        # convert tensors to ndarrays
        state = _tensordict_to_object(tensordict.get("state"), self._state_example)
        action = _tensor_to_ndarray(tensordict.get("action"))

        # flatten batch size
        state = _tree_flatten(state, self.batch_size)
        action = _tree_flatten(action, self.batch_size)

        # call env step with jit and vmap
        next_state = self._vmap_jit_env_step(state, action)

        # reshape batch size and convert ndarrays to tensors
        next_state = _tree_reshape(next_state, self.batch_size)
        next_state = _object_to_tensordict(next_state, self.device, self.batch_size)

        # build result
        tensordict_out = TensorDict(
            source={
                "observation": next_state.get("obs"),
                "reward": next_state.get("reward"),
                "done": next_state.get("done").bool(),
                "state": next_state,
            },
            batch_size=self.batch_size,
            device=self.device,
            _run_checks=False,
        )
        return tensordict_out

    def _step_with_grad(self, tensordict: TensorDictBase):

        # convert tensors to ndarrays
        action = tensordict.get("action")
        state = tensordict.get("state")
        qp_keys = list(state.get("qp").keys())
        qp_values = list(state.get("qp").values())

        # call env step with autograd function
        next_state_nograd, next_obs, next_reward, *next_qp_values = _BraxEnvStep.apply(
            self, state, action, *qp_values
        )

        # extract done values
        next_done = next_state_nograd["done"].bool()

        # merge with tensors with grad function
        next_state = next_state_nograd
        next_state["obs"] = next_obs
        next_state["reward"] = next_reward
        next_state["qp"].update(dict(zip(qp_keys, next_qp_values)))

        # build result
        tensordict_out = TensorDict(
            source={
                "observation": next_obs,
                "reward": next_reward,
                "done": next_done,
                "state": next_state,
            },
            batch_size=self.batch_size,
            device=self.device,
            _run_checks=False,
        )
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:

        if self.requires_grad:
            return self._step_with_grad(tensordict)
        else:
            return self._step_without_grad(tensordict)


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
        requires_grad = kwargs.pop("requires_grad", False)
        assert not kwargs
        self.wrapper_frame_skip = 1
        env = self.lib.envs.get_environment(env_name, **kwargs)
        return super()._build_env(
            env,
            pixels_only=pixels_only,
            from_pixels=from_pixels,
            requires_grad=requires_grad,
        )

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"


class _BraxEnvStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, env: BraxWrapper, state, action, *qp_values):

        # convert tensors to ndarrays
        state = _tensordict_to_object(state, env._state_example)
        action = _tensor_to_ndarray(action)

        # flatten batch size
        state = _tree_flatten(state, env.batch_size)
        action = _tree_flatten(action, env.batch_size)

        # call vjp with jit and vmap
        next_state, vjp_fn = jax.vjp(env._vmap_jit_env_step, state, action)

        # reshape batch size
        next_state = _tree_reshape(next_state, env.batch_size)

        # convert ndarrays to tensors
        next_state = _object_to_tensordict(
            next_state, device=env.device, batch_size=env.batch_size
        )

        # save context
        ctx.vjp_fn = vjp_fn
        ctx.next_state = next_state
        ctx.env = env

        return (
            next_state,  # no gradient
            next_state["obs"],
            next_state["reward"],
            *next_state["qp"].values(),
        )

    @staticmethod
    def backward(ctx, _, grad_next_obs, grad_next_reward, *grad_next_qp_values):

        # build gradient tensordict with zeros in fields with no grad
        grad_next_state = TensorDict(
            source={
                "qp": dict(zip(ctx.next_state["qp"].keys(), grad_next_qp_values)),
                "obs": grad_next_obs,
                "reward": grad_next_reward,
                "done": torch.zeros_like(ctx.next_state["done"]),
                "metrics": {
                    k: torch.zeros_like(v) for k, v in ctx.next_state["metrics"].items()
                },
                "info": {
                    k: torch.zeros_like(v) for k, v in ctx.next_state["info"].items()
                },
            },
            device=ctx.env.device,
            batch_size=ctx.env.batch_size,
            _run_checks=False,
        )

        # convert tensors to ndarrays
        grad_next_state = _tensordict_to_object(grad_next_state, ctx.env._state_example)

        # flatten batch size
        grad_next_state = _tree_flatten(grad_next_state, ctx.env.batch_size)

        # call vjp to get gradients
        grad_state, grad_action = ctx.vjp_fn(grad_next_state)

        # reshape batch size
        grad_state = _tree_reshape(grad_state, ctx.env.batch_size)
        grad_action = _tree_reshape(grad_action, ctx.env.batch_size)

        # convert ndarrays to tensors
        grad_state_qp = _object_to_tensordict(
            grad_state.qp, device=ctx.env.device, batch_size=ctx.env.batch_size
        )
        grad_action = _ndarray_to_tensor(grad_action)

        return (None, None, grad_action, *grad_state_qp.values())
