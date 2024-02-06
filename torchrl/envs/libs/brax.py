# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util

from typing import Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

_has_brax = importlib.util.find_spec("brax") is not None
from torchrl.envs.libs.jax_utils import (
    _extract_spec,
    _ndarray_to_tensor,
    _object_to_tensordict,
    _tensor_to_ndarray,
    _tensordict_to_object,
    _tree_flatten,
    _tree_reshape,
)


def _get_envs():
    if not _has_brax:
        raise ImportError("BRAX is not installed in your virtual environment.")

    import brax.envs

    return list(brax.envs._envs.keys())


class BraxWrapper(_EnvWrapper):
    """Google Brax environment wrapper.

    Brax offers a vectorized and differentiable simulation framework based on Jax.
    TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
    but computational graphs can still be built on top of the simulated trajectories,
    allowing for backpropagation through the rollout.

    GitHub: https://github.com/google/brax

    Paper: https://arxiv.org/abs/2106.13281

    Args:
        env (brax.envs.base.PipelineEnv): the environment to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.DiscreteTensorSpec`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHotTensorSpec`).
            Defaults to ``False``.

    Keyword Args:
        from_pixels (bool, optional): Not yet supported.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            In ``brax``, this indicates the number of vectorized environments.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments availalbe to build

    Examples:
        >>> import brax.envs
        >>> from torchrl.envs import BraxWrapper
        >>> base_env = brax.envs.get_environment("ant")
        >>> env = BraxWrapper(base_env)
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

    To take advante of Brax, one usually executes multiple environments at the
    same time. In the following example, we iteratively test different batch sizes
    and report the execution time for a short rollout:

    Examples:
        >>> from torch.utils.benchmark import Timer
        >>> for batch_size in [4, 16, 128]:
        ...     timer = Timer('''
        ... env.rollout(100)
        ... ''',
        ...     setup=f'''
        ... import brax.envs
        ... from torchrl.envs import BraxWrapper
        ... env = BraxWrapper(brax.envs.get_environment("ant"), batch_size=[{batch_size}])
        ... env.set_seed(0)
        ... env.rollout(2)
        ... ''')
        ...     print(batch_size, timer.timeit(10))
        4
        env.rollout(100)
        setup: [...]
        310.00 ms
        1 measurement, 10 runs , 1 thread

        16
        env.rollout(100)
        setup: [...]
        268.46 ms
        1 measurement, 10 runs , 1 thread

        128
        env.rollout(100)
        setup: [...]
        433.80 ms
        1 measurement, 10 runs , 1 thread

    One can backpropagate through the rollout and optimize the policy directly:

        >>> import brax.envs
        >>> from torchrl.envs import BraxWrapper
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> import torch
        >>>
        >>> env = BraxWrapper(brax.envs.get_environment("ant"), batch_size=[10], requires_grad=True)
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> policy = TensorDictModule(nn.Linear(27, 8), in_keys=["observation"], out_keys=["action"])
        >>>
        >>> td = env.rollout(10, policy)
        >>>
        >>> td["next", "reward"].mean().backward(retain_graph=True)
        >>> print(policy.module.weight.grad.norm())
        tensor(213.8605)

    """

    git_url = "https://github.com/google/brax"

    @_classproperty
    def available_envs(cls):
        if not _has_brax:
            return []
        return list(_get_envs())

    libname = "brax"

    _lib = None
    _jax = None

    @_classproperty
    def lib(cls):
        if cls._lib is not None:
            return cls._lib

        import brax
        import brax.envs

        cls._lib = brax
        return brax

    @_classproperty
    def jax(cls):
        if cls._jax is not None:
            return cls._jax

        import jax

        cls._jax = jax
        return jax

    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        self._categorical_action_encoding = categorical_action_encoding
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        brax = self.lib

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
            raise NotImplementedError(
                "from_pixels=True is not yest supported within BraxWrapper"
            )
        return env

    def _make_state_spec(self, env: "brax.envs.env.Env"):  # noqa: F821
        jax = self.jax

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict).expand(self.batch_size)
        return state_spec

    def _make_specs(self, env: "brax.envs.env.Env") -> None:  # noqa: F821
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(
                *self.batch_size,
                env.action_size,
            ),
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[
                *self.batch_size,
                1,
            ],
            device=self.device,
        )
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(
                    *self.batch_size,
                    env.observation_size,
                ),
                device=self.device,
            ),
            shape=self.batch_size,
        )
        # extract state spec from instance
        state_spec = self._make_state_spec(env)
        self.state_spec["state"] = state_spec
        self.observation_spec["state"] = state_spec.clone()

    def _make_state_example(self):
        jax = self.jax

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, self.batch_size.numel())
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))
        state = _tree_reshape(state, self.batch_size)
        return state

    def _init_env(self) -> Optional[int]:
        jax = self.jax
        self._key = None
        self._vmap_jit_env_reset = jax.vmap(jax.jit(self._env.reset))
        self._vmap_jit_env_step = jax.vmap(jax.jit(self._env.step))
        self._state_example = self._make_state_example()

    def _set_seed(self, seed: int):
        jax = self.jax
        if seed is None:
            raise Exception("Brax requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        jax = self.jax

        # generate random keys
        self._key, *keys = jax.random.split(self._key, 1 + self.numel())

        # call env reset with jit and vmap
        state = self._vmap_jit_env_reset(jax.numpy.stack(keys))

        # reshape batch size
        state = _tree_reshape(state, self.batch_size)
        state = _object_to_tensordict(state, self.device, self.batch_size)

        # build result
        state["reward"] = state.get("reward").view(*self.reward_spec.shape)
        state["done"] = state.get("done").view(*self.reward_spec.shape)
        done = state["done"].bool()
        tensordict_out = TensorDict(
            source={
                "observation": state.get("obs"),
                # "reward": reward,
                "done": done,
                "terminated": done.clone(),
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
        next_state.set("reward", next_state.get("reward").view(self.reward_spec.shape))
        next_state.set("done", next_state.get("done").view(self.reward_spec.shape))
        done = next_state["done"].bool()
        reward = next_state["reward"]
        tensordict_out = TensorDict(
            source={
                "observation": next_state.get("obs"),
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
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
        qp_keys, qp_values = zip(*state.get("pipeline_state").items())

        # call env step with autograd function
        next_state_nograd, next_obs, next_reward, *next_qp_values = _BraxEnvStep.apply(
            self, state, action, *qp_values
        )

        # extract done values: we assume a shape identical to reward
        next_done = next_state_nograd.get("done").view(*self.reward_spec.shape)
        next_reward = next_reward.view(*self.reward_spec.shape)

        # merge with tensors with grad function
        next_state = next_state_nograd
        next_state["obs"] = next_obs
        next_state.set("reward", next_reward)
        next_state.set("done", next_done)
        next_done = next_done.bool()
        next_state.get("pipeline_state").update(dict(zip(qp_keys, next_qp_values)))

        # build result
        tensordict_out = TensorDict(
            source={
                "observation": next_obs,
                "reward": next_reward,
                "done": next_done,
                "terminated": next_done,
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
            out = self._step_with_grad(tensordict)
        else:
            out = self._step_without_grad(tensordict)
        return out


class BraxEnv(BraxWrapper):
    """Google Brax environment wrapper built with the environment name.

    Brax offers a vectorized and differentiable simulation framework based on Jax.
    TorchRL's wrapper incurs some overhead for the jax-to-torch conversion,
    but computational graphs can still be built on top of the simulated trajectories,
    allowing for backpropagation through the rollout.

    GitHub: https://github.com/google/brax

    Paper: https://arxiv.org/abs/2106.13281

    Args:
        env_name (str): the environment name of the env to wrap. Must be part of
            :attr:`~.available_envs`.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.DiscreteTensorSpec`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHotTensorSpec`).
            Defaults to ``False``.

    Keyword Args:
        from_pixels (bool, optional): Not yet supported.
        frame_skip (int, optional): if provided, indicates for how many steps the
            same action is to be repeated. The observation returned will be the
            last observation of the sequence, whereas the reward will be the sum
            of rewards across steps.
        device (torch.device, optional): if provided, the device on which the data
            is to be cast. Defaults to ``torch.device("cpu")``.
        batch_size (torch.Size, optional): the batch size of the environment.
            In ``brax``, this indicates the number of vectorized environments.
            Defaults to ``torch.Size([])``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`~.reset` is called.
            Defaults to ``False``.

    Attributes:
        available_envs: environments availalbe to build

    Examples:
        >>> from torchrl.envs import BraxEnv
        >>> env = BraxEnv("ant")
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

    To take advante of Brax, one usually executes multiple environments at the
    same time. In the following example, we iteratively test different batch sizes
    and report the execution time for a short rollout:

    Examples:
        >>> for batch_size in [4, 16, 128]:
        ...     timer = Timer('''
        ... env.rollout(100)
        ... ''',
        ...     setup=f'''
        ... from torchrl.envs import BraxEnv
        ... env = BraxEnv("ant", batch_size=[{batch_size}])
        ... env.set_seed(0)
        ... env.rollout(2)
        ... ''')
        ...     print(batch_size, timer.timeit(10))
        4
        env.rollout(100)
        setup: [...]
        310.00 ms
        1 measurement, 10 runs , 1 thread

        16
        env.rollout(100)
        setup: [...]
        268.46 ms
        1 measurement, 10 runs , 1 thread

        128
        env.rollout(100)
        setup: [...]
        433.80 ms
        1 measurement, 10 runs , 1 thread

    One can backpropagate through the rollout and optimize the policy directly:

        >>> from torchrl.envs import BraxEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> import torch
        >>>
        >>> env = BraxEnv("ant", batch_size=[10], requires_grad=True)
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> policy = TensorDictModule(nn.Linear(27, 8), in_keys=["observation"], out_keys=["action"])
        >>>
        >>> td = env.rollout(10, policy)
        >>>
        >>> td["next", "reward"].mean().backward(retain_graph=True)
        >>> print(policy.module.weight.grad.norm())
        tensor(213.8605)

    """

    def __init__(self, env_name, **kwargs):
        kwargs["env_name"] = env_name
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "brax.envs.env.Env":  # noqa: F821
        if not _has_brax:
            raise ImportError(
                f"brax not found, unable to create {env_name}. "
                f"Consider downloading and installing brax from"
                f" {self.git_url}"
            )
        from_pixels = kwargs.pop("from_pixels", False)
        pixels_only = kwargs.pop("pixels_only", True)
        requires_grad = kwargs.pop("requires_grad", False)
        if kwargs:
            raise ValueError("kwargs not supported.")
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
    def forward(ctx, env: BraxWrapper, state_td, action_tensor, *qp_values):
        import jax

        # convert tensors to ndarrays
        state_obj = _tensordict_to_object(state_td, env._state_example)
        action_nd = _tensor_to_ndarray(action_tensor)

        # flatten batch size
        state = _tree_flatten(state_obj, env.batch_size)
        action = _tree_flatten(action_nd, env.batch_size)

        # call vjp with jit and vmap
        next_state, vjp_fn = jax.vjp(env._vmap_jit_env_step, state, action)

        # reshape batch size
        next_state_reshape = _tree_reshape(next_state, env.batch_size)

        # convert ndarrays to tensors
        next_state_tensor = _object_to_tensordict(
            next_state_reshape, device=env.device, batch_size=env.batch_size
        )

        # save context
        ctx.vjp_fn = vjp_fn
        ctx.next_state = next_state_tensor
        ctx.env = env

        return (
            next_state_tensor,  # no gradient
            next_state_tensor["obs"],
            next_state_tensor["reward"],
            *next_state_tensor["pipeline_state"].values(),
        )

    @staticmethod
    def backward(ctx, _, grad_next_obs, grad_next_reward, *grad_next_qp_values):

        pipeline_state = dict(
            zip(ctx.next_state.get("pipeline_state").keys(), grad_next_qp_values)
        )
        none_keys = []

        def _make_none(key, val):
            if val is not None:
                return val
            none_keys.append(key)
            return torch.zeros_like(ctx.next_state.get(("pipeline_state", key)))

        pipeline_state = {
            key: _make_none(key, val) for key, val in pipeline_state.items()
        }
        metrics = ctx.next_state.get("metrics", None)
        if metrics is None:
            metrics = {}
        info = ctx.next_state.get("info", None)
        if info is None:
            info = {}
        grad_next_state_td = TensorDict(
            source={
                "pipeline_state": pipeline_state,
                "obs": grad_next_obs,
                "reward": grad_next_reward,
                "done": torch.zeros_like(ctx.next_state.get("done")),
                "metrics": {k: torch.zeros_like(v) for k, v in metrics.items()},
                "info": {k: torch.zeros_like(v) for k, v in info.items()},
            },
            device=ctx.env.device,
            batch_size=ctx.env.batch_size,
        )
        # convert tensors to ndarrays
        grad_next_state_obj = _tensordict_to_object(
            grad_next_state_td, ctx.env._state_example
        )

        # flatten batch size
        grad_next_state_flat = _tree_flatten(grad_next_state_obj, ctx.env.batch_size)

        # call vjp to get gradients
        grad_state, grad_action = ctx.vjp_fn(grad_next_state_flat)

        # reshape batch size
        grad_state = _tree_reshape(grad_state, ctx.env.batch_size)
        grad_action = _tree_reshape(grad_action, ctx.env.batch_size)

        # convert ndarrays to tensors
        grad_state_qp = _object_to_tensordict(
            grad_state.pipeline_state,
            device=ctx.env.device,
            batch_size=ctx.env.batch_size,
        )
        grad_action = _ndarray_to_tensor(grad_action)
        grad_state_qp = {
            key: val if key not in none_keys else None
            for key, val in grad_state_qp.items()
        }
        return (None, None, grad_action, *grad_state_qp.values())
