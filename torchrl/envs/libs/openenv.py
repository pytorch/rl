# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import inspect
import json
import warnings
from collections.abc import Callable
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack

from torchrl.data.tensor_specs import Categorical, Composite, NonTensor, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

_has_openenv = importlib.util.find_spec("openenv") is not None

if _has_openenv:
    from openenv import AutoAction, AutoEnv  # type: ignore[import-not-found]
else:
    AutoAction = None  # type: ignore[assignment]
    AutoEnv = None  # type: ignore[assignment]

__all__ = ["OpenEnvEnv", "OpenEnvWrapper"]


ActionAdapter = Callable[[Any], Any]
ObservationAdapter = Callable[[Any], Any]


def _normalize_batch_size(
    batch_size: int | list[int] | tuple[int, ...] | torch.Size | None,
    *,
    default: torch.Size,
) -> torch.Size:
    if batch_size is None:
        return default
    if isinstance(batch_size, int):
        return torch.Size([batch_size])
    return torch.Size(batch_size)


def _example_from_cls(cls: type | None) -> Any:
    if cls is None:
        return None
    for method in ("model_construct", "construct"):
        constructor = getattr(cls, method, None)
        if constructor is not None:
            return constructor()
    return None


def _object_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _unwrap_nontensor(value: Any) -> Any:
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, NonTensorStack):
        return value.tolist()
    if type(value).__name__ == "LinkedList":
        return [_unwrap_nontensor(item) for item in value]
    return value


def _squeeze_singleton(value: Any) -> Any:
    value = _unwrap_nontensor(value)
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = _unwrap_nontensor(value[0])
    return value


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    value = _unwrap_nontensor(value)
    if isinstance(value, dict):
        return value.get(name, default)
    value_dict = _object_to_dict(value)
    if isinstance(value_dict, dict):
        return value_dict.get(name, default)
    return getattr(value, name, default)


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _single_model_field(cls: type | None) -> str | None:
    if cls is None:
        return None
    fields = getattr(cls, "model_fields", None)
    if fields is None:
        fields = getattr(cls, "__fields__", None)
    if fields and len(fields) == 1:
        return next(iter(fields))
    return None


def _coerce_action(
    action: Any,
    *,
    action_cls: type | None,
    action_adapter: ActionAdapter | None,
) -> Any:
    action = _squeeze_singleton(action)
    if isinstance(action, TensorDictBase):
        action = action.to_dict()
    if action_adapter is not None:
        action = action_adapter(action)
        action = _squeeze_singleton(action)
    action = _maybe_json_loads(action)
    if action_cls is None:
        return action
    if isinstance(action, action_cls):
        return action
    if isinstance(action, dict):
        return action_cls(**action)
    action_field = _single_model_field(action_cls)
    if action_field is not None:
        return action_cls(**{action_field: action})
    return action


def _format_object(
    value: Any,
    *,
    return_object_dict: bool,
    adapter: ObservationAdapter | None = None,
) -> Any:
    if adapter is not None:
        value = adapter(value)
    if return_object_dict:
        value = _object_to_dict(value)
    return value


def _extract_observation(result: Any) -> Any:
    return getattr(result, "observation", result)


def _extract_reward_done(result: Any, observation: Any) -> tuple[Any, Any, Any, Any]:
    reward = getattr(result, "reward", None)
    if reward is None:
        reward = _get_field(observation, "reward", None)
    if reward is None:
        reward = 0.0

    result_done = getattr(result, "done", None)
    obs_done = _get_field(observation, "done", None)
    terminated = getattr(result, "terminated", None)
    if terminated is None:
        terminated = _get_field(observation, "terminated", None)
    truncated = getattr(result, "truncated", None)
    if truncated is None:
        truncated = _get_field(observation, "truncated", None)

    if result_done is not None:
        done = result_done
    elif obs_done is not None:
        done = obs_done
    elif terminated is not None or truncated is not None:
        done = bool(terminated) or bool(truncated)
    else:
        done = False

    if terminated is None:
        terminated = bool(done) and not bool(truncated)
    if truncated is None:
        truncated = False
    return reward, done, terminated, truncated


def _ensure_sync_result(result: Any, method_name: str) -> Any:
    if inspect.isawaitable(result):
        raise RuntimeError(
            f"OpenEnv returned an awaitable from {method_name}(). Pass a sync "
            "client or keep sync=True so the OpenEnv client is converted with sync()."
        )
    return result


def _auto_action_from_env(env_name: str, env_kwargs: dict[str, Any]) -> type:
    try:
        parameters = inspect.signature(AutoAction.from_env).parameters
    except (TypeError, ValueError):
        action_kwargs = {}
    else:
        action_kwargs = {
            key: value for key, value in env_kwargs.items() if key in parameters
        }
    return AutoAction.from_env(env_name, **action_kwargs)


def _as_reward_tensor(
    reward: Any, *, batch_size: torch.Size, device: torch.device | None
) -> torch.Tensor:
    return torch.as_tensor(reward, dtype=torch.float32, device=device).view(
        *batch_size, 1
    )


def _as_done_tensor(
    value: Any, *, batch_size: torch.Size, device: torch.device | None
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool, device=device).view(*batch_size, 1)


class OpenEnvWrapper(_EnvWrapper):
    """OpenEnv environment wrapper.

    This wrapper exposes a sync OpenEnv client through the regular TorchRL
    environment API. Observations, actions and optional state are represented as
    non-tensor leaves, while reward and done signals are regular tensors.

    Args:
        env: OpenEnv client instance to wrap. Async OpenEnv clients exposing a
            ``sync()`` method are converted to sync clients by default.

    Keyword Args:
        action_cls (type, optional): Pydantic-style OpenEnv action class. Dict
            actions are expanded into this class, and scalar actions are mapped
            to its only field when it has exactly one field.
        observation_cls (type, optional): Optional class used only to build
            example observation data for specs.
        action_adapter (Callable[[Any], Any], optional): Callable applied to raw
            TorchRL actions before ``action_cls`` coercion.
        observation_adapter (Callable[[Any], Any], optional): Callable applied to
            OpenEnv observations before storing them in the tensordict.
        return_observation_dict (bool, optional): If ``True``, converts
            Pydantic-like observations and states to dictionaries. Defaults to
            ``False``.
        include_state (bool, optional): If ``True``, calls ``env.state()`` after
            reset and step and stores the result under ``"state"``. Defaults to
            ``False``.
        sync (bool, optional): If ``True``, calls ``env.sync()`` when available.
            Defaults to ``True``.
        device (torch.device | str, optional): Tensor device.
        batch_size (torch.Size, optional): Must be empty. Use ``SerialEnv`` or
            ``ParallelEnv`` to run multiple OpenEnv clients.
        allow_done_after_reset (bool, optional): Whether done after reset is
            tolerated. Defaults to ``False``.

    Examples:
        >>> from dataclasses import dataclass
        >>> from torchrl.envs.libs.openenv import OpenEnvWrapper
        >>> @dataclass
        ... class Result:
        ...     observation: str
        ...     reward: float = 0.0
        ...     done: bool = False
        >>> class CounterEnv:
        ...     def __init__(self):
        ...         self.count = 0
        ...     def reset(self):
        ...         self.count = 0
        ...         return Result("start")
        ...     def step(self, action):
        ...         self.count += 1
        ...         return Result(f"{action}:{self.count}", reward=1.0, done=True)
        >>> env = OpenEnvWrapper(env=CounterEnv())
        >>> td = env.reset()
        >>> td["action"] = "answer"
        >>> env.step(td)["next", "reward"].shape
        torch.Size([1])
    """

    git_url = "https://github.com/huggingface/openenv"
    libname = "openenv"

    @_classproperty
    def available_envs(cls) -> list[str]:
        # OpenEnv discovery is registry/hub based; no stable local list exists.
        return []

    def __init__(
        self,
        env: Any,
        *,
        action_cls: type | None = None,
        observation_cls: type | None = None,
        action_adapter: ActionAdapter | None = None,
        observation_adapter: ObservationAdapter | None = None,
        return_observation_dict: bool = False,
        include_state: bool = False,
        sync: bool = True,
        device: torch.device | str | None = None,
        batch_size: torch.Size | tuple[int, ...] | list[int] | None = None,
        allow_done_after_reset: bool = False,
        **kwargs: Any,
    ) -> None:
        batch_size = _normalize_batch_size(batch_size, default=torch.Size([]))
        if batch_size != torch.Size([]):
            raise ValueError(
                "OpenEnvWrapper only supports an empty batch size. Use SerialEnv "
                "or ParallelEnv to run multiple OpenEnv clients."
            )
        self._action_cls = action_cls
        self._observation_cls = observation_cls
        self._action_adapter = action_adapter
        self._observation_adapter = observation_adapter
        self._return_observation_dict = return_observation_dict
        self._include_state = include_state
        self._sync_env = sync
        super().__init__(
            env=env,
            device=device,
            batch_size=batch_size,
            allow_done_after_reset=allow_done_after_reset,
            **kwargs,
        )

    def _check_kwargs(self, kwargs: dict) -> None:
        if kwargs.get("env") is None:
            raise TypeError("OpenEnvWrapper requires a non-null 'env' argument.")

    def _build_env(self, env: Any, **_: Any) -> Any:
        if self._sync_env and hasattr(env, "sync") and callable(env.sync):
            env = env.sync()
        return env

    def _init_env(self) -> None:
        connect = getattr(self._env, "connect", None)
        if callable(connect):
            _ensure_sync_result(connect(), "connect")

    def _make_specs(self, env: Any) -> None:  # noqa: ARG002
        obs_example = _example_from_cls(self._observation_cls)
        act_example = _example_from_cls(self._action_cls)
        self.full_observation_spec = Composite(
            observation=NonTensor(
                shape=self.batch_size, device=self.device, example_data=obs_example
            ),
            shape=self.batch_size,
            device=self.device,
        )
        if self._include_state:
            self.full_observation_spec.set(
                "state",
                NonTensor(shape=self.batch_size, device=self.device),
            )
        self.full_action_spec = Composite(
            action=NonTensor(
                shape=self.batch_size, device=self.device, example_data=act_example
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.full_state_spec = Composite(shape=self.batch_size, device=self.device)
        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(*self.batch_size, 1), dtype=torch.float32, device=self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        done_leaf = Categorical(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_leaf.clone(),
            terminated=done_leaf.clone(),
            truncated=done_leaf.clone(),
            shape=self.batch_size,
            device=self.device,
        )

    def close(self, *, raise_if_closed: bool = True) -> None:
        self.is_closed = True
        for method_name in ("close", "disconnect"):
            close_fn = getattr(self._env, method_name, None)
            if callable(close_fn):
                _ensure_sync_result(close_fn(), method_name)
                return
        super().close(raise_if_closed=raise_if_closed)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        for method in ("seed", "set_seed"):
            setter = getattr(self._env, method, None)
            if callable(setter):
                _ensure_sync_result(setter(seed), method)
                return
        warnings.warn("OpenEnv client does not expose seed() or set_seed().")

    def _wrap_nontensor(self, value: Any) -> NonTensorData:
        value = _format_object(
            value,
            return_object_dict=self._return_observation_dict,
            adapter=self._observation_adapter,
        )
        return NonTensorData(value, batch_size=self.batch_size, device=self.device)

    def _get_state(self) -> NonTensorData | None:
        if not self._include_state:
            return None
        state_fn = getattr(self._env, "state", None)
        if not callable(state_fn):
            return None
        state = _ensure_sync_result(state_fn(), "state")
        state = _format_object(
            state,
            return_object_dict=self._return_observation_dict,
            adapter=None,
        )
        return NonTensorData(state, batch_size=self.batch_size, device=self.device)

    def _make_td(self, result: Any, *, include_reward_done: bool = True) -> TensorDict:
        raw_observation = _extract_observation(result)
        data = {
            "observation": self._wrap_nontensor(raw_observation),
        }
        if include_reward_done:
            reward, done, terminated, truncated = _extract_reward_done(
                result, raw_observation
            )
            data.update(
                {
                    "reward": _as_reward_tensor(
                        reward, batch_size=self.batch_size, device=self.device
                    ),
                    "done": _as_done_tensor(
                        done, batch_size=self.batch_size, device=self.device
                    ),
                    "terminated": _as_done_tensor(
                        terminated, batch_size=self.batch_size, device=self.device
                    ),
                    "truncated": _as_done_tensor(
                        truncated, batch_size=self.batch_size, device=self.device
                    ),
                }
            )
        state = self._get_state()
        if state is not None:
            data["state"] = state
        return TensorDict(data, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs: Any):
        result = _ensure_sync_result(self._env.reset(), "reset")
        return self._make_td(result, include_reward_done=False)

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        action = tensordict.get("action", None)
        action = _coerce_action(
            action, action_cls=self._action_cls, action_adapter=self._action_adapter
        )
        result = _ensure_sync_result(self._env.step(action), "step")
        return self._make_td(result)


class OpenEnvEnv(OpenEnvWrapper):
    """OpenEnv environment built with OpenEnv auto-discovery.

    See also :class:`~torchrl.trainers.algorithms.configs.OpenEnvEnvConfig`.

    Args:
        env_name (str): OpenEnv environment name, package name variant, or hub
            repository identifier accepted by ``openenv.AutoEnv.from_env``.

    Keyword Args:
        auto_action (bool, optional): If ``True`` and ``action_cls`` is not
            provided, uses ``openenv.AutoAction.from_env`` to discover the action
            class. Defaults to ``True``.
        env_kwargs (dict, optional): Extra keyword arguments forwarded to
            ``AutoEnv.from_env``.
        action_cls (type, optional): Explicit action class.
        observation_cls (type, optional): Optional class used only to build spec
            example data.
        action_adapter (Callable[[Any], Any], optional): Callable applied before
            action-class coercion.
        return_observation_dict (bool, optional): Convert Pydantic-like
            observations and states to dictionaries. Defaults to ``False``.
        include_state (bool, optional): Include ``env.state()`` in outputs when
            available. Defaults to ``False``.
        sync (bool, optional): Convert async clients with ``sync()``. Defaults to
            ``True``.

    Examples:
        >>> from torchrl.envs.libs.openenv import OpenEnvEnv
        >>> env = OpenEnvEnv("my-openenv", env_kwargs={"skip_install": True})  # doctest: +SKIP
        >>> env.reset()  # doctest: +SKIP
    """

    def __init__(
        self,
        env_name: str,
        *,
        auto_action: bool = True,
        env_kwargs: dict[str, Any] | None = None,
        action_cls: type | None = None,
        observation_cls: type | None = None,
        action_adapter: ActionAdapter | None = None,
        observation_adapter: ObservationAdapter | None = None,
        return_observation_dict: bool = False,
        include_state: bool = False,
        sync: bool = True,
        **kwargs: Any,
    ) -> None:
        if not _has_openenv:
            raise ImportError(
                "openenv python package was not found. Install it with "
                "`pip install openenv-core` or `pip install torchrl[openenv]`."
            )
        if env_kwargs is None:
            env_kwargs = {}
        env = AutoEnv.from_env(env_name, **env_kwargs)
        if auto_action and action_cls is None:
            action_cls = _auto_action_from_env(env_name, env_kwargs)
        self.env_name = env_name
        super().__init__(
            env=env,
            action_cls=action_cls,
            observation_cls=observation_cls,
            action_adapter=action_adapter,
            observation_adapter=observation_adapter,
            return_observation_dict=return_observation_dict,
            include_state=include_state,
            sync=sync,
            **kwargs,
        )
