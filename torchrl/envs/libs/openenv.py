from __future__ import annotations

import importlib.util
import inspect
import warnings
from typing import Any

import torch
from tensordict import NonTensorData, TensorDict, TensorDictBase

from torchrl.data.tensor_specs import Categorical, Composite, NonTensor, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

__all__ = ["OpenEnvWrapper", "OpenEnvEnv"]

_has_openenv = importlib.util.find_spec("openenv") is not None

if _has_openenv:
    from openenv import AutoAction, AutoEnv  # type: ignore
else:
    AutoAction = None  # type: ignore
    AutoEnv = None  # type: ignore


def _unwrap_nontensor(value: Any) -> Any:
    if isinstance(value, NonTensorData):
        return value.data
    return value


def _example_from_cls(cls: type | None) -> Any:
    if cls is None:
        return None
    for method in ("model_construct", "construct"):
        if hasattr(cls, method):
            try:
                return getattr(cls, method)()
            except Exception:
                continue
    return None


class OpenEnvWrapper(_EnvWrapper):
    """OpenEnv environment wrapper.

    Wraps an existing OpenEnv client instance (sync or async) and exposes it
    under the TorchRL environment API.

    Observations and actions are stored as non-tensor data under the
    ``"observation"`` and ``"action"`` keys.

    Args:
        env: An OpenEnv client instance. If the client is async and exposes
            a ``sync()`` method, it will be converted to a sync client by default.

    Keyword Args:
        action_cls (type, optional): Optional class used to build actions from dicts.
        observation_cls (type, optional): Optional class used only to seed example
            data for specs.
        return_observation_dict (bool, optional): If ``True``, attempts to
            convert Pydantic observation objects to dicts. Defaults to ``False``.
        sync (bool, optional): If ``True`` (default), calls ``env.sync()`` when
            available.
        device (torch.device | str, optional): device on which tensors are placed.
        batch_size (torch.Size, optional): batch size. OpenEnv environments are
            single-instance; non-empty batch sizes are not supported.
        allow_done_after_reset (bool, optional): tolerate done right after reset.
    """

    git_url = "https://github.com/openenv/openenv-core"
    libname = "openenv"

    @_classproperty
    def available_envs(cls) -> list[str]:
        # OpenEnv env discovery is remote/hub-based; we do not list here.
        return []

    def __init__(
        self,
        *,
        env: Any,
        action_cls: type | None = None,
        observation_cls: type | None = None,
        return_observation_dict: bool = False,
        sync: bool = True,
        **kwargs,
    ) -> None:
        self._action_cls = action_cls
        self._observation_cls = observation_cls
        self._return_observation_dict = return_observation_dict
        self._sync_env = sync
        self._warned_reward_none = False
        super().__init__(env=env, **kwargs)

    def _check_kwargs(self, kwargs: dict) -> None:
        if "env" not in kwargs:
            raise TypeError("OpenEnvWrapper requires an 'env' argument.")

    def _build_env(self, env, **_) -> Any:
        if self._sync_env and hasattr(env, "sync") and callable(env.sync):
            try:
                env = env.sync()
            except Exception:
                pass
        return env

    def _make_specs(self, env) -> None:  # noqa: ARG002
        if len(self.batch_size):
            raise ValueError(
                "OpenEnvWrapper does not support batched environments. "
                "Use ParallelEnv to create multiple OpenEnv instances."
            )

        obs_example = _example_from_cls(self._observation_cls)
        act_example = _example_from_cls(self._action_cls)

        self.observation_spec = Composite(
            observation=NonTensor(
                shape=self.batch_size, device=self.device, example_data=obs_example
            ),
            shape=self.batch_size,
        )

        self.action_spec = Composite(
            action=NonTensor(
                shape=self.batch_size, device=self.device, example_data=act_example
            ),
            shape=self.batch_size,
        )

        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(*self.batch_size, 1), dtype=torch.float32, device=self.device
            ),
            shape=self.batch_size,
        )

        done_leaf = Categorical(
            n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device
        )
        self.done_spec = Composite(
            done=done_leaf.clone(),
            terminated=done_leaf.clone(),
            truncated=done_leaf.clone(),
            shape=self.batch_size,
        )

    def _init_env(self) -> None:
        connect = getattr(self._env, "connect", None)
        if callable(connect) and not inspect.iscoroutinefunction(connect):
            try:
                connect()
            except Exception:
                pass

    def close(self, *, raise_if_closed: bool = True) -> None:
        self.is_closed = True
        disconnect = getattr(self._env, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass
            return
        super().close(raise_if_closed=raise_if_closed)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        for method in ("seed", "set_seed"):
            setter = getattr(self._env, method, None)
            if callable(setter):
                try:
                    setter(seed)
                    return
                except Exception:
                    warnings.warn("OpenEnvWrapper: seeding failed (best-effort).")
                    return
        warnings.warn("OpenEnvWrapper: seeding is not supported by this client.")

    def _format_observation(self, obs: Any) -> Any:
        if self._return_observation_dict:
            if hasattr(obs, "model_dump"):
                try:
                    return obs.model_dump()
                except Exception:
                    pass
            if hasattr(obs, "dict"):
                try:
                    return obs.dict()
                except Exception:
                    pass
        return obs

    def _format_action(self, action: Any) -> Any:
        action = _unwrap_nontensor(action)
        if isinstance(action, TensorDictBase):
            action = action.to_dict()
        if self._action_cls is None:
            return action
        if isinstance(action, self._action_cls):
            return action
        if isinstance(action, dict):
            try:
                return self._action_cls(**action)
            except Exception:
                pass
        return action

    def _reward_to_tensor(self, reward: Any) -> torch.Tensor:
        if reward is None:
            if not self._warned_reward_none:
                warnings.warn(
                    "OpenEnvWrapper received reward=None; defaulting to 0.0.",
                    stacklevel=2,
                )
                self._warned_reward_none = True
            reward = 0.0
        return torch.as_tensor(
            reward, dtype=torch.float32, device=self.device
        ).view(*self.batch_size, 1)

    def _done_to_tensor(self, done: Any) -> torch.Tensor:
        return torch.as_tensor(done, dtype=torch.bool, device=self.device).view(
            *self.batch_size, 1
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:  # noqa: ARG002
        result = self._env.reset()
        if inspect.isawaitable(result):
            raise RuntimeError(
                "OpenEnvWrapper received an awaitable from reset(). "
                "Pass a sync client or set sync=True."
            )
        obs = self._format_observation(result.observation)
        reward = self._reward_to_tensor(getattr(result, "reward", None))
        done = self._done_to_tensor(getattr(result, "done", False))

        td = TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
                "truncated": torch.zeros_like(done),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return td

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:  # noqa: ARG002
        action = tensordict.get("action")
        action = self._format_action(action)
        result = self._env.step(action)
        if inspect.isawaitable(result):
            raise RuntimeError(
                "OpenEnvWrapper received an awaitable from step(). "
                "Pass a sync client or set sync=True."
            )
        obs = self._format_observation(result.observation)
        reward = self._reward_to_tensor(getattr(result, "reward", None))
        done = self._done_to_tensor(getattr(result, "done", False))

        td = TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
                "truncated": torch.zeros_like(done),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return td


class OpenEnvEnv(OpenEnvWrapper):
    """OpenEnv environment.

    Convenience class that constructs an OpenEnv client via the auto-discovery
    API. The resulting environment exposes OpenEnv observations/actions as
    non-tensor data under the TorchRL API.

    Args:
        env_name (str): OpenEnv environment name (e.g. ``"echo-env"``).

    Keyword Args:
        auto_action (bool, optional): If ``True`` (default), uses
            :class:`openenv.AutoAction` to fetch the action class for the env.
        env_kwargs (dict, optional): Extra keyword arguments forwarded to
            ``AutoEnv.from_env`` (e.g. base URLs or hub parameters).
        action_cls (type, optional): Action class to use instead of auto-discovery.
        observation_cls (type, optional): Observation class, used only for
            example data in specs.
        return_observation_dict (bool, optional): If ``True``, converts Pydantic
            observations to dicts. Defaults to ``False``.
        sync (bool, optional): If ``True`` (default), calls ``env.sync()`` when
            available.
        device (torch.device | str, optional): device on which tensors are placed.
        batch_size (torch.Size, optional): batch size (must be empty).
        allow_done_after_reset (bool, optional): tolerate done right after reset.
    """

    def __init__(
        self,
        env_name: str,
        *,
        auto_action: bool = True,
        env_kwargs: dict[str, Any] | None = None,
        action_cls: type | None = None,
        observation_cls: type | None = None,
        return_observation_dict: bool = False,
        sync: bool = True,
        **kwargs,
    ) -> None:
        if not _has_openenv:
            raise ImportError(
                "openenv python package was not found. "
                "Install it with `pip install openenv-core`."
            )

        if env_kwargs is None:
            env_kwargs = {}

        env = AutoEnv.from_env(env_name, **env_kwargs)

        if auto_action and action_cls is None:
            try:
                action_cls = AutoAction.from_env(env_name)
            except Exception:
                warnings.warn(
                    "OpenEnvEnv: failed to auto-load action class; "
                    "actions will be passed through as-is."
                )

        self.env_name = env_name
        super().__init__(
            env=env,
            action_cls=action_cls,
            observation_cls=observation_cls,
            return_observation_dict=return_observation_dict,
            sync=sync,
            **kwargs,
        )
