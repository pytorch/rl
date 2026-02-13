from __future__ import annotations

import importlib.util
import inspect
import json
import warnings
from typing import Any

import torch
from tensordict import lazy_stack, NonTensorData, TensorDict, TensorDictBase

from torchrl.data.llm import History
from torchrl.data.tensor_specs import Categorical, Composite, NonTensor, Unbounded
from torchrl.envs.llm.chat import ChatEnv
from torchrl.envs.utils import _classproperty
from torchrl.modules.llm.policies.common import ChatHistory

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
        fn = getattr(cls, method, None)
        if fn is not None:
            return fn()
    return None


def _to_history_content(value: Any) -> Any:
    value = _unwrap_nontensor(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return str(value)


class OpenEnvWrapper(ChatEnv):
    """OpenEnv environment wrapper.

    Wraps an existing OpenEnv client instance (sync or async) and exposes it
    under the TorchRL environment API.

    Observations and actions are stored as non-tensor data under the
    ``"observation"`` and ``"action"`` keys. This wrapper also exposes a
    chat-style interface (history mode) under the ``"history"`` key to
    integrate with TorchRL's LLM stack.

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
        input_mode (str, optional): Chat input mode. Only ``"history"`` is
            supported.
        system_prompt (str, optional): Optional system prompt prepended to the
            history.
        system_role (str, optional): Role name for system messages. Defaults to
            ``"system"``.
        user_role (str, optional): Role name for environment observations.
            Defaults to ``"user"``.
        policy_role (str, optional): Role name for agent responses. Defaults to
            ``"assistant"``.
        tokenizer (optional): Tokenizer stored on the instance for downstream
            transforms.
        template_kwargs (dict, optional): Template kwargs stored on the instance.
        device (torch.device | str, optional): device on which tensors are placed.
        batch_size (torch.Size, optional): batch size. OpenEnv environments are
            single-instance; only ``(1,)`` is supported to match the ChatEnv
            contract. Use ParallelEnv to create multiple OpenEnv instances.
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
        input_mode: str = "history",
        system_prompt: str | None = None,
        system_role: str = "system",
        user_role: str = "user",
        policy_role: str | None = "assistant",
        tokenizer: Any | None = None,
        template_kwargs: dict[str, Any] | None = None,
        data_key: str | None = None,
        **kwargs,
    ) -> None:
        device = kwargs.pop("device", None)
        batch_size = kwargs.pop("batch_size", None)
        allow_done_after_reset = kwargs.pop("allow_done_after_reset", False)

        if input_mode != "history":
            raise ValueError(
                f"{type(self).__name__} only supports input_mode='history'. Got {input_mode!r}."
            )

        if batch_size is None:
            batch_size = torch.Size((1,))
        elif isinstance(batch_size, int):
            batch_size = torch.Size([batch_size])
        elif isinstance(batch_size, list):
            batch_size = torch.Size(batch_size)
        else:
            batch_size = torch.Size(batch_size)

        if batch_size != torch.Size((1,)):
            raise ValueError(
                "OpenEnvWrapper only supports batch_size=(1,) to match ChatEnv. "
                "Use ParallelEnv to create multiple OpenEnv instances."
            )

        self._action_cls = action_cls
        self._observation_cls = observation_cls
        self._return_observation_dict = return_observation_dict
        self._sync_env = sync
        self._warned_reward_none = False
        self._history_prompt: History | None = None

        super().__init__(
            input_mode=input_mode,
            batch_size=batch_size,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            system_role=system_role,
            user_role=user_role,
            policy_role=policy_role,
            data_key=data_key,
            device=device,
        )
        self._allow_done_after_reset = allow_done_after_reset

        self._constructor_kwargs = kwargs
        self._check_kwargs({"env": env})
        self._env = self._build_env(env=env, **kwargs)
        self.is_closed = False
        self._init_env()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(attr)
        if attr.startswith("__"):
            raise AttributeError(
                "passing built-in private methods is "
                f"not permitted with type {type(self)}. "
                f"Got attribute {attr}."
            )
        if "_env" in self.__dir__():
            env = self.__getattribute__("_env")
            return getattr(env, attr)
        return super().__getattr__(attr)

    def _check_kwargs(self, kwargs: dict) -> None:
        if "env" not in kwargs or kwargs["env"] is None:
            raise TypeError("OpenEnvWrapper requires an 'env' argument.")

    def _build_env(self, env, **_) -> Any:
        if self._sync_env and hasattr(env, "sync") and callable(env.sync):
            env = env.sync()
        return env

    def _make_specs(self, env: Any | None = None) -> None:  # noqa: ARG002
        if self.batch_size != torch.Size((1,)):
            raise ValueError(
                "OpenEnvWrapper only supports batch_size=(1,) to match ChatEnv. "
                "Use ParallelEnv to create multiple OpenEnv instances."
            )

        obs_example = _example_from_cls(self._observation_cls)
        act_example = _example_from_cls(self._action_cls)

        observation_leaf = NonTensor(
            shape=self.batch_size, device=self.device, example_data=obs_example
        )
        action_leaf = NonTensor(
            shape=self.batch_size, device=self.device, example_data=act_example
        )

        obs_spec = Composite(shape=self.batch_size, device=self.device)
        obs_spec.set("observation", observation_leaf)
        if self.input_mode == "history":
            obs_spec.set(
                "history",
                ChatHistory.default_spec(shape=self.batch_size, keys=["prompt"]).to(
                    self.device
                ),
            )
        self.full_observation_spec = obs_spec

        action_spec = Composite(shape=self.batch_size, device=self.device)
        action_spec.set("action", action_leaf)
        if self.input_mode == "history":
            action_spec.set(
                "history",
                ChatHistory.default_spec(shape=self.batch_size, keys=["full"]).to(
                    self.device
                ),
            )
        self.full_action_spec = action_spec

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
            connect()

    def close(self, *, raise_if_closed: bool = True) -> None:
        self.is_closed = True
        disconnect = getattr(self._env, "disconnect", None)
        if callable(disconnect):
            disconnect()
            return
        super().close(raise_if_closed=raise_if_closed)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        for method in ("seed", "set_seed"):
            setter = getattr(self._env, method, None)
            if callable(setter):
                setter(seed)
                return
        warnings.warn("OpenEnvWrapper: seeding is not supported by this client.")

    def _format_observation(self, obs: Any) -> Any:
        if self._return_observation_dict:
            if hasattr(obs, "model_dump"):
                obs = obs.model_dump()
            elif hasattr(obs, "dict"):
                obs = obs.dict()
        return self._wrap_nontensor(obs)

    def _wrap_nontensor(self, value: Any) -> NonTensorData:
        if isinstance(value, NonTensorData):
            if value.batch_size == self.batch_size and value.device == self.device:
                return value
            value = value.data
        return NonTensorData(value, batch_size=self.batch_size, device=self.device)

    def _format_action(self, action: Any) -> Any:
        action = _unwrap_nontensor(action)
        if isinstance(action, TensorDictBase):
            action = action.to_dict()
        if self._action_cls is None:
            return action
        if isinstance(action, self._action_cls):
            return action
        if isinstance(action, dict):
            return self._action_cls(**action)
        action_field = self._get_action_field()
        if action_field is not None:
            return self._action_cls(**{action_field: action})
        return action

    def _get_action_field(self) -> str | None:
        cls = self._action_cls
        if cls is None:
            return None
        fields = getattr(cls, "model_fields", None)
        if fields is None:
            fields = getattr(cls, "__fields__", None)
        if fields and len(fields) == 1:
            return next(iter(fields))
        return None

    def _broadcast_to_batch(self, value: Any) -> Any:
        if not self.batch_size:
            return value
        if getattr(value, "batch_size", None) == self.batch_size:
            return value
        if (
            isinstance(value, list)
            and len(self.batch_size) == 1
            and len(value) == self.batch_size[0]
        ):
            return value
        for s in reversed(self.batch_size):
            value = [value for _ in range(s)]
        return value

    def _make_history_message(self, role: str, content: Any) -> History:
        role = self._broadcast_to_batch(role)
        content = self._broadcast_to_batch(_to_history_content(content))
        return History(
            role=role, content=content, batch_size=self.batch_size, device=self.device
        )

    def _build_prompt_history(self, obs: Any) -> History:
        history = self._make_history_message(self.user_role, obs)
        if self.system_prompt is not None:
            history_system = self._make_history_message(
                self.system_role, self.system_prompt
            )
            history = lazy_stack([history_system, history], -1)
        else:
            history = history.unsqueeze(-1)
        return history

    def _extract_history_parts(
        self, chat_history: ChatHistory | None
    ) -> tuple[History | None, History | None]:
        if chat_history is None:
            return None, None
        prompt = getattr(chat_history, "prompt", None)
        response = getattr(chat_history, "response", None)
        full = getattr(chat_history, "full", None)
        if response is None:
            if full is not None and prompt is not None:
                prompt_len = prompt.shape[-1]
                if full.shape[-1] >= prompt_len:
                    response = full[..., prompt_len:]
            if response is None and full is not None:
                response = full[..., -1:]
        return prompt, response

    def _history_to_action(self, response: History | None) -> Any:
        if response is None:
            return None
        if response.shape[-1] > 1:
            response = response[..., -1]
        content = response.content
        return _unwrap_nontensor(content)

    def _merge_history(self, prompt: History, new: History) -> History:
        if new.batch_dims == prompt.batch_dims:
            return prompt.extend(new, inplace=False, dim=-1)
        if new.batch_dims == prompt.batch_dims - 1:
            return prompt.append(new, inplace=False, dim=-1)
        return prompt.append(new, inplace=False)

    def _reward_to_tensor(self, reward: Any) -> torch.Tensor:
        if reward is None:
            if not self._warned_reward_none:
                warnings.warn(
                    "OpenEnvWrapper received reward=None; defaulting to 0.0.",
                    stacklevel=2,
                )
                self._warned_reward_none = True
            reward = 0.0
        return torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(
            *self.batch_size, 1
        )

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
        data = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "terminated": done.clone(),
            "truncated": torch.zeros_like(done),
        }
        if self.input_mode == "history":
            prompt_history = self._build_prompt_history(obs)
            self._history_prompt = prompt_history
            chat_history = ChatHistory._from_tensordict(
                TensorDict({}, batch_size=self.batch_size, device=self.device)
            )
            chat_history.prompt = prompt_history
            data["history"] = chat_history

        td = TensorDict(data, batch_size=self.batch_size, device=self.device)
        return td

    def _step(self, tensordict: TensorDict, **kwargs) -> TensorDict:  # noqa: ARG002
        chat_history = tensordict.get("history", None)
        prompt_history, response_history = self._extract_history_parts(chat_history)
        action_from_history = self._history_to_action(response_history)
        action = action_from_history
        if action is None:
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
        data = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "terminated": done.clone(),
            "truncated": torch.zeros_like(done),
        }
        if self.input_mode == "history":
            prompt = (
                prompt_history if prompt_history is not None else self._history_prompt
            )
            if prompt is None:
                prompt = self._build_prompt_history(obs)
            else:
                assistant_history = response_history
                if assistant_history is None and action_from_history is not None:
                    role = self.policy_role or "assistant"
                    assistant_history = self._make_history_message(
                        role, action_from_history
                    )
                if assistant_history is not None:
                    prompt = self._merge_history(prompt, assistant_history)
                obs_history = self._make_history_message(self.user_role, obs)
                prompt = self._merge_history(prompt, obs_history)
            self._history_prompt = prompt
            chat_out = ChatHistory._from_tensordict(
                TensorDict({}, batch_size=self.batch_size, device=self.device)
            )
            chat_out.prompt = prompt
            data["history"] = chat_out

        td = TensorDict(data, batch_size=self.batch_size, device=self.device)
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
            action_cls = AutoAction.from_env(env_name)

        self.env_name = env_name
        super().__init__(
            env=env,
            action_cls=action_cls,
            observation_cls=observation_cls,
            return_observation_dict=return_observation_dict,
            sync=sync,
            **kwargs,
        )
