# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from tensordict import lazy_stack, TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData

from torchrl.data.llm import History
from torchrl.data.tensor_specs import Categorical, Composite, NonTensor, Unbounded
from torchrl.envs.libs.openenv import (
    _as_done_tensor,
    _as_reward_tensor,
    _auto_action_from_env,
    _coerce_action,
    _ensure_sync_result,
    _example_from_cls,
    _extract_observation,
    _extract_reward_done,
    _format_object,
    _has_openenv,
    _normalize_batch_size,
    _squeeze_singleton,
    ActionAdapter,
    AutoEnv,
    ObservationAdapter,
)
from torchrl.envs.llm.chat import ChatEnv
from torchrl.modules.llm.policies.common import ChatHistory

__all__ = ["OpenEnvChatEnv"]


HistoryContentAdapter = Callable[[Any], str]


def _to_history_content(value: Any) -> str:
    value = _squeeze_singleton(value)
    if isinstance(value, str):
        return value
    return str(value)


def _nested_none(shape: torch.Size) -> Any:
    if not shape:
        return None
    return [_nested_none(shape[1:]) for _ in range(shape[0])]


class OpenEnvChatEnv(ChatEnv):
    """ChatEnv-compatible adapter for OpenEnv environments.

    ``OpenEnvChatEnv`` maps OpenEnv observations to ``history.prompt`` and maps
    LLM assistant responses back to OpenEnv actions. It is intended for LLM
    collectors and GRPO-style training while preserving the regular ChatEnv
    history contract.

    Args:
        env: OpenEnv client instance to wrap. Mutually exclusive with
            ``env_name``.

    Keyword Args:
        env_name (str, optional): OpenEnv environment name accepted by
            ``openenv.AutoEnv.from_env``. Used when ``env`` is not provided.
        env_kwargs (dict, optional): Extra arguments forwarded to
            ``AutoEnv.from_env``.
        action_cls (type, optional): Pydantic-style action class.
        auto_action (bool, optional): Discover ``action_cls`` with
            ``AutoAction.from_env`` when using ``env_name``. Defaults to ``True``.
        action_adapter (Callable[[Any], Any], optional): Callable applied to the
            assistant response before action-class coercion.
        observation_adapter (Callable[[Any], Any], optional): Callable applied to
            OpenEnv observations before storing them.
        history_content_adapter (Callable[[Any], str], optional): Callable used
            to render OpenEnv observations as user messages. Defaults to
            ``str(...)`` with singleton unbatching.
        return_observation_dict (bool, optional): Convert Pydantic-like
            observations to dictionaries before storing/rendering. Defaults to
            ``True``.
        sync (bool, optional): Convert async clients with ``sync()``. Defaults to
            ``True``.
        input_mode (Literal["history"], optional): Only ``"history"`` is
            supported.
        batch_size (torch.Size, optional): Must be ``(1,)``. Use ``SerialEnv``
            or ``ParallelEnv`` to create multiple OpenEnv chat clients.
        system_prompt (str, optional): System message prepended at reset.
        tokenizer: Tokenizer stored for downstream transforms.
        template_kwargs (dict, optional): Template kwargs for downstream
            transforms.
        device (torch.device | str, optional): Tensor device.

    Examples:
        >>> from dataclasses import dataclass
        >>> from torchrl.envs.llm import OpenEnvChatEnv
        >>> @dataclass
        ... class Result:
        ...     observation: str
        ...     reward: float = 0.0
        ...     done: bool = False
        >>> class EchoEnv:
        ...     def reset(self):
        ...         return Result("say hello")
        ...     def step(self, action):
        ...         return Result(f"heard {action}", reward=1.0, done=True)
        >>> env = OpenEnvChatEnv(env=EchoEnv())
        >>> td = env.reset()
        >>> td["history"].prompt.content[0][0]
        'say hello'
    """

    def __init__(
        self,
        *,
        env: Any | None = None,
        env_name: str | None = None,
        env_kwargs: dict[str, Any] | None = None,
        action_cls: type | None = None,
        auto_action: bool = True,
        action_adapter: ActionAdapter | None = None,
        observation_adapter: ObservationAdapter | None = None,
        history_content_adapter: HistoryContentAdapter | None = None,
        return_observation_dict: bool = True,
        sync: bool = True,
        input_mode: Literal["history"] = "history",
        system_prompt: str | None = None,
        system_role: str = "system",
        user_role: str = "user",
        policy_role: str | None = "assistant",
        tokenizer: Any | None = None,
        template_kwargs: dict[str, Any] | None = None,
        data_key: str | None = None,
        device: torch.device | str | None = None,
        batch_size: torch.Size | tuple[int, ...] | list[int] | int | None = None,
        allow_done_after_reset: bool = False,
        **kwargs: Any,
    ) -> None:
        if input_mode != "history":
            raise ValueError(
                f"{type(self).__name__} only supports input_mode='history', got {input_mode!r}."
            )
        batch_size = _normalize_batch_size(batch_size, default=torch.Size([1]))
        if batch_size != torch.Size([1]):
            raise ValueError(
                "OpenEnvChatEnv only supports batch_size=(1,). Use SerialEnv "
                "or ParallelEnv to run multiple OpenEnv clients."
            )
        if (env is None) == (env_name is None):
            raise TypeError("Pass exactly one of 'env' or 'env_name'.")
        if env_name is not None:
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
        self._env = self._build_env(env=env, sync=sync)
        self._action_cls = action_cls
        self._action_adapter = action_adapter
        self._observation_adapter = observation_adapter
        self._history_content_adapter = history_content_adapter or _to_history_content
        self._return_observation_dict = return_observation_dict
        self._current_prompt: History | None = None
        self._action_example = _example_from_cls(action_cls)
        self._constructor_kwargs = kwargs
        self.is_closed = False
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
        self._init_env()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(attr)
        if attr.startswith("__"):
            raise AttributeError(
                "passing built-in private methods is "
                f"not permitted with type {type(self)}. Got attribute {attr}."
            )
        if "_env" in self.__dir__():
            return getattr(self.__getattribute__("_env"), attr)
        return super().__getattr__(attr)

    def _build_env(self, *, env: Any, sync: bool) -> Any:
        if sync and hasattr(env, "sync") and callable(env.sync):
            env = env.sync()
        return env

    def _init_env(self) -> None:
        connect = getattr(self._env, "connect", None)
        if callable(connect):
            _ensure_sync_result(connect(), "connect")

    def _make_specs(self) -> None:
        self.full_observation_spec = Composite(
            history=ChatHistory.default_spec(shape=self.batch_size, keys=["prompt"]).to(
                self.device
            ),
            observation=NonTensor(shape=self.batch_size, device=self.device),
            shape=self.batch_size,
            device=self.device,
        )
        self.full_action_spec = Composite(
            history=ChatHistory.default_spec(shape=self.batch_size, keys=["full"]).to(
                self.device
            ),
            action=NonTensor(
                shape=self.batch_size,
                device=self.device,
                example_data=self._action_example,
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

    def _format_observation(self, observation: Any) -> Any:
        return _format_object(
            observation,
            return_object_dict=self._return_observation_dict,
            adapter=self._observation_adapter,
        )

    def _wrap_observation(self, observation: Any) -> NonTensorData:
        return NonTensorData(
            observation, batch_size=self.batch_size, device=self.device
        )

    def _make_history_message(self, role: str, content: Any) -> History:
        return History(
            role=[role],
            content=[content],
            tool_calls=[None],
            tool_responses=[None],
            batch_size=self.batch_size,
            device=self.device,
        )

    def _build_prompt_history(self, observation: Any) -> History:
        content = self._history_content_adapter(observation)
        user_history = self._make_history_message(self.user_role, content)
        if self.system_prompt is not None:
            system_history = self._make_history_message(
                self.system_role, self.system_prompt
            )
            return lazy_stack([system_history, user_history], -1)
        return user_history.unsqueeze(-1)

    def _extract_history_parts(
        self, chat_history: ChatHistory | None
    ) -> tuple[History | None, History | None]:
        if chat_history is None:
            return None, None
        prompt = getattr(chat_history, "prompt", None)
        response = getattr(chat_history, "response", None)
        full = getattr(chat_history, "full", None)
        if response is None and full is not None:
            if prompt is not None:
                prompt_len = prompt.shape[-1]
                if full.shape[-1] > prompt_len:
                    response = full[..., prompt_len:]
            elif full.shape[-1] > 0:
                response = full[..., -1:]
        return prompt, response

    def _history_to_action(self, response: History | None) -> Any:
        if response is None or response.shape[-1] == 0:
            return None
        if response.shape[-1] > 1:
            response = response[..., -1:]
        return _squeeze_singleton(response.content)

    def _merge_history(self, prompt: History, new: History) -> History:
        prompt = self._ensure_history_optional_fields(prompt)
        new = self._ensure_history_optional_fields(new)
        if new.batch_dims == prompt.batch_dims:
            return prompt.extend(new, inplace=False, dim=-1)
        if new.batch_dims == prompt.batch_dims - 1:
            return prompt.append(new, inplace=False, dim=-1)
        return prompt.append(new, inplace=False)

    def _ensure_history_optional_fields(self, history: History) -> History:
        if (
            getattr(history, "tool_calls", None) is not None
            and getattr(history, "tool_responses", None) is not None
        ):
            return history
        history = history.clone()
        if getattr(history, "tool_calls", None) is None:
            history.tool_calls = _nested_none(history.shape)
        if getattr(history, "tool_responses", None) is None:
            history.tool_responses = _nested_none(history.shape)
        return history

    def _result_to_data(
        self, result: Any, *, include_reward_done: bool = True
    ) -> tuple[dict[str, Any], Any, Any, Any, Any]:
        raw_observation = _extract_observation(result)
        observation = self._format_observation(raw_observation)
        data = {
            "observation": self._wrap_observation(observation),
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
        else:
            reward = done = terminated = None
        return data, observation, reward, done, terminated

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs: Any):
        result = _ensure_sync_result(self._env.reset(), "reset")
        data, observation, *_ = self._result_to_data(result, include_reward_done=False)
        prompt = self._build_prompt_history(observation)
        self._current_prompt = prompt
        chat_history = ChatHistory._from_tensordict(
            TensorDict({}, batch_size=self.batch_size, device=self.device)
        )
        chat_history.prompt = prompt
        data["history"] = chat_history
        return TensorDict(data, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDictBase, **kwargs: Any) -> TensorDict:
        chat_history = tensordict.get("history", None)
        prompt, response = self._extract_history_parts(chat_history)
        action_from_history = self._history_to_action(response)
        action = action_from_history
        if action is None:
            action = tensordict.get("action", None)
        action = _coerce_action(
            action, action_cls=self._action_cls, action_adapter=self._action_adapter
        )
        result = _ensure_sync_result(self._env.step(action), "step")
        data, observation, *_ = self._result_to_data(result)
        if prompt is None:
            prompt = self._current_prompt
        if prompt is None:
            prompt = self._build_prompt_history(observation)
        else:
            if (
                response is not None
                and response.shape[-1] > 0
                and action_from_history is not None
            ):
                prompt = self._merge_history(prompt, response)
            elif action_from_history is not None:
                role = self.policy_role or "assistant"
                assistant = self._make_history_message(role, action_from_history)
                prompt = self._merge_history(prompt, assistant)
            obs_msg = self._make_history_message(
                self.user_role, self._history_content_adapter(observation)
            )
            prompt = self._merge_history(prompt, obs_msg)
        self._current_prompt = prompt
        chat_out = ChatHistory._from_tensordict(
            TensorDict({}, batch_size=self.batch_size, device=self.device)
        )
        chat_out.prompt = prompt
        data["history"] = chat_out
        return TensorDict(data, batch_size=self.batch_size, device=self.device)
