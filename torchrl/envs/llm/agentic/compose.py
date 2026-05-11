# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""ToolCompose: parallel async tool dispatch over a ChatEnv history.

A :class:`ToolCompose` is a :class:`~torchrl.envs.transforms.Compose`
subclass that:

1. owns a :class:`~torchrl.envs.llm.agentic.ToolCallParser`,
2. holds a fixed set of :class:`~torchrl.envs.llm.agentic.Tool` instances
   (raises :class:`TypeError` on non-Tool insert),
3. on each step parses the latest assistant message *once*, dispatches
   matched tools concurrently via :func:`asyncio.gather`, renders each
   result through the parser, and extends the
   :class:`~torchrl.data.llm.History` with the resulting tool messages,
4. surfaces ``("agentic", "any_tool_calls")`` and
   ``("agentic", "stop_requested")`` keys in the step output for the env
   to use as termination signals.

The ChatEnv is unchanged; ``ToolCompose`` lives entirely in transform
space.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import lazy_stack, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.llm import History
from torchrl.envs.transforms import Compose, Transform

from .protocols import (
    ParsedCall,
    ParseResult,
    Tool,
    ToolCallParser,
    ToolContext,
    ToolError,
    ToolResult,
)
from .rate_limit import RateLimiter
from .schema import validate_args
from .tools.builtin import StopSignal


@dataclass
class DispatchResult:
    """Aggregate outcome of one :meth:`ToolCompose._dispatch_one` call.

    Attributes:
        cleaned_text: Assistant text with tool-call syntax stripped.
        calls: Parsed calls in emission order.
        results: Tool results, aligned with ``calls``.
        any_error: ``True`` if any tool failed (including unknown tool).
        stop_requested: ``True`` if any tool raised :class:`StopSignal`.
    """

    cleaned_text: str = ""
    calls: tuple[ParsedCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    any_error: bool = False
    stop_requested: bool = False


class _ToolTransformShim(Transform):
    """Thin :class:`Transform` wrapper around a :class:`Tool`.

    Stored inside :class:`ToolCompose.transforms` so the parent ``Compose``
    machinery (module hierarchy, device moves, training-mode propagation)
    keeps working. The shim has no ``_step`` of its own -- ``ToolCompose``
    drives dispatch directly and consults this shim only as a holder.
    """

    def __init__(self, tool: Tool) -> None:
        super().__init__()
        # Do NOT register the tool as a submodule -- tools are not nn.Module.
        # Storing in __dict__ keeps it out of state_dict.
        object.__setattr__(self, "tool", tool)

    @property
    def name(self) -> str:
        return self.tool.name

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:  # noqa: D401
        # ToolCompose drives dispatch directly; the shim is a pass-through.
        return next_tensordict


class ToolCompose(Compose):
    """A :class:`Compose` of :class:`~torchrl.envs.llm.agentic.Tool` objects
    with parallel async dispatch.

    Args:
        tools: The set of tools the LLM may call. Each must conform to the
            :class:`~torchrl.envs.llm.agentic.Tool` protocol (``name``,
            ``input_schema``, async ``run``). Inserting a non-Tool (e.g. a
            plain :class:`Transform`) raises :class:`TypeError`.
        parser: The :class:`~torchrl.envs.llm.agentic.ToolCallParser` used
            to extract calls from each assistant message and render
            results back as tool messages.
        rate_limits: Optional per-tool :class:`RateLimiter` map (keyed by
            tool name). Tools without an entry are unthrottled.
        per_call_timeout: Default per-call timeout in seconds. ``None``
            means rely on tool/repl timeouts only.
        pass_state_to_tools: If ``True`` and a tool has
            ``wants_state=True``, ``ctx.state`` is populated with a
            filtered read-only view of the env tensordict (mirrors the
            legacy ``ExecuteToolsInOrder`` knob).
        tool_role: Role string used when injecting tool messages into
            :class:`~torchrl.data.llm.History` (default ``"tool"``).
        validate_inputs: If ``True`` (default), validate ``args`` against
            ``tool.input_schema`` before dispatch. Schema mismatches are
            reported as :class:`ToolResult` with ``is_error=True``.

    Example:
        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.llm import ChatEnv  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic import ToolCompose  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.parsers import XMLToolCallParser
        >>> from torchrl.envs.llm.agentic.tools import StopTool
        >>> compose = ToolCompose(
        ...     tools=[StopTool()],
        ...     parser=XMLToolCallParser(),
        ... )
    """

    def __init__(
        self,
        *,
        tools: Iterable[Tool],
        parser: ToolCallParser,
        rate_limits: Mapping[str, RateLimiter] | None = None,
        per_call_timeout: float | None = None,
        pass_state_to_tools: bool = False,
        tool_role: str = "tool",
        validate_inputs: bool = True,
    ) -> None:
        tool_list = list(tools)
        for t in tool_list:
            if not _is_tool(t):
                raise TypeError(
                    f"ToolCompose accepts Tool objects only; got "
                    f"{type(t).__name__!r}. Wrap a legacy transform with "
                    "torchrl.envs.llm.agentic.tools.as_tool(...)."
                )
        if not isinstance(parser, ToolCallParser):
            raise TypeError(
                "parser must implement ToolCallParser; got "
                f"{type(parser).__name__!r}"
            )
        names = [t.name for t in tool_list]
        if len(names) != len(set(names)):
            seen: set[str] = set()
            dups = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
            raise ValueError(f"duplicate tool names: {dups!r}")
        shims = [_ToolTransformShim(t) for t in tool_list]
        super().__init__(*shims)
        self._tool_list: list[Tool] = tool_list
        self._tools_by_name: dict[str, Tool] = {t.name: t for t in tool_list}
        self.parser = parser
        self._rate_limits: dict[str, RateLimiter] = dict(rate_limits or {})
        self._per_call_timeout = per_call_timeout
        self._pass_state = pass_state_to_tools
        self._tool_role = tool_role
        self._validate_inputs = validate_inputs
        self._setup_done = False

    # ----- introspection helpers -----

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tool_list)

    def __getitem__(self, key: str | int):  # type: ignore[override]
        if isinstance(key, str):
            return self._tools_by_name[key]
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if isinstance(key, str):
            return key in self._tools_by_name
        return super().__contains__(key)

    # ----- enforce Tool-only insertion -----

    def append_transform(self, transform):  # type: ignore[override]
        # ``Compose.append_transform`` accepts arbitrary Transforms. We
        # constrain to Tools to keep the dispatch invariants intact.
        raise TypeError(
            "ToolCompose does not accept arbitrary transforms. Use "
            "append_tool(tool) instead."
        )

    def append_tool(self, tool: Tool) -> None:
        """Append a :class:`Tool` to the dispatch set."""
        if not _is_tool(tool):
            raise TypeError(
                f"append_tool requires a Tool; got {type(tool).__name__!r}"
            )
        if tool.name in self._tools_by_name:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        shim = _ToolTransformShim(tool)
        self.transforms.append(shim)
        shim.set_container(self)
        self._tool_list.append(tool)
        self._tools_by_name[tool.name] = tool

    # ----- lifecycle -----

    async def _setup_tools(self) -> None:
        if self._setup_done:
            return
        for tool in self._tool_list:
            try:
                await tool.setup()
            except Exception:  # pragma: no cover -- per-tool setup is best-effort
                torchrl_logger.exception(
                    "tool %r setup raised; continuing", tool.name
                )
        self._setup_done = True

    async def _teardown_tools(self) -> None:
        for tool in self._tool_list:
            try:
                await tool.teardown()
            except Exception:  # pragma: no cover
                torchrl_logger.exception(
                    "tool %r teardown raised; continuing", tool.name
                )
        self._setup_done = False

    def close(self) -> None:  # type: ignore[override]
        super().close()
        try:
            _run_async(self._teardown_tools())
        except Exception:  # pragma: no cover
            torchrl_logger.exception("teardown_tools raised; continuing")

    # ----- the step path -----

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as flat:
                flat = self._step(tensordict, flat)
            return next_tensordict
        history = self._extract_history(next_tensordict)
        last = history[..., -1]
        contents = last.content
        if isinstance(contents, str):
            contents = [contents]
        # asyncio.gather across batch items.
        dispatch_results: list[DispatchResult] = _run_async(
            self._dispatch_batch(list(contents), next_tensordict)
        )
        return self._inject_results(history, dispatch_results, next_tensordict)

    def _extract_history(self, td: TensorDictBase) -> History:
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "ToolCompose must be used inside a TransformedEnv"
            )
        base_env = parent.base_env
        if getattr(base_env, "input_mode", None) != "history":
            raise RuntimeError(
                "ToolCompose requires the underlying ChatEnv to use "
                "input_mode='history' (got "
                f"{getattr(base_env, 'input_mode', None)!r})"
            )
        return td["history"].prompt

    async def _dispatch_batch(
        self, contents: list[str], td: TensorDictBase
    ) -> list[DispatchResult]:
        await self._setup_tools()
        # Each batch item runs concurrently with every other; within a
        # batch item, calls also run concurrently.
        return list(
            await asyncio.gather(
                *(
                    self._dispatch_one(content, td, batch_index=i)
                    for i, content in enumerate(contents)
                )
            )
        )

    async def _dispatch_one(
        self, content: str, td: TensorDictBase, *, batch_index: int
    ) -> DispatchResult:
        parsed: ParseResult = self.parser.parse(content)
        if not parsed.calls:
            return DispatchResult(cleaned_text=parsed.text)
        # Parallel run all calls in this batch item.
        coros = [
            self._run_one(call, td, batch_index=batch_index)
            for call in parsed.calls
        ]
        outcomes = await asyncio.gather(*coros, return_exceptions=False)
        results: list[ToolResult] = []
        any_error = False
        stop = False
        for r in outcomes:
            if isinstance(r, _StopMarker):
                results.append(r.result)
                stop = True
            else:
                results.append(r)
                if r.is_error:
                    any_error = True
        return DispatchResult(
            cleaned_text=parsed.text,
            calls=parsed.calls,
            results=tuple(results),
            any_error=any_error,
            stop_requested=stop,
        )

    async def _run_one(
        self, call: ParsedCall, td: TensorDictBase, *, batch_index: int
    ):
        tool = self._tools_by_name.get(call.tool)
        if tool is None:
            return ToolResult.from_text(
                f"unknown tool: {call.tool!r}",
                is_error=True,
                meta={"call_id": call.call_id},
            )
        if self._validate_inputs:
            try:
                validate_args(call.args, tool.input_schema)
            except Exception as e:
                return ToolResult.from_text(
                    f"argument validation failed for {tool.name!r}: {e}",
                    is_error=True,
                    meta={"call_id": call.call_id},
                )
        ctx = ToolContext(
            call_id=call.call_id,
            tag=call.tag,
            state=self._filter_state(td, batch_index) if self._pass_state and getattr(
                tool, "wants_state", False
            ) else None,
            sandbox=getattr(tool, "sandbox", None),
            repl=getattr(tool, "repl", None),
            compose=self,
        )
        limiter = self._rate_limits.get(tool.name)
        timeout = self._per_call_timeout
        try:
            if limiter is not None:
                async with limiter.slot():
                    coro = tool.run(call.args, ctx)
                    if timeout is None:
                        return await coro
                    return await asyncio.wait_for(coro, timeout=timeout)
            coro = tool.run(call.args, ctx)
            if timeout is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=timeout)
        except StopSignal as s:
            return _StopMarker(
                ToolResult.from_text(
                    f"[stop] {s}", meta={"call_id": call.call_id, "stop": True}
                )
            )
        except ToolError as e:
            return ToolResult.from_text(
                str(e), is_error=True, meta={"call_id": call.call_id}
            )
        except asyncio.TimeoutError:
            return ToolResult.from_text(
                f"tool {tool.name!r} timed out after {timeout}s",
                is_error=True,
                meta={"call_id": call.call_id, "timed_out": True},
            )
        except Exception as e:  # noqa: BLE001 -- failure isolation
            torchrl_logger.exception(
                "tool %r raised; reporting as error", tool.name
            )
            return ToolResult.from_text(
                f"{type(e).__name__}: {e}",
                is_error=True,
                meta={"call_id": call.call_id},
            )

    def _filter_state(
        self, td: TensorDictBase, batch_index: int
    ) -> TensorDictBase | None:
        try:
            view = td[batch_index] if td.batch_dims else td
        except Exception:  # pragma: no cover
            return None
        try:
            return view.exclude("history")
        except Exception:  # pragma: no cover
            return view

    # ----- result injection -----

    def _inject_results(
        self,
        history: History,
        dispatches: list[DispatchResult],
        td: TensorDictBase,
    ) -> TensorDictBase:
        # Build per-batch-item History extension lists and the agentic
        # signals.
        any_calls = [bool(d.calls) for d in dispatches]
        any_error = [d.any_error for d in dispatches]
        any_stop = [d.stop_requested for d in dispatches]

        per_item_messages: list[list[History]] = []
        for d in dispatches:
            messages: list[History] = []
            for call, result in zip(d.calls, d.results):
                rendered = self.parser.render_result(call.call_id, result)
                content = rendered.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                messages.append(
                    History(role=self._tool_role, content=str(content))
                )
            per_item_messages.append(messages)

        max_len = max((len(m) for m in per_item_messages), default=0)
        if max_len > 0:
            padded: list[list[History]] = []
            for messages in per_item_messages:
                if len(messages) == max_len:
                    padded.append(messages)
                else:
                    padded.append(
                        messages
                        + [History(role="<none>", content="")]
                        * (max_len - len(messages))
                    )
            stacked = lazy_stack([lazy_stack(m) for m in padded])
            history.extend(stacked, dim=-1)
            td["history"].prompt = history

        device = td.device
        td.set(
            ("agentic", "any_tool_calls"),
            torch.tensor(any_calls, dtype=torch.bool, device=device),
        )
        td.set(
            ("agentic", "any_error"),
            torch.tensor(any_error, dtype=torch.bool, device=device),
        )
        td.set(
            ("agentic", "stop_requested"),
            torch.tensor(any_stop, dtype=torch.bool, device=device),
        )
        return td


@dataclass
class _StopMarker:
    """Internal carrier flagging a :class:`StopSignal` from a tool."""

    result: ToolResult


def _is_tool(obj: Any) -> bool:
    """Duck-typed Tool conformance check.

    ``Tool`` is a runtime-checkable Protocol but Protocol checks treat
    every required attribute as needing to be present at the *instance*
    level; class-level ``ClassVar``s satisfy this in CPython but the check
    is unreliable across versions when adapters set per-instance ``name``.
    Reuse a simpler attribute-presence test.
    """
    required = ("name", "input_schema", "run", "setup", "teardown")
    return all(hasattr(obj, a) for a in required) and callable(obj.run)


# ----- async runner: nested-loop safe -----

def _run_async(coro):
    """Run ``coro`` to completion regardless of whether the caller is
    inside an event loop.

    - No running loop: :func:`asyncio.run`.
    - Running loop: dispatch on a worker thread that owns its own loop
      and join. (Necessary because Compose._step is sync and may be
      called from inside a Jupyter-style outer loop.)
    """
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(coro)
    fut: Future = Future()

    def _target():
        try:
            fut.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return fut.result()


__all__ = ["DispatchResult", "ToolCompose"]
