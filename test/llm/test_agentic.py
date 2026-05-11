# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the agentic toolkit under :mod:`torchrl.envs.llm.agentic`.

Split out from ``test/llm/test_llm_transforms.py``: the new package is
large enough -- parsers, sandbox backends, REPLs, and (in follow-up
commits) ToolCompose plus built-in tools -- that bundling its tests
into the legacy file made discovery awkward.
"""
from __future__ import annotations

import asyncio
import json
import sys
import warnings

import pytest
from tensordict import set_list_to_stack

from torchrl.envs.llm.agentic import (
    ParsedCall,
    Tool,
    ToolCallParser,
    ToolResult,
    validate_args,
)
from torchrl.envs.llm.agentic.parsers import (
    AnthropicToolUseParser,
    JSONToolCallParser,
    OpenAIToolCallParser,
    XMLToolCallParser,
)
from torchrl.envs.llm.agentic.repl import _has_jupyter_client, SubprocessRepl
from torchrl.envs.llm.agentic.sandbox import (
    BubblewrapSandbox,
    default_sandbox,
    ResourceLimits,
    SeatbeltSandbox,
    UnsafeSubprocessSandbox,
)
from torchrl.envs.llm.agentic.sandbox.subprocess_bwrap import _has_bwrap
from torchrl.envs.llm.agentic.sandbox.subprocess_seatbelt import _has_sandbox_exec


@pytest.fixture(scope="module", autouse=True)
def list_to_stack_fixture():
    with set_list_to_stack(True):
        yield


def _run(coro):
    return asyncio.run(coro)


class TestAgenticParsers:
    """Per-parser conformance: parse, render_call round-trip, render_result,
    stable call_id (parser-supplied or assigned).
    """

    @pytest.mark.parametrize(
        "parser_cls",
        [
            XMLToolCallParser,
            JSONToolCallParser,
            OpenAIToolCallParser,
            AnthropicToolUseParser,
        ],
    )
    def test_implements_protocol(self, parser_cls):
        p = parser_cls()
        assert isinstance(p, ToolCallParser)
        assert isinstance(p.name, str) and p.name

    def test_xml_parse_and_call_id(self):
        p = XMLToolCallParser()
        r = p.parse('<tool name="echo" tag="t1">{"text": "hi"}</tool>tail')
        assert len(r.calls) == 1
        c = r.calls[0]
        assert c.tool == "echo"
        assert c.args == {"text": "hi"}
        assert c.call_id == "t1"  # tag becomes call_id when present
        assert c.tag == "t1"
        assert r.text == "tail"

    def test_xml_assigns_call_id_when_no_tag(self):
        p = XMLToolCallParser()
        r = p.parse('<tool name="echo">{}</tool>')
        assert r.calls[0].call_id  # non-empty
        assert r.calls[0].tag is None

    def test_xml_round_trip(self):
        p = XMLToolCallParser()
        call = ParsedCall(tool="echo", args={"text": "hi"}, call_id="abc", tag="abc")
        rendered = p.render_call(call)
        re_parsed = p.parse(rendered)
        assert re_parsed.calls[0].tool == "echo"
        assert re_parsed.calls[0].args == {"text": "hi"}
        assert re_parsed.calls[0].call_id == "abc"

    def test_xml_render_result(self):
        p = XMLToolCallParser()
        msg = p.render_result("c1", ToolResult.from_text("output"))
        assert msg["role"] == "tool"
        assert "c1" in msg["content"]
        assert "output" in msg["content"]

    def test_json_block_parse_with_id(self):
        p = JSONToolCallParser()
        resp = json.dumps(
            {
                "message": "ok",
                "tools": [{"tool": "echo", "args": {"x": 1}, "id": "j1"}],
            }
        )
        r = p.parse(resp)
        assert r.text == "ok"
        assert r.calls[0].tool == "echo"
        assert r.calls[0].args == {"x": 1}
        assert r.calls[0].call_id == "j1"

    def test_json_block_assigns_call_id(self):
        p = JSONToolCallParser()
        resp = json.dumps({"message": "", "tools": [{"tool": "x", "args": {}}]})
        r = p.parse(resp)
        assert r.calls[0].call_id  # uuid hex

    def test_json_block_invalid_json_falls_back_to_text(self):
        p = JSONToolCallParser()
        r = p.parse("not json at all")
        assert r.text == "not json at all"
        assert r.calls == ()

    def test_openai_preserves_id_and_decodes_args(self):
        p = OpenAIToolCallParser()
        r = p.parse(
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "torchrl"}',
                        },
                    }
                ],
            }
        )
        assert r.calls[0].tool == "search"
        assert r.calls[0].args == {"q": "torchrl"}
        assert r.calls[0].call_id == "call_a"

    def test_openai_render_result_uses_tool_call_id(self):
        p = OpenAIToolCallParser()
        msg = p.render_result("call_a", ToolResult.from_text("done"))
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_a"
        assert msg["content"] == "done"

    def test_anthropic_extracts_text_and_tool_use(self):
        p = AnthropicToolUseParser()
        r = p.parse(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "search",
                        "input": {"q": "x"},
                    },
                ],
            }
        )
        assert r.text == "Let me search."
        assert r.calls[0].tool == "search"
        assert r.calls[0].args == {"q": "x"}
        assert r.calls[0].call_id == "toolu_a"

    def test_anthropic_render_result_uses_tool_use_id(self):
        p = AnthropicToolUseParser()
        msg = p.render_result("toolu_a", ToolResult.from_text("hit", is_error=False))
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_a"

    def test_validate_args_required(self):
        schema = {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }
        validate_args({"code": "print(1)"}, schema)
        with pytest.raises(Exception):
            validate_args({}, schema)

    def test_validate_args_type_mismatch(self):
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        }
        validate_args({"n": 3}, schema)
        with pytest.raises(Exception):
            validate_args({"n": "three"}, schema)

    def test_tool_protocol_runtime_check(self):
        class _T:
            name = "t"
            description = "d"
            input_schema = {"type": "object", "properties": {}}
            output_schema = None
            wants_state = False

            async def run(self, args, ctx):
                return ToolResult.from_text("ok")

            async def setup(self):
                pass

            async def teardown(self):
                pass

        assert isinstance(_T(), Tool)


class TestAgenticSandbox:
    """Sandbox protocol conformance + sandbox-escape negatives."""

    def test_unsafe_warns_on_open(self):
        async def go():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                async with UnsafeSubprocessSandbox() as _s:
                    pass
            assert any(issubclass(w.category, UserWarning) for w in caught)

        _run(go())

    def test_unsafe_runs_simple_command(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=5)) as s:
                r = await s.run(["/bin/echo", "hello"])
                assert r.exit_code == 0
                assert r.stdout.strip() == "hello"
                assert not r.timed_out

        _run(go())

    def test_unsafe_timeout(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=0.2)) as s:
                r = await s.run(["/bin/sleep", "5"])
                assert r.timed_out

        _run(go())

    def test_resource_limits_narrow(self):
        a = ResourceLimits(wall_seconds=10, network="full")
        b = ResourceLimits(wall_seconds=2, network="none")
        c = a.narrow(b)
        assert c.wall_seconds == 2
        assert c.network == "none"
        # Reverse direction: narrow keeps the strictest.
        c2 = b.narrow(a)
        assert c2.wall_seconds == 2
        assert c2.network == "none"

    def test_default_sandbox_picks_platform(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = default_sandbox()
        if sys.platform.startswith("linux") and _has_bwrap:
            assert isinstance(s, BubblewrapSandbox)
        elif sys.platform == "darwin" and _has_sandbox_exec:
            assert isinstance(s, SeatbeltSandbox)
        else:
            assert isinstance(s, UnsafeSubprocessSandbox)

    @pytest.mark.skipif(
        not (sys.platform.startswith("linux") and _has_bwrap),
        reason="bubblewrap not available",
    )
    def test_bubblewrap_blocks_fs_escape(self, tmp_path):
        """Writes outside fs_write_roots must fail."""
        write_root = tmp_path / "work"
        write_root.mkdir()
        outside = tmp_path / "forbidden"

        async def go():
            limits = ResourceLimits(
                wall_seconds=5,
                network="none",
                fs_write_roots=(str(write_root),),
            )
            async with BubblewrapSandbox(limits=limits) as s:
                # Inside the write root: must succeed.
                inside_path = write_root / "inside.txt"
                r = await s.run(
                    [
                        "/bin/sh",
                        "-c",
                        f"echo hi > {inside_path}",
                    ]
                )
                assert r.exit_code == 0
                assert inside_path.read_text().strip() == "hi"
                # Outside the write root: must fail.
                r2 = await s.run(
                    [
                        "/bin/sh",
                        "-c",
                        f"echo nope > {outside}",
                    ]
                )
                assert r2.exit_code != 0
                assert not outside.exists()

        _run(go())

    @pytest.mark.skipif(
        not (sys.platform.startswith("linux") and _has_bwrap),
        reason="bubblewrap not available",
    )
    def test_bubblewrap_blocks_network(self):
        """network='none' must block outbound TCP."""

        async def go():
            limits = ResourceLimits(wall_seconds=5, network="none")
            async with BubblewrapSandbox(limits=limits) as s:
                r = await s.run(
                    [
                        "python3",
                        "-c",
                        "import socket; "
                        "socket.create_connection(('1.1.1.1', 80), timeout=2)",
                    ]
                )
                assert r.exit_code != 0

        _run(go())

    @pytest.mark.skipif(
        not (sys.platform == "darwin" and _has_sandbox_exec),
        reason="sandbox-exec not available",
    )
    def test_seatbelt_blocks_fs_escape(self, tmp_path):
        write_root = tmp_path / "work"
        write_root.mkdir()
        outside = tmp_path / "forbidden"

        async def go():
            limits = ResourceLimits(
                wall_seconds=5,
                network="none",
                fs_write_roots=(str(write_root),),
            )
            async with SeatbeltSandbox(limits=limits) as s:
                inside_path = write_root / "inside.txt"
                r = await s.run(["/bin/sh", "-c", f"echo hi > {inside_path}"])
                assert r.exit_code == 0
                r2 = await s.run(["/bin/sh", "-c", f"echo nope > {outside}"])
                assert r2.exit_code != 0
                assert not outside.exists()

        _run(go())


class TestAgenticRepl:
    """REPL state, error capture, restart, timeout."""

    def test_subprocess_repl_state_persists(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=10)) as s:
                async with SubprocessRepl(s) as r:
                    r1 = await r.execute("x = 41")
                    assert r1.error is None
                    r2 = await r.execute("print(x + 1)")
                    assert r2.error is None
                    assert r2.stdout.strip() == "42"

        _run(go())

    def test_subprocess_repl_captures_errors(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=10)) as s:
                async with SubprocessRepl(s) as r:
                    res = await r.execute("1/0")
                    assert res.error is not None
                    assert res.error.ename == "ZeroDivisionError"

        _run(go())

    def test_subprocess_repl_restart_clears_state(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=10)) as s:
                async with SubprocessRepl(s) as r:
                    await r.execute("y = 99")
                    await r.restart()
                    res = await r.execute("print(y)")
                    assert res.error is not None  # NameError

        _run(go())

    def test_subprocess_repl_timeout(self):
        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=10)) as s:
                async with SubprocessRepl(s) as r:
                    res = await r.execute("import time; time.sleep(5)", timeout=0.3)
                    assert res.timed_out

        _run(go())

    @pytest.mark.skipif(not _has_jupyter_client, reason="jupyter_client not installed")
    @pytest.mark.slow
    def test_jupyter_repl_state_persists(self):
        from torchrl.envs.llm.agentic.repl import JupyterRepl

        async def go():
            async with UnsafeSubprocessSandbox(ResourceLimits(wall_seconds=60)) as s:
                async with JupyterRepl(s) as r:
                    r1 = await r.execute("x = 41", timeout=30)
                    assert r1.error is None, r1
                    r2 = await r.execute("print(x + 1)", timeout=30)
                    assert r2.error is None
                    assert r2.stdout.strip() == "42"

        _run(go())
