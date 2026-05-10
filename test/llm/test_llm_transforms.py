# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util

import json

import pytest
from tensordict import set_list_to_stack, TensorDict

from torchrl.data.llm import History
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms import (
    ExecuteToolsInOrder,
    IncrementalTokenizer,
    JSONCallParser,
    ToolCall,
    ToolRegistry,
    XMLBlockParser,
)
from torchrl.envs.transforms import TransformedEnv

_has_transformers = importlib.util.find_spec("transformers") is not None


@pytest.fixture(scope="module")
def tokenizer():
    """Get a tokenizer for testing."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module", autouse=True)
def list_to_stack_fixture():
    with set_list_to_stack(True):
        yield


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""

        class DummyService:
            name = "dummy"
            schema_in = {"x": int}
            schema_out = {"y": int}

            def __call__(self, x, **kwargs):
                return {"y": x * 2}

        service = DummyService()
        registry = ToolRegistry([service])

        retrieved = registry.get("dummy")
        assert retrieved.name == "dummy"
        result = retrieved(x=5)
        assert result["y"] == 10

    def test_register_after_init(self):
        """Test registering services after initialization."""

        class ServiceA:
            name = "a"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                return {"result": "a"}

        class ServiceB:
            name = "b"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                return {"result": "b"}

        registry = ToolRegistry([ServiceA()])
        assert "a" in registry
        assert "b" not in registry

        registry.register(ServiceB())
        assert "b" in registry

    def test_unknown_tool_raises_error(self):
        """Test that requesting an unknown tool raises KeyError."""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool: nonexistent"):
            registry.get("nonexistent")


class TestXMLBlockParser:
    """Test the XMLBlockParser."""

    def test_parse_single_tool(self):
        """Test parsing a single tool call."""
        parser = XMLBlockParser()
        response = '<tool name="search">{"query": "test"}</tool>'
        result = parser(response)

        assert result["text"] == ""
        assert len(result["calls"]) == 1
        assert result["calls"][0].tool == "search"
        assert result["calls"][0].args == {"query": "test"}
        assert result["calls"][0].tag is None

    def test_parse_multiple_tools(self):
        """Test parsing multiple tool calls."""
        parser = XMLBlockParser()
        response = """Some text
<tool name="search" tag="A">{"query": "first"}</tool>
More text
<tool name="calculate">{"expr": "1+1"}</tool>
Final text"""
        result = parser(response)

        assert "Some text" in result["text"]
        assert "More text" in result["text"]
        assert "Final text" in result["text"]
        assert len(result["calls"]) == 2
        assert result["calls"][0].tool == "search"
        assert result["calls"][0].tag == "A"
        assert result["calls"][1].tool == "calculate"

    def test_parse_with_tag(self):
        """Test parsing with optional tag attribute."""
        parser = XMLBlockParser()
        response = '<tool name="test" tag="my_tag">{"a": 1}</tool>'
        result = parser(response)

        assert result["calls"][0].tag == "my_tag"

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in tool body."""
        parser = XMLBlockParser()
        response = '<tool name="test">{invalid json}</tool>'
        result = parser(response)

        # Should fall back to raw body
        assert len(result["calls"]) == 1
        assert "raw" in result["calls"][0].args

    def test_parse_empty_body(self):
        """Test parsing with empty tool body."""
        parser = XMLBlockParser()
        response = '<tool name="test"></tool>'
        result = parser(response)

        assert len(result["calls"]) == 1
        assert result["calls"][0].args == {}


class TestJSONCallParser:
    """Test the JSONCallParser."""

    def test_parse_dict_response(self):
        """Test parsing a dictionary response."""
        parser = JSONCallParser()
        response = {
            "message": "Here's the result",
            "tools": [{"tool": "search", "args": {"query": "test"}}],
        }
        result = parser(response)

        assert result["text"] == "Here's the result"
        assert len(result["calls"]) == 1
        assert result["calls"][0].tool == "search"

    def test_parse_json_string(self):
        """Test parsing a JSON string."""
        parser = JSONCallParser()
        response = json.dumps(
            {
                "message": "Processing",
                "tools": [
                    {"tool": "calc", "args": {"expr": "1+1"}},
                    {"tool": "search", "args": {"q": "test"}, "tag": "T1"},
                ],
            }
        )
        result = parser(response)

        assert len(result["calls"]) == 2
        assert result["calls"][1].tag == "T1"

    def test_parse_invalid_json_string(self):
        """Test handling of invalid JSON string."""
        parser = JSONCallParser()
        response = "Not valid JSON"
        result = parser(response)

        assert result["text"] == "Not valid JSON"
        assert len(result["calls"]) == 0

    def test_parse_no_tools(self):
        """Test parsing response with no tools."""
        parser = JSONCallParser()
        response = {"message": "Just a message"}
        result = parser(response)

        assert result["text"] == "Just a message"
        assert len(result["calls"]) == 0


class TestExecuteToolsInOrder:
    """Test the ExecuteToolsInOrder transform."""

    def test_basic_tool_execution(self):
        """Test basic tool execution with XML parser."""

        class AddService:
            name = "add"
            schema_in = {"a": int, "b": int}
            schema_out = {"result": int}

            def __call__(self, a, b, **kwargs):
                return {"result": a + b}

        registry = ToolRegistry([AddService()])
        parser = XMLBlockParser()

        env = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(env, ExecuteToolsInOrder(registry=registry, parser=parser))

        # Reset
        reset_data = TensorDict({"query": "Calculate something"}, batch_size=(1,))
        obs = env.reset(reset_data)

        # Simulate LLM response with tool call
        llm_response = '<tool name="add">{"a": 3, "b": 5}</tool>'
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=llm_response).view(1, 1), dim=-1
        )

        # Step
        next_obs = env.step(obs)

        # Check that tool result is in history
        final_history = next_obs[("next", "history")].prompt
        assert len(final_history[0]) > len(obs["history"].prompt[0])

        # Find the tool result message
        tool_result_found = False
        for msg in final_history[0]:
            if msg.role == "tool" and "result" in msg.content:
                tool_result_found = True
                assert "8" in msg.content or "result" in msg.content
                break
        assert tool_result_found

    def test_multiple_tools_in_order(self):
        """Test that multiple tools execute in order of appearance."""
        execution_order = []

        class FirstService:
            name = "first"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                execution_order.append("first")
                return {"executed": "first"}

        class SecondService:
            name = "second"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                execution_order.append("second")
                return {"executed": "second"}

        class ThirdService:
            name = "third"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                execution_order.append("third")
                return {"executed": "third"}

        registry = ToolRegistry([FirstService(), SecondService(), ThirdService()])
        parser = XMLBlockParser()

        env = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(env, ExecuteToolsInOrder(registry=registry, parser=parser))

        reset_data = TensorDict({"query": "Test"}, batch_size=(1,))
        obs = env.reset(reset_data)

        # Tools in specific order
        llm_response = """
<tool name="second">{}</tool>
<tool name="first">{}</tool>
<tool name="third">{}</tool>
"""
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=llm_response).view(1, 1), dim=-1
        )

        env.step(obs)

        # Check execution order matches appearance order
        assert execution_order == ["second", "first", "third"]

    def test_error_handling_continue(self):
        """Test error handling with stop_on_error=False."""

        class WorkingService:
            name = "working"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                return {"status": "ok"}

        class FailingService:
            name = "failing"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                raise ValueError("Intentional failure")

        registry = ToolRegistry([WorkingService(), FailingService()])
        parser = XMLBlockParser()

        env = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(
            env,
            ExecuteToolsInOrder(registry=registry, parser=parser, stop_on_error=False),
        )

        reset_data = TensorDict({"query": "Test"}, batch_size=(1,))
        obs = env.reset(reset_data)

        llm_response = """
<tool name="working">{}</tool>
<tool name="failing">{}</tool>
<tool name="working">{}</tool>
"""
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=llm_response).view(1, 1), dim=-1
        )

        # Should not raise, continues execution
        next_obs = env.step(obs)

        # Check that result contains both successes and failure
        final_history = next_obs[("next", "history")].prompt
        tool_msg = None
        for msg in final_history[0]:
            if msg.role == "tool":
                tool_msg = msg.content
                break

        assert tool_msg is not None
        assert "succeeded" in tool_msg
        assert "failed" in tool_msg or "error" in tool_msg.lower()

    def test_state_passing_to_tools(self):
        """Test that state is passed to tools when enabled."""
        received_state = {}

        class StateCheckService:
            name = "check"
            schema_in = {}
            schema_out = {}

            def __call__(self, _state=None, **kwargs):
                received_state["state"] = _state
                return {"received_state": _state is not None}

        registry = ToolRegistry([StateCheckService()])
        parser = XMLBlockParser()

        env = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(
            env,
            ExecuteToolsInOrder(
                registry=registry, parser=parser, pass_state_to_tools=True
            ),
        )

        reset_data = TensorDict({"query": "Check state"}, batch_size=(1,))
        obs = env.reset(reset_data)

        llm_response = '<tool name="check">{}</tool>'
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=llm_response).view(1, 1), dim=-1
        )

        env.step(obs)

        # Check that state was received
        assert received_state["state"] is not None
        assert isinstance(received_state["state"], dict)

    def test_no_tools_in_response(self):
        """Test behavior when response contains no tool calls."""

        class DummyService:
            name = "dummy"
            schema_in = {}
            schema_out = {}

            def __call__(self, **kwargs):
                return {}

        registry = ToolRegistry([DummyService()])
        parser = XMLBlockParser()

        env = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(env, ExecuteToolsInOrder(registry=registry, parser=parser))

        reset_data = TensorDict({"query": "Just chat"}, batch_size=(1,))
        obs = env.reset(reset_data)

        # No tool calls in response
        llm_response = "Just a normal response without tools."
        obs["history"].full = obs["history"].prompt.extend(
            History(role="assistant", content=llm_response).view(1, 1), dim=-1
        )

        # Should work fine without errors
        next_obs = env.step(obs)

        # History should not contain tool messages
        final_history = next_obs[("next", "history")].prompt
        for msg in final_history[0]:
            assert msg.role != "tool"


class TestToolCall:
    """Test the ToolCall dataclass."""

    def test_creation(self):
        """Test creating ToolCall instances."""
        call = ToolCall(tool="test", args={"x": 1})
        assert call.tool == "test"
        assert call.args == {"x": 1}
        assert call.tag is None

    def test_with_tag(self):
        """Test ToolCall with optional tag."""
        call = ToolCall(tool="test", args={}, tag="my_tag")
        assert call.tag == "my_tag"


class TestIncrementalTokenizer:
    """Tests for the IncrementalTokenizer transform."""

    def test_reset_tokenizes_history(self, tokenizer):
        """Test that reset produces correct tokens from history."""
        system_prompt = "You are a helpful assistant."
        user_query = "Hello, how are you?"

        env = ChatEnv(
            batch_size=(1,),
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
        env = TransformedEnv(env, IncrementalTokenizer(tokenizer))

        td = TensorDict({"query": user_query}, batch_size=(1,))
        result = env.reset(td)

        # Check that tokens.prompt exists
        assert ("tokens", "prompt") in result.keys(True, True)

        # Verify tokens are valid
        tokens = result.get(("tokens", "prompt"), as_list=True)
        assert len(tokens) == 1
        assert tokens[0].ndim == 1
        assert tokens[0].numel() > 0

        # Verify semantic correctness: decoded tokens should contain the expected content
        decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=False)
        assert system_prompt in decoded_text, (
            f"System prompt not found in decoded tokens. "
            f"Expected '{system_prompt}' in '{decoded_text}'"
        )
        assert user_query in decoded_text, (
            f"User query not found in decoded tokens. "
            f"Expected '{user_query}' in '{decoded_text}'"
        )

    def test_reset_tokenizes_batched_history(self, tokenizer):
        """Test that reset works with batched inputs."""
        system_prompt = "You are a helpful assistant."
        queries = ["Hello!", "What is the weather?"]

        env = ChatEnv(
            batch_size=(2,),
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
        env = TransformedEnv(env, IncrementalTokenizer(tokenizer))

        td = TensorDict({"query": queries}, batch_size=(2,))
        result = env.reset(td)

        # Check that tokens.prompt exists
        assert ("tokens", "prompt") in result.keys(True, True)

        # Verify we have tokens for each batch element
        tokens = result.get(("tokens", "prompt"), as_list=True)
        assert len(tokens) == 2

        # Verify semantic correctness: each batch element's tokens decode to correct content
        for i, query in enumerate(queries):
            decoded_text = tokenizer.decode(tokens[i], skip_special_tokens=False)
            assert (
                system_prompt in decoded_text
            ), f"Batch {i}: System prompt not found in decoded tokens"
            assert (
                query in decoded_text
            ), f"Batch {i}: Query '{query}' not found in decoded tokens. Got: '{decoded_text}'"

    def test_step_incremental_tokenization(self, tokenizer):
        """Test that step incrementally appends tokens and they decode correctly."""
        user_query = "Hello"
        assistant_response = "Hi there! How can I help you?"

        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
        )
        env = TransformedEnv(env, IncrementalTokenizer(tokenizer))

        # Reset with initial query
        td = TensorDict({"query": user_query}, batch_size=(1,))
        result = env.reset(td)
        initial_tokens = result.get(("tokens", "prompt"), as_list=True)

        # Verify initial tokens decode to contain the user query
        initial_decoded = tokenizer.decode(initial_tokens[0], skip_special_tokens=False)
        assert (
            user_query in initial_decoded
        ), f"User query '{user_query}' not found in initial tokens. Got: '{initial_decoded}'"

        # Simulate LLM response - create a full history with response
        # Need to match batch_dims: history_prompt has batch_dims=1, so response needs batch_dims=1 too
        history_prompt = result.get(("history", "prompt"))
        # Create response with batch_size=1 (message dimension), then add batch dimension
        history_response = History(
            role="assistant",
            content=assistant_response,
            batch_size=1,
        ).unsqueeze(
            0
        )  # Add batch dimension to match history_prompt
        history_full = history_prompt.extend(history_response, inplace=False, dim=-1)

        # Create action tensordict with the full history
        action_td = result.clone()
        action_td.set(("history", "full"), history_full)

        # Step the environment
        step_result = env.step(action_td)
        next_td = step_result["next"]

        # Verify tokens.prompt has been updated (next turn's prompt)
        assert ("tokens", "prompt") in next_td.keys(True, True)
        new_tokens = next_td.get(("tokens", "prompt"), as_list=True)

        # New tokens should be longer (more messages in the new prompt)
        assert new_tokens[0].numel() > initial_tokens[0].numel()

        # Verify semantic correctness: new tokens should decode to contain both
        # the original user query AND the assistant response
        new_decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
        assert user_query in new_decoded, (
            f"User query '{user_query}' not found in new tokens after step. "
            f"Got: '{new_decoded}'"
        )
        assert assistant_response in new_decoded, (
            f"Assistant response '{assistant_response}' not found in new tokens after step. "
            f"Got: '{new_decoded}'"
        )

    def test_with_tokenizer_constructor_arg(self, tokenizer):
        """Test that ChatEnv(with_tokenizer=True) creates a TransformedEnv with IncrementalTokenizer."""
        env = ChatEnv(
            tokenizer=tokenizer,
            batch_size=(1,),
            system_prompt="You are a helpful assistant.",
            with_tokenizer=True,
        )

        # Should be a TransformedEnv
        assert isinstance(env, TransformedEnv)

        # The transform should be an IncrementalTokenizer
        assert isinstance(env.transform, IncrementalTokenizer)

        # Test that it works
        td = TensorDict({"query": "Hello!"}, batch_size=(1,))
        result = env.reset(td)
        assert ("tokens", "prompt") in result.keys(True, True)

    def test_with_tokenizer_provides_tokens(self, tokenizer):
        """Test that with_tokenizer factory creates env that provides tokens."""
        env = ChatEnv.with_tokenizer(
            tokenizer=tokenizer,
            batch_size=(1,),
        )
        td = TensorDict({"query": "Test"}, batch_size=(1,))
        result = env.reset(td)
        assert ("tokens", "prompt") in result.keys(True, True)

    def test_tokens_consistency_across_multiple_steps(self, tokenizer):
        """Test that tokens remain consistent across multiple steps."""
        env = ChatEnv.with_tokenizer(
            tokenizer=tokenizer,
            batch_size=(1,),
        )

        # Reset
        td = TensorDict({"query": "What is 2+2?"}, batch_size=(1,))
        result = env.reset(td)
        history = result.get(("history", "prompt"))

        # First response - create with batch_size=1 (message dim), then add batch dimension
        response1 = History(
            role="assistant", content="The answer is 4.", batch_size=1
        ).unsqueeze(0)
        history_full1 = history.extend(response1, inplace=False, dim=-1)
        action_td = result.clone()
        action_td.set(("history", "full"), history_full1)
        step1 = env.step(action_td)
        next1 = step1["next"]
        tokens1 = next1.get(("tokens", "prompt"), as_list=True)[0]

        # Second message - need proper batch dimensions
        history2 = next1.get(("history", "prompt"))
        user2 = History(
            role="user", content="And what is 3+3?", batch_size=1
        ).unsqueeze(0)
        history_with_user2 = history2.extend(user2, inplace=False, dim=-1)
        response2 = History(
            role="assistant", content="That would be 6.", batch_size=1
        ).unsqueeze(0)
        history_full2 = history_with_user2.extend(response2, inplace=False, dim=-1)

        # We need to manually update history.prompt to include user2
        action_td2 = next1.clone()
        action_td2.set(("history", "prompt"), history_with_user2)
        action_td2.set(("history", "full"), history_full2)
        step2 = env.step(action_td2)
        next2 = step2["next"]
        tokens2 = next2.get(("tokens", "prompt"), as_list=True)[0]

        # Tokens should keep growing
        assert tokens2.numel() > tokens1.numel()

        # Verify the content makes sense by decoding
        decoded = tokenizer.decode(tokens2, skip_special_tokens=True)
        assert "2+2" in decoded or "2 + 2" in decoded
        assert "3+3" in decoded or "3 + 3" in decoded

    def test_incremental_tokenizer_spec_transform(self, tokenizer):
        """Test that IncrementalTokenizer correctly transforms observation spec."""
        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
        )
        transform = IncrementalTokenizer(tokenizer)
        env = TransformedEnv(env, transform)

        # The spec should include tokens.prompt
        obs_spec = env.observation_spec
        assert ("tokens", "prompt") in obs_spec.keys(True, True)

    def test_custom_keys(self, tokenizer):
        """Test that custom history_key and tokens_key work."""
        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
        )
        transform = IncrementalTokenizer(
            tokenizer,
            history_key=("history", "prompt"),
            tokens_key=("my_tokens", "all"),
        )
        env = TransformedEnv(env, transform)

        td = TensorDict({"query": "Hello"}, batch_size=(1,))
        result = env.reset(td)

        # Should use custom tokens key
        assert ("my_tokens", "all") in result.keys(True, True)

    def test_empty_history_handling(self, tokenizer):
        """Test handling of minimal history (just user message)."""
        env = ChatEnv(
            batch_size=(1,),
            tokenizer=tokenizer,
            # No system prompt
        )
        env = TransformedEnv(env, IncrementalTokenizer(tokenizer))

        td = TensorDict({"query": "Hi"}, batch_size=(1,))
        result = env.reset(td)

        # Should still produce valid tokens
        assert ("tokens", "prompt") in result.keys(True, True)
        tokens = result.get(("tokens", "prompt"), as_list=True)
        assert tokens[0].numel() > 0


# ---------------------------------------------------------------------------
# Agentic toolkit (torchrl.envs.llm.agentic)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

from torchrl.envs.llm.agentic import (  # noqa: E402
    ParsedCall,
    TextPart,
    Tool,
    ToolCallParser,
    ToolContext,
    ToolResult,
    validate_args,
)
from torchrl.envs.llm.agentic.parsers import (  # noqa: E402
    AnthropicToolUseParser,
    JSONToolCallParser,
    OpenAIToolCallParser,
    XMLToolCallParser,
)
from torchrl.envs.llm.agentic.repl import (  # noqa: E402
    SubprocessRepl,
    _has_jupyter_client,
)
from torchrl.envs.llm.agentic.sandbox import (  # noqa: E402
    BubblewrapSandbox,
    ResourceLimits,
    SandboxError,
    SeatbeltSandbox,
    UnsafeSubprocessSandbox,
    default_sandbox,
)
from torchrl.envs.llm.agentic.sandbox.subprocess_bwrap import _has_bwrap  # noqa: E402
from torchrl.envs.llm.agentic.sandbox.subprocess_seatbelt import (  # noqa: E402
    _has_sandbox_exec,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestAgenticParsers:
    """Per-parser conformance: parse, render_call round-trip, render_result,
    stable call_id (parser-supplied or assigned).
    """

    @pytest.mark.parametrize(
        "parser_cls",
        [XMLToolCallParser, JSONToolCallParser, OpenAIToolCallParser, AnthropicToolUseParser],
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
        call = ParsedCall(
            tool="echo", args={"text": "hi"}, call_id="abc", tag="abc"
        )
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
        msg = p.render_result(
            "toolu_a", ToolResult.from_text("hit", is_error=False)
        )
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
            assert any(
                issubclass(w.category, UserWarning) for w in caught
            )

        _run(go())

    def test_unsafe_runs_simple_command(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=5)
            ) as s:
                r = await s.run(["/bin/echo", "hello"])
                assert r.exit_code == 0
                assert r.stdout.strip() == "hello"
                assert not r.timed_out

        _run(go())

    def test_unsafe_timeout(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=0.2)
            ) as s:
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
                r = await s.run(
                    ["/bin/sh", "-c", f"echo hi > {inside_path}"]
                )
                assert r.exit_code == 0
                r2 = await s.run(
                    ["/bin/sh", "-c", f"echo nope > {outside}"]
                )
                assert r2.exit_code != 0
                assert not outside.exists()

        _run(go())


class TestAgenticRepl:
    """REPL state, error capture, restart, timeout."""

    def test_subprocess_repl_state_persists(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                async with SubprocessRepl(s) as r:
                    r1 = await r.execute("x = 41")
                    assert r1.error is None
                    r2 = await r.execute("print(x + 1)")
                    assert r2.error is None
                    assert r2.stdout.strip() == "42"

        _run(go())

    def test_subprocess_repl_captures_errors(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                async with SubprocessRepl(s) as r:
                    res = await r.execute("1/0")
                    assert res.error is not None
                    assert res.error.ename == "ZeroDivisionError"

        _run(go())

    def test_subprocess_repl_restart_clears_state(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                async with SubprocessRepl(s) as r:
                    await r.execute("y = 99")
                    await r.restart()
                    res = await r.execute("print(y)")
                    assert res.error is not None  # NameError

        _run(go())

    def test_subprocess_repl_timeout(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                async with SubprocessRepl(s) as r:
                    res = await r.execute(
                        "import time; time.sleep(5)", timeout=0.3
                    )
                    assert res.timed_out

        _run(go())

    @pytest.mark.skipif(
        not _has_jupyter_client, reason="jupyter_client not installed"
    )
    @pytest.mark.slow
    def test_jupyter_repl_state_persists(self):
        from torchrl.envs.llm.agentic.repl import JupyterRepl

        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=60)
            ) as s:
                async with JupyterRepl(s) as r:
                    r1 = await r.execute("x = 41", timeout=30)
                    assert r1.error is None, r1
                    r2 = await r.execute("print(x + 1)", timeout=30)
                    assert r2.error is None
                    assert r2.stdout.strip() == "42"

        _run(go())


# ----- ToolCompose, builtins, legacy adapter -----

import time as _time  # noqa: E402

from typing import ClassVar  # noqa: E402

from torchrl.envs.llm.agentic import (  # noqa: E402
    PythonTool,
    RateLimiter,
    ShellTool,
    StopTool,
    ToolCompose,
)
from torchrl.envs.llm.agentic.tools import as_tool  # noqa: E402


class _Sleeper:
    description: ClassVar[str] = "sleep N ms"
    input_schema = {
        "type": "object",
        "properties": {"ms": {"type": "integer"}},
    }
    output_schema = None
    wants_state = False

    def __init__(self, name):
        self.name = name

    async def setup(self):
        pass

    async def teardown(self):
        pass

    async def run(self, args, ctx):
        await asyncio.sleep(args.get("ms", 100) / 1000)
        return ToolResult.from_text(f"{self.name}-done")


class _Stateful:
    name: ClassVar[str] = "stateful"
    description: ClassVar[str] = "needs state"
    input_schema = {"type": "object"}
    output_schema = None
    wants_state = True
    received_state = None

    async def setup(self):
        pass

    async def teardown(self):
        pass

    async def run(self, args, ctx):
        type(self).received_state = ctx.state
        return ToolResult.from_text("ok")


class _Boom:
    name: ClassVar[str] = "boom"
    description = "always fails"
    input_schema = {"type": "object"}
    output_schema = None
    wants_state = False

    async def setup(self):
        pass

    async def teardown(self):
        pass

    async def run(self, args, ctx):
        raise RuntimeError("boom")


def _agentic_env(tools, parser=None):
    from tensordict import set_list_to_stack as _slts

    _slts(True).set()
    parser = parser or XMLToolCallParser()
    base = ChatEnv(batch_size=(1,), input_mode="history")
    return TransformedEnv(base, ToolCompose(tools=tools, parser=parser))


def _push_assistant(obs, response: str):
    obs["history"].full = obs["history"].prompt.extend(
        History(role="assistant", content=response).view(1, 1), dim=-1
    )


class TestToolCompose:
    def test_rejects_non_tool(self):
        with pytest.raises(TypeError):
            ToolCompose(
                tools=[object()], parser=XMLToolCallParser()
            )

    def test_rejects_duplicate_names(self):
        with pytest.raises(ValueError):
            ToolCompose(
                tools=[_Sleeper("dup"), _Sleeper("dup")],
                parser=XMLToolCallParser(),
            )

    def test_append_transform_blocked(self):
        compose = ToolCompose(
            tools=[StopTool()], parser=XMLToolCallParser()
        )
        with pytest.raises(TypeError):
            compose.append_transform(IncrementalTokenizer)

    def test_lookup_by_name(self):
        compose = ToolCompose(
            tools=[StopTool()], parser=XMLToolCallParser()
        )
        assert "stop" in compose
        assert compose["stop"].name == "stop"

    def test_parallel_dispatch_wall_time(self):
        env = _agentic_env(
            [_Sleeper("a"), _Sleeper("b"), _Sleeper("c")]
        )
        obs = env.reset(TensorDict({"query": "go"}, batch_size=(1,)))
        _push_assistant(
            obs,
            '<tool name="a" tag="1">{"ms": 500}</tool>'
            '<tool name="b" tag="2">{"ms": 500}</tool>'
            '<tool name="c" tag="3">{"ms": 500}</tool>',
        )
        t0 = _time.monotonic()
        nxt = env.step(obs)
        elapsed = _time.monotonic() - t0
        # Three 500ms tools must run concurrently: total < 0.8s.
        assert elapsed < 0.9, (
            f"parallel dispatch took {elapsed:.2f}s; expected < 0.8s"
        )
        assert bool(nxt.get(("next", "agentic", "any_tool_calls")).item())
        assert not bool(
            nxt.get(("next", "agentic", "stop_requested")).item()
        )

    def test_stop_tool_terminates(self):
        env = _agentic_env([StopTool()])
        obs = env.reset(TensorDict({"query": "stop"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="stop">{"reason":"done"}</tool>')
        nxt = env.step(obs)
        assert bool(
            nxt.get(("next", "agentic", "stop_requested")).item()
        )

    def test_no_tool_calls_passthrough(self):
        env = _agentic_env([StopTool()])
        obs = env.reset(TensorDict({"query": "nothing"}, batch_size=(1,)))
        _push_assistant(obs, "I have nothing to call.")
        nxt = env.step(obs)
        assert not bool(
            nxt.get(("next", "agentic", "any_tool_calls")).item()
        )

    def test_unknown_tool_reports_error(self):
        env = _agentic_env([StopTool()])
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="ghost" tag="1">{}</tool>')
        nxt = env.step(obs)
        assert bool(nxt.get(("next", "agentic", "any_error")).item())

    def test_failure_isolation(self):
        env = _agentic_env([_Sleeper("a"), _Boom()])
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(
            obs,
            '<tool name="a" tag="1">{"ms": 5}</tool>'
            '<tool name="boom" tag="2">{}</tool>',
        )
        nxt = env.step(obs)
        # both calls fired; one failed, one succeeded.
        assert bool(nxt.get(("next", "agentic", "any_error")).item())
        assert bool(
            nxt.get(("next", "agentic", "any_tool_calls")).item()
        )
        # The history should have both tool messages appended.
        prompt = nxt[("next", "history")].prompt
        # Assistant message + 2 tool messages = at least 3 entries beyond
        # the original prompt.
        assert len(prompt[0]) >= 3

    def test_stable_call_id_round_trip(self):
        captured: list[str] = []

        class _Recorder:
            name: ClassVar[str] = "rec"
            description = "records call_id"
            input_schema = {"type": "object"}
            output_schema = None
            wants_state = False

            async def setup(self):
                pass

            async def teardown(self):
                pass

            async def run(self, args, ctx):
                captured.append(ctx.call_id)
                return ToolResult.from_text(f"id={ctx.call_id}")

        env = _agentic_env([_Recorder()])
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(
            obs,
            '<tool name="rec" tag="my-id">{}</tool>',
        )
        nxt = env.step(obs)
        assert captured == ["my-id"]
        # The rendered tool message must reference the same call_id.
        prompt = nxt[("next", "history")].prompt
        last_msg = prompt[0][-1]
        assert "my-id" in last_msg.content

    def test_pass_state_to_tools(self):
        _Stateful.received_state = None
        from tensordict import set_list_to_stack as _slts

        _slts(True).set()
        base = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(
            base,
            ToolCompose(
                tools=[_Stateful()],
                parser=XMLToolCallParser(),
                pass_state_to_tools=True,
            ),
        )
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="stateful">{}</tool>')
        env.step(obs)
        assert _Stateful.received_state is not None

    def test_pass_state_off_means_no_state(self):
        _Stateful.received_state = None
        env = _agentic_env([_Stateful()])  # pass_state_to_tools defaults False
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="stateful">{}</tool>')
        env.step(obs)
        assert _Stateful.received_state is None

    def test_rate_limit_serializes_concurrent_calls(self):
        # max_concurrent=1 forces 3 calls of 200ms each to take >= 600ms.
        slow = _Sleeper("slow")
        compose = ToolCompose(
            tools=[slow],
            parser=XMLToolCallParser(),
            rate_limits={"slow": RateLimiter(max_concurrent=1)},
        )
        from tensordict import set_list_to_stack as _slts

        _slts(True).set()
        base = ChatEnv(batch_size=(1,), input_mode="history")
        env = TransformedEnv(base, compose)
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(
            obs,
            '<tool name="slow" tag="1">{"ms": 200}</tool>'
            '<tool name="slow" tag="2">{"ms": 200}</tool>'
            '<tool name="slow" tag="3">{"ms": 200}</tool>',
        )
        t0 = _time.monotonic()
        env.step(obs)
        elapsed = _time.monotonic() - t0
        assert elapsed >= 0.55, (
            f"rate-limited dispatch should serialize: got {elapsed:.2f}s"
        )

    def test_argument_validation(self):
        class _NeedsCode:
            name: ClassVar[str] = "needs"
            description = ""
            input_schema = {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            }
            output_schema = None
            wants_state = False

            async def setup(self):
                pass

            async def teardown(self):
                pass

            async def run(self, args, ctx):  # pragma: no cover - never reached
                return ToolResult.from_text("hit")

        env = _agentic_env([_NeedsCode()])
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="needs">{}</tool>')  # missing required
        nxt = env.step(obs)
        assert bool(nxt.get(("next", "agentic", "any_error")).item())

    def test_nested_loop_safety(self):
        # When the caller already owns an event loop, ToolCompose._step
        # must still complete (offload to a worker thread).
        async def go():
            env = _agentic_env([StopTool()])
            obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
            _push_assistant(obs, '<tool name="stop">{}</tool>')
            return env.step(obs)

        nxt = _run(go())
        assert bool(
            nxt.get(("next", "agentic", "stop_requested")).item()
        )


class TestPythonTool:
    def test_state_persists_across_calls(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                tool = PythonTool(repl=SubprocessRepl(s))
                await tool.setup()
                ctx = ToolContext(call_id="c1")
                r1 = await tool.run({"code": "x = 41"}, ctx)
                assert not r1.is_error
                r2 = await tool.run({"code": "print(x + 1)"}, ctx)
                assert not r2.is_error
                assert "42" in r2.text
                await tool.teardown()

        _run(go())

    def test_error_marked_is_error(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=10)
            ) as s:
                tool = PythonTool(repl=SubprocessRepl(s))
                await tool.setup()
                r = await tool.run({"code": "1/0"}, ToolContext(call_id="c"))
                assert r.is_error
                assert "ZeroDivisionError" in r.text
                await tool.teardown()

        _run(go())


class TestShellTool:
    def test_runs_argv(self):
        async def go():
            async with UnsafeSubprocessSandbox(
                ResourceLimits(wall_seconds=5)
            ) as s:
                tool = ShellTool(s)
                await tool.setup()
                r = await tool.run(
                    {"argv": ["/bin/echo", "hi"]}, ToolContext(call_id="c")
                )
                assert not r.is_error
                assert "hi" in r.text
                # Don't tear down s twice -- ShellTool teardown closes it.

        _run(go())


class TestLegacyAdapter:
    def test_lifts_legacy_transform(self):
        # Use a tiny duck-typed legacy class instead of pulling in the
        # full PythonInterpreter, which has its own subprocess pool.
        class _LegacyAdder:
            tool_role = "tool"

            def _process_batch_item(self, content, index):
                # Echo the captured XML so the assertion can find it.
                if "<tool name=" in content:
                    return [f"legacy got: {content}"]
                return None

        tool = as_tool(
            _LegacyAdder(),
            name="adder",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"},
                               "b": {"type": "integer"}},
            },
        )
        assert tool.name == "adder"
        env = _agentic_env([tool])
        obs = env.reset(TensorDict({"query": "?"}, batch_size=(1,)))
        _push_assistant(obs, '<tool name="adder">{"a": 1, "b": 2}</tool>')
        nxt = env.step(obs)
        prompt = nxt[("next", "history")].prompt
        # The last appended message should be the tool result containing
        # the legacy output.
        assert "legacy got" in prompt[0][-1].content
