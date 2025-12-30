# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest
from tensordict import set_list_to_stack, TensorDict

from torchrl.data.llm import History
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms import (
    ExecuteToolsInOrder,
    JSONCallParser,
    ToolCall,
    ToolRegistry,
    XMLBlockParser,
)
from torchrl.envs.transforms import TransformedEnv


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
