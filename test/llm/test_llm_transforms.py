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
    IncrementalTokenizer,
    JSONCallParser,
    ToolCall,
    ToolRegistry,
    XMLBlockParser,
)
from torchrl.envs.transforms import TransformedEnv


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
