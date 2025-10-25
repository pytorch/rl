#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example demonstrating the ExecuteToolsInOrder transform with pluggable services.

This example shows how to:
1. Define tool services with schemas
2. Register them in a ToolRegistry
3. Use different parsers (XML and JSON style)
4. Integrate with ChatEnv and LLM wrappers
"""

from tensordict import TensorDict
from torchrl.data.llm import History
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms import (
    ExecuteToolsInOrder,
    JSONCallParser,
    ToolRegistry,
    XMLBlockParser,
)
from torchrl.envs.transforms import TransformedEnv


# --- Define Example Tool Services ---


class WebSearchService:
    """Example web search service."""

    name = "search"
    schema_in = {"query": str}
    schema_out = {"results": list}

    def __call__(self, query: str, _state=None, **kwargs):
        """Simulate a web search."""
        # In a real implementation, this would call an actual search API
        return {
            "results": [
                {
                    "title": "TorchRL Documentation",
                    "url": "https://pytorch.org/rl/",
                    "snippet": "TorchRL is a PyTorch library for reinforcement learning",
                },
                {
                    "title": f"Results for: {query}",
                    "url": "https://example.com",
                    "snippet": f"Search results for query: {query}",
                },
            ]
        }


class SummarizeService:
    """Example text summarization service."""

    name = "summarize"
    schema_in = {"text": str, "max_length": int}
    schema_out = {"summary": str}

    def __call__(self, text: str, max_length: int = 200, _state=None, **kwargs):
        """Summarize text to max_length characters."""
        if len(text) <= max_length:
            return {"summary": text}
        return {"summary": text[:max_length] + "..."}


class CalculatorService:
    """Example calculator service."""

    name = "calculate"
    schema_in = {"expression": str}
    schema_out = {"result": float}

    def __call__(self, expression: str, _state=None, **kwargs):
        """Safely evaluate a mathematical expression."""
        try:
            # Only allow safe math operations
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            result = eval(expression)
            return {"result": float(result)}
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")


# --- Example 1: Using XML-style parser ---


def example_xml_parser():
    """Demonstrate using XML-style tool blocks."""
    print("\n" + "=" * 60)
    print("Example 1: XML-style Parser")
    print("=" * 60 + "\n")

    # Create registry with services
    registry = ToolRegistry(
        [WebSearchService(), SummarizeService(), CalculatorService()]
    )

    # Create parser
    parser = XMLBlockParser()

    # Create environment
    env = ChatEnv(
        batch_size=(1,),
        system_prompt="You are a helpful assistant with access to tools.",
        input_mode="history",
    )

    # Add transform
    env = TransformedEnv(env, ExecuteToolsInOrder(registry=registry, parser=parser))

    # Reset with initial query
    reset_data = TensorDict({"query": ["Hello, can you help me?"]}, batch_size=(1,))
    obs = env.reset(reset_data)

    # Simulate LLM response with tool calls
    llm_response = """I'll search for information about TorchRL and calculate something.

<tool name="search" tag="info">{"query": "TorchRL transforms"}</tool>

<tool name="calculate" tag="math">{"expression": "42 * 3.14"}</tool>

Let me process these results for you."""

    # Create response tensordict
    history = obs["history"].prompt
    response = History(role="assistant", content=llm_response).view(1, 1)
    full = history.extend(
        response,
        dim=-1,
        inplace=False,
    )
    obs["history"].full = full
    obs["history"].response = response

    # Step with the response
    next_obs = env.step(obs)

    # Print the conversation
    print("Final conversation history:")
    final_history = next_obs[("next", "history")].prompt
    for i, msg in enumerate(final_history[0]):
        print(f"\n[{i}] Role: {msg.role}")
        print(f"Content: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")


# --- Example 2: Using JSON-style parser ---


def example_json_parser():
    """Demonstrate using JSON-style function calling."""
    print("\n" + "=" * 60)
    print("Example 2: JSON-style Parser")
    print("=" * 60 + "\n")

    # Create registry with services
    registry = ToolRegistry([WebSearchService(), CalculatorService()])

    # Create JSON parser
    parser = JSONCallParser()

    # Create environment
    env = ChatEnv(
        batch_size=(1,),
        system_prompt="You are a helpful assistant.",
        input_mode="history",
    )

    # Add transform
    env = TransformedEnv(env, ExecuteToolsInOrder(registry=registry, parser=parser))

    # Reset with initial query
    reset_data = TensorDict({"query": ["Calculate 15 + 27"]}, batch_size=(1,))
    obs = env.reset(reset_data)

    # Simulate LLM response in JSON format
    # Note: In practice, the LLM would generate this
    import json

    llm_response_dict = {
        "message": "I'll calculate that for you.",
        "tools": [{"tool": "calculate", "args": {"expression": "15 + 27"}}],
    }

    # For the JSON parser, we need to pass the content as JSON string
    llm_response = json.dumps(llm_response_dict)

    # Create response tensordict
    history = obs["history"].prompt
    response = History(role="assistant", content=llm_response).view(1, 1)
    full = history.extend(
        response,
        dim=-1,
        inplace=False,
    )
    obs["history"].full = full
    obs["history"].response = response

    # Step with the response
    next_obs = env.step(obs)

    # Print the conversation
    print("Final conversation history:")
    final_history = next_obs[("next", "history")].prompt
    for i, msg in enumerate(final_history[0]):
        print(f"\n[{i}] Role: {msg.role}")
        print(f"Content: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")


# --- Example 3: State passing to tools ---


class StateAwareService:
    """Example service that uses the environment state."""

    name = "check_step"
    schema_in = {}
    schema_out = {"message": str}

    def __call__(self, _state=None, **kwargs):
        """Return information about the current environment state."""
        if _state:
            step_info = _state.get("env/step", "unknown")
            return {
                "message": f"Current environment state: step={step_info}",
                "state_keys": list(_state.keys()),
            }
        return {"message": "No state information available"}


def example_state_passing():
    """Demonstrate state passing to tools."""
    print("\n" + "=" * 60)
    print("Example 3: State Passing to Tools")
    print("=" * 60 + "\n")

    # Create registry with state-aware service
    registry = ToolRegistry([StateAwareService()])
    parser = XMLBlockParser()

    # Create environment
    env = ChatEnv(batch_size=(1,), input_mode="history")

    # Add transform with state passing enabled (default)
    env = TransformedEnv(
        env,
        ExecuteToolsInOrder(registry=registry, parser=parser, pass_state_to_tools=True),
    )

    # Reset and step
    reset_data = TensorDict({"query": ["Check the state"]}, batch_size=(1,))
    obs = env.reset(reset_data)

    llm_response = '<tool name="check_step">{}</tool>'
    history = obs["history"].prompt
    response = History(role="assistant", content=llm_response).view(1, 1)
    full = history.extend(
        response,
        dim=-1,
        inplace=False,
    )
    obs["history"].full = full
    obs["history"].response = response

    next_obs = env.step(obs)

    print("Tool received state information:")
    final_history = next_obs[("next", "history")].prompt
    print(final_history[0][-1].content)


# --- Example 4: Error handling ---


def example_error_handling():
    """Demonstrate error handling with stop_on_error."""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60 + "\n")

    # Create registry
    registry = ToolRegistry([CalculatorService()])
    parser = XMLBlockParser()

    # Create environment with stop_on_error=True
    env = ChatEnv(batch_size=(1,), input_mode="history")
    env = TransformedEnv(
        env,
        ExecuteToolsInOrder(registry=registry, parser=parser, stop_on_error=False),
    )

    # Reset
    reset_data = TensorDict({"query": ["Do some calculations"]}, batch_size=(1,))
    obs = env.reset(reset_data)

    # Response with both valid and invalid calculations
    llm_response = """Let me calculate:

<tool name="calculate">{"expression": "2 + 2"}</tool>
<tool name="calculate">{"expression": "invalid + syntax"}</tool>
<tool name="calculate">{"expression": "5 * 5"}</tool>"""

    history = obs["history"].prompt
    response = History(role="assistant", content=llm_response).view(1, 1)
    full = history.extend(
        response,
        dim=-1,
        inplace=False,
    )
    obs["history"].full = full
    obs["history"].response = response

    next_obs = env.step(obs)

    print("Results (note: execution continues after error):")
    final_history = next_obs[("next", "history")].prompt
    for msg in final_history[0]:
        if msg.role == "tool":
            print(msg.content)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Tool Service Transform Examples")
    print("=" * 60)

    try:
        example_xml_parser()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_json_parser()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_state_passing()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_error_handling()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")
