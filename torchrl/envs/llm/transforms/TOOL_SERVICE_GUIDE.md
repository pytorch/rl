# Tool Service Transform Guide

This guide explains the `ExecuteToolsInOrder` transform and the tool service library for TorchRL's LLM environments.

## Overview

The tool service library provides a clean, extensible way to add tool-calling capabilities to LLM agents in TorchRL. It consists of:

1. **`ToolService` Protocol** - Defines the interface for tool services
2. **`ToolRegistry`** - Manages available tool services
3. **`LLMToolParser` Protocol** - Defines how to parse tool calls from LLM output
4. **`ExecuteToolsInOrder` Transform** - Executes tools in the order they appear in LLM responses

## Key Features

- **Order-preserving execution**: Tools execute in the order they appear in the LLM output, not the order of transforms in the stack
- **Pluggable parsers**: Support for different tool-calling formats (XML, JSON, custom)
- **Clean separation of concerns**: Services, registry, parser, and execution are all independent
- **State passing**: Optional filtered TD state access for tools
- **Error handling**: Configurable fail-fast or continue-on-error behavior
- **Protocol-based**: Easy to extend with custom services and parsers

## Basic Usage

### 1. Define Tool Services

```python
class WebSearchService:
    """Example web search service."""
    name = "search"
    schema_in = {"query": str}
    schema_out = {"results": list}
    
    def __call__(self, query: str, _state=None, **kwargs):
        # Implement your search logic
        return {"results": [...]}

class CalculatorService:
    """Example calculator service."""
    name = "calculate"
    schema_in = {"expression": str}
    schema_out = {"result": float}
    
    def __call__(self, expression: str, _state=None, **kwargs):
        # Implement calculation logic
        return {"result": eval(expression)}
```

### 2. Create Registry and Parser

```python
from torchrl.envs.llm.transforms import (
    ExecuteToolsInOrder,
    ToolRegistry,
    XMLBlockParser,
)

# Register your services
registry = ToolRegistry([
    WebSearchService(),
    CalculatorService(),
])

# Choose a parser (XML or JSON style)
parser = XMLBlockParser()
```

### 3. Add Transform to Environment

```python
from torchrl.envs.llm import ChatEnv
from torchrl.envs.transforms import TransformedEnv

env = ChatEnv(
    batch_size=(1,),
    system_prompt="You are a helpful assistant with access to tools.",
    input_mode="history",
)

env = TransformedEnv(
    env,
    ExecuteToolsInOrder(
        registry=registry,
        parser=parser,
        stop_on_error=False,
        pass_state_to_tools=True,
    )
)
```

### 4. Use in Your LLM Loop

The transform automatically:
- Intercepts LLM responses
- Parses tool calls using the provided parser
- Executes tools in order via the registry
- Injects results back into the conversation history

```python
# The LLM generates responses with tool calls
llm_response = '''I'll search for information about TorchRL.

<tool name="search">{"query": "TorchRL transforms"}</tool>

Let me process those results for you.'''

# The transform automatically handles tool execution
# and adds results to the conversation history
```

## Parsers

### XMLBlockParser

Parses XML-style tool blocks:

```xml
<tool name="search" tag="optional_tag">{"query": "test"}</tool>
```

Features:
- Optional `tag` attribute for correlation IDs
- JSON body for arguments
- Removes tool blocks from user-facing text

### JSONCallParser

Parses JSON-style function calls:

```json
{
  "message": "Let me search for that.",
  "tools": [
    {"tool": "search", "args": {"query": "test"}, "tag": "A"}
  ]
}
```

Features:
- Structured format common in function-calling APIs
- Clean separation of message and tool calls
- Native support for ordered tool lists

### Custom Parsers

Implement the `LLMToolParser` protocol:

```python
class CustomParser:
    def __call__(self, response: str | dict[str, Any]) -> ParseResult:
        # Your parsing logic
        result = ParseResult()
        result["text"] = "cleaned message"
        result["calls"] = [ToolCall(tool="name", args={...})]
        result["meta"] = {"count": 1}
        return result
```

## Advanced Features

### State Passing

Tools can access environment state when `pass_state_to_tools=True`:

```python
class StateAwareService:
    name = "check_state"
    schema_in = {}
    schema_out = {"info": dict}
    
    def __call__(self, _state=None, **kwargs):
        # _state contains filtered TD keys
        step = _state.get("env/step", 0)
        return {"info": {"current_step": step}}
```

The transform exports a filtered view of the TensorDict to tools. You can customize this by overriding `_export_state_for_tool()`.

### Error Handling

```python
env = TransformedEnv(
    env,
    ExecuteToolsInOrder(
        registry=registry,
        parser=parser,
        stop_on_error=False,  # Continue even if tools fail
    )
)
```

- `stop_on_error=False` (default): Continues executing remaining tools even if one fails
- `stop_on_error=True`: Stops execution on first error

Errors are captured and included in tool results with an `"error"` key.

### Custom Keys

All input/output keys are configurable:

```python
ExecuteToolsInOrder(
    registry=registry,
    parser=parser,
    in_keys=("custom", "input"),          # Where to read LLM response
    out_keys=("custom", "output"),        # Where to write results
    message_key=("custom", "message"),    # Cleaned message text
    history_key=("custom", "history"),    # Conversation history
    write_calls_key=("custom", "calls"),  # Parsed tool calls
)
```

## Integration with Transforms

The `ExecuteToolsInOrder` transform integrates seamlessly with other TorchRL transforms:

```python
from torchrl.envs.transforms import Compose, StepCounter
from torchrl.envs.llm.transforms import KLRewardTransform, AddThinkingPrompt

env = TransformedEnv(
    base_env,
    Compose(
        StepCounter(),
        AddThinkingPrompt(),
        ExecuteToolsInOrder(registry=registry, parser=parser),
        KLRewardTransform(actor),
    )
)
```

Tool execution order is determined by appearance in the LLM output, not by transform position in the `Compose` stack.

## Design Rationale

### Why Protocols?

Using `Protocol` for `ToolService` and `LLMToolParser` provides:
- Duck typing - no need to inherit from base classes
- Clear contracts - explicit method signatures
- Easy testing - mock services trivially
- Flexibility - any class with the right interface works

### Why Order-Preserving?

LLMs may emit tools in a specific order for good reasons:
1. Data dependencies (search before summarize)
2. Logical flow (calculate then display)
3. Error recovery (try primary then fallback)

Executing in appearance order respects the LLM's reasoning.

### Why Separate Parser?

Different LLMs use different tool-calling formats:
- OpenAI: JSON function calls
- Anthropic: XML-style blocks
- Custom: Your own format

Pluggable parsers let you support all of them with the same transform.

## Examples

See `examples/llm/tool_service_example.py` for complete working examples including:
- XML-style parsing
- JSON-style parsing
- State passing to tools
- Error handling
- Multiple tools in sequence

## Testing

Tests are in `test/llm/test_transforms.py`. Run with:

```bash
pytest test/llm/test_llm_transforms.py -v
```

## Comparison with MCPToolTransform

The `ExecuteToolsInOrder` transform is a more general, better-architected version of `MCPToolTransform`:

| Feature | MCPToolTransform | ExecuteToolsInOrder |
|---------|------------------|---------------------|
| Parser | Hardcoded XML | Pluggable (XML, JSON, custom) |
| Services | Dict of callables | Protocol-based ToolRegistry |
| State access | No | Yes (optional, filtered) |
| Key configuration | Limited | All keys configurable |
| Error handling | Basic | Configurable (stop or continue) |
| Extensibility | Moderate | High (protocols everywhere) |

`MCPToolTransform` is still available for backward compatibility, but new code should use `ExecuteToolsInOrder`.

## Future Extensions

Potential future enhancements:
- Async tool execution
- Parallel tool execution (when order doesn't matter)
- Tool result validation against schemas
- Automatic prompt generation from tool schemas
- Streaming tool results
- Tool execution timeouts per-tool
- Rate limiting for API-based tools

## References

- [TorchRL Transform Documentation](https://pytorch.org/rl/main/reference/generated/torchrl.envs.transforms.Transform.html)
- [TorchRL LLM Interface](https://pytorch.org/rl/main/reference/llms.html)
- [TransformedEnv](https://pytorch.org/rl/main/reference/generated/torchrl.envs.transforms.TransformedEnv.html)

