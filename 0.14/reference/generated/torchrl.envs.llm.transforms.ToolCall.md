# ToolCall

*class*torchrl.envs.llm.transforms.ToolCall(*tool: str*, *args: dict[str, Any]*, *tag: str | None = None*)[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#ToolCall)

Representation of a parsed tool call from LLM output.

Variables:

- **tool** (*str*) - The name of the tool to call.
- **args** (*dict**[**str**,**Any**]*) - Arguments to pass to the tool.
- **tag** (*str**|**None*) - Optional user-visible label or correlation ID.