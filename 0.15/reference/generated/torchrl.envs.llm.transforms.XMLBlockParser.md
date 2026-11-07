# XMLBlockParser

*class*torchrl.envs.llm.transforms.XMLBlockParser[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#XMLBlockParser)

Parser for XML-style tool blocks in LLM responses.

Parses tool calls in the format:

<tool name="tool_name" tag="optional_tag">{"arg": "value"}</tool>

Examples

```
>>> parser = XMLBlockParser()
>>> response = '<tool name="search" tag="A">{"query": "torchrl"}</tool>\\nSome text.'
>>> result = parser(response)
>>> print(result["text"])
Some text.
>>> print(result["calls"][0].tool)
search
>>> print(result["calls"][0].args)
{"query": "torchrl"}
```