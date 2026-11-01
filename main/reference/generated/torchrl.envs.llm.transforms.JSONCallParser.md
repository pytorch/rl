# JSONCallParser

*class*torchrl.envs.llm.transforms.JSONCallParser[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#JSONCallParser)

Parser for JSON-style function-calling responses.

Expects responses in the format:

```
{
 "message": "...",
 "tools": [
 {"tool": "search", "args": {"query": "..."}, "tag": "A"},
 {"tool": "summarize", "args": {"text": "..."}}
 ]
}
```

Examples

```
>>> parser = JSONCallParser()
>>> response = {
... "message": "Let me search for that.",
... "tools": [{"tool": "search", "args": {"query": "torchrl"}}]
... }
>>> result = parser(response)
>>> print(result["text"])
Let me search for that.
>>> print(result["calls"][0].tool)
search
```