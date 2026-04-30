# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute Python code using MCP server with mcp-run-python."""

import json
import os

from tensordict import set_list_to_stack, TensorDict

from torchrl.data.llm import History
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.transforms import MCPToolTransform

set_list_to_stack(True).set()

deno_path = os.path.expanduser("~/.deno/bin")
if deno_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{deno_path}:{os.environ['PATH']}"

servers = {
    "python": {
        "command": "uvx",
        "args": ["mcp-run-python", "stdio"],
        "env": os.environ.copy(),
    }
}

env = ChatEnv(batch_size=(1,))
env = env.append_transform(MCPToolTransform(servers=servers))

reset_data = TensorDict(query="You are a helpful assistant", batch_size=(1,))
td = env.reset(reset_data)

history = td.get("history")

code = """
import math
result = math.sqrt(144) + math.pi
print(f"Result: {result}")
result
"""

response = (
    History(
        role="assistant",
        content=f'Let me calculate that.\n<tool>python.run_python_code\n{json.dumps({"python_code": code})}</tool>',
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

history.full = history.prompt.extend(response, inplace=True, dim=-1)
history.response = response

result = env.step(td.set("history", history))

print("Python code executed via MCP!")
print("\nTool response:")
tool_response = result["next", "history"].prompt[0, -1]
print(f"Role: {tool_response.role}")
print(f"Content: {tool_response.content}")

fibonacci_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(10)]
print(f"Fibonacci sequence: {result}")
result
"""

history = result["next", "history"]
response2 = (
    History(
        role="assistant",
        content=f'Now calculating Fibonacci.\n<tool>python.run_python_code\n{json.dumps({"python_code": fibonacci_code})}</tool>',
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

history.full = history.prompt.extend(response2, inplace=True, dim=-1)
history.response = response2

result2 = env.step(result["next"].set("history", history))

print("\n\nSecond execution:")
print("\nTool response:")
tool_response2 = result2["next", "history"].prompt[0, -1]
print(f"Role: {tool_response2.role}")
print(f"Content: {tool_response2.content[:500]}...")
