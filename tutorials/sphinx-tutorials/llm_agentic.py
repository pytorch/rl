"""
Agentic ChatEnv: parallel tool dispatch with sandboxed REPL
===========================================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _llm_agentic:

This tutorial walks through building a SOTA agentic loop on top of
:class:`~torchrl.envs.llm.ChatEnv`: register a few tools, drop a
:class:`~torchrl.envs.llm.agentic.ToolCompose` into the env, and let the
LLM call them. Tool calls within a single response run **concurrently**,
Python execution is sandboxed, and any existing tool transform
(``PythonInterpreter``, ``BrowserTransform``, ``MCPToolTransform``,
``SimpleToolTransform``) plugs in alongside native tools via
:func:`~torchrl.envs.llm.agentic.tools.as_tool`.

What you will learn
-------------------

- How to compose :class:`~torchrl.envs.llm.agentic.Tool` instances under
  :class:`~torchrl.envs.llm.agentic.ToolCompose`.
- How to pick a sandbox backend and a stateful REPL.
- How to mix multiple parser families (XML / JSON-block / OpenAI tool
  calls / Anthropic tool use) under one orchestrator.
- How to migrate an existing tool transform into the new stack with
  zero rewriting.
"""

#####################################################################
# Why this exists
# ---------------
#
# The legacy :class:`~torchrl.envs.llm.transforms.ExecuteToolsInOrder`
# is a clean orchestrator but its dispatch is strictly sequential.
# Modern agent loops issue several independent calls per turn (search +
# read + compute) and pay a large wall-clock cost when those run one
# after the other.
#
# The agentic toolkit fixes this by:
#
# 1. Making tools async-first. Each
#    :meth:`~torchrl.envs.llm.agentic.Tool.run` is a coroutine; the
#    dispatcher uses :func:`asyncio.gather`.
# 2. Owning the parser at the
#    :class:`~torchrl.envs.llm.agentic.ToolCompose` level so the
#    response is parsed once, not once per transform.
# 3. Pinning a stable ``call_id`` for every parsed call so results
#    correlate across the dispatch boundary -- crucial for OpenAI /
#    Anthropic round-trips.
# 4. Defaulting to a hardened sandbox for code execution (bubblewrap
#    on Linux, sandbox-exec on macOS) instead of running a bare
#    subprocess in the host process.

#####################################################################
# A minimal agentic loop
# ----------------------
#
# We register two tools: a sandboxed Python REPL and an explicit
# :class:`~torchrl.envs.llm.agentic.StopTool`. The LLM is expected to
# emit XML-style calls.

from tensordict import TensorDict, set_list_to_stack
from torchrl.data.llm import History
from torchrl.envs import TransformedEnv
from torchrl.envs.llm import ChatEnv
from torchrl.envs.llm.agentic import (
    PythonTool,
    StopTool,
    ToolCompose,
)
from torchrl.envs.llm.agentic.parsers import XMLToolCallParser
from torchrl.envs.llm.agentic.repl import SubprocessRepl
from torchrl.envs.llm.agentic.sandbox import default_sandbox

set_list_to_stack(True).set()

sandbox = default_sandbox()
repl = SubprocessRepl(sandbox)
env = TransformedEnv(
    ChatEnv(batch_size=(1,), input_mode="history"),
    ToolCompose(
        tools=[PythonTool(repl=repl), StopTool()],
        parser=XMLToolCallParser(),
    ),
)

obs = env.reset(
    TensorDict({"query": "Compute 2+2 in python."}, batch_size=(1,))
)

# Stand-in for an LLM response; in real use this comes from a policy.
fake_response = '<tool name="python" tag="c1">{"code": "print(2+2)"}</tool>'
obs["history"].full = obs["history"].prompt.extend(
    History(role="assistant", content=fake_response).view(1, 1), dim=-1,
)
nxt = env.step(obs)
print(nxt[("next", "history")].prompt[0][-1].content)

#####################################################################
# Switching parser family
# -----------------------
#
# The same env shape works against any policy that emits structured
# tool calls. Swap the parser to match the model's protocol:

from torchrl.envs.llm.agentic.parsers import (  # noqa: E402
    AnthropicToolUseParser,
    JSONToolCallParser,
    OpenAIToolCallParser,
)

# OpenAI / vLLM-with-tools:
#   ToolCompose(tools=[...], parser=OpenAIToolCallParser())
# Anthropic Messages API:
#   ToolCompose(tools=[...], parser=AnthropicToolUseParser())
# Plain JSON envelope:
#   ToolCompose(tools=[...], parser=JSONToolCallParser())

#####################################################################
# Parallel dispatch in action
# ---------------------------
#
# Three independent tools, each waiting 500ms on a network call,
# complete in roughly 500ms total -- not 1.5s -- because
# :class:`~torchrl.envs.llm.agentic.ToolCompose` runs them concurrently.
# The benchmark group ``agentic-dispatch`` in
# ``benchmarks/test_llm.py`` pins this property in CI.

#####################################################################
# Migrating from legacy transforms
# --------------------------------
#
# If you have existing user code built on
# :class:`~torchrl.envs.llm.transforms.PythonInterpreter`,
# :class:`~torchrl.envs.llm.transforms.BrowserTransform`,
# :class:`~torchrl.envs.llm.transforms.MCPToolTransform`, or
# :class:`~torchrl.envs.llm.transforms.SimpleToolTransform`, you don't
# have to rewrite it. Lift the existing transform into a new-style
# :class:`~torchrl.envs.llm.agentic.Tool` via
# :func:`~torchrl.envs.llm.agentic.tools.as_tool`:
#
# .. code-block:: python
#
#     from torchrl.envs.llm.transforms import PythonInterpreter
#     from torchrl.envs.llm.agentic import ToolCompose
#     from torchrl.envs.llm.agentic.tools import as_tool
#     from torchrl.envs.llm.agentic.parsers import XMLToolCallParser
#
#     legacy_python = as_tool(
#         PythonInterpreter(persistent=True),
#         name="python",
#         input_schema={
#             "type": "object",
#             "properties": {"code": {"type": "string"}},
#             "required": ["code"],
#         },
#     )
#
#     env = TransformedEnv(
#         ChatEnv(batch_size=(1,), input_mode="history"),
#         ToolCompose(tools=[legacy_python], parser=XMLToolCallParser()),
#     )
#
# The legacy transform keeps its existing semantics; it now participates
# in parallel dispatch alongside any native :class:`Tool` you add.

#####################################################################
# Connecting an MCP server
# ------------------------
#
# The Model Context Protocol turns one server into many tools. Use
# :class:`~torchrl.envs.llm.agentic.MCPToolset` to discover them and
# drop the result straight into
# :class:`~torchrl.envs.llm.agentic.ToolCompose`:
#
# .. code-block:: python
#
#     import asyncio
#     from torchrl.envs.llm.agentic import (
#         MCPServerConfig, MCPToolset, ToolCompose,
#     )
#     from torchrl.envs.llm.agentic.parsers import XMLToolCallParser
#
#     async def make_env():
#         pool = MCPToolset(
#             MCPServerConfig(command="npx",
#                             args=("@browsermcp/mcp@latest",))
#         )
#         await pool.open()
#         compose = ToolCompose(
#             tools=list(pool.tools),
#             parser=XMLToolCallParser(),
#         )
#         return compose, pool
#
#     compose, pool = asyncio.run(make_env())

#####################################################################
# Conclusion
# ----------
#
# The combination of *one parse per turn*, *parallel dispatch*, and
# *hardened sandboxes* turns ChatEnv into a SOTA agent backbone without
# introducing a parallel taxonomy of "agent" classes. Tools stay
# composable, ``ChatEnv`` stays minimal, and existing tool transforms
# keep working.

#####################################################################
# Further reading
# ---------------
#
# - :class:`~torchrl.envs.llm.agentic.ToolCompose` -- API reference.
# - :class:`~torchrl.envs.llm.agentic.sandbox.Sandbox` and
#   :class:`~torchrl.envs.llm.agentic.repl.Repl` -- protocol details.
# - The migration table in the LLM Environments reference page.
# - ``benchmarks/test_llm.py::test_toolcompose_parallel_dispatch`` --
#   performance bench.
