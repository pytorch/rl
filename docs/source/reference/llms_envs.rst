:orphan:

.. _llm_envs:

.. currentmodule:: torchrl.envs.llm

LLM Environments
================

The environment layer orchestrates data loading, tool execution, reward computation, and formatting.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatEnv
    CountdownEnv
    CountdownRewardParser
    DatasetChatEnv
    GSM8KEnv
    make_gsm8k_env
    GSM8KPrepareQuestion
    GSM8KRewardParser
    IFEvalEnv
    IfEvalScorer
    IFEvalScoreData
    MATHEnv
    MATHRewardParser
    LLMEnv
    LLMHashingEnv
    make_mlgym
    MLGymWrapper

Agentic toolkit (preview)
-------------------------

.. currentmodule:: torchrl.envs.llm.agentic

The :mod:`torchrl.envs.llm.agentic` package provides a SOTA, async-first
substrate for tool-calling agents on top of an unmodified
:class:`~torchrl.envs.llm.ChatEnv`: structured parsers for the major
provider protocols (XML, JSON-block, OpenAI ``tool_calls``, Anthropic
``tool_use``), hardened :class:`Sandbox` backends, and stateful
:class:`Repl` sessions.

This preview ships the substrate the headline orchestrator
(``ToolCompose``) is built on. A minimal end-to-end sketch -- usable
today against the substrate, formalised by the orchestrator -- looks
like:

.. code-block:: python

    from torchrl.envs.llm.agentic.parsers import XMLToolCallParser
    from torchrl.envs.llm.agentic.sandbox import default_sandbox, ResourceLimits
    from torchrl.envs.llm.agentic.repl import SubprocessRepl

    parser = XMLToolCallParser()
    parsed = parser.parse('<tool name="python" tag="c1">{"code": "print(2+2)"}</tool>')
    # -> parsed.calls[0].tool == "python", parsed.calls[0].call_id == "c1"

    sandbox = default_sandbox(ResourceLimits(wall_seconds=10, network="none"))
    async def run():
        async with sandbox, SubprocessRepl(sandbox) as repl:
            result = await repl.execute("print(2+2)")
            assert result.stdout.strip() == "4"

Tool contracts
~~~~~~~~~~~~~~

A :class:`Tool` is a pure async object with a name, a JSON Schema
``input_schema``, and an async ``run(args, ctx)`` method returning a
:class:`ToolResult`. Calls flow through a :class:`ToolCallParser` (one of
the four built-ins below) which guarantees a stable ``call_id`` for
every invocation.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Tool
    ToolContext
    ToolResult
    TextPart
    JsonPart
    ImagePart
    FileRefPart
    ParsedCall
    ParseResult
    ToolCallParser

Parsers
~~~~~~~

.. currentmodule:: torchrl.envs.llm.agentic.parsers

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    XMLToolCallParser
    JSONToolCallParser
    OpenAIToolCallParser
    AnthropicToolUseParser

Sandboxing
~~~~~~~~~~

.. currentmodule:: torchrl.envs.llm.agentic.sandbox

A :class:`Sandbox` is an async context manager that runs subprocess
commands with bounded resources, controlled filesystem access, and
opt-in network egress. The default backends are
:class:`BubblewrapSandbox` on Linux and :class:`SeatbeltSandbox` on
macOS; pick one explicitly or use :func:`default_sandbox`.

For environments without those binaries, :class:`UnsafeSubprocessSandbox`
provides a no-isolation fallback that warns loudly on every
``open()``. Do not use it with untrusted model output.

.. note::
   Apple has officially deprecated ``sandbox-exec``, but it still ships
   with macOS 14+ and remains the most portable in-process isolation
   primitive on that platform. For stronger or cross-platform
   isolation, prefer :class:`DockerSandbox` (currently a stub --
   contributions welcome).

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Sandbox
    SandboxResult
    ResourceLimits
    BubblewrapSandbox
    SeatbeltSandbox
    UnsafeSubprocessSandbox
    DockerSandbox
    E2BSandbox
    ModalSandbox
    default_sandbox

Stateful REPLs
~~~~~~~~~~~~~~

.. currentmodule:: torchrl.envs.llm.agentic.repl

A :class:`Repl` runs stateful code inside a :class:`Sandbox` so an
agent can build up variables across multiple tool calls. The default
:class:`JupyterRepl` uses an IPython kernel for rich outputs (images,
JSON, plots) and clean restarts (optional dependency:
``jupyter_client``). :class:`SubprocessRepl` is a no-dep fallback that
trades rich display for portability.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Repl
    ReplResult
    ReplDisplay
    ReplError
    JupyterRepl
    SubprocessRepl

Built-in tools and adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.envs.llm.agentic

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ToolCompose
    DispatchResult
    PythonTool
    ShellTool
    FileReadTool
    StopTool
    HttpTool
    MCPServerConfig
    MCPToolset
    RateLimiter
    as_tool

Migration from legacy tool transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing code built on :mod:`torchrl.envs.llm.transforms` keeps working:
no ``DeprecationWarning`` is emitted in this release. Each legacy class
has a ``.. seealso::`` block in its docstring pointing at the
recommended replacement, summarised here.

.. list-table:: Legacy transform → agentic counterpart
   :header-rows: 1
   :widths: 30 30 40

   * - Legacy
     - Agentic
     - Adapter recipe
   * - ``ExecuteToolsInOrder``
     - :class:`ToolCompose`
     - Replace at the env stack level. ``ToolCompose`` runs calls
       concurrently; pin sequential execution per-tool with
       :class:`RateLimiter` ``max_concurrent=1`` if you depend on
       ordering.
   * - ``PythonInterpreter``
     - :class:`PythonTool` + :class:`Sandbox` + :class:`Repl`
     - For a soft migration, lift the existing transform: ``as_tool(PythonInterpreter(persistent=True), name="python", input_schema=...)``.
   * - ``SimpleToolTransform``
     - Native :class:`Tool` subclass
     - Or ``as_tool(transform, name=..., input_schema=...)``.
   * - ``BrowserTransform``
     - :func:`tools.as_tool` of the existing transform
     - A native :class:`Tool` for browser automation may land later;
       until then the adapter is the recommended path.
   * - ``MCPToolTransform``
     - :class:`MCPToolset`
     - One :class:`Tool` per remote tool, schemas auto-discovered.
       Drops directly into ``ToolCompose``.
   * - ``XMLBlockParser`` / ``JSONCallParser``
     - :class:`parsers.XMLToolCallParser` / :class:`parsers.JSONToolCallParser`
     - Same syntax; the agentic versions enforce a stable ``call_id``.
   * - ``ToolService`` / ``ToolRegistry``
     - The ``tools=[...]`` argument to :class:`ToolCompose`
     - The registry pattern collapses into the compose container.

For a guided walkthrough, see the
:ref:`agentic ChatEnv tutorial <llm_agentic>`.
