:orphan:

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
