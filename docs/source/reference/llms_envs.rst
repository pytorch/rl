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
substrate for tool-calling agents. The headline orchestrator
(``ToolCompose``) lands in a follow-up commit; this preview ships the
contracts, parsers, sandboxing, and stateful REPLs that it builds on.

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
   isolation, prefer :class:`DockerSandbox` (real implementation
   tracked in the package TODO list).

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
