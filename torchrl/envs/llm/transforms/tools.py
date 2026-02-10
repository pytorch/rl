# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

import torch

from tensordict import lazy_stack, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.llm import History

from torchrl.envs.transforms import Transform
from typing_extensions import TypedDict


# --- Base Class for Tool Transforms ---


class ToolTransformBase(Transform):
    """Base class for tool transforms that parse and execute tools from LLM output.

    This class handles all the common boilerplate for tool transforms:
    - History extraction and validation
    - Batch dimension flattening
    - Result collection and padding
    - History extension with tool results

    Subclasses only need to implement:
    - :meth:`_process_batch_item`: Extract and execute tools from one response
    - :meth:`_format_result`: Format one tool result as string (optional)

    Attributes:
        use_step (bool): Whether to use _step() vs _call(). Defaults to True.
        tool_role (str): Role name for results in history. Defaults to "tool".

    Examples:
        >>> class SimpleCalculator(ToolTransformBase):
        ...     tool_role = "calculator"
        ...
        ...     def _process_batch_item(self, content: str, index: int):
        ...         # Extract math expressions and evaluate
        ...         if "2+2" in content:
        ...             return ["2+2=4"]
        ...         return None
    """

    use_step: bool = True  # Use _step() vs _call()
    tool_role: str = "tool"  # Role name for results in history

    def _validate_and_extract_history(
        self, next_tensordict: TensorDictBase
    ) -> tuple[History, History]:
        """Validate environment and extract history.

        Args:
            next_tensordict: The tensordict containing history.

        Returns:
            tuple: (full_history, local_history) where local_history is the last message.

        Raises:
            RuntimeError: If parent env doesn't exist or isn't in history mode.
        """
        # Check that base_env is in history mode
        parent = self.parent
        if parent is None:
            raise RuntimeError(f"{self.__class__.__name__} must be used with a ChatEnv")
        base_env = parent.base_env
        if base_env.input_mode != "history":
            raise RuntimeError(
                f"{self.__class__.__name__} must be used with a ChatEnv in history mode"
            )

        # Get history and isolate last element (the LLM's response)
        history = next_tensordict["history"].prompt
        local_history = history[..., -1]

        return history, local_history

    def _process_batch_item(self, content: str, index: int) -> list[str] | None:
        """Process one item in the batch to extract and execute tools.

        This is the main method subclasses must implement.

        Args:
            content: The text content from the LLM response.
            index: The index of this item in the batch.

        Returns:
            list[str] or None: List of result strings for each tool executed,
                or None if no tools were found/executed.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _process_batch_item()"
        )

    def _format_result(self, result: str) -> str:
        """Format a single result string.

        Override this to customize result formatting. Default is identity.

        Args:
            result: Raw result string from tool execution.

        Returns:
            str: Formatted result string.
        """
        return result

    def _inject_results_to_history(
        self,
        history: History,
        results: list[list[str] | None],
        next_tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """Inject tool results back into history with proper batching.

        Args:
            history: The full conversation history.
            results: List of results per batch item (can contain None).
            next_tensordict: The tensordict to update.

        Returns:
            TensorDictBase: Updated tensordict with results in history.
        """
        # Convert string results to History objects
        procs = []
        for batch_results in results:
            if batch_results is None or len(batch_results) == 0:
                procs.append(None)
            else:
                formatted_results = [self._format_result(r) for r in batch_results]
                procs.append(
                    [
                        History(role=self.tool_role, content=result)
                        for result in formatted_results
                    ]
                )

        # If there are no tool responses, skip
        if all(p is None for p in procs):
            return next_tensordict

        # Fill None entries with empty lists for consistent batching
        if any(p is None for p in procs):
            procs = [p if p is not None else [] for p in procs]

        # Pad all results to same length (required for batching)
        if len(procs) > 1 and not all(len(p) == len(procs[0]) for p in procs):

            def fill_procs(proc: list[History], max_len: int) -> list[History]:
                if len(proc) == max_len:
                    return proc
                return proc + [History(role="<none>", content="")] * (
                    max_len - len(proc)
                )

            max_len = max(len(p) for p in procs)
            procs = [fill_procs(p, max_len) for p in procs]

        # Stack and extend history
        procs = lazy_stack([lazy_stack(p) for p in procs])
        history.extend(procs, dim=-1)
        next_tensordict["history"].prompt = history

        return next_tensordict

    def _process_tensordict(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        """Main processing logic for tool transforms.

        Handles batch flattening, history extraction, tool processing, and result injection.

        Args:
            next_tensordict: The tensordict to process.

        Returns:
            TensorDictBase: Updated tensordict with tool results.
        """
        # Flatten batch dimensions if needed
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as next_tensordict_flat:
                next_tensordict_flat = self._process_tensordict(next_tensordict_flat)
            return next_tensordict

        # Extract and validate history
        history, local_history = self._validate_and_extract_history(next_tensordict)

        # Handle content as string or list
        content = local_history.content
        if isinstance(content, str):
            content = [content]

        # Process each batch item
        results = []
        for i, text in enumerate(content):
            batch_results = self._process_batch_item(text, i)
            results.append(batch_results)

        # Inject results back into history
        return self._inject_results_to_history(history, results, next_tensordict)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Handle step with tool processing.

        Args:
            tensordict: Input tensordict.
            next_tensordict: Output tensordict.

        Returns:
            TensorDictBase: Updated next_tensordict.
        """
        if not self.use_step:
            raise RuntimeError(
                f"{self.__class__.__name__} uses _call(), not _step(). Set use_step=False."
            )
        return self._process_tensordict(next_tensordict)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        """Handle call with tool processing.

        Args:
            next_tensordict: The tensordict to process.

        Returns:
            TensorDictBase: Updated tensordict.
        """
        if self.use_step:
            raise RuntimeError(
                f"{self.__class__.__name__} uses _step(), not _call(). Set use_step=True."
            )
        return self._process_tensordict(next_tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Handle reset (no-op for base class).

        Args:
            tensordict (TensorDictBase): Input tensordict.
            tensordict_reset (TensorDictBase): Reset tensordict.

        Returns:
            TensorDictBase: Unchanged reset tensordict.
        """
        return tensordict_reset


# --- Tool Service Library: Pluggable Services & Parsers ---


class ToolService(Protocol):
    """Protocol for side-effecting service callable with structured IO.

    A tool service is a callable that can be invoked with keyword arguments
    and returns a dictionary of results. It has a name and input/output schemas.

    Attributes:
        name (str): The name of the tool service.
        schema_in (dict[str, Any]): Input schema describing expected parameters.
        schema_out (dict[str, Any]): Output schema describing returned data.
    """

    name: str
    schema_in: dict[str, Any]
    schema_out: dict[str, Any]

    def __call__(self, **kwargs) -> dict[str, Any]:
        """Execute the tool service.

        Args:
            **kwargs: Keyword arguments matching the input schema.

        Returns:
            dict[str, Any]: Results matching the output schema.
        """
        ...


class ToolRegistry:
    """Registry for managing available tool services.

    This class maintains a collection of tool services that can be looked up
    by name for execution.

    Args:
        services (Sequence[ToolService], optional): Initial services to register.
            Defaults to an empty sequence.

    Examples:
        >>> class AddService:
        ...     name = "add"
        ...     schema_in = {"a": int, "b": int}
        ...     schema_out = {"result": int}
        ...     def __call__(self, a, b, **kwargs):
        ...         return {"result": a + b}
        >>> registry = ToolRegistry([AddService()])
        >>> service = registry.get("add")
        >>> result = service(a=1, b=2)
        >>> print(result)
        {"result": 3}
    """

    def __init__(self, services: Sequence[ToolService] = ()):
        self._svc: dict[str, ToolService] = {s.name: s for s in services}

    def register(self, service: ToolService) -> None:
        """Register a new service.

        Args:
            service (ToolService): The service to register.
        """
        self._svc[service.name] = service

    def get(self, name: str) -> ToolService:
        """Retrieve a service by name.

        Args:
            name (str): The name of the service to retrieve.

        Returns:
            ToolService: The requested service.

        Raises:
            KeyError: If the service is not found.
        """
        if name not in self._svc:
            raise KeyError(f"Unknown tool: {name}")
        return self._svc[name]

    def __contains__(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if the service exists, False otherwise.
        """
        return name in self._svc


@dataclass
class ToolCall:
    """Representation of a parsed tool call from LLM output.

    Attributes:
        tool (str): The name of the tool to call.
        args (dict[str, Any]): Arguments to pass to the tool.
        tag (str | None): Optional user-visible label or correlation ID.
    """

    tool: str
    args: dict[str, Any]
    tag: str | None = None


class ParseResult(TypedDict):
    """Result of parsing an LLM response for tool calls.

    This is a TypedDict-style class that contains:
        text (str): The final message to user (post tool blocks removal).
        calls (list[ToolCall]): Ordered tool calls as they appear.
        meta (dict[str, Any]): Optional parser metadata.
    """

    text: str
    calls: list[ToolCall]
    meta: dict[str, Any]


class LLMToolParser(Protocol):
    """Protocol for parsing LLM responses into ordered tool calls.

    A tool parser takes the LLM's response (as string or structured data)
    and extracts ordered tool calls, along with the cleaned user-facing text.
    """

    def __call__(self, response: str | dict[str, Any]) -> ParseResult:
        """Parse an LLM response.

        Args:
            response (str | dict[str, Any]): The LLM's response to parse.

        Returns:
            ParseResult: Parsed result with text, calls, and metadata.
        """
        ...


class XMLBlockParser:
    r"""Parser for XML-style tool blocks in LLM responses.

    Parses tool calls in the format:
        <tool name="tool_name" tag="optional_tag">{"arg": "value"}</tool>

    Examples:
        >>> parser = XMLBlockParser()
        >>> response = '<tool name="search" tag="A">{"query": "torchrl"}</tool>\\nSome text.'
        >>> result = parser(response)
        >>> print(result["text"])
        Some text.
        >>> print(result["calls"][0].tool)
        search
        >>> print(result["calls"][0].args)
        {"query": "torchrl"}
    """

    _re = re.compile(
        r'<tool\s+name="(?P<name>[^"]+)"(?:\s+tag="(?P<tag>[^"]+)")?\s*>\s*(?P<body>.*?)\s*</tool>',
        re.DOTALL,
    )

    def __call__(self, response: str | dict[str, Any]) -> ParseResult:
        """Parse XML-style tool blocks from response.

        Args:
            response (str | dict[str, Any]): The response to parse.

        Returns:
            ParseResult: Parsed result with cleaned text and tool calls.
        """
        text = response if isinstance(response, str) else response.get("text", "")
        calls: list[ToolCall] = []

        def repl(m: re.Match) -> str:
            name = m.group("name")
            tag = m.group("tag")
            body = m.group("body")
            try:
                args = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                # If JSON parsing fails, pass the raw body as a "raw" argument
                args = {"raw": body}
            calls.append(ToolCall(tool=name, args=args, tag=tag))
            return ""  # Remove block from final user-visible message

        cleaned = self._re.sub(repl, text).strip()
        result = ParseResult()
        result["text"] = cleaned
        result["calls"] = calls
        result["meta"] = {"count": len(calls)}
        return result


class JSONCallParser:
    """Parser for JSON-style function-calling responses.

    Expects responses in the format::

        {
          "message": "...",
          "tools": [
            {"tool": "search", "args": {"query": "..."}, "tag": "A"},
            {"tool": "summarize", "args": {"text": "..."}}
          ]
        }

    Examples:
        >>> parser = JSONCallParser()
        >>> response = {
        ...     "message": "Let me search for that.",
        ...     "tools": [{"tool": "search", "args": {"query": "torchrl"}}]
        ... }
        >>> result = parser(response)
        >>> print(result["text"])
        Let me search for that.
        >>> print(result["calls"][0].tool)
        search
    """

    def __call__(self, response: str | dict[str, Any]) -> ParseResult:
        """Parse JSON-style function calls from response.

        Args:
            response (str | dict[str, Any]): The response to parse.

        Returns:
            ParseResult: Parsed result with message and tool calls.
        """
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as plain text with no tools
                result = ParseResult()
                result["text"] = response
                result["calls"] = []
                result["meta"] = {"count": 0}
                return result

        tools_data = response.get("tools", [])
        calls = [ToolCall(**c) for c in tools_data]

        result = ParseResult()
        result["text"] = response.get("message", "")
        result["calls"] = calls
        result["meta"] = {"count": len(calls)}
        return result


class ExecuteToolsInOrder(ToolTransformBase):
    """A Transform that executes tools in the order they appear in LLM output.

    This transform reads the LLM response, parses ordered tool blocks using a
    pluggable parser, and executes tools via a ToolRegistry strictly in the
    order they appear in the response (independent of transform stacking order).

    The transform integrates naturally with TorchRL's LLM environments and can
    read/write conversation history alongside other transforms.

    Args:
        registry (ToolRegistry): Registry containing available tool services.
        parser (LLMToolParser): Parser for extracting tool calls from LLM output.
        stop_on_error (bool, optional): Whether to stop execution on first error.
            Defaults to ``False``.
        pass_state_to_tools (bool, optional): Whether to pass TD state to tools.
            Defaults to ``True``.

    Examples:
        >>> from torchrl.envs.llm import ChatEnv
        >>> from torchrl.envs.transforms import TransformedEnv, Compose
        >>> from torchrl.envs.llm.transforms import ExecuteToolsInOrder, ToolRegistry, XMLBlockParser
        >>>
        >>> # Define a simple service
        >>> class WebSearch:
        ...     name = "search"
        ...     schema_in = {"query": str}
        ...     schema_out = {"results": list}
        ...     def __call__(self, query: str, **kwargs):
        ...         return {"results": [{"title": "TorchRL docs", "url": "https://..."}]}
        >>>
        >>> # Create registry and parser
        >>> registry = ToolRegistry([WebSearch()])
        >>> parser = XMLBlockParser()
        >>>
        >>> # Create environment with transform
        >>> env = ChatEnv(batch_size=(1,))
        >>> env = TransformedEnv(
        ...     env,
        ...     ExecuteToolsInOrder(registry=registry, parser=parser)
        ... )

    .. note::
        This transform operates in the forward direction only; inverse is a no-op.
        Tool execution order is determined by appearance in the LLM output,
        not by the order of transforms in the Compose stack.
    """

    use_step = True  # Use _step() method

    def __init__(
        self,
        registry: ToolRegistry,
        parser: LLMToolParser,
        stop_on_error: bool = False,
        pass_state_to_tools: bool = True,
    ):
        super().__init__()
        self.registry = registry
        self.parser = parser
        self.stop_on_error = stop_on_error
        self.pass_state_to_tools = pass_state_to_tools
        self.tool_role = "tool"

    def _process_batch_item(self, content: str, index: int) -> list[str] | None:
        """Process one batch item to extract and execute tools.

        This is the main method required by ToolTransformBase.

        Args:
            content: The text content from the LLM response.
            index: The index of this item in the batch.

        Returns:
            list[str] or None: List of result strings for each tool executed,
                or None if no tools were found.
        """
        # Parse the response for tool calls
        parse: ParseResult = self.parser(content)
        ordered_calls = parse["calls"]

        if not ordered_calls:
            return None

        tool_outputs: list[dict[str, Any]] = []

        # Execute tools IN ORDER OF APPEARANCE
        for j, call in enumerate(ordered_calls):
            try:
                service = self.registry.get(call.tool)
                kwargs = dict(call.args)
                if self.pass_state_to_tools:
                    # Get tensordict from parent context if available
                    # For now, pass empty state - can be enhanced later
                    kwargs["_state"] = {}

                out = service(**kwargs)
                out["_tool"] = call.tool
                out["_index"] = j
                if call.tag:
                    out["_tag"] = call.tag
                tool_outputs.append(out)
            except Exception as e:
                err = {"_tool": call.tool, "_index": j, "error": str(e)}
                tool_outputs.append(err)
                if self.stop_on_error:
                    break

        # Format tool results as a single string
        # Format tool results as a single string
        if tool_outputs:
            results_text = self._format_tool_results(tool_outputs)
            return [results_text] if results_text else None

    def _format_tool_results(self, tool_outputs: list[dict[str, Any]]) -> str:
        """Format tool execution results as text.

        Args:
            tool_outputs (list[dict[str, Any]]): List of tool execution results.

        Returns:
            str: Formatted text representation of results.
        """
        if not tool_outputs:
            return ""

        lines = ["<tool_results>"]
        for output in tool_outputs:
            tool_name = output.pop("_tool", "unknown")
            index = output.pop("_index", 0)
            tag = output.pop("_tag", None)

            if "error" in output:
                lines.append(f"Tool {tool_name} (call {index + 1}) failed:")
                lines.append(f"  Error: {output['error']}")
            else:
                header = f"Tool {tool_name} (call {index + 1})"
                if tag:
                    header += f" [tag: {tag}]"
                header += " succeeded:"
                lines.append(header)
                lines.append(f"  Result: {json.dumps(output, indent=2)}")

        lines.append("</tool_results>")
        return "\n".join(lines)


class PersistentPythonProcess:
    """A persistent Python process that can execute code blocks."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._output_queue = queue.Queue()
        self._error_queue = queue.Queue()
        self._accumulated_errors = []
        self._init_script = None
        self.process = None  # Initialize to None to avoid AttributeError in __del__

        # Start the process
        self._start_process()

    def _start_process(self):
        """Start the Python process with the initialization script."""
        # Create a temporary file for initialization
        init_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        self._init_script = init_file.name

        # Write a script that creates a continuous execution environment
        init_file.write(
            """
import sys
import traceback

def run_code(code_str):
    # Create a dictionary to store the local variables
    locals_dict = {}
    try:
        # First try to compile the code to catch syntax errors
        compiled = compile(code_str, '<string>', 'exec')
        # Execute the code with the locals dictionary
        exec(compiled, globals(), locals_dict)
        # Ensure output is flushed
        sys.stdout.flush()
        sys.stderr.flush()
        return locals_dict
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Ensure error output is flushed immediately
        sys.stdout.flush()
        sys.stderr.flush()
        return locals_dict

# Signal that we're ready to accept commands
print('---READY---')
sys.stdout.flush()

# Main loop to handle commands
while True:
    try:
        # Read a line that signals the start of a command
        line = input()
        if line.strip() == '---EXEC---':
            # Read the code until we see the end marker
            code_lines = []
            while True:
                line = input()
                if line.strip() == '---END---':
                    break
                code_lines.append(line)

            # Execute the code
            code_str = '\\n'.join(code_lines)
            print('---START---')  # Signal start of execution
            sys.stdout.flush()
            locals_dict = run_code(code_str)
            # Update globals with new locals for persistence
            globals().update(locals_dict)
            print('---END---')  # Signal end of execution
            # Ensure all output is flushed
            sys.stdout.flush()
            sys.stderr.flush()
    except (EOFError, KeyboardInterrupt):
        break
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        break
"""
        )
        init_file.close()

        # Start the process
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", self._init_script],  # -u for unbuffered output
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start output reading threads
            self._stdout_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stdout, self._output_queue, "stdout"),
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stderr, self._error_queue, "stderr"),
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            # Wait for the process to be ready
            ready = False
            timeout = self.timeout
            while timeout > 0 and not ready:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Process failed to start: {self.process.returncode}"
                    )

                try:
                    line = self._output_queue.get_nowait()
                    torchrl_logger.info(f"Output: {line}")
                    if "---READY---" in line:
                        ready = True
                        break
                except queue.Empty:
                    timeout -= 0.1
                    time.sleep(0.1)

            if not ready:
                raise RuntimeError("Process failed to initialize within timeout")

        except Exception:
            # Clean up if process creation failed
            if self._init_script:
                try:
                    os.unlink(self._init_script)
                    self._init_script = None
                except Exception:
                    pass
            raise

    def _read_output(self, pipe: TextIO, q: queue.Queue, pipe_name: str) -> None:
        """Read output from a pipe and put it in a queue."""
        try:
            for line in iter(pipe.readline, ""):
                if pipe_name == "stderr":
                    self._accumulated_errors.append(line)
                q.put(line)
        except (ValueError, OSError) as e:
            # Pipe has been closed
            torchrl_logger.info(f"{pipe_name} pipe closed: {str(e)}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def execute(self, prompt: str) -> dict[str, Any]:
        """Execute code in the persistent process."""
        if not self.process or self.process.poll() is not None:
            # Get any accumulated errors
            errors = "".join(self._accumulated_errors)
            torchrl_logger.info(
                f"Process state: poll={self.process.poll() if self.process else 'No process'}, accumulated errors: {errors}"
            )
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Process not initialized or terminated. Accumulated errors: {errors}",
                "returncode": self.process.returncode if self.process else -1,
            }

        if not self.process.stdin:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Process stdin not available",
                "returncode": -1,
            }

        try:
            # Clear accumulated errors before new execution
            self._accumulated_errors.clear()

            # Send the execution markers and code
            try:
                self.process.stdin.write("---EXEC---\n")
                torchrl_logger.info(f"Writing to stdin: {prompt}")
                self.process.stdin.write(f"{prompt}\n")
                self.process.stdin.write("---END---\n")
                self.process.stdin.flush()
            except OSError as e:
                torchrl_logger.info(f"Failed to write to stdin: {str(e)}")
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Failed to write to process: {str(e)}",
                    "returncode": -1,
                }

            # Collect output until we see the end marker
            output = []
            error = []
            start_found = False
            timeout_val = self.timeout

            while timeout_val > 0:
                if self.process.poll() is not None:
                    # Process has terminated - get accumulated errors
                    errors = "".join(self._accumulated_errors)
                    torchrl_logger.info(
                        f"Process terminated with return code {self.process.returncode} - accumulated errors: {errors}"
                    )
                    error.append(
                        f"Process terminated with return code {self.process.returncode} - {errors}"
                    )
                    break

                try:
                    # Check for errors first
                    try:
                        while True:  # Drain all available error output
                            line = self._error_queue.get_nowait()
                            torchrl_logger.info(f"Error: {line}")
                            error.append(line)
                    except queue.Empty:
                        pass

                    # Then check for output
                    try:
                        line = self._output_queue.get_nowait()
                        torchrl_logger.info(f"Output: {line}")
                        if "---START---" in line:
                            start_found = True
                            continue
                        if "---END---" in line:
                            break
                        if start_found:
                            output.append(line)
                    except queue.Empty:
                        pass

                    # Always sleep a bit to avoid busy-waiting and give subprocess time
                    timeout_val -= 0.01
                    time.sleep(0.01)

                except Exception as e:
                    return {
                        "success": False,
                        "stdout": "",
                        "stderr": f"Execution error: {str(e)}",
                        "returncode": -1,
                    }

            if timeout_val <= 0:
                # Kill the process and create a new one
                self.cleanup()
                self.__init__(self.timeout)
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Code execution timed out - process restarted",
                    "returncode": -1,
                }

            return {
                "success": len(error) == 0,
                "stdout": "".join(output),
                "stderr": "".join(error),
                "returncode": 0 if len(error) == 0 else 1,
            }

        except Exception as e:
            # If we encounter any error, restart the process
            self.cleanup()
            self.__init__(self.timeout)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)} - process restarted",
                "returncode": -1,
            }

    def cleanup(self):
        """Clean up the persistent process."""
        import signal

        if self.process:
            try:
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                self.process.kill()
            self.process = None

        # Clean up the init script
        if self._init_script:
            try:
                os.unlink(self._init_script)
                self._init_script = None
            except Exception:
                pass

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class PythonExecutorService:
    """Ray actor that manages a pool of persistent Python interpreters.

    This service allows multiple environments to share a pool of Python
    interpreters, reducing resource usage and improving efficiency.

    Args:
        pool_size (int): Number of Python interpreter processes to maintain.
        timeout (float): Timeout for code execution in seconds.

    Examples:
        >>> # Register the service
        >>> from torchrl.services import get_services
        >>> services = get_services(backend="ray")
        >>> services.register(
        ...     "python_executor",
        ...     PythonExecutorService,
        ...     pool_size=32,
        ...     timeout=10.0,
        ...     num_cpus=32,
        ...     max_concurrency=32
        ... )
        >>>
        >>> # Use in transform
        >>> env = env.append_transform(
        ...     PythonInterpreter(services="ray")
        ... )
    """

    def __init__(self, pool_size: int = 32, timeout: float = 10.0):
        self.pool_size = pool_size
        self.timeout = timeout
        self.processes = [
            PersistentPythonProcess(timeout=timeout) for _ in range(pool_size)
        ]
        # Create a lock for each process to prevent concurrent access
        self.process_locks = [threading.Lock() for _ in range(pool_size)]
        self.next_idx = 0
        self._selection_lock = threading.Lock()

    def execute(self, code: str) -> dict:
        """Execute Python code using next available process (round-robin).

        Args:
            code: Python code to execute.

        Returns:
            dict: Execution result with keys 'success', 'stdout', 'stderr', 'returncode'.
        """
        # Select a process using round-robin
        with self._selection_lock:
            process_idx = self.next_idx
            self.next_idx = (self.next_idx + 1) % self.pool_size

        # Lock the selected process for the duration of execution
        with self.process_locks[process_idx]:
            return self.processes[process_idx].execute(code)

    def cleanup(self):
        """Cleanup all processes in the pool."""
        if hasattr(self, "processes"):
            for process in self.processes:
                if process:
                    process.cleanup()
            self.processes = []

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during cleanup - we might be in Ray actor context
            pass


class PythonInterpreter(ToolTransformBase):
    r"""A transform that executes Python code in the LLM response.

    This transform inherits from :class:`ToolTransformBase` and handles all the
    boilerplate for history extraction, batch processing, and result injection.

    Args:
        tokenizer: The tokenizer to use. Defaults to `None` (no tokenizer).
        tool_name: The name of the tool in the chat history. Defaults to `"tool"`.
        persistent: Whether to use persistent processes. Defaults to `False`.
        timeout: The timeout for the persistent processes. Defaults to `10.0`.
        services: Backend for shared Python executor service. If `"ray"`, uses
            a shared Ray actor service for execution. If `None`, uses local
            processes. Defaults to `None`.
        service_name: Name of the service in the registry. Only used if
            `services="ray"`. Defaults to `"python_executor"`.
        namespace: Ray namespace for the service. Only used if `services="ray"`.
            If `None`, uses the default namespace. Defaults to `None`.

    Examples:
        >>> from torchrl.envs.llm.transforms import PythonInterpreter
        >>> from transformers import AutoTokenizer
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> from torchrl.envs.llm import ChatEnv
        >>> set_list_to_stack(True).set()
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = ChatEnv(
        ...     batch_size=(1,),
        ...     system_prompt="I'm the system, do as I say",
        ...     apply_template=True,
        ...     tokenizer=tokenizer,
        ... )
        >>> env = env.append_transform(PythonInterpreter())
        >>> r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        >>> r["text_response"] = ["Here is a python code to execute:\n```python\na=1\nprint(f'{a=}')\n```<|im_end|>\n"]
        >>> s = env.step(r)
        >>> print(s['next', 'history'].apply_chat_template(tokenizer=tokenizer))
        ['<|im_start|>system\n'
         "I'm the system, do as I say<|im_end|>\n"
         '<|im_start|>user\n'
         'This is the user prompt<|im_end|>\n'
         '<|im_start|>assistant\n'
         'Here is a python code to execute:\n'
         '```python\n'
         'a=1\n'
         "print(f'{a=}')\n"
         '```<|im_end|>\n'
         '<|im_start|>user\n'
         '<tool_response>\n'
         'Code block 1 executed successfully:\n'
         'a=1\n'
         '\n'
         '</tool_response><|im_end|>\n'
         '<|im_start|>assistant\n']

        Using shared Ray service:
        >>> from torchrl.services import get_services
        >>>
        >>> # Register service once (e.g., in main process)
        >>> services = get_services(backend="ray")
        >>> if "python_executor" not in services:
        ...     services.register(
        ...         "python_executor",
        ...         PythonExecutorService,
        ...         pool_size=32,
        ...         timeout=10.0,
        ...         num_cpus=32,
        ...         max_concurrency=32
        ...     )
        >>>
        >>> # Use in transform (all 128 envs share the 32 interpreters)
        >>> env = env.append_transform(PythonInterpreter(services="ray"))
    """

    use_step = True  # Use _step() method

    def __init__(
        self,
        tokenizer=None,  # type: ignore
        tool_name: str = "tool",
        persistent: bool = False,
        timeout: float = 10.0,
        services: str | None = None,
        service_name: str = "python_executor",
        namespace: str | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tool_role = tool_name  # Set the role for history entries
        self.persistent = persistent
        self.timeout = timeout
        self.services = services
        self.service_name = service_name
        self.namespace = namespace

        # Initialize attributes to avoid AttributeError in __del__
        self.python_service = None
        self.processes = None

        # Initialize based on service mode
        if services == "ray":
            # Use shared Ray service
            try:
                from torchrl.services import get_services

                service_registry = get_services(backend="ray", namespace=namespace)
                self.python_service = service_registry[service_name]
                self.processes = None
                torchrl_logger.info(
                    f"PythonInterpreter using Ray service '{service_name}'"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get Ray service '{service_name}'. "
                    f"Make sure the service is registered. Error: {e}"
                ) from e
        elif services is None:
            # Use local processes
            self.python_service = None
            self.processes = [] if persistent else []
        else:
            raise ValueError(
                f"Invalid services backend: {services}. Must be 'ray' or None."
            )

    def close(self):
        """Close the transform."""
        if self.python_service is None and self.processes:
            for process in self.processes:
                if process:
                    process.cleanup()
            self.processes = []

    def clone(self):
        """Clone the transform."""
        return self.__class__(
            tokenizer=self.tokenizer,
            tool_name=self.tool_role,  # tool_role is the instance attribute
            persistent=self.persistent,
            timeout=self.timeout,
            services=self.services,
            service_name=self.service_name,
            namespace=self.namespace,
        )

    def _ensure_processes(self, batch_size: int):
        """Ensure we have the right number of persistent processes."""
        if not self.persistent:
            return

        # Create new processes if needed
        while len(self.processes) < batch_size:
            self.processes.append(PersistentPythonProcess(timeout=self.timeout))

        if any(p is None for p in self.processes):
            self.processes = [
                p if p is not None else PersistentPythonProcess(timeout=self.timeout)
                for p in self.processes
            ]

        # Remove extra processes if batch size decreased
        if len(self.processes) > batch_size:
            raise RuntimeError(
                f"Too many processes: {len(self.processes)} > {batch_size}"
            )

    def _execute_python_code(self, code: str, i: int) -> dict:
        """Safely execute Python code and return results."""
        if self.python_service is not None:
            # Use shared Ray service
            try:
                import ray

                result = ray.get(self.python_service.execute.remote(code))
                return result
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Ray service execution failed: {str(e)}",
                    "returncode": -1,
                }
        elif self.persistent:
            # Use local persistent process
            # Ensure we have enough processes
            if i >= len(self.processes):
                self._ensure_processes(i + 1)
            # Use persistent process
            process = self.processes[i]
            if process is None:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Process not initialized",
                    "returncode": -1,
                }
            return process.execute(code)
        else:
            # Use temporary file approach
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    temp_file = f.name

                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                os.unlink(temp_file)

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Code execution timed out",
                    "returncode": -1,
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "returncode": -1,
                }

    def _extract_python_code(self, text: str) -> list[str]:
        """Extract Python code blocks from markdown-style formatting."""
        # Pattern to match ```python ... ``` blocks
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _process_batch_item(self, content: str, index: int) -> list[str] | None:
        """Process one batch item to extract and execute Python code.

        This is the main method required by ToolTransformBase.

        Args:
            content: The text content from the LLM response.
            index: The index of this item in the batch.

        Returns:
            list[str] or None: List of result strings for each code block executed,
                or None if no code blocks were found.
        """
        # Ensure we have enough processes for persistent mode
        if self.persistent:
            if index >= len(self.processes):
                self._ensure_processes(index + 1)

        # Extract code blocks
        code_blocks = self._extract_python_code(content)
        if not code_blocks:
            return None

        # Execute each code block
        results = []
        for block_idx, code in enumerate(code_blocks):
            result = self._execute_python_code(code, index)

            if result["success"]:
                results.append(
                    f"Code block {block_idx + 1} executed successfully:\n{result['stdout']}"
                )
            else:
                results.append(
                    f"Code block {block_idx + 1} failed:\n{result['stderr']}"
                )

        return results if results else None

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Override to handle batch size management for persistent processes."""
        # Ensure we have enough processes for the entire batch (only for local persistent mode)
        if (
            self.python_service is None
            and self.persistent
            and next_tensordict.batch_dims == 1
        ):
            self._ensure_processes(len(next_tensordict))

        # Delegate to base class for all the heavy lifting
        return super()._step(tensordict, next_tensordict)

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            if hasattr(self, "python_service") and self.python_service is None:
                if hasattr(self, "processes") and self.processes:
                    for process in self.processes:
                        if process:
                            process.cleanup()
        except Exception:
            # Ignore errors during cleanup
            pass

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # Get the '_reset' key from the tensordict_reset
        reset = tensordict.get("_reset")
        if reset is not None:
            reset = reset.view(tensordict.shape)
        else:
            reset = torch.ones(
                tensordict.shape, device=tensordict.device, dtype=torch.bool
            )

        # Only reset local persistent processes, not the shared service
        if self.python_service is None and self.persistent:
            for i, process in enumerate(self.processes):
                if reset[i] and process is not None:
                    process.cleanup()
            self.processes = [
                process
                if not reset[i]
                else PersistentPythonProcess(timeout=self.timeout)
                for i, process in enumerate(self.processes)
            ]
        return tensordict_reset


class SimpleToolTransform(ToolTransformBase):
    r"""A simple transform that executes tools from a dictionary of callables.

    This is a lightweight alternative to MCPToolTransform for simple use cases
    where you don't need the full Model Context Protocol infrastructure.

    Args:
        tools (dict[str, Callable]): Dictionary mapping tool names to their implementation functions.
            Each function should accept kwargs matching its expected parameters.
        tool_schemas (dict[str, dict], optional): Dictionary mapping tool names to their schemas.
            Used for documentation purposes only.
        parser (LLMToolParser | None, optional): Parser for extracting tool calls. If None,
            uses a simple XML-style parser. Defaults to None.
        tool_call_pattern (str | None, optional): Regex pattern for extracting tool calls.
            Only used if parser is None. Format should capture (tool_name, args_json).
            Defaults to ``r"<tool>(.*?)\\n(.*?)</tool>"``.
        tool_name (str, optional): Role name for tool results in history. Defaults to "tool".
        timeout (float, optional): Timeout for tool execution in seconds. Defaults to 10.0.

    Examples:
        >>> from torchrl.envs.llm.transforms import SimpleToolTransform, XMLBlockParser
        >>> from torchrl.envs.llm import ChatEnv
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> set_list_to_stack(True).set()
        >>>
        >>> # Define a simple tool
        >>> def calculator(operation: str, a: float, b: float):
        ...     if operation == "add":
        ...         return {"result": a + b}
        ...     return {"error": "unknown operation"}
        >>>
        >>> tools = {"calculator": calculator}
        >>> env = ChatEnv(batch_size=(1,))
        >>>
        >>> # Option 1: Use default XML-style pattern
        >>> env = env.append_transform(SimpleToolTransform(tools=tools))
        >>>
        >>> # Option 2: Use XMLBlockParser for more features
        >>> parser = XMLBlockParser()
        >>> env = env.append_transform(SimpleToolTransform(tools=tools, parser=parser))
        >>>
        >>> # Option 3: Custom pattern
        >>> env = env.append_transform(
        ...     SimpleToolTransform(
        ...         tools=tools,
        ...         tool_call_pattern=r"CALL\[(.*?)\]\((.*?)\)"
        ...     )
        ... )
    """

    use_step = True

    def __init__(
        self,
        tools: dict[str, Callable],
        tool_schemas: dict[str, dict] | None = None,
        parser: LLMToolParser | None = None,
        tool_call_pattern: str | None = None,
        tool_name: str = "tool",
        timeout: float = 10.0,
    ):
        super().__init__()
        self.tools = tools
        self.tool_schemas = tool_schemas or {}
        self.parser = parser
        self.tool_call_pattern = tool_call_pattern or r"<tool>(.*?)\n(.*?)</tool>"
        self.tool_role = tool_name
        self.timeout = timeout

    def _extract_tool_calls(self, text: str) -> list[tuple[str, str]]:
        """Extract tool calls from text.

        Uses parser if provided, otherwise falls back to regex pattern.
        """
        if self.parser is not None:
            # Use the parser (e.g., XMLBlockParser)
            result: ParseResult = self.parser(text)
            calls = result.get("calls", [])
            return [(call.tool, json.dumps(call.args)) for call in calls]
        else:
            # Use regex pattern
            matches = re.findall(self.tool_call_pattern, text, re.DOTALL)
            return matches

    def _execute_tool(self, tool_name: str, args_json: str) -> dict:
        """Execute a tool with the given arguments."""
        try:
            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                }

            # Parse arguments
            try:
                args = json.loads(args_json) if args_json.strip() else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse tool arguments: {str(e)}",
                }

            # Execute tool
            result = self.tools[tool_name](**args)
            return {
                "success": True,
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    def _process_batch_item(self, content: str, index: int) -> list[str] | None:
        """Process one batch item to extract and execute simple tools."""
        tool_calls = self._extract_tool_calls(content)
        if not tool_calls:
            return None

        results = []
        for tool_name, args_json in tool_calls:
            result = self._execute_tool(tool_name, args_json)

            if result["success"]:
                results.append(
                    f"Tool {tool_name} executed successfully:\n{result['result']}"
                )
            else:
                results.append(f"Tool {tool_name} failed:\n{result['error']}")

        return results if results else None


class MCPToolTransform(ToolTransformBase):
    r"""A transform that executes tools via the Model Context Protocol (MCP).

    This transform connects to MCP servers and executes tools through the official
    MCP library. It runs async operations in a background thread to work with
    TorchRL's synchronous transform API.

    Args:
        servers (dict[str, dict]): Dictionary mapping server names to their configurations.
            Each config should have:
            - "command" (str): Command to launch the server (e.g., "npx", "uvx")
            - "args" (list[str]): Arguments for the command
            Example: {"browser": {"command": "npx", "args": ["@browsermcp/mcp@latest"]}}
        tool_call_pattern (str, optional): Regex pattern for extracting tool calls.
            Should capture (tool_name_with_server, args_json).
            Defaults to ``r"<tool>([\\w.]+)\\n(.*?)</tool>"``.
        tool_name (str, optional): Role name for tool results in history. Defaults to "tool".
        timeout (float, optional): Timeout for tool execution in seconds. Defaults to 10.0.

    Examples:
        >>> import os
        >>> import json
        >>> from torchrl.envs.llm import ChatEnv
        >>> from torchrl.envs.llm.transforms import MCPToolTransform
        >>> from torchrl.data.llm import History
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> set_list_to_stack(True).set()
        >>>
        >>> # Add Deno to PATH (required for mcp-run-python)
        >>> environ = os.environ.copy()
        >>> deno_path = os.path.expanduser("~/.deno/bin")
        >>> if deno_path not in os.environ.get('PATH', ''):
        ...     environ['PATH'] = f"{deno_path}:{os.environ['PATH']}"
        >>>
        >>> # Define MCP servers
        >>> servers = {
        ...     "browser": {
        ...         "command": "npx",
        ...         "args": ["@browsermcp/mcp@latest"]
        ...     },
        ...     "python": {
        ...         "command": "uvx",
        ...         "args": ["mcp-run-python", "stdio"],
        ...         "env": environ
        ...     }
        ... }
        >>>
        >>> # Create environment with MCP transform
        >>> env = ChatEnv(batch_size=(1,))
        >>> env = env.append_transform(MCPToolTransform(servers=servers))  # doctest: +SKIP
        [torchrl][INFO] Connecting to MCP server 'browser' (npx @browsermcp/mcp@latest)
        [torchrl][INFO] Connected to MCP server 'browser' with 12 tools
        [torchrl][INFO] Connecting to MCP server 'python' (uvx mcp-run-python stdio)
        [torchrl][INFO] Connected to MCP server 'python' with 1 tools
        >>>
        >>> # Execute Python code via MCP
        >>> reset_data = TensorDict(query="You are a useful assistant", batch_size=(1,))
        >>> td = env.reset(reset_data)
        >>> history = td.get("history")
        >>> code = '''
        ... import math
        ... result = math.sqrt(144) + math.pi
        ... print(f"Result: {result}")
        ... result
        ... '''
        >>> response = History(
        ...     role="assistant",
        ...     content=f'Let me calculate that.\n<tool>python.run_python_code\n{json.dumps({"python_code": code})}</tool>',
        ... ).unsqueeze(0).unsqueeze(0)
        >>> history.full = history.prompt.extend(response, inplace=True, dim=-1)
        >>> history.response = response
        >>> result = env.step(td.set("history", history))  # doctest: +SKIP
        >>> print(result["next", "history", "prompt"][..., -1].content)  # doctest: +SKIP
        LinkedList(LinkedList(["Tool python.run_python_code executed successfully:\n[TextContent(type='text', text='<status>success</status>\\n<output>\\nResult: 15.141592653589793\\n</output>\\n<return_value>\\n15.141592653589793\\n</return_value>', annotations=None, meta=None)]"]))

    .. note::
        This requires the `mcp` package to be installed: `pip install mcp`
        The transform manages async MCP connections in a background thread.

    .. note::
        Some MCP servers have additional requirements:
        - `mcp-run-python` requires Deno: `curl -fsSL https://deno.land/install.sh | sh`
        - Server-specific dependencies should be installed before use
    """

    use_step = True  # Use _step() method

    def __init__(
        self,
        servers: dict[str, dict],
        tool_call_pattern: str | None = None,
        tool_name: str = "tool",
        timeout: float = 10.0,
    ):
        super().__init__()
        self.server_configs = servers
        self.tool_call_pattern = tool_call_pattern or r"<tool>([\w.]+)\n(.*?)</tool>"
        self.tool_role = tool_name
        self.timeout = timeout

        # MCP session management
        self._loop = None
        self._thread = None
        self._sessions = {}
        self._tools_cache = {}
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()
        self._connection_error = None

        # Start the async event loop in a background thread
        self._start_mcp_thread()

    def _start_mcp_thread(self):
        """Start a background thread running an async event loop for MCP, since it's made of coroutines."""

        def run_loop():
            try:
                import asyncio
            except ImportError:
                self._connection_error = "asyncio not available for MCPToolTransform"
                torchrl_logger.error(self._connection_error)
                self._ready_event.set()
                return

            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                # Connect to all MCP servers
                self._loop.run_until_complete(self._connect_servers())

                # Signal that initialization is complete
                self._ready_event.set()

                # Keep loop running until shutdown
                while not self._shutdown_event.is_set():
                    self._loop.run_until_complete(asyncio.sleep(0.1))

                # Cleanup
                self._loop.run_until_complete(self._disconnect_servers())
                self._loop.close()
            except Exception as e:
                self._connection_error = f"MCP thread failed: {str(e)}"
                torchrl_logger.error(self._connection_error)
                self._ready_event.set()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for initialization to complete (with timeout)
        if not self._ready_event.wait(timeout=10.0):
            torchrl_logger.warning("MCP initialization timed out after 10 seconds")

        if self._connection_error:
            torchrl_logger.warning(
                f"MCP initialization had errors: {self._connection_error}"
            )

    async def _connect_servers(self):
        """Connect to all configured MCP servers."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            torchrl_logger.error(
                f"MCP library not installed. Install with: pip install mcp\nError: {e}"
            )
            return

        for server_name, config in self.server_configs.items():
            try:
                # Create stdio transport
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", None),
                )

                torchrl_logger.info(
                    f"Connecting to MCP server '{server_name}' ({config['command']} {' '.join(config.get('args', []))})"
                )

                # Connect and initialize session
                stdio = stdio_client(server_params)
                try:
                    read, write = await stdio.__aenter__()
                except Exception as e:
                    error_msg = str(e).lower()
                    if (
                        "deno" in error_msg
                        or "no such file or directory: 'deno'" in error_msg
                    ):
                        torchrl_logger.error(
                            f"Failed to start stdio for '{server_name}': Deno is not installed.\n"
                            f"  Install Deno: curl -fsSL https://deno.land/install.sh | sh\n"
                            f"  After installing, restart your terminal/shell."
                        )
                    else:
                        torchrl_logger.error(
                            f"Failed to start stdio for '{server_name}': {type(e).__name__}: {e}"
                        )
                    raise

                session = ClientSession(read, write)
                try:
                    await session.__aenter__()
                except Exception as e:
                    error_msg = str(e).lower()
                    if "connection closed" in error_msg:
                        # Subprocess likely crashed - check for common issues
                        torchrl_logger.error(
                            f"Failed to initialize session for '{server_name}': Subprocess terminated.\n"
                            f"  The MCP server '{config['command']}' started but immediately crashed.\n"
                            f"  Common causes:\n"
                            f"    - Missing dependencies (e.g., Deno for mcp-run-python)\n"
                            f"    - Invalid server configuration\n"
                            f"  Try running manually: {config['command']} {' '.join(config.get('args', []))}\n"
                            f"  Error: {e}"
                        )
                    else:
                        torchrl_logger.error(
                            f"Failed to initialize session for '{server_name}': {type(e).__name__}: {e}"
                        )
                    # Try to close stdio
                    try:
                        await stdio.__aexit__(None, None, None)
                    except Exception:
                        pass
                    raise

                self._sessions[server_name] = {
                    "session": session,
                    "stdio": stdio,
                }

                # Discover tools
                try:
                    tools_response = await session.list_tools()
                    tools = {tool.name: tool for tool in tools_response.tools}
                    self._tools_cache[server_name] = tools
                    torchrl_logger.info(
                        f"Connected to MCP server '{server_name}' with {len(tools)} tools"
                    )
                except Exception as e:
                    error_msg = str(e).lower()
                    if "connection closed" in error_msg:
                        torchrl_logger.error(
                            f"Could not list tools for server '{server_name}': Connection closed.\n"
                            f"  The MCP server started but crashed immediately.\n"
                            f"  This often means missing dependencies (e.g., Deno for mcp-run-python).\n"
                            f"  Test manually: {config['command']} {' '.join(config.get('args', []))}\n"
                            f"  For mcp-run-python, install Deno: curl -fsSL https://deno.land/install.sh | sh"
                        )
                    else:
                        torchrl_logger.warning(
                            f"Could not list tools for server '{server_name}': {e}"
                        )
                    self._tools_cache[server_name] = {}
                    # Don't keep a session we can't list tools from
                    try:
                        await session.__aexit__(None, None, None)
                        await stdio.__aexit__(None, None, None)
                    except Exception:
                        pass
                    if server_name in self._sessions:
                        del self._sessions[server_name]

            except FileNotFoundError as e:
                # Check if it's a Deno dependency issue
                if "deno" in str(e).lower():
                    torchrl_logger.error(
                        f"Failed to connect to MCP server '{server_name}': Deno is not installed.\n"
                        f"  Install Deno: curl -fsSL https://deno.land/install.sh | sh\n"
                        f"  Or use a different MCP server that doesn't require Deno.\n"
                        f"  Error: {e}"
                    )
                else:
                    torchrl_logger.error(
                        f"Failed to connect to MCP server '{server_name}': Command not found.\n"
                        f"  Make sure '{config['command']}' is installed and in your PATH.\n"
                        f"  Error: {e}"
                    )
            except Exception as e:
                torchrl_logger.error(
                    f"Failed to connect to MCP server '{server_name}': {type(e).__name__}: {e}"
                )

    async def _disconnect_servers(self):
        """Disconnect from all MCP servers."""
        for server_name, server_data in self._sessions.items():
            try:
                session = server_data["session"]
                stdio = server_data["stdio"]
                await session.__aexit__(None, None, None)
                await stdio.__aexit__(None, None, None)
            except Exception as e:
                torchrl_logger.warning(f"Error disconnecting from '{server_name}': {e}")

        self._sessions.clear()
        self._tools_cache.clear()

    def _extract_tool_calls(self, text: str) -> list[tuple[str, str, str]]:
        r"""Extract tool calls from text in format <tool>server.tool_name\nargs_json</tool>."""
        matches = re.findall(self.tool_call_pattern, text, re.DOTALL)

        # Parse into (server_name, tool_name, args_json)
        parsed = []
        for full_name, args_json in matches:
            if "." in full_name:
                server_name, tool_name = full_name.split(".", 1)
            else:
                # Default to first server if no prefix
                server_name = next(iter(self.server_configs.keys()), None)
                tool_name = full_name

            if server_name:
                parsed.append((server_name, tool_name, args_json))

        return parsed

    def _execute_tool_sync(
        self, server_name: str, tool_name: str, args_json: str
    ) -> dict:
        """Execute a tool via MCP (blocking call that schedules async work)."""
        if not self._loop or not self._thread or not self._thread.is_alive():
            return {
                "success": False,
                "error": "MCP thread not running",
            }

        # Schedule the async call in the background thread
        future = asyncio.run_coroutine_threadsafe(
            self._execute_tool_async(server_name, tool_name, args_json), self._loop
        )

        try:
            result = future.result(timeout=self.timeout)
            return result
        except TimeoutError:
            return {
                "success": False,
                "error": f"Tool execution timed out after {self.timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    async def _execute_tool_async(
        self, server_name: str, tool_name: str, args_json: str
    ) -> dict:
        """Execute a tool via MCP (async implementation)."""
        try:
            # Check if server exists
            if server_name not in self._sessions:
                return {
                    "success": False,
                    "error": f"MCP server '{server_name}' not connected",
                }

            session = self._sessions[server_name]["session"]

            # Parse arguments
            try:
                args = json.loads(args_json) if args_json.strip() else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse tool arguments: {str(e)}",
                }

            # Call the tool via MCP
            result = await session.call_tool(tool_name, arguments=args)

            return {
                "success": True,
                "result": result.content if hasattr(result, "content") else str(result),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"MCP tool call failed: {str(e)}",
            }

    def _process_batch_item(self, content: str, index: int) -> list[str] | None:
        """Process one batch item to extract and execute MCP tools.

        This is the main method required by ToolTransformBase.

        Args:
            content: The text content from the LLM response.
            index: The index of this item in the batch.

        Returns:
            list[str] or None: List of result strings for each tool executed,
                or None if no tools were found.
        """
        # Extract tool calls
        tool_calls = self._extract_tool_calls(content)
        if not tool_calls:
            return None

        # Execute each tool via MCP
        results = []
        for server_name, tool_name, args_json in tool_calls:
            result = self._execute_tool_sync(server_name, tool_name, args_json)

            if result["success"]:
                results.append(
                    f"Tool {server_name}.{tool_name} executed successfully:\n{result['result']}"
                )
            else:
                results.append(
                    f"Tool {server_name}.{tool_name} failed:\n{result['error']}"
                )

        return results if results else None

    def close(self):
        """Shutdown the MCP connections and background thread."""
        if self._thread and self._thread.is_alive():
            self._shutdown_event.set()
            self._thread.join(timeout=2.0)

        self._loop = None
        self._thread = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
