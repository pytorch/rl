# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

import torch

from tensordict import lazy_stack, TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.data.llm import History

from torchrl.envs import Transform


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


class ParseResult(dict):
    """Result of parsing an LLM response for tool calls.

    This is a TypedDict-style class that contains:
        text (str): The final message to user (post tool blocks removal).
        calls (list[ToolCall]): Ordered tool calls as they appear.
        meta (dict[str, Any]): Optional parser metadata.
    """


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
    """Parser for XML-style tool blocks in LLM responses.

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

    Expects responses in the format:
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


class ExecuteToolsInOrder(Transform):
    """A Transform that executes tools in the order they appear in LLM output.

    This transform reads the LLM response, parses ordered tool blocks using a
    pluggable parser, and executes tools via a ToolRegistry strictly in the
    order they appear in the response (independent of transform stacking order).

    The transform integrates naturally with TorchRL's LLM environments and can
    read/write conversation history alongside other transforms.

    Args:
        registry (ToolRegistry): Registry containing available tool services.
        parser (LLMToolParser): Parser for extracting tool calls from LLM output.
        in_keys (tuple[str, ...], optional): Key where LLM response is read.
            Defaults to ``("history", "prompt")``.
        out_keys (tuple[str, ...], optional): Key where tool results are written.
            Defaults to ``("tools", "results")``.
        message_key (tuple[str, ...], optional): Key for cleaned message text.
            Defaults to ``("llm", "message")``.
        history_key (tuple[str, ...], optional): Key for conversation history.
            Defaults to ``("history", "prompt")``.
        write_calls_key (tuple[str, ...], optional): Key for storing parsed calls.
            Defaults to ``("tools", "calls")``.
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

    def __init__(
        self,
        registry: ToolRegistry,
        parser: LLMToolParser,
        in_keys: tuple[str, ...] | None = None,
        out_keys: tuple[str, ...] | None = None,
        message_key: tuple[str, ...] | None = None,
        history_key: tuple[str, ...] | None = None,
        write_calls_key: tuple[str, ...] | None = None,
        stop_on_error: bool = False,
        pass_state_to_tools: bool = True,
    ):
        # Set defaults
        if in_keys is None:
            in_keys = ("history", "prompt")
        if out_keys is None:
            out_keys = ("tools", "results")
        if message_key is None:
            message_key = ("llm", "message")
        if history_key is None:
            history_key = ("history", "prompt")
        if write_calls_key is None:
            write_calls_key = ("tools", "calls")

        super().__init__(in_keys=[in_keys], out_keys=[out_keys])
        self.registry = registry
        self.parser = parser
        self._in = in_keys
        self._out = out_keys
        self._msg = message_key
        self._hist = history_key
        self._calls = write_calls_key
        self.stop_on_error = stop_on_error
        self.pass_state_to_tools = pass_state_to_tools

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Execute tools during environment step.

        Args:
            tensordict (TensorDictBase): Input tensordict before step.
            next_tensordict (TensorDictBase): Output tensordict after step.

        Returns:
            TensorDictBase: Modified next_tensordict with tool results.
        """
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as next_tensordict_flat:
                next_tensordict_flat = self._step(tensordict, next_tensordict_flat)
            return next_tensordict

        # Check that we're in history mode
        parent = self.parent
        if parent is None:
            raise RuntimeError("ExecuteToolsInOrder must be used with a ChatEnv")
        base_env = parent.base_env
        if base_env.input_mode != "history":
            raise RuntimeError(
                "ExecuteToolsInOrder must be used with a ChatEnv in history mode"
            )

        # Get the history and extract the last message (LLM response)
        history = next_tensordict["history"].prompt
        local_history = history[..., -1]

        procs = []
        # Iterate over batch
        for i, response_text in enumerate(local_history.content):
            # Parse the response for tool calls
            parse: ParseResult = self.parser(response_text)
            ordered_calls = parse["calls"]
            tool_outputs: list[dict[str, Any]] = []

            # Execute tools IN ORDER OF APPEARANCE
            for j, call in enumerate(ordered_calls):
                try:
                    service = self.registry.get(call.tool)
                    kwargs = dict(call.args)
                    if self.pass_state_to_tools:
                        kwargs["_state"] = self._export_state_for_tool(next_tensordict)

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

            # Store results and calls in tensordict
            if tool_outputs:
                # Format tool results as history entries
                results_text = self._format_tool_results(tool_outputs)
                if results_text:
                    procs.append([History(role="tool", content=results_text)])
                else:
                    procs.append(None)
            else:
                procs.append(None)

        # Add tool results to history if any tools were executed
        if not all(p is None for p in procs):
            if any(p is None for p in procs):
                procs = [p if p is not None else [] for p in procs]

            # Ensure all batch elements have same length
            if len(procs) > 1 and not all(len(p) == len(procs[0]) for p in procs):

                def fill_procs(proc: list[History], max_len: int) -> list[History]:
                    if len(proc) == max_len:
                        return proc
                    return proc + [History(role="tool", content="")] * (
                        max_len - len(proc)
                    )

                max_len = max(len(p) for p in procs)
                procs = [fill_procs(p, max_len) for p in procs]

            # Stack and extend history
            procs = lazy_stack([lazy_stack(p) for p in procs])
            history.extend(procs, dim=-1)
            next_tensordict["history"].prompt = history

        return next_tensordict

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

    def _export_state_for_tool(self, td: TensorDictBase) -> dict[str, Any]:
        """Export a filtered, read-only view of TD state for tools.

        Args:
            td (TensorDictBase): The tensordict to export from.

        Returns:
            dict[str, Any]: Filtered state dictionary.
        """
        # Minimal, safe view; customize as needed
        keys_for_tools = [("history", "prompt"), ("env", "step"), ("episode", "id")]
        out = {}
        for k in keys_for_tools:
            if td.get(k, None) is not None:
                value = td.get(k)
                # Convert to Python types if needed
                if isinstance(value, torch.Tensor):
                    value = value.tolist() if value.numel() > 1 else value.item()
                out["/".join(k if isinstance(k, tuple) else (k,))] = value
        return out

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Handle reset (no-op for this transform).

        Args:
            tensordict (TensorDictBase): Input tensordict.
            tensordict_reset (TensorDictBase): Reset tensordict.

        Returns:
            TensorDictBase: Unchanged reset tensordict.
        """
        return tensordict_reset


class PersistentPythonProcess:
    """A persistent Python process that can execute code blocks."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._output_queue = queue.Queue()
        self._error_queue = queue.Queue()
        self._accumulated_errors = []
        self._init_script = None

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
                ["python", "-u", self._init_script],  # -u for unbuffered output
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

    def execute(self, prompt: str) -> dict[str, any]:
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

                    if not start_found:
                        timeout_val -= 0.1
                        time.sleep(0.1)

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


class PythonInterpreter(Transform):
    r"""A transform that executes Python code in the LLM response.

    Args:
        tokenizer: The tokenizer to use. Defaults to `None` (no tokenizer).
        tool_name: The name of the tool in the chat history. Defaults to `"tool"`.
        persistent: Whether to use persistent processes. Defaults to `False`.
        timeout: The timeout for the persistent processes. Defaults to `10.0`.

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
    """

    def __init__(
        self,
        tokenizer=None,  # type: ignore
        tool_name: str = "tool",
        persistent: bool = False,
        timeout: float = 10.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tool_name = tool_name
        self.persistent = persistent
        # Initialize as empty list if persistent, None otherwise
        self.processes: list[PersistentPythonProcess | None] = [] if persistent else []

    def close(self):
        """Close the transform."""
        if self.processes:
            for process in self.processes:
                process.cleanup()
        self.processes = []

    def clone(self):
        """Clone the transform."""
        return self.__class__(
            tokenizer=self.tokenizer,
            tool_name=self.tool_name,
            persistent=self.persistent,
        )

    def _ensure_processes(self, batch_size: int):
        """Ensure we have the right number of persistent processes."""
        if not self.persistent:
            return

        # Create new processes if needed
        while len(self.processes) < batch_size:
            self.processes.append(PersistentPythonProcess())

        if any(p is None for p in self.processes):
            self.processes = [
                p if p is not None else PersistentPythonProcess()
                for p in self.processes
            ]

        # Remove extra processes if batch size decreased
        if len(self.processes) > batch_size:
            raise RuntimeError(
                f"Too many processes: {len(self.processes)} > {batch_size}"
            )

    def _execute_python_code(self, code: str, i: int) -> dict:
        """Safely execute Python code and return results."""
        if self.persistent:
            # Ensure we have enough processes
            if i >= len(self.processes):
                self._ensure_processes(i + 1)
            # Use persistent process
            return self.processes[i].execute(code)
        else:
            # Use temporary file approach
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    temp_file = f.name

                result = subprocess.run(
                    ["python", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
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

    def _process_llm_response(self, response: str, i: int) -> list[str]:
        """Process LLM response and execute any Python code found.

        Args:
            response (str): The response from the LLM.
            i (int): The index of the response in the batch.

        Returns:
            list[str]: A list of strings, each containing the result of the execution of the code block.
        """
        code_blocks = self._extract_python_code(response)

        results = []
        for i, code in enumerate(code_blocks):
            result = self._execute_python_code(code, i)

            if result["success"]:
                results.append(
                    f"Code block {i + 1} executed successfully:\n{result['stdout']}"
                )
            else:
                results.append(f"Code block {i + 1} failed:\n{result['stderr']}")

        return results

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as next_tensordict_flat, tensordict.view(
                -1
            ) as tensordict_flat:
                # Call the transform on the flattened tensordict
                next_tensordict_flat = self._step(tensordict_flat, next_tensordict_flat)
            return next_tensordict

        # Ensure we have enough processes for the batch
        if self.persistent:
            self._ensure_processes(len(next_tensordict))

        # Convert text to a history
        history = next_tensordict["history"].prompt
        # Isolate last element, which should be our action
        local_history = history[..., -1]

        procs = []
        # Iterate over env batch-size
        content = local_history.content
        if isinstance(content, str):
            content = [content]
        for i, t in enumerate(content):
            results = self._process_llm_response(t, i)
            if len(results) == 0:
                procs.append(None)
                continue
            procs.append(
                [History(role=self.tool_name, content=result) for result in results]
            )

        # If there is no tool response, just skip entire batch
        if all(p is None for p in procs):
            return next_tensordict
        if any(p is None for p in procs):
            procs = [p if p is not None else [] for p in procs]
        # We need to have the same number of items for eache element in the batch
        if len(procs) > 1 and not all(len(p) == len(procs[0]) for p in procs):

            def fill_procs(proc: list[History], max_len: int) -> list[History]:
                if len(proc) == max_len:
                    return proc
                return proc + [History(role="<none>", content="")] * (
                    max_len - len(proc)
                )

            max_len = max(len(p) for p in procs)
            procs = [fill_procs(p, max_len) for p in procs]
        # Procs has the shape of the batch-size. We can cat along dim=-1
        procs = lazy_stack([lazy_stack(p) for p in procs])
        history.extend(procs, dim=-1)
        next_tensordict["history"].prompt = history
        return next_tensordict

    def __del__(self):
        """Cleanup persistent processes on deletion."""
        if self.processes:
            for process in self.processes:
                if process:
                    process.cleanup()

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
        if self.persistent:
            for i, process in enumerate(self.processes):
                if reset[i] and process is not None:
                    process.cleanup()
            self.processes = [
                process if not reset[i] else PersistentPythonProcess()
                for i, process in enumerate(self.processes)
            ]
        return tensordict_reset


class MCPToolTransform(Transform):
    r"""A transform that executes MCP-style tools in response to LLM actions.

    This transform allows execution of tools following the Mission Control Protocol pattern,
    where tools are defined with clear input/output schemas and executed in a controlled manner.

    Args:
        tools (dict[str, callable]): A dictionary mapping tool names to their implementation functions.
            Each function should accept kwargs matching its schema and return a dict with results.
        tool_schemas (dict[str, dict]): A dictionary mapping tool names to their JSON schemas.
            Each schema should define the tool's parameters and return type.
        tokenizer: The tokenizer to use. Defaults to `None` (no tokenizer).
        tool_name: The name of the tool in the chat history. Defaults to `"tool"`.
        timeout: The timeout for tool execution in seconds. Defaults to `10.0`.

    Examples:
        >>> from torchrl.envs.llm.transforms import MCPToolTransform
        >>> from transformers import AutoTokenizer
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> from torchrl.envs.llm import ChatEnv
        >>> set_list_to_stack(True).set()
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> # Define a simple tool
        >>> def add_numbers(a: int, b: int) -> dict:
        ...     return {"result": a + b}
        >>> # Define its schema
        >>> add_schema = {
        ...     "name": "add_numbers",
        ...     "description": "Add two numbers",
        ...     "parameters": {
        ...         "type": "object",
        ...         "properties": {
        ...             "a": {"type": "integer"},
        ...             "b": {"type": "integer"}
        ...         },
        ...         "required": ["a", "b"]
        ...     }
        ... }
        >>> tools = {"add_numbers": add_numbers}
        >>> schemas = {"add_numbers": add_schema}
        >>> env = ChatEnv(
        ...     batch_size=(1,),
        ...     system_prompt="I'm the system, do as I say",
        ...     apply_template=True,
        ...     tokenizer=tokenizer,
        ... )
        >>> env = env.append_transform(MCPToolTransform(tools, schemas))
        >>> r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        >>> r["text_response"] = ["Let me add two numbers:\n<tool>add_numbers\n{\"a\": 1, \"b\": 2}</tool>"]
        >>> s = env.step(r)
        >>> print(s['next', 'history'].apply_chat_template(tokenizer=tokenizer))
        ['<|im_start|>system\n'
         "I'm the system, do as I say<|im_end|>\n"
         '<|im_start|>user\n'
         'This is the user prompt<|im_end|>\n'
         '<|im_start|>assistant\n'
         'Let me add two numbers:\n'
         '<tool>add_numbers\n'
         '{"a": 1, "b": 2}</tool><|im_end|>\n'
         '<|im_start|>user\n'
         '<tool_response>\n'
         'Tool add_numbers executed successfully:\n'
         '{"result": 3}\n'
         '</tool_response><|im_end|>\n'
         '<|im_start|>assistant\n']
    """

    def __init__(
        self,
        tools: dict[str, callable],
        tool_schemas: dict[str, dict],
        tokenizer=None,  # type: ignore
        tool_name: str = "tool",
        timeout: float = 10.0,
    ):
        super().__init__()
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.tokenizer = tokenizer
        self.tool_name = tool_name
        self.timeout = timeout

    def _extract_tool_calls(
        self, text: str
    ) -> list[tuple[str, str]]:  # noqa: D415, D301, D209, D205
        """Extract tool calls from text in the format <tool>tool_name\nargs_json</tool>."""
        import re

        pattern = r"<tool>(.*?)\n(.*?)</tool>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _execute_tool(self, tool_name: str, args_json: str) -> dict:
        """Execute a tool with the given arguments."""
        import json
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_context(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Tool execution timed out")

            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(int(seconds))
            try:
                yield
            finally:
                # Disable the alarm
                signal.alarm(0)

        try:
            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                }

            # Parse arguments
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse tool arguments: {str(e)}",
                }

            # Execute with timeout
            with timeout_context(self.timeout):
                result = self.tools[tool_name](**args)
                return {
                    "success": True,
                    "result": result,
                }

        except TimeoutError:
            return {
                "success": False,
                "error": f"Tool execution timed out after {self.timeout} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    def _process_llm_response(self, response: str) -> list[str]:
        """Process LLM response and execute any tool calls found.

        Args:
            response (str): The response from the LLM.

        Returns:
            list[str]: A list of strings, each containing the result of a tool execution.
        """
        tool_calls = self._extract_tool_calls(response)

        results = []
        for tool_name, args_json in tool_calls:
            result = self._execute_tool(tool_name, args_json)

            if result["success"]:
                results.append(
                    f"Tool {tool_name} executed successfully:\n{result['result']}"
                )
            else:
                results.append(f"Tool {tool_name} failed:\n{result['error']}")

        return results

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as next_tensordict_flat:
                # Call the transform on the flattened tensordict
                next_tensordict_flat = self._call(next_tensordict_flat)
            return next_tensordict

        # Check that base_env is on history mode
        parent = self.parent
        if parent is None:
            raise RuntimeError("MCPToolTransform must be used with a ChatEnv")
        base_env = parent.base_env
        if base_env.input_mode != "history":
            raise RuntimeError(
                "MCPToolTransform must be used with a ChatEnv in history mode"
            )

        # Convert text to a history
        history = next_tensordict["history"].prompt
        # Isolate last element, which should be our action
        local_history = history[..., -1]

        procs = []
        # Iterate over env batch-size
        for t in local_history.content:
            results = self._process_llm_response(t)
            if len(results) == 0:
                procs.append(None)
                continue
            procs.append(
                [History(role=self.tool_name, content=result) for result in results]
            )

        # If there is no tool response, just skip entire batch
        if all(p is None for p in procs):
            return next_tensordict
        if any(p is None for p in procs):
            procs = [p if p is not None else [] for p in procs]
        # We need to have the same number of items for each element in the batch
        if len(procs) > 1 and not all(len(p) == len(procs[0]) for p in procs):

            def fill_procs(proc: list[History], max_len: int) -> list[History]:
                if len(proc) == max_len:
                    return proc
                return proc + [History(role="<none>", content="")] * (
                    max_len - len(proc)
                )

            max_len = max(len(p) for p in procs)
            procs = [fill_procs(p, max_len) for p in procs]
        # Procs has the shape of the batch-size. We can cat along dim=-1
        procs = lazy_stack([lazy_stack(p) for p in procs])
        history.extend(procs, dim=-1)
        next_tensordict["history"].prompt = history
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset
