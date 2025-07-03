# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from typing import TextIO

import torch

from tensordict import lazy_stack, TensorDictBase
from torchrl import torchrl_logger
from torchrl.data.llm import History

from torchrl.envs import Transform


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
                    f"Code block {i+1} executed successfully:\n{result['stdout']}"
                )
            else:
                results.append(f"Code block {i+1} failed:\n{result['stderr']}")

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
        for i, t in enumerate(local_history.content):
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
