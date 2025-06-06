# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from tensordict import lazy_stack, TensorDictBase
from torchrl.data.llm import History

from torchrl.envs import Transform


class PythonInterpreter(Transform):
    r"""A transform that executes Python code in the LLM response.

    Args:
        tokenizer (transformers.PretrainedTokenizerBase or str, optional): the tokenizer to use. If ``None``,
            it will attempt to retrieve the tokenizer from the parent :class:`~torchrl.envs.llm.ChatEnv`.
        tool_name (str, optional): the name of the tool to use. Defaults to `"tool"`.

    Example:
        >>> from torchrl.envs.llm.transforms import PythonInterpreter
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> # ChatEnv is a base class for all LLM environments.
        >>> base_env = ChatEnv(
        ...     batch_size=(1,),
        ...     system_prompt="I'm the system, do as I say",
        ...     apply_template=True,
        ...     tokenizer=tokenizer,
        >>> )
        >>> # Append the PythonInterpreter transform to the environment.
        >>> env = base_env.append_transform(PythonInterpreter())
        >>> # Reset the environment.
        >>> r = env.reset(TensorDict(text=["This is the user prompt"], batch_size=(1,)))
        >>> h = r["history"]
        >>> # Apply the chat template to the history.
        >>> history_from_text = h.apply_chat_template(tokenizer=tokenizer)
        >>> assert history_from_text == [
        ...     "<|im_start|>system\nI'm the system, do as I say<|im_end|>\n<|im_start|>user\nThis is the user prompt<|im_end|>\n<|im_start|>assistant\n"
        >>> ]
        >>> # Set the text response to the environment and step the environment.
        >>> r["text_response"] = [
        ...     '''Here is a python code to execute:
        ... ```python
        ... print(1 + 1)
        ... ```<|im_end|>\n
        ... '''
        >>> ]
        >>> s = env.step(r)
        >>> history_str = s["next", "history"].apply_chat_template(tokenizer=tokenizer)
        >>> # Check the history, it should contain the tool response.
        >>> assert history_str == [
        ...     "<|im_start|>system\n"
        ...     "I'm the system, do as I say<|im_end|>\n"
        ...     "<|im_start|>user\n"
        ...     "This is the user prompt<|im_end|>\n"
        ...     "<|im_start|>assistant\n"
        ...     "Here is a python code to execute:\n"
        ...     "```python\n"
        ...     "print(1 + 1)\n"
        ...     "```<|im_end|>\n"
        ...     "<|im_start|>user\n"
        ...     "<tool_response>\n"
        ...     "Code block 1 executed successfully:\n"
        ...     "2\n"
        ...     "\n"
        ...     "</tool_response><|im_end|>\n"
        ...     "<|im_start|>assistant\n"
        >>> ]
        >>> # Convert the history as text back to a History object
        >>> history_from_text = History.from_text(history_str, chat_template_name="qwen")
        >>> assert (
        ...     history_from_text
        ...     == lazy_stack(
        ...         [
        ...             History(role="system", content="I'm the system, do as I say"),
        ...             History(role="user", content="This is the user prompt"),
        ...             History(
        ...                 role="assistant",
        ...                 content="Here is a python code to execute:\n```python\nprint(1 + 1)\n```",
        ...             ),
        ...             History(
        ...                 role="user",
        ...                 content="<tool_response>\nCode block 1 executed successfully:\n2\n\n</tool_response>",
        ...                 tool_responses=["Code block 1 executed successfully:\n2\n"],
        ...             ),
        ...         ]
        ...     ).unsqueeze(0)
        >>> ).all()
    """

    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa, type: ignore
        tool_name: str = "tool",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tool_name = tool_name

    @property
    def tokenizer(self) -> transformers.AutoTokenizer | None:  # noqa, type: ignore
        """The tokenizer to use.

        If ``None``, it will attempt to retrieve the tokenizer from the parent :class:`~torchrl.envs.llm.ChatEnv`.
        """
        if self._tokenizer is None and self.parent is not None:
            self._tokenizer = getattr(self.parent.base_env, "tokenizer", None)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(
        self, tokenizer: transformers.AutoTokenizer | None  # noqa, type: ignore
    ):
        self._tokenizer = tokenizer

    def _extract_python_code(self, text: str) -> list[str]:
        """Extract Python code blocks from markdown-style formatting."""
        # Pattern to match ```python ... ``` blocks
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _execute_python_code(self, code: str) -> dict:
        """Safely execute Python code and return results."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute the code
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout
            )

            # Clean up
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
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def _process_llm_response(self, response: str) -> list[str]:
        """Process LLM response and execute any Python code found."""
        code_blocks = self._extract_python_code(response)

        results = []
        for i, code in enumerate(code_blocks):
            result = self._execute_python_code(code)

            if result["success"]:
                results.append(
                    f"Code block {i+1} executed successfully:\n{result['stdout']}"
                )
            else:
                results.append(f"Code block {i+1} failed:\n{result['stderr']}")

        return results

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if next_tensordict.batch_dims > 1:
            with next_tensordict.view(-1) as next_tensordict_flat:
                # Call the transform on the flattened tensordict
                next_tensordict_flat = self._call(next_tensordict_flat)
            return next_tensordict

        # Convert text to a history
        history = next_tensordict["history"]
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
        next_tensordict["history"] = history
        return next_tensordict
