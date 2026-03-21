# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import re
from typing import Literal

import torch
from tensordict import lazy_stack, NestedKey, TensorDict, TensorDictBase
from tensordict.utils import _zip_strict, is_non_tensor
from torchrl.data import Composite, Unbounded
from torchrl.envs import Transform
from torchrl.envs.common import EnvBase


class CountdownRewardParser(Transform):
    """Reward parser for the Countdown numbers game.

    The Countdown game gives the model a set of source numbers and a target.
    The model must construct an arithmetic expression using each source number
    *at most once* that evaluates to the target.

    The reward follows the standard GRPO convention:

    - ``correct_reward`` (default ``1.0``) when the expression is valid and
      evaluates to the target.
    - ``format_reward`` (default ``0.1``) when the response has a valid
      ``<answer>`` tag but the expression is wrong.
    - ``0.0`` otherwise.

    The ground-truth data is expected to carry a JSON-like string with keys
    ``"target"`` and ``"numbers"`` (stored in the ``"answer"`` field by
    :class:`CountdownEnv`).

    Args:
        tokenizer: the tokenizer associated with the model (optional).
        in_keys (list of NestedKey): the input keys.
        out_keys (list of NestedKey): the output keys.
        eos_token (str): the end-of-sentence token.
        set_done_if_answer (bool): whether to set done when an answer is present.
        input_mode: the input mode of the parent environment.
        format_reward (float): reward for correct format but wrong answer.
        correct_reward (float): reward for a correct answer.
    """

    def __init__(
        self,
        tokenizer=None,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
        eos_token: str | None = None,
        set_done_if_answer: bool = True,
        input_mode: Literal["history", "text", "tokens"] | None = None,
        format_reward: float = 0.1,
        correct_reward: float = 1.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token = (
            eos_token
            if eos_token is not None
            else tokenizer.eos_token
            if tokenizer is not None
            else None
        )
        self.set_done_if_answer = set_done_if_answer
        self._input_mode = input_mode
        self.format_reward = format_reward
        self.correct_reward = correct_reward

        if out_keys is None:
            out_keys = [
                "reward_answer",
                "reward_think",
                "reward_right",
                "reward",
                "success",
            ]
        super().__init__()
        if in_keys is not None:
            self.in_keys = in_keys
        self.out_keys = out_keys

    # ------------------------------------------------------------------
    # input_mode / in_keys discovery
    # ------------------------------------------------------------------

    def _maybe_get_in_keys(self):
        if not self.in_keys:
            parent = getattr(self, "parent", None)
            if parent is not None:
                base_env = getattr(parent, "base_env", None)
                mode = getattr(base_env, "input_mode", None) if base_env else None
                if mode == "history":
                    self.in_keys = [("history", "full"), "answer"]
                elif mode == "text":
                    self.in_keys = [("text", "full"), "answer"]
                elif mode == "tokens":
                    self.in_keys = [("tokens", "full"), "answer"]
            else:
                raise ValueError(
                    f"No base env found for {self} with container {self.container}"
                )

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        self._maybe_get_in_keys()
        return result

    _input_mode = None

    @property
    def input_mode(self):
        if self._input_mode is None:
            input_mode = (
                getattr(self.parent, "input_mode", "history")
                if hasattr(self, "parent") and self.parent is not None
                else "history"
            )
            self._input_mode = input_mode
        return self._input_mode

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if next_tensordict.batch_dims > 1:
            with tensordict.view(-1) as td_view, next_tensordict.view(
                -1
            ) as next_td_view:
                self._step(td_view, next_td_view)
            return next_tensordict

        self._maybe_get_in_keys()
        responses = tensordict[self.in_keys[0]]

        input_mode = self.input_mode
        if input_mode == "history":
            responses = lazy_stack([r[..., -1] for r in responses.unbind(0)])
            if hasattr(responses, "content"):
                text_completion = responses.content
                if is_non_tensor(text_completion):
                    text_completion = text_completion.tolist()
                if not isinstance(text_completion, list):
                    text_completion = [text_completion]
            elif hasattr(responses, "apply_chat_template"):
                text_completion = responses.apply_chat_template(
                    tokenizer=self.tokenizer, add_generation_prompt=False
                )
                if not isinstance(text_completion, list):
                    text_completion = [text_completion]
            else:
                text_completion = [str(responses)]
        elif input_mode == "text":
            if isinstance(responses, str):
                text_completion = [
                    responses for _ in range(next_tensordict.batch_size[0])
                ]
            elif not isinstance(responses, list):
                text_completion = [responses]
            else:
                text_completion = responses
        elif input_mode == "tokens":
            if isinstance(responses, torch.Tensor):
                text_completion = self.tokenizer.decode(
                    responses.flatten(0, 1).tolist()
                )
                if not isinstance(text_completion, list):
                    text_completion = [
                        text_completion for _ in range(next_tensordict.batch_size[0])
                    ]
            else:
                text_completion = []
                for token_seq in responses:
                    if isinstance(token_seq, torch.Tensor):
                        text_completion.append(
                            self.tokenizer.decode(token_seq.tolist())
                        )
                    else:
                        text_completion.append(str(token_seq))
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        if self.eos_token is not None:
            text_completion = [r.removesuffix(self.eos_token) for r in text_completion]
        answers = next_tensordict[self.in_keys[1]]

        tds = []
        for answer, compl in _zip_strict(answers, text_completion):
            if compl.endswith("<|im_end|>"):
                compl = compl.removesuffix("<|im_end|>")
            cot, potential_answer = self.extract_tags(compl)
            target, numbers = self._parse_ground_truth(answer)
            tds.append(
                self._single_correctness_reward(target, numbers, potential_answer, cot)
            )
        tds = torch.stack(tds)
        if isinstance(responses, torch.Tensor) and responses.ndim == 3:
            batch_size, grpo_size, _ = responses.shape
            tds = tds.reshape(batch_size, grpo_size)
        tds = tds.apply(lambda t: t.unsqueeze(-1).unsqueeze(-1))
        next_td_exist = next_tensordict.select(*tds.keys(True, True), strict=False)
        if not next_td_exist.is_empty():
            tds = tds.add(
                next_td_exist, default=torch.zeros((), device=next_tensordict.device)
            )
        next_tensordict = next_tensordict.update(tds)
        if (
            self.set_done_if_answer
            and (reward_answer := (next_tensordict["reward_answer"] > 0)).any()
        ):
            done = next_tensordict.get("done")
            if done is not None:
                next_tensordict.set("done", reward_answer.view_as(done) | done)
            terminated = next_tensordict.get("terminated")
            if terminated is not None:
                next_tensordict.set(
                    "terminated", reward_answer.view_as(terminated) | terminated
                )
        return next_tensordict

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        shape = reward_spec.shape + (1, 1)
        reward_spec.update(
            Composite(
                reward_answer=Unbounded(shape),
                reward_think=Unbounded(shape),
                reward_right=Unbounded(shape),
                reward=Unbounded(shape),
                success=Unbounded(shape, dtype=torch.bool),
            )
        )
        return reward_spec

    # ------------------------------------------------------------------
    # reward logic
    # ------------------------------------------------------------------

    def _single_correctness_reward(
        self,
        target: int,
        numbers: list[int],
        expression: str,
        cot: str,
    ) -> TensorDict:
        has_answer = bool(expression)
        has_think = bool(cot)
        correct = has_answer and self.validate_expression(expression, target, numbers)

        reward_answer = float(has_answer)
        reward_think = float(has_think)

        if correct:
            reward_right = self.correct_reward
        elif has_answer:
            reward_right = self.format_reward
        else:
            reward_right = 0.0

        return TensorDict(
            reward_answer=reward_answer,
            reward_think=reward_think,
            reward_right=reward_right,
            reward=reward_right,
            success=correct,
        )

    # ------------------------------------------------------------------
    # expression validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_expression(expression: str, target: int, numbers: list[int]) -> bool:
        """Check that *expression* evaluates to *target* using only the given *numbers*.

        Each source number may be used at most once.  Only ``+``, ``-``,
        ``*``, ``/`` and parentheses are allowed.
        """
        if not re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", expression):
            return False
        used = [int(n) for n in re.findall(r"\d+", expression)]
        available = list(numbers)
        for n in used:
            if n in available:
                available.remove(n)
            else:
                return False
        try:
            result = eval(expression)  # noqa: S307
        except Exception:
            return False
        return abs(result - target) < 1e-9

    # ------------------------------------------------------------------
    # parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ground_truth(answer: str) -> tuple[int, list[int]]:
        """Parse the ground-truth string produced by :class:`CountdownEnv`.

        Expected format: ``"target=<int>, numbers=<int>,<int>,..."``.
        """
        target_match = re.search(r"target\s*=\s*(\d+)", answer)
        numbers_match = re.search(r"numbers\s*=\s*([\d,\s]+)", answer)
        target = int(target_match.group(1))
        numbers = [int(n.strip()) for n in numbers_match.group(1).split(",")]
        return target, numbers

    @staticmethod
    def extract_tags(text: str) -> tuple[str, str]:
        """Extract think and answer content from a response using regex."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return (
            think_match.group(1).strip() if think_match else "",
            answer_match.group(1).strip() if answer_match else "",
        )
