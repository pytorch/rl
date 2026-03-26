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


class GSM8KRewardParser(Transform):
    """Reward parser for GSM8KEnv or make_gsm8k_env.

    This parser automatically detects the input_mode from the parent environment and handles
    responses accordingly:
    - "history" mode: response is in ("history", "response") and is a History object
    - "text" mode: response is in ("text", "response") and is text
    - "tokens" mode: response is in ("tokens", "response") and is tokens

    The reward follows the standard GRPO convention:

    - ``1.0`` if the extracted answer matches the ground truth (after normalization).
    - ``format_reward`` (default ``0.1``) if the response has a valid ``<answer>`` tag but
      the answer is wrong.
    - ``0.0`` otherwise (no parseable answer).

    Args:
        tokenizer (AutoTokenizer from transformers): the tokenizer associated with the model.
        in_keys (list of NestedKey): the input keys. If None, will be automatically determined based on parent's input_mode.
        out_keys (list of NestedKey): the output keys. Defaults to
            ``["reward_answer", "reward_think", "reward_right", "reward", "success"]``.
        eos_token (str): the end of sentence token. Defaults to ``tokenizer.eos_token`` if not provided.
        set_done_if_answer (bool): whether to set the done flag to ``True`` when an answer is present. Defaults to ``True``.
        input_mode (Literal["history", "text", "tokens"]): the input mode of the parent environment.
            Defaults to ``None`` (will be automatically determined based on parent's input_mode).
        format_reward (float): reward for correct format but wrong answer. Defaults to ``0.1``.
        correct_reward (float): reward for correct answer. Defaults to ``1.0``.
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

    def _maybe_get_in_keys(self):
        if not self.in_keys:
            parent = getattr(self, "parent", None)
            if parent is not None:
                if getattr(parent, "base_env", None) is not None:
                    if getattr(parent.base_env, "input_mode", None) == "history":
                        self.in_keys = [("history", "full"), "answer"]
                    elif getattr(parent.base_env, "input_mode", None) == "text":
                        self.in_keys = [("text", "full"), "answer"]
                    elif getattr(parent.base_env, "input_mode", None) == "tokens":
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
                if responses.ndim == 3:
                    batch_size, grpo_size, _ = responses.shape
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
            _unused, answer = answer.split("#### ")
            tds.append(self._single_correctness_reward(answer, potential_answer, cot))
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

    def _single_correctness_reward(
        self, true_answer: str, potential_answer: str, cot: str
    ) -> TensorDict:
        has_answer = bool(potential_answer)
        has_think = bool(cot)

        norm_true = self.normalize_answer(true_answer)
        correct = has_answer and self.normalize_answer(potential_answer) == norm_true

        reward_answer = float(has_answer)
        reward_think = float(has_think)

        if correct:
            reward_right = self.correct_reward
        elif has_answer:
            reward_right = self.format_reward
        else:
            reward_right = 0.0

        reward = reward_right

        return TensorDict(
            reward_answer=reward_answer,
            reward_think=reward_think,
            reward_right=reward_right,
            reward=reward,
            success=correct,
        )

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize a numerical answer string for comparison.

        Strips whitespace, removes commas/dollar signs/percent signs, and
        normalizes trailing decimal zeros (e.g. ``"120.0"`` becomes ``"120"``).
        """
        answer = answer.strip()
        answer = answer.replace(",", "").replace("$", "").replace("%", "")
        answer = answer.rstrip(".")
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        return answer

    @staticmethod
    def extract_tags(text: str) -> tuple[str, str]:
        """Extract think and answer content from a response using regex.

        More robust than XML parsing since LLM outputs frequently contain
        malformed markup.

        Returns:
            A ``(think_content, answer_content)`` tuple.  Empty strings are
            returned when the corresponding tag is absent.
        """
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return (
            think_match.group(1).strip() if think_match else "",
            answer_match.group(1).strip() if answer_match else "",
        )
