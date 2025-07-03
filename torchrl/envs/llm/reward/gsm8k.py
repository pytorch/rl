# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

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

    Args:
        tokenizer (AutoTokenizer from transformers): the tokenizer associated with the model.
        in_keys (list of NestedKey): the input keys. If None, will be automatically determined based on parent's input_mode.
        out_keys (list of NestedKey): the output keys. Defaults to `[ "reward_answer", "reward_think", "reward_right", "reward_contained", "reward", "success"]`.
        eos_token (str): the end of sentence token. Defaults to `tokenizer.eos_token` if not provided.
        set_done_if_answer (bool): whether to set the done flag to `True` when an answer is present. Defaults to `True`.
        input_mode (Literal["history", "text", "tokens"]): the input mode of the parent environment.
            Defaults to `None` (will be automatically determined based on parent's input_mode).
    """

    def __init__(
        self,
        tokenizer,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
        eos_token: str | None = None,
        set_done_if_answer: bool = True,
        input_mode: Literal["history", "text", "tokens"] | None = None,
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

        if out_keys is None:
            out_keys = [
                "reward_answer",
                "reward_think",
                "reward_right",
                "reward_contained",
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
                raise ValueError(f"No base env found for {self}")

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
        from xml.etree import ElementTree as ET

        if next_tensordict.batch_dims > 1:
            with tensordict.view(-1) as td_view, next_tensordict.view(
                -1
            ) as next_td_view:
                self._step(td_view, next_td_view)
            # did update in place
            return next_tensordict

        # Get the completion based on input_mode
        self._maybe_get_in_keys()
        responses = tensordict[self.in_keys[0]]  # batch_size, grpo_size, L

        # Handle different response types based on input_mode
        input_mode = self.input_mode
        if input_mode == "history":
            # responses is a History object, extract the text content
            responses = lazy_stack([r[..., -1] for r in responses.unbind(0)])
            if hasattr(responses, "content"):
                # If it's a History object with content attribute
                text_completion = responses.content
                if is_non_tensor(text_completion):
                    text_completion = text_completion.tolist()
                if not isinstance(text_completion, list):
                    text_completion = [text_completion]
            elif hasattr(responses, "apply_chat_template"):
                # If it's a History object, apply chat template to get text
                text_completion = responses.apply_chat_template(
                    tokenizer=self.tokenizer, add_generation_prompt=False
                )
                if not isinstance(text_completion, list):
                    text_completion = [text_completion]
            else:
                # Fallback: try to convert to string
                text_completion = [str(responses)]
        elif input_mode == "text":
            # responses is already text
            if isinstance(responses, str):
                text_completion = [
                    responses for _ in range(next_tensordict.batch_size[0])
                ]
            elif not isinstance(responses, list):
                text_completion = [responses]
            else:
                text_completion = responses
        elif input_mode == "tokens":
            # responses is tokens, need to decode
            if isinstance(responses, torch.Tensor):
                if responses.ndim == 3:
                    batch_size, grpo_size, _ = responses.shape
                # decode
                text_completion = self.tokenizer.decode(
                    responses.flatten(0, 1).tolist()
                )
                if not isinstance(text_completion, list):
                    text_completion = [
                        text_completion for _ in range(next_tensordict.batch_size[0])
                    ]
            else:
                # Assume it's already a list of token sequences
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
        answers = next_tensordict[self.in_keys[1]]  # batch_size, grpo_size

        # Decomposed reward
        tds = []
        # torchrl_logger.info(f"{answers=}")
        # torchrl_logger.info(f"{text_completion=}")
        for answer, compl in _zip_strict(answers, text_completion):
            try:
                if not compl.startswith("<think>"):
                    compl = "<think>" + compl
                if compl.endswith("<|im_end|>"):
                    compl = compl.removesuffix("<|im_end|>")
                cot, potential_answer = self.extract_tags(compl)
            except ET.ParseError:
                cot, potential_answer = ("", "")
            if potential_answer is None:
                potential_answer = ""
            if cot is None:
                cot = ""
            # TODO: in tune, the answer is parsed during dataloading
            #  we could create a similar dataclass for both proposed and real answer
            #  With tensorclass comparison should be easy
            cot_orig, answer = answer.split("#### ")
            tds.append(
                self._single_shaped_correctness_reward(
                    answer, [potential_answer], [cot]
                )
            )
        tds = torch.stack(tds)
        if isinstance(responses, torch.Tensor) and responses.ndim == 3:
            batch_size, grpo_size, _ = responses.shape
            tds = tds.reshape(batch_size, grpo_size)
        # Rewards need to have shape broadcastable to [batch x tokens x 1]
        tds = tds.apply(lambda t: t.unsqueeze(-1).unsqueeze(-1))
        # Add the rewards, in case some have already been written
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
                reward_contained=Unbounded(shape),
                reward=Unbounded(shape),
                success=Unbounded(shape, dtype=torch.bool),
            )
        )
        return reward_spec

    @classmethod
    def _single_shaped_correctness_reward(
        cls, true_answer: str, potential_answer: list[str], cot: list[str]
    ) -> TensorDict:
        # TODO: In tune, these end up being lists
        # torchrl_logger.info(f"{potential_answer=}")
        # torchrl_logger.info(f"{true_answer=}")
        if isinstance(potential_answer, str):
            potential_answer = [potential_answer]
        if isinstance(cot, str):
            cot = [cot]

        # Format quality rewards (always applied)
        reward_answer = 5.0 * (len(potential_answer) == 1)
        reward_think = 5.0 * (len(cot) == 1)

        # Answer correctness rewards
        reward_right = 20.0 * (
            any(attempt == true_answer for attempt in potential_answer)
        )
        reward_contained = 10.0 * (
            any((true_answer in attempt) for attempt in potential_answer)
        )

        success = len(potential_answer) > 0 and potential_answer[-1] == true_answer

        # Base success reward (lower than before to make format quality more important)
        base_success_reward = 60.0 if success else 0.0

        # Compose the rewards - always include format quality, even when successful
        reward = (
            base_success_reward
            + reward_answer
            + reward_think
            + reward_contained
            + reward_right
        )

        rewards = TensorDict(
            reward_answer=reward_answer,
            reward_think=reward_think,
            reward_right=reward_right,
            reward_contained=reward_contained,
            reward=reward,
            success=success,
        )
        return rewards

    @staticmethod
    def extract_tags(text: str) -> tuple[str, str]:
        """Parse XML-like tags from text.

        Returns: a dictionary with keys 'think' and 'answer'.
            The values are lists of strings, with each string being the content of a tag.

        """
        from xml.etree import ElementTree as ET

        xml_string = f"<root>{text}</root>"
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError:
            return ("", "")

        think_elem = root.find("think")
        answer_elem = root.find("answer")
        return (
            think_elem.text
            if think_elem is not None and think_elem.text is not None
            else "",
            answer_elem.text
            if answer_elem is not None and answer_elem.text is not None
            else "",
        )
