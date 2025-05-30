# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.data import Composite, Unbounded
from torchrl.envs import Transform


class GSM8KRewardParser(Transform):
    """Reward parser for GSM8KEnv or make_gsm8k_env.

    Args:
        tokenizer (AutoTokenizer from transformers): the tokenizer asssociated with the model.
        in_keys (list of NestedKey): the input keys. Defaults to `["text_response", "answer"]`.
        out_keys (list of NestedKey): the output keys. Defaults to `[ "reward_answer", "reward_think", "reward_right", "reward_contained", "reward", "success"]`.
        eos_token (str): the end of sentence token. Defaults to `tokenizer.eos_token` if not provided.

    """

    def __init__(
        self,
        tokenizer,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
        eos_token: str | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token = eos_token if eos_token is not None else tokenizer.eos_token
        if in_keys is None:
            in_keys = ["text_response", "answer"]
        if not isinstance(in_keys, list) or len(in_keys) != 2:
            raise ValueError(
                f"{type(self).__name__} requires in_keys to be of type list and have 2 elements."
            )
        if out_keys is None:
            out_keys = [
                "reward_answer",
                "reward_think",
                "reward_right",
                "reward_contained",
                "reward",
                "success",
            ]
        super().__init__(in_keys, out_keys)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        from xml.etree import ElementTree as ET

        # Get the completion

        responses = tensordict[self.in_keys[0]]  # batch_size, grpo_size, L

        # if isinstance(prompts, str):
        #     prompts = [prompts]

        if isinstance(responses, str):
            responses = [responses]

        # responses = [p + r for p, r in zip(prompts, responses)]

        if self.eos_token is not None:
            responses = [r.removesuffix(self.eos_token) for r in responses]
        answers = next_tensordict[self.in_keys[1]]  # batch_size, grpo_size
        if isinstance(responses, torch.Tensor):
            if responses.ndim == 3:
                batch_size, grpo_size, _ = responses.shape
            # decode
            text_completion = self.tokenizer.decode(responses.flatten(0, 1).tolist())
        else:
            text_completion = responses
        if not isinstance(text_completion, list):
            text_completion = [text_completion]
        # Decomposed reward
        tds = []
        torchrl_logger.info(f"{answers=}")
        torchrl_logger.info(f"{text_completion=}")
        for answer, compl in zip(answers, text_completion):
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
                self.single_shaped_correctness_reward(answer, potential_answer, cot)
            )
        tds = torch.stack(tds)
        if isinstance(responses, torch.Tensor) and responses.ndim == 3:
            tds = tds.reshape(batch_size, grpo_size)
        # Rewards need to have shape broadcastable to [batch x tokens x 1]
        tds = tds.apply(lambda t: t.unsqueeze(-1).unsqueeze(-1))
        # Add the rewards, in case some have already been written
        next_td_exist = next_tensordict.select(*tds.keys(True, True), strict=False)
        if not next_td_exist.is_empty():
            tds = tds.add(
                next_td_exist, default=torch.zeros((), device=next_tensordict.device)
            )
        return next_tensordict.update(tds)

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
    def single_shaped_correctness_reward(
        cls, true_answer: str, potential_answer: list[str], cot: list[str]
    ) -> TensorDict:
        # TODO: In tune, these end up being lists
        torchrl_logger.info(f"{potential_answer=}")
        torchrl_logger.info(f"{true_answer=}")
        if isinstance(potential_answer, str):
            potential_answer = [potential_answer]
        if isinstance(cot, str):
            cot = [cot]
        reward_answer = 5.0 * (len(potential_answer) == 1)

        reward_think = 5.0 * (len(cot) == 1)

        # One of the answer tags has the right answer
        reward_right = 20.0 * (
            any(attempt == true_answer for attempt in potential_answer)
        )

        # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
        reward_contained = 10.0 * (
            any((true_answer in attempt) for attempt in potential_answer)
        )

        success = len(potential_answer) > 0 and potential_answer[-1] == true_answer
        # Compose the rewards
        reward = 100.0 * float(success) + (
            reward_answer + reward_think + reward_contained + reward_right
        ) * (1 - float(success))

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

        return (
            root.find("think").text if root.find("think") is not None else "",
            root.find("answer").text if root.find("answer") is not None else "",
        )
