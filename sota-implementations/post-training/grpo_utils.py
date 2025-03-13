# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.envs import Transform
from torchrl.data import Composite, TensorSpec, Unbounded
from tensordict.utils import _zip_strict
from tensordict import TensorDictBase, TensorDict
from tensordict import NestedKey
BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)

class PrepareQuestion(Transform):
    def __init__(self, in_keys: list[NestedKey] | None = None, out_keys: list[NestedKey] | None = None):
        if in_keys is None:
            in_keys = ["text"]
        if out_keys is None:
            out_keys = list(in_keys)
        super().__init__(in_keys, out_keys)

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            string = tensordict.get(in_key)
            tensordict.set(out_key, self._modify_str(string))
        return tensordict

    def _modify_str(self, obs: str | list[str] | NonTensorData | NonTensorStack) -> NonTensorData | NonTensorStack:
        if isinstance(obs, NonTensorData):
            return self._modify_str(obs.data)
        if isinstance(obs, NonTensorStack):
            return self._modify_str(obs.tolist())
        if isinstance(obs, list):
            return NonTensorStack(
                *[BASE_PROMPT % obs for obs in obs]
            )
        return NonTensorData(BASE_PROMPT % obs)

    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if out_key != in_key:
                observation_spec[out_key] = observation_spec[in_key].clone()
        return observation_spec

class ShapedCorrectnessReward(Transform):
    def __init__(self, tokenizer, in_keys: list[NestedKey] | None=None, out_keys: list[NestedKey] | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        if in_keys is None:
            in_keys = ["text", "answer"]
        if not isinstance(in_keys, list) or len(in_keys) != 2:
            raise ValueError("ShapedCorrectnessReward requires in_keys to be of type list and have 2 elements.")
        if out_keys is None:
            out_keys = ["reward_answer", "reward_think", "reward_right", "reward_contained", "reward", "success"]
        super().__init__(in_keys, out_keys)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        from xml.etree import ElementTree as ET
        # Get the completion
        responses = next_tensordict[self.in_keys[0]]  # batch_size, grpo_size, L
        answers = next_tensordict[self.in_keys[1]]  # batch_size, grpo_size
        if isinstance(responses, torch.Tensor):
            if responses.ndim  == 3:
                batch_size, grpo_size, _ = responses.shape
            # decode
            text_completion = self.tokenizer.decode(
                responses.flatten(0, 1).tolist()
            )
        else:
            text_completion = responses
        # Decomposed reward
        tds = []
        for answer, compl in zip(answers, text_completion):
            try:
                cot, potential_answer = self.extract_tags("<think>" + compl) #.replace("<<", "").replace(">>", ""))
            except ET.ParseError:
                cot, potential_answer = ("", "")
            tds.append(self.single_shaped_correctness_reward(potential_answer, cot))
        tds = torch.stack(tds)
        if isinstance(responses, torch.Tensor) and responses.ndim  == 3:
            tds = tds.reshape(batch_size, grpo_size)
        tds = tds.apply(lambda t: t.unsqueeze(-1))
        return next_tensordict.update(tds)

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        shape = reward_spec.shape + (1,)
        reward_spec.update(Composite(
            reward_answer=Unbounded(shape),
            reward_think=Unbounded(shape),
            reward_right=Unbounded(shape),
            reward_contained=Unbounded(shape),
            reward=Unbounded(shape),
            success=Unbounded(shape, dtype=torch.bool),
        ))
        return reward_spec

    @classmethod
    def single_shaped_correctness_reward(cls, answer: str, cot: str) -> TensorDict:

        reward_answer = 5.0 * (len(answer) == 1)

        reward_think = 5.0 * (len(cot) == 1)

        # One of the answer tags has the right answer
        reward_right = 20.0 * (any(attempt == answer for attempt in answer))

        # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
        reward_contained = 10.0 * (any((answer in attempt) for attempt in answer))

        success = len(answer) > 0 and answer[-1] == answer
        # Compose the rewards
        reward = 100.0 * float(success) + (reward_answer + reward_think + reward_contained + reward_right) * (1- float(success))

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
    def extract_tags(text: str) -> Tuple[str, str]:
        """
        Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
        The values are lists of strings, with each string being the content of a tag.
        """
        from xml.etree import ElementTree as ET

        xml_string = f"<root>{text}</root>"
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            return ("", "")

        return (
            root.find("think").text if root.find("think") is not None else "",
            root.find("answer").text if root.find("answer") is not None else "",
        )
