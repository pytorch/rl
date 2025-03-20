# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict
from torch import nn

from torchrl.collectors import LocalWeightUpdaterBase
from torchrl.data import Composite, TensorSpec, Unbounded
from torchrl.envs import Transform


class HF2vLLMLocalWeightUpdater(LocalWeightUpdaterBase):
    hf_params: TensorDictBase | None = None
    vllm_params: TensorDictBase | None = None

    def __init__(self, hf_model: nn.Module, vllm_model: vllm.LLM):  # noqa
        self.vllm_model = vllm_model
        self.hf_model = hf_model

    def _get_server_weights(self) -> TensorDictBase:
        # Get weight from hf model
        if self.hf_params is None:
            self.hf_params = TensorDict.from_module(self.hf_model).data.lock_()
        return self.hf_params

    def _get_local_weights(self) -> TensorDictBase:
        if self.vllm_params is None:
            self.vllm_model = TensorDict.from_module(
                self.vllm_model.llm_engine.model_executor.driver_worker.model_runner.inference_model
            ).data.lock_()
        return self.vllm_model

    def _maybe_map_weights(
        self, server_weights: TensorDictBase, local_weights: TensorDictBase
    ) -> TensorDictBase:
        return self.format_hf_weights(td=server_weights, to_vllm=True)

    @classmethod
    def format_hf_weights(
        cls,
        model: nn.Module | None = None,
        td: TensorDictBase | None = None,
        *,
        from_,
        to_vllm=False,
        update_model=False,
    ):
        if td is None:
            if model is None:
                raise TypeError("A model or a tensordict is required.")
            td = TensorDict.from_module(model)
        if to_vllm and update_model:
            raise TypeError(
                "Cannot update model with to_vllm=True as the weight format has changed."
            )
        for k in list(td.keys(True)):
            if k[-1] == "q_proj":
                keys = [k, k[:-1] + ("k_proj",), k[:-1] + ("v_proj",)]
                qkv = torch.stack([td[_k].data for _k in keys])
                if not to_vllm:
                    splits = qkv.chunk(3)
                    for _k, split in zip(keys, splits):
                        td.pop(_k)
                        td.set(_k, split.apply(lambda x: torch.nn.Parameter(x)))
                else:
                    qkv = qkv.apply(lambda x: torch.nn.Parameter(x))
                    td.set(k[:-1] + ("qkv_proj",), qkv)
        if update_model:
            if model is None:
                raise TypeError("The model must be provided to be updated.")
            td.to_module(model)
        return td


BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)


class PrepareQuestion(Transform):
    def __init__(
        self,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
    ):
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

    def _modify_str(
        self, obs: str | list[str] | NonTensorData | NonTensorStack
    ) -> NonTensorData | NonTensorStack:
        if isinstance(obs, NonTensorData):
            return self._modify_str(obs.data)
        if isinstance(obs, NonTensorStack):
            return self._modify_str(obs.tolist())
        if isinstance(obs, list):
            return NonTensorStack(*[BASE_PROMPT % obs for obs in obs])
        return NonTensorData(BASE_PROMPT % obs)

    def _apply_transform(self, obs: torch.Tensor) -> None:
        return obs

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if out_key != in_key:
                observation_spec[out_key] = observation_spec[in_key].clone()
        return observation_spec


class ShapedCorrectnessReward(Transform):
    def __init__(
        self,
        tokenizer,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if in_keys is None:
            in_keys = ["text", "answer"]
        if not isinstance(in_keys, list) or len(in_keys) != 2:
            raise ValueError(
                "ShapedCorrectnessReward requires in_keys to be of type list and have 2 elements."
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
        responses = next_tensordict[self.in_keys[0]]  # batch_size, grpo_size, L
        answers = next_tensordict[self.in_keys[1]]  # batch_size, grpo_size
        if isinstance(responses, torch.Tensor):
            if responses.ndim == 3:
                batch_size, grpo_size, _ = responses.shape
            # decode
            text_completion = self.tokenizer.decode(responses.flatten(0, 1).tolist())
        else:
            text_completion = responses
        # Decomposed reward
        tds = []
        for answer, compl in zip(answers, text_completion):
            try:
                cot, potential_answer = self.extract_tags(
                    "<think>" + compl
                )  # .replace("<<", "").replace(">>", ""))
            except ET.ParseError:
                cot, potential_answer = ("", "")
            tds.append(
                self.single_shaped_correctness_reward(answer, potential_answer, cot)
            )
        tds = torch.stack(tds)
        if isinstance(responses, torch.Tensor) and responses.ndim == 3:
            tds = tds.reshape(batch_size, grpo_size)
        tds = tds.apply(lambda t: t.unsqueeze(-1))
        return next_tensordict.update(tds)

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        shape = reward_spec.shape + (1,)
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
        """
        Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
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
