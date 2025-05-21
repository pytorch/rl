# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Modifications from original script.

Modifications include:

- TensorDict embedding
- Modification of key names
- make IfEvalScorer a TorchRL transform

"""

from __future__ import annotations

import importlib.util
import re

import torch
from jedi.inference.gradual.typing import Callable

from tensordict import NestedKey, NonTensorData, TensorClass, TensorDict, TensorDictBase
from tensordict.tensorclass import is_non_tensor

from torchrl.data.tensor_specs import Composite, TensorSpec, Unbounded
from torchrl.envs import Transform

_has_langdetect = importlib.util.find_spec("langdetect") is not None
_has_nltk = importlib.util.find_spec("nltk") is not None
_has_immutabledict = importlib.util.find_spec("immutabledict") is not None


class IFEvalScoreData(TensorClass):
    """IFEval score container."""

    prompt_level_strict_acc: torch.Tensor
    inst_level_strict_acc: torch.Tensor
    prompt_level_loose_acc: torch.Tensor
    inst_level_loose_acc: torch.Tensor


def _process_results(data: TensorDict, response: str | NonTensorData) -> TensorDict:
    if not _has_langdetect:
        raise ImportError("langdetect must be installed to user IFEvalScorer.")
    if not _has_immutabledict:
        raise ImportError("immutabledict must be installed to user IFEvalScorer.")

    from ._instructions_main import (
        _InputExample,
        _test_instruction_following_loose,
        _test_instruction_following_strict,
    )

    inp = _InputExample(
        key=data["key"],
        instruction_id_list=data["instruction_id_list"],
        prompt=data["text"],
        kwargs=data["kwargs"],
    )

    out_strict = _test_instruction_following_strict(inp, response)
    out_loose = _test_instruction_following_loose(inp, response)

    return IFEvalScoreData(
        prompt_level_strict_acc=out_strict.follow_all_instructions,
        inst_level_strict_acc=out_strict.follow_instruction_list,
        prompt_level_loose_acc=out_loose.follow_all_instructions,
        inst_level_loose_acc=out_loose.follow_instruction_list,
    )


class IfEvalScorer(Transform):
    """Scorer for the IF-Eval task.

    For the IFEval dataset format, see https://huggingface.co/datasets/google/IFEval

    The score data is written under the `score_key` using the :class:`~torchrl.envs.llm.IFEvalScoreData` data structure.
    Scores can be aggregated on a single reward by using the `aggregate_reward` keyword argument in the constructor, which
    can be a bool or a function.

    Keyword Args:
        instruction_ids_key (NestedKey, optional): The column name for the list of instruction ids. Defaults to "instruction_id_list".
        prompt_key (NestedKey, optional): The column name for the prompt. Defaults to "text".
        keyword_args_key (NestedKey, optional): The column name for the keyword arguments to the instruction builder. Defaults to "kwargs".
        id_key (NestedKey, optional): The column name for the unique identifier. Defaults to "key".
        response_column (NestedKey, optional): The column name for the response. Defaults to "text_response".
        score_key (NestedKey, optional): The key to store the score. Defaults to "ifeval_score".
        aggregate_reward (bool, callable, optional): Whether to aggregate the reward or not. If a Callable is passed,
            it must take as input an :class:`~torchrl.envs.llm.IFEvalScoreData` instance and return a tensor with shape
            identical to the env batch-size with an additional trailing singleton dimension.
            Defaults to `True`. The default aggregator is a simple sum over the fields of :class:`~torchrl.envs.llm.IFEvalScoreData`.

    .. note:: `IFEvalScorer` requires the following libraries to be installed: `langdetect`, `nltk` and `immutabledict`.

    """

    def __init__(
        self,
        *,
        instruction_ids_key: NestedKey = "instruction_id_list",
        prompt_key: NestedKey = "text",
        keyword_args_key: NestedKey = "kwargs",
        id_key: NestedKey = "key",
        response_column: NestedKey = "text_response",
        score_key: NestedKey = "ifeval_score",
        aggregate_reward: bool | Callable[[IFEvalScoreData], torch.Tensor] = True,
    ):
        self.aggregate_reward = aggregate_reward
        self.score_key = score_key
        out_keys = [self.score_key]
        if aggregate_reward:
            out_keys.append("reward")
        super().__init__(
            in_keys=[
                instruction_ids_key,
                prompt_key,
                keyword_args_key,
                id_key,
                response_column,
            ],
            out_keys=out_keys,
        )
        if not _has_langdetect:
            raise ImportError("langdetect must be installed to user IFEvalScorer.")
        if not _has_nltk:
            raise ImportError("nltk must be installed to user IFEvalScorer.")
        self.instruction_ids_key = instruction_ids_key
        self.response_key = response_column
        self.keyword_args_key = keyword_args_key
        self.prompt_key = prompt_key
        self.id_key = id_key

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if tensordict.ndim:
            return torch.stack(
                [
                    self._step(td, next_td)
                    for td, next_td in zip(
                        tensordict.unbind(0), next_tensordict.unbind(0)
                    )
                ]
            )
        response = tensordict.get(self.response_key)
        if is_non_tensor(response):
            response = response.data

        # TODO: This should be a distinct module
        # Regular expression patterns to match think and answer blocks
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        # Extract think block
        think_match = re.search(think_pattern, response, re.DOTALL)
        if think_match:
            think_match.group(1).strip()
        else:
            pass

        # Extract answer block
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        if answer_match:
            answer_block = answer_match.group(1).strip()
        else:
            answer_block = ""

        score = _process_results(tensordict, answer_block)
        next_tensordict.set(
            self.score_key,
            score,
        )
        if self.aggregate_reward:
            if callable(self.aggregate_reward):
                reward_func = self.aggregate_reward
            else:
                reward_func = (
                    lambda td: td.sum(reduce=True, dim="feature")
                    .to(torch.get_default_dtype())
                    .unsqueeze(-1)
                )
            reward = reward_func(score)
            next_tensordict.set("reward", reward)

        return next_tensordict

    @property
    def expected_keys(self) -> list[str]:
        return [
            self.instruction_ids_key,
            self.prompt_key,
            self.keyword_args_key,
            self.id_key,
            self.response_key,
        ]

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_spec["reward"] = Unbounded(
            reward_spec.shape + (1,), dtype=torch.get_default_dtype()
        )
        return reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec[self.score_key] = Composite(
            prompt_level_strict_acc=Unbounded(
                shape=observation_spec.shape, dtype=torch.bool
            ),
            inst_level_strict_acc=Unbounded(
                shape=observation_spec.shape, dtype=torch.bool
            ),
            prompt_level_loose_acc=Unbounded(
                shape=observation_spec.shape, dtype=torch.bool
            ),
            inst_level_loose_acc=Unbounded(
                shape=observation_spec.shape, dtype=torch.bool
            ),
        )
        return observation_spec
