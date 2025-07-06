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
from typing import Callable

import torch
from tensordict import NestedKey, NonTensorData, TensorClass, TensorDict, TensorDictBase
from tensordict.tensorclass import is_non_tensor
from torchrl._utils import logger as torchrl_logger

from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs import Transform

_has_langdetect = importlib.util.find_spec("langdetect") is not None
_has_nltk = importlib.util.find_spec("nltk") is not None
_has_immutabledict = importlib.util.find_spec("immutabledict") is not None


class IFEvalScoreData(TensorClass):
    """IFEval score container."""

    prompt_level_strict_acc: torch.Tensor | None
    inst_level_strict_acc: torch.Tensor | None
    prompt_level_loose_acc: torch.Tensor | None
    inst_level_loose_acc: torch.Tensor | None

    def __post_init__(self):
        prompt_level_loose_acc = self.get(
            "prompt_level_loose_acc", as_padded_tensor=True
        )
        inst_level_loose_acc = self.get("inst_level_loose_acc", as_padded_tensor=True)
        prompt_level_strict_acc = self.get(
            "prompt_level_strict_acc", as_padded_tensor=True
        )
        inst_level_strict_acc = self.get("inst_level_strict_acc", as_padded_tensor=True)

        if prompt_level_loose_acc is None:
            self.prompt_level_loose_acc = torch.zeros(self.batch_size + (1,))
        elif prompt_level_loose_acc.ndim == self.ndim:
            self.prompt_level_loose_acc = prompt_level_loose_acc.unsqueeze(-1)

        if inst_level_loose_acc is None:
            self.inst_level_loose_acc = torch.zeros(self.batch_size + (1,))
        elif inst_level_loose_acc.ndim == self.ndim:
            self.inst_level_loose_acc = inst_level_loose_acc.unsqueeze(-1)

        if prompt_level_strict_acc is None:
            self.prompt_level_strict_acc = torch.zeros(self.batch_size + (1,))
        elif prompt_level_strict_acc.ndim == self.ndim:
            self.prompt_level_strict_acc = prompt_level_strict_acc.unsqueeze(-1)

        if inst_level_strict_acc is None:
            self.inst_level_strict_acc = torch.zeros(self.batch_size + (1,))
        elif inst_level_strict_acc.ndim == self.ndim:
            self.inst_level_strict_acc = inst_level_strict_acc.unsqueeze(-1)


def _process_results(
    data: TensorDict, response: str | NonTensorData, verbose: bool = False
) -> IFEvalScoreData:
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

    if verbose:
        torchrl_logger.info(f"Processing {inp=} {response=}")
    out_strict = _test_instruction_following_strict(inp, response)
    out_loose = _test_instruction_following_loose(inp, response)

    result = IFEvalScoreData(
        prompt_level_strict_acc=out_strict.follow_all_instructions,
        inst_level_strict_acc=out_strict.follow_instruction_list,
        prompt_level_loose_acc=out_loose.follow_all_instructions,
        inst_level_loose_acc=out_loose.follow_instruction_list,
        batch_size=data.batch_size,
        device=data.device,
    )

    if verbose:
        torchrl_logger.info(f"Result: {result.to_dict()=}")
    return result


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
            it must take as input an :class:`~torchrl.envs.llm.IFEvalScoreData` instance, and optionally `think_blocks`, `answer_blocks` and `complete` keyword arguments
            containing the list of think and answer blocks, respectively.
            It must return a tensor with shape identical to the env batch-size with an additional trailing singleton dimension.
            Defaults to `True`. The default aggregator is a simple sum over the fields of :class:`~torchrl.envs.llm.IFEvalScoreData`.
        format_weights (list[float], optional): The weights for the format fields (`prompt_level_strict_acc`, `inst_level_strict_acc`,
            `prompt_level_loose_acc`, `inst_level_loose_acc`, in that order). Defaults to `[0.4, 0.3, 0.2, 0.1]`.
            This is only used if `aggregate_reward` is `True` and the default aggregator is used.
        verbose (bool, optional): Whether to print verbose information. Defaults to `False`.

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
        aggregate_reward: bool
        | Callable[
            [IFEvalScoreData, list[str] | None, list[str] | None], torch.Tensor
        ] = True,
        format_weights: list[float] | None = None,
        verbose: bool = False,
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
        self.format_weights = (
            format_weights if format_weights is not None else [0.4, 0.3, 0.2, 0.1]
        )
        self.verbose = verbose

    def default_reward_aggregator(
        self,
        score: IFEvalScoreData,
        think_blocks: list[str] | None = None,
        answer_blocks: list[str] | None = None,
        complete: bool | torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Default reward aggregation function that provides a more nuanced scoring system.

        Args:
            score (IFEvalScoreData): The score data.
            think_blocks (list[str], optional): The list of think blocks.
            answer_blocks (list[str], optional): The list of answer blocks.
            complete (bool, optional): Whether the response is complete (ends with a eos token).

        The reward is composed of three main components:
        1. Format score (max 1.0):
           - prompt_level_strict_acc: 0.4 (highest weight for strict adherence to all instructions)
           - inst_level_strict_acc: 0.3 (high weight for strict adherence to individual instructions)
           - prompt_level_loose_acc: 0.2 (medium weight for loose adherence to all instructions)
           - inst_level_loose_acc: 0.1 (lowest weight for loose adherence to individual instructions)
           All instruction-level metrics are averaged to ensure balanced contribution.

        2. Structure score (max 1.0):
           - think_block: 0.5 (presence of exactly one think block)
           - answer_block: 0.5 (presence of exactly one answer block)

        3. Completion bonus (max 0.2):
           - complete: 0.2 (response ends with eos token)

        The overall formula for the reward is:

          .. math::

            reward = format\_score + structure\_score + completion\_bonus

        Therefore, the maximum value the reward can take is 2.2, with:
        - 1.0 from format adherence
        - 1.0 from structural elements (think/answer blocks)
        - 0.2 from completion bonus
        """
        default_dtype = torch.get_default_dtype()
        score = score.to(default_dtype)

        # Format score calculation - using mean for instruction-level metrics
        format_components = torch.stack(
            [
                score.prompt_level_strict_acc.sum(-1, keepdim=True),  # Single value
                score.inst_level_strict_acc.mean(
                    -1, keepdim=True
                ),  # Average across instructions
                score.prompt_level_loose_acc.sum(-1, keepdim=True),  # Single value
                score.inst_level_loose_acc.mean(
                    -1, keepdim=True
                ),  # Average across instructions
            ],
            -1,
        )
        weights = torch.tensor(
            self.format_weights,
            device=format_components.device,
            dtype=torch.get_default_dtype(),
        )
        format_score = (format_components * weights).sum(dim=-1, keepdim=True)

        # Structure score calculation
        if think_blocks is not None:
            think_score = float(len(think_blocks) == 1) * 0.5
        else:
            think_score = 0.0

        if answer_blocks is not None:
            answer_score = float(len(answer_blocks) == 1) * 0.5
        else:
            answer_score = 0.0

        structure_score = think_score + answer_score

        # Completion bonus
        if complete is None:
            completion_bonus = 0.0
        elif isinstance(complete, torch.Tensor):
            completion_bonus = complete.to(default_dtype) * 0.2
        else:
            completion_bonus = float(complete) * 0.2

        # Combine all components
        final_reward = format_score + structure_score + completion_bonus
        final_reward = final_reward.to(default_dtype)

        return final_reward

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
        h = next_tensordict["history"][..., -1]
        response = h.content
        complete = h.is_complete
        # response = tensordict.get(self.response_key)
        if is_non_tensor(response):
            response = response.data

        # TODO: This should be a distinct module
        # Regular expression patterns to match think and answer blocks
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        # Extract think block
        think_blocks = re.findall(think_pattern, response, re.DOTALL)

        # Extract answer block
        answer_blocks = re.findall(answer_pattern, response, re.DOTALL)

        score = _process_results(
            tensordict.copy().auto_device_(),
            answer_blocks[0] if answer_blocks else "",
            verbose=self.verbose,
        )
        next_tensordict.set(
            self.score_key,
            score,
        )
        if self.aggregate_reward:
            if callable(self.aggregate_reward):
                reward_func = self.aggregate_reward
            else:
                reward_func = self.default_reward_aggregator
            reward = reward_func(
                score,
                think_blocks=think_blocks,
                answer_blocks=answer_blocks,
                complete=complete,
            )
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

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        reward_spec["reward"] = Unbounded(
            reward_spec.shape + (1,), dtype=torch.get_default_dtype()
        )
        return reward_spec

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
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
