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
from tensordict import (
    lazy_stack,
    NestedKey,
    NonTensorData,
    TensorClass,
    TensorDict,
    TensorDictBase,
)
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

    @classmethod
    def default_spec(
        cls, shape: torch.Size, device: torch.device | None = None
    ) -> Composite:
        return Composite(
            prompt_level_strict_acc=Unbounded(
                shape=shape, dtype=torch.bool, device=device
            ),
            inst_level_strict_acc=Unbounded(
                shape=shape, dtype=torch.bool, device=device
            ),
            prompt_level_loose_acc=Unbounded(
                shape=shape, dtype=torch.bool, device=device
            ),
            inst_level_loose_acc=Unbounded(
                shape=shape, dtype=torch.bool, device=device
            ),
            data_cls=cls,
            step_mdp_static=True,
        )

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
    data: TensorDict,
    response: str | NonTensorData,
    verbose: bool = False,
    prompt: str | None = None,
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

    if prompt is None:
        prompt = data["text"]

    inp = _InputExample(
        key=data["key"],
        instruction_id_list=data["instruction_id_list"],
        prompt=prompt if prompt is not None else "",
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
        set_done_if_answer (bool): whether to set the done flag to `True` when an answer is present. Defaults to `True`.

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
        set_done_if_answer: bool = True,
    ):
        self.aggregate_reward = aggregate_reward
        self.score_key = score_key
        self.set_done_if_answer = set_done_if_answer
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
        r"""Improved reward aggregation function with tiered multiplicative scoring.

        Args:
            score (IFEvalScoreData): The score data.
            think_blocks (list[str], optional): The list of think blocks.
            answer_blocks (list[str], optional): The list of answer blocks.
            complete (bool, optional): Whether the response is complete (ends with a eos token).

        The reward uses a tiered multiplicative system:

        1. Critical failure check: No answer blocks = 0 reward
        2. Base format score (0-1): Weighted average of format metrics
        3. Structure multiplier (0.1-1.0): Penalties for missing/multiple blocks
        4. Quality bonus (0-0.5): Rewards for high quality and completion
        5. Task complexity scaling: More requirements = higher potential rewards

        The final formula is:
            reward = (format_score + quality_bonus) * structure_multiplier * complexity_scale

        This provides better learning signals by:
        - Requiring critical elements (answer tags) for meaningful rewards
        - Using multiplicative scaling to reward doing everything well
        - Scaling rewards based on task complexity
        - Providing clear failure modes and success incentives

        Reward range: 0.0 to ~1.5-2.7 depending on task complexity (more instructions = higher max reward).
        """
        default_dtype = torch.get_default_dtype()
        score = score.to(default_dtype)

        # Critical failure check - no answer = no reward
        if not answer_blocks:
            return torch.zeros(
                score.batch_size + (1,), device=score.device, dtype=default_dtype
            )

        # Base format score calculation (0-1)
        format_components = torch.stack(
            [
                score.prompt_level_strict_acc.sum(-1, keepdim=True)
                if score.prompt_level_strict_acc is not None
                else torch.zeros(
                    score.batch_size + (1,), device=score.device, dtype=default_dtype
                ),  # Single value
                score.inst_level_strict_acc.mean(-1, keepdim=True)
                if score.inst_level_strict_acc is not None
                else torch.zeros(
                    score.batch_size + (1,), device=score.device, dtype=default_dtype
                ),  # Average across instructions
                score.prompt_level_loose_acc.sum(-1, keepdim=True)
                if score.prompt_level_loose_acc is not None
                else torch.zeros(
                    score.batch_size + (1,), device=score.device, dtype=default_dtype
                ),  # Single value
                score.inst_level_loose_acc.mean(-1, keepdim=True)
                if score.inst_level_loose_acc is not None
                else torch.zeros(
                    score.batch_size + (1,), device=score.device, dtype=default_dtype
                ),  # Average across instructions
            ],
            -1,
        )
        weights = torch.tensor(
            self.format_weights,
            device=format_components.device,
            dtype=default_dtype,
        )
        format_score = (format_components * weights).sum(dim=-1, keepdim=True)

        # Structure multiplier (0.1-1.0)
        structure_multiplier = 1.0

        # Heavy penalty for missing think blocks (but not zero)
        if not think_blocks:
            structure_multiplier *= 0.3
        elif len(think_blocks) > 1:
            structure_multiplier *= 0.7  # Penalty for multiple think blocks

        # Penalty for multiple answer blocks
        if len(answer_blocks) > 1:
            structure_multiplier *= 0.7

        # Quality bonus (0-0.5)
        quality_bonus = torch.zeros_like(format_score)

        # Bonus for high quality responses
        if format_score > 0.8:
            quality_bonus += 0.3

        # Completion bonus
        if complete is not None:
            if isinstance(complete, torch.Tensor):
                completion_bonus = complete.to(default_dtype) * 0.2
            else:
                completion_bonus = float(complete) * 0.2
            quality_bonus += completion_bonus

        # Task complexity scaling based on number of instructions
        # More instructions = higher potential rewards
        if (
            score.inst_level_strict_acc is not None
            and score.inst_level_strict_acc.numel() > 0
        ):
            num_instructions = score.inst_level_strict_acc.shape[-1]
        else:
            num_instructions = 1
        complexity_scale = (
            1.0 + (num_instructions - 1) * 0.2
        )  # 1.0 for 1 instruction, 1.2 for 2, etc.

        # Final reward: (format + quality) * structure_multiplier * complexity_scale
        final_reward = (
            (format_score + quality_bonus) * structure_multiplier * complexity_scale
        )
        final_reward = final_reward.to(default_dtype)

        return final_reward

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not getattr(self.parent.base_env, "input_mode", "history") == "history":
            raise ValueError("IFEvalScorer only supports history input mode")

        if tensordict.ndim:
            return lazy_stack(
                [
                    self._step(td, next_td)
                    for td, next_td in zip(
                        tensordict.unbind(0), next_tensordict.unbind(0)
                    )
                ]
            )
        h = tensordict["history", "full"][..., -1]
        prompt = tensordict["history", "prompt"][..., -1].content
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
            prompt=prompt,
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
            reward = reward.view(
                next_tensordict.batch_size
                + (
                    1,
                    1,
                )
            )
            next_tensordict.set("reward", reward)
        if self.set_done_if_answer and bool(answer_blocks):
            next_tensordict.set(
                "done",
                torch.ones(
                    next_tensordict.batch_size + (1,),
                    device=next_tensordict.device,
                    dtype=torch.bool,
                ),
            )
            next_tensordict.set(
                "terminated",
                torch.ones(
                    next_tensordict.batch_size + (1,),
                    device=next_tensordict.device,
                    dtype=torch.bool,
                ),
            )
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
            reward_spec.shape + (1, 1),
            dtype=torch.get_default_dtype(),
            device=reward_spec.device,
        )
        return reward_spec

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec[self.score_key] = IFEvalScoreData.default_spec(
            observation_spec.shape, device=observation_spec.device
        )
        return observation_spec
