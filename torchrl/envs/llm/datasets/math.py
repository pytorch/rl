# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import TensorDict
from torchrl.envs import StepCounter
from torchrl.envs.llm.chat import DatasetChatEnv
from torchrl.envs.llm.reward.math import MATHRewardParser

if TYPE_CHECKING:
    import transformers


def _collate_fn(batch):
    batch = torch.stack([TensorDict.from_dict(_batch) for _batch in batch])
    batch.rename_key_("problem", "query")
    batch.rename_key_("solution", "answer")
    return batch


class MATHEnv(DatasetChatEnv):
    r"""MATH (competition mathematics) dataset environment.

    Uses the ``DigitalLearningGmbH/MATH-lighteval`` dataset on Hugging Face
    (a drop-in replacement for the original ``hendrycks/competition_math``).

    Answers are in LaTeX ``\boxed{}`` format.  When ``math-verify`` is
    installed the reward parser uses symbolic equivalence checking; otherwise
    it falls back to normalised string comparison.

    Keyword Args:
        dataset (str, optional): HuggingFace dataset name.
            Defaults to ``"DigitalLearningGmbH/MATH-lighteval"``.
        shuffle (bool, optional): Shuffle the dataset. Defaults to ``True``.
        num_envs (int, optional): Number of parallel envs. Defaults to ``1``.
        repeats (int | None, optional): Repeats per sample for MC estimation.
        batch_size_dl (int, optional): Dataloader batch size. Defaults to ``1``.
        seed (int | None, optional): Random seed.
        group_repeats (bool, optional): Group repeated samples. Defaults to ``False``.
        tokenizer: Tokenizer for text processing.
        device: Device for computation.
        template_kwargs: Extra kwargs for ``apply_chat_template``.
        apply_template (bool): Apply chat template. Defaults to ``False``.
        compute_reward (bool): Compute rewards. Defaults to ``True``.
        collate_fn: Custom collate function.
        max_steps (int): Max steps per episode. Defaults to ``1``.
        input_mode: ``"history"``, ``"text"``, or ``"tokens"``.
        ray_backend (bool): Use Ray backend for data loading.
        dataloader_actor_name (str): Ray actor name for data loading.

    Examples:
        >>> import transformers
        >>> from torchrl.envs.llm.datasets.math import MATHEnv
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = MATHEnv(tokenizer=tokenizer, apply_template=True)
        >>> r = env.reset()
        >>> assert "history" in r

    """

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a math problem, "
        "and the Assistant solves it.\n"
        "The assistant first thinks about the reasoning process in the mind and "
        "then provides the user with the answer.\n"
        "The reasoning process and answer are enclosed within <think></think> and "
        "<answer></answer> tags, respectively, i.e.,\n"
        "<think>reasoning process here</think> <answer>answer here</answer>.\n"
        "The answer should be a mathematical expression (use LaTeX if needed)."
    )

    def __init__(
        self,
        *,
        dataset: str = "DigitalLearningGmbH/MATH-lighteval",
        shuffle: bool = True,
        num_envs: int = 1,
        repeats: int | None = None,
        batch_size_dl: int = 1,
        seed: int | None = None,
        group_repeats: bool = False,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa
        device: torch.device | None = None,
        template_kwargs: dict[str, Any] | None = None,
        apply_template: bool | None = False,
        compute_reward: bool = True,
        collate_fn: Callable | None = None,
        max_steps: int = 1,
        input_mode: Literal["history", "text", "tokens"] = "history",
        ray_backend: bool = False,
        dataloader_actor_name: str | None = None,
    ):
        if ray_backend and dataloader_actor_name is None:
            dataloader_actor_name = "math_dataloader"
        if collate_fn is None:
            collate_fn = _collate_fn
        super().__init__(
            dataset=dataset,
            shuffle=shuffle,
            num_envs=num_envs,
            repeats=repeats,
            batch_size_dl=batch_size_dl,
            seed=seed,
            group_repeats=group_repeats,
            tokenizer=tokenizer,
            device=device,
            template_kwargs=template_kwargs,
            apply_template=apply_template,
            collate_fn=collate_fn,
            input_mode=input_mode,
            ray_backend=ray_backend,
            dataloader_actor_name=dataloader_actor_name,
        )
        if max_steps:
            self.append_transform(StepCounter(max_steps=max_steps))
        if compute_reward:
            self.append_transform(MATHRewardParser(tokenizer=tokenizer))
