# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset
from torchrl.envs import StepCounter
from torchrl.envs.llm.chat import DatasetChatEnv
from torchrl.envs.llm.reward.countdown import CountdownRewardParser

if TYPE_CHECKING:
    import transformers


class _CountdownProblemGenerator(IterableDataset):
    """Infinite procedural generator for Countdown problems.

    Each problem picks ``num_count`` numbers from [1, max_number] and
    generates a target that is reachable from those numbers using
    ``+``, ``-``, ``*``, ``/``.
    """

    def __init__(
        self,
        num_count: int = 4,
        max_number: int = 100,
        max_target: int = 1000,
        seed: int | None = None,
    ):
        self.num_count = num_count
        self.max_number = max_number
        self.max_target = max_target
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        numbers = [self.rng.randint(1, self.max_number) for _ in range(self.num_count)]
        target = self._make_target(numbers)
        query = (
            f"Using the numbers {numbers}, create an arithmetic expression that "
            f"equals {target}. You may use each number at most once. "
            f"Only use +, -, *, / and parentheses."
        )
        answer = f"target={target}, numbers={','.join(str(n) for n in numbers)}"
        return {"query": query, "answer": answer}

    def _make_target(self, numbers: list[int]) -> int:
        """Generate a reachable target by randomly combining numbers."""
        ops = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
        ]
        pool = list(numbers)
        self.rng.shuffle(pool)
        result = pool[0]
        for n in pool[1:]:
            op = self.rng.choice(ops)
            result = op(result, n)
        result = abs(result)
        if result == 0:
            result = sum(numbers)
        if result > self.max_target:
            result = sum(numbers)
        return result


def _collate_fn(batch):
    return torch.stack([TensorDict.from_dict(b) for b in batch])


class CountdownEnv(DatasetChatEnv):
    """Countdown numbers-game environment for LLM post-training.

    Given a set of source numbers and a target, the model must construct an
    arithmetic expression that evaluates to the target using each source number
    at most once.

    Problems are generated procedurally (no external dataset required), making
    this environment ideal for quick experimentation and debugging of RL
    training loops.

    Keyword Args:
        num_count (int): How many source numbers per problem. Defaults to ``4``.
        max_number (int): Maximum value for each source number. Defaults to ``100``.
        max_target (int): Ceiling for the generated target. Defaults to ``1000``.
        shuffle (bool): Ignored (procedural generation is always random).
        num_envs (int): Number of parallel environments. Defaults to ``1``.
        repeats (int | None): Repeats per sample for MC estimation.
        batch_size_dl (int): Dataloader batch size. Defaults to ``1``.
        seed (int | None): Random seed for reproducibility.
        group_repeats (bool): Group repeated samples. Defaults to ``False``.
        tokenizer: Tokenizer for text processing.
        device: Device for computation.
        template_kwargs: Extra kwargs for ``apply_chat_template``.
        apply_template (bool): Apply chat template. Defaults to ``False``.
        compute_reward (bool): Compute rewards. Defaults to ``True``.
        collate_fn: Custom collate function.
        max_steps (int): Max steps per episode. Defaults to ``1``.
        input_mode: ``"history"``, ``"text"``, or ``"tokens"``.

    Examples:
        >>> import transformers
        >>> from torchrl.envs.llm.datasets.countdown import CountdownEnv
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = CountdownEnv(tokenizer=tokenizer, apply_template=True, seed=42)
        >>> r = env.reset()
        >>> assert "history" in r

    """

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user gives a set of "
        "numbers and a target. The Assistant must find an arithmetic expression "
        "using each given number at most once that equals the target.\n"
        "The reasoning process and answer are enclosed within <think></think> "
        "and <answer></answer> tags, respectively.\n"
        "The answer should contain ONLY the arithmetic expression (e.g. "
        "(25 + 3) * 4)."
    )

    def __init__(
        self,
        *,
        num_count: int = 4,
        max_number: int = 100,
        max_target: int = 1000,
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
    ):
        if collate_fn is None:
            collate_fn = _collate_fn

        self._num_count = num_count
        self._max_number = max_number
        self._max_target = max_target
        self._seed = seed

        batch_size = (num_envs,)
        dataloader = DataLoader(  # noqa: TOR401
            _CountdownProblemGenerator(
                num_count=num_count,
                max_number=max_number,
                max_target=max_target,
                seed=seed,
            ),
            batch_size=batch_size_dl,
            collate_fn=collate_fn,
        )

        self._from_dataloader(
            self,
            dataloader=dataloader,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            input_mode=input_mode,
        )

        if max_steps:
            self.append_transform(StepCounter(max_steps=max_steps))
        if compute_reward:
            self.append_transform(CountdownRewardParser(tokenizer=tokenizer))
