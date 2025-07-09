# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, Callable, Literal, TYPE_CHECKING

import torch
from tensordict import NonTensorData, NonTensorStack, TensorClass, TensorDict
from torchrl.data import Composite, NonTensor, Unbounded
from torchrl.envs import StepCounter
from torchrl.envs.llm.chat import DatasetChatEnv
from torchrl.envs.llm.reward.ifeval import IfEvalScorer

if TYPE_CHECKING:
    import transformers


class IFEvalData(TensorClass["nocast"]):
    """A tensorclass for IFEval dta."""

    key: torch.Tensor
    instruction_id_list: list[str]
    kwargs: list[dict]
    query: str

    # Reponses and additional fields
    response: str | None = None
    tokens: torch.Tensor | None = None
    tokens_response: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    reward: torch.Tensor | None = None

    @classmethod
    def default_spec(
        cls, shape: torch.Size, device: torch.device | None = None
    ) -> Composite:
        return Composite(
            key=Unbounded(shape=shape, dtype=torch.int64, device=device),
            instruction_id_list=NonTensor(
                shape=shape,
                device=device,
                feature_dims=0,
                example_data=["punctuation:no_comma"],
            ),
            kwargs=NonTensor(
                shape=shape,
                device=device,
                feature_dims=0,
                example_data={
                    "num_highlights": None,
                    "relation": None,
                    "num_placeholders": None,
                },
            ),
            query=NonTensor(
                shape=shape,
                device=device,
                example_data="Plan a 2 week Europe trip and visit London, Paris, and Rome. Answer in all caps. The response must contain at least 8 placeholders (i.e., [restaurant]).",
            ),
            shape=shape,
            step_mdp_static=True,
            data_cls=cls,
        )


def _collate_fn(batch):
    batch = torch.stack([TensorDict.from_any(_batch) for _batch in batch])
    batch.rename_key_("prompt", "query")
    # we want instruction_id_list and kwargs to be lists, but not NonTensorStacks
    instruction_id_list = batch["instruction_id_list"]
    # instruction_id_list should be a list of lists
    instruction_id_list = NonTensorStack(
        *[
            NonTensorData([item] if not isinstance(item, list) else item)
            for item in instruction_id_list
        ]
    )
    kwargs = batch["kwargs"]
    kwargs = NonTensorStack(
        *[
            NonTensorData([item] if not isinstance(item, list) else item)
            for item in kwargs
        ]
    )
    batch.set("instruction_id_list", instruction_id_list)
    batch.set("kwargs", kwargs)
    # we don't need a tensorclass here
    return batch
    # return IFEvalData.from_tensordict(batch)


class IFEvalEnv(DatasetChatEnv):
    r"""A chat environment based on the IFEval dataset.

    Keyword Args:
        dataset (str, optional): The name of the dataset. Defaults to `"google/IFeval"`.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to `True`.
        num_envs (int, optional): The number of environments to create. Defaults to `1`.
        repeats (int | None, optional): The number of times to repeat each sample from the dataset (mainly for Monte-Carlo
            based value estimation). If `None`, the dataset is not repeated. Defaults to `None`.
        batch_size_dl (int, optional): The batch size for data loading. Defaults to `1`.
        seed (int | None, optional): The random seed for reproducibility. If `None`, a random seed is used. Defaults to `None`.
        group_repeats (bool, optional): Whether to group repeated samples together. Defaults to `False`.
        tokenizer (transformers.AutoTokenizer | None, optional): The tokenizer to use for text processing. Defaults to `None`.

            .. note:: It is recommended to pass a tokenizer to the environment. This is an easy way to ensure that the
                template applied to the chat history is consistent with the format required by the model.

        device (torch.device | None, optional): The device to use for computations. Defaults to None.
        template_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the template. Defaults to `None`.
        apply_template (bool | None, optional): Whether to apply the template to the text. Defaults to `False`.
        compute_reward (bool, optional): Whether to compute rewards. Defaults to `True`.
        collate_fn (Callable | None, optional): A custom collate function for data loading. If `None`, a default
            collate function is used. Defaults to `None`.
        max_steps (int, optional): The maximum number of steps allowed in an episode. Defaults to `1`.
        input_mode (Literal["history", "text", "tokens"], optional): The mode of input to use. Defaults to `"history"`.

    Examples:
        >>> import transformers
        >>> from pprint import pprint
        >>> from torchrl.envs.llm.datasets import IFEvalEnv
        >>> from tensordict import set_list_to_stack
        >>> set_list_to_stack(True).set()
        >>>
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = IFEvalEnv(tokenizer=tokenizer, apply_template=True)
        >>> r = env.reset()
        >>> print(r)
        LazyStackedTensorDict(
            fields={
                done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                history: History(
                    content=NonTensorStack(
                        [['A conversation between User and Assistant.\nYou...,
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    role=NonTensorStack(
                        [['system', 'user']],
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    batch_size=torch.Size([1, 2]),
                    device=None,
                    is_shared=False),
                instruction_id_list: NonTensorStack(
                    [['detectable_content:number_placeholders']],
                    batch_size=torch.Size([1, 1]),
                    device=None),
                key: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False),
                kwargs: NonTensorStack(
                    [[{'num_highlights': None, 'relation': None, 'num_...,
                    batch_size=torch.Size([1, 1]),
                    device=None),
                step_count: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                text: NonTensorStack(
                    ['<|im_start|>system\nA conversation between User ...,
                    batch_size=torch.Size([1]),
                    device=None),
                truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> # Print content of conversation so far
        >>> pprint(r["history", "content"])
        [['A conversation between User and Assistant.\n'
          'You are tasked with responding to user queries in a very specific format. \n'
          'When given a task or question, first think through the problem and provide '
          'your thought process between <think> and </think> tags.\n'
          'Then, give your final answer or response between <answer> and </answer> '
          'tags.\n'
          'You will be assessed by the content of the answer block only, so make sure '
          'it contains all the required information, and only that.',
          'Plan a 2 week Europe trip and visit London, Paris, and Rome. Answer in all '
          'caps. The response must contain at least 8 placeholders (i.e., '
          '[restaurant]).']]
        >>> # Actions space: the environment expects an action with key "text_response" containing a (list of) strings
        >>> print(env.action_spec)
        Composite(
            text_response: NonTensor(
                shape=torch.Size([1]),
                space=None,
                device=None,
                dtype=None,
                domain=None,
                example_data=a string),
            device=None,
            shape=torch.Size([1]))

    """

    SYSTEM_PROMPT = """You are a helpful AI assistant that follows instructions extremely well.

IMPORTANT: You must respond in a specific format for every task:

1. First, think through the problem step by step and write your reasoning between <think> and </think> tags
2. Then, provide your final answer between <answer> and </answer> tags

CRITICAL RULES:
- ALWAYS use <think>...</think> and <answer>...</answer> tags exactly as shown
- Do NOT use <thought>, <reasoning>, or any other tag variations
- Your <answer> section will be evaluated, so make it complete and accurate
- Follow ALL specific requirements in the user's request (formatting, content, etc.)
- If the user asks for placeholders like [restaurant], include them exactly as requested
- Pay attention to capitalization, punctuation, and other formatting requirements

Example format:
<think>
I need to analyze what the user is asking for...
[Your reasoning here]
</think>
<answer>
[Your final answer here, following all user requirements]
</answer>"""

    def __init__(
        self,
        *,
        dataset: str = "google/IFeval",
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
            data_key="query",
            primers=IFEvalData.default_spec((num_envs,), device),
        )
        if max_steps:
            self.append_transform(StepCounter(max_steps=max_steps))
        if compute_reward:
            self.append_transform(IfEvalScorer())
