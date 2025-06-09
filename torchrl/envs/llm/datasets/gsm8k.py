# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from typing import Any, Callable

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict
from torch.utils.data import DataLoader
from torchrl.data import TensorSpec
from torchrl.envs import StepCounter, Transform

from torchrl.envs.llm.chat import DatasetChatEnv

from torchrl.envs.llm.envs import LLMEnv
from torchrl.envs.llm.reward.gsm8k import GSM8KRewardParser

BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)


class GSM8KPrepareQuestion(Transform):
    """A transform to prepare the prompt when using GSM8k within an LLMEnv."""

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


def _collate_fn(batch):
    batch = torch.stack([TensorDict.from_dict(_batch) for _batch in batch])
    batch.rename_key_("question", "text")
    return batch


def make_gsm8k_env(
    dataset: str = "gsm8k",
    num_envs: int = 1,
    repeats: int | None = None,
    batch_size_dl: int = 1,
    seed: int | None = None,
    group_repeats: bool = False,
    tokenizer: transformers.PretrainedTokenizer | None = None,  # noqa
):
    """A builder for an LLMEnv-based GSM8K environment.

    .. note:: Prefer `torchrl.envs.llm.GSM8KEnv` to interact with this dataset.

    """
    warnings.warn("This constructor is to be deprecated. Use GSM8KEnv instead.")
    from datasets import load_dataset

    dataset = load_dataset(dataset, "main")
    train_dataset = dataset["train"]

    # Env
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(seed)

    dataloader = DataLoader(  # noqa: TOR401
        train_dataset,
        batch_size=batch_size_dl,
        shuffle=True,
        collate_fn=_collate_fn,
        generator=generator,
    )
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        # tokenizer=tokenizer,
        from_text=True,
        batch_size=(num_envs,),
        repeats=repeats,
        group_repeats=group_repeats,
        # assign_reward=True,
    )
    env.insert_transform(0, GSM8KPrepareQuestion())

    # Finally, we want the env to stop after the first step
    env.append_transform(StepCounter(max_steps=1))

    if tokenizer is not None:
        env.append_transform(GSM8KRewardParser(tokenizer=tokenizer))
    else:
        warnings.warn("No tokenizer specified - reward will not be assigned.")

    return env


class GSM8KEnv(DatasetChatEnv):
    r"""GSM8K dataset environment.

    Keyword Args:
        dataset (str, optional): The name of the dataset. Defaults to `"gsm8k"`.
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

    Examples:
        >>> import transformers
        >>> from torchrl.envs.llm.datasets.gsm8k import GSM8KEnv
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> env = GSM8KEnv(tokenizer=tokenizer, apply_template=True)
        >>> r = env.reset()
        >>> assert "history" in r
        >>> # We have an instruction step (role="system") and a question (role="user")
        >>> assert r["history"].shape == (1, 2)
        >>> assert "text" in r
        >>> r = r.clone()
        >>> print(r)
        LazyStackedTensorDict(
            fields={
                answer: NonTensorStack(
                    ['Adam bought 3 sandwiches, so he paid 3 * 3 = $<<...,
                    batch_size=torch.Size([1]),
                    device=None),
                done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                history: History(
                    content=NonTensorStack(
                        [['A conversation between User and Assistant. The ...,
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    role=NonTensorStack(
                        [['system', 'user']],
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    batch_size=torch.Size([1, 2]),
                    device=None,
                    is_shared=False),
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
        >>> response = "<think>First, calculate the total number of snakes in the breeding balls. There are 3 breeding balls with 8 snakes each, so 3 * 8 = 24 snakes. Next, calculate the number of snakes in the additional pairs. There are 6 pairs of snakes, and each pair has 2 snakes, so 6 * 2 = 12 snakes. Finally, add the number of snakes from the breeding balls and the additional pairs: 24 + 12 = 36 snakes.</think> <answer>Mary saw a total of 36 snakes.</answer><|im_end|>"
        >>> r["text_response"] = [response]
        >>> s = env.step(r)
        >>> print(s)
        LazyStackedTensorDict(
            fields={
                answer: NonTensorStack(
                    ['Adam bought 3 sandwiches, so he paid 3 * 3 = $<<...,
                    batch_size=torch.Size([1]),
                    device=None),
                done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                history: History(
                    content=NonTensorStack(
                        [['A conversation between User and Assistant. The ...,
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    role=NonTensorStack(
                        [['system', 'user']],
                        batch_size=torch.Size([1, 2]),
                        device=None),
                    batch_size=torch.Size([1, 2]),
                    device=None,
                    is_shared=False),
                next: LazyStackedTensorDict(
                    fields={
                        answer: NonTensorStack(
                            ['Adam bought 3 sandwiches, so he paid 3 * 3 = $<<...,
                            batch_size=torch.Size([1]),
                            device=None),
                        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        history: History(
                            content=NonTensorStack(
                                [['A conversation between User and Assistant. The ...,
                                batch_size=torch.Size([1, 3]),
                                device=None),
                            role=NonTensorStack(
                                [['system', 'user', 'assistant']],
                                batch_size=torch.Size([1, 3]),
                                device=None),
                            batch_size=torch.Size([1, 3]),
                            device=None,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_answer: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_contained: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_right: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_think: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        success: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
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
                    stack_dim=0),
                step_count: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                text: NonTensorStack(
                    ['<|im_start|>system\nA conversation between User ...,
                    batch_size=torch.Size([1]),
                    device=None),
                text_response: NonTensorStack(
                    ['<think>First, calculate the total number of snak...,
                    batch_size=torch.Size([1]),
                    device=None),
                truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> assert s["next", "reward"] >= 10
        >>> assert s["next", "done"].all()

    """

    SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively,
i.e., <think>reasoning process here</think> <answer>answer here</answer>."""

    def __init__(
        self,
        *,
        dataset: str = "gsm8k",
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
    ):
        if collate_fn is None:
            collate_fn = _collate_fn
        super().__init__(
            dataset=dataset,
            name="main",
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
        )
        if max_steps:
            self.append_transform(StepCounter(max_steps=max_steps))
        if compute_reward:
            self.append_transform(GSM8KRewardParser(tokenizer=tokenizer))
