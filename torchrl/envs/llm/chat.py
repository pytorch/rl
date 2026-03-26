# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import lazy_stack, TensorDictBase
from tensordict.utils import _zip_strict
from torch.utils.data import DataLoader
from torchrl.data import Composite, NonTensor
from torchrl.data.llm.history import History
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.common import _EnvPostInit
from torchrl.envs.llm.transforms.dataloading import (
    DataLoadingPrimer,
    RayDataLoadingPrimer,
)
from torchrl.modules.llm.policies.common import ChatHistory, Text, Tokens

if TYPE_CHECKING:
    import transformers


class _ChatEnvMeta(_EnvPostInit):
    """Metaclass for ChatEnv that handles with_tokenizer wrapping."""

    def __call__(cls, *args, with_tokenizer: bool = False, **kwargs):
        # Create the instance using parent metaclass logic
        instance = super().__call__(*args, **kwargs)

        # Wrap with IncrementalTokenizer if requested
        if with_tokenizer:
            tokenizer = kwargs.get("tokenizer")
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when with_tokenizer=True")
            from torchrl.envs.llm.transforms import IncrementalTokenizer

            return TransformedEnv(instance, IncrementalTokenizer(tokenizer))

        return instance


def _default_collate_fn(batch):
    # We want to rename the "text" key to "query"
    #  otherwise it will conflict with the "text" key in the tensordict returned by TorchRL components
    if isinstance(batch, dict) and "text" in batch:
        batch["query"] = batch.pop("text")
    elif isinstance(batch, list):
        for item in batch:
            if "text" in item:
                item["query"] = item.pop("text")
    return batch


class ChatEnv(EnvBase, metaclass=_ChatEnvMeta):
    r"""A chat-based environment for LLMs, designed as a blank canvas for conversation and RL.

    This environment is designed to work seamlessly with both :class:`~torchrl.modules.llm.policies.TransformersWrapper` and
    :class:`~torchrl.modules.llm.policies.vLLMWrapper`. It provides the fundamental structure for managing conversation state
    using the :class:`~torchrl.data.llm.History` format (or, alternatively, tokens or text), but is intentionally minimal to allow
    maximum flexibility through transforms.

    Core Functionality
        The environment operates in three main modes:

            - **History mode**: Uses :class:`~torchrl.data.llm.History` objects for conversation management
            - **Text mode**: Uses simple text strings for input/output
            - **Tokens mode**: Uses tokenized data for input/output

    Reset Operation
        During reset, the environment:

            1. Takes input text from the `data_key` (default: `"query"`) in the tensordict
            2. Creates a :class:`~torchrl.data.llm.History` object with the user's message
            3. Optionally prepends a system prompt if provided
            4. Formats the conversation according to the selected input mode (history, text, or tokens)
            5. Returns the formatted prompt ready for the LLM

    Step Operation
        During step, the environment:

            1. Takes the LLM's response (containing both prompt and generated text)
            2. Extracts the full conversation history
            3. Prepares the next prompt by setting the full history as the new prompt
            4. Returns the updated conversation state

    This design enables natural multi-turn conversations where each step extends the conversation
    history, making it ideal for dialogue systems and reinforcement learning applications.

    Integration with Transforms
        ChatEnv is designed to be extended with transforms that add specific capabilities:

            - **Reward computation**: :class:`~torchrl.envs.llm.transforms.KLRewardTransform` for KL divergence rewards
            - **Tool execution**: :class:`~torchrl.envs.llm.transforms.PythonInterpreter` for Python code execution
            - **Data loading**: :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer` for loading prompts from datasets
            - **Thinking prompts**: :class:`~torchrl.envs.llm.transforms.AddThinkingPrompt` for chain-of-thought reasoning
            - **Token maintenance**: :class:`~torchrl.envs.llm.transforms.IncrementalTokenizer` for token-first inference

    Keyword Args:
        input_mode (Literal["history", "text", "tokens"]): The mode of input to the environment.
            Defaults to `"history"`.
        batch_size (torch.Size): Expected batch size of the input. Defaults to `(1,)` (null batch sizes such as `()`
            are not recommended as they don't play well with generators).
        system_prompt (str, optional): An optional `"system"` prompt string to use during reset calls.
            Defaults to `None`.
        tokenizer (transformers.PreTrainedTokenizer, optional): A tokenizer that will be used to tokenize the text.
            Defaults to `None`.
        template_kwargs (dict[str, any], optional): Keyword arguments passed to :meth:`~torchrl.data.llm.History.apply_chat_template`.
            Defaults to `None`.
        system_role (str, optional): The role of the system (at reset time). Defaults to `"system"`.
        user_role (str, optional): The role of the user (at reset time). Defaults to `"user"`.
        policy_role (str, optional): The role of the policy/assistant. Defaults to `"assistant"`.
        data_key (str, optional): The key of the data input to the env at reset time (from dataloader). Defaults to `"query"`.
        device (torch.device, optional): The device to use for computations. Defaults to `None`.
        with_tokenizer (bool, optional): If ``True``, the environment is automatically wrapped with
            :class:`~torchrl.envs.llm.transforms.IncrementalTokenizer` to maintain ``tokens.prompt`` synchronized
            with ``history.prompt``. This enables token-first inference in LLM wrappers with ``prefer_tokens=True``,
            ensuring KV cache consistency across multi-turn conversations. Requires ``tokenizer`` to be provided.
            Defaults to ``False``.

    Methods:
        reset (TensorDict): Resets the state of the environment. A tensordict or equivalent with a `"query"` entry
            (originating from the dataloader) must be passed. This key name is defined as a class attribute `data_key`.
        step (TensorDict): Makes a step in the environment. A tensordict or equivalent with the LLM's response must be passed.
            The response key is defined as a class attribute `response_key`.

    .. seealso:: To see examples of a `ChatEnv` in action, see :class:`~torchrl.envs.llm.chat.DatasetChatEnv`,
        :class:`~torchrl.envs.llm.GSM8KEnv` and :class:`~torchrl.envs.llm.IFEvalEnv`.

    Examples:
        >>> from torchrl.envs.llm import ChatEnv
        >>> from torchrl.data.llm import History
        >>> from tensordict import TensorDict
        >>>
        >>> # Create a basic chat environment
        >>> env = ChatEnv(
        ...     system_prompt="You are a helpful assistant.",
        ...     input_mode="history"
        ... )
        >>>
        >>> # Reset with a user query
        >>> reset_data = TensorDict({"query": "Hello, how are you?"}, batch_size=(1,))
        >>> obs = env.reset(reset_data)
        >>> print(obs["history"].prompt)  # History with system prompt + user message
        >>>
        >>> # Simulate LLM response and step
        >>> response_data = TensorDict({
        ...     "history": History.from_chats([[
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Hello, how are you?"},
        ...         {"role": "assistant", "content": "I'm doing well, thank you!"}
        ...     ]])
        ... }, batch_size=(1,))
        >>> next_obs = env.step(response_data)
        >>> print(next_obs["history"].prompt)  # Full conversation history
        >>>
        >>> # Create environment with token maintenance for KV cache consistency
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> env = ChatEnv(
        ...     tokenizer=tokenizer,
        ...     system_prompt="You are a helpful assistant.",
        ...     with_tokenizer=True,  # Automatically wraps with IncrementalTokenizer
        ... )
        >>> # Now tokens.prompt will be available and synchronized with history.prompt

    """

    # Nested key corresponding to the text input to the LLM
    text_key = ("text", "prompt")
    # Nested key corresponding to the response from the LLM
    response_key = ("text", "response")
    # Nested key corresponding to the data input to the env at reset time (from dataloader)
    data_key = "query"

    @classmethod
    def with_tokenizer(
        cls,
        tokenizer: transformers.AutoTokenizer,  # noqa: F821
        **kwargs,
    ) -> TransformedEnv:
        """Create a ChatEnv wrapped with IncrementalTokenizer for token maintenance.

        This is a convenience method equivalent to ``ChatEnv(..., with_tokenizer=True)``.

        Args:
            tokenizer: The tokenizer to use for tokenization.
            **kwargs: Additional arguments passed to ChatEnv constructor.

        Returns:
            TransformedEnv: A ChatEnv wrapped with IncrementalTokenizer.

        Example:
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            >>> env = ChatEnv.with_tokenizer(
            ...     tokenizer=tokenizer,
            ...     batch_size=(1,),
            ...     system_prompt="You are a helpful assistant.",
            ... )
            >>> # Now tokens.prompt will be maintained automatically
        """
        return cls(tokenizer=tokenizer, with_tokenizer=True, **kwargs)

    def __init__(
        self,
        *,
        input_mode: Literal["history", "text"] = "history",
        batch_size: tuple | torch.Size | None = None,
        system_prompt: str | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        template_kwargs: dict[str, Any] | None = None,
        system_role: str = "system",
        user_role: str = "user",
        policy_role: str | None = "assistant",
        data_key: str | None = None,
        device: torch.device | None = None,
    ):
        self.input_mode = input_mode
        if batch_size is None:
            batch_size = (1,)
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        if isinstance(batch_size, list):
            batch_size = torch.Size(batch_size)
        if batch_size == ():
            raise ValueError(f"{type(self).__name__} must have at least one dimension")
        if data_key is not None:
            self.data_key = data_key
        super().__init__(batch_size=batch_size, device=device)
        self.batch_size = batch_size

        self.system_prompt = system_prompt

        if template_kwargs is None:
            template_kwargs = {}
        self.template_kwargs = template_kwargs

        self.system_role = system_role
        self.user_role = user_role
        self.policy_role = policy_role
        self.tokenizer = tokenizer

        self._make_specs()

    def _make_specs(self):
        if self.input_mode == "history":
            self._make_specs_history()
        elif self.input_mode == "text":
            self._make_specs_text()
        elif self.input_mode == "tokens":
            self._make_specs_tokens()
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}")

    def _make_specs_history(self):
        # we output prompt
        self.full_observation_spec = Composite(
            history=ChatHistory.default_spec(shape=self.batch_size, keys=["prompt"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        # We receive prompt, response and full
        self.full_action_spec = Composite(
            history=ChatHistory.default_spec(shape=self.batch_size, keys=["full"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.full_state_spec = Composite(
            {
                self.data_key: NonTensor(
                    example_data="a string", shape=self.batch_size, device=self.device
                )
            },
            shape=self.batch_size,
            device=self.device,
        )

    def _make_specs_text(self):
        # we output prompt
        self.full_observation_spec = Composite(
            text=Text.default_spec(shape=self.batch_size, keys=["prompt"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        # We receive prompt, response and full
        self.full_action_spec = Composite(
            text=Text.default_spec(shape=self.batch_size, keys=["full"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.full_state_spec = Composite(
            {
                self.data_key: NonTensor(
                    example_data="a string", shape=self.batch_size, device=self.device
                )
            },
            shape=self.batch_size,
            device=self.device,
        )

    def _make_specs_tokens(self):
        # we output prompt
        self.full_observation_spec = Composite(
            tokens=Tokens.default_spec(shape=self.batch_size, keys=["prompt"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        # We receive prompt, response and full
        self.full_action_spec = Composite(
            tokens=Tokens.default_spec(shape=self.batch_size, keys=["full"]).to(
                self.device
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.full_state_spec = Composite(
            {
                self.data_key: NonTensor(
                    example_data="a string", shape=self.batch_size, device=self.device
                )
            },
            shape=self.batch_size,
            device=self.device,
        )

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        *,
        repeats: int | None = None,
        device: torch.device | None = None,
        group_repeats: bool = False,
        batch_size: tuple | torch.Size | None = None,
        primers: Composite | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        template_kwargs: dict[str, Any] | None = None,
        input_mode: Literal["history", "text", "tokens"] = "history",
        data_key: str | None = None,
        system_prompt: str | None = None,
    ):
        """Create a chat environment from a dataloader.

        Args:
            dataloader (DataLoader): The dataloader to use.

        Keyword Args:
            repeats (int | None, optional): The number of times to repeat each sample from the dataset (mainly for Monte-Carlo
                based value estimation). If `None`, the dataset is not repeated. Defaults to `None`.
            device (torch.device | None, optional): The device to use for computations. Defaults to None.
            group_repeats (bool, optional): Whether to group repeated samples together. Defaults to `False`.
            batch_size (tuple | torch.Size | None, optional): The batch size for data loading. Defaults to `1`.
            primers (Composite | None, optional): The primers to use for data loading. Defaults to `None`.
            tokenizer (transformers.AutoTokenizer | None, optional): The tokenizer to use for text processing. Defaults to `None`.
            template_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the template. Defaults to `None`.
            input_mode (Literal["history", "text", "tokens"], optional): The mode of input to the environment. Defaults to `"history"`.
            data_key (str, optional): The spec of the data returned by the dataloader (or better, its collate_fn).
                Defaults to `None` (automatically determined based on the input_mode).
            system_prompt (str | None, optional): The system prompt to use for the environment. Defaults to `None`.

        Returns:
            DatasetChatEnv: The chat environment.
        """
        return DatasetChatEnv.from_dataloader(
            dataloader=dataloader,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
            primers=primers,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            input_mode=input_mode,
            data_key=data_key,
            system_prompt=system_prompt,
        )

    # def _post_step_mdp_hooks(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     """Allows modification of the tensordict after the step_mdp."""
    # if self.input_mode == "history":
    #     tensordict.exclude(
    #         ("history", "response"), ("history", "full"), inplace=True
    #     )
    # if self.input_mode in ("text", "history"):
    #     tensordict.exclude(("text", "response"), ("text", "full"), inplace=True)
    # if self.input_mode in ("tokens", "history", "text"):
    #     tensordict.exclude(("tokens", "response"), ("tokens", "full"), inplace=True)
    # if "log_probs" in tensordict.keys():
    #     tensordict.exclude(
    #             ("log_probs", "response"), ("log_probs", "full"), inplace=True
    #         )
    #     return tensordict

    def _step(self, tensordict):
        if self.input_mode == "history":
            return self._step_history(tensordict)
        if self.input_mode in ("text", "history"):
            return self._step_text(tensordict)
        if self.input_mode in ("tokens", "history", "text"):
            return self._step_tokens(tensordict)
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}")

    def _step_history(self, tensordict):
        """Step the environment in history mode."""
        # get history from tensordict
        chat_history: ChatHistory = tensordict["history"]
        # prompt = chat_history.prompt
        full = chat_history.full
        # response = chat_history.response
        empty_td = tensordict.empty(device=self.device)
        # Old full will be new prompt - can be modified at will
        new_history = ChatHistory(prompt=full)
        empty_td.set("history", new_history)
        return empty_td

    def _step_text(self, tensordict):
        """Step the environment in text mode."""
        # get text from tensordict
        text: Text = tensordict["text"]
        full = text.full
        empty_td = tensordict.empty(device=self.device)
        new_history = Text(prompt=full)
        empty_td.set("text", new_history)
        return empty_td

    def _step_tokens(self, tensordict):
        """Step the environment in tokens mode."""
        # get tokens from tensordict
        tokens: Tokens = tensordict["tokens"]
        full = tokens.full
        empty_td = tensordict.empty(device=self.device)
        new_history = Tokens(prompt=full)
        empty_td.set("tokens", new_history)
        return empty_td

    def _reset(self, tensordict: TensorDictBase | None, **kwargs):
        if tensordict is None:
            raise RuntimeError(
                f"{type(self).__name__} expects a tensordict as input. Got `None`."
            )
        # Find the total text
        content = tensordict.get(self.data_key)
        if content is None:
            raise RuntimeError(
                f"{type(self).__name__} expects a tensordict with a {self.data_key} key, got {tensordict.keys()}"
            )
        if content.batch_size != self.batch_size:
            for s in reversed(self.batch_size):
                content = [content for _ in range(s)]

        # FIXME: Assume the text is not formatted and this is just content
        role = self.user_role
        for s in reversed(self.batch_size):
            role = [role for _ in range(s)]
        history = History(role=role, content=content, batch_size=self.batch_size)
        if self.system_prompt is not None:
            system_role = self.system_role
            history_system = History(
                role=system_role,
                content=self.system_prompt,
            )
            for s in reversed(self.batch_size):
                history_system = lazy_stack([history_system for _ in range(s)])
            history = lazy_stack([history_system, history], -1)
        else:
            history = history.unsqueeze(-1)

        # Now that we have the history, call the specific reset method
        if self.input_mode == "history":
            return (
                self._reset_history(tensordict, history)
                .update(tensordict)
                .to_lazystack(0)
            )
        elif self.input_mode == "text":
            return (
                self._reset_text(tensordict, history).update(tensordict).to_lazystack(0)
            )
        elif self.input_mode == "tokens":
            return (
                self._reset_tokens(tensordict, history)
                .update(tensordict)
                .to_lazystack(0)
            )
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}")

    def _reset_history(self, tensordict: TensorDictBase, history: History):
        # Simplest case: history is the prompt
        chat_history = ChatHistory._from_tensordict(
            tensordict.empty(device=self.device)
        )
        chat_history.prompt = history
        return tensordict.empty(device=self.device).set("history", chat_history)

    def _reset_text(self, tensordict: TensorDictBase, history: History):
        # We need to parse the history to a text
        text = history.apply_chat_template(
            tokenizer=self.tokenizer, add_generation_prompt=True, **self.template_kwargs
        )
        txt = Text._from_tensordict(tensordict.empty())
        txt.prompt = text
        result = tensordict.empty(device=self.device).set("text", txt)
        return result

    def _reset_tokens(self, tensordict: TensorDictBase, history: History):
        # We need to parse the history to a tokens
        tokens = history.apply_chat_template(
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **self.template_kwargs,
        )
        tokens_obj = Tokens._from_tensordict(tensordict.empty().to_lazystack(0))
        for to, tok in _zip_strict(tokens_obj.unbind(0), tokens["input_ids"]):
            to.prompt = tok
        result = tensordict.empty(device=self.device).set("tokens", tokens_obj)
        return result

    def _set_seed(self, seed):
        return


class DatasetChatEnv(TransformedEnv):
    """Base class for chat environment with queries pulled from a dataset.

    Typical usage include RLHF (Reinforcement Learning from Human feedback) or RLVR (Reinforcement learning with Verifiable rewards).

    Keyword Args:
        dataset (str): The name of the dataset.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to `True`.
        name (str, optional): name of the dataset configuration.
        split (str, optional): the split to use (usually from `"train"`, `"val"` or `"test"`). Defaults to `None` (no split).
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
        collate_fn (Callable | None, optional): A custom collate function for data loading. If `None`, a default
            collate function is used that renames the `"text"` key to `"query"` to avoid conflicts with the `"text"` key
            in the tensordict returned by TorchRL components. Defaults to `None`.
        input_mode (Literal["history", "text", "tokens"], optional): The mode of input to the environment. Defaults to `"history"`.
        data_key (str, optional): The spec of the data returned by the dataloader (or better, its collate_fn).
            Defaults to `None` (automatically determined based on the input_mode).
        system_prompt (str | None, optional): The system prompt to use for the environment. Defaults to `None`.
        ray_backend (bool, optional): Whether to use the Ray backend for data loading. Defaults to `False`.
            Using this backend allows for explicit resource control and avoids serialization issues, as well as
            sharing the same dataloader across multiple environments and actors.
        dataloader_actor_name (str | None, optional): Name of the Ray actor to use for data loading.
            Ignored if `ray_backend` is `None`.

    .. seealso:: `DatasetChatEnv` is a thin wrapper around :class:`~torchrl.envs.llm.ChatEnv` bucketed with a
        :class:`~torchrl.envs.llm.DataLoadingPrimer` transform. See these two classes for more insight on data format
        and functionality.

    .. seealso:: Examples of `DatasetChatEnv` include :class:`~torchrl.envs.llm.GSM8KEnv` and :class:`~torchrl.envs.llm.IFEvalEnv`.

    """

    SYSTEM_PROMPT: str | None = None

    def __init__(
        self,
        *,
        dataset: str,
        shuffle: bool = True,
        name: str | None = None,
        split: Literal["train", "val", "test"] | None = None,
        num_envs: int = 1,
        repeats: int | None = None,
        batch_size_dl: int = 1,
        seed: int | None = None,
        group_repeats: bool = False,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        device: torch.device | None = None,
        template_kwargs: dict[str, Any] | None = None,
        apply_template: bool | None = False,
        collate_fn: Callable[[Any], Any] | None = None,
        input_mode: Literal["history", "text", "tokens"] = "history",
        data_key: str | None = None,
        primers: Composite | None = None,
        system_prompt: str | None = None,
        ray_backend: bool = False,
        dataloader_actor_name: str | None = None,
    ):
        from tensordict import list_to_stack

        if not list_to_stack():
            raise RuntimeError(
                "list_to_stack() must return True. Use LIST_TO_STACK=1 or `tensordict.set_list_to_stack(True).set()` "
                "at the beginning of the script."
            )

        batch_size = (num_envs,)

        dataloader_factory = functools.partial(
            self._dataloader_factory,
            dataset=dataset,
            name=name,
            split=split,
            seed=seed,
            batch_size_dl=batch_size_dl,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        self._from_dataloader(
            self,
            dataloader=None,
            dataloader_factory=dataloader_factory,
            ray_backend=ray_backend,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
            primers=primers,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            input_mode=input_mode,
            data_key=data_key,
            system_prompt=system_prompt,
            dataloader_actor_name=dataloader_actor_name,
        )

    @staticmethod
    def _dataloader_factory(
        dataset, name, split, seed, batch_size_dl, shuffle, collate_fn
    ):
        from datasets import load_dataset

        dataset_obj = load_dataset(dataset, name)
        if split is None and "train" in dataset_obj:
            split = "train"
        if split is not None:
            dataset_obj = dataset_obj[split]
        # Env
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator(device=torch.get_default_device())
        generator.manual_seed(seed)

        dataloader = DataLoader(  # noqa: TOR401
            dataset_obj,
            batch_size=batch_size_dl,
            shuffle=shuffle,
            collate_fn=collate_fn if collate_fn is not None else _default_collate_fn,
            generator=generator,
        )
        return dataloader

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        *,
        repeats: int | None = None,
        device: torch.device | None = None,
        group_repeats: bool = False,
        batch_size: tuple | torch.Size | None = None,
        primers: Composite | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        template_kwargs: dict[str, Any] | None = None,
        input_mode: Literal["history", "text", "tokens"] = "history",
        data_key: str | None = None,
        system_prompt: str | None = None,
    ):
        """Create a chat environment from a dataloader.

        Args:
            dataloader (DataLoader): The dataloader to use.

        Keyword Args:
            repeats (int | None, optional): The number of times to repeat each sample from the dataset (mainly for Monte-Carlo
                based value estimation). If `None`, the dataset is not repeated. Defaults to `None`.
            device (torch.device | None, optional): The device to use for computations. Defaults to None.
            group_repeats (bool, optional): Whether to group repeated samples together. Defaults to `False`.
            batch_size (tuple | torch.Size | None, optional): The batch size for data loading. Defaults to `1`.
            primers (Composite | None, optional): The primers to use for data loading. Defaults to `None`.
            tokenizer (transformers.AutoTokenizer | None, optional): The tokenizer to use for text processing. Defaults to `None`.
            template_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the template. Defaults to `None`.
            input_mode (Literal["history", "text", "tokens"], optional): The mode of input to the environment. Defaults to `"history"`.
            data_key (str, optional): The spec of the data returned by the dataloader (or better, its collate_fn).
                Defaults to `None` (automatically determined based on the input_mode).
            system_prompt (str | None, optional): The system prompt to use for the environment. Defaults to `None`.

        Returns:
            ChatEnv: The chat environment.
        """
        self = cls.__new__(cls)
        return cls._from_dataloader(
            self,
            dataloader=dataloader,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
            primers=primers,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            input_mode=input_mode,
            data_key=data_key,
            system_prompt=system_prompt,
        )

    @classmethod
    def _from_dataloader(
        cls,
        self,
        dataloader=None,
        *,
        dataloader_factory=None,
        repeats: int | None = None,
        device: torch.device | None = None,
        group_repeats: bool = False,
        batch_size: tuple | torch.Size | None = None,
        primers: Composite | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        template_kwargs: dict[str, Any] | None = None,
        input_mode: Literal["history", "text", "tokens"] = "history",
        data_key: str | None = None,
        system_prompt: str | None = None,
        ray_backend: bool = False,
        dataloader_actor_name: str | None = None,
    ):
        if ray_backend:
            dl_cls = functools.partial(
                RayDataLoadingPrimer, actor_name=dataloader_actor_name
            )
        else:
            if dataloader_actor_name is not None:
                raise ValueError(
                    "dataloader_actor_name must be None if ray_backend is False"
                )
            dl_cls = DataLoadingPrimer
        primer = dl_cls(
            dataloader=dataloader,
            dataloader_factory=dataloader_factory,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
            primers=primers,
        )
        env_base = ChatEnv(
            batch_size=batch_size,
            system_prompt=cls.SYSTEM_PROMPT if system_prompt is None else system_prompt,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            input_mode=input_mode,
            data_key=data_key,
            device=device,
        )
        TransformedEnv.__init__(self, env_base, primer)
        return self

    def reset_dataloader(self):
        """Reset the dataloader.

        This is useful when the dataloader is not infinite and we want to reset it.

        Returns:
            self: The environment itself.
        """
        if hasattr(self.transform, "__getitem__"):
            self.transform[0].reset_dataloader()
        return self
