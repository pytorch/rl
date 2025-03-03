# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase, unravel_key
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict
from torch.utils.data import DataLoader
from torchrl.data.map.hash import SipHash
from torchrl.data.tensor_specs import (
    Bounded,
    Categorical as CategoricalSpec,
    Composite,
    NonTensor,
    TensorSpec,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import _StepMDP


class LLMEnv(EnvBase):
    """A text generation environment.

    This environment is designed to work with language models, where the observation is a string or a tensor of
    integers representing a sequence of tokens.
    The action is also a string or a tensor of integers, which is concatenated to the previous observation to form the
    new observation.

    By default, this environment is meant to track history for a prompt. Users can append transforms to tailor
    this to their use case, such as Chain of Thought (CoT) reasoning or other custom processing.

    Users must append a transform to set the "done" condition, which would trigger the loading of the next prompt.

    Prompts to the language model can be loaded when the environment is ``reset`` if the environment is created via :meth:`~from_dataloader`

    Args:
        observation_key (NestedKey, optional): The key in the tensordict where the observation is stored. Defaults to
            ``"observation"``.
        action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to ``"action"``.
        str2str (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``False``.
        device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
        vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
            unbounded vocabulary. Defaults to ``None``.

    .. seealso:: :class:`~torchrl.envs.DataLoadingPrimer` for examples.

    Methods:
        from_dataloader: Creates an LLMEnv instance from a dataloader.

    """

    def __init__(
        self,
        *,
        observation_key: NestedKey = "observation",
        action_key: NestedKey = "action",
        str2str: bool = False,
        device: torch.device | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(device=device)
        self._batch_locked = False
        self.str2str = str2str
        self.vocab_size = vocab_size
        self.observation_key = unravel_key(observation_key)
        # self.action_key = unravel_key(action_key)
        if str2str:
            self.observation_spec = Composite(
                {
                    observation_key: NonTensor(
                        example_data="a string", batched=True, shape=()
                    )
                }
            )
            self.action_spec = Composite(
                {action_key: NonTensor(example_data="a string", batched=True, shape=())}
            )
        else:
            if vocab_size is None:
                self.observation_spec = Composite(
                    {
                        observation_key: Unbounded(
                            shape=(-1,), dtype=torch.int64, device=device
                        )
                    }
                )
                self.action_spec = Composite(
                    {
                        action_key: Unbounded(
                            shape=(-1,), dtype=torch.int64, device=device
                        )
                    }
                )
            else:
                self.observation_spec = Composite(
                    {
                        observation_key: Bounded(
                            shape=(-1,),
                            dtype=torch.int64,
                            low=0,
                            high=vocab_size,
                            device=device,
                        )
                    }
                )
                self.action_spec = Composite(
                    {
                        action_key: Bounded(
                            shape=(-1,),
                            dtype=torch.int64,
                            low=0,
                            high=vocab_size,
                            device=device,
                        )
                    }
                )
        self.full_done_spec = Composite(
            done=Unbounded(shape=(1,), dtype=torch.bool),
            truncated=Unbounded(shape=(1,), dtype=torch.bool),
            terminated=Unbounded(shape=(1,), dtype=torch.bool),
        )

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        *,
        observation_key: NestedKey = "observation",
        action_key: NestedKey = "action",
        str2str: bool = False,
        device: torch.device | None = None,
        vocab_size: int | None = None,
        primers: Composite | None = None,
        data_keys: list[NestedKey] | None = None,
        data_specs: list[TensorSpec] | None = None,
        example_data: Any = None,
        stack_method: Callable[[Any], Any]
        | Literal["as_nested_tensor", "as_padded_tensor"] = None,
    ) -> LLMEnv:
        """Creates an LLMEnv instance from a dataloader.

        This method creates an LLMEnv instance and appends a DataLoadingPrimer to it, which populates ``data_keys`` (by default ``observation_key``) with data from the provided dataloader when the environment is reset.

        Args:
            dataloader (DataLoader): The dataloader to load data from.
            observation_key (NestedKey, optional): The key in the tensordict where the observation is stored. Defaults
                to ``"observation"``.
            action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to ``"action"``.
            str2str (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``False``.
            device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
            vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
                unbounded vocabulary. Defaults to ``None``.
            primers (Composite | None, optional): The primers to use for each key in the dataloader.
                Defaults to ``None``.
            data_keys (list[NestedKey] | None, optional): The keys to use for each item in the dataloader. If not passed ``observation_key`` will be populated with the data.
                Defaults to ``None``.
            data_specs (list[TensorSpec] | None, optional): The specs to use for each item in the dataloader.
                Defaults to ``None``.
            example_data (Any, optional): Example data to use for initializing the primer. Defaults to ``None``.
            stack_method (Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"], optional): The
                method to use for stacking the data. Defaults to ``None``.

        Returns:
            LLMEnv: The created LLMEnv instance.
        """
        from torchrl.envs import DataLoadingPrimer

        primer = DataLoadingPrimer(
            dataloader=dataloader,
            primers=primers,
            data_keys=data_keys if data_keys is not None else [observation_key],
            data_specs=data_specs,
            example_data=example_data,
            stack_method=stack_method,
        )
        env = LLMEnv(
            str2str=str2str,
            device=device,
            observation_key=observation_key,
            action_key=action_key,
            vocab_size=vocab_size,
        )
        return env.append_transform(primer)

    @staticmethod
    def _check_obs_act_and_cat(obs, action):
        if not isinstance(obs, str):
            raise TypeError(f"Observation must be a string, got {type(obs)}.")
        if not isinstance(action, str):
            raise TypeError(f"Action must be a string, got {type(action)}.")
        return obs + action

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        # Cat action entry with prev obs
        if self.str2str:
            obs = tensordict[self.observation_key]
            action = tensordict[self.action_key]
            if not tensordict.batch_size:
                if not isinstance(obs, str) or not isinstance(action, str):
                    raise TypeError(
                        "The tensordict is batchless, yet the action and/or observations are not "
                        f"strings but {type(action)} and {type(obs)}, respectivly."
                    )
                observation = self._check_obs_act_and_cat(obs, action)
            else:
                observation = NonTensorStack(
                    *[
                        self._check_obs_act_and_cat(_obs, _action)
                        for (_obs, _action) in _zip_strict(obs, action)
                    ]
                )
        else:
            try:
                obs: torch.Tensor = tensordict.get(self.observation_key)
                action = tensordict.get(self.action_key)
                if getattr(obs, "is_nested", False):
                    observation = torch.nested.as_nested_tensor(
                        [
                            torch.cat(
                                [
                                    _obs,
                                    _action,
                                ],
                                -1,
                            )
                            for _obs, _action in _zip_strict(
                                obs.unbind(0), action.unbind(0)
                            )
                        ],
                        layout=obs.layout,
                    )
                else:
                    observation = torch.cat(
                        [
                            obs,
                            action,
                        ],
                        -1,
                    )
            except TypeError:
                raise TypeError(
                    "Failed to cat action and observation tensors. Check that str2str argument is correctly "
                    f"set in {type(self).__name__}."
                )
        return tensordict.empty().set(self.observation_key, observation)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # We should have an observation by this time, if not raise an exception
        if tensordict is None or self.observation_key not in tensordict.keys(
            isinstance(self.observation_key, tuple)
        ):
            raise KeyError(
                f"Observation key {self.observation_key} is not defined. Make sure a TensorDictPrimer (eg, "
                f"torchrl.envs.DataLoadingPrimer) is appended to the env transforms."
            )
        return tensordict.copy()

    def _set_seed(self, seed: int | None):
        return seed


class LLMHashingEnv(EnvBase):
    """A text generation environment that uses a hashing module to identify unique observations.

    The primary goal of this environment is to identify token chains using a hashing function.
    This allows the data to be stored in a :class:`~torchrl.data.MCTSForest` using nothing but hashes as node
    identifiers, or easily prune repeated token chains in a data structure.
    The following figure gives an overview of this workflow:

    .. figure:: /_static/img/rollout-llm.png
        :alt: Data collection loop with our LLM environment.

    Args:
        vocab_size (int): The size of the vocabulary. Can be omitted if the tokenizer is passed.

    Keyword Args:
        hashing_module (Callable[[torch.Tensor], torch.Tensor], optional):
            A hashing function that takes a tensor as input and returns a hashed tensor.
            Defaults to :class:`~torchrl.data.SipHash` if not provided.
        observation_key (NestedKey, optional): The key for the observation in the TensorDict.
            Defaults to "observation".
        text_output (bool, optional): Whether to include the text output in the observation.
            Defaults to True.
        tokenizer (transformers.Tokenizer | None, optional):
            A tokenizer function that converts text to tensors.
            Only used when `text_output` is `True`.
            Must implement the following methods: `decode` and `batch_decode`.
            Defaults to ``None``.
        text_key (NestedKey | None, optional): The key for the text output in the TensorDict.
            Defaults to "text".

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import LLMHashingEnv
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        >>> x = tokenizer(["Check out TorchRL!"])["input_ids"]
        >>> env = LLMHashingEnv(tokenizer=tokenizer)
        >>> td = TensorDict(observation=x, batch_size=[1])
        >>> td = env.reset(td)
        >>> print(td)
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                hash: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                observation: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                text: NonTensorStack(
                    ['Check out TorchRL!'],
                    batch_size=torch.Size([1]),
                    device=None)},
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        vocab_size: int | None = None,
        *,
        hashing_module: Callable[[torch.Tensor], torch.Tensor] = None,
        observation_key: NestedKey = "observation",
        text_output: bool = True,
        tokenizer: Callable[[str | list[str]], torch.Tensor] | None = None,
        text_key: NestedKey | None = "text",
    ):
        super().__init__()
        if vocab_size is None:
            if tokenizer is None:
                raise TypeError(
                    "You must provide a vocab_size integer if tokenizer is `None`."
                )
            vocab_size = tokenizer.vocab_size
        self._batch_locked = False
        if hashing_module is None:
            hashing_module = SipHash()

        self._hashing_module = hashing_module
        self._tokenizer = tokenizer
        self.observation_key = observation_key
        observation_spec = {
            observation_key: CategoricalSpec(n=vocab_size, shape=(-1,)),
            "hashing": Unbounded(shape=(1,), dtype=torch.int64),
        }
        self.text_output = text_output
        if not text_output:
            text_key = None
        elif text_key is None:
            text_key = "text"
        if text_key is not None:
            observation_spec[text_key] = NonTensor(shape=())
            self.text_key = text_key
        self.observation_spec = Composite(observation_spec)
        self.action_spec = Composite(action=CategoricalSpec(vocab_size, shape=(1,)))
        _StepMDP(self)

    def make_tensordict(self, input: str | list[str]) -> TensorDict:
        """Converts a string or list of strings in a TensorDict with appropriate shape and device."""
        list_len = len(input) if isinstance(input, list) else 0
        tensordict = TensorDict(
            {self.observation_key: self._tokenizer(input)}, device=self.device
        )
        if list_len:
            tensordict.batch_size = [list_len]
        return self.reset(tensordict)

    def _reset(self, tensordict: TensorDictBase):
        """Initializes the environment with a given observation.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the initial observation.

        Returns:
            A TensorDict containing the initial observation, its hash, and other relevant information.

        """
        out = tensordict.empty()
        obs = tensordict.get(self.observation_key, None)
        if obs is None:
            raise RuntimeError(
                f"Resetting the {type(self).__name__} environment requires a prompt."
            )
        if self.text_output:
            if obs.ndim > 1:
                text = self._tokenizer.batch_decode(obs)
                text = NonTensorStack.from_list(text)
            else:
                text = self._tokenizer.decode(obs)
                text = NonTensorData(text)
            out.set(self.text_key, text)

        if obs.ndim > 1:
            out.set("hashing", self._hashing_module(obs).unsqueeze(-1))
        else:
            out.set("hashing", self._hashing_module(obs.unsqueeze(0)).transpose(0, -1))

        if not self.full_done_spec.is_empty():
            out.update(self.full_done_spec.zero(tensordict.shape))
        else:
            out.set("done", torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool))
            out.set(
                "terminated", torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool)
            )
        return out

    def _step(self, tensordict):
        """Takes an action (i.e., the next token to generate) and returns the next observation and reward.

        Args:
            tensordict: A TensorDict containing the current observation and action.

        Returns:
            A TensorDict containing the next observation, its hash, and other relevant information.
        """
        out = tensordict.empty()
        action = tensordict.get("action")
        obs = torch.cat([tensordict.get(self.observation_key), action], -1)
        kwargs = {self.observation_key: obs}

        catval = torch.cat([tensordict.get("hashing"), action], -1)
        if obs.ndim > 1:
            new_hash = self._hashing_module(catval).unsqueeze(-1)
        else:
            new_hash = self._hashing_module(catval.unsqueeze(0)).transpose(0, -1)

        if self.text_output:
            if obs.ndim > 1:
                text = self._tokenizer.batch_decode(obs)
                text = NonTensorStack.from_list(text)
            else:
                text = self._tokenizer.decode(obs)
                text = NonTensorData(text)
            kwargs[self.text_key] = text
        kwargs.update(
            {
                "hashing": new_hash,
                "done": torch.zeros((*tensordict.batch_size, 1), dtype=torch.bool),
                "terminated": torch.zeros(
                    (*tensordict.batch_size, 1), dtype=torch.bool
                ),
            }
        )
        return out.update(kwargs)

    def _set_seed(self, *args):
        """Sets the seed for the environment's randomness.

        .. note:: This environment has no randomness, so this method does nothing.
        """
