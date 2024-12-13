# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Callable, List, Union

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack

from torchrl.data import (
    Categorical as CategoricalSpec,
    Composite,
    NonTensor,
    SipHash,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import _StepMDP


class LLMHashingEnv(EnvBase):
    """A text generation environment that uses a hashing module to identify unique observations.

    The primary goal of this environment is to identify token chains using a hashing function.
    This allows the data to be stored in a :class:`~torchrl.data.MCTSForest` using nothing but hashes as node
    identifiers, or easily prune repeated token chains in a data structure.
    The following figure gives an overview of this workflow:

    .. figure:: /_static/img/rollout-llm.png
        :alt: Data collection loop with our LLM environment.

    .. seealso:: the :ref:`Beam Search <beam_search>` tutorial gives a practical example of how this env can be used.

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
        tokenizer: Callable[[Union[str, List[str]]], torch.Tensor] | None = None,
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

    def make_tensordict(self, input: str | List[str]) -> TensorDict:
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
        pass
