# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, List

import torch
from tensordict import NestedKey, TensorDictBase
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

    Args:
        vocab_size (int): The size of the vocabulary.
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

    """

    def __init__(
        self,
        vocab_size: int,
        hashing_module: Callable[[torch.Tensor], torch.Tensor] = None,
        observation_key: NestedKey = "observation",
        text_output: bool = True,
        tokenizer: Callable[[str | List[str]], torch.Tensor] | None = None,
        text_key: NestedKey | None = "text",
    ):
        super().__init__()
        self._batch_locked = False
        if hashing_module is None:
            hashing_module = SipHash()

        self._hashing_module = hashing_module
        self._tokenizer = tokenizer
        self.observation_key = observation_key
        observation_spec = {
            observation_key: CategoricalSpec(n=vocab_size, shape=(-1,)),
            "hash": Unbounded(shape=(1,), dtype=torch.int64),
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

    def _reset(self, tensordict: TensorDictBase):
        """Initializes the environment with a given observation.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the initial observation.

        Returns:
            A TensorDict containing the initial observation, its hash, and other relevant information.

        """
        out = tensordict.empty()
        obs = tensordict.get(self.observation_key)
        if self.text_output:
            if obs.ndim > 1:
                text = self._tokenizer.batch_decode(obs)
                text = NonTensorStack.from_list(text)
            else:
                text = self._tokenizer.decode(obs)
                text = NonTensorData(text)
            out.set(self.text_key, text)

        if obs.ndim > 1:
            out.set("hash", self._hashing_module(obs).unsqueeze(-1))
        else:
            out.set("hash", self._hashing_module(obs.unsqueeze(0)).transpose(0, -1))

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

        catval = torch.cat([tensordict.get("hash"), action], -1)
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
                "hash": new_hash,
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
