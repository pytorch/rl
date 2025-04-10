# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

from typing import Any, Callable, Literal

import torch

from tensordict import (
    is_leaf_nontensor,
    LazyStackedTensorDict,
    NestedKey,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict
from torch.utils.data import DataLoader

from torchrl._utils import _replace_last
from torchrl.data.map.hash import SipHash
from torchrl.data.tensor_specs import (
    Bounded,
    Categorical as CategoricalSpec,
    Composite,
    NonTensor,
    Unbounded,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import _StepMDP
from torchrl.modules.utils.utils import _unpad_tensors


class LLMEnv(EnvBase):
    """A text generation environment for language models.

    This environment is designed to work with language models, where the observation is a string or a tensor of
    integers representing a sequence of tokens. The action is also a string or a tensor of integers, which is
    concatenated to the previous observation to form the new observation.

    By default, this environment is meant to track history for a prompt. Users can append transforms to tailor
    this to their use case, such as Chain of Thought (CoT) reasoning or other custom processing.

    Users must append a transform to set the "done" condition, which would trigger the loading of the next prompt.
    Prompts to the language model can be loaded when the environment is ``reset`` if the environment is created via
    :meth:`~from_dataloader`.

    .. note:: The default arguments of the `LLMEnv` class are set to make it easy to run this environment with
        the vllm backend (:class:`~torchrl.modules.vLLMWrapper`).

    Keyword Args:
        token_key (NestedKey, optional): The key in the tensordict where the tokens are stored (when `from_text=False`).
            Defaults to ``"tokens"``.
        str_key (NestedKey, optional): The key in the tensordict where the string input is stored (when `from_text=True`).
            Defaults to ``"text"``.
        attention_key (NestedKey, optional): The key in the tensordict where the attention mask is stored.
            Defaults to ``"attention_mask"``.
        action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to
            ``"tokens_response"`` or ``"text_response"``.
        reward_key (NestedKey, optional): The key in the tensordict where the reward is stored if `assign_reward=True`.
            Defaults to  ``"reward"``.
        from_text (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``True``.
        device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
        vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
            unbounded vocabulary. Defaults to ``None``.
        has_attention (bool, optional): If ``True``, an attention mask is to be used under the key indicated by
            :attr:`attention_key`. Defaults to ``True``.
        assign_reward (bool, optional): If ``True``, a zero-valued reward of shape equal to the action shape
            is written during calls to `step()`. Defaults to ``False``.
        assign_done (bool, optional): If ``True``, a zero-valued done and terminated state of shape equal to the
            action shape is written during calls to `step()`. Defaults to ``False``.
            .. note:: Regardless of the value assigned to `assign_done`, a done state will be written at the root
                as it is a requirement for all TorchRL environments.
        batch_size (int or torch.Size, optional): Batch size of the environment.
            If left empty, an empty batch-size is assumed.
            The batch size can be null (`torch.Size([])`) or one-dimensional. Batchless environments are not supported.

            .. note:: When using a :class:`~torchrl.envs.DataLoadingPrimer` transform, the batch-size of the env
                and the transform should match.

        eos_token_id (int, optional): The token id of the end of the sequence. If passed, the `done` state
            is set to `True` when detected. Defaults to `None`.

    .. seealso:: :class:`~torchrl.envs.DataLoadingPrimer` for examples.

    Methods:
        from_dataloader: Creates an LLMEnv instance from a dataloader.

    """

    _DEFAULT_TOKEN_KEY = "tokens"
    _DEFAULT_STR_KEY = "text"
    _DEFAULT_ATTENTION_KEY = "attention_mask"
    _DEFAULT_ACTION_TOKENS_KEY = "tokens_response"
    _DEFAULT_ACTION_STR_KEY = "text_response"

    def __init__(
        self,
        *,
        token_key: NestedKey | None = None,
        str_key: NestedKey | None = None,
        attention_key: NestedKey | None = None,
        action_key: NestedKey | None = None,
        reward_key: NestedKey = "reward",
        from_text: bool = True,
        device: torch.device | None = None,
        vocab_size: int | None = None,
        assign_reward: bool = False,
        assign_done: bool = False,
        batch_size: int | torch.Size | None = None,
        has_attention: bool = True,
        # Experimental
        as_llm_data: bool = False,
        eos_token_id: int | None = None,
    ) -> None:
        self.as_llm_data = as_llm_data
        if token_key is None:
            token_key = self._DEFAULT_TOKEN_KEY
        if str_key is None:
            str_key = self._DEFAULT_STR_KEY
        if attention_key is None:
            attention_key = self._DEFAULT_ATTENTION_KEY
        if action_key is None:
            if from_text:
                action_key = self._DEFAULT_ACTION_STR_KEY
            else:
                action_key = self._DEFAULT_ACTION_TOKENS_KEY
        self._batch_locked = True
        if batch_size is None:
            batch_size = ()
        else:
            if not isinstance(batch_size, (tuple, list)):
                batch_size = (batch_size,)
            elif len(batch_size) > 1:
                raise TypeError(
                    f"batch-size of LLMEnv must be 0 or 1d. Got batch_size={batch_size}."
                )
        super().__init__(
            device=device,
            batch_size=batch_size,
        )
        self.has_attention = has_attention
        self.from_text = from_text
        self.vocab_size = vocab_size
        self.token_key = unravel_key(token_key)
        self.str_key = unravel_key(str_key)
        if attention_key is not None:
            attention_key = unravel_key(attention_key)
        self.attention_key = attention_key
        self.assign_reward = assign_reward
        self.assign_done = assign_done
        self.eos_token_id = eos_token_id
        if eos_token_id is None:
            warnings.warn(
                "eos_token_id is missing. This means that the environment will not be able to capture its "
                "done state automatically. This may lead to undefined behaviors when the generated text reaches "
                "an eos_token.",
                category=UserWarning,
            )

        # self.action_key = unravel_key(action_key)
        if from_text:
            self.full_observation_spec_unbatched = Composite(
                {
                    self.str_key: NonTensor(
                        example_data="a string",
                        batched=True,
                        shape=(),
                        device=device,
                    )
                }
            )
            self.full_action_spec_unbatched = Composite(
                {
                    action_key: NonTensor(
                        example_data="a string", batched=True, shape=(), device=device
                    )
                }
            )
        else:
            if vocab_size is None:
                observation_spec = {
                    token_key: Unbounded(shape=(-1,), dtype=torch.int64, device=device)
                }
                if self.has_attention:
                    observation_spec[attention_key] = Unbounded(
                        shape=(-1,), dtype=torch.int64, device=device
                    )
                self.full_observation_spec_unbatched = Composite(observation_spec)
                self.full_action_spec_unbatched = Composite(
                    {
                        action_key: Unbounded(
                            shape=(-1,), dtype=torch.int64, device=device
                        )
                    }
                )
            else:
                self.full_observation_spec_unbatched = Composite(
                    {
                        token_key: Bounded(
                            shape=(-1,),
                            dtype=torch.int64,
                            low=0,
                            high=vocab_size,
                            device=device,
                        )
                    }
                )
                self.full_action_spec_unbatched = Composite(
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
        STR2STR_ERR = ValueError(
            "from_text cannot be True when either of assign_reward / assign_done are True. "
            "Tokens are required to compute the reward shape."
        )
        if self.assign_reward:
            if self.from_text:
                raise STR2STR_ERR
            self.full_reward_spec_unbatched = Composite(
                {reward_key: Unbounded(shape=(-1,), device=device)}
            )
        else:
            self.full_reward_spec_unbatched = Composite(device=device)

        if not self.assign_done:
            # Use single done
            self.full_done_spec_unbatched = Composite(
                done=Unbounded(shape=(1,), dtype=torch.bool, device=device),
                terminated=Unbounded(shape=(1,), dtype=torch.bool, device=device),
            )
        elif self.from_text:
            raise STR2STR_ERR
        else:
            # Use single done
            self.full_done_spec_unbatched = Composite(
                tokens_data=Composite(
                    done=Unbounded(shape=(-1,), dtype=torch.bool, device=device),
                    terminated=Unbounded(shape=(-1,), dtype=torch.bool, device=device),
                ),
                done=Unbounded(shape=(1,), dtype=torch.bool, device=device),
                terminated=Unbounded(shape=(1,), dtype=torch.bool, device=device),
            )

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        *,
        tokenizer: transformers.PretrainedTokenizerBase | None = None,  # noqa
        token_key: NestedKey | None = None,
        str_key: NestedKey | None = None,
        attention_key: NestedKey | None = None,
        action_key: NestedKey | None = None,
        reward_key: NestedKey = "reward",
        from_text: bool = True,
        device: torch.device | None = None,
        vocab_size: int | None = None,
        batch_size: int | torch.Size | None = None,
        has_attention: bool = True,
        assign_reward: bool = False,
        assign_done: bool = False,
        primers: Composite | None = None,
        example_data: Any = None,
        stack_method: Callable[[Any], Any]
        | Literal["as_nested_tensor", "as_padded_tensor"] = None,
        repeats: int | None = None,
        group_repeats: bool = True,
        eos_token_id: int | None = None,
    ) -> LLMEnv:
        """Creates an LLMEnv instance from a dataloader.

        This method creates an LLMEnv instance and appends a DataLoadingPrimer to it, which populates ``data_keys`` (by default ``observation_key``) with data from the provided dataloader when the environment is reset.

        Args:
            dataloader (DataLoader): The dataloader to load data from.

        Keyword Args:
            tokenizer (transformers.PretrainedTokenizerBase or str, optional): the tokenizer to use. If ``None``,
                "bert-base-uncased" will be used by default. If a string is provided, it should be the name of a
                pre-trained tokenizer.

                .. note:: Using the `tokenizer` will append a :class:`~torchrl.envs.Tokenizer` transform to the environment.
                    If `from_text` is set to `True`, the tokenizer will be called during every iteration and the rollout
                    will contain both tokens and text data.
                    If `from_text` is set to `False`, the tokenizer will be called during reset only, and the only
                    text data in the rollout will be the text sampled from the dataset.

            token_key (NestedKey, optional): The key in the tensordict where the tokens are stored (when `from_text=False`).
                Defaults to ``("tokens_in", "input_ids")``.
            str_key (NestedKey, optional): The key in the tensordict where the string input is stored (when `from_text=True`).
                Defaults to ``"test"``.
            attention_key (NestedKey, optional): The key in the tensordict where the attention mask is stored.
                Defaults to ``("tokens_in", "input_ids")``
            action_key (NestedKey, optional): The key in the tensordict where the action is stored. Defaults to
                ``("tokens_out", "sequences")``.
            reward_key (NestedKey, optional): The key in the tensordict where the reward is stored if `assign_reward=True`.
                Defaults to  ``"reward"``.
            from_text (bool, optional): Whether the environment should expect strings as input and output. Defaults to ``True``.
            device (torch.device | None, optional): The device on which the environment should run. Defaults to ``None``.
            vocab_size (int | None, optional): The size of the vocabulary. If None, the environment will assume an
                unbounded vocabulary. Defaults to ``None``.
            has_attention (bool, optional): if ``True``, an attention mask is to be used under the key indicated by
                :attr:`attention_key`. Defaults to ``True``.
            assign_reward (bool, optional): if ``True``, a zero-valued reward of shape equal to to the action shape
                is written during calls to `step()`. Defaults to ``False``.
            assign_done (bool, optional): if ``True``, a zero-valued done and terminated state of shape equal to to the
                action shape is written during calls to `step()`. Defaults to ``False``.

                .. note:: regardless of the value assigned to `assign_done`, a done state will be written at the root
                    as it is a requirement for all TorchRL environments.

            batch_size (int or torch.Size, optional): Batch size of the environment.
                If left empty, the batch size is inferred from `dataloader.batch_size` if that attribute exists, otherwise
                it is set to `()`.
                The batch size can be null (`torch.Size([])`) or one-dimensional. Batchless environments are not supported.

                .. note:: When using a :class:`~torchrl.envs.DataLoadingPrimer` transform, the batch-size of the env
                    and the transform should match.

            primers (Composite | None, optional): The primers to use for each key in the dataloader.
                Defaults to ``None`` (inferred automatically from the first batch of data).
            stack_method (Callable[[Any], Any] | Literal["as_nested_tensor", "as_padded_tensor"], optional): The
                method to use for stacking the data. Defaults to ``None``.
            repeats (int, optional): How many times the same sample needs to appear successively. This can be useful in
                situations like GRPO where a single prompt is used multiple times to estimate the advantage using Monte-Carlo
                samples (rather than an advantage module).
            group_repeats (bool, optional): if ``True``, the batch-size is multiplied by the number of repeats such that
                all repeats are grouped in a single batch collected from the buffer. Defaults to ``True``.
            eos_token_id (int, optional): The token id of the end of the sequence. If passed, the `done` state
                is set to `True` when detected. Defaults to `None`.

        Returns:
            LLMEnv: The created LLMEnv instance.
        """
        from torchrl.envs import DataLoadingPrimer, Tokenizer

        if str_key is None:
            str_key = LLMEnv._DEFAULT_STR_KEY
        if token_key is None:
            token_key = LLMEnv._DEFAULT_TOKEN_KEY
        if attention_key is None:
            attention_key = LLMEnv._DEFAULT_ATTENTION_KEY
        elif tokenizer is not None and attention_key != _replace_last(
            token_key, "attention_mask"
        ):
            raise ValueError(
                "When using the Tokenizer, attention key must match `(*token_key[:-1], 'attention_mask')` where "
                f"`token_key` is a tuple-typed nested key. Got attention_key={attention_key} while expecting "
                f"{_replace_last(token_key, 'attention_mask')}."
            )

        if tokenizer is not None:
            if from_text:
                # In this case, the tokenizer is appended to the env after each step
                if action_key is None:
                    action_key = cls._DEFAULT_ACTION_STR_KEY
                tokenizer_transform = Tokenizer(
                    tokenizer=tokenizer,
                    in_keys=[str_key],
                    out_keys=[token_key],
                    # Assume that the tokens are named according to _DEFAULT_ACTION_TOKENS_KEY
                    in_keys_inv=[action_key],
                    out_keys_inv=[cls._DEFAULT_ACTION_TOKENS_KEY],
                    call_before_reset=False,
                    # We should always see the required entries
                    missing_tolerance=False,
                )
            else:
                # FIXME: This is broken - do we need it anyway?
                raise RuntimeError(
                    "tokenizers can only be used whenever from_text is set to `True`."
                )

        primer = DataLoadingPrimer(
            dataloader=dataloader,
            primers=primers,
            stack_method=stack_method,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
        )
        env = LLMEnv(
            from_text=from_text,
            device=device,
            token_key=token_key,
            str_key=str_key,
            attention_key=attention_key,
            action_key=action_key,
            reward_key=reward_key,
            vocab_size=vocab_size,
            assign_reward=assign_reward,
            assign_done=assign_done,
            batch_size=primer.batch_size,
            has_attention=has_attention,
            eos_token_id=eos_token_id,
        )
        if tokenizer is not None:
            env = env.append_transform(tokenizer_transform)
        return env.append_transform(primer)

    @staticmethod
    def _check_obs_act_and_cat(obs, action, *, device):
        if not isinstance(obs, str):
            raise TypeError(f"Observation must be a string, got {type(obs)}.")
        if not isinstance(action, str):
            raise TypeError(f"Action must be a string, got {type(action)}.")
        return NonTensorData(obs + action, device=device)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        next_td = tensordict.empty()
        self._make_next_obs(tensordict, next_td)
        self._maybe_make_reward(tensordict, next_td)
        self._maybe_make_done(tensordict, next_td)
        if self.as_llm_data:
            raise NotImplementedError()
        return next_td

    def _maybe_make_reward(
        self, tensordict: TensorDictBase, next_td: TensorDictBase
    ) -> TensorDictBase:
        if self.assign_reward:
            next_td.set(
                self.reward_key,
                torch.zeros_like(
                    tensordict.get(self.action_key), dtype=self.reward_spec.dtype
                ),
            )
        return next_td

    def _maybe_make_done(
        self,
        tensordict: TensorDictBase,
        next_td: TensorDictBase,
        resetting: bool = False,
    ) -> TensorDictBase:
        if self.assign_done:
            action = tensordict.get(self.action_key)
            if action is None:
                done = torch.zeros(
                    tensordict.shape + (1,), dtype=torch.bool, device=self.device
                )
            else:
                done = torch.zeros_like(action, dtype=torch.bool)
            next_td.set(("tokens_data", "terminated"), done)
            next_td.set(("tokens_data", "done"), done.clone())
            next_td.set(
                "done", next_td.get(("tokens_data", "done")).any(-1, keepdim=True)
            )
            next_td.set(
                "terminated",
                next_td.get(("tokens_data", "terminated")).any(-1, keepdim=True),
            )
        if not resetting and self.eos_token_id is not None:
            if self.from_text:
                token_action_key = self._DEFAULT_ACTION_TOKENS_KEY
            else:
                token_action_key = self.action_key
            action = tensordict.get(
                token_action_key, as_padded_tensor=True, padding_value=-1
            )
            mask = action == -1

            if action is None:
                raise RuntimeError(
                    f"Couldn't find the tokenized action with key {token_action_key} to set the done state in tensordict "
                    f"with keys {list(tensordict.keys(True))}."
                )
            full_done = action == self.eos_token_id
            done = full_done.any(-1, keepdim=True)
            next_td.set("done", done)
            next_td.set("terminated", done)
            if self.assign_done:
                full_done = _unpad_tensors(full_done, mask)
                next_td.set(("tokens_data", "terminated"), full_done)
                next_td.set(("tokens_data", "done"), full_done)
        return next_td

    def _make_next_obs(
        self, tensordict: TensorDictBase, nex_td: TensorDictBase
    ) -> TensorDictBase:
        # Cat action entry with prev obs
        if self.from_text:
            obs = tensordict[self.str_key]
            action = tensordict[self.action_key]
            if not tensordict.batch_size:
                if not isinstance(obs, str) or not isinstance(action, str):
                    raise TypeError(
                        "The tensordict is batchless, yet the action and/or observations are not "
                        f"strings but {type(action)} and {type(obs)}, respectivly."
                    )
                observation = self._check_obs_act_and_cat(
                    obs, action, device=self.device
                )
            else:
                observation = NonTensorStack(
                    *[
                        self._check_obs_act_and_cat(_obs, _action, device=self.device)
                        for (_obs, _action) in _zip_strict(obs, action)
                    ]
                )
            return nex_td.set(self.str_key, observation)
        else:
            try:
                obs: torch.Tensor = tensordict.get(self.token_key)
                action = tensordict.get(self.action_key)
                if getattr(obs, "is_nested", False):
                    observation = torch.nested.as_nested_tensor(
                        [
                            torch.cat([_obs, _action], -1)
                            for _obs, _action in _zip_strict(
                                obs.unbind(0), action.unbind(0)
                            )
                        ],
                        layout=obs.layout,
                    )
                else:
                    observation = torch.cat([obs, action], -1)
                    if self.has_attention:
                        attention_mask = tensordict.get(self.attention_key)
                        attention_mask = torch.cat(
                            [attention_mask, attention_mask.new_ones(action.shape)], -1
                        )
                        nex_td.set(self.attention_key, attention_mask)
            except TypeError:
                raise TypeError(
                    "Failed to cat action and observation tensors. Check that from_text argument is correctly "
                    f"set in {type(self).__name__}."
                )
            return nex_td.set(self.token_key, observation)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # We should have an observation by this time, if not raise an exception
        def check_token():
            return not self.from_text and (
                self.token_key not in tensordict.keys(isinstance(self.token_key, tuple))
            )

        def check_str():
            return self.from_text and (
                self.str_key not in tensordict.keys(isinstance(self.str_key, tuple))
            )

        if tensordict is None or check_token() or check_str():
            raise KeyError(
                f"Observation key {self.token_key}/{self.str_key} is not defined in tensordict with keys "
                f"{list(tensordict.keys(True, True, is_leaf=is_leaf_nontensor))}. Make sure a TensorDictPrimer (eg, "
                f"torchrl.envs.DataLoadingPrimer) is appended to the env transforms."
            )
        if not isinstance(tensordict, LazyStackedTensorDict) and tensordict.ndim:
            tensordict = LazyStackedTensorDict(*tensordict.unbind(0))
        td_reset = tensordict.copy()
        if td_reset.device != self.device:
            if self.device is None:
                td_reset.clear_device_()
            else:
                td_reset = td_reset.to(self.device)
        tensordict = self._maybe_make_done(tensordict, td_reset, resetting=True)
        if self.as_llm_data:
            raise NotImplementedError()
        return tensordict

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

    @set_list_to_stack(True)
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
