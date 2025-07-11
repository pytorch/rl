# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

from contextlib import nullcontext
from copy import copy
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import NestedKey, set_list_to_stack, TensorDictBase, unravel_key
from tensordict.utils import _zip_strict, is_seq_of_nested_key, logger as torchrl_logger
from torch.nn.utils.rnn import pad_sequence
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, Transform
from torchrl.envs.transforms.transforms import Compose
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.modules.llm.policies.common import LLMWrapperBase

if TYPE_CHECKING:
    import transformers


class KLRewardTransform(Transform):
    """A legacy transform for computing KL divergence-based rewards.

    **Deprecated**: This transform is maintained for backward compatibility but is no longer
    the recommended approach. Use :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL` instead,
    which provides better modularity and integration with the new wrapper design.

    **Recent Changes:**
    - **Legacy Status**: This transform is now considered legacy and may not work optimally
      with the new modular wrapper design.
    - **ChatHistory Integration**: Limited support for the new :class:`~torchrl.modules.llm.policies.ChatHistory` objects.
    - **Input Mode Support**: May not handle all input modes (`"history"`, `"text"`, `"tokens"`) consistently.

    **Recommendation**:
    Use :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL` for new code, which provides:
    - Better integration with the new wrapper design
    - Consistent support for all input modes
    - Proper handling of ChatHistory objects
    - More modular and composable architecture

    Args:
        gen_model (LLMWrapperBase): the generation model.
        ref_model (LLMWrapperBase): the reference model.

    Keyword Args:
        assistant_only (bool): whether to only compute KL on assistant tokens. Defaults to `True`.
        tokenizer (transformers.AutoTokenizer): the tokenizer to use. Defaults to `None`.
        detach (bool): whether to detach the KL from the computation graph. Defaults to `True`.
        device (torch.device): the device to use. Defaults to `None`.
        padding_side (str): the side of the padding when using pad_sequence. Defaults to `"left"`.

    Examples:
        >>> # Legacy usage (not recommended for new code)
        >>> transform = KLRewardTransform(gen_model, ref_model)
        >>>
        >>> # Recommended approach using RetrieveKL
        >>> from torchrl.envs.llm.transforms.kl import RetrieveKL
        >>> transform = RetrieveKL(gen_model, ref_model, assistant_only=True)

    .. seealso::
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL`: The recommended transform for KL divergence computation.
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveLogProb`: Base transform for retrieving log-probabilities.
        :class:`~torchrl.envs.llm.transforms.kl.KLComputation`: Transform for computing KL divergence between log-prob tensors.
    """

    DEFAULT_IN_KEYS = ["reward"]

    def __init__(
        self,
        ref_model: LLMWrapperBase,
        *,
        coef=1.0,
        in_keys=None,
        out_keys=None,
        log_prob_key: NestedKey = ("log_probs", "full"),
        device: torch.device | None = None,
        add_to_reward: bool = True,
        tokenizer: transformers.AutoTokenizer | None = None,
        assistant_only: bool = True,
        padding_side: str = "left",
    ):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        if len(out_keys) == len(in_keys):
            out_keys = out_keys + ["kl_penalty", "ref_log_prob"]
        elif len(out_keys) != len(in_keys) + 2:
            raise ValueError(
                "The out_keys must have the same length as the in_keys (plus two additional optional kl entries for logging)."
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not is_seq_of_nested_key(self.in_keys) or not is_seq_of_nested_key(
            self.out_keys
        ):
            raise ValueError(
                f"invalid in_keys / out_keys:\nin_keys={self.in_keys} \nout_keys={self.out_keys}"
            )
        if len(self.in_keys) != 1 or len(self.out_keys) != 3:
            raise ValueError(
                f"Only one in_key/out_key is allowed, got in_keys={self.in_keys}, out_keys={self.out_keys}."
            )
        self._out_keys = [unravel_key(out_key) for out_key in self._out_keys]

        if getattr(ref_model, "generate", False):
            raise ValueError(
                "The actor is configured to generate text, not compute the log-probs."
            )

        # update the in_keys for dispatch etc
        self.in_keys = self.in_keys + ref_model.in_keys
        self.in_keys = [unravel_key(in_key) for in_key in self.in_keys]

        self.add_to_reward = add_to_reward
        # check that the model has parameters
        self.__dict__["ref_model"] = ref_model

        # self._buffers["actor_params"] = params.clone().detach()

        self.device = device

        # find the sample log-prob key
        self.log_prob_full_key = log_prob_key

        self._tokenizer = tokenizer
        self.assistant_only = assistant_only
        self.padding_side = padding_side

        if not isinstance(coef, torch.Tensor):
            coef = torch.as_tensor(coef)
        self.register_buffer("coef", coef)
        # sanity check for the ref_model
        if not getattr(ref_model, "input_mode", "tokens") == "tokens":
            raise ValueError(
                "The ref_model must be configured to use tokens as input. Please set the `input_mode` argument to `tokens`."
            )

    @property
    def pad_output(self):
        # We need pad_output to match the pad_output of the inference model
        return self.ref_model.pad_output

    @property
    def tokenizer(self):
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer
        try:
            return self.ref_model.tokenizer
        except AttributeError:
            raise AttributeError(
                "The ref_model does not have a tokenizer. Please pass the tokenizer to the constructor."
            )

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        if self.action_key is None:
            parent = getattr(self, "parent", None)
            if parent is not None:
                action_keys = parent.action_keys
                if len(action_keys) != 1:
                    raise ValueError(
                        f"More than one action_key found. Please pass the `action_key` argument directly to {type(self).__name__}."
                    )
                action_key = action_keys[0]
                self.action_key = action_key
        return result

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._step(tensordict_reset, tensordict_reset)
        return tensordict_reset

    @property
    def action_key(self) -> NestedKey:
        # Get the action from the base env (a ChatEnv).
        if self.parent.base_env.input_mode == "history":
            return ("history", "full")
        if self.parent.base_env.input_mode == "text":
            return ("text", "full")
        if self.parent.base_env.input_mode == "tokens":
            return ("tokens", "full")
        raise ValueError(f"Invalid input mode: {self.parent.base_env.input_mode}")

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if self.device is not None:
            tensordict = tensordict.to(self.device)
            next_tensordict = next_tensordict.to(self.device)
        # tensordict = self._get_text_response(tensordict, next_tensordict)
        response = tensordict.get(self.action_key, None)
        if response is None:
            if not self.missing_tolerance:
                raise RuntimeError(
                    f"Action with key {self.action_key} not found data {tensordict}"
                )
            # being called after reset or without action, skipping
            if self.out_keys[0] != "reward" and self.parent is not None:
                next_tensordict.set(self.out_keys[0], self.parent.reward_spec.zero())
            return next_tensordict

        # We use the ("tokens", "full") key to get the log-probs of the reference model
        with torch.device(self.device) if self.device is not None else nullcontext():
            td_input = tensordict.copy()
            ref_log_prob_td = self.ref_model(td_input)
        if self.pad_output:
            ref_log_prob_padded = ref_log_prob_td.get(self.log_prob_full_key)
        else:
            ref_log_prob_unpadded = ref_log_prob_td.get(
                self.log_prob_full_key, as_list=True
            )
        if self.assistant_only:
            # Get the assistant mask
            mask = tensordict.get(("masks", "all_assistant_mask"))
            # mask will often be None - fall back on prompt / response separation
            if mask is None:
                if self.pad_output:
                    # simple case: just take the prompt length
                    prompt_length = tensordict.get(("tokens", "prompt")).shape[-1]
                    mask = tensordict.get(("masks", "all_attention_mask")).clone()
                    mask[..., :prompt_length] = False
                else:
                    # simple case: just take the prompt length
                    prompt_length = [
                        t.size(-1)
                        for t in tensordict.get(("tokens", "prompt"), as_list=True)
                    ]
                    mask = tensordict.get(("masks", "all_attention_mask"), as_list=True)
                    for i in range(len(prompt_length)):
                        mask[i] = mask[i].clone()
                        mask[i][..., : prompt_length[i]] = False

            # we want to keep the batch dimension
            ref_log_prob_list = []
            if self.pad_output:
                for i in range(ref_log_prob_padded.size(0)):
                    ref_log_prob_list.append(
                        ref_log_prob_padded[i].masked_fill(~mask[i], 0)
                    )
            else:
                for i in range(len(ref_log_prob_unpadded)):
                    ref_log_prob_list.append(
                        ref_log_prob_unpadded[i].masked_fill(~mask[i], 0)
                    )
            if self.pad_output:
                ref_log_prob = pad_sequence(
                    ref_log_prob_list,
                    batch_first=True,
                    padding_value=0,
                    padding_side=self.padding_side,
                )
            else:
                ref_log_prob = torch.nested.nested_tensor(
                    ref_log_prob_list, layout=torch.strided
                )

        # we obtain the current log-probs (already computed) from the current tensordict
        if self.pad_output:
            curr_log_prob_padded = tensordict.get(self.log_prob_full_key)
        else:
            curr_log_prob_unpadded = tensordict.get(
                self.log_prob_full_key, as_list=True
            )
        if self.assistant_only:
            # we want to keep the batch dimension
            curr_log_prob_list = []
            if self.pad_output:
                for i in range(curr_log_prob_padded.size(0)):
                    curr_log_prob_list.append(
                        curr_log_prob_padded[i].masked_fill(~mask[i], 0)
                    )
            else:
                for i in range(len(curr_log_prob_unpadded)):
                    curr_log_prob_list.append(
                        curr_log_prob_unpadded[i].masked_fill(~mask[i], 0)
                    )
            if self.pad_output:
                curr_log_prob = pad_sequence(
                    curr_log_prob_list,
                    batch_first=True,
                    padding_value=0,
                    padding_side=self.padding_side,
                )
            else:
                curr_log_prob = torch.nested.nested_tensor(
                    curr_log_prob_list, layout=torch.strided
                )

        ref_log_prob = ref_log_prob.to(curr_log_prob.device)
        # We want the log-probs to have a similar dim to the reward
        curr_log_prob = curr_log_prob.unsqueeze(-1)
        ref_log_prob = ref_log_prob.unsqueeze(-1)

        for i in range(ref_log_prob.size(0)):
            if ref_log_prob[i].shape != curr_log_prob[i].shape:
                # Don't check shapes if nested
                raise ValueError(
                    f"the log-probability tensor shapes must match, got cur_log_prob.shape={curr_log_prob[i].shape} and log_prob.shape={ref_log_prob[i].shape}. "
                    f"One possible reason is that the padding token is identical to the eos token, which means that the eos_token log_prob is truncated from the "
                    f"reference model output."
                )
        kl = curr_log_prob - ref_log_prob
        if self.add_to_reward:
            reward_key = self.in_keys[0]
            reward = next_tensordict.get(reward_key)
            # we use the unbiased consistent estimator of the KL: log_p(x) - log_q(x) when x ~ p(x)
            if not reward.is_nested and ref_log_prob.is_nested:
                reward = torch.nested.nested_tensor(
                    [rew.expand(lp.shape) for rew, lp in zip(reward, ref_log_prob)],
                    layout=torch.strided,
                )
            if reward is not None and reward.ndim != curr_log_prob.ndim:
                raise ValueError(
                    "The number of dimensions of reward must be the same as the number of dimensions of the KL "
                    f"term. Got ndim={reward.ndim} and {curr_log_prob.ndim} respectively."
                )
            if reward is None:
                reward = 0
            reward = reward - self.coef * kl
            next_tensordict.set(self.out_keys[0], reward)
        next_tensordict.set(self.out_keys[1], kl)
        next_tensordict.set(self.out_keys[2], ref_log_prob)
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = tensordict.pop("next")
        next_td = self._step(tensordict, next_td)
        return tensordict.set("next", next_td)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        in_key = unravel_key(self.in_keys[0])
        out_key = unravel_key(self.out_keys[0])

        if "full_observation_spec" in output_spec.keys():
            observation_spec = output_spec["full_observation_spec"]
        else:
            observation_spec = Composite(
                shape=output_spec.shape, device=output_spec.device
            )
            output_spec["full_observation_spec"] = observation_spec

        if in_key == "reward" and out_key == "reward":
            parent = self.parent

            reward_keys = parent.reward_keys
            if len(reward_keys) == 1:
                reward_key = reward_keys[0]
                shape = output_spec["full_reward_spec"].shape
            elif "reward" in reward_keys:
                reward_key = "reward"
                shape = output_spec["full_reward_spec"].shape
            else:
                shape = output_spec.shape
                reward_key = "reward"
            # For LLMs, the shape of the reward is (batch, -1, 1)
            shape = (*shape, -1, 1)
            reward_spec = Unbounded(
                device=output_spec.device,
                shape=shape,
            )
            output_spec["full_reward_spec"] = Composite(
                {reward_key: reward_spec},
                shape=output_spec["full_reward_spec"].shape,
            )
        elif in_key == "reward":
            # TODO: we should at least allow to make this a component of the reward specs, to avoid a call during reset
            parent = self.parent
            reward_spec = output_spec["full_reward_spec"][parent.reward_key]

            shape = output_spec["full_reward_spec"].shape
            # For LLMs, the shape of the reward is (batch, -1, 1)
            shape = (*shape, -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["full_observation_spec"]
            reward_spec = observation_spec[in_key]

            shape = observation_spec.shape
            shape = (*shape, -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec

        observation_spec[self.out_keys[1]] = reward_spec.clone()

        return output_spec


class RetrieveLogProb(Transform):
    """A transform to retrieve log-probabilities from a model for KL divergence computation.

    This transform computes log-probabilities from a reference model, which can then be used
    to compute KL divergence with another model's log-probabilities. It's designed to work
    with the :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL` and :class:`~torchrl.envs.llm.transforms.kl.KLComputation` transforms.

    Args:
        model (LLMWrapperBase): the model to use to compute the log-probs.

    Keyword Args:
        log_probs_full_key (NestedKey): the key where the log-probs are stored.
            If not provided, the key will be retrieved from the model's `log_probs_key` attribute
            (i.e., `(model.log_probs_key, "full")`).
        assistant_only (bool): whether to zero out the log-probs of the non-assistant tokens (i.e., steps of history
            where the role is not `"assistant"`). Defaults to `True`.

            .. note:: When `assistant_only=True`, the model must have `input_mode='history'` to properly identify
                assistant tokens. For other input modes (`"text"` or `"tokens"`), set `assistant_only=False`.
                This ensures users are conscious of the limitation that assistant token identification requires
                structured conversation history.

        tokenizer_kwargs (dict): the keyword arguments to pass to the tokenizer to be used to apply the chat template to the history when `assistant_only` is `True`.
            To control the tokenization in the ref_model, pass the tokenizer kwargs to the ref_model constructor.
            Defaults to `{"return_assistant_tokens_mask": True, "tokenize": True, "return_dict": True, "padding": False, "add_generation_prompt": False}`.
        tokenizer (transformers.AutoTokenizer): the tokenizer to be used to tokenize the input and compute the assitant mask. If not provided, the tokenizer will be inferred from the `ref_model`.
        detach (bool): whether to exclude the log-probs from the gradient computation. Defaults to `True`.
        device (torch.device): the device to use for tensor creation. Defaults to `None`.
        padding_side (str): the side of the padding when using pad_sequence. Defaults to `"left"`.

    Examples:
        >>> from torchrl.data.llm import History
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.modules.llm.policies import ChatHistory
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> import torch
        >>>
        >>> # Set up list to stack for History
        >>> set_list_to_stack(True).set()
        >>>
        >>> # Create chat data
        >>> chats = [
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Hello, how are you?"},
        ...         {"role": "assistant", "content": "I'm doing well, thank you!"},
        ...     ],
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What's the weather like?"},
        ...         {"role": "assistant", "content": "I can't check the weather for you."},
        ...     ],
        ... ]
        >>> history = History.from_chats(chats)
        >>> print(f"Created history with shape: {history.shape}")
        Created history with shape: torch.Size([2, 3])
        >>>
        >>> # Setup tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> model = OPTForCausalLM(OPTConfig()).eval()
        >>>
        >>> # Create reference model
        >>> ref_model = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     input_mode="history",
        ...     generate=False,
        ...     return_log_probs=True,
        ...     pad_output=True,
        ... )
        >>>
        >>> # Create the RetrieveLogProb transform
        >>> transform = RetrieveLogProb(
        ...     ref_model,
        ...     assistant_only=True,
        ...     tokenizer=tokenizer,
        ... )
        >>>
        >>> # Prepare data using ChatHistory
        >>> chat_history = ChatHistory(full=history)
        >>> data = TensorDict(history=chat_history, batch_size=(2,))
        >>>
        >>> # Apply the transform to get reference log probabilities
        >>> result = transform(data)
        >>> log_probs_key = (ref_model.log_probs_key, "full")
        >>> ref_log_probs = result.get(log_probs_key)
        >>> print(f"Log-probs shape: {ref_log_probs.shape}")
        Log-probs shape: torch.Size([2, 26])

    .. note::
        By default, the log-probabilities are stored as a list of tensors (one per sample, with variable length).
        Use `as_padded_tensor=True` in `.get()` to obtain a batchable tensor (with padding).
        The reference log probabilities are computed only for assistant tokens when `assistant_only=True`.

        **Input Mode Compatibility:**
        - When `assistant_only=True` (default), the model must have `input_mode='history'` to properly identify assistant tokens.
        - When `assistant_only=False`, the transform works with any input mode (`"history"`, `"text"`, or `"tokens"`).
        - This design ensures users are conscious of the limitation that assistant token identification requires structured conversation history.

    .. seealso::
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL`: A higher-level transform that combines two `RetrieveLogProb` instances with `KLComputation`.
        :class:`~torchrl.envs.llm.transforms.kl.KLComputation`: A transform that computes KL divergence between two log-prob tensors.
        :class:`~torchrl.envs.llm.transforms.kl.KLRewardTransform`: A legacy transform for KL reward computation (use `RetrieveKL` instead).
    """

    def __init__(
        self,
        model: LLMWrapperBase,
        *,
        log_probs_full_key: NestedKey | None = None,
        assistant_only: bool = True,
        tokenizer_kwargs: dict | None = None,
        detach: bool = True,
        device: torch.device | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,
        padding_side: str = "left",
    ):
        # Set up keys
        if log_probs_full_key is None:
            log_probs_full_key = (model.log_probs_key, "full")
        elif (
            not isinstance(log_probs_full_key, tuple)
            or log_probs_full_key[-1] != "full"
        ):
            warnings.warn(
                f"The log_probs_full_key {log_probs_full_key} is not a tuple or does not end with 'full'. "
                "This may cause issues with the KL computation. "
                "Please use a tuple with the log_probs_key and 'full' as the last element."
            )
        self.log_probs_full_key = log_probs_full_key

        # Set up input/output keys
        in_keys = list(model.in_keys)
        out_keys = [self.log_probs_full_key]
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        # Store model and configuration
        self.model = model
        self.assistant_only = assistant_only
        self.detach = detach
        self.device = device
        self.tokenizer = tokenizer
        self.padding_side = padding_side

        # Set up tokenizer kwargs
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("return_dict", True)
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("add_generation_prompt", False)
        self.tokenizer_kwargs = tokenizer_kwargs

        # Validate model configuration (after setting assistant_only)
        self._validate_model_config(model)

    def _validate_model_config(self, model: LLMWrapperBase):
        """Validate model configuration."""
        if not getattr(model, "return_log_probs", True):
            raise ValueError(
                "The model must have `return_log_probs=True` to use the `RetrieveLogProb` transform."
            )
        if getattr(model, "generate", True):
            raise ValueError(
                "The model must have `generate=False` to use the `RetrieveLogProb` transform."
            )

        # Check input mode compatibility with assistant_only
        input_mode = getattr(model, "input_mode", "history")
        if self.assistant_only and input_mode != "history":
            raise ValueError(
                f"The model must have `input_mode='history'` when `assistant_only=True`. "
                f"Current input_mode is '{input_mode}'. "
                f"To use input_mode '{input_mode}', set `assistant_only=False`."
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = tensordict.get("next")
        next_is_none = False
        if next_td is None:
            next_is_none = True
            next_td = tensordict
        output = self._step(tensordict, next_td)
        if next_is_none:
            return output
        return tensordict.set("next", output)

    def _mask_assistant_tokens(
        self, td: TensorDictBase, lp_key: NestedKey
    ) -> torch.Tensor:
        """Mask log-probs to only include assistant tokens.

        Args:
            td: TensorDict containing the data
            lp_key: Key for log-probs in the TensorDict

        Returns:
            Masked log-probs tensor
        """
        with torch.device(self.device) if self.device is not None else nullcontext():
            # Get assistant mask
            assistant_masks = td.get(("masks", "all_assistant_mask"), as_list=True)
            log_probs = td.get(lp_key, as_list=True)
            log_probs = [
                torch.masked_fill(lp, ~mask, 0.0)
                for lp, mask in _zip_strict(log_probs, assistant_masks)
            ]
            if self.model.pad_output:
                log_probs = pad_sequence(
                    log_probs,
                    batch_first=True,
                    padding_value=0.0,
                    padding_side=self.padding_side,
                )
            else:
                log_probs = torch.nested.as_nested_tensor(
                    log_probs, layout=self.model.layout
                )
            return log_probs

    @set_list_to_stack(True)
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # Compute log-probs using the model
        # Use tensordict since we want to process the "full" entry
        ref_td = self.model(tensordict.copy())
        tmp_log_probs_key = (self.model.log_probs_key, "full")

        # Apply assistant masking if requested
        if self.assistant_only:
            log_probs = self._mask_assistant_tokens(ref_td, tmp_log_probs_key)
            ref_td.set(tmp_log_probs_key, log_probs)

        # Rename and store the log-probs
        if tmp_log_probs_key != self.log_probs_full_key:
            ref_td.rename_key_(tmp_log_probs_key, self.log_probs_full_key)
        next_tensordict.update(ref_td, keys_to_update=(self.log_probs_full_key,))

        return next_tensordict

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        # Add kl to observation spec
        observation_spec["kl_penalty"] = Unbounded(
            device=observation_spec.device,
            shape=observation_spec.shape,
        )
        return observation_spec


class RetrieveKL(Compose):
    """A transform to retrieve the KL divergence between two models' log-probabilities.

    This transform combines two :class:`~torchrl.envs.llm.transforms.kl.RetrieveLogProb` instances
    with a :class:`~torchrl.envs.llm.transforms.kl.KLComputation` to compute KL divergence
    between a generation model and a reference model.

    .. note::
        Both gen_model and ref_model must use the same pad_output value (True or False), otherwise KL computation will fail.

    Args:
        gen_model (LLMWrapperBase): the generation model, wrapped in such a way that it does not generate but computes the log-probs.
            In cases where the transform is used within a :class:`~torchrl.collectors.llm.LLMCollector` run on a remote worker, the
            policy may not be available ahead of time. In this case, the `gen_model` can be set to `"from_collector"` (default) to retrieve the
            policy from the collector. See :meth:`~torchrl.modules.llm.policies.LLMWrapperBase.get_new_version` for more details
            about generating a new version of the policy to gather the log-probs.
        ref_model (LLMWrapperBase): the reference model, wrapped in such a way that it does not generate but computes the log-probs.

    Keyword Args:
        assistant_only (bool): whether to only retrieve the log-probs of the assistant tokens (i.e., steps of history
            where the role is `"assistant"`). Defaults to `True`.

            .. note:: When `assistant_only=True`, both models must have `input_mode='history'` to properly identify assistant tokens.
                For other input modes (`"text"` or `"tokens"`), set `assistant_only=False`.
                This ensures users are conscious of the limitation that assistant token identification requires structured conversation history.

        gen_log_probs_full_key (str): the key where the log-probs of the generation model are stored. Defaults to `("log_probs", "full")`.
        ref_log_probs_full_key (str): the key where the log-probs of the reference model are stored. Defaults to `("ref_log_probs", "full")`.
        history_key (str): the key where the history is stored. Defaults to `"history"`.
        tokenizer_kwargs (dict): the keyword arguments to pass to the tokenizer to be used to apply the chat template to the history when `assistant_only` is `True`.
            To control the tokenization in the actor, pass the tokenizer kwargs to the actor constructor.
            Defaults to `{"return_assistant_tokens_mask": True, "tokenize": True, "return_tensors": "pt", "padding": True, "add_generation_prompt": False}`.
        detach (bool): whether to exclude the log-probs from the gradient computation. Defaults to `True`.
        device (torch.device): the device to use for tensor creation. Defaults to `None`.
        tokenizer (transformers.AutoTokenizer): the tokenizer to be used to tokenize the input and compute the assitant mask. If not provided, the tokenizer will be inferred from the `actor`.
        padding_side (str): the side of the padding when using pad_sequence. Defaults to `"left"`.
        kl_key (NestedKey): the key where the KL divergence is stored. Defaults to `"kl_penalty"`.
        add_to_reward (bool): whether to add the KL divergence to the reward. Defaults to `True`.
        coeff (float): the coefficient for the KL term when adding to reward. Defaults to `1.0`.
        padding_side (str): the side of the padding when using pad_sequence. Defaults to `"left"`.
        **kwargs: additional arguments to pass to the `RetrieveLogProb` transform.

    Examples:
        >>> from torchrl.data.llm import History
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.modules.llm.policies import ChatHistory
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> import torch
        >>>
        >>> # Set up list to stack for History
        >>> set_list_to_stack(True).set()
        >>>
        >>> # Create chat data
        >>> chats = [
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "Hello, how are you?"},
        ...         {"role": "assistant", "content": "I'm doing well, thank you!"},
        ...     ],
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What's the weather like?"},
        ...         {"role": "assistant", "content": "I can't check the weather for you."},
        ...     ],
        ... ]
        >>> history = History.from_chats(chats)
        >>> print(f"Created history with shape: {history.shape}")
        Created history with shape: torch.Size([2, 3])
        >>>
        >>> # Setup tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> model = OPTForCausalLM(OPTConfig()).eval()
        >>>
        >>> # Create generation and reference models
        >>> gen_model = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     input_mode="history",
        ...     generate=False,
        ...     return_log_probs=True,
        ...     pad_output=True,
        ...     log_probs_key="gen_log_probs",
        ... )
        >>> ref_model = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     input_mode="history",
        ...     generate=False,
        ...     return_log_probs=True,
        ...     pad_output=True,
        ...     log_probs_key="ref_log_probs",
        ... )
        >>>
        >>> # Create RetrieveKL transform
        >>> transform = RetrieveKL(
        ...     gen_model=gen_model,
        ...     ref_model=ref_model,
        ...     assistant_only=True,
        ...     tokenizer=tokenizer,
        ... )
        >>>
        >>> # Prepare data with next tensordict using ChatHistory
        >>> chat_history = ChatHistory(full=history)
        >>> next_td = TensorDict(history=chat_history, batch_size=(2,))
        >>> data = TensorDict(history=chat_history, next=next_td, batch_size=(2,))
        >>>
        >>> # Apply transform
        >>> result = transform(data)
        >>> kl = result["next"].get("kl_penalty")
        >>> print(f"KL shape: {kl.shape}")
        KL shape: torch.Size([2, 26])

    Note:
        **Input Mode Compatibility:**
        - When `assistant_only=True`, both models must have `input_mode='history'` to properly identify assistant tokens.
        - When `assistant_only=False`, the transform works with any input mode (`"history"`, `"text"`, or `"tokens"`).
        - This design ensures users are conscious of the limitation that assistant token identification requires structured conversation history.

    .. seealso::
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveLogProb`: The base transform for retrieving log-probabilities from a single model.
        :class:`~torchrl.envs.llm.transforms.kl.KLComputation`: The transform that computes KL divergence between two log-prob tensors.
        :class:`~torchrl.envs.llm.transforms.kl.KLRewardTransform`: A legacy transform for KL reward computation (use `RetrieveKL` instead).
    """

    def __init__(
        self,
        gen_model: LLMWrapperBase | Literal["from_collector"] = "from_collector",
        ref_model: LLMWrapperBase | None = None,
        *,
        assistant_only: bool | None = True,
        history_key: str = "history",
        tokenizer_kwargs: dict[str, Any] | None = None,
        detach: bool = True,
        device: torch.device | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,
        padding_side: str = "left",
        gen_log_probs_full_key: NestedKey = ("log_probs", "full"),
        ref_log_probs_full_key: NestedKey = ("ref_log_probs", "full"),
        kl_key: NestedKey = "kl_penalty",
        add_to_reward: bool = True,
        coeff: float = 1.0,
        **kwargs,
    ):
        if isinstance(gen_model, str) and gen_model == "from_collector":
            # Lazy init
            self._initialized = False
            self._init_params = {
                "ref_model": ref_model,
                "assistant_only": assistant_only,
                "history_key": history_key,
                "tokenizer_kwargs": tokenizer_kwargs,
                "detach": detach,
                "device": device,
                "tokenizer": tokenizer,
                "gen_log_probs_full_key": gen_log_probs_full_key,
                "ref_log_probs_full_key": ref_log_probs_full_key,
                "kl_key": kl_key,
                "add_to_reward": add_to_reward,
                "coeff": coeff,
                "padding_side": padding_side,
                **kwargs,
            }
            super().__init__()
            return

        self._initialized = True

        # Check pad_output consistency if both models are provided
        if hasattr(gen_model, "pad_output") and hasattr(ref_model, "pad_output"):
            if gen_model.pad_output != ref_model.pad_output:
                raise ValueError(
                    f"pad_output mismatch: gen_model.pad_output={gen_model.pad_output}, "
                    f"ref_model.pad_output={ref_model.pad_output}. "
                    "Both models must use the same padding strategy for KL computation."
                )

        if not getattr(gen_model, "return_log_probs", True):
            raise ValueError(
                "The generation model must have `return_log_probs=True` to use the `RetrieveKL` transform."
            )
        elif getattr(gen_model, "generate", False):
            raise ValueError(
                "The generation model must have `generate=False` to use the `RetrieveKL` transform."
            )

        if not getattr(ref_model, "return_log_probs", True):
            raise ValueError(
                "The reference model must have `return_log_probs=True` to use the `RetrieveKL` transform."
            )
        elif getattr(ref_model, "generate", False):
            raise ValueError(
                "The reference model must have `generate=False` to use the `RetrieveKL` transform."
            )
        if getattr(gen_model, "log_probs_key", "gen_log_probs") == getattr(
            ref_model, "log_probs_key", "log_probs"
        ):
            raise ValueError(
                "The generation and reference models must have different `log_prob_key` values to use the `RetrieveKL` transform."
            )
        t1 = RetrieveLogProb(
            gen_model,
            log_probs_full_key=gen_log_probs_full_key,
            assistant_only=assistant_only,
            tokenizer_kwargs=tokenizer_kwargs,
            detach=detach,
            device=device,
            tokenizer=tokenizer,
            padding_side=padding_side,
            **kwargs,
        )
        t2 = RetrieveLogProb(
            ref_model,
            log_probs_full_key=ref_log_probs_full_key,
            assistant_only=assistant_only,
            tokenizer_kwargs=tokenizer_kwargs,
            detach=detach,
            device=device,
            tokenizer=tokenizer,
            padding_side=padding_side,
            **kwargs,
        )
        t3 = KLComputation(
            gen_log_probs_full_key=gen_log_probs_full_key,
            ref_log_probs_full_key=ref_log_probs_full_key,
            kl_key=kl_key,
            add_to_reward=add_to_reward,
            coeff=coeff,
        )
        super().__init__(t1, t2, t3)

    def _init_deferred(self):
        torchrl_logger.info("Initializing RetrieveKL transform")
        container = self.container
        if container is None:
            # also logging, since this will be sometimes hidden within the AttributeError
            torchrl_logger.warning(
                "The container is not set. Please set the container before calling this method."
            )
            raise ValueError(
                "The container is not set. Please set the container before calling this method."
            )
        container.empty_cache()
        self.empty_cache()
        collector = self.collector
        if collector is None:
            # also logging, since this will be sometimes hidden within the AttributeError
            torchrl_logger.warning(
                "The collector is not set. Please set the collector before calling this method."
            )
            raise ValueError(
                "The collector is not set. Please set the collector before calling this method."
            )
        ref_model = self._init_params["ref_model"]
        pad_output = getattr(ref_model, "pad_output", None)
        gen_log_probs_full_key = self._init_params["gen_log_probs_full_key"]
        if (
            not isinstance(gen_log_probs_full_key, tuple)
            or gen_log_probs_full_key[-1] != "full"
        ):
            raise ValueError(
                f"The gen_log_probs_full_key {gen_log_probs_full_key} is not a tuple or does not end with 'full'. "
                "This may cause issues with the KL computation. "
                "Please use a tuple with the log_probs_key and 'full' as the last element."
            )
        log_probs_key = gen_log_probs_full_key[:-1]
        gen_model = collector.policy.get_new_version(
            generate=False,
            return_log_probs=True,
            log_probs_key=log_probs_key,
            input_mode=ref_model.input_mode,
            input_key=(ref_model.input_mode, "full"),
            pad_output=pad_output,  # Pass pad_output from ref_model
        )
        # Create the transforms manually instead of calling __init__
        t1 = RetrieveLogProb(
            gen_model,
            log_probs_full_key=gen_log_probs_full_key,
            assistant_only=self._init_params["assistant_only"],
            tokenizer_kwargs=self._init_params["tokenizer_kwargs"],
            detach=self._init_params["detach"],
            device=self._init_params["device"],
            tokenizer=self._init_params["tokenizer"],
            padding_side=self._init_params["padding_side"],
        )
        ref_log_probs_full_key = self._init_params["ref_log_probs_full_key"]
        if (
            not isinstance(ref_log_probs_full_key, tuple)
            or ref_log_probs_full_key[-1] != "full"
        ):
            raise ValueError(
                f"The ref_log_probs_full_key {ref_log_probs_full_key} is not a tuple or does not end with 'full'. "
                "This may cause issues with the KL computation. "
                "Please use a tuple with the log_probs_key and 'full' as the last element."
            )
        t2 = RetrieveLogProb(
            ref_model,
            log_probs_full_key=ref_log_probs_full_key,
            assistant_only=self._init_params["assistant_only"],
            tokenizer_kwargs=self._init_params["tokenizer_kwargs"],
            detach=self._init_params["detach"],
            device=self._init_params["device"],
            tokenizer=self._init_params["tokenizer"],
            padding_side=self._init_params["padding_side"],
        )
        t3 = KLComputation(
            gen_log_probs_full_key=gen_log_probs_full_key,
            ref_log_probs_full_key=ref_log_probs_full_key,
            kl_key=self._init_params["kl_key"],
            add_to_reward=self._init_params["add_to_reward"],
            coeff=self._init_params["coeff"],
        )
        # Replace the transforms in the Compose
        self.transforms.extend([t1, t2, t3])
        del self._init_params
        self._initialized = True
        torchrl_logger.info("Successfully initialized")

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            self._init_deferred()
        return super()._step(tensordict, next_tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            self._init_deferred()
        return super()._reset(tensordict, tensordict_reset)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self._init_deferred()
        return super().forward(tensordict)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_observation_spec(observation_spec)

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_reward_spec(reward_spec)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self._init_deferred()
        return super()._inv_call(tensordict)

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_action_spec(action_spec)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_input_spec(input_spec)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_output_spec(output_spec)

    def transform_state_spec(self, state_spec: Composite) -> Composite:
        if not self._initialized:
            self._init_deferred()
        return super().transform_state_spec(state_spec)


class KLComputation(Transform):
    """A transform to compute KL divergence between two log-prob tensors and optionally add it to the reward.

    This transform computes KL divergence between generation and reference log-probabilities
    and can optionally subtract it from the reward (for KL penalty). It's designed to work
    with the :class:`~torchrl.envs.llm.transforms.kl.RetrieveLogProb` and :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL` transforms.

    .. note::
        Both input log-prob tensors must use the same padding strategy (pad_output) for correct KL computation.

    Args:
        gen_log_probs_full_key (NestedKey): the key where the generation model log-probs are stored.
            Defaults to `("gen_log_probs", "full")`.
        ref_log_probs_full_key (NestedKey): the key where the reference model log-probs are stored.
            Defaults to `("ref_log_probs", "full")`.
        kl_key (NestedKey): the key where the KL divergence is stored. Defaults to `"kl_penalty"`.
        add_to_reward (bool): whether to add the KL divergence to the reward. Defaults to `True`.
        coeff (float): the coefficient for the KL term when adding to reward. Defaults to `1.0`.
        padding_side (str): the side of the padding when using pad_sequence. Defaults to `"left"`.

    Examples:
        >>> from tensordict import TensorDict
        >>> import torch
        >>>
        >>> # Create sample log-probs
        >>> gen_log_probs = torch.randn(2, 10)  # 2 samples, 10 tokens each
        >>> ref_log_probs = torch.randn(2, 10)
        >>>
        >>> # Create data with next tensordict
        >>> next_td = TensorDict(
        ...     {
        ...         ("gen_log_probs", "full"): gen_log_probs,
        ...         ("ref_log_probs", "full"): ref_log_probs,
        ...         "reward": torch.randn(2, 10, 1),
        ...     },
        ...     batch_size=(2,)
        ... )
        >>> data = TensorDict(next=next_td, batch_size=(2,))
        >>>
        >>> # Create KLComputation transform
        >>> kl_transform = KLComputation(
        ...     gen_log_probs_key=("gen_log_probs", "full"),
        ...     ref_log_probs_key=("ref_log_probs", "full"),
        ...     kl_key="kl_penalty",
        ...     add_to_reward=True,
        ...     coef=1.0,
        ... )
        >>>
        >>> # Apply transform
        >>> result = kl_transform(data)
        >>> kl = result["next"].get("kl_penalty")
        >>> print(f"KL shape: {kl.shape}")
        KL shape: torch.Size([2, 10])

    .. seealso::
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveLogProb`: The base transform for retrieving log-probabilities from a single model.
        :class:`~torchrl.envs.llm.transforms.kl.RetrieveKL`: A higher-level transform that combines two `RetrieveLogProb` instances with `KLComputation`.
        :class:`~torchrl.envs.llm.transforms.kl.KLRewardTransform`: A legacy transform for KL reward computation (use `RetrieveKL` instead).

    """

    def __init__(
        self,
        gen_log_probs_full_key: NestedKey = ("log_probs", "full"),
        ref_log_probs_full_key: NestedKey = ("ref_log_probs", "full"),
        *,
        kl_key: NestedKey = "kl_penalty",
        add_to_reward: bool = True,
        coeff: float = 1.0,
        padding_side: str = "left",
    ):
        in_keys = [gen_log_probs_full_key, ref_log_probs_full_key]
        if add_to_reward:
            in_keys.append("reward")
        out_keys = [kl_key]
        if add_to_reward:
            out_keys.append("reward")
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.gen_log_probs_full_key = gen_log_probs_full_key
        self.ref_log_probs_full_key = ref_log_probs_full_key
        self.kl_key = kl_key
        self.add_to_reward = add_to_reward
        self.coeff = coeff
        self.padding_side = padding_side

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = tensordict.get("next")
        has_next_td = True
        if next_td is None:
            next_td = tensordict
            has_next_td = False
        next_td = self._step(tensordict, next_td)
        if has_next_td:
            return tensordict.set("next", next_td)
        return next_td

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # Get log-probs
        gen_log_probs = next_tensordict.get(self.gen_log_probs_full_key, as_list=True)
        ref_log_probs = next_tensordict.get(self.ref_log_probs_full_key, as_list=True)

        if gen_log_probs is None or ref_log_probs is None:
            raise ValueError(
                f"Log-probs not found. Expected keys: {self.gen_log_probs_key}, {self.ref_log_probs_key}"
            )

        # Debug: Check lengths and shapes
        if len(gen_log_probs) != len(ref_log_probs):
            raise ValueError(
                f"Batch size mismatch: gen_log_probs has {len(gen_log_probs)} samples, ref_log_probs has {len(ref_log_probs)} samples"
            )

        # Check individual sequence lengths
        for i, (gen_lp, ref_lp) in enumerate(_zip_strict(gen_log_probs, ref_log_probs)):
            if gen_lp.shape != ref_lp.shape:
                raise ValueError(
                    f"Sample {i} has different shapes: gen_log_probs[{i}].shape={gen_lp.shape}, ref_log_probs[{i}].shape={ref_lp.shape}"
                )

        # Compute KL divergence: KL(p||q) = E_p[log p - log q]
        # Here gen_log_probs = log p, ref_log_probs = log q
        kl = [
            gen_lp - ref_lp
            for gen_lp, ref_lp in _zip_strict(gen_log_probs, ref_log_probs)
        ]

        kl = torch.nested.as_nested_tensor(kl, layout=torch.strided)

        next_tensordict.set(self.kl_key, kl)

        # Add to reward if requested
        if self.add_to_reward:
            reward = next_tensordict.get("reward", as_list=True)
            if reward is not None:
                if isinstance(reward, list):
                    if reward[0].ndim != kl[0].ndim + 1:
                        raise ValueError(
                            f"The rewards have shape {reward[0].shape} but the kl has shape {kl[0].shape}. "
                            f"The rewards should have one more dimension than the KL."
                        )
                    reward = [
                        r - self.coeff * k.unsqueeze(-1)
                        for r, k in _zip_strict(reward, kl)
                    ]
                    next_tensordict.set(
                        "reward",
                        torch.nested.as_nested_tensor(reward, layout=torch.strided),
                    )
                else:
                    if reward.ndim != kl.ndim + 1:
                        raise ValueError(
                            f"The rewards have shape {reward.shape} but the kl has shape {kl.shape}. "
                            f"The rewards should have one more dimension than the KL."
                        )
                    reward = reward - self.coeff * kl.unsqueeze(-1)
                    next_tensordict.set("reward", reward)

        return next_tensordict

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        # Add kl to observation spec
        observation_spec[self.kl_key] = Unbounded(
            device=observation_spec.device,
            shape=observation_spec.shape,
        )
        return observation_spec

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        # Optionally adjust reward spec if KL is added to reward
        if self.add_to_reward:
            shape = reward_spec["reward"].shape
            # For LLMs, the shape of the reward is (batch, -1, 1)
            shape = (*shape, -1, 1)
            reward_spec["reward"] = reward_spec["reward"].clone()
            reward_spec["reward"].shape = torch.Size(shape)
        return reward_spec
