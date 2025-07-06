# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import gc

from copy import copy

import torch
from tensordict import NestedKey, set_list_to_stack, TensorDictBase, unravel_key
from tensordict.nn import ProbabilisticTensorDictModule
from tensordict.utils import _zip_strict, is_seq_of_nested_key
from torchrl.data import Composite, Unbounded
from torchrl.data.llm.chat import History
from torchrl.envs import EnvBase, Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.modules.llm.policies.common import CategoricalSequential

try:
    import transformers
except ImportError:
    transformers = None


class KLRewardTransform(Transform):
    """A transform to add a KL[pi_current||pi_0] correction term to the reward.

    This transform is used to constrain the policy to remain close to its original
    configuration which limits overfitting when fine-tuning using RLHF.

    Args:
        actor (ProbabilisticTensorDictModule): a frozen probabilistic actor. It must
            have the following features: it must have a set of input (``in_keys``)
            and output keys (``out_keys``). It must have a ``get_dist`` method
            that outputs the distribution of the action.
        coef (:obj:`float`): the coefficient of the KL term. Defaults to ``1.0``.
        in_keys (str or list of str/tuples of str): the input key where the
            reward should be fetched. Defaults to ``"reward"``.
        out_keys (str or list of str/tuples of str): the output key where the
            reward should be written. Defaults to ``["reward", "kl_penalty", "ref_log_prob"]``.
        add_to_reward (bool): whether to add the reward term to the reward.
            Defaults to ``True``.

    .. note:: If the parameters are not differentiable (default), they will *not*
        follow the module when dtype or device casting operations will be called
        (such as :meth:`cuda`, :meth:`to` etc.). When ``requires_grad=True``,
        casting operations will work as expected.

    Examples:
        TODO

    .. note:: Because the KL formula is not always available and the parameters of the
      original distribution may not have been recorded, we use a stochastic estimate
      of the KL divergence.

    """

    DEFAULT_IN_KEYS = ["reward"]

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        coef=1.0,
        in_keys=None,
        out_keys=None,
        log_prob_key: NestedKey = "log_probs",
        action_key: NestedKey | None = None,
        device: torch.device | None = None,
        add_to_reward: bool = True,
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

        # update the in_keys for dispatch etc
        self.in_keys = self.in_keys + actor.in_keys
        self.in_keys = [unravel_key(in_key) for in_key in self.in_keys]

        self.add_to_reward = add_to_reward
        # check that the model has parameters
        self.__dict__["actor"] = actor

        # self._buffers["actor_params"] = params.clone().detach()

        self.device = device
        self.action_key = action_key

        # find the sample log-prob key
        self.sample_log_prob_key = log_prob_key

        def find_sample_log_prob(module):
            if hasattr(module, "log_prob_key"):
                self.sample_log_prob_key = module.log_prob_key

        self.actor.apply(find_sample_log_prob)

        if not isinstance(coef, torch.Tensor):
            coef = torch.as_tensor(coef)
        self.register_buffer("coef", coef)

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

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # run the actor on the tensordict
        action_key = self.action_key
        if action_key is None:
            raise ValueError(
                f"action_key is required. Please set a parent for the {type(self).__name__} to recover the action keys automatically, "
                f"or pass the action_key argument directly to {type(self).__name__} constructor."
            )
        response_txt = tensordict.get(action_key, None)
        if response_txt is None:
            if not self.missing_tolerance:
                raise RuntimeError(
                    f"Action with key {action_key} not found data {tensordict}"
                )
            # being called after reset or without action, skipping
            if self.out_keys[0] != "reward" and self.parent is not None:
                next_tensordict.set(self.out_keys[0], self.parent.reward_spec.zero())
            return next_tensordict
        if hasattr(self.actor, "log_prob"):
            if self.device is not None and tensordict.device != self.device:
                td_device = tensordict.to(self.device)
            else:
                td_device = tensordict.copy()
            ref_log_prob = self.actor.log_prob(
                td_device, as_nested_tensor=True, layout=torch.strided
            )
        else:
            ref_log_prob_td = self.actor(tensordict)
            ref_log_prob = ref_log_prob_td.get(self.sample_log_prob_key)

        reward_key = self.in_keys[0]
        reward = next_tensordict.get(reward_key)
        curr_log_prob = tensordict.get(
            self.sample_log_prob_key, as_nested_tensor=True, layout=torch.strided
        )
        ref_log_prob = ref_log_prob.to(curr_log_prob.device)
        # We want the log-probs to have a similar dim to the reward
        curr_log_prob = curr_log_prob.unsqueeze(-1)
        ref_log_prob = ref_log_prob.unsqueeze(-1)

        # we use the unbiased consistent estimator of the KL: log_p(x) - log_q(x) when x ~ p(x)
        if not reward.is_nested and ref_log_prob.is_nested:
            reward = torch.nested.nested_tensor(
                [rew.expand(lp.shape) for rew, lp in zip(reward, ref_log_prob)],
                layout=torch.strided,
            )
        for i in range(ref_log_prob.size(0)):
            if ref_log_prob[i].shape != curr_log_prob[i].shape:
                # Don't check shapes if nested
                raise ValueError(
                    f"the log-probability tensor shapes must match, got cur_log_prob.shape={curr_log_prob[i].shape} and log_prob.shape={ref_log_prob[i].shape}. "
                    f"One possible reason is that the padding token is identical to the eos token, which means that the eos_token log_prob is truncated from the "
                    f"reference model output."
                )
        if reward is not None and reward.ndim != curr_log_prob.ndim:
            raise ValueError(
                "The number of dimensions of reward must be the same as the number of dimensions of the KL "
                f"term. Got ndim={reward.ndim} and {curr_log_prob.ndim} respectively."
            )
        kl = curr_log_prob - ref_log_prob
        if self.add_to_reward:
            if reward is None:
                reward = 0
            next_tensordict.set(self.out_keys[0], reward - self.coef * kl)
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
    """A transform to retrieve the log-probs of a text given a reference model.

    Args:
        actor (CategoricalSequential): the reference model.

    Keyword Args:
        history_key (NestedKey): the key where the history is stored. Defaults to `"history"`.
        log_prob_key (NestedKey): the key where the log-probs are stored. Defaults to `"ref_log_prob"`.
        assistant_only (bool): whether to only retrieve the log-probs of the assistant tokens (i.e., steps of history
            where the role is `"assistant"`). Defaults to `False`.

            .. note:: The template must accommodate the `return_assistant_tokens_mask` keyword argument.
                This may not be the case for all templates. In this case, you can pass a custom template to the `apply_chat_template` method
                via the `tokenizer_kwargs` argument: `tokenizer_kwargs = {"chat_template_name": "qwen"}` or `tokenizer_kwargs = {"chat_template": my_template}.

        tokenizer_kwargs (dict): the keyword arguments to pass to the tokenizer to be used to apply the chat template to the history when `assistant_only` is `True`.
            To control the tokenization in the actor, pass the tokenizer kwargs to the actor constructor.
            Defaults to `{"return_assistant_tokens_mask": True, "tokenize": True, "return_tensors": "pt", "padding": True, "add_generation_prompt": False}`.
        tokenizer (transformers.AutoTokenizer): the tokenizer to be used to tokenize the input and compute the assitant mask. If not provided, the tokenizer will be inferred from the `actor`.
        detach (bool): whether to exclude the log-probs from the gradient computation. Defaults to `True`.
        device (torch.device): the device to use for tensor creation. Defaults to `None`.

    Examples:
        >>> from torchrl.data.llm.chat import History, _CHAT_TEMPLATES
        >>> from torchrl.modules.llm import TransformersWrapper
        >>> from torchrl.objectives.llm.sft import SFTLoss
        >>> from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
        >>> from tensordict import TensorDict, lazy_stack, set_list_to_stack
        >>> import torch
        >>>
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
        >>> tokenizer.chat_template = _CHAT_TEMPLATES["chatml_format"]
        >>> model = OPTForCausalLM(OPTConfig()).eval()
        >>>
        >>> # Create training and reference policies
        >>> policy_train = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     from_text=True,
        ...     chat_template_name="qwen",
        ... )
        >>> policy_ref = TransformersWrapper(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     generate=False,
        ...     from_text=True,
        ...     return_log_probs=True,
        ...     chat_template_name="qwen",
        ... )
        >>>
        >>> # Create the RetrieveLogProb transform
        >>> transform = RetrieveLogProb(
        ...     policy_ref,
        ...     assistant_only=True,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ...     tokenizer=tokenizer,
        ... )
        >>>
        >>> # Prepare data
        >>> text = history[:, :-1].apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=True
        ... )
        >>> text_response = history.apply_chat_template(
        ...     tokenizer=tokenizer, chat_template_name="qwen", add_generation_prompt=False
        ... )
        >>> text_response = [
        ...     txt[len(txt_start):] for txt, txt_start in zip(text_response, text)
        ... ]
        >>> td = TensorDict(
        ...     text=text,
        ...     text_response=text_response,
        ...     history=history,
        ...     next=TensorDict(
        ...         reward=torch.randn(2, 1),
        ...         done=torch.zeros(2, dtype=torch.bool),
        ...         history=history,
        ...     ),
        ...     batch_size=(2,),
        ... )
        >>> data = lazy_stack(list(td.unbind(0)))
        >>>
        >>> # Apply the transform to get reference log probabilities
        >>> data = transform(data)
        >>> # You can get a padded tensor for batching:
        >>> ref_log_probs = data.get(("next", "ref_log_prob"), as_padded_tensor=True)
        >>> print(f"Type: {type(ref_log_probs)}, Length: {len(ref_log_probs)}")
        Type: <class 'torch.Tensor'>, Length: 2
        >>> print(f"Example shapes: {[x.shape for x in ref_log_probs]}")
        Example shapes: [torch.Size([35]), torch.Size([35])]
        >>> print(ref_log_probs.shape)  # (batch, max_seq_len)
        torch.Size([2, 35])
        >>>
        >>> # Use with SFTLoss for KL regularization
        >>> loss = SFTLoss(
        ...     actor_network=policy_train,
        ...     tokenizer=tokenizer,
        ...     reduction="mean",
        ...     normalize_by_seq_length=True,
        ...     kl_to_ref_coeff=0.1,
        ...     tokenizer_kwargs={"chat_template_name": "qwen"},
        ... )
        >>> loss_vals = loss(data)
        >>> print(f"SFT Loss: {loss_vals.loss_sft.item():.4f}")
        SFT Loss: 10.7856
        >>> print(f"KL to Reference Loss: {loss_vals.loss_kl_to_ref.item():.4f}")
        KL to Reference Loss: 0.0000
        >>> print(f"Total Loss: {loss_vals.sum(reduce=True).item():.4f}")
        Total Loss: 10.7856

    Note:
        By default, the log-probabilities are stored as a list of tensors (one per sample, with variable length).
        Use `as_padded_tensor=True` in `.get()` to obtain a batchable tensor (with padding).
        The reference log probabilities are computed only for assistant tokens when `assistant_only=True`.

    """

    def __init__(
        self,
        actor: CategoricalSequential,
        *,
        history_key: NestedKey | None = None,
        log_prob_key: NestedKey = "ref_log_prob",
        assistant_only: bool = False,
        tokenizer_kwargs: dict | None = None,
        detach: bool = True,
        device: torch.device | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,
    ):
        if history_key is None:
            history_key = "history"
        self.history_key = history_key
        self.log_prob_key = log_prob_key
        super().__init__(in_keys=[history_key], out_keys=[log_prob_key])
        self.actor = actor
        if not getattr(actor, "return_log_probs", True):
            raise ValueError(
                "The actor must have `return_log_probs=True` to use the `AssistantLogProb` transform."
            )
        if getattr(actor, "generate", True):
            raise ValueError(
                "The actor must have `generate=False` to use the `AssistantLogProb` transform."
            )
        if not getattr(actor, "from_text", False):
            raise ValueError(
                "The actor must have `from_text=True` to use the `AssistantLogProb` transform. If `from_text=False` is required, please file an issue on GitHub."
            )
        # if getattr(self.actor, "tokenizer_kwargs", {}).get("add_generation_prompt", True):
        # raise ValueError("The actor must have `tokenizer_kwargs['add_generation_prompt']=False` to use the `AssistantLogProb` transform.")
        self.assistant_only = assistant_only
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer_kwargs.setdefault("return_assistant_tokens_mask", True)
        tokenizer_kwargs.setdefault("tokenize", True)
        tokenizer_kwargs.setdefault("return_tensors", "pt")
        tokenizer_kwargs.setdefault("padding", False)
        tokenizer_kwargs.setdefault("add_generation_prompt", False)
        self.tokenizer_kwargs = tokenizer_kwargs
        self.tokenizer = tokenizer
        self.detach = detach
        self.device = device

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = self._step(tensordict, tensordict.get("next"))
        return tensordict.set("next", next_td)

    @set_list_to_stack(True)
    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        td = next_tensordict.select(self.history_key)
        with torch.device(
            self.device
        ) if self.device is not None else contextlib.nullcontext(), torch.no_grad() if self.detach else contextlib.nullcontext():
            result = self.actor(td.select(self.history_key))
            td.update(result.select(getattr(self.actor, "log_prob_key", "log_probs")))
            td.rename_key_(
                getattr(self.actor, "log_prob_key", "log_probs"), self.log_prob_key
            )
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        if self.assistant_only:
            with torch.device(
                self.device
            ) if self.device is not None else contextlib.nullcontext():
                # Get assistant mask
                history: History = td.get(self.history_key)
                proc = history.apply_chat_template(
                    tokenizer=self.actor.tokenizer
                    if self.tokenizer is None
                    else self.tokenizer,
                    **self.tokenizer_kwargs,
                )
                assistant_masks = proc.get("assistant_masks", as_list=True)
                log_probs = td.get(self.log_prob_key, as_list=True)
                log_probs = [
                    lp[mask.bool()]
                    for lp, mask in _zip_strict(log_probs, assistant_masks)
                ]
                td = td.set(self.log_prob_key, log_probs)
        return next_tensordict.update(td)
