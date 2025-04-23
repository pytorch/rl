# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import collections
import importlib

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn, Tensor
from torch.nn import functional as F

from torchrl.data.llm.prompt import PromptData

_has_transformers = importlib.util.find_spec("transformers") is not None


class KLControllerBase(abc.ABC):
    """Base class for KL controllers.

    Each controller must implement an update method that takes the current KL value and
    the number of steps and updates the kl_coef attribute of the wrapped model,
    which will multiply the KL during calculation of the reward.
    """

    @abc.abstractmethod
    def update(self, kl_values: list[float]) -> float:
        ...


class ConstantKLController(KLControllerBase):
    """Constant KL Controller.

    This controller maintains a fixed coefficient no matter what values it is updated
    with.

    Keyword Arguments:
        kl_coef (:obj:`float`): The coefficient to multiply KL with when calculating the
            reward.
        model (nn.Module, optional): wrapped model that needs to be controlled.
            Must have an attribute ``"kl_coef"``. If provided, the ``"kl_coef"`` will
            be updated in-place.
    """

    def __init__(
        self,
        *,
        kl_coef: float = None,
        model: nn.Module | None = None,
    ):
        self.model = model
        if model is not None and not hasattr(model, "kl_coef"):
            raise AttributeError(
                "Model input to ConstantKLController doesn't have attribute 'kl_coef'"
            )
        self.coef = kl_coef
        if model is not None:
            self.model.kl_coef = self.coef

    def update(self, kl_values: list[float] = None) -> float:
        if self.model is not None:
            self.model.kl_coef = self.coef
        return self.coef


class AdaptiveKLController(KLControllerBase):
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences".

    Keyword Arguments:
        init_kl_coef (:obj:`float`): The starting value of the coefficient.
        target (:obj:`float`): The target KL value. When the observed KL is smaller, the
            coefficient is decreased, thereby relaxing the KL penalty in the training
            objective and allowing the model to stray further from the reference model.
            When the observed KL is greater than the target, the KL coefficient is
            increased, thereby pulling the model back towards the reference model.
        horizon (int): Scaling factor to control how aggressively we update the
            coefficient.
        model (nn.Module, optional): wrapped model that needs to be controlled.
            Must have an attribute ``"kl_coef"``. If provided, the ``"kl_coef"`` will
            be updated in-place.

    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(
        self,
        *,
        init_kl_coef: float,
        target: float,
        horizon: int,
        model: nn.Module | None = None,
    ):
        self.model = model
        self.coef = init_kl_coef
        self.target = target
        self.horizon = horizon
        if model is not None:
            self.model.kl_coef = self.coef

    def update(self, kl_values: list[float]):
        """Update ``self.coef`` adaptively.

        Arguments:
            kl_values (sequence of float): The current KL value between the newest policy and the initial
                policy.

        """
        if kl_values is None:
            raise ValueError(
                f"The kl_values were not provided to {type(self)}. "
                f"Make sure these values are provided for the scheduler to be updated "
                f"accordingly. "
            )
        n_steps = len(kl_values)
        # renormalize kls
        kl_value = -torch.as_tensor(kl_values).mean() / self.coef
        proportional_error = np.clip(kl_value / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.coef *= mult  # βₜ₊₁
        if self.model is not None:
            self.model.kl_coef = self.coef
        return self.coef


class RolloutFromModel:
    """A class for performing rollouts with causal language models.

    It is assumed that the model this class wraps takes as input tokenized text and
    whose task is to predict the next word in a sentence having read the n previous
    words.

    Args:
        model (transformers.Transformer): the model to be used. Should have a
            :meth:`generate` method.
        ref_model (transformers.Transformer): a frozen version of ``model``
            where params are in their initial configuration. This is used to compute a
            KL penalty for the reward, to stop the model from straying too far from the
            reference model during training.
        reward_model: (nn.Module, tensordict.nn.TensorDictModule): a model which, given
            ``input_ids`` and ``attention_mask``, calculates rewards for each token and
            end_scores (the reward for the final token in each sequence).
        kl_coef: (:obj:`float`, optional): initial kl coefficient.
        max_new_tokens (int, optional): the maximum length of the sequence.
            Defaults to 50.
        score_clip (:obj:`float`, optional): Scores from the reward model are clipped to the
            range ``(-score_clip, score_clip)``. Defaults to 10.
        kl_scheduler (KLControllerBase, optional): the KL coefficient scheduler.
        num_steps (int, optional): number of steps between two optimization.

    Examples:
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models.llm import GPT2RewardModel
        >>> from torchrl.data.llm.utils import RolloutFromModel
        >>> from torchrl.data.llm.dataset import get_dataloader
        >>> from torchrl.data.llm.prompt import PromptData
        >>> from transformers import GPT2LMHeadModel
        >>>
        >>> dl = get_dataloader(
        ...     batch_size=4,
        ...     block_size=550,
        ...     tensorclass_type=PromptData,
        ...     device="cpu",
        ...     dataset_name="CarperAI/openai_summarize_tldr",
        ... )
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> # we load ref_model with random weights so it differs from model
        >>> ref_model = GPT2LMHeadModel(GPT2LMHeadModel.config_class())
        >>> reward_model = GPT2RewardModel(model_path="gpt2")
        >>> rollout_from_model = RolloutFromModel(model, ref_model, reward_model)
        >>>
        >>> batch = next(dl)
        >>> rollout = rollout_from_model.rollout_from_data(batch)
        >>> rollout
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([4, 50]), device=cpu, dtype=torch.int64, is_shared=False),
                attention_mask: Tensor(shape=torch.Size([4, 50, 600]), device=cpu, dtype=torch.bool, is_shared=False),
                input_ids: Tensor(shape=torch.Size([4, 50, 600]), device=cpu, dtype=torch.int64, is_shared=False),
                next: TensorDict(
                    fields={
                        attention_mask: Tensor(shape=torch.Size([4, 50, 600]), device=cpu, dtype=torch.bool, is_shared=False),
                        done: Tensor(shape=torch.Size([4, 50, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        input_ids: Tensor(shape=torch.Size([4, 50, 600]), device=cpu, dtype=torch.int64, is_shared=False),
                        reward: Tensor(shape=torch.Size([4, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_kl: Tensor(shape=torch.Size([4, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward_raw: Tensor(shape=torch.Size([4, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4, 50]),
                    device=cpu,
                    is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([4, 50, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4, 50]),
            device=cpu,
            is_shared=False)

    """

    EOS_TOKEN_ID = 50256

    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        kl_coef=0.1,
        max_new_tokens=50,
        score_clip=10.0,
        kl_scheduler: KLControllerBase | None = None,
        num_steps: int | None = None,
    ):
        if not _has_transformers:
            raise ImportError(
                "transformers module couldn't be found. Make sure it is installed in your "
                "environment."
            )
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.max_new_tokens = max_new_tokens
        self.score_clip = score_clip
        self.kl_coef = kl_coef
        self.kl_scheduler = kl_scheduler
        if num_steps is not None:
            self._kl_queue = collections.deque(maxlen=num_steps)
        else:
            # we create a list. Value appended to it will be detached scalars so very cheap to store,
            # even if the update is not called.
            # The scheduler update will take care of erasing these values.
            self._kl_queue = []

    @torch.no_grad()
    def rollout_from_data(self, batch):
        generated, log_probs, log_ratio = self.generate(batch)
        return self.create_rollout_td(batch, generated, log_probs, log_ratio)

    @torch.no_grad()
    def create_rollout_td(self, batch, generated, log_probs, log_ratio):
        """A TensorDict wrapper for generated data.

        This function takes a batch plus the generated tokens and replicates the
        tensordict structure that would have been obtained from a rollout with a TorchRL
        env that sampled one token each timestep.

        Args:
            batch (TensorDict): A batch of data containing the original prompt together with a field
                "rindex" indicating the right index of the prompt.
            generated (torch.Tensor): Tokenized prompt followed by generated tokens. This can be obtained
                by calling the ``generate`` method.
            log_probs (torch.Tensor): The log probabilities of the generated tokens. Can be obtained by
                calling the ``generate`` method.
            log_ratio (torch.Tensor): The log ratio of the probabilities of the generated tokens
                according to the generative model and the reference model. Can be
                obtained by calling the ``generate`` method.

        Returns:
            A :class:`~tensordict.TensorDict` with the following keys:

            - ``"action"``: the sequence of actions (generated tokens)
            - ``"input_ids"``: the input_ids passed to the generative model at each time
              step.
            - ``"attention_mask"``: the attention_masks passed to the generative model at
              each time step
            - ``"sample_log_prob"``: the log probability of each token during generation
            - ``("next", "input_ids")``: the sequence of tokens after generation. Makes up
              part of the inputs that will be used for generating the next token.
            - ``("next", "attention_mask")``: updated attention_mask after token has been
              generated. Passed to the generative model on the next time step
            - ``("next", "terminated")``: Boolean array indicating whether we've reached a
              terminal state (either because we generated EOS token or because we
              reached the token limit)
            - ``("next", "done")``: Boolean array indicating whether we've reached a
              final state. Currently a copy of ``"terminated"``.
            - ``("next", "reward")``: The reward received at each time step
            - ``("next", "reward_raw")``: The raw reward from the reward model, without the
              KL term. This is mainly for debugging and logging, it is not used in
              training
            - ``("next", "reward_kl")``: The KL term from the reward. This is mainly for
              debugging and logging, it is not used in training.

        """
        rollout_generated = self._get_rollout_generated(generated, batch)
        rollout_attention_mask = (rollout_generated != self.EOS_TOKEN_ID).bool()

        done, terminated = self._get_done_status(generated, batch)
        action = self._get_action(generated, batch)
        end_scores, end_scores_labels = self._get_end_scores(
            rollout_generated, rollout_attention_mask, batch
        )

        # the reward is zero except for the timestep where we reached a stopping condition
        clipped_scores = torch.clip(
            end_scores - end_scores_labels, -self.score_clip, self.score_clip
        )
        reward_raw = clipped_scores.unsqueeze(-1).unsqueeze(-1)
        reward_raw = reward_raw * done
        reward_kl = -self.kl_coef * log_ratio.unsqueeze(-1)
        reward = reward_raw + reward_kl
        td = {
            "action": action,
            "input_ids": rollout_generated[:, :-1].clone(),
            "attention_mask": rollout_attention_mask[:, :-1].clone(),
            "sample_log_prob": log_probs,
            "next": {
                "input_ids": rollout_generated[:, 1:].clone(),
                "attention_mask": rollout_attention_mask[:, 1:].clone(),
                "done": done,
                "terminated": terminated,
                "reward": reward,
                "reward_raw": reward_raw,
                "reward_kl": reward_kl,
            },
        }
        self._kl_queue.append(reward_kl.detach().mean())
        return TensorDict(
            td, batch_size=done.shape[:2], device=generated.device
        ).refine_names(..., "time")

    def _get_rollout_generated(self, generated, batch):
        # stack the individual timesteps during generation into a single tensor
        rollout_generated = []
        arange = torch.arange(generated.shape[1], device=generated.device)
        for rindex, row in zip(batch.prompt_rindex, generated):
            tokens = []
            for i in range(self.max_new_tokens + 1):
                tokens.append(torch.where(arange < rindex + i, row, self.EOS_TOKEN_ID))
            rollout_generated.append(torch.stack(tokens))
        rollout_generated = torch.stack(rollout_generated)
        return rollout_generated

    def _get_done_status(self, generated, batch):
        # done is True when we either first sample an EOS token or reach the maximum number
        # of generated tokens
        done_idx = torch.minimum(
            (generated != self.EOS_TOKEN_ID).sum(dim=-1) - batch.prompt_rindex,
            torch.as_tensor(self.max_new_tokens) - 1,
        )
        truncated_idx = (
            torch.as_tensor(self.max_new_tokens, device=generated.device).expand_as(
                done_idx
            )
            - 1
        )
        zeros = torch.zeros(
            done_idx.numel(),
            self.max_new_tokens,
            dtype=torch.bool,
            device=generated.device,
        )
        truncated = zeros.scatter(-1, truncated_idx.unsqueeze(-1), 1).unsqueeze(-1)
        done = zeros.scatter(-1, done_idx.unsqueeze(-1), 1).unsqueeze(-1)
        terminated = (
            done & ~truncated
        )  # we assume that if it's not truncated, it was terminated
        return truncated | terminated, terminated

    def _get_action(self, generated, batch):
        # the sequence of actions for each trajectory is just the generated token ids
        action_idx = torch.arange(self.max_new_tokens, device=generated.device)
        action_idx = action_idx + batch.prompt_rindex.unsqueeze(-1)
        return generated.gather(-1, action_idx)

    def _get_end_scores(self, rollout_generated, rollout_attention_mask, batch):
        # calculate the reward for the finished sequence
        _, end_scores = self.reward_model(
            input_ids=rollout_generated[:, -1],
            attention_mask=rollout_attention_mask[:, -1],
        )
        _, end_scores_labels = self.reward_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        return end_scores, end_scores_labels

    @classmethod
    def _padded_right_to_left(cls, tensor, *, eos_token_id=None, dim=1):
        if eos_token_id is None:
            eos_token_id = cls.EOS_TOKEN_ID
        mask = tensor != eos_token_id
        out = torch.full_like(tensor, eos_token_id)
        out[mask.flip(dim)] = tensor[mask]
        return out

    @classmethod
    def _padded_left_to_right(
        cls, tensor, *, sequence_length=None, eos_token_id=None, dim=1
    ):
        # some care must be taken here, because generated sequences may have both left
        # and right padding, and also may not terminated early if all sequences in the
        # batch reached EOS before reaching the token limit
        if sequence_length is None:
            sequence_length = tensor.size(dim)
        if dim < 0:
            dim = tensor.ndim + dim
        if eos_token_id is None:
            eos_token_id = cls.EOS_TOKEN_ID
        mask = tensor != eos_token_id
        # convert [0, 0, 1, 1, 0] to [0, 0, 1, 1, 1] to avoid right eos
        mask = ~((~mask).to(torch.uint8).cumprod(dim).bool())
        shape = list(mask.shape)
        shape[dim] = sequence_length
        out = torch.full(torch.Size(shape), eos_token_id, device=tensor.device)
        index = (slice(None),) * dim + (slice(tensor.size(dim)),)
        out[index][mask.flip(dim)] = tensor[mask]
        return out

    @property
    def _default_conf(self):
        from transformers import GenerationConfig

        return GenerationConfig(
            pad_token_id=self.EOS_TOKEN_ID,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )

    def _get_scores(
        self, scores: tuple, generated_tokens: Tensor = None, use_max=False, pad_to=None
    ):
        scores = torch.stack(scores, 1)
        if scores.shape[1] != self.max_new_tokens:
            scores = F.pad(
                scores,
                (0, 0, 0, self.max_new_tokens - scores.shape[1]),
                value=float("-inf"),
            )
        scores = F.log_softmax(scores, dim=-1)
        if use_max:
            scores = scores.max(dim=-1).values
        else:
            index = generated_tokens.unsqueeze(-1)
            scores = torch.gather(scores, dim=-1, index=index)
        if pad_to is not None:
            pad = pad_to - scores.shape[1]
            return F.pad(scores, (0, pad), value=-float("inf"))
        return scores

    @staticmethod
    def logprobs_of_labels(logits, labels):
        """Log probabilities of the labels.

        These are calculated from the logits. The labels (token ids) are used to index
        the logits along the relevant dimension.
        """
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
        return logprobs_labels.squeeze(-1)

    @torch.no_grad()
    def _log_ratio(self, generated, prompt_rindex):
        # get the scores and normalise for log probabilities
        attention_mask = (generated != self.EOS_TOKEN_ID).bool()
        logits = self.model(
            input_ids=generated, attention_mask=attention_mask, return_dict=True
        ).logits
        logprobs = self.logprobs_of_labels(logits[:, :-1], generated[:, 1:])
        ref_logits = self.ref_model(
            input_ids=generated.to(self.ref_model.device),
            attention_mask=attention_mask.to(self.ref_model.device),
            return_dict=True,
        ).logits.to(logits.device)
        ref_logprobs = self.logprobs_of_labels(ref_logits[:, :-1], generated[:, 1:])
        log_ratio = logprobs - ref_logprobs
        log_ratio = log_ratio.masked_fill(~attention_mask[:, :-1], 0)
        log_ratio = torch.stack(
            [
                row[rindex - 1 : rindex + self.max_new_tokens - 1]
                for row, rindex in zip(log_ratio, prompt_rindex)
            ],
            dim=0,
        )
        return log_ratio

    def _get_generated_tokens(self, generated, rindex):
        # extracts the generated tokens from the full sequence of prompt + generated
        idx = torch.arange(generated.shape[1], device=generated.device)
        rindex = rindex.unsqueeze(-1)
        mask = (idx >= rindex) & (idx < rindex + self.max_new_tokens)
        return generated[mask].reshape(-1, self.max_new_tokens)

    @torch.no_grad()
    def generate(self, batch: PromptData, generation_config=None):
        """Generates a sequence of tokens from a batch of data sampled from the data collector.

        Args:
            batch (PromptData): the data to be used. Must have ``input_ids``
                and ``prompt_rindex`` fields.
            generation_config (GenerationConfig, optional): the configuration for the
                call to generate.

        Returns:
            generated (torch.Tensor): a [B x (Ti +To)] sequence of integers (tokens),
                where Ti is the length of the input sequence and To is the length
                of the generated sequence.
            log_probs_gen: the log-probabilities of the token generated.
            log_ratio: the log ratio between probabilities under the generative
                model and the frozen version.

        """
        input_ids = batch.mask_label().input_ids

        # move padding tokens to left pad
        # huggingface models expect left padding for generation
        input_ids = self._padded_right_to_left(input_ids)

        # generate and capture scores
        if generation_config is None:
            generation_config = self._default_conf

        attention_mask = (input_ids != self.EOS_TOKEN_ID).bool()
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        samples = outputs.sequences

        # we'll insert generated tokens into a tensor prepopulated with padding tokens,
        # thereby moving back to right padding for reward model
        generated = self._padded_left_to_right(
            samples,
            sequence_length=input_ids.shape[1] + self.max_new_tokens,
            eos_token_id=self.EOS_TOKEN_ID,
        )
        generated_tokens = self._get_generated_tokens(generated, batch.prompt_rindex)
        # get the scores and normalise for log probabilities
        log_probs_gen = self._get_scores(outputs.scores, generated_tokens)

        log_ratio = self._log_ratio(generated, batch.prompt_rindex)
        return generated, log_probs_gen, log_ratio

    def step_scheduler(self):
        # recover true kl
        self.kl_coef = self.kl_scheduler.update(self._kl_queue)
        if isinstance(self._kl_queue, (list, collections.deque)):
            # remove all values
            while len(self._kl_queue):
                self._kl_queue.remove(self._kl_queue[0])
