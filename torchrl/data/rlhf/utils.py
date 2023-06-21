from typing import Tuple

import torch
import transformers
from torch import Tensor

from tensordict import TensorDict
from torch.nn import functional as F
from transformers import GenerationConfig

from torchrl.data.rlhf.tldr import PromptData


class RolloutFromModel:
    """
    Args:
        model (transformers.Transformer): the model to be used. Should have a
            :meth:`generate` method.
        ref_model (transformers.Transformer): a frozen version of ``model``
            where params are in their initial configuration.
        max_new_tokens (int, optional): the maximum length of the sequence.
            Defaults to 50.

    """
    EOS_TOKEN_ID = 50256

    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        max_new_tokens=50,
        kl_coef=0.1,
                 ):
        self.model =  model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.max_new_tokens = max_new_tokens
        self.kl_coef = kl_coef

    def rollout_from_data(self,
        batch,
    ):
        generated, log_probs, log_ratio = self.generate(batch)
        return self.create_rollout_td(
            batch, generated, log_probs, log_ratio, self.max_new_tokens, self.kl_coef
        )


    @torch.no_grad()
    def create_rollout_td(self,
        batch, generated, reward_model, log_probs, log_ratio, max_new_tokens=50, kl_coef=0.1
    ):
        """A TensorDict wrapper for generated data.

        This function takes a batch plus the generated tokens and replicates the tensordict
        structure that would have been obtained from a rollout with a TorchRL env that
        sampled one token each timestep.

        Args:
            batch:
        """
        rollout_generated = []
        for rindex, row in zip(batch.prompt_rindex, generated):
            arange = torch.arange(row.shape[0], device=generated.device)
            tokens = []
            for i in range(max_new_tokens + 1):
                tokens.append(
                    torch.where(
                        arange < rindex + i,
                        row,
                        self.EOS_TOKEN_ID,
                    )
                )
            rollout_generated.append(torch.stack(tokens))
        rollout_generated = torch.stack(rollout_generated)
        rollout_attention_mask = (rollout_generated != self.EOS_TOKEN_ID).bool()

        # done is True when we either first sample an EOS token or reach the maximum number
        # of generated tokens
        done_idx = torch.minimum(
            (generated != self.EOS_TOKEN_ID).sum(dim=-1) - batch.prompt_rindex,
            torch.tensor(max_new_tokens),
        )
        done = torch.zeros(done_idx.numel(), max_new_tokens, dtype=torch.bool)
        done = done.scatter(-1, done_idx.unsqueeze(-1), 1)

        # the sequence of actions for each trajectory is just the generated token ids
        action_idx = torch.arange(max_new_tokens, device=generated.device)
        action_idx = action_idx + batch.prompt_rindex.unsqueeze(-1)
        action = generated.gather(-1, action_idx)

        # calculate the reward for the finished sequence
        _, end_scores = reward_model(
            input_ids=rollout_generated[:, -1], attention_mask=rollout_attention_mask[:, -1]
        )
        _, end_scores_labels = reward_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        # the reward is zero except for the timestep where we reached a stopping condition
        reward = end_scores - end_scores_labels
        reward = reward.unsqueeze(-1).unsqueeze(-1)
        reward = reward.masked_scatter(~done, 0)
        reward = reward - kl_coef * log_ratio.unsqueeze(-1)
        td = {
            "action": action,
            "input_ids": rollout_generated[:, :-1].clone(),
            "attention_mask": rollout_attention_mask[:, :-1].clone(),
            "sample_log_prob": log_probs,
            "next": {
                "input_ids": rollout_generated[:, 1:].clone(),
                "attention_mask": rollout_attention_mask[:, 1:].clone(),
                "done": done,
                "reward": reward,
            },
        }
        return TensorDict(td, batch_size=done.shape[:2], device=generated.device)

    @classmethod
    def _padded_right_to_left(cls, tensor, eos_token_id=None, dim=1):
        # this is about 2x slower than masking
        # tensor = torch.stack(
        #     [torch.roll(row, (row == eos_token_id).sum().item(), 0) for row in tensor]
        # )
        if eos_token_id is None:
            eos_token_id = cls.EOS_TOKEN_ID
        mask = tensor != eos_token_id
        out = torch.full_like(tensor, eos_token_id)
        out[mask.flip(dim)] = tensor[mask]
        return out

    @classmethod
    def _padded_left_to_right(cls, tensor, eos_token_id=None, dim=1):
        # mask = tensor != eos_token_id
        # splits = tensor[mask].split(mask.sum(-1).tolist(), 0)
        # tensor = torch.nested.as_nested_tensor(list(splits))
        # tensor = torch.nested.to_padded_tensor(tensor, eos_token_id)
        if eos_token_id is None:
            eos_token_id = cls.EOS_TOKEN_ID
        return cls._padded_right_to_left(tensor, eos_token_id, dim=dim)

    @property
    def _default_conf(self):
        return GenerationConfig(
            pad_token_id=self.EOS_TOKEN_ID,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )

    @staticmethod
    def _get_scores(scores: Tuple, generated_tokens: Tensor=None, use_max=False, pad_to=None):
        scores = torch.stack(scores, 1)
        if use_max:
            scores = F.log_softmax(scores, dim=-1).max(dim=-1).values
        else:
            index = generated_tokens.unsqueeze(-1)
            scores = torch.gather(scores, -1, index)
        if pad_to is not None:
            pad = pad_to - scores.shape[1]
            return F.pad(scores, (0, pad), value=-float("inf"))
        return scores

    @staticmethod
    def logprobs_of_labels(logits, labels):
        """Log probabilities of the labels.

        These are calculated from the logits."""
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
        return logprobs_labels.squeeze(-1)

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

    @torch.no_grad()
    def generate(self,
        batch: PromptData,
        generation_config=None
    ):
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
        generated = self._padded_left_to_right(samples, self.EOS_TOKEN_ID)

        # get the scores and normalise for log probabilities
        log_probs_gen = self._get_scores(outputs.scores, generated)

        log_ratio = self._log_ratio(generated, batch.prompt_rindex)
        return generated, log_probs_gen, log_ratio

