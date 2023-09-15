# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn as nn

_has_transformers = importlib.util.find_spec("transformers") is not None


class GPT2RewardModel(nn.Module):
    """Wrapper around GPT2-like models to enable their use as reward models.

    This wrapper replaces the language modelling head of the GPT2 model with a new
    linear layer with 1 output that can be used as a reward signal. It also exposes the
    method ``compute_reward_loss`` which calculates the reward loss by comparing two
    batches, one chosen, one rejected.

    Examples:
        >>> from transformers import GPT2Tokenizer
        >>> from torchrl.modules.models.rlhf import GPT2RewardModel
        >>> reward_model = GPT2RewardModel(model_path="gpt2")
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model_inputs = tokenizer(
        ...     ["This is a test sentence"], return_tensors="pt"
        ... )
        >>> rewards, end_scores = reward_model(**model_inputs)
        >>> assert rewards.shape == model_inputs["input_ids"].shape
        >>> assert end_scores.shape[1] == 1

    """

    def __init__(self, model_path=None):
        if not _has_transformers:
            raise ImportError("The transformers library is missing.")

        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        super().__init__()
        if model_path:
            model = GPT2LMHeadModel.from_pretrained(model_path, return_dict=False)
        else:
            model = GPT2LMHeadModel(GPT2LMHeadModel.config_class())

        self.config = model.config
        self.transformer = model.transformer

        # replace last layer with the reward layer
        self.lm_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.pad_id = GPT2TokenizerFast.from_pretrained("gpt2").eos_token_id

    def forward(self, input_ids, attention_mask):
        """Returns a tuple (rewards, end_scores) where `rewards` contains all rewards computed at each timestep, `end_scores` contains the reward computed at the last-non-padding token."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        rewards = self.lm_head(hidden_states).squeeze(-1)
        end_scores = self._compute_end_scores(rewards, input_ids)

        return rewards, end_scores

    def _compute_end_scores(self, rewards, input_ids):
        end_scores = []
        bs = input_ids.shape[0]

        for i in range(bs):
            pad_inds = (input_ids[i] == self.pad_id).nonzero()
            first_pad_ind = (
                pad_inds[0].item() if len(pad_inds) > 0 else input_ids.shape[1]
            )
            end_scores.append(rewards[i, first_pad_ind - 1])

        return torch.stack(end_scores)

    @staticmethod
    def compute_reward_loss(chosen_batch, rejected_batch, pad_token_id=50256):
        """Compute the reward loss given a chosen and rejected batch.

        The loss is computed as ``loss = -log_sigmoid(chosen_reward - rejected_reward)``.
        This loss is small when the reward model favours the chosen data and large if
        the model favours the rejected data.

          .. note:: The loss is computed excluding the common "prefix" subsequence to effectively disregard contribution of the original prompt.

        Examples:
            >>> import torch
            >>> from tensordict.nn import TensorDictModule
            >>> from torchrl.data.rlhf.dataset import get_dataloader
            >>> from torchrl.data.rlhf.reward import PairwiseDataset
            >>> from torchrl.modules.models.rlhf import GPT2RewardModel
            >>>
            >>> reward_model = TensorDictModule(
            ...     GPT2RewardModel(model_path="gpt2"),
            ...     in_keys=["input_ids", "attention_mask"],
            ...     out_keys=["rewards", "end_scores"],
            ... )
            >>> dl = get_dataloader(
            ...     batch_size=4,
            ...     block_size=550,
            ...     tensorclass_type=PairwiseDataset,
            ...     device="cpu",
            ...     dataset_name="CarperAI/openai_summarize_comparisons",
            ... )
            >>> batch = next(dl)
            >>> reward_model(batch.chosen_data)
            >>> reward_model(batch.rejected_data)
            >>> loss = reward_model.compute_reward_loss(
            ...     batch.chosen_data, batch.rejected_data
            ... )
            >>> assert isinstance(loss, torch.Tensor)
            >>> assert loss.shape == torch.Size([])
        """
        chosen_ids = chosen_batch.input_ids
        rejected_ids = rejected_batch.input_ids
        chosen_rewards = chosen_batch.rewards
        rejected_rewards = rejected_batch.rewards

        bs = chosen_rewards.shape[0]
        loss = 0

        # TODO: this loop can likely be made more efficient
        for i in range(bs):
            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_ids[i] == pad_token_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_ids.shape[1]
            r_inds = (rejected_ids[i] == pad_token_id).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_ids.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_ids[i] != rejected_ids[i]).nonzero()[0]

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            loss += -F.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
        return loss / bs

    @classmethod
    def from_pretrained(cls, path):
        filename = Path(path) / "reward_model.pt"
        if filename.exists():
            return torch.load(filename)
        return cls(path)

    def save_pretrained(self, path):
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        torch.save(self, save_dir / "reward_model.pt")
