from pathlib import Path

import torch
from torch import nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class GPT2RewardModel(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        if model_path:
            model = GPT2LMHeadModel.from_pretrained(model_path, return_dict=False)
        else:
            model = GPT2LMHeadModel(GPT2LMHeadModel.config_class())

        self.config = model.config
        self.transformer = model.transformer

        # replace last layer with the reward layer
        self.lm_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.PAD_ID = GPT2TokenizerFast.from_pretrained("gpt2").eos_token_id

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        rewards = self.lm_head(hidden_states).squeeze(-1)
        end_scores = []
        bs = input_ids.shape[0]

        for i in range(bs):
            pad_inds = (input_ids[i] == self.PAD_ID).nonzero()
            first_pad_ind = (
                pad_inds[0].item() if len(pad_inds) > 0 else input_ids.shape[1]
            )
            end_scores.append(rewards[i, first_pad_ind - 1])

        return rewards, torch.stack(end_scores)

    @staticmethod
    def compute_reward_loss(chosen_batch, rejected_batch, pad_token_id=50256):
        chosen_ids = chosen_batch.input_ids
        rejected_ids = rejected_batch.input_ids
        chosen_rewards = chosen_batch.rewards
        rejected_rewards = rejected_batch.rewards

        bs = chosen_rewards.shape[0]
        loss = 0

        for i in range(bs):
            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_ids[i] == pad_token_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_ids.shape[1]
            r_inds = (rejected_ids[i] == pad_token_id).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_ids.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_ids[i] != rejected_ids[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
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
