"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import torch
import torch.nn as nn
from nanoGPT.model import GPT, GPTConfig
from torch.nn import functional as F

__all__ = ["GPT", "GPTConfig", "RLHF"]


class RLHF(nn.Module):
    def __init__(self, model, mode, discrete_reward=False):
        super().__init__()
        self.model = model
        self.config = model.config

        # reward model
        self.n_embd = model.lm_head.in_features
        self.block_size = model.config.block_size
        model.policy_head = nn.Linear(
            model.lm_head.in_features, model.lm_head.out_features, bias=False
        )
        self.mode = mode
        self.discrete_reward = discrete_reward
        if discrete_reward:
            model.reward_head = nn.Linear(model.lm_head.in_features, 2, bias=False)
        else:
            model.reward_head = nn.Linear(model.lm_head.in_features, 1, bias=False)

    def forward_reward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.model.transformer.wte(idx)
        # position embeddings of shape (1, t, n_embd)
        pos_emb = self.model.transformer.wpe(pos)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)

        rewards = self.model.reward_head(x[:, -1, :])

        if self.discrete_reward:
            probs = torch.softmax(rewards, 1)
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                loss = F.cross_entropy(probs, targets, ignore_index=-1)
            else:
                loss = None
            return probs, loss
        else:
            return rewards

    def forward(self, idx, targets=None):
        if self.mode == "reward":
            return self.forward_reward(idx, targets)
        else:
            return self.model(idx, targets)

    def generate(
        self,
        idx,
        max_new_tokens,
        device,
        block_size,
        use_reference=True,
        reward_model=None,
        hard_code_reward=True,
    ):
        # idx is (B, T) array of indices in the current context
        log_probs = torch.tensor([]).to(device)
        log_probs_ref = torch.tensor([]).to(device)

        values_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        advantages_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)

        gamma = 1
        lam = 1

        # TODO: Critic, PPO
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # block_size = 256
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities

            probs_next = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs_next, num_samples=1)  # (B, 1)

            probs_idx_next = torch.gather(probs_next, 1, idx_next)
            log_probs_idx_next = torch.log(probs_idx_next)
            log_probs = torch.cat((log_probs, log_probs_idx_next), dim=1)

            if use_reference:
                logits_ref, _ = self.model(idx_cond)
                logits_ref = logits_ref[:, -1, :]  # becomes (B, C)
                probs_ref_next = F.softmax(logits_ref, dim=-1)  # (B, C)
                probs_ref_idx_next = torch.gather(probs_ref_next, 1, idx_next)
                log_probs_ref_idx_next = torch.log(probs_ref_idx_next)
                log_probs_ref = torch.cat(
                    (log_probs_ref, log_probs_ref_idx_next), dim=1
                )

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if i == max_new_tokens - 1:
                states = idx[:, -max_new_tokens:]
                if hard_code_reward:
                    # simple test where reward for outputting the letter 'z' (89)
                    rewards = torch.zeros_like(states, dtype=torch.float16)
                    rewards[states == 89] = 1.0
                    rewards = torch.sum(rewards, 1, keepdim=True)
                    rewards[rewards > 1] = 1

                else:
                    if self.discrete_reward:
                        rewards = reward_model.forward_reward(torch.tensor(states))[0][
                            :, 1
                        ].unsqueeze(-1)
                    else:
                        rewards = reward_model.forward_reward(torch.tensor(states))

                for t in reversed(range(max_new_tokens)):
                    if t == max_new_tokens - 1:
                        # value at last state is 0
                        delta = rewards[:].squeeze() - values_all[:, t]
                        advantages_all[:, t] = delta
                        # returns_all[:, t] = rewards[:]
                    else:
                        # rewards can only be non-zero at the last state
                        delta = gamma * values_all[:, t + 1] - values_all[:, t]
                        advantages_all[:, t] = (
                            delta + gamma * lam * advantages_all[:, t + 1]
                        )
                        # returns_all[:, t] += gamma * returns_all[:, t + 1]
        return (
            idx,
            log_probs[:, -max_new_tokens:],
            log_probs_ref[:, -max_new_tokens:],
            rewards,
            advantages_all,
        )
