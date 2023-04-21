from typing import Optional

import torch
import torch.nn as nn
import transformers
from torchrl.modules.models.gpt2_transformer import GPT2Model


class DecisionTransformer(nn.Module):
    """Decion Transformer as described in https://arxiv.org/abs/2202.05607 ."""

    def __init__(
        self, state_dim, action_dim, hidden_size=512, max_ep_len=1000, ordering=False
    ):
        super(DecisionTransformer, self).__init__()
        assert hidden_size == 512, "Only hidden_size=512 is supported"
        gpt_config = transformers.GPT2Config(
            n_embd=512,
            n_layer=4,
            n_head=4,
            n_inner=4 * 512,
            activation_function="relu",
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            vocab_size=1,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.ordering = ordering
        self.train_context = 20
        self.inference_context = 5

        self.transformer = GPT2Model(config=gpt_config)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.action_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = observation.shape[0], observation.shape[1]

        if seq_length == self.inference_context:
            observation, action, return_to_go, timesteps, seq_length = self.pad_context(
                observation, action, return_to_go, timesteps
            )

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(observation)
        action_embeddings = self.embed_action(action)
        returns_embeddings = self.embed_return(return_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return x[:, 1]  # only state tokens

    def pad_context(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        observation = torch.nn.functional.pad(
            observation,
            (0, 0, self.train_context - self.inference_context, 0),
            mode="constant",
            value=0,
        )
        action = torch.nn.functional.pad(
            action,
            (0, 0, self.train_context - self.inference_context - 1, 1),
            mode="constant",
            value=0,
        )  # pad first action with 0
        return_to_go = torch.nn.functional.pad(
            return_to_go,
            (0, 0, self.train_context - self.inference_context - 1, 1),
            mode="constant",
            value=0,
        )
        timesteps = torch.nn.functional.pad(
            timesteps,
            (0, 0, self.train_context - self.inference_context, 0),
            mode="constant",
            value=0,
        )
        return observation, action, return_to_go, timesteps, self.train_context
