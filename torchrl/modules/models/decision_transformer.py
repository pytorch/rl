# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class DecisionTransformer(nn.Module):
    """Online Decion Transformer.

    Desdescribed in https://arxiv.org/abs/2202.05607 .

    Args:
        state_dim (int): dimension of the state space
        action_dim (int): dimension of the action space
        config (dict): transformer architecture configuration, used to create the GPT2Config from transformers.


    Example:
        >>> config = {
        >>>     "n_embd": 256,
        >>>     "n_layer": 4,
        >>>     "n_head": 4,
        >>>     "n_inner": 1024,
        >>>     "activation": "relu",
        >>>     "n_positions": 1024,
        >>>     "resid_pdrop": 0.1,
        >>>     "attn_pdrop": 0.1,
        >>> }
        >>> model = DecisionTransformer(state_dim=4, action_dim=2, config=config)
        >>> observation = torch.randn(32, 10, 4)
        >>> action = torch.randn(32, 10, 2)
        >>> return_to_go = torch.randn(32, 10, 1)
        >>> output = model(observation, action, return_to_go)
        >>> output.shape
        torch.Size([32, 10, 256])


    """

    def __init__(
        self,
        state_dim,
        action_dim,
        config,
    ):
        super(DecisionTransformer, self).__init__()

        gpt_config = transformers.GPT2Config(
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=["config.n_inner"],
            activation_function=config["activation"],
            n_positions=config["n_positions"],
            resid_pdrop=config["resid_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            vocab_size=1,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = config["n_embd"]

        self.transformer = GPT2Model(config=gpt_config)

        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.action_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
    ):
        batch_size, seq_length = observation.shape[0], observation.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(observation)
        action_embeddings = self.embed_action(action)
        returns_embeddings = self.embed_return(return_to_go)

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

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return x[:, 1]  # only state tokens
