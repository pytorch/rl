# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import dataclasses

import importlib
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

_has_transformers = importlib.util.find_spec("transformers") is not None
import transformers
from transformers.models.gpt2.modeling_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    GPT2Model,
)


class ModifiedGPT2Model(GPT2Model):
    """Wrapper around the GPT2Model from transformers.

    This class is a modified version of the GPT2Model from transformers
    as for the Decision Transformer we dont need the wpe layer.

    """

    def __init__(self, config):
        super(ModifiedGPT2Model, self).__init__(config)

        # Remove the wpe layer
        del self.wpe

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        input_shape = inputs_embeds.size()[:-1]

        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        use_cache = self.config.use_cache
        return_dict = self.config.use_return_dict

        head_mask = self.get_head_mask(None, self.config.n_layer)

        hidden_states = inputs_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        past_key_values = tuple([None] * len(self.h))
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class DecisionTransformer(nn.Module):
    """Online Decion Transformer.

    Desdescribed in https://arxiv.org/abs/2202.05607 .

    The transformer utilizes a default config to create the GPT2 model if the user does not provide a specific config.
    default_config = {
        "n_embd": 256,
        "n_layer": 4,
        "n_head": 4,
        "n_inner": 1024,
        "activation": "relu",
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1,
    }

    Args:
        state_dim (int): dimension of the state space
        action_dim (int): dimension of the action space
        config (:obj:`~.DTConfig` or dict, optional): transformer architecture configuration,
            used to create the GPT2Config from transformers.
            Defaults to :obj:`~.default_config`.


    Example:
        >>> config = DecisionTransformer.default_config()
        >>> config.n_embd = 128
        >>> print(config)
        DTConfig(n_embd: 128, n_layer: 4, n_head: 4, n_inner: 1024, activation: relu, n_positions: 1024, resid_pdrop: 0.1, attn_pdrop: 0.1)
        >>> # alternatively
        >>> config = DecisionTransformer.DTConfig(n_embd=128)
        >>> model = DecisionTransformer(state_dim=4, action_dim=2, config=config)
        >>> batch_size = [3, 32]
        >>> length = 10
        >>> observation = torch.randn(*batch_size, length, 4)
        >>> action = torch.randn(*batch_size, length, 2)
        >>> return_to_go = torch.randn(*batch_size, length, 1)
        >>> output = model(observation, action, return_to_go)
        >>> output.shape
        torch.Size([3, 32, 10, 128])

    """

    @dataclass
    class DTConfig:
        """Default configuration for DecisionTransformer."""

        n_embd: Any = 256
        n_layer: Any = 4
        n_head: Any = 4
        n_inner: Any = 1024
        activation: Any = "relu"
        n_positions: Any = 1024
        resid_pdrop: Any = 0.1
        attn_pdrop: Any = 0.1

        def __repr__(self):
            fields = []
            for f in dataclasses.fields(self):
                value = getattr(self, f.name)
                fields.append(f"{f.name}: {value}")
            fields = ", ".join(fields)
            return f"{self.__class__.__name__}({fields})"

    @classmethod
    def default_config(cls):
        return cls.DTConfig()

    def __init__(
        self,
        state_dim,
        action_dim,
        config: dict | DTConfig = None,
    ):
        if not _has_transformers:
            raise ImportError(
                "transformers is not installed. Please install it with `pip install transformers`."
            )

        if config is None:
            config = self.default_config()
        if isinstance(config, self.DTConfig):
            config = dataclasses.asdict(config)
        if not isinstance(config, dict):
            try:
                config = dict(config)
            except Exception as err:
                raise TypeError(
                    f"Config of type {type(config)} is not supported."
                ) from err

        super(DecisionTransformer, self).__init__()

        gpt_config = transformers.GPT2Config(
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=config["n_inner"],
            activation_function=config["activation"],
            n_positions=config["n_positions"],
            resid_pdrop=config["resid_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            vocab_size=1,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = config["n_embd"]

        self.transformer = ModifiedGPT2Model(config=gpt_config)

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
        batch_size, seq_length = observation.shape[:-2], observation.shape[-2]
        batch_size_orig = batch_size
        if len(batch_size) != 1:
            # TODO: vmap over transformer once this is possible
            observation = observation.view(-1, *observation.shape[-2:])
            action = action.view(-1, *action.shape[-2:])
            return_to_go = return_to_go.view(-1, *return_to_go.shape[-2:])
            batch_size = torch.Size([batch_size.numel()])

        # embed each modality with a different head
        state_embeddings = self.embed_state(observation)
        action_embeddings = self.embed_action(action)
        returns_embeddings = self.embed_return(return_to_go)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=-3
            )
            .permute(*range(len(batch_size)), -2, -3, -1)
            .reshape(*batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(*batch_size, seq_length, 3, self.hidden_size).permute(
            *range(len(batch_size)), -2, -3, -1
        )
        if batch_size_orig is batch_size:
            return x[..., 1, :, :]  # only state tokens
        return x[..., 1, :, :].view(*batch_size_orig, *x.shape[-2:])
