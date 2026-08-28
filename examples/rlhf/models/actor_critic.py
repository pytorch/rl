# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from types import SimpleNamespace

from torch import nn
from torchrl.modules.tensordict_module.actors import LMHeadActorValueOperator
from torchrl.modules.tensordict_module.common import VmapModule

from .transformer import init_transformer

__all__ = ["init_actor_critic"]


class _ActorValueTransformer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, attention_mask):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )


def init_actor_critic(model_cfg, sys_cfg):

    transformer_name_or_path = model_cfg.name_or_path
    dropout = model_cfg.dropout

    device = sys_cfg.device
    compile_model = sys_cfg.compile
    base_model = init_transformer(
        transformer_name_or_path,
        dropout,
        device,
        as_tensordictmodule=False,
        compile_model=compile_model,
        inference=True,
    )
    model = LMHeadActorValueOperator(
        SimpleNamespace(
            transformer=_ActorValueTransformer(base_model.transformer),
            lm_head=base_model.lm_head,
        )
    )
    # Recent Transformers releases require structured transformer outputs in
    # generation, while the actor-value operator consumes its tuple form.
    base_model.config.return_dict = True
    model.to(device)
    model.eval()
    actor = model.get_policy_operator()
    critic = model.get_value_operator()
    critic_head = model.get_value_head()

    return actor, VmapModule(critic, mock=True), critic_head, base_model
