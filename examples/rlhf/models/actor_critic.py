# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.modules.tensordict_module.actors import LMActorCritic
from torchrl.modules.tensordict_module.common import VmapModule

from .transformer import init_transformer

__all__ = ["init_actor_critic"]


def init_actor_critic(transformer_name_or_path, dropout, device, compile_):
    base_model = init_transformer(
        transformer_name_or_path,
        dropout,
        device,
        as_tensordictmodule=False,
        compile_=compile_,
        inference=True,
    )
    model = LMActorCritic(base_model)
    model.to(device)
    model.eval()
    actor = model.get_policy_operator()
    critic = model.get_value_operator()
    critic_head = model.get_value_head()

    return actor, VmapModule(critic), critic_head, base_model
