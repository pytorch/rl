# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torch.distributions.categorical import Categorical

from torchrl.modules import (
    ActorValueOperator,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)

from .transformer import init_transformer

__all__ = ["ActorCritic", "init_actor_critic"]


class VmapCritic(TensorDictModuleBase):
    def __init__(self, critic):
        super().__init__()
        self.in_keys = critic.in_keys
        self.out_keys = critic.out_keys
        self.module = critic

    def forward(self, tensordict):
        ndim = tensordict.ndim
        training = self.module.training
        self.module.eval()
        td = torch.vmap(self.module, (ndim - 1,))(tensordict)
        self.module.train(training)
        # vmap sends this dim to the beginning so we need to send it back where it belongs
        td = td.permute(*range(1, ndim), 0)
        return tensordict.update(td)


class ActorCritic(ActorValueOperator):
    def __init__(self, base_model):
        actor_head = base_model.lm_head
        value_head = nn.Linear(actor_head.in_features, 1, bias=False)

        common = TensorDictSequential(
            TensorDictModule(
                base_model.transformer,
                in_keys={"input_ids": "input_ids", "attention_mask": "attention_mask"},
                out_keys=["x"],
            ),
            TensorDictModule(lambda x: x[:, -1], in_keys=["x"], out_keys=["x"]),
        )
        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["logits"])
        actor_head = SafeProbabilisticTensorDictSequential(
            actor_head,
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=Categorical,
                return_log_prob=True,
            ),
        )
        value_head = TensorDictModule(
            value_head, in_keys=["x"], out_keys=["state_value"]
        )

        super().__init__(common, actor_head, value_head)


def init_actor_critic(transformer_name_or_path, dropout, device, compile_):
    base_model = init_transformer(
        transformer_name_or_path,
        dropout,
        device,
        as_tensordictmodule=False,
        compile_=compile_,
        inference=True,
    )
    a2c_model = ActorCritic(base_model)
    a2c_model.to(device)
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()
    critic_head = a2c_model.get_value_head()

    return actor, VmapCritic(critic), critic_head, base_model
