import copy

import torch

import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions.categorical import Categorical

from torchrl.modules import (
    ActorValueOperator,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)

from .transformer import init_transformer

__all__ = ["ActorCritic", "init_actor_critic"]


class ActorCritic(ActorValueOperator):
    def __init__(self, base_model):
        base_model = copy.deepcopy(base_model)
        n_embd = base_model.lm_head.in_features

        # actor network
        # extract last layer to be reused by actor
        actor_head = base_model.lm_head
        base_model.lm_head = nn.Identity()

        # critic network
        value_head = nn.Linear(n_embd, 1, bias=False)

        common = TensorDictSequential(
            TensorDictModule(base_model, in_keys=["prompt"], out_keys=["x"]),
            TensorDictModule(lambda x: x[:, -1, :], in_keys=["x"], out_keys=["x"]),
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


def init_actor_critic(config):
    model_base = init_transformer(
        config, as_tensordictmodule=False, skip_compilation=True
    )
    a2c_model = ActorCritic(model_base)
    a2c_model.to(config["device"])
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()
    critic_head = a2c_model.get_value_head()

    return actor, critic, critic_head
