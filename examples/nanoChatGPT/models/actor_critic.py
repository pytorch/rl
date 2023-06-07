import copy

import torch

import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions.categorical import Categorical

from torchrl.modules import (
    ActorValueOperator,
    MaskedCategorical,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)

from .transformer import init_transformer

__all__ = ["ActorCritic", "init_actor_critic"]


class Masker(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, logits):
        _, top_indices = torch.topk(logits, k=self.k, sorted=False)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.arange(logits.shape[0])[:, None], top_indices] = True
        return mask


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
            TensorDictModule(
                base_model,
                in_keys={"input_ids": "input_ids", "attention_mask": "attention_mask"},
                out_keys=["x"],
            ),
            TensorDictModule(lambda x: x[:, -1, :], in_keys=["x"], out_keys=["x"]),
        )
        masker = TensorDictModule(Masker(k=50), in_keys=["logits"], out_keys=["mask"])
        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["logits"])
        actor_head = SafeProbabilisticTensorDictSequential(
            actor_head,
            masker,
            SafeProbabilisticModule(
                in_keys=["logits", "mask"],
                out_keys=["action"],
                distribution_class=MaskedCategorical,
                return_log_prob=True,
            ),
        )
        value_head = TensorDictModule(
            value_head, in_keys=["x"], out_keys=["state_value"]
        )

        super().__init__(common, actor_head, value_head)


def init_actor_critic(config):
    base_model = init_transformer(
        config, as_tensordictmodule=False, skip_compilation=True, inference=True
    )
    a2c_model = ActorCritic(base_model)
    a2c_model.to(config["device"])
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()
    critic_head = a2c_model.get_value_head()

    return actor, critic, critic_head
