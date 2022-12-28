from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.modules import SafeModule, SafeProbabilisticSequential
from torchrl.objectives import distance_loss
from torchrl.objectives.common import LossModule


class ReinforceLoss(LossModule):
    """Reinforce loss module.

    Presented in "Simple statistical gradient-following algorithms for connectionist reinforcement learning", Williams, 1992
    https://doi.org/10.1007/BF00992696

    """

    def __init__(
        self,
        actor_network: SafeProbabilisticSequential,
        critic: Optional[SafeModule] = None,
        delay_value: bool = False,
        gamma: float = 0.99,
        advantage_key: str = "advantage",
        value_target_key: str = "value_target",
        loss_critic_type: str = "smooth_l1",
    ) -> None:
        super().__init__()

        self.delay_value = delay_value
        self.advantage_key = advantage_key
        self.value_target_key = value_target_key
        self.loss_critic_type = loss_critic_type
        self.register_buffer("gamma", torch.tensor(gamma))

        # Actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
        )

        # Value
        if critic is not None:
            self.convert_to_functional(
                critic,
                "critic",
                create_target_params=self.delay_value,
                compare_against=list(actor_network.parameters()),
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        advantage = tensordict.get(self.advantage_key)

        # compute log-prob
        tensordict = self.actor_network(
            tensordict,
            params=self.actor_network_params,
        )

        log_prob = tensordict.get("sample_log_prob")
        loss_actor = -log_prob * advantage.detach()
        loss_actor = loss_actor.mean()
        td_out = TensorDict({"loss_actor": loss_actor}, [])

        td_out.set("loss_value", self.loss_critic(tensordict).mean())

        return td_out

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        try:
            target_return = tensordict.get(self.value_target_key)
            tensordict_select = tensordict.select(*self.critic.in_keys)
            state_value = self.critic(
                tensordict_select,
                params=self.critic_params,
            ).get("state_value")
            loss_value = distance_loss(
                target_return,
                state_value,
                loss_function=self.loss_critic_type,
            )
        except KeyError:
            raise KeyError(
                f"the key {self.value_target_key} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )
        return loss_value
