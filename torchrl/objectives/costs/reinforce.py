from typing import Optional

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.modules import ProbabilisticTDModule, TDModule
from torchrl.objectives import GAE
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.returns.a2c import A2C


class ReinforceLoss(_LossModule):
    def __init__(
        self,
        actor_network: ProbabilisticTDModule,
        advantage_module: str,
        value_network: Optional[TDModule] = None,
        gamma: float = 0.99,
        lamda: float = 0.95,
        delay_value: bool = False,
    ) -> None:
        super().__init__()

        self.delay_value = delay_value

        # Actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
        )

        # Value
        if value_network is not None:
            self.convert_to_functional(
                value_network,
                "value_network",
                create_target_params=self.delay_value,
            )

        if advantage_module == "gae":
            self.advantage_module = GAE(
                gamma,
                lamda,
                self.value_network,
                gradient_mode=True)
        elif advantage_module == "a2c":
            self.advantage_module = A2C(
                gamma,
                self.value_network,
                gradient_mode=True)
        else:
            raise NotImplementedError

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        # get advantage
        tensordict = self.advantage_module(
            tensordict,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
            target_params=self.target_value_network_params,
            target_buffers=self.target_value_network_buffers,
        )
        advantage = tensordict.get("advantage")
        value_target = tensordict.get("value_target")

        # compute log-prob
        tensordict = self.actor_network(
            tensordict,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers
        )

        log_prob = tensordict.get("action_log_prob")
        loss_actor = - log_prob * advantage.detach()
        loss_actor = loss_actor.mean()
        td_out = TensorDict({'loss_actor': loss_actor}, [])
        loss_value = value_target.pow(2).mean()
        td_out.set("loss_value", loss_value)

        return td_out
