from typing import Optional

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import ProbabilisticTDModule, TDModule
from torchrl.objectives import GAE, distance_loss
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.returns.a2c import A2C


class ReinforceLoss(_LossModule):
    def __init__(
        self,
        actor_network: ProbabilisticTDModule,
        advantage_module: str,
        critic: Optional[TDModule] = None,
        gamma: float = 0.99,
        lamda: float = 0.95,
        delay_value: bool = False,
        advantage_key: str = "advantage",
        advantage_diff_key: str = "advantage_diff",
        loss_critic_type: str = "smooth_l1",
    ) -> None:
        super().__init__()

        self.delay_value = delay_value
        self.advantage_key = advantage_key
        self.advantage_diff_key = advantage_diff_key
        self.loss_critic_type = loss_critic_type

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
            )

        if advantage_module == "gae":
            self.advantage_module = GAE(
                gamma,
                lamda,
                value_network=self.critic,
                gradient_mode=True
            )
        elif advantage_module == "a2c":
            self.advantage_module = A2C(
                gamma,
                value_network=self.critic,
                gradient_mode=True
            )
        else:
            raise NotImplementedError

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        # get advantage
        tensordict = self.advantage_module(
            tensordict,
            params=self.critic_params,
            buffers=self.critic_buffers,
            target_params=self.target_critic_params,
            target_buffers=self.target_critic_buffers,
        )
        advantage = tensordict.get(self.advantage_key)

        # compute log-prob
        tensordict = self.actor_network(
            tensordict,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers,
        )

        log_prob = tensordict.get("action_log_prob")
        loss_actor = -log_prob * advantage.detach()
        loss_actor = loss_actor.mean()
        td_out = TensorDict({"loss_actor": loss_actor}, [])

        td_out.set("loss_value", self.loss_critic(tensordict).mean())

        return td_out

    def loss_critic(self, tensordict: _TensorDict) -> torch.Tensor:
        if self.advantage_diff_key in tensordict.keys():
            advantage_diff = tensordict.get(self.advantage_diff_key)
            if not advantage_diff.requires_grad:
                raise RuntimeError(
                    "value_target retrieved from tensordict does not requires grad."
                )
            loss_value = distance_loss(
                advantage_diff, torch.zeros_like(advantage_diff),
                loss_function=self.loss_critic_type
            )
        else:
            with torch.no_grad():
                reward = tensordict.get("reward")
                next_td = step_tensordict(tensordict)
                next_value = self.critic(
                    next_td,
                    params=self.critic_params,
                    buffers=self.critic_buffers,
                ).get("state_value")
                value_target = reward + next_value * self.gamma
            tensordict_select = tensordict.select(*self.critic.in_keys).clone()
            value = self.critic(
                tensordict_select,
                params=self.critic_params,
                buffers=self.critic_buffers,
            ).get("state_value")

            loss_value = distance_loss(
                value, value_target, loss_function=self.loss_critic_type
            )
        return loss_value
