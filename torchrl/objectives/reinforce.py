from typing import Callable, Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.envs.utils import step_mdp
from torchrl.modules import SafeModule, SafeProbabilisticModule
from torchrl.objectives import distance_loss
from torchrl.objectives.common import LossModule


class ReinforceLoss(LossModule):
    """Reinforce loss module.

    Presented in "Simple statistical gradient-following algorithms for connectionist reinforcement learning", Williams, 1992
    https://doi.org/10.1007/BF00992696

    """

    def __init__(
        self,
        actor_network: SafeProbabilisticModule,
        advantage_module: Callable[[TensorDictBase], TensorDictBase],
        critic: Optional[SafeModule] = None,
        delay_value: bool = False,
        gamma: float = 0.99,
        advantage_key: str = "advantage",
        advantage_diff_key: str = "value_error",
        loss_critic_type: str = "smooth_l1",
    ) -> None:
        super().__init__()

        self.delay_value = delay_value
        self.advantage_key = advantage_key
        self.advantage_diff_key = advantage_diff_key
        self.loss_critic_type = loss_critic_type
        self.register_buffer("gamma", torch.tensor(gamma))

        if (
            hasattr(advantage_module, "is_functional")
            and not advantage_module.is_functional
        ):
            raise RuntimeError(
                "The advantage module must be functional, as it must support params and target params arguments"
            )

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

        self.advantage_module = advantage_module

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # get advantage
        tensordict = self.advantage_module(
            tensordict,
            params=self.critic_params,
            target_params=self.target_critic_params,
        )
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
        if self.advantage_diff_key in tensordict.keys():
            advantage_diff = tensordict.get(self.advantage_diff_key)
            if not advantage_diff.requires_grad:
                raise RuntimeError(
                    "value_target retrieved from tensordict does not requires grad."
                )
            loss_value = distance_loss(
                advantage_diff,
                torch.zeros_like(advantage_diff),
                loss_function=self.loss_critic_type,
            )
        else:
            with torch.no_grad():
                reward = tensordict.get("reward")
                next_td = step_mdp(tensordict)
                next_value = self.critic(
                    next_td,
                    params=self.critic_params,
                ).get("state_value")
                value_target = reward + next_value * self.gamma
            tensordict_select = tensordict.select(*self.critic.in_keys).clone()
            value = self.critic(
                tensordict_select,
                params=self.critic_params,
            ).get("state_value")

            loss_value = distance_loss(
                value, value_target, loss_function=self.loss_critic_type
            )
        return loss_value
