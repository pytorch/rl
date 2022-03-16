from __future__ import annotations

from typing import Tuple

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.modules import TDModule
from torchrl.modules.td_module.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import (
    distance_loss,
    hold_out_params,
    next_state_value,
)
from .common import _LossModule


class DDPGLoss(_LossModule):
    """
    The DDPG Loss class.
    Args:
        actor_network (TDModule): a policy operator.
        value_network (TDModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
    """

    delay_actor: bool = False
    delay_value: bool = False

    def __init__(
        self,
        actor_network: TDModule,
        value_network: TDModule,
        gamma: float,
        loss_function: str = "l2",
    ) -> None:
        super().__init__()
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function

    def forward(self, input_tensor_dict: _TensorDict) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensor_dict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns: a tuple of 2 tensors containing the DDPG loss.

        """
        if not input_tensor_dict.device == self.device:
            raise RuntimeError(
                f"Got device={input_tensor_dict.device} but actor_network.device={self.device} "
                f"(self.device={self.device})"
            )

        loss_value, td_error, pred_val, target_value = self._loss_value(
            input_tensor_dict,
        )
        td_error = td_error.detach()
        td_error = td_error.unsqueeze(input_tensor_dict.ndimension())
        td_error = td_error.to(input_tensor_dict.device)
        input_tensor_dict.set(
            "td_error",
            td_error,
            inplace=True,
        )
        loss_actor = self._loss_actor(input_tensor_dict)
        return TensorDict(
            source={
                "loss_actor": loss_actor.mean(),
                "loss_value": loss_value.mean(),
                "pred_value": pred_val.mean().detach(),
                "target_value": target_value.mean().detach(),
                "pred_value_max": pred_val.max().detach(),
                "target_value_max": target_value.max().detach(),
            },
            batch_size=[],
        )

    def _loss_actor(
        self,
        tensor_dict: _TensorDict,
    ) -> torch.Tensor:
        td_copy = tensor_dict.select(*self.actor_in_keys).detach()
        td_copy = self.actor_network(
            td_copy,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers,
        )
        with hold_out_params(self.value_network_params) as params:
            td_copy = self.value_network(
                td_copy, params=params, buffers=self.value_network_buffers
            )
        return -td_copy.get("state_action_value")

    def _loss_value(
        self,
        tensor_dict: _TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # value loss
        td_copy = tensor_dict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        actor_critic = ActorCriticWrapper(
            self.actor_network, self.value_network
        )
        target_params = list(self.target_actor_network_params) + list(
            self.target_value_network_params
        )
        target_buffers = list(self.target_actor_network_buffers) + list(
            self.target_value_network_buffers
        )
        target_value = next_state_value(
            tensor_dict,
            actor_critic,
            gamma=self.gamma,
            params=target_params,
            buffers=target_buffers,
        )

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        return loss_value, abs(pred_val - target_value), pred_val, target_value


class DoubleDDPGLoss(DDPGLoss):
    """
    A Double DDPG loss class.
    As for Double DQN loss, this class separates the target value/actor networks from the value/actor networks used for
    data collection. Those target networks should be updated from their original counterparts with some delay using
    dedicated classes (SoftUpdate and HardUpdate in objectives.cost.utils).
    Note that the original networks will be copied at initialization using the copy.deepcopy method: in some rare cases
    this may lead to unexpected behaviours (for instance if the networks change in a way that won't be reflected by their
    state_dict). Please report any such bug if encountered.

    """

    delay_actor: bool = True
    delay_value: bool = True
