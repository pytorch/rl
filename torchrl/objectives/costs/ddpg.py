from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Tuple

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.modules import TDModule, reset_noise, TDModule
from torchrl.modules.td_module.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import distance_loss, next_state_value, hold_out_net
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

    def __init__(
        self,
        actor_network: TDModule,
        value_network: TDModule,
        gamma: float,
        loss_function: str = "l2",
    ) -> None:
        super().__init__()
        self.value_network = value_network
        self.actor_network = actor_network
        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function

    def _get_networks(
        self,
    ) -> Tuple[TDModule, TDModule, TDModule, TDModule,]:
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self.actor_network
        target_value_network = self.value_network
        return actor_network, value_network, target_actor_network, target_value_network

    def forward(self, input_tensor_dict: _TensorDict) -> TensorDict:
        """
        Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensor_dict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns: a tuple of 2 tensors containing the DDPG loss.

        """
        (
            actor_network,
            value_network,
            target_actor_network,
            target_value_network,
        ) = self._get_networks()
        if not input_tensor_dict.device == actor_network.device:
            raise RuntimeError(
                f"Got device={input_tensor_dict.device} but actor_network.device={actor_network.device} "
                f"(self.device={self.device})"
            )
        if not input_tensor_dict.device == value_network.device:
            raise RuntimeError(
                f"Got device={input_tensor_dict.device} but value_network.device={value_network.device} "
                f"(self.device={self.device})"
            )

        loss_value, td_error, pred_val, target_value = self._loss_value(
            input_tensor_dict,
            value_network,
            target_actor_network,
            target_value_network,
        )
        input_tensor_dict.set(
            "td_error",
            td_error.detach()
            .unsqueeze(input_tensor_dict.ndimension())
            .to(input_tensor_dict.device),
            inplace=True,
        )
        loss_actor = self._loss_actor(input_tensor_dict, actor_network, value_network)
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

    @property
    def target_value_network(self) -> TDModule:
        return self.value_network

    @property
    def target_actor_network(self) -> TDModule:
        return self.actor_network

    def _loss_actor(
        self,
        tensor_dict: _TensorDict,
        actor_network: TDModule,
        value_network: TDModule,
    ) -> torch.Tensor:
        td_copy = tensor_dict.select(*self.actor_in_keys).detach()
        td_copy = actor_network(td_copy)
        with hold_out_net(value_network):
            td_copy = value_network(td_copy)
        return -td_copy.get("state_action_value")

    def _loss_value(
        self,
        tensor_dict: _TensorDict,
        value_network: TDModule,
        target_actor_network: TDModule,
        target_value_network: TDModule,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # value loss
        td_copy = tensor_dict.select(*value_network.in_keys).detach()
        value_network(td_copy)
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        actor_critic = ActorCriticWrapper(target_actor_network, target_value_network)
        target_value = next_state_value(tensor_dict, actor_critic, gamma=self.gamma)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self._target_actor_network = deepcopy(self.actor_network)
        self._target_actor_network.requires_grad_(False)

    def _get_networks(
        self,
    ) -> Tuple[TDModule, TDModule, TDModule, TDModule,]:
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self.target_actor_network
        target_value_network = self.target_value_network
        return actor_network, value_network, target_actor_network, target_value_network

    @property
    def target_value_network(self) -> TDModule:
        self._target_value_network.apply(reset_noise)
        return self._target_value_network

    @property
    def target_actor_network(self) -> TDModule:
        self._target_actor_network.apply(reset_noise)
        return self._target_actor_network
