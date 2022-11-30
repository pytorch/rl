# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy

from typing import Tuple

import torch
from tensordict.nn import make_functional, repopulate_module
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.modules import SafeModule
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.utils import distance_loss, hold_out_params, next_state_value

from ..envs.utils import set_exploration_mode
from .common import LossModule


class DDPGLoss(LossModule):
    """The DDPG Loss class.

    Args:
        actor_network (SafeModule): a policy operator.
        value_network (SafeModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_actor (bool, optional): whether to separate the target actor networks from the actor networks used for
            data collection. Default is :obj:`False`.
        delay_value (bool, optional): whether to separate the target value networks from the value networks used for
            data collection. Default is :obj:`False`.
    """

    def __init__(
        self,
        actor_network: SafeModule,
        value_network: SafeModule,
        gamma: float,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
    ) -> None:
        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value

        actor_critic = ActorCriticWrapper(actor_network, value_network)
        params = make_functional(actor_critic)
        self.actor_critic = deepcopy(actor_critic)
        repopulate_module(actor_network, params["module", "0"])
        repopulate_module(value_network, params["module", "1"])

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=list(actor_network.parameters()),
        )
        self.actor_critic.module[0] = self.actor_network
        self.actor_critic.module[1] = self.value_network

        self.actor_in_keys = actor_network.in_keys

        self.register_buffer("gamma", torch.tensor(gamma))
        self.loss_funtion = loss_function

    def forward(self, input_tensordict: TensorDictBase) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (TensorDictBase): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of 2 tensors containing the DDPG loss.

        """
        if not input_tensordict.device == self.device:
            raise RuntimeError(
                f"Got device={input_tensordict.device} but "
                f"actor_network.device={self.device} (self.device={self.device})"
            )

        loss_value, td_error, pred_val, target_value = self._loss_value(
            input_tensordict,
        )
        td_error = td_error.detach()
        td_error = td_error.unsqueeze(input_tensordict.ndimension())
        if input_tensordict.device is not None:
            td_error = td_error.to(input_tensordict.device)
        input_tensordict.set(
            "td_error",
            td_error,
            inplace=True,
        )
        loss_actor = self._loss_actor(input_tensordict)
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
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        td_copy = tensordict.select(*self.actor_in_keys).detach()
        td_copy = self.actor_network(
            td_copy,
            params=self.actor_network_params,
        )
        with hold_out_params(self.value_network_params) as params:
            td_copy = self.value_network(
                td_copy,
                params=params,
            )
        return -td_copy.get("state_action_value")

    def _loss_value(
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        actor_critic = self.actor_critic
        target_params = TensorDict(
            {
                "module": {
                    "0": self.target_actor_network_params,
                    "1": self.target_value_network_params,
                }
            },
            batch_size=self.target_actor_network_params.batch_size,
            device=self.target_actor_network_params.device,
        )
        with set_exploration_mode("mode"):
            target_value = next_state_value(
                tensordict,
                actor_critic,
                gamma=self.gamma,
                params=target_params,
            )

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        return loss_value, (pred_val - target_value).pow(2), pred_val, target_value
