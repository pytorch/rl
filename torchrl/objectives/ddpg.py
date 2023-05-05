# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from copy import deepcopy

from typing import Tuple

import torch
from tensordict.nn import make_functional, repopulate_module, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    hold_out_params,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


class DDPGLoss(LossModule):
    """The DDPG Loss class.

    Args:
        actor_network (TensorDictModule): a policy operator.
        value_network (TensorDictModule): a Q value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_actor (bool, optional): whether to separate the target actor networks from the actor networks used for
            data collection. Default is ``False``.
        delay_value (bool, optional): whether to separate the target value networks from the value networks used for
            data collection. Default is ``False``.
    """

    default_value_estimator: ValueEstimators = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        *,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
        gamma: float = None,
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

        self.loss_funtion = loss_function

        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

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
        target_value = self.value_estimator.value_estimate(
            tensordict, target_params=target_params
        ).squeeze(-1)

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        return loss_value, (pred_val - target_value).pow(2), pred_val, target_value

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        value_key = "state_action_value"
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.actor_critic, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.actor_critic, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.actor_critic, value_key=value_key, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
