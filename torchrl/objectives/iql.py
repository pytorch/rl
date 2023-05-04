# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional, Tuple

import torch
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import Tensor

from torchrl.modules import ProbabilisticActor
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator

try:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    _has_functorch = True
    err = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERROR = err


class IQLLoss(LossModule):
    r"""TorchRL implementation of the IQL loss.

    Presented in "Offline Reinforcement Learning with Implicit Q-Learning" https://arxiv.org/abs/2110.06169

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model
        value_network (TensorDictModule, optional): V(s) parametric model.
        num_qvalue_nets (integer, optional): number of Q-Value networks used.
            Defaults to ``2``.
        priority_key (str, optional): tensordict key where to write the
            priority (for prioritized replay buffer usage). Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        temperature (float, optional):  Inverse temperature (beta).
            For smaller hyperparameter values, the objective behaves similarly to
            behavioral cloning, while for larger values, it attempts to recover the
            maximum of the Q-function.
        expectile (float, optional): expectile :math:`\tau`. A larger value of :math:`\tau` is crucial
            for antmaze tasks that require dynamical programming ("stichting").

    """

    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        value_network: Optional[TensorDictModule],
        *,
        num_qvalue_nets: int = 2,
        priority_key: str = "td_error",
        loss_function: str = "smooth_l1",
        temperature: float = 1.0,
        expectile: float = 0.5,
        gamma: float = None,
    ) -> None:
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()

        # IQL parameter
        self.temperature = temperature
        self.expectile = expectile

        # Actor Network
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )

        # Value Function Network
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=False,
            compare_against=list(actor_network.parameters()),
        )

        # Q Function Network
        self.delay_qvalue = True
        self.num_qvalue_nets = num_qvalue_nets

        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=True,
            compare_against=list(actor_network.parameters())
            + list(value_network.parameters()),
        )

        self.priority_key = priority_key
        self.loss_function = loss_function
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    @staticmethod
    def loss_value_diff(diff, expectile=0.8):
        """Loss function for iql expectile value difference."""
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        device = self.device
        td_device = tensordict_reshape.to(device)

        loss_actor = self._loss_actor(td_device)
        loss_qvalue, priority = self._loss_qvalue(td_device)
        loss_value = self._loss_value(td_device)

        tensordict_reshape.set(self.priority_key, priority)
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        out = {
            "loss_actor": loss_actor.mean(),
            "loss_qvalue": loss_qvalue.mean(),
            "loss_value": loss_value.mean(),
            "entropy": -td_device.get("_log_prob").mean().detach(),
        }

        return TensorDict(
            out,
            [],
        )

    def _loss_actor(self, tensordict: TensorDictBase) -> Tensor:
        # KL loss
        dist = self.actor_network.get_dist(
            tensordict,
            params=self.actor_network_params,
        )

        log_prob = dist.log_prob(tensordict["action"])

        # Min Q value
        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q = vmap(self.qvalue_network, (None, 0))(
            td_q, self.target_qvalue_network_params
        )
        min_q = td_q.get("state_action_value").min(0)[0].squeeze(-1)

        if log_prob.shape != min_q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q.shape}"
            )
        # state value
        with torch.no_grad():
            td_copy = tensordict.select(*self.value_network.in_keys).detach()
            self.value_network(
                td_copy,
                params=self.value_network_params,
            )
            value = td_copy.get("state_value").squeeze(-1)  # assert has no gradient

        exp_a = torch.exp((min_q - value) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device))

        # write log_prob in tensordict for alpha loss
        tensordict.set("_log_prob", log_prob.detach())
        return -(exp_a * log_prob).mean()

    def _loss_value(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        # Min Q value
        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q = vmap(self.qvalue_network, (None, 0))(
            td_q, self.target_qvalue_network_params
        )
        min_q = td_q.get("state_action_value").min(0)[0].squeeze(-1)
        # state value
        td_copy = tensordict.select(*self.value_network.in_keys)
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )
        value = td_copy.get("state_value").squeeze(-1)
        value_loss = self.loss_value_diff(min_q - value, self.expectile).mean()
        return value_loss

    def _loss_qvalue(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        obs_keys = self.actor_network.in_keys
        tensordict = tensordict.select("next", *obs_keys, "action")

        target_value = self.value_estimator.value_estimate(
            tensordict, target_params=self.target_value_network_params
        ).squeeze(-1)
        tensordict_expand = vmap(self.qvalue_network, (None, 0))(
            tensordict.select(*self.qvalue_network.in_keys),
            self.qvalue_network_params,
        )
        pred_val = tensordict_expand.get("state_action_value").squeeze(-1)
        td_error = abs(pred_val - target_value)
        loss_qval = (
            distance_loss(
                pred_val,
                target_value.expand_as(pred_val),
                loss_function=self.loss_function,
            )
            .sum(0)
            .mean()
        )
        return loss_qval, td_error.detach().max(0)[0]

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        value_net = self.value_network

        value_key = "state_value"
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
