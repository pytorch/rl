# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional

import torch

from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator


class ReinforceLoss(LossModule):
    """Reinforce loss module.

    Presented in "Simple statistical gradient-following algorithms for connectionist reinforcement learning", Williams, 1992
    https://doi.org/10.1007/BF00992696


    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.
        delay_value (bool, optional): if ``True``, a target network is needed
            for the critic. Defaults to ``False``.
        advantage_key (str): the input tensordict key where the advantage is
            expected to be written.
            Defaults to ``"advantage"``.
        value_target_key (str): the input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> reinforce_loss = ReinforceLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = reinforce_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = reinforce_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> reinforce_loss = ReinforceLoss(actor, critic)
        >>> reinforce_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = reinforce_loss(data)

    """

    default_value_estimator = ValueEstimators.GAE

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: Optional[TensorDictModule] = None,
        *,
        delay_value: bool = False,
        advantage_key: str = "advantage",
        value_target_key: str = "value_target",
        loss_critic_type: str = "smooth_l1",
        gamma: float = None,
    ) -> None:
        super().__init__()

        self.delay_value = delay_value
        self.advantage_key = advantage_key
        self.value_target_key = value_target_key
        self.loss_critic_type = loss_critic_type

        # Actor
        self.convert_to_functional(
            actor,
            "actor_network",
            create_target_params=False,
        )

        # Value
        if critic is not None:
            self.convert_to_functional(
                critic,
                "critic",
                create_target_params=self.delay_value,
                compare_against=list(actor.parameters()),
            )
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        advantage = tensordict.get(self.advantage_key, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
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

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        value_key = "state_value"
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.critic, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.critic, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(
                value_network=self.critic, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.critic, value_key=value_key, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
