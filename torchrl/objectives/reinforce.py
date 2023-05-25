# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
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
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        advantage_key (str): [Deprecated, use .set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is expected to be written.
            Defaults to ``"advantage"``.
        value_target_key (str): [Deprecated, use .set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.

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

    @dataclass
    class _AcceptedKeys:
        advantage_key: NestedKey = "advantage"
        value_target_key: NestedKey = "value_target"
        value_key: NestedKey = "state_value"
        sample_log_prob_key: NestedKey = "sample_log_prob"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.GAE

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._tensor_keys = cls._AcceptedKeys()
        return super().__new__(cls)

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: Optional[TensorDictModule] = None,
        *,
        delay_value: bool = False,
        loss_critic_type: str = "smooth_l1",
        gamma: float = None,
        advantage_key: str = None,
        value_target_key: str = None,
    ) -> None:
        super().__init__()
        self._set_deprecated_ctor_keys(
            advantage_key=advantage_key, value_target_key=value_target_key
        )

        self.delay_value = delay_value
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

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        return self._tensor_keys

    def set_keys(self, **kwargs) -> None:
        """TODO"""
        for key, _ in kwargs.items():
            if key not in self._AcceptedKeys.__dict__:
                raise ValueError(f"{key} it not an accepted tensordict key")
        self._tensor_keys = self._AcceptedKeys(**kwargs)

        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage_key=self._tensor_keys.advantage_key,
                value_target_key=self._tensor_keys.value_target_key,
                value_key=self._tensor_keys.value_key,
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        advantage = tensordict.get(self.tensor_keys.advantage_key, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage_key)

        # compute log-prob
        tensordict = self.actor_network(
            tensordict,
            params=self.actor_network_params,
        )

        log_prob = tensordict.get(self.tensor_keys.sample_log_prob_key)
        loss_actor = -log_prob * advantage.detach()
        loss_actor = loss_actor.mean()
        td_out = TensorDict({"loss_actor": loss_actor}, [])

        td_out.set("loss_value", self.loss_critic(tensordict).mean())

        return td_out

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        try:
            target_return = tensordict.get(self.tensor_keys.value_target_key)
            tensordict_select = tensordict.select(*self.critic.in_keys)
            state_value = self.critic(
                tensordict_select,
                params=self.critic_params,
            ).get(self.tensor_keys.value_key)
            loss_value = distance_loss(
                target_return,
                state_value,
                loss_function=self.loss_critic_type,
            )
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.loss_key('value_target_key')} was not found in the input tensordict. "
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
        ve_args = {
            "value_network": self.critic,
            "advantage_key": self.tensor_keys.advantage_key,
            "value_key": self.tensor_keys.value_key,
            "value_target_key": self.tensor_keys.value_target_key,
        }
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(**ve_args, **hp)
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(**ve_args, **hp)
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(**ve_args, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(**ve_args, **hp)
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
