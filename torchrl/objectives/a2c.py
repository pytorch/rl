# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Tuple

import torch
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import distributions as d

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator


class A2CLoss(LossModule):
    """TorchRL implementation of the A2C loss.

    A2C (Advantage Actor Critic) is a model-free, online RL algorithm that uses parallel rollouts of n steps to
    update the policy, relying on the REINFORCE estimator to compute the gradient. It also adds an entropy term to the
    objective function to improve exploration.

    For more details regarding A2C, refer to: "Asynchronous Methods for Deep Reinforcment Learning",
    https://arxiv.org/abs/1602.01783v2

    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        value_target_key (str): the input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        entropy_bonus (bool): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (float): the weight of the entropy loss.
        critic_coef (float): the weight of the critic loss.
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.
      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

    """

    default_value_estimator: ValueEstimators = ValueEstimators.GAE

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: TensorDictModule,
        *,
        advantage_key: str = "advantage",
        value_target_key: str = "value_target",
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        gamma: float = None,
        separate_losses: bool = False,
    ):
        super().__init__()
        self.convert_to_functional(
            actor, "actor", funs_to_decorate=["forward", "get_dist"]
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor.parameters())
        else:
            policy_params = None
        self.convert_to_functional(critic, "critic", compare_against=policy_params)
        self.advantage_key = advantage_key
        self.value_target_key = value_target_key
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_coef
        self.register_buffer(
            "entropy_coef", torch.tensor(entropy_coef, device=self.device)
        )
        self.register_buffer(
            "critic_coef", torch.tensor(critic_coef, device=self.device)
        )
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma
        self.loss_critic_type = loss_critic_type

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def _log_probs(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get("action")
        if action.requires_grad:
            raise RuntimeError("tensordict stored action require grad.")
        tensordict_clone = tensordict.select(*self.actor.in_keys).clone()

        dist = self.actor.get_dist(tensordict_clone, params=self.actor_params)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.unsqueeze(-1)
        return log_prob, dist

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        try:
            # TODO: if the advantage is gathered by forward, this introduces an
            # overhead that we could easily reduce.
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
        return self.critic_coef * loss_value

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.advantage_key, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.advantage_key)
        log_probs, dist = self._log_probs(tensordict)
        loss = -(log_probs * advantage)
        td_out = TensorDict({"loss_objective": loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict).mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
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
