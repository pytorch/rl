# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType, set_interaction_type, TensorDictModule
from tensordict.utils import NestedKey
from torchrl.data import CompositeSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import ValueEstimators


class TQCLoss(LossModule):
    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"advantage"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"_log_prob"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey = "sample_log_prob"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        top_quantiles_to_drop: float = 10,
        alpha_init: float = 1.0,
        # no need to pass device, should be handled by actor/qvalue nets
        # device: torch.device,
        # gamma should be passed to the value estimator construction
        # for consistency with other losses
        # gamma: float=None,
        target_entropy=None,
        action_spec=None,
    ):
        super().__init__()

        self.convert_to_functional(
            actor_network,
            "actor",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )

        self.convert_to_functional(
            qvalue_network,
            "critic",
            create_target_params=True,  # Create a target critic network
        )

        # self.device = device
        for p in self.parameters():
            device = p.device
            break
        else:
            # this should never be reached unless both network have 0 parameter
            raise RuntimeError
        self.log_alpha = torch.nn.Parameter(
            torch.tensor([np.log(alpha_init)], requires_grad=True, device=device)
        )
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.target_entropy = target_entropy
        self._action_spec = action_spec
        self.make_value_estimator()

    @property
    def target_entropy(self):
        target_entropy = self.__dict__.get("_target_entropy", None)
        if target_entropy is None:
            # Compute target entropy
            action_spec = self._action_spec
            if action_spec is None:
                action_spec = getattr(self.actor, "spec", None)
            if action_spec is None:
                raise RuntimeError(
                    "Could not deduce action spec neither from "
                    "the actor network nor from the constructor kwargs. "
                    "Please provide the target entropy during construction."
                )
            if not isinstance(action_spec, CompositeSpec):
                action_spec = CompositeSpec({self.tensor_keys.action: action_spec})
            action_container_len = len(action_spec.shape)

            target_entropy = -float(
                action_spec[self.tensor_keys.action]
                .shape[action_container_len:]
                .numel()
            )
            self.target_entropy = target_entropy
        return target_entropy

    @target_entropy.setter
    def target_entropy(self, value):
        if value is not None:
            value = float(value)
        self._target_entropy = value

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def value_loss(self, tensordict):
        tensordict_copy = tensordict.clone(False)
        td_next = tensordict_copy.get("next")
        reward = td_next.get(self.tensor_keys.reward)
        not_done = td_next.get(self.tensor_keys.done).logical_not()
        alpha = self.alpha

        # Q-loss
        with torch.no_grad():
            # get policy action
            self.actor(td_next, params=self.actor_params)
            self.critic(td_next, params=self.target_critic_params)
            next_log_pi = td_next.get(self.tensor_keys.log_prob)
            next_log_pi = torch.unsqueeze(next_log_pi, dim=-1)

            # compute and cut quantiles at the next state
            next_z = td_next.get(self.tensor_keys.state_action_value)
            sorted_z, _ = torch.sort(next_z.reshape(*tensordict_copy.batch_size, -1))
            sorted_z_part = sorted_z[..., : -self.top_quantiles_to_drop]

            # compute target
            # --- Note ---
            # This is computed manually here, since the built-in value estimators in the library
            # currently do not support a critic of a shape different from the reward.
            # ------------
            target = reward + not_done * self.gamma * (
                sorted_z_part - alpha * next_log_pi
            )

        self.critic(tensordict_copy, params=self.critic_params)
        cur_z = tensordict_copy.get(self.tensor_keys.state_action_value)
        critic_loss = quantile_huber_loss_f(cur_z, target)
        metadata = {}
        return critic_loss, metadata

    def actor_loss(self, tensordict):
        tensordict_copy = tensordict.clone(False)
        alpha = self.alpha
        self.actor(tensordict_copy, params=self.actor_params)
        self.critic(tensordict_copy, params=self.critic_params)
        new_log_pi = tensordict_copy.get(self.tensor_keys.log_prob)
        tensordict.set(self.tensor_keys.log_prob, new_log_pi)
        actor_loss = (
            alpha * new_log_pi
            - tensordict_copy.get(self.tensor_keys.state_action_value)
            .mean(-1)
            .mean(-1, keepdim=True)
        ).mean()
        metadata = {}
        return actor_loss, metadata

    def alpha_loss(self, tensordict):
        log_prob = tensordict.get(self.tensor_keys.log_prob)
        alpha_loss = -self.log_alpha * (log_prob + self.target_entropy).detach().mean()
        return alpha_loss, {}

    def entropy(self, tensordict):
        with set_exploration_type(ExplorationType.RANDOM):
            dist = self.actor.get_dist(
                tensordict,
                params=self.actor_params,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm).detach()
        entropy = -log_prob.mean()
        return entropy

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        critic_loss, metadata_value = self.value_loss(tensordict)
        actor_loss, metadata_actor = self.actor_loss(
            tensordict
        )  # Compute actor loss AFTER critic loss
        alpha_loss, metadata_alpha = self.alpha_loss(tensordict)
        metadata = {
            "alpha": self.alpha,
            "entropy": self.entropy(tensordict),
        }
        metadata.update(metadata_alpha)
        metadata.update(metadata_value)
        metadata.update(metadata_actor)
        losses = {
            "loss_critic": critic_loss,
            "loss_actor": actor_loss,
            "loss_alpha": alpha_loss,
        }
        losses.update(metadata)
        return TensorDict(losses, batch_size=[])

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        """Value estimator settor for TQC.

        The only value estimator supported is ``ValueEstimators.TD0``.

        This method can also be used to set the ``gamma`` factor.

        Args:
            value_type (ValueEstimators, optional): the value estimator to be used.
                Will raise an exception if it differs from ``ValueEstimators.TD0``.
            gamma (float, optional): the gamma factor for the target computation.
                Defaults to 0.99.
        """
        if value_type not in (ValueEstimators.TD0, None):
            raise NotImplementedError(
                f"Value type {value_type} is not currently implemented."
            )
        self.gamma = hyperparams.pop("gamma", 0.99)


# ====================================================================
# Quantile Huber Loss
# -------------------


def quantile_huber_loss_f(quantiles, samples):
    """
    Quantile Huber loss from the original PyTorch TQC implementation.
    See: https://github.com/SamsungLabs/tqc_pytorch/blob/master/tqc/functions.py

    quantiles is assumed to be of shape [batch size, n_nets, n_quantiles]
    samples is assumed to be of shape [batch size, n_samples]
    Arbitrary batch sizes are allowed.
    """
    pairwise_delta = (
        samples[..., None, None, :] - quantiles[..., None]
    )  # batch x n_nets x n_quantiles x n_samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )
    n_quantiles = quantiles.shape[-1]
    tau = (
        torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles
        + 1 / 2 / n_quantiles
    )
    loss = (
        torch.abs(tau[..., None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss
