# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from examples.tqc.utils import quantile_huber_loss_f
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import set_interaction_type, InteractionType
from torchrl.data import CompositeSpec
from torchrl.objectives import LossModule, ValueEstimators


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
            actor_network,
            qvalue_network,
            gamma,
            top_quantiles_to_drop,
            alpha_init,
            device
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
            create_target_params=True  # Create a target critic network
        )

        self.device = device
        self.log_alpha = torch.tensor([np.log(alpha_init)], requires_grad=True, device=self.device)
        self.gamma = gamma
        self.top_quantiles_to_drop = top_quantiles_to_drop

        # Compute target entropy
        action_spec = getattr(self.actor, "spec", None)
        if action_spec is None:
            print("Could not deduce action spec from actor network.")
        if not isinstance(action_spec, CompositeSpec):
            action_spec = CompositeSpec({"action": action_spec})
        action_container_len = len(action_spec.shape)
        self.target_entropy = -float(action_spec["action"].shape[action_container_len:].numel())

    def value_loss(self, tensordict):
        td_next = tensordict.get("next")
        reward = td_next.get(self.tensor_keys.reward)
        not_done = tensordict.get(self.tensor_keys.done_key).logical_not()
        alpha = torch.exp(self.log_alpha)

        # Q-loss
        with torch.no_grad():
            # get policy action
            self.actor(td_next, params=self.actor_params)
            self.critic(td_next, params=self.target_critic_params)

            next_log_pi = td_next.get(self.tensor_keys.log_prob)
            next_log_pi = torch.unsqueeze(next_log_pi, dim=-1)

            # compute and cut quantiles at the next state
            next_z = td_next.get(self.tensor_keys.state_action_value)
            sorted_z, _ = torch.sort(next_z.reshape(*tensordict.batch_size, -1))
            sorted_z_part = sorted_z[..., :-self.top_quantiles_to_drop]

            # compute target
            target = reward + not_done * self.gamma * (sorted_z_part - alpha * next_log_pi)

        self.critic(tensordict, params=self.critic_params)
        cur_z = tensordict.get(self.tensor_keys.state_action_value)
        critic_loss = quantile_huber_loss_f(cur_z, target)
        return critic_loss

    def actor_loss(self, tensordict):
        alpha = torch.exp(self.log_alpha)
        self.actor(tensordict, params=self.actor_params)
        self.critic(tensordict, params=self.critic_params)
        new_log_pi = tensordict.get(self.tensor_keys.log_prob)
        actor_loss = (alpha * new_log_pi - tensordict.get(self.tensor_keys.state_action_value).mean(-1).mean(-1, keepdim=True)).mean()
        return actor_loss, new_log_pi

    def alpha_loss(self, log_prob):
        alpha_loss = -self.log_alpha * (log_prob + self.target_entropy).detach().mean()
        return alpha_loss

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
        alpha = torch.exp(self.log_alpha)
        critic_loss = self.value_loss(tensordict)
        actor_loss, log_prob = self.actor_loss(tensordict)  # Compute actor loss AFTER critic loss
        alpha_loss = self.alpha_loss(log_prob)
        entropy = self.entropy(tensordict)

        return TensorDict(
            {
                "loss_critic": critic_loss,
                "loss_actor": actor_loss,
                "loss_alpha": alpha_loss,
                "alpha": alpha,
                "entropy": entropy,
            },
            batch_size=[]
        )
