# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from numbers import Number

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.modules import TensorDictModule

from torchrl.objectives.utils import (
    distance_loss,
    hold_out_params,
    next_state_value as get_next_state_value,
)
from ..envs.utils import set_exploration_mode
from .common import _has_functorch, LossModule


class TD3Loss(LossModule):
    """The TD3 Loss class.

    Args:
        actor_network (TensorDictModule): a policy operator.
        value_network (TensorDictModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        priotity_key (str, optional): Key where to write the priority value for prioritized replay buffers. Default is
            `"td_error"`.
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
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        gamma: Number = 0.99,
        max_action: float = 1.0,
        priority_key: str = "td_error",
        policy_update_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
    ) -> None:

        if not _has_functorch:
            raise ImportError("TD3 requires functorch to be installed.")

        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value
        self.max_action = max_action
        self.policy_update_delay = policy_update_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_iter = 0
        self.priority_key = priority_key
        self.num_qvalue_nets = 2

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            self.num_qvalue_nets,
            create_target_params=self.delay_value,
            compare_against=list(actor_network.parameters()),
        )

        self.actor_in_keys = actor_network.in_keys
        self.value_net_in_keys = value_network.in_keys
        self.register_buffer("gamma", torch.tensor(gamma))
        self.loss_function = loss_function

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the TD3 losses given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of tensors containing the TD3 loss.

        """
        self.update_iter += 1

        observation_td = tensordict.select(*self.actor_in_keys)
        next_observation_td = tensordict["next"].select(*self.actor_in_keys)

        observations_td = torch.stack([observation_td, next_observation_td], 0)
        observations_td = observations_td.contiguous()

        # cat params
        target_actor_network_params_detach = hold_out_params(
            self.target_actor_network_params
        ).params
        actor_params = [
            torch.stack([p1, p2], 0)
            for p1, p2 in zip(
                self.actor_network_params, target_actor_network_params_detach
            )
        ]
        actor_buffers = [
            torch.stack([p1, p2], 0)
            for p1, p2 in zip(
                self.actor_network_buffers, self.target_actor_network_buffers
            )
        ]
        # forward policy path
        with set_exploration_mode("mode"):
            actor_output_td = self.actor_network(
                observations_td,
                params=actor_params,
                buffers=actor_buffers,
                vmap=(0, 0, 0),
            )

        # add noise to target policy
        noise = torch.normal(
            mean=torch.zeros(actor_output_td[1]["action"].shape),
            std=torch.ones(actor_output_td[1]["action"].shape) * self.policy_noise,
        ).to(actor_output_td[1].device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = (actor_output_td[1]["action"] + noise).clamp(
            -self.max_action, self.max_action
        )
        actor_output_td[1].set("action", next_action, inplace=True)
        # repeat tensordict_actor to match the qvalue size

        tensordict_qval = torch.cat(
            [  # state with current policy action for actor loss
                actor_output_td[0]
                .select(*self.value_net_in_keys)
                .expand(
                    self.num_qvalue_nets, *actor_output_td[0].batch_size
                ),  # next state with noisy next action for target q value estimation
                actor_output_td[1]
                .select(*self.value_net_in_keys)
                .expand(
                    self.num_qvalue_nets, *actor_output_td[1].batch_size
                ),  # state and action for q value prediction and q loss
                tensordict.select(*self.value_net_in_keys).expand(
                    self.num_qvalue_nets,
                    *tensordict.select(*self.value_net_in_keys).batch_size,
                ),
            ],
            0,
        )

        # cat params
        q_params_detach = hold_out_params(self.value_network_params).params
        target_value_network_params_detach = hold_out_params(
            self.target_value_network_params
        ).params
        value_params = [
            torch.cat([p1, p2, p3], 0)
            for p1, p2, p3 in zip(
                q_params_detach,
                target_value_network_params_detach,
                self.value_network_params,
            )
        ]
        value_buffers = [
            torch.cat([p1, p2, p3], 0)
            for p1, p2, p3 in zip(
                self.value_network_buffers,
                self.target_value_network_buffers,
                self.value_network_buffers,
            )
        ]

        tensordict_qval = self.value_network(
            tensordict_qval,
            tensordict_out=TensorDict({}, tensordict_qval.shape),
            params=value_params,
            buffers=value_buffers,
            vmap=(
                0,
                0,
                0,
            ),
            # TensorDict vmap will take care of expanding the tuple as needed
        )

        state_action_value = tensordict_qval.get("state_action_value").squeeze(-1)

        # need to split as we stacked different values
        (
            state_action_value_actor,
            target_qvalues,
            pred_qval,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )

        # calc q loss
        target_qvalue = target_qvalues.min(0)[0].detach()
        target_value = get_next_state_value(
            tensordict,
            gamma=self.gamma,
            pred_next_val=target_qvalue,
        )

        td_error = (pred_qval - target_value).pow(2)
        loss_qval = distance_loss(
            pred_qval,
            target_value.expand_as(pred_qval),
            loss_function=self.loss_function,
        ).mean(0)

        tensordict.set("td_error", td_error.detach().max(0)[0])
        out_dict = {
            "loss_qvalue": loss_qval.mean(),
            "td_error": td_error.detach().max(0)[0],
            "state_action_value_actor": state_action_value_actor.mean().detach(),
            "next_state_value": target_qvalues.mean().detach(),
            "target_value": target_value.mean().detach(),
        }

        # calc actor loss every policy delayed steps
        if self.update_iter % self.policy_update_delay == 0:
            loss_actor = -state_action_value_actor[0].mean()
            out_dict.update({"loss_actor": loss_actor.mean()})
        else:
            out_dict.update({"loss_actor": 0.0})

        td_out = TensorDict(
            out_dict,
            [],
        )

        return td_out
