# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import SafeModule
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    distance_loss,
    next_state_value as get_next_state_value,
)

try:
    from functorch import vmap

    FUNCTORCH_ERR = ""
    _has_functorch = True
except ImportError as err:
    FUNCTORCH_ERR = str(err)
    _has_functorch = False


class TD3Loss(LossModule):
    """TD3 Loss module.

    Args:
        actor_network (SafeModule): the actor to be trained
        qvalue_network (SafeModule): a single Q-value network that will be multiplicated as many times as needed.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        gamma (Number, optional): gamma decay factor. Default is 0.99.
        max_action (float, optional): Maximum action, in MuJoCo environments typically 1.0.
        policy_noise (float, optional): Standard deviation for the target policy action noise. Default is 0.2.
        noise_clip (float, optional): Clipping range value for the sampled target policy action noise. Default is 0.5.
        priotity_key (str, optional): Key where to write the priority value for prioritized replay buffers. Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of  `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        delay_actor (bool, optional): whether to separate the target actor networks from the actor networks used for
            data collection. Default is :obj:`False`.
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is :obj:`False`.
    """

    def __init__(
        self,
        actor_network: SafeModule,
        qvalue_network: SafeModule,
        num_qvalue_nets: int = 2,
        gamma: Number = 0.99,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        delay_actor: bool = False,
        delay_qvalue: bool = False,
    ) -> None:
        if not _has_functorch:
            raise ImportError(
                f"Failed to import functorch with error message:\n{FUNCTORCH_ERR}"
            )

        super().__init__()

        self.delay_actor = delay_actor
        self.delay_qvalue = delay_qvalue

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()),
        )

        self.num_qvalue_nets = num_qvalue_nets
        self.register_buffer("gamma", torch.tensor(gamma))
        self.priority_key = priotity_key
        self.loss_function = loss_function
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = actor_network.spec["action"].space.maximum.max().item()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.select(
            "reward", "done", "next", *obs_keys, "action"
        )

        actor_params = torch.stack(
            [self.actor_network_params, self.target_actor_network_params], 0
        )

        tensordict_actor_grad = tensordict_select.select(
            *obs_keys
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict_select).select(
            *self.actor_network.in_keys
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)
        tensordict_actor = tensordict_actor.contiguous()

        with set_exploration_mode("mode"):
            actor_output_td = vmap(self.actor_network)(
                tensordict_actor,
                actor_params,
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
        tensordict_actor["action"] = actor_output_td["action"]

        # repeat tensordict_actor to match the qvalue size
        _actor_loss_td = (
            tensordict_actor[0]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[0].batch_size)
        )  # for actor loss
        _qval_td = tensordict_select.select(*self.qvalue_network.in_keys).expand(
            self.num_qvalue_nets,
            *tensordict_select.select(*self.qvalue_network.in_keys).batch_size,
        )  # for qvalue loss
        _next_val_td = (
            tensordict_actor[1]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[1].batch_size)
        )  # for next value estimation
        tensordict_qval = torch.cat(
            [
                _actor_loss_td,
                _next_val_td,
                _qval_td,
            ],
            0,
        )

        # cat params
        q_params_detach = self.qvalue_network_params.detach()
        qvalue_params = torch.cat(
            [
                q_params_detach,
                self.target_qvalue_network_params,
                self.qvalue_network_params,
            ],
            0,
        )
        tensordict_qval = vmap(self.qvalue_network)(
            tensordict_qval,
            qvalue_params,
        )

        state_action_value = tensordict_qval.get("state_action_value").squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )

        loss_actor = -(state_action_value_actor.min(0)[0]).mean()

        next_state_value = next_state_action_value_qvalue.min(0)[0]

        target_value = get_next_state_value(
            tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_value,
        )
        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = (
            distance_loss(
                pred_val,
                target_value.expand_as(pred_val),
                loss_function=self.loss_function,
            )
            .mean(-1)
            .sum()
            * 0.5
        )

        tensordict.set("td_error", td_error.detach().max(0)[0])

        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            source={
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "pred_value": pred_val.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "next_state_value": next_state_value.mean().detach(),
                "target_value": target_value.mean().detach(),
            },
            batch_size=[],
        )

        return td_out
