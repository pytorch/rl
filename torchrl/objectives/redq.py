# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from numbers import Number
from typing import Union

import numpy as np
import torch
from torch import Tensor

from torchrl.data.tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import TensorDictModule
from torchrl.objectives.common import LossModule, _has_functorch
from torchrl.objectives.utils import (
    distance_loss,
    hold_out_params,
    next_state_value as get_next_state_value,
)


class REDQLoss(LossModule):
    """REDQ Loss module.

    REDQ (RANDOMIZED ENSEMBLED DOUBLE Q-LEARNING: LEARNING FAST WITHOUT A MODEL
    https://openreview.net/pdf?id=AY8zfZm0tDd) generalizes the idea of using an ensemble of Q-value functions to
    train a SAC-like algorithm.

    Args:
        actor_network (TensorDictModule): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will be multiplicated as many times as needed.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        sub_sample_len (int, optional): number of Q-value networks to be subsampled to evaluate the next state value
            Default is 2.
        gamma (Number, optional): gamma decay factor. Default is 0.99.
        priotity_key (str, optional): Key where to write the priority value for prioritized replay buffers. Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of  `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is 0.1.
        max_alpha (float, optional): max value of alpha.
            Default is 10.0.
        fixed_alpha (bool, optional): whether alpha should be trained to match a target entropy. Default is :obj:`False`.
        target_entropy (Union[str, Number], optional): Target entropy for the stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is :obj:`False`.
        gSDE (bool, optional): Knowing if gSDE is used is necessary to create random noise variables.
            Default is False

    """

    delay_actor: bool = False

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        num_qvalue_nets: int = 10,
        sub_sample_len: int = 2,
        gamma: Number = 0.99,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        gSDE: bool = False,
    ):
        if not _has_functorch:
            raise ImportError("REDQ requires functorch to be installed.")

        super().__init__()
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        # let's make sure that actor_network has `return_log_prob` set to True
        self.actor_network.return_log_prob = True

        self.delay_qvalue = delay_qvalue
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()),
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.sub_sample_len = max(1, min(sub_sample_len, num_qvalue_nets - 1))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.priority_key = priotity_key
        self.loss_function = loss_function

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        self.register_buffer(
            "min_log_alpha", torch.tensor(min_alpha, device=device).log()
        )
        self.register_buffer(
            "max_log_alpha", torch.tensor(max_alpha, device=device).log()
        )
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if target_entropy == "auto":
            if actor_network.spec["action"] is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        self.gSDE = gSDE

    @property
    def alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        next_obs_keys = [key for key in tensordict.keys() if key.startswith("next_")]
        tensordict_select = tensordict.select(
            "reward", "done", *next_obs_keys, *obs_keys, "action"
        )
        selected_models_idx = torch.randperm(self.num_qvalue_nets)[
            : self.sub_sample_len
        ].sort()[0]
        selected_q_params = [
            p[selected_models_idx] for p in self.target_qvalue_network_params
        ]
        selected_q_buffers = [
            b[selected_models_idx] for b in self.target_qvalue_network_buffers
        ]

        actor_params = [
            torch.stack([p1, p2], 0)
            for p1, p2 in zip(
                self.actor_network_params, self.target_actor_network_params
            )
        ]
        actor_buffers = [
            torch.stack([p1, p2], 0)
            for p1, p2 in zip(
                self.actor_network_buffers, self.target_actor_network_buffers
            )
        ]

        tensordict_actor_grad = tensordict_select.select(
            *obs_keys
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict_select).select(
            *self.actor_network.in_keys
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)
        tensordict_actor = tensordict_actor.contiguous()

        with set_exploration_mode("random"):
            if self.gSDE:
                tensordict_actor.set(
                    "_eps_gSDE",
                    torch.zeros(tensordict_actor.shape, device=tensordict_actor.device),
                )
            tensordict_actor = self.actor_network(
                tensordict_actor,
                params=actor_params,
                buffers=actor_buffers,
                vmap=(0, 0, 0),
            )

        # repeat tensordict_actor to match the qvalue size
        tensordict_qval = torch.cat(
            [
                tensordict_actor[0]
                .select(*self.qvalue_network.in_keys)
                .expand(
                    self.num_qvalue_nets, *tensordict_actor[0].batch_size
                ),  # for actor loss
                tensordict_actor[1]
                .select(*self.qvalue_network.in_keys)
                .expand(
                    self.sub_sample_len, *tensordict_actor[1].batch_size
                ),  # for next value estimation
                tensordict_select.select(*self.qvalue_network.in_keys).expand(
                    self.num_qvalue_nets,
                    *tensordict_select.select(*self.qvalue_network.in_keys).batch_size,
                ),  # for qvalue loss
            ],
            0,
        )

        # cat params
        q_params_detach = hold_out_params(self.qvalue_network_params).params
        qvalue_params = [
            torch.cat([p1, p2, p3], 0)
            for p1, p2, p3 in zip(
                q_params_detach, selected_q_params, self.qvalue_network_params
            )
        ]
        qvalue_buffers = [
            torch.cat([p1, p2, p3], 0)
            for p1, p2, p3 in zip(
                self.qvalue_network_buffers,
                selected_q_buffers,
                self.qvalue_network_buffers,
            )
        ]
        tensordict_qval = self.qvalue_network(
            tensordict_qval,
            tensordict_out=TensorDict({}, tensordict_qval.shape),
            params=qvalue_params,
            buffers=qvalue_buffers,
            vmap=(
                0,
                0,
                0,
            ),  # TensorDict vmap will take care of expanding the tuple as needed
        )

        state_action_value = tensordict_qval.get("state_action_value").squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.sub_sample_len, self.num_qvalue_nets],
            dim=0,
        )
        sample_log_prob = tensordict_actor.get("sample_log_prob").squeeze(-1)
        (
            action_log_prob_actor,
            next_action_log_prob_qvalue,
        ) = sample_log_prob.unbind(0)

        loss_actor = -(
            state_action_value_actor - self.alpha * action_log_prob_actor
        ).mean(0)

        next_state_value = (
            next_state_action_value_qvalue - self.alpha * next_action_log_prob_qvalue
        )
        next_state_value = next_state_value.min(0)[0]

        target_value = get_next_state_value(
            tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_value,
        )
        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).mean(0)

        tensordict.set("td_error", td_error.detach().max(0)[0])

        loss_alpha = self._loss_alpha(sample_log_prob)
        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self.alpha.detach(),
                "entropy": -sample_log_prob.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "action_log_prob_actor": action_log_prob_actor.mean().detach(),
                "next_state_value": next_state_value.mean().detach(),
                "target_value": target_value.mean().detach(),
            },
            [],
        )

        return td_out

    def _loss_alpha(self, log_pi: Tensor) -> Tensor:
        if torch.is_grad_enabled() and not log_pi.requires_grad:
            raise RuntimeError(
                "expected log_pi to require gradient for the alpha loss)"
            )
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss
