import math
from numbers import Number

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import TDModule, ActorCriticWrapper
from torchrl.objectives import next_state_value, hold_out_params, distance_loss
from torchrl.objectives.costs.common import _LossModule

__all__ = ["REDQLoss", "DoubleREDQLoss"]


class REDQLoss(_LossModule):
    delay_actor: bool = False
    delay_qvalue: bool = False

    def __init__(
        self,
        actor_network: TDModule,
        qvalue_network: TDModule,
        num_qvalue_nets: int = 10,
        sub_sample_len: int = 2,
        gamma: Number = 0.99,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: Number = 1.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
    ):
        super().__init__()
        self.convert_to_functional(
            actor_network, "actor_network", create_target_params=self.delay_actor
        )
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.sub_sample_len = min(sub_sample_len, num_qvalue_nets - 1)
        self.gamma = gamma
        self.priority_key = priotity_key
        self.loss_function = loss_function

        self.register_buffer("alpha_init", torch.tensor(alpha_init))
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer("log_alpha", torch.tensor(math.log(alpha_init)))
        else:
            self.register_parameter(
                "log_alpha", torch.nn.Parameter(torch.tensor(math.log(alpha_init)))
            )

        if target_entropy == "auto":
            target_entropy = -float(np.prod(actor_network.spec.shape))
        self.register_buffer("target_entropy", torch.tensor(target_entropy))

    @property
    def alpha(self):
        with torch.no_grad():
            alpha = self.log_alpha.detach().exp()
        return alpha

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        loss_actor, action_log_prob = self._actor_loss(tensordict)
        loss_qval = self._qvalue_loss(tensordict)
        loss_alpha = self._loss_alpha(action_log_prob)
        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qval": loss_qval.mean(),
                "loss_alpha": loss_alpha,
                "alpha": self.alpha,
            },
            [],
        )

        return td_out

    def _actor_loss(self, tensordict: _TensorDict) -> Tuple[Tensor, Tensor]:
        tensordict_clone = tensordict.clone()  # to avoid overwriting keys
        self.actor_network(
            tensordict_clone,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers,
        )
        with hold_out_params(self.qvalue_network_params) as params:
            tensordict_expand = self.qvalue_network(
                tensordict_clone,
                params=params,
                buffers=self.qvalue_network_buffers,
                vmap=True,
            )
            state_action_value = tensordict_expand.get("state_action_value").squeeze(-1)
        loss_actor = -(
            state_action_value
            - self.alpha * tensordict_clone.get("action_log_prob").squeeze(-1)
        ).mean(0)
        return loss_actor, tensordict_clone.get("action_log_prob")

    def _qvalue_loss(self, tensordict: _TensorDict) -> Tensor:
        tensordict_save = tensordict
        tensordict = tensordict.clone()
        selected_models_idx = torch.randperm(self.num_qvalue_nets)[
            : self.sub_sample_len
        ].sort()[0]
        with torch.no_grad():
            selected_q_params = [
                p[selected_models_idx] for p in self.target_qvalue_network_params
            ]
            selected_q_buffers = [
                b[selected_models_idx] for b in self.target_qvalue_network_buffers
            ]

            next_td = step_tensor_dict(tensordict)  # next_observation -> observation
            # select pseudo-action
            self.actor_network(
                next_td,
                params=list(self.target_actor_network_params),
                buffers=self.target_actor_network_buffers,
            )
            # get q-values
            next_td = self.qvalue_network(
                next_td, params=selected_q_params, buffers=selected_q_buffers, vmap=True
            )
            state_value = next_td.get("state_action_value") - self.alpha * next_td.get(
                "action_log_prob"
            )
            state_value = state_value.min(0)[0]

        tensordict.set("next_state_value", state_value)
        target_value = next_state_value(
            tensordict,
            gamma=self.gamma,
            vmap=True,
            next_val_key="state_action_value",
            pred_next_val=state_value,
        )
        tensordict_expand = self.qvalue_network(
            tensordict,
            params=list(self.qvalue_network_params),
            buffers=self.qvalue_network_buffers,
            vmap=True,
        )
        pred_val = tensordict_expand.get("state_action_value").squeeze(-1)
        td_error = abs(pred_val - target_value)
        tensordict_save.set("td_error", td_error.detach().max(0)[0])
        loss_qval = distance_loss(
            pred_val, target_value.expand_as(pred_val), loss_function=self.loss_function
        ).mean(0)
        return loss_qval

    def _loss_alpha(self, log_pi: Tensor) -> Tensor:
        if not log_pi.requires_grad:
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


class DoubleREDQLoss(REDQLoss):
    delay_qvalue: bool = True
