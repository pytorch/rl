# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from numbers import Number
from typing import Union

import numpy as np
import torch

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import Tensor

from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
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

    FUNCTORCH_ERR = ""
    _has_functorch = True
except ImportError as err:
    FUNCTORCH_ERR = str(err)
    _has_functorch = False


class REDQLoss(LossModule):
    """REDQ Loss module.

    REDQ (RANDOMIZED ENSEMBLED DOUBLE Q-LEARNING: LEARNING FAST WITHOUT A MODEL
    https://openreview.net/pdf?id=AY8zfZm0tDd) generalizes the idea of using an ensemble of Q-value functions to
    train a SAC-like algorithm.

    Args:
        actor_network (TensorDictModule): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will
            be multiplicated as many times as needed.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained.
            Default is ``10``.
        sub_sample_len (int, optional): number of Q-value networks to be
            subsampled to evaluate the next state value
            Default is ``2``.
        loss_function (str, optional): loss function to be used for the Q-value.
            Can be one of  ``"smooth_l1"``, ``"l2"``,
            ``"l1"``, Default is ``"smooth_l1"``.
        alpha_init (float, optional): initial entropy multiplier.
            Default is ``1.0``.
        min_alpha (float, optional): min value of alpha.
            Default is ``0.1``.
        max_alpha (float, optional): max value of alpha.
            Default is ``10.0``.
        fixed_alpha (bool, optional): whether alpha should be trained to match
            a target entropy. Default is ``False``.
        target_entropy (Union[str, Number], optional): Target entropy for the
            stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used
            for data collection. Default is ``False``.
        gSDE (bool, optional): Knowing if gSDE is used is necessary to create
            random noise variables.
            Default is ``False``.
        priority_key (str, optional): [Deprecated, use .set_keys() instead] Key where to write the priority value
            for prioritized replay buffers. Default is
            ``"td_error"``.

    """

    delay_actor: bool = False
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        *,
        num_qvalue_nets: int = 10,
        sub_sample_len: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        gSDE: bool = False,
        gamma: float = None,
        priority_key: str = None,
    ):
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERR

        super().__init__()

        self.tensordict_keys = {
            "priority_key": "td_error",
            "action_key": "action",
            "value_key": "state_value",
            "sample_log_prob_key": "sample_log_prob",
            "state_action_value_key": "state_action_value",
        }
        if priority_key is not None:
            warnings.warn(
                "Setting 'priority_key' via ctor is deprecated, use .set_keys(priotity_key='some_key') instead.",
                category=DeprecationWarning,
            )
            self.tensordict_keys["priority_key"] = priority_key
        self.set_keys(**self.tensordict_keys)

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist_params"],
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
            target_entropy = -float(np.prod(actor_network.spec[self.action_key].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        self.gSDE = gSDE
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    @property
    def alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.clone(False).select(
            "next", *obs_keys, self.action_key
        )
        selected_models_idx = torch.randperm(self.num_qvalue_nets)[
            : self.sub_sample_len
        ].sort()[0]
        selected_q_params = self.target_qvalue_network_params[selected_models_idx]

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

        with set_exploration_type(ExplorationType.RANDOM):
            if self.gSDE:
                tensordict_actor.set(
                    "_eps_gSDE",
                    torch.zeros(tensordict_actor.shape, device=tensordict_actor.device),
                )
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = vmap(self.actor_network.get_dist_params)(
                tensordict_actor,
                actor_params,
            )
            if isinstance(self.actor_network, TensorDictSequential):
                sample_key = self.action_key
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            else:
                sample_key = self.action_key
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            tensordict_actor.set(sample_key, tensordict_actor_dist.rsample())
            tensordict_actor.set(
                self.sample_log_prob_key,
                tensordict_actor_dist.log_prob(tensordict_actor.get(sample_key)),
            )

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
            .expand(self.sub_sample_len, *tensordict_actor[1].batch_size)
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
            [q_params_detach, selected_q_params, self.qvalue_network_params], 0
        )
        tensordict_qval = vmap(self.qvalue_network)(
            tensordict_qval,
            qvalue_params,
        )

        state_action_value = tensordict_qval.get(self.state_action_value_key).squeeze(
            -1
        )
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.sub_sample_len, self.num_qvalue_nets],
            dim=0,
        )
        sample_log_prob = tensordict_actor.get(self.sample_log_prob_key).squeeze(-1)
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

        tensordict_select.set(("next", self.value_key), next_state_value.unsqueeze(-1))
        target_value = self.value_estimator.value_estimate(tensordict_select).squeeze(
            -1
        )

        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).mean(0)

        tensordict.set(self.priority_key, td_error.detach().max(0)[0])

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
                "next.state_value": next_state_value.mean().detach(),
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

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        value_key = "state_value"
        # we do not need a value network bc the next state value is already passed
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=None, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=None, value_key=value_key, **hp
            )
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=None, value_key=value_key, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
