# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from numbers import Number
from typing import Tuple, Union

import numpy as np
import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDictBase
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.objectives import (
    default_value_kwargs,
    distance_loss,
    hold_out_params,
    ValueEstimators,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _GAMMA_LMBDA_DEPREC_WARNING
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


class REDQLoss_deprecated(LossModule):
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
        priority_key (str, optional): Key where to write the priority value
            for prioritized replay buffers. Default is
            ``"td_error"``.
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
        priority_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        gSDE: bool = False,
        gamma: float = None,
    ):
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERR
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
            expand_dim=num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=actor_network.parameters(),
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.sub_sample_len = max(1, min(sub_sample_len, num_qvalue_nets - 1))
        self.priority_key = priority_key
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

        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    @property
    def alpha(self):
        # keep alpha is a reasonable range
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss_actor, sample_log_prob = self._actor_loss(tensordict)

        loss_qval = self._qvalue_loss(tensordict)
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
                "alpha": self.alpha,
                "entropy": -sample_log_prob.mean().detach(),
            },
            [],
        )

        return td_out

    def _actor_loss(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        obs_keys = self.actor_network.in_keys
        tensordict_clone = tensordict.select(*obs_keys)  # to avoid overwriting keys
        with set_exploration_type(ExplorationType.RANDOM):
            self.actor_network(
                tensordict_clone,
                params=self.actor_network_params,
            )

        with hold_out_params(self.qvalue_network_params) as params:
            tensordict_expand = vmap(self.qvalue_network, (None, 0))(
                tensordict_clone.select(*self.qvalue_network.in_keys),
                params,
            )
            state_action_value = tensordict_expand.get("state_action_value").squeeze(-1)
        loss_actor = -(
            state_action_value
            - self.alpha * tensordict_clone.get("sample_log_prob").squeeze(-1)
        ).mean(0)
        return loss_actor, tensordict_clone.get("sample_log_prob")

    def _qvalue_loss(self, tensordict: TensorDictBase) -> Tensor:
        tensordict_save = tensordict

        obs_keys = self.actor_network.in_keys
        tensordict = tensordict.clone(False).select("next", *obs_keys, "action")

        selected_models_idx = torch.randperm(self.num_qvalue_nets)[
            : self.sub_sample_len
        ].sort()[0]
        with torch.no_grad():
            selected_q_params = self.target_qvalue_network_params[selected_models_idx]

            next_td = step_mdp(tensordict).select(
                *self.actor_network.in_keys
            )  # next_observation ->
            # observation
            # select pseudo-action
            with set_exploration_type(ExplorationType.RANDOM):
                self.actor_network(
                    next_td,
                    params=self.target_actor_network_params,
                )
            sample_log_prob = next_td.get("sample_log_prob")
            # get q-values
            next_td = vmap(self.qvalue_network, (None, 0))(
                next_td,
                selected_q_params,
            )
            state_action_value = next_td.get("state_action_value")
            if (
                state_action_value.shape[-len(sample_log_prob.shape) :]
                != sample_log_prob.shape
            ):
                sample_log_prob = sample_log_prob.unsqueeze(-1)
            next_state_value = (
                next_td.get("state_action_value") - self.alpha * sample_log_prob
            )
            next_state_value = next_state_value.min(0)[0]

        tensordict.set(("next", "state_value"), next_state_value)
        target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
        tensordict_expand = vmap(self.qvalue_network, (None, 0))(
            tensordict.select(*self.qvalue_network.in_keys),
            self.qvalue_network_params,
        )
        pred_val = tensordict_expand.get("state_action_value").squeeze(-1)
        td_error = abs(pred_val - target_value)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).mean(0)
        tensordict_save.set("td_error", td_error.detach().max(0)[0])
        return loss_qval

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


class DoubleREDQLoss_deprecated(REDQLoss_deprecated):
    """[Deprecated] Class for delayed target-REDQ (which should be the default behaviour)."""

    delay_qvalue: bool = True
