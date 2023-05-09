# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict.nn import make_functional, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import Tensor

from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
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

    _has_functorch = True
    err = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERROR = err


class SACLoss(LossModule):
    """TorchRL implementation of the SAC loss.

    Presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/abs/1801.01290
    and "Soft Actor-Critic Algorithms and Applications" https://arxiv.org/abs/1812.05905

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model.
            This module typically outputs a ``"state_action_value"`` entry.
        value_network (TensorDictModule, optional): V(s) parametric model.
            This module typically outputs a ``"state_value"`` entry.

            .. note::
              If not provided, the second version of SAC is assumed, where
              only the Q-Value network is needed.

        num_qvalue_nets (integer, optional): number of Q-Value networks used.
            Defaults to ``2``.
        priority_key (str, optional): tensordict key where to write the
            priority (for prioritized replay buffer usage). Defaults to
            ``"td_error"``.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is 0.1.
        max_alpha (float, optional): max value of alpha.
            Default is 10.0.
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        delay_actor (bool, optional): Whether to separate the target actor
            networks from the actor networks used for data collection.
            Default is ``False``.
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
            Default is ``False``.
        delay_value (bool, optional): Whether to separate the target value
            networks from the value networks used for data collection.
            Default is ``False``.
    """

    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        value_network: Optional[TensorDictModule] = None,
        *,
        num_qvalue_nets: int = 2,
        priority_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = False,
        delay_value: bool = False,
        gamma: float = None,
    ) -> None:
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()

        # Actor
        self.delay_actor = delay_actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist"],
        )

        # Value
        if value_network is not None:
            self._version = 1
            self.delay_value = delay_value
            self.convert_to_functional(
                value_network,
                "value_network",
                create_target_params=self.delay_value,
                compare_against=list(actor_network.parameters()),
            )
        else:
            self._version = 2

        # Q value
        self.delay_qvalue = delay_qvalue
        self.num_qvalue_nets = num_qvalue_nets
        if self._version == 1:
            value_params = list(value_network.parameters())
        else:
            value_params = []
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()) + value_params,
        )

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
            if actor_network.spec is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        if self._version == 1:
            self.actor_critic = ActorCriticWrapper(
                self.actor_network, self.value_network
            )
            make_functional(self.actor_critic)
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        if self._version == 1:
            value_net = self.actor_critic
        elif self._version == 2:
            # we will take care of computing the next value inside this module
            value_net = None
        else:
            # unreachable
            raise NotImplementedError

        value_key = "state_value"
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        device = self.device
        td_device = tensordict_reshape.to(device)

        loss_actor = self._loss_actor(td_device)
        if self._version == 1:
            loss_qvalue, priority = self._loss_qvalue_v1(td_device)
            loss_value = self._loss_value(td_device)
        else:
            loss_qvalue, priority = self._loss_qvalue_v2(td_device)
            loss_value = None
        loss_alpha = self._loss_alpha(td_device)
        tensordict_reshape.set(self.priority_key, priority)
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        out = {
            "loss_actor": loss_actor.mean(),
            "loss_qvalue": loss_qvalue.mean(),
            "loss_alpha": loss_alpha.mean(),
            "alpha": self._alpha,
            "entropy": -td_device.get("_log_prob").mean().detach(),
        }
        if self._version == 1:
            out["loss_value"] = loss_value.mean()
        return TensorDict(out, [])

    def _loss_actor(self, tensordict: TensorDictBase) -> Tensor:
        # KL lossa
        with set_exploration_type(ExplorationType.RANDOM):
            dist = self.actor_network.get_dist(
                tensordict,
                params=self.actor_network_params,
            )
            a_reparm = dist.rsample()
        # if not self.actor_network.spec.is_in(a_reparm):
        #     a_reparm.data.copy_(self.actor_network.spec.project(a_reparm.data))
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q.set("action", a_reparm)
        td_q = vmap(self.qvalue_network, (None, 0))(
            td_q, self.target_qvalue_network_params
        )
        min_q_logprob = td_q.get("state_action_value").min(0)[0].squeeze(-1)

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        # write log_prob in tensordict for alpha loss
        tensordict.set("_log_prob", log_prob.detach())
        return self._alpha * log_prob - min_q_logprob

    def _loss_qvalue_v1(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        target_params = TensorDict(
            {
                "module": {
                    "0": self.target_actor_network_params,
                    "1": self.target_value_network_params,
                }
            },
            torch.Size([]),
            _run_checks=False,
        )
        with set_exploration_type(ExplorationType.MODE):
            target_value = self.value_estimator.value_estimate(
                tensordict, target_params=target_params
            ).squeeze(-1)

        # value loss
        qvalue_network = self.qvalue_network

        # Q-nets must be trained independently: as such, we split the data in 2
        # if required and train each q-net on one half of the data.
        shape = tensordict.shape
        if shape[0] % self.num_qvalue_nets != 0:
            raise RuntimeError(
                f"Batch size={tensordict.shape} is incompatible "
                f"with num_qvqlue_nets={self.num_qvalue_nets}."
            )
        tensordict_chunks = torch.stack(
            tensordict.chunk(self.num_qvalue_nets, dim=0), 0
        )
        target_chunks = torch.stack(target_value.chunk(self.num_qvalue_nets, dim=0), 0)

        # if vmap=True, it is assumed that the input tensordict must be cast to the param shape
        tensordict_chunks = vmap(qvalue_network)(
            tensordict_chunks, self.qvalue_network_params
        )
        pred_val = tensordict_chunks.get("state_action_value").squeeze(-1)
        loss_value = distance_loss(
            pred_val, target_chunks, loss_function=self.loss_function
        ).view(*shape)
        priority_value = torch.cat((pred_val - target_chunks).pow(2).unbind(0), 0)

        return loss_value, priority_value

    def _get_value_v2(self, tensordict, _alpha, actor_params, qval_params):
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(ExplorationType.RANDOM):
                dist = self.actor_network.get_dist(tensordict, params=actor_params)
                tensordict.set("action", dist.rsample())
                log_prob = dist.log_prob(tensordict.get("action"))
                tensordict.set("sample_log_prob", log_prob)
            sample_log_prob = tensordict.get("sample_log_prob")

            # get q-values
            tensordict_expand = vmap(self.qvalue_network, (None, 0))(
                tensordict, qval_params
            )
            state_action_value = tensordict_expand.get("state_action_value")
            if (
                state_action_value.shape[-len(sample_log_prob.shape) :]
                != sample_log_prob.shape
            ):
                sample_log_prob = sample_log_prob.unsqueeze(-1)
            state_value = state_action_value - _alpha * sample_log_prob
            state_value = state_value.min(0)[0]
            tensordict.set(("next", self.value_estimator.value_key), state_value)
            target_value = self.value_estimator.value_estimate(
                tensordict,
                _alpha=self._alpha,
                actor_params=self.target_actor_network_params,
                qval_params=self.target_qvalue_network_params,
            ).squeeze(-1)
            return target_value

    def _loss_qvalue_v2(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        # we pass the alpha value to the tensordict. Since it's a scalar, we must erase the batch-size first.
        target_value = self._get_value_v2(
            tensordict,
            self._alpha,
            self.target_actor_network_params,
            self.target_qvalue_network_params,
        )

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
        return loss_qval, td_error.detach().max(0)[0]

    def _loss_value(self, tensordict: TensorDictBase) -> Tensor:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )
        pred_val = td_copy.get("state_value").squeeze(-1)

        action_dist = self.actor_network.get_dist(
            td_copy,
            params=self.target_actor_network_params,
        )  # resample an action
        action = action_dist.rsample()

        td_copy.set("action", action, inplace=False)

        qval_net = self.qvalue_network
        td_copy = vmap(qval_net, (None, 0))(
            td_copy,
            self.target_qvalue_network_params,
        )

        min_qval = td_copy.get("state_action_value").squeeze(-1).min(0)[0]

        log_p = action_dist.log_prob(action)
        if log_p.shape != min_qval.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {min_qval.shape} and {log_p.shape}"
            )
        target_val = min_qval - self._alpha * log_p

        loss_value = distance_loss(
            pred_val, target_val, loss_function=self.loss_function
        )
        return loss_value

    def _loss_alpha(self, tensordict: TensorDictBase) -> Tensor:
        log_pi = tensordict.get("_log_prob")
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    @property
    def _alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha


class DiscreteSACLoss(LossModule):
    """Discrete SAC Loss module.

    Args:
        actor_network (ProbabilisticActor): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will be multiplicated as many times as needed.
        num_actions (int): number of actions in the action space.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        priority_key (str, optional): Key where to write the priority value for prioritized replay buffers. Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of  `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is 0.1.
        max_alpha (float, optional): max value of alpha.
            Default is 10.0.
        fixed_alpha (bool, optional): whether alpha should be trained to match a target entropy. Default is ``False``.
        target_entropy_weight (float, optional): weight for the target entropy term.
        target_entropy (Union[str, Number], optional): Target entropy for the stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is ``False``.

    """

    default_value_estimator = ValueEstimators.TD0
    delay_actor: bool = False

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        num_actions: int,
        *,
        num_qvalue_nets: int = 2,
        priority_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy_weight: float = 0.98,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
    ):
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist_params"],
        )

        self.delay_qvalue = delay_qvalue
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()),
        )
        self.num_qvalue_nets = num_qvalue_nets
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
            target_entropy = -float(np.log(1.0 / num_actions) * target_entropy_weight)
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )

    @property
    def alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.clone(False).select("next", *obs_keys, "action")

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
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = vmap(self.actor_network.get_dist_params)(
                tensordict_actor,
                actor_params,
            )
            if isinstance(self.actor_network, ProbabilisticActor):
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            else:
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            probs = tensordict_actor_dist.probs
            z = (probs == 0.0).float() * 1e-8
            logp_pi = torch.log(probs + z)
            logp_pi_pol = torch.sum(probs * logp_pi, dim=-1, keepdim=True)

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

        state_action_value = tensordict_qval.get("state_value").squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )

        loss_actor = -(
            (state_action_value_actor.min(0)[0] * probs[0]).sum(-1, keepdim=True)
            - self.alpha * logp_pi_pol[0]
        ).mean()

        pred_next_val = (
            probs[1]
            * (next_state_action_value_qvalue.min(0)[0] - self.alpha * logp_pi[1])
        ).sum(dim=-1, keepdim=True)

        tensordict_select.set(("next", self.value_estimator.value_key), pred_next_val)
        target_value = self.value_estimator.value_estimate(tensordict_select).squeeze(
            -1
        )

        actions = torch.argmax(tensordict_select["action"], dim=-1)

        pred_val_1 = (
            state_action_value_qvalue[0].gather(-1, actions.unsqueeze(-1)).unsqueeze(0)
        )
        pred_val_2 = (
            state_action_value_qvalue[1].gather(-1, actions.unsqueeze(-1)).unsqueeze(0)
        )
        pred_val = torch.cat([pred_val_1, pred_val_2], dim=0).squeeze()
        td_error = (pred_val - target_value.expand_as(pred_val)).pow(2)
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

        loss_alpha = self._loss_alpha(logp_pi_pol)
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
                "entropy": -logp_pi.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "action_log_prob_actor": logp_pi.mean().detach(),
                "next.state_value": pred_next_val.mean().detach(),
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
        value_net = None
        value_key = "state_value"
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
                value_target_key="value_target",
                value_key=value_key,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")
