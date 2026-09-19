# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey

from torchrl.modules import FlowMatchingPolicy, OneStepPolicy
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _vmap_func,
    dispatch_value_estimator,
    ValueEstimators,
)


class FQLLoss(LossModule):
    """Flow Q-learning for normalized continuous actions.

    Implements https://arxiv.org/abs/2502.02538. The flow learns the behavior
    distribution; the one-step actor balances distillation and mean critic value.
    Only the critics have target parameters, updated with :class:`SoftUpdate`.

    Args:
        flow_policy (FlowMatchingPolicy): behavior flow policy.
        actor_network (OneStepPolicy): noise-conditioned one-step policy.
        qvalue_network (TensorDictModule or list): critic reading observations
            and actions and writing ``state_action_value``. A single critic is
            duplicated; a list supplies independently initialized critics.

    Keyword Args:
        num_qvalue_nets (int, optional): number of critics. Defaults to 2.
        alpha (float, optional): distillation weight. Defaults to 10.
        q_aggregation (str, optional): target critic aggregation, ``mean`` or
            ``min``. The actor always uses the mean. Defaults to ``mean``.
        normalize_q_loss (bool, optional): divide actor Q loss by the detached
            mean absolute Q value. Defaults to False.
        reduction (str, optional): ``none``, ``mean`` or ``sum``. Defaults to
            ``mean``. Action and critic coordinates are always averaged first.
    """

    @dataclass
    class _AcceptedKeys:
        """TensorDict keys configurable through :meth:`set_keys`."""

        observation: NestedKey = "observation"
        action: NestedKey = "action"
        value: NestedKey = "state_action_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        priority: NestedKey = "td_error"
        priority_weight: NestedKey = "priority_weight"

    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0
    out_keys = ["loss_flow", "loss_actor", "loss_qvalue"]
    flow_policy: FlowMatchingPolicy
    actor_network: OneStepPolicy
    qvalue_network: TensorDictModule
    flow_policy_params: TensorDictParams
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    target_flow_policy_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        flow_policy: FlowMatchingPolicy,
        actor_network: OneStepPolicy,
        qvalue_network: TensorDictModule | list[TensorDictModule],
        *,
        num_qvalue_nets: int = 2,
        alpha: float = 10.0,
        q_aggregation: str = "mean",
        normalize_q_loss: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        if q_aggregation not in ("mean", "min"):
            raise ValueError("q_aggregation must be 'mean' or 'min'")
        self.convert_to_functional(flow_policy, "flow_policy")
        self.convert_to_functional(actor_network, "actor_network")
        self.convert_to_functional(
            qvalue_network, "qvalue_network", num_qvalue_nets, create_target_params=True
        )
        self.alpha = alpha
        self.q_aggregation = q_aggregation
        self.normalize_q_loss = normalize_q_loss
        self.reduction = reduction
        self.set_vmap_randomness(self.vmap_randomness)

    @property
    def in_keys(self):
        keys = self.tensor_keys
        current = [keys.observation, keys.action, *self.qvalue_network.in_keys]
        following = [
            ("next", key)
            for key in [*current, keys.reward, keys.done, keys.terminated]
            if key != keys.action
        ]
        return list(dict.fromkeys(current + following))

    def set_vmap_randomness(self, value):
        if value not in ("error", "same", "different"):
            raise ValueError("vmap randomness must be 'error', 'same' or 'different'")
        self._vmap_randomness = value
        self.qvalue_vmap = _vmap_func(
            self.qvalue_network, in_dims=(None, 0), randomness=value
        )

    def reduce_loss(self, loss, tensordict, *, reduction=None):
        return self._reduce_loss(
            loss,
            tensordict,
            reduction=reduction,
            weights=self._maybe_get_priority_weight(tensordict),
        )

    def flow_loss(self, tensordict):
        action = tensordict.get(self.tensor_keys.action)
        observation = tensordict.get(self.tensor_keys.observation)
        noise = torch.randn_like(action)
        time = torch.rand_like(action[..., :1])
        interpolated = (1 - time) * noise + time * action
        with self.flow_policy_params.to_module(self.flow_policy):
            velocity = self.flow_policy.velocity(observation, interpolated, time)
        loss = (velocity - (action - noise)).square().mean(-1)
        return self.reduce_loss(loss, tensordict)

    def actor_loss(self, tensordict):
        observation = tensordict.get(self.tensor_keys.observation)
        noise = torch.randn_like(tensordict.get(self.tensor_keys.action))
        with torch.no_grad(), self.flow_policy_params.to_module(self.flow_policy):
            target_action = self.flow_policy(observation, noise)
        with self.actor_network_params.to_module(self.actor_network):
            action = self.actor_network(observation, noise, clamp=False)
        distillation = (action - target_action).square().mean(-1)
        actor_td = tensordict.select(*self.qvalue_network.in_keys)
        actor_td.set(self.tensor_keys.action, action.clamp(-1, 1))
        # Frozen critic weights still transmit gradients to the sampled action.
        qvalue = self.qvalue_vmap(actor_td, self.qvalue_network_params.detach())
        qvalue = qvalue.get(self.tensor_keys.value).squeeze(-1).mean(0)
        q_loss = -qvalue
        if self.normalize_q_loss:
            scale = self.reduce_loss(
                qvalue.detach().abs(), tensordict, reduction="mean"
            )
            q_loss = q_loss / scale.clamp_min(torch.finfo(qvalue.dtype).eps)
        loss = self.reduce_loss(self.alpha * distillation + q_loss, tensordict)
        return loss, {
            "distillation_loss": distillation.detach(),
            "q_loss": q_loss.detach(),
        }

    def qvalue_loss(self, tensordict):
        keys = self.tensor_keys
        td = tensordict.clone(False)
        with torch.no_grad():
            next_td = td.get("next")
            with self.actor_network_params.to_module(self.actor_network):
                action = self.actor_network(next_td.get(keys.observation))
            next_td.set(keys.action, action)
            next_q = self.qvalue_vmap(next_td, self.target_qvalue_network_params)
            next_q = next_q.get(keys.value)
            next_value = (
                next_q.mean(0) if self.q_aggregation == "mean" else next_q.min(0).values
            )
            td.set(("next", keys.value), next_value)
            target = self.value_estimator.value_estimate(td).squeeze(-1)
        prediction = self.qvalue_vmap(
            td.select(*self.qvalue_network.in_keys), self.qvalue_network_params
        )
        error = (prediction.get(keys.value).squeeze(-1) - target).square()
        loss = self.reduce_loss(error.mean(0), tensordict)
        return loss, {"td_error": error.detach().max(0).values, "target_value": target}

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        loss_flow = self.flow_loss(tensordict)
        loss_actor, actor_metadata = self.actor_loss(tensordict)
        loss_qvalue, critic_metadata = self.qvalue_loss(tensordict)
        tensordict.set(self.tensor_keys.priority, critic_metadata.pop("td_error"))
        result = TensorDict(
            {
                "loss_flow": loss_flow,
                "loss_actor": loss_actor,
                "loss_qvalue": loss_qvalue,
                **actor_metadata,
                **critic_metadata,
            }
        )
        self._clear_weakrefs(
            tensordict,
            result,
            "flow_policy_params",
            "actor_network_params",
            "qvalue_network_params",
            "target_qvalue_network_params",
        )
        return result

    def make_value_estimator(self, value_type=None, **hyperparams):
        value_type, hp = self._prepare_value_estimator_kwargs(value_type, **hyperparams)
        if value_type is not None:
            dispatch_value_estimator(
                self,
                value_type,
                supported=(ValueEstimators.TD0,),
                value_network=None,
                **hp,
            )
        self._forward_value_estimator_keys()
        return self
