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

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce, distance_loss


class BCLoss(LossModule):
    """Behavior Cloning Loss Module.

    Implements behavior cloning loss for both stochastic and deterministic policies.
    For stochastic policies, minimizes the negative log-likelihood: -E[log π(a_expert | s)].
    For deterministic policies, minimizes the distance between predicted and expert actions.

    The policy type is auto-detected based on whether the actor network outputs a log_prob key.

    Args:
        actor_network (TensorDictModule): the actor network to be trained.

    Keyword Args:
        loss_function (str, optional): loss function for deterministic policies.
            Can be one of "smooth_l1", "l2", "l1". Default is "l2".
        reduction (str, optional): Specifies the reduction to apply to the output:
            "none" | "mean" | "sum". "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output, "sum": the output will be summed. Default: "mean".

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.modules.tensordict_module.actors import Actor
        >>> from torchrl.objectives.bc import BCLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> module = nn.Linear(n_obs, n_act)
        >>> actor = Actor(module=module, spec=spec)
        >>> loss = BCLoss(actor)
        >>> batch = [2, ]
        >>> data = TensorDict({
        ...     "observation": torch.randn(*batch, n_obs),
        ...     "action": spec.rand(batch),
        ... }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss_bc: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are the actor's in_keys + ["action"].
    The return value is a tensor corresponding to the loss.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.modules.tensordict_module.actors import Actor
        >>> from torchrl.objectives.bc import BCLoss
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> module = nn.Linear(n_obs, n_act)
        >>> actor = Actor(module=module, spec=spec)
        >>> loss = BCLoss(actor)
        >>> _ = loss.select_out_keys("loss_bc")
        >>> batch = [2, ]
        >>> loss_bc = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=spec.rand(batch))
        >>> loss_bc.backward()

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to "action".
        """

        action: NestedKey = "action"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = ["loss_bc"]

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModule,
        *,
        loss_function: str = "l2",
        reduction: str | None = None,
    ) -> None:
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._in_keys = None

        self.convert_to_functional(
            actor_network,
            "actor_network",
        )

        self.loss_function = loss_function
        self.reduction = reduction

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            *self.actor_network.in_keys,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the behavior cloning loss.

        Args:
            tensordict (TensorDictBase): input data containing observations and expert actions.

        Returns:
            TensorDict with key "loss_bc".
        """
        tensordict = tensordict.clone(False)

        # Get expert action
        action_expert = tensordict.get(self.tensor_keys.action)

        # Forward pass through actor
        with self.actor_network_params.to_module(self.actor_network):
            tensordict_actor = self.actor_network(tensordict)

        # Check if stochastic (has log_prob) or deterministic
        if "log_prob" in tensordict_actor.keys():
            # Stochastic policy: minimize -log π(a_expert | s)
            log_prob = tensordict_actor.get("log_prob")
            loss = -log_prob
        else:
            # Deterministic policy: minimize distance(a_pred, a_expert)
            action_pred = tensordict_actor.get(self.tensor_keys.action)
            loss = distance_loss(
                action_pred,
                action_expert,
                loss_function=self.loss_function,
            )

        loss = _reduce(loss, reduction=self.reduction)

        td_out = TensorDict({"loss_bc": loss}, batch_size=[])
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out
