# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce


class BCLoss(LossModule):
    """Behavior Cloning Loss Module.

    Implements behavior cloning loss for both stochastic and deterministic policies.
    Minimizes the negative log-likelihood: -E[log π(a_expert | s)] where π is the
    policy being trained and a_expert are the expert actions from the demonstration dataset.

    Works with any actor network that implements :meth:`~tensordict.nn.TensorDictModule.get_dist`
    method, including both
    stochastic and deterministic policies.

    Reference:
        "Integrating Behavior Cloning and Reinforcement Learning for Improved
        Performance in Dense and Sparse Reward Environments"
        https://arxiv.org/abs/1910.04281

    Args:
        actor_network (TensorDictModule): the actor network to be trained.

    Keyword Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

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
    the expected keyword arguments are the actor's ``in_keys`` + ``["action"]``.
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
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | Literal["l1", "l2", "mse", "smooth_l1", "cross_entropy"]
        | None = None,
        reduction: Literal["mean", "sum", "none"] | None = None,
    ) -> None:
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._in_keys = None

        self.convert_to_functional(
            actor_network,
            "actor_network",
        )

        self.reduction = reduction
        self.loss_function = loss_function

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
        tensordict = tensordict.copy()

        # Get expert action
        action_expert = tensordict.get(self.tensor_keys.action)

        # Forward pass through actor
        with self.actor_network_params.to_module(self.actor_network):
            tensordict = self.actor_network(tensordict)

            if self.loss_function is not None:
                # Use provided loss function on predicted and expert actions
                action_pred = tensordict.get("action")
                if isinstance(self.loss_function, str):
                    if self.loss_function == "l1":
                        loss = F.l1_loss(action_pred, action_expert, reduction="none")
                    elif self.loss_function == "l2" or self.loss_function == "mse":
                        loss = F.mse_loss(action_pred, action_expert, reduction="none")
                    elif self.loss_function == "smooth_l1":
                        loss = F.smooth_l1_loss(
                            action_pred, action_expert, reduction="none"
                        )
                    elif self.loss_function == "cross_entropy":
                        loss = F.cross_entropy(
                            action_pred,
                            action_expert.squeeze(-1)
                            if action_expert.ndim > 1
                            else action_expert,
                            reduction="none",
                        )
                    else:
                        raise ValueError(
                            f"Unsupported loss_function: {self.loss_function}"
                        )
                else:
                    loss = self.loss_function(action_pred, action_expert)
            elif "action" in tensordict:
                # Determine loss type based on action dtype and actor structure
                action_pred = tensordict.get("action")

                # Priority 1: If expert actions are discrete (integers), use cross-entropy
                if action_expert.dtype in (torch.long, torch.int32, torch.int64):
                    # For discrete actions: target is 1D class indices, prediction is [batch, num_classes]
                    loss = F.cross_entropy(
                        action_pred,
                        action_expert.squeeze(-1)
                        if action_expert.ndim > 1
                        else action_expert,
                        reduction="none",
                    )
                # Priority 2: Check if actor has distributional outputs (stochastic actor)
                elif hasattr(self.actor_network, "out_keys") and any(
                    k in self.actor_network.out_keys
                    for k in ["loc", "scale", "logits", "probs"]
                ):
                    # Stochastic actor: use NLL loss
                    dist = self.actor_network.get_dist(tensordict)
                    log_prob = dist.log_prob(action_expert)
                    loss = -log_prob
                else:
                    # Default: use MSE for continuous deterministic actions
                    loss = F.mse_loss(action_pred, action_expert, reduction="none")
            else:
                # Use distribution-based negative log probability
                dist = self.actor_network.get_dist(tensordict)
                log_prob = dist.log_prob(action_expert)
                loss = -log_prob

        loss = _reduce(loss, reduction=self.reduction)

        td_out = TensorDict({"loss_bc": loss})
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out
