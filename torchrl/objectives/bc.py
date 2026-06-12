# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections.abc import Callable

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule


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

    Chunked (VLA-style) behavior cloning is the same loss with the action
    chunk as the ``action`` and the padding mask excluded via ``pad_mask``:

    Examples:
        >>> from tensordict.nn import TensorDictModule
        >>> chunk_actor = TensorDictModule(
        ...     nn.Sequential(nn.Linear(n_obs, 8), nn.Unflatten(-1, (2, 4))),
        ...     in_keys=["observation"],
        ...     out_keys=["action_chunk"],
        ... )
        >>> loss = BCLoss(chunk_actor, loss_function="l1")
        >>> loss.set_keys(action="action_chunk", pad_mask="action_is_pad")
        >>> data = TensorDict({
        ...     "observation": torch.randn(2, n_obs),
        ...     "action_chunk": torch.randn(2, 2, 4),
        ...     "action_is_pad": torch.tensor([[False, False], [False, True]]),
        ... }, [2])
        >>> loss(data)["loss_bc"].shape
        torch.Size([])

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to "action".

                .. versionchanged:: 0.14
                    The action key now also selects the actor's prediction:
                    the actor should write its prediction at this key (it
                    previously was read from the hardcoded ``"action"`` key,
                    regardless of ``set_keys``). An actor writing ``"action"``
                    while the loss uses a different action key still works
                    but emits a ``FutureWarning`` (removal in v0.16); an
                    actor writing neither key raises a ``RuntimeError``
                    instead of silently comparing the expert action with
                    itself.
            pad_mask (NestedKey, optional): a boolean entry marking padded
                action elements to exclude from the loss (``True`` = padded),
                e.g. the ``action_is_pad`` mask of chunked (VLA-style)
                behavior cloning. Trailing dimensions are broadcast: a
                ``[*B, H]`` mask applies to a ``[*B, H, action_dim]`` loss.
                Only supported with elementwise losses (not the
                distribution-based NLL or cross-entropy paths). A wholly
                padded batch yields ``NaN`` under ``"mean"`` reduction, and
                ``reduction="none"`` returns the flat 1D tensor of unmasked
                loss elements. Defaults to ``None`` (no masking).
        """

        action: NestedKey = "action"
        pad_mask: NestedKey | None = None

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

    def _get_prediction(
        self, tensordict: TensorDictBase, action_expert: torch.Tensor
    ) -> torch.Tensor:
        """Return the actor's prediction at the configured action key.

        Falls back to the legacy hardcoded ``"action"`` key (with a
        ``FutureWarning``) when the actor did not write the configured key,
        and raises if no prediction was written at all.
        """
        action_pred = tensordict.get(self.tensor_keys.action)
        if action_pred is not action_expert:
            return action_pred
        legacy = tensordict.get("action", default=None)
        if legacy is not None and legacy is not action_expert:
            warnings.warn(
                f"The actor wrote its prediction at 'action' while the loss's "
                f"action key is {self.tensor_keys.action!r}. Reading the "
                "prediction from the hardcoded 'action' key is deprecated and "
                "will be removed in v0.16: make the actor's out_keys match "
                "the loss's action key.",
                FutureWarning,
            )
            return legacy
        raise RuntimeError(
            f"The actor did not write a prediction at "
            f"{self.tensor_keys.action!r}: make sure the actor's "
            "out_keys match the loss's action key."
        )

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            *self.actor_network.in_keys,
        ]
        if self.tensor_keys.pad_mask is not None:
            keys.append(self.tensor_keys.pad_mask)
        self._in_keys = list(set(keys))

    def set_keys(self, **kwargs) -> None:
        super().set_keys(**kwargs)
        self._in_keys = None  # invalidate the cached in_keys

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
        with self.actor_network_params.to_module(
            self.actor_network, preserve_module_state=False
        ):
            tensordict = self.actor_network(tensordict)

            if self.loss_function is not None:
                # Use provided loss function on predicted and expert actions
                action_pred = self._get_prediction(tensordict, action_expert)
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
            elif self.tensor_keys.action in tensordict:
                # Determine loss type based on action dtype and actor structure

                # Priority 1: If expert actions are discrete (integers), use cross-entropy
                if action_expert.dtype in (torch.long, torch.int32, torch.int64):
                    # For discrete actions: target is 1D class indices, prediction is [batch, num_classes]
                    action_pred = self._get_prediction(tensordict, action_expert)
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
                    action_pred = self._get_prediction(tensordict, action_expert)
                    loss = F.mse_loss(action_pred, action_expert, reduction="none")
            else:
                # Use distribution-based negative log probability
                dist = self.actor_network.get_dist(tensordict)
                log_prob = dist.log_prob(action_expert)
                loss = -log_prob

        # Route through ``_reduce_loss`` so that, when the input tensordict
        # carries the ``("collector", "mask")`` key written by SliceSampler
        # (PR #3695, ``pad_output=True``), padded positions are excluded from
        # the time-averaging. Behavior is byte-identical to the old
        # ``_reduce(loss, reduction=self.reduction)`` path when no mask key
        # is present.
        loss = self._reduce_loss(loss, tensordict=tensordict)
        mask = None
        if self.tensor_keys.pad_mask is not None:
            pad = tensordict.get(self.tensor_keys.pad_mask, default=None)
            if pad is not None:
                mask = ~pad
                if mask.ndim > loss.ndim:
                    raise RuntimeError(
                        f"pad_mask {self.tensor_keys.pad_mask!r} has more "
                        f"dimensions ({mask.ndim}) than the computed loss "
                        f"({loss.ndim}): per-element masking requires an "
                        "elementwise loss (e.g. loss_function='l1'), not a "
                        "distribution-based (NLL) or cross-entropy loss."
                    )
                while mask.ndim < loss.ndim:
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(loss)
        loss = _reduce(loss, reduction=self.reduction, mask=mask)

        td_out = TensorDict({"loss_bc": loss})
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
        )
        return td_out
