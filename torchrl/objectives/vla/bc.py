# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Chunked behavior-cloning loss for Vision-Language-Action policies."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey

from torchrl.data.vla.schema import ACTION_CHUNK_KEY, ACTION_IS_PAD_KEY
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce

if TYPE_CHECKING:
    from torchrl.modules.vla import VLAWrapperBase

__all__ = ["VLABCLoss"]


class VLABCLoss(LossModule):
    """Chunked behavior-cloning loss for VLA policies.

    Regresses a VLA policy's predicted action chunk onto the expert action chunk
    with an L1 / L2 / smooth-L1 objective, masking the padded chunk steps via
    ``action_is_pad``. This is the OpenVLA-OFT / ACT fine-tuning objective:
    parallel-decoded continuous action chunks trained with a simple regression
    loss (`OpenVLA-OFT <https://arxiv.org/abs/2502.19645>`_,
    `ACT <https://arxiv.org/abs/2304.13705>`_).

    The actor is a continuous-head :class:`~torchrl.modules.vla.VLAWrapperBase`
    (e.g. :class:`~torchrl.modules.vla.TinyVLA`) whose ``action_chunk`` out-key
    matches the expert ``action_chunk`` key read here. Unlike most RL losses the
    actor is stored as a plain submodule (no functional parameter copy), so it
    works with lazily-initialized policies and exposes its parameters through
    ``loss.parameters()``.

    Args:
        actor_network (VLAWrapperBase): the continuous-head VLA policy to train.

    Keyword Args:
        loss_function (str or Callable): ``"l1"`` (default), ``"l2"``/``"mse"``,
            ``"smooth_l1"``, or a custom ``(pred, target) -> per-element loss``
            callable.
        reduction (str): ``"mean"`` (default), ``"sum"`` or ``"none"``.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorStack, TensorDict
        >>> from torchrl.modules.vla import TinyVLA
        >>> from torchrl.objectives.vla import VLABCLoss
        >>> policy = TinyVLA(action_dim=7, chunk_size=4)
        >>> loss = VLABCLoss(policy)
        >>> td = TensorDict(
        ...     {
        ...         "observation": {
        ...             "image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
        ...             "state": torch.zeros(2, 5),
        ...         },
        ...         "language_instruction": NonTensorStack("pick", "place"),
        ...         "action_chunk": torch.zeros(2, 4, 7),
        ...         "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        ...     },
        ...     batch_size=[2],
        ... )
        >>> loss(td)["loss_vla_bc"].shape
        torch.Size([])

    .. note::
        The expert chunk and the policy's prediction share the ``action_chunk``
        key: the expert is read before the actor runs, so the actor must not
        list ``action_chunk`` among its ``in_keys`` (VLA policies do not).

    .. seealso:: :class:`~torchrl.modules.vla.VLAWrapperBase`.
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys for :class:`VLABCLoss`.

        Attributes:
            action_chunk (NestedKey): expert action chunk (and the policy's
                prediction out-key). Defaults to ``"action_chunk"``.
            action_is_pad (NestedKey): boolean chunk-padding mask (optional;
                padded steps are excluded from the loss). Defaults to
                ``"action_is_pad"``. A wholly-padded batch yields NaN under
                ``"mean"`` reduction.
        """

        action_chunk: NestedKey = ACTION_CHUNK_KEY
        action_is_pad: NestedKey = ACTION_IS_PAD_KEY

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = ["loss_vla_bc"]

    def __init__(
        self,
        actor_network: VLAWrapperBase,
        *,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | Literal["l1", "l2", "mse", "smooth_l1"] = "l1",
        reduction: Literal["mean", "sum", "none"] | None = None,
    ) -> None:
        super().__init__()
        action_head = getattr(actor_network, "action_head", "continuous")
        if action_head != "continuous":
            raise ValueError(
                "VLABCLoss requires a continuous-head VLA policy "
                f"(action_head='continuous'), got action_head={action_head!r}. The "
                "token head emits action_tokens, not action_chunk -- use a token "
                "objective instead."
            )
        self._in_keys = None
        self.actor_network = actor_network
        self.loss_function = loss_function
        self.reduction = "mean" if reduction is None else reduction

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def set_keys(self, **kwargs) -> None:
        super().set_keys(**kwargs)
        self._in_keys = None  # invalidate the cached in_keys

    def _set_in_keys(self) -> None:
        keys = [
            self.tensor_keys.action_chunk,
            self.tensor_keys.action_is_pad,
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

    def _elementwise_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(self.loss_function, str):
            return self.loss_function(pred, target)
        if self.loss_function == "l1":
            return F.l1_loss(pred, target, reduction="none")
        if self.loss_function in ("l2", "mse"):
            return F.mse_loss(pred, target, reduction="none")
        if self.loss_function == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unsupported loss_function: {self.loss_function!r}.")

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the chunked behavior-cloning loss.

        Returns a TensorDict with the key ``"loss_vla_bc"``.
        """
        tensordict = tensordict.copy()
        expert = tensordict.get(self.tensor_keys.action_chunk)
        is_pad = tensordict.get(self.tensor_keys.action_is_pad, default=None)
        # the actor writes its prediction to the same action_chunk key; the
        # expert target was read above, and the actor does not read that key.
        tensordict = self.actor_network(tensordict)
        pred = tensordict.get(self.tensor_keys.action_chunk)
        if pred is expert:
            raise RuntimeError(
                f"The actor did not write a prediction at "
                f"{self.tensor_keys.action_chunk!r}: ensure it is a continuous-head "
                "VLA policy whose action_chunk out-key matches the loss key."
            )

        loss = self._elementwise_loss(pred, expert)
        valid = None if is_pad is None else (~is_pad).unsqueeze(-1).expand_as(loss)
        loss = _reduce(loss, self.reduction, mask=valid)
        return TensorDict({"loss_vla_bc": loss}, batch_size=[])
