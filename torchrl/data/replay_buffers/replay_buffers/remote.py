# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch


from typing import TYPE_CHECKING, TypeVar

from tensordict import TensorDictBase

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl._utils import accept_remote_rref_udf_invocation

T = TypeVar("T")
if TYPE_CHECKING:
    from typing import Self
else:
    Self = T


from .tensordict import TensorDictReplayBuffer


@accept_remote_rref_udf_invocation
class RemoteTensorDictReplayBuffer(TensorDictReplayBuffer):
    """A remote invocation friendly ReplayBuffer class. Public methods can be invoked by remote agents using `torch.rpc` or called locally as normal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        batch_size: int | None = None,
        include_info: bool | None = None,
        return_info: bool = False,
    ) -> TensorDictBase:
        return super().sample(
            batch_size=batch_size, include_info=include_info, return_info=return_info
        )

    def add(self, data: TensorDictBase) -> int:
        return super().add(data)

    def extend(
        self, tensordicts: list | TensorDictBase, *, update_priority: bool | None = None
    ) -> torch.Tensor:
        return super().extend(tensordicts, update_priority=update_priority)

    def update_priority(
        self, index: int | torch.Tensor, priority: int | torch.Tensor
    ) -> None:
        return super().update_priority(index, priority)

    def update_tensordict_priority(self, data: TensorDictBase) -> None:
        return super().update_tensordict_priority(data)
