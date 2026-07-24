# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchrl.data.replay_buffers.storages import Storage

__all__ = ["SampleUnit", "Transition"]


class SampleUnit(abc.ABC):
    """Expands sampled anchors into the records a batch is made of.

    Replay sampling combines two orthogonal decisions: which anchors are
    selected (the sampler's probability distribution) and what each anchor
    expands into (a single transition, a fixed-length sequence, a complete
    trajectory). A ``SampleUnit`` owns the second decision. The buffer calls
    :meth:`expand` inside its sampling critical section, after the anchor
    sampler ran and before the storage is read or any index bookkeeping
    happens, so the indices it returns are the ones the batch is built from
    and the ones reported in the sample info.

    Contract for implementations:

    - ``expand`` receives the anchor index (a tensor, or a tuple of
      coordinate tensors for multidimensional storages), the sampler's info
      dictionary and the storage. It returns the expanded index and info,
      which may be new objects; it must not mutate the storage.
    - Entries of ``info`` that are aligned with the anchors (for example
      priority weights) are the unit's responsibility: a unit that changes
      the number of records must expand or reduce those entries so they stay
      aligned with the index it returns.
    - Metadata describing the expansion (validity masks, learning masks,
      per-record anchor provenance) is communicated by adding entries to
      ``info``; scalar-per-record tensors are surfaced as keys of
      TensorDict samples automatically.

    .. seealso:: :class:`Transition`, the identity unit reproducing classic
        one-anchor-one-transition sampling.
    """

    @abc.abstractmethod
    def expand(
        self,
        index: torch.Tensor | tuple,
        info: dict[str, Any],
        storage: Storage,
    ) -> tuple[torch.Tensor | tuple, dict[str, Any]]:
        """Expands anchor indices into the final record indices of the batch.

        Args:
            index (torch.Tensor or tuple of torch.Tensor): the anchor indices
                selected by the sampler.
            info (dict): the sampler's info dictionary.
            storage (Storage): the storage the batch will be read from.

        Returns:
            A tuple ``(index, info)`` with the expanded indices and the
            (possibly augmented) info dictionary.
        """
        ...


class Transition(SampleUnit):
    """The identity sample unit: every anchor is one transition.

    This unit reproduces the classic replay-buffer behavior exactly and is
    the implicit default when no ``sample_unit`` is passed to the buffer:
    anchors selected by the sampler are the records of the batch, and the
    info dictionary is returned untouched.

    Examples:
        >>> import torch
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
        >>> from torchrl.data.replay_buffers import Transition
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(10),
        ...     batch_size=4,
        ...     sample_unit=Transition(),
        ... )
        >>> rb.extend(torch.arange(10))
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> sample = rb.sample()
        >>> sample.shape
        torch.Size([4])
    """

    def expand(
        self,
        index: torch.Tensor | tuple,
        info: dict[str, Any],
        storage: Storage,
    ) -> tuple[torch.Tensor | tuple, dict[str, Any]]:
        return index, info
