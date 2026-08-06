# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import is_tensor_collection
from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.utils import _derive_end_flags, _end_to_start_stop

if TYPE_CHECKING:
    from torchrl.data.replay_buffers.storages import Storage

__all__ = ["SampleUnit", "Transition", "Sequence"]


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

    .. seealso:: :class:`~torchrl.trainers.algorithms.configs.data.TransitionConfig`
        for the Hydra configuration companion.

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


class Sequence(SampleUnit):
    """Expands anchors into a window of records around each anchor.

    Each anchor expands into ``burn_in + length + bootstrap`` records:
    ``burn_in`` records preceding the anchor, the learning region of
    ``length`` records starting at the anchor, then ``bootstrap`` records
    following it. ``stride`` spaces the records of the window uniformly.

    This unit requires a :class:`~torchrl.data.replay_buffers.TensorStorage`
    backed by a TensorDict (e.g. :class:`~torchrl.data.LazyTensorStorage`
    filled with TensorDict data), since episode boundaries are read from the
    stored ``done_key`` entry.

    Args:
        length (int): the length of the learning region of the sequences.
        episode_boundary (str, optional): boundary policy. One of:

            - ``"pad"``: repeat the last valid state if a boundary is reached,
              marking padded steps as invalid in the ``"validity_mask"`` info
              entry.
            - ``"stop"``: shift the anchor backward so the sequence ends
              exactly at the boundary, falling back to pad if the episode is
              shorter than ``length``.
            - ``"include_reset"``: cross episode boundaries. The write seam
              (the boundary between the newest and the oldest record of the
              ring buffer) and unwritten slots are never crossed: records
              beyond it are clamped and marked invalid.

            Defaults to ``"pad"``.
        done_key (NestedKey, optional): the key for the end-of-episode flag.
            If ``None``, the written storage span is treated as one trajectory,
            bounded only by the replay buffer's write seam. Defaults to
            ``("next", "done")``.
        burn_in (int, optional): number of records preceding the anchor,
            marked False in the ``"learning_mask"`` info entry. Burn-in never
            shifts the anchor: entries before the anchor's episode start (or
            before the oldest written record) are invalid and clamp to that
            boundary. Defaults to 0.
        bootstrap (int, optional): number of records following the learning
            region, marked False in ``"learning_mask"`` and subject to the
            ``episode_boundary`` policy at episode ends. Defaults to 0.
        stride (int, optional): spacing between the records of the window.
            Defaults to 1.

    After expansion, ``info["index"]`` (and the ``"index"`` entry of
    TensorDict samples) holds the expanded per-record storage indices of the
    window records, not the anchors. The unit additionally reports a
    per-record ``"anchor_index"`` info entry holding the storage index of
    each record's sampled anchor, so priorities of sampled sequences can be
    updated per anchor through ``update_priority``.
    :meth:`TensorDictReplayBuffer.update_tensordict_priority
    <torchrl.data.TensorDictReplayBuffer.update_tensordict_priority>` uses it
    automatically: per-record priorities are reduced (max over the valid
    records of each window) and written to the anchors only, so padded or
    bootstrap records never pollute the priorities of unrelated anchors.

    .. note:: ``"anchor_index"`` always reports the anchor the sampler drew.
        With ``episode_boundary="stop"`` the window may be shifted backward,
        and with ``stride > 1`` the shifted window is laid out on the stride
        grid of the shifted anchor: the reported (pre-shift) anchor is then
        not necessarily one of the window's records.

    .. seealso:: :class:`~torchrl.trainers.algorithms.configs.data.SequenceConfig`
        for the Hydra configuration companion.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer, Sequence
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(10),
        ...     batch_size=2,
        ...     sample_unit=Sequence(length=3),
        ... )
        >>> done = torch.zeros(10, 1, dtype=torch.bool)
        >>> done[4] = done[9] = True
        >>> rb.extend(TensorDict(
        ...     {
        ...         "obs": torch.arange(10, dtype=torch.float32),
        ...         ("next", "done"): done,
        ...     },
        ...     batch_size=[10],
        ... ))
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> sample, info = rb.sample(return_info=True)
        >>> sample["obs"].shape  # 2 anchors x 3 records each
        torch.Size([6])
        >>> sorted(info.keys())
        ['anchor_index', 'index', 'learning_mask', 'sequence_id', 'step_in_sequence', 'validity_mask']
        >>> # burn-in and bootstrap extend the window around the anchor:
        >>> unit = Sequence(length=2, burn_in=1, bootstrap=1)
        >>> index, info = unit.expand(torch.tensor([5]), {}, rb._storage)
        >>> index.tolist()  # burn-in clamps at the episode start (5)
        [5, 5, 6, 7]
        >>> info["learning_mask"].tolist()
        [False, True, True, False]
        >>> info["validity_mask"].tolist()
        [False, True, True, True]
        >>> # stride spaces the window records uniformly:
        >>> index, _ = Sequence(length=3, stride=2).expand(
        ...     torch.tensor([0]), {}, rb._storage
        ... )
        >>> index.tolist()
        [0, 2, 4]
    """

    def __init__(
        self,
        length: int,
        episode_boundary: Literal["pad", "stop", "include_reset"] = "pad",
        done_key: NestedKey | None = ("next", "done"),
        burn_in: int = 0,
        bootstrap: int = 0,
        stride: int = 1,
    ):
        if length <= 0:
            raise ValueError(f"length must be strictly positive, got {length}.")
        if episode_boundary not in ("pad", "stop", "include_reset"):
            raise ValueError(f"Unknown episode_boundary {episode_boundary}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}.")
        if bootstrap < 0:
            raise ValueError(f"bootstrap must be non-negative, got {bootstrap}.")
        if stride < 1:
            raise ValueError(f"stride must be strictly positive, got {stride}.")
        if done_key is not None and not isinstance(done_key, str):
            # normalize sequence-form nested keys (e.g. lists or omegaconf
            # containers coming from Hydra configs) to plain tuples
            done_key = tuple(done_key)
        self.length = length
        self.episode_boundary = episode_boundary
        self.done_key = done_key
        self.burn_in = burn_in
        self.bootstrap = bootstrap
        self.stride = stride

    @staticmethod
    def _newest_index(storage: Storage, written: int) -> int:
        """Physical index of the most recently written record."""
        cursor = getattr(storage, "_last_cursor", None)
        if isinstance(cursor, torch.Tensor):
            cursor = cursor.reshape(-1)
            if cursor.numel():
                return int(cursor[-1].item()) % written
        elif isinstance(cursor, range):
            if len(cursor):
                return int(cursor[-1]) % written
        elif isinstance(cursor, int):
            return cursor % written
        return written - 1

    def _check_storage(self, storage: Storage) -> None:
        if not isinstance(storage, TensorStorage):
            raise TypeError(
                f"{type(self).__name__} requires a TensorDict-backed TensorStorage "
                f"(e.g. LazyTensorStorage or LazyMemmapStorage written with "
                f"TensorDict data) to recover episode boundaries and the write "
                f"cursor; got {type(storage).__name__}."
            )
        contents = getattr(storage, "_storage", None)
        if contents is not None and not is_tensor_collection(contents):
            raise TypeError(
                f"{type(self).__name__} requires the TensorStorage to hold a "
                f"TensorDict (or other tensor collection) so that the "
                f"'{self.done_key}' entry can be read; the storage holds "
                f"{type(contents).__name__} instead."
            )

    def expand(
        self,
        index: torch.Tensor | tuple,
        info: dict[str, Any],
        storage: Storage,
    ) -> tuple[torch.Tensor | tuple, dict[str, Any]]:
        if isinstance(index, tuple):
            raise NotImplementedError(
                "Multidimensional storage not yet supported by Sequence."
            )
        self._check_storage(storage)

        anchor = index.clone()
        B = anchor.shape[0]
        # All bookkeeping happens on the sampler's index device so that the
        # returned indices live on the same device as the ones Transition
        # (identity) would return.
        device = anchor.device
        total = self.burn_in + self.length + self.bootstrap

        expanded_info = {}
        for k, v in info.items():
            val = torch.as_tensor(v)
            if val.ndim == 0:
                # scalar metadata is not per-anchor: leave it untouched
                expanded_info[k] = v
            else:
                expanded_info[k] = val.repeat_interleave(total, dim=0)

        steps = torch.arange(total, device=device, dtype=torch.long)
        learning = (steps >= self.burn_in) & (steps < self.burn_in + self.length)

        expanded_info["sequence_id"] = torch.arange(B, device=device).repeat_interleave(
            total
        )
        expanded_info["step_in_sequence"] = steps.repeat(B)
        expanded_info["learning_mask"] = learning.repeat(B)
        expanded_info["anchor_index"] = anchor.repeat_interleave(total)

        offset = ((steps - self.burn_in) * self.stride).unsqueeze(0).expand(B, total)

        if self.episode_boundary in ("pad", "stop"):
            done = (
                storage.get(self.done_key)
                if self.done_key is not None
                else torch.zeros(len(storage), dtype=torch.bool, device=device)
            )
            end, max_len = _derive_end_flags(
                end=done,
                at_capacity=storage._is_full,
                cursor=getattr(storage, "_last_cursor", None),
            )
            # _end_to_start_stop returns its indices on the device of ``end``
            # (the storage device): move the flags first so start/stop live
            # on the anchor device.
            end = end.to(device)
            start, stop, _ = _end_to_start_stop(end=end, length=max_len, device=device)
            start = start[:, 0]
            stop = stop[:, 0]

            start_exp = start.unsqueeze(0)
            stop_exp = stop.unsqueeze(0)
            a_exp = anchor.unsqueeze(1)

            cond1 = (start_exp <= stop_exp) & (start_exp <= a_exp) & (a_exp <= stop_exp)
            cond2 = (start_exp > stop_exp) & (
                (a_exp >= start_exp) | (a_exp <= stop_exp)
            )
            mask = cond1 | cond2

            traj_idx = mask.float().argmax(dim=1)
            a_start = start[traj_idx]
            a_stop = stop[traj_idx]

            dist_to_stop = ((a_stop - anchor) % max_len).to(torch.long)
            dist_from_start = ((anchor - a_start) % max_len).to(torch.long)

            anchor_eff = anchor
            if self.episode_boundary == "stop":
                shortfall = (
                    self.stride * (self.length + self.bootstrap - 1) - dist_to_stop
                )
                shift = torch.clamp(
                    shortfall, min=torch.zeros_like(shortfall), max=dist_from_start
                )
                anchor_eff = anchor - shift
                dist_to_stop = dist_to_stop + shift
                dist_from_start = dist_from_start - shift

            validity = (offset <= dist_to_stop.unsqueeze(1)) & (
                offset >= -dist_from_start.unsqueeze(1)
            )
            clamped_offset = torch.minimum(offset, dist_to_stop.unsqueeze(1))
            clamped_offset = torch.maximum(
                clamped_offset, -dist_from_start.unsqueeze(1)
            )
            indices = (anchor_eff.unsqueeze(1) + clamped_offset) % max_len
        else:
            # "include_reset": cross episode boundaries, but never cross the
            # write seam (between the newest and the oldest record of the
            # ring buffer) nor read slots that were never written -- in
            # either direction, since burn-in walks backward from the anchor.
            written = len(storage)
            newest = self._newest_index(storage, written)
            oldest = (newest + 1) % written if storage._is_full else 0
            newest = torch.as_tensor(newest, device=device, dtype=torch.long)
            oldest = torch.as_tensor(oldest, device=device, dtype=torch.long)
            dist_forward = torch.remainder(newest - anchor, written)
            dist_backward = torch.remainder(anchor - oldest, written)
            validity = (offset <= dist_forward.unsqueeze(1)) & (
                offset >= -dist_backward.unsqueeze(1)
            )
            clamped_offset = torch.minimum(offset, dist_forward.unsqueeze(1))
            clamped_offset = torch.maximum(clamped_offset, -dist_backward.unsqueeze(1))
            indices = torch.remainder(anchor.unsqueeze(1) + clamped_offset, written)

        expanded_info["validity_mask"] = validity.flatten()

        return indices.flatten(), expanded_info
