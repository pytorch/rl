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
from torchrl.data.replay_buffers.utils import _ReplayBoundaryIndex

if TYPE_CHECKING:
    from torchrl.data.replay_buffers.storages import Storage

__all__ = ["SampleUnit", "Transition", "Sequence", "SubgoalTriplet"]


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

    @staticmethod
    def _repeat_info(info: dict[str, Any], repeats: int) -> dict[str, Any]:
        """Repeats every per-anchor entry of ``info``; scalar metadata is kept as is."""
        return {
            k: v
            if torch.as_tensor(v).ndim == 0
            else torch.as_tensor(v).repeat_interleave(repeats, dim=0)
            for k, v in info.items()
        }

    @staticmethod
    def _episode_boundaries(
        storage: Storage, done_key: NestedKey | None, device: torch.device
    ) -> _ReplayBoundaryIndex:
        """Episode boundaries of ``storage``, read from its ``done_key`` entry.

        Without a ``done_key`` the storage holds a single episode.
        """
        done = (
            storage.get(done_key)
            if done_key is not None
            else torch.zeros(storage.shape, dtype=torch.bool, device=device)
        )
        while done.ndim > storage.ndim and done.shape[-1] == 1:
            done = done.squeeze(-1)
        return _ReplayBoundaryIndex(
            end=done,
            at_capacity=storage._is_full,
            cursor=getattr(storage, "_last_cursor_index", None),
            device=device,
            storage=storage,
            source=("end", done_key),
            cache_values=True,
        )


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
    following it. ``dilation`` spaces the records of the window uniformly.

    This is useful when a learner needs temporal context around the records
    that contribute to its loss. For example, a recurrent Q-learning learner
    can replay the burn-in prefix to reconstruct its hidden state, compute
    losses only over the learning region, and use the bootstrap suffix as
    future context for a multi-step target. The unit only selects stored
    records and reports masks: it does not run the recurrent model or compute
    bootstrap targets.

    The configuration is fixed when the unit is constructed. If the replay
    buffer samples ``B`` anchors, the returned flat batch contains
    ``B * (burn_in + length + bootstrap)`` records. The corresponding storage
    span is ``1 + dilation * (burn_in + length + bootstrap - 1)`` records.

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
            boundary. A recurrent learner can process these records to
            reconstruct its hidden state while excluding them from the loss.
            Defaults to 0.
        bootstrap (int, optional): number of records following the learning
            region, marked False in ``"learning_mask"`` and subject to the
            ``episode_boundary`` policy at episode ends. These records provide
            future context to a target estimator; this unit does not compute a
            bootstrap value. Defaults to 0.
        dilation (int, optional): distance in storage records between
            consecutive records of the returned window. For example,
            ``dilation=2`` selects every other stored record. Dilation does not
            aggregate skipped transitions and does not control the spacing or
            overlap between independently sampled windows. Defaults to 1.

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
        and with ``dilation > 1`` the shifted window is laid out on the
        dilation grid of the shifted anchor: the reported (pre-shift) anchor
        is then not necessarily one of the window's records.

    For multidimensional storage, coordinate zero is time and the others are
    preserved lanes. ``"index"`` and ``"anchor_index"`` carry full storage
    coordinates.

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
        >>> # A recurrent learner could use one record to warm up its hidden
        >>> # state, learn on two records, and keep one future record for its
        >>> # target. It runs over the full window and applies the loss only
        >>> # where learning_mask and validity_mask are both true.
        >>> unit = Sequence(length=2, burn_in=1, bootstrap=1)
        >>> index, info = unit.expand(torch.tensor([5]), {}, rb.storage)
        >>> index.tolist()  # burn-in clamps at the episode start (5)
        [5, 5, 6, 7]
        >>> info["learning_mask"].tolist()
        [False, True, True, False]
        >>> info["validity_mask"].tolist()
        [False, True, True, True]
        >>> # Dilation temporally subsamples records inside the window:
        >>> index, _ = Sequence(length=3, dilation=2).expand(
        ...     torch.tensor([0]), {}, rb.storage
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
        dilation: int = 1,
    ):
        if length <= 0:
            raise ValueError(f"length must be strictly positive, got {length}.")
        if episode_boundary not in ("pad", "stop", "include_reset"):
            raise ValueError(f"Unknown episode_boundary {episode_boundary}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}.")
        if bootstrap < 0:
            raise ValueError(f"bootstrap must be non-negative, got {bootstrap}.")
        if dilation < 1:
            raise ValueError(f"dilation must be strictly positive, got {dilation}.")
        if done_key is not None and not isinstance(done_key, str):
            # normalize sequence-form nested keys (e.g. lists or omegaconf
            # containers coming from Hydra configs) to plain tuples
            done_key = tuple(done_key)
        self.length = length
        self.episode_boundary = episode_boundary
        self.done_key = done_key
        self.burn_in = burn_in
        self.bootstrap = bootstrap
        self.dilation = dilation

    @staticmethod
    def _newest_index(storage: Storage, written: int) -> int:
        """Physical index of the most recently written record."""
        cursor = getattr(storage, "_last_cursor_index", None)
        if cursor is not None:
            return cursor % written
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

    def expand(
        self,
        index: torch.Tensor | tuple,
        info: dict[str, Any],
        storage: Storage,
    ) -> tuple[torch.Tensor | tuple, dict[str, Any]]:
        self._check_storage(storage)
        multidimensional = storage.ndim > 1
        if isinstance(index, tuple):
            anchor = torch.stack(index, -1)
        else:
            anchor = index.clone()
        if multidimensional:
            if anchor.ndim != 2 or anchor.shape[-1] != storage.ndim:
                raise RuntimeError(
                    "Sequence expected multidimensional anchors with shape "
                    f"[batch, {storage.ndim}]."
                )
            anchor_time = anchor[:, 0]
        else:
            anchor_time = anchor
        B = anchor.shape[0]
        # All bookkeeping happens on the sampler's index device so that the
        # returned indices live on the same device as the ones Transition
        # (identity) would return.
        device = anchor.device
        total = self.burn_in + self.length + self.bootstrap

        expanded_info = self._repeat_info(info, total)

        steps = torch.arange(total, device=device, dtype=torch.long)
        learning = (steps >= self.burn_in) & (steps < self.burn_in + self.length)

        expanded_info["sequence_id"] = torch.arange(B, device=device).repeat_interleave(
            total
        )
        expanded_info["step_in_sequence"] = steps.repeat(B)
        expanded_info["learning_mask"] = learning.repeat(B)
        expanded_info["anchor_index"] = anchor.repeat_interleave(total, dim=0)

        offset = ((steps - self.burn_in) * self.dilation).unsqueeze(0).expand(B, total)

        if self.episode_boundary in ("pad", "stop"):
            boundary = self._episode_boundaries(storage, self.done_key, device)
            dist_from_start, dist_to_stop = boundary.distances(anchor)
            max_len = boundary.length

            anchor_eff = anchor_time
            if self.episode_boundary == "stop":
                shortfall = (
                    self.dilation * (self.length + self.bootstrap - 1) - dist_to_stop
                )
                shift = torch.clamp(
                    shortfall, min=torch.zeros_like(shortfall), max=dist_from_start
                )
                anchor_eff = anchor_time - shift
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
            written = storage.shape[0]
            newest = self._newest_index(storage, written)
            oldest = (newest + 1) % written if storage._is_full else 0
            newest = torch.as_tensor(newest, device=device, dtype=torch.long)
            oldest = torch.as_tensor(oldest, device=device, dtype=torch.long)
            dist_forward = torch.remainder(newest - anchor_time, written)
            dist_backward = torch.remainder(anchor_time - oldest, written)
            validity = (offset <= dist_forward.unsqueeze(1)) & (
                offset >= -dist_backward.unsqueeze(1)
            )
            clamped_offset = torch.minimum(offset, dist_forward.unsqueeze(1))
            clamped_offset = torch.maximum(clamped_offset, -dist_backward.unsqueeze(1))
            indices = torch.remainder(
                anchor_time.unsqueeze(1) + clamped_offset, written
            )

        expanded_info["validity_mask"] = validity.flatten()
        if multidimensional:
            lane = anchor[:, 1:].unsqueeze(1).expand(B, total, storage.ndim - 1)
            coordinates = torch.cat([indices.unsqueeze(-1), lane], -1).flatten(0, 1)
            return tuple(coordinates.unbind(-1)), expanded_info
        return indices.flatten(), expanded_info


class SubgoalTriplet(SampleUnit):
    """Expands anchors into in-trajectory ``(state, subgoal, goal, policy goal)`` records.

    Each anchor ``i`` expands into four records of its own episode, in this
    order:

    - the anchor ``i`` itself;
    - a subgoal ``k`` drawn uniformly in ``[i, j)``;
    - a goal ``j > i``;
    - a policy goal ``g > i`` drawn independently of ``j``.

    This is the sampling scheme of Transitive RL (Park et al., 2026), whose
    divide-and-conquer critic update regresses ``Q(s_i, a_i, s_j)`` toward
    ``Q(s_i, a_i, s_k) Q(s_k, a_k, s_j)``. Restricting ``k`` to states the
    trajectory actually visited between ``i`` and ``j`` is what keeps the
    update from exploiting arbitrary subgoals.

    If the buffer samples ``B`` anchors, the batch holds ``4 * B`` records.
    ``batch.unflatten(0, (B, 4))`` recovers one row per anchor. The unit
    reports a per-record ``"step_offset"`` info entry holding the distance in
    steps between each record and its anchor (``0`` for the anchor), so that
    losses can read the ``k - i`` and ``j - k`` horizons from the data. An
    anchor sampled on the last record of its episode is moved one step back,
    since no goal lies after it; ``"validity_mask"`` is ``False`` for the four
    records of an anchor whose episode has a single record.

    This unit requires a :class:`~torchrl.data.replay_buffers.TensorStorage`
    backed by a TensorDict with a one-dimensional batch, since episode
    boundaries are read from the stored ``done_key`` entry. Offsets are drawn
    from the buffer's generator when it has one on the anchors' device, so
    seeding the buffer makes its batches reproducible.

    Reference: Park, Oberai, Atreya, Levine. "Transitive RL: Value Learning
    via Divide and Conquer." ICLR 2026. https://arxiv.org/abs/2510.22512

    See also :class:`~torchrl.trainers.algorithms.configs.data.SubgoalTripletConfig`.

    Keyword Args:
        goal_discount (float or None, optional): distribution of the goal
            offset ``j - i``. A float ``gamma`` in ``(0, 1)`` draws the offset
            from a geometric distribution with success probability
            ``1 - gamma``, truncated at the episode end. ``None`` draws ``j``
            uniformly among the records after ``i``. Defaults to ``0.99``.
        policy_goal_discount (float or None, optional): same as
            ``goal_discount`` for the policy goal. Defaults to ``None``
            (uniform).
        done_key (NestedKey, optional): the key of the end-of-episode flag.
            Defaults to ``("next", "done")``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers import SubgoalTriplet
        >>> done = torch.zeros(20, 1, dtype=torch.bool)
        >>> done[9] = done[19] = True
        >>> data = TensorDict(
        ...     {"step": torch.arange(20), ("next", "done"): done}, batch_size=[20]
        ... )
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(20),
        ...     batch_size=8,
        ...     sample_unit=SubgoalTriplet(goal_discount=0.9),
        ... )
        >>> _ = rb.extend(data)
        >>> batch = rb.sample().unflatten(0, (8, 4))
        >>> batch.shape
        torch.Size([8, 4])
        >>> state, subgoal, goal = batch["step"].unbind(-1)[:3]
        >>> bool(((state <= subgoal) & (subgoal < goal)).all())
        True
        >>> bool((batch["step_offset"][:, 2] == goal - state).all())
        True
    """

    num_records = 4

    def __init__(
        self,
        *,
        goal_discount: float | None = 0.99,
        policy_goal_discount: float | None = None,
        done_key: NestedKey = ("next", "done"),
    ):
        for name, value in (
            ("goal_discount", goal_discount),
            ("policy_goal_discount", policy_goal_discount),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1) or None, got {value}.")
        if not isinstance(done_key, str):
            done_key = tuple(done_key)
        self.goal_discount = goal_discount
        self.policy_goal_discount = policy_goal_discount
        self.done_key = done_key

    @staticmethod
    def _offsets(
        horizon: torch.Tensor,
        discount: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Draws offsets in ``[1, horizon]`` (``horizon >= 1``)."""
        if discount is None:
            uniform = torch.rand(
                horizon.shape, device=horizon.device, generator=generator
            )
            return 1 + (uniform * horizon).long().clamp_max(horizon - 1)
        geometric = torch.empty(horizon.shape, device=horizon.device).geometric_(
            1.0 - discount, generator=generator
        )
        return torch.minimum(geometric.long(), horizon)

    def expand(
        self,
        index: torch.Tensor | tuple,
        info: dict[str, Any],
        storage: Storage,
    ) -> tuple[torch.Tensor | tuple, dict[str, Any]]:
        self._check_storage(storage)
        if storage.ndim != 1:
            raise TypeError(
                f"{type(self).__name__} requires a one-dimensional storage, got "
                f"{storage.ndim} dimensions."
            )
        if isinstance(index, tuple):
            (index,) = index
        anchor = index.to(torch.long)
        device = anchor.device
        generator = getattr(storage, "_rng", None)
        if generator is not None and generator.device.type != device.type:
            generator = None
        boundary = self._episode_boundaries(storage, self.done_key, device)
        dist_from_start, dist_to_stop = boundary.distances(anchor)
        length = boundary.length

        at_end = (dist_to_stop == 0) & (dist_from_start > 0)
        anchor = torch.where(at_end, anchor - 1, anchor) % length
        dist_to_stop = torch.where(at_end, torch.ones_like(dist_to_stop), dist_to_stop)
        valid = dist_to_stop > 0
        horizon = dist_to_stop.clamp_min(1)

        goal_offset = self._offsets(horizon, self.goal_discount, generator)
        subgoal_offset = (
            (
                torch.rand(goal_offset.shape, device=device, generator=generator)
                * goal_offset
            )
            .long()
            .clamp_max(goal_offset - 1)
        )
        policy_goal_offset = self._offsets(
            horizon, self.policy_goal_discount, generator
        )
        offsets = torch.stack(
            [
                torch.zeros_like(goal_offset),
                subgoal_offset,
                goal_offset,
                policy_goal_offset,
            ],
            -1,
        )
        offsets = torch.where(valid.unsqueeze(-1), offsets, torch.zeros_like(offsets))

        expanded_info = self._repeat_info(info, self.num_records)
        expanded_info["step_offset"] = offsets.flatten()
        expanded_info["anchor_index"] = anchor.repeat_interleave(self.num_records)
        expanded_info["validity_mask"] = valid.repeat_interleave(self.num_records)
        indices = (anchor.unsqueeze(-1) + offsets) % length
        return indices.flatten(), expanded_info
