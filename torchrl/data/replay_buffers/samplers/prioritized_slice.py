# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from tensordict.utils import NestedKey
from torchrl._utils import _replace_last
from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import unravel_index

from .prioritized import PrioritizedSampler
from .slice import SliceSampler


class PrioritizedSliceSampler(SliceSampler, PrioritizedSampler):
    r"""Samples slices of data along the first dimension, given start and stop signals, using prioritized sampling.

    This class combines trajectory sampling with Prioritized Experience Replay (PER) as presented in
    "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay."
    (https://arxiv.org/abs/1511.05952)

    **Core Idea**: Instead of sampling trajectory slices uniformly, this sampler prioritizes
    trajectory start points based on the importance of the transitions at those positions.
    This allows focusing learning on the most informative parts of trajectories.

    **How it works**:
    1. Each transition is assigned a priority based on its TD error: :math:`p_i = |\\delta_i| + \\epsilon`
    2. Trajectory start points are sampled with probability: :math:`P(i) = \frac{p_i^\alpha}{\\sum_j p_j^\alpha}`
    3. Importance sampling weights correct for bias: :math:`w_i = (N \\cdot P(i))^{-\beta}`
    4. Complete trajectory slices are extracted from the sampled start points

    For more info see :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` and :class:`~torchrl.data.replay_buffers.samplers.PrioritizedSampler`.

    .. warning:: PrioritizedSliceSampler will look at the priorities of the individual transitions and sample the
        start points accordingly. This means that transitions with a low priority may as well appear in the
        samples if they follow another of higher priority, and transitions with a high priority but closer to the
        end of a trajectory may never be sampled if they cannot be used as start points.
        Currently, it is the user responsibility to aggregate priorities across items of a trajectory using
        :meth:`update_priority`.

    Args:
        max_capacity (int): maximum capacity of the buffer.
        alpha (:obj:`float`): exponent :math:`\alpha` determines how much prioritization is used.
            - :math:`\alpha = 0`: uniform sampling of trajectory start points
            - :math:`\alpha = 1`: full prioritization based on TD error magnitude at start points
            - Typical values: 0.4-0.7 for balanced prioritization
            - Higher :math:`\alpha` means more aggressive prioritization of high-error trajectory regions
        beta (:obj:`float`): importance sampling negative exponent :math:`\beta`.
            - :math:`\beta` controls the correction for the bias introduced by prioritization
            - :math:`\beta = 0`: no correction (biased towards high-priority trajectory regions)
            - :math:`\beta = 1`: full correction (unbiased but potentially unstable)
            - Typical values: start at 0.4-0.6 and anneal to 1.0 during training
            - Lower :math:`\beta` early in training provides stability, higher :math:`\beta` later reduces bias
        eps (:obj:`float`, optional): small constant added to priorities to ensure
            no transition has zero priority. This prevents trajectory regions from never
            being sampled. Defaults to 1e-8.
        reduction (str, optional): the reduction method for multidimensional
            tensordicts (i.e., stored trajectory). Can be one of "max", "min",
            "median" or "mean".

    **Parameter Guidelines**:

    - **:math:`\alpha` (alpha)**: Controls how much to prioritize high-error trajectory regions.
      0.4-0.7: Good balance between learning speed and stability.
      1.0: Maximum prioritization (may be unstable).
      0.0: Uniform sampling (no prioritization benefit).

    - **:math:`\beta` (beta)**: Controls importance sampling correction.
      Start at 0.4-0.6 for training stability.
      Anneal to 1.0 over training to reduce bias.
      Lower values = more stable but biased.
      Higher values = less biased but potentially unstable.

    - **:math:`\\epsilon`**: Small constant to prevent zero priorities.
      1e-8: Good default value.
      Too small: may cause numerical issues.
      Too large: reduces prioritization effect.

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
            Exclusive with ``end_keys``.

            .. note:: A single ``end_key`` misses trajectories whose end is
                marked by another flag only (e.g. datasets carrying
                ``truncated=True`` ends without an aggregate ``done`` entry
                -- those get silently merged with the next trajectory). Pass
                ``end_keys`` to apply the
                :data:`~torchrl.data.DEFAULT_DONE_KEYS` union convention.
        end_keys (sequence of NestedKey, optional): a sequence of keys whose
            entries are OR-ed together to build the end-of-trajectory signal.
            Keys absent from the storage are skipped (at least one must be
            present). Use
            ``[("next", key) for key in DEFAULT_DONE_KEYS]`` to union
            ``done``, ``truncated`` and ``terminated``. Exclusive with
            ``end_key``. Defaults to ``None`` (use ``end_key``).
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        cache_values (bool, optional): to be used with static datasets.
            Caches trajectory boundaries until the storage revision changes,
            including writes from another buffer or process. Direct backing
            tensor mutations cannot be detected.

        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch. If ``True``, trajectories shorted
            than required will be filtered out.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.
        compile (bool or dict of kwargs, optional): if ``True``, the bottleneck of
            the :meth:`~sample` method will be compiled with :func:`~torch.compile`.
            Keyword arguments can also be passed to torch.compile with this arg.
            Defaults to ``False``.
        span (bool, int, Tuple[bool | int, bool | int], optional): if provided, the sampled
            trajectory will span across the left and/or the right. This means that possibly
            fewer elements will be provided than what was required. A boolean value means
            that at least one element will be sampled per trajectory. An integer `i` means
            that at least `slice_len - i` samples will be gathered for each sampled trajectory.
            Using tuples allows a fine grained control over the span on the left (beginning
            of the stored trajectory) and on the right (end of the stored trajectory).
        max_priority_within_buffer (bool, optional): if ``True``, the max-priority
            is tracked within the buffer. When ``False``, the max-priority tracks
            the maximum value since the instantiation of the sampler.
            Defaults to ``False``.
        max_pending (int, optional): maximum number of :meth:`mark_update` calls
            whose priority writes may be deferred before the sampler flushes them
            to the segment trees. See
            :class:`~torchrl.data.replay_buffers.PrioritizedSampler`.
            Defaults to ``64``.

    Examples:
        >>> import torch
        >>> from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage, PrioritizedSliceSampler
        >>> from tensordict import TensorDict
        >>> sampler = PrioritizedSliceSampler(max_capacity=9, num_slices=3, alpha=0.7, beta=0.9)
        >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(9), sampler=sampler, batch_size=6)
        >>> data = TensorDict(
        ...     {
        ...         "observation": torch.randn(9,16),
        ...         "action": torch.randn(9, 1),
        ...         "episode": torch.tensor([0,0,0,1,1,1,2,2,2], dtype=torch.long),
        ...         "steps": torch.tensor([0,1,2,0,1,2,0,1,2], dtype=torch.long),
        ...         ("next", "observation"): torch.randn(9,16),
        ...         ("next", "reward"): torch.randn(9,1),
        ...         ("next", "done"): torch.tensor([0,0,1,0,0,1,0,0,1], dtype=torch.bool).unsqueeze(1),
        ...     },
        ...     batch_size=[9],
        ... )
        >>> rb.extend(data)
        >>> sample, info = rb.sample(return_info=True)
        >>> print("episode", sample["episode"].tolist())
        episode [2, 2, 2, 2, 1, 1]
        >>> print("steps", sample["steps"].tolist())
        steps [1, 2, 0, 1, 1, 2]
        >>> print("weight", info["priority_weight"].tolist())
        weight [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> priority = torch.tensor([0,3,3,0,0,0,1,1,1])
        >>> rb.update_priority(torch.arange(0,9,1), priority=priority)
        >>> sample, info = rb.sample(return_info=True)
        >>> print("episode", sample["episode"].tolist())
        episode [2, 2, 2, 2, 2, 2]
        >>> print("steps", sample["steps"].tolist())
        steps [1, 2, 0, 1, 0, 1]
        >>> print("weight", info["priority_weight"].tolist())
        weight [9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06]

    .. seealso::

        Trajectory boundaries are recovered at sampling time with
        :func:`~torchrl.data.find_start_stop_traj`; see
        :ref:`the trajectory-boundary documentation <ref_traj_boundaries>`
        for the conventions collectors, storages and samplers follow.

    """

    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        *,
        num_slices: int | None = None,
        slice_len: int | None = None,
        end_key: NestedKey | None = None,
        end_keys: Sequence[NestedKey] | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
        compile: bool | dict = False,
        span: bool | int | tuple[bool | int, bool | int] = False,
        max_priority_within_buffer: bool = False,
        max_pending: int = 64,
    ):
        SliceSampler.__init__(
            self,
            num_slices=num_slices,
            slice_len=slice_len,
            end_key=end_key,
            end_keys=end_keys,
            traj_key=traj_key,
            cache_values=cache_values,
            truncated_key=truncated_key,
            strict_length=strict_length,
            ends=ends,
            trajectories=trajectories,
            compile=compile,
            span=span,
        )
        PrioritizedSampler.__init__(
            self,
            max_capacity=max_capacity,
            alpha=alpha,
            beta=beta,
            eps=eps,
            dtype=dtype,
            reduction=reduction,
            max_priority_within_buffer=max_priority_within_buffer,
            max_pending=max_pending,
        )
        if self.span[0]:
            # Span left is hard to achieve because we need to sample 'negative' starts, but to sample
            # the start we rely on PrioritizedSampler which has no idea it's looking at trajectories.
            #
            # Another way to go about this would be to stochastically decrease the seq_length to
            # accommodate this but that would require to over-sample the starts too.
            #
            warnings.warn(
                f"Left spanning is disabled for {type(self).__name__} and will be automatically turned off. "
                f"If this feature is required, please file an issue on torchrl GitHub repo."
            )
            self.span = (0, self.span[1])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_slices={self.num_slices}, "
            f"slice_len={self.slice_len}, "
            f"end_key={self.end_key}, "
            f"traj_key={self.traj_key}, "
            f"truncated_key={self.truncated_key}, "
            f"strict_length={self.strict_length},"
            f"alpha={self._alpha}, "
            f"beta={self._beta}, "
            f"eps={self._eps}"
        )

    def __getstate__(self):
        state = SliceSampler.__getstate__(self)
        state.update(PrioritizedSampler.__getstate__(self))
        return state

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        return PrioritizedSampler.mark_update(self, index, storage=storage)

    def _padded_indices(self, shapes, arange) -> torch.Tensor:
        # this complex mumbo jumbo creates a left padded tensor with valid indices on the right, e.g.
        # tensor([[ 0,  1,  2,  3,  4],
        #         [-1, -1,  5,  6,  7],
        #         [-1,  8,  9, 10, 11]])
        # where the -1 items on the left are padded values
        num_groups = shapes.shape[0]
        max_group_len = shapes.max()
        pad_lengths = max_group_len - shapes

        # Get all the start and end indices within arange for each group
        group_ends = shapes.cumsum(0)
        group_starts = torch.empty_like(group_ends)
        group_starts[0] = 0
        group_starts[1:] = group_ends[:-1]
        pad = torch.empty(
            (num_groups, max_group_len), dtype=arange.dtype, device=arange.device
        )
        for pad_row, group_start, group_end, pad_len in zip(
            pad, group_starts, group_ends, pad_lengths
        ):
            pad_row[:pad_len] = -1
            pad_row[pad_len:] = arange[group_start:group_end]

        return pad

    def _preceding_stop_idx(self, storage, lengths, seq_length, start_idx):
        preceding_stop_idx = self._cache.get("preceding_stop_idx")
        if preceding_stop_idx is not None:
            return preceding_stop_idx
        arange = torch.arange(storage.shape.numel())
        shapes = lengths.view(-1, 1).cpu()
        if not shapes.sum() - 1 == arange[-1]:
            raise RuntimeError("Wrong shapes / arange configuration")
        if not self.strict_length:
            # First, remove the starts from the arange
            # We do this because each traj can be sampled
            all_but_starts = torch.ones(arange.shape, dtype=torch.bool)
            starts = lengths.cumsum(0)
            starts = torch.cat([torch.zeros_like(starts[:1]), starts[:-1]])
            all_but_starts[starts] = False
            arange = arange[all_but_starts]
            shapes = shapes - 1
        pad = self._padded_indices(shapes, arange)
        _, span_right = self.span[0], self.span[1]
        if span_right and isinstance(span_right, bool):
            preceding_stop_idx = pad[:, -1:]
        else:
            # Mask the rightmost values of that padded tensor
            preceding_stop_idx = pad[:, -seq_length + 1 + span_right :]
        preceding_stop_idx = preceding_stop_idx[preceding_stop_idx >= 0]
        if storage._is_full:
            preceding_stop_idx = (
                preceding_stop_idx
                + np.ravel_multi_index(
                    tuple(start_idx[0].tolist()), storage._total_shape
                )
            ) % storage._total_shape.numel()
        if self.cache_values:
            self._cache["preceding_stop_idx"] = preceding_stop_idx
        return preceding_stop_idx

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        self._flush_pending_updates()
        # Sample `batch_size` indices representing the start of a slice.
        # The sampling is based on a weight vector.
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)

        preceding_stop_idx = self._preceding_stop_idx(
            storage, lengths, seq_length, start_idx
        )
        if storage.ndim > 1:
            # we need to convert indices of the permuted, flatten storage to indices in a flatten storage (not permuted)
            # This is because the lengths come as they would for a permuted storage
            preceding_stop_idx = unravel_index(
                preceding_stop_idx, (storage.shape[-1], *storage.shape[:-1])
            )
            preceding_stop_idx = (preceding_stop_idx[-1], *preceding_stop_idx[:-1])
            preceding_stop_idx = torch.as_tensor(
                np.ravel_multi_index(preceding_stop_idx, storage.shape)
            )

        # force to not sample index at the end of a trajectory
        vals = torch.tensor(self._sum_tree[preceding_stop_idx.cpu().numpy()])
        self._sum_tree[preceding_stop_idx.cpu().numpy()] = 0.0
        # and no need to update self._min_tree

        starts, info = PrioritizedSampler.sample(
            self, storage=storage, batch_size=batch_size // seq_length
        )
        self._sum_tree[preceding_stop_idx.cpu().numpy()] = vals
        # We must truncate the seq_length if (1) not strict length or (2) span[1]
        if self.span[1] or not self.strict_length:
            if not isinstance(starts, torch.Tensor):
                starts_tensor = torch.stack(list(starts), dim=-1).to(stop_idx.device)
            else:
                starts_tensor = starts.unsqueeze(1).to(stop_idx.device)
            # Find the stop that comes after the start index
            # say start_tensor has shape [N, X] and stop_idx has shape [M, X]
            # diff will have shape [M, N, X]
            stop_idx_corr = stop_idx.clone()
            stop_idx_corr[:, 0] = torch.where(
                stop_idx[:, 0] < start_idx[:, 0],
                stop_idx[:, 0] + storage._len_along_dim0,
                stop_idx[:, 0],
            )
            diff = stop_idx_corr.unsqueeze(1) - starts_tensor.unsqueeze(0)
            # filter out all items that don't belong to the same dim in the storage
            mask = (diff[:, :, 1:] != 0).any(-1)
            diff = diff[:, :, 0]
            diff[mask] = diff.max() + 1
            diff = diff.reshape(-1, starts_tensor.shape[0])
            # We remove all neg values from consideration
            diff[diff < 0] = diff.max() + 1
            # Take the arg min along dim 0 (thereby reducing dim M)
            idx = diff.argmin(dim=0)
            stops = stop_idx_corr[idx, 0]
            # TODO: here things may not work bc we could have spanning trajs,
            #  though I cannot show that it breaks in the tests
            if starts_tensor.ndim > 1:
                starts_tensor = starts_tensor[:, 0]
            seq_length = (stops - starts_tensor + 1).clamp_max(seq_length)
            if (seq_length <= 0).any():
                raise RuntimeError(
                    "failed to compute seq_length, please report this bug"
                )

        if isinstance(starts, tuple):
            starts = torch.stack(starts, -1)
        # starts = torch.as_tensor(starts, device=lengths.device)
        info["priority_weight"] = torch.as_tensor(
            info["priority_weight"], device=lengths.device
        )

        # extends starting indices of each slice with sequence_length to get indices of all steps
        index = self._tensor_slices_from_startend(
            seq_length, starts, storage_length=storage.shape[0]
        )

        # repeat the weight of each slice to match the number of steps
        info["priority_weight"] = torch.repeat_interleave(
            info["priority_weight"], seq_length
        )

        if self.truncated_key is not None:
            # following logics borrowed from SliceSampler
            truncated_key = self.truncated_key

            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (index.shape[0], 1), dtype=torch.bool, device=index.device
            )
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            index = index.to(torch.long).unbind(-1)
            st_index = storage[index]
            try:
                done = st_index[done_key] | truncated
            except KeyError:
                done = truncated.clone()
            try:
                terminated = st_index[terminated_key]
            except KeyError:
                terminated = torch.zeros_like(truncated)
            info.update(
                {
                    truncated_key: truncated,
                    done_key: done,
                    terminated_key: terminated,
                }
            )
            return index, info
        return index.to(torch.long).unbind(-1), info

    def _empty(self):
        # no op for SliceSampler
        PrioritizedSampler._empty(self)

    def dumps(self, path):
        # no op for SliceSampler
        PrioritizedSampler.dumps(self, path)

    def loads(self, path):
        # no op for SliceSampler
        return PrioritizedSampler.loads(self, path)

    def state_dict(self):
        # no op for SliceSampler
        return PrioritizedSampler.state_dict(self)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # no op for SliceSampler
        return PrioritizedSampler.load_state_dict(self, state_dict)

    def add(self, index: torch.Tensor) -> None:
        PrioritizedSampler.add(self, index)
        return SliceSampler.add(self, index)

    def extend(self, index: torch.Tensor) -> None:
        PrioritizedSampler.extend(self, index)
        return SliceSampler.extend(self, index)
