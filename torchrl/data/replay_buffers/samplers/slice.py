# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from multiprocessing.context import get_spawning_popen
from typing import Any

import torch
from tensordict import is_tensor_collection
from tensordict.utils import NestedKey, unravel_key

from torchrl._utils import _replace_last, logger
from torchrl.data.replay_buffers.storages import Storage, TensorStorage
from torchrl.data.replay_buffers.utils import (
    _auto_device,
    _end_to_start_stop,
    _ReplayBoundaryIndex,
)

from ._trajectory import _FragmentedTrajectoryIndex
from .base import Sampler


class SliceSampler(Sampler):
    """Samples slices of data along the first dimension, given start and stop signals.

    This class samples sub-trajectories with replacement. For a version without
    replacement, see :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`.
    Equivalently, ``SliceSampler(replacement=False, ...)`` dispatches to
    :class:`SliceSamplerWithoutReplacement` and forwards the remaining keyword
    arguments (including ``drop_last`` and ``shuffle``).

    .. note:: `SliceSampler` can be slow to retrieve the trajectory indices. To accelerate
        its execution, prefer using `end_key` over `traj_key`, and consider the following
        keyword arguments: :attr:`compile`, :attr:`cache_values` and :attr:`use_gpu`.

    Keyword Args:
        replacement (bool, optional): if ``False``, the call is dispatched to
            :class:`SliceSamplerWithoutReplacement` (which accepts the same
            keyword arguments as well as ``drop_last`` and ``shuffle``).
            Defaults to ``True``.
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
        step_key (NestedKey, optional): key containing a non-negative integer step
            number for each item. Used only when ``fragmented=True`` and defaults
            to ``"step_count"``.
        fragmented (bool, optional): if ``True``, reconstructs logical trajectory
            slices from ``traj_key`` and ``step_key`` even when consecutive steps
            occupy non-adjacent storage positions. Missing logical steps split a
            trajectory into separate sampleable runs. This mode currently supports
            single-dimensional TensorDict-backed storages and sampling with
            replacement. Trajectory ids and step numbers must be scalar integer
            tensors, and every live trajectory-step pair must be unique. Defaults
            to ``False``. The ``span`` and ``compile`` options are not currently
            supported in fragmented mode.
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
            If provided, it is assumed that the storage is at capacity and that
            if the last element of the ``ends`` tensor is ``False``,
            the same trajectory spans across end and beginning.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
            If provided, it is assumed that the storage is at capacity and that
            if the last element of the trajectory tensor is identical to the first,
            the same trajectory spans across end and beginning.
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
        pad_output (bool, optional): **discouraged. Prefer the default
            (``False``).** When ``True`` (and ``strict_length=False``),
            short trajectories are padded by *duplicating their last real
            timestep* up to ``slice_len`` so the output's ``B * T`` is a
            fixed product. The output is still a 1D batch of shape
            ``[B * T]`` — the sample is not reshaped to ``[B, T]``. A 1D
            boolean mask of shape ``[B * T]`` is written to
            ``("collector", "mask")`` flagging real (``True``) vs
            duplicated-last-step (``False``) positions. TorchRL's primitives
            (recurrent modules under
            :func:`~torchrl.modules.set_recurrent_mode`, mask-aware loss
            modules, ``split_trajectories``, etc.) are all designed to
            consume concatenated variable-length slices directly via the
            ``is_init`` / ``truncated`` markers the sampler already emits,
            so padding is a niche escape hatch for downstream code that
            genuinely cannot accept a ragged batch (e.g. a custom op that
            requires a fixed time dimension before a manual reshape).
            Combining ``pad_output=True`` with ``strict_length=True`` raises
            :class:`ValueError`. Defaults to ``False``.
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
        use_gpu (bool or torch.device): if ``True`` (or is a device is passed), an accelerator
            will be used to retrieve the indices of the trajectory starts. This can significantly
            accelerate the sampling when the buffer content is large.
            Defaults to ``False``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    .. note:: When using a multi-process collector
        (:class:`~torchrl.collectors.MultiSyncCollector` or
        :class:`~torchrl.collectors.MultiAsyncCollector`) with a shared replay
        buffer, adjacent transitions in the buffer may come from different
        workers and different episodes. A ``SliceSampler`` that relies on
        ``end_key`` can then sample slices that straddle unrelated trajectories.

        To avoid this, either:

        - set ``fragmented=True`` and provide both ``traj_key`` and ``step_key``
          so that logical adjacency is reconstructed independently of storage
          order,
        - set ``trajs_per_batch`` on the collector so that only **complete**
          trajectories (each ending with ``done=True``) are written to the
          buffer (use ``ndim=1`` on the storage — ``ndim >= 2`` is
          incompatible with the variable-length flat sequences that
          ``trajs_per_batch`` produces), or
        - set ``set_truncated=True`` on the collector so that every batch
          boundary carries a ``done`` signal (note: this introduces artificial
          truncations that value estimators must account for).

    .. note:: When using `strict_length=False`, it is recommended to use
        :func:`~torchrl.collectors.utils.split_trajectories` to split the sampled trajectories.
        However, if two samples from the same episode are placed next to each other,
        this may produce incorrect results. To avoid this issue, consider one of these solutions:

        - using a :class:`~torchrl.data.TensorDictReplayBuffer` instance with the slice sampler

            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.collectors.utils import split_trajectories
            >>> from torchrl.data import TensorDictReplayBuffer, ReplayBuffer, LazyTensorStorage, SliceSampler, SliceSamplerWithoutReplacement
            >>>
            >>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=1000),
            ...                   sampler=SliceSampler(
            ...                       slice_len=5, traj_key="episode",strict_length=False,
            ...                   ))
            ...
            >>> ep_1 = TensorDict(
            ...     {"obs": torch.arange(100),
            ...     "episode": torch.zeros(100),},
            ...     batch_size=[100]
            ... )
            >>> ep_2 = TensorDict(
            ...     {"obs": torch.arange(4),
            ...     "episode": torch.ones(4),},
            ...     batch_size=[4]
            ... )
            >>> rb.extend(ep_1)
            >>> rb.extend(ep_2)
            >>>
            >>> s = rb.sample(50)
            >>> print(s)
            TensorDict(
                fields={
                    episode: Tensor(shape=torch.Size([46]), device=cpu, dtype=torch.float32, is_shared=False),
                    index: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                    next: TensorDict(
                        fields={
                            done: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            terminated: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                            truncated: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        batch_size=torch.Size([46]),
                        device=cpu,
                        is_shared=False),
                    obs: Tensor(shape=torch.Size([46]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([46]),
                device=cpu,
                is_shared=False)
            >>> t = split_trajectories(s, done_key="truncated")
            >>> print(t["obs"])
            tensor([[73, 74, 75, 76, 77],
                    [ 0,  1,  2,  3,  0],
                    [ 0,  1,  2,  3,  0],
                    [41, 42, 43, 44, 45],
                    [ 0,  1,  2,  3,  0],
                    [67, 68, 69, 70, 71],
                    [27, 28, 29, 30, 31],
                    [80, 81, 82, 83, 84],
                    [17, 18, 19, 20, 21],
                    [ 0,  1,  2,  3,  0]])
            >>> print(t["episode"])
            tensor([[0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 0.],
                    [1., 1., 1., 1., 0.],
                    [0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 0.]])

        - using a :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`

            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.collectors.utils import split_trajectories
            >>> from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler, SliceSamplerWithoutReplacement
            >>>
            >>> rb = ReplayBuffer(storage=LazyTensorStorage(max_size=1000),
            ...                   sampler=SliceSamplerWithoutReplacement(
            ...                       slice_len=5, traj_key="episode",strict_length=False
            ...                   ))
            ...
            >>> ep_1 = TensorDict(
            ...     {"obs": torch.arange(100),
            ...     "episode": torch.zeros(100),},
            ...     batch_size=[100]
            ... )
            >>> ep_2 = TensorDict(
            ...     {"obs": torch.arange(4),
            ...     "episode": torch.ones(4),},
            ...     batch_size=[4]
            ... )
            >>> rb.extend(ep_1)
            >>> rb.extend(ep_2)
            >>>
            >>> s = rb.sample(50)
            >>> t = split_trajectories(s, trajectory_key="episode")
            >>> print(t["obs"])
            tensor([[75, 76, 77, 78, 79],
                    [ 0,  1,  2,  3,  0]])
            >>> print(t["episode"])
            tensor([[0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 0.]])

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSampler
        >>> torch.manual_seed(0)
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1_000_000),
        ...     sampler=SliceSampler(cache_values=True, num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> print("sample:", sample)
        >>> print("episodes", sample.get("episode").unique())
        episodes tensor([1, 2, 3, 4], dtype=torch.int32)

    :class:`~torchrl.data.replay_buffers.SliceSampler` is default-compatible with
    most of TorchRL's datasets:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSampler
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320, sampler=SliceSampler(num_slices=num_slices))
        >>> for batch in data:
        ...     batch = batch.reshape(num_slices, -1)
        ...     break
        >>> print("check that each batch only has one episode:", batch["episode"].unique(dim=1))
        check that each batch only has one episode: tensor([[19],
                [14],
                [ 8],
                [10],
                [13],
                [ 4],
                [ 2],
                [ 3],
                [22],
                [ 8]])

    .. seealso::

        Trajectory boundaries are recovered at sampling time with
        :func:`~torchrl.data.find_start_stop_traj`, which documents how
        trajectory ids, end flags, the write cursor and the storage capacity
        interact. See also :ref:`the trajectory-boundary documentation
        <ref_traj_boundaries>` for the conventions collectors, storages and
        samplers follow.

    """

    # We use this whenever we need to sample N times too many transitions to then select only a 1/N fraction of them
    _batch_size_multiplier: int | None = 1

    # Class-level defaults keep samplers pickled by earlier torchrl versions
    # (whose __dict__ lacks these attributes) working after an upgrade.
    fragmented: bool = False
    step_key: NestedKey | None = "step_count"
    _fragmented_index: _FragmentedTrajectoryIndex | None = None

    def __init__(
        self,
        *,
        num_slices: int | None = None,
        slice_len: int | None = None,
        end_key: NestedKey | None = None,
        end_keys: Sequence[NestedKey] | None = None,
        traj_key: NestedKey | None = None,
        step_key: NestedKey | None = "step_count",
        fragmented: bool = False,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
        pad_output: bool = False,
        compile: bool | dict = False,
        span: bool | int | tuple[bool | int, bool | int] = False,
        use_gpu: torch.device | bool = False,
    ):
        if isinstance(span, (bool, int)):
            span = (span, span)
        if fragmented:
            if type(self).sample is not SliceSampler.sample:
                raise NotImplementedError(
                    "fragmented=True currently supports SliceSampler with "
                    "replacement only."
                )
            if step_key is None:
                raise ValueError("step_key must be provided when fragmented=True.")
            if ends is not None:
                raise ValueError(
                    "fragmented=True requires trajectory and step identifiers; "
                    "the static ends argument cannot reconstruct interleaved "
                    "trajectories."
                )
            if trajectories is not None:
                raise ValueError(
                    "The static trajectories argument is not supported with "
                    "fragmented=True; provide traj_key and step_key."
                )
            if any(span):
                raise NotImplementedError("span is not supported with fragmented=True.")
            if compile:
                raise NotImplementedError(
                    "compile is not supported with fragmented=True."
                )
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.step_key = step_key
        self.fragmented = fragmented
        self._fragmented_index: _FragmentedTrajectoryIndex | None = None
        if end_keys is not None and end_key is not None:
            raise RuntimeError(
                "`end_key` and `end_keys` are exclusive arguments: pass the "
                "single boundary key through `end_key`, or the sequence of "
                "keys to be OR-ed together through `end_keys`."
            )
        self.end_key = end_key
        self.end_keys = (
            tuple(unravel_key(key) for key in end_keys)
            if end_keys is not None
            else None
        )
        self.traj_key = traj_key
        self.truncated_key = truncated_key
        self.cache_values = cache_values
        self._fetch_traj = True
        self.strict_length = strict_length
        if pad_output and strict_length:
            raise ValueError(
                "pad_output=True is incompatible with strict_length=True: "
                "padding only happens when short trajectories are kept, which "
                "requires strict_length=False."
            )
        self.pad_output = pad_output
        self._cache = {}
        self.use_gpu = bool(use_gpu)
        self._gpu_device = (
            None
            if not self.use_gpu
            else (
                torch.device(use_gpu)
                if not isinstance(use_gpu, bool)
                else _auto_device()
            )
        )

        self.span = span

        if trajectories is not None:
            if traj_key is not None or end_key or end_keys:
                raise RuntimeError(
                    "`trajectories` and `end_key`, `end_keys` or `traj_key` are exclusive arguments."
                )
            if ends is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, trajectories requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(
                trajectory=trajectories,
                at_capacity=True,
            )
            self._cache["static-stop-and-length"] = vals

        elif ends is not None:
            if traj_key is not None or end_key or end_keys:
                raise RuntimeError(
                    "`ends` and `end_key`, `end_keys` or `traj_key` are exclusive arguments."
                )
            if trajectories is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, ends requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(end=ends, at_capacity=True)
            self._cache["static-stop-and-length"] = vals

        else:
            if traj_key is not None:
                self._fetch_traj = True
                self._traj_key_auto = False
            elif end_key is not None or end_keys is not None:
                self._fetch_traj = False
                self._traj_key_auto = False
            else:
                # Neither provided: auto-detect from storage on first sample call.
                # Prefer ("collector", "traj_ids") (written by collectors) over "episode".
                self._fetch_traj = True
                self._traj_key_auto = True
            if end_key is None and end_keys is None:
                end_key = ("next", "done")
            self.end_key = end_key
            self.traj_key = traj_key  # may be None when _traj_key_auto=True

        if not ((num_slices is None) ^ (slice_len is None)):
            raise TypeError(
                "Either num_slices or slice_len must be not None, and not both. "
                f"Got num_slices={num_slices} and slice_len={slice_len}."
            )
        self.compile = bool(compile)
        if self.compile:
            if isinstance(compile, dict):
                kwargs = compile
            else:
                kwargs = {}
            self._get_index = torch.compile(self._get_index, **kwargs)

    def __getstate__(self):
        if get_spawning_popen() is not None and self.cache_values:
            logger.warning(
                f"It seems you are sharing a {type(self).__name__} across processes with "
                f"cache_values=True. "
                f"While this isn't forbidden and could perfectly work if your dataset "
                f"is unaltered on both processes, remember that calling extend/add on "
                f"one process will NOT erase the cache on another process's sampler, "
                f"which will cause synchronization issues."
            )
        state = super().__getstate__()
        state["_cache"] = {}
        return state

    def extend(self, index: torch.Tensor) -> None:
        if self.fragmented:
            return
        super().extend(index)
        if self.cache_values:
            self._cache.clear()

    def add(self, index: torch.Tensor) -> None:
        if self.fragmented:
            return
        super().add(index)
        if self.cache_values:
            self._cache.clear()

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        if self.fragmented and self._fragmented_index is not None:
            self._fragmented_index.mark_update(index, storage=storage)
        # Delegate cooperatively so classes mixing SliceSampler with e.g.
        # PrioritizedSampler keep their write hook.
        super().mark_update(index, storage=storage)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_slices={self.num_slices}, "
            f"slice_len={self.slice_len}, "
            f"end_key={self.end_key}, "
            f"end_keys={getattr(self, 'end_keys', None)}, "
            f"traj_key={self.traj_key}, "
            f"step_key={self.step_key}, "
            f"fragmented={self.fragmented}, "
            f"truncated_key={self.truncated_key}, "
            f"strict_length={self.strict_length}, "
            f"pad_output={getattr(self, 'pad_output', False)})"
        )

    def _find_start_stop_traj(
        self,
        *,
        trajectory=None,
        end=None,
        at_capacity: bool,
        cursor=None,
        storage=None,
        source=None,
    ):
        # Thin wrapper around the shared module-level utilities (see
        # torchrl.data.find_start_stop_traj). Kept as a method, dispatching
        # through self._end_to_start_stop, so that subclasses overriding
        # either hook keep working and the sampler's GPU device is applied.
        boundary = _ReplayBoundaryIndex(
            trajectory=trajectory,
            end=end,
            at_capacity=at_capacity,
            cursor=cursor,
            device=self._gpu_device,
            end_to_start_stop=lambda end, length: self._end_to_start_stop(
                end=end, length=length
            ),
            storage=storage,
            source=source,
            cache_values=(
                self.cache_values
                and storage is not None
                and type(self)._end_to_start_stop is SliceSampler._end_to_start_stop
            ),
        )
        return boundary.boundaries()

    def _end_to_start_stop(self, end, length):
        return _end_to_start_stop(end=end, length=length, device=self._gpu_device)

    def _start_to_end(self, st: torch.Tensor, length: int):

        arange = torch.arange(length, device=st.device, dtype=st.dtype)
        ndims = st.shape[-1] - 1 if st.ndim else 0
        if ndims:
            arange = torch.stack([arange] + [torch.zeros_like(arange)] * ndims, -1)
        else:
            arange = arange.unsqueeze(-1)
        if st.shape != arange.shape:
            # we do this to make sure that we're not broadcasting the start
            # wrong as a tensor with shape [N] can't be expanded to [N, 1]
            # without getting an error
            st = st.expand_as(arange)
        return arange + st

    def _tensor_slices_from_startend(self, seq_length, start, storage_length):
        # start is a 2d tensor resulting from nonzero()
        # seq_length is a 1d tensor indicating the desired length of each sequence

        if isinstance(seq_length, int):
            arange = torch.arange(seq_length, device=start.device, dtype=start.dtype)
            ndims = start.shape[-1] - 1 if (start.ndim - 1) else 0
            if ndims:
                arange_reshaped = torch.empty(
                    arange.shape + torch.Size([ndims + 1]),
                    device=start.device,
                    dtype=start.dtype,
                )
                arange_reshaped[..., 0] = arange
                arange_reshaped[..., 1:] = 0
            else:
                arange_reshaped = arange.unsqueeze(-1)
            arange_expanded = arange_reshaped.expand(
                torch.Size([start.shape[0]]) + arange_reshaped.shape
            )
            if start.shape != arange_expanded.shape:
                n_missing_dims = arange_expanded.dim() - start.dim()
                start_expanded = start[
                    (slice(None),) + (None,) * n_missing_dims
                ].expand_as(arange_expanded)
            result = (start_expanded + arange_expanded).flatten(0, 1)

        else:
            # when padding is needed
            result = torch.cat(
                [
                    self._start_to_end(_start, _seq_len)
                    for _start, _seq_len in zip(start, seq_length)
                ]
            )
        result[:, 0] = result[:, 0] % storage_length
        return result

    def _resolve_traj_key(self, storage):
        """Auto-detect traj_key from storage on first sample call.

        Probes for ``("collector", "traj_ids")`` first (written by TorchRL
        collectors), then falls back to ``"episode"``, then to ``end_key``
        reconstruction. Schema is read from ``storage._storage`` keys when
        possible to avoid materialising any data.

        Note: this method runs *once* per sampler lifetime (the
        ``_traj_key_auto`` flag is cleared on the first call). If the storage
        schema changes after the first sample call — e.g. data with a different
        traj key is added later — the resolved key won't be updated. In
        practice this only matters if the user is mixing storages, which is
        unusual.
        """
        self._traj_key_auto = False
        keys = None
        # Cheap path: read the schema from the underlying TensorDict without
        # fetching any data.
        underlying = getattr(storage, "_storage", None)
        if is_tensor_collection(underlying):
            try:
                keys = set(underlying.keys(include_nested=True))
            except (RuntimeError, AttributeError, TypeError):
                keys = None
        if keys is None:
            # Fallback: read one row. May be costly for remote storages.
            try:
                sample = storage[0:1]
            except (IndexError, KeyError, RuntimeError):
                sample = None
            if sample is not None and hasattr(sample, "keys"):
                try:
                    keys = set(sample.keys(include_nested=True))
                except (RuntimeError, AttributeError, TypeError):
                    keys = None

        if keys is None:
            # Could not introspect schema: fall back to end_key reconstruction.
            self._fetch_traj = False
            return

        has_collector = ("collector", "traj_ids") in keys
        has_episode = "episode" in keys
        if has_collector:
            self.traj_key = ("collector", "traj_ids")
            self._fetch_traj = True
            return
        if has_episode:
            self.traj_key = "episode"
            self._fetch_traj = True
            return
        # Neither traj key found: reconstruct from end_key
        self._fetch_traj = False

    def _get_stop_and_length(self, storage, fallback=True):
        if self.cache_values and "static-stop-and-length" in self._cache:
            return self._cache.get("static-stop-and-length")

        if getattr(self, "_traj_key_auto", False):
            self._resolve_traj_key(storage)

        if self._fetch_traj:
            # We first try with the traj_key
            try:
                if isinstance(storage, TensorStorage):
                    trajectory = storage[:][self._used_traj_key]
                else:
                    try:
                        trajectory = storage[:][self.traj_key]
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                vals = self._find_start_stop_traj(
                    trajectory=trajectory,
                    at_capacity=storage._is_full,
                    cursor=getattr(storage, "_last_cursor_index", None),
                    storage=storage,
                    source=("trajectory", self._used_traj_key),
                )
                return vals
            except KeyError:
                if fallback:
                    logger.info(
                        "SliceSampler could not find traj_key %r in storage. "
                        "Falling back to end_key %r to reconstruct trajectory boundaries.",
                        self.traj_key,
                        self.end_key,
                    )
                    self._fetch_traj = False
                    return self._get_stop_and_length(storage, fallback=False)
                raise

        else:
            try:
                if self.end_keys is not None:
                    done = self._end_signal_from_keys(storage)
                else:
                    try:
                        done = storage[:][self.end_key]
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                    done = done.squeeze()
                vals = self._find_start_stop_traj(
                    end=done[: len(storage)],
                    at_capacity=storage._is_full,
                    cursor=getattr(storage, "_last_cursor_index", None),
                    storage=storage,
                    source=(
                        "end",
                        tuple(self.end_keys)
                        if self.end_keys is not None
                        else self.end_key,
                    ),
                )
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = True
                    if self.traj_key is None:
                        # No trajectory key was configured either: probe the
                        # storage for the usual candidates on the retry.
                        self._traj_key_auto = True
                    return self._get_stop_and_length(storage, fallback=False)
                raise

    def _end_signal_from_keys(self, storage) -> torch.Tensor:
        """Union (logical OR) of all ``end_keys`` entries present in the storage.

        Missing keys are skipped; if none of the keys is present a
        ``KeyError`` is raised so that :meth:`_get_stop_and_length` can fall
        back to trajectory-id-based recovery.
        """
        try:
            data = storage[:]
        except Exception:
            raise RuntimeError(
                "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
            )
        done = None
        for key in self.end_keys:
            val = data.get(key, default=None)
            if val is None:
                continue
            val = val.squeeze()
            done = val if done is None else done | val
        if done is None:
            raise KeyError(
                f"None of the end_keys {self.end_keys} could be found in the storage."
            )
        return done

    def _adjusted_batch_size(self, batch_size):
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the number of slices, got "
                    f"batch_size={batch_size} and num_slices={self.num_slices}."
                )
            seq_length = batch_size // self.num_slices
            num_slices = self.num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the slice length, got "
                    f"batch_size={batch_size} and slice_len={self.slice_len}."
                )
            seq_length = self.slice_len
            num_slices = batch_size // self.slice_len
        return seq_length, num_slices

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if self._batch_size_multiplier is not None:
            batch_size = batch_size * self._batch_size_multiplier
        if self.fragmented:
            return self._sample_fragmented(storage, batch_size)
        # pick up as many trajs as we need
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        # we have to make sure that the number of dims of the storage
        # is the same as the stop/start signals since we will
        # use these to sample the storage
        if start_idx.shape[1] != storage.ndim:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be "
                f"{storage.ndim}-dimensional. Got a tensor with shape[1]={start_idx.shape[1]} "
                "instead."
            )
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        storage_length = storage.shape[0]
        return self._sample_slices(
            lengths,
            start_idx,
            stop_idx,
            seq_length,
            num_slices,
            storage_length=storage_length,
            storage=storage,
        )

    def _sample_fragmented(
        self, storage: Storage, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        if len(storage) == 0:
            raise RuntimeError("Cannot sample from an empty storage.")
        if getattr(self, "_traj_key_auto", False):
            self._resolve_traj_key(storage)
        if not self._fetch_traj or self.traj_key is None:
            raise RuntimeError(
                "fragmented=True requires a trajectory identifier under traj_key; "
                "end-of-trajectory flags alone cannot reconstruct interleaved data."
            )
        indexer = self._fragmented_index
        if (
            indexer is None
            or indexer.trajectory_key != self.traj_key
            or indexer.step_key != self.step_key
        ):
            indexer = self._fragmented_index = _FragmentedTrajectoryIndex(
                self.traj_key, self.step_key
            )
        indexer.refresh(storage)
        ordered_slots, run_offsets, lengths = indexer.runs()
        if not lengths.numel():
            raise RuntimeError("No logical trajectory runs are available for sampling.")

        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        if self.strict_length:
            eligible = lengths >= seq_length
            if not eligible.any():
                raise RuntimeError(
                    "Did not find a single fragmented trajectory run with "
                    f"sufficient length (length range: {lengths.min()} - "
                    f"{lengths.max()} / required={seq_length})."
                )
            run_offsets = run_offsets[eligible]
            lengths = lengths[eligible]
            run_idx = torch.randint(
                lengths.shape[0],
                (num_slices,),
                device=lengths.device,
                generator=self._rng,
            )
            sampled_lengths: int | torch.Tensor = seq_length
            target_seq_length = None
        else:
            run_idx = torch.randint(
                lengths.shape[0],
                (num_slices,),
                device=lengths.device,
                generator=self._rng,
            )
            sampled_lengths = lengths[run_idx].clamp_max(seq_length)
            target_seq_length = seq_length if self.pad_output else None

        selected_run_lengths = lengths[run_idx]
        available_starts = selected_run_lengths - sampled_lengths + 1
        relative_starts = (
            (
                torch.rand(num_slices, device=lengths.device, generator=self._rng)
                * available_starts
            )
            .floor()
            .to(run_offsets.dtype)
        )
        packed_starts = run_offsets[run_idx] + relative_starts

        if target_seq_length is not None:
            offsets = torch.arange(target_seq_length, device=lengths.device)
            real_mask = offsets.unsqueeze(0) < sampled_lengths.unsqueeze(1)
            offsets = torch.minimum(
                offsets.unsqueeze(0), (sampled_lengths - 1).unsqueeze(1)
            )
            packed_indices = packed_starts.unsqueeze(1) + offsets
            index = ordered_slots[packed_indices].reshape(-1, 1)
            mask_flat = real_mask.reshape(-1)
        elif isinstance(sampled_lengths, int):
            offsets = torch.arange(sampled_lengths, device=lengths.device)
            packed_indices = packed_starts.unsqueeze(1) + offsets
            index = ordered_slots[packed_indices].reshape(-1, 1)
            mask_flat = None
        else:
            packed_indices = torch.cat(
                [
                    torch.arange(length, device=lengths.device) + start
                    for start, length in zip(packed_starts, sampled_lengths)
                ]
            )
            index = ordered_slots[packed_indices].reshape(-1, 1)
            mask_flat = None

        return self._finalize_index(
            index=index,
            num_slices=num_slices,
            seq_length=sampled_lengths,
            target_seq_length=target_seq_length,
            mask_flat=mask_flat,
            storage=storage,
        )

    def _sample_slices(
        self,
        lengths: torch.Tensor,
        start_idx: torch.Tensor,
        stop_idx: torch.Tensor,
        seq_length: int,
        num_slices: int,
        storage_length: int,
        traj_idx: torch.Tensor | None = None,
        *,
        storage,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        # start_idx and stop_idx are 2d tensors organized like a non-zero

        def get_traj_idx(maxval):
            return torch.randint(
                maxval, (num_slices,), device=lengths.device, generator=self._rng
            )

        _target_seq_length = None
        if (lengths < seq_length).any():
            if self.strict_length:
                idx = lengths >= seq_length
                if not idx.any():
                    raise RuntimeError(
                        f"Did not find a single trajectory with sufficient length (length range: {lengths.min()} - {lengths.max()} / required={seq_length}))."
                    )
                if (
                    isinstance(seq_length, torch.Tensor)
                    and seq_length.shape == lengths.shape
                ):
                    seq_length = seq_length[idx]
                lengths_idx = lengths[idx]
                start_idx = start_idx[idx]
                stop_idx = stop_idx[idx]

                if traj_idx is None:
                    traj_idx = get_traj_idx(lengths_idx.shape[0])
                else:
                    # Here we must filter out the indices that correspond to trajectories
                    # we don't want to keep. That could potentially lead to an empty sample.
                    # The difficulty with this adjustment is that traj_idx points to a full
                    # sequences of lengths, but we filter out part of it so we must
                    # convert traj_idx to a boolean mask, index this mask with the
                    # valid indices and then recover the nonzero.
                    idx_mask = torch.zeros_like(idx)
                    idx_mask[traj_idx] = True
                    traj_idx = idx_mask[idx].nonzero().squeeze(-1)
                    if not traj_idx.numel():
                        raise RuntimeError(
                            "None of the provided indices pointed to a trajectory of "
                            "sufficient length. Consider using strict_length=False for the "
                            "sampler instead."
                        )
                    num_slices = traj_idx.shape[0]

                del idx
                lengths = lengths_idx
            else:
                if traj_idx is None:
                    traj_idx = get_traj_idx(lengths.shape[0])
                else:
                    num_slices = traj_idx.shape[0]

                # make seq_length a tensor with values clamped by lengths
                _target_seq_length = seq_length if self.pad_output else None
                seq_length = lengths[traj_idx].clamp_max(seq_length)
        else:
            if traj_idx is None:
                traj_idx = get_traj_idx(lengths.shape[0])
            else:
                num_slices = traj_idx.shape[0]
            _target_seq_length = None
        return self._get_index(
            lengths=lengths,
            start_idx=start_idx,
            stop_idx=stop_idx,
            num_slices=num_slices,
            seq_length=seq_length,
            storage_length=storage_length,
            traj_idx=traj_idx,
            target_seq_length=_target_seq_length,
            storage=storage,
        )

    def _get_index(
        self,
        lengths: torch.Tensor,
        start_idx: torch.Tensor,
        stop_idx: torch.Tensor,
        seq_length: int,
        num_slices: int,
        storage_length: int,
        traj_idx: torch.Tensor | None = None,
        *,
        target_seq_length: int | None = None,
        storage,
    ) -> tuple[torch.Tensor, dict]:
        # end_point is the last possible index for start
        last_indexable_start = lengths[traj_idx] - seq_length + 1
        if not self.span[1]:
            end_point = last_indexable_start
        elif self.span[1] is True:
            end_point = lengths[traj_idx] + 1
        else:
            span_left = self.span[1]
            if span_left >= seq_length:
                raise ValueError(
                    "The right and left span must be strictly lower than the sequence length"
                )
            end_point = lengths[traj_idx] - span_left

        if not self.span[0]:
            start_point = 0
        elif self.span[0] is True:
            start_point = -seq_length + 1
        else:
            span_right = self.span[0]
            if span_right >= seq_length:
                raise ValueError(
                    "The right and left span must be strictly lower than the sequence length"
                )
            start_point = -span_right

        relative_starts = (
            torch.rand(num_slices, device=lengths.device, generator=self._rng)
            * (end_point - start_point)
        ).floor().to(start_idx.dtype) + start_point

        if self.span[0]:
            out_of_traj = relative_starts < 0
            if out_of_traj.any():
                # a negative start means sampling fewer elements
                # Convert seq_length to tensor to avoid torch.compile inductor C++ codegen
                # bug with mixed scalar/tensor int64 in blendv operations (see PyTorch #xyz)
                seq_length_t = torch.as_tensor(
                    seq_length,
                    dtype=relative_starts.dtype,
                    device=relative_starts.device,
                )
                seq_length = torch.where(
                    ~out_of_traj, seq_length_t, seq_length_t + relative_starts
                )
                relative_starts = torch.where(
                    ~out_of_traj, relative_starts, torch.zeros_like(relative_starts)
                )
        if self.span[1]:
            out_of_traj = relative_starts + seq_length > lengths[traj_idx]
            if out_of_traj.any():
                # a negative start means sampling fewer elements
                # Convert seq_length to tensor if it's still a scalar
                if not isinstance(seq_length, torch.Tensor):
                    seq_length = torch.as_tensor(
                        seq_length,
                        dtype=relative_starts.dtype,
                        device=relative_starts.device,
                    )
                seq_length = torch.minimum(
                    seq_length, lengths[traj_idx] - relative_starts
                )

        starts = torch.cat(
            [
                (start_idx[traj_idx, 0] + relative_starts).unsqueeze(1),
                start_idx[traj_idx, 1:],
            ],
            1,
        )

        # When strict_length=False produced variable per-slice lengths, pad all
        # slices to target_seq_length and emit a boolean mask so callers can
        # distinguish real timesteps from padded ones.
        if target_seq_length is not None and isinstance(seq_length, torch.Tensor):
            T = target_seq_length
            n_extra_dims = starts.shape[1] - 1

            # time offsets: [T, n_extra_dims+1] — only the first dim advances
            time_offsets = torch.zeros(
                T, n_extra_dims + 1, device=starts.device, dtype=starts.dtype
            )
            time_offsets[:, 0] = torch.arange(
                T, device=starts.device, dtype=starts.dtype
            )

            # full index before masking: [B, T, n_extra_dims+1]
            index_full = starts.unsqueeze(1) + time_offsets.unsqueeze(0)

            # real_mask[i, t] == True iff timestep t is a real (non-padded) step
            arange = torch.arange(T, device=seq_length.device)
            real_mask = arange.unsqueeze(0) < seq_length.unsqueeze(1)  # [B, T]

            # padded positions repeat the last real index so storage access is safe
            last_valid = starts.clone()
            last_valid[:, 0] = starts[:, 0] + (seq_length - 1).clamp(min=0)
            index_full = torch.where(
                real_mask.unsqueeze(-1),
                index_full,
                last_valid.unsqueeze(1).expand_as(index_full),
            )
            index_full[:, :, 0] = index_full[:, :, 0] % storage_length
            index = index_full.reshape(-1, n_extra_dims + 1)
            mask_flat = real_mask.reshape(-1)  # [B*T]
        else:
            index = self._tensor_slices_from_startend(
                seq_length, starts, storage_length
            )
            mask_flat = None

        return self._finalize_index(
            index=index,
            num_slices=num_slices,
            seq_length=seq_length,
            target_seq_length=target_seq_length,
            mask_flat=mask_flat,
            storage=storage,
        )

    def _finalize_index(
        self,
        *,
        index: torch.Tensor,
        num_slices: int,
        seq_length: int | torch.Tensor,
        target_seq_length: int | None,
        mask_flat: torch.Tensor | None,
        storage: Storage,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        if self.truncated_key is not None:
            truncated_key = self.truncated_key
            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (index.shape[0], 1), dtype=torch.bool, device=index.device
            )
            if target_seq_length is not None and isinstance(seq_length, torch.Tensor):
                # mark the last *real* timestep of each slice as truncated
                T = target_seq_length
                trunc_positions = torch.arange(
                    num_slices, device=seq_length.device
                ) * T + (seq_length - 1).clamp(min=0)
                truncated[trunc_positions] = 1
            elif isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            index = index.to(torch.long).unbind(-1)
            st_index = storage[index]
            done = st_index.get(done_key, default=None)
            if done is None:
                done = truncated.clone()
            else:
                done = done | truncated
            terminated = st_index.get(terminated_key, default=None)
            if terminated is None:
                terminated = torch.zeros_like(truncated)
            info = {
                truncated_key: truncated,
                done_key: done,
                terminated_key: terminated,
            }
            if mask_flat is not None:
                info[("collector", "mask")] = mask_flat
            self._maybe_emit_init_marker(
                info, st_index, num_slices, seq_length, target_seq_length
            )
            return index, info
        index = index.to(torch.long).unbind(-1)
        info = {}
        if mask_flat is not None:
            info[("collector", "mask")] = mask_flat
        # Fetch st_index lazily so the marker logic can OR with existing
        # is_init. Cheap because index is already computed.
        st_index = storage[index]
        self._maybe_emit_init_marker(
            info, st_index, num_slices, seq_length, target_seq_length
        )
        return index, info

    def _maybe_emit_init_marker(
        self,
        info: dict,
        st_index,
        num_slices: int,
        seq_length,
        target_seq_length: int | None,
    ) -> None:
        """Mark every slice start with ``is_init=True``.

        Recurrent modules (:class:`~torchrl.modules.LSTMModule`,
        :class:`~torchrl.modules.GRUModule`) under
        :func:`~torchrl.modules.set_recurrent_mode` ``("recurrent")`` split a
        flat sequence on ``is_init`` and use the stored hidden state at each
        split as the initial state. By marking the first timestep of every
        slice we let the user pass the concatenated flat sample straight to a
        recurrent policy: the RNN sees the slices as independent
        sub-trajectories and uses each slice's stored ``recurrent_state[0]``
        as its initial hidden state.

        We OR our markers with the storage's existing ``is_init`` so episode
        resets that fall *inside* a slice are preserved. If the storage
        doesn't carry an ``is_init`` field (no :class:`InitTracker`), we don't
        introduce one — we'd be lying about real resets we can't see.
        """
        existing_is_init = (
            st_index.get("is_init", default=None) if hasattr(st_index, "get") else None
        )
        if existing_is_init is None:
            return
        init_marker = torch.zeros_like(existing_is_init)
        device = init_marker.device
        if target_seq_length is not None:
            # pad_output path: every slice is target_seq_length long.
            slice_starts = (
                torch.arange(num_slices, device=device, dtype=torch.long)
                * target_seq_length
            )
        elif isinstance(seq_length, int):
            # Uniform slices (strict_length=True, or strict_length=False with
            # all sufficiently-long trajectories).
            slice_starts = (
                torch.arange(num_slices, device=device, dtype=torch.long) * seq_length
            )
        else:
            # Variable per-slice length: starts are at cumulative offsets,
            # i.e. [0, len_0, len_0+len_1, ...].
            slice_starts = torch.zeros(num_slices, device=device, dtype=torch.long)
            slice_starts[1:] = seq_length.to(device).cumsum(0)[:-1].to(torch.long)
        init_marker[slice_starts] = True
        info["is_init"] = init_marker | existing_is_init

    @property
    def _used_traj_key(self):
        return self.__dict__.get("__used_traj_key", self.traj_key)

    @_used_traj_key.setter
    def _used_traj_key(self, value):
        self.__dict__["__used_traj_key"] = value

    @property
    def _used_end_key(self):
        return self.__dict__.get("__used_end_key", self.end_key)

    @_used_end_key.setter
    def _used_end_key(self, value):
        self.__dict__["__used_end_key"] = value

    def _empty(self):
        if self._fragmented_index is not None:
            self._fragmented_index.clear()

    def dumps(self, path):
        # no op - cache does not need to be saved
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._fragmented_index is not None:
            self._fragmented_index.clear()
