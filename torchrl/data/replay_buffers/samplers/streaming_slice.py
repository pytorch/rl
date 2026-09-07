# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tensordict import NestedKey, TensorDictBase

from torchrl.data.replay_buffers.storages import Storage

from .slice import SliceSampler


class StreamingSliceSampler(SliceSampler):
    """A slice sampler that prioritizes newly completed streaming windows.

    Incoming records are partitioned into non-overlapping, fixed-length
    windows without crossing trajectory or done boundaries. Each completed
    window is queued once and returned before the sampler falls back to
    regular uniform :class:`SliceSampler` sampling. Generation-like write
    stamps maintained by the sampler discard queued windows whose ring slots
    were overwritten before they could be sampled.

    The sampler observes writes through the standard replay-buffer
    ``mark_update`` notification. It is intended for one-dimensional storages
    whose writes arrive in chronological order, such as one replay-buffer
    member per environment stream.

    See Also
    :class:`~torchrl.trainers.algorithms.configs.StreamingSliceSamplerConfig`.

    Keyword Args:
        slice_len (int): fixed length of every queued and sampled slice.
        end_key (NestedKey, optional): single end-of-trajectory key. Defaults
            to ``("next", "done")`` when no trajectory key is available.
        end_keys (sequence of NestedKey, optional): boundary keys to combine
            with a logical OR. Exclusive with ``end_key``.
        traj_key (NestedKey, optional): trajectory identifier. When no
            boundary key is configured, the usual collector trajectory key is
            detected automatically before falling back to done markers.
        cache_values (bool, optional): cache the uniform fallback trajectory
            index. Defaults to ``False``.
        truncated_key (NestedKey, optional): key populated in sampling info at
            the final record of each sampled slice. Defaults to
            ``("next", "truncated")``.
        strict_length (bool, optional): whether uniform fallback sampling
            rejects trajectories shorter than ``slice_len``. Defaults to
            ``True``.
        pad_output (bool, optional): whether short uniform fallback slices are
            padded when ``strict_length=False``. Defaults to ``False``.
        compile (bool or dict, optional): compile options forwarded to
            :class:`SliceSampler`. Defaults to ``False``.
        span (bool, int or pair, optional): span options forwarded to
            :class:`SliceSampler`. Defaults to ``False``.
        use_gpu (torch.device or bool, optional): boundary-index device option
            forwarded to :class:`SliceSampler`. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import (
        ...     LazyTensorStorage,
        ...     StreamingSliceSampler,
        ...     TensorDictReplayBuffer,
        ... )
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(16),
        ...     sampler=StreamingSliceSampler(slice_len=3),
        ...     batch_size=6,
        ... )
        >>> done = torch.zeros(6, 1, dtype=torch.bool)
        >>> rb.extend(
        ...     TensorDict(
        ...         {"value": torch.arange(6), ("next", "done"): done},
        ...         batch_size=[6],
        ...     )
        ... )
        >>> rb.sample()["value"].reshape(2, 3)
        tensor([[0, 1, 2],
                [3, 4, 5]])
    """

    def __init__(
        self,
        *,
        slice_len: int,
        end_key: NestedKey | None = None,
        end_keys: Sequence[NestedKey] | None = None,
        traj_key: NestedKey | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
        pad_output: bool = False,
        compile: bool | dict = False,
        span: bool | int | tuple[bool | int, bool | int] = False,
        use_gpu: torch.device | bool = False,
    ):
        if isinstance(slice_len, bool) or not isinstance(slice_len, int):
            raise TypeError("slice_len must be a positive integer.")
        if slice_len < 1:
            raise ValueError("slice_len must be a positive integer.")
        super().__init__(
            slice_len=slice_len,
            end_key=end_key,
            end_keys=end_keys,
            traj_key=traj_key,
            cache_values=cache_values,
            truncated_key=truncated_key,
            strict_length=strict_length,
            pad_output=pad_output,
            compile=compile,
            span=span,
            use_gpu=use_gpu,
        )
        self._queued_slices = collections.deque()
        self._pending_indices = torch.empty(0, dtype=torch.long)
        self._pending_versions = torch.empty(0, dtype=torch.long)
        self._slot_versions = torch.empty(0, dtype=torch.long)
        self._last_traj = None
        self._last_was_done = False

    def _ensure_slot_versions(self, min_size: int, capacity: int) -> None:
        if self._slot_versions.numel() >= min_size:
            return
        size = min(capacity, max(min_size, 2 * self._slot_versions.numel(), 1024))
        versions = torch.zeros(size, dtype=torch.long)
        versions[: self._slot_versions.numel()] = self._slot_versions
        self._slot_versions = versions

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        super().mark_update(index, storage=storage)
        if storage is None:
            return
        if storage.ndim != 1:
            raise RuntimeError(
                "StreamingSliceSampler requires a one-dimensional storage."
            )
        if storage.max_size < self.slice_len:
            raise ValueError(
                f"Storage capacity {storage.max_size} cannot hold one "
                f"slice_len={self.slice_len} window."
            )
        if isinstance(index, tuple):
            raise RuntimeError(
                "StreamingSliceSampler requires one-dimensional write indices."
            )
        storage_index = torch.as_tensor(index, dtype=torch.long).reshape(-1)
        if not storage_index.numel():
            return
        if storage_index.numel() > storage.max_size:
            storage_index = storage_index[-storage.max_size :]
        indices = storage_index.cpu()
        self._ensure_slot_versions(int(indices.max()) + 1, storage.max_size)

        unique = indices.unique()
        occurrence_versions = torch.empty_like(indices)
        if unique.numel() == indices.numel():
            occurrence_versions = self._slot_versions[indices] + 1
            self._slot_versions[indices] = occurrence_versions
        else:
            for position, slot in enumerate(indices.tolist()):
                self._slot_versions[slot] += 1
                occurrence_versions[position] = self._slot_versions[slot]

        if self._pending_indices.numel() and not torch.equal(
            self._slot_versions[self._pending_indices], self._pending_versions
        ):
            self._pending_indices = torch.empty(0, dtype=torch.long)
            self._pending_versions = torch.empty(0, dtype=torch.long)

        data = storage.get(storage_index)
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                "StreamingSliceSampler requires a tensordict-backed storage."
            )
        num_records = indices.numel()
        if getattr(self, "_traj_key_auto", False):
            self._resolve_traj_key(storage)

        trajectory = None
        if self._fetch_traj and self.traj_key is not None:
            trajectory = data.get(self.traj_key, default=None)
            if trajectory is not None:
                trajectory = trajectory.reshape(num_records, -1).cpu()

        done = torch.zeros(num_records, dtype=torch.bool)
        boundary_keys = self.end_keys or (self.end_key,)
        for key in boundary_keys:
            value = data.get(key, default=None)
            if value is not None:
                done |= value.reshape(num_records, -1).any(-1).cpu()
        is_init = data.get("is_init", default=None)
        if is_init is None:
            is_init = torch.zeros_like(done)
        else:
            is_init = is_init.reshape(num_records, -1).any(-1).cpu()

        boundary_before = is_init.clone()
        if num_records > 1:
            boundary_before[1:] |= done[:-1]
            if trajectory is not None:
                boundary_before[1:] |= (trajectory[1:] != trajectory[:-1]).any(-1)
        if self._last_was_done:
            boundary_before[0] = True
        if trajectory is not None and self._last_traj is not None:
            boundary_before[0] |= bool((trajectory[0] != self._last_traj).any())

        if self._pending_indices.numel() and not boundary_before[0]:
            pending_size = self._pending_indices.numel()
            combined_indices = torch.cat((self._pending_indices, indices))
            combined_versions = torch.cat((self._pending_versions, occurrence_versions))
            combined_done = torch.cat(
                (torch.zeros(pending_size, dtype=torch.bool), done)
            )
        else:
            pending_size = 0
            combined_indices = indices
            combined_versions = occurrence_versions
            combined_done = done

        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                boundary_before.nonzero().flatten() + pending_size,
            )
        ).unique(sorted=True)
        stops = torch.cat(
            (starts[1:], torch.tensor([combined_indices.numel()], dtype=torch.long))
        )
        self._pending_indices = torch.empty(0, dtype=torch.long)
        self._pending_versions = torch.empty(0, dtype=torch.long)
        for segment_id, (start, stop) in enumerate(
            zip(starts.tolist(), stops.tolist())
        ):
            segment_length = stop - start
            complete_length = segment_length // self.slice_len * self.slice_len
            if complete_length:
                window_indices = combined_indices[
                    start : start + complete_length
                ].reshape(-1, self.slice_len)
                window_versions = combined_versions[
                    start : start + complete_length
                ].reshape(-1, self.slice_len)
                self._queued_slices.extend(
                    zip(window_indices.unbind(0), window_versions.unbind(0))
                )
            is_last_segment = segment_id == starts.numel() - 1
            if (
                is_last_segment
                and complete_length < segment_length
                and not combined_done[stop - 1]
            ):
                self._pending_indices = combined_indices[
                    start + complete_length : stop
                ].clone()
                self._pending_versions = combined_versions[
                    start + complete_length : stop
                ].clone()

        # Chronological windows are disjoint. Older excess windows must have
        # been overwritten; bound their metadata even when no samples are taken.
        while len(self._queued_slices) > storage.max_size // self.slice_len:
            self._queued_slices.popleft()
        if trajectory is not None:
            self._last_traj = trajectory[-1].clone()
        self._last_was_done = bool(done[-1])

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        if seq_length != self.slice_len:
            raise RuntimeError(
                f"StreamingSliceSampler expected slice length {self.slice_len}, "
                f"got {seq_length}."
            )
        fresh = []
        while self._queued_slices and len(fresh) < num_slices:
            indices, versions = self._queued_slices.popleft()
            if indices.max() >= self._slot_versions.numel():
                continue
            if torch.equal(self._slot_versions[indices], versions):
                fresh.append(indices)
        if not fresh:
            return super().sample(storage, batch_size)

        fresh_index = torch.stack(fresh).reshape(-1, 1)
        storage_device = getattr(storage, "device", None)
        if storage_device is not None and storage_device != "auto":
            fresh_index = fresh_index.to(storage_device)
        fresh_index, fresh_info = self._finalize_index(
            index=fresh_index,
            num_slices=len(fresh),
            seq_length=self.slice_len,
            target_seq_length=None,
            mask_flat=(
                torch.ones(
                    len(fresh) * self.slice_len,
                    dtype=torch.bool,
                    device=fresh_index.device,
                )
                if self.pad_output and not self.strict_length
                else None
            ),
            storage=storage,
        )
        if len(fresh) == num_slices:
            return fresh_index, fresh_info

        fallback_index, fallback_info = super().sample(
            storage, (num_slices - len(fresh)) * self.slice_len
        )
        index = tuple(
            torch.cat((fresh_part.to(fallback_part.device), fallback_part))
            for fresh_part, fallback_part in zip(fresh_index, fallback_index)
        )
        info = {
            key: torch.cat(
                (fresh_info[key].to(fallback_info[key].device), fallback_info[key])
            )
            for key in fresh_info.keys() | fallback_info.keys()
        }
        return index, info

    def _empty(self):
        super()._empty()
        self._cache.clear()
        self._queued_slices.clear()
        self._pending_indices = torch.empty(0, dtype=torch.long)
        self._pending_versions = torch.empty(0, dtype=torch.long)
        self._slot_versions = torch.empty(0, dtype=torch.long)
        self._last_traj = None
        self._last_was_done = False

    def state_dict(self) -> dict[str, Any]:
        if self._queued_slices:
            queued_indices, queued_versions = zip(*self._queued_slices)
            queued_indices = torch.stack(queued_indices)
            queued_versions = torch.stack(queued_versions)
        else:
            queued_indices = torch.empty((0, self.slice_len), dtype=torch.long)
            queued_versions = torch.empty((0, self.slice_len), dtype=torch.long)
        return {
            "slot_versions": self._slot_versions.clone(),
            "queued_indices": queued_indices,
            "queued_versions": queued_versions,
            "pending_indices": self._pending_indices.clone(),
            "pending_versions": self._pending_versions.clone(),
            "last_traj": (
                self._last_traj.clone()
                if self._last_traj is not None
                else torch.empty(0)
            ),
            "has_last_traj": torch.tensor(self._last_traj is not None),
            "last_was_done": torch.tensor(self._last_was_done),
            "slice_len": torch.tensor(self.slice_len),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        slice_len = int(state_dict["slice_len"])
        if slice_len != self.slice_len:
            raise RuntimeError(
                f"Cannot restore slice_len={slice_len} into a "
                f"slice_len={self.slice_len} StreamingSliceSampler."
            )
        self._slot_versions = state_dict["slot_versions"].clone()
        self._queued_slices = collections.deque(
            zip(
                state_dict["queued_indices"].clone().unbind(0),
                state_dict["queued_versions"].clone().unbind(0),
            )
        )
        self._pending_indices = state_dict["pending_indices"].clone()
        self._pending_versions = state_dict["pending_versions"].clone()
        self._last_traj = (
            state_dict["last_traj"].clone()
            if bool(state_dict["has_last_traj"])
            else None
        )
        self._last_was_done = bool(state_dict["last_was_done"])
        self._cache.clear()

    def dumps(self, path: Path):
        path = Path(path).absolute()
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "streaming_slice.pt")

    def loads(self, path: Path):
        path = Path(path).absolute()
        self.load_state_dict(torch.load(path / "streaming_slice.pt"))
