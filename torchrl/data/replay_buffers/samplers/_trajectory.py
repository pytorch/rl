# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref
from collections import defaultdict

import torch
from tensordict import is_tensor_collection
from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.storages import Storage


class _FragmentedTrajectoryIndex:
    """Indexes logical trajectory adjacency independently of storage order."""

    def __init__(self, trajectory_key: NestedKey, step_key: NestedKey):
        self.trajectory_key = trajectory_key
        self.step_key = step_key
        self._trajectory_positions: dict[int, dict[int, int]] | None = None
        # Per-slot record columns (CPU, long). A step of -1 marks a slot
        # without a live record.
        self._slot_traj: torch.Tensor | None = None
        self._slot_step: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._runs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._pending_indices: list[torch.Tensor] = []
        self._pending_revisions: list[int] = []
        # Weak references rather than id(): a recycled address of a dead
        # storage must not be mistaken for the cached one.
        self._pending_storage_ref: weakref.ref | None = None
        self._cache_storage_ref: weakref.ref | None = None
        self._cache_revision: int | None = None

    @staticmethod
    def _storage_device(storage: Storage) -> torch.device | None:
        device = getattr(storage, "device", None)
        if device is None or device == "auto":
            return None
        return torch.device(device)

    def _validate_metadata(
        self,
        trajectory: torch.Tensor | None,
        step: torch.Tensor | None,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
        expected = index.numel()
        for key, value in (
            (self.trajectory_key, trajectory),
            (self.step_key, step),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"key={key!r} must contain a tensor, got {type(value).__name__}."
                )
            value = value.reshape(expected, -1)
            if value.shape[1] != 1:
                raise RuntimeError(
                    f"Expected scalar values under key={key!r}, got shape "
                    f"{tuple(value.shape)}."
                )
            if (
                value.dtype == torch.bool
                or torch.is_floating_point(value)
                or torch.is_complex(value)
            ):
                raise TypeError(
                    f"key={key!r} must contain integer values, got dtype={value.dtype}."
                )
        trajectory = trajectory.reshape(expected, -1)[:, 0]
        step = step.reshape(expected, -1)[:, 0]
        metadata = torch.stack(
            (
                index.to(device=step.device, dtype=torch.long),
                trajectory.to(device=step.device, dtype=torch.long),
                step.to(torch.long),
            ),
            dim=-1,
        ).cpu()
        slots, trajectories, steps = metadata.unbind(-1)
        return slots, trajectories, steps, step.device

    def _read_records(
        self, storage: Storage, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device]:
        storage_device = self._storage_device(storage)
        if storage_device is not None and index.device != storage_device:
            index = index.to(storage_device)
        lookup_index = index.clamp(0, len(storage) - 1)
        # Prefer column reads: advanced-indexing the storage itself would
        # materialize every stored key just to read two integer columns.
        trajectory = step = None
        try:
            source = storage[:]
        except Exception:
            source = None
        if is_tensor_collection(source):
            try:
                trajectory = source.get(self.trajectory_key, default=None)
                step = source.get(self.step_key, default=None)
            except Exception:
                trajectory = step = None
        if trajectory is not None and step is not None:
            trajectory = trajectory[lookup_index.to(trajectory.device)]
            step = step[lookup_index.to(step.device)]
        else:
            data = storage.get(lookup_index)
            if not is_tensor_collection(data):
                raise TypeError(
                    "Fragmented trajectory indexing requires a single-dimensional "
                    "storage whose slices return a tensor collection, such as "
                    "LazyTensorStorage or LazyStackStorage."
                )
            trajectory = data.get(self.trajectory_key)
            step = data.get(self.step_key)
        return self._validate_metadata(trajectory, step, index)

    def clear(self) -> None:
        self._trajectory_positions = None
        self._slot_traj = None
        self._slot_step = None
        self._device = None
        self._runs = None
        self._pending_indices.clear()
        self._pending_revisions.clear()
        self._pending_storage_ref = None
        self._cache_storage_ref = None
        self._cache_revision = None

    def _full_rebuild(self, storage: Storage, revision: int) -> None:
        if storage.ndim > 1:
            raise NotImplementedError(
                "Fragmented trajectory indexing only supports single-dimensional "
                f"storages, got storage.ndim={storage.ndim}."
            )
        storage_length = len(storage)
        index = torch.arange(storage_length, dtype=torch.long)
        slots, trajectories, steps, device = self._read_records(storage, index)
        trajectory_positions: dict[int, dict[int, int]] = defaultdict(dict)
        for position, trajectory, step in zip(
            slots.tolist(), trajectories.tolist(), steps.tolist()
        ):
            if step < 0:
                raise ValueError(
                    f"Step numbers must be non-negative, got {step} under "
                    f"step_key={self.step_key!r}."
                )
            positions = trajectory_positions[trajectory]
            if step in positions:
                raise RuntimeError(
                    "Found duplicate records for trajectory "
                    f"{trajectory!r} at step {step}. Trajectory-step pairs must "
                    "be unique in the live storage."
                )
            positions[step] = position

        # The read index is arange(storage_length), so the metadata rows are
        # already slot-aligned columns.
        self._slot_traj = trajectories.clone()
        self._slot_step = steps.clone()
        self._trajectory_positions = dict(trajectory_positions)
        self._device = device
        self._runs = None
        self._pending_indices.clear()
        self._pending_revisions.clear()
        self._pending_storage_ref = None
        self._cache_storage_ref = weakref.ref(storage)
        self._cache_revision = revision

    def _consume_pending(self) -> torch.Tensor:
        if len(self._pending_indices) == 1:
            index = self._pending_indices[0]
        elif all(
            item.device == self._pending_indices[0].device
            for item in self._pending_indices
        ):
            index = torch.cat(self._pending_indices)
        else:
            index = torch.cat([item.cpu() for item in self._pending_indices])
        index = index.reshape(-1)
        self._pending_indices.clear()
        self._pending_revisions.clear()
        self._pending_storage_ref = None
        return index

    def _apply_pending(self, storage: Storage, revision: int) -> None:
        index = self._consume_pending()
        storage_length = len(storage)
        if not index.numel():
            self._cache_revision = revision
            return
        slots, trajectories, steps, _ = self._read_records(storage, index)
        records = {
            slot: (trajectory, step)
            for slot, trajectory, step in zip(
                slots.tolist(), trajectories.tolist(), steps.tolist()
            )
            if 0 <= slot < storage_length
        }
        slots = list(records)
        if not slots:
            self._cache_revision = revision
            return
        trajectories, steps = zip(*records.values())

        try:
            size = self._slot_step.numel()
            max_slot = max(slots)
            if max_slot >= size:
                grow = max_slot + 1 - size
                self._slot_traj = torch.cat(
                    [self._slot_traj, self._slot_traj.new_zeros(grow)]
                )
                self._slot_step = torch.cat(
                    [self._slot_step, self._slot_step.new_full((grow,), -1)]
                )

            for slot in slots:
                step = int(self._slot_step[slot])
                if step < 0:
                    continue
                trajectory = int(self._slot_traj[slot])
                positions = self._trajectory_positions[trajectory]
                if positions.get(step) == slot:
                    del positions[step]
                if not positions:
                    del self._trajectory_positions[trajectory]
                self._slot_step[slot] = -1

            new_records = []
            seen = set()
            for slot, trajectory, step in zip(slots, trajectories, steps):
                if step < 0:
                    raise ValueError(
                        f"Step numbers must be non-negative, got {step} under "
                        f"step_key={self.step_key!r}."
                    )
                record = (trajectory, step)
                positions = self._trajectory_positions.get(trajectory)
                if record in seen or (positions is not None and step in positions):
                    raise RuntimeError(
                        "Found duplicate records for trajectory "
                        f"{trajectory!r} at step {step}. Trajectory-step pairs must "
                        "be unique in the live storage."
                    )
                seen.add(record)
                new_records.append((slot, trajectory, step))

            for slot, trajectory, step in new_records:
                self._trajectory_positions.setdefault(trajectory, {})[step] = slot
                self._slot_traj[slot] = trajectory
                self._slot_step[slot] = step
        except Exception:
            # The pending list is already consumed and the maps may be half
            # mutated; drop the index so the next refresh rebuilds instead of
            # serving a partially applied update.
            self.clear()
            raise

        self._runs = None
        self._cache_revision = revision

    def refresh(self, storage: Storage) -> None:
        revision = int(storage._mutation_revision)
        if (
            self._trajectory_positions is None
            or self._cache_storage_ref is None
            or self._cache_storage_ref() is not storage
        ):
            return self._full_rebuild(storage, revision)
        if (
            self._pending_indices
            and self._pending_storage_ref is not None
            and self._pending_storage_ref() is storage
        ):
            changed_revisions = {
                pending_revision
                for pending_revision in self._pending_revisions
                if pending_revision > self._cache_revision
            }
            expected_revisions = set(range(self._cache_revision + 1, revision + 1))
            if changed_revisions == expected_revisions:
                return self._apply_pending(storage, revision)
            return self._full_rebuild(storage, revision)
        if self._cache_revision != revision:
            return self._full_rebuild(storage, revision)
        return None

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        if storage is None:
            self.clear()
            return
        if (
            self._pending_storage_ref is not None
            and self._pending_storage_ref() is not storage
        ):
            self.clear()
        self._pending_storage_ref = weakref.ref(storage)
        self._pending_indices.append(
            torch.as_tensor(index, dtype=torch.long).detach().reshape(-1)
        )
        self._pending_revisions.append(int(storage._mutation_revision))

    def runs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns packed physical slots, run offsets, and logical run lengths."""
        if self._trajectory_positions is None:
            raise RuntimeError("The fragmented trajectory index has not been built.")
        if self._runs is not None:
            return self._runs

        live = (self._slot_step >= 0).nonzero().squeeze(-1)
        trajectory = self._slot_traj[live]
        step = self._slot_step[live]
        # Stable lexsort by (trajectory, step): the packing is then a pure
        # function of the stored values, so samples drawn with a given RNG
        # state do not depend on the write or rebuild history.
        order = torch.argsort(step, stable=True)
        order = order[torch.argsort(trajectory[order], stable=True)]
        sorted_trajectory = trajectory[order]
        sorted_step = step[order]
        new_run = torch.ones_like(sorted_step, dtype=torch.bool)
        if sorted_step.numel() > 1:
            new_run[1:] = (sorted_trajectory[1:] != sorted_trajectory[:-1]) | (
                sorted_step[1:] != sorted_step[:-1] + 1
            )
        run_offsets = new_run.nonzero().squeeze(-1)
        boundaries = torch.cat(
            [run_offsets, run_offsets.new_tensor([sorted_step.numel()])]
        )
        run_lengths = boundaries[1:] - boundaries[:-1]

        device = self._device
        self._runs = (
            live[order].to(device),
            run_offsets.to(device),
            run_lengths.to(device),
        )
        return self._runs

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in (
            "_trajectory_positions",
            "_slot_traj",
            "_slot_step",
            "_device",
            "_runs",
        ):
            state[key] = None
        state["_pending_indices"] = []
        state["_pending_revisions"] = []
        state["_pending_storage_ref"] = None
        state["_cache_storage_ref"] = None
        state["_cache_revision"] = None
        return state
