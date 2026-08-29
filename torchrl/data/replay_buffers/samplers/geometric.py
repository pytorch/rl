# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.storages import Storage

from ._trajectory import _FragmentedTrajectoryIndex
from .base import Sampler

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


class GeometricTrajectoryWindowSampler(Sampler):
    r"""Samples windows around trajectory steps with a geometric future offset.

    This sampler is intended for replay buffers populated by asynchronous
    sources, where consecutive steps of one trajectory need not be adjacent in
    storage. Each stored item must carry a trajectory identifier and a step
    number. At sampling time, the sampler draws a shared future offset
    :math:`k` with probability proportional to

    .. math::

        P(k) = (1-y)y^k,

    then samples anchors uniformly from all stored trajectory steps that have
    the requested history and at least :math:`k` consecutive future steps.
    The returned storage indices have the fixed shape
    ``[batch_size, history + max_future + 1]``. Positions after :math:`t+k` are
    padded, which keeps sampling compatible with :func:`torch.compile` and CUDA
    graphs.

    Steps before the beginning of a trajectory are represented by repeating
    its step-zero storage index. The ``"validity_mask"`` sampler metadata marks
    those entries as ``False`` so callers can replace them with zeroes. The
    sampled offset is repeated over the window under ``"future_offset"``, and
    ``"anchor_index"`` contains the storage index of step :math:`t`. Positions
    after :math:`t+k` repeat its storage index and are also marked ``False``.

    The geometric distribution is truncated at ``max_future`` and at the
    largest offset currently feasible for at least one anchor. Equivalently,
    :math:`k` is sampled from the geometric distribution conditioned on the
    current buffer contents and the configured upper bound.

    Args:
        history (int): Number of steps preceding the anchor. The returned
            history including the anchor therefore has length ``history + 1``.
        max_future (int): Maximum sampled future offset. This also fixes the
            returned window length to ``history + max_future + 1``.
        continuation_probability (float): The geometric continuation
            probability :math:`y`. Must satisfy ``0 <= y < 1``.

    Keyword Args:
        trajectory_key (NestedKey, optional): Key containing a trajectory id
            for each stored item. IDs must be unique across trajectories.
            Defaults to ``("collector", "traj_ids")``.
        step_key (NestedKey, optional): Key containing a non-negative integer
            step number for each item. Step numbers must start at zero and be
            unique within each trajectory. Defaults to ``"step_count"``.
        compile (bool or dict of kwargs, optional): If truthy, compiles the
            tensor-only sampling kernel with :func:`torch.compile`. A dictionary
            is forwarded as keyword arguments. Defaults to ``False``.

    .. note:: This sampler supports single-dimensional TensorDict-backed
        storages. Trajectory ids and step numbers must be scalar integer tensors.
        Missing non-negative steps are treated as unavailable data, not as
        padding, so anchors whose required window crosses a hole or an
        overwritten prefix are excluded.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import (
        ...     GeometricTrajectoryWindowSampler,
        ...     LazyTensorStorage,
        ...     TensorDictReplayBuffer,
        ... )
        >>> sampler = GeometricTrajectoryWindowSampler(
        ...     history=1,
        ...     max_future=2,
        ...     continuation_probability=0.0,
        ...     trajectory_key="trajectory",
        ...     step_key="step",
        ... )
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(10), sampler=sampler, batch_size=2
        ... )
        >>> rb.extend(TensorDict(
        ...     {
        ...         "trajectory": torch.tensor([0, 1, 0, 1]),
        ...         "step": torch.tensor([0, 0, 1, 1]),
        ...         "observation": torch.tensor([0.0, 10.0, 1.0, 11.0]),
        ...     },
        ...     batch_size=[4],
        ... ))
        tensor([0, 1, 2, 3])
        >>> sample = rb.sample()
        >>> sample.shape
        torch.Size([2, 4])
        >>> sample["validity_mask"].shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        history: int,
        max_future: int,
        continuation_probability: float,
        *,
        trajectory_key: NestedKey = ("collector", "traj_ids"),
        step_key: NestedKey = "step_count",
        compile: bool | dict = False,
    ):
        if isinstance(history, bool) or not isinstance(history, (int, np.integer)):
            raise TypeError(f"history must be a non-negative integer, got {history!r}.")
        if history < 0:
            raise ValueError(f"history must be non-negative, got {history}.")
        if isinstance(max_future, bool) or not isinstance(
            max_future, (int, np.integer)
        ):
            raise TypeError(
                f"max_future must be a non-negative integer, got {max_future!r}."
            )
        if max_future < 0:
            raise ValueError(f"max_future must be non-negative, got {max_future}.")
        if isinstance(continuation_probability, bool) or not isinstance(
            continuation_probability, (float, int, np.floating, np.integer)
        ):
            raise TypeError(
                "continuation_probability must be a real number in [0, 1), got "
                f"{continuation_probability!r}."
            )
        continuation_probability = float(continuation_probability)
        if not 0.0 <= continuation_probability < 1.0:
            raise ValueError(
                "continuation_probability must be in [0, 1), got "
                f"{continuation_probability}."
            )
        self.history = int(history)
        self.max_future = int(max_future)
        self.continuation_probability = continuation_probability
        self.trajectory_key = trajectory_key
        self.step_key = step_key
        self.compile = bool(compile)
        self._trajectory_index = _FragmentedTrajectoryIndex(trajectory_key, step_key)
        self._anchor_slots: set[int] | None = None
        self._max_future_by_slot: torch.Tensor | None = None
        self._sample_index = self._sample_kernel
        if self.compile:
            kwargs = compile if isinstance(compile, dict) else {}
            self._sample_index = torch.compile(self._sample_kernel, **kwargs)

    def _clear_index(self) -> None:
        self._trajectory_index.clear()
        self._anchor_slots = None
        self._max_future_by_slot = None

    def _rebuild_window_metadata(self) -> None:
        trajectory_positions = self._trajectory_index.trajectory_positions
        capacity = self._trajectory_index.following.numel()
        max_future_by_slot = [-1] * capacity
        anchor_slots = set()
        for positions in trajectory_positions.values():
            sorted_steps = sorted(positions)
            run_start = 0
            for run_end, step in enumerate(sorted_steps):
                is_run_end = (
                    run_end == len(sorted_steps) - 1
                    or sorted_steps[run_end + 1] != step + 1
                )
                if not is_run_end:
                    continue
                first_step = sorted_steps[run_start]
                last_step = step
                for anchor_step in sorted_steps[run_start : run_end + 1]:
                    if max(0, anchor_step - self.history) < first_step:
                        continue
                    anchor_position = positions[anchor_step]
                    max_future_by_slot[anchor_position] = min(
                        self.max_future, last_step - anchor_step
                    )
                    anchor_slots.add(anchor_position)
                run_start = run_end + 1

        self._anchor_slots = anchor_slots
        self._max_future_by_slot = torch.tensor(
            max_future_by_slot,
            dtype=torch.long,
            device=self._trajectory_index.following.device,
        )

    def _apply_window_update(
        self, affected: dict[int, set[int]], removed_slots: set[int]
    ) -> None:
        future_updates = {slot: -1 for slot in removed_slots}
        for slot in removed_slots:
            self._anchor_slots.discard(slot)
        for trajectory, changed_steps in affected.items():
            positions = self._trajectory_index.trajectory_positions.get(trajectory, {})
            metadata_steps = set()
            for step in changed_steps:
                metadata_steps.update(
                    range(max(0, step - self.max_future), step + self.history + 1)
                )
            for step in metadata_steps:
                slot = positions.get(step)
                if slot is None:
                    continue
                history_start = max(0, step - self.history)
                if all(
                    history_step in positions
                    for history_step in range(history_start, step + 1)
                ):
                    available_future = 0
                    while (
                        available_future < self.max_future
                        and step + available_future + 1 in positions
                    ):
                        available_future += 1
                    future_updates[slot] = available_future
                    self._anchor_slots.add(slot)
                else:
                    future_updates[slot] = -1
                    self._anchor_slots.discard(slot)

        if future_updates:
            updates = torch.tensor(
                list(future_updates.items()),
                dtype=torch.long,
                device=self._max_future_by_slot.device,
            )
            self._max_future_by_slot[updates[:, 0]] = updates[:, 1]

    def _maybe_refresh_index(self, storage: Storage) -> None:
        update = self._trajectory_index.refresh(storage)
        if update is None:
            return
        if update.full_rebuild or self._max_future_by_slot is None:
            self._rebuild_window_metadata()
        else:
            self._apply_window_update(update.affected, update.removed_slots)

    def _sample_kernel(
        self,
        max_future_by_slot: torch.Tensor,
        previous: torch.Tensor,
        following: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = max_future_by_slot.device
        offsets = torch.arange(self.max_future + 1, device=device)
        max_available = max_future_by_slot.max()
        offset_weights = self.continuation_probability ** offsets.to(torch.float32)
        offset_weights = offset_weights * (offsets <= max_available)
        future_offset = torch.multinomial(
            offset_weights, 1, replacement=True, generator=self._rng
        )
        anchor_weights = (max_future_by_slot >= future_offset).to(torch.float32)
        anchors = torch.multinomial(
            anchor_weights, batch_size, replacement=True, generator=self._rng
        )

        indices = [anchors]
        validity = [torch.ones_like(anchors, dtype=torch.bool)]
        current = anchors
        for _ in range(self.history):
            candidate = previous[current]
            is_valid = candidate >= 0
            current = torch.where(is_valid, candidate, current)
            indices.append(current)
            validity.append(is_valid)
        indices.reverse()
        validity.reverse()

        current = anchors
        for offset in range(1, self.max_future + 1):
            take_step = offsets[offset] <= future_offset
            candidate = following[current]
            current = torch.where(take_step, candidate, current)
            indices.append(current)
            validity.append(take_step.expand_as(current))

        index = torch.stack(indices, dim=-1)
        validity_mask = torch.stack(validity, dim=-1)
        expanded_offset = future_offset.expand_as(index)
        anchor_index = anchors.unsqueeze(-1).expand_as(index)
        return index, validity_mask, expanded_offset, anchor_index

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        self._maybe_refresh_index(storage)
        if not self._anchor_slots:
            raise RuntimeError(
                "No trajectory step has a complete stored history. Add more "
                "consecutive steps or lower history."
            )
        index, validity, future_offset, anchor_index = self._sample_index(
            self._max_future_by_slot,
            self._trajectory_index.previous,
            self._trajectory_index.following,
            batch_size,
        )
        info = {
            "validity_mask": validity,
            "future_offset": future_offset,
            "anchor_index": anchor_index,
        }
        return index, info

    def add(self, index: int) -> None:
        return

    def extend(self, index: torch.Tensor) -> None:
        return

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        self._trajectory_index.mark_update(index, storage=storage)

    def _empty(self) -> None:
        self._clear_index()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._clear_index()

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        TensorDict(self.state_dict()).memmap(path)

    def loads(self, path):
        self.load_state_dict(TensorDict.load_memmap(path).to_dict())

    def __getstate__(self):
        state = super().__getstate__()
        state["_anchor_slots"] = None
        state["_max_future_by_slot"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(history={self.history}, "
            f"max_future={self.max_future}, "
            f"continuation_probability={self.continuation_probability}, "
            f"trajectory_key={self.trajectory_key}, step_key={self.step_key}, "
            f"compile={self.compile})"
        )
