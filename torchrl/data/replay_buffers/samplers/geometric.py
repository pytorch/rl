# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import is_tensor_collection, TensorDict
from tensordict.utils import NestedKey

from torchrl.data.replay_buffers.storages import Storage

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
    The returned storage indices have shape ``[batch_size, history + k + 1]``
    and cover ``[t-history, ..., t+k]``.

    Steps before the beginning of a trajectory are represented by repeating
    its step-zero storage index. The ``"validity_mask"`` sampler metadata marks
    those entries as ``False`` so callers can replace them with zeroes. The
    sampled offset is repeated over the window under ``"future_offset"``, and
    ``"anchor_index"`` contains the storage index of step :math:`t`.

    Since replay storage is finite, the geometric distribution is truncated at
    the largest offset currently feasible for at least one anchor. Equivalently,
    :math:`k` is sampled from the geometric distribution conditioned on the
    current buffer contents.

    Args:
        history (int): Number of steps preceding the anchor. The returned
            history including the anchor therefore has length ``history + 1``.
        continuation_probability (float): The geometric continuation
            probability :math:`y`. Must satisfy ``0 <= y < 1``.

    Keyword Args:
        trajectory_key (NestedKey, optional): Key containing a trajectory id
            for each stored item. IDs must be unique across trajectories.
            Defaults to ``("collector", "traj_ids")``.
        step_key (NestedKey, optional): Key containing a non-negative integer
            step number for each item. Step numbers must start at zero and be
            unique within each trajectory. Defaults to ``"step_count"``.

    .. note:: This sampler supports single-dimensional TensorDict-backed
        storages. Missing non-negative steps are treated as unavailable data,
        not as padding, so anchors whose required window crosses a hole or an
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
        torch.Size([2, 2])
        >>> sample["validity_mask"].shape
        torch.Size([2, 2])
    """

    def __init__(
        self,
        history: int,
        continuation_probability: float,
        *,
        trajectory_key: NestedKey = ("collector", "traj_ids"),
        step_key: NestedKey = "step_count",
    ):
        if isinstance(history, bool) or not isinstance(history, (int, np.integer)):
            raise TypeError(f"history must be a non-negative integer, got {history!r}.")
        if history < 0:
            raise ValueError(f"history must be non-negative, got {history}.")
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
        self.continuation_probability = continuation_probability
        self.trajectory_key = trajectory_key
        self.step_key = step_key
        self._trajectory_positions: dict[Any, dict[int, int]] | None = None
        self._anchors: list[tuple[Any, int, int]] | None = None
        self._anchor_max_future: torch.Tensor | None = None
        self._cache_storage_id: int | None = None
        self._cache_revision: int | None = None

    @staticmethod
    def _to_key_list(values: Any) -> list:
        if hasattr(values, "tolist"):
            values = values.tolist()
        else:
            values = list(values)
        return [tuple(value) if isinstance(value, list) else value for value in values]

    def _invalidate_cache(self) -> None:
        self._trajectory_positions = None
        self._anchors = None
        self._anchor_max_future = None
        self._cache_storage_id = None
        self._cache_revision = None

    def _maybe_build_index(self, storage: Storage) -> None:
        revision = int(storage._mutation_revision)
        if (
            self._trajectory_positions is not None
            and self._cache_storage_id == id(storage)
            and self._cache_revision == revision
        ):
            return
        if storage.ndim > 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports single-dimensional storages, "
                f"got storage.ndim={storage.ndim}."
            )
        data = storage[:]
        if not is_tensor_collection(data):
            raise TypeError(
                f"{type(self).__name__} requires a single-dimensional storage "
                "whose slices return a tensor collection, such as "
                "LazyTensorStorage or LazyStackStorage."
            )
        trajectory_values = data.get(self.trajectory_key)
        trajectories = self._to_key_list(trajectory_values)
        steps = data.get(self.step_key)
        if not isinstance(steps, torch.Tensor):
            raise TypeError(
                f"step_key={self.step_key!r} must contain a tensor, got "
                f"{type(steps).__name__}."
            )
        if steps.shape[0] != len(storage):
            raise RuntimeError(
                f"Expected step_key={self.step_key!r} to contain one value per "
                f"storage item, got shape {tuple(steps.shape)} for {len(storage)} items."
            )
        steps = steps.reshape(len(storage), -1)
        if steps.shape[1] != 1:
            raise RuntimeError(
                f"Expected scalar step numbers under step_key={self.step_key!r}, "
                f"got shape {tuple(steps.shape)}."
            )
        steps = steps[:, 0]
        if (
            steps.dtype == torch.bool
            or torch.is_floating_point(steps)
            or torch.is_complex(steps)
        ):
            raise TypeError(
                f"step_key={self.step_key!r} must contain integer step numbers, "
                f"got dtype={steps.dtype}."
            )
        steps = steps.cpu().tolist()
        if len(trajectories) != len(storage):
            raise RuntimeError(
                f"Expected trajectory_key={self.trajectory_key!r} to contain one "
                f"value per storage item, got {len(trajectories)} values for "
                f"{len(storage)} items."
            )

        trajectory_positions: dict[Any, dict[int, int]] = defaultdict(dict)
        for position, (trajectory, step) in enumerate(zip(trajectories, steps)):
            step = int(step)
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

        anchors: list[tuple[Any, int, int]] = []
        max_futures = []
        for trajectory, positions in trajectory_positions.items():
            sorted_steps = sorted(positions)
            run_start = 0
            for run_end in range(len(sorted_steps)):
                is_last = run_end == len(sorted_steps) - 1
                if (
                    not is_last
                    and sorted_steps[run_end + 1] == sorted_steps[run_end] + 1
                ):
                    continue
                first_step = sorted_steps[run_start]
                last_step = sorted_steps[run_end]
                for step in sorted_steps[run_start : run_end + 1]:
                    if max(0, step - self.history) < first_step:
                        continue
                    max_future = last_step - step
                    anchors.append((trajectory, step, positions[step]))
                    max_futures.append(max_future)
                run_start = run_end + 1

        self._trajectory_positions = dict(trajectory_positions)
        self._anchors = anchors
        self._anchor_max_future = torch.tensor(max_futures, dtype=torch.long)
        self._cache_storage_id = id(storage)
        self._cache_revision = revision

    def _sample_future_offset(self, max_future: int) -> int:
        if not max_future or not self.continuation_probability:
            return 0
        device = self._rng.device if self._rng is not None else None
        offsets = torch.arange(max_future + 1, dtype=torch.float64, device=device)
        weights = self.continuation_probability**offsets
        return int(torch.multinomial(weights, 1, generator=self._rng).cpu().item())

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        self._maybe_build_index(storage)
        if not self._anchors:
            raise RuntimeError(
                "No trajectory step has a complete stored history. Add more "
                "consecutive steps or lower history."
            )
        max_future = int(self._anchor_max_future.max())
        future_offset = self._sample_future_offset(max_future)
        eligible = (self._anchor_max_future >= future_offset).nonzero(as_tuple=True)[0]
        device = self._rng.device if self._rng is not None else None
        selected = torch.randint(
            eligible.numel(),
            (batch_size,),
            generator=self._rng,
            device=device,
        ).cpu()
        selected = eligible[selected]

        window_length = self.history + future_offset + 1
        index = torch.empty((batch_size, window_length), dtype=torch.long)
        validity = torch.empty((batch_size, window_length), dtype=torch.bool)
        anchor_index = torch.empty_like(index)
        relative_steps = torch.arange(-self.history, future_offset + 1)
        for row, anchor_id in enumerate(selected.tolist()):
            trajectory, step, anchor_position = self._anchors[anchor_id]
            target_steps = step + relative_steps
            row_validity = target_steps >= 0
            safe_steps = target_steps.clamp_min(0).tolist()
            positions = self._trajectory_positions[trajectory]
            index[row] = torch.tensor(
                [positions[target_step] for target_step in safe_steps],
                dtype=torch.long,
            )
            validity[row] = row_validity
            anchor_index[row].fill_(anchor_position)

        info = {
            "validity_mask": validity,
            "future_offset": torch.full_like(index, future_offset),
            "anchor_index": anchor_index,
        }
        return index, info

    def add(self, index: int) -> None:
        self._invalidate_cache()

    def extend(self, index: torch.Tensor) -> None:
        self._invalidate_cache()

    def mark_update(
        self, index: int | torch.Tensor, *, storage: Storage | None = None
    ) -> None:
        self._invalidate_cache()

    def _empty(self) -> None:
        self._invalidate_cache()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._invalidate_cache()

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        TensorDict(self.state_dict()).memmap(path)

    def loads(self, path):
        self.load_state_dict(TensorDict.load_memmap(path).to_dict())

    def __getstate__(self):
        state = super().__getstate__()
        state["_trajectory_positions"] = None
        state["_anchors"] = None
        state["_anchor_max_future"] = None
        state["_cache_storage_id"] = None
        state["_cache_revision"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(history={self.history}, "
            f"continuation_probability={self.continuation_probability}, "
            f"trajectory_key={self.trajectory_key}, step_key={self.step_key})"
        )
