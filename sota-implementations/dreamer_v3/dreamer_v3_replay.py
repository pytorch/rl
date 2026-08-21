# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Driver-step accounting and the continuous replay stream of the example."""
from __future__ import annotations

from typing import TypeAlias

import torch
from tensordict import TensorDictBase

from torchrl.data import LazyTensorStorage, ReplayBuffer, SliceSampler

ReplayIndex: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
ReplaySampleInfo: TypeAlias = dict[str, ReplayIndex]
_REPLAY_CONTEXT_VALID_KEY = ("collector", "context_valid")


# --- Driver step accounting --------------------------------------------------


def driver_step_for_action(
    action_index: int,
    env_index: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Return the driver step of a one-based action index, with reset records."""
    reset_records = 1 + (action_index - 1) // max_episode_steps
    vector_record = action_index + reset_records
    return (vector_record - 1) * num_envs + env_index + 1


def collector_action_budget(
    record_budget: int,
    num_envs: int,
    max_episode_steps: int,
) -> int:
    """Return the actions in a driver-record budget that also holds resets."""
    if record_budget % num_envs:
        raise ValueError(
            "A driver-record budget must be divisible by the number of "
            f"environments, got {record_budget} and {num_envs}."
        )
    vector_records = record_budget // num_envs
    reset_records = (vector_records + max_episode_steps) // (max_episode_steps + 1)
    return (vector_records - reset_records) * num_envs


class DreamerV3UpdateRatio:
    """Schedule learner updates from a ratio of updates to driver records.

    Each call truncates the count from the cumulative driver-record count and
    keeps the remainder. The first call returns one update.

    Args:
        ratio (float): Learner updates for each driver record.
    """

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self._previous: float | None = None

    def __call__(self, record_count: int) -> int:
        if self.ratio <= 0:
            return 0
        if self._previous is None:
            self._previous = float(record_count)
            return 1
        repeats = int((record_count - self._previous) * self.ratio)
        self._previous += repeats / self.ratio
        return repeats


# --- Replay: record stream, writeback, sampling ------------------------------


def _refresh_replay_context(
    replay_buffer: ReplayBuffer,
    sample_indices: ReplayIndex,
    sample_generations: torch.Tensor,
    state: torch.Tensor,
    belief: torch.Tensor,
) -> None:
    if not isinstance(sample_indices, tuple):
        sample_indices = (sample_indices,)
    batch_size, sequence_length = state.shape[:2]
    context_length = sequence_length + 1
    destination_indices = tuple(
        index.reshape(batch_size, context_length)[:, 1:].reshape(-1)
        for index in sample_indices
    )
    destination_generation = sample_generations.reshape(batch_size, context_length)[
        :, 1:
    ].reshape(-1)
    # Slices overlap, and a CUDA index write leaves duplicate coordinates
    # undefined. Keep the last value of each coordinate.
    coordinates = torch.stack(destination_indices, -1)
    linear_coordinate = coordinates[:, 0]
    for coordinate, size in zip(
        coordinates[:, 1:].unbind(-1), replay_buffer.storage.shape[1:]
    ):
        linear_coordinate = linear_coordinate * int(size) + coordinate
    order = linear_coordinate.argsort(stable=True)
    ordered_coordinate = linear_coordinate[order]
    keep_ordered = torch.ones_like(ordered_coordinate, dtype=torch.bool)
    keep_ordered[:-1] = ordered_coordinate[:-1] != ordered_coordinate[1:]
    keep = order[keep_ordered]
    destination_indices = tuple(index[keep] for index in destination_indices)
    destination_index = (
        torch.stack(destination_indices, -1)
        if len(destination_indices) > 1
        else destination_indices[0]
    )
    destination_generation = destination_generation[keep]
    replay_buffer.update_if_present(
        index=destination_index,
        generation=destination_generation,
        patch={
            "state": state.detach()
            .float()
            .reshape(-1, state.shape[-1])[keep.to(state.device)],
            "belief": belief.detach()
            .float()
            .reshape(-1, belief.shape[-1])[keep.to(belief.device)],
            _REPLAY_CONTEXT_VALID_KEY: torch.ones(
                (keep.numel(), 1), dtype=torch.bool, device=state.device
            ),
        },
    )


class DreamerV3ReplayPipeline:
    """Sample one batch ahead, and apply each latent refresh one update behind."""

    def __init__(self) -> None:
        self._prefetched: tuple[TensorDictBase, ReplaySampleInfo] | None = None
        self._pending_context: tuple[
            ReplaySampleInfo, torch.Tensor, torch.Tensor
        ] | None = None

    @property
    def has_prefetched(self) -> bool:
        return self._prefetched is not None

    @property
    def has_pending_context(self) -> bool:
        return self._pending_context is not None

    def prefetch(self, replay_buffer: ReplayBuffer) -> None:
        if self._prefetched is None:
            self._prefetched = replay_buffer.sample(return_info=True)

    def take(
        self, replay_buffer: ReplayBuffer
    ) -> tuple[TensorDictBase, ReplaySampleInfo]:
        """Return the prefetched batch, and sample the next one."""
        self.prefetch(replay_buffer)
        current = self._prefetched
        self._prefetched = replay_buffer.sample(return_info=True)
        return current

    def apply_pending_context(self, replay_buffer: ReplayBuffer) -> None:
        """Apply the previous refresh, after ``take`` samples the next batch."""
        if self._pending_context is not None:
            pending_info, pending_state, pending_belief = self._pending_context
            _refresh_replay_context(
                replay_buffer,
                pending_info["index"],
                pending_info["index_generation"],
                pending_state,
                pending_belief,
            )
            self._pending_context = None

    def stage_context(
        self,
        sample_info: ReplaySampleInfo,
        state: torch.Tensor,
        belief: torch.Tensor,
    ) -> None:
        if self._pending_context is not None:
            raise RuntimeError(
                "The preceding replay context must be applied before staging "
                "another learner output."
            )
        self._pending_context = (sample_info, state, belief)


class DreamerV3ReplayRecordBuilder:
    """Convert collector transitions into the replay stream."""

    def __init__(self, num_streams: int) -> None:
        self.num_streams = num_streams
        self._started = False

    def __call__(self, data: TensorDictBase) -> TensorDictBase:
        if self.num_streams == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2 or data.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected collector data with shape [num_streams, time], got "
                f"{tuple(data.shape)} for {self.num_streams} streams."
            )

        records = []
        record_keys = (
            "action",
            "is_init",
            "state",
            "belief",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
            ("next", "terminated"),
        )
        for time_index in range(data.shape[1]):
            collector_step = data[:, time_index]
            reset = collector_step.get("is_init").reshape(self.num_streams, -1).any(-1)
            insert_reset = reset if self._started else torch.zeros_like(reset)
            if insert_reset.any() and not insert_reset.all():
                raise RuntimeError(
                    "The 2D DreamerV3 replay stream requires synchronized episode "
                    "resets across collector environments."
                )

            transition = collector_step.select(*record_keys, strict=True).clone()
            # This record models the transition into next.observation, so it
            # keeps its action; the separate reset record marks the reset.
            transition.get("is_init").zero_()
            transition.set(
                _REPLAY_CONTEXT_VALID_KEY,
                torch.ones_like(transition.get("is_init"), dtype=torch.bool),
            )

            if insert_reset.any():
                reset_transition = transition.clone()
                for key in (
                    "action",
                    "state",
                    "belief",
                    ("next", "reward"),
                    ("next", "done"),
                    ("next", "terminated"),
                ):
                    reset_transition.get(key).zero_()
                reset_transition.get("is_init").fill_(True)
                reset_transition.set(
                    ("next", "observation"),
                    collector_step.get("observation").clone(),
                )
                records.append(reset_transition)

            records.append(transition)
            self._started = True

        return torch.stack(records, 1)


class DreamerV3ShiftedRecordExtender:
    """Keep a placeholder record for the posterior of the newest transition.

    The next collector batch completes it in place, at the same slot.
    """

    def __init__(self, num_streams: int) -> None:
        self.num_streams = num_streams
        self._tail_index: torch.Tensor | None = None
        self._tail_generation: torch.Tensor | None = None

    @staticmethod
    def _tail_placeholder(records: TensorDictBase) -> TensorDictBase:
        tail = records[:, -1].clone()
        tail.get("action").zero_()
        tail.get("is_init").zero_()
        tail.get("state").zero_()
        tail.get("belief").zero_()
        tail.get(("next", "reward")).zero_()
        tail.get(("next", "done")).zero_()
        tail.get(("next", "terminated")).zero_()
        tail.get(_REPLAY_CONTEXT_VALID_KEY).zero_()
        return tail.unsqueeze(1)

    def _finalize_tail(
        self, replay_buffer: ReplayBuffer, records: TensorDictBase
    ) -> None:
        tail_index = self._tail_index
        tail_generation = self._tail_generation
        if tail_index is None or tail_generation is None:
            return

        storage = replay_buffer.storage
        stored = (
            storage[tail_index]
            if storage.ndim == 1
            else storage[tuple(tail_index.unbind(-1))]
        )
        incoming = records[:, 0].clone().to(stored.device)
        context_valid = stored.get(_REPLAY_CONTEXT_VALID_KEY)
        incoming.set(
            "state",
            torch.where(context_valid, stored.get("state"), incoming.get("state")),
        )
        incoming.set(
            "belief",
            torch.where(context_valid, stored.get("belief"), incoming.get("belief")),
        )
        incoming.set(
            _REPLAY_CONTEXT_VALID_KEY,
            torch.ones_like(context_valid, dtype=torch.bool),
        )
        result = replay_buffer.update_if_present(
            index=tail_index,
            generation=tail_generation,
            patch=incoming,
        )
        if result.updated_count != self.num_streams:
            raise RuntimeError(
                "The mutable DreamerV3 replay tail was recycled before it "
                "could be finalized."
            )

    def extend(
        self,
        replay_buffer: ReplayBuffer,
        replay_sampler: DreamerV3ReplaySampler,
        records: TensorDictBase,
    ) -> torch.Tensor:
        if records.ndim != 2 or records.shape[0] != self.num_streams:
            raise RuntimeError(
                "Expected replay records with shape [num_streams, time], got "
                f"{tuple(records.shape)} for {self.num_streams} streams."
            )
        self._finalize_tail(replay_buffer, records)
        placeholder = self._tail_placeholder(records)
        if self._tail_index is None:
            appended = torch.cat([records, placeholder], 1)
        else:
            appended = torch.cat([records[:, 1:], placeholder], 1)
        replay_indices = replay_buffer.extend(
            appended if self.num_streams > 1 else appended.reshape(-1)
        )
        replay_sampler.observe_extend(replay_indices, replay_buffer.storage)

        coordinates = torch.as_tensor(replay_indices, dtype=torch.long).reshape(
            appended.shape[1], self.num_streams, replay_buffer.storage.ndim
        )
        tail_index = coordinates[-1]
        if replay_buffer.storage.ndim == 1:
            tail_index = tail_index[:, 0]
        self._tail_index = tail_index.clone()
        self._tail_generation = replay_buffer.writer.generations_of(
            self._tail_index
        ).clone()
        return replay_indices


class DreamerV3ReplaySampler(SliceSampler):
    """Slice sampler that takes the oldest queued blocks before uniform ones."""

    def __init__(self, *args, online: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.online = online
        self._stream_lengths: torch.Tensor | None = None
        self._online_queue: list[torch.Tensor] = []

    @property
    def online_queue_size(self) -> int:
        return len(self._online_queue)

    def observe_extend(self, index: torch.Tensor, storage: LazyTensorStorage) -> None:
        """Queue the start of each new, non-overlapping ``slice_len`` block."""
        if not self.online:
            return
        index = torch.as_tensor(index, dtype=torch.long)
        if storage.ndim == 1:
            coordinates = index.reshape(-1, 1, 1)
            num_streams = 1
        else:
            num_streams = storage.shape[1:].numel()
            coordinates = index.reshape(-1, num_streams, storage.ndim)
        if self._stream_lengths is None:
            self._stream_lengths = torch.zeros(num_streams, dtype=torch.long)
        elif self._stream_lengths.numel() != num_streams:
            raise RuntimeError(
                "The number of replay streams changed after initialization."
            )

        max_time = storage._max_size_along_dim0()
        for coordinate_row in coordinates:
            self._stream_lengths.add_(1)
            enqueue = (self._stream_lengths > self.slice_len) & (
                (self._stream_lengths - 1).remainder(self.slice_len) == 0
            )
            if enqueue.any():
                starts = coordinate_row[enqueue].clone()
                starts[:, 0].sub_(self.slice_len - 1).remainder_(max_time)
                self._online_queue.extend(starts.unbind(0))

    def _drop_stale_online(self, storage: LazyTensorStorage, seq_length: int) -> None:
        """Drop queued starts whose ``seq_length`` window is no longer stored."""
        if not self._online_queue or not storage._is_full:
            return
        stored_time = storage.shape[0]
        oldest = (int(storage._last_cursor_index) + 1) % stored_time
        live = stored_time - seq_length + 1
        self._online_queue = [
            start
            for start in self._online_queue
            if (int(start[0]) - oldest) % stored_time < live
        ]

    def sample(
        self, storage: LazyTensorStorage, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        self._drop_stale_online(storage, seq_length)
        # Each sequence of a batch takes one online block if the queue has one.
        num_online = min(num_slices, len(self._online_queue))
        num_uniform = num_slices - num_online
        if storage.ndim > 2:
            raise RuntimeError("DreamerV3 continuous replay supports 1D or 2D storage.")
        if num_uniform:
            stored_time = storage.shape[0]
            num_starts = stored_time - seq_length + 1
            if num_starts < 1:
                raise RuntimeError(
                    f"Replay streams have length {stored_time}, but sampling "
                    f"requires {seq_length} records."
                )
            num_streams = 1 if storage.ndim == 1 else storage.shape[1]
            flat_start = torch.randint(
                num_starts * num_streams,
                (num_uniform,),
                generator=self._rng,
            )
            relative_time = flat_start.div(num_streams, rounding_mode="floor")
            stream = flat_start.remainder(num_streams)
            oldest_time = (
                (int(storage._last_cursor_index) + 1) % stored_time
                if storage._is_full
                else 0
            )
            start_time = (relative_time + oldest_time).remainder(stored_time)
            if storage.ndim == 1:
                uniform_starts = start_time.unsqueeze(-1)
            else:
                uniform_starts = torch.stack([start_time, stream], -1)
            uniform_coordinates = self._tensor_slices_from_startend(
                seq_length,
                uniform_starts,
                stored_time,
            ).reshape(num_uniform, seq_length, storage.ndim)
            index_device = uniform_starts.device
        else:
            uniform_coordinates = None
            index_device = self._online_queue[0].device

        if num_online:
            online_starts = torch.stack(
                [self._online_queue.pop(0) for _ in range(num_online)]
            ).to(index_device)
            online_coordinates = self._tensor_slices_from_startend(
                seq_length,
                online_starts,
                storage.shape[0],
            ).reshape(num_online, seq_length, storage.ndim)
        else:
            online_coordinates = None
        coordinates = torch.cat(
            [
                candidate
                for candidate in (online_coordinates, uniform_coordinates)
                if candidate is not None
            ],
            0,
        )
        return coordinates.reshape(-1, storage.ndim).unbind(-1), {}

    def _empty(self) -> None:
        super()._empty()
        self._stream_lengths = None
        self._online_queue.clear()
