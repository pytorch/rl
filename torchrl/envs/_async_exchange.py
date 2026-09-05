# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import expand_as_right

from torchrl._utils import timeit


def _receive_batch(
    result_queue,
    min_get: int,
    max_get: int | None,
    timeout: float | None,
) -> list[Any]:
    if min_get < 1:
        raise ValueError(f"min_get must be positive, got {min_get}.")
    if max_get is not None and max_get < min_get:
        raise ValueError(
            f"max_get must be greater than or equal to min_get, got "
            f"min_get={min_get} and max_get={max_get}."
        )
    if timeout is not None and timeout < 0:
        raise ValueError(f"timeout must be non-negative, got {timeout}.")

    # The deadline is anchored at call entry and bounds the entire call,
    # including the wait for the first result.
    deadline_timer = (
        None if timeout is None else timeit("async_env_batch_deadline").start()
    )

    items: list[Any] = []
    while len(items) < min_get:
        if deadline_timer is None:
            items.append(result_queue.get())
            continue
        remaining = timeout - deadline_timer.elapsed()
        try:
            if remaining <= 0:
                # Deadline passed: drain what is already available without
                # waiting before giving up.
                items.append(result_queue.get_nowait())
            else:
                items.append(result_queue.get(timeout=remaining))
        except queue.Empty:
            # Requeue the partial harvest so no result is lost, then signal
            # the missed deadline. The pool's pending-result accounting is
            # only updated by the caller on success, so state stays
            # consistent.
            for item in items:
                result_queue.put(item)
            raise TimeoutError(
                f"async recv timed out: {len(items)}/{min_get} results after "
                f"{timeout}s; partial results were requeued and remain "
                f"available to the next call."
            ) from None

    limit = max_get if max_get is not None else float("inf")
    while len(items) < limit:
        try:
            if deadline_timer is None:
                items.append(result_queue.get_nowait())
                continue
            remaining = timeout - deadline_timer.elapsed()
            if remaining <= 0:
                items.append(result_queue.get_nowait())
            else:
                items.append(result_queue.get(timeout=remaining))
        except queue.Empty:
            break
    return items


class _SharedSlotExchange:
    def __init__(self, fake_tensordicts: list[TensorDictBase]) -> None:
        self._validate(fake_tensordicts)
        template = torch.stack(fake_tensordicts, 0)
        self.input_buffer = template.clone().share_memory_()
        self.result_buffer = template.clone().share_memory_()
        self.next_buffer = template.clone().share_memory_()
        self.input_slots = self.input_buffer.unbind(0)
        self.result_slots = self.result_buffer.unbind(0)
        self.next_slots = self.next_buffer.unbind(0)
        self._input_keys = set(self.input_buffer.keys(True, True))
        self._lock = threading.Lock()
        self._clock = timeit("async_env_shared_exchange").start()
        self._leased_at: dict[int, float] = {}
        self._batch_count = 0
        self._batch_items = 0
        self._batch_capacity = 0
        self._partial_batch_count = 0
        self._ready_dwell_s = 0.0
        self._ready_dwell_count = 0
        self._action_dwell_s = 0.0
        self._action_dwell_count = 0
        self._consumer_busy_s = 0.0
        self._consumer_busy_started_s: float | None = None
        self._metric_started_s: float | None = None

    @staticmethod
    def _validate(fake_tensordicts: list[TensorDictBase]) -> None:
        if not fake_tensordicts:
            raise ValueError("Shared slot exchange requires at least one environment.")
        batch_size = fake_tensordicts[0].batch_size
        keys = set(fake_tensordicts[0].keys(True, True))
        for index, tensordict in enumerate(fake_tensordicts):
            if tensordict.batch_size != batch_size:
                raise ValueError(
                    "Shared slot exchange requires identical child batch sizes, "
                    f"got {batch_size} and {tensordict.batch_size} at index {index}."
                )
            if set(tensordict.keys(True, True)) != keys:
                raise ValueError(
                    "Shared slot exchange requires identical TensorDict keys across "
                    f"workers; worker {index} differs from worker 0."
                )
            # Leaves-only iteration silently skips non-tensor entries, so walk
            # every key explicitly: a NonTensorData leaf must reject the
            # exchange rather than end up in a shared slot.
            for key in tensordict.keys(True):
                value = tensordict.get(key)
                if isinstance(value, (NonTensorData, NonTensorStack)):
                    raise TypeError(
                        "Shared slot exchange only supports tensor leaves, "
                        f"got {type(value).__name__} at key {key!r}."
                    )
            for key, value in tensordict.items(True, True):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "Shared slot exchange only supports tensor leaves, "
                        f"got {type(value).__name__} at key {key!r}."
                    )
                if value.device.type != "cpu":
                    raise ValueError(
                        "Shared slot exchange requires CPU tensors, "
                        f"got device {value.device} at key {key!r}."
                    )
                reference = fake_tensordicts[0].get(key)
                if value.shape != reference.shape or value.dtype != reference.dtype:
                    raise ValueError(
                        "Shared slot exchange requires identical tensor schemas, "
                        f"but worker {index} has shape {value.shape} and dtype "
                        f"{value.dtype} at key {key!r}; worker 0 has shape "
                        f"{reference.shape} and dtype {reference.dtype}."
                    )

    def worker_slots(
        self, env_index: int
    ) -> tuple[TensorDictBase, TensorDictBase, TensorDictBase, timeit]:
        return (
            self.input_slots[env_index],
            self.result_slots[env_index],
            self.next_slots[env_index],
            self._clock,
        )

    def write_input(
        self,
        env_index: int,
        tensordict: TensorDictBase,
        *,
        record_action: bool = True,
    ) -> tuple:
        tensor_keys = []
        unsupported = []
        for key, value in tensordict.items(True, True):
            if key == "env_index":
                continue
            if not isinstance(value, torch.Tensor) or key not in self._input_keys:
                unsupported.append(key)
            else:
                tensor_keys.append(key)
        if unsupported:
            raise KeyError(
                "Shared slot exchange received keys absent from the fixed exchange "
                f"schema: {unsupported}. Use exchange='queue' for dynamic data."
            )
        self.input_slots[env_index].update_(
            tensordict.select(*tensor_keys, strict=True)
        )
        if record_action:
            self.record_action(env_index)
        return tuple(tensor_keys)

    @staticmethod
    def publish(
        slot: TensorDictBase, tensordict: TensorDictBase, clock: timeit
    ) -> tuple[tuple, float]:
        keys = tuple(tensordict.keys(True, True))
        slot.update_(tensordict)
        return keys, clock.elapsed()

    @staticmethod
    def publish_pair(
        result_slot: TensorDictBase,
        next_slot: TensorDictBase,
        result: TensorDictBase,
        next_result: TensorDictBase,
        clock: timeit,
    ) -> tuple[tuple, tuple, float]:
        result_keys = tuple(result.keys(True, True))
        next_keys = tuple(next_result.keys(True, True))
        result_slot.update_(result)
        next_slot.update_(next_result)
        return result_keys, next_keys, clock.elapsed()

    def receive(
        self,
        result_queue,
        min_get: int,
        max_get: int | None,
        timeout: float | None,
        *,
        track_action: bool,
    ) -> list[tuple]:
        descriptors = _receive_batch(result_queue, min_get, max_get, timeout)
        self._record_received(descriptors, max_get=max_get, track_action=track_action)
        return sorted(descriptors, key=lambda descriptor: descriptor[0])

    def receive_one(self, result_queue, *, track_action: bool) -> tuple:
        descriptor = result_queue.get()
        self._record_received([descriptor], max_get=1, track_action=track_action)
        return descriptor

    def _record_received(
        self,
        descriptors: list[tuple],
        *,
        max_get: int | None,
        track_action: bool,
    ) -> None:
        now = self._clock.elapsed()
        with self._lock:
            if self._metric_started_s is None:
                self._metric_started_s = now
            self._batch_count += 1
            self._batch_items += len(descriptors)
            self._batch_capacity += len(descriptors) if max_get is None else max_get
            if max_get is not None and len(descriptors) < max_get:
                self._partial_batch_count += 1
            if track_action and not self._leased_at:
                self._consumer_busy_started_s = now
            for descriptor in descriptors:
                env_index = descriptor[0]
                ready_s = descriptor[-1]
                if track_action:
                    self._leased_at[env_index] = now
                self._ready_dwell_s += now - ready_s
                self._ready_dwell_count += 1

    def read(self, descriptors: list[tuple], stack_func) -> TensorDictBase:
        results = []
        indices = []
        for env_index, keys, _ in descriptors:
            indices.append(env_index)
            results.append(self.result_slots[env_index].select(*keys, strict=True))
        result = stack_func(results)
        indices_data = NonTensorStack(*indices)
        while indices_data.batch_dims < result.batch_dims:
            indices_data = expand_as_right(indices_data, result)
        result.set("env_index", indices_data)
        return result

    def read_one(self, descriptor: tuple) -> TensorDictBase:
        env_index, keys, _ = descriptor
        return (
            self.result_slots[env_index]
            .select(*keys, strict=True)
            .clone()
            .set("env_index", NonTensorData(env_index))
        )

    def read_pair(
        self, descriptors: list[tuple], stack_func
    ) -> tuple[TensorDictBase, TensorDictBase]:
        results = []
        next_results = []
        indices = []
        for env_index, result_keys, next_keys, _ in descriptors:
            indices.append(env_index)
            results.append(
                self.result_slots[env_index].select(*result_keys, strict=True)
            )
            next_results.append(
                self.next_slots[env_index].select(*next_keys, strict=True)
            )
        result = stack_func(results)
        next_result = stack_func(next_results)
        indices_data = NonTensorStack(*indices)
        while indices_data.batch_dims < result.batch_dims:
            indices_data = expand_as_right(indices_data, result)
        result.set("env_index", indices_data)
        next_result.set("env_index", indices_data.clone())
        return result, next_result

    def read_pair_one(self, descriptor: tuple) -> tuple[TensorDictBase, TensorDictBase]:
        env_index, result_keys, next_keys, _ = descriptor
        index_data = NonTensorData(env_index)
        result = (
            self.result_slots[env_index]
            .select(*result_keys, strict=True)
            .clone()
            .set("env_index", index_data)
        )
        next_result = (
            self.next_slots[env_index]
            .select(*next_keys, strict=True)
            .clone()
            .set("env_index", index_data.clone())
        )
        return result, next_result

    def record_action(self, env_index: int) -> None:
        now = self._clock.elapsed()
        with self._lock:
            leased_at = self._leased_at.pop(env_index, None)
            if leased_at is not None:
                self._action_dwell_s += now - leased_at
                self._action_dwell_count += 1
                if not self._leased_at:
                    self._consumer_busy_s += now - self._consumer_busy_started_s
                    self._consumer_busy_started_s = None

    def stats(self, reset: bool = False) -> dict[str, float | int]:
        with self._lock:
            now = self._clock.elapsed()
            batch_count = self._batch_count
            busy_s = self._consumer_busy_s
            if self._consumer_busy_started_s is not None:
                busy_s += now - self._consumer_busy_started_s
            elapsed_s = (
                now - self._metric_started_s
                if self._metric_started_s is not None
                else 0
            )
            metrics = {
                "batches": batch_count,
                "items": self._batch_items,
                "avg_batch_size": self._batch_items / batch_count
                if batch_count
                else 0.0,
                "batch_fill_ratio": self._batch_items / self._batch_capacity
                if self._batch_capacity
                else 0.0,
                "partial_batch_fraction": self._partial_batch_count / batch_count
                if batch_count
                else 0.0,
                "avg_observation_to_batch_ms": self._ready_dwell_s
                / self._ready_dwell_count
                * 1e3
                if self._ready_dwell_count
                else 0.0,
                "avg_batch_to_action_ms": self._action_dwell_s
                / self._action_dwell_count
                * 1e3
                if self._action_dwell_count
                else 0.0,
                "consumer_busy_fraction": busy_s / elapsed_s if elapsed_s else 0.0,
            }
            if reset:
                self._batch_count = 0
                self._batch_items = 0
                self._batch_capacity = 0
                self._partial_batch_count = 0
                self._ready_dwell_s = 0.0
                self._ready_dwell_count = 0
                self._action_dwell_s = 0.0
                self._action_dwell_count = 0
                self._consumer_busy_s = 0.0
                self._metric_started_s = now if self._leased_at else None
                self._consumer_busy_started_s = now if self._leased_at else None
                self._leased_at = {env_index: now for env_index in self._leased_at}
        return metrics
