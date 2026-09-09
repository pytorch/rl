# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Driver-step accounting and replay-context updates for DreamerV3."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from tensordict import NestedKey, TensorDictBase


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


def _last_rows_per_coordinate(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the last row holding each distinct coordinate, in coordinate order.

    Packs the non-negative integer columns into one key so a single stable sort
    orders the rows lexicographically with ties in their original order; the
    last row of every run of equal keys then wins. When the packed key would
    overflow ``int64``, the columns are sorted one at a time instead.
    """
    coordinates = coordinates.long()
    n_rows, n_columns = coordinates.shape
    radices = coordinates.amax(0) + 1
    if bool(radices.double().log2().sum() < 62):
        strides = torch.ones_like(radices)
        strides[:-1] = radices[1:].flip(0).cumprod(0).flip(0)
        key = (coordinates * strides).sum(-1)
        order = key.argsort(stable=True)
        ordered = key[order].unsqueeze(-1)
    else:
        order = torch.arange(n_rows, device=coordinates.device)
        for column in range(n_columns - 1, -1, -1):
            order = order[coordinates[order, column].argsort(stable=True)]
        ordered = coordinates[order]
    last = torch.ones(n_rows, dtype=torch.bool, device=coordinates.device)
    last[:-1] = (ordered[:-1] != ordered[1:]).any(-1)
    return order[last]


def _index_on(index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a host index to ``device`` without synchronizing the current stream.

    A pageable host-to-device copy waits for every kernel already enqueued,
    which after a learner step is the whole step; staging through pinned memory
    keeps the copy asynchronous.
    """
    if index.device == device:
        return index
    if device.type == "cuda" and index.device.type == "cpu":
        return index.pin_memory().to(device, non_blocking=True)
    return index.to(device)


def replay_context_update(
    sample: TensorDictBase,
    state: torch.Tensor,
    belief: torch.Tensor,
) -> tuple[TensorDictBase, torch.Tensor, Mapping[NestedKey, torch.Tensor]]:
    """Build a deduplicated update for the context rows after a sampled step.

    Nothing here synchronizes with the device holding ``state`` and
    ``belief``; the returned patch stays on that device and may still be in
    flight, so pass it to an asynchronous replay update.
    """
    if sample.ndim != 2:
        raise RuntimeError(
            "Expected a replay sample with shape [batch, time], got "
            f"{tuple(sample.shape)}."
        )
    learner_shape = torch.Size((sample.shape[0], sample.shape[1] - 1))
    if state.shape[:2] != learner_shape or belief.shape[:2] != learner_shape:
        raise RuntimeError(
            "Refreshed state and belief must match the learner sample batch and "
            "time dimensions."
        )

    handles = sample.get("index")[:, 1:].reshape(-1)
    generations = sample.get("index_generation")[:, 1:].reshape(-1)
    buffer_ids = handles.get("buffer_ids")
    local_indices = handles.get("index")
    coordinates = torch.cat(
        (buffer_ids.reshape(-1, 1), local_indices.reshape(buffer_ids.numel(), -1)),
        -1,
    )

    # Sampled slices may overlap. Keep the last value for each destination so
    # indexed writes have deterministic semantics on every device.
    keep = _last_rows_per_coordinate(coordinates)
    state = state.detach().reshape(-1, state.shape[-1])
    belief = belief.detach().reshape(-1, belief.shape[-1])
    return (
        handles[keep],
        generations[keep],
        {
            "state": state[_index_on(keep, state.device)].float(),
            "belief": belief[_index_on(keep, belief.device)].float(),
        },
    )
