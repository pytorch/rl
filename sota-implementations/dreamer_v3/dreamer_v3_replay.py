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


def replay_context_update(
    sample: TensorDictBase,
    state: torch.Tensor,
    belief: torch.Tensor,
) -> tuple[TensorDictBase, torch.Tensor, Mapping[NestedKey, torch.Tensor]]:
    """Build a deduplicated update for the context rows after a sampled step."""
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
    order = torch.arange(coordinates.shape[0], device=coordinates.device)
    for dimension in range(coordinates.shape[1] - 1, -1, -1):
        order = order[coordinates[order, dimension].argsort(stable=True)]
    ordered_coordinates = coordinates[order]
    keep_ordered = torch.ones(
        ordered_coordinates.shape[0], dtype=torch.bool, device=coordinates.device
    )
    keep_ordered[:-1] = (ordered_coordinates[:-1] != ordered_coordinates[1:]).any(-1)
    keep = order[keep_ordered]

    state = state.detach().float().reshape(-1, state.shape[-1])
    belief = belief.detach().float().reshape(-1, belief.shape[-1])
    return (
        handles[keep],
        generations[keep],
        {
            "state": state[keep.to(state.device)],
            "belief": belief[keep.to(belief.device)],
        },
    )
