# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sample paired history windows from interleaved asynchronous trajectories.

Run this example from the repository root with::

    python examples/replay-buffers/geometric_trajectory_windows.py

Each item is written as soon as one of the simulated sources produces it. The
storage order therefore interleaves trajectories. ``GeometricTrajectoryWindowSampler``
uses the explicit trajectory and step keys to reconstruct temporal neighbours,
draw a geometric future offset, and sample eligible anchor steps uniformly.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from torchrl.data import (
    GeometricTrajectoryWindowSampler,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)

HISTORY = 3
BATCH_SIZE = 4

sampler = GeometricTrajectoryWindowSampler(
    history=HISTORY,
    continuation_probability=0.7,
    trajectory_key="trajectory_id",
    step_key="step_id",
)
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(64),
    sampler=sampler,
    batch_size=BATCH_SIZE,
    generator=torch.Generator().manual_seed(0),
)

# Simulate three asynchronous sources. Steps from each individual trajectory
# stay ordered, but their arrival order in the shared replay buffer is mixed.
trajectory_lengths = [7, 5, 8]
arrival_order = [
    (trajectory_id, step_id)
    for step_id in range(max(trajectory_lengths))
    for trajectory_id, length in enumerate(trajectory_lengths)
    if step_id < length
]
for trajectory_id, step_id in arrival_order:
    replay_buffer.add(
        TensorDict(
            {
                "trajectory_id": torch.tensor(trajectory_id),
                "step_id": torch.tensor(step_id),
                "observation": torch.tensor([float(trajectory_id), float(step_id)]),
                "action": torch.tensor([float(step_id)]),
            },
            batch_size=[],
        )
    )

# The raw sample is the union [t-h, ..., t+k]. Padded positions repeat step
# zero so the storage lookup remains valid; validity_mask identifies them.
window = replay_buffer.sample()
k = int(window["future_offset"][0, 0])
validity = window["validity_mask"]
observation = window["observation"]
observation = observation.masked_fill(~validity.unsqueeze(-1), 0)

# [o_{t-h}, ..., o_t]
current_observation_history = observation[:, : HISTORY + 1]
current_observation_mask = validity[:, : HISTORY + 1]

# [a_t, ..., a_{t+k}]
actions = window["action"][:, HISTORY : HISTORY + k + 1]

# [o_{t+k-h}, ..., o_{t+k}]
future_observation_history = observation[:, k : k + HISTORY + 1]
future_observation_mask = validity[:, k : k + HISTORY + 1]

assert current_observation_history.shape[:2] == (BATCH_SIZE, HISTORY + 1)
assert actions.shape[:2] == (BATCH_SIZE, k + 1)
assert future_observation_history.shape[:2] == (BATCH_SIZE, HISTORY + 1)

print(f"sampled future offset: {k}")
print("current observation history:\n", current_observation_history)
print("current observation mask:\n", current_observation_mask)
print("actions:\n", actions)
print("future observation history:\n", future_observation_history)
print("future observation mask:\n", future_observation_mask)
