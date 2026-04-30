# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end recurrent RL pipeline with SliceSampler auto-detection and padding.

The point of this example is that the data plumbing for sequence-based RL is
almost invisible — every key, every reset, every padding decision is handled
by a building block, and the training loop only does the work that actually
needs to happen there.

Pieces, in the order they appear:

* :func:`~torchrl.modules.get_env_transforms_from_module(policy)` returns the
  :class:`~torchrl.envs.transforms.InitTracker` and
  :class:`~torchrl.envs.transforms.TensorDictPrimer` the recurrent policy
  needs, in a single call. Append it to the env and ``is_init`` and the
  hidden-state key both materialise in every rollout.
* :class:`~torchrl.collectors.SyncDataCollector` is constructed with
  ``replay_buffer=rb`` so the collector populates the buffer itself — no
  ``rb.extend(...)`` in the training loop. ``collector.start()`` runs it in a
  background thread; ``rb.write_count`` is the source of truth for "how much
  data have we collected".
* :class:`~torchrl.data.replay_buffers.SliceSampler` is built *without* a
  ``traj_key`` argument: on the first sample it probes the storage and picks
  ``("collector", "traj_ids")`` automatically (the key the collector writes).
* ``strict_length=False, pad_output=True`` make every sample uniform
  ``[B * T]`` (short trajectories are padded instead of dropped) and write
  ``is_init=True`` at every slice start, OR-ed with whatever ``InitTracker``
  already wrote.  A ``("collector", "mask")`` flagging real vs padded
  timesteps is also emitted; mask-aware loss modules consume it on their own
  — the training loop never has to touch it.
* :func:`~torchrl.modules.set_recurrent_mode` ``("recurrent")`` lets the GRU
  consume the flat ``[B * T]`` sample directly. The RNN's existing
  ``is_init``-based split path (``_get_num_per_traj_init``) recovers the
  per-slice trajectory structure on its own and uses each slice's stored
  ``recurrent_state[0]`` as the initial hidden state. No reshape, no manual
  splitting, no per-slice book-keeping.

Run it::

    python examples/replay-buffers/recurrent_slice_sampler_pipeline.py
"""
from __future__ import annotations

import time
from collections import OrderedDict

import torch
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SliceSampler
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.modules import (
    get_env_transforms_from_module,
    GRUModule,
    MLP,
    QValueModule,
    set_recurrent_mode,
)

torch.manual_seed(0)

HIDDEN = 32
N_OBS = 4
N_ACT = 2
NUM_SLICES = 4
SLICE_LEN = 16
BATCH_SIZE = NUM_SLICES * SLICE_LEN

# ---------------------------------------------------------------------------
# Recurrent policy: linear embedding -> GRU -> linear -> Q-value head.
# ---------------------------------------------------------------------------
embed = Mod(nn.Linear(N_OBS, HIDDEN), in_keys=["observation"], out_keys=["embed"])
gru = GRUModule(
    input_size=HIDDEN,
    hidden_size=HIDDEN,
    num_layers=1,
    in_keys=["embed", "recurrent_state", "is_init"],
    out_keys=["embed_out", ("next", "recurrent_state")],
)
mlp = Mod(
    MLP(in_features=HIDDEN, out_features=N_ACT, num_cells=[]),
    in_keys=["embed_out"],
    out_keys=["action_value"],
)
qval = QValueModule(action_space="categorical")
policy = Seq(OrderedDict(embed=embed, gru=gru, mlp=mlp, qval=qval))

# ---------------------------------------------------------------------------
# Env: one call hooks up InitTracker + TensorDictPrimer for the GRU.
# ---------------------------------------------------------------------------
env = TransformedEnv(GymEnv("CartPole-v1"), get_env_transforms_from_module(policy))

# ---------------------------------------------------------------------------
# Replay buffer: no traj_key — the sampler auto-detects it on first sample.
# ---------------------------------------------------------------------------
rb = TensorDictReplayBuffer(
    storage=LazyTensorStorage(2_000),
    sampler=SliceSampler(
        num_slices=NUM_SLICES,
        strict_length=False,
        pad_output=True,
    ),
    batch_size=BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Collector: writes directly into the buffer in a background thread.
# total_frames=-1 means "run forever" — we stop it ourselves when we've
# trained enough.
# ---------------------------------------------------------------------------
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=200,
    total_frames=-1,
    replay_buffer=rb,
)
collector.start()

# Wait until the buffer has enough data to draw a first batch.
while rb.write_count < BATCH_SIZE:
    time.sleep(0.05)

# ---------------------------------------------------------------------------
# Training loop. Sample, reshape to [B, T] for the GRU, that's it.
# ---------------------------------------------------------------------------
N_TRAINING_STEPS = 5
for step in range(N_TRAINING_STEPS):
    sample = rb.sample()
    if step == 0:
        # First sample triggered the auto-detect — confirm what was picked.
        print("Auto-detected traj_key:", rb.sampler.traj_key)

    # The flat [B*T] sample carries is_init=True at every slice start, so the
    # GRU in recurrent_mode splits the sequence into independent slices on its
    # own and returns action_value shaped exactly like the input.
    with set_recurrent_mode("recurrent"):
        out = policy(sample)
    assert out["action_value"].shape == torch.Size([NUM_SLICES * SLICE_LEN, N_ACT])
    print(f"step {step}: write_count={rb.write_count}")

collector.async_shutdown()
print("\nDone.")
