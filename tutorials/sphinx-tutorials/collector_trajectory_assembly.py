"""
TorchRL Collectors Deep Dive: Trajectory IDs, Partial Chunks, and Emission
=========================================================================

**Author**: `Jay Prajapati <https://github.com/coder-jayp>`_

.. _collector_trajectory_assembly:

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Why collectors return fixed-size chunks instead of full episodes
      * How ``split_trajectories()`` reassembles trajectories with padding and masking
      * How to request complete trajectories using ``trajs_per_batch``
      * How to store complete trajectories in a replay buffer
      * The meaning of ``("collector", "traj_ids")`` and ``("collector", "mask")``
      * Proper handling of ``done`` versus ``truncated``
      * When to use ``as_nested=True`` versus zero-padded tensors

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * TorchRL >= 0.5
      * gym or gymnasium
"""

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import time

import torch
from tensordict import TensorDict
from torchrl.collectors.utils import split_trajectories
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules import RandomPolicy

######################################################################
# Overview
# --------
#
# TorchRL collectors are designed to return batches of a fixed number of frames.
# Because real episodes have very different lengths, a single batch often contains
# pieces from multiple different trajectories.
#
# This can be inconvenient when training recurrent policies or when you want
# clean, complete episodes in your replay buffer.
#
# In this tutorial you will learn how TorchRL handles trajectory assembly,
# how to request full trajectories, and how to use them effectively with
# replay buffers.

######################################################################
# Collecting data from a real environment
# ---------------------------------------

env = GymEnv("CartPole-v1")
policy = RandomPolicy(env.action_spec)

######################################################################
# Raw collector output
# --------------------

data = TensorDict(
    {
        "observation": torch.randn(15, 4),
        ("next", "observation"): torch.randn(15, 4),
        ("next", "reward"): torch.randn(15, 1),
        ("next", "done"): torch.tensor(
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=torch.bool
        ).unsqueeze(-1),
        ("collector", "traj_ids"): torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.int64
        ),
    },
    batch_size=[15],
)

print("Raw collector batch shape:", data.shape)
print("Trajectory IDs:", data.get(("collector", "traj_ids")))
print("Done signals: ", data.get(("next", "done")).squeeze(-1))

######################################################################
# The problem without trajectory assembly
# ---------------------------------------

# Without assembly, a batch can contain fragments from several trajectories.
# This makes it difficult to train recurrent policies, which expect continuous
# sequences from the same episode.

######################################################################
# Reassembling trajectories with split_trajectories
# -------------------------------------------------

split_data = split_trajectories(data, trajectory_key=("collector", "traj_ids"))

print("After split_trajectories:")
print(f"  Shape: {split_data.shape} → (num_trajectories, max_length, ...)")
print("\nPadded traj_ids:\n", split_data.get(("collector", "traj_ids")))
print("Mask (True = real data):\n", split_data.get(("collector", "mask")))

######################################################################
# Getting complete trajectories
# -----------------------------

# If you want full episodes instead of fixed-size frame batches, you can use the
# ``trajs_per_batch`` argument when creating the collector.

######################################################################
# Using complete trajectories with replay buffers
# -----------------------------------------------

# Once you have complete trajectories, storing them in a replay buffer is
# straightforward:

rb = ReplayBuffer(storage=LazyTensorStorage(max_size=100_000))

# rb.extend(split_data)   # this is the usual pattern

# This approach keeps your buffer clean and avoids manual padding logic.

######################################################################
# done vs truncated and the mask
# ------------------------------

# The ``('collector', 'mask')`` indicates which timesteps are real data.
# You should always use it when training recurrent models so padding is ignored.

# About done and truncated:
# - 'done' means the episode ended naturally
# - 'truncated' means the episode was cut off early (e.g. by a time limit)
# If a trajectory is incomplete at the end of a batch, the last step
# is typically marked as truncated. ``split_trajectories()`` handles this
# correctly by respecting the mask.

######################################################################
# Padded vs nested tensors
# ------------------------

t0 = time.time()
padded = split_trajectories(
    data, trajectory_key=("collector", "traj_ids"), as_nested=False
)
t_padded = time.time() - t0

t0 = time.time()
nested = split_trajectories(
    data, trajectory_key=("collector", "traj_ids"), as_nested=True
)
t_nested = time.time() - t0

print(f"Default padded   → shape: {str(padded.shape):20} time: {t_padded:.4f}s")
print(f"Nested output    → type : {type(nested).__name__:20} time: {t_nested:.4f}s")

######################################################################
# Recommendation
# --------------

# - Use the default (zero-padded) for simplicity and broad compatibility
# - Use as_nested=True when trajectory lengths vary a lot and you want to save memory

######################################################################
# Conclusion
# ----------

# In this tutorial we have covered how TorchRL collectors handle trajectories.
# You now understand why they return fixed-size chunks instead of full episodes,
# how to reassemble them using ``split_trajectories()``, how to request complete
# trajectories with ``trajs_per_batch``, and how to store them cleanly in a replay buffer.

# Useful next resources:
# - tutorials/sphinx-tutorials/getting-started-3.py (basic data collection)
# - tutorials/sphinx-tutorials/dqn_with_rnn.py (RNN policy example)
# - The main TorchRL documentation: https://pytorch.org/rl/

# Thank you for reading!
