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

import time

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from tensordict import TensorDict
from torchrl.collectors.utils import split_trajectories
from torchrl.data import LazyTensorStorage, ReplayBuffer

######################################################################
# Overview
# --------
#
# TorchRL collectors are designed to return batches of a fixed number of frames.
# Because real episodes have very different lengths, a single batch often contains
# pieces from multiple different trajectories.
#
# **In this tutorial you will learn:**
#
# - Why collectors return fixed-size chunks instead of full episodes
# - How ``split_trajectories()`` reassembles trajectories with padding and masking
# - How to request complete trajectories using ``trajs_per_batch``
# - How to store complete trajectories in a replay buffer
# - The meaning of ``("collector", "traj_ids")`` and ``("collector", "mask")``
# - Proper handling of ``done`` versus ``truncated``
# - When to use ``as_nested=True`` versus zero-padded tensors

######################################################################
# Section 1: Why trajectories get split
# -------------------------------------

print("=== Section 1: Why trajectories get split ===\n")

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
# Section 2: Reassembling trajectories
# ------------------------------------

print("\n" + "=" * 80)
print("=== Section 2: Reassembling trajectories ===\n")

split_data = split_trajectories(data, trajectory_key=("collector", "traj_ids"))

print("After split_trajectories:")
print(f"  Shape: {split_data.shape} → (num_trajectories, max_length, ...)")
print("\nPadded traj_ids:\n", split_data.get(("collector", "traj_ids")))
print("Mask (True = real data):\n", split_data.get(("collector", "mask")))

######################################################################
# Edge Case: Trajectory spanning multiple batches
# -----------------------------------------------

print("\nEdge Case: Trajectory spanning multiple collector iterations")

batch1 = TensorDict(
    {
        "obs": torch.randn(8, 4),
        ("next", "done"): torch.zeros(8, 1, dtype=torch.bool),
        ("collector", "traj_ids"): torch.zeros(8, dtype=torch.int64),
    },
    [8],
)

batch2 = TensorDict(
    {
        "obs": torch.randn(7, 4),
        ("next", "done"): torch.tensor([0, 0, 0, 0, 0, 0, 1]).unsqueeze(-1),
        ("collector", "traj_ids"): torch.zeros(7, dtype=torch.int64),
    },
    [7],
)

combined = torch.cat([batch1, batch2], dim=0)
split_combined = split_trajectories(combined, trajectory_key=("collector", "traj_ids"))

print(f"Combined shape : {combined.shape}")
print(f"After split    : {split_combined.shape}")
print(
    "Mask shows full trajectory is valid:",
    split_combined.get(("collector", "mask")).all().item(),
)

######################################################################
# Section 3: Getting complete trajectories
# ----------------------------------------

print("\n" + "=" * 80)
print("=== Section 3: Getting complete trajectories ===\n")

print("If you want full episodes instead of fixed-size frame batches,")
print("you can use the ``trajs_per_batch`` argument when creating the collector:")

print(
    """
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=1000,
    trajs_per_batch=8,          # ask for 8 complete trajectories
    ...
)
"""
)

print("The collector will automatically buffer partial trajectories and")
print("only return them once they are finished.")

######################################################################
# Section 4: Populating replay buffers
# ------------------------------------

print("\n" + "=" * 80)
print("=== Section 4: Populating replay buffers ===\n")

print(
    "Once you have complete trajectories, storing them in a replay buffer is straightforward:"
)

rb = ReplayBuffer(storage=LazyTensorStorage(max_size=100_000))

# rb.extend(split_data)   # this is the usual pattern

print("This approach keeps your buffer clean and avoids manual padding logic.")

######################################################################
# Section 5: done vs truncated and the mask
# -----------------------------------------

print("\n" + "=" * 80)
print("=== Section 5: done vs truncated and the mask ===\n")

print("The ``('collector', 'mask')`` indicates which timesteps are real data.")
print("You should always use it when training recurrent models so padding is ignored.")

print("\nAbout done and truncated:")
print("- 'done' means the episode ended naturally")
print("- 'truncated' means the episode was cut off early (e.g. by a time limit)")
print("If a trajectory is incomplete at the end of a batch, the last step")
print("is typically marked as truncated. ``split_trajectories()`` handles this")
print("correctly by respecting the mask.")

######################################################################
# Section 6: Padded vs nested tensors
# -----------------------------------

print("\n" + "=" * 80)
print("=== Section 6: Padded vs nested tensors ===\n")

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

print("\nRecommendation:")
print("- Use the default (zero-padded) for simplicity and broad compatibility")
print(
    "- Use as_nested=True when trajectory lengths vary a lot and you want to save memory"
)

######################################################################
# Conclusion
# ----------

print("\n" + "=" * 80)
print("Conclusion")
print("=" * 80)

print("In this tutorial we have covered how TorchRL collectors handle trajectories.")
print("You now understand why they return fixed-size chunks instead of full episodes,")
print("how to reassemble them using ``split_trajectories()``, how to request complete")
print(
    "trajectories with ``trajs_per_batch``, and how to store them cleanly in a replay buffer."
)

print("\nUseful next resources:")
print("- tutorials/sphinx-tutorials/getting-started-3.py (basic data collection)")
print("- tutorials/sphinx-tutorials/dqn_with_rnn.py (RNN policy example)")
print("- The main TorchRL documentation: https://pytorch.org/rl/")

print("\nThank you for reading!")
