"""
TorchRL Collectors Deep Dive: Trajectory IDs, Partial Chunks, and Emission
=========================================================================

**Author**: `Jay Prajapati <https://github.com/coder-jayp>`_

.. _collector_trajectory_assembly:

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Why collectors return fixed-size chunks instead of full episodes
      * How ``split_trajectories()`` reassembles trajectories with padding and masking
      * How partial trajectories are buffered and emitted using ``_traj_ingest()`` and ``_traj_emit()``
      * The role of ``("collector", "traj_ids")`` and ``("collector", "mask")``
      * ``done``/``terminated`` boundary handling and ``as_nested=True`` vs padded outputs

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * TorchRL >= 0.5
      * gym or gymnasium
"""

# sphinx_gallery_start_ignore
import warnings
import time

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from tensordict import TensorDict
from torchrl.collectors.utils import split_trajectories, _traj_ingest, _traj_emit

######################################################################
# Overview
# --------
#
# TorchRL collectors return fixed-size batches of frames (`frames_per_batch`). 
# Because episodes can have very different lengths, a single batch often contains 
# pieces from several different trajectories.
#
# This creates a challenge when training recurrent or sequence-based policies, 
# which need clean, complete trajectories. TorchRL solves this with two main tools:
#
# - :func:`~torchrl.collectors.utils.split_trajectories` — the main public utility 
#   for reassembling and masking trajectories.
# - Internal buffering (`_traj_ingest` and `_traj_emit`) — used when you want 
#   full trajectories via `trajs_per_batch`.

######################################################################
# Section 1: Synthetic Collector Output
# -------------------------------------

print("=== Section 1: Synthetic Collector Output ===\n")

data = TensorDict(
    {
        "observation": torch.randn(15, 4),
        ("next", "observation"): torch.randn(15, 4),
        ("next", "reward"): torch.randn(15, 1),
        ("next", "done"): torch.tensor([0,0,0,1,0,0,0,0,1,0,0,0,0,0,1], dtype=torch.bool).unsqueeze(-1),
        ("collector", "traj_ids"): torch.tensor([0,0,0,0,1,1,1,1,1,2,2,2,2,2,2], dtype=torch.int64),
    },
    batch_size=[15],
)

print("Raw batch shape:", data.shape)
print("traj_ids:", data.get(("collector", "traj_ids")))
print("done:     ", data.get(("next", "done")).squeeze(-1))

######################################################################
# Section 2: split_trajectories()
# -------------------------------

print("\n" + "="*80)
print("=== Section 2: split_trajectories() ===\n")

split_data = split_trajectories(data, trajectory_key=("collector", "traj_ids"))

print("After split_trajectories:")
print(f"  Shape: {split_data.shape} → (num_trajectories, max_length, ...)")
print("\ntraj_ids (padded):\n", split_data.get(("collector", "traj_ids")))
print("mask:\n", split_data.get(("collector", "mask")))

######################################################################
# Edge Case: Trajectory Spanning Multiple Batches
# -----------------------------------------------

print("\nEdge Case: Trajectory spanning multiple collector iterations")

batch1 = TensorDict({
    "obs": torch.randn(8, 4),
    ("next", "done"): torch.zeros(8, 1, dtype=torch.bool),
    ("collector", "traj_ids"): torch.zeros(8, dtype=torch.int64)
}, [8])

batch2 = TensorDict({
    "obs": torch.randn(7, 4),
    ("next", "done"): torch.tensor([0,0,0,0,0,0,1]).unsqueeze(-1),
    ("collector", "traj_ids"): torch.zeros(7, dtype=torch.int64)
}, [7])

combined = torch.cat([batch1, batch2], dim=0)
split_combined = split_trajectories(combined, trajectory_key=("collector", "traj_ids"))

print(f"Combined shape : {combined.shape}")
print(f"After split    : {split_combined.shape}")
print("Mask (single long trajectory):", split_combined.get(("collector", "mask")).all().item())

######################################################################
# Section 3: Internal Buffering (_traj_ingest & _traj_emit)
# ---------------------------------------------------------

print("\n" + "="*80)
print("=== Section 3: Internal Buffering (_traj_ingest & _traj_emit) ===\n")

partial_trajs = {}
complete_trajs = []

b1 = TensorDict({
    "obs": torch.randn(5,4),
    ("next","done"): torch.zeros(5,1,dtype=torch.bool),
    ("collector","traj_ids"): torch.full((5,), 100, dtype=torch.int64)
}, [5])

b2 = TensorDict({
    "obs": torch.randn(6,4),
    ("next","done"): torch.tensor([0,0,0,0,0,1]).unsqueeze(-1),
    ("collector","traj_ids"): torch.full((6,), 100, dtype=torch.int64)
}, [6])

_traj_ingest(b1, partial_trajs, complete_trajs)
_traj_ingest(b2, partial_trajs, complete_trajs)

print(f"After second batch → Complete trajectories ready: {len(complete_trajs)}")

if complete_trajs:
    emitted = _traj_emit(complete_trajs, num_trajectories=1)
    print(f"Emitted full trajectory shape: {emitted.shape}")
    print("Mask added by _traj_emit:\n", emitted.get(("collector", "mask")))

######################################################################
# Section 4: Mask Semantics and Downstream Usage
# ----------------------------------------------

print("\n" + "="*80)
print("=== Section 4: Mask Semantics and Downstream Usage ===\n")

print("The ``('collector', 'mask')`` tells the model which timesteps are real data.")
print("This is especially important for recurrent models, where padding should not affect training.\n")

mask = emitted.get(("collector", "mask")) if 'emitted' in locals() else None
if mask is not None:
    print(f"Mask shape: {mask.shape}")

print("Recommended way to compute masked loss:")
print("    mask = mask.float()")
print("    loss = ((pred - target)**2 * mask).sum() / mask.sum()")

######################################################################
# Section 5: Padded vs Nested Outputs
# -----------------------------------

print("\n" + "="*80)
print("=== Section 5: Padded vs Nested Outputs ===\n")

t0 = time.time()
padded = split_trajectories(data, trajectory_key=("collector", "traj_ids"), as_nested=False)
t_padded = time.time() - t0

t0 = time.time()
nested = split_trajectories(data, trajectory_key=("collector", "traj_ids"), as_nested=True)
t_nested = time.time() - t0

print(f"Default padded   → shape: {str(padded.shape):20} time: {t_padded:.4f}s")
print(f"Nested output    → type : {type(nested).__name__:20} time: {t_nested:.4f}s")

print("\nRecommendation:")
print("- Use the default (zero-padded) for simplicity and maximum compatibility")
print("- Use `as_nested=True` when you have highly variable trajectory lengths and want to save memory")

######################################################################
# Troubleshooting
# ---------------

print("\n" + "="*80)
print("=== Troubleshooting ===\n")

print("Q: KeyError: ('collector', 'traj_ids')")
print("A: This usually happens when using custom data instead of a TorchRL collector.")
print("\nSolutions:")
print("1. Use SyncDataCollector (it adds traj_ids automatically)")
print("2. Use done_key instead:")
print("   split_trajectories(data, done_key=('next', 'done'))")
print("3. Manually add ('collector', 'traj_ids') before calling split_trajectories")

######################################################################
# Conclusion
# ----------

print("\n" + "="*80)
print("Conclusion")
print("="*80)

print("This tutorial covered how TorchRL handles trajectory assembly under the hood.")
print("You now understand why trajectories get split, how they are reassembled, and")
print("the important role of the mask when training recurrent policies.")

print("\nYou should now be comfortable:")
print("• Debugging trajectory-related issues")
print("• Using split_trajectories() with recurrent models")
print("• Choosing between padded and nested outputs")

print("\nThank you for reading!")