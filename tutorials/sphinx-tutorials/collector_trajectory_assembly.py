"""
Collectors Deep Dive: Trajectory Assembly
==========================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _collector_trajectory_assembly:

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * Why collectors return fixed-size batches that mix multiple trajectories
      * How ``split_trajectories()`` reassembles them into padded, per-episode tensors
      * What ``("collector", "traj_ids")`` and ``("collector", "mask")`` mean
      * How ``done`` and ``truncated`` interact with trajectory splitting
      * When to use ``as_nested=True`` for memory-efficient ragged batches
      * How to request complete trajectories with ``trajs_per_batch``
      * How to store complete trajectories in a replay buffer

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * `TorchRL <https://github.com/pytorch/rl>`_ and
        `gymnasium <https://gymnasium.farama.org>`_ installed
      * Familiarity with :class:`~torchrl.collectors.SyncDataCollector`
        (see :ref:`the data-collection tutorial <gs_storage_collector>`)
"""

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules import RandomPolicy

torch.manual_seed(0)

######################################################################
# Why collectors return fixed-size chunks
# ---------------------------------------
#
# In reinforcement learning, episodes can have wildly different lengths.
# A CartPole episode may last 10 steps or 500, depending on the policy.
# To keep training loops predictable, TorchRL collectors always return
# batches of exactly ``frames_per_batch`` transitions, regardless of how
# many episodes those transitions span.
#
# This means a single batch will typically contain **fragments of
# multiple trajectories** stitched together. Let's see this in practice.

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

for data in collector:
    print(data)
    break

######################################################################
# The batch has exactly 200 transitions. Let's inspect its trajectory
# IDs — each integer labels which episode a given transition belongs to:

print(data["collector", "traj_ids"])

######################################################################
# Multiple trajectory IDs appear because several short episodes were
# packed into a single 200-frame batch. The ``("next", "done")`` key
# marks where each episode ends:

print(data["next", "done"].squeeze(-1))

######################################################################
# We can count how many complete episodes fell within this batch:

n_episodes = data["next", "done"].sum().item()
print(f"This batch of {data.shape[0]} frames contains {n_episodes} episodes.")

######################################################################
# Reassembling trajectories with ``split_trajectories``
# -----------------------------------------------------
#
# For many algorithms (especially those involving recurrent networks or
# episode-level returns), you need data organized **per episode**, not
# as a flat interleaved stream. :func:`~torchrl.collectors.utils.split_trajectories`
# takes a flat batch with ``("collector", "traj_ids")`` and returns a
# zero-padded ``TensorDict`` of shape ``(num_trajectories, max_length)``.

split_data = split_trajectories(data)

print(split_data)
print(f"Shape: {split_data.shape}  →  (num_trajectories, max_episode_length)")

######################################################################
# Because episodes have different lengths, shorter ones are padded with
# zeros. The ``("collector", "mask")`` key tells you which time-steps
# contain real data (``True``) and which are padding (``False``):

print(split_data["collector", "mask"])

######################################################################
# When computing losses on this padded tensor, **always multiply by the
# mask** (or index with it) so that padding does not leak into your
# gradients. This is especially important for recurrent models.

######################################################################
# ``done`` vs ``truncated`` and the mask
# ---------------------------------------
#
# TorchRL distinguishes two flavours of episode termination:
#
# * ``("next", "done")`` is ``True`` whenever an episode ends, for any
#   reason.
# * ``("next", "truncated")`` is ``True`` only when the episode was cut
#   short by an external limit (a time limit, or the collector running
#   out of frames before the environment signalled a natural end).
#
# When a trajectory is still in-flight at the edge of a batch, its last
# step will be ``truncated=True, done=True``. ``split_trajectories``
# handles this correctly: the mask covers exactly the valid steps,
# and the ``done`` / ``truncated`` flags are preserved so that you can
# treat natural terminations and artificial truncations differently in
# your value-function bootstrap.

print("done shape:     ", split_data["next", "done"].shape)
print("truncated shape:", split_data["next", "truncated"].shape)

######################################################################
# Padded vs nested tensors
# ------------------------
#
# By default ``split_trajectories`` zero-pads to the length of the
# longest trajectory. If your episodes vary a lot in length this wastes
# memory. Passing ``as_nested=True`` returns a
# :class:`~tensordict.TensorDict` backed by nested tensors instead:

padded = split_trajectories(data, as_nested=False)
nested = split_trajectories(data, as_nested=True)

print(f"Padded shape : {padded.shape}")
print(f"Nested result: {type(nested).__name__}, batch_size={nested.batch_size}")

######################################################################
# **Recommendation:** use the default (padded) for simplicity and broad
# compatibility. Switch to ``as_nested=True`` when episode lengths are
# highly variable and memory is a concern.

######################################################################
# Getting complete trajectories with ``trajs_per_batch``
# -------------------------------------------------------
#
# Sometimes you want the collector itself to hand you **complete
# episodes** rather than fixed-frame chunks. The ``trajs_per_batch``
# argument tells the collector to buffer partial trajectories internally
# and yield only once it has accumulated the requested number of
# finished episodes.

collector_trajs = SyncDataCollector(
    env,
    policy,
    frames_per_batch=200,
    total_frames=-1,
    trajs_per_batch=5,
)

for traj_data in collector_trajs:
    print(traj_data)
    break
print(f"Shape: {traj_data.shape}  →  (trajs_per_batch, max_episode_length)")

######################################################################
# Every row is a **complete** episode. The mask confirms this — each
# trajectory starts at step 0 and runs until the episode's natural (or
# truncated) end:

print(traj_data["collector", "mask"])

######################################################################
# Storing trajectories in a replay buffer
# ----------------------------------------
#
# Once you have per-episode data (from ``split_trajectories`` or
# ``trajs_per_batch``), storing it in a
# :class:`~torchrl.data.ReplayBuffer` is straightforward. Each episode
# becomes a single entry in the buffer, and sampling returns
# ready-to-use episode tensors.

rb = ReplayBuffer(storage=LazyTensorStorage(max_size=10_000))
rb.extend(traj_data)

print(f"Buffer length after one batch: {len(rb)}")

sample = rb.sample(batch_size=3)
print(sample)

######################################################################
# A typical training loop combines all of the above:
#
# .. code-block:: python
#
#     collector = SyncDataCollector(env, policy, ..., trajs_per_batch=32)
#     rb = ReplayBuffer(storage=LazyTensorStorage(max_size=100_000))
#
#     for traj_batch in collector:
#         rb.extend(traj_batch)
#         for _ in range(n_optim):
#             sample = rb.sample(batch_size=16)
#             loss = loss_fn(sample)
#             loss.backward()
#             optim.step()

######################################################################
# Conclusion
# ----------
#
# In this tutorial we covered how TorchRL collectors handle trajectories:
#
# * Collectors return **fixed-size batches** that interleave fragments of
#   multiple episodes.
# * :func:`~torchrl.collectors.utils.split_trajectories` reassembles them
#   into a ``(num_trajectories, max_length)`` padded tensor with a mask.
# * ``done`` marks any episode end; ``truncated`` flags artificial
#   cut-offs. The mask covers valid time-steps only.
# * ``as_nested=True`` gives memory-efficient ragged tensors.
# * ``trajs_per_batch`` makes the collector yield complete episodes
#   directly.
# * Complete episodes slot naturally into a
#   :class:`~torchrl.data.ReplayBuffer`.
#
# Useful next resources
# ~~~~~~~~~~~~~~~~~~~~~
#
# * :ref:`Get started with data collection <gs_storage>` — basic collector
#   and replay-buffer workflow.
# * :ref:`Recurrent DQN tutorial <coding_dqn_rnn>` — training a recurrent
#   policy where per-episode data is essential.
# * `TorchRL documentation <https://pytorch.org/rl/>`_

# sphinx_gallery_start_ignore
collector.shutdown()
collector_trajs.shutdown()
# sphinx_gallery_end_ignore
