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
# Storing transitions and sampling trajectory slices
# ---------------------------------------------------
#
# In off-policy training the standard pattern is to store **flat
# transitions** in a :class:`~torchrl.data.ReplayBuffer` and let a
# :class:`~torchrl.data.SliceSampler` carve out contiguous
# sub-sequences that respect episode boundaries. The sampler uses
# ``("next", "done")`` to locate where episodes end, so you never get a
# slice that straddles two unrelated trajectories.
#
# This is the approach used in the
# :ref:`Recurrent DQN tutorial <RNN_tuto>`.
#
# .. seealso::
#   The :ref:`replay buffer tutorial <tuto_rb_traj>` covers trajectory
#   storage in more depth, including alternative samplers such as
#   :class:`~torchrl.data.PrioritizedSliceSampler` and
#   :class:`~torchrl.data.SliceSamplerWithoutReplacement`.

from torchrl.data import SliceSampler

rb = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10_000),
    sampler=SliceSampler(
        slice_len=16,
        end_key=("next", "done"),
    ),
    batch_size=32,
)

######################################################################
# We extend the buffer with the **flat** collector batch (``data``, shape
# ``(200,)``), not with the pre-assembled trajectory tensor.  The
# ``SliceSampler`` reads the ``("next", "done")`` flags in this flat
# storage to figure out where episodes start and stop.

rb.extend(data)

print(f"Buffer length after one batch: {len(rb)}")

sample = rb.sample()
print(sample)

######################################################################
# With ``batch_size=32`` and ``slice_len=16`` the sampler must draw
# exactly ``32 // 16 = 2`` contiguous trajectory slices per call:

traj_ids = sample["collector", "traj_ids"]
print(f"Unique trajectories in sample: {traj_ids.unique().numel()}")

######################################################################
# Each sampled batch contains contiguous slices of 16 steps drawn from
# the stored transitions. A typical training loop looks like this:
#
# .. code-block:: python
#
#     collector = SyncDataCollector(env, policy, frames_per_batch=200, ...)
#     rb = ReplayBuffer(
#         storage=LazyTensorStorage(max_size=100_000),
#         sampler=SliceSampler(slice_len=16, end_key=("next", "done")),
#         batch_size=64,
#     )
#
#     for batch in collector:
#         rb.extend(batch)
#         for _ in range(n_optim):
#             sample = rb.sample()
#             loss = loss_fn(sample)
#             loss.backward()
#             optim.step()

######################################################################
# Asynchronous collection with ``collector.start()``
# ---------------------------------------------------
#
# When a replay buffer is passed directly to the collector, you can
# decouple collection from training entirely using
# :meth:`~torchrl.collectors.Collector.start`. The collector runs in a
# background thread and writes flat transitions into the buffer
# continuously while your training loop samples from it.

import time

from torchrl.collectors import Collector

rb_async = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10_000),
    sampler=SliceSampler(
        slice_len=16,
        end_key=("next", "done"),
    ),
    shared=True,
)

collector_async = Collector(
    env,
    policy,
    replay_buffer=rb_async,
    frames_per_batch=200,
    total_frames=-1,
)

collector_async.start()

######################################################################
# The collector is now filling ``rb_async`` in the background with flat
# transitions. The ``SliceSampler`` will carve contiguous 16-step slices
# out of this flat storage, respecting episode boundaries.

for _ in range(10):
    time.sleep(0.1)
    if len(rb_async) > 0:
        break

print(f"Buffer length after background collection: {len(rb_async)}")

if len(rb_async) >= 16:
    sample = rb_async.sample(batch_size=32)
    print(sample)

######################################################################
# When you are done, shut the collector down:

collector_async.async_shutdown()

######################################################################
# This pattern is especially useful when environment stepping is slow
# (e.g. physics simulators or LLM inference): the training loop never
# idles waiting for new data, and the buffer is always fresh.

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
# * Passing a replay buffer and calling
#   :meth:`~torchrl.collectors.Collector.start` enables fully
#   asynchronous collection in a background thread.
#
# Useful next resources
# ~~~~~~~~~~~~~~~~~~~~~
#
# * :ref:`Get started with data collection <gs_storage>` — basic collector
#   and replay-buffer workflow.
# * :ref:`Recurrent DQN tutorial <RNN_tuto>` — training a recurrent
#   policy where per-episode data is essential.
# * `TorchRL documentation <https://pytorch.org/rl/>`_

# sphinx_gallery_start_ignore
collector.shutdown()
collector_trajs.shutdown()
# sphinx_gallery_end_ignore
