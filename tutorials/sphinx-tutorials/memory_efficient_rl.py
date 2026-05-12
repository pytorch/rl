"""
Memory-Efficient RL Training
============================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _memory_efficient_rl:

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      * The cost of keeping ``("next", obs)`` in rollouts and replay buffers
      * Why TorchRL keeps it by default (bootstrap targets and MultiStep)
      * Halving the observation footprint with
        :class:`~torchrl.collectors.SyncDataCollector` ``compact_obs=True``
      * Rebuilding ``("next", obs)`` on the consumer side with
        :class:`~torchrl.envs.transforms.NextStateReconstructor`
      * Why the resulting ``NaN`` at trajectory ends does not crash GAE / TD
      * When *not* to take this path (MultiStep DQN, truncated transitions)
      * Other knobs: :class:`~torchrl.data.LazyMemmapStorage`,
        :class:`~torchrl.data.SliceSampler`, and the new padding-free RNN
        backends

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * `TorchRL <https://github.com/pytorch/rl>`_ and
        `gymnasium <https://gymnasium.farama.org>`_ installed
      * Familiarity with :class:`~torchrl.collectors.Collector` and
        :class:`~torchrl.data.ReplayBuffer`
        (see :ref:`the data-collection tutorial <gs_storage>` and
        :ref:`the replay-buffer tutorial <rb_tuto>`)
"""

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import tempfile

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import GymEnv
from torchrl.envs.transforms import NextStateReconstructor
from torchrl.modules import RandomPolicy
from torchrl.objectives.value import GAE

torch.manual_seed(0)

######################################################################
# Where the memory goes
# ---------------------
#
# A typical RL rollout returns a tensordict with both the current
# observation (``"observation"``) and the next observation
# (``("next", "observation")``). The two overlap by ``T - 1`` entries
# within a trajectory of length ``T``: ``data["observation"][1:]`` is
# bit-for-bit equal to ``data[("next", "observation")][:-1]``. We are
# storing roughly *two copies of every observation*.
#
# Let's measure this directly on a tiny CartPole rollout.

env_maker = lambda: GymEnv("CartPole-v1")  # noqa: E731
collector = SyncDataCollector(
    create_env_fn=env_maker,
    frames_per_batch=200,
    total_frames=200,
)

data = next(iter(collector))
collector.shutdown()

total_bytes = data.bytes()
obs_bytes = data.get("observation").numel() * data.get("observation").element_size()
next_obs_bytes = (
    data.get(("next", "observation")).numel()
    * data.get(("next", "observation")).element_size()
)

print(f"Full rollout:               {total_bytes:>6d} B")
print(f"  observation share:        {obs_bytes:>6d} B")
print(f"  ('next','observation'):   {next_obs_bytes:>6d} B")
print(
    f"  duplicated obs:           "
    f"{int(next_obs_bytes * (data.shape[-1] - 1) / data.shape[-1]):>6d} B "
    f"(≈ (T-1)/T of the next-obs share)"
)

######################################################################
# CartPole's 4-dim float observation is small, but the same pattern
# applies to vision policies (84×84×3 frames), critic features
# (hundreds of dimensions), or LLM hidden states (thousands).
# Multiplied by a 10⁶-step replay buffer, the duplication is the
# difference between fitting on a single GPU and not.

######################################################################
# Why we keep ``("next", obs)`` by default
# ----------------------------------------
#
# Before we drop anything we should be explicit about what the
# duplicated tensor is worth. There are two main consumers:
#
# 1. **Bootstrap target at trajectory ends.** TD(0), TD(λ) and GAE all
#    compute ``target = r_t + γ (1 - done_t) V(next_obs_t)``. On *every*
#    transition we need the canonical next observation — including the
#    very last frame of a *truncated* episode, where the bootstrap is
#    still applied because the trajectory was artificially cut.
# 2. **MultiStep n-step fallback.**
#    :class:`~torchrl.envs.transforms.MultiStepTransform` places
#    ``data[t + n]`` into ``data[("next", obs)][t]``. For the last
#    ``n - 1`` frames of every trajectory it falls back to
#    ``data[t + n - 1]``, ``data[t + n - 2]``, ..., down to ``data[t + 1]``
#    — and it can only do that because the genuine
#    ``("next", obs)`` lives in storage.
#
# Both of these consumers need *information that is not present in
# ``data["observation"][t + 1]``* once the trajectory ends. That is why
# the default is to keep both copies.

######################################################################
# Knob 1 — drop the duplicates at the collector
# ---------------------------------------------
#
# If your loss does not depend on a *bootable* terminal next-obs
# (vanilla policy-gradient losses, on-policy GAE with terminated-only
# transitions, …), the trade-off flips. The
# :class:`~torchrl.collectors.SyncDataCollector` exposes a
# ``compact_obs=True`` flag that drops every observation / state key
# under ``("next", ...)`` *before* stacking per-step data.
# ``("next", "reward")``, ``("next", "done")`` and
# ``("next", "truncated")`` are preserved — they cannot be reconstructed
# from the root keys. The flag works for ``MultiSyncCollector`` and
# ``MultiAsyncCollector`` too.

compact_collector = SyncDataCollector(
    create_env_fn=env_maker,
    frames_per_batch=200,
    total_frames=200,
    compact_obs=True,
)
compact_data = next(iter(compact_collector))
compact_collector.shutdown()

print(f"Default rollout bytes:    {data.bytes():>6d}")
print(f"compact_obs=True bytes:   {compact_data.bytes():>6d}")
print(
    f"saving:                   {data.bytes() - compact_data.bytes():>6d} B  "
    f"({100 * (data.bytes() - compact_data.bytes()) / data.bytes():.1f} %)"
)
print()
print("Keys dropped from the rollout:")
print(set(data.keys(True, True)) - set(compact_data.keys(True, True)))

######################################################################
# The collector queries ``env._observation_keys_step_mdp`` and
# ``env._state_keys_step_mdp`` to discover *which* keys are duplicated,
# so nested obs (``("agents", "pos")``, dict-shaped vision obs, …) are
# handled automatically.

######################################################################
# Knob 2 — rehydrate at sampling time
# -----------------------------------
#
# Many losses *do* read ``("next", obs)`` (notably GAE / TD). The
# consumer-side counterpart of ``compact_obs`` is
# :class:`~torchrl.envs.transforms.NextStateReconstructor`. The rule is
# simple: for each sampled position ``i``, the canonical next is
# position ``i + 1`` of the same batch *iff* it belongs to the same
# trajectory and the trajectory hasn't ended; otherwise the slot is
# filled with ``NaN`` (configurable).
#
# "Same trajectory" is decided from a trajectory id (default
# ``("collector", "traj_ids")``, which
# :class:`~torchrl.collectors.SyncDataCollector` populates by default)
# and ``("next", "done")``. The transform is sampler-agnostic — it does
# not require :class:`~torchrl.data.SliceSampler` — but
# :class:`~torchrl.data.SliceSampler` is the natural pairing because
# adjacent positions inside a slice are also adjacent in trajectory
# time.

rb = ReplayBuffer(
    storage=LazyTensorStorage(200),
    sampler=SliceSampler(slice_len=20, traj_key=("collector", "traj_ids")),
    transform=NextStateReconstructor(),
    batch_size=40,
)
rb.extend(compact_data)
sample = rb.sample()

# ``("next", "observation")`` is back in the sample, even though it was
# absent from the storage.
print("sample keys:", sorted(str(k) for k in sample.keys(True, True))[:6])
print("any NaN in ('next', observation')?", torch.isnan(sample[("next", "observation")]).any().item())

######################################################################
# The NaN entries land exactly where the *real* next observation is no
# longer reconstructable — slice boundaries that coincide with
# trajectory ends. We can see them by looking at the rows where the
# trajectory id changes (or where the trajectory ended):

traj = sample[("collector", "traj_ids")]
done = sample[("next", "done")].squeeze(-1)
boundary = torch.cat([(traj[1:] != traj[:-1]), torch.tensor([True])]) | done
print(
    "rows with NaN next-obs:                ",
    torch.isnan(sample[("next", "observation")]).any(-1).nonzero(as_tuple=True)[0].tolist(),
)
print(
    "rows flagged as trajectory boundaries: ",
    boundary.nonzero(as_tuple=True)[0].tolist(),
)

######################################################################
# Knob 2.5 — value-estimator NaN safety
# -------------------------------------
#
# ``NaN`` propagating through GAE / TD would be catastrophic:
# ``V(NaN) = NaN`` and the canonical ``(1 - done) * V_next`` masking
# does *not* save us because IEEE 754 has ``0 * NaN = NaN``. The
# value-estimator pipeline therefore sanitises the input before calling
# the value network — see
# :meth:`~torchrl.objectives.value.ValueEstimatorBase._sanitize_next_obs_nan`
# — substituting the corresponding root observation at every NaN
# position. At *terminated* steps the substitute is masked out
# downstream by ``(1 - done)``; at *truncated-only* steps it acts as
# an approximate bootstrap ``V(obs[t]) ≈ V(real_next_obs)``.
#
# The upshot: ``compact_obs`` + ``NextStateReconstructor`` + GAE / TD
# is numerically safe out of the box.

# Tiny value net for illustration: V(s) = ‖s‖₂.
value_net = TensorDictModule(
    lambda x: x.pow(2).sum(-1, keepdim=True).sqrt(),
    in_keys=["observation"],
    out_keys=["state_value"],
)
gae = GAE(gamma=0.99, lmbda=0.95, value_network=value_net, shifted=False)

# Reshape the flat slice batch into (num_slices, slice_len) for GAE.
gae_in = sample.reshape(-1, 20)
out = gae(gae_in.clone())
print("GAE advantage finite everywhere?", torch.isfinite(out["advantage"]).all().item())
print("any -inf or +inf?", torch.isinf(out["advantage"]).any().item())

######################################################################
# When *not* to rehydrate
# -----------------------
#
# Two situations call for keeping the canonical ``("next", obs)``:
#
# 1. :class:`~torchrl.envs.transforms.MultiStepTransform`. The n-step
#    next observation is the *original* ``data[t + n]``, not
#    ``data[t + 1]``, and the in-trajectory fallback at the last
#    ``n - 1`` frames depends on having every ``data[t + k]`` written
#    to ``("next", obs)`` at extend time. Rehydration cannot
#    reconstruct that.
# 2. Losses that bootstrap on *truncated* transitions and need the
#    real next observation, not the
#    ``V(obs[t]) ≈ V(real_next_obs)`` approximation that
#    :meth:`~torchrl.objectives.value.ValueEstimatorBase._sanitize_next_obs_nan`
#    falls back to. The approximation is fine for many tasks (it's
#    consistent and finite) but it *is* an approximation.
#
# A second, smaller knob in the same area is the
# ``shifted=True`` mode of the value estimators
# (:class:`~torchrl.objectives.value.GAE`,
# :class:`~torchrl.objectives.value.TD0Estimator`,
# :class:`~torchrl.objectives.value.TDLambdaEstimator`, …). ``shifted``
# folds the two value-net forward passes (one on root obs, one on
# ``("next", obs)``) into a single pass on a length-``T + 1``
# interleaved sequence. It saves roughly half of the value-net
# compute, but requires the same parameters for root and next — no
# distinct target network — and consumes the canonical
# ``("next", obs)`` at trajectory ends, which means it inherits the
# same approximation as the compact path at truncated steps.

######################################################################
# Knob 3 — memory-mapped replay buffer storage
# --------------------------------------------
#
# Even after halving the observation footprint, the replay buffer can
# easily outgrow VRAM (and RAM). :class:`~torchrl.data.LazyMemmapStorage`
# is a drop-in replacement for :class:`~torchrl.data.LazyTensorStorage`
# that allocates each leaf tensor as a memory-mapped file on disk.
# Reading is fast (the OS page cache keeps hot pages in memory), and
# the buffer can be larger than physical memory.

with tempfile.TemporaryDirectory() as tmpdir:
    rb_mmap = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=1_000, scratch_dir=tmpdir),
        sampler=SliceSampler(slice_len=20, traj_key=("collector", "traj_ids")),
        transform=NextStateReconstructor(),
        batch_size=40,
    )
    rb_mmap.extend(compact_data)
    mmap_sample = rb_mmap.sample()
    print("memmap sample shape:", mmap_sample.shape)

######################################################################
# The data went through disk, but the public API is identical to the
# in-memory case. See the :ref:`replay buffer tutorial <rb_tuto>` for
# more on storage choices.

######################################################################
# Knob 4 — sequence training without padding
# ------------------------------------------
#
# RNN-based policies and value heads classically train on
# zero-padded ``(batch, max_T, feature)`` tensors, with a mask telling
# the loss which timesteps are real. Padding wastes both memory (every
# trajectory pays for the longest one) and compute (the RNN unrolls
# through the padding tokens).
#
# Two recent additions sidestep both:
#
# * :class:`~torchrl.data.SliceSampler` returns *contiguous* slices of
#   pre-specified length. There is no padding; every entry is a real
#   transition. The trajectory-id key lets the sampler align slices to
#   trajectory boundaries.
# * :class:`~torchrl.modules.LSTMModule` and
#   :class:`~torchrl.modules.GRUModule` accept a
#   ``recurrent_backend`` argument with three non-default values:
#
#     * ``"scan"`` — built on
#       ``torch._higher_order_ops.scan`` (PyTorch ≥ 2.6). Resets the
#       hidden state at each ``is_init=True`` frame inside the kernel,
#       so trajectories of different lengths can be concatenated end
#       to end with no padding.
#     * ``"triton"`` — same idea, implemented as a custom Triton
#       kernel (requires CUDA and ``triton >= 2.2``). Fastest of the
#       three on GPU.
#     * ``"auto"`` — picks ``"scan"`` under ``torch.compile`` and
#       falls back to the classical ``"pad"`` path otherwise.
#
# A typical configuration looks like this:
#
# .. code-block:: python
#
#     from torchrl.modules import GRUModule
#
#     rnn = GRUModule(
#         input_size=64,
#         hidden_size=128,
#         in_keys=["obs", "rhs"],
#         out_keys=["features", ("next", "rhs")],
#         recurrent_backend="scan",  # or "triton" on CUDA
#         default_recurrent_mode=True,
#     )
#
# Combined with :class:`~torchrl.data.SliceSampler`, the trained
# sequence is exactly the concatenation of the slices — no padding
# allocated, no hidden states wasted on zero tokens.

######################################################################
# Putting it together
# -------------------
#
# A memory-conscious value-based pipeline (off-policy actor / critic,
# GAE bootstraps, slice-sampled sequence training):
#
# .. code-block:: python
#
#     collector = SyncDataCollector(
#         create_env_fn=env_maker,
#         policy=policy,
#         frames_per_batch=1024,
#         total_frames=1_000_000,
#         compact_obs=True,                       # halve obs memory
#     )
#     rb = ReplayBuffer(
#         storage=LazyMemmapStorage(1_000_000),   # spill to disk
#         sampler=SliceSampler(                   # no padding
#             slice_len=64,
#             traj_key=("collector", "traj_ids"),
#         ),
#         transform=NextStateReconstructor(),     # rehydrate ('next', obs)
#         batch_size=8 * 64,
#     )
#     loss = ClipPPOLoss(actor=actor, critic=critic)
#     advantage = GAE(                            # NaN-safe at boundaries
#         gamma=0.99, lmbda=0.95,
#         value_network=critic, shifted=True,     # one V-net call per step
#     )
#
# Every knob is independent — adopt them à la carte depending on what
# your loss needs. The ones that *interact* are highlighted in the
# *When not to rehydrate* section above.

######################################################################
# Conclusion
# ----------
#
# * ``("next", obs)`` is a duplicate of ``obs[t + 1]`` *within* a
#   trajectory, but it is *not* a duplicate at trajectory boundaries.
#   That is why TorchRL keeps it by default.
# * :class:`~torchrl.collectors.SyncDataCollector`'s ``compact_obs``
#   flag drops it at the producer side, halving the observation
#   footprint of rollouts and replay buffers.
# * :class:`~torchrl.envs.transforms.NextStateReconstructor` rebuilds
#   it on the consumer side, with ``NaN`` at the (genuinely missing)
#   trajectory ends.
# * The value-estimator pipeline keeps GAE / TD targets numerically
#   defined via
#   :meth:`~torchrl.objectives.value.ValueEstimatorBase._sanitize_next_obs_nan`.
# * :class:`~torchrl.envs.transforms.MultiStepTransform` is the main
#   loss-side reason to *not* take this path.
# * :class:`~torchrl.data.LazyMemmapStorage`,
#   :class:`~torchrl.data.SliceSampler`, and the ``"scan"`` / ``"triton"``
#   recurrent backends compose orthogonally for further memory wins.
#
# Useful next resources
# ~~~~~~~~~~~~~~~~~~~~~
#
# * :ref:`Replay buffer tutorial <rb_tuto>` — storage and sampler
#   choices in depth.
# * :ref:`Recurrent DQN tutorial <RNN_tuto>` — sequence training with
#   RNN policies; pair with the ``"scan"`` / ``"triton"`` backends for
#   padding-free training.
# * :ref:`Trajectory assembly tutorial <collector_trajectory_assembly>`
#   — how collectors lay out trajectory ids, masks, and slices.
# * `TorchRL documentation <https://pytorch.org/rl/>`_
