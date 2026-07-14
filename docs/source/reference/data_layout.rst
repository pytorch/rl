.. _data-layout:

Data layout: contiguous trajectories
====================================

This page describes how trajectory data is laid out across TorchRL's
collectors, replay buffers, and recurrent modules — and explains why the
**default and recommended** layout is a single, flat 1-D tensordict in
which trajectories are concatenated end-to-end and their boundaries are
recovered from the per-step ``is_init`` / ``("next", "done")`` /
``("next", "truncated")`` / ``("next", "terminated")`` markers, **not**
from a fixed ``[B, T]`` shape with a padding mask.

Two main patterns coexist in TorchRL:

* **Padded ``[B, T]`` tensordict + mask.** Every batch has a fixed time
  dimension; short trajectories are padded with garbage and a boolean
  ``mask`` flags real vs padded steps. Every downstream consumer (loss,
  advantage estimator, normalizer) must mask-out the padded entries. This
  is the layout produced by
  :func:`~torchrl.collectors.utils.split_trajectories`,
  :class:`~torchrl.collectors.MultiCollector` with ``split_trajs=True``,
  and :class:`~torchrl.data.replay_buffers.SliceSampler` with
  ``pad_output=True``. It is **discouraged for new code** — see
  :ref:`data-layout-padded-discouraged` below.

* **Concatenated 1-D tensordict.** A single flat tensordict where
  trajectories sit next to each other and their boundaries are encoded
  by per-step markers (``is_init`` at the first step of each
  trajectory / slice, ``("next", "done")`` at the last). No padding, no
  mask, no wasted FLOPs. Every TorchRL primitive that needs to know about
  trajectory structure (recurrent modules under
  :class:`~torchrl.modules.set_recurrent_mode`,
  :class:`~torchrl.data.replay_buffers.SliceSampler`, value estimators in
  ``shifted=True`` value estimators) consumes this layout natively.

The rest of this page walks through the building blocks.

Trajectory boundary keys
------------------------

.. _data-layout-boundary-keys:

Four per-step boolean keys jointly describe a trajectory:

``is_init``
    *Marks the first step of a trajectory or slice.* Written by
    :class:`~torchrl.envs.transforms.InitTracker` at every reset, and
    additionally written by
    :class:`~torchrl.data.replay_buffers.SliceSampler` at the start of every
    sampled slice (OR-ed with whatever ``InitTracker`` already produced, so
    real episode resets that fall *inside* a slice are preserved). This is
    the key recurrent modules
    (:class:`~torchrl.modules.LSTMModule`,
    :class:`~torchrl.modules.GRUModule`) split on under
    :class:`~torchrl.modules.set_recurrent_mode` ``("recurrent")``: each
    ``is_init=True`` position resets the hidden state to whatever was stored
    at that index, letting the RNN process a flat batch of concatenated
    trajectories as if it had been called recursively on each one.

``("next", "done")``
    *Marks the last step of a trajectory.* The union of ``terminated`` and
    ``truncated`` (TorchRL's
    :class:`~torchrl.envs.EnvBase` metaclass guarantees both are flanked
    with their dual). Used by collectors to decide when to reset, by
    :class:`~torchrl.data.replay_buffers.SliceSampler` to reconstruct trajectory boundaries
    when no ``traj_ids`` key is available, and by
    :func:`~torchrl.collectors.utils.split_trajectories` (legacy).
    Datasets sometimes carry only a subset of the three flags; consumers
    that detect trajectory ends from flags should use the union of
    :data:`~torchrl.data.DEFAULT_DONE_KEYS` rather than ``done`` alone.

``("next", "terminated")``
    *Trajectory ended because the MDP says so* (goal reached, agent
    died, etc.). The bootstrap value of the next state is **zero**.
    Value estimators rely on this to decide whether to bootstrap.

``("next", "truncated")``
    *Trajectory ended because a time limit (or other external clock)
    cut it off.* The bootstrap value of the next state is the predicted
    value, **not** zero. Conflating ``truncated`` and ``terminated`` is a
    classic source of value-estimation bugs.

``("collector", "traj_ids")``
    *Optional integer per-step trajectory identifier.* Written by every
    :class:`~torchrl.collectors.BaseCollector` subclass by default
    (``track_traj_ids=False`` disables it). When present,
    :class:`~torchrl.data.replay_buffers.SliceSampler` uses this directly instead of
    reconstructing boundaries from ``done``. Auto-detected on the first
    sample call when no ``traj_key`` is passed at construction.

The "1-D contiguous" layout uses these keys *exclusively* — no shape-based
padding, no mask. Every primitive in TorchRL that needs to know where
trajectories start and stop reads them. The next section describes how they
are *consumed* at read time.

.. _ref_traj_boundaries:

Trajectory boundaries: recovering episodes from storage
-------------------------------------------------------

A replay-buffer storage holds steps, not trajectories: nothing in the
storage layer knows where an episode starts or ends. Components that need
trajectories (:class:`~torchrl.data.replay_buffers.SliceSampler` and its
variants, trajectory-aware transforms, offline dataset tooling) recover the
boundaries at *read time* from the markers described above. The contract
between producers and consumers is the following:

- **Collectors stamp trajectory ids** under ``("collector", "traj_ids")``
  by default (``track_traj_ids=False`` disables it). This is the most
  robust boundary marker: a change of id between two consecutive steps in
  storage order is a boundary, whether or not the episode ended with a
  ``done`` flag.
- **End flags mark trajectory ends.** A step can be the last of its
  trajectory because of any of the :data:`~torchrl.data.DEFAULT_DONE_KEYS`
  entries (``"done"``, ``"truncated"``, ``"terminated"``, typically read
  under ``("next", ...)``). Consumers that reconstruct boundaries from
  flags should use the union of these signals: a dataset that only carries
  ``truncated=True`` ends would otherwise silently merge consecutive
  episodes. :class:`~torchrl.data.replay_buffers.SliceSampler` reads a
  single ``end_key`` (default ``("next", "done")``) for backward
  compatibility; pass
  ``end_keys=[("next", key) for key in DEFAULT_DONE_KEYS]`` to apply the
  union convention.
- **Collectors can mark batch ends as truncations.** Passing
  ``set_truncated=True`` to a collector marks the last step of every
  rollout batch as truncated. This introduces artificial trajectory ends,
  but guarantees that batch boundaries are never silently crossed.
  Multi-process collectors warn when a
  :class:`~torchrl.data.replay_buffers.SliceSampler` is used and neither
  ``trajs_per_batch`` nor ``set_truncated`` is set, because different
  workers' batches interleave in the shared buffer and adjacent frames can
  then belong to different episodes (see :ref:`collectors_replay_trajs`
  for the trade-offs and the recommended ``trajs_per_batch`` alternative).
  Single-process collectors do not need this: they write batches in
  temporal order, so a batch boundary is not a seam — the next batch
  continues exactly where the previous one ended, and the only
  mid-trajectory edge is the live write cursor (handled below).
- **Writers never mutate stored data.** No flag is written into the
  storage when the ring buffer wraps or when a write stops mid-trajectory.
  Instead, samplers resolve the missing boundaries at read time from the
  storage state, as described next.

**Circular-storage semantics.** Once a storage is full it behaves as a ring
buffer, and its *physical* order (index 0 to N-1) no longer matches the
*chronological* order in which the steps were written. Boundary recovery —
implemented by :func:`~torchrl.data.find_start_stop_traj`, which
:class:`~torchrl.data.replay_buffers.SliceSampler` uses under the hood —
resolves this as follows:

- The storage's ``_last_cursor`` records where the last write landed. The
  step under the cursor is the oldest remaining step of a
  partially-overwritten trajectory, so the cursor position is treated as an
  implicit truncation (an end flag is forced there at read time).
- When the storage is *not* full, the last valid element is always treated
  as a trajectory end (the write head is an implicit truncation).
- A trajectory with no intervening end flag can span the wrap point of a
  full storage. The recovered ``(start, stop, lengths)`` indices represent
  this as ``start > stop`` with ``stop`` *inclusive*: a trajectory spanning
  rows ``[8, 9, 0, 1, 2]`` of a 10-row storage has ``start=8``, ``stop=2``
  and ``lengths=5``.
- If a full storage carries no end marker at all in some batch column, a
  single trajectory ending at the last row is assumed for that column.

.. warning::

    There is one blind spot: if the stored data carries no trajectory ids
    and an episode ended mid-buffer without any end flag set (e.g. data
    collected without ``set_truncated`` and stripped of its
    ``("collector", "traj_ids")`` entry), that boundary is unrecoverable —
    the two episodes are indistinguishable from a single longer one, and
    any consumer will merge them. Keep the trajectory ids, or make sure
    every trajectory ends with one of the
    :data:`~torchrl.data.DEFAULT_DONE_KEYS` flags set.

New components that need trajectory boundaries should call
:func:`~torchrl.data.find_start_stop_traj` rather than reimplement these
conventions (naive reimplementations typically mishandle the wrap point and
truncated-only episode ends). Which API to reach for:

.. list-table::
   :header-rows: 1

   * - Input / use case
     - API
   * - Fresh contiguous rollout, needing padded or nested per-trajectory views
     - :func:`~torchrl.collectors.utils.split_trajectories` (padded output is
       discouraged unless explicitly needed — see
       :ref:`data-layout-split-trajectories`)
   * - Physical replay-storage markers, needing boundary indices / lengths
     - :func:`~torchrl.data.find_start_stop_traj`
   * - Sampling contiguous trajectory slices from a buffer
     - :class:`~torchrl.data.replay_buffers.SliceSampler` (and variants)
   * - Collecting only complete trajectories in the first place
     - ``trajs_per_batch`` (see :ref:`collectors_replay_trajs`)

The replay buffer ``ndim`` arg and why it doesn't multi-process well
--------------------------------------------------------------------

.. _data-layout-storage-ndim:

:class:`~torchrl.data.replay_buffers.LazyTensorStorage` (and friends)
accepts an ``ndim`` argument that tells the storage how many dimensions to
preserve when extending. The natural mapping is:

* ``ndim=1`` (default) — flat 1-D buffer; ``extend(td_of_shape_[N])`` writes
  ``N`` rows.
* ``ndim=2`` — buffer of shape ``[N, T]``; ``extend(td_of_shape_[B, T])``
  writes ``B`` rows of ``T`` consecutive frames each. Useful when the
  collector itself produces ``[num_envs, frames_per_env]`` batches (e.g.
  :class:`~torchrl.envs.ParallelEnv` rollouts), because that lets the
  :class:`~torchrl.data.replay_buffers.SliceSampler` infer one trajectory per row without
  scanning ``done`` keys.
* ``ndim=3`` and beyond — when both an outer worker dim and an env dim
  exist, e.g. ``MultiSyncCollector([ParallelEnv(2, …)] * 4, …)``.

It looks attractive: the buffer stores its data in the same shape the
collector produces, no reshape needed.

**It runs into trouble with multi-process collectors that share a single
storage.** With ``ndim >= 2`` every ``extend`` call commits one row's
worth of frames along the time axis, and that row is implicitly assumed to
be a contiguous run of frames from a single env. When several worker
processes write into the same storage concurrently — e.g.
:class:`~torchrl.collectors.MultiCollector` ``(sync=False)``,
:class:`~torchrl.collectors.distributed.RayCollector`, an external pool of
producers, or any cluster setup where a learner aggregates batches from
many actors — the inter-worker write order is uncontrolled. Without
boundary markers, a given row of the ``[N, T]`` storage can stitch
together frames from a worker's two consecutive but unrelated episodes
(or from different workers if a postprocessing step rearranges the extend
order), and a :class:`~torchrl.data.replay_buffers.SliceSampler` drawing whole rows would
silently span them.

Two existing knobs mitigate this without giving up ``ndim >= 2``:

* **``Collector(set_truncated=True)``.** Marks the last frame of every
  collected batch as ``truncated`` (and therefore ``done``). With this on,
  the boundary keys *do* delimit each row, so a sampler that respects
  ``done`` / ``truncated`` no longer spans batch boundaries. The cost is
  that every batch boundary becomes an *artificial* trajectory cut: it
  shortens the effective length of the trajectories the buffer can serve,
  and downstream value estimators see truncations that did not actually
  happen in the env (so they bootstrap where they shouldn't have had to).
  Useful when the alternative is wrong, but pure overhead in the cases
  where complete-trajectory writes are an option.

* **One buffer per worker, glued by a**
  :class:`~torchrl.data.ReplayBufferEnsemble`. Each member
  storage is written by exactly one worker, so its write order is
  deterministic and ``ndim >= 2`` is sound for that member. The ensemble
  samples uniformly across members at training time:

  .. code-block:: python

      from torchrl.collectors import MultiCollector
      from torchrl.data import (
          LazyTensorStorage, ReplayBufferEnsemble, TensorDictReplayBuffer,
      )
      from torchrl.data.replay_buffers import SliceSampler

      num_workers = 4
      buffers = [
          TensorDictReplayBuffer(
              storage=LazyTensorStorage(N, ndim=2),     # one row = one rollout
              sampler=SliceSampler(num_slices=4),
          )
          for _ in range(num_workers)
      ]
      rb = ReplayBufferEnsemble(
          *buffers, sample_from_all=True, batch_size=256,
      )
      collector = MultiCollector(
          [make_env] * num_workers, policy,
          replay_buffer=rb,                              # see note below
          frames_per_batch=200, total_frames=-1,
      )

  .. note::
      Routing each worker's writes to *its own* member buffer requires
      an indexed-extend hook the parent ``ReplayBufferEnsemble`` does not
      provide out of the box. The pattern is sketched here for the
      ``ndim >= 2`` shape; if you go this route you will likely either
      build a thin per-worker collector (each owning its buffer) or wire
      a custom ``post_collect_hook`` that dispatches to the right member.

The simpler default is to keep ``ndim = 1`` and rely on the boundary keys
above to recover trajectory structure — see
:ref:`data-layout-buffer-to-collector` below.

Concretely, ``ndim >= 2`` is straightforward for:

* Single-process :class:`~torchrl.collectors.Collector` with a batched
  env (one process writes; the env dim is stable).
* Synchronous :class:`~torchrl.collectors.MultiCollector` ``(sync=True)``
  with ``cat_results="stack"``, which delivers one ``[num_workers, T]``
  batch at a time *atomically*.

It needs the ``set_truncated`` or ensemble mitigation above for:

* :class:`~torchrl.collectors.MultiCollector` ``(sync=False)``.
* :class:`~torchrl.collectors.distributed.RayCollector`,
  :class:`~torchrl.collectors.distributed.DistributedSyncCollector`,
  RPC-based collectors.
* Any setup where multiple producers ``extend`` the same shared storage
  with no synchronisation guarantee.

The buffer-to-collector handoff: complete-trajectory writes
-----------------------------------------------------------

.. _data-layout-buffer-to-collector:

The clean solution to the multi-process ordering problem is to make every
``extend`` call a **complete, well-formed trajectory** rather than a
fixed-frame batch. Then the ordering of those extends across workers
becomes irrelevant: each row of the 1-D storage is a self-contained
episode segment, and any slice sampled from it is correct by construction.

This is what passing the buffer directly to the collector and setting
``trajs_per_batch`` does:

.. code-block:: python

    from torchrl.collectors import MultiCollector
    from torchrl.data import LazyTensorStorage, SliceSampler, TensorDictReplayBuffer

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000),         # ndim=1 — the only safe choice
        sampler=SliceSampler(slice_len=32),         # auto-detects ("collector", "traj_ids")
        batch_size=256,
    )
    collector = MultiCollector(
        [make_env] * 4,
        policy,
        replay_buffer=rb,
        frames_per_batch=200,
        total_frames=-1,
        trajs_per_batch=8,    # each worker writes COMPLETE trajectories only
        sync=False,
    )
    collector.start()
    for _ in range(train_steps):
        batch = rb.sample()    # variable-length contiguous slices, no leaks
        # ...

How the buffer is actually populated when ``replay_buffer=`` is passed:

* In a multi-process collector, **each worker process calls
  ``rb.extend(...)`` directly on the shared storage** — the parent does
  not aggregate batches and re-extend.
* This is true in both ``sync=True`` and ``sync=False`` mode. The
  ``sync=`` flag controls *iteration delivery* (whether ``for data in
  collector`` blocks for all workers vs first-come), not how the buffer
  is populated.
* Consequently the inter-worker write order on the shared buffer is
  uncontrolled in both modes — it is the same condition that makes
  ``ndim >= 2`` shared storages unsafe (see
  :ref:`data-layout-storage-ndim`).

What ``trajs_per_batch`` adds is a guarantee on the *contents* of each
extend: with ``trajs_per_batch=N``, every ``rb.extend`` call commits one
or more **complete trajectories** (last step has
``("next", "done") == True``). The buffer never sees a partial episode,
so even when worker A's flush interleaves with worker B's, the resulting
storage is just a concatenation of complete episodes. No intra-episode
boundary ever sits at a worker-handoff seam, and the
:class:`~torchrl.data.replay_buffers.SliceSampler`'s boundary detection works on a flat
``ndim = 1`` buffer regardless of write order.

If complete-trajectory writes are not an option (e.g. very long episodes,
where waiting for ``done`` is impractical), ``set_truncated=True``
provides a lighter mitigation by inserting an artificial ``truncated``
at every batch boundary — see
:ref:`collectors_replay_trajs` for the trade-offs.

See :ref:`collectors_replay_trajs` for the full ``trajs_per_batch`` API
and the synchronous-iteration pattern.

SliceSampler: variable-length contiguous slices
-----------------------------------------------

.. _data-layout-slice-sampler:

:class:`~torchrl.data.replay_buffers.SliceSampler` consumes a flat 1-D
buffer and emits **a flat 1-D batch of concatenated slices**. It does not
reshape to ``[num_slices, slice_len]``; it concatenates slices end-to-end
along the only batch dim and writes ``is_init=True`` at the first step of
each slice (OR-ed with any pre-existing ``is_init`` from
:class:`~torchrl.envs.transforms.InitTracker`).

Defaults that match the recommended layout:

* ``strict_length=False`` — short trajectories are kept and produce
  shorter slices; the resulting batch can be smaller than
  ``num_slices * slice_len``. *This is a feature, not a defect.*
* ``pad_output=False`` (default) — no padding, no ``mask`` key.
* ``traj_key`` not specified — the sampler probes the storage on the
  first sample call and prefers ``("collector", "traj_ids")`` over
  ``"episode"``, falling back to reconstructing trajectory boundaries
  from ``("next", "done")`` if neither key is present.

The flat 1-D output plugs directly into a recurrent policy under
:class:`~torchrl.modules.set_recurrent_mode` ``("recurrent")``: the
RNN splits on ``is_init``, treats each slice as an independent
sub-trajectory, and uses each slice's stored hidden state at position 0
as its initial hidden state. The output is identical (bitwise) to what a
manually-reshaped ``[num_slices, slice_len]`` call would produce.

.. code-block:: python

    from torchrl.data.replay_buffers import SliceSampler
    from torchrl.modules import set_recurrent_mode

    sampler = SliceSampler(num_slices=4, slice_len=32, strict_length=False)
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(100_000),
                                sampler=sampler, batch_size=128)
    # ... extend rb from a collector ...
    sample = rb.sample()              # shape [<= 128]
    with set_recurrent_mode("recurrent"):
        out = recurrent_policy(sample)  # consumes the flat sample directly

.. _data-layout-padded-discouraged:

``pad_output=True`` is available as an escape hatch for code that
genuinely cannot accept a ragged batch (a custom op that requires a fixed
time dimension before a manual reshape, for instance). It pads short
slices by *duplicating their last real timestep* and emits a 1-D bool
``("collector", "mask")`` of length ``B * T`` flagging real vs padded
positions. **This is discouraged for new code.** All TorchRL-provided
primitives consume the unpadded layout natively, so padding is pure
overhead and adds a key the caller has to remember to honour everywhere.

Auto-discoverability for recurrent policies
-------------------------------------------

.. _data-layout-rnn-auto:

A recurrent policy needs two things from its env that a vanilla env does
not provide: a per-step ``is_init`` marker (so the RNN knows where
trajectories start) and a hidden-state placeholder in the env's reset
output (so the policy can read a sensible initial state on step 0).
TorchRL wires both up automatically. There are two equivalent entry
points; either is fine:

1. **Pass ``policy=`` to the env constructor.** The
   :class:`~torchrl.envs.EnvBase` metaclass post-init hook walks the
   policy looking for recurrent submodules (anything implementing
   ``make_tensordict_primer()``) and appends an
   :class:`~torchrl.envs.transforms.InitTracker` plus the matching
   :class:`~torchrl.envs.transforms.TensorDictPrimer` to the env when
   the env's specs don't already provide them.

   .. code-block:: python

       from torchrl.envs import GymEnv
       from torchrl.modules import GRUModule

       gru = GRUModule(input_size=4, hidden_size=8, num_layers=1,
                       in_keys=["observation", "recurrent_state", "is_init"],
                       out_keys=["features", ("next", "recurrent_state")])
       env = GymEnv("CartPole-v1", policy=gru)   # InitTracker + primer attached

2. **Pass a bare env to the collector.** When
   :class:`~torchrl.collectors.Collector` (or any subclass) is constructed
   with ``auto_register_policy_transforms=True``, it runs the same
   spec-based detection and appends what's missing. The check is
   idempotent, so combining the env hook with the collector hook is a
   no-op (no double-wrapping).

The detection reads the env's ``full_observation_spec`` and
``full_state_spec``, so it sees through
:class:`~torchrl.envs.SerialEnv` / :class:`~torchrl.envs.ParallelEnv`
where transforms live in child envs. Limitations and the v0.15 default
flip are documented in the
:ref:`policy= argument page <Environment-policy-arg>`.

Net effect: a recurrent training pipeline rarely needs to touch
``InitTracker`` or ``TensorDictPrimer`` by hand. The user wires the
policy, hands it to the env or collector, and the boundary keys + hidden
state the rest of the stack expects are all in place.

Legacy: ``split_trajectories``
------------------------------

.. _data-layout-split-trajectories:

:func:`~torchrl.collectors.utils.split_trajectories` and the
``split_trajs=True`` collector kwarg implement the older, padded layout:
they take a tensordict that contains multiple trajectories (delineated by
``done`` or ``("collector", "traj_ids")``) and produce an
``[N_traj, T_max]`` zero-padded tensordict with a boolean ``mask`` entry.

Aside from the padding+mask cost shared with every padded layout
(:ref:`data-layout-padded-discouraged`), this introduces a more subtle
problem: it **bakes collector hyperparameters into the data shape**. A
trajectory that spans the boundary between two collected batches gets
cut at the batch boundary — the part before is artificially marked
``truncated`` / ``done``, the part after is artificially marked
``is_init``. Downstream code then sees boundary signals that do not
reflect real env transitions, and changing ``frames_per_batch`` silently
changes the trajectory shape the trainer consumes (effective lengths,
number of returns, where value bootstraps land). The contiguous 1-D
layout sidesteps this entirely: trajectory boundaries come exclusively
from the env's own ``done`` signal.

This is **discouraged for new code**. ``split_trajs=True`` will emit an
advisory warning and is scheduled for full deprecation in a future
release. The recommended replacement is to keep the collector output
flat (``split_trajs=False``, the default) and:

* If you need to draw sub-sequences for training, write the data into a
  buffer with a :class:`~torchrl.data.replay_buffers.SliceSampler`.
* If you need per-trajectory aggregates (returns, lengths) for logging,
  group by ``("collector", "traj_ids")`` directly on the flat tensor.
* If you need a ragged ``[N_traj, T_var]`` view for a custom op, prefer
  ``tensordict.split`` on a ``done`` boundary or
  ``split_trajectories(..., as_nested=True)`` (no zero-padding, no mask)
  rather than the padded form.

Narrow canonicalization for recurrent inputs
--------------------------------------------

The recurrent backends (``scan`` and ``triton``) expect the RNN input and
recurrent-state leaves to be in canonical (contiguous, predictable-stride)
layout. Calling ``data.contiguous(canonical=True)`` on the whole TensorDict
before feeding a recurrent learner is the simplest way to satisfy that, but
it materializes a full-batch copy of every leaf — including rewards, dones,
log-probs, advantages, and value targets the RNN never reads.

:meth:`~torchrl.modules.LSTMModule.canonicalize` (and its
:class:`~torchrl.modules.GRUModule` twin) canonicalize only the subset of
keys the module actually reads/writes (:attr:`canonical_keys`), leaving
unrelated leaves untouched:

.. code-block:: python

    from torchrl.modules import LSTMModule, canonicalize_rnn_subset

    actor = LSTMModule(input_size=..., hidden_size=..., in_key="obs",
                       out_key="actor_h")
    critic = LSTMModule(input_size=..., hidden_size=..., in_key="obs",
                        out_key="critic_h")

    # Before GAE / PPO update: canonicalize only the RNN keys.
    data = canonicalize_rnn_subset(data, [actor, critic])

Pair with :class:`~torchrl.cuda_memory_profile` to verify the win:

.. code-block:: python

    from torchrl import cuda_memory_profile

    with cuda_memory_profile("learner-canonicalize"):
        data = canonicalize_rnn_subset(data, [actor, critic])
        advantages = gae(data)
        update(data)

See also
--------

* :ref:`Recurrent training on sequence batches <recurrent_sequence_tuto>` —
  a runnable tutorial that samples ragged trajectory slices with
  :class:`~torchrl.data.replay_buffers.SliceSampler` and feeds them straight
  into a recurrent policy, the contiguous-batch pattern described on this
  page.
* :doc:`collectors_replay` — concrete ``ndim`` patterns and the full
  ``trajs_per_batch`` API.
* :ref:`Auto-wrapping recurrent transforms <Environment-policy-arg>` —
  the ``policy=`` env argument and the collector-side equivalent.
* :class:`~torchrl.data.replay_buffers.SliceSampler` — reference for the
  sampler used throughout this page.
* :class:`~torchrl.modules.LSTMModule`,
  :class:`~torchrl.modules.GRUModule`,
  :class:`~torchrl.modules.set_recurrent_mode` — the recurrent modules
  that consume the contiguous-trajectory layout natively.
* :func:`~torchrl.modules.canonicalize_rnn_subset` — narrow
  canonicalization for multi-RNN learners.
* :doc:`modules_rnn` — recurrent execution modes, backend selection, and
  Triton precision controls.
