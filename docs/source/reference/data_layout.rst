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
  :func:`~torchrl.modules.set_recurrent_mode`,
  :class:`~torchrl.data.SliceSampler`, value estimators in
  ``single_call=True`` mode) consumes this layout natively.

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
    :func:`~torchrl.modules.set_recurrent_mode` ``("recurrent")``: each
    ``is_init=True`` position resets the hidden state to whatever was stored
    at that index, letting the RNN process a flat batch of concatenated
    trajectories as if it had been called recursively on each one.

``("next", "done")``
    *Marks the last step of a trajectory.* The union of ``terminated`` and
    ``truncated`` (TorchRL's
    :class:`~torchrl.envs.EnvBase` metaclass guarantees both are flanked
    with their dual). Used by collectors to decide when to reset, by
    :class:`~torchrl.data.SliceSampler` to reconstruct trajectory boundaries
    when no ``traj_ids`` key is available, and by
    :func:`~torchrl.collectors.utils.split_trajectories` (legacy).

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
    :class:`~torchrl.collectors.BaseCollector` subclass. When present,
    :class:`~torchrl.data.SliceSampler` uses this directly instead of
    reconstructing boundaries from ``done``. Auto-detected on the first
    sample call when no ``traj_key`` is passed at construction.

The "1-D contiguous" layout uses these keys *exclusively* — no shape-based
padding, no mask. Every primitive in TorchRL that needs to know where
trajectories start and stop reads them.

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
  :class:`~torchrl.data.SliceSampler` infer one trajectory per row without
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
order), and a :class:`~torchrl.data.SliceSampler` drawing whole rows would
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
  :class:`~torchrl.data.replay_buffers.ReplayBufferEnsemble`. Each member
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
:class:`~torchrl.data.SliceSampler`'s boundary detection works on a flat
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
:func:`~torchrl.modules.set_recurrent_mode` ``("recurrent")``: the
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
  buffer with a :class:`~torchrl.data.SliceSampler`.
* If you need per-trajectory aggregates (returns, lengths) for logging,
  group by ``("collector", "traj_ids")`` directly on the flat tensor.
* If you need a ragged ``[N_traj, T_var]`` view for a custom op, prefer
  ``tensordict.split`` on a ``done`` boundary or
  ``split_trajectories(..., as_nested=True)`` (no zero-padding, no mask)
  rather than the padded form.

See also
--------

* :doc:`collectors_replay` — concrete ``ndim`` patterns and the full
  ``trajs_per_batch`` API.
* :ref:`Auto-wrapping recurrent transforms <Environment-policy-arg>` —
  the ``policy=`` env argument and the collector-side equivalent.
* :class:`~torchrl.data.replay_buffers.SliceSampler` — reference for the
  sampler used throughout this page.
* :class:`~torchrl.modules.LSTMModule`,
  :class:`~torchrl.modules.GRUModule`,
  :func:`~torchrl.modules.set_recurrent_mode` — the recurrent modules
  that consume the contiguous-trajectory layout natively.
