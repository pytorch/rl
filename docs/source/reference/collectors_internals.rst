.. currentmodule:: torchrl.collectors

.. _ref_collectors_internals:

Collector Internals
===================

This page describes how :class:`~torchrl.collectors.SyncDataCollector` (aliased
as :class:`Collector`) actually steps through an environment.  It is meant for
contributors and for users debugging unexpected rollout behaviour — the device
casts, the per-step bookkeeping, and the trajectory-tracking machinery are not
visible from the public API and have, until now, only been documented in
inline comments inside ``torchrl/collectors/_single.py``.

The multi-process collectors (:class:`MultiSyncCollector`,
:class:`MultiAsyncCollector`) delegate their per-worker rollouts to
:class:`SyncDataCollector`, so everything on this page applies to them too.

Per-timestep flow
-----------------

A single iteration of :meth:`SyncDataCollector.rollout` corresponds to one
environment step.  ``frames_per_batch`` such iterations are stacked into the
batch yielded to the user (or written to the replay buffer).

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │  for t in range(frames_per_batch):                                  │
    │                                                                     │
    │    ┌─ carrier ──────────────────────────────────────────────────┐   │
    │    │  TensorDict — observation + collector metadata,            │   │
    │    │  device-cleared if policy_device != env_device             │   │
    │    └────────────┬───────────────────────────────────────────────┘   │
    │                 │                                                   │
    │                 │  (1) cast to policy_device if needed              │
    │                 │      → _sync_policy()                             │
    │                 ▼                                                   │
    │           ┌──────────┐                                              │
    │           │  policy  │  ← reads obs, writes action + log_prob       │
    │           └────┬─────┘                                              │
    │                │                                                   │
    │                │  carrier.update(policy_output)                     │
    │                ▼                                                   │
    │    ┌─ carrier (now has action) ──────────────────────────────┐     │
    │    └────────────┬────────────────────────────────────────────┘     │
    │                 │                                                  │
    │                 │  (2) cast to env_device if needed                │
    │                 │      → _sync_env()                               │
    │                 ▼                                                  │
    │           ┌──────────────┐                                         │
    │           │  env.step_   │  ← returns (env_output, env_next_output)│
    │           │  and_maybe_  │    auto-resets done envs                │
    │           │  reset       │                                         │
    │           └────┬─────────┘                                         │
    │                │                                                  │
    │                │  carrier.set("next", env_output["next"])         │
    │                ▼                                                  │
    │    ┌─ carrier_for_out (snapshot for this step) ──────────────┐    │
    │    └────────────┬────────────────────────────────────────────┘    │
    │                 │                                                 │
    │                 │  (3) cast to storing_device if needed           │
    │                 │      → _sync_storage()                          │
    │                 ▼                                                 │
    │           append to tensordicts list  OR  replay_buffer.add(...)  │
    │                 │                                                 │
    │                 │  carrier = env_next_output  (post-reset state)  │
    │                 │  update traj_ids if any env finished            │
    │                 └─→ next iteration                                │
    └─────────────────────────────────────────────────────────────────────┘

Implementation: :meth:`SyncDataCollector.rollout` in
``torchrl/collectors/_single.py``.

The carrier
-----------

The **carrier** (formerly called the *shuttle*) is the single
:class:`~tensordict.TensorDictBase` instance that survives across timesteps and
carries data between the environment and the policy.  It is allocated once by
:meth:`SyncDataCollector._make_shuttle` and stored as ``self._carrier``.

Why it exists
~~~~~~~~~~~~~

- **Allocation amortization.**  Reallocating a tensordict every timestep would
  dominate the wall-clock cost of fast environments.  Reusing one tensordict
  keeps the per-step overhead at the cost of a few in-place updates.
- **Deviceless semantics.**  When the policy and environment live on
  *different* devices (e.g. policy on ``cuda:0``, env on ``cpu``), the carrier
  is cleared of any device pin via ``clear_device_()`` so that subsequent
  ``.to(device, non_blocking=True)`` calls do the right thing regardless of
  which side wrote to it last.  The boolean flag
  ``self._shuttle_has_no_device`` records whether this clearing happened —
  see :meth:`_make_shuttle`.
- **Single source of truth for collector metadata.**  Trajectory IDs and
  any other ``("collector", ...)`` keys live on the carrier and persist
  across steps without round-tripping through the env.

Reading the carrier
~~~~~~~~~~~~~~~~~~~

You should not normally touch ``self._carrier`` directly — it is an
implementation detail.  If you need to instrument what the policy sees on
step ``t``, use :attr:`SyncDataCollector.pre_collect_hook` or read the data
yielded by iteration (which is a copy).  Mutating the carrier from a hook is
undefined behaviour.

Sync points
-----------

Three explicit synchronisation functions are installed at construction time
in :meth:`SyncDataCollector._setup_devices` and called inside the rollout
loop:

``_sync_policy``
    Called after copying the carrier to ``policy_device`` (rollout loop,
    after the ``self._carrier.to(self.policy_device, ...)`` call).  Ensures
    the host has seen the GPU→CPU transfer of whatever was on the carrier
    before the policy reads it.

``_sync_env``
    Called after copying the carrier to ``env_device`` (rollout loop, after
    the ``self._carrier.to(self.env_device, ...)`` call).  Same role on the
    env side.

``_sync_storage``
    Called after copying ``carrier_for_out`` to ``storing_device`` (rollout
    loop, when appending to the per-step ``tensordicts`` list).  Ensures the
    stored batch is consistent before it is returned or stacked.

What ``_sync_*`` actually is depends on the destination device — see
:meth:`SyncDataCollector._get_sync_fn`:

+-------------------+------------------------------------------------------+
| Destination       | Sync function                                        |
+===================+======================================================+
| ``cuda`` device   | ``_do_nothing`` (CUDA handles ordering itself)       |
+-------------------+------------------------------------------------------+
| Non-CUDA, CUDA    | ``_cuda_sync_if_initialized`` (safe to call after a  |
| available         | GPU→CPU transfer; no-op in fork subprocesses where   |
|                   | CUDA was not initialised)                            |
+-------------------+------------------------------------------------------+
| Non-CUDA, MPS     | ``torch.mps.synchronize``                            |
| available         |                                                      |
+-------------------+------------------------------------------------------+
| ``cpu`` (no GPU)  | ``_do_nothing``                                      |
+-------------------+------------------------------------------------------+
| ``None``          | ``_do_nothing``                                      |
+-------------------+------------------------------------------------------+

Setting ``no_cuda_sync=True`` on the collector skips the explicit ``_sync_*``
calls — only do this if you know all your transfers are CUDA-stream-ordered
or if you are running pure-CPU.

Device casting flags
--------------------

Two cached booleans short-circuit the per-step device logic:

``_cast_to_policy_device``
    Set once in :meth:`_setup_devices`.  ``True`` iff
    ``policy_device != env_device``.  When ``True``, the carrier is copied
    to ``policy_device`` before the policy is invoked.

``_cast_to_env_device``
    Set once in :meth:`_setup_devices` (with an extra refinement at
    :meth:`_make_final_rollout`-time).  ``True`` iff a cast is required on
    the env side — either because the policy device differs from the env
    device, or because the policy lives on a device the env cannot consume
    directly.

These are computed once so that the per-step branches degenerate into a
single bool check when everything lives on the same device — the common
single-GPU case pays essentially no device-management overhead.

The companion flag ``_shuttle_has_no_device`` (set in :meth:`_make_shuttle`)
records whether the carrier was stripped of its device.  When ``True``, any
new ``"next"`` data merged into the carrier after an env step is also
device-stripped (see the ``if self._shuttle_has_no_device`` block in the
rollout loop) so the deviceless invariant is preserved.

Trajectory IDs
--------------

When ``track_traj_ids=True`` (the default), every frame carries a
``("collector", "traj_ids")`` integer that uniquely identifies the trajectory
it belongs to.  Two pieces of machinery cooperate:

- :meth:`SyncDataCollector._traj_pool` returns a process-local
  :class:`_TrajectoryPool` that hands out fresh IDs and guarantees they do
  not collide across resets.
- :meth:`SyncDataCollector._update_traj_ids` runs after each env step.  It
  reads the aggregated end-of-trajectory signal from ``("next", "done")``
  via :func:`_aggregate_end_of_traj`, draws as many fresh IDs from the pool
  as there are envs that finished, and ``masked_scatter``-s them into the
  per-env ``traj_ids`` tensor on the carrier.

Setting ``track_traj_ids=False`` skips both the per-step bookkeeping and the
allocation of the ``traj_ids`` tensor — worth it in throughput-sensitive
setups that do not need trajectory-aware sampling.  Note that
``split_trajs=True`` requires ``track_traj_ids=True``; the constructor will
raise if you ask for the former without the latter.

Collection hooks
----------------

Two opt-in callbacks let you instrument the rollout without subclassing:

``pre_collect_hook``
    Called once at the top of :meth:`rollout`, before the per-timestep loop
    starts (and before any ``reset_at_each_iter`` reset).  Receives no
    arguments.  Use it to step a profiler, mark a section in NVTX, or update
    a worker-local counter.

``post_collect_hook``
    Called with the batch tensordict immediately before it is yielded to
    the consumer.  Receives the :class:`~tensordict.TensorDictBase` that
    will be yielded.  Return value is ignored.  Use it to log metrics
    derived from the batch.

Hooks are worker-local: in :class:`MultiSyncCollector` /
:class:`MultiAsyncCollector` they run inside each worker process, not on
the training worker.  Exceptions raised by a hook propagate up and stop
collection — they are not swallowed.

For batch *transformations* (rather than instrumentation), use ``postproc``
on the collector constructor instead.

Where to look in the code
-------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Concept
     - File / function
   * - Per-step rollout loop
     - :meth:`SyncDataCollector.rollout` in ``torchrl/collectors/_single.py``
   * - Carrier allocation
     - :meth:`SyncDataCollector._make_shuttle`
   * - Device setup, cast flags
     - :meth:`SyncDataCollector._setup_devices`
   * - Sync function dispatch
     - :meth:`SyncDataCollector._get_sync_fn`
   * - Trajectory ID update
     - :meth:`SyncDataCollector._update_traj_ids`
   * - Trajectory ID pool
     - :class:`torchrl.collectors.utils._TrajectoryPool`
   * - Hooks
     - ``pre_collect_hook`` / ``post_collect_hook`` constructor arguments

See also
--------

- :ref:`ref_collectors` for the high-level API
- :ref:`ref_profiling` for ``TORCHRL_PROFILING=1`` instrumentation that emits
  named ranges on the carrier / policy / env transitions described above
- :doc:`data_layout` for the shape and key conventions the carrier follows
