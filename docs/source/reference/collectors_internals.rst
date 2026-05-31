.. currentmodule:: torchrl.collectors

.. _ref_collectors_internals:

Collector Internals
===================

This page describes how :class:`~torchrl.collectors.SyncDataCollector` (aliased
as :class:`Collector`) steps through an environment.  It is meant for
contributors and for users debugging unexpected rollout behaviour: device
casts, per-step bookkeeping, and trajectory tracking are implementation details
that are not visible from the public API.

The multi-process collectors (:class:`MultiSyncCollector` and
:class:`MultiAsyncCollector`) delegate their per-worker rollouts to
:class:`SyncDataCollector`, so the per-worker flow on this page applies to
them too.

Per-timestep flow
-----------------

A single iteration of :meth:`SyncDataCollector.rollout` corresponds to one
environment step.  ``frames_per_batch`` such iterations are stacked into the
batch yielded to the user, extended into a replay buffer, or written directly
with ``replay_buffer.add(...)`` when direct replay-buffer writes are enabled.

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │  for t in range(frames_per_batch):                                  │
    │                                                                     │
    │    ┌─ carrier ──────────────────────────────────────────────────┐   │
    │    │  TensorDict — observation + collector metadata;            │   │
    │    │  device-cleared when needed for cross-device stepping      │   │
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
    │                 │  (3a) replay_buffer.add(carrier_for_out)        │
    │                 │       for direct writes                         │
    │                 │                                                 │
    │                 │  (3b) otherwise cast to storing_device          │
    │                 │       if needed → _sync_storage(), then append  │
    │                 ▼                                                 │
    │           direct replay-buffer write OR append to tensordicts     │
    │                 │                                                 │
    │                 │  carrier = env_next_output  (post-reset state)  │
    │                 │  update traj_ids if any env finished            │
    │                 └─→ next iteration                                │
    └─────────────────────────────────────────────────────────────────────┘

Implementation: :meth:`SyncDataCollector.rollout` in
``torchrl/collectors/_single.py``.

The carrier
-----------

The **carrier** is the :class:`~tensordict.TensorDictBase` instance stored as
``self._carrier``. It persists across calls to ``next(iter(collector))`` and
holds the post-reset result of the previous environment step, which is the
state that the next policy call must consume. It is initialized by
:meth:`SyncDataCollector._make_carrier` and then advanced at the end of every
timestep by assigning ``env_next_output`` back to ``self._carrier``.

Why it exists
~~~~~~~~~~~~~

- **State persistence across batches.**  Collection may stop at a batch
  boundary while the environment trajectory continues. The carrier preserves
  the latest reset-aware environment output so the next rollout resumes from
  the correct observation and recurrent state.
- **Allocation amortization.**  Reusing the same tensordict-shaped state avoids
  allocating a fresh container for every policy/env exchange.
- **Device-neutral handoff.**  When the policy and environment cannot share a
  single device-owned tensordict, the carrier is cleared of its device with
  ``clear_device_()``. The boolean flag ``self._carrier_has_no_device`` records
  whether this invariant must be preserved when new ``"next"`` data is merged.
- **Collector metadata.**  Trajectory IDs and other ``("collector", ...)`` keys
  live on the carrier and persist across steps without round-tripping through
  the env.

Reading the carrier
~~~~~~~~~~~~~~~~~~~

You should not normally touch ``self._carrier`` directly; it is an
implementation detail.  If you need to instrument collected data, use
:attr:`SyncDataCollector.post_collect_hook` or read the data yielded by
iteration. Mutating the carrier from a hook is undefined behaviour.

Sync points
-----------

Three explicit synchronisation functions are installed at construction time in
:meth:`SyncDataCollector._setup_devices` and called inside the rollout loop
when the corresponding explicit sync is not disabled by ``no_cuda_sync=True``:

``_sync_policy``
    Called after copying the carrier to ``policy_device`` and before the
    policy reads it.

``_sync_env``
    Called after copying the carrier to ``env_device`` and before the
    environment reads it.

``_sync_storage``
    Called after copying ``carrier_for_out`` to ``storing_device`` on the
    append-to-list path. The direct ``replay_buffer.add(...)`` path does not
    perform this cast or sync.

What ``_sync_*`` actually is depends on the destination device; see
:meth:`SyncDataCollector._get_sync_fn`:

+-------------------+------------------------------------------------------+
| Destination       | Sync function                                        |
+===================+======================================================+
| ``cuda`` device   | ``_do_nothing`` (CUDA handles ordering itself)       |
+-------------------+------------------------------------------------------+
| Non-CUDA, CUDA    | ``_cuda_sync_if_initialized`` (safe to call after a  |
| available         | GPU-to-host transfer; no-op in fork subprocesses     |
|                   | where CUDA was not initialised)                      |
+-------------------+------------------------------------------------------+
| Non-CUDA, MPS     | ``torch.mps.synchronize``                            |
| available         |                                                      |
+-------------------+------------------------------------------------------+
| Non-CUDA, NPU     | ``torch.npu.synchronize``                            |
| available         |                                                      |
+-------------------+------------------------------------------------------+
| ``cpu`` (no GPU)  | ``_do_nothing``                                      |
+-------------------+------------------------------------------------------+
| ``None``          | ``_do_nothing``                                      |
+-------------------+------------------------------------------------------+

Setting ``no_cuda_sync=True`` on the collector skips the explicit ``_sync_*``
calls. Only do this if you know the transfers are already correctly ordered or
if you are running pure CPU.

Device casting flags
--------------------

Two cached booleans short-circuit the per-step device logic:

``_cast_to_policy_device``
    Set in :meth:`SyncDataCollector._setup_devices`.  ``True`` iff
    ``policy_device != env_device``.  When ``True``, the carrier is copied to
    ``policy_device`` before the policy is invoked.

``_cast_to_env_device``
    Set in :meth:`SyncDataCollector._apply_env_device`, after the environment
    device has been applied or inferred.  It is ``True`` when
    ``_cast_to_policy_device`` is already ``True`` or when
    ``env.device != storing_device``.  When ``True``, the carrier is copied to
    ``env_device`` before ``env.step_and_maybe_reset``.

These are computed once so that the per-step branches degenerate into a single
bool check when everything lives on the same device.

The companion flag ``_carrier_has_no_device`` (set in
:meth:`SyncDataCollector._make_carrier`) records whether the carrier was
stripped of its device.  When ``True``, any new ``"next"`` data merged into the
carrier after an env step is also device-stripped so the deviceless invariant
is preserved.

Trajectory IDs
--------------

When ``track_traj_ids=True`` (the default), every frame carries a
``("collector", "traj_ids")`` integer that uniquely identifies the trajectory
it belongs to.  Two pieces of machinery cooperate:

- :meth:`SyncDataCollector._traj_pool` returns a process-local
  :class:`torchrl.collectors.utils._TrajectoryPool` that hands out fresh IDs.
  In multi-process collectors, workers share a locked pool created by the
  parent collector so IDs do not collide across worker resets.
- :meth:`SyncDataCollector._update_traj_ids` runs after each env step. It reads
  the aggregated end-of-trajectory signal from ``("next", "done")`` via
  :func:`_aggregate_end_of_traj`, draws as many fresh IDs from the pool as
  there are envs that finished, and ``masked_scatter``-s them into the per-env
  ``traj_ids`` tensor on the carrier.

Setting ``track_traj_ids=False`` skips both the per-step bookkeeping and the
allocation of the ``traj_ids`` tensor. This is useful in throughput-sensitive
setups that do not need trajectory-aware sampling. Note that
``split_trajs=True`` requires ``track_traj_ids=True``; the constructor will
raise if you ask for the former without the latter.

Collection hooks
----------------

Two opt-in callbacks let you instrument collection without subclassing:

``pre_collect_hook``
    Called once at the top of :meth:`rollout`, before the per-timestep loop
    starts and before any ``reset_at_each_iter`` reset. Receives no arguments.
    Use it to step a profiler, mark a section in NVTX, or update a
    worker-local counter.

``post_collect_hook``
    Called with the batch tensordict immediately before it is yielded to the
    consumer. Receives the :class:`~tensordict.TensorDictBase` that will be
    yielded. Return value is ignored. Use it to log metrics derived from the
    batch.

Hooks are worker-local: in :class:`MultiSyncCollector` /
:class:`MultiAsyncCollector` they run inside each worker process, not on the
training worker. Exceptions raised by a hook propagate up and stop collection;
they are not swallowed.

For batch *transformations* (rather than instrumentation), use ``postproc`` on
the collector constructor instead.

Where to look in the code
-------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Concept
     - File / function
   * - Per-step rollout loop
     - :meth:`SyncDataCollector.rollout` in ``torchrl/collectors/_single.py``
   * - Carrier initialization
     - :meth:`SyncDataCollector._make_carrier`
   * - Device setup and policy cast flag
     - :meth:`SyncDataCollector._setup_devices`
   * - Environment device application and env cast flag
     - :meth:`SyncDataCollector._apply_env_device`
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
