.. currentmodule:: torchrl.collectors

Single Node Collectors
======================

Use :class:`Collector` as the construction entry point for direct and local
multi-process collection. The concrete classes below remain public for
advanced use and document the implementation returned by each selection.

Single node data collectors
---------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    BaseCollector
    Collector
    AsyncCollector
    AsyncBatchedCollector
    MultiCollector
    MultiSyncCollector
    MultiAsyncCollector

.. _collectors_single_traj:

Trajectory batching
-------------------

Pass ``trajs_per_batch=N`` to any collector to receive batches of exactly *N*
complete, zero-padded trajectories instead of fixed-frame batches.
Trajectories that span multiple internal collection steps are automatically
reassembled. Each yielded :class:`~tensordict.TensorDict` has shape
``(N, max_traj_len)`` and includes a ``("collector", "mask")`` boolean tensor
marking valid time steps.

``frames_per_batch`` still controls how frequently the environment is polled
internally; it does **not** determine the output batch size when
``trajs_per_batch`` is set.

.. code-block:: python

    from torchrl.collectors import Collector
    from torchrl.envs import GymEnv

    collector = Collector(
        GymEnv("CartPole-v1"),
        policy=my_policy,
        frames_per_batch=200,  # controls internal polling frequency
        total_frames=10000,
        trajs_per_batch=4,
        traj_format="padded",
    )

    for batch in collector:
        # batch.shape == (4, max_traj_len)
        valid = batch[("collector", "mask")]  # (4, max_traj_len) bool
        loss = compute_loss(batch, valid)
        collector.update_policy_weights_()

**Unpadded batches**: with ``traj_format="cat"`` the *N* trajectories are
concatenated along time instead of stacked and padded.  Each yield is then a
flat ``[sum_i T_i]`` batch with no mask: trajectories are contiguous, in
completion order, with ``("next", "done")`` ``True`` at the last step of each
and ``("collector", "traj_ids")`` identifying them.  Prefer it when episode
lengths vary widely or frames are large — the padded layout materializes
``N * max_traj_len`` frames, the flat one only the steps actually collected.

.. note::
    The current default layout is ``"padded"``, but it will change to
    ``"cat"`` in torchrl v0.16.  Omitting ``traj_format`` while yielding
    ``trajs_per_batch`` batches emits a :class:`FutureWarning`; pass the
    layout explicitly.

**Replay buffer integration**: when a ``replay_buffer`` is also provided,
complete trajectories are written to the buffer as **flat 1-D sequences** (no
padding) instead of being yielded.  This is the recommended pattern for
off-policy training with :class:`~torchrl.data.replay_buffers.SliceSampler`, especially
with multi-process collectors where fixed-frame batches can silently mix
episodes.  See :ref:`collectors_replay_trajs` for full details and examples.

.. note::
    The deprecated collector aliases were removed in v0.13. Construct new
    collectors with ``Collector``. ``BaseCollector``, ``AsyncCollector``,
    ``MultiCollector``, ``MultiSyncCollector``, and ``MultiAsyncCollector``
    remain available as concrete implementation APIs.

Using AsyncBatchedCollector
---------------------------

The :class:`AsyncBatchedCollector` pairs an :class:`~torchrl.envs.AsyncEnvPool`
with an :class:`~torchrl.modules.inference_server.InferenceServer` to pipeline environment
stepping and batched GPU inference.  You only need to supply **env factories**
and a **policy** -- all internal wiring is handled automatically:

.. code-block:: python

    from torchrl.collectors import AsyncBatchedCollector
    from torchrl.envs import GymEnv
    from tensordict.nn import TensorDictModule
    import torch.nn as nn

    policy = TensorDictModule(
        nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
        in_keys=["observation"],
        out_keys=["action"],
    )

    collector = AsyncBatchedCollector(
        create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
        policy=policy,
        frames_per_batch=200,
        total_frames=10000,
        max_batch_size=8,
        env_backend="multiprocessing",
    )

    for data in collector:
        # data is a lazy-stacked TensorDict of collected transitions
        pass

    collector.shutdown()

Compile modules and run their first warm-up calls before iterating the
collector whenever possible. Compiler initialization can create process-wide
worker resources, and overlapping a first compilation with the collector's
coordinator and inference threads can stall either workload. If lazy
compilation after collection has started is unavoidable, pause the collector
around that call:

.. code-block:: python

    for data in collector:
        with collector.pause():
            compiled_learner(data)
        break

The pause context finishes in-flight environment and policy requests, parks
the coordinator threads, and leaves the inference server idle. Collection
resumes automatically when the context exits.

With ``replay_buffer=buffer``, calling ``collector.start()`` writes complete
batches in a background thread until ``total_frames`` is reached. The ordinary
iterator still writes synchronously and yields ``None``. Choose one mode per
collector. Background mode runs post-processing and post-collect hooks on the
writer thread; ``collector.pause()`` waits for writes as well as environment and
policy work to finish. Call ``collector.async_shutdown()`` to join the writer,
close workers, and propagate collection or replay errors.

.. _async_batched_collector_cpu_affinity:

On Linux, ``worker_affinity`` assigns CPU masks to the multiprocessing
environment workers, while ``driver_affinity`` assigns one mask to the
inference-server and coordinator threads, as well as the pool's parent-side
queue feeder threads. Dedicated process-backed inference servers use a spawned
process and are not covered by ``driver_affinity``. For example, with two
driver CPUs followed by four two-CPU worker windows:

.. code-block:: python

    import os

    available_cpus = sorted(os.sched_getaffinity(0))
    driver_cpus = available_cpus[:2]
    worker_cpus = available_cpus[2:10]
    worker_masks = [
        tuple(worker_cpus[start : start + 2])
        for start in range(0, len(worker_cpus), 2)
    ]
    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * len(worker_masks),
        policy=policy,
        frames_per_batch=200,
        env_backend="multiprocessing",
        driver_affinity=driver_cpus,
        worker_affinity=worker_masks,
    )

Build both masks from the CPUs visible through ``os.sched_getaffinity(0)``.
See :ref:`async_env_pool_cpu_affinity` for container cpuset, CFS quota, and
Kubernetes CPU Manager considerations.

**Key advantages over direct collection through** :class:`Collector`:

- The inference server automatically **batches policy forward passes** from
  all environments, maximising GPU utilisation.
- Environment stepping and inference run in **overlapping fashion**, reducing
  idle time.
- Supports ``yield_completed_trajectories=True`` for episode-level yields.

For many fixed-schema CPU environments, set ``env_backend="multiprocessing"``.
The default exchange (``env_exchange="auto"``) then uses shared memory whenever
the environment schema allows, and the collector drains ready shared-memory
slots in batches from one coordinator thread while keeping faster environments
independent of slower ones. This path is intended for environments whose step
latency dominates its millisecond-scale coordinator polling interval.

When both environment stepping and inference should leave the driver process,
pass ``transport="auto"`` together with ``env_backend="multiprocessing"`` and a
``policy_factory``. The collector derives the fixed request and response
layouts from one environment's ``fake_tensordict()`` and one policy pass, builds
a :class:`~torchrl.modules.inference_server.ProcessSlotTransport` and serves the
policy from a dedicated process. Each environment process then performs its own
reset/infer/step loop against a fixed shared-memory inference slot; the driver
receives completed transitions only. When the conditions do not hold (threaded
environment workers, grouped workers, no ``policy_factory``, or a policy whose
inputs the environment does not produce), the policy is served from a thread of
the driver process and the reason is logged. A pre-built
:class:`~torchrl.modules.inference_server.ProcessSlotTransport` can be passed
instead; it implies the process inference server and multiprocessing workers.
``transport="driver"`` keeps the driver-mediated path explicitly: coordinator
threads relay requests to the transport derived from ``policy_backend``. In
v0.15 ``transport="auto"`` becomes the default; until then the collector emits
a :class:`FutureWarning` when the default would change its behavior.
This mode bounds completed and in-flight transitions together to twice the
environment count. Workers park before reserving capacity when the driver stops
consuming, and ``pause()`` still works with a full buffer. Environment workers
are daemonic and both the workers and inference server exit if their owning
process dies, including while environment or policy calls are blocked. Call
``shutdown()`` for normal cleanup; abrupt owner death cannot guarantee
environment cleanup hooks run. Environment factories in this mode must not start
multiprocessing children.

With process workers, ``transition_chunk_size="auto"`` (the default) makes each
worker accumulate at least 64 consecutive transitions, or
``frames_per_batch // num_envs`` when that is larger and never more than
``frames_per_batch``, and send them as one dense message. The driver then
receives one message per chunk, concatenates whole chunks into each batch and
performs a single routed replay write per batch, so its cost per transition is
amortized over the chunk; that work competes with training, and on a
64-environment pixel workload 16-transition messages collected 17% slower than
64-transition ones. A transition reaches the driver once its chunk is complete,
about 64 environment steps later. Pass ``1`` to send every transition as soon as
it completes, or another value to trade delivery latency against driver work; up
to ``transition_chunk_size - 1`` transitions per environment remain in the worker
while collection is paused or stopped. Chunked batches are dense
:class:`~tensordict.TensorDict` instances whose ``env_index`` is a tensor.

Scaling ``Collector`` across local processes
--------------------------------------------

Pass ``num_collectors`` to :class:`Collector` to run parallel local collection.
The ``sync`` parameter selects synchronous or asynchronous delivery and the
constructor returns :class:`MultiSyncCollector` or :class:`MultiAsyncCollector`,
respectively:

.. code-block:: python

    from torchrl.collectors import Collector
    from torchrl.envs import GymEnv

    def make_env():
        return GymEnv("CartPole-v1")

    # Synchronous multi-worker collection (recommended for on-policy algorithms)
    sync_collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
        policy=my_policy,
        frames_per_batch=1000,
        total_frames=100000,
        sync=True,  # ← All workers complete before delivering batch
    )

    # Asynchronous multi-worker collection (recommended for off-policy algorithms)
    async_collector = Collector(
        create_env_fn=make_env,
        num_collectors=4,
        policy=my_policy,
        frames_per_batch=1000,
        total_frames=100000,
        sync=False,  # ← First-come-first-serve delivery
    )

    # Iterate over collected data
    for data in sync_collector:
        # Train on data...
        pass

    sync_collector.shutdown()

**Comparison:**

+------------------------+----------------------------------+----------------------------------+
| Feature                | ``sync=True``                    | ``sync=False``                   |
+========================+==================================+==================================+
| Batch delivery         | All workers complete first       | First available worker           |
+------------------------+----------------------------------+----------------------------------+
| Policy consistency     | All data from same policy version| Data may be from older policy    |
+------------------------+----------------------------------+----------------------------------+
| Best for               | On-policy (PPO, A2C)             | Off-policy (SAC, DQN)            |
+------------------------+----------------------------------+----------------------------------+
| Throughput             | Limited by slowest worker        | Higher throughput                |
+------------------------+----------------------------------+----------------------------------+

Running the Collector Asynchronously
------------------------------------

Passing replay buffers to a collector allows us to start the collection and get rid of the iterative nature of the
collector.
If you want to run a data collector in the background, simply run :meth:`~torchrl.collectors.BaseCollector.start`:

    >>> collector = Collector(..., replay_buffer=rb) # pass your replay buffer
    >>> collector.start()
    >>> # little pause
    >>> time.sleep(10)
    >>> # Start training
    >>> for i in range(optim_steps):
    ...     data = rb.sample()  # Sampling from the replay buffer
    ...     # rest of the training loop

Direct collectors (``Collector(backend="direct")``) run background collection
using multithreading,
so be mindful of Python's GIL and related multithreading restrictions.

Multiprocessed collectors will on the other hand let the child processes handle the filling of the buffer on their own,
which truly decouples the data collection and training.

Data collectors that have been started with `start()` should be shut down using
:meth:`~torchrl.collectors.BaseCollector.async_shutdown`.

.. tip::

    For maximum throughput with trajectory-based training (e.g. recurrent
    policies, decision transformers), combine ``start()`` with
    ``trajs_per_batch`` and a :class:`~torchrl.data.replay_buffers.SliceSampler`:

    .. code-block:: python

        rb = ReplayBuffer(
            storage=LazyTensorStorage(100_000),
            sampler=SliceSampler(slice_len=32, end_key=("next", "done")),
            batch_size=256,
            shared=True,
        )
        collector = Collector(
            make_env,
            policy,
            num_collectors=4,
            replay_buffer=rb,
            frames_per_batch=200,
            total_frames=-1,
            trajs_per_batch=8,
            sync=False,
        )
        collector.start()
        for step in range(train_steps):
            batch = rb.sample()  # clean trajectory slices
            # ...
        collector.async_shutdown()

    Each worker writes only **complete trajectories** to the buffer, so the
    sampler never draws slices that cross episode boundaries.  See
    :ref:`collectors_replay_trajs` for a full discussion.

.. warning:: Running a collector asynchronously decouples the collection from training, which means that the training
    performance may be drastically different depending on the hardware, load and other factors (although it is generally
    expected to provide significant speed-ups). Make sure you understand how this may affect your algorithm and if it
    is a legitimate thing to do! (For example, on-policy algorithms such as PPO should not be run asynchronously
    unless properly benchmarked).
