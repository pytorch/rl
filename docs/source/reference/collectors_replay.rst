.. currentmodule:: torchrl.collectors

Collectors and Replay Buffers
=============================

Collectors and replay buffers interoperability
----------------------------------------------

In the simplest scenario where single transitions have to be sampled
from the replay buffer, little attention has to be given to the way
the collector is built. Flattening the data after collection will
be a sufficient preprocessing step before populating the storage:

    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N),
    ...     transform=lambda data: data.reshape(-1))
    >>> for data in collector:
    ...     memory.extend(data)

If trajectory slices have to be collected, the recommended way to achieve this is to create
a multidimensional buffer and sample using the :class:`~torchrl.data.replay_buffers.SliceSampler`
sampler class. One must ensure that the data passed to the buffer is properly shaped, with the
``time`` and ``batch`` dimensions clearly separated. In practice, the following configurations
will work:

    >>> # Single environment: no need for a multi-dimensional buffer
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = Collector(env, policy, frames_per_batch=N, total_frames=-1)
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # Batched environments: a multi-dim buffer is required
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=2),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> env = ParallelEnv(4, make_env)
    >>> collector = Collector(env, policy, frames_per_batch=N, total_frames=-1)
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # MultiSyncCollector + regular env: behaves like a ParallelEnv if cat_results="stack"
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=2),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = MultiSyncCollector([make_env] * 4,
    ...     policy,
    ...     frames_per_batch=N,
    ...     total_frames=-1,
    ...     cat_results="stack")
    >>> for data in collector:
    ...     memory.extend(data)
    >>> # MultiSyncCollector + parallel env: the ndim must be adapted accordingly
    >>> memory = ReplayBuffer(
    ...     storage=LazyTensorStorage(N, ndim=3),
    ...     sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
    ... )
    >>> collector = MultiSyncCollector([ParallelEnv(2, make_env)] * 4,
    ...     policy,
    ...     frames_per_batch=N,
    ...     total_frames=-1,
    ...     cat_results="stack")
    >>> for data in collector:
    ...     memory.extend(data)

.. important::

    The ``ndim=2`` and ``ndim=3`` examples above apply to **fixed-frame
    batches** (the default, without ``trajs_per_batch``).  When
    ``trajs_per_batch`` is set, each trajectory is written to the buffer as a
    **flat 1-D sequence** of variable length.  A storage with ``ndim >= 2``
    expects a fixed second dimension that variable-length trajectories cannot
    satisfy.  Always use the default ``ndim=1`` when combining
    ``trajs_per_batch`` with a replay buffer.

.. _collectors_replay_trajs:

Complete trajectory collection with ``trajs_per_batch``
-------------------------------------------------------

When using a multi-process collector
(:class:`~torchrl.collectors.MultiSyncCollector` or
:class:`~torchrl.collectors.MultiAsyncCollector`) with fixed-frame batches
and a :class:`~torchrl.data.SliceSampler`, adjacent frames in the buffer can
come from **different workers and different episodes** without an intervening
``done`` signal.  The sampler has no way to detect these invisible boundaries,
so it may draw slices that straddle unrelated trajectories — silently
corrupting the training data.

Setting ``trajs_per_batch`` on the collector solves this.  Each worker
assembles **complete trajectories** (episodes whose last step carries
``("next", "done") == True``) before writing them to the buffer as flat 1-D
sequences — no padding, no artificial boundaries.  Every trajectory in the
buffer is guaranteed to be a genuine episode segment, making it directly
compatible with :class:`~torchrl.data.SliceSampler`.

**Synchronous iteration (for-loop)**

.. code-block:: python

    from torchrl.collectors import MultiCollector
    from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler

    rb = ReplayBuffer(
        storage=LazyTensorStorage(100_000),
        sampler=SliceSampler(slice_len=32, end_key=("next", "done")),
        batch_size=256,
    )
    collector = MultiCollector(
        [make_env] * 4,
        policy,
        replay_buffer=rb,
        frames_per_batch=200,
        total_frames=500_000,
        trajs_per_batch=8,   # each worker writes complete trajectories
        sync=True,
    )
    for _ in collector:          # yields None (data goes straight to rb)
        batch = rb.sample()      # contiguous sub-sequences, no cross-episode leaks
        loss = loss_fn(batch)
        # ...

**Asynchronous collection (``start()``)**

For off-policy algorithms where data collection and training run
concurrently, use :meth:`~torchrl.collectors.BaseCollector.start`:

.. code-block:: python

    collector = MultiCollector(
        [make_env] * 4,
        policy,
        replay_buffer=rb,
        frames_per_batch=200,
        total_frames=-1,
        trajs_per_batch=8,
        sync=False,
    )
    collector.start()           # workers fill rb in background threads/processes
    for step in range(train_steps):
        batch = rb.sample()
        loss = loss_fn(batch)
        # ...
        collector.update_policy_weights_()
    collector.async_shutdown()

This pattern fully decouples data collection from training and is the
recommended way to maximise inference throughput on multi-core machines or
GPU-accelerated environments.

**Single-process collectors** also support ``trajs_per_batch`` with the same
replay-buffer semantics:

.. code-block:: python

    collector = Collector(
        env, policy,
        replay_buffer=rb,
        frames_per_batch=200,
        total_frames=-1,
        trajs_per_batch=8,
    )
    collector.start()
    # ...

.. warning::

    Without ``trajs_per_batch``, a multi-process collector writes fixed-frame
    batches from each worker.  If the buffer uses a
    :class:`~torchrl.data.SliceSampler`, the sampler will reconstruct episode
    boundaries from ``done`` signals, but worker batch boundaries are invisible
    — consecutive frames in the buffer may belong to completely different
    episodes.

    A partial mitigation is ``set_truncated=True``, which marks every batch
    boundary with a ``truncated`` (and therefore ``done``) signal.  This
    prevents cross-episode slices but introduces artificial truncations that
    value estimators must handle correctly.

    ``trajs_per_batch`` is the recommended solution: it guarantees clean
    episode boundaries in the buffer without artificial truncations.

.. seealso::

    - :class:`~torchrl.collectors.BaseCollector` for the full ``trajs_per_batch``
      API, completeness guarantee, and batched-environment behaviour.
    - :class:`~torchrl.data.SliceSampler` for configuring sub-sequence sampling
      from the buffer.
    - :ref:`The trajectory batching section <collectors_single>` in the
      single-node collector docs for the non-replay-buffer usage
      (padded ``(trajs, max_len)`` batches).

Helper functions
----------------

.. currentmodule:: torchrl.collectors.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    split_trajectories
