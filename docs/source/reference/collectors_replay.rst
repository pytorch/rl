.. currentmodule:: torchrl.collectors

Collectors and Replay Buffers
=============================

.. seealso::

    For the conceptual story behind the patterns on this page —
    contiguous 1-D trajectories, the boundary keys (``is_init``,
    ``done``, ``terminated``, ``truncated``), the limits of
    ``ndim>=2`` storages with multi-process collectors, and why
    :func:`~torchrl.collectors.utils.split_trajectories` is no longer
    recommended — see :ref:`Data layout: contiguous trajectories
    <data-layout>`.

Preferred layout: flat 1-D storage
----------------------------------

TorchRL recommends storing replay data in a single flat, 1-D buffer
(``ndim=1``). Collector batch dimensions describe how data was produced; they
do not need to become replay-buffer dimensions. Trajectory boundaries remain
available through ``("collector", "traj_ids")``, ``("next", "done")``,
``("next", "terminated")``, and ``("next", "truncated")``, and trajectory
slices can be reconstructed at sampling time with
:class:`~torchrl.data.replay_buffers.SliceSampler`.

When the collector writes directly to the replay buffer, set
``flatten_data=True``. A batched rollout with shape ``[N, T]`` is then reshaped
once to ``[N * T]`` and written with one ``extend`` call:

.. code-block:: python

    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    memory = ReplayBuffer(storage=LazyTensorStorage(100_000))
    collector = Collector(
        env,
        policy,
        frames_per_batch=200,
        total_frames=-1,
        replay_buffer=memory,
        flatten_data=True,
    )
    for _ in collector:  # yields None; transitions are written directly
        batch = memory.sample(256)

The same argument is available on :class:`~torchrl.collectors.AsyncCollector`,
:class:`~torchrl.collectors.MultiSyncCollector`,
:class:`~torchrl.collectors.MultiAsyncCollector`, and
:class:`~torchrl.collectors.distributed.RayCollector`. If the caller owns the
write instead, flatten explicitly before extending:

.. code-block:: python

    for data in collector:
        memory.extend(data.reshape(-1))

Higher-dimensional storage (``ndim >= 2``) remains supported for applications
that deliberately store fixed-shape chunks. It is a specialized layout rather
than the recommended default: the chunk dimensions become part of every replay
item, variable-length trajectories cannot be represented directly, and shared
multi-producer buffers require extra care around trajectory boundaries. See
:ref:`The replay buffer ndim arg and why it doesn't multi-process well
<data-layout-storage-ndim>` for details.

.. _collectors_replay_trajs:

Complete trajectory collection with ``trajs_per_batch``
-------------------------------------------------------

When using a multi-process collector
(:class:`~torchrl.collectors.MultiSyncCollector` or
:class:`~torchrl.collectors.MultiAsyncCollector`) with fixed-frame batches
and a :class:`~torchrl.data.replay_buffers.SliceSampler`, adjacent frames in the buffer can
come from **different workers and different episodes** without an intervening
``done`` signal.  The sampler has no way to detect these invisible boundaries,
so it may draw slices that straddle unrelated trajectories — silently
corrupting the training data.

Setting ``trajs_per_batch`` on the collector solves this.  Each worker
assembles **complete trajectories** (episodes whose last step carries
``("next", "done") == True``) before writing them to the buffer as flat 1-D
sequences — no padding, no artificial boundaries.  Every trajectory in the
buffer is guaranteed to be a genuine episode segment, making it directly
compatible with :class:`~torchrl.data.replay_buffers.SliceSampler`.

``trajs_per_batch`` is not tied to replay buffers or multi-process
collection: on any collector — including the single-process
:class:`~torchrl.collectors.Collector` — setting it (without a
``replay_buffer``) switches the iterator from fixed-frame batches to batches
of exactly that many **complete trajectories**, zero-padded along time with
a ``("collector", "mask")`` entry marking the valid steps.  Episodes
spanning internal collection steps are reassembled, and in-flight episodes
are held back for the next batch.  This is the natural fit for on-policy
algorithms whose training unit is the episode (e.g. GRPO-style
group-relative advantages):

.. code-block:: python

    from torchrl.collectors import Collector

    collector = Collector(
        env,
        policy,
        frames_per_batch=200,    # internal polling granularity only
        total_frames=-1,
        trajs_per_batch=16,      # one yield = 16 whole episodes
        traj_format="padded",    # default until v0.16, then "cat"
    )
    for batch in collector:      # batch: [16, max_traj_len]
        mask = batch["collector", "mask"]
        returns = (batch["next", "reward"].squeeze(-1) * mask).sum(-1)
        # ...

The padded layout is convenient for per-trajectory reductions but
materializes ``16 * max_traj_len`` frames even when most episodes are short.
With ``traj_format="cat"`` the same batches come out **flat and unpadded**
instead: trajectories are concatenated along time (shape ``[sum_i T_i]``),
contiguous and in completion order, with ``("next", "done")`` ``True`` at the
last step of each and ``("collector", "traj_ids")`` telling them apart — the
same layout the replay-buffer write path produces.  Prefer it when episode
lengths vary a lot or frames are large (images, token sequences):

.. code-block:: python

    collector = Collector(
        env,
        policy,
        frames_per_batch=200,
        total_frames=-1,
        trajs_per_batch=16,
        traj_format="cat",
    )
    for batch in collector:      # batch: [sum of the 16 episode lengths]
        done = batch["next", "done"].squeeze(-1)
        episode_idx = done.long().cumsum(0) - done.long()
        # per-episode return without any padding
        returns = torch.zeros(16).index_add_(
            0, episode_idx, batch["next", "reward"].squeeze(-1)
        )
        # ...


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
    :class:`~torchrl.data.replay_buffers.SliceSampler`, the sampler will reconstruct episode
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
    - :class:`~torchrl.data.replay_buffers.SliceSampler` for configuring sub-sequence sampling
      from the buffer.
    - :ref:`Trajectory boundaries <ref_traj_boundaries>` for the contract the
      sampler relies on: which markers delimit trajectories in storage, how
      boundaries are recovered at read time
      (:func:`~torchrl.data.find_start_stop_traj`), and the blind spot when
      neither ids nor end flags are present.
    - :ref:`The trajectory batching section <collectors_single>` in the
      single-node collector docs for the non-replay-buffer usage
      (padded ``(trajs, max_len)`` batches, or flat unpadded batches with
      ``traj_format="cat"``).

Helper functions
----------------

.. currentmodule:: torchrl.collectors.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    split_trajectories
