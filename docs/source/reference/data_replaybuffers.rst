.. currentmodule:: torchrl.data

Replay Buffers
==============

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:

Core Replay Buffer Classes
--------------------------

Replay buffers use ``service_backend="direct"`` by default, where
``buffer.client() is buffer``. ``service_backend="ray"`` constructs a
:class:`RayReplayBuffer` owner and ``client()`` returns the restricted,
picklable handle intended for collector workers. Only the owner can shut down
the actor.

.. code-block:: python

    from functools import partial
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    buffer = ReplayBuffer(
        storage=partial(LazyTensorStorage, 1000),
        service_backend="ray",
        service_backend_options={"remote_config": {"num_cpus": 1}},
    )
    worker_buffer = buffer.client()
    buffer.shutdown()

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBuffer
    OfflineToOnlineReplayBuffer
    ReplayBufferEnsemble
    PrioritizedReplayBuffer
    TensorDictReplayBuffer
    TensorDictPrioritizedReplayBuffer
    RayReplayBuffer
    DataParallelReplayBufferClient
    RemoteTensorDictReplayBuffer

Data-parallel replay clients
----------------------------

A Ray replay-buffer client can be viewed by replicated learner ranks with
:meth:`~torchrl.data.RayReplayBuffer.client` followed by ``data_parallel``.
The configured and explicit sample batch sizes are **global**: with global
batch size ``B`` and ``W`` ranks, every rank request samples ``B / W`` items
from the single Ray owner. ``B`` must be divisible by ``W``.
Phase 1 rejects data-parallel sampling when owner-side prefetching is enabled,
because the prefetch queue is not keyed by requested batch size. Construct the
owner with ``prefetch=0``.

.. code-block:: python

    from functools import partial
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    global_batch_size = 128
    owner = ReplayBuffer(
        storage=partial(LazyTensorStorage, 10_000),
        batch_size=global_batch_size,
        service_backend="ray",
    )
    replay_buffer = owner.client().data_parallel(
        rank=rank,
        world_size=world_size,
    )
    local_batch = replay_buffer.sample(return_info=False)

Writes and priority updates are forwarded unchanged and ``len(replay_buffer)``
reports the owner's global sampleable length. The Ray actor serializes calls
from every rank, so without-replacement and consuming sampler state, free-list
reuse, and prioritized trees remain owner-coherent. Random and prioritized
draws are distribution-equivalent to a global sample; independent rank calls
do not promise a globally reproducible order.

Shared ``next`` and iteration are unsupported because independent callers
cannot safely delimit finite sampler epochs. For consuming workloads, every
rank should wait until the global sampleable length is at least ``B`` before
sampling. Strict cached global-round sampling is not part of this first
implementation.


Offline-to-online helpers
-------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    prefill_replay_buffer

Trajectory queries
------------------

Stored transitions can be regrouped into trajectories and filtered with a
small query language. :data:`~torchrl.data.traj` builds predicates over
trajectory fields, and :meth:`ReplayBuffer.query` returns the matching
:class:`~torchrl.data.Trajectory` views:

    >>> from torchrl.data import traj
    >>> good = rb.query((traj.reward.sum() > 100) & (traj.length >= 50))
    >>> good[0].observation, good[0].action

Trajectory boundaries are recovered with the same machinery
:class:`~torchrl.data.replay_buffers.SliceSampler` uses, so queries and
samplers always agree on where trajectories start and stop, including for
storages that have wrapped around and for multi-dimensional storages
(``LazyTensorStorage(..., ndim=2)``). Predicates built from
:data:`~torchrl.data.traj` report the entries they read through
:meth:`TrajectoryPredicate.required_keys
<torchrl.data.TrajectoryPredicate.required_keys>`, letting ``query()`` fetch
only those entries (and run only the transforms that can affect them) while
evaluating, instead of materializing the whole buffer content.

:class:`~torchrl.data.Trajectory` is a tensorclass: slicing and indexing
return :class:`~torchrl.data.Trajectory` instances, and query results of
different lengths can be assembled into a single ragged batch with
:func:`~tensordict.lazy_stack`.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Trajectory
    TrajectoryPredicate

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    filter_trajectories
    iter_trajectories

Composable Replay Buffers
-------------------------

.. _ref_buffers:

We also give users the ability to compose a replay buffer.
We provide a wide panel of solutions for replay buffer usage, including support for
almost any data type; storage in memory, on device or on physical memory;
several sampling strategies; usage of transforms etc.

Supported data types and choosing a storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In theory, replay buffers support any data type but we can't guarantee that each
component will support any data type. The most crude replay buffer implementation
is made of a :class:`~torchrl.data.ReplayBuffer` base with a
:class:`~torchrl.data.replay_buffers.ListStorage` storage. This is very inefficient
but it will allow you to store complex data structures with non-tensor data.
Storages in contiguous memory include :class:`~torchrl.data.replay_buffers.TensorStorage`,
:class:`~torchrl.data.replay_buffers.LazyTensorStorage` and
:class:`~torchrl.data.replay_buffers.LazyMemmapStorage`.

Sampling and indexing
~~~~~~~~~~~~~~~~~~~~~

Replay buffers can be indexed and sampled.
Indexing and sampling collect data at given indices in the storage and then process them
through a series of transforms and ``collate_fn`` that can be passed to the `__init__`
function of the replay buffer.

The full physical storage can be read with ``rb[:]``. This is useful when all
stored items must be processed in storage order, for example to recompute value
targets after collection. :meth:`~torchrl.data.ReplayBuffer.read_all_in_order`
is an explicit equivalent to ``rb[:]``, and
:meth:`~torchrl.data.ReplayBuffer.write_all` is an explicit equivalent to
``rb[:] = data``. Passing ``end=...`` to these helpers updates only the leading
storage entries.

    >>> from tensordict import TensorDict
    >>> import torch
    >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    >>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
    >>> rb.extend(TensorDict({"obs": torch.arange(3)}, [3]))
    tensor([0, 1, 2])
    >>> data = rb.read_all_in_order()
    >>> assert (data == rb[:]).all()
    >>> data["target"] = data["obs"] + 1
    >>> rb.write_all(data)
    >>> assert (rb[:] == data).all()

Consuming replay buffers
~~~~~~~~~~~~~~~~~~~~~~~~

Replay buffers can consume items as they are sampled by passing
``consume_after_n_samples``. This is useful in online loops where a collector
keeps writing new data while the trainer should avoid reusing old samples after
they have contributed to an update.

    >>> import torch
    >>> from torchrl.data import ListStorage, ReplayBuffer
    >>> rb = ReplayBuffer(
    ...     storage=ListStorage(8),
    ...     batch_size=2,
    ...     consume_after_n_samples=1,
    ... )
    >>> rb.extend([torch.tensor(i) for i in range(3)])
    tensor([0, 1, 2])
    >>> batch = rb.sample()
    >>> assert len(batch) == 2
    >>> assert len(rb) == 1
    >>> rb.extend([torch.tensor(3), torch.tensor(4)])
    tensor([3, 4])
    >>> assert len(rb) == 3

The consumed entries remain in physical storage until they are overwritten, but
they are removed from the sampleable set and are not returned by future calls to
:meth:`~torchrl.data.ReplayBuffer.sample`. New writes reuse consumed slots before
falling back to the writer's normal cursor, so consumed data behaves as freed
capacity without scanning the full storage on every write. This mode supports
1-dimensional ``ListStorage``,
``TensorStorage``, ``LazyTensorStorage`` and ``LazyMemmapStorage`` with uniform
random sampling. Prefetching, prioritized replay and multidimensional storages
are rejected explicitly.

Trajectory boundaries
~~~~~~~~~~~~~~~~~~~~~

Replay buffers store steps, not trajectories: components that need
trajectories (:class:`~torchrl.data.replay_buffers.SliceSampler` and its
variants, trajectory-aware transforms, offline dataset tooling) recover
episode boundaries at *read time* from markers present in the stored data.
The full producer/consumer contract — which markers exist, who writes them,
how circular storage (wraparound, write cursor) interacts with boundary
recovery, and its blind spots — is documented in
:ref:`Trajectory boundaries <ref_traj_boundaries>` on the data-layout page.
The associated APIs are:

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    find_start_stop_traj

.. py:data:: DEFAULT_DONE_KEYS
    :value: ("done", "truncated", "terminated")

    Canonical end-of-trajectory signal keys in TED format. A step can be
    marked as the last of its trajectory by any of these entries (typically
    read under the ``"next"`` sub-tensordict); ``"done"`` is the union of the
    other two, but datasets sometimes carry only a subset of the entries, so
    consumers detecting trajectory ends from flags should use the union of
    all three. Shared default of :class:`~torchrl.data.TED2Flat`,
    :class:`~torchrl.data.TED2Nested`, :class:`~torchrl.data.postprocs.MultiStep`
    and :class:`~torchrl.envs.transforms.MultiStepTransform`; accepted by
    :class:`~torchrl.data.replay_buffers.SliceSampler` through its
    ``end_keys`` argument.

TED-format conversion
~~~~~~~~~~~~~~~~~~~~~

The following helpers convert between the TorchRL Episode Data (TED) layout and
a flat, storage-friendly representation when serializing or restoring a buffer:

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TED2Flat
    Flat2TED

Video-backed replay buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video-backed datasets are dominated by frames; materializing every decoded frame
as a dense tensor throws away the video codec's compression. :class:`VideoClipRef`
is a lightweight, picklable reference to frames inside an encoded video (mp4, ...):
it stores only *where* the frames are (the file(s) it spans plus a per-frame
``frame_index`` and ``file_id``), so indexing the whole buffer stays cheap. Frames
are decoded on-demand with
torchcodec by :class:`~torchrl.envs.transforms.DecodeVideoTransform`, appended on
the replay-buffer sample path, so ``rb.sample()`` returns decoded frames aligned to
the sampled steps. It composes with :class:`SliceSampler`: a contiguous window of
sampled steps maps to consecutive frame indices and decodes as a single ranged
read. Decoders are opened lazily and cached per worker process (see
:func:`set_video_decoder_cache_size` and :func:`clear_video_decoder_cache`); the
references stored in the buffer never hold an open decoder.

**Temporal alignment / binning.** Video frames usually outnumber a lower-rate
signal (e.g. 100 frames for 30 proprioceptive steps). :meth:`VideoClipRef.rebin`
(also ``VideoClipRef.from_file(..., num_bins=...)``) resamples the frames onto
``num_bins`` non-overlapping temporal bins:

- ``frames_per_bin=None`` keeps one **center** frame per bin -> ``[num_bins]``,
  decoding to ``[num_bins, C, H, W]`` (subsample);
- ``frames_per_bin=k`` keeps ``k`` frames spanning each bin -> ``[num_bins, k]``,
  decoding to ``[num_bins, k, C, H, W]`` (a dense, non-overlapping stack; frames are
  dropped/repeated to stay rectangular).

For *overlapping* (sliding-window) stacking, subsample first and then apply
:class:`~torchrl.envs.transforms.CatFrames` to the decoded frames on the sample
path -- ``CatFrames`` concatenates along an existing dim
(``[B, C, H, W] -> [B, N*C, H, W]``), giving classic frame-stacking with
trajectory-edge padding, while ``rebin``'s stack keeps a separate frame axis::

    >>> from torchrl.data import VideoClipRef, ReplayBuffer, LazyTensorStorage, SliceSampler
    >>> from torchrl.envs.transforms import CatFrames, Compose, DecodeVideoTransform
    >>> # one frame per step, then a sliding stack of the last 4 along the channel dim
    >>> rb = ReplayBuffer(
    ...     storage=LazyTensorStorage(1000),
    ...     sampler=SliceSampler(slice_len=16, traj_key="episode"),
    ...     transform=Compose(
    ...         DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"]),
    ...         CatFrames(N=4, dim=-3, in_keys=["pixels"]),
    ...     ),
    ... )  # doctest: +SKIP

**Multiple files.** A clip is often split across many small files (one per episode)
rather than one large mp4. :meth:`VideoClipRef.from_files` addresses a list of files
as a single logical sequence, so slicing, :meth:`rebin` and decoding work across
file boundaries (a window that straddles two files decodes per file and
concatenates), with one cached decoder per file. No ``LazyStacked`` / ``LazyCat``
container is needed -- it is just a longer ``frame_index`` plus a per-frame
``file_id``. The index is stored compactly: the unique file paths live once in the
``sources`` tuple and each frame carries a single ``int64`` ``file_id`` into it, so
references spanning thousands of files stay light on the replay-buffer sample path
(the resolved path is still available via the ``VideoClipRef.source`` property).

When camera and control loops run at different rates, prefer
:meth:`VideoClipRef.from_timestamps` to align frames by time rather than by index.

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    VideoClipRef
    clear_video_decoder_cache
    set_video_decoder_cache_size
