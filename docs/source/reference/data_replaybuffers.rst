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
the actor. A Ray-owned replay buffer accepts either the flexible Ray payload
transport or a fixed-layout Gloo/NCCL tensor transport. See
:ref:`ref_service_transports` for the compatibility table, payload
restrictions, and expected performance trade-offs, and
:ref:`ref_distributed_transport_layouts` for the per-operation layout
discovery and buffer lifecycle.

.. code-block:: python

    from functools import partial
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    buffer = ReplayBuffer(
        storage=partial(LazyTensorStorage, 1000),
        service_backend="ray",
        service_backend_options={"remote_config": {"num_cpus": 1}},
        transport="distributed",
        transport_options={"backend": "gloo"},
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
    RemoteTensorDictReplayBuffer


Sample units
------------

Replay sampling combines two orthogonal decisions: which anchors are selected
(the sampler's probability distribution) and what each anchor expands into.
A :class:`~torchrl.data.replay_buffers.SampleUnit` passed through the
``sample_unit`` argument owns the second decision. The default behavior,
equivalent to :class:`~torchrl.data.replay_buffers.Transition`, keeps every
anchor as a single transition;
:class:`~torchrl.data.replay_buffers.Sequence` expands each anchor into a
fixed-length sequence of records with explicit episode-boundary policies
(``"pad"``, ``"stop"`` or ``"include_reset"``).

For multidimensional storage, coordinate zero is time and the others are
preserved lanes. ``index`` and ``anchor_index`` contain full coordinates, so
priority reduction remains lane-specific.

A sequence can include context outside its loss-bearing region. For example,
in a recurrent Q-learning learner, a ``burn_in`` prefix reconstructs the
model's hidden state, ``length`` records contribute to the loss, and a
``bootstrap`` suffix supplies future records required by the target estimator.
The learner runs over the complete window and applies its loss only where both
the returned ``learning_mask`` and ``validity_mask`` are true. The sampling unit
selects records and produces these masks; it does not run the model or compute
bootstrap targets.

``dilation`` controls temporal subsampling *inside* a window. For example,
``dilation=2`` selects every other stored record. It neither aggregates the
skipped transitions nor controls spacing or overlap between sampled windows.
With ``B`` sampled anchors, the flat output contains
``B * (burn_in + length + bootstrap)`` records.

.. code-block:: python

    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.data.replay_buffers import Sequence

    rb = ReplayBuffer(
        storage=LazyTensorStorage(1000),
        batch_size=64,
        sample_unit=Sequence(
            length=32,
            burn_in=8,
            bootstrap=5,  # future records needed by this target estimator
            dilation=1,
            episode_boundary="pad",
        ),
    )

    # After filling the buffer, run the recurrent model over the entire sample
    # and restrict the loss to real records in the learning region.
    sample, info = rb.sample(return_info=True)
    loss_mask = info["learning_mask"] & info["validity_mask"]

Conditional record updates
--------------------------

Round-robin writers recycle storage slots, so a physical index captured at
sampling time can point to a different record by the time an asynchronous
computation writes back. Writers constructed with ``track_generations=True``
stamp every slot with a generation counter (see
:ref:`ref_buffers_generations`): samples then expose it as an
``"index_generation"`` entry next to ``"index"``, and
:meth:`~torchrl.data.ReplayBuffer.update_if_present` applies a patch only to
records whose ``(index, generation)`` pair is still live, skipping recycled
slots instead of corrupting them. This supports algorithms that refresh
stored fields after sampling, such as recurrent-state refreshes or
asynchronously computed labels, without pinning the buffer or racing against
collection. Generation tracking is opt-in, and
:meth:`~torchrl.data.ReplayBuffer.update_if_present` raises when the buffer's
writer does not track generations.

.. code-block:: python

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(1000),
        writer=TensorDictRoundRobinWriter(track_generations=True),
        batch_size=32,
    )
    ...
    sample = buffer.sample()
    refreshed = compute_refreshed_state(sample)
    result = buffer.update_if_present(
        index=sample["index"],
        generation=sample["index_generation"],
        patch={"recurrent_state": refreshed},
    )
    print(f"updated {result.updated_count}, skipped {result.stale_count} stale records")

Version-compared updates
~~~~~~~~~~~~~~~~~~~~~~~~

Generation stamps answer "is this still my record?"; they say nothing about
*which* of several concurrent writers holds the freshest result. When
multiple asynchronous workers write back to the same records, pass
``version_key`` and ``version`` to
:meth:`~torchrl.data.ReplayBuffer.update_if_present`: a generation-live
record is then only patched when the incoming version compares favorably
against the value stored under ``version_key`` (strictly greater with
``require_newer=True``, greater-or-equal otherwise), and the accepted
version is written back atomically with the patch. Outdated writers lose
deterministically -- retrying an outdated update mutates nothing and returns
the same result. When one call addresses the same slot several times, only
the row carrying the highest incoming version is applied and the others are
rejected, so the reported result always reflects what was written.

.. code-block:: python

    result = buffer.update_if_present(
        index=sample["index"],
        generation=sample["index_generation"],
        patch={"recurrent_state": refreshed},
        version_key="state_version",
        version=worker_step,
        require_newer=True,
    )
    print(
        f"updated {result.updated_count}, "
        f"outdated {result.version_rejected_count}, "
        f"stale {result.stale_count}"
    )

``version_key`` must name a stored per-record scalar field (nested keys in
tuple form), may not appear in ``patch``, and ``version`` can be a scalar or
one entry per record. The result's ``version_rejected`` mask marks
generation-live records that lost the comparison; ``stale_count`` keeps
counting only generation-stale handles.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SampleUnit
    Sequence
    Transition
    ConditionalUpdateResult

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

Detecting overwritten slots: generation stamps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _ref_buffers_generations:

A replay buffer index is a *physical slot number*, not a handle on a piece of
data. A round-robin writer reuses slots, so an index sampled at one point in
time may name completely different data a moment later. That matters whenever
something outside the buffer holds an index across a write:

- asynchronous training, where an inference worker samples, computes, and only
  then writes results back at the index it was given;
- prioritized replay, where priorities are updated after the forward pass;
- any conditional write ("update this record only if it is still the one I
  read").

Generation stamps make that staleness detectable. With
``track_generations=True``, the writer keeps one counter per storage slot and
advances it on every write to that slot. Comparing the stamp you captured
against the current stamp answers "is this still my data?":

    >>> import torch
    >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
    >>> from torchrl.data import RoundRobinWriter
    >>> rb = ReplayBuffer(
    ...     storage=LazyTensorStorage(8),
    ...     writer=RoundRobinWriter(track_generations=True),
    ... )
    >>> _ = rb.extend(torch.arange(8))
    >>> _, info = rb.sample(4, return_info=True)
    >>> index, generation = info["index"], info["index_generation"]
    >>> _ = rb.extend(torch.arange(8, 11))   # overwrites slots 0, 1, 2
    >>> stale = rb.writer.generations_of(index) != generation
    >>> # `index[stale]` no longer holds the sampled data

:meth:`~torchrl.data.ReplayBuffer.sample` adds ``"index_generation"`` to its
``info`` (and, for tensordict buffers, to the sample itself) whenever the writer
tracks generations, alongside the existing ``"index"``.

Semantics
^^^^^^^^^

- **One stamp per write, not per ``extend`` call.** A single ``extend`` that
  wraps the storage advances a reused slot once for each write it receives, so
  a slot written twice in one call advances by two.
- **``-1`` means "no usable stamp"**: a never-written slot, an
  out-of-range index, or a writer that does not track generations. It is not
  "generation zero".
- **Monotonic across** :meth:`~torchrl.data.ReplayBuffer.empty`. Emptying
  advances every written slot's stamp rather than resetting it, so handles taken
  before the ``empty()`` correctly read as stale. Never-written slots keep
  ``-1``.
- **Stamps are for detection, not for ordering across slots.** Two slots'
  stamps are independent counters; a higher stamp on slot 3 than on slot 7 says
  nothing about write order between them.

Implementation notes
^^^^^^^^^^^^^^^^^^^^

- **Opt-in.** The default is ``track_generations=False``: enabling it allocates
  one ``int64`` per storage slot and adds a key to the sampler output, neither
  of which should be imposed on buffers that do not need it.
- **The counters live on the storage, not on the writer.** Two buffers sharing
  one storage overwrite each other's slots, so a per-writer counter would let
  one buffer's handles read as live after the other overwrote them. The buffer
  is attached to the storage object, and a writer registered against a storage
  that already has one adopts it rather than replacing it.
- **Allocation.** Storages small enough to allocate up front get a single
  allocation, so the buffer's shape never changes and the ``torch.compile``
  extend/sample path does not recompile. Larger and unbounded storages
  (``ListStorage`` with no ``max_size`` reports ``torch.iinfo(torch.int64).max``)
  grow geometrically on demand instead.
- **Process-local.** The counters are not shared across processes: the buffer is
  replaced rather than mutated when it grows, so a shared mapping would silently
  stop tracking after the first growth. A slot overwritten by another process is
  not reflected. Cross-process staleness detection needs a storage-owned,
  fixed-size mapping and is not implemented yet.
- **Multidimensional storages.** A generation stamps a whole dim-0 slot. A 1-D
  index tensor is therefore always read as a batch of slot indices; to identify
  a single cell of an ``ndim > 1`` storage, pass the ``tuple`` of per-dimension
  indices that :meth:`~torchrl.data.ReplayBuffer.extend` returns.
- **Checkpointing.** Stamps are part of ``state_dict``/``dumps`` when tracking
  is on, and a checkpoint written without them (or by an older version) loads
  fine -- tracking simply starts from scratch.

The relevant APIs are :attr:`~torchrl.data.Writer.tracks_generations` and
:meth:`~torchrl.data.Writer.generations_of`, and the ``track_generations``
argument of :class:`~torchrl.data.RoundRobinWriter`.

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

The same ``dim=-3`` convention, and the collector / ``extend`` / ``sample``
wiring, is spelled out in :ref:`catframes-collector-replay`.

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

.. _catframes-collector-replay:

Frame stacking images with CatFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visual off-policy training almost always stacks the last ``N`` frames so a
CNN can see motion. :class:`~torchrl.envs.transforms.CatFrames` can do that
in two places; pick **one** of them for the tensor that is stored.

- **Env-side** (policy / inference): a stateful rolling buffer on the
  :class:`~torchrl.envs.TransformedEnv`. Documented in
  :ref:`CatFrames for visual RL <catframes-images>`.
- **Buffer-side** (this section): store **unstacked** raw pixels and
  rebuild the stack in ``rb.sample()``. A stack of ``N`` float32 frames
  occupies ``N`` times the RAM of one uint8 image; putting
  :class:`~torchrl.envs.transforms.CatFrames` on the sample path is how
  you avoid paying that cost in the storage.

The two placements share ``N`` and ``dim`` so the policy and the loss see
the same layout. They must not both write the stacked key: env stacking
**and** buffer stacking of the already-stacked tensor produces
``[N * N * C, H, W]``.

For CHW images the stack dimension is ``dim=-3`` (channel), not the
vector default ``dim=-1``. After :class:`~torchrl.envs.transforms.ToTensorImage`
(and optional :class:`~torchrl.envs.transforms.GrayScale`) a frame is
``[C, H, W]``; concatenating along ``-3`` yields ``[N * C, H, W]``.

Why the raw frame is what you store
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the env transform and the stored tensor on **different keys**.
:class:`~torchrl.envs.transforms.ToTensorImage` should write a processed
copy (``out_keys=["pixels_trsf"]``) and leave the env's ``"pixels"``
untouched. :class:`~torchrl.envs.transforms.CatFrames` then stacks
``pixels_trsf`` for the policy. On ``rb.extend(...)`` you drop
``pixels_trsf`` (and its ``("next", ...)`` counterpart) so the storage
holds only the uint8 frame. The buffer transform recreates
``pixels_trsf`` from ``"pixels"`` at sample time.

:meth:`~torchrl.envs.transforms.CatFrames.make_rb_transform_and_sampler`
builds the sample-time half of that pipeline: a
:class:`~torchrl.data.replay_buffers.SliceSampler` with ``slice_len=N``,
and a transform that reshapes each sampled slice to ``[B, N]``, unfolds
:class:`~torchrl.envs.transforms.CatFrames` along time, keeps the last
step of every window, and -- on the inverse / write path -- excludes the
stacked ``out_keys``. Offline :class:`~torchrl.envs.transforms.CatFrames`
(``forward`` / ``unfolding``) needs a time dimension and
``("next", "done")`` to stop stacks at episode boundaries; the helper
sampler is what provides that time axis. Sampling independent transitions
and then unfolding treats the batch index as time and mixes unrelated
frames.

Collector, ``extend`` and ``sample``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~torchrl.collectors.Collector` writes ``("collector", "traj_ids")``
on every batch. Pass that key to the helper so the
:class:`~torchrl.data.replay_buffers.SliceSampler` agrees with the
collector on trajectory boundaries. Then ``extend`` each collected batch
and ``sample`` as usual:

.. code-block:: python

    from torchrl.collectors import Collector, RandomPolicy
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.envs import (
        CatFrames,
        Compose,
        GrayScale,
        GymEnv,
        InitTracker,
        Resize,
        StepCounter,
        ToTensorImage,
        TransformedEnv,
    )

    frame_stack = 4
    batch_size = 32

    catframes = CatFrames(
        N=frame_stack,
        dim=-3,
        in_keys=["pixels_trsf"],
        out_keys=["pixels_trsf"],
    )
    env = TransformedEnv(
        GymEnv("CartPole-v1", from_pixels=True, pixels_only=True),
        Compose(
            InitTracker(),
            ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
            GrayScale(in_keys=["pixels_trsf"]),
            Resize(84, 84, in_keys=["pixels_trsf"]),
            catframes,
            StepCounter(),
        ),
    )  # doctest: +SKIP

    rb_catframes, sampler = catframes.make_rb_transform_and_sampler(
        batch_size=batch_size,
        traj_key=("collector", "traj_ids"),
    )
    # Sample-path processing must match the env: same ToTensorImage /
    # GrayScale / Resize, but applied to both the root and the "next"
    # pixels so CatFrames sees the layout it saw during collection.
    rb = ReplayBuffer(
        storage=LazyTensorStorage(100_000),
        sampler=sampler,
        batch_size=batch_size,
        transform=Compose(
            ToTensorImage(
                in_keys=["pixels", ("next", "pixels")],
                out_keys=["pixels_trsf", ("next", "pixels_trsf")],
            ),
            GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
            Resize(84, 84, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
            rb_catframes,
        ),
    )  # doctest: +SKIP

    collector = Collector(
        env,
        RandomPolicy(env.action_spec),
        frames_per_batch=64,
        total_frames=10_000,
    )  # doctest: +SKIP
    for data in collector:
        # Inverse ExcludeTransform on rb_catframes already drops the
        # stacked keys on write; excluding them here is equivalent and
        # makes the stored tensordict obvious.
        rb.extend(data.exclude("pixels_trsf", ("next", "pixels_trsf")))
    batch = rb.sample()
    # batch["pixels_trsf"] has shape [32, 4, 84, 84] (grayscale, dim=-3)

``rb.extend(data)`` is the only write API you need: the collector yields
a tensordict of shape ``[frames_per_batch]`` (or
``[batch, time]`` for a batched env -- then give the storage
``ndim=2``, see :ref:`collectors and replay buffers <ref_collectors>`).
Do not flatten away ``("collector", "traj_ids")`` or ``("next", "done")``
before extending; the sampler uses them to keep each slice inside one
episode.

The same ``extend`` / ``sample`` pairing without the helper looks like
this. The extra ``SliceSampler(slice_len=N)`` is not optional: without
it, :meth:`~torchrl.envs.transforms.CatFrames.unfolding` has no time
dimension.

.. code-block:: python

    from torchrl.data import SliceSampler
    from torchrl.envs import ExcludeTransform

    rb = ReplayBuffer(
        storage=LazyTensorStorage(100_000),
        sampler=SliceSampler(
            slice_len=frame_stack,
            traj_key=("collector", "traj_ids"),
        ),
        batch_size=batch_size * frame_stack,  # B windows of length N
        transform=Compose(
            ToTensorImage(
                in_keys=["pixels", ("next", "pixels")],
                out_keys=["pixels_trsf", ("next", "pixels_trsf")],
            ),
            GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
            Resize(84, 84, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
            CatFrames(
                N=frame_stack,
                dim=-3,
                in_keys=["pixels_trsf", ("next", "pixels_trsf")],
                out_keys=["pixels_trsf", ("next", "pixels_trsf")],
            ),
            ExcludeTransform("pixels_trsf", ("next", "pixels_trsf"), inverse=True),
        ),
    )  # doctest: +SKIP
    rb.extend(data)          # stores raw "pixels" only
    batch = rb.sample()      # rebuilds the N-frame stack

The helper is the preferred form: it multiplies the requested
``batch_size`` by ``N`` internally, reshapes to ``[B, N]``, and keeps
``batch[:, -1]``, so ``rb.sample()`` returns ``batch_size`` stacked
transitions rather than a flat ``batch_size * N`` window.

Common pitfalls
^^^^^^^^^^^^^^^

- **Stacking twice.** Env :class:`~torchrl.envs.transforms.CatFrames`
  writing ``pixels_trsf`` **and** a buffer
  :class:`~torchrl.envs.transforms.CatFrames` that reads that same
  already-stacked key. Store raw ``"pixels"`` and rebuild, or store the
  stack and do not attach :class:`~torchrl.envs.transforms.CatFrames` to
  the buffer -- not both.
- **Wrong ``dim``.** Images after
  :class:`~torchrl.envs.transforms.ToTensorImage` are CHW: use
  ``dim=-3``. ``dim=-1`` is for vector observations. ``dim=-4`` is only
  correct after an
  :class:`~torchrl.envs.transforms.UnsqueezeTransform` that inserted that
  axis (the ``[N, C, H, W]`` variant in
  ``examples/replay-buffers/catframes-in-buffer.py``).
- **No time axis.** Offline :class:`~torchrl.envs.transforms.CatFrames`
  unfolds along time. A uniform random sample of transitions has no
  time axis, so the stack is assembled from unrelated steps. Always pair
  the buffer transform with a
  :class:`~torchrl.data.replay_buffers.SliceSampler` (or
  :meth:`~torchrl.envs.transforms.CatFrames.make_rb_transform_and_sampler`).
- **Missing ``("next", ...)`` keys.** A buffer transform does not walk
  into ``"next"`` on its own. List both ``"pixels"`` and
  ``("next", "pixels")`` on every sample-path transform that should
  apply to both.
- **Same in/out key on**
  :class:`~torchrl.envs.transforms.ToTensorImage`. If the processed
  tensor overwrites ``"pixels"``, there is no cheap raw frame left to
  store and you cannot drop the stack on ``extend``.

.. seealso::

    :ref:`CatFrames for visual RL <catframes-images>` for the env-side
    placement, reset / :class:`~torchrl.envs.transforms.InitTracker`
    behaviour, and the ``dim=-3`` convention.
    :ref:`Collectors and replay buffers <ref_collectors>` for
    ``ndim`` / ``traj_key`` when the collector is batched or
    multi-process. A runnable script of the extra-axis (``dim=-4``)
    variant is ``examples/replay-buffers/catframes-in-buffer.py``.
