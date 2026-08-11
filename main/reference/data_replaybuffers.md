# Replay Buffers

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:

## Core Replay Buffer Classes

Replay buffers use `service_backend="direct"` by default, where
`buffer.client() is buffer`. `service_backend="ray"` constructs a
[`RayReplayBuffer`](generated/torchrl.data.RayReplayBuffer.html#torchrl.data.RayReplayBuffer) owner and `client()` returns the restricted,
picklable handle intended for collector workers. Only the owner can shut down
the actor. A Ray-owned replay buffer accepts either the flexible Ray payload
transport or a fixed-layout Gloo/NCCL tensor transport. See
[Choosing a payload transport](services_workflow.html#ref-service-transports) for the compatibility table, payload
restrictions, and expected performance trade-offs, and
[Distributed transport implementation notes](services_workflow.html#ref-distributed-transport-layouts) for the per-operation layout
discovery and buffer lifecycle.

```
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
```

| [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)(*args[, use_ray_service, ...]) | A generic, composable replay buffer class. |
| --- | --- |
| [`OfflineToOnlineReplayBuffer`](generated/torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer)(offline_dataset, *) | A replay buffer combining an immutable offline dataset with a growing online buffer. |
| [`ReplayBufferEnsemble`](generated/torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble)(*args[, ...]) | An ensemble of replay buffers. |
| [`PrioritizedReplayBuffer`](generated/torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer)(*args[, ...]) | Prioritized replay buffer. |
| [`TensorDictReplayBuffer`](generated/torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer)(*args[, ...]) | TensorDict-specific wrapper around the [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) class. |
| [`TensorDictPrioritizedReplayBuffer`](generated/torchrl.data.TensorDictPrioritizedReplayBuffer.html#torchrl.data.TensorDictPrioritizedReplayBuffer)(*args[, ...]) | TensorDict-specific wrapper around the [`PrioritizedReplayBuffer`](generated/torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer) class. |
| [`RayReplayBuffer`](generated/torchrl.data.RayReplayBuffer.html#torchrl.data.RayReplayBuffer)(*args[, use_ray_service, ...]) | A Ray implementation of the Replay Buffer that can be extended and sampled remotely. |
| [`RemoteTensorDictReplayBuffer`](generated/torchrl.data.RemoteTensorDictReplayBuffer.html#torchrl.data.RemoteTensorDictReplayBuffer)(*args[, ...]) | A remote invocation friendly ReplayBuffer class. |

## Sample units

Replay sampling combines two orthogonal decisions: which anchors are selected
(the sampler's probability distribution) and what each anchor expands into.
A `SampleUnit` passed through the
`sample_unit` argument owns the second decision. The default behavior,
equivalent to `Transition`, keeps every
anchor as a single transition;
`Sequence` expands each anchor into a
fixed-length sequence of records with explicit episode-boundary policies
(`"pad"`, `"stop"` or `"include_reset"`).

A sequence can include context outside its loss-bearing region. For example,
in a recurrent Q-learning learner, a `burn_in` prefix reconstructs the
model's hidden state, `length` records contribute to the loss, and a
`bootstrap` suffix supplies future records required by the target estimator.
The learner runs over the complete window and applies its loss only where both
the returned `learning_mask` and `validity_mask` are true. The sampling unit
selects records and produces these masks; it does not run the model or compute
bootstrap targets.

`dilation` controls temporal subsampling *inside* a window. For example,
`dilation=2` selects every other stored record. It neither aggregates the
skipped transitions nor controls spacing or overlap between sampled windows.
With `B` sampled anchors, the flat output contains
`B * (burn_in + length + bootstrap)` records.

```
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import Sequence

rb = ReplayBuffer(
 storage=LazyTensorStorage(1000),
 batch_size=64,
 sample_unit=Sequence(
 length=32,
 burn_in=8,
 bootstrap=5, # future records needed by this target estimator
 dilation=1,
 episode_boundary="pad",
 ),
)

# After filling the buffer, run the recurrent model over the entire sample
# and restrict the loss to real records in the learning region.
sample, info = rb.sample(return_info=True)
loss_mask = info["learning_mask"] & info["validity_mask"]
```

## Conditional record updates

Round-robin writers recycle storage slots, so a physical index captured at
sampling time can point to a different record by the time an asynchronous
computation writes back. Writers constructed with `track_generations=True`
stamp every slot with a generation counter (see
ref_buffers_generations): samples then expose it as an
`"index_generation"` entry next to `"index"`, and
[`update_if_present()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.update_if_present) applies a patch only to
records whose `(index, generation)` pair is still live, skipping recycled
slots instead of corrupting them. This supports algorithms that refresh
stored fields after sampling, such as recurrent-state refreshes or
asynchronously computed labels, without pinning the buffer or racing against
collection. Generation tracking is opt-in, and
[`update_if_present()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.update_if_present) raises when the buffer's
writer does not track generations.

```
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
```

### Version-compared updates

Generation stamps answer "is this still my record?"; they say nothing about
*which* of several concurrent writers holds the freshest result. When
multiple asynchronous workers write back to the same records, pass
`version_key` and `version` to
[`update_if_present()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.update_if_present): a generation-live
record is then only patched when the incoming version compares favorably
against the value stored under `version_key` (strictly greater with
`require_newer=True`, greater-or-equal otherwise), and the accepted
version is written back atomically with the patch. Outdated writers lose
deterministically - retrying an outdated update mutates nothing and returns
the same result. When one call addresses the same slot several times, only
the row carrying the highest incoming version is applied and the others are
rejected, so the reported result always reflects what was written.

```
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
```

`version_key` must name a stored per-record scalar field (nested keys in
tuple form), may not appear in `patch`, and `version` can be a scalar or
one entry per record. The result's `version_rejected` mask marks
generation-live records that lost the comparison; `stale_count` keeps
counting only generation-stale handles.

| [`SampleUnit`](generated/torchrl.data.SampleUnit.html#torchrl.data.SampleUnit)() | Expands sampled anchors into the records a batch is made of. |
| --- | --- |
| [`Sequence`](generated/torchrl.data.Sequence.html#torchrl.data.Sequence)(length[, episode_boundary, ...]) | Expands anchors into a window of records around each anchor. |
| [`Transition`](generated/torchrl.data.Transition.html#torchrl.data.Transition)() | The identity sample unit: every anchor is one transition. |
| [`ConditionalUpdateResult`](generated/torchrl.data.ConditionalUpdateResult.html#torchrl.data.ConditionalUpdateResult)(updated[, ...]) | |

## Offline-to-online helpers

| [`prefill_replay_buffer`](generated/torchrl.data.prefill_replay_buffer.html#torchrl.data.prefill_replay_buffer)(rb, dataset[, ...]) | Copy samples from an offline dataset into a mutable replay buffer. |
| --- | --- |

## Trajectory queries

Stored transitions can be regrouped into trajectories and filtered with a
small query language. `traj` builds predicates over
trajectory fields, and [`ReplayBuffer.query()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.query) returns the matching
[`Trajectory`](generated/torchrl.data.Trajectory.html#torchrl.data.Trajectory) views:

```
>>> from torchrl.data import traj
>>> good = rb.query((traj.reward.sum() > 100) & (traj.length >= 50))
>>> good[0].observation, good[0].action
```

Trajectory boundaries are recovered with the same machinery
[`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) uses, so queries and
samplers always agree on where trajectories start and stop, including for
storages that have wrapped around and for multi-dimensional storages
(`LazyTensorStorage(..., ndim=2)`). Predicates built from
`traj` report the entries they read through
[`TrajectoryPredicate.required_keys`](generated/torchrl.data.TrajectoryPredicate.html#torchrl.data.TrajectoryPredicate.required_keys), letting `query()` fetch
only those entries (and run only the transforms that can affect them) while
evaluating, instead of materializing the whole buffer content.

[`Trajectory`](generated/torchrl.data.Trajectory.html#torchrl.data.Trajectory) is a tensorclass: slicing and indexing
return [`Trajectory`](generated/torchrl.data.Trajectory.html#torchrl.data.Trajectory) instances, and query results of
different lengths can be assembled into a single ragged batch with
`lazy_stack()`.

| [`Trajectory`](generated/torchrl.data.Trajectory.html#torchrl.data.Trajectory)(data, *, batch_size[, device, names]) | |
| --- | --- |
| [`TrajectoryPredicate`](generated/torchrl.data.TrajectoryPredicate.html#torchrl.data.TrajectoryPredicate)(fn[, description, keys]) | A boolean predicate over a [`Trajectory`](generated/torchrl.data.Trajectory.html#torchrl.data.Trajectory). |

| [`filter_trajectories`](generated/torchrl.data.filter_trajectories.html#torchrl.data.filter_trajectories)(data[, predicate, ...]) | Split `data` into trajectories and keep those matching `predicate`. |
| --- | --- |
| [`iter_trajectories`](generated/torchrl.data.iter_trajectories.html#torchrl.data.iter_trajectories)(data[, trajectory_key]) | Iterate over the trajectories stored in a flat batch of transitions. |

## Composable Replay Buffers

We also give users the ability to compose a replay buffer.
We provide a wide panel of solutions for replay buffer usage, including support for
almost any data type; storage in memory, on device or on physical memory;
several sampling strategies; usage of transforms etc.

### Supported data types and choosing a storage

In theory, replay buffers support any data type but we can't guarantee that each
component will support any data type. The most crude replay buffer implementation
is made of a [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) base with a
[`ListStorage`](generated/torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage) storage. This is very inefficient
but it will allow you to store complex data structures with non-tensor data.
Storages in contiguous memory include [`TensorStorage`](generated/torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage),
[`LazyTensorStorage`](generated/torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage) and
[`LazyMemmapStorage`](generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage).

### Sampling and indexing

Replay buffers can be indexed and sampled.
Indexing and sampling collect data at given indices in the storage and then process them
through a series of transforms and `collate_fn` that can be passed to the __init__
function of the replay buffer.

The full physical storage can be read with `rb[:]`. This is useful when all
stored items must be processed in storage order, for example to recompute value
targets after collection. [`read_all_in_order()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.read_all_in_order)
is an explicit equivalent to `rb[:]`, and
[`write_all()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.write_all) is an explicit equivalent to
`rb[:] = data`. Passing `end=...` to these helpers updates only the leading
storage entries.

```
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
```

### Consuming replay buffers

Replay buffers can consume items as they are sampled by passing
`consume_after_n_samples`. This is useful in online loops where a collector
keeps writing new data while the trainer should avoid reusing old samples after
they have contributed to an update.

```
>>> import torch
>>> from torchrl.data import ListStorage, ReplayBuffer
>>> rb = ReplayBuffer(
... storage=ListStorage(8),
... batch_size=2,
... consume_after_n_samples=1,
... )
>>> rb.extend([torch.tensor(i) for i in range(3)])
tensor([0, 1, 2])
>>> batch = rb.sample()
>>> assert len(batch) == 2
>>> assert len(rb) == 1
>>> rb.extend([torch.tensor(3), torch.tensor(4)])
tensor([3, 4])
>>> assert len(rb) == 3
```

The consumed entries remain in physical storage until they are overwritten, but
they are removed from the sampleable set and are not returned by future calls to
[`sample()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample). New writes reuse consumed slots before
falling back to the writer's normal cursor, so consumed data behaves as freed
capacity without scanning the full storage on every write. This mode supports
1-dimensional `ListStorage`,
`TensorStorage`, `LazyTensorStorage` and `LazyMemmapStorage` with uniform
random sampling. Prefetching, prioritized replay and multidimensional storages
are rejected explicitly.

### Detecting overwritten slots: generation stamps

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
`track_generations=True`, the writer keeps one counter per storage slot and
advances it on every write to that slot. Comparing the stamp you captured
against the current stamp answers "is this still my data?":

```
>>> import torch
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.data import RoundRobinWriter
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(8),
... writer=RoundRobinWriter(track_generations=True),
... )
>>> _ = rb.extend(torch.arange(8))
>>> _, info = rb.sample(4, return_info=True)
>>> index, generation = info["index"], info["index_generation"]
>>> _ = rb.extend(torch.arange(8, 11)) # overwrites slots 0, 1, 2
>>> stale = rb.writer.generations_of(index) != generation
>>> # `index[stale]` no longer holds the sampled data
```

[`sample()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample) adds `"index_generation"` to its
`info` (and, for tensordict buffers, to the sample itself) whenever the writer
tracks generations, alongside the existing `"index"`.

#### Semantics

- **One stamp per write, not per ``extend`` call.** A single `extend` that
wraps the storage advances a reused slot once for each write it receives, so
a slot written twice in one call advances by two.
- **``-1`` means "no usable stamp"**: a never-written slot, an
out-of-range index, or a writer that does not track generations. It is not
"generation zero".
- **Monotonic across** [`empty()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.empty). Emptying
advances every written slot's stamp rather than resetting it, so handles taken
before the `empty()` correctly read as stale. Never-written slots keep
`-1`.
- **Stamps are for detection, not for ordering across slots.** Two slots'
stamps are independent counters; a higher stamp on slot 3 than on slot 7 says
nothing about write order between them.

#### Implementation notes

- **Opt-in.** The default is `track_generations=False`: enabling it allocates
one `int64` per storage slot and adds a key to the sampler output, neither
of which should be imposed on buffers that do not need it.
- **The counters live on the storage, not on the writer.** Two buffers sharing
one storage overwrite each other's slots, so a per-writer counter would let
one buffer's handles read as live after the other overwrote them. The buffer
is attached to the storage object, and a writer registered against a storage
that already has one adopts it rather than replacing it.
- **Allocation.** Storages small enough to allocate up front get a single
allocation, so the buffer's shape never changes and the `torch.compile`
extend/sample path does not recompile. Larger and unbounded storages
(`ListStorage` with no `max_size` reports `torch.iinfo(torch.int64).max`)
grow geometrically on demand instead.
- **Process-local.** The counters are not shared across processes: the buffer is
replaced rather than mutated when it grows, so a shared mapping would silently
stop tracking after the first growth. A slot overwritten by another process is
not reflected. Cross-process staleness detection needs a storage-owned,
fixed-size mapping and is not implemented yet.
- **Multidimensional storages.** A generation stamps a whole dim-0 slot. A 1-D
index tensor is therefore always read as a batch of slot indices; to identify
a single cell of an `ndim > 1` storage, pass the `tuple` of per-dimension
indices that [`extend()`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend) returns.
- **Checkpointing.** Stamps are part of `state_dict`/`dumps` when tracking
is on, and a checkpoint written without them (or by an older version) loads
fine - tracking simply starts from scratch.

The relevant APIs are `tracks_generations` and
`generations_of()`, and the `track_generations`
argument of `RoundRobinWriter`.

### Trajectory boundaries

Replay buffers store steps, not trajectories: components that need
trajectories ([`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) and its
variants, trajectory-aware transforms, offline dataset tooling) recover
episode boundaries at *read time* from markers present in the stored data.
The full producer/consumer contract -- which markers exist, who writes them,
how circular storage (wraparound, write cursor) interacts with boundary
recovery, and its blind spots -- is documented in
[Trajectory boundaries](data_layout.html#ref-traj-boundaries) on the data-layout page.
The associated APIs are:

| [`find_start_stop_traj`](generated/torchrl.data.find_start_stop_traj.html#torchrl.data.find_start_stop_traj)(*[, trajectory, end, ...]) | Recover trajectory boundaries from trajectory ids or end-of-trajectory flags. |
| --- | --- |

torchrl.data.DEFAULT_DONE_KEYS*= ("done", "truncated", "terminated")*

Canonical end-of-trajectory signal keys in TED format. A step can be
marked as the last of its trajectory by any of these entries (typically
read under the `"next"` sub-tensordict); `"done"` is the union of the
other two, but datasets sometimes carry only a subset of the entries, so
consumers detecting trajectory ends from flags should use the union of
all three. Shared default of [`TED2Flat`](generated/torchrl.data.TED2Flat.html#torchrl.data.TED2Flat),
`TED2Nested`, `MultiStep`
and `MultiStepTransform`; accepted by
[`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) through its
`end_keys` argument.

### TED-format conversion

The following helpers convert between the TorchRL Episode Data (TED) layout and
a flat, storage-friendly representation when serializing or restoring a buffer:

| [`TED2Flat`](generated/torchrl.data.TED2Flat.html#torchrl.data.TED2Flat)([done_key, shift_key, is_full_key, ...]) | A storage saving hook to serialize TED data in a compact format. |
| --- | --- |
| [`Flat2TED`](generated/torchrl.data.Flat2TED.html#torchrl.data.Flat2TED)([done_key, shift_key, is_full_key, ...]) | A storage loading hook to deserialize flattened TED data to TED format. |

### Video-backed replay buffers

Video-backed datasets are dominated by frames; materializing every decoded frame
as a dense tensor throws away the video codec's compression. [`VideoClipRef`](generated/torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef)
is a lightweight, picklable reference to frames inside an encoded video (mp4, ...):
it stores only *where* the frames are (the file(s) it spans plus a per-frame
`frame_index` and `file_id`), so indexing the whole buffer stays cheap. Frames
are decoded on-demand with
torchcodec by [`DecodeVideoTransform`](generated/torchrl.envs.transforms.DecodeVideoTransform.html#torchrl.envs.transforms.DecodeVideoTransform), appended on
the replay-buffer sample path, so `rb.sample()` returns decoded frames aligned to
the sampled steps. It composes with `SliceSampler`: a contiguous window of
sampled steps maps to consecutive frame indices and decodes as a single ranged
read. Decoders are opened lazily and cached per worker process (see
[`set_video_decoder_cache_size()`](generated/torchrl.data.set_video_decoder_cache_size.html#torchrl.data.set_video_decoder_cache_size) and [`clear_video_decoder_cache()`](generated/torchrl.data.clear_video_decoder_cache.html#torchrl.data.clear_video_decoder_cache)); the
references stored in the buffer never hold an open decoder.

**Temporal alignment / binning.** Video frames usually outnumber a lower-rate
signal (e.g. 100 frames for 30 proprioceptive steps). [`VideoClipRef.rebin()`](generated/torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef.rebin)
(also `VideoClipRef.from_file(..., num_bins=...)`) resamples the frames onto
`num_bins` non-overlapping temporal bins:

- `frames_per_bin=None` keeps one **center** frame per bin -> `[num_bins]`,
decoding to `[num_bins, C, H, W]` (subsample);
- `frames_per_bin=k` keeps `k` frames spanning each bin -> `[num_bins, k]`,
decoding to `[num_bins, k, C, H, W]` (a dense, non-overlapping stack; frames are
dropped/repeated to stay rectangular).

For *overlapping* (sliding-window) stacking, subsample first and then apply
[`CatFrames`](generated/torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames) to the decoded frames on the sample
path - `CatFrames` concatenates along an existing dim
(`[B, C, H, W] -> [B, N*C, H, W]`), giving classic frame-stacking with
trajectory-edge padding, while `rebin`'s stack keeps a separate frame axis:

```
>>> from torchrl.data import VideoClipRef, ReplayBuffer, LazyTensorStorage, SliceSampler
>>> from torchrl.envs.transforms import CatFrames, Compose, DecodeVideoTransform
>>> # one frame per step, then a sliding stack of the last 4 along the channel dim
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(1000),
... sampler=SliceSampler(slice_len=16, traj_key="episode"),
... transform=Compose(
... DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"]),
... CatFrames(N=4, dim=-3, in_keys=["pixels"]),
... ),
... )
```

**Multiple files.** A clip is often split across many small files (one per episode)
rather than one large mp4. [`VideoClipRef.from_files()`](generated/torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef.from_files) addresses a list of files
as a single logical sequence, so slicing, `rebin()` and decoding work across
file boundaries (a window that straddles two files decodes per file and
concatenates), with one cached decoder per file. No `LazyStacked` / `LazyCat`
container is needed - it is just a longer `frame_index` plus a per-frame
`file_id`. The index is stored compactly: the unique file paths live once in the
`sources` tuple and each frame carries a single `int64` `file_id` into it, so
references spanning thousands of files stay light on the replay-buffer sample path
(the resolved path is still available via the `VideoClipRef.source` property).

When camera and control loops run at different rates, prefer
[`VideoClipRef.from_timestamps()`](generated/torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef.from_timestamps) to align frames by time rather than by index.

| [`VideoClipRef`](generated/torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef)(sources[, frame_index, ...]) | |
| --- | --- |
| [`clear_video_decoder_cache`](generated/torchrl.data.clear_video_decoder_cache.html#torchrl.data.clear_video_decoder_cache)() | Closes and clears all cached torchcodec decoders in the current process. |
| [`set_video_decoder_cache_size`](generated/torchrl.data.set_video_decoder_cache_size.html#torchrl.data.set_video_decoder_cache_size)(maxsize) | Sets the maximum number of open torchcodec decoders cached per process. |