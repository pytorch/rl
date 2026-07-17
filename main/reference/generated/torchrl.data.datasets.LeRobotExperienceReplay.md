# LeRobotExperienceReplay

*class*torchrl.data.datasets.LeRobotExperienceReplay(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/lerobot.html#LeRobotExperienceReplay)

Experience replay over a [LeRobot](https://github.com/huggingface/lerobot) dataset.

LeRobot is the de-facto open format for robot-learning datasets (Parquet for
state/action + MP4 for video), hosting many community datasets and the data
used to train SmolVLA / pi0 / ACT. This adapter maps a LeRobot dataset into
the canonical VLA TensorDict schema and serves it as a TorchRL replay buffer
with trajectory-aware slice sampling.

There are three ways to build it:

- `LeRobotExperienceReplay(repo_id, download=True)` downloads the hub
snapshot and reads the on-disk LeRobot format (v2.x and v3.x) directly -
only the `huggingface_hub` and `datasets` packages are required
(installed by the `vla` extra), not the `lerobot` package itself;
- `LeRobotExperienceReplay(repo_id, root=..., download=False)` loads a
previously-converted memory-mapped copy from disk;
- `from_columns()` builds directly from an in-memory LeRobot-style
columnar dict (no download), which is also the path used in tests.

Parameters:

**repo_id** (*str*) - the Hugging Face dataset repo id (e.g.
`"lerobot/aloha_sim_insertion_human"`).

Keyword Arguments:

- **root** (*str**or**Path**,**optional*) - local cache root. Defaults to the TorchRL
LeRobot cache directory.
- **download** (*bool*) - whether to download+convert the dataset if it is not
already cached. Defaults to `True`.
- **batch_size** (*int**,**optional*) - the batch size for sampling.
- **num_slices** (*int**,**optional*) - number of trajectory slices per batch
(exclusive with `slice_len`).
- **slice_len** (*int**,**optional*) - length of each trajectory slice.
- **sampler** ([*Sampler*](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*,**optional*) - a custom sampler. Defaults to a
`SliceSampler` over the (key-mapped) episode
key - `episode` unless `key_map` remaps `episode_index` -
when `num_slices`/`slice_len` is given.
- **writer** ([*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*,**optional*) - a custom writer.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*,**optional*) - a transform applied on sampling.
- **key_map** (*dict**,**optional*) - overrides the default LeRobot-to-canonical key
mapping (see [`lerobot_columns_to_tensordict()`](torchrl.data.datasets.lerobot_columns_to_tensordict.html#torchrl.data.datasets.lerobot_columns_to_tensordict)).
- **decode_video** (*bool*) - if `True` (default) and the dataset carries lazy
[`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef) video columns, a
[`DecodeVideoTransform`](torchrl.envs.transforms.DecodeVideoTransform.html#torchrl.envs.transforms.DecodeVideoTransform) is appended so
that `sample()` returns decoded frames (requires `torchcodec`).
Set to `False` to keep the raw references and decode them yourself.
- **rehydrate** (*bool*) - if `True`, sampled batches are made fully
TED-compliant by re-hydrating `("next", "observation", ...)`
entries from the following row of each sampled slice
([`NextStateReconstructor`](torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor)
instances are appended after the video decode and before
`transform`). Boundaries are detected from the episode id
(required) plus the per-episode frame counter (`frame`) when
present, so positions whose in-batch successor is not the true
next step - slice ends and splices between back-to-back slices -
are filled with `NaN` for floating-point leaves and `0` for
integer leaves (e.g. decoded `uint8` frames); mask the filled
positions with the slice sampler's `("next", "truncated")` flag
in the returned batch when consuming `next`. Video references
left undecoded (`decode_video=False` or `torchcodec` not
installed) are skipped with a warning. Defaults to `False`.
- **strict_length** (*bool*) - passed to the slice sampler. Defaults to `True`.
- **collate_fn** (*Callable**,**optional*) - merges samples; defaults to the
identity collation used by offline datasets.
- **pin_memory** (*bool*) - whether to pin memory on sampling. Defaults to `False`.
- **prefetch** (*int**,**optional*) - number of batches to prefetch with a background
thread.

Note

Sampled batches are *flat* `[num_slices * slice_len]` like any
`SliceSampler` output; reshape to
`[num_slices, slice_len, ...]` before applying
[`ActionChunkTransform`](torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform).

Note

MP4 video columns are loaded lazily as [`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef)
leaves - no frames are materialized in storage. With `decode_video=True`
(the default) they are decoded on `sample()` via
[`DecodeVideoTransform`](torchrl.envs.transforms.DecodeVideoTransform.html#torchrl.envs.transforms.DecodeVideoTransform) (requires
`torchcodec`).

Warning

The `download=True` path reads the documented LeRobot on-disk format
(validated against `lerobot/pusht`, format `v3.0`) but is **not
exercised in CI** (`huggingface_hub`/`datasets` are optional
dependencies and CI does not download datasets). For fully
reproducible behavior, build offline via `from_columns()`.

Examples

```
>>> import torch
>>> from torchrl.data.datasets import LeRobotExperienceReplay
>>> columns = {
... "observation.state": torch.zeros(8, 7),
... "action": torch.zeros(8, 7),
... "episode_index": torch.arange(2).repeat_interleave(4),
... "task": ["pick"] * 8,
... }
>>> rb = LeRobotExperienceReplay.from_columns(
... columns, slice_len=4, batch_size=8
... )
>>> sample = rb.sample()
>>> sample["action"].shape
torch.Size([8, 7])
>>> # rehydrate=True re-hydrates ("next", "observation", ...) from the
>>> # following row of each slice (slice ends are filled and flagged
>>> # by ("next", "truncated"))
>>> rb = LeRobotExperienceReplay.from_columns(
... columns, slice_len=4, batch_size=8, rehydrate=True
... )
>>> sample = rb.sample()
>>> sample["next", "observation", "state"].shape
torch.Size([8, 7])
```

See also

[`OpenXExperienceReplay`](torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay) for the
Open X-Embodiment equivalent.

add(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → int

Add a single element to the replay buffer.

Parameters:

**data** (*Any*) - data to be added to the replay buffer

Returns:

index where the data lives in the replay buffer.

append_transform(*transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*, ***, *invert: bool = False*) → [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)

Appends transform at the end.

Transforms are applied in order when sample is called.

Parameters:

**transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)) - The transform to be appended

Keyword Arguments:

**invert** (*bool**,**optional*) - if `True`, the transform will be inverted (forward calls will be called
during writing and inverse calls during reading). Defaults to `False`.

Example

```
>>> rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
>>> data = TensorDict({"a": torch.zeros(10)}, [10])
>>> def t(data):
... data += 1
... return data
>>> rb.append_transform(t, invert=True)
>>> rb.extend(data)
>>> assert (data == 1).all()
```

*classmethod*as_remote(*remote_config=None*)

Creates an instance of a remote ray class.

Parameters:

- **cls** (*Python Class*) - class to be remotely instantiated.
- **remote_config** (*dict*) - the quantity of CPU cores to reserve for this class.
Defaults to torchrl.collectors.distributed.ray.DEFAULT_REMOTE_CLASS_CONFIG.

Returns:

A function that creates ray remote class instances.

*property*batch_size

The batch size of the replay buffer.

The batch size can be overridden by setting the batch_size parameter in the `sample()` method.

It defines both the number of samples returned by `sample()` and the number of samples that are
yielded by the `ReplayBuffer` iterator.

client() → T

Return `self` for the zero-overhead direct backend.

*property*data_path*: Path*

Path to the dataset, including split.

*property*data_path_root*: Path*

Path to the dataset root.

delete()

Deletes a dataset storage from disk.

dump(**args*, ***kwargs*)

Alias for `dumps()`.

dumps(*path*)

Saves the replay buffer on disk at the specified path.

Parameters:

**path** (*Path**or**str*) - path where to save the replay buffer.

Examples

```
>>> import tempfile
>>> import tqdm
>>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
>>> from torchrl.data.replay_buffers.samplers import PrioritizedSampler, RandomSampler
>>> import torch
>>> from tensordict import TensorDict
>>> # Build and populate the replay buffer
>>> S = 1_000_000
>>> sampler = PrioritizedSampler(S, 1.1, 1.0)
>>> # sampler = RandomSampler()
>>> storage = LazyMemmapStorage(S)
>>> rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)
>>>
>>> for _ in tqdm.tqdm(range(100)):
... td = TensorDict({"obs": torch.randn(100, 3, 4), "next": {"obs": torch.randn(100, 3, 4)}, "td_error": torch.rand(100)}, [100])
... rb.extend(td)
... sample = rb.sample(32)
... rb.update_tensordict_priority(sample)
>>> # save and load the buffer
>>> with tempfile.TemporaryDirectory() as tmpdir:
... rb.dumps(tmpdir)
...
... sampler = PrioritizedSampler(S, 1.1, 1.0)
... # sampler = RandomSampler()
... storage = LazyMemmapStorage(S)
... rb_load = TensorDictReplayBuffer(storage=storage, sampler=sampler)
... rb_load.loads(tmpdir)
... assert len(rb) == len(rb_load)
```

empty(*empty_write_count: bool = True*)

Empties the replay buffer and reset cursor to 0.

Parameters:

**empty_write_count** (*bool**,**optional*) - Whether to empty the write_count attribute. Defaults to True.

extend(*tensordicts: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *update_priority: bool | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Extends the replay buffer with a batch of data.

Parameters:

**tensordicts** (*TensorDictBase*) - The data to extend the replay buffer with.

Keyword Arguments:

**update_priority** (*bool**,**optional*) - Whether to update the priority of the data. Defaults to True.

Returns:

The indices of the data that were added to the replay buffer.

*classmethod*from_columns(*columns: dict[str, Any]*, ***, *repo_id: str = 'local'*, *key_map: dict[str, NestedKey] | None = None*, ***kwargs*) → LeRobotExperienceReplay[[source]](../../_modules/torchrl/data/datasets/lerobot.html#LeRobotExperienceReplay.from_columns)

Build directly from an in-memory LeRobot-style columnar dict.

Converts `columns` with [`lerobot_columns_to_tensordict()`](torchrl.data.datasets.lerobot_columns_to_tensordict.html#torchrl.data.datasets.lerobot_columns_to_tensordict) and
wraps the result in an in-memory storage - no download or `lerobot`
install required.

*property*initialized*: bool*

Whether the replay buffer has been initialized.

insert_transform(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*, ***, *invert: bool = False*) → [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)

Inserts transform.

Transforms are executed in order when sample is called.

Parameters:

- **index** (*int*) - Position to insert the transform.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)) - The transform to be appended

Keyword Arguments:

**invert** (*bool**,**optional*) - if `True`, the transform will be inverted (forward calls will be called
during writing and inverse calls during reading). Defaults to `False`.

*property*is_alive*: bool*

Whether this direct replay buffer remains available.

load(**args*, ***kwargs*)

Alias for `loads()`.

loads(*path*)

Loads a replay buffer state at the given path.

The buffer should have matching components and be saved using `dumps()`.

Parameters:

**path** (*Path**or**str*) - path where the replay buffer was saved.

See `dumps()` for more info.

next()

Returns the next item in the replay buffer.

This method is used to iterate over the replay buffer in contexts where __iter__ is not available,
such as `RayReplayBuffer`.

preprocess(*fn: Callable[[TensorDictBase], TensorDictBase]*, *dim: int = 0*, *num_workers: int | None = None*, ***, *chunksize: int | None = None*, *num_chunks: int | None = None*, *pool: mp.Pool | None = None*, *generator: [torch.Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | None = None*, *max_tasks_per_child: int | None = None*, *worker_threads: int = 1*, *index_with_generator: bool = False*, *pbar: bool = False*, *mp_start_method: str | None = None*, *num_frames: int | None = None*, *dest: str | Path*) → [TensorStorage](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage)

Preprocesses a dataset and returns a new storage with the formatted data.

The data transform must be unitary (work on a single sample of the dataset).

The dataset can subsequently be deleted using `delete()`.

Parameters:

- **fn** (*Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - transform to apply to each sample.
- **dim** (*int**,**optional*) - dimension along which the dataset is mapped. Defaults to `0`.
- **num_workers** (*int**,**optional*) - number of worker processes to use.
Defaults to `None`.

Keyword Arguments:

- **chunksize** (*int**,**optional*) - chunk size forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **num_chunks** (*int**,**optional*) - number of chunks forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **pool** (*multiprocessing.Pool**,**optional*) - worker pool forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) - random generator forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **max_tasks_per_child** (*int**,**optional*) - maximum number of tasks per child
process forwarded to [`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **worker_threads** (*int**,**optional*) - number of threads per worker forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map). Defaults to `1`.
- **index_with_generator** (*bool**,**optional*) - whether to index with the generator
when mapping. Defaults to `False`.
- **pbar** (*bool**,**optional*) - whether to display a progress bar. Defaults to
`False`.
- **mp_start_method** (*str**,**optional*) - multiprocessing start method forwarded to
[`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).
- **dest** (*path**or**equivalent*) - a path to the location of the new dataset.
- **num_frames** (*int**,**optional*) - if provided, only the first num_frames will be
transformed. This is useful to debug the transform at first.

Returns: A new storage to be used within a [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) instance.

Examples

```
>>> from torchrl.data.datasets import MinariExperienceReplay
>>>
>>> data = MinariExperienceReplay(
... list(MinariExperienceReplay.available_datasets)[0],
... batch_size=32
... )
>>> print(data)
MinariExperienceReplay(
 storages=TensorStorage(TensorDict(
 fields={
 action: MemoryMappedTensor(shape=torch.Size([1000000, 8]), device=cpu, dtype=torch.float32, is_shared=True),
 episode: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.int64, is_shared=True),
 info: TensorDict(
 fields={
 distance_from_origin: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 forward_reward: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 qpos: MemoryMappedTensor(shape=torch.Size([1000000, 15]), device=cpu, dtype=torch.float64, is_shared=True),
 qvel: MemoryMappedTensor(shape=torch.Size([1000000, 14]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_ctrl: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_forward: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_survive: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 success: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.bool, is_shared=True),
 x_position: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 x_velocity: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 y_position: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 y_velocity: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 next: TensorDict(
 fields={
 done: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 info: TensorDict(
 fields={
 distance_from_origin: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 forward_reward: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 qpos: MemoryMappedTensor(shape=torch.Size([1000000, 15]), device=cpu, dtype=torch.float64, is_shared=True),
 qvel: MemoryMappedTensor(shape=torch.Size([1000000, 14]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_ctrl: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_forward: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 reward_survive: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 success: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.bool, is_shared=True),
 x_position: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 x_velocity: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 y_position: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True),
 y_velocity: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float64, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 observation: TensorDict(
 fields={
 achieved_goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 desired_goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 observation: MemoryMappedTensor(shape=torch.Size([1000000, 27]), device=cpu, dtype=torch.float64, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 reward: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.float64, is_shared=True),
 terminated: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 truncated: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 observation: TensorDict(
 fields={
 achieved_goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 desired_goal: MemoryMappedTensor(shape=torch.Size([1000000, 2]), device=cpu, dtype=torch.float64, is_shared=True),
 observation: MemoryMappedTensor(shape=torch.Size([1000000, 27]), device=cpu, dtype=torch.float64, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False)),
 samplers=RandomSampler,
 writers=ImmutableDatasetWriter(),
batch_size=32,
transform=Compose(
),
collate_fn=<function _collate_id at 0x120e21dc0>)
>>> from torchrl.envs import CatTensors, Compose
>>> from tempfile import TemporaryDirectory
>>>
>>> cat_tensors = CatTensors(
... in_keys=[("observation", "observation"), ("observation", "achieved_goal"),
... ("observation", "desired_goal")],
... out_key="obs"
... )
>>> cat_next_tensors = CatTensors(
... in_keys=[("next", "observation", "observation"),
... ("next", "observation", "achieved_goal"),
... ("next", "observation", "desired_goal")],
... out_key=("next", "obs")
... )
>>> t = Compose(cat_tensors, cat_next_tensors)
>>>
>>> def func(td):
... td = td.select(
... "action",
... "episode",
... ("next", "done"),
... ("next", "observation"),
... ("next", "reward"),
... ("next", "terminated"),
... ("next", "truncated"),
... "observation"
... )
... td = t(td)
... return td
>>> with TemporaryDirectory() as tmpdir:
... new_storage = data.preprocess(func, num_workers=4, pbar=True, mp_start_method="fork", dest=tmpdir)
... rb = ReplayBuffer(storage=new_storage)
... print(rb)
ReplayBuffer(
 storage=TensorStorage(
 data=TensorDict(
 fields={
 action: MemoryMappedTensor(shape=torch.Size([1000000, 8]), device=cpu, dtype=torch.float32, is_shared=True),
 episode: MemoryMappedTensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.int64, is_shared=True),
 next: TensorDict(
 fields={
 done: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 obs: MemoryMappedTensor(shape=torch.Size([1000000, 31]), device=cpu, dtype=torch.float64, is_shared=True),
 observation: TensorDict(
 fields={
 },
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 reward: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.float64, is_shared=True),
 terminated: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True),
 truncated: MemoryMappedTensor(shape=torch.Size([1000000, 1]), device=cpu, dtype=torch.bool, is_shared=True)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 obs: MemoryMappedTensor(shape=torch.Size([1000000, 31]), device=cpu, dtype=torch.float64, is_shared=True),
 observation: TensorDict(
 fields={
 },
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([1000000]),
 device=cpu,
 is_shared=False),
 shape=torch.Size([1000000]),
 len=1000000,
 max_size=1000000),
 sampler=RandomSampler(),
 writer=RoundRobinWriter(cursor=0, full_storage=True),
 batch_size=None,
 collate_fn=<function _collate_id at 0x168406fc0>)
```

query(*predicate: Callable[[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)], bool] | None = None*, ***, *trajectory_key: NestedKey | None = None*) → list[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)]

Filters the stored trajectories with a query predicate.

Splits the buffer content into trajectories (see
`iter_trajectories()`) and
returns those matching the predicate as
[`Trajectory`](torchrl.data.Trajectory.html#torchrl.data.Trajectory) views.

Parameters:

**predicate** (*Callable**[**[*[*Trajectory*](torchrl.data.Trajectory.html#torchrl.data.Trajectory)*]**,**bool**]**,**optional*) - a
[`TrajectoryPredicate`](torchrl.data.TrajectoryPredicate.html#torchrl.data.TrajectoryPredicate)
built from `traj`, or
any callable mapping a trajectory to a boolean. Defaults to
None (return all trajectories).

Keyword Arguments:

**trajectory_key** (*NestedKey**,**optional*) - entry holding
per-transition trajectory ids. Defaults to None
(auto-detection from `("collector", "traj_ids")`,
`"traj_ids"`, `"episode"` or the done/terminated/truncated
flags).

Returns:

A list of matching trajectory views, ordered chronologically
(oldest trajectory first; for multi-dimensional storages, grouped
by batch coordinate).

The trajectory boundaries are computed from the stored (untransformed)
data with the same machinery
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) uses, so
samplers and queries always agree on where trajectories start and
stop. This includes storages with `ndim > 1` (e.g.
`LazyTensorStorage(..., ndim=2)` holding `[B, T]` batches), whose
trajectories are recovered per batch coordinate.

Predicates built from `traj`
report the keys they read via
`required_keys()`;
evaluation then only fetches those entries from the storage and only
runs the transforms that can affect them. Matching trajectories are
extracted in full with the complete transform chain applied, so
predicates and results see the same values a sampler would produce.
Opaque callables are evaluated against the fully transformed content.

Note

Once the buffer has wrapped around (it is at capacity and older
entries have been overwritten), the oldest trajectory may have
lost its first transitions to overwriting and will appear
truncated at the front. A trajectory written across the wrap
point is followed through it and returned whole, in time order.

Examples

```
>>> from torchrl.data import traj
>>> good_trajs = rb.query((traj.reward.sum() > 100) & (traj.length >= 50))
>>> observations = good_trajs[0].observation
```

read_all_in_order(*end: int | None = None*) → Any

Read storage contents in physical order.

This is equivalent to `rb[:]` when `end` is `None`.

Parameters:

**end** (*int**,**optional*) - Number of leading storage entries to read.
Defaults to the entire storage slice.

Returns:

A storage slice containing entries `[:end]`.

register_load_hook(*hook: Callable[[Any], Any]*)

Registers a load hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

register_save_hook(*hook: Callable[[Any], Any]*)

Registers a save hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

sample(*batch_size: int | None = None*, *return_info: bool = False*, *include_info: bool | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Samples a batch of data from the replay buffer.

Uses Sampler to sample indices, and retrieves them from Storage.

Parameters:

- **batch_size** (*int**,**optional*) - size of data to be collected. If none
is provided, this method will sample a batch-size as indicated
by the sampler.
- **return_info** (*bool*) - whether to return info. If True, the result
is a tuple (data, info). If False, the result is the data.
- **include_info** (*bool**,**optional*) - deprecated alias for `return_info`.

Returns:

A tensordict containing a batch of data selected in the replay buffer.
A tuple containing this tensordict and info if return_info flag is set to True.

*property*sampler*: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*

The sampler of the replay buffer.

The sampler must be an instance of [`Sampler`](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler).

save(**args*, ***kwargs*)

Alias for `dumps()`.

*property*service_backend*: str*

The canonical deployment backend for this replay buffer.

set_(*key*, *value*)

Sets the value of a key across the entire replay buffer in-place.

Parameters:

- **key** (*NestedKey*) - the key to set.
- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the value to write.

Returns:

self

set_at_(*key*, *value*, *index*)

Sets the value of a key at specified indices in the replay buffer.

Parameters:

- **key** (*NestedKey*) - the key to set.
- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the value to write.
- **index** - the indices where to write the value.

Returns:

self

set_sampler(*sampler: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*)

Sets a new sampler in the replay buffer and returns the previous sampler.

set_storage(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *collate_fn: Callable | None = None*)

Sets a new storage in the replay buffer and returns the previous storage.

Parameters:

- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the new storage for the buffer.
- **collate_fn** (*callable**,**optional*) - if provided, the collate_fn is set to this
value. Otherwise it is reset to a default value.

set_writer(*writer: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*)

Sets a new writer in the replay buffer and returns the previous writer.

shutdown(*timeout: float | None = None*) → None

Mark this direct replay-buffer owner as shut down.

start() → T

Return this already-started direct replay buffer.

stats() → dict[str, int | float | bool]

Returns a cheap, serializable snapshot of the buffer's operational state.

The snapshot only contains scalar counters and gauges. It never
includes the storage content, does not modify the buffer state and is
safe to call concurrently with writes and samples. Cumulative
counters such as `write_count` are meant to be converted into rates
by an external monitor such as
[`LoggerMonitor`](torchrl.record.loggers.monitoring.LoggerMonitor.html#torchrl.record.loggers.monitoring.LoggerMonitor).

Calling this method on an uninitialized buffer does not trigger its
initialization; an empty snapshot with `initialized=False` is
returned instead (`capacity` is still reported when the storage
already advertises it).

Returns:

- `"size"`: current number of elements in the buffer (mirrors `len(buffer)`);
- `"write_count"`: total number of items written through `add` and
`extend` (`0` for writers that do not track writes, such as
[`ImmutableDatasetWriter`](torchrl.data.replay_buffers.ImmutableDatasetWriter.html#torchrl.data.replay_buffers.ImmutableDatasetWriter));
- `"prefetch_queue_size"`: number of pending prefetched batches;
- `"initialized"`: whether the buffer components are initialized;
- `"capacity"`: maximum number of elements the storage can hold
(only present when the storage advertises a `max_size`);
- `"utilization"`: `size / capacity` (only present alongside `capacity`).

Remote clients backed by the distributed transport report a subset
of these entries (`size` and `write_count`).

Return type:

A dictionary with the following entries

Examples

```
>>> import torch
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> rb = ReplayBuffer(storage=LazyTensorStorage(10))
>>> rb.extend(torch.arange(5))
>>> snapshot = rb.stats()
>>> print(snapshot["size"], snapshot["write_count"], snapshot["capacity"])
5 5 10
```

*property*storage*: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*

The storage of the replay buffer.

The storage must be an instance of [`Storage`](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage).

*property*transform*: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*

The transform of the replay buffer.

The transform must be an instance of [`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform).

update_(*input_dict_or_td*, *clone=False*, ***, *keys_to_update=None*)

Updates the replay buffer in-place with the given dict or TensorDict.

Parameters:

- **input_dict_or_td** (*dict**or**TensorDictBase*) - the data to update with.
- **clone** (*bool**,**optional*) - whether to clone the values before writing.
Defaults to `False`.
- **keys_to_update** (*sequence**of**NestedKey**,**optional*) - if provided, only
these keys will be updated.

Returns:

self

write_all(*data: Any*, *end: int | None = None*) → None

Write data back to storage in physical order.

This is equivalent to `rb[:end] = data`. If `end` is `None`,
`end` defaults to `data.shape[0]` for tensor collections and
`len(data)` otherwise. If `data` spans the full storage, this is
equivalent to `rb[:] = data`.

Parameters:

- **data** - Data to write to storage.
- **end** (*int**,**optional*) - Number of leading storage entries to update.
Defaults to `data.shape[0]` for tensor collections and
`len(data)` otherwise.

*property*write_count*: int*

The total number of items written so far in the buffer through add and extend.

*property*writer*: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*

The writer of the replay buffer.

The writer must be an instance of [`Writer`](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer).