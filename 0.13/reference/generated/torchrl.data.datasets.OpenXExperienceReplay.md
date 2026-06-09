# OpenXExperienceReplay

*class*torchrl.data.datasets.OpenXExperienceReplay(*dataset_id*, *batch_size: int | None = None*, ***, *shuffle: bool = True*, *num_slices: int | None = None*, *slice_len: int | None = None*, *pad: float | bool | None = None*, *replacement: bool | None = None*, *streaming: bool | None = None*, *root: str | Path | None = None*, *download: bool | None = None*, *sampler: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler) | None = None*, *writer: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer) | None = None*, *collate_fn: Callable | None = None*, *pin_memory: bool = False*, *prefetch: int | None = None*, *transform: torchrl.envs.Transform | None = None*, *split_trajs: bool = False*, *strict_length: bool = True*)[[source]](../../_modules/torchrl/data/datasets/openx.html#OpenXExperienceReplay)

Open X-Embodiment datasets experience replay.

The Open X-Embodiment Dataset contains 1M+ real robot trajectories
spanning 22 robot embodiments, collected through a collaboration between
21 institutions, demonstrating 527 skills (160266 tasks).

Website: [https://robotics-transformer-x.github.io/](https://robotics-transformer-x.github.io/)

GitHub: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)

Paper: [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)

The data format follows the [TED convention](../data_datasets.html#ted-format).

Note

Non-tensor data will be written in the tensordict data using the
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) primitive.
For instance, the language_instruction field in the data will
be stored in data.get_non_tensor("language_instruction") (or equivalently
data.get("language_instruction").data). See the documentation of this
class for more information on how to interact with non-tensor data
stored in a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

Parameters:

- **dataset_id** (*str*) - The dataset to be downloaded.
Must be part of `OpenXExperienceReplay.available_datasets`.
- **batch_size** (*int*) - Batch-size used during sampling.
Can be overridden by data.sample(batch_size) if necessary.
See `num_slices` and `slice_len` keyword arguments for a refined
sampling strategy.
If the `batch_size` is `None` (default), iterating over the
dataset will deliver trajectories one at a time *whereas* calling
`sample()` will *still* require a batch-size to be provided.

Keyword Arguments:

- **shuffle** (*bool**,**optional*) -

if `True`, trajectories are delivered in a
random order when the dataset is iterated over.
If `False`, the dataset is iterated over in the pre-defined order.

Warning

shuffle=False will also impact the sampling. We advice users to
create a copy of the dataset where the `shuffle` attribute of the
sampler is set to `False` if they wish to enjoy the two different
behaviors (shuffled and not) within the same code base.
- **num_slices** (*int**,**optional*) - the number of slices in a batch. This
corresponds to the number of trajectories present in a batch.
Once collected, the batch is presented as a concatenation of
sub-trajectories that can be recovered through batch.reshape(num_slices, -1).
The batch_size must be divisible by num_slices if provided.
This argument is exclusive with `slice_len`.
If the `num_slices` argument equates the `batch_size`, each sample
will belong to a different trajectory.
If neither `slice_len` nor `num_slice` are provided:
whenever a trajectory has a length shorter than the
batch-size, a contiguous slice of it of length batch_size will be
sampled. If the trajectory length is insufficient, an exception will
be raised unless pad is not None.
- **slice_len** (*int**,**optional*) -

the length of slices in a batch. This
corresponds to the length of trajectories present in a batch.
Once collected, the batch is presented as a concatenation of
sub-trajectories that can be recovered through batch.reshape(-1, slice_len).
The batch_size must be divisible by slice_len if provided.
This argument is exclusive with `num_slice`.
If the `slice_len` argument equates `1`, each sample
will belong to a different trajectory.
If neither `slice_len` nor `num_slice` are provided:
whenever a trajectory has a length shorter than the
batch-size, a contiguous slice of it of length batch_size will be
sampled. If the trajectory length is insufficient, an exception will
be raised unless pad is not None.

Note

The `slice_len` (but not `num_slices`) can be used when
iterating over a dataset without passing a batch-size in the,
constructor. In these cases, a random sub-sequence of the
trajectory will be chosen.
- **replacement** (*bool**,**optional*) - if `False`, sampling will be done
without replacement. Defaults to `True` for downloaded datasets,
`False` for streamed datasets.
- **pad** (bool, `float` or None) - if `True`, trajectories of insufficient length
given the slice_len or num_slices arguments will be padded with
0s. If another value is provided, it will be used for padding. If
`False` or `None` (default) any encounter with a trajectory of
insufficient length will raise an exception.
- **root** (*Path**or**str**,**optional*) - The OpenX dataset root directory.
The actual dataset memory-mapped files will be saved under
<root>/<dataset_id>. If none is provided, it defaults to
~/.cache/torchrl/atari.openx`.
- **streaming** (*bool**,**optional*) -

if `True`, the data won't be downloaded but
read from a stream instead.

Note

The formatting of the data **will change** when download=True
compared to streaming=True. If the data is downloaded and
the sampler is left untouched (ie, num_slices=None, slice_len=None
and sampler=None, transitions will be sampled randomly from
the dataset. This isn't possible at a reasonable cost with
streaming=True: in this case, trajectories will be sampled
one at a time and delivered as such (with cropping to comply with
the batch-size etc). The behavior of the two modalities is
much more similar when num_slices and slice_len are specified,
as in these cases, views of sub-episodes will be returned in both
cases.
- **download** (*bool**or**str**,**optional*) - Whether the dataset should be downloaded if
not found. Defaults to `True`. Download can also be passed as "force",
in which case the downloaded data will be overwritten.
- **sampler** ([*Sampler*](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*,**optional*) - the sampler to be used. If none is provided
a default RandomSampler() will be used.
- **writer** ([*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*,**optional*) - the writer to be used. If none is provided
a default [`ImmutableDatasetWriter`](torchrl.data.replay_buffers.ImmutableDatasetWriter.html#torchrl.data.replay_buffers.ImmutableDatasetWriter) will be used.
- **collate_fn** (*callable**,**optional*) - merges a list of samples to form a
mini-batch of Tensor(s)/outputs. Used when using batched
loading from a map-style dataset.
- **pin_memory** (*bool*) - whether pin_memory() should be called on the rb
samples.
- **prefetch** (*int**,**optional*) - number of next batches to be prefetched
using multithreading.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*,**optional*) - Transform to be executed when sample() is called.
To chain transforms use the `Compose` class.
- **split_trajs** (*bool**,**optional*) - if `True`, the trajectories will be split
along the first dimension and padded to have a matching shape.
To split the trajectories, the `"done"` signal will be used, which
is recovered via `done = truncated | terminated`. In other words,
it is assumed that any `truncated` or `terminated` signal is
equivalent to the end of a trajectory.
Defaults to `False`.
- **strict_length** (*bool**,**optional*) - if `False`, trajectories of length
shorter than slice_len (or batch_size // num_slices) will be
allowed to appear in the batch.
Be mindful that this can result in effective batch_size shorter
than the one asked for! Trajectories can be split using
`torchrl.collectors.split_trajectories()`. Defaults to `True`.

Examples

```
>>> from torchrl.data.datasets import OpenXExperienceReplay
>>> import tempfile
>>> # Download the data, and sample 128 elements in each batch out of two trajectories
>>> num_slices = 2
>>> with tempfile.TemporaryDirectory() as root:
... dataset = OpenXExperienceReplay("cmu_stretch", batch_size=128,
... num_slices=num_slices, download=True, streaming=False,
... root=root,
... )
... for batch in dataset:
... print(batch.reshape(num_slices, -1))
... break
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 64, 8]), device=cpu, dtype=torch.float64, is_shared=False),
 discount: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 episode: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int32, is_shared=False),
 index: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False),
 is_init: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.bool, is_shared=False),
 language_embedding: Tensor(shape=torch.Size([2, 64, 512]), device=cpu, dtype=torch.float64, is_shared=False),
 language_instruction: NonTensorData(
 data='lift open green garbage can lid',
 batch_size=torch.Size([2, 64]),
 device=cpu,
 is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: TensorDict(
 fields={
 image: Tensor(shape=torch.Size([2, 64, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
 state: Tensor(shape=torch.Size([2, 64, 4]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([2, 64]),
 device=cpu,
 is_shared=False),
 reward: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2, 64]),
 device=cpu,
 is_shared=False),
 observation: TensorDict(
 fields={
 image: Tensor(shape=torch.Size([2, 64, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
 state: Tensor(shape=torch.Size([2, 64, 4]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([2, 64]),
 device=cpu,
 is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2, 64]),
 device=cpu,
 is_shared=False)
>>> # Read data from a stream. Deliver entire trajectories when iterating
>>> dataset = OpenXExperienceReplay("cmu_stretch",
... num_slices=num_slices, download=False, streaming=True)
>>> for data in dataset: # data does not have a consistent shape
... break
>>> # Define batch-size dynamically
>>> data = dataset.sample(128) # delivers 2 sub-trajectories of length 64
```

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

*property*data_path

Path to the dataset, including split.

*property*data_path_root

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

Args and Keyword Args are forwarded to [`map()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.map).

The dataset can subsequently be deleted using `delete()`.

Keyword Arguments:

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

Returns:

A tensordict containing a batch of data selected in the replay buffer.
A tuple containing this tensordict and info if return_info flag is set to True.

*property*sampler*: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*

The sampler of the replay buffer.

The sampler must be an instance of [`Sampler`](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler).

save(**args*, ***kwargs*)

Alias for `dumps()`.

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