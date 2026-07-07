# VD4RLExperienceReplay

*class*torchrl.data.datasets.VD4RLExperienceReplay(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/vd4rl.html#VD4RLExperienceReplay)

V-D4RL experience replay dataset.

This class downloads the H5/npz data from V-D4RL and processes it in a mmap
format, which makes indexing (and therefore sampling) faster.

Learn more about V-D4RL here: [https://arxiv.org/abs/2206.04779](https://arxiv.org/abs/2206.04779)

The "pixels" entry is located at the root of the data, and all the data
that is not reward, done-state, action or pixels is moved under a "state"
node.

The data format follows the [TED convention](../data_datasets.html#ted-format).

Parameters:

- **dataset_id** (*str*) - the dataset to be downloaded. Must be part of
VD4RLExperienceReplay.available_datasets.
- **batch_size** (*int*) - Batch-size used during sampling. Can be overridden by
data.sample(batch_size) if necessary.

Keyword Arguments:

- **root** (*Path**or**str**,**optional*) - The V-D4RL dataset root directory.
The actual dataset memory-mapped files will be saved under
<root>/<dataset_id>. If none is provided, it defaults to
~/.cache/torchrl/atari.vd4rl`.
- **download** (*bool**or**str**,**optional*) - Whether the dataset should be downloaded if
not found. Defaults to `True`. Download can also be passed as `"force"`,
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
equivalent to the end of a trajectory. For some datasets from
`D4RL`, this may not be true. It is up to the user to make
accurate choices regarding this usage of `split_trajs`.
Defaults to `False`.
- **totensor** (*bool**,**optional*) - if `True`, a [`ToTensorImage`](torchrl.envs.transforms.ToTensorImage.html#torchrl.envs.transforms.ToTensorImage)
transform will be included in the transform list (if not automatically
detected). Defaults to `True`.
- **image_size** (*int**,*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**ints**or**None*) - if not `None`, this argument
will be used to create a [`Resize`](torchrl.envs.transforms.Resize.html#torchrl.envs.transforms.Resize)
transform that will be appended to the transform list. Supports
int types (square resizing) or a list/tuple of int (rectangular
resizing). Defaults to `None` (no resizing).
- **num_workers** (*int**,**optional*) - the number of workers to download the files.
Defaults to `0` (no multiprocessing).

Variables:

**available_datasets** - a list of accepted entries to be downloaded. These
names correspond to the directory path in the huggingface dataset
repository. If possible, the list will be dynamically retrieved from
huggingface. If no internet connection is available, it a cached
version will be used.

Note

Since not all experience replay have start and stop signals, we
do not mark the episodes in the retrieved dataset.

Examples

```
>>> import torch
>>> torch.manual_seed(0)
>>> from torchrl.data.datasets import VD4RLExperienceReplay
>>> d = VD4RLExperienceReplay("main/walker_walk/random/64px", batch_size=32,
... image_size=50)
>>> for batch in d:
... break
>>> print(batch)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 index: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
 is_init: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: TensorDict(
 fields={
 height: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False),
 orientations: Tensor(shape=torch.Size([32, 14]), device=cpu, dtype=torch.float32, is_shared=False),
 velocity: Tensor(shape=torch.Size([32, 9]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([32, 3, 50, 50]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 observation: TensorDict(
 fields={
 height: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False),
 orientations: Tensor(shape=torch.Size([32, 14]), device=cpu, dtype=torch.float32, is_shared=False),
 velocity: Tensor(shape=torch.Size([32, 9]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([32, 3, 50, 50]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32]),
 device=cpu,
 is_shared=False)
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

client() → T

Return `self` for the zero-overhead direct backend.

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