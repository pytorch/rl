# AtariDQNExperienceReplay

*class*torchrl.data.datasets.AtariDQNExperienceReplay(*dataset_id: str*, *batch_size: int | None = None*, ***, *root: str | Path | None = None*, *download: bool | str = True*, *sampler=None*, *writer=None*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | None = None*, *num_procs: int = 0*, *num_slices: int | None = None*, *slice_len: int | None = None*, *strict_len: bool = True*, *replacement: bool = True*, *mp_start_method: str = 'fork'*, ***kwargs*)[[source]](../../_modules/torchrl/data/datasets/atari_dqn.html#AtariDQNExperienceReplay)

Atari DQN Experience replay class.

The Atari DQN dataset ([https://offline-rl.github.io/](https://offline-rl.github.io/)) is a collection of 5 training
iterations of DQN over each of the Arari 2600 games for a total of 200 million frames.
The sub-sampling rate (frame-skip) is equal to 4, meaning that each game dataset
has 50 million steps in total.

The data format follows the [TED convention](../data_datasets.html#ted-format). Since the dataset is quite heavy,
the data formatting is done on-line, at sampling time.

To make training more modular, we split the dataset in each of the Atari games
and separate each training round. Consequently, each dataset is presented as
a Storage of length 50x10^6 elements. Under the hood, this dataset is split
in 50 memory-mapped tensordicts of length 1 million each.

Parameters:

- **dataset_id** (*str*) - The dataset to be downloaded.
Must be part of `AtariDQNExperienceReplay.available_datasets`.
- **batch_size** (*int*) - Batch-size used during sampling.
Can be overridden by data.sample(batch_size) if necessary.

Keyword Arguments:

- **root** (*Path**or**str**,**optional*) - The AtariDQN dataset root directory.
The actual dataset memory-mapped files will be saved under
<root>/<dataset_id>. If none is provided, it defaults to
~/.cache/torchrl/atari.atari`.
- **num_procs** (*int**,**optional*) - number of processes to launch for preprocessing.
Has no effect whenever the data is already downloaded. Defaults to 0
(no multiprocessing used).
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
- **num_slices** (*int**,**optional*) - the number of slices to be sampled. The batch-size
must be greater or equal to the `num_slices` argument. Exclusive
with `slice_len`. Defaults to `None` (no slice sampling).
The `sampler` arg will override this value.
- **slice_len** (*int**,**optional*) - the length of the slices to be sampled. The batch-size
must be greater or equal to the `slice_len` argument and divisible
by it. Exclusive with `num_slices`. Defaults to `None` (no slice sampling).
The `sampler` arg will override this value.
- **strict_length** (*bool**,**optional*) - if `False`, trajectories of length
shorter than slice_len (or batch_size // num_slices) will be
allowed to appear in the batch.
Be mindful that this can result in effective batch_size shorter
than the one asked for! Trajectories can be split using
`torchrl.collectors.split_trajectories()`. Defaults to `True`.
The `sampler` arg will override this value.
- **replacement** (*bool**,**optional*) - if `False`, sampling will occur without replacement.
The `sampler` arg will override this value.
- **mp_start_method** (*str**,**optional*) - the start method for multiprocessed
download. Defaults to `"fork"`.

Variables:

- **available_datasets** - list of available datasets, formatted as <game_name>/<run>. Example:
"Pong/5", "Krull/2", ...
- **dataset_id** (*str*) - the name of the dataset.
- **episodes** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a 1d tensor indicating to what run each of the
1M frames belongs. To be used with [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)
to cheaply sample slices of episodes.

Examples

```
>>> from torchrl.data.datasets import AtariDQNExperienceReplay
>>> dataset = AtariDQNExperienceReplay("Pong/5", batch_size=128)
>>> for data in dataset:
... print(data)
... break
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 index: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int64, is_shared=False),
 metadata: NonTensorData(
 data={'invalid_range': MemoryMappedTensor([999998, 999999, 0, 1, 2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'}},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False)
```

Warning

Atari-DQN does not provide the next observation after a termination signal.
In other words, there is no way to obtain the `("next", "observation")` state
when `("next", "done")` is `True`. This value is filled with 0s but should
not be used in practice. If TorchRL's value estimators (`ValueEstimator`)
are used, this should not be an issue.

Note

Because the construction of the sampler for episode sampling is slightly
convoluted, we made it easy for users to pass the arguments of the
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) directly to the
`AtariDQNExperienceReplay` dataset: any of the `num_slices` or
`slice_len` arguments will make the sampler an instance of
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler). The `strict_length`
can also be passed.

```
>>> from torchrl.data.datasets import AtariDQNExperienceReplay
>>> from torchrl.data.replay_buffers import SliceSampler
>>> dataset = AtariDQNExperienceReplay("Pong/5", batch_size=128, slice_len=64)
>>> for data in dataset:
... print(data)
... print(data.get("index")) # indices are in 4 groups of consecutive values
... break
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 index: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.int64, is_shared=False),
 metadata: NonTensorData(
 data={'invalid_range': MemoryMappedTensor([999998, 999999, 0, 1, 2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'}},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([128, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([128, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([128]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([128]),
 device=None,
 is_shared=False)
tensor([2657628, 2657629, 2657630, 2657631, 2657632, 2657633, 2657634, 2657635,
 2657636, 2657637, 2657638, 2657639, 2657640, 2657641, 2657642, 2657643,
 2657644, 2657645, 2657646, 2657647, 2657648, 2657649, 2657650, 2657651,
 2657652, 2657653, 2657654, 2657655, 2657656, 2657657, 2657658, 2657659,
 2657660, 2657661, 2657662, 2657663, 2657664, 2657665, 2657666, 2657667,
 2657668, 2657669, 2657670, 2657671, 2657672, 2657673, 2657674, 2657675,
 2657676, 2657677, 2657678, 2657679, 2657680, 2657681, 2657682, 2657683,
 2657684, 2657685, 2657686, 2657687, 2657688, 2657689, 2657690, 2657691,
 1995687, 1995688, 1995689, 1995690, 1995691, 1995692, 1995693, 1995694,
 1995695, 1995696, 1995697, 1995698, 1995699, 1995700, 1995701, 1995702,
 1995703, 1995704, 1995705, 1995706, 1995707, 1995708, 1995709, 1995710,
 1995711, 1995712, 1995713, 1995714, 1995715, 1995716, 1995717, 1995718,
 1995719, 1995720, 1995721, 1995722, 1995723, 1995724, 1995725, 1995726,
 1995727, 1995728, 1995729, 1995730, 1995731, 1995732, 1995733, 1995734,
 1995735, 1995736, 1995737, 1995738, 1995739, 1995740, 1995741, 1995742,
 1995743, 1995744, 1995745, 1995746, 1995747, 1995748, 1995749, 1995750])
```

Note

As always, datasets should be composed using `ReplayBufferEnsemble`:

```
>>> from torchrl.data.datasets import AtariDQNExperienceReplay
>>> from torchrl.data.replay_buffers import ReplayBufferEnsemble
>>> # we change this parameter for quick experimentation, in practice it should be left untouched
>>> AtariDQNExperienceReplay._max_runs = 2
>>> dataset_asterix = AtariDQNExperienceReplay("Asterix/5", batch_size=128, slice_len=64, num_procs=4)
>>> dataset_pong = AtariDQNExperienceReplay("Pong/5", batch_size=128, slice_len=64, num_procs=4)
>>> dataset = ReplayBufferEnsemble(dataset_pong, dataset_asterix, batch_size=128, sample_from_all=True)
>>> sample = dataset.sample()
>>> print("first sample, Asterix", sample[0])
first sample, Asterix TensorDict(
 fields={
 action: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
 index: TensorDict(
 fields={
 buffer_ids: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False),
 index: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 metadata: NonTensorData(
 data={'invalid_range': MemoryMappedTensor([999998, 999999, 0, 1, 2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Pong/5'},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False)
>>> print("second sample, Pong", sample[1])
second sample, Pong TensorDict(
 fields={
 action: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
 index: TensorDict(
 fields={
 buffer_ids: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False),
 index: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 metadata: NonTensorData(
 data={'invalid_range': MemoryMappedTensor([999998, 999999, 0, 1, 2]), 'add_count': MemoryMappedTensor(999999), 'dataset_id': 'Asterix/5'},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False)
>>> print("Aggregate (metadata hidden)", sample)
Aggregate (metadata hidden) LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
 index: LazyStackedTensorDict(
 fields={
 buffer_ids: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False),
 index: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2, 64]),
 device=None,
 is_shared=False,
 stack_dim=0),
 metadata: LazyStackedTensorDict(
 fields={
 },
 exclusive_fields={
 },
 batch_size=torch.Size([2, 64]),
 device=None,
 is_shared=False,
 stack_dim=0),
 next: LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2, 64]),
 device=None,
 is_shared=False,
 stack_dim=0),
 observation: Tensor(shape=torch.Size([2, 64, 84, 84]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.uint8, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2, 64]),
 device=None,
 is_shared=False,
 stack_dim=0)
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

*abstract property*data_path*: Path*

Path to the dataset, including split.

*abstract property*data_path_root*: Path*

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

preprocess(*fn: Callable[[TensorDictBase], TensorDictBase]*, *dim: int = 0*, *num_workers: int | None = None*, ***, *chunksize: int | None = None*, *num_chunks: int | None = None*, *pool: mp.Pool | None = None*, *generator: [torch.Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | None = None*, *max_tasks_per_child: int | None = None*, *worker_threads: int = 1*, *index_with_generator: bool = False*, *pbar: bool = False*, *mp_start_method: str | None = None*, *dest: str | Path*, *num_frames: int | None = None*)[[source]](../../_modules/torchrl/data/datasets/atari_dqn.html#AtariDQNExperienceReplay.preprocess)

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