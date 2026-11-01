# TensorDictReplayBuffer

*class*torchrl.data.TensorDictReplayBuffer(***, *priority_key: str = 'td_error'*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#TensorDictReplayBuffer)

TensorDict-specific wrapper around the [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) class.

Keyword Arguments:

- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*,**Callable**[**[**]**,*[*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*]**,**optional*) - the storage to be used.
If a callable is passed, it is used as constructor for the storage.
If none is provided a default [`ListStorage`](torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage) with
`max_size` of `1_000` will be created.
- **sampler** ([*Sampler*](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*,**Callable**[**[**]**,*[*Sampler*](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*]**,**optional*) - the sampler to be used.
If a callable is passed, it is used as constructor for the sampler.
If none is provided, a default [`RandomSampler`](torchrl.data.replay_buffers.RandomSampler.html#torchrl.data.replay_buffers.RandomSampler)
will be used.
- **writer** ([*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*,**Callable**[**[**]**,*[*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*]**,**optional*) - the writer to be used.
If a callable is passed, it is used as constructor for the writer.
If none is provided a default [`TensorDictRoundRobinWriter`](torchrl.data.replay_buffers.TensorDictRoundRobinWriter.html#torchrl.data.replay_buffers.TensorDictRoundRobinWriter)
will be used.
- **collate_fn** (*callable**,**optional*) - merges a list of samples to form a
mini-batch of Tensor(s)/outputs. Used when using batched
loading from a map-style dataset. The default value will be decided
based on the storage type.
- **pin_memory** (*bool*) - whether pin_memory() should be called on the rb
samples.
- **prefetch** (*int**,**optional*) - number of next batches to be prefetched
using multithreading. Defaults to None (no prefetching).
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*or**Callable**[**[**Any**]**,**Any**]**,**optional*) - Transform to be executed when
`sample()` is called.
To chain transforms use the `Compose` class.
Transforms should be used with [`tensordict.TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
content. A generic callable can also be passed if the replay buffer
is used with PyTree structures (see example below).
Unlike storages, writers and samplers, transform constructors must
be passed as separate keyword argument `transform_factory`,
as it is impossible to distinguish a constructor from a transform.
- **transform_factory** (*Callable**[**[**]**,**Callable**]**,**optional*) - a factory for the
transform. Exclusive with `transform`.
- **batch_size** (*int**,**optional*) -

the batch size to be used when sample() is
called.

Note

The batch-size can be specified at construction time via the
`batch_size` argument, or at sampling time. The former should
be preferred whenever the batch-size is consistent across the
experiment. If the batch-size is likely to change, it can be
passed to the `sample()` method. This option is
incompatible with prefetching (since this requires to know the
batch-size in advance) as well as with samplers that have a
`drop_last` argument.
- **priority_key** (*str**,**optional*) - the key at which priority is assumed to
be stored within TensorDicts added to this ReplayBuffer.
This is to be used when the sampler is of type
`PrioritizedSampler`.
Defaults to `"td_error"`.
- **dim_extend** (*int**,**optional*) -

indicates the dim to consider for
extension when calling `extend()`. Defaults to `storage.ndim-1`.
When using `dim_extend > 0`, we recommend using the `ndim`
argument in the storage instantiation if that argument is
available, to let storages know that the data is
multi-dimensional and keep consistent notions of storage-capacity
and batch-size during sampling.

Note

This argument has no effect on `add()` and
therefore should be used with caution when both `add()`
and `extend()` are used in a codebase. For example:

```
>>> data = torch.zeros(3, 4)
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(10, ndim=2),
... dim_extend=1)
>>> # these two approaches are equivalent:
>>> for d in data.unbind(1):
... rb.add(d)
>>> rb.extend(data)
```
- **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) -

a generator to use for sampling.
Using a dedicated generator for the replay buffer can allow a fine-grained control
over seeding, for instance keeping the global seed different but the RB seed identical
for distributed jobs.
Defaults to `None` (global default generator).

Warning

As of now, the generator has no effect on the transforms.
- **shared** (*bool**,**optional*) - whether the buffer will be shared using multiprocessing or not.
Defaults to `False`.
- **compilable** (*bool**,**optional*) - whether the writer is compilable.
If `True`, the writer cannot be shared between multiple processes.
Defaults to `False`.
- **delayed_init** (*bool**,**optional*) - whether to initialize storage, writer, sampler and transform
the first time the buffer is used rather than during construction.
This is useful when the replay buffer needs to be pickled and sent to remote workers,
particularly when using transforms with modules that require gradients.
If not specified, defaults to `True` when `transform_factory` is provided,
and `False` otherwise.

Examples

```
>>> import torch
>>>
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
>>> from tensordict import TensorDict
>>>
>>> torch.manual_seed(0)
>>>
>>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=5)
>>> data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 1, 1)}, [10])
>>> rb.extend(data)
>>> sample = rb.sample(3)
>>> # samples keep track of the index
>>> print(sample)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False),
 index: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False)
>>> # we can iterate over the buffer
>>> for i, data in enumerate(rb):
... print(i, data)
... if i == 2:
... break
0 TensorDict(
 fields={
 a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False)
1 TensorDict(
 fields={
 a: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([5, 1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 index: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.int32, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False)
```

add(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → int[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#TensorDictReplayBuffer.add)

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
yielded by the [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) iterator.

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

extend(*tensordicts: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *update_priority: bool | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#TensorDictReplayBuffer.extend)

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

sample(*batch_size: int | None = None*, *return_info: bool = False*, *include_info: bool | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#TensorDictReplayBuffer.sample)

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