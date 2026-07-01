# ReplayBuffer

*class*torchrl.data.ReplayBuffer(***, *storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage) | Callable[[], [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)] | None = None*, *sampler: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler) | Callable[[], [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)] | None = None*, *writer: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer) | Callable[[], [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)] | None = None*, *collate_fn: Callable | None = None*, *pin_memory: bool = False*, *prefetch: int | None = None*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable | None = None*, *transform_factory: Callable[[], [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable] | None = None*, *batch_size: int | None = None*, *dim_extend: int | None = None*, *checkpointer: [StorageCheckpointerBase](torchrl.data.replay_buffers.StorageCheckpointerBase.html#torchrl.data.replay_buffers.StorageCheckpointerBase) | Callable[[], [StorageCheckpointerBase](torchrl.data.replay_buffers.StorageCheckpointerBase.html#torchrl.data.replay_buffers.StorageCheckpointerBase)] | None = None*, *generator: [torch.Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | None = None*, *consume_after_n_samples: int | None = None*, *shared: bool = False*, *compilable: bool | None = None*, *delayed_init: bool | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer)

A generic, composable replay buffer class.

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
If none is provided a default [`RoundRobinWriter`](torchrl.data.replay_buffers.RoundRobinWriter.html#torchrl.data.replay_buffers.RoundRobinWriter)
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
- **dim_extend** (*int**,**optional*) -

indicates the dim to consider for
extension when calling `extend()`. Defaults to `storage.ndim-1`.
When using `dim_extend > 0`, we recommend using the `ndim`
argument in the storage instantiation if that argument is
available, to let storages know that the data is
multi-dimensional and keep consistent notions of storage-capacity
and batch-size during sampling.

Important

When using a collector with `trajs_per_batch`,
trajectories are written as flat 1-D sequences of variable
length. Do not set `dim_extend > 0` or `ndim >= 2` in
this case -- the storage must be 1-dimensional.

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
- **consume_after_n_samples** (*int**,**optional*) - if provided, sampled items are
removed from the sampleable set after they have been returned this
many times. The default value of `None` keeps the standard replay
buffer behavior. Passing `1` makes each item available for a
single sample before it is consumed.
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
>>> from torchrl.data import ReplayBuffer, ListStorage
>>>
>>> torch.manual_seed(0)
>>> rb = ReplayBuffer(
... storage=ListStorage(max_size=1000),
... batch_size=5,
... )
>>> # populate the replay buffer and get the item indices
>>> data = range(10)
>>> indices = rb.extend(data)
>>> # sample will return as many elements as specified in the constructor
>>> sample = rb.sample()
>>> print(sample)
tensor([4, 9, 3, 0, 3])
>>> # Passing the batch-size to the sample method overrides the one in the constructor
>>> sample = rb.sample(batch_size=3)
>>> print(sample)
tensor([9, 7, 3])
>>> # one cans sample using the ``sample`` method or iterate over the buffer
>>> for i, batch in enumerate(rb):
... print(i, batch)
... if i == 3:
... break
0 tensor([7, 3, 1, 6, 6])
1 tensor([9, 8, 6, 6, 8])
2 tensor([4, 3, 6, 9, 1])
3 tensor([4, 4, 1, 9, 9])
```

Replay buffers accept *any* kind of data. Not all storage types
will work, as some expect numerical data only, but the default
`ListStorage` will:

Examples

```
>>> torch.manual_seed(0)
>>> buffer = ReplayBuffer(storage=ListStorage(100), collate_fn=lambda x: x)
>>> indices = buffer.extend(["a", 1, None])
>>> buffer.sample(3)
[None, 'a', None]
```

The [`TensorStorage`](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage), [`LazyMemmapStorage`](torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage)
and [`LazyTensorStorage`](torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage) also work
with any PyTree structure (a PyTree is a nested structure of arbitrary depth made of dicts,
lists or tuples where the leaves are tensors) provided that it only contains
tensor data.

Examples

```
>>> from torch.utils._pytree import tree_map
>>> def transform(x):
... # Zeros all the data in the pytree
... return tree_map(lambda y: y * 0, x)
>>> rb = ReplayBuffer(storage=LazyMemmapStorage(100), transform=transform)
>>> data = {
... "a": torch.randn(3),
... "b": {"c": (torch.zeros(2), [torch.ones(1)])},
... 30: -torch.ones(()),
... }
>>> rb.add(data)
>>> # The sample has a similar structure to the data (with a leading dimension of 10 for each tensor)
>>> s = rb.sample(10)
>>> # let's check that our transform did its job:
>>> def assert0(x):
>>> assert (x == 0).all()
>>> tree_map(assert0, s)
```

add(*data: Any*) → int[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.add)

Add a single element to the replay buffer.

Parameters:

**data** (*Any*) - data to be added to the replay buffer

Returns:

index where the data lives in the replay buffer.

append_transform(*transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*, ***, *invert: bool = False*) → ReplayBuffer[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.append_transform)

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

dump(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.dump)

Alias for `dumps()`.

dumps(*path*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.dumps)

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

empty(*empty_write_count: bool = True*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.empty)

Empties the replay buffer and reset cursor to 0.

Parameters:

**empty_write_count** (*bool**,**optional*) - Whether to empty the write_count attribute. Defaults to True.

extend(*data: Sequence*, ***, *update_priority: bool | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.extend)

Extends the replay buffer with one or more elements contained in an iterable.

If present, the inverse transforms will be called.`

Parameters:

**data** (*iterable*) - collection of data to be added to the replay
buffer.

Keyword Arguments:

**update_priority** (*bool**,**optional*) - Whether to update the priority of the data. Defaults to True.
Without effect in this class. See [`extend()`](torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer.extend) for more details.

Returns:

Indices of the data added to the replay buffer.

Warning

`extend()` can have an
ambiguous signature when dealing with lists of values, which should be interpreted
either as PyTree (in which case all elements in the list will be put in a slice
in the stored PyTree in the storage) or a list of values to add one at a time.
To solve this, TorchRL makes the clear-cut distinction between list and tuple:
a tuple will be viewed as a PyTree, a list (at the root level) will be interpreted
as a stack of values to add one at a time to the buffer.
For [`ListStorage`](torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage) instances, only
unbound elements can be provided (no PyTrees).

*property*initialized*: bool*

Whether the replay buffer has been initialized.

insert_transform(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*, ***, *invert: bool = False*) → ReplayBuffer[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.insert_transform)

Inserts transform.

Transforms are executed in order when sample is called.

Parameters:

- **index** (*int*) - Position to insert the transform.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)) - The transform to be appended

Keyword Arguments:

**invert** (*bool**,**optional*) - if `True`, the transform will be inverted (forward calls will be called
during writing and inverse calls during reading). Defaults to `False`.

load(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.load)

Alias for `loads()`.

loads(*path*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.loads)

Loads a replay buffer state at the given path.

The buffer should have matching components and be saved using `dumps()`.

Parameters:

**path** (*Path**or**str*) - path where the replay buffer was saved.

See `dumps()` for more info.

next()[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.next)

Returns the next item in the replay buffer.

This method is used to iterate over the replay buffer in contexts where __iter__ is not available,
such as `RayReplayBuffer`.

read_all_in_order(*end: int | None = None*) → Any[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.read_all_in_order)

Read storage contents in physical order.

This is equivalent to `rb[:]` when `end` is `None`.

Parameters:

**end** (*int**,**optional*) - Number of leading storage entries to read.
Defaults to the entire storage slice.

Returns:

A storage slice containing entries `[:end]`.

register_load_hook(*hook: Callable[[Any], Any]*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.register_load_hook)

Registers a load hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

register_save_hook(*hook: Callable[[Any], Any]*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.register_save_hook)

Registers a save hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

sample(*batch_size: int | None = None*, *return_info: bool = False*) → Any[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.sample)

Samples a batch of data from the replay buffer.

Uses Sampler to sample indices, and retrieves them from Storage.

Parameters:

- **batch_size** (*int**,**optional*) - size of data to be collected. If none
is provided, this method will sample a batch-size as indicated
by the sampler.
- **return_info** (*bool*) - whether to return info. If True, the result
is a tuple (data, info). If False, the result is the data.

Returns:

A batch of data selected in the replay buffer.
A tuple containing this batch and info if return_info flag is set to True.

*property*sampler*: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*

The sampler of the replay buffer.

The sampler must be an instance of [`Sampler`](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler).

save(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.save)

Alias for `dumps()`.

set_(*key*, *value*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.set_)

Sets the value of a key across the entire replay buffer in-place.

Parameters:

- **key** (*NestedKey*) - the key to set.
- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the value to write.

Returns:

self

set_at_(*key*, *value*, *index*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.set_at_)

Sets the value of a key at specified indices in the replay buffer.

Parameters:

- **key** (*NestedKey*) - the key to set.
- **value** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the value to write.
- **index** - the indices where to write the value.

Returns:

self

set_sampler(*sampler: [Sampler](torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.set_sampler)

Sets a new sampler in the replay buffer and returns the previous sampler.

set_storage(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *collate_fn: Callable | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.set_storage)

Sets a new storage in the replay buffer and returns the previous storage.

Parameters:

- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the new storage for the buffer.
- **collate_fn** (*callable**,**optional*) - if provided, the collate_fn is set to this
value. Otherwise it is reset to a default value.

set_writer(*writer: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.set_writer)

Sets a new writer in the replay buffer and returns the previous writer.

*property*storage*: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*

The storage of the replay buffer.

The storage must be an instance of [`Storage`](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage).

*property*transform*: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*

The transform of the replay buffer.

The transform must be an instance of [`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform).

update_(*input_dict_or_td*, *clone=False*, ***, *keys_to_update=None*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.update_)

Updates the replay buffer in-place with the given dict or TensorDict.

Parameters:

- **input_dict_or_td** (*dict**or**TensorDictBase*) - the data to update with.
- **clone** (*bool**,**optional*) - whether to clone the values before writing.
Defaults to `False`.
- **keys_to_update** (*sequence**of**NestedKey**,**optional*) - if provided, only
these keys will be updated.

Returns:

self

write_all(*data: Any*, *end: int | None = None*) → None[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBuffer.write_all)

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