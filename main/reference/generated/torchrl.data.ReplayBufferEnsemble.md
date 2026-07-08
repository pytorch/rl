# ReplayBufferEnsemble

*class*torchrl.data.ReplayBufferEnsemble(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/replay_buffers.html#ReplayBufferEnsemble)

An ensemble of replay buffers.

This class allows to read and sample from multiple replay buffers at once.
It automatically composes ensemble of storages ([`StorageEnsemble`](torchrl.data.replay_buffers.StorageEnsemble.html#torchrl.data.replay_buffers.StorageEnsemble)),
writers ([`WriterEnsemble`](torchrl.data.replay_buffers.WriterEnsemble.html#torchrl.data.replay_buffers.WriterEnsemble)) and
samplers ([`SamplerEnsemble`](torchrl.data.replay_buffers.SamplerEnsemble.html#torchrl.data.replay_buffers.SamplerEnsemble)).

Note

Writing directly to this class is forbidden, but it can be indexed to retrieve
the nested nested-buffer and extending it.

There are two distinct ways of constructing a `ReplayBufferEnsemble`:
one can either pass a list of replay buffers, or directly pass the components
(storage, writers and samplers) like it is done for other replay buffer subclasses.

Parameters:

- **rbs** (*sequence**of**ReplayBuffer instances**,**optional*) - the replay buffers to ensemble.
- **storages** ([*StorageEnsemble*](torchrl.data.replay_buffers.StorageEnsemble.html#torchrl.data.replay_buffers.StorageEnsemble)*,**optional*) - the ensemble of storages, if the replay
buffers are not passed.
- **samplers** ([*SamplerEnsemble*](torchrl.data.replay_buffers.SamplerEnsemble.html#torchrl.data.replay_buffers.SamplerEnsemble)*,**optional*) - the ensemble of samplers, if the replay
buffers are not passed.
- **writers** ([*WriterEnsemble*](torchrl.data.replay_buffers.WriterEnsemble.html#torchrl.data.replay_buffers.WriterEnsemble)*,**optional*) - the ensemble of writers, if the replay
buffers are not passed.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*,**optional*) - if passed, this will be the transform
of the ensemble of replay buffers. Individual transforms for each
replay buffer is retrieved from its parent replay buffer, or directly
written in the [`StorageEnsemble`](torchrl.data.replay_buffers.StorageEnsemble.html#torchrl.data.replay_buffers.StorageEnsemble)
object.
- **batch_size** (*int**,**optional*) - the batch-size to use during sampling.
- **collate_fn** (*callable**,**optional*) - the function to use to collate the
data after each individual collate_fn has been called and the data
is placed in a list (along with the buffer id).
- **collate_fns** (*list**of**callables**,**optional*) - collate_fn of each nested
replay buffer. Retrieved from the [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) instances
if not provided.
- **p** (*list**of**float**or**Tensor**,**optional*) - a list of floating numbers
indicating the relative weight of each replay buffer. Can also
be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
if the buffer is built explicitly.
- **sample_from_all** (*bool**,**optional*) - if `True`, each dataset will be sampled
from. This is not compatible with the `p` argument. Defaults to `False`.
Can also be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
if the buffer is built explicitly.
- **num_buffer_sampled** (*int**,**optional*) - the number of buffers to sample.
if `sample_from_all=True`, this has no effect, as it defaults to the
number of buffers. If `sample_from_all=False`, buffers will be
sampled according to the probabilities `p`. Can also
be passed to torchrl.data.replay_buffers.samplers.SamplerEnsemble`
if the buffer is built explicitly.
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
- **delayed_init** (*bool**,**optional*) - whether to initialize storage, writer, sampler and transform
the first time the buffer is used rather than during construction.
This is useful when the replay buffer needs to be pickled and sent to remote workers,
particularly when using transforms with modules that require gradients.
If not specified, defaults to `True` when `transform_factory` is provided,
and `False` otherwise.

Examples

```
>>> from torchrl.envs import Compose, ToTensorImage, Resize, RenameTransform
>>> from torchrl.data import TensorDictReplayBuffer, ReplayBufferEnsemble, LazyMemmapStorage
>>> from tensordict import TensorDict
>>> import torch
>>> rb0 = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(10),
... transform=Compose(
... ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
... Resize(32, in_keys=["pixels", ("next", "pixels")]),
... RenameTransform([("some", "key")], ["renamed"]),
... ),
... )
>>> rb1 = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(10),
... transform=Compose(
... ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
... Resize(32, in_keys=["pixels", ("next", "pixels")]),
... RenameTransform(["another_key"], ["renamed"]),
... ),
... )
>>> rb = ReplayBufferEnsemble(
... rb0,
... rb1,
... p=[0.5, 0.5],
... transform=Resize(33, in_keys=["pixels"], out_keys=["pixels33"]),
... )
>>> print(rb)
ReplayBufferEnsemble(
 storages=StorageEnsemble(
 storages=(<torchrl.data.replay_buffers.storages.LazyMemmapStorage object at 0x13a2ef430>, <torchrl.data.replay_buffers.storages.LazyMemmapStorage object at 0x13a2f9310>),
 transforms=[Compose(
 ToTensorImage(keys=['pixels', ('next', 'pixels')]),
 Resize(w=32, h=32, interpolation=InterpolationMode.BILINEAR, keys=['pixels', ('next', 'pixels')]),
 RenameTransform(keys=[('some', 'key')])), Compose(
 ToTensorImage(keys=['pixels', ('next', 'pixels')]),
 Resize(w=32, h=32, interpolation=InterpolationMode.BILINEAR, keys=['pixels', ('next', 'pixels')]),
 RenameTransform(keys=['another_key']))]),
 samplers=SamplerEnsemble(
 samplers=(<torchrl.data.replay_buffers.samplers.RandomSampler object at 0x13a2f9220>, <torchrl.data.replay_buffers.samplers.RandomSampler object at 0x13a2f9f70>)),
 writers=WriterEnsemble(
 writers=(<torchrl.data.replay_buffers.writers.TensorDictRoundRobinWriter object at 0x13a2d9b50>, <torchrl.data.replay_buffers.writers.TensorDictRoundRobinWriter object at 0x13a2f95b0>)),
batch_size=None,
transform=Compose(
 Resize(w=33, h=33, interpolation=InterpolationMode.BILINEAR, keys=['pixels'])),
collate_fn=<built-in method stack of type object at 0x128648260>)
>>> data0 = TensorDict(
... {
... "pixels": torch.randint(255, (10, 244, 244, 3)),
... ("next", "pixels"): torch.randint(255, (10, 244, 244, 3)),
... ("some", "key"): torch.randn(10),
... },
... batch_size=[10],
... )
>>> data1 = TensorDict(
... {
... "pixels": torch.randint(255, (10, 64, 64, 3)),
... ("next", "pixels"): torch.randint(255, (10, 64, 64, 3)),
... "another_key": torch.randn(10),
... },
... batch_size=[10],
... )
>>> rb[0].extend(data0)
>>> rb[1].extend(data1)
>>> for _ in range(2):
... sample = rb.sample(10)
... assert sample["next", "pixels"].shape == torch.Size([2, 5, 3, 32, 32])
... assert sample["pixels"].shape == torch.Size([2, 5, 3, 32, 32])
... assert sample["pixels33"].shape == torch.Size([2, 5, 3, 33, 33])
... assert sample["renamed"].shape == torch.Size([2, 5])
```

add(*data: Any*) → int

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

client() → T

Return `self` for the zero-overhead direct backend.

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

extend(*data: Sequence*, ***, *update_priority: bool | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

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

sample(*batch_size: int | None = None*, *return_info: bool = False*) → Any

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