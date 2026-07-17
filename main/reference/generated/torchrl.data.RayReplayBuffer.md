# RayReplayBuffer

*class*torchrl.data.RayReplayBuffer(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer)

A Ray implementation of the Replay Buffer that can be extended and sampled remotely.

Keyword Arguments:

- **replay_buffer_cls** ([*type*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.type)*[*[*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*]**,**optional*) - the class to use for the replay buffer.
Defaults to [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer).
- **ray_init_config** (*dict**[**str**,**Any**]**,**optional*) - keyword arguments to pass to ray.init().
- **remote_config** (*dict**[**str**,**Any**]**,**optional*) - keyword arguments to pass to cls.as_remote().
Defaults to torchrl.collectors.distributed.ray.DEFAULT_REMOTE_CLASS_CONFIG.
- ****kwargs** - keyword arguments to pass to the replay buffer class.

See also

[`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) for a list of other keyword arguments.

The writer, sampler and storage should be passed as constructors to prevent serialization issues.
Transforms constructors should be passed through the transform_factory argument.

Example

```
>>> from functools import partial
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> buffer = ReplayBuffer(
... storage=partial(LazyTensorStorage, 100),
... batch_size=4,
... service_backend="ray",
... service_backend_options={"remote_config": {"num_cpus": 0}},
... transport="auto",
... )
>>> buffer.extend(TensorDict({"value": torch.arange(8)}, batch_size=[8]))
>>> buffer.sample().shape
torch.Size([4])
>>> buffer.shutdown()
```

add(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.add)

Add a single element to the replay buffer.

Parameters:

**data** (*Any*) - data to be added to the replay buffer

Returns:

index where the data lives in the replay buffer.

append_transform(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.append_transform)

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

client() → Any[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.client)

Return a picklable client without actor shutdown rights.

clients(*num_clients: int*) → list[Any][[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.clients)

Return one independently routed client per concurrent consumer.

close() → None[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.close)

Terminates the Ray actor associated with this replay buffer.

dump(*path*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.dump)

Alias for `dumps()`.

dumps(*path*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.dumps)

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

empty(*empty_write_count: bool = True*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.empty)

Empties the replay buffer and reset cursor to 0.

Parameters:

**empty_write_count** (*bool**,**optional*) - Whether to empty the write_count attribute. Defaults to True.

extend(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.extend)

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

insert_transform(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*, ***, *invert: bool = False*) → [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.insert_transform)

Inserts transform.

Transforms are executed in order when sample is called.

Parameters:

- **index** (*int*) - Position to insert the transform.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)) - The transform to be appended

Keyword Arguments:

**invert** (*bool**,**optional*) - if `True`, the transform will be inverted (forward calls will be called
during writing and inverse calls during reading). Defaults to `False`.

*property*is_alive*: bool*

Whether the owned Ray replay-buffer actor is available.

load(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.load)

Alias for `loads()`.

loads(*path*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.loads)

Loads a replay buffer state at the given path.

The buffer should have matching components and be saved using `dumps()`.

Parameters:

**path** (*Path**or**str*) - path where the replay buffer was saved.

See `dumps()` for more info.

next()[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.next)

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

register_load_hook(*hook: Callable[[Any], Any]*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.register_load_hook)

Registers a load hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

register_save_hook(*hook: Callable[[Any], Any]*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.register_save_hook)

Registers a save hook for the storage.

Note

Hooks are currently not serialized when saving a replay buffer: they must
be manually re-initialized every time the buffer is created.

sample(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.sample)

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

save(*path: str*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.save)

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

set_sampler(*sampler*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.set_sampler)

Sets a new sampler in the replay buffer and returns the previous sampler.

set_storage(*storage*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.set_storage)

Sets a new storage in the replay buffer and returns the previous storage.

Parameters:

- **storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the new storage for the buffer.
- **collate_fn** (*callable**,**optional*) - if provided, the collate_fn is set to this
value. Otherwise it is reset to a default value.

set_writer(*writer*)[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.set_writer)

Sets a new writer in the replay buffer and returns the previous writer.

shutdown(*timeout: float | None = None*) → None[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.shutdown)

Terminate the owned Ray actor.

start() → RayReplayBuffer[[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.start)

Return this already-started Ray replay-buffer owner.

stats() → dict[str, int | float | bool][[source]](../../_modules/torchrl/data/replay_buffers/ray_buffer.html#RayReplayBuffer.stats)

Returns the buffer stats snapshot through a single actor round-trip.

See [`stats()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.stats).

*property*storage*: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*

The storage of the replay buffer.

The storage must be an instance of [`Storage`](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage).

*property*transform*: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*

The transform of the replay buffer.

The transform must be an instance of [`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform).

*property*transport_kind*: str*

Physical transport used for replay payloads.

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

*property*write_count

The total number of items written so far in the buffer through add and extend.

*property*writer*: [Writer](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)*

The writer of the replay buffer.

The writer must be an instance of [`Writer`](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer).