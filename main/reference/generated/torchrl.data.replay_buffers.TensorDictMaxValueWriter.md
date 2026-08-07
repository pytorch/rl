# TensorDictMaxValueWriter

*class*torchrl.data.replay_buffers.TensorDictMaxValueWriter(*rank_key=None*, *reduction: str = 'sum'*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictMaxValueWriter)

A Writer class for composable replay buffers that keeps the top elements based on some ranking key.

Parameters:

- **rank_key** (*str**or**tuple**of**str*) - the key to rank the elements by. Defaults to `("next", "reward")`.
- **reduction** (*str*) - the reduction method to use if the rank key has more than one element.
Can be `"max"`, `"min"`, `"mean"`, `"median"` or `"sum"`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, TensorDictMaxValueWriter
>>> from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(1),
... sampler=SamplerWithoutReplacement(),
... batch_size=1,
... writer=TensorDictMaxValueWriter(rank_key="key"),
... )
>>> td = TensorDict({
... "key": torch.tensor(range(10)),
... "obs": torch.tensor(range(10))
... }, batch_size=10)
>>> rb.extend(td)
>>> print(rb.sample().get("obs").item())
9
>>> td = TensorDict({
... "key": torch.tensor(range(10, 20)),
... "obs": torch.tensor(range(10, 20))
... }, batch_size=10)
>>> rb.extend(td)
>>> print(rb.sample().get("obs").item())
19
>>> td = TensorDict({
... "key": torch.tensor(range(10)),
... "obs": torch.tensor(range(10))
... }, batch_size=10)
>>> rb.extend(td)
>>> print(rb.sample().get("obs").item())
19
```

Note

This class isn't compatible with storages with more than one dimension.
This doesn't mean that storing trajectories is prohibited, but that
the trajectories stored must be stored on a per-trajectory basis.
Here are some examples of valid and invalid usages of the class.
First, a flat buffer where we store individual transitions:

```
>>> from torchrl.data import TensorStorage
>>> # Simplest use case: data comes in 1d and is stored as such
>>> data = TensorDict({
... "obs": torch.zeros(10, 3),
... "reward": torch.zeros(10, 1),
... }, batch_size=[10])
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(max_size=100),
... writer=TensorDictMaxValueWriter(rank_key="reward")
... )
>>> # We initialize the buffer: a total of 100 *transitions* can be stored
>>> rb.extend(data)
>>> # Samples 5 *transitions* at random
>>> sample = rb.sample(5)
>>> assert sample.shape == (5,)
```

Second, a buffer where we store trajectories. The max signal is aggregated
in each batch (e.g. the reward of each rollout is summed):

```
>>> # One can also store batches of data, each batch being a sub-trajectory
>>> env = ParallelEnv(2, lambda: GymEnv("Pendulum-v1"))
>>> # Get a batch of [2, 10] -- format is [Batch, Time]
>>> rollout = env.rollout(max_steps=10)
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(max_size=100),
... writer=TensorDictMaxValueWriter(rank_key="reward")
... )
>>> # We initialize the buffer: a total of 100 *trajectories* (!) can be stored
>>> rb.extend(rollout)
>>> # Sample 5 trajectories at random
>>> sample = rb.sample(5)
>>> assert sample.shape == (5, 10)
```

If data come in batch but a flat buffer is needed, we can simply flatten
the data before extending the buffer:

```
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(max_size=100),
... writer=TensorDictMaxValueWriter(rank_key="reward")
... )
>>> # We initialize the buffer: a total of 100 *transitions* can be stored
>>> rb.extend(rollout.reshape(-1))
>>> # Sample 5 trajectories at random
>>> sample = rb.sample(5)
>>> assert sample.shape == (5,)
```

It is not possible to create a buffer that is extended along the time
dimension, which is usually the recommended way of using buffers with
batches of trajectories. Since trajectories are overlapping, it's hard
if not impossible to aggregate the reward values and compare them.
This constructor isn't valid (notice the ndim argument):

```
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(max_size=100, ndim=2), # Breaks!
... writer=TensorDictMaxValueWriter(rank_key="reward")
... )
```

add(*data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictMaxValueWriter.add)

Inserts a single element of data at an appropriate index, and returns that index.

The `rank_key` in the data passed to this module should be structured as [].
If it has more dimensions, it will be reduced to a single value using the `reduction` method.

extend(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictMaxValueWriter.extend)

Inserts a series of data points at appropriate indices.

The `rank_key` in the data passed to this module should be structured as [B].
If it has more dimensions, it will be reduced to a single value using the `reduction` method.

generations_of(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the generation stamp for each physical slot in `index`.

A slot's stamp advances once per write it receives, so a single
`extend` that wraps the storage advances a reused slot once per write.
Comparing a stamp captured at sampling time against the current stamp
tells you whether the slot still holds the data you sampled.

Writers that do not track slot reuse - and writers constructed with
`track_generations=False`, which is the default - report `-1`
everywhere. Never-written slots also report `-1`, so `-1` means
"no usable stamp" rather than "generation zero".

Parameters:

**index** (*int**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - dim-0 slot indices. A 1-D tensor is
always read as a batch of slot indices; for a storage with
`ndim > 1`, pass a `tuple` of per-dimension indices (as
[`extend()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend) returns) to identify
a single cell - only its dim-0 component is used, since a
generation stamps a whole dim-0 slot.

Returns:

`int64` stamps shaped like the dim-0 component of
`index`, on `index`'s device.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

get_insert_index(*data: Any*) → int[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictMaxValueWriter.get_insert_index)

Returns the index where the data should be inserted, or `None` if it should not be inserted.

tracks_generations*: bool**= False*

Whether this writer stamps storage slots with a reuse generation. Always
`False` unless the writer both supports generation tracking and was
constructed with it enabled (see
`RoundRobinWriter`).