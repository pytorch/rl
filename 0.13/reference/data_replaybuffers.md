# Replay Buffers

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:

## Core Replay Buffer Classes

| [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)(*[, storage, sampler, writer, ...]) | A generic, composable replay buffer class. |
| --- | --- |
| [`ReplayBufferEnsemble`](generated/torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble)(*rbs[, storages, ...]) | An ensemble of replay buffers. |
| [`PrioritizedReplayBuffer`](generated/torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer)(*, alpha, beta[, ...]) | Prioritized replay buffer. |
| [`TensorDictReplayBuffer`](generated/torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer)(*[, priority_key]) | TensorDict-specific wrapper around the [`ReplayBuffer`](generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) class. |
| [`TensorDictPrioritizedReplayBuffer`](generated/torchrl.data.TensorDictPrioritizedReplayBuffer.html#torchrl.data.TensorDictPrioritizedReplayBuffer)(*, alpha, beta) | TensorDict-specific wrapper around the [`PrioritizedReplayBuffer`](generated/torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer) class. |
| [`RayReplayBuffer`](generated/torchrl.data.RayReplayBuffer.html#torchrl.data.RayReplayBuffer)(*args, replay_buffer_cls, ...) | A Ray implementation of the Replay Buffer that can be extended and sampled remotely. |
| [`RemoteTensorDictReplayBuffer`](generated/torchrl.data.RemoteTensorDictReplayBuffer.html#torchrl.data.RemoteTensorDictReplayBuffer)(*args, **kwargs) | A remote invocation friendly ReplayBuffer class. |

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

### TED-format conversion

The following helpers convert between the TorchRL Episode Data (TED) layout and
a flat, storage-friendly representation when serializing or restoring a buffer:

| [`TED2Flat`](generated/torchrl.data.TED2Flat.html#torchrl.data.TED2Flat)([done_key, shift_key, is_full_key, ...]) | A storage saving hook to serialize TED data in a compact format. |
| --- | --- |
| [`Flat2TED`](generated/torchrl.data.Flat2TED.html#torchrl.data.Flat2TED)([done_key, shift_key, is_full_key, ...]) | A storage loading hook to deserialize flattened TED data to TED format. |