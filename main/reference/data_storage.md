# Storage Backends

TorchRL provides various storage backends for replay buffers, each optimized for different use cases.

| [`CompressedListStorage`](generated/torchrl.data.replay_buffers.CompressedListStorage.html#torchrl.data.replay_buffers.CompressedListStorage)(max_size, *[, ...]) | A storage that compresses and decompresses data. |
| --- | --- |
| [`CompressedListStorageCheckpointer`](generated/torchrl.data.replay_buffers.CompressedListStorageCheckpointer.html#torchrl.data.replay_buffers.CompressedListStorageCheckpointer)() | A storage checkpointer for CompressedListStorage. |
| [`FlatStorageCheckpointer`](generated/torchrl.data.replay_buffers.FlatStorageCheckpointer.html#torchrl.data.replay_buffers.FlatStorageCheckpointer)([done_keys, reward_keys]) | Saves the storage in a compact form, saving space on the TED format. |
| [`H5StorageCheckpointer`](generated/torchrl.data.replay_buffers.H5StorageCheckpointer.html#torchrl.data.replay_buffers.H5StorageCheckpointer)(*[, checkpoint_file, ...]) | Saves the storage in a compact form, saving space on the TED format and using H5 format to save the data. |
| [`ImmutableDatasetWriter`](generated/torchrl.data.replay_buffers.ImmutableDatasetWriter.html#torchrl.data.replay_buffers.ImmutableDatasetWriter)([compilable]) | A blocking writer for immutable datasets. |
| [`LazyMemmapStorage`](generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage)(max_size, *[, ...]) | A memory-mapped storage for tensors and tensordicts. |
| [`LazyTensorStorage`](generated/torchrl.data.replay_buffers.LazyTensorStorage.html#torchrl.data.replay_buffers.LazyTensorStorage)(max_size, *[, device, ...]) | A pre-allocated tensor storage for tensors and tensordicts. |
| [`ListStorage`](generated/torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage)([max_size, compilable, device]) | A storage stored in a list. |
| [`LazyStackStorage`](generated/torchrl.data.replay_buffers.LazyStackStorage.html#torchrl.data.replay_buffers.LazyStackStorage)([max_size, compilable, ...]) | A ListStorage that returns LazyStackTensorDict instances. |
| [`ListStorageCheckpointer`](generated/torchrl.data.replay_buffers.ListStorageCheckpointer.html#torchrl.data.replay_buffers.ListStorageCheckpointer)() | A storage checkpointer for ListStoage. |
| [`NestedStorageCheckpointer`](generated/torchrl.data.replay_buffers.NestedStorageCheckpointer.html#torchrl.data.replay_buffers.NestedStorageCheckpointer)([done_keys, ...]) | Saves the storage in a compact form, saving space on the TED format and using memory-mapped nested tensors. |
| [`Storage`](generated/torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)(max_size[, checkpointer, compilable]) | A Storage is the container of a replay buffer. |
| [`StorageCheckpointerBase`](generated/torchrl.data.replay_buffers.StorageCheckpointerBase.html#torchrl.data.replay_buffers.StorageCheckpointerBase)() | Public base class for storage checkpointers. |
| [`StorageEnsemble`](generated/torchrl.data.replay_buffers.StorageEnsemble.html#torchrl.data.replay_buffers.StorageEnsemble)(*storages[, transforms]) | An ensemble of storages. |
| [`StorageEnsembleCheckpointer`](generated/torchrl.data.replay_buffers.StorageEnsembleCheckpointer.html#torchrl.data.replay_buffers.StorageEnsembleCheckpointer)() | Checkpointer for ensemble storages. |
| [`TensorStorage`](generated/torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage)(storage[, max_size, device, ...]) | A storage for tensors and tensordicts. |
| [`TensorStorageCheckpointer`](generated/torchrl.data.replay_buffers.TensorStorageCheckpointer.html#torchrl.data.replay_buffers.TensorStorageCheckpointer)() | A storage checkpointer for TensorStorages. |

## Storage Performance

Storage choice is very influential on replay buffer sampling latency, especially
in distributed reinforcement learning settings with larger data volumes.
[`LazyMemmapStorage`](generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage) is highly
advised in distributed settings with shared storage due to the lower serialization
cost of MemoryMappedTensors as well as the ability to specify file storage locations
for improved node failure recovery.