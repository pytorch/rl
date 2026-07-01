# CompressedListStorageCheckpointer

*class*torchrl.data.replay_buffers.CompressedListStorageCheckpointer[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#CompressedListStorageCheckpointer)

A storage checkpointer for CompressedListStorage.

This checkpointer saves compressed data and metadata using memory-mapped storage
for efficient disk I/O and memory usage.

dumps(*storage*, *path*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#CompressedListStorageCheckpointer.dumps)

Save compressed storage to disk using memory-mapped storage.

Parameters:

- **storage** - The CompressedListStorage instance to save
- **path** - Directory path where to save the storage

loads(*storage*, *path*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#CompressedListStorageCheckpointer.loads)

Load compressed storage from disk.

Parameters:

- **storage** - The CompressedListStorage instance to load into
- **path** - Directory path where the storage was saved

register_load_hook(*hook*)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)

Registers a save hook for this checkpointer.