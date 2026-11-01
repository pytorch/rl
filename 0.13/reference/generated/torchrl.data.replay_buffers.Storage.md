# Storage

*class*torchrl.data.replay_buffers.Storage(*max_size: int*, *checkpointer: [StorageCheckpointerBase](torchrl.data.replay_buffers.StorageCheckpointerBase.html#torchrl.data.replay_buffers.StorageCheckpointerBase) | None = None*, *compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage)

A Storage is the container of a replay buffer.

Every storage must have a set, get and __len__ methods implemented.
Get and set should support integers as well as list of integers.

The storage does not need to have a definite size, but if it does one should
make sure that it is compatible with the buffer size.

attach(*buffer: Any*) → None[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.attach)

This function attaches a sampler to this storage.

Buffers that read from this storage must be included as an attached
entity by calling this method. This guarantees that when data
in the storage changes, components are made aware of changes even if the storage
is shared with other buffers (eg. Priority Samplers).

Parameters:

**buffer** - the object that reads from this storage.

dump(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.dump)

Alias for `dumps()`.

load(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.load)

Alias for `loads()`.

register_load_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.register_load_hook)

Register a load hook for this storage.

The hook is forwarded to the checkpointer.

register_save_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.register_save_hook)

Register a save hook for this storage.

The hook is forwarded to the checkpointer.

save(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#Storage.save)

Alias for `dumps()`.