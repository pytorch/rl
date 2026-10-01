# StorageCheckpointerBase

*class*torchrl.data.replay_buffers.StorageCheckpointerBase[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#StorageCheckpointerBase)

Public base class for storage checkpointers.

Each storage checkpointer must implement a save and load method that take as input a storage and a
path.

register_load_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#StorageCheckpointerBase.register_load_hook)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#StorageCheckpointerBase.register_save_hook)

Registers a save hook for this checkpointer.