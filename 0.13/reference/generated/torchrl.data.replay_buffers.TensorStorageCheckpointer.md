# TensorStorageCheckpointer

*class*torchrl.data.replay_buffers.TensorStorageCheckpointer[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#TensorStorageCheckpointer)

A storage checkpointer for TensorStorages.

This class supports TensorDict-based storages as well as pytrees.

This class will call save and load hooks if provided. These hooks should take as input the
data being transformed as well as the path where the data should be saved.

register_load_hook(*hook*)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)

Registers a save hook for this checkpointer.