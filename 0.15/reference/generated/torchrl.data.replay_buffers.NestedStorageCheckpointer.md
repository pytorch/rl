# NestedStorageCheckpointer

*class*torchrl.data.replay_buffers.NestedStorageCheckpointer(*done_keys=None*, *reward_keys=None*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#NestedStorageCheckpointer)

Saves the storage in a compact form, saving space on the TED format and using memory-mapped nested tensors.

This class explicitly assumes and does NOT check that:

> - done states (including terminated and truncated) at the root are always False;
> - observations in the "next" tensordict are shifted by one step in the future (this
> is not the case when a multi-step transform is used for instance).

register_load_hook(*hook*)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)

Registers a save hook for this checkpointer.