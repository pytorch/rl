# FlatStorageCheckpointer

*class*torchrl.data.replay_buffers.FlatStorageCheckpointer(*done_keys=None*, *reward_keys=None*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#FlatStorageCheckpointer)

Saves the storage in a compact form, saving space on the TED format.

This class explicitly assumes and does NOT check that:

> - done states (including terminated and truncated) at the root are always False;
> - observations in the "next" tensordict are shifted by one step in the future (this
> is not the case when a multi-step transform is used for instance) unless done is True
> in which case the observation in ("next", key) at time t and the one in key at time
> t+1 should not match.

register_load_hook(*hook*)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)

Registers a save hook for this checkpointer.