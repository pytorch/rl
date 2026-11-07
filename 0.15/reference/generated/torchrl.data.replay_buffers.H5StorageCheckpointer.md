# H5StorageCheckpointer

*class*torchrl.data.replay_buffers.H5StorageCheckpointer(***, *checkpoint_file: str = 'checkpoint.h5'*, *done_keys=None*, *reward_keys=None*, *h5_kwargs=None*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/checkpointers.html#H5StorageCheckpointer)

Saves the storage in a compact form, saving space on the TED format and using H5 format to save the data.

This class explicitly assumes and does NOT check that:

> - done states (including terminated and truncated) at the root are always False;
> - observations in the "next" tensordict are shifted by one step in the future (this
> is not the case when a multi-step transform is used for instance).

Keyword Arguments:

- **checkpoint_file** - the filename where to save the checkpointed data.
This will be ignored iff the path passed to dumps / loads ends with the `.h5`
suffix. Defaults to `"checkpoint.h5"`.
- **h5_kwargs** (*Dict**[**str**,**Any**] or**Tuple**[**Tuple**[**str**,**Any**]**,**...**]*) - kwargs to be
passed to `h5py.File.create_dataset()`.

Note

To prevent out-of-memory issues, the data of the H5 file will be temporarily written
on memory-mapped tensors stored in shared file system. The physical memory usage may increase
during loading as a consequence.

register_load_hook(*hook*)

Registers a load hook for this checkpointer.

register_save_hook(*hook*)

Registers a save hook for this checkpointer.