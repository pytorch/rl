# WriterEnsemble

*class*torchrl.data.replay_buffers.WriterEnsemble(**writers*)

An ensemble of writers.

This class is designed to work with [`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble).
It contains the member writers used by a replay-buffer ensemble.

Parameters:

**writers** (*sequence**of*[*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)) - the writers to make the composite writer.

Warning

This class does not write directly. A routed
[`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble) dispatches writes to its
member writers; otherwise, index the parent ensemble before writing.

add()[[source]](../../_modules/torchrl/data/replay_buffers/writers/ensemble.html#WriterEnsemble.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend()[[source]](../../_modules/torchrl/data/replay_buffers/writers/ensemble.html#WriterEnsemble.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.

generations_of(*index: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/ensemble.html#WriterEnsemble.generations_of)

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

*property*tracks_generations*: bool*

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.