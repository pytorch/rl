# TensorDictRoundRobinWriter

*class*torchrl.data.replay_buffers.TensorDictRoundRobinWriter(*compilable: bool = False*, ***, *track_generations: bool = False*)

A RoundRobin Writer class for composable, tensordict-based replay buffers.

See Also:
`TensorDictRoundRobinWriterConfig`.

Takes the same arguments as [`RoundRobinWriter`](torchrl.data.replay_buffers.RoundRobinWriter.html#torchrl.data.replay_buffers.RoundRobinWriter), including
`track_generations`. When enabled, `"index_generation"` is written into
the sampled tensordict alongside `"index"`.

add(*data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#TensorDictRoundRobinWriter.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#TensorDictRoundRobinWriter.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.

generations_of(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

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

write_at(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#TensorDictRoundRobinWriter.write_at)

Writes data at explicit storage indices without moving the cursor.

The generation of every written slot is bumped, so handles previously
handed out for those slots are stale once this returns.