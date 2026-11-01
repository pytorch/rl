# RoundRobinWriter

*class*torchrl.data.replay_buffers.RoundRobinWriter(*compilable: bool = False*, ***, *track_generations: bool = False*)

A RoundRobin Writer class for composable replay buffers.

See also `RoundRobinWriterConfig`.

Parameters:

**compilable** (*bool**,**optional*) - whether the writer is compilable.
If `True`, the writer cannot be shared between multiple processes.
Defaults to `False`.

Keyword Arguments:

**track_generations** (*bool**,**optional*) - if `True`, stamp every storage
slot with a counter that advances each time the slot is written, so
a consumer holding an index can tell whether the slot still holds
the data it sampled. Reads are exposed through
`generations_of()`, and [`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample)
adds an `"index_generation"` entry to its `info` (and, for
tensordict buffers, to the sample). Defaults to `False`: enabling
it allocates one `int64` slot per storage slot and adds a key to
the sampler output, so it is opt-in.

Note

The generation buffer lives on the *storage*, not on the writer, so two
buffers sharing one storage observe each other's writes. It is
process-local: a slot overwritten in another process is not reflected
here. See ref_buffers_generations.

Examples

```
>>> import torch
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(4),
... writer=RoundRobinWriter(track_generations=True),
... )
>>> index = rb.extend(torch.arange(4))
>>> rb.writer.generations_of(index)
tensor([0, 0, 0, 0])
>>> _ = rb.extend(torch.arange(4, 6)) # overwrites slots 0 and 1
>>> rb.writer.generations_of(index)
tensor([1, 1, 0, 0])
```

add(*data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#RoundRobinWriter.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#RoundRobinWriter.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.

generations_of(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#RoundRobinWriter.generations_of)

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

write_at(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers/round_robin.html#RoundRobinWriter.write_at)

Writes data at explicit storage indices without moving the cursor.

The generation of every written slot is bumped, so handles previously
handed out for those slots are stale once this returns.