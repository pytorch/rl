# RoundRobinWriter

*class*torchrl.data.replay_buffers.RoundRobinWriter(*compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#RoundRobinWriter)

A RoundRobin Writer class for composable replay buffers.

Parameters:

**compilable** (*bool**,**optional*) - whether the writer is compilable.
If `True`, the writer cannot be shared between multiple processes.
Defaults to `False`.

add(*data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#RoundRobinWriter.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#RoundRobinWriter.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.

write_at(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#RoundRobinWriter.write_at)

Writes data at explicit storage indices without moving the cursor.