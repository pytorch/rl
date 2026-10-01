# TensorDictRoundRobinWriter

*class*torchrl.data.replay_buffers.TensorDictRoundRobinWriter(*compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictRoundRobinWriter)

A RoundRobin Writer class for composable, tensordict-based replay buffers.

add(*data: Any*) → int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictRoundRobinWriter.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#TensorDictRoundRobinWriter.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.