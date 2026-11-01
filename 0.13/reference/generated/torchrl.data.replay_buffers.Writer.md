# Writer

*class*torchrl.data.replay_buffers.Writer(*compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#Writer)

A ReplayBuffer base Writer class.

*abstract*add(*data: Any*) → int[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#Writer.add)

Inserts one piece of data at an appropriate index, and returns that index.

*abstract*extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#Writer.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.