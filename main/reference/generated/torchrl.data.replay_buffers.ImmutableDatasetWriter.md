# ImmutableDatasetWriter

*class*torchrl.data.replay_buffers.ImmutableDatasetWriter(*compilable: bool = False*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#ImmutableDatasetWriter)

A blocking writer for immutable datasets.

add(*data: Any*) → int[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#ImmutableDatasetWriter.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend(*data: Sequence*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#ImmutableDatasetWriter.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.