# WriterEnsemble

*class*torchrl.data.replay_buffers.WriterEnsemble(**writers*)[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#WriterEnsemble)

An ensemble of writers.

This class is designed to work with [`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble).
It contains the writers but blocks writing with any of them.

Parameters:

**writers** (*sequence**of*[*Writer*](torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)) - the writers to make the composite writer.

Warning

This class does not support writing.
To extend one of the replay buffers, simply index the parent
[`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble) object.

add()[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#WriterEnsemble.add)

Inserts one piece of data at an appropriate index, and returns that index.

extend()[[source]](../../_modules/torchrl/data/replay_buffers/writers.html#WriterEnsemble.extend)

Inserts a series of data points at appropriate indices, and returns a tensor containing the indices.