# ConsumingSampler

*class*torchrl.data.replay_buffers.ConsumingSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#ConsumingSampler)

A random sampler that consumes entries after they have been sampled.

`ConsumingSampler` tracks how many times each storage index has been
returned by `sample()`. Once an index has been returned
`max_sample_count` times, it is removed from the set of sampleable
indices until that slot is overwritten by the replay-buffer writer. When
used through [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer), consumed indices are kept
in a free-list so writes can reuse those slots before advancing the normal
writer cursor.

Parameters:

**max_sample_count** (*int**,**optional*) - number of returned samples after which
an item is consumed. Defaults to `1`.

Examples

```
>>> import torch
>>>
>>> from torchrl.data import ConsumingSampler, ListStorage, ReplayBuffer
>>> rb = ReplayBuffer(
... storage=ListStorage(10),
... sampler=ConsumingSampler(),
... batch_size=3,
... )
>>> rb.extend([torch.tensor(i) for i in range(4)])
tensor([0, 1, 2, 3])
>>> sample = rb.sample()
>>> len(sample)
3
>>> len(rb)
1
```

Note

`ConsumingSampler` only supports 1-dimensional storages and uniform
random sampling without replacement within each sampled batch.
Prefetching and prioritized replay are not supported.