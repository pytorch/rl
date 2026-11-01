# StreamingSliceSampler

*class*torchrl.data.replay_buffers.StreamingSliceSampler(**args*, ***kwargs*)

A slice sampler that prioritizes newly completed streaming windows.

Incoming records are partitioned into non-overlapping, fixed-length
windows without crossing trajectory or done boundaries. Each completed
window is queued once and returned before the sampler falls back to
regular uniform [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) sampling. Generation-like write
stamps maintained by the sampler discard queued windows whose ring slots
were overwritten before they could be sampled.

The sampler observes writes through the standard replay-buffer
`mark_update` notification. It is intended for one-dimensional storages
whose writes arrive in chronological order, such as one replay-buffer
member per environment stream.

See Also
`StreamingSliceSamplerConfig`.

Keyword Arguments:

- **slice_len** (*int*) - fixed length of every queued and sampled slice.
- **end_key** (*NestedKey**,**optional*) - single end-of-trajectory key. Defaults
to `("next", "done")` when no trajectory key is available.
- **end_keys** (*sequence**of**NestedKey**,**optional*) - boundary keys to combine
with a logical OR. Exclusive with `end_key`.
- **traj_key** (*NestedKey**,**optional*) - trajectory identifier. When no
boundary key is configured, the usual collector trajectory key is
detected automatically before falling back to done markers.
- **cache_values** (*bool**,**optional*) - cache the uniform fallback trajectory
index. Defaults to `False`.
- **truncated_key** (*NestedKey**,**optional*) - key populated in sampling info at
the final record of each sampled slice. Defaults to
`("next", "truncated")`.
- **init_key** (*NestedKey**,**optional*) - If not `None`, the sampler marks the
first step of every slice with `True` under this key (OR-ed with
the flags stored in the buffer, when present) so that recurrent
modules restart from the stored hidden state at each slice start.
Pass `None` to leave the stored flags untouched, as required by
models that reset their state wherever `is_init` is set, such as
the DreamerV3 RSSM rollout. Defaults to `"is_init"`.
- **strict_length** (*bool**,**optional*) - whether uniform fallback sampling
rejects trajectories shorter than `slice_len`. Defaults to
`True`.
- **pad_output** (*bool**,**optional*) - whether short uniform fallback slices are
padded when `strict_length=False`. Defaults to `False`.
- **compile** (*bool**or**dict**,**optional*) - compile options forwarded to
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler). Defaults to `False`.
- **span** (*bool**,**int**or**pair**,**optional*) - span options forwarded to
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler). Defaults to `False`.
- **use_gpu** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**bool**,**optional*) - boundary-index device option
forwarded to [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler). Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import (
... LazyTensorStorage,
... StreamingSliceSampler,
... TensorDictReplayBuffer,
... )
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(16),
... sampler=StreamingSliceSampler(slice_len=3),
... batch_size=6,
... )
>>> done = torch.zeros(6, 1, dtype=torch.bool)
>>> rb.extend(
... TensorDict(
... {"value": torch.arange(6), ("next", "done"): done},
... batch_size=[6],
... )
... )
>>> rb.sample()["value"].reshape(2, 3)
tensor([[0, 1, 2],
 [3, 4, 5]])
```

can_sample(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *batch_size: int*) → bool

Returns whether the sampler can draw the requested batch.