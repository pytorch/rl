# PromptGroupSampler

*class*torchrl.data.replay_buffers.PromptGroupSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#PromptGroupSampler)

A sampler that draws complete groups of items sharing a common key.

This sampler partitions the storage into groups whose items share the same
value under `group_key` (for LLM post-training, the prompt or query). Every
call to [`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample) returns
`samples_per_group` items for each of `num_groups` selected groups, so a
batch is laid out as balanced groups rather than independent items. This is
the layout required by group-relative objectives such as
[`GRPOLoss`](torchrl.objectives.llm.GRPOLoss.html#torchrl.objectives.llm.GRPOLoss).

Sampling never consumes the storage, so past generations for a group remain
available and can be replayed across policy updates. Combined with a
persistent replay buffer (one that is not emptied between iterations), this
turns an on-policy GRPO loop into the replay-enhanced regime of RePO
("RePO: Replay-Enhanced Policy Optimization", Li et al. 2025,
[https://arxiv.org/abs/2506.09340](https://arxiv.org/abs/2506.09340)), where each update mixes fresh on-policy
groups with off-policy groups retrieved from the buffer.

Keyword Arguments:

- **num_groups** (*int**,**optional*) - the number of distinct groups to draw per
batch. Exactly one of `num_groups` or `samples_per_group` must be
provided; the other is inferred from the `batch_size` passed to
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).
- **samples_per_group** (*int**,**optional*) - the number of items to draw from each
selected group. Exactly one of `num_groups` or
`samples_per_group` must be provided.
- **group_key** (*NestedKey**,**optional*) - the tensordict key identifying the group
each item belongs to. Stored values may be integers (e.g. a prompt
id) or strings (e.g. the prompt text). Defaults to `"query"`.
- **strategy** (*str**,**optional*) -

the retrieval strategy. One of:

- `"random"` (default): groups are chosen uniformly at random and
items within a group are drawn uniformly at random.
- `"recency"`: the most recently inserted items are drawn from
each group.
- `"reward"`: the highest-reward items are drawn from each group.
- `"variance"`: the fixed-size subset that maximizes reward
variance is drawn from each group, breaking ties by total reward.
This targets the vanishing-gradient case described by RePO.
- **reward_key** (*NestedKey**,**optional*) - the key holding a numeric reward,
required by the `"reward"` and `"variance"` strategies. It is
reduced to one scalar per item by averaging over any trailing
dimensions. Defaults to `("next", "reward")`.
- **cache_groups** (*bool**,**optional*) - if `True` (default), the group index is
cached and rebuilt only when items are added to the storage. Set to
`False` if the stored group values may change in place.

Note

This sampler supports single-dimensional TensorDict-backed
storages, including `LazyTensorStorage`,
`LazyMemmapStorage`, and
`LazyStackStorage`. Plain
`ListStorage` is unsupported because its slices
return Python lists.

Warning

When a group holds fewer than `samples_per_group` items (or
the storage holds fewer than `num_groups` groups), the missing draws
are completed by sampling with replacement and a warning is raised once.

See also

[`MCAdvantage`](torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage), the group-relative
advantage engine these batches feed into.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import LazyStackStorage, ReplayBuffer
>>> from torchrl.data.replay_buffers.samplers import PromptGroupSampler
>>> rb = ReplayBuffer(
... storage=LazyStackStorage(100),
... sampler=PromptGroupSampler(num_groups=2, group_key="prompt"),
... batch_size=8,
... )
>>> data = TensorDict(
... {
... "prompt": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
... "reward": torch.arange(12.0),
... },
... batch_size=[12],
... )
>>> _ = rb.extend(data)
>>> sample = rb.sample()
>>> int(sample["prompt"].unique().numel())
2
>>> int(sample.shape[0])
8
```