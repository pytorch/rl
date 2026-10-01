# PrioritizedSliceSampler

*class*torchrl.data.replay_buffers.PrioritizedSliceSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#PrioritizedSliceSampler)

Samples slices of data along the first dimension, given start and stop signals, using prioritized sampling.

This class combines trajectory sampling with Prioritized Experience Replay (PER) as presented in
"Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay."
([https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952))

**Core Idea**: Instead of sampling trajectory slices uniformly, this sampler prioritizes
trajectory start points based on the importance of the transitions at those positions.
This allows focusing learning on the most informative parts of trajectories.

**How it works**:
1. Each transition is assigned a priority based on its TD error: \(p_i = |\\delta_i| + \\epsilon\)
2. Trajectory start points are sampled with probability: \(P(i) = \frac{p_i^\alpha}{\\sum_j p_j^\alpha}\)
3. Importance sampling weights correct for bias: \(w_i = (N \\cdot P(i))^{-\beta}\)
4. Complete trajectory slices are extracted from the sampled start points

For more info see [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) and [`PrioritizedSampler`](torchrl.data.replay_buffers.PrioritizedSampler.html#torchrl.data.replay_buffers.PrioritizedSampler).

Warning

PrioritizedSliceSampler will look at the priorities of the individual transitions and sample the
start points accordingly. This means that transitions with a low priority may as well appear in the
samples if they follow another of higher priority, and transitions with a high priority but closer to the
end of a trajectory may never be sampled if they cannot be used as start points.
Currently, it is the user responsibility to aggregate priorities across items of a trajectory using
`update_priority()`.

Parameters:

- **max_capacity** (*int*) - maximum capacity of the buffer.
- **alpha** (`float`) - exponent \(\alpha\) determines how much prioritization is used.
- \(\alpha = 0\): uniform sampling of trajectory start points
- \(\alpha = 1\): full prioritization based on TD error magnitude at start points
- Typical values: 0.4-0.7 for balanced prioritization
- Higher \(\alpha\) means more aggressive prioritization of high-error trajectory regions
- **beta** (`float`) - importance sampling negative exponent \(\beta\).
- \(\beta\) controls the correction for the bias introduced by prioritization
- \(\beta = 0\): no correction (biased towards high-priority trajectory regions)
- \(\beta = 1\): full correction (unbiased but potentially unstable)
- Typical values: start at 0.4-0.6 and anneal to 1.0 during training
- Lower \(\beta\) early in training provides stability, higher \(\beta\) later reduces bias
- **eps** (`float`, optional) - small constant added to priorities to ensure
no transition has zero priority. This prevents trajectory regions from never
being sampled. Defaults to 1e-8.
- **reduction** (*str**,**optional*) - the reduction method for multidimensional
tensordicts (i.e., stored trajectory). Can be one of "max", "min",
"median" or "mean".

**Parameter Guidelines**:

- **:math:`alpha` (alpha)**: Controls how much to prioritize high-error trajectory regions.
0.4-0.7: Good balance between learning speed and stability.
1.0: Maximum prioritization (may be unstable).
0.0: Uniform sampling (no prioritization benefit).
- **:math:`beta` (beta)**: Controls importance sampling correction.
Start at 0.4-0.6 for training stability.
Anneal to 1.0 over training to reduce bias.
Lower values = more stable but biased.
Higher values = less biased but potentially unstable.
- **:math:`\epsilon`**: Small constant to prevent zero priorities.
1e-8: Good default value.
Too small: may cause numerical issues.
Too large: reduces prioritization effect.

Keyword Arguments:

- **num_slices** (*int*) - the number of slices to be sampled. The batch-size
must be greater or equal to the `num_slices` argument. Exclusive
with `slice_len`.
- **slice_len** (*int*) - the length of the slices to be sampled. The batch-size
must be greater or equal to the `slice_len` argument and divisible
by it. Exclusive with `num_slices`.
- **end_key** (*NestedKey**,**optional*) - the key indicating the end of a
trajectory (or episode). Defaults to `("next", "done")`.
- **traj_key** (*NestedKey**,**optional*) - the key indicating the trajectories.
Defaults to `"episode"` (commonly used across datasets in TorchRL).
- **ends** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a 1d boolean tensor containing the end of run signals.
To be used whenever the `end_key` or `traj_key` is expensive to get,
or when this signal is readily available. Must be used with `cache_values=True`
and cannot be used in conjunction with `end_key` or `traj_key`.
- **trajectories** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a 1d integer tensor containing the run ids.
To be used whenever the `end_key` or `traj_key` is expensive to get,
or when this signal is readily available. Must be used with `cache_values=True`
and cannot be used in conjunction with `end_key` or `traj_key`.
- **cache_values** (*bool**,**optional*) -

to be used with static datasets.
Will cache the start and end signal of the trajectory. This can be safely used even
if the trajectory indices change during calls to [`extend`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend)
as this operation will erase the cache.

Warning

`cache_values=True` will not work if the sampler is used with a
storage that is extended by another buffer. For instance:

```
>>> buffer0 = ReplayBuffer(storage=storage,
... sampler=SliceSampler(num_slices=8, cache_values=True),
... writer=ImmutableWriter())
>>> buffer1 = ReplayBuffer(storage=storage,
... sampler=other_sampler)
>>> # Wrong! Does not erase the buffer from the sampler of buffer0
>>> buffer1.extend(data)
```

Warning

`cache_values=True` will not work as expected if the buffer is
shared between processes and one process is responsible for writing
and one process for sampling, as erasing the cache can only be done locally.
- **truncated_key** (*NestedKey**,**optional*) - If not `None`, this argument
indicates where a truncated signal should be written in the output
data. This is used to indicate to value estimators where the provided
trajectory breaks. Defaults to `("next", "truncated")`.
This feature only works with `TensorDictReplayBuffer`
instances (otherwise the truncated key is returned in the info dictionary
returned by the `sample()` method).
- **strict_length** (*bool**,**optional*) - if `False`, trajectories of length
shorter than slice_len (or batch_size // num_slices) will be
allowed to appear in the batch. If `True`, trajectories shorted
than required will be filtered out.
Be mindful that this can result in effective batch_size shorter
than the one asked for! Trajectories can be split using
`split_trajectories()`. Defaults to `True`.
- **compile** (*bool**or**dict**of**kwargs**,**optional*) - if `True`, the bottleneck of
the `sample()` method will be compiled with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
Keyword arguments can also be passed to torch.compile with this arg.
Defaults to `False`.
- **span** (*bool**,**int**,**Tuple**[**bool**|**int**,**bool**|**int**]**,**optional*) - if provided, the sampled
trajectory will span across the left and/or the right. This means that possibly
fewer elements will be provided than what was required. A boolean value means
that at least one element will be sampled per trajectory. An integer i means
that at least slice_len - i samples will be gathered for each sampled trajectory.
Using tuples allows a fine grained control over the span on the left (beginning
of the stored trajectory) and on the right (end of the stored trajectory).
- **max_priority_within_buffer** (*bool**,**optional*) - if `True`, the max-priority
is tracked within the buffer. When `False`, the max-priority tracks
the maximum value since the instantiation of the sampler.
Defaults to `False`.

Examples

```
>>> import torch
>>> from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage, PrioritizedSliceSampler
>>> from tensordict import TensorDict
>>> sampler = PrioritizedSliceSampler(max_capacity=9, num_slices=3, alpha=0.7, beta=0.9)
>>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(9), sampler=sampler, batch_size=6)
>>> data = TensorDict(
... {
... "observation": torch.randn(9,16),
... "action": torch.randn(9, 1),
... "episode": torch.tensor([0,0,0,1,1,1,2,2,2], dtype=torch.long),
... "steps": torch.tensor([0,1,2,0,1,2,0,1,2], dtype=torch.long),
... ("next", "observation"): torch.randn(9,16),
... ("next", "reward"): torch.randn(9,1),
... ("next", "done"): torch.tensor([0,0,1,0,0,1,0,0,1], dtype=torch.bool).unsqueeze(1),
... },
... batch_size=[9],
... )
>>> rb.extend(data)
>>> sample, info = rb.sample(return_info=True)
>>> print("episode", sample["episode"].tolist())
episode [2, 2, 2, 2, 1, 1]
>>> print("steps", sample["steps"].tolist())
steps [1, 2, 0, 1, 1, 2]
>>> print("weight", info["priority_weight"].tolist())
weight [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
>>> priority = torch.tensor([0,3,3,0,0,0,1,1,1])
>>> rb.update_priority(torch.arange(0,9,1), priority=priority)
>>> sample, info = rb.sample(return_info=True)
>>> print("episode", sample["episode"].tolist())
episode [2, 2, 2, 2, 2, 2]
>>> print("steps", sample["steps"].tolist())
steps [1, 2, 0, 1, 0, 1]
>>> print("weight", info["priority_weight"].tolist())
weight [9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06, 9.120110917137936e-06]
```

*property*alpha

The priority exponent.

Note

Changing `alpha` on a sampler that already holds priorities
(e.g. when annealing it with a
`ParameterScheduler`)
does not re-transform the `(p + eps) ** alpha` values already
written to the sum/min trees: old and new exponents mix until every
entry's priority is updated again.

update_priority(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *priority: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *storage: [TensorStorage](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage) | None = None*) → None

Updates the priority of the data pointed by the index.

Parameters:

- **index** (*int**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - indexes of the priorities to be
updated.
- **priority** (*Number**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - new priorities of the
indexed elements.

Keyword Arguments:

**storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*,**optional*) - a storage used to map the Nd index size to
the 1d size of the sum_tree and min_tree. Only required whenever
`index.ndim > 2`.