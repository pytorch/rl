# PrioritizedSampler

*class*torchrl.data.replay_buffers.PrioritizedSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#PrioritizedSampler)

Prioritized sampler for replay buffer.

This sampler implements Prioritized Experience Replay (PER) as presented in
"Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay."
([https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952))

**Core Idea**: Instead of sampling experiences uniformly from the replay buffer,
PER samples experiences with probability proportional to their "importance" - typically
measured by the magnitude of their temporal-difference (TD) error. This prioritization
can lead to faster learning by focusing on experiences that are most informative.

**How it works**:
1. Each experience is assigned a priority based on its TD error: \(p_i = |\delta_i| + \epsilon\)
2. Sampling probability is computed as: \(P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}\)
3. Importance sampling weights correct for the bias: \(w_i = (N \cdot P(i))^{-\beta}\)

Parameters:

- **max_capacity** (*int*) - maximum capacity of the buffer.
- **alpha** (`float`) - exponent \(\alpha\) determines how much prioritization is used.
- \(\alpha = 0\): uniform sampling (no prioritization)
- \(\alpha = 1\): full prioritization based on TD error magnitude
- Typical values: 0.4-0.7 for balanced prioritization
- Higher \(\alpha\) means more aggressive prioritization of high-error experiences
- **beta** (`float`) - importance sampling negative exponent \(\beta\).
- \(\beta\) controls the correction for the bias introduced by prioritization
- \(\beta = 0\): no correction (biased towards high-priority samples)
- \(\beta = 1\): full correction (unbiased but potentially unstable)
- Typical values: start at 0.4-0.6 and anneal to 1.0 during training
- Lower \(\beta\) early in training provides stability, higher \(\beta\) later reduces bias
- **eps** (`float`, optional) - small constant added to priorities to ensure
no experience has zero priority. This prevents experiences from never
being sampled. Defaults to 1e-8.
- **reduction** (*str**,**optional*) - the reduction method for multidimensional
tensordicts (ie stored trajectory). Can be one of "max", "min",
"median" or "mean".
- **max_priority_within_buffer** (*bool**,**optional*) - if `True`, the max-priority
is tracked within the buffer. When `False`, the max-priority tracks
the maximum value since the instantiation of the sampler.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device that holds the priority
trees. Defaults to `None`, in which case CUDA storage selects a CUDA
tree when the installed TorchRL extension was built with CUDA support,
and CPU storage keeps the existing CPU tree.

**Parameter Guidelines**:

- **:math:`alpha` (alpha)**: Controls how much to prioritize high-error experiences.
0.4-0.7: Good balance between learning speed and stability.
1.0: Maximum prioritization (may be unstable).
0.0: Uniform sampling (no prioritization benefit).
- **:math:`beta` (beta)**: Controls importance sampling correction.
Start at 0.4-0.6 for training stability.
Anneal to 1.0 over training to reduce bias.
Lower values = more stable but biased.
Higher values = less biased but potentially unstable.
- **:math:`epsilon`**: Small constant to prevent zero priorities.
1e-8: Good default value.
Too small: may cause numerical issues.
Too large: reduces prioritization effect.

Examples

```
>>> from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, PrioritizedSampler
>>> from tensordict import TensorDict
>>> rb = ReplayBuffer(storage=LazyTensorStorage(10), sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0))
>>> priority = torch.tensor([0, 1000])
>>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
>>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
>>> rb.add(data_0)
>>> rb.add(data_1)
>>> rb.update_priority(torch.tensor([0, 1]), priority=priority)
>>> sample, info = rb.sample(10, return_info=True)
>>> print(sample)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 obs: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 priority: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
 reward: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False)
>>> print(info)
{'priority_weight': array([1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11, 1.e-11,
 1.e-11, 1.e-11], dtype=float32), 'index': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
```

Note

Using a `TensorDictReplayBuffer` can smoothen the
process of updating the priorities:

```
>>> from torchrl.data.replay_buffers import TensorDictReplayBuffer as TDRB, LazyTensorStorage, PrioritizedSampler
>>> from tensordict import TensorDict
>>> rb = TDRB(
... storage=LazyTensorStorage(10),
... sampler=PrioritizedSampler(max_capacity=10, alpha=1.0, beta=1.0),
... priority_key="priority", # This kwarg isn't present in regular RBs
... )
>>> priority = torch.tensor([0, 1000])
>>> data_0 = TensorDict({"reward": 0, "obs": [0], "action": [0], "priority": priority[0]}, [])
>>> data_1 = TensorDict({"reward": 1, "obs": [1], "action": [2], "priority": priority[1]}, [])
>>> data = torch.stack([data_0, data_1])
>>> rb.extend(data)
>>> rb.update_priority(data) # Reads the "priority" key as indicated in the constructor
>>> sample, info = rb.sample(10, return_info=True)
>>> print(sample['index']) # The index is packed with the tensordict
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

*property*alpha

The priority exponent.

Note

Setting `alpha` on a sampler that already holds priorities
(e.g. when annealing it with a
`ParameterScheduler`)
re-transforms the `(p + eps) ** alpha` values stored in the
sum/min trees to the new exponent in a single O(capacity) pass, so
sampling probabilities stay consistent with the new value. The one
exception is changing `alpha` away from exactly `0`: the raw
priorities cannot be recovered from the trees in that regime, so the
stored (uniform) values are kept - and a warning is emitted -
until each entry's priority is next updated.

update_priority(*index: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *priority: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *storage: [TensorStorage](torchrl.data.replay_buffers.TensorStorage.html#torchrl.data.replay_buffers.TensorStorage) | None = None*) → None[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#PrioritizedSampler.update_priority)

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