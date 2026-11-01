# SliceSamplerWithoutReplacement

*class*torchrl.data.replay_buffers.SliceSamplerWithoutReplacement(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#SliceSamplerWithoutReplacement)

Samples slices of data along the first dimension, given start and stop signals, without replacement.

In this context, `without replacement` means that the same element (NOT trajectory) will not be sampled twice
before the counter is automatically reset. Within a single sample, however, only one slice of a given trajectory
will appear (see example below).

This class is to be used with static replay buffers or in between two
replay buffer extensions. Extending the replay buffer will reset the
the sampler, and continuous sampling without replacement is currently not
allowed.

Note

SliceSamplerWithoutReplacement can be slow to retrieve the trajectory indices. To accelerate
its execution, prefer using end_key over traj_key, and consider the following
keyword arguments: `compile`, `cache_values` and `use_gpu`.

Keyword Arguments:

- **drop_last** (*bool**,**optional*) - if `True`, the last incomplete sample (if any) will be dropped.
If `False`, this last sample will be kept.
Defaults to `False`.
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
- **shuffle** (*bool**,**optional*) - if `False`, the order of the trajectories
is not shuffled. Defaults to `True`.
- **compile** (*bool**or**dict**of**kwargs**,**optional*) - if `True`, the bottleneck of
the `sample()` method will be compiled with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
Keyword arguments can also be passed to torch.compile with this arg.
Defaults to `False`.
- **use_gpu** (*bool**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - if `True` (or is a device is passed), an accelerator
will be used to retrieve the indices of the trajectory starts. This can significantly
accelerate the sampling when the buffer content is large.
Defaults to `False`.

Note

To recover the trajectory splits in the storage,
`SliceSamplerWithoutReplacement` will first
attempt to find the `traj_key` entry in the storage. If it cannot be
found, the `end_key` will be used to reconstruct the episodes.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
>>> from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
>>>
>>> rb = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(1000),
... # asking for 10 slices for a total of 320 elements, ie, 10 trajectories of 32 transitions each
... sampler=SliceSamplerWithoutReplacement(num_slices=10),
... batch_size=320,
... )
>>> episode = torch.zeros(1000, dtype=torch.int)
>>> episode[:300] = 1
>>> episode[300:550] = 2
>>> episode[550:700] = 3
>>> episode[700:] = 4
>>> data = TensorDict(
... {
... "episode": episode,
... "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
... "act": torch.randn((20,)).expand(1000, 20),
... "other": torch.randn((20, 50)).expand(1000, 20, 50),
... }, [1000]
... )
>>> rb.extend(data)
>>> sample = rb.sample()
>>> # since we want trajectories of 32 transitions but there are only 4 episodes to
>>> # sample from, we only get 4 x 32 = 128 transitions in this batch
>>> print("sample:", sample)
>>> print("trajectories in sample", sample.get("episode").unique())
```

`SliceSamplerWithoutReplacement` is default-compatible with
most of TorchRL's datasets, and allows users to consume datasets in a dataloader-like fashion:

Examples

```
>>> import torch
>>>
>>> from torchrl.data.datasets import RobosetExperienceReplay
>>> from torchrl.data import SliceSamplerWithoutReplacement
>>>
>>> torch.manual_seed(0)
>>> num_slices = 10
>>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
>>> data = RobosetExperienceReplay(dataid, batch_size=320,
... sampler=SliceSamplerWithoutReplacement(num_slices=num_slices))
>>> # the last sample is kept, since drop_last=False by default
>>> for i, batch in enumerate(data):
... print(batch.get("episode").unique())
tensor([ 5, 6, 8, 11, 12, 14, 16, 17, 19, 24])
tensor([ 1, 2, 7, 9, 10, 13, 15, 18, 21, 22])
tensor([ 0, 3, 4, 20, 23])
```

When requesting a large total number of samples with few trajectories and small span, the batch will contain
only at most one sample of each trajectory:

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.collectors.utils import split_trajectories
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler, SliceSamplerWithoutReplacement
>>>
>>> rb = ReplayBuffer(storage=LazyTensorStorage(max_size=1000),
... sampler=SliceSamplerWithoutReplacement(
... slice_len=5, traj_key="episode",strict_length=False
... ))
...
>>> ep_1 = TensorDict(
... {"obs": torch.arange(100),
... "episode": torch.zeros(100),},
... batch_size=[100]
... )
>>> ep_2 = TensorDict(
... {"obs": torch.arange(51),
... "episode": torch.ones(51),},
... batch_size=[51]
... )
>>> rb.extend(ep_1)
>>> rb.extend(ep_2)
>>>
>>> s = rb.sample(50)
>>> t = split_trajectories(s, trajectory_key="episode")
>>> print(t["obs"])
tensor([[14, 15, 16, 17, 18],
 [ 3, 4, 5, 6, 7]])
>>> print(t["episode"])
tensor([[0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 1.]])
>>>
>>> s = rb.sample(50)
>>> t = split_trajectories(s, trajectory_key="episode")
>>> print(t["obs"])
tensor([[ 4, 5, 6, 7, 8],
 [26, 27, 28, 29, 30]])
>>> print(t["episode"])
tensor([[0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 1.]])
```