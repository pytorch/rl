# SliceSampler

*class*torchrl.data.replay_buffers.SliceSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#SliceSampler)

Samples slices of data along the first dimension, given start and stop signals.

This class samples sub-trajectories with replacement. For a version without
replacement, see [`SliceSamplerWithoutReplacement`](torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement).
Equivalently, `SliceSampler(replacement=False, ...)` dispatches to
[`SliceSamplerWithoutReplacement`](torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement) and forwards the remaining keyword
arguments (including `drop_last` and `shuffle`).

Note

SliceSampler can be slow to retrieve the trajectory indices. To accelerate
its execution, prefer using end_key over traj_key, and consider the following
keyword arguments: `compile`, `cache_values` and `use_gpu`.

Keyword Arguments:

- **replacement** (*bool**,**optional*) - if `False`, the call is dispatched to
[`SliceSamplerWithoutReplacement`](torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement) (which accepts the same
keyword arguments as well as `drop_last` and `shuffle`).
Defaults to `True`.
- **num_slices** (*int*) - the number of slices to be sampled. The batch-size
must be greater or equal to the `num_slices` argument. Exclusive
with `slice_len`.
- **slice_len** (*int*) - the length of the slices to be sampled. The batch-size
must be greater or equal to the `slice_len` argument and divisible
by it. Exclusive with `num_slices`.
- **end_key** (*NestedKey**,**optional*) -

the key indicating the end of a
trajectory (or episode). Defaults to `("next", "done")`.
Exclusive with `end_keys`.

Note

A single `end_key` misses trajectories whose end is
marked by another flag only (e.g. datasets carrying
`truncated=True` ends without an aggregate `done` entry
- those get silently merged with the next trajectory). Pass
`end_keys` to apply the
[`DEFAULT_DONE_KEYS`](../data_replaybuffers.html#torchrl.data.DEFAULT_DONE_KEYS) union convention.
- **end_keys** (*sequence**of**NestedKey**,**optional*) - a sequence of keys whose
entries are OR-ed together to build the end-of-trajectory signal.
Keys absent from the storage are skipped (at least one must be
present). Use
`[("next", key) for key in DEFAULT_DONE_KEYS]` to union
`done`, `truncated` and `terminated`. Exclusive with
`end_key`. Defaults to `None` (use `end_key`).
- **traj_key** (*NestedKey**,**optional*) - the key indicating the trajectories.
Defaults to `"episode"` (commonly used across datasets in TorchRL).
- **ends** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a 1d boolean tensor containing the end of run signals.
To be used whenever the `end_key` or `traj_key` is expensive to get,
or when this signal is readily available. Must be used with `cache_values=True`
and cannot be used in conjunction with `end_key` or `traj_key`.
If provided, it is assumed that the storage is at capacity and that
if the last element of the `ends` tensor is `False`,
the same trajectory spans across end and beginning.
- **trajectories** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a 1d integer tensor containing the run ids.
To be used whenever the `end_key` or `traj_key` is expensive to get,
or when this signal is readily available. Must be used with `cache_values=True`
and cannot be used in conjunction with `end_key` or `traj_key`.
If provided, it is assumed that the storage is at capacity and that
if the last element of the trajectory tensor is identical to the first,
the same trajectory spans across end and beginning.
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
- **pad_output** (*bool**,**optional*) - **discouraged. Prefer the default
(``False``).** When `True` (and `strict_length=False`),
short trajectories are padded by *duplicating their last real
timestep* up to `slice_len` so the output's `B * T` is a
fixed product. The output is still a 1D batch of shape
`[B * T]` -- the sample is not reshaped to `[B, T]`. A 1D
boolean mask of shape `[B * T]` is written to
`("collector", "mask")` flagging real (`True`) vs
duplicated-last-step (`False`) positions. TorchRL's primitives
(recurrent modules under
[`set_recurrent_mode()`](torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode), mask-aware loss
modules, `split_trajectories`, etc.) are all designed to
consume concatenated variable-length slices directly via the
`is_init` / `truncated` markers the sampler already emits,
so padding is a niche escape hatch for downstream code that
genuinely cannot accept a ragged batch (e.g. a custom op that
requires a fixed time dimension before a manual reshape).
Combining `pad_output=True` with `strict_length=True` raises
`ValueError`. Defaults to `False`.
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
- **use_gpu** (*bool**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - if `True` (or is a device is passed), an accelerator
will be used to retrieve the indices of the trajectory starts. This can significantly
accelerate the sampling when the buffer content is large.
Defaults to `False`.

Note

To recover the trajectory splits in the storage,
`SliceSampler` will first
attempt to find the `traj_key` entry in the storage. If it cannot be
found, the `end_key` will be used to reconstruct the episodes.

Note

When using a multi-process collector
([`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) or
[`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)) with a shared replay
buffer, adjacent transitions in the buffer may come from different
workers and different episodes. A `SliceSampler` that relies on
`end_key` can then sample slices that straddle unrelated trajectories.

To avoid this, either:

- set `trajs_per_batch` on the collector so that only **complete**
trajectories (each ending with `done=True`) are written to the
buffer (use `ndim=1` on the storage -- `ndim >= 2` is
incompatible with the variable-length flat sequences that
`trajs_per_batch` produces), or
- set `set_truncated=True` on the collector so that every batch
boundary carries a `done` signal (note: this introduces artificial
truncations that value estimators must account for).

Note

When using strict_length=False, it is recommended to use
[`split_trajectories()`](torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) to split the sampled trajectories.
However, if two samples from the same episode are placed next to each other,
this may produce incorrect results. To avoid this issue, consider one of these solutions:

- using a [`TensorDictReplayBuffer`](torchrl.data.TensorDictReplayBuffer.html#torchrl.data.TensorDictReplayBuffer) instance with the slice sampler

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.collectors.utils import split_trajectories
>>> from torchrl.data import TensorDictReplayBuffer, ReplayBuffer, LazyTensorStorage, SliceSampler, SliceSamplerWithoutReplacement
>>>
>>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=1000),
... sampler=SliceSampler(
... slice_len=5, traj_key="episode",strict_length=False,
... ))
...
>>> ep_1 = TensorDict(
... {"obs": torch.arange(100),
... "episode": torch.zeros(100),},
... batch_size=[100]
... )
>>> ep_2 = TensorDict(
... {"obs": torch.arange(4),
... "episode": torch.ones(4),},
... batch_size=[4]
... )
>>> rb.extend(ep_1)
>>> rb.extend(ep_2)
>>>
>>> s = rb.sample(50)
>>> print(s)
TensorDict(
 fields={
 episode: Tensor(shape=torch.Size([46]), device=cpu, dtype=torch.float32, is_shared=False),
 index: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([46, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([46]),
 device=cpu,
 is_shared=False),
 obs: Tensor(shape=torch.Size([46]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([46]),
 device=cpu,
 is_shared=False)
>>> t = split_trajectories(s, done_key="truncated")
>>> print(t["obs"])
tensor([[73, 74, 75, 76, 77],
 [ 0, 1, 2, 3, 0],
 [ 0, 1, 2, 3, 0],
 [41, 42, 43, 44, 45],
 [ 0, 1, 2, 3, 0],
 [67, 68, 69, 70, 71],
 [27, 28, 29, 30, 31],
 [80, 81, 82, 83, 84],
 [17, 18, 19, 20, 21],
 [ 0, 1, 2, 3, 0]])
>>> print(t["episode"])
tensor([[0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 0.],
 [1., 1., 1., 1., 0.],
 [0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 0.]])
```
- using a [`SliceSamplerWithoutReplacement`](torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement)

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
... {"obs": torch.arange(4),
... "episode": torch.ones(4),},
... batch_size=[4]
... )
>>> rb.extend(ep_1)
>>> rb.extend(ep_2)
>>>
>>> s = rb.sample(50)
>>> t = split_trajectories(s, trajectory_key="episode")
>>> print(t["obs"])
tensor([[75, 76, 77, 78, 79],
 [ 0, 1, 2, 3, 0]])
>>> print(t["episode"])
tensor([[0., 0., 0., 0., 0.],
 [1., 1., 1., 1., 0.]])
```

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
>>> from torchrl.data.replay_buffers.samplers import SliceSampler
>>> torch.manual_seed(0)
>>> rb = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(1_000_000),
... sampler=SliceSampler(cache_values=True, num_slices=10),
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
>>> print("sample:", sample)
>>> print("episodes", sample.get("episode").unique())
episodes tensor([1, 2, 3, 4], dtype=torch.int32)
```

`SliceSampler` is default-compatible with
most of TorchRL's datasets:

Examples

```
>>> import torch
>>>
>>> from torchrl.data.datasets import RobosetExperienceReplay
>>> from torchrl.data import SliceSampler
>>>
>>> torch.manual_seed(0)
>>> num_slices = 10
>>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
>>> data = RobosetExperienceReplay(dataid, batch_size=320, sampler=SliceSampler(num_slices=num_slices))
>>> for batch in data:
... batch = batch.reshape(num_slices, -1)
... break
>>> print("check that each batch only has one episode:", batch["episode"].unique(dim=1))
check that each batch only has one episode: tensor([[19],
 [14],
 [ 8],
 [10],
 [13],
 [ 4],
 [ 2],
 [ 3],
 [22],
 [ 8]])
```

See also

Trajectory boundaries are recovered at sampling time with
[`find_start_stop_traj()`](torchrl.data.find_start_stop_traj.html#torchrl.data.find_start_stop_traj), which documents how
trajectory ids, end flags, the write cursor and the storage capacity
interact. See also [the trajectory-boundary documentation](../data_layout.html#ref-traj-boundaries) for the conventions collectors, storages and
samplers follow.