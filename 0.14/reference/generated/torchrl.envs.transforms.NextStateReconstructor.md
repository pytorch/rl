# NextStateReconstructor

*class*torchrl.envs.transforms.NextStateReconstructor(*keys: Sequence[NestedKey] = ('observation',)*, ***, *traj_key: NestedKey | None = ('collector', 'traj_ids')*, *done_key: NestedKey | None = ('next', 'done')*, *step_count_key: NestedKey | None = None*, *fill_value: float = nan*, *strict: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/rb_transforms.html#NextStateReconstructor)

Re-hydrate `("next", obs)` keys at sampling time by shifting along the batch.

Pairs with [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) configured with
`compact_obs=True` (and the analogous flag on the multi-process collectors):
the collector drops the observation and state keys from the
`("next", ...)` sub-tensordict before stacking because those values are
bit-for-bit identical to the root keys at `t + 1` within the same
trajectory; this transform rebuilds them on the consumer side.

**Core rule.** For each registered root key `k` and each position `i`
of the flat sampled batch:

- if position `i + 1` is in the batch *and* belongs to the same
trajectory as position `i`, write
`data[("next", k)][i] = data[k][i + 1]`;
- otherwise write `data[("next", k)][i] = fill_value` (`NaN` by
default).

"Same trajectory" is decided from a trajectory id key in the sample,
by default `("collector", "traj_ids")` -- the key that
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) populates when
`track_traj_ids=True` (the default). The semantics fall out cleanly for
every common sampler:

- [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) with
`traj_key`: positions inside a slice mirror to the next position;
slice boundaries differ in trajectory id and become `NaN`.
- A full rollout sampled as one contiguous batch: every transition inside
a trajectory is reconstructed; trajectory ends become `NaN`.
- [`RandomSampler`](torchrl.data.replay_buffers.RandomSampler.html#torchrl.data.replay_buffers.RandomSampler) and similar:
adjacent batch positions almost never share a trajectory id, so the
result is mostly `NaN`. This is correct -- the next observation is
genuinely not available in the sampled batch -- and it makes the
mis-use loud rather than silent.

The trajectory-id check alone is *not* enough: a sampler is allowed to
place two slices of the *same* trajectory back-to-back in one batch
(e.g. [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) sampling
with replacement when there are fewer trajectories than slices). In that
case the two positions across the splice would share a trajectory id
without being consecutive in time. The transform therefore also consults
`("next", "done")` (if present): when `done[i]` is `True` the
trajectory ended at step `i`, so position `i + 1` is never the next
step of trajectory `traj_id[i]` no matter what.

An additional, stricter `step_count_key` cross-check is available for
setups where neither `traj_id` nor `done` are bulletproof -- see below.

Parameters:

**keys** (*sequence**of**NestedKey**,**optional*) - the root keys whose
`("next", k)` counterparts should be reconstructed. Defaults to
`("observation",)`. For environments with nested observation
specs, pass the full leaf list, e.g.
`[("agents", "pos"), ("agents", "vel")]`.

Keyword Arguments:

- **traj_key** (*NestedKey**,**optional*) - key carrying the trajectory id used
to detect boundaries. Defaults to `("collector", "traj_ids")`.
Set to `None` to skip the trajectory check and treat the entire
sampled batch as one trajectory (only the very last position is
then filled with `fill_value`).
- **done_key** (*NestedKey**,**optional*) - key whose `True` entries indicate
that the trajectory terminated at position `i`, so position
`i + 1` is not the next step. Defaults to `("next", "done")`.
Set to `None` to disable the check.
- **step_count_key** (*NestedKey**,**optional*) - if not `None`, also require
`data[step_count_key][i + 1] == data[step_count_key][i] + 1` to
consider position `i + 1` as the canonical next step. The
collector populates `("collector", "step_count")` only when a
[`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) is in the env
transform chain. Defaults to `None`.
- **fill_value** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - value written wherever the next
observation is not available. Defaults to `float("nan")`. For
integer-typed observation keys, NaN cannot be represented; pass
an explicit integer (e.g. `0`).
- **strict** (*bool**,**optional*) - if `True` (default) and any configured
marker key (`traj_key`, `done_key`, `step_count_key`) is
missing from the sampled batch, raise. If `False`, silently
drop that check.

Example

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage
>>> from torchrl.data.replay_buffers.samplers import SliceSampler
>>> from torchrl.envs.transforms.rb_transforms import (
... NextStateReconstructor,
... )
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(100),
... sampler=SliceSampler(
... slice_len=4, traj_key=("collector", "traj_ids"),
... ),
... transform=NextStateReconstructor(),
... batch_size=8,
... )
>>> # populate `rb` with a collector configured with `compact_obs=True`
>>> # so that ``("next", "observation")`` is absent from storage:
>>> data = TensorDict({
... "observation": torch.arange(8, dtype=torch.float32).view(8, 1),
... ("next", "reward"): torch.zeros(8, 1),
... ("next", "done"): torch.tensor([[False]] * 7 + [[True]]),
... ("collector", "traj_ids"): torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
... }, batch_size=[8])
>>> rb.extend(data)
>>> sample = rb.sample() # ('next', 'observation') is reconstructed
```

See also

[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector)'s `compact_obs` flag
is the producer side of this transform -- it drops the duplicated
`("next", obs)` before stacking. Trajectory ends carry `NaN` after
rehydration; the value-estimator pipeline keeps GAE / TD targets
numerically defined via
`_sanitize_next_obs_nan()`.
`MultiStepTransform` is **not**
compatible with the compact path: it needs the canonical `("next", obs)`
to read the n-step neighbour (and to keep working at the last
`n - 1` frames of every trajectory, where the n-step lookup falls
back to the in-trajectory neighbours). For a lossy alternative that
reconstructs the *real* boundary transition (smaller memory saving,
no `NaN`), see
[`NextObservationDelta`](torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta). See the
*Memory-efficient RL training* tutorial for an end-to-end pipeline.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/rb_transforms.html#NextStateReconstructor.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```