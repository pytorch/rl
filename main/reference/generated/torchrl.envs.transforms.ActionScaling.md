# ActionScaling

*class*torchrl.envs.transforms.ActionScaling(*in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, ***, *loc: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float | None = None*, *scale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | float | None = None*, *standard_normal: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling)

Affine-scale a continuous action using the bounds of the action spec.

Given a bounded action spec with bounds `[low, high]`, this transform exposes
a normalized action space to the policy and rescales actions back to the
original env range before they are passed to the environment.

The `loc` and `scale` are derived from the spec:

\[loc = \frac{high + low}{2}, \quad scale = \frac{high - low}{2}.\]

When `standard_normal=True` (default) the normalized action space is
`[-1, 1]` and the inverse mapping (policy action -> env action) is

\[a_{env} = a_{norm} \cdot scale + loc.\]

The forward mapping (env action -> normalized action, used by replay buffer
transforms) is the inverse:

\[a_{norm} = (a_{env} - loc) / scale.\]

When `standard_normal=False` the normalized space is `[0, 1]` and the
mapping is rescaled accordingly so that `0` maps to `low` and `1` to
`high`.

Parameters:

- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys read during the
`inv` direction (policy -> env). Defaults to `["action"]`. A
single key per `ActionScaling` instance is supported; compose
several instances to scale several actions. Pass an empty list for
a forward-only transform (normalize raw dataset actions on the
replay-buffer sample path while leaving `extend` and the env-side
action interface untouched); this requires explicit `loc` and
`scale`.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys written during the
`inv` direction. Defaults to `in_keys_inv`.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys read during the forward
direction (env action -> normalized action, used by replay buffers
and inside [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) chains). Defaults to
`in_keys_inv`, or `["action"]` when `in_keys_inv=[]`
(forward-only mode).
- **out_keys** (*sequence**of**NestedKey**,**optional*) - keys written during the
forward direction. Defaults to `in_keys`.

Keyword Arguments:

- **loc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - explicit location of the affine
transform. If both `loc` and `scale` are provided the values are
used as-is and no derivation from the spec is performed (useful when
no parent environment is available, e.g. inside a replay buffer).
Defaults to `None`.
- **scale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - explicit scale of the affine
transform. Must be provided together with `loc`.
Defaults to `None`.
- **standard_normal** (*bool**,**optional*) - if `True` (default), the normalized
action space is `[-1, 1]`. If `False`, the normalized action
space is `[0, 1]`.

Raises:

**RuntimeError** - if `loc` and `scale` are derived from the spec (no
 explicit values passed) and the action spec is unbounded or
 partially unbounded (any bound is non-finite). With explicit
 `loc`/`scale`, a bounded spec is mapped through the affine
 transform and an unbounded (or partially unbounded) spec is
 advertised as `Unbounded` instead of raising.

With explicit `loc` and `scale` the transform is fully spec-independent
- the standard workflow when training on dataset action statistics, e.g.
for VLA policies. Use `from_stats()` (`mean`/`std` or
`low`/`high`) or `from_metadata()` to build such an instance from
dataset statistics. Attached to an environment, it denormalizes the
policy's actions on the inverse path: a bounded action spec is mapped
through the affine transform (and an unbounded action spec stays
unbounded), so the advertised normalized space reflects the actual
statistics rather than being assumed `[-1, 1]`. Appended to a replay
buffer, it normalizes actions on the `sample` path; beware that
`ReplayBuffer.extend` applies the *inverse* transform, so when raw
(env-scale) data is written through `extend`, use a forward-only
instance (`in_keys_inv=[]`) to leave the stored data untouched - the
default bidirectional keys suit the env side and pre-populated dataset
storages.

Examples

```
>>> import torch
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.envs.transforms import ActionScaling, TransformedEnv
>>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
>>> base_env = ContinuousActionVecMockEnv(
... action_spec=Bounded(low=-2.0, high=4.0, shape=(7,))
... )
>>> env = TransformedEnv(base_env, ActionScaling())
>>> env.action_spec.space.low
tensor([-1., -1., -1., -1., -1., -1., -1.])
>>> env.action_spec.space.high
tensor([1., 1., 1., 1., 1., 1., 1.])
>>> # dataset-statistics-driven normalization (no env required): the
>>> # forward pass maps raw actions to the normalized space
>>> from tensordict import TensorDict
>>> t = ActionScaling.from_stats(
... mean=torch.tensor([1.0, 2.0]), std=torch.tensor([2.0, 4.0])
... )
>>> td = TensorDict({"action": torch.tensor([[3.0, 6.0]])}, batch_size=[1])
>>> t(td)["action"]
tensor([[1., 1.]])
>>> # on a replay buffer, a forward-only instance (in_keys_inv=[])
>>> # normalizes on sample and leaves data written through extend
>>> # untouched (extend applies the inverse pass)
>>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
>>> t = ActionScaling.from_stats(
... mean=torch.tensor([1.0, 2.0]),
... std=torch.tensor([2.0, 4.0]),
... in_keys_inv=[],
... )
>>> rb = TensorDictReplayBuffer(
... storage=LazyTensorStorage(10), transform=t, batch_size=2
... )
>>> raw = TensorDict(
... {"action": torch.tensor([[3.0, 6.0]]).expand(10, 2)}, batch_size=[10]
... )
>>> indices = rb.extend(raw) # stored as-is
>>> rb.sample()["action"] # normalized with the dataset statistics
tensor([[1., 1.],
 [1., 1.]])
>>> # the same affine map is exposed on raw tensors for execution-time
>>> # use, e.g. mapping a policy's normalized prediction to the robot
>>> t.denormalize(torch.tensor([[1.0, 1.0]]))
tensor([[3., 6.]])
```

denormalize(*action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.denormalize)

Map a normalized action back to the env scale (the inverse map).

*classmethod*from_metadata(*metadata: [RobotDatasetMetadata](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata)*, ***kwargs*) → ActionScaling[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.from_metadata)

Build from the action statistics of a [`RobotDatasetMetadata`](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata).

Uses `action_mean`/`action_std` when available, falling back to
`action_low`/`action_high`. The action key defaults to the
metadata's `action_key`.

*classmethod*from_stats(***, *mean: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *std: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *eps: float = 1e-06*, ***kwargs*) → ActionScaling[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.from_stats)

Build an `ActionScaling` from dataset action statistics.

Provide exactly one complete pair: `mean` and `std` (zero-mean,
unit-std normalized space) or `low` and `high` (maps the range to
`[-1, 1]`).

Keyword Arguments:

- **mean** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension action mean.
- **std** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension action std.
- **low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension action minimum.
- **high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - per-dimension action maximum.
- **eps** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - floor applied to the scale to avoid division
by zero on constant action dimensions. Defaults to `1e-6`.
- ****kwargs** - forwarded to the constructor (e.g. `in_keys_inv`,
`standard_normal`).

normalize(*action: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.normalize)

Map an env-scale action to the normalized space (the forward map).

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform