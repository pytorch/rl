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
several instances to scale several actions.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys written during the
`inv` direction. Defaults to `in_keys_inv`.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys read during the forward
direction (env action -> normalized action, used by replay buffers
and inside [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) chains). Defaults to
`in_keys_inv`.
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

**RuntimeError** - if the action spec is unbounded or partially unbounded
 (any bound is non-finite).

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
```

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionScaling.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform