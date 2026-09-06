# FlattenAction

*class*torchrl.envs.transforms.FlattenAction(*first_dim: int = -2*, *last_dim: int = -1*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *allow_positive_dim: bool = False*, ***, *action_shape: Sequence[int] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#FlattenAction)

Flatten adjacent dimensions of an action.

Mirrors [`FlattenObservation`](torchrl.envs.transforms.FlattenObservation.html#torchrl.envs.transforms.FlattenObservation), but applies
to actions: the policy sees a flattened action space and the original
multi-dimensional shape is restored on the inv direction before the action
is passed to the base environment.

On the inv direction (policy -> env), a 1-D `flattened` action is
unflattened to the original `(dim_first, ..., dim_last)` span of the env
action. On the forward direction (env action -> flattened, used inside
replay buffers and [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) chains), the adjacent dims
`[first_dim, last_dim]` are flattened.

Parameters:

- **first_dim** (*int*) - first dimension to flatten. Must be negative unless
`allow_positive_dim` is `True`.
- **last_dim** (*int*) - last dimension to flatten (inclusive). Must be negative
unless `allow_positive_dim` is `True`.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys read during the
`inv` direction (policy -> env). Defaults to `["action"]`.
Multiple keys are supported - the same flatten span is applied to
each one, which is useful for dict-structured action spaces.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys written during the
`inv` direction. Defaults to `in_keys_inv`.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys read during the forward
direction (env action -> flattened). Defaults to `in_keys_inv`.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - keys written during the
forward direction. Defaults to `in_keys`.
- **allow_positive_dim** (*bool**,**optional*) - if `True`, positive dimensions
are accepted. Defaults to `False` so that the same transform
works regardless of the parent environment's batch size.

Keyword Arguments:

**action_shape** (*sequence**of**int**,**optional*) - explicit pre-flatten shape
of the dimensions `[first_dim, last_dim]`. Useful when the
transform is used outside a [`TransformedEnv`](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv) (e.g. inside
a replay buffer) and the original action shape cannot be derived
from a parent env. The same span is applied to every entry of
`in_keys_inv`. Defaults to `None`, in which case the shape is
derived lazily from the parent env's action spec.

Examples

```
>>> import torch
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.envs.transforms import FlattenAction, TransformedEnv
>>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
>>> base_env = ContinuousActionVecMockEnv(
... action_spec=Bounded(low=-1.0, high=1.0, shape=(3, 5))
... )
>>> env = TransformedEnv(base_env, FlattenAction(first_dim=-2, last_dim=-1))
>>> env.action_spec.shape
torch.Size([15])
```

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_action.html#FlattenAction.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform