# MacroPrimitiveTransform

*class*torchrl.envs.transforms.MacroPrimitiveTransform(**args: Any*, *execute: bool = False*, *multi_action_dim: int = 1*, *stack_rewards: bool = True*, *stack_observations: bool = False*, ***kwargs: Any*)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitiveTransform)

Expand a high-level macro action into a low-level action sequence.

The base transform is deliberately agnostic to robots, grippers and MuJoCo
models. Its inverse-action path reads one macro action from `action_key`,
resolves a `(start, target)` pair of low-level actions, linearly
interpolates between them over `macro_steps` (plus `settle_steps` held
repeats), and writes the resulting `(..., T, action_dim)` sequence back
under `action_key`. When `execute=True` the constructor returns
`Compose(MultiAction(...), self)` so the sequence is executed by the parent
environment in a single high-level step.

The policy-facing action accepted under `action_key` may be:

- a [`MacroAction`](torchrl.envs.transforms.MacroAction.html#torchrl.envs.transforms.MacroAction) / [`TargetMacroAction`](torchrl.envs.transforms.TargetMacroAction.html#torchrl.envs.transforms.TargetMacroAction) (or a plain
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with the same `mode` / `target` /
`steps` / `settle_steps` schema); or
- a raw tensor, treated as a direct low-level action target (`MOVE`).

Domain specializations override three hooks rather than configuring adapter,
solver and library objects:

- `_resolve()` - map a macro action to `(start, target, steps,
settle_steps)` low-level tensors;
- `current_action()` - read the low-level action used as the
interpolation start (defaults to zeros or a tensor already at
`action_key`);
- `transform_input_spec()` - advertise the policy-facing action spec.

Parameters:

- **action_key** - low-level action key consumed by the inner environment and
also the key carrying the macro action on the way in.
- **macro_steps** - number of interpolated low-level actions per primitive.
- **settle_steps** - number of repeated final actions appended after each
primitive.
- **action_dim** - low-level action dimension. Required when it cannot be
inferred from specs or from the macro action target.
- **execute** - if `True`, return `Compose(MultiAction(...), transform)` so
emitted action sequences are executed by the parent environment.
- **multi_action_dim** - stack dimension consumed by `MultiAction` when
`execute=True`.
- **stack_rewards** - whether `MultiAction` returns each low-level reward.
- **stack_observations** - whether `MultiAction` returns each low-level
observation.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs.transforms import MacroPrimitiveTransform
>>> td = TensorDict({"action": torch.ones(1, 3)}, batch_size=[1])
>>> transform = MacroPrimitiveTransform(macro_steps=2, action_dim=3)
>>> transform.inv(td)["action"].shape
torch.Size([1, 2, 3])
```

action_sequence(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *mode: int | IntEnum | None = None*, ***, *target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *target_qpos: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *steps: int | None = None*, *settle_steps: int | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitiveTransform.action_sequence)

Expand a macro action into its low-level sequence without executing.

When `mode`/`target` are given, a primitive is built first;
otherwise `tensordict` is expected to already carry a macro action
under `action_key`.

current_action(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *batch_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *action_dim: int*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitiveTransform.current_action)

Return the low-level action used as the interpolation start.

The base implementation starts every macro from the zero action: in the
inverse path `action_key` carries the incoming macro action (the
*target*), so it must not be read back here as the start. Subclasses that
can read the controlled state from observations (e.g. joint positions)
override this hook.

make_primitive(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *mode: int | IntEnum = MacroPrimitive.MOVE*, ***, *target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *target_qpos: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *steps: int | None = None*, *settle_steps: int | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitiveTransform.make_primitive)

Return a copy of `tensordict` carrying one macro action.

This is a small scripting helper: it builds a
[`TargetMacroAction`](torchrl.envs.transforms.TargetMacroAction.html#torchrl.envs.transforms.TargetMacroAction) and stores it under `action_key` so the
result can be passed to `action_sequence()` or executed.

primitive_enum

alias of [`MacroPrimitive`](torchrl.envs.transforms.MacroPrimitive.html#torchrl.envs.transforms.MacroPrimitive)

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitiveTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform