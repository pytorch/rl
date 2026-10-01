# UnaryTransform

*class*torchrl.envs.transforms.UnaryTransform(*in_keys: Sequence[NestedKey]*, *out_keys: Sequence[NestedKey]*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, ***, *fn: Callable[[Any], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *inv_fn: Callable[[Any], Any] | None = None*, *use_raw_nontensor: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform)

Applies a unary operation on the specified inputs.

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - the keys of inputs to the unary operation.
- **out_keys** (*sequence**of**NestedKey*) - the keys of the outputs of the unary operation.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of inputs to the unary operation during inverse call.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of the outputs of the unary operation during inverse call.

Keyword Arguments:

- **fn** (*Callable**[**[**Any**]**,**Tensor**|**TensorDictBase**]*) - the function to use as the unary operation. If it accepts
a non-tensor input, it must also accept `None`.
- **inv_fn** (*Callable**[**[**Any**]**,**Any**]**,**optional*) - the function to use as the unary operation during inverse calls.
If it accepts a non-tensor input, it must also accept `None`.
Can be omitted, in which case `fn` will be used for inverse maps.
- **use_raw_nontensor** (*bool**,**optional*) - if `False`, data is extracted from
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) inputs before `fn` is called
on them. If `True`, the raw [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)
inputs are given directly to `fn`, which must support those
inputs. Default is `False`.

Example

```
>>> from torchrl.envs import GymEnv, UnaryTransform
>>> env = GymEnv("Pendulum-v1")
>>> env = env.append_transform(
... UnaryTransform(
... in_keys=["observation"],
... out_keys=["observation_trsf"],
... fn=lambda tensor: str(tensor.numpy().tobytes())))
>>> env.observation_spec
Composite(
 observation: BoundedContinuous(
 shape=torch.Size([3]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 observation_trsf: NonTensor(
 shape=torch.Size([]),
 space=None,
 device=cpu,
 dtype=None,
 domain=None),
 device=None,
 shape=torch.Size([]))
>>> env.rollout(3)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_trsf: NonTensorStack(
 ["b'\\xbe\\xbc\\x7f?8\\x859=/\\x81\\xbe;'", "b'\\x...,
 batch_size=torch.Size([3]),
 device=None),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_trsf: NonTensorStack(
 ["b'\\x9a\\xbd\\x7f?\\xb8T8=8.c>'", "b'\\xbe\\xbc\...,
 batch_size=torch.Size([3]),
 device=None),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> env.check_env_specs()
[torchrl][INFO] check_env_specs succeeded!
```

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*, *test_input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_done_spec(*done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*, *test_output_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_done_spec)

Transforms the done spec such that the resulting spec matches transform mapping.

Parameters:

**done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*, *test_output_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*, *test_output_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*, *test_input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#UnaryTransform.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform