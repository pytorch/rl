# PermuteTransform

*class*torchrl.envs.transforms.PermuteTransform(*dims*, *in_keys=None*, *out_keys=None*, *in_keys_inv=None*, *out_keys_inv=None*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#PermuteTransform)

Permutation transform.

Permutes input tensors along the desired dimensions. The permutations
must be provided along the feature dimension (not batch dimension).

Parameters:

- **dims** (*list**of**int*) - the permuted order of the dimensions. Must be a reordering
of the dims `[-(len(dims)), ..., -1]`.
- **in_keys** (*list**of**NestedKeys*) - input entries (read).
- **out_keys** (*list**of**NestedKeys*) - input entries (write). Defaults to `in_keys` if
not provided.
- **in_keys_inv** (*list**of**NestedKeys*) - input entries (read) during `inv` calls.
- **out_keys_inv** (*list**of**NestedKeys*) - input entries (write) during `inv` calls. Defaults to `in_keys_in` if
not provided.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> base_env = GymEnv("ALE/Pong-v5")
>>> base_env.rollout(2)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
>>> env = TransformedEnv(base_env, PermuteTransform((-1, -3, -2), in_keys=["pixels"]))
>>> env.rollout(2) # channels are at the end
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
```

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#PermuteTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#PermuteTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform