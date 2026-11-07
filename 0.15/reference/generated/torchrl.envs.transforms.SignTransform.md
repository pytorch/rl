# SignTransform

*class*torchrl.envs.transforms.SignTransform(*in_keys=None*, *out_keys=None*, *in_keys_inv=None*, *out_keys_inv=None*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#SignTransform)

A transform to compute the signs of TensorDict values.

This transform reads the tensors in `in_keys` and `in_keys_inv`, computes the
signs of their elements and writes the resulting sign tensors to `out_keys` and
`out_keys_inv` respectively.

Parameters:

- **in_keys** (*list**of**NestedKeys*) - input entries (read)
- **out_keys** (*list**of**NestedKeys*) - input entries (write)
- **in_keys_inv** (*list**of**NestedKeys*) - input entries (read) during `inv` calls.
- **out_keys_inv** (*list**of**NestedKeys*) - input entries (write) during `inv` calls.

Examples

```
>>> from torchrl.envs import GymEnv, TransformedEnv, SignTransform
>>> base_env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(base_env, SignTransform(in_keys=['observation']))
>>> r = env.rollout(100)
>>> obs = r["observation"]
>>> assert (torch.logical_or(torch.logical_or(obs == -1, obs == 1), obs == 0.0)).all()
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#SignTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#SignTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform