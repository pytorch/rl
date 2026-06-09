# ClipTransform

*class*torchrl.envs.transforms.ClipTransform(*in_keys=None*, *out_keys=None*, *in_keys_inv=None*, *out_keys_inv=None*, ***, *low=None*, *high=None*)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ClipTransform)

A transform to clip input (state, action) or output (observation, reward) values.

This transform can take multiple input or output keys but only one value per
transform. If multiple clipping values are needed, several transforms should
be appended one after the other.

Parameters:

- **in_keys** (*list**of**NestedKeys*) - input entries (read)
- **out_keys** (*list**of**NestedKeys*) - input entries (write)
- **in_keys_inv** (*list**of**NestedKeys*) - input entries (read) during `inv` calls.
- **out_keys_inv** (*list**of**NestedKeys*) - input entries (write) during `inv` calls.

Keyword Arguments:

- **low** (*scalar**,**optional*) - the lower bound of the clipped space.
- **high** (*scalar**,**optional*) - the higher bound of the clipped space.

Note

Providing just one of the arguments `low` or `high` is permitted,
but at least one must be provided.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> base_env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(base_env, ClipTransform(in_keys=['observation'], low=-1, high=0.1))
>>> r = env.rollout(100)
>>> assert (r["observation"] <= 0.1).all()
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ClipTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_clip.html#ClipTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform