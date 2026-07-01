# RewardClipping

*class*torchrl.envs.transforms.RewardClipping(*clamp_min: float | None = None*, *clamp_max: float | None = None*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#RewardClipping)

Clips the reward between clamp_min and clamp_max.

Parameters:

- **clip_min** (*scalar*) - minimum value of the resulting reward.
- **clip_max** (*scalar*) - maximum value of the resulting reward.

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#RewardClipping.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform