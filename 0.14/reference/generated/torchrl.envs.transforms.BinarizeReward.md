# BinarizeReward

*class*torchrl.envs.transforms.BinarizeReward(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#BinarizeReward)

Maps the reward to a binary value (0 or 1) if the reward is null or non-null, respectively.

Parameters:

- **in_keys** (*List**[**NestedKey**]*) - input keys
- **out_keys** (*List**[**NestedKey**]**,**optional*) - output keys. Defaults to value
of `in_keys`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the binerized reward.
Defaults to `torch.int8`.

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#BinarizeReward.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform