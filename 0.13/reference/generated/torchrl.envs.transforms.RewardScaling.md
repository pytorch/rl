# RewardScaling

*class*torchrl.envs.transforms.RewardScaling(*loc: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *scale: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *standard_normal: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#RewardScaling)

Affine transform of the reward.

> The reward is transformed according to:

\[reward = reward * scale + loc\]

Parameters:

- **loc** (*number**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - location of the affine transform
- **scale** (*number**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - scale of the affine transform
- **standard_normal** (*bool**,**optional*) -

if `True`, the transform will be

\[reward = (reward-loc)/scale\]

as it is done for standardization. Default is False.

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#RewardScaling.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform