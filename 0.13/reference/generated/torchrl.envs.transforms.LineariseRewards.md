# LineariseRewards

*class*torchrl.envs.transforms.LineariseRewards(*in_keys: Sequence[NestedKey]*, *out_keys: Sequence[NestedKey] | None = None*, ***, *weights: Sequence[float] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#LineariseRewards)

Transforms a multi-objective reward signal to a single-objective one via a weighted sum.

Parameters:

- **in_keys** (*List**[**NestedKey**]*) - The keys under which the multi-objective rewards are found.
- **out_keys** (*List**[**NestedKey**]**,**optional*) - The keys under which single-objective rewards should be written. Defaults to `in_keys`.
- **weights** (*List**[**float**]**,**Tensor**,**optional*) - Dictates how to weight each reward when summing them. Defaults to [1.0, 1.0, ...].

Warning

If a sequence of in_keys of length strictly greater than one is passed (e.g. one group for each agent in a
multi-agent set-up), the same weights will be applied for each entry. If you need to aggregate rewards
differently for each group, use several `LineariseRewards` in a row.

Example

```
>>> import mo_gymnasium as mo_gym
>>> from torchrl.envs import MOGymWrapper
>>> mo_env = MOGymWrapper(mo_gym.make("deep-sea-treasure-v0"))
>>> mo_env.reward_spec
BoundedContinuous(
 shape=torch.Size([2]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True)),
 ...)
>>> so_env = TransformedEnv(mo_env, LineariseRewards(in_keys=("reward",)))
>>> so_env.reward_spec
BoundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
 ...)
>>> td = so_env.rollout(5)
>>> td["next", "reward"].shape
torch.Size([5, 1])
```

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#LineariseRewards.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform