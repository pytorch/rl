# Resize

*class*torchrl.envs.transforms.Resize(*w: int*, *h: int | None = None*, *interpolation: str = 'bilinear'*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#Resize)

Resizes a pixel observation.

Parameters:

- **w** (*int*) - resulting width.
- **h** (*int**,**optional*) - resulting height. If not provided, the value of w
is taken.
- **interpolation** (*str*) - interpolation method

Examples

```
>>> from torchrl.envs import GymEnv
>>> t = Resize(64, 84)
>>> base_env = GymEnv("HalfCheetah-v4", from_pixels=True)
>>> env = TransformedEnv(base_env, Compose(ToTensorImage(), t))
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#Resize.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform