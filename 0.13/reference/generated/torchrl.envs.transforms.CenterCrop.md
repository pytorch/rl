# CenterCrop

*class*torchrl.envs.transforms.CenterCrop(*w: int*, *h: int | None = None*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#CenterCrop)

Crops the center of an image.

Parameters:

- **w** (*int*) - resulting width
- **h** (*int**,**optional*) - resulting height. If None, then w is used (square crop).
- **in_keys** (*sequence**of**NestedKey**,**optional*) - the entries to crop. If none is provided,
`["pixels"]` is assumed.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - the cropped images keys. If none is
provided, `in_keys` is assumed.

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#CenterCrop.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform