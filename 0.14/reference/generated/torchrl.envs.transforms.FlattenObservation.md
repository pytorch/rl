# FlattenObservation

*class*torchrl.envs.transforms.FlattenObservation(*first_dim: int*, *last_dim: int*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *allow_positive_dim: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#FlattenObservation)

Flatten adjacent dimensions of a tensor.

Parameters:

- **first_dim** (*int*) - first dimension of the dimensions to flatten.
- **last_dim** (*int*) - last dimension of the dimensions to flatten.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - the entries to flatten. If none is provided,
`["pixels"]` is assumed.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - the flatten observation keys. If none is
provided, `in_keys` is assumed.
- **allow_positive_dim** (*bool**,**optional*) - if `True`, positive dimensions are accepted.
`FlattenObservation` will map these to the n^th feature dimension
(ie n^th dimension after batch size of parent env) of the input tensor.
Defaults to False, ie. non-negative dimensions are not permitted.

forward(*next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Reads the input tensordict, and for the selected keys, applies the transform.

`_call` can be re-written whenever a modification of the output of env.step needs to be modified independently
of the data collected in the previous step (including actions and states).

For any operation that relates exclusively to the parent env (e.g. `FrameSkip`),
modify the `_step()` method instead.
`_call()` should only be overwritten if a modification of the input tensordict is needed.

`_call()` will be called by `step()` and
`reset()` but not during `forward()`.

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#FlattenObservation.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform