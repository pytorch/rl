# ObservationNorm

*class*torchrl.envs.transforms.ObservationNorm(*loc: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *scale: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, *standard_normal: bool = False*, *eps: float | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#ObservationNorm)

Observation affine transformation layer.

Normalizes an observation according to

\[obs = obs * scale + loc\]

Parameters:

- **loc** (*number**or**tensor*) - location of the affine transform
- **scale** (*number**or**tensor*) - scale of the affine transform
- **in_keys** (*sequence**of**NestedKey**,**optional*) - entries to be normalized. Defaults to ["observation", "pixels"].
All entries will be normalized with the same values: if a different behavior is desired
(e.g. a different normalization for pixels and states) different `ObservationNorm`
objects should be used.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - output entries. Defaults to the value of in_keys.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - ObservationNorm also supports inverse transforms. This will
only occur if a list of keys is provided to `in_keys_inv`. If none is provided,
only the forward transform will be called.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - output entries for the inverse transform.
Defaults to the value of in_keys_inv.
- **standard_normal** (*bool**,**optional*) -

if `True`, the transform will be

\[obs = (obs-loc)/scale\]

as it is done for standardization. Default is False.
- **eps** (`float`, optional) - epsilon increment for the scale in the `standard_normal` case.
Defaults to `1e-6` if not recoverable directly from the scale dtype.

Examples

```
>>> torch.set_default_tensor_type(torch.DoubleTensor)
>>> r = torch.randn(100, 3)*torch.randn(3) + torch.randn(3)
>>> td = TensorDict({'obs': r}, [100])
>>> transform = ObservationNorm(
... loc = td.get('obs').mean(0),
... scale = td.get('obs').std(0),
... in_keys=["obs"],
... standard_normal=True)
>>> _ = transform(td)
>>> print(torch.isclose(td.get('obs').mean(0),
... torch.zeros(3)).all())
tensor(True)
>>> print(torch.isclose(td.get('next_obs').std(0),
... torch.ones(3)).all())
tensor(True)
```

The normalization stats can be automatically computed:
.. rubric:: Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> torch.manual_seed(0)
>>> env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(env, ObservationNorm(in_keys=["observation"]))
>>> env.set_seed(0)
>>> env.transform.init_stats(100)
>>> print(env.transform.loc, env.transform.scale)
tensor([-1.3752e+01, -6.5087e-03, 2.9294e-03], dtype=torch.float32) tensor([14.9636, 2.5608, 0.6408], dtype=torch.float32)
```

init_stats(*num_iter: int*, *reduce_dim: int | tuple[int] = 0*, *cat_dim: int | None = None*, *key: NestedKey | None = None*, *keep_dims: tuple[int] | None = None*) → None[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#ObservationNorm.init_stats)

Initializes the loc and scale stats of the parent environment.

Normalization constant should ideally make the observation statistics approach
those of a standard Gaussian distribution. This method computes a location
and scale tensor that will empirically compute the mean and standard
deviation of a Gaussian distribution fitted on data generated randomly with
the parent environment for a given number of steps.

Parameters:

- **num_iter** (*int*) - number of random iterations to run in the environment.
- **reduce_dim** (*int**or**tuple**of**int**,**optional*) - dimension to compute the mean and std over.
Defaults to 0.
- **cat_dim** (*int**,**optional*) - dimension along which the batches collected will be concatenated.
It must be part equal to reduce_dim (if integer) or part of the reduce_dim tuple.
Defaults to the same value as reduce_dim.
- **key** (*NestedKey**,**optional*) - if provided, the summary statistics will be
retrieved from that key in the resulting tensordicts.
Otherwise, the first key in `ObservationNorm.in_keys` will be used.
- **keep_dims** (*tuple**of**int**,**optional*) - the dimensions to keep in the loc and scale.
For instance, one may want the location and scale to have shape [C, 1, 1]
when normalizing a 3D tensor over the last two dimensions, but not the
third. Defaults to None.

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#ObservationNorm.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#ObservationNorm.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_normalization.html#ObservationNorm.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform