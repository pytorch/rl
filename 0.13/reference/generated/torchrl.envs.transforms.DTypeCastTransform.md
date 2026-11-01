# DTypeCastTransform

*class*torchrl.envs.transforms.DTypeCastTransform(*dtype_in: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *dtype_out: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_device.html#DTypeCastTransform)

Casts one dtype to another for selected keys.

Depending on whether the `in_keys` or `in_keys_inv` are provided
during construction, the class behavior will change:

> - If the keys are provided, those entries and those entries only will be
> transformed from `dtype_in` to `dtype_out` entries;
> - If the keys are not provided and the object is within an environment
> register of transforms, the input and output specs that have a dtype
> set to `dtype_in` will be used as in_keys_inv / in_keys respectively.
> - If the keys are not provided and the object is used without an
> environment, the `forward` / `inverse` pass will scan through the
> input tensordict for all `dtype_in` values and map them to a `dtype_out`
> tensor. For large data structures, this can impact performance as this
> scanning doesn't come for free. The keys to be
> transformed will not be cached.
> Note that, in this case, the out_keys (resp.
> out_keys_inv) cannot be passed as the order on which the keys are processed
> cannot be anticipated precisely.

Parameters:

- **dtype_in** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the input dtype (from the env).
- **dtype_out** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the output dtype (for model training).
- **in_keys** (*sequence**of**NestedKey**,**optional*) - list of `dtype_in` keys to be converted to
`dtype_out` before being exposed to external objects and functions.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - list of destination keys.
Defaults to `in_keys` if not provided.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - list of `dtype_out` keys to be converted to
`dtype_in` before being passed to the contained base_env or storage.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - list of destination keys for inverse
transform.
Defaults to `in_keys_inv` if not provided.

Examples

```
>>> td = TensorDict(
... {'obs': torch.ones(1, dtype=torch.double),
... 'not_transformed': torch.ones(1, dtype=torch.double),
... }, [])
>>> transform = DTypeCastTransform(torch.double, torch.float, in_keys=["obs"])
>>> _ = transform(td)
>>> print(td.get("obs").dtype)
torch.float32
>>> print(td.get("not_transformed").dtype)
torch.float64
```

In "automatic" mode, all float64 entries are transformed:

Examples

```
>>> td = TensorDict(
... {'obs': torch.ones(1, dtype=torch.double),
... 'not_transformed': torch.ones(1, dtype=torch.double),
... }, [])
>>> transform = DTypeCastTransform(torch.double, torch.float)
>>> _ = transform(td)
>>> print(td.get("obs").dtype)
torch.float32
>>> print(td.get("not_transformed").dtype)
torch.float32
```

The same behavior is the rule when environments are constructed without
specifying the transform keys:

Examples

```
>>> class MyEnv(EnvBase):
... def __init__(self):
... super().__init__()
... self.observation_spec = Composite(obs=Unbounded((), dtype=torch.float64))
... self.action_spec = Unbounded((), dtype=torch.float64)
... self.reward_spec = Unbounded((1,), dtype=torch.float64)
... self.done_spec = Unbounded((1,), dtype=torch.bool)
... def _reset(self, data=None):
... return TensorDict({"done": torch.zeros((1,), dtype=torch.bool), **self.observation_spec.rand()}, [])
... def _step(self, data):
... assert data["action"].dtype == torch.float64
... reward = self.reward_spec.rand()
... done = torch.zeros((1,), dtype=torch.bool)
... obs = self.observation_spec.rand()
... assert reward.dtype == torch.float64
... assert obs["obs"].dtype == torch.float64
... return obs.empty().set("next", obs.update({"reward": reward, "done": done}))
... def _set_seed(self, seed) -> None:
... pass
>>> env = TransformedEnv(MyEnv(), DTypeCastTransform(torch.double, torch.float))
>>> assert env.action_spec.dtype == torch.float32
>>> assert env.observation_spec["obs"].dtype == torch.float32
>>> assert env.reward_spec.dtype == torch.float32, env.reward_spec.dtype
>>> print(env.rollout(2))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 obs: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
>>> assert env.transform.in_keys == ["obs", "reward"]
>>> assert env.transform.in_keys_inv == ["action"]
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_device.html#DTypeCastTransform.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_device.html#DTypeCastTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec*)[[source]](../../_modules/torchrl/envs/transforms/_device.html#DTypeCastTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_device.html#DTypeCastTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform