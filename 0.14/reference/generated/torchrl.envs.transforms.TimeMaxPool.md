# TimeMaxPool

*class*torchrl.envs.transforms.TimeMaxPool(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *T: int = 1*, *reset_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#TimeMaxPool)

Take the maximum value in each position over the last T observations.

This transform take the maximum value in each position for all in_keys tensors over the last T time steps.

Parameters:

- **in_keys** (*sequence**of**NestedKey**,**optional*) - input keys on which the max pool will be applied. Defaults to "observation" if left empty.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - output keys where the output will be written. Defaults to in_keys if left empty.
- **T** (*int**,**optional*) - Number of time steps over which to apply max pooling.
- **reset_key** (*NestedKey**,**optional*) - the reset key to be used as partial
reset indicator. Must be unique. If not provided, defaults to the
only reset key of the parent environment (if it has only one)
and raises an exception otherwise.

Examples

```
>>> from torchrl.envs import GymEnv
>>> base_env = GymEnv("Pendulum-v1")
>>> env = TransformedEnv(base_env, TimeMaxPool(in_keys=["observation"], T=10))
>>> torch.manual_seed(0)
>>> env.set_seed(0)
>>> rollout = env.rollout(10)
>>> print(rollout["observation"]) # values should be increasing up until the 10th step
tensor([[ 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0216, 0.0000],
 [ 0.0000, 0.1149, 0.0000],
 [ 0.0000, 0.1990, 0.0000],
 [ 0.0000, 0.2749, 0.0000],
 [ 0.0000, 0.3281, 0.0000],
 [-0.9290, 0.3702, -0.8978]])
```

Note

`TimeMaxPool` currently only supports `done` signal at the root.
Nested `done`, such as those found in MARL settings, are currently not supported.
If this feature is needed, please raise an issue on TorchRL repo.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#TimeMaxPool.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#TimeMaxPool.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform