# Timer

*class*torchrl.envs.transforms.Timer(*out_keys: Sequence[NestedKey] = None*, *time_key: str = 'time'*)[[source]](../../_modules/torchrl/envs/transforms/_timer.html#Timer)

A transform that measures the time intervals between inv and call operations in an environment.

The Timer transform is used to track the time elapsed between the inv call and the call,
and between the call and the inv call. This is useful for performance monitoring and debugging
within an environment. The time is measured in seconds and stored as a tensor with the default
dtype from PyTorch. If the tensordict has a batch size (e.g., in batched environments), the time will be expended
to the size of the input tensordict.

Variables:

- **out_keys** - The keys of the output tensordict for the inverse transform. Defaults to
out_keys = [f"{time_key}_step", f"{time_key}_policy", f"{time_key}_reset"], where the first key represents
the time it takes to make a step in the environment, and the second key represents the
time it takes to execute the policy, the third the time for the call to reset.
- **time_key** - A prefix for the keys where the time intervals will be stored in the tensordict.
Defaults to "time".

Note

During a succession of rollouts, the time marks of the reset are written at the root (the "time_reset"
entry or equivalent key is always 0 in the "next" tensordict). At the root, the "time_policy" and "time_step"
entries will be 0 when there is a reset. they will never be 0 in the "next".

Examples

```
>>> from torchrl.envs import Timer, GymEnv
>>>
>>> env = GymEnv("Pendulum-v1").append_transform(Timer())
>>> r = env.rollout(10)
>>> print("time for policy", r["time_policy"])
time for policy tensor([0.0000, 0.0882, 0.0004, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
 0.0002])
>>> print("time for step", r["time_step"])
time for step tensor([9.5797e-04, 1.6289e-03, 9.7990e-05, 8.0824e-05, 9.0837e-05, 7.6056e-05,
 8.2016e-05, 7.6056e-05, 8.1062e-05, 7.7009e-05])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_timer.html#Timer.forward)

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

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_timer.html#Timer.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform