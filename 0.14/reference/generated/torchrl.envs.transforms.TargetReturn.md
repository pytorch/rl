# TargetReturn

*class*torchrl.envs.transforms.TargetReturn(*target_return: float*, *mode: str = 'reduce'*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *reset_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#TargetReturn)

Sets a target return for the agent to achieve in the environment.

In goal-conditioned RL, the `TargetReturn` is defined as the
expected cumulative reward obtained from the current state to the goal state
or the end of the episode. It is used as input for the policy to guide its behavior.
For a trained policy typically the maximum return in the environment is
chosen as the target return.
However, as it is used as input to the policy module, it should be scaled
accordingly.
With the `TargetReturn` transform, the tensordict can be updated
to include the user-specified target return.
The `mode` parameter can be used to specify
whether the target return gets updated at every step by subtracting the
reward achieved at each step or remains constant.

Parameters:

- **target_return** (`float`) - target return to be achieved by the agent.
- **mode** (*str*) - mode to be used to update the target return. Can be either "reduce" or "constant". Default: "reduce".
- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys pointing to the reward
entries. Defaults to the reward keys of the parent env.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - keys pointing to the
target keys. Defaults to a copy of in_keys where the last element
has been substituted by `"target_return"`, and raises an exception
if these keys aren't unique.
- **reset_key** (*NestedKey**,**optional*) - the reset key to be used as partial
reset indicator. Must be unique. If not provided, defaults to the
only reset key of the parent environment (if it has only one)
and raises an exception otherwise.

Examples

```
>>> from torchrl.envs import GymEnv
>>> env = TransformedEnv(
... GymEnv("CartPole-v1"),
... TargetReturn(10.0, mode="reduce"))
>>> env.set_seed(0)
>>> torch.manual_seed(0)
>>> env.rollout(20)['target_return'].squeeze()
tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0., -1., -2., -3.])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#TargetReturn.forward)

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

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#TargetReturn.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_reward.html#TargetReturn.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform