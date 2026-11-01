# EndOfLifeTransform

*class*torchrl.envs.transforms.EndOfLifeTransform(*eol_key: NestedKey = 'end-of-life'*, *lives_key: NestedKey = 'lives'*, *done_key: NestedKey = 'done'*, *eol_attribute='unwrapped.ale.lives'*)[[source]](../../_modules/torchrl/envs/transforms/gym_transforms.html#EndOfLifeTransform)

Registers the end-of-life signal from a Gym env with a lives method.

Proposed by DeepMind for the DQN and co. It helps value estimation.

Parameters:

- **eol_key** (*NestedKey**,**optional*) - the key where the end-of-life signal should
be written. Defaults to `"end-of-life"`.
- **done_key** (*NestedKey**,**optional*) - a "done" key in the parent env done_spec,
where the done value can be retrieved. This key must be unique and its
shape must match the shape of the end-of-life entry. Defaults to `"done"`.
- **eol_attribute** (*str**,**optional*) - the location of the "lives" in the gym env.
Defaults to `"unwrapped.ale.lives"`. Supported attribute types are
integer/array-like objects or callables that return these values.

Note

This transform should be used with gym envs that have a `env.unwrapped.ale.lives`.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.envs.transforms.transforms import TransformedEnv
>>> env = GymEnv("ALE/Breakout-v5")
>>> env.rollout(100)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([100, 4]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([100]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([100]),
 device=cpu,
 is_shared=False)
>>> eol_transform = EndOfLifeTransform()
>>> env = TransformedEnv(env, eol_transform)
>>> env.rollout(100)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([100, 4]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 eol: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 lives: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.int64, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 end-of-life: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 lives: Tensor(shape=torch.Size([100]), device=cpu, dtype=torch.int64, is_shared=False),
 pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 reward: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([100]),
 device=cpu,
 is_shared=False),
 pixels: Tensor(shape=torch.Size([100, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 terminated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([100]),
 device=cpu,
 is_shared=False)
```

The typical usage of this transform is to replace the "done" state by "end-of-life"
within the loss module. The end-of-life signal isn't registered within the `done_spec`
because it should not instruct the env to reset.

Examples

```
>>> from torchrl.objectives import DQNLoss
>>> module = torch.nn.Identity() # used as a placeholder
>>> loss = DQNLoss(module, action_space="categorical")
>>> loss.set_keys(done="end-of-life", terminated="end-of-life")
>>> # equivalently
>>> eol_transform.register_keys(loss)
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/gym_transforms.html#EndOfLifeTransform.forward)

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

register_keys(*loss_or_advantage: [torchrl.objectives.common.LossModule](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*)[[source]](../../_modules/torchrl/envs/transforms/gym_transforms.html#EndOfLifeTransform.register_keys)

Registers the end-of-life key at appropriate places within the loss.

Parameters:

**loss_or_advantage** ([*torchrl.objectives.LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)*or*[*torchrl.objectives.value.ValueEstimatorBase*](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase)) - a module to instruct what the end-of-life key is.

transform_observation_spec(*observation_spec*)[[source]](../../_modules/torchrl/envs/transforms/gym_transforms.html#EndOfLifeTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform