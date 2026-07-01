# BatchSizeTransform

*class*torchrl.envs.transforms.BatchSizeTransform(***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *reshape_fn: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *reset_func: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase), [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *env_kwarg: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BatchSizeTransform)

A transform to modify the batch-size of an environment.

This transform has two distinct usages: it can be used to set the
batch-size for non-batch-locked (e.g. stateless) environments to
enable data collection using data collectors. It can also be used
to modify the batch-size of an environment (e.g. squeeze, unsqueeze or
reshape).

This transform modifies the environment batch-size to match the one provided.
It expects the parent environment batch-size to be expandable to the
provided one.

Keyword Arguments:

- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**equivalent**,**optional*) - the new batch-size of the environment.
Exclusive with `reshape_fn`.
- **reshape_fn** (*callable**,**optional*) -

a callable to modify the environment batch-size.
Exclusive with `batch_size`.

Note

Currently, transformations involving
`reshape`, `flatten`, `unflatten`, `squeeze` and `unsqueeze`
are supported. If another reshape operation is required, please submit
a feature request on TorchRL github.
- **reset_func** (*callable**,**optional*) - a function that produces a reset tensordict.
The signature must match `Callable[[TensorDictBase, TensorDictBase], TensorDictBase]`
where the first input argument is the optional tensordict passed to the
environment during the call to `reset()` and the second
is the output of `TransformedEnv.base_env.reset`. It can also support an
optional `env` keyword argument if `env_kwarg=True`.
- **env_kwarg** (*bool**,**optional*) - if `True`, `reset_func` must support a
`env` keyword argument. Defaults to `False`. The env passed will
be the env accompanied by its transform.

Example

```
>>> # Changing the batch-size with a function
>>> from torchrl.envs import GymEnv
>>> base_env = GymEnv("CartPole-v1")
>>> env = TransformedEnv(base_env, BatchSizeTransform(reshape_fn=lambda data: data.reshape(1, 1)))
>>> env.rollout(4)
>>> # Setting the shape of a stateless environment
>>> class MyEnv(EnvBase):
... batch_locked = False
... def __init__(self):
... super().__init__()
... self.observation_spec = Composite(observation=Unbounded(3))
... self.reward_spec = Unbounded(1)
... self.action_spec = Unbounded(1)
...
... def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
... tensordict_batch_size = tensordict.batch_size if tensordict is not None else torch.Size([])
... result = self.observation_spec.rand(tensordict_batch_size)
... result.update(self.full_done_spec.zero(tensordict_batch_size))
... return result
...
... def _step(
... self,
... tensordict: TensorDictBase,
... ) -> TensorDictBase:
... result = self.observation_spec.rand(tensordict.batch_size)
... result.update(self.full_done_spec.zero(tensordict.batch_size))
... result.update(self.full_reward_spec.zero(tensordict.batch_size))
... return result
...
... def _set_seed(self, seed: Optional[int]) -> None:
... pass
...
>>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5]))
>>> assert env.batch_size == torch.Size([5])
>>> assert env.rollout(10).shape == torch.Size([5, 10])
```

The `reset_func` can create a tensordict with the desired batch-size, allowing for
a fine-grained reset call:

```
>>> def reset_func(tensordict, tensordict_reset, env):
... result = env.observation_spec.rand()
... result.update(env.full_done_spec.zero())
... assert result.batch_size != torch.Size([])
... return result
>>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5], reset_func=reset_func, env_kwarg=True))
>>> print(env.rollout(2))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 2]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 2]),
 device=None,
 is_shared=False)
```

This transform can be used to deploy non-batch-locked environments within data
collectors:

```
>>> from torchrl.collectors import Collector
>>> collector = Collector(env, lambda td: env.rand_action(td), frames_per_batch=10, total_frames=-1)
>>> for data in collector:
... print(data)
... break
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([5, 2]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([5, 2]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 2]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 2]),
 device=None,
 is_shared=False)
>>> collector.shutdown()
```

forward(*next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

transform_env_batch_size(*batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BatchSizeTransform.transform_env_batch_size)

Transforms the batch-size of the parent env.

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BatchSizeTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BatchSizeTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform