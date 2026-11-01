# ModuleTransform

*class*torchrl.envs.transforms.ModuleTransform(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform)

A transform that wraps a module.

Keyword Arguments:

- **module** (*TensorDictModuleBase*) - The module to wrap. Exclusive with module_factory. At least one of module or module_factory must be provided.
- **module_factory** (*Callable**[**[**]**,**TensorDictModuleBase**]*) - The factory to create the module. Exclusive with module. At least one of module or module_factory must be provided.
- **no_grad** (*bool**,**optional*) - Whether to use gradient computation. Default is False.
- **inverse** (*bool**,**optional*) - Whether to use the inverse of the module. Default is False.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to use. Default is None.
- **use_ray_service** (*bool**,**optional*) - Whether to use Ray service. Default is False.
- **num_gpus** (*int**,**optional*) - The number of GPUs to use if using Ray. Default is None.
- **num_cpus** (*int**,**optional*) - The number of CPUs to use if using Ray. Default is None.
- **actor_name** (*str**,**optional*) - The name of the actor to use. Default is None. If an actor name is provided and
an actor with this name already exists, the existing actor will be used.
- **observation_spec_transform** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*or**Callable**[**[*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]**,*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]*) - either a new spec for the observation
after it has been transformed by the module, or a function that modifies the existing spec.
Defaults to None (observation specs remain unchanged).
- **done_spec_transform** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*or**Callable**[**[*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]**,*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]*) - either a new spec for the done
after it has been transformed by the module, or a function that modifies the existing spec.
Defaults to None (done specs remain unchanged).
- **reward_spec_transform** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*or**Callable**[**[*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]**,*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]*) - either a new spec for the reward
after it has been transformed by the module, or a function that modifies the existing spec.
Defaults to None (reward specs remain unchanged).
- **state_spec_transform** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*or**Callable**[**[*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]**,*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]*) - either a new spec for the state
after it has been transformed by the module, or a function that modifies the existing spec.
Defaults to None (state specs remain unchanged).
- **action_spec_transform** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*or**Callable**[**[*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]**,*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*]*) - either a new spec for the action
after it has been transformed by the module, or a function that modifies the existing spec.
Defaults to None (action specs remain unchanged).

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.forward)

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

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_done_spec(*done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.transform_done_spec)

Transforms the done spec such that the resulting spec matches transform mapping.

Parameters:

**done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform