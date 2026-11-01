# ModuleTransform

*class*torchrl.envs.transforms.ModuleTransform(**args*, *use_ray_service=False*, *service_backend=None*, *service_backend_options=None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform)

A transform that wraps a module.

Keyword Arguments:

- **module** (*TensorDictModuleBase*) - The module to wrap. Exclusive with module_factory. At least one of module or module_factory must be provided.
- **module_factory** (*Callable**[**[**]**,**TensorDictModuleBase**]*) - The factory to create the module. Exclusive with module. At least one of module or module_factory must be provided.
- **no_grad** (*bool**,**optional*) - Whether to use gradient computation. Default is False.
- **inverse** (*bool**,**optional*) - Whether to use the inverse of the module. Default is False.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device used to place the wrapped module at
construction time. Until v0.17 this value is also the explicit
input/output tensordict placement policy: incoming tensordicts are
moved to this device for the call (TensorDict `.to()` copy-back).
A later `.to(device)` updates this policy to the same destination;
a dtype-only `.to()` does not. `ModuleTransform.device`
returns this value and will be removed in v0.17. Defaults to None
(the module is left where it is and incoming tensordicts are not
moved).
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

Warning

`ModuleTransform.device` is deprecated and will be removed in v0.17.
It is an explicit input/output tensordict placement policy, not the
location of the wrapped module. Setting it moves incoming tensordicts to
that device for the call (TensorDict `.to()` copy-back). The
constructor `device=` argument places the module at initialization and,
until v0.17, also sets this I/O policy. A later `.to(device)` updates
the policy to that destination.

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

to(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/module.html#ModuleTransform.to)

Its signature is similar to [`torch.Tensor.to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to), but only accepts
floating point or complex `dtype`s. In addition, this method will
only cast the floating point or complex parameters and buffers to `dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

Note

This method modifies the module in-place.

Parameters:

- **device** ([`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the desired device of the parameters
and buffers in this module
- **dtype** ([`torch.dtype`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the desired floating point or complex dtype of
the parameters and buffers in this module
- **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Tensor whose dtype and device are the desired
dtype and device for all parameters and buffers in this module
- **memory_format** ([`torch.memory_format`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)) - the desired memory
format for 4D parameters and buffers in this module (keyword
only argument)

Returns:

self

Return type:

Module

Examples:

```
>>> # xdoctest: +IGNORE_WANT("non-deterministic")
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]], dtype=torch.float64)
>>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j, 0.2382+0.j],
 [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
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