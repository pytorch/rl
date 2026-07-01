# Compose

*class*torchrl.envs.transforms.Compose(*transforms: list[[Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)]*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose)

Composes a chain of transforms.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or ``callable``s are accepted.

The class can be instantiated in several ways:

Parameters:

- ***transforms** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)) - Variable number of transforms to compose.
- **transforms** (*list**[*[*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*]**,**optional*) - A list of transforms to compose.
This can be passed as a keyword argument.

Examples

```
>>> env = GymEnv("Pendulum-v0")
>>>
>>> # Method 1: Using positional arguments
>>> transforms = Compose(RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0))
>>> transformed_env = TransformedEnv(env, transforms)
>>>
>>> # Method 2: Using a list with positional argument
>>> transform_list = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
>>> transforms = Compose(transform_list)
>>> transformed_env = TransformedEnv(env, transforms)
>>>
>>> # Method 3: Using keyword argument
>>> transforms = Compose(transforms=[RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)])
>>> transformed_env = TransformedEnv(env, transforms)
```

append(*transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.append)

Appends a transform in the chain.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or callable are accepted.

close()[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.close)

Close the transform.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.forward)

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

init(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.init)

Runs init steps for the transform.

insert(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.insert)

Inserts a transform in the chain at the desired index.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or callable are accepted.

pop(*index: int | None = None*) → [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.pop)

Pop a transform from the chain.

Parameters:

**index** (*int**,**optional*) - The index of the transform to pop. If None, the last transform is popped.

Returns:

The popped transform.

to(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.to)

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

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_env_batch_size(*batch_size: torch.batch_size*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_env_batch_size)

Transforms the batch-size of the parent env.

transform_env_device(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_env_device)

Transforms the device of the parent env.

transform_fake_tensordict(*fake_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_fake_tensordict)

Adjust the env's `fake_tensordict` after it is built from specs.

[`fake_tensordict()`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.fake_tensordict) constructs a zero-filled
tensordict from the env's specs, which is used by data collectors to
pre-allocate the rollout storage. The TorchRL spec system shares the
observation spec between the root and `("next", ...)` leaves, so
transforms that want the runtime `("next", k)` dtype to differ from
the root `k` dtype need a way to fix up the fake tensordict here.

The default is a no-op. Override only when the runtime tensordict your
transform produces does not match what the spec-derived fake
tensordict would imply.

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Compose.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform