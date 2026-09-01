# Transform

*class*torchrl.envs.transforms.Transform(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform)

Base class for environment transforms, which modify or create new data in a tensordict.

Transforms are used to manipulate the input and output data of an environment. They can be used to preprocess
observations, modify rewards, or transform actions. Transforms can be composed together to create more complex
transformations.

A transform receives a tensordict as input and returns (the same or another) tensordict as output, where a series
of values have been modified or created with a new key.

Variables:

- **parent** - The parent environment of the transform.
- **container** - The container that holds the transform.
- **in_keys** - The keys of the input tensordict that the transform will read from.
- **out_keys** - The keys of the output tensordict that the transform will write to.

See also

[TorchRL transforms](../envs_transforms.html#transforms).

Subclassing Transform:

> There are various ways of subclassing a transform. The things to take into considerations are:
> 
> 
> 
> 
> - Is the transform identical for each tensor / item being transformed? Use
> `_apply_transform()` and `_inv_apply_transform()`.
> - The transform needs access to the input data to env.step as well as output? Rewrite
> `_step()`.
> Otherwise, rewrite `_call()` (or `_inv_call()`).
> - Is the transform to be used within a replay buffer? Overwrite `forward()`,
> `inv()`, `_apply_transform()` or
> `_inv_apply_transform()`.
> - Within a transform, you can access (and make calls to) the parent environment using
> `parent` (the base env + all transforms till this one) or
> `container()` (The object that encapsulates the transform).
> - Don't forget to edits the specs if needed: top level: `transform_output_spec()`,
> `transform_input_spec()`.
> Leaf level: `transform_observation_spec()`,
> `transform_action_spec()`, `transform_state_spec()`,
> `transform_reward_spec()` and
> `transform_reward_spec()`.
> 
> 
> 
> 
> For practical examples, see the methods listed above.

clone()[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.clone)

creates a copy of the tensordict, without parent (a transform object can only have one parent).

set_container()[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.set_container)

Sets the container for the transform, and in turn the parent if the container is or has one
an environment within.

reset_parent()[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.reset_parent)

resets the parent and container caches.

close()[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.close)

Close the transform.

*property*collector*: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) | None*

Returns the collector associated with the container, if it exists.

This can be used whenever the transform needs to be made aware of the collector or the policy associated with it.

Make sure to call this property only on transforms that are not nested in sub-processes.
The collector reference will not be passed to the workers of a [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) or
similar batched environments.

*property*container*: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | None*

Returns the env containing the transform.

Examples

```
>>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
>>> env.transform[0].container is env
True
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.forward)

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

init(*tensordict*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.init)

Runs init steps for the transform.

inv(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.inv)

Reads the input tensordict, and for the selected keys, applies the inverse transform.

By default, this method:

- calls directly `_inv_apply_transform()`.
- does not call `_inv_call()`.

Note

`inv` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Note

`inv` is called by [`extend()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend).

*property*parent*: [TransformedEnv](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv) | None*

Returns the parent env of the transform.

The parent env is the env that contains all the transforms up until the current one.

Examples

```
>>> from torchrl.envs import TransformedEnv, Compose, RewardSum, StepCounter
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(RewardSum(), StepCounter()))
>>> env.transform[1].parent
TransformedEnv(
 env=GymEnv(env=Pendulum-v1, batch_size=torch.Size([]), device=cpu),
 transform=Compose(
 RewardSum(keys=['reward'])))
```

to(**args*, ***kwargs*) → Transform[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.to)

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

transform_action_spec(*action_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_action_spec)

Transforms the action spec such that the resulting spec matches transform mapping.

Parameters:

**action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_done_spec(*done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_done_spec)

Transforms the done spec such that the resulting spec matches transform mapping.

Parameters:

**done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_env_batch_size(*batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*) → [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_env_batch_size)

Transforms the batch-size of the parent env.

transform_env_device(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*) → [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_env_device)

Transforms the device of the parent env.

transform_fake_tensordict(*fake_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_fake_tensordict)

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

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_state_spec(*state_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_base.html#Transform.transform_state_spec)

Transforms the state spec such that the resulting spec matches transform mapping.

Parameters:

**state_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform