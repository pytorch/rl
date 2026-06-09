# VC1Transform

*class*torchrl.envs.transforms.VC1Transform(*in_keys*, *out_keys*, *model_name*, *del_keys: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform)

VC1 Transform class.

VC1 provides pre-trained ResNet weights aimed at facilitating visual
embedding for robotic tasks. The models are trained using Ego4d.

See the paper:
VC1: A Universal Visual Representation for Robot Manipulation (Suraj Nair,

Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta)
[https://arxiv.org/abs/2203.12601](https://arxiv.org/abs/2203.12601)

The VC1Transform is created in a lazy manner: the object will be initialized
only when an attribute (a spec or the forward method) will be queried.
The reason for this is that the `_init()` method requires some attributes of
the parent environment (if any) to be accessed: by making the class lazy we
can ensure that the following code snippet works as expected:

Examples

```
>>> transform = VC1Transform("default", in_keys=["pixels"])
>>> env.append_transform(transform)
>>> # the forward method will first call _init which will look at env.observation_spec
>>> env.reset()
```

Parameters:

- **in_keys** (*list**of**NestedKeys*) - list of input keys. If left empty, the
"pixels" key is assumed.
- **out_keys** (*list**of**NestedKeys**,**optional*) - list of output keys. If left empty,
"VC1_vec" is assumed.
- **model_name** (*str*) - One of `"large"`, `"base"` or any other compatible
model name (see the [github repo](https://github.com/facebookresearch/eai-vc) for more info). Defaults to `"default"`
which provides a small, untrained model for testing.
- **del_keys** (*bool**,**optional*) - If `True` (default), the input key will be
discarded from the returned tensordict.

forward(*next_tensordict*)

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

*classmethod*make_noload_model()[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.make_noload_model)

Creates an naive model at a custom destination.

to(*dest: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.to)

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

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/vc1.html#VC1Transform.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform