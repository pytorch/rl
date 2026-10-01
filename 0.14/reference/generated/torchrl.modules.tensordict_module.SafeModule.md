# SafeModule

*class*torchrl.modules.tensordict_module.SafeModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule)

[`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) subclass that accepts a [`TensorSpec`](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec) as argument to control the output domain.

Parameters:

- **module** (*nn.Module*) - a nn.Module used to map the input to the output
parameter space. Can be a functional
module (FunctionalModule or FunctionalModuleWithBuffers), in which
case the `forward` method will expect
the params (and possibly) buffers keyword arguments.
- **in_keys** (*iterable**of**str*) - keys to be read from input tensordict and
passed to the module. If it
contains more than one element, the values will be passed in the
order given by the in_keys iterable.
- **out_keys** (*iterable**of**str*) - keys to be written to the input tensordict.
The length of out_keys must match the
number of tensors returned by the embedded module. Using "_" as a
key avoid writing tensor to output.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - specs of the output tensor. If the module
outputs multiple output tensors,
spec characterize the space of the first output tensor.
- **safe** (*bool*) - if `True`, the value of the output is checked against the
input spec. Out-of-domain sampling can
occur because of exploration policies or numerical under/overflow issues.
If this value is out of bounds, it is projected back onto the
desired space using the `TensorSpec.project`
method. Default is `False`.
- **inplace** (*bool**or**str**,**optional*) - if True, the input tensordict is modified in-place. If False, a new empty
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instance is created. If "empty", input.empty() is used instead (ie, the
output preserves type, device and batch-size). Defaults to True.

Embedding a neural network in a TensorDictModule only requires to specify the input and output keys. The domain spec can
be passed along if needed. TensorDictModule support functional and regular `nn.Module` objects. In the functional
case, the 'params' (and 'buffers') keyword argument must be specified:

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import Unbounded
>>> from torchrl.modules import TensorDictModule
>>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
>>> spec = Unbounded(8)
>>> module = torch.nn.GRUCell(4, 8)
>>> td_fmodule = TensorDictModule(
... module=module,
... spec=spec,
... in_keys=["input", "hidden"],
... out_keys=["output"],
... )
>>> params = TensorDict.from_module(td_fmodule)
>>> with params.to_module(td_module):
... td_functional = td_fmodule(td.clone())
>>> print(td_functional)
TensorDict(
 fields={
 hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
 input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
 output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

In the stateful case:

```
>>> td_module = TensorDictModule(
... module=torch.nn.GRUCell(4, 8),
... spec=spec,
... in_keys=["input", "hidden"],
... out_keys=["output"],
... )
>>> td_stateful = td_module(td.clone())
>>> print(td_stateful)
TensorDict(
 fields={
 hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
 input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
 output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

One can use a vmap operator to call the functional module. In this case the tensordict is expanded to match the
batch size (i.e. the tensordict isn't modified in-place anymore):

```
>>> # Model ensemble using vmap
>>> from torch import vmap
>>> params_repeat = params.expand(4, *params.shape)
>>> td_vmap = vmap(td_fmodule, (None, 0))(td.clone(), params_repeat)
>>> print(td_vmap)
TensorDict(
 fields={
 hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
 input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
 output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
```

random(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.random)

Samples a random element in the target space, irrespective of any input.

If multiple output keys are present, only the first will be written in the input `tensordict`.

Parameters:

**tensordict** (*TensorDictBase*) - tensordict where the output value should be written.

Returns:

the original tensordict with a new/updated value for the output key.

random_sample(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.random_sample)

See `TensorDictModule.random(...)`.

to(*dest: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int*) → [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/modules/tensordict_module/common.html#SafeModule.to)

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