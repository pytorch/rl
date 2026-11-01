# TensorDictPrimer

*class*torchrl.envs.transforms.TensorDictPrimer(*primers: dict | [Composite](torchrl.data.Composite.html#torchrl.data.Composite) | None = None*, *random: bool | None = None*, *default_value: float | Callable | dict[NestedKey, float] | dict[NestedKey, Callable] | None = None*, *reset_key: NestedKey | None = None*, *expand_specs: bool | None = None*, *single_default_value: bool = False*, *call_before_env_reset: bool = False*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer)

A primer for TensorDict initialization at reset time.

This transform will populate the tensordict at reset with values drawn from
the relative tensorspecs provided at initialization.
If the transform is used out of the env context (e.g. as an nn.Module or
appended to a replay buffer), a call to forward will also populate the
tensordict with the desired features.

Parameters:

- **primers** (*dict**or*[*Composite*](torchrl.data.Composite.html#torchrl.data.Composite)*,**optional*) - a dictionary containing
key-spec pairs which will be used to populate the input tensordict.
[`Composite`](torchrl.data.Composite.html#torchrl.data.Composite) instances are supported too.
- **random** (*bool**,**optional*) - if `True`, the values will be drawn randomly from
the TensorSpec domain (or a unit Gaussian if unbounded). Otherwise a fixed value will be assumed.
Defaults to False.
- **default_value** (`float`, Callable, Dict[NestedKey, float], Dict[NestedKey, Callable], optional) -

If non-random
filling is chosen, default_value will be used to populate the tensors.

- If default_value is a float or any other scala, all elements of the tensors will be set to that value.
- If it is a callable and single_default_value=False (default), this callable is expected to return a tensor
fitting the specs (ie, `default_value()` will be called independently for each leaf spec).
- If it is a callable and `single_default_value=True`, then the callable will be called just once and it is expected
that the structure of its returned TensorDict instance or equivalent will match the provided specs.
The `default_value` must accept an optional reset keyword argument indicating which envs are to be reset.
The returned TensorDict must have as many elements as the number of envs to reset.

See also

`DataLoadingPrimer`
- Finally, if default_value is a dictionary of tensors or a dictionary of callables with keys matching
those of the specs, these will be used to generate the corresponding tensors. Defaults to 0.0.
- **reset_key** (*NestedKey**,**optional*) - the reset key to be used as partial
reset indicator. Must be unique. If not provided, defaults to the
only reset key of the parent environment (if it has only one)
and raises an exception otherwise.
- **single_default_value** (*bool**,**optional*) - if `True` and default_value is a callable, it will be expected that
`default_value` returns a single tensordict matching the specs. If False, default_value() will be
called independently for each leaf. Defaults to `False`.
- **call_before_env_reset** (*bool**,**optional*) - if `True`, the tensordict is populated before env.reset is called.
Defaults to `False`.
- ****kwargs** - each keyword argument corresponds to a key in the tensordict.
The corresponding value has to be a TensorSpec instance indicating
what the value must be.

When used in a TransformedEnv, the spec shapes must match the environment's shape if
the parent environment is batch-locked (env.batch_locked=True). If the spec shapes and
parent shapes do not match, the spec shapes are modified in-place to match the leading
dimensions of the parent's batch size. This adjustment is made for cases where the parent
batch size dimension is not known during instantiation.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.envs import SerialEnv
>>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
>>> env = TransformedEnv(base_env)
>>> # the env is batch-locked, so the leading dims of the spec must match those of the env
>>> env.append_transform(TensorDictPrimer(mykey=Unbounded([2, 3])))
>>> td = env.reset()
>>> print(td)
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 mykey: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
>>> # the entry is populated with 0s
>>> print(td.get("mykey"))
tensor([[0., 0., 0.],
 [0., 0., 0.]])
```

When calling `env.step()`, the current value of the key will be carried
in the `"next"` tensordict __unless it already exists__.

Examples

```
>>> td = env.rand_step(td)
>>> print(td.get(("next", "mykey")))
tensor([[0., 0., 0.],
 [0., 0., 0.]])
>>> # with another value for "mykey", the previous value is not carried on
>>> td = env.reset()
>>> td = td.set(("next", "mykey"), torch.ones(2, 3))
>>> td = env.rand_step(td)
>>> print(td.get(("next", "mykey")))
tensor([[1., 1., 1.],
 [1., 1., 1.]])
```

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.envs import SerialEnv, TransformedEnv
>>> from torchrl.modules.utils import get_primers_from_module
>>> from torchrl.modules import GRUModule
>>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
>>> env = TransformedEnv(base_env)
>>> model = GRUModule(input_size=2, hidden_size=2, in_key="observation", out_key="action")
>>> primers = get_primers_from_module(model)
>>> print(primers) # Primers shape is independent of the env batch size
TensorDictPrimer(primers=Composite(
 recurrent_state: UnboundedContinuous(
 shape=torch.Size([1, 2]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 device=None,
 shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)
>>> env.append_transform(primers)
>>> print(env.reset()) # The primers are automatically expanded to match the env batch size
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 recurrent_state: Tensor(shape=torch.Size([2, 1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
```

Note

Some TorchRL modules rely on specific keys being present in the environment TensorDicts,
like `LSTM` or `GRU`.
To facilitate this process, the method `get_primers_from_module()`
automatically checks for required primer transforms in a module and its submodules and
generates them.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.forward)

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

to(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.to)

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

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TensorDictPrimer.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform