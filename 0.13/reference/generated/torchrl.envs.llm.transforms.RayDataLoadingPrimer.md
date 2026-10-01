# RayDataLoadingPrimer

*class*torchrl.envs.llm.transforms.RayDataLoadingPrimer(***, *dataloader=None*, *dataloader_factory=None*, *num_cpus=None*, *num_gpus=0*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *actor_name: str | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/llm/transforms/dataloading.html#RayDataLoadingPrimer)

A [`DataLoadingPrimer`](torchrl.envs.llm.transforms.DataLoadingPrimer.html#torchrl.envs.llm.transforms.DataLoadingPrimer) that creates a single actor that can be shared by multiple environments.

This class creates a Ray remote actor from DataLoadingPrimer that can be shared across multiple workers.
All method calls are delegated to the remote actor, ensuring that multiple environments iterate over
the same shared dataloader.

Keyword Arguments:

- **dataloader** - A dataloader object to be used directly. Ray will handle serialization.
- **dataloader_factory** - A callable that returns a dataloader. This allows for explicit
resource control and avoids serialization issues.
- **num_cpus** (*int**,**optional*) - Number of CPUs to allocate to the Ray actor.
Defaults to the dataloader's num_workers if available, otherwise 1.
- **num_gpus** (*int**,**optional*) - Number of GPUs to allocate to the Ray actor. Defaults to 0.
- **actor_name** (*str**,**optional*) - Name of the Ray actor to use. If provided, the actor will be reused if it already exists.
- ****kwargs** - Additional keyword arguments to pass to DataLoadingPrimer.

Note

Exactly one of dataloader or dataloader_factory must be provided.

Examples

```
>>> # Option 1: Using a dataloader factory for explicit resource control
>>> def create_dataloader():
... return torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
>>> primer1 = RayDataLoadingPrimer(dataloader_factory=create_dataloader, num_cpus=4)
>>> primer2 = RayDataLoadingPrimer(dataloader_factory=create_dataloader, num_cpus=4) # Same shared actor
```

```
>>> # Option 2: Pass dataloader directly (Ray handles serialization)
>>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
>>> primer1 = RayDataLoadingPrimer(dataloader=dataloader) # num_cpus=4 inferred from num_workers
>>> primer2 = RayDataLoadingPrimer(dataloader=dataloader) # Same shared actor
```

add_module(*name: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None*) → None

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Parameters:

- **name** (*str*) - name of the child module. The child module can be
accessed from this module using the given name
- **module** (*Module*) - child module to be added to the module.

apply(*fn: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)], None]*) → Self

Apply `fn` recursively to every submodule (as returned by `.children()`) as well as self.

Typical use includes initializing the parameters of a model
(see also [torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html#nn-init-doc)).

Parameters:

**fn** (`Module` -> None) - function to be applied to each submodule

Returns:

self

Return type:

Module

Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>> print(m)
>>> if type(m) is nn.Linear:
>>> m.weight.fill_(1.0)
>>> print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
 [1., 1.]], requires_grad=True)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
 [1., 1.]], requires_grad=True)
Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
)
```

*property*base_env

Returns the base environment. This traverses the parent chain locally.

bfloat16() → Self

Casts all floating point parameters and buffers to `bfloat16` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

buffers(*recurse: bool = True*) → Iterator[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

Return an iterator over module buffers.

Parameters:

**recurse** (*bool*) - if True, then yields buffers of this module
and all submodules. Otherwise, yields only buffers that
are direct members of this module.

Yields:

*torch.Tensor* - module buffer

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for buf in model.buffers():
>>> print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

children() → Iterator[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]

Return an iterator over immediate children modules.

Yields:

*Module* - a child module

clone()

Clone the transform.

close()

Close the transform.

*property*collector*: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) | None*

Returns the collector associated with the container, if it exists.

This can be used whenever the transform needs to be made aware of the collector or the policy associated with it.

Make sure to call this property only on transforms that are not nested in sub-processes.
The collector reference will not be passed to the workers of a [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) or
similar batched environments.

compile(**args*, ***kwargs*) → None

Compile this Module's forward using [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

This Module's __call__ method is compiled and all arguments are passed as-is
to [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

See [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) for details on the arguments for this function.

*property*container*: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | None*

Returns the env containing the transform. This is handled locally.

cpu() → Self

Move all model parameters and buffers to the CPU.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

cuda(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

*property*data_keys

Get data_keys property.

*property*dataloader

Get dataloader property.

*property*device

Get device property.

double() → Self

Casts all floating point parameters and buffers to `double` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

dump(***kwargs*)

Dump method.

empty_cache()

Empty cache.

*property*endless_dataloader

Get endless_dataloader property.

eval() → Self

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with [`self.train(False)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train).

See [Locally disabling gradient computation](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for a comparison between
.eval() and several similar mechanisms that may be confused with it.

Returns:

self

Return type:

Module

extra_repr() → str

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

float() → Self

Casts all floating point parameters and buffers to `float` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None

Forward pass.

get_buffer(*target: str*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Return the buffer given by `target` if it exists, otherwise throw an error.

See the docstring for `get_submodule` for a more detailed
explanation of this method's functionality as well as how to
correctly specify `target`.

Parameters:

**target** - The fully-qualified string name of the buffer
to look for. (See `get_submodule` for how to specify a
fully-qualified string.)

Returns:

The buffer referenced by `target`

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Raises:

**AttributeError** - If the target string references an invalid
 path or resolves to something that is not a
 buffer

get_extra_state() → Any

Return any extra state to include in the module's state_dict.

Implement this and a corresponding `set_extra_state()` for your module
if you need to store extra state. This function is called when building the
module's state_dict().

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:

Any extra state to store in the module's state_dict

Return type:

object

get_parameter(*target: str*) → [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)

Return the parameter given by `target` if it exists, otherwise throw an error.

See the docstring for `get_submodule` for a more detailed
explanation of this method's functionality as well as how to
correctly specify `target`.

Parameters:

**target** - The fully-qualified string name of the Parameter
to look for. (See `get_submodule` for how to specify a
fully-qualified string.)

Returns:

The Parameter referenced by `target`

Return type:

torch.nn.Parameter

Raises:

**AttributeError** - If the target string references an invalid
 path or resolves to something that is not an
 `nn.Parameter`

get_submodule(*target: str*) → [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Return the submodule given by `target` if it exists, otherwise throw an error.

For example, let's say you have an `nn.Module` `A` that
looks like this:

```
A(
 (net_b): Module(
 (net_c): Module(
 (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
 )
 (linear): Linear(in_features=100, out_features=200, bias=True)
 )
)
```

(The diagram shows an `nn.Module` `A`. `A` which has a nested
submodule `net_b`, which itself has two submodules `net_c`
and `linear`. `net_c` then has a submodule `conv`.)

To check whether or not we have the `linear` submodule, we
would call `get_submodule("net_b.linear")`. To check whether
we have the `conv` submodule, we would call
`get_submodule("net_b.net_c.conv")`.

The runtime of `get_submodule` is bounded by the degree
of module nesting in `target`. A query against
`named_modules` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, `get_submodule` should always be
used.

Parameters:

**target** - The fully-qualified string name of the submodule
to look for. (See above example for how to specify a
fully-qualified string.)

Returns:

The submodule referenced by `target`

Return type:

[torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Raises:

**AttributeError** - If at any point along the path resulting from
 the target string the (sub)path resolves to a non-existent
 attribute name or an object that is not an instance of `nn.Module`.

half() → Self

Casts all floating point parameters and buffers to `half` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

*property*in_keys

Get in_keys property.

*property*in_keys_inv

Get in_keys_inv property.

init(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None*)[[source]](../../_modules/torchrl/envs/llm/transforms/dataloading.html#RayDataLoadingPrimer.init)

Initialize.

inv(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None

Inverse.

ipu(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

load_state_dict(*state_dict: Mapping[str, Any]*, *strict: bool = True*, *assign: bool = False*)

Copy parameters and buffers from `state_dict` into this module and its descendants.

If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module's [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function.

Warning

If `assign` is `True` the optimizer must be created after
the call to `load_state_dict` unless
[`get_swap_module_params_on_conversion()`](https://docs.pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion) is `True`.

Parameters:

- **state_dict** (*dict*) - a dict containing parameters and
persistent buffers.
- **strict** (*bool**,**optional*) - whether to strictly enforce that the keys
in `state_dict` match the keys returned by this module's
[`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function. Default: `True`
- **assign** (*bool**,**optional*) - When set to `False`, the properties of the tensors
in the current module are preserved whereas setting it to `True` preserves
properties of the Tensors in the state dict. The only
exception is the `requires_grad` field of `Parameter`
for which the value from the module is preserved. Default: `False`

Returns:

- `missing_keys` is a list of str containing any keys that are expected

by this module but missing from the provided `state_dict`.
- `unexpected_keys` is a list of str containing the keys that are not

expected by this module but present in the provided `state_dict`.

Return type:

`NamedTuple` with `missing_keys` and `unexpected_keys` fields

Note

If a parameter or buffer is registered as `None` and its corresponding key
exists in `state_dict`, `load_state_dict()` will raise a
`RuntimeError`.

*property*missing_tolerance

Get missing tolerance.

modules(*remove_duplicate: bool = True*) → Iterator[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]

Return an iterator over all modules in the network.

Parameters:

**remove_duplicate** - whether to remove the duplicated module instances in the result
or not.

Yields:

*Module* - a module in the network

Note

Duplicate modules are returned only once by default. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
... print(idx, '->', m)

0 -> Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```

mtia(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

named_buffers(*prefix: str = ''*, *recurse: bool = True*, *remove_duplicate: bool = True*) → Iterator[tuple[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]]

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Parameters:

- **prefix** (*str*) - prefix to prepend to all buffer names.
- **recurse** (*bool**,**optional*) - if True, then yields buffers of this module
and all submodules. Otherwise, yields only buffers that
are direct members of this module. Defaults to True.
- **remove_duplicate** (*bool**,**optional*) - whether to remove the duplicated buffers in the result. Defaults to True.

Yields:

*(str, torch.Tensor)* - Tuple containing the name and buffer

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>> if name in ['running_var']:
>>> print(buf.size())
```

named_children() → Iterator[tuple[str, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]]

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:

*(str, Module)* - Tuple containing a name and child module

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, module in model.named_children():
>>> if name in ['conv4', 'conv5']:
>>> print(module)
```

named_modules(*memo: set[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)] | None = None*, *prefix: str = ''*, *remove_duplicate: bool = True*)

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Parameters:

- **memo** - a memo to store the set of modules already added to the result
- **prefix** - a prefix that will be added to the name of the module
- **remove_duplicate** - whether to remove the duplicated module instances in the result
or not

Yields:

*(str, Module)* - Tuple of name and module

Note

Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
... print(idx, '->', m)

0 -> ('', Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```

named_parameters(*prefix: str = ''*, *recurse: bool = True*, *remove_duplicate: bool = True*) → Iterator[tuple[str, [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]]

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Parameters:

- **prefix** (*str*) - prefix to prepend to all parameter names.
- **recurse** (*bool*) - if True, then yields parameters of this module
and all submodules. Otherwise, yields only parameters that
are direct members of this module.
- **remove_duplicate** (*bool**,**optional*) - whether to remove the duplicated
parameters in the result. Defaults to True.

Yields:

*(str, Parameter)* - Tuple containing the name and parameter

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>> if name in ['bias']:
>>> print(param.size())
```

*property*out_keys

Get out_keys property.

*property*out_keys_inv

Get out_keys_inv property.

parameters(*recurse: bool = True*) → Iterator[[Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]

Return an iterator over module parameters.

This is typically passed to an optimizer.

Parameters:

**recurse** (*bool*) - if True, then yields parameters of this module
and all submodules. Otherwise, yields only parameters that
are direct members of this module.

Yields:

*Parameter* - module parameter

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for param in model.parameters():
>>> print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

*property*parent*: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | None*

Returns the parent env of the transform. This is handled locally.

*property*primers

Get primers property.

register_backward_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*) → RemovableHandle

Register a backward hook on the module.

This function is deprecated in favor of [`register_full_backward_hook()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook) and
the behavior of this function will change in future versions.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_buffer(*name: str*, *tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None*, *persistent: bool = True*) → None

Add a buffer to the module.

This is typically used to register a buffer that should not be
considered a model parameter. For example, BatchNorm's `running_mean`
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
`state_dict`.

Buffers can be accessed as attributes using given names.

Parameters:

- **name** (*str*) - name of the buffer. The buffer can be accessed
from this module using the given name
- **tensor** (*Tensor**or**None*) - buffer to be registered. If `None`, then operations
that run on buffers, such as `cuda`, are ignored. If `None`,
the buffer is **not** included in the module's `state_dict`.
- **persistent** (*bool*) - whether the buffer is part of this module's
`state_dict`.

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```

register_forward_hook(*hook: Callable[[T, tuple[Any, ...], Any], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any], Any], Any | None]*, ***, *prepend: bool = False*, *with_kwargs: bool = False*, *always_call: bool = False*) → RemovableHandle

Register a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.

If `with_kwargs` is `False` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the `forward`. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after `forward()` is called. The hook
should have the following signature:

```
hook(module, args, output) -> None or modified output
```

If `with_kwargs` is `True`, the forward hook will be passed the
`kwargs` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature:

```
hook(module, args, kwargs, output) -> None or modified output
```

Parameters:

- **hook** (*Callable*) - The user defined hook to be registered.
- **prepend** (*bool*) - If `True`, the provided `hook` will be fired
before all existing `forward` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `forward` hooks on
this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`forward` hooks registered with
`register_module_forward_hook()` will fire before all hooks
registered by this method.
Default: `False`
- **with_kwargs** (*bool*) - If `True`, the `hook` will be passed the
kwargs given to the forward function.
Default: `False`
- **always_call** (*bool*) - If `True` the `hook` will be run regardless of
whether an exception is raised while calling the Module.
Default: `False`

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_forward_pre_hook(*hook: Callable[[T, tuple[Any, ...]], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None]*, ***, *prepend: bool = False*, *with_kwargs: bool = False*) → RemovableHandle

Register a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.

If `with_kwargs` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the `forward`. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature:

```
hook(module, args) -> None or modified input
```

If `with_kwargs` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature:

```
hook(module, args, kwargs) -> None or a tuple of modified input and kwargs
```

Parameters:

- **hook** (*Callable*) - The user defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `forward_pre` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `forward_pre` hooks
on this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`forward_pre` hooks registered with
`register_module_forward_pre_hook()` will fire before all
hooks registered by this method.
Default: `False`
- **with_kwargs** (*bool*) - If true, the `hook` will be passed the kwargs
given to the forward function.
Default: `False`

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_full_backward_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*, *prepend: bool = False*) → RemovableHandle

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:

> 1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.
> 2. If none of the module inputs require gradients, the hook will fire when the gradients are computed
> with respect to module outputs.
> 3. If none of the module outputs require gradients, then the hooks will not fire.

The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

Warning

Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.

Parameters:

- **hook** (*Callable*) - The user-defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `backward` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `backward` hooks on
this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`backward` hooks registered with
`register_module_full_backward_hook()` will fire before
all hooks registered by this method.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_full_backward_pre_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*, *prepend: bool = False*) → RemovableHandle

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature:

```
hook(module, grad_output) -> tuple[Tensor, ...], Tensor or None
```

The `grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of `grad_output` in
subsequent computations. Entries in `grad_output` will be `None` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

Warning

Modifying inputs inplace is not allowed when using backward hooks and
will raise an error.

Parameters:

- **hook** (*Callable*) - The user-defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `backward_pre` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `backward_pre` hooks
on this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`backward_pre` hooks registered with
`register_module_full_backward_pre_hook()` will fire before
all hooks registered by this method.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_load_state_dict_post_hook(*hook*)

Register a post-hook to be run after module's `load_state_dict()` is called.

It should have the following signature::

hook(module, incompatible_keys) -> None

The `module` argument is the current module that this hook is registered
on, and the `incompatible_keys` argument is a `NamedTuple` consisting
of attributes `missing_keys` and `unexpected_keys`. `missing_keys`
is a `list` of `str` containing the missing keys and
`unexpected_keys` is a `list` of `str` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling `load_state_dict()` with
`strict=True` are affected by modifications the hook makes to
`missing_keys` or `unexpected_keys`, as expected. Additions to either
set of keys will result in an error being thrown when `strict=True`, and
clearing out both missing and unexpected keys will avoid an error.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_load_state_dict_pre_hook(*hook*)

Register a pre-hook to be run before module's `load_state_dict()` is called.

It should have the following signature::

hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None # noqa: B950

Parameters:

**hook** (*Callable*) - Callable hook that will be invoked before
loading the state dict.

register_module(*name: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None*) → None

Alias for `add_module()`.

register_parameter(*name: str*, *param: [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) | None*) → None

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Parameters:

- **name** (*str*) - name of the parameter. The parameter can be accessed
from this module using the given name
- **param** (*Parameter**or**None*) - parameter to be added to the module. If
`None`, then operations that run on parameters, such as `cuda`,
are ignored. If `None`, the parameter is **not** included in the
module's `state_dict`.

register_state_dict_post_hook(*hook*)

Register a post-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method.

It should have the following signature::

hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the `state_dict` inplace.

register_state_dict_pre_hook(*hook*)

Register a pre-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method.

It should have the following signature::

hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the `state_dict`
call is made.

*property*repeats

Get repeats property.

requires_grad_(*requires_grad: bool = True*) → Self

Change if autograd should record operations on parameters in this module.

This method sets the parameters' `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See [Locally disabling gradient computation](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for a comparison between
.requires_grad_() and several similar mechanisms that may be confused with it.

Parameters:

**requires_grad** (*bool*) - whether autograd should record operations on
parameters in this module. Default: `True`.

Returns:

self

Return type:

Module

reset_dataloader()[[source]](../../_modules/torchrl/envs/llm/transforms/dataloading.html#RayDataLoadingPrimer.reset_dataloader)

Reset the dataloader.

reset_parent() → None

Reset the parent. This is handled locally.

set_container(*container: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*) → None

Set the container for this transform. This is handled locally.

set_extra_state(*state: Any*) → None

Set extra state contained in the loaded state_dict.

This function is called from `load_state_dict()` to handle any extra state
found within the state_dict. Implement this function and a corresponding
`get_extra_state()` for your module if you need to store extra state within its
state_dict.

Parameters:

**state** (*dict*) - Extra state from the state_dict

set_missing_tolerance(*mode=False*)

Set missing tolerance.

set_submodule(*target: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *strict: bool = False*) → None

Set the submodule given by `target` if it exists, otherwise throw an error.

Note

If `strict` is set to `False` (default), the method will replace an existing submodule
or create a new submodule if the parent module exists. If `strict` is set to `True`,
the method will only attempt to replace an existing submodule and throw an error if
the submodule does not exist.

For example, let's say you have an `nn.Module` `A` that
looks like this:

```
A(
 (net_b): Module(
 (net_c): Module(
 (conv): Conv2d(3, 3, 3)
 )
 (linear): Linear(3, 3)
 )
)
```

(The diagram shows an `nn.Module` `A`. `A` has a nested
submodule `net_b`, which itself has two submodules `net_c`
and `linear`. `net_c` then has a submodule `conv`.)

To override the `Conv2d` with a new submodule `Linear`, you
could call `set_submodule("net_b.net_c.conv", nn.Linear(1, 1))`
where `strict` could be `True` or `False`

To add a new submodule `Conv2d` to the existing `net_b` module,
you would call `set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))`.

In the above if you set `strict=True` and call
`set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)`, an AttributeError
will be raised because `net_b` does not have a submodule named `conv`.

Parameters:

- **target** - The fully-qualified string name of the submodule
to look for. (See above example for how to specify a
fully-qualified string.)
- **module** - The module to set the submodule to.
- **strict** - If `False`, the method will replace an existing submodule
or create a new submodule if the parent module exists. If `True`,
the method will only attempt to replace an existing submodule and throw an error
if the submodule doesn't already exist.

Raises:

- **ValueError** - If the `target` string is empty or if `module` is not an instance of `nn.Module`.
- **AttributeError** - If at any point along the path resulting from
 the `target` string the (sub)path resolves to a non-existent
 attribute name or an object that is not an instance of `nn.Module`.

share_memory() → Self

See [`torch.Tensor.share_memory_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch.Tensor.share_memory_).

*property*stack_method

Get stack_method property.

state_dict(**args*, *destination=None*, *prefix=''*, *keep_vars=False*)

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to `None` are not included.

Note

The returned object is a shallow copy. It contains references
to the module's parameters and buffers.

Warning

Currently `state_dict()` also accepts positional arguments for
`destination`, `prefix` and `keep_vars` in order. However,
this is being deprecated and keyword arguments will be enforced in
future releases.

Warning

Please avoid the use of argument `destination` as it is not
designed for end-users.

Parameters:

- **destination** (*dict**,**optional*) - If provided, the state of module will
be updated into the dict and the same object is returned.
Otherwise, an `OrderedDict` will be created and returned.
Default: `None`.
- **prefix** (*str**,**optional*) - a prefix added to parameter and buffer
names to compose the keys in state_dict. Default: `''`.
- **keep_vars** (*bool**,**optional*) - by default the [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) s
returned in the state dict are detached from autograd. If it's
set to `True`, detaching will not be performed.
Default: `False`.

Returns:

a dictionary containing a whole state of the module

Return type:

dict

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight']
```

to(**args*, ***kwargs*)

Move to device.

to_empty(***, *device: str | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | None*, *recurse: bool = True*) → Self

Move the parameters and buffers to the specified device without copying storage.

Parameters:

- **device** ([`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - The desired device of the parameters
and buffers in this module.
- **recurse** (*bool*) - Whether parameters and buffers of submodules should
be recursively moved to the specified device.

Returns:

self

Return type:

Module

train(*mode: bool = True*) → Self

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

Parameters:

**mode** (*bool*) - whether to set training mode (`True`) or evaluation
mode (`False`). Default: `True`.

Returns:

self

Return type:

Module

transform_action_spec(*action_spec*)

Transform action spec.

transform_done_spec(*done_spec*)

Transform done spec.

transform_env_batch_size(*batch_size*)

Transform environment batch size.

transform_env_device(*device*)

Transform environment device.

transform_fake_tensordict(*fake_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

transform_input_spec(*input_spec*)

Transform input spec.

transform_observation_spec(*observation_spec*)

Transform observation spec.

transform_output_spec(*output_spec*)

Transform output spec.

transform_reward_spec(*reward_spec*)

Transform reward spec.

transform_state_spec(*state_spec*)

Transform state spec.

type(*dst_type: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | str*) → Self

Casts all parameters and buffers to `dst_type`.

Note

This method modifies the module in-place.

Parameters:

**dst_type** (*type**or**string*) - the desired type

Returns:

self

Return type:

Module

xpu(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

zero_grad(*set_to_none: bool = True*) → None

Reset gradients of all model parameters.

See similar function under [`torch.optim.Optimizer`](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) for more context.

Parameters:

**set_to_none** (*bool*) - instead of setting to zero, set the grads to None.
See [`torch.optim.Optimizer.zero_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) for details.