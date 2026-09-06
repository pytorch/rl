# TransformedEnv

*class*torchrl.envs.transforms.TransformedEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv)

A transformed environment.

Parameters:

- **base_env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - original environment to be transformed.
- **transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*or**callable**,**optional*) -

transform to apply to the tensordict resulting
from `base_env.step(td)`. If none is provided, an empty Compose
placeholder in an eval mode is used.

Note

If `transform` is a callable, it must receive as input a single tensordict
and output a tensordict as well. The callable will be called at `step`
and `reset` time: if it acts on the reward (which is absent at
reset time), a check needs to be implemented to ensure that
the transform will run smoothly:

```
>>> def add_1(data):
... if "reward" in data.keys():
... return data.set("reward", data.get("reward") + 1)
... return data
>>> env = TransformedEnv(base_env, add_1)
```
- **cache_specs** (*bool**,**optional*) - if `True`, the specs will be cached once
and for all after the first call (i.e. the specs will be
transformed only once). If the transform changes during
training, the original spec transform may not be valid anymore,
in which case this value should be set to False. Default is
True.

Keyword Arguments:

**auto_unwrap** (*bool**,**optional*) -

if `True`, wrapping a transformed env in transformed env
unwraps the transforms of the inner TransformedEnv in the outer one (the new instance).
Defaults to `True`.

Note

This behavior will switch to `False` in v0.9.

See also

[`set_auto_unwrap_transformed_env`](torchrl.set_auto_unwrap_transformed_env.html#torchrl.set_auto_unwrap_transformed_env)

Examples

```
>>> env = GymEnv("Pendulum-v0")
>>> transform = RewardScaling(0.0, 1.0)
>>> transformed_env = TransformedEnv(env, transform)
>>> # check auto-unwrap
>>> transformed_env = TransformedEnv(transformed_env, StepCounter())
>>> # The inner env has been unwrapped
>>> assert isinstance(transformed_env.base_env, GymEnv)
```

add_truncated_keys() → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.add_truncated_keys)

Adds truncated keys to the environment.

append_transform(*transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*) → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.append_transform)

Appends a transform to the env.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or callable are accepted.

*property*batch_locked*: bool*

Whether the environment can be used with a batch size different from the one it was initialized with or not.

If True, the env needs to be used with a tensordict having the same batch size as the env.
batch_locked is an immutable property.

*property*batch_size*: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*

Number of envs batched in this environment instance organised in a torch.Size() object.

Environment may be similar or different but it is assumed that they have little if
not no interactions between them (e.g., multi-task or batched execution
in parallel).

empty_cache()[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.empty_cache)

Erases all the cached values.

For regular envs, the key lists (reward, done etc) are cached, but in some cases
they may change during the execution of the code (eg, when adding a transform).

eval() → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.eval)

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

fake_tensordict() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.fake_tensordict)

Build a fake tensordict and let the transform chain post-process it.

*property*input_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Observation spec of the transformed environment.

insert_transform(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*) → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.insert_transform)

Inserts a transform to the env at the desired index.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or callable are accepted.

load_state_dict(*state_dict: OrderedDict*, ***kwargs*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.load_state_dict)

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

*property*output_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Observation spec of the transformed environment.

rand_action(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.rand_action)

Performs a random action given the action_spec attribute.

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - tensordict where the resulting action should be written.

Returns:

a tensordict object with the "action" entry updated with a random
sample from the action-spec.

set_missing_tolerance(*mode=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.set_missing_tolerance)

Indicates if an KeyError should be raised whenever an in_key is missing from the input tensordict.

set_seed(*seed: int | None = None*, *static_seed: bool = False*) → int | None[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.set_seed)

Set the seeds of the environment.

state_dict(**args*, ***kwargs*) → OrderedDict[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.state_dict)

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

to(**args*, ***kwargs*) → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.to)

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

train(*mode: bool = True*) → TransformedEnv[[source]](../../_modules/torchrl/envs/transforms/_base.html#TransformedEnv.train)

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