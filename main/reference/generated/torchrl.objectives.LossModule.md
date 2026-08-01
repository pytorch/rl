# LossModule

*class*torchrl.objectives.LossModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/common.html#LossModule)

A parent class for RL losses.

LossModule inherits from nn.Module. It is designed to read an input
TensorDict and return another tensordict
with loss keys named `"loss_*"`.

Splitting the loss in its component can then be used by the trainer to log
the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Variables:

**default_value_estimator** - The default value type of the class.
Losses that require a value estimation are equipped with a default value
pointer. This class attribute indicates which value estimator will be
used if none other is specified.
The value estimator can be changed using the `make_value_estimator()` method.

By default, the forward method is always decorated with a
gh `torchrl.envs.ExplorationType.MEAN`

To utilize the ability configuring the tensordict keys via
`set_keys()` a subclass must define an _AcceptedKeys dataclass.
This dataclass should include all keys that are intended to be configurable.
The default `_forward_value_estimator_keys()` implementation forwards
common value-estimator keys when present. Subclasses should override it when
the loss's key names need to be remapped before being forwarded to the
underlying value estimator.

Subclasses can declare a `_schedulable_buffers` frozenset to allow direct
scalar assignment (e.g. `loss.entropy_coeff = 0.003`) for registered
buffers that are commonly scheduled during training. The assignment performs
an in-place update, preserving the buffer's device and dtype.

Examples

```
>>> class MyLoss(LossModule):
>>> @dataclass
>>> class _AcceptedKeys:
>>> action = "action"
>>>
>>> def _forward_value_estimator_keys(self, **kwargs) -> None:
>>> pass
>>>
>>> loss = MyLoss()
>>> loss.set_keys(action="action2")
```

Note

When a policy that is wrapped or augmented with an exploration module is passed
to the loss, we want to deactivate the exploration through `set_exploration_type(<exploration>)` where
`<exploration>` is either `ExplorationType.MEAN`, `ExplorationType.MODE` or
`ExplorationType.DETERMINISTIC`. The default value is `DETERMINISTIC` and it is set
through the `deterministic_sampling_mode` loss attribute. If another
exploration mode is required (or if `DETERMINISTIC` is not available), one can
change the value of this attribute which will change the mode.

convert_to_functional(*module: [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)*, *module_name: str*, *expand_dim: int | None = None*, *create_target_params: bool = False*, *compare_against: list[[Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)] | None = None*, ***kwargs*) → None[[source]](../../_modules/torchrl/objectives/common.html#LossModule.convert_to_functional)

Converts a module to functional to be used in the loss.

Parameters:

- **module** (*TensorDictModule**or**compatible*) - a stateful tensordict module.
Parameters from this module will be isolated in the <module_name>_params
attribute and a stateless version of the module will be registered
under the module_name attribute.
- **module_name** (*str*) - name where the module will be found.
The parameters of the module will be found under `loss_module.<module_name>_params`
whereas the module will be found under `loss_module.<module_name>`.
- **expand_dim** (*int**,**optional*) -

if provided, the parameters of the module
will be expanded `N` times, where `N = expand_dim` along the
first dimension. This option is to be used whenever a target
network with more than one configuration is to be used.

Note

If a `compare_against` list of values is provided, the
resulting parameters will simply be a detached expansion
of the original parameters. If `compare_against` is not
provided, the value of the parameters will be resampled uniformly
between the minimum and maximum value of the parameter content.
- **create_target_params** (*bool**,**optional*) - if `True`, a detached
copy of the parameter will be available to feed a target network
under the name `loss_module.<module_name>_target_params`.
If `False` (default), this attribute will still be available
but it will be a detached instance of the parameters, not a copy.
In other words, any modification of the parameter value
will directly be reflected in the target parameters.
- **compare_against** (*iterable**of**parameters**,**optional*) - if provided,
this list of parameters will be used as a comparison set for
the parameters of the module. If the parameters are expanded
(`expand_dim > 0`), the resulting parameters for the module
will be a simple expansion of the original parameter. Otherwise,
the resulting parameters will be a detached version of the
original parameters. If `None`, the resulting parameters
will carry gradients as expected.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/common.html#LossModule.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

from_stateful_net(*network_name: str*, *stateful_net: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*)[[source]](../../_modules/torchrl/objectives/common.html#LossModule.from_stateful_net)

Populates the parameters of a model given a stateful version of the network.

See `get_stateful_net()` for details on how to gather a stateful version of the network.

Parameters:

- **network_name** (*str*) - the network name to reset.
- **stateful_net** (*nn.Module*) - the stateful network from which the params should be
gathered.

*property*functional

Whether the module is functional.

Unless it has been specifically designed not to be functional, all losses are functional.

get_stateful_net(*network_name: str*, *copy: bool | None = None*)[[source]](../../_modules/torchrl/objectives/common.html#LossModule.get_stateful_net)

Returns a stateful version of the network.

This can be used to initialize parameters.

Such networks will often not be callable out-of-the-box and will require a vmap call
to be executable.

Parameters:

- **network_name** (*str*) - the network name to gather.
- **copy** (*bool**,**optional*) -

if `True`, a deepcopy of the network is made.
Defaults to `True`.

Note

if the module is not functional, no copy is made.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/common.html#LossModule.make_value_estimator)

Value-function constructor.

If the non-default value function is wanted, it must be built using
this method.

Parameters:

- **value_type** ([*ValueEstimators*](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators)*,*[*ValueEstimatorBase*](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase)*, or**type*) -

The value
estimator to use. This can be one of the following:

- A [`ValueEstimators`](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) enum type
indicating which value function to use. If none is provided,
the default stored in the `default_value_estimator`
attribute will be used.
- A [`ValueEstimatorBase`](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase) instance,
which will be used directly as the value estimator.
- A [`ValueEstimatorBase`](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase) subclass,
which will be instantiated with the provided `hyperparams`.

The resulting value estimator class will be registered in
`self.value_type`, allowing future refinements.
- ****hyperparams** - hyperparameters to use for the value function.
If not provided, the value indicated by
`default_value_kwargs()` will be
used. When passing a `ValueEstimatorBase` subclass, these
hyperparameters are passed directly to the class constructor.

Returns:

Returns the loss module for method chaining.

Return type:

self

Examples

```
>>> from torchrl.objectives import DQNLoss
>>> # initialize the DQN loss
>>> actor = torch.nn.Linear(3, 4)
>>> dqn_loss = DQNLoss(actor, action_space="one-hot")
>>> # updating the parameters of the default value estimator
>>> dqn_loss.make_value_estimator(gamma=0.9)
>>> dqn_loss.make_value_estimator(
... ValueEstimators.TD1,
... gamma=0.9)
>>> # if we want to change the gamma value
>>> dqn_loss.make_value_estimator(dqn_loss.value_type, gamma=0.9)
```

Using a [`ValueEstimatorBase`](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase) subclass:

```
>>> from torchrl.objectives.value import TD0Estimator
>>> dqn_loss.make_value_estimator(TD0Estimator, gamma=0.99, value_network=value_net)
```

Using a [`ValueEstimatorBase`](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase) instance:

```
>>> from torchrl.objectives.value import GAE
>>> gae = GAE(gamma=0.99, lmbda=0.95, value_network=value_net)
>>> ppo_loss.make_value_estimator(gae)
```

named_parameters(*prefix: str = ''*, *recurse: bool = True*) → Iterator[tuple[str, [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]][[source]](../../_modules/torchrl/objectives/common.html#LossModule.named_parameters)

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

parameters(*recurse: bool = True*) → Iterator[[Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)][[source]](../../_modules/torchrl/objectives/common.html#LossModule.parameters)

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

register_coeff_buffer(*name: str*, *value: float | int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*) → None[[source]](../../_modules/torchrl/objectives/common.html#LossModule.register_coeff_buffer)

Register a scalar coefficient as a buffer, converting it to a tensor.

Eliminates the recurring `if not isinstance(value, Tensor): value =
torch.tensor(value); self.register_buffer(name, value)` boilerplate in
loss `__init__` methods.

If `value` is `None` the attribute is set to `None` instead of a
buffer being registered, matching the common optional-coefficient idiom
(e.g. `critic_coeff` / `clip_value`).

Parameters:

- **name** (*str*) - the buffer / attribute name.
- **value** (*float**,**int**,**Tensor**or**None*) - the coefficient. `None` sets
the attribute to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device for the buffer.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype for the buffer.

reset_parameters_recursive()[[source]](../../_modules/torchrl/objectives/common.html#LossModule.reset_parameters_recursive)

Reset the parameters of the module.

set_keys(***kwargs*) → None[[source]](../../_modules/torchrl/objectives/common.html#LossModule.set_keys)

Set tensordict key names.

Examples

```
>>> from torchrl.objectives import DQNLoss
>>> # initialize the DQN loss
>>> actor = torch.nn.Linear(3, 4)
>>> dqn_loss = DQNLoss(actor, action_space="one-hot")
>>> dqn_loss.set_keys(priority_key="td_error", action_value_key="action_value")
```

*property*value_estimator*: [ValueEstimatorBase](torchrl.objectives.value.ValueEstimatorBase.html#torchrl.objectives.value.ValueEstimatorBase)*

The value function blends in the reward and value estimate(s) from upcoming state(s)/state-action pair(s) into a target value estimate for the value network.

*property*vmap_randomness

Vmap random mode.

The vmap randomness mode controls what [`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) should do when dealing with
functions with a random outcome such as [`randn()`](https://docs.pytorch.org/docs/stable/generated/torch.randn.html#torch.randn) and [`rand()`](https://docs.pytorch.org/docs/stable/generated/torch.rand.html#torch.rand).
If "error", any random function will raise an exception indicating that vmap does not
know how to handle the random call.

If "different", every element of the batch along which vmap is being called will
behave differently. If "same", vmaps will copy the same result across all elements.

`vmap_randomness` defaults to "error" if no random module is detected, and to "different" in
other cases. By default, only a limited number of modules are listed as random, but the list can be extended
using the [`add_random_module()`](torchrl.objectives.add_random_module.html#torchrl.objectives.add_random_module) function.

This property supports setting its value.