# CrossQLoss

*class*torchrl.objectives.CrossQLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss)

TorchRL implementation of the CrossQ loss.

Presented in "CROSSQ: BATCH NORMALIZATION IN DEEP REINFORCEMENT LEARNING
FOR GREATER SAMPLE EFFICIENCY AND SIMPLICITY" [https://openreview.net/pdf?id=PczQtTsTIX](https://openreview.net/pdf?id=PczQtTsTIX)

This class has three loss functions that will be called sequentially by the forward method:
`qvalue_loss()`, `actor_loss()` and `alpha_loss()`. Alternatively, they can
be called by the user that order.

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor
- **qvalue_network** (*TensorDictModule*) -

Q(s, a) parametric model.
This module typically outputs a `"state_action_value"` entry.
If a single instance of qvalue_network is provided, it will be duplicated `num_qvalue_nets`
times. If a list of modules is passed, their
parameters will be stacked unless they share the same identity (in which case
the original parameter will be expanded).

Warning

When a list of parameters if passed, it will **not** be compared against the policy parameters
and all the parameters will be considered as untied.

Keyword Arguments:

- **num_qvalue_nets** (*integer**,**optional*) - number of Q-Value networks used.
Defaults to `2`.
- **loss_function** (*str**,**optional*) - loss function to be used with
the value function loss. Default is "smooth_l1".
- **alpha_init** (`float`, optional) - initial entropy multiplier.
Default is 1.0.
- **min_alpha** (`float`, optional) - min value of alpha.
Default is None (no minimum value).
- **max_alpha** (`float`, optional) - max value of alpha.
Default is None (no maximum value).
- **action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - the action tensor spec. If not provided
and the target entropy is `"auto"`, it will be retrieved from
the actor.
- **fixed_alpha** (*bool**,**optional*) - if `True`, alpha will be fixed to its
initial value. Otherwise, alpha will be optimized to
match the 'target_entropy' value.
Default is `False`.
- **target_entropy** (`float` or str, optional) - Target entropy for the
stochastic policy. Default is "auto", where target entropy is
computed as `-prod(n_actions)`.
- **priority_key** (*str**,**optional*) - [Deprecated, use .set_keys(priority_key=priority_key) instead]
Tensordict key where to write the
priority (for prioritized replay buffer usage). Defaults to `"td_error"`.
- **separate_losses** (*bool**,**optional*) - if `True`, shared parameters between
policy and critic will only be trained on the policy loss.
Defaults to `False`, i.e., gradients are propagated to shared
parameters for both policy and critic losses.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.
- **deactivate_vmap** (*bool**,**optional*) - whether to deactivate vmap calls and replace them with a plain for loop.
Defaults to `False`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.crossq import CrossQLoss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["loc", "scale"],
... spec=spec,
... distribution_class=TanhNormal)
>>> class ValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> module = ValueClass()
>>> qvalue = ValueOperator(
... module=module,
... in_keys=['observation', 'action'])
>>> loss = CrossQLoss(actor, qvalue)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> data = TensorDict({
... "observation": torch.randn(*batch, n_obs),
... "action": action,
... ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "reward"): torch.randn(*batch, 1),
... ("next", "observation"): torch.randn(*batch, n_obs),
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor and qvalue network.
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]`

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives import CrossQLoss
>>> _ = torch.manual_seed(42)
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["loc", "scale"],
... spec=spec,
... distribution_class=TanhNormal)
>>> class ValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> module = ValueClass()
>>> qvalue = ValueOperator(
... module=module,
... in_keys=['observation', 'action'])
>>> loss = CrossQLoss(actor, qvalue)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_actor, loss_qvalue, _, _, _ = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `CrossQLoss.select_out_keys()`
method.

Examples

```
>>> _ = loss.select_out_keys('loss_actor', 'loss_qvalue')
>>> loss_actor, loss_qvalue = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

actor_loss(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), dict[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]][[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.actor_loss)

Compute the actor loss.

The actor loss should be computed after the `qvalue_loss()` and before the ~.alpha_loss which
requires the log_prob field of the metadata returned by this method.

Parameters:

**tensordict** (*TensorDictBase*) - the input data for the loss. Check the class's in_keys to see what fields
are required for this to be computed.

Returns: a differentiable tensor with the alpha loss along with a metadata dictionary containing the detached "log_prob" of the sampled action.

alpha_loss(*log_prob: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.alpha_loss)

Compute the entropy loss.

The entropy loss should be computed last.

Parameters:

- **log_prob** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a log-probability as computed by the `actor_loss()` and returned in the metadata.
- **tensordict** (*TensorDictBase**,**optional*) - the input tensordict.

Returns: a differentiable tensor with the entropy loss.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.forward)

The forward method.

Computes successively the `qvalue_loss()`, `actor_loss()` and `alpha_loss()`, and returns
a tensordict with these values along with the "alpha" value and the "entropy" value (detached).
To see what keys are expected in the input tensordict and what keys are expected as output, check the
class's "in_keys" and "out_keys" attributes.

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

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.make_value_estimator)

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

maybe_init_target_entropy(*fault_tolerant=True*)[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.maybe_init_target_entropy)

Initialize the target entropy.

Parameters:

**fault_tolerant** (*bool**,**optional*) - if `True`, returns None if the target entropy
cannot be determined. Raises an exception otherwise. Defaults to `True`.

qvalue_loss(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), dict[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]][[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.qvalue_loss)

Compute the q-value loss.

The q-value loss should be computed before the `actor_loss()`.

Parameters:

**tensordict** (*TensorDictBase*) - the input data for the loss. Check the class's in_keys to see what fields
are required for this to be computed.

Returns: a differentiable tensor with the qvalue loss along with a metadata dictionary containing

the detached "td_error" to be used for prioritized sampling.

set_keys(***kwargs*) → None[[source]](../../_modules/torchrl/objectives/crossq.html#CrossQLoss.set_keys)

Set tensordict key names.

Examples

```
>>> from torchrl.objectives import DQNLoss
>>> # initialize the DQN loss
>>> actor = torch.nn.Linear(3, 4)
>>> dqn_loss = DQNLoss(actor, action_space="one-hot")
>>> dqn_loss.set_keys(priority_key="td_error", action_value_key="action_value")
```

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

*property*target_entropy_buffer

The target entropy.

This value can be controlled via the target_entropy kwarg in the constructor.