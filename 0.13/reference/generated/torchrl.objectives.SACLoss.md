# SACLoss

*class*torchrl.objectives.SACLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/sac.html#SACLoss)

TorchRL implementation of the SAC loss.

Presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
Reinforcement Learning with a Stochastic Actor" [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)
and "Soft Actor-Critic Algorithms and Applications" [https://arxiv.org/abs/1812.05905](https://arxiv.org/abs/1812.05905)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor
- **qvalue_network** (*TensorDictModule**|**list**[**TensorDictModule**]*) -

Q(s, a) parametric model.
This module typically outputs a `"state_action_value"` entry.
If a single instance of qvalue_network is provided, it will be duplicated `num_qvalue_nets`
times. If a list of modules is passed, their
parameters will be stacked unless they share the same identity (in which case
the original parameter will be expanded).
When a list is provided, the first module is used as the functional forward
reference (its `in_keys`/`out_keys` are used), so all modules must share
the same signature.

Warning

When a list of parameters if passed, it will **not** be compared against the policy parameters
and all the parameters will be considered as untied.
- **value_network** (*TensorDictModule**,**optional*) -

V(s) parametric model.
This module typically outputs a `"state_value"` entry.

Note

If not provided, the second version of SAC is assumed, where
only the Q-Value network is needed.

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
- **delay_actor** (*bool**,**optional*) - Whether to separate the target actor
networks from the actor networks used for data collection.
Default is `False`.
- **delay_qvalue** (*bool**,**optional*) - Whether to separate the target Q value
networks from the Q value networks used for data collection.
Default is `True`.
- **delay_value** (*bool**,**optional*) - Whether to separate the target value
networks from the value networks used for data collection.
Default is `True`.
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
- **skip_done_states** (*bool**,**optional*) - whether the actor network used for value computation should only be run on
valid, non-terminating next states. If `True`, it is assumed that the done state can be broadcast to the
shape of the data and that masking the data results in a valid data structure. Among other things, this may
not be true in MARL settings or when using RNNs. Defaults to `False`.
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
>>> from torchrl.objectives.sac import SACLoss
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
>>> module = nn.Linear(n_obs, 1)
>>> value = ValueOperator(
... module=module,
... in_keys=["observation"])
>>> loss = SACLoss(actor, qvalue, value)
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
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor, value, and qvalue network.
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]` + `"loss_value"` if version one is used.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.sac import SACLoss
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
>>> module = nn.Linear(n_obs, 1)
>>> value = ValueOperator(
... module=module,
... in_keys=["observation"])
>>> loss = SACLoss(actor, qvalue, value)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_actor, loss_qvalue, _, _, _, _ = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `SACLoss.select_out_keys()`
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

alpha_loss(*log_prob: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/sac.html#SACLoss.alpha_loss)

Compute the alpha loss for SAC.

This method computes the alpha loss which adapts the entropy coefficient
to maintain the target entropy level.

Parameters:

**log_prob** (*Tensor*) - The log probability of actions from the actor network.

Returns:

The alpha loss tensor

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/sac.html#SACLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

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

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/sac.html#SACLoss.make_value_estimator)

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