# IQLLoss

*class*torchrl.objectives.IQLLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/iql.html#IQLLoss)

TorchRL implementation of the IQL loss.

Presented in "Offline Reinforcement Learning with Implicit Q-Learning" [https://arxiv.org/abs/2110.06169](https://arxiv.org/abs/2110.06169)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor
- **qvalue_network** (*TensorDictModule*) -

Q(s, a) parametric model
If a single instance of qvalue_network is provided, it will be duplicated `num_qvalue_nets`
times. If a list of modules is passed, their
parameters will be stacked unless they share the same identity (in which case
the original parameter will be expanded).

Warning

When a list of parameters if passed, it will **not** be compared against the policy parameters
and all the parameters will be considered as untied.
- **value_network** (*TensorDictModule**,**optional*) - V(s) parametric model.

Keyword Arguments:

- **num_qvalue_nets** (*integer**,**optional*) - number of Q-Value networks used.
Defaults to `2`.
- **loss_function** (*str**,**optional*) - loss function to be used with
the value function loss. Default is "smooth_l1".
- **temperature** (`float`, optional) - Inverse temperature (beta).
For smaller hyperparameter values, the objective behaves similarly to
behavioral cloning, while for larger values, it attempts to recover the
maximum of the Q-function.
- **expectile** (`float`, optional) - expectile \(\tau\). A larger value of \(\tau\) is crucial
for antmaze tasks that require dynamical programming ("stichting").
- **priority_key** (*str**,**optional*) - [Deprecated, use .set_keys(priority_key=priority_key) instead]
tensordict key where to write the priority (for prioritized replay
buffer usage). Default is "td_error".
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
>>> from torchrl.objectives.iql import IQLLoss
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
>>> class QValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> qvalue = SafeModule(
... QValueClass(),
... in_keys=["observation", "action"],
... out_keys=["state_action_value"],
... )
>>> value = SafeModule(
... nn.Linear(n_obs, 1),
... in_keys=["observation"],
... out_keys=["state_value"],
... )
>>> loss = IQLLoss(actor, qvalue, value)
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
 entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor, value, and qvalue network
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "loss_value", "entropy"]`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.iql import IQLLoss
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
>>> class QValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> qvalue = SafeModule(
... QValueClass(),
... in_keys=["observation", "action"],
... out_keys=["state_action_value"],
... )
>>> value = SafeModule(
... nn.Linear(n_obs, 1),
... in_keys=["observation"],
... out_keys=["state_value"],
... )
>>> loss = IQLLoss(actor, qvalue, value)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_actor, loss_qvalue, loss_value, entropy = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `IQLLoss.select_out_keys()`
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

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/iql.html#IQLLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

*static*loss_value_diff(*diff*, *expectile=0.8*)[[source]](../../_modules/torchrl/objectives/iql.html#IQLLoss.loss_value_diff)

Loss function for iql expectile value difference.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/iql.html#IQLLoss.make_value_estimator)

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