# CQLLoss

*class*torchrl.objectives.CQLLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/cql.html#CQLLoss)

TorchRL implementation of the continuous CQL loss.

Presented in "Conservative Q-Learning for Offline Reinforcement Learning" [https://arxiv.org/abs/2006.04779](https://arxiv.org/abs/2006.04779)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor
- **qvalue_network** (*TensorDictModule**or**list**of**TensorDictModule*) -

Q(s, a) parametric model.
This module typically outputs a `"state_action_value"` entry.
If a single instance of qvalue_network is provided, it will be duplicated `N`
times (where `N=2` for this loss). If a list of modules is passed, their
parameters will be stacked unless they share the same identity (in which case
the original parameter will be expanded).

Warning

When a list of parameters if passed, it will **not** be compared against the policy parameters
and all the parameters will be considered as untied.

Keyword Arguments:

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
- **gamma** (`float`, optional) - Discount factor. Default is `None`.
- **temperature** (`float`, optional) - CQL temperature. Default is 1.0.
- **min_q_weight** (`float`, optional) - Minimum Q weight. Default is 1.0.
- **max_q_backup** (*bool**,**optional*) - Whether to use the max-min Q backup.
Default is `False`.
- **deterministic_backup** (*bool**,**optional*) - Whether to use the deterministic. Default is `True`.
- **num_random** (*int**,**optional*) - Number of random actions to sample for the CQL loss.
Default is 10.
- **with_lagrange** (*bool**,**optional*) - Whether to use the Lagrange multiplier.
Default is `False`.
- **lagrange_thresh** (`float`, optional) - Lagrange threshold. Default is 0.0.
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
>>> from torchrl.objectives.cql import CQLLoss
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
>>> loss = CQLLoss(actor, qvalue)
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
 loss_actor_bc: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_cql: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor, value, and qvalue network.
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "loss_alpha", "loss_alpha_prime", "alpha", "entropy"]`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.cql import CQLLoss
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
>>> loss = CQLLoss(actor, qvalue)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_actor, loss_actor_bc, loss_qvalue, loss_cql, *_ = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `CQLLoss.select_out_keys()`
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

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/cql.html#CQLLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/cql.html#CQLLoss.make_value_estimator)

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