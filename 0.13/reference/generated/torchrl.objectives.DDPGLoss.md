# DDPGLoss

*class*torchrl.objectives.DDPGLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/ddpg.html#DDPGLoss)

The DDPG Loss class.

Parameters:

- **actor_network** (*TensorDictModule*) - a policy operator.
- **value_network** (*TensorDictModule*) - a Q value operator.
- **loss_function** (*str*) - loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
- **delay_actor** (*bool**,**optional*) - whether to separate the target actor networks from the actor networks used for
data collection. Default is `False`.
- **delay_value** (*bool**,**optional*) - whether to separate the target value networks from the value networks used for
data collection. Default is `True`.
- **separate_losses** (*bool**,**optional*) - if `True`, shared parameters between
policy and critic will only be trained on the policy loss.
Defaults to `False`, i.e., gradients are propagated to shared
parameters for both policy and critic losses.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
>>> from torchrl.objectives.ddpg import DDPGLoss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> actor = Actor(spec=spec, module=nn.Linear(n_obs, n_act))
>>> class ValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> module = ValueClass()
>>> value = ValueOperator(
... module=module,
... in_keys=["observation", "action"])
>>> loss = DDPGLoss(actor, value)
>>> batch = [2, ]
>>> data = TensorDict({
... "observation": torch.randn(*batch, n_obs),
... "action": spec.rand(batch),
... ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "reward"): torch.randn(*batch, 1),
... ("next", "observation"): torch.randn(*batch, n_obs),
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["next_reward", "next_done", "next_terminated"]` + in_keys of the actor_network and value_network.
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_value", "pred_value", "target_value", "pred_value_max", "target_value_max"]`

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
>>> from torchrl.objectives.ddpg import DDPGLoss
>>> _ = torch.manual_seed(42)
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> actor = Actor(spec=spec, module=nn.Linear(n_obs, n_act))
>>> class ValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs + n_act, 1)
... def forward(self, obs, act):
... return self.linear(torch.cat([obs, act], -1))
>>> module = ValueClass()
>>> value = ValueOperator(
... module=module,
... in_keys=["observation", "action"])
>>> loss = DDPGLoss(actor, value)
>>> loss_actor, loss_value, pred_value, target_value, pred_value_max, target_value_max = loss(
... observation=torch.randn(n_obs),
... action=spec.rand(),
... next_done=torch.zeros(1, dtype=torch.bool),
... next_terminated=torch.zeros(1, dtype=torch.bool),
... next_observation=torch.randn(n_obs),
... next_reward=torch.randn(1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `DDPGLoss.select_out_keys()`
method.

Examples

```
>>> loss.select_out_keys('loss_actor', 'loss_value')
>>> loss_actor, loss_value = loss(
... observation=torch.randn(n_obs),
... action=spec.rand(),
... next_done=torch.zeros(1, dtype=torch.bool),
... next_terminated=torch.zeros(1, dtype=torch.bool),
... next_observation=torch.randn(n_obs),
... next_reward=torch.randn(1))
>>> loss_actor.backward()
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/ddpg.html#DDPGLoss.forward)

Computes the DDPG losses given a tensordict sampled from the replay buffer.

This function will also write a "td_error" key that can be used by prioritized replay buffers to assign

a priority to items in the tensordict.

Parameters:

**tensordict** (*TensorDictBase*) - a tensordict with keys ["done", "terminated", "reward"] and the in_keys of the actor
and value networks.

Returns:

a tuple of 2 tensors containing the DDPG loss.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/ddpg.html#DDPGLoss.make_value_estimator)

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