# TD3Loss

*class*torchrl.objectives.TD3Loss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/td3.html#TD3Loss)

TD3 Loss module.

Parameters:

- **actor_network** (*TensorDictModule*) - the actor to be trained
- **qvalue_network** (*TensorDictModule*) -

a single Q-value network or a list of
Q-value networks.
If a single instance of qvalue_network is provided, it will be duplicated `num_qvalue_nets`
times. If a list of modules is passed, their
parameters will be stacked unless they share the same identity (in which case
the original parameter will be expanded).

Warning

When a list of parameters if passed, it will **not** be compared against the policy parameters
and all the parameters will be considered as untied.

Keyword Arguments:

- **bounds** (*tuple**of*[*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - the bounds of the action space.
Exclusive with action_spec. Either this or `action_spec` must
be provided.
- **action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - the action spec.
Exclusive with bounds. Either this or `bounds` must be provided.
- **num_qvalue_nets** (*int**,**optional*) - Number of Q-value networks to be
trained. Default is `10`.
- **policy_noise** (`float`, optional) - Standard deviation for the target
policy action noise. Default is `0.2`.
- **noise_clip** (`float`, optional) - Clipping range value for the sampled
target policy action noise. Default is `0.5`.
- **priority_key** (*str**,**optional*) - Key where to write the priority value
for prioritized replay buffers. Default is
"td_error".
- **loss_function** (*str**,**optional*) - loss function to be used for the Q-value.
Can be one of `"smooth_l1"`, `"l2"`,
`"l1"`, Default is `"smooth_l1"`.
- **delay_actor** (*bool**,**optional*) - whether to separate the target actor
networks from the actor networks used for
data collection. Default is `True`.
- **delay_qvalue** (*bool**,**optional*) - Whether to separate the target Q value
networks from the Q value networks used
for data collection. Default is `True`.
- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - the action tensor spec. If not provided
and the target entropy is `"auto"`, it will be retrieved from
the actor.
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
>>> from torchrl.modules.tensordict_module.actors import Actor, ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.td3 import TD3Loss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> module = nn.Linear(n_obs, n_act)
>>> actor = Actor(
... module=module,
... spec=spec)
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
>>> loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
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
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 next_state_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 state_action_value_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor and qvalue network
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "pred_value", "state_action_value_actor", "next_state_value", "target_value",]`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data import Bounded
>>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
>>> from torchrl.objectives.td3 import TD3Loss
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> module = nn.Linear(n_obs, n_act)
>>> actor = Actor(
... module=module,
... spec=spec)
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
>>> loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
>>> _ = loss.select_out_keys("loss_actor", "loss_qvalue")
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_actor, loss_qvalue = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_reward=torch.randn(*batch, 1),
... next_observation=torch.randn(*batch, n_obs))
>>> loss_actor.backward()
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/td3.html#TD3Loss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/td3.html#TD3Loss.make_value_estimator)

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