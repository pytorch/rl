# DiscreteCQLLoss

*class*torchrl.objectives.DiscreteCQLLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/cql.html#DiscreteCQLLoss)

TorchRL implementation of the discrete CQL loss.

This class implements the discrete conservative Q-learning (CQL) loss function, as presented in the paper
"Conservative Q-Learning for Offline Reinforcement Learning" ([https://arxiv.org/abs/2006.04779](https://arxiv.org/abs/2006.04779)).

Parameters:

**value_network** (*Union**[*[*QValueActor*](torchrl.modules.QValueActor.html#torchrl.modules.QValueActor)*,**nn.Module**]*) - The Q-value network used to estimate state-action values.

Keyword Arguments:

- **loss_function** (*Optional**[**str**]*) - The distance function used to calculate the distance between the predicted
Q-values and the target Q-values. Defaults to `l2`.
- **delay_value** (*bool*) - Whether to separate the target Q value
networks from the Q value networks used for data collection.
Default is `True`.
- **gamma** (`float`, optional) - Discount factor. Default is `None`.
- **action_space** - The action space of the environment. If None, it is inferred from the value network.
Defaults to None.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

Examples

```
>>> from torchrl.modules import MLP, QValueActor
>>> from torchrl.data import OneHot
>>> from torchrl.objectives import DiscreteCQLLoss
>>> n_obs, n_act = 4, 3
>>> value_net = MLP(in_features=n_obs, out_features=n_act)
>>> spec = OneHot(n_act)
>>> actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
>>> loss = DiscreteCQLLoss(actor, action_space=spec)
>>> batch = [10,]
>>> data = TensorDict({
... "observation": torch.randn(*batch, n_obs),
... "action": spec.rand(batch),
... ("next", "observation"): torch.randn(*batch, n_obs),
... ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "reward"): torch.randn(*batch, 1)
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 loss_cql: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 td_error: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["observation", "next_observation", "action", "next_reward", "next_done", "next_terminated"]`,
and a single loss value is returned.

Examples

```
>>> from torchrl.objectives import DiscreteCQLLoss
>>> from torchrl.data import OneHot
>>> from torch import nn
>>> import torch
>>> n_obs = 3
>>> n_action = 4
>>> action_spec = OneHot(n_action)
>>> value_network = nn.Linear(n_obs, n_action) # a simple value model
>>> dcql_loss = DiscreteCQLLoss(value_network, action_space=action_spec)
>>> # define data
>>> observation = torch.randn(n_obs)
>>> next_observation = torch.randn(n_obs)
>>> action = action_spec.rand()
>>> next_reward = torch.randn(1)
>>> next_done = torch.zeros(1, dtype=torch.bool)
>>> next_terminated = torch.zeros(1, dtype=torch.bool)
>>> loss_val = dcql_loss(
... observation=observation,
... next_observation=next_observation,
... next_reward=next_reward,
... next_done=next_done,
... next_terminated=next_terminated,
... action=action)
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/cql.html#DiscreteCQLLoss.forward)

Computes the (DQN) CQL loss given a tensordict sampled from the replay buffer.

This function will also write a "td_error" key that can be used by prioritized replay buffers to assign

a priority to items in the tensordict.

Parameters:

**tensordict** (*TensorDictBase*) - a tensordict with keys ["action"] and the in_keys of
the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

Returns:

a tensor containing the CQL loss.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/cql.html#DiscreteCQLLoss.make_value_estimator)

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