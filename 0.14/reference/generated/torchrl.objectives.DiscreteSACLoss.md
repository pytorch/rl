# DiscreteSACLoss

*class*torchrl.objectives.DiscreteSACLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/sac.html#DiscreteSACLoss)

Discrete SAC Loss module.

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - the actor to be trained
- **qvalue_network** (*TensorDictModule*) - a single Q-value network that will be multiplicated as many times as needed.
- **action_space** (*str**or*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - Action space. Must be one of
`"one-hot"`, `"mult_one_hot"`, `"binary"` or `"categorical"`,
or an instance of the corresponding specs ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot),
[`torchrl.data.MultiOneHot`](torchrl.data.MultiOneHot.html#torchrl.data.MultiOneHot),
[`torchrl.data.Binary`](torchrl.data.Binary.html#torchrl.data.Binary) or [`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)).
- **num_actions** (*int**,**optional*) - number of actions in the action space.
To be provided if target_entropy is set to "auto".
- **num_qvalue_nets** (*int**,**optional*) - Number of Q-value networks to be trained. Default is 2.
- **loss_function** (*str**,**optional*) - loss function to be used for the Q-value. Can be one of "smooth_l1", "l2",
"l1", Default is "smooth_l1".
- **alpha_init** (`float`, optional) - initial entropy multiplier.
Default is 1.0.
- **min_alpha** (`float`, optional) - min value of alpha.
Default is None (no minimum value).
- **max_alpha** (`float`, optional) - max value of alpha.
Default is None (no maximum value).
- **fixed_alpha** (*bool**,**optional*) - whether alpha should be trained to match a target entropy. Default is `False`.
- **target_entropy_weight** (`float`, optional) - weight for the target entropy term.
- **target_entropy** (*Union**[**str**,**Number**]**,**optional*) - Target entropy for the
stochastic policy. Default is "auto", where target entropy is
computed as `-target_entropy_weight * log(1 / num_actions)`.
- **delay_qvalue** (*bool**,**optional*) - Whether to separate the target Q value networks from the Q value networks used
for data collection. Default is `False`.
- **priority_key** (*str**,**optional*) - [Deprecated, use .set_keys(priority_key=priority_key) instead]
Key where to write the priority value for prioritized replay buffers.
Default is "td_error".
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
>>> from torchrl.data.tensor_specs import OneHot
>>> from torchrl.modules.distributions import NormalParamExtractor, OneHotCategorical
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.sac import DiscreteSACLoss
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> n_act, n_obs = 4, 3
>>> spec = OneHot(n_act)
>>> module = TensorDictModule(nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["logits"],
... out_keys=["action"],
... spec=spec,
... distribution_class=OneHotCategorical)
>>> qvalue = TensorDictModule(
... nn.Linear(n_obs, n_act),
... in_keys=["observation"],
... out_keys=["action_value"],
... )
>>> loss = DiscreteSACLoss(actor, qvalue, action_space=spec, num_actions=spec.space.n)
>>> batch = [2,]
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
`["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]`.
The output keys can also be filtered using `DiscreteSACLoss.select_out_keys()` method.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import OneHot
>>> from torchrl.modules.distributions import NormalParamExtractor, OneHotCategorical
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.sac import DiscreteSACLoss
>>> n_act, n_obs = 4, 3
>>> spec = OneHot(n_act)
>>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["logits"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["logits"],
... out_keys=["action"],
... spec=spec,
... distribution_class=OneHotCategorical)
>>> class ValueClass(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(n_obs, n_act)
... def forward(self, obs):
... return self.linear(obs)
>>> module = ValueClass()
>>> qvalue = ValueOperator(
... module=module,
... in_keys=['observation'])
>>> loss = DiscreteSACLoss(actor, qvalue, num_actions=actor.spec["action"].space.n)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> # filter output keys to "loss_actor", and "loss_qvalue"
>>> _ = loss.select_out_keys("loss_actor", "loss_qvalue")
>>> loss_actor, loss_qvalue = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

alpha_loss(*log_prob: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/sac.html#DiscreteSACLoss.alpha_loss)

Compute the alpha loss for discrete SAC.

This method computes the alpha loss which adapts the entropy coefficient
to maintain the target entropy level for discrete actions.

Parameters:

**log_prob** (*Tensor*) - The log probability of actions from the actor network.

Returns:

The alpha loss tensor

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/sac.html#DiscreteSACLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/sac.html#DiscreteSACLoss.make_value_estimator)

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