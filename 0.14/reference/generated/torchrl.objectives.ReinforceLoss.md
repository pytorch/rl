# ReinforceLoss

*class*torchrl.objectives.ReinforceLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/reinforce.html#ReinforceLoss)

Reinforce loss module.

Presented in "Simple statistical gradient-following sota-implementations for connectionist reinforcement learning", Williams, 1992
[https://doi.org/10.1007/BF00992696](https://doi.org/10.1007/BF00992696)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - policy operator.
- **critic_network** ([*ValueOperator*](torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)) - value operator.

Keyword Arguments:

- **delay_value** (*bool**,**optional*) - if `True`, a target network is needed
for the critic. Defaults to `False`. Incompatible with `functional=False`.
- **loss_critic_type** (*str*) - loss function for the value discrepancy.
Can be one of "l1", "l2" or "smooth_l1". Defaults to `"smooth_l1"`.
- **advantage_key** (*str*) - [Deprecated, use .set_keys(advantage_key=advantage_key) instead]
The input tensordict key where the advantage is expected to be written.
Defaults to `"advantage"`.
- **value_target_key** (*str*) - [Deprecated, use .set_keys(value_target_key=value_target_key) instead]
The input tensordict key where the target state
value is expected to be written. Defaults to `"value_target"`.
- **separate_losses** (*bool**,**optional*) - if `True`, shared parameters between
policy and critic will only be trained on the policy loss.
Defaults to `False`, i.e., gradients are propagated to shared
parameters for both policy and critic losses.
- **functional** (*bool**,**optional*) - whether modules should be functionalized.
Functionalizing permits features like meta-RL, but makes it
impossible to use distributed models (DDP, FSDP, ...) and comes
with a little cost. Defaults to `True`.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.
- **clip_value** (`float`, optional) - If provided, it will be used to compute a clipped version of the value
prediction with respect to the input tensordict value estimate and use it to calculate the value loss.
The purpose of clipping is to limit the impact of extreme value predictions, helping stabilize training
and preventing large updates. However, it will have no impact if the value estimate was done by the current
version of the value estimator. Defaults to `None`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Unbounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.reinforce import ReinforceLoss
>>> from tensordict import TensorDict
>>> n_obs, n_act = 3, 5
>>> value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
>>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor_net = ProbabilisticActor(
... module,
... distribution_class=TanhNormal,
... return_log_prob=True,
... in_keys=["loc", "scale"],
... spec=Unbounded(n_act),)
>>> loss = ReinforceLoss(actor_net, value_net)
>>> batch = 2
>>> data = TensorDict({
... "observation": torch.randn(batch, n_obs),
... "next": {
... "observation": torch.randn(batch, n_obs),
... "reward": torch.randn(batch, 1),
... "done": torch.zeros(batch, 1, dtype=torch.bool),
... "terminated": torch.zeros(batch, 1, dtype=torch.bool),
... },
... "action": torch.randn(batch, n_act),
... }, [batch])
>>> loss(data)
TensorDict(
 fields={
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor and critic network
The return value is a tuple of tensors in the following order: `["loss_actor", "loss_value"]`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Unbounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.reinforce import ReinforceLoss
>>> n_obs, n_act = 3, 5
>>> value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
>>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor_net = ProbabilisticActor(
... module,
... distribution_class=TanhNormal,
... return_log_prob=True,
... in_keys=["loc", "scale"],
... spec=Unbounded(n_act),)
>>> loss = ReinforceLoss(actor_net, value_net)
>>> batch = 2
>>> loss_actor, loss_value = loss(
... observation=torch.randn(batch, n_obs),
... next_observation=torch.randn(batch, n_obs),
... next_reward=torch.randn(batch, 1),
... next_done=torch.zeros(batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(batch, 1, dtype=torch.bool),
... action=torch.randn(batch, n_act),)
>>> loss_actor.backward()
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/reinforce.html#ReinforceLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

*property*functional

Whether the module is functional.

Unless it has been specifically designed not to be functional, all losses are functional.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/reinforce.html#ReinforceLoss.make_value_estimator)

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