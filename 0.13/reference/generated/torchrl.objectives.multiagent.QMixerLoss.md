# QMixerLoss

*class*torchrl.objectives.multiagent.QMixerLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/multiagent/qmixer.html#QMixerLoss)

The QMixer loss class.

Mixes local agent q values into a global q value according to a mixing network and then
uses DQN updates on the global value.
This loss is for multi-agent applications.
Therefore, it expects the 'local_value', 'action_value' and 'action' keys
to have an agent dimension (this is visible in the default AcceptedKeys).
This dimension will be mixed by the mixer which will compute a 'global_value' key, used for a DQN objective.
The premade mixers of type `torchrl.modules.models.multiagent.Mixer` will expect the multi-agent
dimension to be the penultimate one.

Parameters:

- **local_value_network** ([*QValueActor*](torchrl.modules.QValueActor.html#torchrl.modules.QValueActor)*or**nn.Module*) - a local Q value operator.
- **mixer_network** (*TensorDictModule**or**nn.Module*) - a mixer network mapping the agents' local Q values
and an optional state to the global Q value. It is suggested to provide a TensorDictModule
wrapping a mixer from `torchrl.modules.models.multiagent.Mixer`.

Keyword Arguments:

- **loss_function** (*str**,**optional*) - loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
Defaults to "l2".
- **delay_value** (*bool**,**optional*) - whether to duplicate the value network
into a new target value network to
create a double DQN. Default is `False`.
- **action_space** (*str**or*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - Action space. Must be one of
`"one-hot"`, `"mult_one_hot"`, `"binary"` or `"categorical"`,
or an instance of the corresponding specs ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot),
[`torchrl.data.MultiOneHot`](torchrl.data.MultiOneHot.html#torchrl.data.MultiOneHot),
[`torchrl.data.Binary`](torchrl.data.Binary.html#torchrl.data.Binary) or [`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)).
If not provided, an attempt to retrieve it from the value network
will be made.
- **priority_key** (*NestedKey**,**optional*) - [Deprecated, use .set_keys(priority_key=priority_key) instead]
The key at which priority is assumed to be stored within TensorDicts added
to this ReplayBuffer. This is to be used when the sampler is of type
`PrioritizedSampler`. Defaults to `"td_error"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import QValueModule, SafeSequential
>>> from torchrl.modules.models.multiagent import QMixer
>>> from torchrl.objectives.multiagent import QMixerLoss
>>> n_agents = 4
>>> module = TensorDictModule(
... nn.Linear(10,3), in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
... )
>>> value_module = QValueModule(
... action_value_key=("agents", "action_value"),
... out_keys=[
... ("agents", "action"),
... ("agents", "action_value"),
... ("agents", "chosen_action_value"),
... ],
... action_space="categorical",
... )
>>> qnet = SafeSequential(module, value_module)
>>> qmixer = TensorDictModule(
... module=QMixer(
... state_shape=(64, 64, 3),
... mixing_embed_dim=32,
... n_agents=n_agents,
... device="cpu",
... ),
... in_keys=[("agents", "chosen_action_value"), "state"],
... out_keys=["chosen_action_value"],
... )
>>> loss = QMixerLoss(qnet, qmixer, action_space="categorical")
>>> td = TensorDict(
... {
... "agents": TensorDict(
... {"observation": torch.zeros(32, n_agents, 10)}, [32, n_agents]
... ),
... "state": torch.zeros(32, 64, 64, 3),
... "next": TensorDict(
... {
... "agents": TensorDict(
... {"observation": torch.zeros(32, n_agents, 10)}, [32, n_agents]
... ),
... "state": torch.zeros(32, 64, 64, 3),
... "reward": torch.zeros(32, 1),
... "done": torch.zeros(32, 1, dtype=torch.bool),
... "terminated": torch.zeros(32, 1, dtype=torch.bool),
... },
... [32],
... ),
... },
... [32],
... )
>>> loss(qnet(td))
TensorDict(
 fields={
 loss: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/multiagent/qmixer.html#QMixerLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/multiagent/qmixer.html#QMixerLoss.make_value_estimator)

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