# DreamerV3ActorLoss

*class*torchrl.objectives.DreamerV3ActorLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ActorLoss)

DreamerV3 Actor Loss.

Rolls out imagined trajectories in latent space using the world model
environment, then computes:

```
loss_actor = -E[log pi(a_t | z_t) * sg(A_t)] - eta * H[pi(. | z_t)]
```

where `A_t = V_lambda(z_t) - v(z_t)` is the advantage (lambda return
minus baseline) and `eta` is the entropy bonus weight.

When the actor is a reparameterizable (continuous) policy the
reparameterization gradient is used directly instead of REINFORCE.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **actor_model** (*TensorDictModule*) - The actor / policy network.
- **value_model** (*TensorDictModule*) - The value network.
- **model_based_env** (*DreamerEnv*) - The imagination environment.
- **imagination_horizon** (*int**,**optional*) - Rollout length inside imagination.
Default: 15.
- **discount_loss** (*bool**,**optional*) - If `True`, discount the actor loss
with a cumulative gamma factor. Default: `True`.
- **entropy_bonus** (*float**,**optional*) - Weight for the entropy regularisation
term `eta`. Default: `3e-4`.
- **use_reinforce** (*bool**,**optional*) - If `True`, uses REINFORCE (log-prob
* stop-gradient advantage). If `False`, uses the straight
reparameterization gradient (suitable for continuous Gaussian
actors). Default: `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... InteractionType,
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... TensorDictModule,
... )
>>> from torchrl.data import Unbounded
>>> from torchrl.envs import TransformedEnv
>>> from torchrl.envs.model_based.dreamer import DreamerEnv
>>> from torchrl.envs.transforms import TensorDictPrimer
>>> from torchrl.modules import MLP, SafeSequential, WorldModelWrapper
>>> from torchrl.modules.distributions.continuous import TanhNormal
>>> from torchrl.modules.models.model_based import DreamerActor
>>> from torchrl.modules.models.model_based_v3 import RSSMPriorV3
>>> from torchrl.objectives import DreamerV3ActorLoss
>>> from torchrl.objectives.utils import ValueEstimators
>>> from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv
>>> base_env = TransformedEnv(
... ContinuousActionConvMockEnv(pixel_shape=[3, 16, 16]),
... TensorDictPrimer(
... random=False, default_value=0,
... state=Unbounded(16), belief=Unbounded(8),
... ),
... )
>>> action_dim = base_env.action_spec.shape[0]
>>> rssm_prior = RSSMPriorV3(
... action_shape=base_env.action_spec.shape,
... hidden_dim=8, rnn_hidden_dim=8,
... num_categoricals=4, num_classes=4, action_dim=action_dim,
... )
>>> transition = SafeSequential(
... TensorDictModule(
... rssm_prior,
... in_keys=["state", "belief", "action"],
... out_keys=["_", "state", "belief"],
... ),
... )
>>> reward = TensorDictModule(
... MLP(out_features=1, depth=1, num_cells=8),
... in_keys=["state", "belief"], out_keys=["reward"],
... )
>>> mb_env = DreamerEnv(
... world_model=WorldModelWrapper(transition, reward),
... prior_shape=torch.Size([16]),
... belief_shape=torch.Size([8]),
... )
>>> mb_env.set_specs_from_env(base_env)
>>> with torch.no_grad():
... _ = mb_env.rollout(3)
>>> actor_module = DreamerActor(out_features=action_dim, depth=1, num_cells=8)
>>> actor = ProbabilisticTensorDictSequential(
... TensorDictModule(
... actor_module, in_keys=["state", "belief"], out_keys=["loc", "scale"],
... ),
... ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"],
... default_interaction_type=InteractionType.RANDOM,
... distribution_class=TanhNormal,
... ),
... )
>>> warmup = TensorDict(
... {"state": torch.randn(1, 2, 16), "belief": torch.randn(1, 2, 8)}, [1]
... )
>>> _ = actor(warmup)
>>> value = TensorDictModule(
... MLP(out_features=1, depth=1, num_cells=8),
... in_keys=["state", "belief"], out_keys=["state_value"],
... )
>>> _ = value(warmup)
>>> loss = DreamerV3ActorLoss(actor, value, mb_env, imagination_horizon=3)
>>> loss.make_value_estimator(ValueEstimators.TDLambda)
>>> td = TensorDict(
... {"state": torch.randn(2, 16), "belief": torch.randn(2, 8)}, [2]
... )
>>> loss_td, _ = loss(td)
>>> "loss_actor" in loss_td.keys()
True
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)*) → tuple[[TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)][[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ActorLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#DreamerV3ActorLoss.make_value_estimator)

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