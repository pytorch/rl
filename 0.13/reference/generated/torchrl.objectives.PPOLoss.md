# PPOLoss

*class*torchrl.objectives.PPOLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/ppo.html#PPOLoss)

A parent PPO loss class.

PPO (Proximal Policy Optimization) is a model-free, online RL algorithm
that makes use of a recorded (batch of)
trajectories to perform several optimization steps, while actively
preventing the updated policy to deviate too
much from its original parameter configuration.

PPO loss can be found in different flavors, depending on the way the
constrained optimization is implemented: ClipPPOLoss and KLPENPPOLoss.
Unlike its subclasses, this class does not implement any regularization
and should therefore be used cautiously.

For more details regarding PPO, refer to: "Proximal Policy Optimization Algorithms",
[https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - policy operator.
Typically, a [`ProbabilisticTensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) subclass taking observations
as input and outputting an action (or actions) as well as its log-probability value.
- **critic_network** ([*ValueOperator*](torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)) - value operator. The critic will usually take the observations as input
and return a scalar value (`state_value` by default) in the output keys.

Note

While this loss module does not enforce any specific model mode (train/eval), it is highly recommended
to keep your model in eval mode during RL training to ensure deterministic behavior.
A failure to learn due to a train/eval mode mismatch is often observed when the Effective Sample Size (ESS)
drops or increases significantly (see note below).

Note

The PPO loss exposes a couple of additional metrics that can be used to monitor the training process:

- The clip fraction is the ratio of the number of clipped weights in the PPO loss (i.e. the ratio of the number of weights that were clipped to the total number of weights).
- The Effective Sample Size (ESS) is a measure of the effective number of samples in the batch, computed as the inverse of the sum of the squared importance weights.
A value of 1 indicates that the importance weights are all equal to 1 (i.e., the samples are equally weighted).
Any value below 1 indicates that the samples are not equally weighted, and the ESS is a measure of the effective number of samples.
If the value drops or increases significantly, it often indicates issues with the model configuration (such as a train/eval mode mismatch, or a large policy update).

Keyword Arguments:

- **entropy_bonus** (*bool**,**optional*) - if `True`, an entropy bonus will be added to the
loss to favour exploratory policies.
- **samples_mc_entropy** (*int**,**optional*) - if the distribution retrieved from the policy
operator does not have a closed form
formula for the entropy, a Monte-Carlo estimate will be used.
`samples_mc_entropy` will control how many
samples will be used to compute this estimate.
Defaults to `1`.
- **entropy_coeff** -

scalar | Mapping[NestedKey, scalar], optional): entropy multiplier when computing the total loss.
* **Scalar**: one value applied to the summed entropy of every action head.
* **Mapping** `{head_name: coeff}` gives an individual coefficient for each action-head's entropy.
Defaults to `0.01`.

See ppo_entropy_coefficients for detailed usage examples and troubleshooting.
- **log_explained_variance** (*bool**,**optional*) - if `True`, the explained variance of the critic
predictions w.r.t. value targets will be computed and logged as `"explained_variance"`.
This can help monitor critic quality during training. Best possible score is 1.0, lower values are worse. Defaults to `True`.
- **critic_coeff** (*scalar**,**optional*) - critic loss multiplier when computing the total
loss. Defaults to `1.0`. Set `critic_coeff` to `None` to exclude the value
loss from the forward outputs.
- **loss_critic_type** (*str**,**optional*) - loss function for the value discrepancy.
Can be one of "l1", "l2" or "smooth_l1". Defaults to `"smooth_l1"`.
- **normalize_advantage** (*bool**,**optional*) - if `True`, the advantage will be normalized
before being used. Defaults to `False`.
- **normalize_advantage_exclude_dims** (*Tuple**[**int**]**,**optional*) - dimensions to exclude from the advantage
standardization. Negative dimensions are valid. This is useful in multiagent (or multiobjective) settings
where the agent (or objective) dimension may be excluded from the reductions. Default: ().
- **separate_losses** (*bool**,**optional*) - if `True`, shared parameters between
policy and critic will only be trained on the policy loss.
Defaults to `False`, i.e., gradients are propagated to shared
parameters for both policy and critic losses.
- **advantage_key** (*str**,**optional*) - [Deprecated, use set_keys(advantage_key=advantage_key) instead]
The input tensordict key where the advantage is
expected to be written. Defaults to `"advantage"`.
- **value_target_key** (*str**,**optional*) - [Deprecated, use set_keys(value_target_key=value_target_key) instead]
The input tensordict key where the target state
value is expected to be written. Defaults to `"value_target"`.
- **value_key** (*str**,**optional*) - [Deprecated, use set_keys(value_key) instead]
The input tensordict key where the state
value is expected to be written. Defaults to `"state_value"`.
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
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) -

device of the buffers. Defaults to `None`.

Note

Parameters and buffers from the policy / critic will not be cast to that device to ensure that
the storages match the ones that are passed to other components, such as data collectors.

Note

The advantage (typically GAE) can be computed by the loss function or
in the training loop. The latter option is usually preferred, but this is
up to the user to choose which option is to be preferred.
If the advantage key (`"advantage` by default) is not present in the
input tensordict, the advantage will be computed by the `forward()`
method.

```
>>> ppo_loss = PPOLoss(actor, critic)
>>> advantage = GAE(critic)
>>> data = next(datacollector)
>>> losses = ppo_loss(data)
>>> # equivalent
>>> advantage(data)
>>> losses = ppo_loss(data)
```

A custom advantage module can be built using `make_value_estimator()`.
The default is [`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE) with hyperparameters
dictated by `default_value_kwargs()`.

```
>>> ppo_loss = PPOLoss(actor, critic)
>>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
>>> data = next(datacollector)
>>> losses = ppo_loss(data)
```

Note

If the actor and the value function share parameters, one can avoid
calling the common module multiple times by passing only the head of the
value network to the PPO loss module:

```
>>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
>>> actor_head = SomeActor(in_keys=["hidden"])
>>> value_head = SomeValue(in_keys=["hidden"])
>>> # first option, with 2 calls on the common module
>>> model = ActorValueOperator(common, actor_head, value_head)
>>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
>>> # second option, with a single call to the common module
>>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)
```

This will work regardless of whether separate_losses is activated or not.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.ppo import PPOLoss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> base_layer = nn.Linear(n_obs, 5)
>>> net = nn.Sequential(base_layer, nn.Linear(5, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor = ProbabilisticActor(
... module=module,
... distribution_class=TanhNormal,
... in_keys=["loc", "scale"],
... spec=spec)
>>> module = nn.Sequential(base_layer, nn.Linear(5, 1))
>>> value = ValueOperator(
... module=module,
... in_keys=["observation"])
>>> loss = PPOLoss(actor, value)
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> data = TensorDict({"observation": torch.randn(*batch, n_obs),
... "action": action,
... "action_log_prob": torch.randn_like(action[..., 1]),
... ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "reward"): torch.randn(*batch, 1),
... ("next", "observation"): torch.randn(*batch, n_obs),
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 explained_variance: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 kl_approx: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_critic: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_objective: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "sample_log_prob", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor and value network.
The return value is a tuple of tensors in the following order:
`["loss_objective"]` + `["entropy", "loss_entropy"]` if entropy_bonus is set + `"loss_critic"` if critic_coeff is not `None`.
The output keys can also be filtered using `PPOLoss.select_out_keys()` method.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.ppo import PPOLoss
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> base_layer = nn.Linear(n_obs, 5)
>>> net = nn.Sequential(base_layer, nn.Linear(5, 2 * n_act), NormalParamExtractor())
>>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
>>> actor = ProbabilisticActor(
... module=module,
... distribution_class=TanhNormal,
... in_keys=["loc", "scale"],
... spec=spec)
>>> module = nn.Sequential(base_layer, nn.Linear(5, 1))
>>> value = ValueOperator(
... module=module,
... in_keys=["observation"])
>>> loss = PPOLoss(actor, value)
>>> loss.set_keys(sample_log_prob="sampleLogProb")
>>> _ = loss.select_out_keys("loss_objective")
>>> batch = [2, ]
>>> action = spec.rand(batch)
>>> loss_objective = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... sampleLogProb=torch.randn_like(action[..., 1]) / 10,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_reward=torch.randn(*batch, 1),
... next_observation=torch.randn(*batch, n_obs))
>>> loss_objective.backward()
```

**Simple Entropy Coefficient Examples**:

```
>>> # Scalar entropy coefficient (default behavior)
>>> loss = PPOLoss(actor, critic, entropy_coeff=0.01)
>>>
>>> # Per-head entropy coefficients (for composite action spaces)
>>> entropy_coeff = {
... ("agent0", "action_log_prob"): 0.01, # Low exploration
... ("agent1", "action_log_prob"): 0.05, # High exploration
... }
>>> loss = PPOLoss(actor, critic, entropy_coeff=entropy_coeff)
```

Note

There is an exception regarding compatibility with non-tensordict-based modules.
If the actor network is probabilistic and uses a [`CompositeDistribution`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution),
this class must be used with tensordicts and cannot function as a tensordict-independent module.
This is because composite action spaces inherently rely on the structured representation of data provided by
tensordicts to handle their actions.

Note

**Entropy Bonus and Coefficient Management**

The entropy bonus encourages exploration by adding the negative entropy of the policy to the loss.
This can be configured in two ways:

**Scalar Coefficient (Default)**: Use a single coefficient for all action heads:

```
>>> loss = PPOLoss(actor, critic, entropy_coeff=0.01)
```

**Per-Head Coefficients**: Use different coefficients for different action components:

```
>>> # For a robot with movement and gripper actions
>>> entropy_coeff = {
... ("agent0", "action_log_prob"): 0.01, # Movement: low exploration
... ("agent1", "action_log_prob"): 0.05, # Gripper: high exploration
... }
>>> loss = PPOLoss(actor, critic, entropy_coeff=entropy_coeff)
```

**Key Requirements**: When using per-head coefficients, you must provide the full nested key
path to each action head's log probability (e.g., ("agent0", "action_log_prob")).

**Monitoring Entropy Loss**:

When using composite action spaces, the loss output includes:
- "entropy": Summed entropy across all action heads (for logging)
- "composite_entropy": Individual entropy values for each action head
- "loss_entropy": The weighted entropy loss term

Example output:

```
>>> result = loss(data)
>>> print(result["entropy"]) # Total entropy: 2.34
>>> print(result["composite_entropy"]) # Per-head: {"movement": 1.2, "gripper": 1.14}
>>> print(result["loss_entropy"]) # Weighted loss: -0.0234
```

**Common Issues**:

**KeyError: "Missing entropy coeff for head 'head_name'"**:

- Ensure you provide coefficients for ALL action heads
- Use full nested keys: ("head_name", "action_log_prob")
- Check that your action space structure matches the coefficient mapping

**Incorrect Entropy Calculation**:

- Call set_composite_lp_aggregate(False).set() before creating your policy
- Verify that your action space uses [`CompositeDistribution`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution)

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/ppo.html#PPOLoss.forward)

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

loss_critic(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), ...][[source]](../../_modules/torchrl/objectives/ppo.html#PPOLoss.loss_critic)

Returns the critic loss multiplied by `critic_coeff`, if it is not `None`.

make_value_estimator(*value_type: [ValueEstimators](torchrl.objectives.ValueEstimators.html#torchrl.objectives.ValueEstimators) = None*, ***hyperparams*)[[source]](../../_modules/torchrl/objectives/ppo.html#PPOLoss.make_value_estimator)

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