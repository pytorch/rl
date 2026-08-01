# KLPENPPOLoss

*class*torchrl.objectives.KLPENPPOLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/ppo.html#KLPENPPOLoss)

KL Penalty PPO loss.

The KL penalty loss has the following formula:

loss = loss - beta * KL(old_policy, new_policy)

The "beta" parameter is adapted on-the-fly to match a target KL divergence between the new and old policy, thus
favouring a certain level of distancing between the two while still preventing them to be too much apart.

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - policy operator.
- **critic_network** ([*ValueOperator*](torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)) - value operator.

Keyword Arguments:

- **dtarg** (*scalar**,**optional*) - target KL divergence. Defaults to `0.01`.
- **samples_mc_kl** (*int**,**optional*) - number of samples used to compute the KL divergence
if no analytical formula can be found. Defaults to `1`.
- **beta** (*scalar**,**optional*) - initial KL divergence multiplier.
Defaults to `1.0`.
- **decrement** (*scalar**,**optional*) - how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
default: `0.5`.
- **increment** (*scalar**,**optional*) - how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
default: `2.0`.
- **entropy_bonus** (*bool**,**optional*) - if `True`, an entropy bonus will be added to the
loss to favour exploratory policies. Defaults to `True`.
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
- **critic_coeff** (*scalar**,**optional*) - critic loss multiplier when computing the total
loss. Defaults to `1.0`.
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

If the actor and the value function share parameters, one can avoid
calling the common module multiple times by passing only the head of the
value network to the PPO loss module:

```
>>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
>>> actor_head = SomeActor(in_keys=["hidden"])
>>> value_head = SomeValue(in_keys=["hidden"])
>>> # first option, with 2 calls on the common module
>>> model = ActorValueOperator(common, actor_head, value_head)
>>> loss_module = KLPENPPOLoss(model.get_policy_operator(), model.get_value_operator())
>>> # second option, with a single call to the common module
>>> loss_module = KLPENPPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)
```

This will work regardless of whether separate_losses is activated or not.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/ppo.html#KLPENPPOLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.