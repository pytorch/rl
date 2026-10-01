# ClipPPOLoss

*class*torchrl.objectives.ClipPPOLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/ppo.html#ClipPPOLoss)

Clipped PPO loss.

The clipped importance weighted loss is computed as follows:

loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - policy operator.
- **critic_network** ([*ValueOperator*](torchrl.modules.ValueOperator.html#torchrl.modules.ValueOperator)) - value operator.

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

- **clip_epsilon** (*scalar**or**tuple**of**scalars**,**optional*) -

weight clipping threshold(s) in the clipped
PPO loss equation.

- float `x`: symmetric clipping `[1 - x, 1 + x]`. Default: `0.2`.
- tuple `(eps_low, eps_high)`: asymmetric clipping `[1 - eps_low, 1 + eps_high]` as in
DAPO Clip-Higher (recommended values `(0.20, 0.28)`; see Eq. (10) of the
[DAPO paper](https://arxiv.org/html/2503.14476)). With a tuple, the thresholds are
exposed (and schedulable) as the `clip_epsilon_low` / `clip_epsilon_high` buffers
instead of `clip_epsilon`, and `clip_value=True` is not allowed (pass an explicit
float threshold instead).
- **entropy_bonus** (*bool**,**optional*) - if `True`, an entropy bonus will be added to the
loss to favour exploratory policies.
- **samples_mc_entropy** (*int**,**optional*) - if the distribution retrieved from the policy
operator does not have a closed form
formula for the entropy, a Monte-Carlo estimate will be used.
`samples_mc_entropy` will control how many
samples will be used to compute this estimate.
Defaults to `1`.
- **entropy_coeff** -

(scalar | Mapping[NestedKey, scalar], optional): entropy multiplier when computing the total loss.
* **Scalar**: one value applied to the summed entropy of every action head.
* **Mapping** `{head_name: coeff}` gives an individual coefficient for each action-head's entropy.
Defaults to `0.01`.

See ppo_entropy_coefficients for detailed usage examples and troubleshooting.
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
- **advantage_norm** ([*ValueNorm*](torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm)*,**optional*) - a stateful
[`ValueNorm`](torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm) used to rescale the advantage,
e.g. [`PercentileValueNorm`](torchrl.modules.PercentileValueNorm.html#torchrl.modules.PercentileValueNorm) for
DreamerV3-style return normalization. The advantage is divided by
`advantage_norm.scale()` without re-centering (the advantage is
already centred by the value baseline). In training mode, the
statistics are updated with the fresh value targets when the loss
runs its own value estimator (i.e. when the input tensordict
carries no advantage entry), so repeated forwards over the same
precomputed rollout do not skew the moving average; rows marked
invalid by the validity masks (see
[`loss_mask_key`](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.loss_mask_key)) are excluded
from the update. When the advantage is precomputed outside the
loss, call `advantage_norm.update(value_target)` once per rollout
yourself. Mutually exclusive with `normalize_advantage`.
Defaults to `None`.
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
- **clip_value** (*bool**or*[*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - If a `float` is provided, it will be used to compute a clipped
version of the value prediction with respect to the input tensordict value estimate and use it to
calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
helping stabilize training and preventing large updates. However, it will have no impact if the value
estimate was done by the current version of the value estimator. If instead `True` is provided, the
`clip_epsilon` parameter will be used as the clipping threshold (this is only compatible with a
scalar `clip_epsilon`; with an asymmetric `(low, high)` tuple, pass an explicit float threshold
instead). If not provided or `False`, no clipping will be performed. Defaults to `False`.
- **delay_actor** (*bool**,**optional*) - if `True`, a detached copy of the actor parameters is kept under
`target_actor_network_params` and used as the *proximal policy*: the clipping acts on
`pi_theta / pi_prox` while the surrogate is re-weighted by `pi_prox / pi_behav`, where
`pi_behav` is the *behavior policy* that collected the data (`sample_log_prob` entry), so the
gradient estimate remains unbiased for the data at hand. This decouples the strength of the trust
region from how (and how recently) the data was collected. Updating the target parameters with
`SoftUpdate` after every optimizer step makes the proximal policy an
exponentially-weighted moving average of the policy, i.e. PPO-EWMA ("Batch size-invariance for
policy optimization", Hilton et al., 2021, [https://arxiv.org/abs/2110.00641](https://arxiv.org/abs/2110.00641)); see the note below.
Requires `functional=True`. Defaults to `False`.
- **max_importance_ratio** (`float`, optional) - if provided, the behavior ratio `pi_theta / pi_behav` is
capped at this value. Stale data, or a proximal policy that drifted away from the behavior policy,
can otherwise produce arbitrarily large ratios. The cap lifts the (gradient-free) behavior
log-probability rather than clamping the ratio, so capped samples keep a rescaled policy gradient,
as in the reference PPO-EWMA implementation (which uses `100.0`). Defaults to `None` (no cap).
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
>>> loss_module = ClipPPOLoss(model.get_policy_operator(), model.get_value_operator())
>>> # second option, with a single call to the common module
>>> loss_module = ClipPPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)
```

This will work regardless of whether separate_losses is activated or not.

Note

**Decoupled proximal policy (PPO-EWMA).** With `delay_actor=True` the loss follows the decoupled
clipped objective of Hilton et al. (2021),

> loss = -(pi_prox / pi_behav) * min(r * advantage, clip(r, 1 - eps, 1 + eps) * advantage),
> r = pi_theta / pi_prox,

where `pi_prox` runs on `target_actor_network_params`. Any `TargetNetUpdater`
controls how old the proximal policy is: a `SoftUpdate` stepped after every
optimizer step gives the EWMA of the paper (`eps` being the decay rate `beta_prox`), a
`HardUpdate` refreshes it every `value_network_update_interval` steps.
`clip_fraction` and `kl_approx` then describe `r`, the ratio the trust region acts on, whereas
`ESS`, `max_ratio` and `mean_ratio` describe the behavior ratio `pi_theta / pi_behav` the data
is actually weighted by.

```
>>> loss_module = ClipPPOLoss(actor, critic, delay_actor=True, max_importance_ratio=100.0)
>>> updater = SoftUpdate(loss_module, eps=0.889)
>>> for batch in replay_buffer:
... losses = loss_module(batch)
... loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
... loss.backward()
... optimizer.step()
... optimizer.zero_grad()
... updater.step()
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/ppo.html#ClipPPOLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.