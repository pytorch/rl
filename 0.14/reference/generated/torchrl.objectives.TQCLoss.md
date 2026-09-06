# TQCLoss

*class*torchrl.objectives.TQCLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/tqc.html#TQCLoss)

Truncated Quantile Critics loss.

TQC critics predict return quantiles instead of one value. Bellman targets
pool every target critic's atoms, discard the highest from the shared pool,
and train each critic on what remains. The actor uses the full mixture.

See [Kuznetsov et al. (2020), Controlling Overestimation Bias with
Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269).

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - policy used to
sample actions and score their entropy.
- **qvalue_network** (*TensorDictModule**or**list**of**TensorDictModule*) - critic whose last output dimension holds return atoms. One module
is copied across the ensemble; a list supplies each critic, as in
[`SACLoss`](torchrl.objectives.SACLoss.html#torchrl.objectives.SACLoss).

Keyword Arguments:

- **num_qvalue_nets** (*int**,**optional*) - ensemble size for one critic module.
Defaults to `5`.
- **top_quantiles_to_drop_per_net** (*int**,**optional*) - upper atoms discarded per
critic from the pooled target. The total is this value times the
ensemble size. Defaults to `2`.
- **alpha_init** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - initial entropy temperature. Defaults to
`1.0`.
- **min_alpha** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - temperature floor. Defaults to `None`.
- **max_alpha** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - temperature ceiling. Defaults to `None`.
- **action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*,**optional*) - action domain for automatic target
entropy. Defaults to the actor's spec.
- **fixed_alpha** (*bool**,**optional*) - disable temperature learning. Defaults to
`False`.
- **target_entropy** (float or `"auto"`, optional) - entropy target.
Defaults to `"auto"`.
- **delay_qvalue** (*bool**,**optional*) - maintain delayed target critics.
Defaults to `True`.
- **separate_losses** (*bool**,**optional*) - exclude shared actor parameters from
critic training. Defaults to `False`.
- **reduction** (*str**,**optional*) - `"none"`, `"mean"`, or
`"sum"`. Defaults to `"mean"`.
- **deactivate_vmap** (*bool**,**optional*) - loop over critics instead of using
`vmap`. Defaults to `False`.
- **skip_done_states** (*bool**,**optional*) - skip terminal next-state evaluation.
Defaults to `False`.
- **use_prioritized_weights** (bool or `"auto"`, optional) - use replay
weights when present. Defaults to `"auto"`.
- **scalar_output_mode** (*str**,**optional*) - scalar handling when
`reduction="none"`. See [`SACLoss`](torchrl.objectives.SACLoss.html#torchrl.objectives.SACLoss).

Note

Among TorchRL's built-in value estimators, TQC supports only
[`TD0Estimator`](torchrl.objectives.value.TD0Estimator.html#torchrl.objectives.value.TD0Estimator). Reward and termination
leaves are explicitly expanded to the target-atom shape so the value
estimator retains its strict shape checks.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import NormalParamExtractor, TensorDictModule
>>> from torchrl.data import Bounded
>>> from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
>>> from torchrl.modules.distributions import TanhNormal
>>> from torchrl.objectives import SoftUpdate, TQCLoss
>>> n_obs, n_act, n_quantiles = 3, 2, 8
>>> action_spec = Bounded(-1, 1, (n_act,))
>>> actor_net = nn.Sequential(
... nn.Linear(n_obs, 2 * n_act), NormalParamExtractor()
... )
>>> actor = ProbabilisticActor(
... TensorDictModule(
... actor_net,
... in_keys=["observation"],
... out_keys=["loc", "scale"],
... ),
... in_keys=["loc", "scale"],
... spec=action_spec,
... distribution_class=TanhNormal,
... )
>>> critic = ValueOperator(
... MLP(
... in_features=n_obs + n_act,
... out_features=n_quantiles,
... num_cells=[],
... ),
... in_keys=["observation", "action"],
... )
>>> loss = TQCLoss(actor, critic, num_qvalue_nets=2)
>>> loss.make_value_estimator(gamma=0.99)
>>> target_updater = SoftUpdate(loss, eps=0.995)
>>> batch = TensorDict(
... {
... "observation": torch.randn(4, n_obs),
... "action": action_spec.rand((4,)),
... "next": {
... "observation": torch.randn(4, n_obs),
... "reward": torch.randn(4, 1),
... "done": torch.zeros(4, 1, dtype=torch.bool),
... "terminated": torch.zeros(4, 1, dtype=torch.bool),
... },
... },
... batch_size=[4],
... )
>>> output = loss(batch)
>>> output.get("loss_qvalue").shape
torch.Size([])
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.