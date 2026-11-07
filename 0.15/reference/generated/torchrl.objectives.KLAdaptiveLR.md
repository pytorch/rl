# KLAdaptiveLR

*class*torchrl.objectives.KLAdaptiveLR(*optimizer: [Optimizer](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)*, *target_kl: float*, ***, *factor: float = 1.5*, *min_lr: float = 1e-05*, *max_lr: float = 0.01*)[[source]](../../_modules/torchrl/objectives/utils.html#KLAdaptiveLR)

Adapt an optimizer's learning rate to a target policy KL divergence.

After each policy update, compare the measured mean KL divergence between
the old and the new policy with `target_kl`: when it exceeds
`2 * target_kl` the learning rate is divided by `factor`, when it is
positive but below `target_kl / 2` it is multiplied by `factor`, and it
is left unchanged in between. The learning rate of every parameter group is
clamped to `[min_lr, max_lr]`. A KL of exactly zero leaves the learning
rate unchanged, so a policy that did not move does not trigger runaway
growth.

This is the schedule used by the `rsl_rl` PPO implementation (Rudin et
al., "Learning to Walk in Minutes Using Massively Parallel Deep
Reinforcement Learning", [https://arxiv.org/abs/2109.11978](https://arxiv.org/abs/2109.11978)). The
`kl_approx` output of [`ClipPPOLoss`](torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) can be
passed directly to `step()`.

Parameters:

- **optimizer** ([*torch.optim.Optimizer*](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)) - optimizer whose parameter groups are
rescaled in place.
- **target_kl** (*float*) - desired mean KL divergence per update.

Keyword Arguments:

- **factor** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - multiplicative change applied when the KL
leaves the `[target_kl / 2, 2 * target_kl]` band. Must be greater
than one. Defaults to `1.5`.
- **min_lr** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - lower bound of the learning rate. Defaults to
`1e-5`.
- **max_lr** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - upper bound of the learning rate. Defaults to
`1e-2`.

Examples

```
>>> import torch
>>> from torchrl.objectives import KLAdaptiveLR
>>> params = [torch.nn.Parameter(torch.zeros(1))]
>>> optimizer = torch.optim.Adam(params, lr=1e-3)
>>> scheduler = KLAdaptiveLR(optimizer, target_kl=0.01, factor=2.0)
>>> scheduler.step(kl=0.05) # the update was too large: halve the lr
>>> optimizer.param_groups[0]["lr"]
0.0005
>>> scheduler.step(kl=0.001) # the update was too small: double it
>>> optimizer.param_groups[0]["lr"]
0.001
```

get_last_lr() → list[float][[source]](../../_modules/torchrl/objectives/utils.html#KLAdaptiveLR.get_last_lr)

Return the current learning rate of each parameter group.

load_state_dict(*state_dict: dict[str, Any]*) → None[[source]](../../_modules/torchrl/objectives/utils.html#KLAdaptiveLR.load_state_dict)

Load a state produced by `state_dict()`.

state_dict() → dict[str, Any][[source]](../../_modules/torchrl/objectives/utils.html#KLAdaptiveLR.state_dict)

Return the scheduler state, excluding the optimizer.

step(*kl: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/objectives/utils.html#KLAdaptiveLR.step)

Rescale the learning rate from the KL divergence of the last update.

Parameters:

**kl** (*float**or**Tensor*) - mean KL divergence between the policy before
and after the update. A zero-dimensional tensor is accepted.