# OneStepPolicy

*class*torchrl.modules.OneStepPolicy(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#OneStepPolicy)

TensorDict policy distilled from a flow policy.

Parameters:

- **network** (*nn.Module*) - maps concatenated observation and Gaussian noise
directly to an action, without an output activation.
- **action_dim** (*int*) - number of action coordinates.

Keyword Arguments:

- **low** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - lower action bound, broadcast over
actions. Must be strictly less than `high`. Defaults to -1.
- **high** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*or**Tensor**,**optional*) - upper action bound, broadcast over
actions. Defaults to 1.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - observation and optional
noise keys, in that order. Must contain one or two keys.
Defaults to `["observation", "noise"]`.
Missing noise is sampled from a standard normal distribution.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - action output key.
Defaults to `["action"]`.

The tensor-only [`OneStepModel`](torchrl.modules.OneStepModel.html#torchrl.modules.OneStepModel) is available as
`module`; call `module(observation, noise, clamp=False)` for distillation.
Actions are clipped to `[low, high]` by default. Both flow policies sample
Gaussian noise even under deterministic exploration; pass noise for repeatability.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import OneStepPolicy
>>> policy = OneStepPolicy(torch.nn.Linear(5, 2), action_dim=2)
>>> td = TensorDict(observation=torch.zeros(4, 3), batch_size=[4])
>>> policy(td)["action"].shape
torch.Size([4, 2])
```