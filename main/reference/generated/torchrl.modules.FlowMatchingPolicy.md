# FlowMatchingPolicy

*class*torchrl.modules.FlowMatchingPolicy(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#FlowMatchingPolicy)

TensorDict policy that samples actions by Euler integration.

Parameters:

- **velocity_network** (*nn.Module*) - maps concatenated observation, action and
scalar time to an action-sized velocity.
- **action_dim** (*int*) - number of action coordinates.
- **num_steps** (*int**,**optional*) - Euler integration steps. Defaults to 10.

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

The tensor-only [`FlowMatchingModel`](torchrl.modules.FlowMatchingModel.html#torchrl.modules.FlowMatchingModel) is available as
`module`. Clipping is applied after the final integration step.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import FlowMatchingPolicy
>>> policy = FlowMatchingPolicy(torch.nn.Linear(6, 2), action_dim=2)
>>> td = TensorDict(
... observation=torch.zeros(4, 3), noise=torch.zeros(4, 2),
... batch_size=[4],
... )
>>> policy(td)["action"].shape
torch.Size([4, 2])
```