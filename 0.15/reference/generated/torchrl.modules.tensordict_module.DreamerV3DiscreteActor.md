# DreamerV3DiscreteActor

*class*torchrl.modules.tensordict_module.DreamerV3DiscreteActor(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3DiscreteActor)

DreamerV3 one-hot categorical policy over stochastic state and belief.

The RMS-normalized SiLU network concatenates belief before state, initializes
its output weights with scale `0.01`, and mixes categorical probabilities
with a uniform distribution. Logits remain in float32 under autocast.
Calling the actor writes logits, a hard one-hot action and its log probability
into the input tensordict. Sampling is random by default and respects
[`set_exploration_type()`](torchrl.envs.set_exploration_type.html#torchrl.envs.set_exploration_type). Use `get_dist()` for
differentiable straight-through sampling with `distribution.rsample()`.

Reference: Hafner et al., DreamerV3 (2023),
[https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

Parameters:

- **in_features** (*int*) - Sum of the flattened stochastic state and belief widths.
- **out_features** (*int*) - Number of discrete actions.

Keyword Arguments:

- **depth** (*int**,**optional*) - Number of hidden layers. Must be positive.
Defaults to `3`.
- **num_cells** (*int**,**optional*) - Width of each hidden layer. Defaults to `1024`.
- **norm_eps** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - RMS normalization epsilon. Defaults to `1e-4`.
- **unimix** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - Uniform probability fraction in `[0, 1)`.
Defaults to `0.01`.
- **in_keys** ([*Sequence*](torchrl.data.Sequence.html#torchrl.data.Sequence)*[**NestedKey**] or**None**,**optional*) - Exactly two input keys,
in stochastic state, then belief order. Defaults to
`["state", "belief"]` when `None`.
- **action_key** (*NestedKey**,**optional*) - Output one-hot action key. Defaults to
`"action"`.
- **logits_key** (*NestedKey**,**optional*) - Output mixed log-probability key.
Defaults to `"logits"`.
- **log_prob_key** (*NestedKey**,**optional*) - Output sampled-action log-probability
key. Defaults to `"action_log_prob"`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**or**None**,**optional*) - Initial parameter device.
Defaults to `None`, using the default torch device.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs import ExplorationType, set_exploration_type
>>> from torchrl.modules import DreamerV3DiscreteActor
>>> actor = DreamerV3DiscreteActor(12, 3, depth=2, num_cells=32)
>>> data = TensorDict({"state": torch.randn(4, 8), "belief": torch.randn(4, 4)}, [4])
>>> with set_exploration_type(ExplorationType.DETERMINISTIC):
... result = actor(data)
>>> result["action"].sum(-1).tolist()
[1, 1, 1, 1]
>>> distribution = actor.get_dist(data)
>>> action = distribution.rsample()
>>> (action * torch.arange(3)).sum().backward()
>>> distribution.log_prob(action).shape
torch.Size([4])
```

See also

`DreamerV3DiscreteActorConfig`